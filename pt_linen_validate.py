""" ImageNet Validation Script
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import fnmatch
import os
import time

import jax
from timm.data import create_dataset, create_loader, resolve_data_config

from jeffnet.common import get_model_cfg, list_models, correct_topk, AverageMeter
from jeffnet.linen import create_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='pt_efficientnet_b0',
                    help='model architecture (default: pt_efficientnet_b0)')
parser.add_argument('-b', '--batch-size', default=250, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--no-jit', action='store_true', default=False,
                    help='Disable jit of model (for comparison).')


def validate(args):
    rng = jax.random.PRNGKey(0)
    model, variables = create_model(args.model, pretrained=True, rng=rng)
    print(f'Created {args.model} model. Validating...')

    if args.no_jit:
        eval_step = lambda images, labels: eval_forward(model, variables, images, labels)
    else:
        eval_step = jax.jit(lambda images, labels: eval_forward(model, variables, images, labels))

    dataset = create_dataset('imagenet', args.data)

    data_config = resolve_data_config(vars(args), model=model)
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=8,
        crop_pct=data_config['crop_pct'])

    batch_time = AverageMeter()
    correct_top1, correct_top5 = 0, 0
    total_examples = 0
    start_time = prev_time = time.time()
    for batch_index, (images, labels) in enumerate(loader):
        images = images.numpy().transpose(0, 2, 3, 1)
        labels = labels.numpy()

        top1_count, top5_count = eval_step(images, labels)
        correct_top1 += int(top1_count)
        correct_top5 += int(top5_count)
        total_examples += images.shape[0]

        batch_time.update(time.time() - prev_time)
        if batch_index % 20 == 0 and batch_index > 0:
            print(
                f'Test: [{batch_index:>4d}/{len(loader)}]  '
                f'Rate: {images.shape[0] / batch_time.val:>5.2f}/s ({images.shape[0] / batch_time.avg:>5.2f}/s) '
                f'Acc@1: {100 * correct_top1 / total_examples:>7.3f} '
                f'Acc@5: {100 * correct_top5 / total_examples:>7.3f}')
        prev_time = time.time()

    acc_1 = 100 * correct_top1 / total_examples
    acc_5 = 100 * correct_top5 / total_examples
    print(f'Validation complete. {total_examples / (prev_time - start_time):>5.2f} img/s. '
          f'Acc@1 {acc_1:>7.3f}, Acc@5 {acc_5:>7.3f}')
    return dict(top1=float(acc_1), top5=float(acc_5))


def eval_forward(model, variables, images, labels):
    logits = model.apply(variables, images, mutable=False, training=False)
    top1_count, top5_count = correct_topk(logits, labels, topk=(1, 5))
    return top1_count, top5_count


def main():
    args = parser.parse_args()
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    jax.config.enable_omnistaging()

    def _try_validate(args):
        res = None
        batch_size = args.batch_size
        while res is None:
            try:
                print(f'Setting validation batch size to {batch_size}')
                args.batch_size = batch_size
                res = validate(args)
            except RuntimeError as e:
                if batch_size <= 1:
                    print("Validation failed with no ability to reduce batch size. Exiting.")
                    raise e
                batch_size = max(batch_size // 2, 1)
                print("Validation failed, reducing batch size by 50%")
        return res

    if get_model_cfg(args.model) is not None:
        _try_validate(args)
    else:
        models = list_models(pretrained=True)
        if args.model != 'all':
            models = fnmatch.filter(models, args.model)
        if not models:
            print(f'ERROR: No models found to validate with pattern {args.model}.')
            exit(1)

        print('Validating:', ', '.join(models))
        results = []
        start_batch_size = args.batch_size
        for m in models:
            args.batch_size = start_batch_size  # reset in case reduced for retry
            args.model = m
            res = _try_validate(args)
            res.update(dict(model=m))
            results.append(res)
        print('Results:')
        for r in results:
            print(f"Model: {r['model']}, Top1: {r['top1']}, Top5: {r['top5']}")


if __name__ == '__main__':
    main()
