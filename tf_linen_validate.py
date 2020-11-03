""" ImageNet Validation Script
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import time
import argparse
import jax
import fnmatch

import jeffnet.data.tf_imagenet_data as imagenet_data
from jeffnet.common import correct_topk, AverageMeter, list_models, get_model_cfg
from jeffnet.linen import create_model


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientnet_b0',
                    help='model architecture (default: tf_efficientnet_b0)')
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

    """Runs evaluation and returns top-1 accuracy."""
    image_size = model.default_cfg['input_size'][-1]
    test_ds, num_batches = imagenet_data.load(
        imagenet_data.Split.TEST,
        is_training=False,
        image_size=image_size,
        batch_dims=[args.batch_size],
        mean=tuple([x * 255 for x in model.default_cfg['mean']]),
        std=tuple([x * 255 for x in model.default_cfg['std']]),
        tfds_data_dir=args.data)

    batch_time = AverageMeter()
    correct_top1, correct_top5 = 0, 0
    total_examples = 0
    start_time = prev_time = time.time()
    for batch_index, batch in enumerate(test_ds):
        images, labels = batch['images'], batch['labels']
        top1_count, top5_count = eval_step(images, labels)
        correct_top1 += int(top1_count)
        correct_top5 += int(top5_count)
        total_examples += images.shape[0]

        batch_time.update(time.time() - prev_time)
        if batch_index % 20 == 0 and batch_index > 0:
            print(
                f'Test: [{batch_index:>4d}/{num_batches}]  '
                f'Rate: {images.shape[0] / batch_time.val:>5.2f}/s ({images.shape[0] / batch_time.avg:>5.2f}/s) '
                f'Acc@1: {100 * correct_top1 / total_examples:>7.3f} '
                f'Acc@5: {100 * correct_top5 / total_examples:>7.3f}')
        prev_time = time.time()

    acc_1 = 100 * correct_top1 / total_examples
    acc_5 = 100 * correct_top5 / total_examples
    print(f'Validation complete. {total_examples / (prev_time - start_time):>5.2f} img/s. '
          f'Acc@1 {acc_1:>7.3f}, Acc@5 {acc_5:>7.3f}')
    return dict(top1=acc_1, top5=acc_5)


def eval_forward(model, variables, images, labels):
    logits = model.apply(variables, images, mutable=False, training=False)
    top1_count, top5_count = correct_topk(logits, labels, topk=(1, 5))
    return top1_count, top5_count


def main():
    args = parser.parse_args()
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)

    if get_model_cfg(args.model) is not None:
        validate(args)
    else:
        models = list_models(pretrained=True)
        if args.model != 'all':
            models = fnmatch.filter(models, args.model)
        print('Validating: ', ', '.join(models))
        results = []
        for m in models:
            args.model = m
            res = validate(args)
            res.update(dict(model=m))
            results.append(res)
        print('Results:')
        for r in results:
            print(f"Model: {r['model']}, Top1: {r['top1']}, Top5: {r['top5']}")

if __name__ == '__main__':
    main()
