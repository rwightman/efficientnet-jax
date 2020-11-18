""" ImageNet Validation Script
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import time
import argparse
import fnmatch

import jax
import flax
import tensorflow_datasets as tfds

import jeffnet.data.tf_imagenet_data as imagenet_data
import jeffnet.data.tf_input_pipeline as input_pipeline
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
parser.add_argument('--half-precision', action='store_true', default=False,
                    help='Evaluate in half (mixed) precision')


def validate(args):
    rng = jax.random.PRNGKey(0)
    platform = jax.local_devices()[0].platform
    if args.half_precision:
        if platform == 'tpu':
            model_dtype = jax.numpy.bfloat16
        else:
            model_dtype = jax.numpy.float16
    else:
        model_dtype = jax.numpy.float32

    model, variables = create_model(args.model, pretrained=True, dtype=model_dtype, rng=rng)
    print(f'Created {args.model} model. Validating...')

    if args.no_jit:
        eval_step = lambda images, labels: eval_forward(model, variables, images, labels)
    else:
        eval_step = jax.jit(lambda images, labels: eval_forward(model, variables, images, labels))

    """Runs evaluation and returns top-1 accuracy."""
    image_size = model.default_cfg['input_size'][-1]

    eval_iter, num_batches = create_eval_iter(
        args.data, args.batch_size, image_size, args.half_precision,
        mean=tuple([x * 255 for x in model.default_cfg['mean']]),
        std=tuple([x * 255 for x in model.default_cfg['std']]),
        interpolation=model.default_cfg['interpolation'],
    )

    batch_time = AverageMeter()
    correct_top1, correct_top5 = 0, 0
    total_examples = 0
    start_time = prev_time = time.time()
    for batch_index, batch in enumerate(eval_iter):
        images, labels = batch['image'], batch['label']
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


def prepare_tf_data(xs):
    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access
        return x
    return jax.tree_map(_prepare, xs)


def create_eval_iter(data_dir, batch_size, image_size, half_precision=False,
                     mean=None, std=None, interpolation='bicubic'):
    dataset_builder = tfds.builder('imagenet2012:5.*.*', data_dir=data_dir)
    assert dataset_builder.info.splits['validation'].num_examples % batch_size == 0
    num_batches = dataset_builder.info.splits['validation'].num_examples // batch_size
    # FIXME currently forcing no host/device-split, I haven't added distributed eval support
    ds = input_pipeline.create_split(
        dataset_builder, batch_size, train=False, half_precision=half_precision,
        image_size=image_size, mean=mean, std=std, interpolation=interpolation, no_split=True, no_repeat=True)
    it = map(prepare_tf_data, ds)
    return it, num_batches


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
        if not models:
            print(f'ERROR: No models found to validate with pattern {args.model}.')
            exit(1)

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
