import time
import argparse

import jax
from absl import logging

import objax
import jeffnet.data.tf_imagenet_data as imagenet_data
from jeffnet.common import load_state_dict, correct_topk, AverageMeter
from jeffnet.objax import create_model


def eval_forward(model, images, labels):
    logits = model(images, training=False)
    top1_count, top5_count = correct_topk(logits, labels, topk=(1, 5))
    return top1_count, top5_count


def validate(args):
    model = create_model('tf_efficientnet_b0')
    model_vars = model.vars()
    jax_state_dict = load_state_dict('./tf_efficientnet_b0_ns.npz')

    # FIXME hack, assuming alignment, currently enforced by my layer customizations
    model_vars.assign(jax_state_dict.values())

    eval_step = objax.Jit(
        lambda images, labels: eval_forward(model, images, labels),
        model.vars())

    """Runs evaluation and returns top-1 accuracy."""
    test_ds, num_batches = imagenet_data.load(
        imagenet_data.Split.TEST,
        is_training=False,
        batch_dims=[args.batch_size],
        chw=True,
        #mean=(255 * 0.5,) * 3,
        #std=(255 * 0.5,) *3,
        tfds_data_dir=args.data)

    batch_time = AverageMeter()
    correct_top1, correct_top5 = 0, 0
    total_examples = 0
    start_time = prev_time = time.time()
    for batch_index, batch in enumerate(test_ds):
        images, labels = batch['images'], batch['labels']
        top1_count, top5_count = eval_step(images, labels)
        correct_top1 += top1_count
        correct_top5 += top5_count
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


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')


def main():
    args = parser.parse_args()
    logging.set_verbosity(logging.ERROR)
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)

    validate(args)


if __name__ == '__main__':
    main()
