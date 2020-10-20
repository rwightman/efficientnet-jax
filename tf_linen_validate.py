import time
import argparse

import flax
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp

import jeffnet.data.tf_imagenet_data as imagenet_data
from jeffnet.common import load_state_dict, split_state_dict, correct_topk, AverageMeter
from jeffnet.linen import tf_efficientnet_b0, pt_efficientnet_b0


def eval_forward(model, variables, images, labels):
    logits = model.apply(variables, images, mutable=False, training=False)
    top1_count, top5_count = correct_topk(logits, labels, topk=(1, 5))
    return top1_count, top5_count


def validate(args):
    rng = jax.random.PRNGKey(0)
    img_size = 224
    input_shape = (1, img_size, img_size, 3)
    model = tf_efficientnet_b0()

    state_dict = load_state_dict('./tf_efficientnet_b0.npz', transpose=True)
    source_params, source_state = split_state_dict(state_dict)

    def _init_model():
        var_init = model.init({'params': rng}, jnp.ones(input_shape, jnp.float32), training=False)
        # FIXME this is all a rather large hack
        var_unfrozen = unfreeze(var_init)
        flat_params = flatten_dict(var_unfrozen['params'])
        flat_state = flatten_dict(var_unfrozen['batch_stats'])
        for ok, ov, sv in zip(flat_params.keys(), flat_params.values(), source_params.values()):
            assert ov.shape == sv.shape
            flat_params[ok] = sv
        for ok, ov, sv in zip(flat_state.keys(), flat_state.values(), source_state.values()):
            assert ov.shape == sv.shape
            flat_state[ok] = sv
        params = unflatten_dict(flat_params)
        batch_stats = unflatten_dict(flat_state)
        return dict(params=params, batch_stats=batch_stats)
    variables = _init_model()

    no_jit = False
    if no_jit:
        eval_step = lambda images, labels: eval_forward(model, variables, images, labels)
    else:
        eval_step = jax.jit(lambda images, labels: eval_forward(model, variables, images, labels))

    """Runs evaluation and returns top-1 accuracy."""
    test_ds, num_batches = imagenet_data.load(
        imagenet_data.Split.TEST,
        is_training=False,
        batch_dims=[args.batch_size],
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
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)

    validate(args)


if __name__ == '__main__':
    main()
