import os
import time
import argparse
import logging

import flax
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config
from jeffnet.common import load_state_dict, split_state_dict, correct_topk, AverageMeter
from jeffnet.linen import tf_efficientnet_b0, pt_efficientnet_b0

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')


def eval_forward(model, variables, images, labels):
    logits = model.apply(variables, images, mutable=False, training=False)
    top1_count, top5_count = correct_topk(logits, labels, topk=(1, 5))
    return top1_count, top5_count


def validate(args):
    rng = jax.random.PRNGKey(0)
    img_size = 224
    input_shape = (1, img_size, img_size, 3)
    model = pt_efficientnet_b0()

    state_dict = load_state_dict('./efficientnet_b0.npz', transpose=True)
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

    if os.path.splitext(args.data)[1] == '.tar' and os.path.isfile(args.data):
        dataset = DatasetTar(args.data)
    else:
        dataset = Dataset(args.data)

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
        correct_top1 += top1_count
        correct_top5 += top5_count
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
    return dict(top1=acc_1, top5=acc_5)


def main():
    args = parser.parse_args()
    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    jax.config.enable_omnistaging()
    validate(args)


if __name__ == '__main__':
    main()
