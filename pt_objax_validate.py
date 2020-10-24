import os
import time
import argparse

import objax
import jax

from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from jeffnet.common import correct_topk, AverageMeter, load_state_dict
from jeffnet.objax import create_model


def validate(args):
    model = create_model('pt_efficientnet_b0')

    model_vars = model.vars()
    jax_state_dict = load_state_dict('./efficientnet_b0.npz')
    model_vars.assign(jax_state_dict.values())

    #model_ut = lambda x: model(x, training=False)
    model_ut = objax.Jit(lambda x: model(x, training=False), model.vars())

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
        images = images.numpy()
        labels = labels.numpy()

        logits = model_ut(images)
        correct = correct_topk(logits, labels, topk=(1, 5))
        correct_top1 += correct[0]
        correct_top5 += correct[1]
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
