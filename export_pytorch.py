import os
import time
import argparse

import torch
import timm
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--model', '-m', metavar='MODEL', default='efficientnet_b0',
                    help='model architecture (default: efficientnet_b0)')
parser.add_argument('--output', '-o', metavar='MODEL', default=None,
                    help='')


def main():
    args = parser.parse_args()

    m = timm.create_model(args.model, pretrained=True)
    d = dict(m.named_modules())

    data = {}
    names = []
    types = []
    for k, v in m.state_dict().items():
        if 'num_batches' in k:
            continue
        data[str(len(data))] = v.numpy()
        names.append(k)
        parent_module = '.'.join(k.split('.')[:-1])
        type_str = ''
        if parent_module in d:
            type_str = type(d[parent_module]).__name__
        types.append(type_str)

    if args.output is None:
        output_file = args.model + '.npz'
    else:
        output_file = args.output

    np.savez(output_file, names=np.array(names), types=types, **data)


if __name__ == '__main__':
    main()
