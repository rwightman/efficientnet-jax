""" timm -> generic npz export script
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import shutil
import tempfile
import hashlib

import numpy as np
import torch
import timm
from jeffnet.common import list_models


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', '-m', metavar='MODEL', default='efficientnet_b0',
                    help='model architecture (default: efficientnet_b0)')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='')


def remap_module(module_type, k, v):
    # remappings for layers I've renamed since PyTorch impl
    # objax/flax specific naming reqs are handled on load
    if module_type == "ConvBnAct":
        k = k.replace('bn1.', 'bn.')
    elif module_type == "InvertedResidual":
        k = k.replace('conv_pw.', 'conv_exp.')
        k = k.replace('bn1.', 'bn_exp.')
        k = k.replace('bn2.', 'bn_dw.')
        k = k.replace('bn3.', 'bn_pwl.')
    elif module_type == "EdgeResidual":
        k = k.replace('bn1.', 'bn_exp.')
        k = k.replace('bn2.', 'bn_pwl.')
    elif module_type == 'DepthwiseSeparableConv':
        k = k.replace('bn1.', 'bn_dw.')
        k = k.replace('bn2.', 'bn_pw.')
    elif module_type == 'SqueezeExcite':
        k = k.replace('conv_reduce.', 'reduce.')
        k = k.replace('conv_expand.', 'expand.')
    elif module_type == 'EdgeResidual':
        k = k.replace('bn1.', 'bn_exp.')
        k = k.replace('bn3.', 'bn_pwl.')
    elif module_type == 'EfficientNet':
        k = k.replace('conv_stem.', 'stem.conv.')
        k = k.replace('bn1.', 'stem.bn.')
        k = k.replace('conv_head.', 'head.conv_pw.')
        k = k.replace('bn2.', 'head.bn.')
        k = k.replace('classifier.', 'head.classifier.')
    elif module_type == "MobileNetV3":
        k = k.replace('conv_stem.', 'stem.conv.')
        k = k.replace('bn1.', 'stem.bn.')
        k = k.replace('conv_head.', 'head.conv_pw.')
        k = k.replace('bn2.', 'head.bn.')
        k = k.replace('classifier.', 'head.classifier.')
    return k, v


def export_model(model_name, output_dir=''):
    timm_model_name = model_name.replace('pt_', '')
    m = timm.create_model(timm_model_name, pretrained=True)
    d = dict(m.named_modules())

    data = {}
    names = []
    types = []
    for k, v in m.state_dict().items():
        if 'num_batches' in k:
            continue

        k_split = k.split('.')
        layer_name = '.'.join(k_split[:-1])
        parent_name = '.'.join(k_split[:-2])
        parent_module = d[parent_name]
        parent_type = type(parent_module).__name__
        if 'MixedConv' in parent_type:
            # need to step back another level in hierarchy to get parent block
            parent_name = '.'.join(k_split[:-3])
            parent_module = d[parent_name]
            parent_type = type(parent_module).__name__
        k, v = remap_module(parent_type, k, v)

        type_str = ''
        if layer_name in d:
            type_str = type(d[layer_name]).__name__
            if type_str == 'Conv2dSame':
                type_str = 'Conv2d'
        types.append(type_str)

        print(k, type_str, v.shape)
        data[str(len(data))] = v.numpy()
        names.append(k)

    # write as npz
    tempf = tempfile.NamedTemporaryFile(delete=False, dir='./')
    np.savez(tempf, names=np.array(names), types=types, **data)
    tempf.close()

    # verify by reading and hashing
    with open(tempf.name, 'rb') as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()

    # move to proper name / location
    if output_dir:
        assert os.path.isdir(output_dir)
    else:
        output_dir = './'
    final_filename = '-'.join([model_name, sha_hash[:8]]) + '.npz'
    shutil.move(tempf.name, os.path.join(output_dir, final_filename))


def main():
    args = parser.parse_args()

    all_models = list_models(pretrained=True)
    if args.model == 'all':
        for model_name in all_models:
            export_model(model_name, args.output)
    else:
        export_model(args.model, args.output)


if __name__ == '__main__':
    main()
