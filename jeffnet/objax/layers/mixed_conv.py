""" Mixed Grouped Convolution

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from objax.module import ModuleList
from jax import numpy as jnp

from .linear import Conv2d


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv(ModuleList):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='',
                 dilation=1, depthwise=False, conv_layer=None, **kwargs):
        super(MixedConv, self).__init__()
        conv_layer = Conv2d if conv_layer is None else conv_layer
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.append(
                conv_layer(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = jnp.array(in_splits).cumsum()[:-1]

    def __call__(self, x):
        x_split = jnp.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self)]
        x = jnp.concatenate(x_out, axis=1)
        return x
