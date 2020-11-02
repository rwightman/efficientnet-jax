from typing import Any, Callable, Sequence, Optional, Tuple, List, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from .linear import conv2d

ModuleDef = Any


def _split_channels(num_feat, num_groups):
    split = [num_feat // num_groups for _ in range(num_groups)]
    split[0] += num_feat - sum(split)
    return split


def _to_list(x):
    if isinstance(x, int):
        return [x]
    return x


class MixedConv(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
        https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    features: int
    kernel_size: Union[List[int], int] = 3
    dilation: int = 1
    stride: int = 1
    padding: Union[str, Tuple[int, int]] = 0
    depthwise: bool = False
    bias: bool = False

    conv_layer: ModuleDef = conv2d

    @nn.compact
    def __call__(self, x):
        num_groups = len(_to_list(self.kernel_size))
        # NOTE need to use np not jnp for calculating splits otherwise abstract value error
        in_splits = np.array(_split_channels(x.shape[-1], num_groups)).cumsum()[:-1]
        out_splits = _split_channels(self.features, num_groups)
        x_split = jnp.split(x, in_splits, axis=3)
        x_out = [self.conv_layer(
            feat, kernel_size=k, stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=feat if self.depthwise else 1, bias=self.bias, name=f'{idx}')(x_split[idx])
                 for idx, (k, feat) in enumerate(zip(self.kernel_size, out_splits))]
        x = jnp.concatenate(x_out, axis=3)
        return x
