""" Linear and Conv Layers

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from typing import Callable, Iterable, Tuple, Optional, Union

from jax import numpy as jnp, lax

from objax import functional, random, util
from objax.module import Module
from objax.nn.init import kaiming_normal, xavier_normal
from objax.typing import JaxArray
from objax.variable import TrainVar

from jeffnet.common.padding import get_like_padding


class Conv2d(Module):
    """Applies a 2D convolution on a 4D-input batch of shape (N,C,H,W)."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int] = 1,
                 padding: Union[str, Tuple[int, int], int] = 0,
                 dilation: Union[Tuple[int, int], int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 kernel_init: Callable = kaiming_normal,
                 bias_init: Callable = jnp.zeros,
                 ):
        """Creates a Conv2D module instance.

        Args:
            in_channels: number of channels of the input tensor.
            out_channels: number of channels of the output tensor.
            kernel_size: size of the convolution kernel, either tuple (height, width) or single number if they're the same.
            stride: convolution strides, either tuple (stride_y, stride_x) or single number if they're the same.
            dilation: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x) or single number if they're the same.
            groups: number of input and output channels group. When groups > 1 convolution operation is applied
                    individually for each group. nin and nout must both be divisible by groups.
            padding: padding of the input tensor, either Padding.SAME or Padding.VALID.
            bias: if True then convolution will have bias term.
            kernel_init: initializer for convolution kernel (a function that takes in a HWIO shape and returns a 4D matrix).
        """
        super().__init__()
        assert in_channels % groups == 0, 'in_chs should be divisible by groups'
        assert out_channels % groups == 0, 'out_chs should be divisible by groups'
        kernel_size = util.to_tuple(kernel_size, 2)
        self.weight = TrainVar(kernel_init((out_channels, in_channels // groups, *kernel_size)))  # OIHW
        self.bias = TrainVar(bias_init((out_channels,))) if bias else None
        self.strides = util.to_tuple(stride, 2)
        self.dilations = util.to_tuple(dilation, 2)
        if isinstance(padding, str):
            if padding == 'LIKE':
                padding = (
                    get_like_padding(kernel_size[0], self.strides[0], self.dilations[0]),
                    get_like_padding(kernel_size[1], self.strides[1], self.dilations[1]))
                padding = [padding, padding]
        else:
            padding = util.to_tuple(padding, 2)
            padding = [padding, padding]
        self.padding = padding
        self.groups = groups

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        y = lax.conv_general_dilated(
            x, self.weight.value, self.strides, self.padding,
            rhs_dilation=self.dilations, feature_group_count=self.groups,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
        if self.bias:
            y += self.bias.value.reshape((1, -1, 1, 1))
        return y


class Linear(Module):
    """Applies a linear transformation on an input batch."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            weight_init: Callable = xavier_normal,
            bias_init: Callable = jnp.zeros,
    ):
        """Creates a Linear module instance.

        Args:
            in_features: number of channels of the input tensor.
            out_features: number of channels of the output tensor.
            bias: if True then linear layer will have bias term.
            weight_init: weight initializer for linear layer (a function that takes in a IO shape and returns a 2D matrix).
        """
        super().__init__()
        self.weight = TrainVar(weight_init((out_features, in_features)))
        self.bias = TrainVar(bias_init(out_features)) if bias else None

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the linear transformation to input x."""
        y = jnp.dot(x, self.weight.value.transpose())
        if self.bias:
            y += self.bias.value
        return y