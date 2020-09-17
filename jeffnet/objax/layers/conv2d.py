import inspect
from typing import Callable, Iterable, Tuple, Optional, Union

from jax import numpy as jn, random as jr, lax

from objax import functional, random, util
from objax.module import ModuleList, Module
from objax.nn.init import kaiming_normal, xavier_normal
from objax.typing import JaxArray
from objax.variable import TrainVar, StateVar

from jeffnet.common.padding import get_like_padding


class Conv2d(Module):
    """Applies a 2D convolution on a 4D-input batch of shape (N,C,H,W)."""

    def __init__(self,
                 nin: int,
                 nout: int,
                 k: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int] = 1,
                 padding: Union[str, Tuple[int, int], int] = 0,
                 dilation: Union[Tuple[int, int], int] = 1,
                 groups: int = 1,
                 use_bias: bool = False,
                 w_init: Callable = kaiming_normal):
        """Creates a Conv2D module instance.

        Args:
            nin: number of channels of the input tensor.
            nout: number of channels of the output tensor.
            k: size of the convolution kernel, either tuple (height, width) or single number if they're the same.
            stride: convolution strides, either tuple (stride_y, stride_x) or single number if they're the same.
            dilation: spacing between kernel points (also known as astrous convolution),
                       either tuple (dilation_y, dilation_x) or single number if they're the same.
            groups: number of input and output channels group. When groups > 1 convolution operation is applied
                    individually for each group. nin and nout must both be divisible by groups.
            padding: padding of the input tensor, either Padding.SAME or Padding.VALID.
            use_bias: if True then convolution will have bias term.
            w_init: initializer for convolution kernel (a function that takes in a HWIO shape and returns a 4D matrix).
        """
        super().__init__()
        assert nin % groups == 0, 'in_chs should be divisible by groups'
        assert nout % groups == 0, 'out_chs should be divisible by groups'
        self.b = TrainVar(jn.zeros((nout, 1, 1))) if use_bias else None
        self.w = TrainVar(w_init((*util.to_tuple(k, 2), nin // groups, nout)))  # HWIO

        if isinstance(padding, str):
            if padding == 'LIKE':
                padding = get_like_padding(k, stride, dilation)
                padding = util.to_tuple(padding, 2)
                # padding = [(0, 0), padding, padding, (0, 0)]
                padding = [padding, padding]
        else:
            padding = util.to_tuple(padding, 2)
            # padding = [(0, 0), padding, padding, (0, 0)]
            padding = [padding, padding]
        self.padding = padding
        self.strides = util.to_tuple(stride, 2)
        self.dilations = util.to_tuple(dilation, 2)
        self.groups = groups

    def __call__(self, x: JaxArray) -> JaxArray:
        """Returns the results of applying the convolution to input x."""
        y = lax.conv_general_dilated(
            x, self.w.value, self.strides, self.padding,
            rhs_dilation=self.dilations, feature_group_count=self.groups,
            dimension_numbers=('NCHW', 'HWIO', 'NCHW'))
        if self.b:
            y += self.b.value
        return y

