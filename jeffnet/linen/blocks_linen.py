""" EfficientNet, MobileNetV3, etc Blocks

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Any, Callable, Union, Optional

from flax import linen as nn
import jax.numpy as jnp

from jeffnet.common.block_defs import *
from .layers import conv2d, batchnorm2d, drop_path, get_act_fn

ModuleDef = Any


class SqueezeExcite(nn.Module):
    num_features: int  # features at input to containing block
    block_features: int = None  # input feature count of containing block
    se_ratio: float = 0.25
    divisor: int = 1
    reduce_from_block: bool = True  # calc se reduction from containing block's input features

    conv_layer: ModuleDef = conv2d
    act_fn: Callable = nn.relu
    gate_fn: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x):
        x_se = x.mean((1, 2), keepdims=True)
        base_features = self.block_features if self.block_features and self.reduce_from_block else self.num_features
        reduce_features: int = make_divisible(base_features * self.se_ratio, self.divisor)
        x_se = self.conv_layer(reduce_features, 1, stride=1, bias=True, name='fc1')(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_layer(self.num_features, 1, stride=1, bias=True, name='fc2')(x_se)
        return x * self.gate_fn(x_se)


class ConvBnAct(nn.Module):
    out_features: int
    in_features: int = None  # note used, currently for generic args support
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    pad_type: str = 'LIKE'

    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        x = self.conv_layer(
            self.out_features, self.kernel_size, stride=self.stride,
            dilation=self.dilation, padding=self.pad_type, name='conv')(x)
        x = self.norm_layer(name='bn', training=training)(x)
        x = self.act_fn(x)
        return x


class DepthwiseSeparable(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    in_features: int
    out_features: int
    dw_kernel_size: int = 3
    pw_kernel_size: int = 1
    stride: int = 1
    dilation: int = 1
    pad_type: str = 'LIKE'
    noskip: bool = False
    pw_act: bool = False
    se_ratio: float = 0.
    drop_path_rate: float = 0.

    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    se_layer: ModuleDef = SqueezeExcite
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        shortcut = x

        x = self.conv_layer(
            self.in_features, self.dw_kernel_size, stride=self.stride, dilation=self.dilation,
            padding=self.pad_type, groups=self.in_features, name='conv_dw')(x)
        x = self.norm_layer(name='bn_dw', training=training)(x)
        x = self.act_fn(x)

        if self.se_layer is not None and self.se_ratio > 0:
            x = self.se_layer(num_features=self.in_features, se_ratio=self.se_ratio, name='se')(x)

        x = self.conv_layer(self.out_features, self.pw_kernel_size, padding=self.pad_type, name='conv_pw')(x)
        x = self.norm_layer(name='bn_pw')(x)
        if self.pw_act:
            x = self.act_fn(x)

        if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
            if self.drop_path_rate > 0.:
                x = drop_path(self.drop_path_rate, self.training)(x)
            x = x + shortcut
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    in_features: int
    out_features: int
    exp_kernel_size: int = 1
    dw_kernel_size: int = 3
    pw_kernel_size: int = 1
    stride: int = 1
    dilation: int = 1
    pad_type: str = 'LIKE'
    noskip: bool = False
    pw_act: bool = False
    exp_ratio: float = 1.0
    se_ratio: float = 0.
    drop_path_rate: float = 0.

    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    se_layer: ModuleDef = SqueezeExcite
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        shortcut = x

        features = make_divisible(self.in_features * self.exp_ratio)

        # Point-wise expansion
        if self.exp_ratio > 1.:
            x = self.conv_layer(features, self.exp_kernel_size, padding=self.pad_type, name='conv_exp')(x)
            x = self.norm_layer(name='bn_exp', training=training)(x)
            x = self.act_fn(x)

        x = self.conv_layer(
            features, self.dw_kernel_size, stride=self.stride, dilation=self.dilation,
            padding=self.pad_type, groups=features, name='conv_dw')(x)
        x = self.norm_layer(name='bn_dw', training=training)(x)
        x = self.act_fn(x)

        if self.se_layer is not None and self.se_ratio > 0:
            x = self.se_layer(
                num_features=features, block_features=self.in_features, se_ratio=self.se_ratio, name='se')(x)

        x = self.conv_layer(self.out_features, self.pw_kernel_size, padding=self.pad_type, name='conv_pw')(x)
        x = self.norm_layer(name='bn_pw')(x)

        if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x = x + shortcut
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""
    in_features: int
    out_features: int
    exp_kernel_size: int = 1
    dw_kernel_size: int = 3
    pw_kernel_size: int = 1
    stride: int = 1
    dilation: int = 1
    pad_type: str = 'LIKE'
    noskip: bool = False
    pw_act: bool = False
    exp_ratio: float = 1.0
    se_ratio: float = 0.
    drop_path_rate: float = 0.

    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    se_layer: ModuleDef = SqueezeExcite
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        shortcut = x

        features = make_divisible(self.in_features * self.exp_ratio)

        # Point-wise expansion
        x = self.conv_layer(features, self.exp_kernel_size, padding=self.pad_type, name='conv_exp')(x)
        x = self.norm_layer(name='bn_exp')(x)
        x = self.act_fn(x)

        if self.se_layer is not None and self.se_ratio > 0:
            x = self.se_layer(
                num_features=features, block_features=self.in_features, se_ratio=self.se_ratio, name='se')(x)

        x = self.conv_layer(self.out_features, self.pw_kernel_size, padding=self.pad_type, name='conv_pw')(x)
        x = self.norm_layer(name='bn_pw')(x)

        if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x = x + shortcut
        return x


def chan_to_features(kwargs):
    in_chs = kwargs.pop('in_chs', None)
    if in_chs is not None:
        kwargs['in_features'] = in_chs
    out_chs = kwargs.pop('out_chs', None)
    if out_chs is not None:
        kwargs['out_features'] = out_chs
    return kwargs


class BlockFactory:

    @staticmethod
    def CondConv(block_idx,**block_args):
        assert False, "Not currently impl"

    @staticmethod
    def InvertedResidual(block_idx, **block_args):
        block_args = chan_to_features(block_args)
        return InvertedResidual(**block_args, name=f'block{block_idx}')

    @staticmethod
    def DepthwiseSeparable(block_idx,**block_args):
        block_args = chan_to_features(block_args)
        return DepthwiseSeparable(**block_args, name=f'block{block_idx}')

    @staticmethod
    def EdgeResidual(block_idx,**block_args):
        block_args = chan_to_features(block_args)
        return EdgeResidual(**block_args, name=f'block{block_idx}')

    @staticmethod
    def ConvBnAct(block_idx,**block_args):
        block_args.pop('drop_path_rate', None)
        block_args.pop('se_layer', None)
        block_args = chan_to_features(block_args)
        return ConvBnAct(**block_args, name=f'block{block_idx}')

    @staticmethod
    def get_act_fn(act_fn: Union[str, Callable]):
        return get_act_fn(act_fn) if isinstance(act_fn, str) else act_fn
