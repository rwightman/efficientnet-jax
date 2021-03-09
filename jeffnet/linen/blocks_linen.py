""" EfficientNet, MobileNetV3, etc Blocks for Flax Linen

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Any, Callable, Union, Optional

from flax import linen as nn
import jax.numpy as jnp

from jeffnet.common.block_utils import *
from .layers import conv2d, batchnorm2d, get_act_fn, linear, DropPath, Dropout, MixedConv

Dtype = Any
ModuleDef = Any


def create_conv(features, kernel_size, conv_layer=None, **kwargs):
    """ Select a convolution implementation based on arguments
    Creates and returns one of Conv, MixedConv, or CondConv (TODO)
    """
    conv_layer = conv2d if conv_layer is None else conv_layer
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        assert 'groups' not in kwargs  # MixedConv groups are defined by kernel list
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv(features, kernel_size, conv_layer=conv_layer, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = features if depthwise else kwargs.pop('groups', 1)
        # if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
        #     m = CondConv(features, kernel_size, groups=groups, conv_layer=conv_layer, **kwargs)
        # else:
        m = conv_layer(features, kernel_size, groups=groups, **kwargs)
    return m


class SqueezeExcite(nn.Module):
    num_features: int  # features at input to containing block
    block_features: int = None  # input feature count of containing block
    se_ratio: float = 0.25
    divisor: int = 1
    reduce_from_block: bool = True  # calc se reduction from containing block's input features

    dtype: Dtype = jnp.float32
    conv_layer: ModuleDef = conv2d
    act_fn: Callable = nn.relu
    bound_act_fn: Optional[Callable] = None  # override the passed in act_fn from parent with a bound fn
    gate_fn: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x):
        x_se = jnp.asarray(x, jnp.float32)
        x_se = x_se.mean((1, 2), keepdims=True)
        x_se = jnp.asarray(x_se, self.dtype)
        base_features = self.block_features if self.block_features and self.reduce_from_block else self.num_features
        reduce_features: int = make_divisible(base_features * self.se_ratio, self.divisor)
        act_fn = self.bound_act_fn if self.bound_act_fn is not None else self.act_fn
        x_se = self.conv_layer(reduce_features, 1, stride=1, bias=True, name='reduce')(x_se)
        x_se = act_fn(x_se)
        x_se = self.conv_layer(self.num_features, 1, stride=1, bias=True, name='expand')(x_se)
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
        x = self.norm_layer(name='bn')(x, training=training)
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

        x = create_conv(
            self.in_features, self.dw_kernel_size, stride=self.stride, dilation=self.dilation,
            padding=self.pad_type, depthwise=True, conv_layer=self.conv_layer, name='conv_dw')(x)
        x = self.norm_layer(name='bn_dw')(x, training=training)
        x = self.act_fn(x)

        if self.se_layer is not None and self.se_ratio > 0:
            x = self.se_layer(
                num_features=self.in_features, se_ratio=self.se_ratio,
                conv_layer=self.conv_layer, act_fn=self.act_fn, name='se')(x)

        x = create_conv(
            self.out_features, self.pw_kernel_size, padding=self.pad_type,
            conv_layer=self.conv_layer, name='conv_pw')(x)
        x = self.norm_layer(name='bn_pw')(x, training=training)
        if self.pw_act:
            x = self.act_fn(x)

        if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
            x = DropPath(self.drop_path_rate)(x, training=training)
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
            x = create_conv(
                features, self.exp_kernel_size, padding=self.pad_type, conv_layer=self.conv_layer, name='conv_exp')(x)
            x = self.norm_layer(name='bn_exp')(x, training=training)
            x = self.act_fn(x)

        x = create_conv(
            features, self.dw_kernel_size, stride=self.stride, dilation=self.dilation,
            padding=self.pad_type, depthwise=True, conv_layer=self.conv_layer, name='conv_dw')(x)
        x = self.norm_layer(name='bn_dw')(x, training=training)
        x = self.act_fn(x)

        if self.se_layer is not None and self.se_ratio > 0:
            x = self.se_layer(
                num_features=features, block_features=self.in_features, se_ratio=self.se_ratio,
                conv_layer=self.conv_layer, act_fn=self.act_fn, name='se')(x)

        x = create_conv(
            self.out_features, self.pw_kernel_size, padding=self.pad_type,
            conv_layer=self.conv_layer, name='conv_pwl')(x)
        x = self.norm_layer(name='bn_pwl')(x, training=training)

        if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
            x = DropPath(self.drop_path_rate)(x, training=training)
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

        # Unlike other blocks, not using the arch def for in_features since it's not reliable for Edge
        features = make_divisible(x.shape[-1] * self.exp_ratio)

        # Point-wise expansion
        x = create_conv(
            features, self.exp_kernel_size, padding=self.pad_type, conv_layer=self.conv_layer, name='conv_exp')(x)
        x = self.norm_layer(name='bn_exp')(x, training=training)
        x = self.act_fn(x)

        if self.se_layer is not None and self.se_ratio > 0:
            x = self.se_layer(
                num_features=features, block_features=self.in_features, se_ratio=self.se_ratio,
                conv_layer=self.conv_layer, act_fn=self.act_fn, name='se')(x)

        x = create_conv(
            self.out_features, self.pw_kernel_size, stride=self.stride, dilation=self.dilation,
            padding=self.pad_type, conv_layer=self.conv_layer, name='conv_pwl')(x)
        x = self.norm_layer(name='bn_pwl')(x, training=training)

        if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
            x = DropPath(self.drop_path_rate)(x, training=training)
            x = x + shortcut
        return x


class Head(nn.Module):
    """ Standard Head from EfficientNet, MixNet, MNasNet, MobileNetV2, etc. """
    num_features: int
    num_classes: int = 1000
    global_pool: str = 'avg'  # FIXME support diff pooling
    drop_rate: float = 0.

    dtype: Dtype = jnp.float32
    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    linear_layer: ModuleDef = linear
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        x = self.conv_layer(self.num_features, 1, name='conv_pw')(x)
        x = self.norm_layer(name='bn')(x, training=training)
        x = self.act_fn(x)
        if self.global_pool == 'avg':
            x = jnp.asarray(x, jnp.float32)
            x = x.mean((1, 2))
            x = jnp.asarray(x, self.dtype)
        x = Dropout(rate=self.drop_rate)(x, training=training)
        if self.num_classes > 0:
            x = self.linear_layer(self.num_classes, bias=True, name='classifier')(x)
        return x


class EfficientHead(nn.Module):
    """ EfficientHead for MobileNetV3. """
    num_features: int
    num_classes: int = 1000
    global_pool: str = 'avg'  # FIXME support diff pooling
    drop_rate: float = 0.

    dtype: Dtype = jnp.float32
    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = None  # ignored, to keep calling code clean
    linear_layer: ModuleDef = linear
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        if self.global_pool == 'avg':
            x = jnp.asarray(x, jnp.float32)
            x = x.mean((1, 2), keepdims=True)
            x = jnp.asarray(x, self.dtype)
        x = self.conv_layer(self.num_features, 1, bias=True, name='conv_pw')(x)
        x = self.act_fn(x)
        x = Dropout(rate=self.drop_rate)(x, training=training)
        if self.num_classes > 0:
            x = self.linear_layer(self.num_classes, bias=True, name='classifier')(x)
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
    def CondConv(stage_idx, block_idx, **block_args):
        assert False, "Not currently impl"

    @staticmethod
    def InvertedResidual(stage_idx, block_idx, **block_args):
        block_args = chan_to_features(block_args)
        return InvertedResidual(**block_args, name=f'blocks_{stage_idx}_{block_idx}')

    @staticmethod
    def DepthwiseSeparable(stage_idx, block_idx, **block_args):
        block_args = chan_to_features(block_args)
        return DepthwiseSeparable(**block_args, name=f'blocks_{stage_idx}_{block_idx}')

    @staticmethod
    def EdgeResidual(stage_idx, block_idx, **block_args):
        block_args = chan_to_features(block_args)
        block_args.pop('fake_in_chs')  # not needed for Linen @nn.compact defs, we can access the real in_features
        return EdgeResidual(**block_args, name=f'blocks_{stage_idx}_{block_idx}')

    @staticmethod
    def ConvBnAct(stage_idx, block_idx, **block_args):
        block_args.pop('drop_path_rate', None)
        block_args.pop('se_layer', None)
        block_args = chan_to_features(block_args)
        return ConvBnAct(**block_args, name=f'blocks_{stage_idx}_{block_idx}')

    @staticmethod
    def get_act_fn(act_fn: Union[str, Callable]):
        return get_act_fn(act_fn) if isinstance(act_fn, str) else act_fn
