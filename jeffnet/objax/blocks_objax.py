""" EfficientNet, MobileNetV3, etc Blocks for Objax

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Union, Callable, Optional

import objax.nn as nn
import objax.functional as F
from objax import Module, nn as nn, functional as F
from objax.typing import JaxArray

from jeffnet.common.block_utils import *
from .layers import Conv2d, MixedConv, BatchNorm2d, drop_path, get_act_fn, Linear


def create_conv(in_channels, out_channels, kernel_size, conv_layer=None, **kwargs):
    """ Select a convolution implementation based on arguments
    Creates and returns one of Conv, MixedConv, or CondConv (TODO)
    """
    conv_layer = Conv2d if conv_layer is None else conv_layer
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        assert 'groups' not in kwargs  # MixedConv groups are defined by kernel list
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv(in_channels, out_channels, kernel_size, conv_layer=conv_layer, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        # if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
        #     m = CondConv(in_channels, out_channels, kernel_size, groups=groups, conv_layer=conv_layer, **kwargs)
        # else:
        m = conv_layer(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


class SqueezeExcite(Module):
    def __init__(self, in_chs, se_ratio=0.25, block_chs=None, reduce_from_block=True,
                 conv_layer=Conv2d, act_fn=F.relu, bound_act_fn=None, gate_fn=F.sigmoid, divisor=1):
        super(SqueezeExcite, self).__init__()
        base_features = block_chs if block_chs and reduce_from_block else in_chs
        reduced_chs = make_divisible(base_features * se_ratio, divisor)
        self.reduce = conv_layer(in_chs, reduced_chs, 1, bias=True)
        self.act_fn = bound_act_fn if bound_act_fn is not None else act_fn
        self.expand = conv_layer(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def __call__(self, x):
        x_se = x.mean((2, 3), keepdims=True)
        x_se = self.reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.expand(x_se)
        return x * self.gate_fn(x_se)


class ConvBnAct(Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='LIKE', conv_layer=Conv2d, norm_layer=BatchNorm2d, act_fn=F.relu):
        super(ConvBnAct, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn = norm_layer(out_chs)
        self.act_fn = act_fn

    def __call__(self, x, training: bool):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act_fn(x)
        return x


class DepthwiseSeparable(Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='LIKE', noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0.,
                 conv_layer=Conv2d, norm_layer=BatchNorm2d, se_layer=None, act_fn=F.relu, drop_path_rate=0.):
        super(DepthwiseSeparable, self).__init__()
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, conv_layer=conv_layer)
        self.bn_dw = norm_layer(in_chs)
        self.act_fn = act_fn

        # Squeeze-and-excitation
        self.se = None
        if se_layer is not None and se_ratio > 0.:
            self.se = se_layer(in_chs, se_ratio=se_ratio, act_fn=act_fn)

        self.conv_pw = create_conv(in_chs, out_chs, pw_kernel_size, padding=pad_type, conv_layer=conv_layer)
        self.bn_pw = norm_layer(out_chs)

    def __call__(self, x, training: bool):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn_dw(x, training=training)
        x = self.act_fn(x)

        if self.se is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn_pw(x, training=training)
        if self.has_pw_act:
            x = self.act_fn(x)

        if self.has_residual:
            if training:
                x = drop_path(x, drop_prob=self.drop_path_rate)
            x += shortcut
        return x


class InvertedResidual(Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='LIKE', noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
                 conv_layer=Conv2d, norm_layer=BatchNorm2d, se_layer=None, act_fn=F.relu, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_exp = create_conv(in_chs, mid_chs, exp_kernel_size, padding=pad_type, conv_layer=conv_layer)
        self.bn_exp = norm_layer(mid_chs)
        self.act_fn = act_fn

        # Depth-wise convolution
        self.conv_dw = create_conv(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, conv_layer=conv_layer)
        self.bn_dw = norm_layer(mid_chs)

        # Squeeze-and-excitation
        self.se = None
        if se_layer is not None and se_ratio > 0.:
            self.se = se_layer(mid_chs, block_chs=in_chs, se_ratio=se_ratio, act_fn=act_fn)

        # Point-wise linear projection
        self.conv_pwl = create_conv(mid_chs, out_chs, pw_kernel_size, padding=pad_type, conv_layer=conv_layer)
        self.bn_pwl = norm_layer(out_chs)

    def __call__(self, x, training: bool):
        shortcut = x

        # Point-wise expansion
        x = self.conv_exp(x)
        x = self.bn_exp(x, training=training)
        x = self.act_fn(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn_dw(x, training=training)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn_pwl(x, training=training)

        if self.has_residual:
            if training:
                x = drop_path(x, drop_prob=self.drop_path_rate)
            x += shortcut

        return x


class EdgeResidual(Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0, fake_in_chs=0,
                 stride=1, dilation=1, pad_type='LIKE', noskip=False, pw_kernel_size=1,
                 se_ratio=0., conv_layer=Conv2d, norm_layer=BatchNorm2d, se_layer=None, act_fn=F.relu,
                 drop_path_rate=0.):
        super(EdgeResidual, self).__init__()
        _in_chs = fake_in_chs if fake_in_chs > 0 else in_chs  # mismatch in arch specs and actual in chs
        mid_chs = make_divisible(_in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Expansion convolution
        self.conv_exp = create_conv(in_chs, mid_chs, exp_kernel_size, padding=pad_type, conv_layer=conv_layer)
        self.bn_exp = norm_layer(mid_chs)
        self.act_fn = act_fn

        # Squeeze-and-excitation
        self.se = None
        if se_layer is not None and se_ratio > 0.:
            self.se = se_layer(mid_chs, block_chs=in_chs, se_ratio=se_ratio, act_fn=act_fn)

        # Point-wise linear projection
        self.conv_pwl = create_conv(
            mid_chs, out_chs, pw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, conv_layer=conv_layer)
        self.bn_pwl = norm_layer(out_chs)

    def __call__(self, x, training: bool):
        shortcut = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn_exp(x, training=training)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn_pwl(x, training=training)

        if self.has_residual:
            if training:
                x = drop_path(x, drop_prob=self.drop_path_rate)
            x = x + shortcut

        return x


class EfficientHead(Module):
    """ EfficientHead from MobileNetV3 """
    def __init__(self, in_chs: int, num_features: int, num_classes: int = 1000, global_pool: str = 'avg',
                 act_fn='relu', conv_layer=Conv2d, norm_layer=None):
        self.global_pool = global_pool  # FIXME support diff pooling

        self.conv_pw = conv_layer(in_chs, num_features, 1, bias=True)
        self.act_fn = act_fn
        if num_classes > 0:
            self.classifier = Linear(num_features, num_classes, bias=True)
        else:
            self.classifier = None

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        if self.global_pool == 'avg':
            x = x.mean((2, 3), keepdims=True)
        x = self.conv_pw(x).reshape(x.shape[0], -1)
        x = self.act_fn(x)
        x = self.classifier(x)
        return x


class Head(Module):
    """ Standard Head from EfficientNet, MixNet, MNasNet, MobileNetV2, etc. """
    def __init__(self, in_chs: int, num_features: int, num_classes: int = 1000, global_pool: str = 'avg',
                 act_fn=F.relu, conv_layer=Conv2d, norm_layer=BatchNorm2d):
        self.global_pool = global_pool  # FIXME support diff pooling

        self.conv_pw = conv_layer(in_chs, num_features, 1)
        self.bn = norm_layer(num_features)
        self.act_fn = act_fn
        if num_classes > 0:
            self.classifier = Linear(num_features, num_classes, bias=True)
        else:
            self.classifier = None

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        x = self.conv_pw(x)
        x = self.bn(x, training=training)
        x = self.act_fn(x)
        if self.global_pool == 'avg':
            x = x.mean((2, 3))
        if self.classifier is not None:
            x = self.classifier(x)
        return x


class BlockFactory:

    @staticmethod
    def CondConv(stage_idx, block_idx, **block_args):
        assert False, "Not currently impl"

    @staticmethod
    def InvertedResidual(stage_idx, block_idx, **block_args):
        return InvertedResidual(**block_args)

    @staticmethod
    def DepthwiseSeparable(stage_idx, block_idx, **block_args):
        return DepthwiseSeparable(**block_args)

    @staticmethod
    def EdgeResidual(stage_idx, block_idx, **block_args):
        return EdgeResidual(**block_args)

    @staticmethod
    def ConvBnAct(stage_idx, block_idx, **block_args):
        block_args.pop('drop_path_rate', None)
        block_args.pop('se_layer', None)
        return ConvBnAct(**block_args)

    @staticmethod
    def get_act_fn(act_fn: Union[str, Callable]):
        return get_act_fn(act_fn) if isinstance(act_fn, str) else act_fn
