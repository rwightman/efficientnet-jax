""" EfficientNet, MobileNetV3, etc Blocks

Hacked together by / Copyright 2020 Ross Wightman
"""
import objax.nn as nn
from objax import Module

from jeffnet.common.block_defs import *
from jeffnet.common.activations import get_act_fn

from .layers import Conv2d, drop_path


class SqueezeExcite(Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_fn='relu', gate_fn='sigmoid', divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.fc1 = Conv2d(in_chs, reduced_chs, 1, use_bias=True)
        self.act_fn = get_act_fn(act_fn)
        self.fc2 = Conv2d(reduced_chs, in_chs, 1, use_bias=True)
        self.gate_fn = get_act_fn(gate_fn)

    def __call__(self, x):
        x_se = x.mean((2, 3), keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate_fn(x_se)


class ConvBnAct(Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='LIKE', act_fn='relu',
                 norm_layer=nn.BatchNorm2D, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = Conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn = norm_layer(out_chs, **norm_kwargs)
        self.act_fn = get_act_fn(act_fn)

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
                 stride=1, dilation=1, pad_type='LIKE', act_fn='relu', noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2D, norm_kwargs=None, drop_path_rate=0.):
        super(DepthwiseSeparable, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = Conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=in_chs)
        self.bn_dw = norm_layer(in_chs, **norm_kwargs)
        self.act_fn = get_act_fn(act_fn)

        # Squeeze-and-excitation
        self.se = None
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_fn)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)

        self.conv_pw = Conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn_pw = norm_layer(out_chs, **norm_kwargs)

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
                 stride=1, dilation=1, pad_type='LIKE', act_fn='relu', noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2D, norm_kwargs=None,
                 conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_exp = Conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn_exp = norm_layer(mid_chs, **norm_kwargs)
        self.act_fn = get_act_fn(act_fn)

        # Depth-wise convolution
        self.conv_dw = Conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, groups=mid_chs, **conv_kwargs)
        self.bn_dw = norm_layer(mid_chs, **norm_kwargs)

        # Squeeze-and-excitation
        self.se = None
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_fn)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)

        # Point-wise linear projection
        self.conv_pw = Conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn_pw = norm_layer(out_chs, **norm_kwargs)

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
        x = self.conv_pw(x)
        x = self.bn_pw(x, training=training)

        if self.has_residual:
            if training:
                x = drop_path(x, drop_prob=self.drop_path_rate)
            x += shortcut

        return x


class EdgeResidual(Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0, fake_in_chs=0,
                 stride=1, dilation=1, padding='LIKE', act_fn='relu', noskip=False, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2D, norm_kwargs=None,
                 drop_path_rate=0.):
        super(EdgeResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        if fake_in_chs > 0:
            mid_chs = make_divisible(fake_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Expansion convolution
        self.conv_exp = Conv2d(in_chs, mid_chs, exp_kernel_size, padding=padding)
        self.bn_exp = norm_layer(mid_chs, **norm_kwargs)
        self.act_fn = get_act_fn(act_fn)

        # Squeeze-and-excitation
        self.se = None
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_fn)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)

        # Point-wise linear projection
        self.conv_pw = Conv2d(
            mid_chs, out_chs, pw_kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.bn_pw = norm_layer(out_chs, **norm_kwargs)

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
        x = self.conv_pw(x)
        x = self.bn_pw(x, training=training)

        if self.has_residual:
            if training:
                x = drop_path(x, drop_prob=self.drop_path_rate)
            x = x + shortcut

        return x


class BlockFactory:

    @staticmethod
    def CondConv(**block_args):
        assert False, "Not currently impl"

    @staticmethod
    def InvertedResidual(**block_args):
        return InvertedResidual(**block_args)

    @staticmethod
    def DepthwiseSeparable(**block_args):
        return DepthwiseSeparable(**block_args)

    @staticmethod
    def EdgeResidual(**block_args):
        return EdgeResidual(**block_args)

    @staticmethod
    def ConvBnAct(**block_args):
        block_args.pop('drop_path_rate', None)
        block_args.pop('se_kwargs', None)
        return ConvBnAct(**block_args)