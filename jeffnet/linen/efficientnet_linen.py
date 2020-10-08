from typing import Any, Callable, Sequence, Dict
from functools import partial
from flax import linen as nn
from flax.linen import Module

from jeffnet.common import round_channels, get_act_fn, resolve_bn_args,\
    decode_arch_def, EfficientNetBuilder

from .layers import conv2d, linear, batchnorm2d, drop_path
from .blocks_linen import ConvBnAct, SqueezeExcite, BlockFactory

ModuleDef = Any

_DEBUG = True


# class EfficientHead(Module):
#     """ EfficientHead from MobileNetV3 """
#     def __init__(self, in_chs: int, num_features: int, num_classes: int=1000, global_pool='avg',
#                  act_fn='relu', norm_layer=nn.BatchNorm2D, norm_kwargs=None):
#         norm_kwargs = norm_kwargs or {}
#         self.global_pool = global_pool
#
#         self.conv_1x1 = Conv2d(in_chs, num_features, 1)
#         self.bn = norm_layer(num_features, **norm_kwargs)
#         self.act_fn = act_fn
#         if num_classes > 0:
#             self.classifier = nn.Linear(num_features, num_classes, use_bias=True)
#         else:
#             self.classifier = None
#
#     def __call__(self, x: JaxArray, training: bool) -> JaxArray:
#         if self.global_pool == 'avg':
#             x = x.mean((2, 3))
#         x = self.conv_1x1(x)
#         x = self.bn(x, training=training)
#         x = self.act_fn(x)
#         x = self.classifier(x)
#         return x


class Head(nn.Module):
    """ Standard Head from EfficientNet, MixNet, MNasNet, MobileNetV2, etc. """
    num_features: int
    num_classes: int = 1000
    global_pool: str = 'avg'

    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    linear_layer: ModuleDef = linear
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        x = self.conv_layer(self.num_features, 1, name='conv1x1')(x)
        x = self.norm_layer(name='bn', training=training)(x)
        x = self.act_fn(x)
        if self.global_pool == 'avg':
            x = x.mean((2, 3))
        if self.num_classes > 0:
            self.linear_layer(self.num_classes, bias=True, name='classifier')(x)
        return x


class EfficientNet(nn.Module):
    """ EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1

    """
    # model config
    block_args: Sequence[Sequence[Dict]]
    stem_features: int = 32
    feat_multiplier: float = 1.0
    feat_divisor: int = 8
    feat_min: int = None
    fix_stem: bool = False
    pad_type: str = 'LIKE'
    output_stride: int = 32

    # classifier config
    num_classes: int = 1000
    num_features: int = 1280
    global_pool: str = 'avg'
    # output_stride: int = 32  # FIXME support variable output strides

    # regularization
    drop_rate: float = 0.
    drop_path_rate: float = 0.

    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    se_layer: ModuleDef = SqueezeExcite
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        if not self.fix_stem:
            stem_features = round_channels(
                self.stem_features, self.feat_multiplier, self.feat_divisor, self.feat_min)
        x = ConvBnAct(
            out_features=stem_features, kernel_size=3, stride=2, pad_type=self.pad_type,
            conv_layer=self.conv_layer, norm_layer=self.norm_layer, act_fn=self.act_fn,
            name='stem')(x, training=training)

        for stage in EfficientNetBuilder(
                stem_features, self.block_args, BlockFactory(),
                feat_multiplier=self.feat_multiplier, feat_divisor=self.feat_divisor, feat_min=self.feat_min,
                output_stride=self.output_stride, pad_type=self.pad_type, conv_layer=self.conv_layer,
                norm_layer=self.norm_layer, se_layer=self.se_layer, act_fn=self.act_fn,
                drop_path_rate=self.drop_path_rate, verbose=_DEBUG)():
            for block in stage:
                x = block(x, training=training)

        x = Head(num_features=self.num_features, num_classes=self.num_classes, name='head')(x, training=training)
        return x


# FIXME refactor model gen/entrypoint fn to be common to linen, objax, etc

def _gen_efficientnet(variant, feat_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (feat_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      feat_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, feat_multiplier, 8, None),
        stem_features=32,
        feat_multiplier=feat_multiplier,
        conv_layer=conv2d,
        norm_layer=batchnorm2d,
        se_layer=SqueezeExcite,
        act_fn=kwargs.pop('act_fn', nn.relu),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    return _gen_efficientnet('efficientnet_b0', pretrained=pretrained, **kwargs)
