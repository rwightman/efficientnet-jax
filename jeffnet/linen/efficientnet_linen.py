from typing import Any, Union, Callable, Sequence, Dict
from functools import partial

import jax
from flax import linen as nn
import jax.nn as jnn

from jeffnet.common import round_channels, decode_arch_def, EfficientNetBuilder
from .layers import conv2d, linear, batchnorm2d
from .blocks_linen import ConvBnAct, SqueezeExcite, BlockFactory

ModuleDef = Any

_DEBUG = True


class Head(nn.Module):
    """ Standard Head from EfficientNet, MixNet, MNasNet, MobileNetV2, etc. """
    num_features: int
    num_classes: int = 1000
    global_pool: str = 'avg'
    drop_rate: float = 0.

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
            x = x.mean((1, 2))
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=not training)
        if self.num_classes > 0:
            x = self.linear_layer(self.num_classes, bias=True, name='classifier')(x)
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
        lkwargs = dict(conv_layer=self.conv_layer, norm_layer=self.norm_layer, act_fn=self.act_fn)
        if not self.fix_stem:
            stem_features = round_channels(
                self.stem_features, self.feat_multiplier, self.feat_divisor, self.feat_min)
        x = ConvBnAct(
            out_features=stem_features, kernel_size=3, stride=2, pad_type=self.pad_type,
            **lkwargs, name='stem')(x, training=training)

        blocks = EfficientNetBuilder(
            stem_features, self.block_args, BlockFactory(),
            feat_multiplier=self.feat_multiplier, feat_divisor=self.feat_divisor, feat_min=self.feat_min,
            output_stride=self.output_stride, pad_type=self.pad_type, se_layer=self.se_layer, **lkwargs,
            drop_path_rate=self.drop_path_rate, verbose=_DEBUG)()
        for stage in blocks:
            for block in stage:
                x = block(x, training=training)

        x = Head(num_features=self.num_features, num_classes=self.num_classes, **lkwargs,
                 drop_rate=self.drop_rate, name='head')(x, training=training)
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
        norm_layer=kwargs.pop('norm_layer', batchnorm2d),
        se_layer=SqueezeExcite,
        act_fn=kwargs.pop('act_fn', jnn.silu),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    return model


def pt_efficientnet_b0(pretrained=False, **kwargs):
    norm_layer = partial(batchnorm2d, eps=1e-5, momentum=0.9)
    return _gen_efficientnet(
        'efficientnet_b0', pretrained=pretrained, pad_type='LIKE', norm_layer=norm_layer, **kwargs)


def tf_efficientnet_b0(pretrained=False, **kwargs):
    norm_layer = partial(batchnorm2d, eps=1e-3, momentum=0.99)
    return _gen_efficientnet(
        'efficientnet_b0', pretrained=pretrained, pad_type='SAME', norm_layer=norm_layer, **kwargs)