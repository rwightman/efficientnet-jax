from typing import Any, Callable, Sequence, Dict
from functools import partial

from flax import linen as nn
import jax.nn as jnn

from jeffnet.common import round_features, decode_arch_def, get_model_cfg, EfficientNetBuilder
from .helpers import load_pretrained
from .layers import conv2d, batchnorm2d, get_act_fn
from .blocks_linen import ConvBnAct, SqueezeExcite, BlockFactory, Head

ModuleDef = Any


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
    block_defs: Sequence[Sequence[Dict]]
    stem_size: int = 32
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
        stem_features = self.stem_size
        if not self.fix_stem:
            stem_features = round_features(self.stem_size, self.feat_multiplier, self.feat_divisor, self.feat_min)
        x = ConvBnAct(
            out_features=stem_features, kernel_size=3, stride=2, pad_type=self.pad_type,
            **lkwargs, name='stem')(x, training=training)

        blocks = EfficientNetBuilder(
            stem_features, self.block_defs, BlockFactory(),
            feat_multiplier=self.feat_multiplier, feat_divisor=self.feat_divisor, feat_min=self.feat_min,
            output_stride=self.output_stride, pad_type=self.pad_type, se_layer=self.se_layer, **lkwargs,
            drop_path_rate=self.drop_path_rate)()
        for stage in blocks:
            for block in stage:
                x = block(x, training=training)

        x = Head(num_features=self.num_features, num_classes=self.num_classes, **lkwargs,
                 drop_rate=self.drop_rate, name='head')(x, training=training)
        return x


def create_model(variant, pretrained=False, **kwargs):
    model_cfg = get_model_cfg(variant)
    model_args = model_cfg['arch_fn'](variant, **model_cfg['arch_cfg'])
    model_args.update(kwargs)

    # resolve some special layers and their arguments
    se_args = model_args.pop('se_cfg', {})  # not consumable by model
    if 'se_layer' not in model_args:
        model_args['se_layer'] = partial(SqueezeExcite, **se_args)

    bn_args = model_args.pop('bn_cfg')  # not consumable by model
    if 'norm_layer' not in model_args:
        model_args['norm_layer'] = partial(batchnorm2d, **bn_args)

    model_args['act_fn'] = get_act_fn(model_args.pop('act_fn'))  # convert str -> fn

    model = EfficientNet(**model_args)

    if pretrained:
        load_pretrained(model)

    return model
