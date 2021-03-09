""" EfficientNet (Objax) Model and Factory

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from typing import Optional
from functools import partial

import objax.nn as nn
import objax.functional as F
from objax import Module
from objax.typing import JaxArray

from jeffnet.common import get_model_cfg, round_features, decode_arch_def, EfficientNetBuilder

from .helpers import load_pretrained
from .layers import Conv2d, BatchNorm2d, get_act_fn
from .blocks_objax import ConvBnAct, SqueezeExcite, BlockFactory, Head, EfficientHead


class EfficientNet(Module):
    """ EfficientNet (and other MBConvNets)
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-Lite
      * MixNet S, M, L, XL
      * MobileNetV3
      * MobileNetV2
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    def __init__(self, block_defs,
                 num_classes: int = 1000, num_features: int = 1280, drop_rate: float = 0., global_pool: str = 'avg',
                 feat_multiplier: float = 1.0, feat_divisor: int = 8, feat_min: Optional[int] = None,
                 in_chs: int = 3, stem_size: int = 32, fix_stem: bool = False, output_stride: int = 32,
                 efficient_head: bool = False, pad_type: str ='LIKE', conv_layer=Conv2d, norm_layer=BatchNorm2d,
                 se_layer=SqueezeExcite, act_fn=F.relu, drop_path_rate: float = 0.):
        super(EfficientNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        cba_kwargs = dict(conv_layer=conv_layer, norm_layer=norm_layer, act_fn=act_fn)
        if not fix_stem:
            stem_size = round_features(stem_size, feat_multiplier, feat_divisor, feat_min)
        self.stem = ConvBnAct(in_chs, stem_size, 3, stride=2, pad_type=pad_type, **cba_kwargs)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            stem_size, block_defs, BlockFactory(),
            feat_multiplier=feat_multiplier, feat_divisor=feat_divisor, feat_min=feat_min,
            output_stride=output_stride, pad_type=pad_type, se_layer=se_layer, **cba_kwargs,
            drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential([nn.Sequential(b) for b in builder()])
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head (1x1 conv + pooling + classifier)
        head_layer = EfficientHead if efficient_head else Head
        self.head = head_layer(head_chs, self.num_features, self.num_classes, global_pool=global_pool, **cba_kwargs)

        # how to init?

    def get_classifier(self):
        return self.head.classifier

    def forward_features(self, x: JaxArray, training: bool) -> JaxArray:
        x = self.stem(x, training=training)
        x = self.blocks(x, training=training)
        return x

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        x = self.forward_features(x, training=training)
        x = self.head(x, training=training)
        return x


def create_model(variant, pretrained=False, **kwargs):
    model_cfg = get_model_cfg(variant)
    model_args = model_cfg['arch_fn'](variant, **model_cfg['arch_cfg'])
    model_args.update(kwargs)

    # resolve some special layers and their arguments
    se_args = model_args.pop('se_cfg', {})  # not consumable by model
    if 'se_layer' not in model_args:
        if 'bound_act_fn' in se_args:
            se_args['bound_act_fn'] = get_act_fn(se_args['bound_act_fn'])
        if 'gate_fn' in se_args:
            se_args['gate_fn'] = get_act_fn(se_args['gate_fn'])
        model_args['se_layer'] = partial(SqueezeExcite, **se_args)

    bn_args = model_args.pop('bn_cfg')  # not consumable by model
    if 'norm_layer' not in model_args:
        model_args['norm_layer'] = partial(BatchNorm2d, **bn_args)

    model_args['act_fn'] = get_act_fn(model_args.pop('act_fn', 'relu'))  # convert str -> fn

    model = EfficientNet(**model_args)
    model.default_cfg = model_cfg['default_cfg']

    if pretrained:
        load_pretrained(model, default_cfg=model.default_cfg)

    return model
