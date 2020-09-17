import objax.nn as nn
from objax import Module, ModuleList
from objax.typing import JaxArray

from jeffnet.common import round_channels, get_act_fn, resolve_bn_args,\
    decode_arch_def, EfficientNetBuilder

from .layers import Conv2d, drop_path
from .blocks_objax import ConvBnAct, BlockFactory

_DEBUG = True


class EfficientHead(Module):
    """ EfficientHead from MobileNetV3 """
    def __init__(self, in_chs: int, num_features: int, num_classes: int=1000, global_pool='avg',
                 act_fn='relu', norm_layer=nn.BatchNorm2D, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        self.global_pool = global_pool

        self.conv_1x1 = Conv2d(in_chs, num_features, 1)
        self.bn = norm_layer(num_features, **norm_kwargs)
        self.act_fn = act_fn
        if num_classes > 0:
            self.classifier = nn.Linear(num_features, num_classes, use_bias=True)
        else:
            self.classifier = None

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        if self.global_pool == 'avg':
            x = x.mean((2, 3))
        x = self.conv_1x1(x)
        x = self.bn(x, training=training)
        x = self.act_fn(x)
        x = self.classifier(x)
        return x


class Head(Module):
    """ Standard Head from EfficientNet, MixNet, MNasNet, MobileNetV2, etc. """
    def __init__(self, in_chs: int, num_features: int, num_classes: int = 1000, global_pool='avg',
                 act_fn='relu', norm_layer=nn.BatchNorm2D, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        self.global_pool = global_pool

        self.conv_1x1 = Conv2d(in_chs, num_features, 1)
        self.bn = norm_layer(num_features, **norm_kwargs)
        self.act_fn = get_act_fn(act_fn)
        if num_classes > 0:
            self.classifier = nn.Linear(num_features, num_classes, use_bias=True)
        else:
            self.classifier = None

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        x = self.conv_1x1(x)
        x = self.bn(x, training=training)
        x = self.act_fn(x)
        if self.global_pool == 'avg':
            x = x.mean((2, 3))
        if self.classifier is not None:
            x = self.classifier(x)
        return x


class EfficientNet(Module):
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

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chs=3, stem_chs=32,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, padding='LIKE', fix_stem=False, act_fn='relu', drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2D, norm_kwargs=None, global_pool='avg'):
        super(EfficientNet, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        if not fix_stem:
            stem_chs = round_channels(stem_chs, channel_multiplier, channel_divisor, channel_min)
        self.stem = ConvBnAct(
            in_chs, stem_chs, 3, stride=2, pad_type=padding, act_fn=act_fn,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            BlockFactory(), channel_multiplier, channel_divisor, channel_min, output_stride, padding,
            act_fn, se_kwargs, norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential([nn.Sequential(b) for b in builder(stem_chs, block_args)])
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head (1x1 conv + pooling + classifier)
        self.head = Head(
            head_chs, self.num_features, self.num_classes, global_pool=global_pool,
            act_fn=act_fn, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

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



def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
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
      channel_multiplier: multiplier to number of channels per layer
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
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_chs=32,
        channel_multiplier=channel_multiplier,
        act_fn=kwargs.pop('act_fn', 'swish'),
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    return _gen_efficientnet('efficientnet_b0', pretrained=pretrained, **kwargs)
