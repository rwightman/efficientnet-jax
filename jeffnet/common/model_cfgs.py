""" Model definitions and default (ImageNet) configurations

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from .arch_defs import *
import fnmatch

__all__ = ['get_model_cfg', 'list_models']

from .constants import IMAGENET_MEAN, IMAGENET_STD, INCEPTION_MEAN, INCEPTION_STD, get_bn_args_tf, get_bn_args_pt


def get_model_cfg(name):
    if name not in _model_cfg:
        return None
    return deepcopy(_model_cfg[name])


def list_models(pattern='', pretrained=True):
    model_names = []
    for k, c in _model_cfg.items():
        if (pretrained and c['default_cfg']['url']) or not pretrained:
            model_names.append(k)
    if pattern:
        model_names = fnmatch.filter(model_names, pattern)  # include these models
    return model_names


def dcfg(url='', **kwargs):
    """ Default Dataset Config (ie ImageNet pretraining config)"""
    cfg = dict(
        url=url, num_classes=1000, input_size=(3, 224, 224), pool_size=(7, 7),
        crop_pct=0.875, interpolation='bicubic', mean=IMAGENET_MEAN, std=IMAGENET_STD)
    cfg.update(kwargs)
    return cfg


def pt_acfg(**kwargs):
    """ Architecture Config (PyTorch model origin) """
    bn_cfg = kwargs.pop('bn_cfg', None) or get_bn_args_pt()
    return {'pad_type': 'LIKE', 'bn_cfg': bn_cfg, **kwargs}


def tf_acfg(**kwargs):
    """ Architecture Config (Tensorflow model origin) """
    bn_cfg = kwargs.pop('bn_cfg', None) or get_bn_args_tf()
    return {'pad_type': 'SAME', 'bn_cfg': bn_cfg, **kwargs}


_model_cfg = dict(
    pt_mnasnet_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mnasnet_100-3580d9fc.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mnasnet_b1,
    ),
    pt_semnasnet_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_semnasnet_100-e8c37d5d.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mnasnet_a1,
    ),

    pt_mobilenetv2_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mobilenetv2_100-39fce4c0.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mobilenet_v2,
    ),
    pt_mobilenetv2_110d=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mobilenetv2_110d-fbaf9685.npz'),
        arch_cfg=pt_acfg(feat_multiplier=1.1, depth_multiplier=1.2, fix_stem_head=True),
        arch_fn=arch_mobilenet_v2,
    ),
    pt_mobilenetv2_120d=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mobilenetv2_120d-eaa01988.npz'),
        arch_cfg=pt_acfg(feat_multiplier=1.2, depth_multiplier=1.4, fix_stem_head=True),
        arch_fn=arch_mobilenet_v2,
    ),
    pt_mobilenetv2_140=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mobilenetv2_140-9b7c5e30.npz'),
        arch_cfg=pt_acfg(feat_multiplier=1.4),
        arch_fn=arch_mobilenet_v2,
    ),

    pt_fbnetc_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_fbnetc_100-21618080.npz',
            interpolation='bilinear'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_fbnetc,
    ),

    pt_spnasnet_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_spnasnet_100-886fba67.npz',
            interpolation='bilinear'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_spnasnet,
    ),

    pt_efficientnet_b0=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_b0-d480c9de.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b1=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_b1-44dd2799.npz',
            input_size=(3, 240, 240), pool_size=(8, 8)),
        arch_cfg=pt_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b2=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_b2-3ab3711f.npz',
            input_size=(3, 260, 260), pool_size=(9, 9)),
        arch_cfg=pt_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b3=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_b3-2c86e358.npz',
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=pt_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b4=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=pt_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b5=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=pt_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b6=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=pt_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b7=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=pt_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b8=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
        arch_cfg=pt_acfg(feat_multiplier=2.2, depth_multiplier=3.6),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_l2=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.961),
        arch_cfg=pt_acfg(feat_multiplier=4.3, depth_multiplier=5.3),
        arch_fn=arch_efficientnet,
    ),

    tf_efficientnet_b0=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b0-71d7115d.npz',
            input_size=(3, 224, 224)),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b1=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b1-83ec238a.npz',
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b2=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b2-1d1f6c3a.npz',
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b3=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b3-2ac427e3.npz',
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b4=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b4-716d81f1.npz',
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b5=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b5-6092c165.npz',
            input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=tf_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b6=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b6-27e6732d.npz',
            input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=tf_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b7=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b7-7bbc9e2b.npz',
            input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=tf_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b8=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b8-67bfcabf.npz',
            input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
        arch_cfg=tf_acfg(feat_multiplier=2.2, depth_multiplier=3.6),
        arch_fn=arch_efficientnet,
    ),

    tf_efficientnet_b0_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b0_ap-d860b743.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD, input_size=(3, 224, 224)),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b1_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b1_ap-95cb042f.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b2_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b2_ap-c2e62974.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b3_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b3_ap-e62198e7.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b4_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b4_ap-e65a2ef8.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b5_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b5_ap-1924f949.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=tf_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b6_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b6_ap-28daec82.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=tf_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b7_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b7_ap-c7218e34.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=tf_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b8_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b8_ap-afc6b624.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
        arch_cfg=tf_acfg(feat_multiplier=2.2, depth_multiplier=3.6),
        arch_fn=arch_efficientnet,
    ),

    tf_efficientnet_b0_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b0_ns-6f64c20c.npz',
            input_size=(3, 224, 224)),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b1_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b1_ns-9f8b5df2.npz',
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b2_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b2_ns-6729f09d.npz',
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b3_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b3_ns-b43b9f62.npz',
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b4_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b4_ns-bfc84391.npz',
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b5_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b5_ns-0dc1453c.npz',
            input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=tf_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b6_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b6_ns-2f311b5f.npz',
            input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=tf_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b7_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_b7_ns-feca4a8a.npz',
            input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=tf_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_l2_ns_475=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_l2_ns_475-72254c39.npz',
            input_size=(3, 475, 475), pool_size=(15, 15), crop_pct=0.936),
        arch_cfg=tf_acfg(feat_multiplier=4.3, depth_multiplier=5.3),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_l2_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_l2_ns-7a2c14f2.npz',
            input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.96),
        arch_cfg=tf_acfg(feat_multiplier=4.3, depth_multiplier=5.3),
        arch_fn=arch_efficientnet,
    ),

    pt_efficientnet_es=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_es-d7c5fd59.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_efficientnet_edge,
    ),
    pt_efficientnet_em=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_em-e36de59a.npz',
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=pt_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet_edge,
    ),
    pt_efficientnet_el=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=pt_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet_edge,
    ),

    tf_efficientnet_es=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_es-6e146728.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 224, 224), ),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet_edge,
    ),
    tf_efficientnet_em=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_em-21a4e831.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet_edge,
    ),
    tf_efficientnet_el=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_el-02387067.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet_edge,
    ),

    pt_efficientnet_lite0=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_efficientnet_lite0-6115ce63.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_efficientnet_lite,
    ),
    pt_efficientnet_lite1=dict(
        default_cfg=dcfg(
            url='',
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=pt_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet_lite,
    ),
    pt_efficientnet_lite2=dict(
        default_cfg=dcfg(
            url='',
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=pt_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet_lite,
    ),
    pt_efficientnet_lite3=dict(
        default_cfg=dcfg(
            url='',
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=pt_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet_lite,
    ),
    pt_efficientnet_lite4=dict(
        default_cfg=dcfg(
            url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=pt_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet_lite,
    ),

    tf_efficientnet_lite0=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_lite0-49febd06.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
        ),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite1=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_lite1-529d9b59.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882,
            interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
        ),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite2=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_lite2-a185f6b2.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890,
            interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
        ),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite3=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_lite3-df8fe453.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, interpolation='bilinear'),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite4=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_efficientnet_lite4-615dfa42.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.920, interpolation='bilinear'),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet_lite,
    ),

    pt_mixnet_s=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mixnet_s-1cf350a7.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mixnet_s,
    ),
    pt_mixnet_m=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mixnet_m-fbbd9e8a.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mixnet_m,
    ),
    pt_mixnet_l=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mixnet_l-27554ae7.npz'),
        arch_cfg=pt_acfg(feat_multiplier=1.3),
        arch_fn=arch_mixnet_m,
    ),
    pt_mixnet_xl=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mixnet_xl-13d7aeef.npz'),
        arch_cfg=pt_acfg(feat_multiplier=1.6, depth_multiplier=1.2),
        arch_fn=arch_mixnet_m,
    ),

    tf_mixnet_s=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mixnet_s-60afe3e1.npz'),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mixnet_s,
    ),
    tf_mixnet_m=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mixnet_m-3c049458.npz'),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mixnet_m,
    ),
    tf_mixnet_l=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mixnet_l-49daedd5.npz'),
        arch_cfg=tf_acfg(feat_multiplier=1.3),
        arch_fn=arch_mixnet_m,
    ),

    pt_mobilenetv3_large_075=dict(
        default_cfg=dcfg(url=''),
        arch_cfg=pt_acfg(feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    pt_mobilenetv3_large_100=dict(
        default_cfg=dcfg(
            interpolation='bicubic',
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/pt_mobilenetv3_large_100-0e2a5d09.npz'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mobilenet_v3,
    ),
    pt_mobilenetv3_small_075=dict(
        default_cfg=dcfg(url=''),
        arch_cfg=pt_acfg(feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    pt_mobilenetv3_small_100=dict(
        default_cfg=dcfg(url=''),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mobilenet_v3,
    ),

    tf_mobilenetv3_large_075=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mobilenetv3_large_075-86b186f9.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_large_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mobilenetv3_large_100-4db08e68.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_large_minimal_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mobilenetv3_large_minimal_100-bb99a4f0.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_small_075=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mobilenetv3_small_075-bb7ad439.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_small_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mobilenetv3_small_100-18a48cb4.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_small_minimal_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/efficientnet-jax/releases/download/weights/tf_mobilenetv3_small_minimal_100-666b8e3b.npz',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mobilenet_v3,
    ),
)