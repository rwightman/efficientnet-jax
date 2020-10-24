from .arch_defs import *

__all__ = ['get_model_cfg']

from .constants import IMAGENET_MEAN, IMAGENET_STD, INCEPTION_MEAN, INCEPTION_STD, get_bn_args_tf, get_bn_args_pt


def get_model_cfg(variant):
    return deepcopy(_model_cfg[variant])


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
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mnasnet_b1,
    ),
    pt_semnasnet_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mnasnet_a1,
    ),

    pt_mobilenetv2_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mobilenet_v2,
    ),
    pt_mobilenetv2_110d=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_110d_ra-77090ade.pth'),
        arch_cfg=pt_acfg(feat_multiplier=1.1, depth_multiplier=1.2, fix_stem_head=True),
        arch_fn=arch_mobilenet_v2,
    ),
    pt_mobilenetv2_120d=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_120d_ra-5987e2ed.pth'),
        arch_cfg=pt_acfg(feat_multiplier=1.2, depth_multiplier=1.4, fix_stem_head=True),
        arch_fn=arch_mobilenet_v2,
    ),
    pt_mobilenetv2_140=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_140_ra-21a4e913.pth'),
        arch_cfg=pt_acfg(feat_multiplier=1.4),
        arch_fn=arch_mobilenet_v2,
    ),

    pt_fbnetc_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
            interpolation='bilinear'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_fbnetc,
    ),

    pt_spnasnet_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
            interpolation='bilinear'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_spnasnet,
    ),

    pt_efficientnet_b0=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_efficientnet,
    ),

    pt_efficientnet_b1=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
            input_size=(3, 240, 240), pool_size=(8, 8)),
        arch_cfg=pt_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b2=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
            input_size=(3, 260, 260), pool_size=(9, 9)),
        arch_cfg=pt_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    pt_efficientnet_b3=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
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
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
            input_size=(3, 224, 224)),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b1=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b2=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b3=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b4=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b5=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
            input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=tf_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b6=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
            input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=tf_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b7=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
            input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=tf_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b8=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth',
            input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
        arch_cfg=tf_acfg(feat_multiplier=2.2, depth_multiplier=3.6),
        arch_fn=arch_efficientnet,
    ),

    tf_efficientnet_b0_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ap-f262efe1.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD, input_size=(3, 224, 224)),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b1_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b2_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ap-2f8e7636.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b3_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ap-aad25bdd.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b4_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ap-dedb23e6.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b5_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=tf_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b6_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ap-4ffb161f.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=tf_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b7_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ap-ddb28fec.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=tf_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b8_ap=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ap-00e169fa.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
        arch_cfg=tf_acfg(feat_multiplier=2.2, depth_multiplier=3.6),
        arch_fn=arch_efficientnet,
    ),

    tf_efficientnet_b0_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth',
            input_size=(3, 224, 224)),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b1_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth',
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b2_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth',
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b3_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth',
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b4_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth',
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b5_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
            input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
        arch_cfg=tf_acfg(feat_multiplier=1.6, depth_multiplier=2.2),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b6_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth',
            input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
        arch_cfg=tf_acfg(feat_multiplier=1.8, depth_multiplier=2.6),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_b7_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth',
            input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
        arch_cfg=tf_acfg(feat_multiplier=2.0, depth_multiplier=3.1),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_l2_ns_475=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth',
            input_size=(3, 475, 475), pool_size=(15, 15), crop_pct=0.936),
        arch_cfg=tf_acfg(feat_multiplier=4.3, depth_multiplier=5.3),
        arch_fn=arch_efficientnet,
    ),
    tf_efficientnet_l2_ns=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth',
            input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.96),
        arch_cfg=tf_acfg(feat_multiplier=4.3, depth_multiplier=5.3),
        arch_fn=arch_efficientnet,
    ),

    pt_efficientnet_es=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_efficientnet_edge,
    ),
    pt_efficientnet_em=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_em_ra2-66250f76.pth',
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
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 224, 224), ),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet_edge,
    ),
    tf_efficientnet_em=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet_edge,
    ),
    tf_efficientnet_el=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet_edge,
    ),

    pt_efficientnet_lite0=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth'),
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
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
        ),
        arch_cfg=tf_acfg(),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite1=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882,
            interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
        ),
        arch_cfg=tf_acfg(feat_multiplier=1.0, depth_multiplier=1.1),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite2=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890,
            interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
        ),
        arch_cfg=tf_acfg(feat_multiplier=1.1, depth_multiplier=1.2),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite3=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, interpolation='bilinear'),
        arch_cfg=tf_acfg(feat_multiplier=1.2, depth_multiplier=1.4),
        arch_fn=arch_efficientnet_lite,
    ),
    tf_efficientnet_lite4=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD,
            input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.920, interpolation='bilinear'),
        arch_cfg=tf_acfg(feat_multiplier=1.4, depth_multiplier=1.8),
        arch_fn=arch_efficientnet_lite,
    ),

    pt_mixnet_s=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mixnet_s,
    ),
    pt_mixnet_m=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
        arch_cfg=pt_acfg(),
        arch_fn=arch_mixnet_m,
    ),
    pt_mixnet_l=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth'),
        arch_cfg=pt_acfg(feat_multiplier=1.3),
        arch_fn=arch_mixnet_m,
    ),
    pt_mixnet_xl=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth'),
        arch_cfg=pt_acfg(feat_multiplier=1.6, depth_multiplier=1.2),
        arch_fn=arch_mixnet_m,
    ),

    tf_mixnet_s=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mixnet_s,
    ),
    tf_mixnet_m=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
        arch_cfg=tf_acfg(),
        arch_fn=arch_mixnet_m,
    ),
    tf_mixnet_l=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),
        arch_cfg=tf_acfg(feat_multiplier=1.3),
        arch_fn=arch_mixnet_m,
    ),

    pt_mobilenetv3_large_075=dict(
        default_cfg=dcfg(url=''),
        arch_cfg=pt_acfg(mnv3se=True, feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    pt_mobilenetv3_large_100=dict(
        default_cfg=dcfg(
            interpolation='bicubic',
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),
        arch_cfg=pt_acfg(mnv3se=True),
        arch_fn=arch_mobilenet_v3,
    ),
    pt_mobilenetv3_small_075=dict(
        default_cfg=dcfg(url=''),
        arch_cfg=pt_acfg(mnv3se=True, feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    pt_mobilenetv3_small_100=dict(
        default_cfg=dcfg(url=''),
        arch_cfg=pt_acfg(mnv3se=True),
        arch_fn=arch_mobilenet_v3,
    ),

    tf_mobilenetv3_large_075=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(mnv3se=True, feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_large_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(mnv3se=True),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_large_minimal_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(mnv3se=True),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_small_075=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(mnv3se=True, feat_multiplier=0.75),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_small_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(mnv3se=True),
        arch_fn=arch_mobilenet_v3,
    ),
    tf_mobilenetv3_small_minimal_100=dict(
        default_cfg=dcfg(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
            mean=INCEPTION_MEAN, std=INCEPTION_STD),
        arch_cfg=tf_acfg(mnv3se=True),
        arch_fn=arch_mobilenet_v3,
    ),
)