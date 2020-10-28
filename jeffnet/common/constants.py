DEFAULT_CROP_PCT = 0.875
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

BN_MOM_TF_DEFAULT = 0.99
BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=BN_MOM_TF_DEFAULT, eps=BN_EPS_TF_DEFAULT)

BN_MOM_PT_DEFAULT = .9
BN_EPS_PT_DEFAULT = 1e-5
_BN_ARGS_PT = dict(momentum=BN_MOM_PT_DEFAULT, eps=BN_EPS_PT_DEFAULT)


def get_bn_args_tf():
    return _BN_ARGS_TF.copy()


def get_bn_args_pt():
    return _BN_ARGS_PT.copy()