""" Drop path layer
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from jax import lax
from jax import random
import jax.numpy as jnp

from flax.linen import Module, compact
from flax.nn import make_rng


def drop_path(x, drop_prob: float = 0., rng=None):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    # FIXME not tested
    if drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    if rng is None:
        rng = make_rng('dropout')
    random_tensor = keep_prob + random.bernoulli(key=rng, p=keep_prob, shape=(x.shape[0], 1, 1, 1))
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output