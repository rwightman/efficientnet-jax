""" Drop Path
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import jax.random as jr
from objax import random
from objax.typing import JaxArray


def drop_path(x: JaxArray, drop_prob: float = 0., generator=random.DEFAULT_GENERATOR) -> JaxArray:
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
    keep_shape = (x.shape[0], 1, 1, 1)
    keep_mask = keep_prob + jr.bernoulli(generator.key(), p=keep_prob, shape=keep_shape)
    output = (x / keep_prob) * keep_mask
    return output
