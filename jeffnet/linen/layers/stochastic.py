""" Dropout, DropPath, DropBLock layers
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from typing import Any

from jax import lax
from jax import random
import jax.numpy as jnp

import flax.linen as nn
from flax.nn import make_rng


PRNGKey = Any


class Dropout(nn.Module):
    """Create a dropout layer.
      Args:
        rate: the dropout probability.  (_not_ the keep rate!)
      """
    rate: float

    @nn.compact
    def __call__(self, x, training=False, rng=None):
        """Applies a random dropout mask to the input.
        Args:
            x: the inputs that should be randomly masked.
            training: if false the inputs are scaled by `1 / (1 - rate)` and
                masked, whereas if true, no mask is applied and the inputs are returned as is.
            rng: an optional `jax.random.PRNGKey`. By default `nn.make_rng()` will be used.
        Returns:
            The masked inputs reweighted to preserve mean.
        """
        if self.rate == 0. or not training:
            return x
        keep_prob = 1. - self.rate
        if rng is None:
            rng = self.make_rng('dropout')
        mask = random.bernoulli(rng, p=keep_prob, shape=x.shape)
        return lax.select(mask, x / keep_prob, jnp.zeros_like(x))


def drop_path(x: jnp.array, drop_rate: float = 0., rng=None) -> jnp.array:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_rate == 0.:
        return x
    keep_prob = 1 - drop_rate
    if rng is None:
        rng = make_rng()
    mask = random.bernoulli(key=rng, p=keep_prob, shape=(x.shape[0], 1, 1, 1))
    mask = jnp.broadcast_to(mask, x.shape)
    return lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class DropPath(nn.Module):
    rate: float = 0.

    @nn.compact
    def __call__(self, x, training: bool = True, rng: PRNGKey = None):
        if not training or self.rate == 0.:
            return x
        if rng is None:
            rng = self.make_rng('dropout')
        return drop_path(x, self.rate, rng)


# TDDO DropBlock