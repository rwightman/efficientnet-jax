""" BatchNorm Layer Wrapper

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from typing import Any, Callable, Tuple, Optional

import flax.linen as nn
import flax.linen.initializers as initializers
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


def batchnorm2d(
        eps=1e-3,
        momentum=0.99,
        affine=True,
        training=True,
        dtype: Dtype = jnp.float32,
        name: Optional[str] = None,
        bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros,
        weight_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones,
):
    return nn.BatchNorm(
        use_running_average=not training,
        momentum=momentum,
        epsilon=eps,
        use_bias=affine,
        use_scale=affine,
        dtype=dtype,
        name=name,
        bias_init=bias_init,
        scale_init=weight_init,
    )
