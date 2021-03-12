from typing import Callable, Tuple, Any, Union

import jax
import optax
from jax import numpy as jnp

ScalarOrSchedule = Union[float, Callable]


def scale_by_learning_rate(learning_rate: ScalarOrSchedule):
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: -learning_rate(count))
    return optax.scale(-learning_rate)


def update_moment(updates, moments, decay, order):
    return jax.tree_multimap(lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


FilterFn = Callable[[Tuple[Any], jnp.ndarray], jnp.ndarray]


def exclude_bias_and_norm(path: Tuple[Any], val: jnp.ndarray) -> jnp.ndarray:
    """Filter to exclude biaises and normalizations weights."""
    del val
    if path[-1] == "bias" or path[-1] == 'scale':
        return False
    return True
