import optax
import jax
import jax.numpy as jnp

from typing import Union, Callable

ScalarOrSchedule = Union[float, Callable]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule):
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: -learning_rate(count))
    return optax.scale(-learning_rate)


def _update_moment(updates, moments, decay, order):
    return jax.tree_multimap(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def scale_by_rms_tf(decay: float = 0.9, eps: float = 1e-8):
    """Rescale updates by the root of the exp. moving avg of the square.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        nu = jax.tree_map(jnp.ones_like, params)  # second moment
        return optax.ScaleByRmsState(nu=nu)

    def update_fn(updates, state, params=None):
        del params
        nu = _update_moment(updates, state.nu, decay, 2)
        updates = jax.tree_multimap(lambda g, n: g * jax.lax.rsqrt(n + eps), updates, nu)
        return updates, optax.ScaleByRmsState(nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def rmsprop_tensorflow(
        learning_rate: ScalarOrSchedule,
        decay: float = 0.9,
        momentum: float = 0.,
        eps: float = 1e-8,
        nesterov: bool = False,
        centered: bool = False) -> optax.GradientTransformation:

    if centered:
        return optax.chain(
            optax.scale_by_stddev(decay=decay, eps=eps),
            _scale_by_learning_rate(learning_rate),
            optax.trace(decay=momentum, nesterov=nesterov)
        )
    return optax.chain(
        scale_by_rms_tf(decay=decay, eps=eps),
        _scale_by_learning_rate(learning_rate),
        optax.trace(decay=momentum, nesterov=nesterov),
    )


def rmsprop_momentum(
        learning_rate: ScalarOrSchedule,
        decay: float = 0.9,
        momentum: float = 0.,
        eps: float = 1e-8,
        nesterov: bool = False,
        centered: bool = False) -> optax.GradientTransformation:

    if centered:
        return optax.chain(
            optax.scale_by_stddev(decay=decay, eps=eps),
            _scale_by_learning_rate(learning_rate),
            optax.trace(decay=momentum, nesterov=nesterov)
        )
    return optax.chain(
        optax.scale_by_rms(decay=decay, eps=eps),
        _scale_by_learning_rate(learning_rate),
        optax.trace(decay=momentum, nesterov=nesterov)
    )
