import optax
import jax
import jax.numpy as jnp

from .helpers import scale_by_learning_rate, update_moment, ScalarOrSchedule


def scale_by_rms(decay: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.):
    """Rescale updates by the root of the exp. moving avg of the square.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        initial_scale: initial value for second moment

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        nu = jax.tree_map(lambda n: jnp.full_like(n, initial_scale), params)  # second moment
        return optax.ScaleByRmsState(nu=nu)

    def update_fn(updates, state, params=None):
        del params
        nu = update_moment(updates, state.nu, decay, 2)
        updates = jax.tree_multimap(lambda g, n: g * jax.lax.rsqrt(n + eps), updates, nu)
        return updates, optax.ScaleByRmsState(nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_stddev(decay: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.) -> optax.GradientTransformation:
    """Rescale updates by the root of the centered exp. moving average of squares.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        initial_scale: initial value for second moment

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)  # First moment
        nu = jax.tree_map(lambda n: jnp.full_like(n, initial_scale), params)  # second moment
        return optax.ScaleByRStdDevState(mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, decay, 1)
        nu = update_moment(updates, state.nu, decay, 2)
        updates = jax.tree_multimap(
            lambda g, m, n: g * jax.lax.rsqrt(n - jnp.square(m) + eps), updates, mu, nu)
        return updates, optax.ScaleByRStdDevState(mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def rmsprop(
        learning_rate: ScalarOrSchedule,
        decay: float = 0.9,
        momentum: float = 0.,
        eps: float = 1e-8,
        centered: bool = False,
        nesterov: bool = False,
        initial_scale: float = 0.) -> optax.GradientTransformation:

    if centered:
        return optax.chain(
            scale_by_stddev(decay=decay, eps=eps, initial_scale=initial_scale),
            scale_by_learning_rate(learning_rate),
            optax.trace(decay=momentum, nesterov=nesterov) if momentum > 0 else optax.identity()
        )
    return optax.chain(
        scale_by_rms(decay=decay, eps=eps, initial_scale=initial_scale),
        scale_by_learning_rate(learning_rate),
        optax.trace(decay=momentum, nesterov=nesterov) if momentum > 0 else optax.identity()
    )
