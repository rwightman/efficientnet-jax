""" BatchNorm Layer Wrapper

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from typing import Any, Callable, Tuple, Optional

from jax import lax
import jax.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as initializers

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(nn.Module):
    """BatchNorm Module.

    NOTE: A BatchNorm layer similar to Flax ver, but with diff of squares for var cal for numerical
    comparisons. Also, removed cross-process reduction in this variation (for now).

    Attributes:
        axis: the feature or non-batch axis of the input.
        momentum: decay rate for the exponential moving average of the batch statistics.
        epsilon: a small float added to variance to avoid dividing by zero.
        dtype: the dtype of the computation (default: float32).
        bias: if True, bias (beta) is added.
        scale: if True, multiply by scale (gamma).
            When the next layer is linear (also e.g. nn.relu), this can be disabled
            since the scaling will be done by the next layer.
        bias_init: initializer for bias, by default, zero.
        scale_init: initializer for scale, by default, one.
    """
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

    @nn.compact
    def __call__(self, x, training: bool):
        """Normalizes the input using batch statistics.
        Args:
            x: the input to be normalized.
        Returns:
            Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # we detect if we're in initialization via empty variable tree.
        initializing = not self.has_variable('batch_stats', 'mean')

        ra_mean = self.variable('batch_stats', 'mean', lambda s: jnp.zeros(s, jnp.float32), reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), reduced_feature_shape)

        if not training:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            var = jnp.mean((x - mean) ** 2, axis=reduction_axis, keepdims=False)
            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale', self.scale_init, reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias', self.bias_init, reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)


class FlaxBatchNorm(nn.Module):
    """ FlaxBatchNorm Module.

    NOTE: A copy of the official Flax BN layer, w/ diff of squares variance and cross-process batch stats syncing.

    Attributes:
        axis: the feature or non-batch axis of the input.
        momentum: decay rate for the exponential moving average of the batch statistics.
        epsilon: a small float added to variance to avoid dividing by zero.
        dtype: the dtype of the computation (default: float32).
        bias: if True, bias (beta) is added.
        scale: if True, multiply by scale (gamma).
            When the next layer is linear (also e.g. nn.relu), this can be disabled
            since the scaling will be done by the next layer.
        bias_init: initializer for bias, by default, zero.
        scale_init: initializer for scale, by default, one.
        axis_name: the axis name used to combine batch statistics from multiple
            devices. See `jax.pmap` for a description of axis names (default: None).
        axis_index_groups: groups of axis indices within that named axis
            representing subsets of devices to reduce over (default: None). For
            example, `[[0, 1], [2, 3]]` would independently batch-normalize over
            the examples on the first two and last two devices. See `jax.lax.psum` for more details.
    """
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x, training: bool):
        """Normalizes the input using batch statistics.
        Args:
            x: the input to be normalized.
        Returns:
            Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # we detect if we're in initialization via empty variable tree.
        initializing = not self.has_variable('batch_stats', 'mean')

        ra_mean = self.variable('batch_stats', 'mean', lambda s: jnp.zeros(s, jnp.float32), reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), reduced_feature_shape)

        if not training:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(concatenated_mean, axis_name=self.axis_name, axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale', self.scale_init, reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias', self.bias_init, reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)


class L1BatchNorm(nn.Module):
    """L1 BatchNorm Module.

    Attributes:
        axis: the feature or non-batch axis of the input.
        momentum: decay rate for the exponential moving average of the batch statistics.
        epsilon: a small float added to variance to avoid dividing by zero.
        dtype: the dtype of the computation (default: float32).
        bias: if True, bias (beta) is added.
        scale: if True, multiply by scale (gamma).
            When the next layer is linear (also e.g. nn.relu), this can be disabled
            since the scaling will be done by the next layer.
        bias_init: initializer for bias, by default, zero.
        scale_init: initializer for scale, by default, one.
    """
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

    @nn.compact
    def __call__(self, x, training: bool):
        """Normalizes the input using batch statistics.
        Args:
            x: the input to be normalized.
        Returns:
            Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, self.dtype)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # we detect if we're in initialization via empty variable tree.
        initializing = not self.has_variable('batch_stats', 'mean')

        ra_mean = self.variable('batch_stats', 'mean', lambda s: jnp.zeros(s, jnp.float32), reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), reduced_feature_shape)

        if not training:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            var = jnp.mean(lax.abs(x - mean), axis=reduction_axis, keepdims=False) * jnp.sqrt(jnp.pi / 2)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, var])
                mean, var = jnp.split(
                    lax.pmean(concatenated_mean, axis_name=self.axis_name, axis_index_groups=self.axis_index_groups), 2)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        mean = jnp.asarray(mean, self.dtype)
        var = jnp.asarray(var, self.dtype)
        y = x - mean.reshape(feature_shape)
        mul = lax.reciprocal(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale', self.scale_init, reduced_feature_shape).reshape(feature_shape)
            scale = jnp.asarray(scale, self.dtype)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias', self.bias_init, reduced_feature_shape).reshape(feature_shape)
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return jnp.asarray(y, self.dtype)


def batchnorm2d(
        eps=1e-3,
        momentum=0.99,
        affine=True,
        dtype: Dtype = jnp.float32,
        name: Optional[str] = None,
        variant: str = '',
        bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros,
        weight_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones,
):
    layer = BatchNorm
    if variant == 'flax':
        layer = FlaxBatchNorm
    elif variant == 'l1':
        layer = L1BatchNorm

    return layer(
        momentum=momentum,
        epsilon=eps,
        use_bias=affine,
        use_scale=affine,
        dtype=dtype,
        name=name,
        bias_init=bias_init,
        scale_init=weight_init,
    )
