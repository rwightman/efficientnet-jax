import jax
from jax import numpy as jnp, lax


# FIXME ended up with multiple cross entropy loss def here while experimenting
# with diff numeric stability issues... will cleanup someday.


def cross_entropy_loss(logits, labels, label_smoothing=0., dtype=jnp.float32):
    """Compute cross entropy for logits and labels w/ label smoothing
    Args:
        logits: [batch, length, num_classes] float array.
        labels: categorical labels [batch, length] int array.
        label_smoothing: label smoothing constant, used to determine the on and off values.
        dtype: dtype to perform loss calcs in, including log_softmax
    """
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes, dtype=dtype)
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
    logp = jax.nn.log_softmax(logits.astype(dtype))
    return -jnp.mean(jnp.sum(logp * labels, axis=-1))


def onehot(labels, num_classes, on_value=1.0, off_value=0.0, dtype=jnp.float32):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(dtype)


def weighted_cross_entropy_loss(logits, labels, weights=None, label_smoothing=0.0, dtype=jnp.float32):
    """Compute weighted cross entropy for logits and labels w/ label smoothing.
    Args:
        logits: [batch, length, num_classes] float array.
        labels: categorical labels [batch, length] int array.
        weights: None or array of shape [batch, length].
        label_smoothing: label smoothing constant, used to determine the on and off values.
        dtype: dtype to perform loss calcs in, including log_softmax
    Returns:
        Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != labels.ndim + 1:
        raise ValueError(f'Incorrect shapes. Got shape {logits.shape} logits and {labels.shape} targets')
    num_classes = logits.shape[-1]
    off_value = label_smoothing / num_classes
    on_value = 1. - label_smoothing + off_value
    soft_targets = onehot(labels, num_classes, on_value=on_value, off_value=off_value, dtype=dtype)
    logp = jax.nn.log_softmax(logits.astype(dtype))
    loss = jnp.sum(logp * soft_targets, axis=-1)
    if weights is not None:
        loss = loss * weights
    return -loss.mean()
