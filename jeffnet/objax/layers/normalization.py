""" Normalization Layer Defs
"""
from typing import Callable, Iterable, Tuple, Optional, Union

from jax import numpy as jnp

from objax import functional
from objax.module import Module
from objax.typing import JaxArray
from objax.variable import TrainVar, StateVar


class _BatchNorm(Module):
    """Applies a batch normalization on different ranks of an input tensor.

    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_.
    """

    def __init__(self, num_features: int, redux: Iterable[int], momentum: float = 0.9, eps: float = 1e-5):
        """Creates a BatchNorm module instance.

        Args:
            dims: shape of the batch normalization state variables.
            redux: list of indices of reduction axes. Batch norm statistics are computed by averaging over these axes.
            momentum: value used to compute exponential moving average of batch statistics.
            eps: small value which is used for numerical stability.
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.redux = tuple(redux)
        self.weight = TrainVar(jnp.ones(num_features))
        self.bias = TrainVar(jnp.zeros(num_features))
        self.running_mean = StateVar(jnp.zeros(num_features))
        self.running_var = StateVar(jnp.ones(num_features))

    def __call__(self, x: JaxArray, training: bool) -> JaxArray:
        """Performs batch normalization of input tensor.

        Args:
            x: input tensor.
            training: if True compute batch normalization in training mode (accumulating batch statistics),
                otherwise compute in evaluation mode (using already accumulated batch statistics).

        Returns:
            Batch normalized tensor.
        """
        shape = (1, -1, 1, 1)
        weight = self.weight.value.reshape(shape)
        bias = self.bias.value.reshape(shape)
        if training:
            mean = x.mean(self.redux, keepdims=True)
            var = (x ** 2).mean(self.redux, keepdims=True) - mean ** 2
            self.running_mean.value += (1 - self.momentum) * (mean.squeeze(axis=self.redux) - self.running_mean.value)
            self.running_var.value += (1 - self.momentum) * (var.squeeze(axis=self.redux) - self.running_var.value)
        else:
            mean, var = self.running_mean.value.reshape(shape), self.running_var.value.reshape(shape)

        y = weight * (x - mean) * functional.rsqrt(var + self.eps) + bias
        return y


class BatchNorm1d(_BatchNorm):
    """Applies a 1D batch normalization on a 3D-input batch of shape (N,C,L).

    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """Creates a BatchNorm1D module instance.

        Args:
            num_features: number of features in the input example.
            momentum: value used to compute exponential moving average of batch statistics.
            eps: small value which is used for numerical stability.
        """
        super().__init__(num_features, (0, 2), momentum, eps)


class BatchNorm2d(_BatchNorm):
    """Applies a 2D batch normalization on a 4D-input batch of shape (N,C,H,W).

    The module follows the operation described in Algorithm 1 of
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    <https://arxiv.org/abs/1502.03167>`_.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """Creates a BatchNorm2D module instance.

        Args:
            num_features: number of features in the input example.
            momentum: value used to compute exponential moving average of batch statistics.
            eps: small value which is used for numerical stability.
        """
        super().__init__(num_features, (0, 2, 3), momentum, eps)