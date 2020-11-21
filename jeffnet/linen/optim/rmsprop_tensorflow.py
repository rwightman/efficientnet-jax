import jax.numpy as jnp
import numpy as onp
import flax.struct as struct
from flax.optim.base import OptimizerDef


@struct.dataclass
class _RMSPropHyperParams:
    """RMSProp hyper parameters"""

    learning_rate: float
    decay: float
    momentum: float
    eps: float


@struct.dataclass
class _RMSPropTfParamState:
    """RMSProp parameter state"""
    rms: onp.ndarray
    mom: onp.ndarray


class RMSPropTensorflow(OptimizerDef):
    """RMSProp optimizer that matches Tensorflow impl."""

    def __init__(self, learning_rate: float = None, decay=0.9, momentum=0., eps=1e-8):
        """Constructor for the RMSProp optimizer

        Args:
            learning_rate: the step size used to update the parameters.
            decay (float): discounting factor for the history/coming gradient
            momentum (float): momentum factor (default: 0)
            eps: the term added to the gradient magnitude estimate for numerical stability.
        """
        hyper_params = _RMSPropHyperParams(learning_rate, decay, momentum, eps)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        """Initialize parameter state"""
        return _RMSPropTfParamState(jnp.ones_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        """Apply per-parameter gradients"""

        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        new_rms = hyper_params.decay * state.rms + (1.0 - hyper_params.decay) * jnp.square(grad)
        new_mom = hyper_params.momentum * state.mom + \
                  hyper_params.learning_rate * grad / jnp.sqrt(new_rms + hyper_params.eps)
        new_param = param - new_mom
        new_state = _RMSPropTfParamState(new_rms, new_mom)

        return new_param, new_state
