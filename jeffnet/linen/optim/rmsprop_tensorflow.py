import jax.numpy as jnp
import numpy as onp
import flax.struct as struct
from flax.optim.base import OptimizerDef


@struct.dataclass
class _RMSPropHyperParams:
    """RMSProp hyper parameters"""

    learning_rate: float
    beta1: float  # named momentum in TF
    beta2: float  # named decay in TF
    eps: float
    weight_decay: float


@struct.dataclass
class _RMSPropTfParamState:
    """RMSProp parameter state"""
    rms: onp.ndarray
    mom: onp.ndarray


class RMSPropTensorflow(OptimizerDef):
    """RMSProp optimizer that matches Tensorflow impl."""

    def __init__(self, learning_rate: float = None, beta1=0., beta2=0.9, eps=1e-8, weight_decay=0.):
        """Constructor for the RMSProp optimizer

        Args:
            learning_rate: the step size used to update the parameters.
            beta1 (float): gradient momentum factor (default: 0.)
            beta2 (float): discounting factor for the history/coming gradient magnitude (default: 0.9)
            eps: the term added to the gradient magnitude estimate for numerical stability.
        """
        hyper_params = _RMSPropHyperParams(learning_rate, beta1, beta2, eps, weight_decay)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        """Initialize parameter state"""
        return _RMSPropTfParamState(jnp.ones_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        """Apply per-parameter gradients"""

        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        new_rms = hyper_params.beta2 * state.rms + (1.0 - hyper_params.beta2) * jnp.square(grad)
        new_mom = hyper_params.beta1 * state.mom + \
                  hyper_params.learning_rate * (grad / jnp.sqrt(new_rms + hyper_params.eps))
        new_param = param - new_mom
        if hyper_params.weight_decay != 0.:
            new_param -= hyper_params.learning_rate * hyper_params.weight_decay * param
        new_state = _RMSPropTfParamState(new_rms, new_mom)

        return new_param, new_state
