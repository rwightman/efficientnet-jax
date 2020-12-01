import flax
import flax.serialization as serialization
import flax.struct as struct
import jax


@struct.dataclass
class EmaState:
    decay: float = struct.field(pytree_node=False, default=0.)
    params: flax.core.FrozenDict = None
    model_state: flax.core.FrozenDict = None

    @staticmethod
    def create(decay, params, model_state):
        """Initialize ema state"""
        if decay == 0.:
            # default state == disabled
            return EmaState()
        ema_params = jax.tree_map(lambda x: x, params)
        ema_model_state = jax.tree_map(lambda x: x, model_state)
        return EmaState(decay, ema_params, ema_model_state)

    def update(self, new_params, new_model_state):
        if self.decay == 0.:
            return self.replace(params=None, model_state=None)

        new_params = jax.tree_multimap(
            lambda ema, p: ema * self.decay + (1. - self.decay) * p, self.params, new_params)
        new_model_state = jax.tree_multimap(
            lambda ema, s: ema * self.decay + (1. - self.decay) * s, self.model_state, new_model_state)
        return self.replace(params=new_params, model_state=new_model_state)
