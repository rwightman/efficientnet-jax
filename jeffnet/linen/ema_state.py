import flax
import flax.serialization as serialization
import flax.struct as struct
import jax


@struct.dataclass
class EmaState:
    ema_decay: float = struct.field(pytree_node=False)
    ema_params: flax.core.FrozenDict
    ema_model_state: flax.core.FrozenDict

    @staticmethod
    def create(decay, params, model_state):
        """Initialize ema state"""
        ema_params = jax.tree_map(lambda x: x, params)
        ema_model_state = jax.tree_map(lambda x: x, model_state)
        return EmaState(decay, ema_params, ema_model_state)

    def update(self, new_params, new_model_state):
        new_ema_params = jax.tree_multimap(
            lambda ema, p: ema * self.ema_decay + (1. - self.ema_decay) * p, self.ema_params, new_params)
        new_ema_model_state = jax.tree_multimap(
            lambda ema, s: ema * self.ema_decay + (1. - self.ema_decay) * s, self.ema_model_state, new_model_state)
        return self.replace(ema_params=new_ema_params, ema_model_state=new_ema_model_state)

    # def state_dict(self):
    #     return serialization.to_state_dict({
    #         'ema_params': serialization.to_state_dict(self.ema_params),
    #         'ema_model_state': serialization.to_state_dict(self.ema_model_state)
    #     })
    #
    # def restore_state(self, state_dict):
    #     ema_params = serialization.from_state_dict(self.ema_params, state_dict['ema_params'])
    #     ema_model_state = serialization.from_state_dict(self.ema_model_state, state_dict['ema_model_state'])
    #     return self.replace(ema_params=ema_params, ema_model_state=ema_model_state)
