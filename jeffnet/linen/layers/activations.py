import flax.linen as nn
import jax.nn.functions as jnnf
from functools import partial

_ACT_FN = dict(
    relu=nn.relu,
    relu6=nn.relu6,
    leaky_relu=nn.leaky_relu,
    gelu=nn.gelu,
    elu=nn.elu,
    softplus=nn.softplus,
    silu=nn.swish,
    swish=nn.swish,
    sigmoid=nn.sigmoid,
    tanh=nn.tanh,
    hard_silu=jnnf.hard_silu,
    hard_swish=jnnf.hard_silu,
    hard_sigmoid=jnnf.hard_sigmoid,
    hard_tanh=jnnf.hard_tanh,
)


def get_act_fn(name='relu', **kwargs):
    name = name.lower()
    assert name in _ACT_FN
    act_fn = _ACT_FN[name]
    if kwargs:
        act_fn = partial(act_fn, **kwargs)
    return act_fn

