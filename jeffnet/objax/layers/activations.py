import objax.functional as F
import jax.nn.functions as jnnf
from functools import partial

_ACT_FN = dict(
    relu=F.relu,
    relu6=jnnf.relu6,
    leaky_relu=F.leaky_relu,
    gelu=jnnf.gelu,
    elu=F.elu,
    softplus=F.softplus,
    silu=jnnf.silu,
    swish=jnnf.silu,
    sigmoid=F.sigmoid,
    tanh=F.tanh,
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

