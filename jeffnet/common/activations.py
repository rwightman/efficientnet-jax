import jax.nn.functions as jnnf
from functools import partial

_ACT_FN = dict(
    relu=jnnf.relu,
    relu6=jnnf.relu6,
    leaky_relu=jnnf.leaky_relu,
    gelu=jnnf.gelu,
    elu=jnnf.elu,
    softplus=jnnf.softplus,
    silu=jnnf.silu,
    swish=jnnf.silu,
    sigmoid=jnnf.sigmoid,
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

