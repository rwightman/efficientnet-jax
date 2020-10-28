""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import flax.linen as nn
import jax.nn as jnn
from functools import partial

_ACT_FN = dict(
    relu=nn.relu,
    relu6=jnn.relu6,
    leaky_relu=nn.leaky_relu,
    gelu=nn.gelu,
    elu=nn.elu,
    softplus=nn.softplus,
    silu=nn.swish,
    swish=nn.swish,
    sigmoid=nn.sigmoid,
    tanh=nn.tanh,
    hard_silu=jnn.hard_silu,
    hard_swish=jnn.hard_silu,
    hard_sigmoid=jnn.hard_sigmoid,
    hard_tanh=jnn.hard_tanh,
)


def get_act_fn(name='relu', **kwargs):
    name = name.lower()
    assert name in _ACT_FN
    act_fn = _ACT_FN[name]
    if kwargs:
        act_fn = partial(act_fn, **kwargs)
    return act_fn

