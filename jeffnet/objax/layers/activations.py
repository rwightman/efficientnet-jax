""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import objax.functional as F
import jax.nn as nn
from functools import partial

_ACT_FN = dict(
    relu=F.relu,
    relu6=nn.relu6,
    leaky_relu=F.leaky_relu,
    gelu=nn.gelu,
    elu=F.elu,
    softplus=F.softplus,
    silu=nn.silu,
    swish=nn.silu,
    sigmoid=F.sigmoid,
    tanh=F.tanh,
    hard_silu=nn.hard_silu,
    hard_swish=nn.hard_silu,
    hard_sigmoid=nn.hard_sigmoid,
    hard_tanh=nn.hard_tanh,
)


def get_act_fn(name='relu', **kwargs):
    name = name.lower()
    assert name in _ACT_FN
    act_fn = _ACT_FN[name]
    if kwargs:
        act_fn = partial(act_fn, **kwargs)
    return act_fn

