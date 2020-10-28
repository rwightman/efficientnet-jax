"""Pretrained State Dict Helpers

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from jeffnet.common import load_state_dict_from_url


def load_pretrained(model, url='', default_cfg=None, filter_fn=None):
    if not url:
        assert default_cfg is not None and default_cfg['url']
        url = default_cfg['url']
    model_vars = model.vars()
    jax_state_dict = load_state_dict_from_url(url=url, transpose=False)
    if filter_fn is not None:
        jax_state_dict = filter_fn(jax_state_dict)
    # FIXME hack, assuming alignment, currently enforced by my layer customizations
    # TODO remap keys
    model_vars.assign(jax_state_dict.values())
