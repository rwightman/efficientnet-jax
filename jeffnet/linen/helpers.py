""" Pretrained State Dict Helpers

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jeffnet.common import load_state_dict_from_url, split_state_dict, load_state_dict


def load_pretrained(variables, url='', default_cfg=None, filter_fn=None):
    if not url:
        assert default_cfg is not None and default_cfg['url']
        url = default_cfg['url']
    state_dict = load_state_dict_from_url(url, transpose=True)

    source_params, source_state = split_state_dict(state_dict)
    if filter_fn is not None:
        # filter after split as we may have modified the split criteria (ie bn running vars)
        source_params = filter_fn(source_params)
        source_state = filter_fn(source_state)

    # FIXME better way to do this?
    var_unfrozen = unfreeze(variables)
    missing_keys = []
    flat_params = flatten_dict(var_unfrozen['params'])
    flat_param_keys = set()
    for k, v in flat_params.items():
        flat_k = '.'.join(k)
        if flat_k in source_params:
            assert flat_params[k].shape == v.shape
            flat_params[k] = source_params[flat_k]
        else:
            missing_keys.append(flat_k)
        flat_param_keys.add(flat_k)
    unexpected_keys = list(set(source_params.keys()).difference(flat_param_keys))
    params = freeze(unflatten_dict(flat_params))

    flat_state = flatten_dict(var_unfrozen['batch_stats'])
    flat_state_keys = set()
    for k, v in flat_state.items():
        flat_k = '.'.join(k)
        if flat_k in source_state:
            assert flat_state[k].shape == v.shape
            flat_state[k] = source_state[flat_k]
        else:
            missing_keys.append(flat_k)
        flat_state_keys.add(flat_k)
    unexpected_keys.extend(list(set(source_state.keys()).difference(flat_state_keys)))
    batch_stats = freeze(unflatten_dict(flat_state))

    if missing_keys:
        print(f' WARNING: {len(missing_keys)} keys missing while loading state_dict. {str(missing_keys)}')
    if unexpected_keys:
        print(f' WARNING: {len(unexpected_keys)} unexpected keys found while loading state_dict. {str(unexpected_keys)}')

    return dict(params=params, batch_stats=batch_stats)
