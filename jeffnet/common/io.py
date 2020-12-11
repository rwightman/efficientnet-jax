""" Numpy State Dict Helpers

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import numpy as np
import jax.numpy as jnp


def load_state_dict(filename, include_type_map=False, transpose=False):
    np_weights = np.load(filename)
    var_names = np_weights['names']
    var_types = []
    if 'types' in np_weights:
        var_types = np_weights['types']
    var_values = [np_weights[str(i)] for i in range(len(var_names))]
    jax_state_dict = {}
    type_map = {}
    for i, (k, v) in enumerate(zip(var_names, var_values)):
        if transpose:
            # FIXME this is narrowly defined and currently only robust to conv2d, linear, typical norm layers
            assert len(v.shape) in (1, 2, 4)
            if len(v.shape) == 4:
                v = v.transpose((2, 3, 1, 0))  # OIHW -> HWIO
            elif len(v.shape) == 2:
                v = v.transpose()  # OI -> IO
        jax_state_dict[k] = jnp.array(v)
        if include_type_map and len(var_types) == len(var_names):
            t = var_types[i]
            type_map[k] = t.lower()

    if len(type_map):
        return jax_state_dict, type_map
    else:
        return jax_state_dict


_STATE_NAMES = ('running_mean', 'running_var', 'moving_mean', 'moving_variance')


def split_state_dict(state_dict):
    """ split a state_dict into params and other state
    FIXME currently other state is assumed to be norm running state
    """
    out_params = {}
    out_state = {}
    for k, v in state_dict.items():
        if any(n in k for n in _STATE_NAMES):
            out_state[k] = v
        else:
            out_params[k] = v
    return out_params, out_state


def get_outdir(path, *paths, retry_inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif retry_inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir
