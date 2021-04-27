import optax

from .lars import lars


def _rename(kwargs, originals, new):
    for o, n in zip(originals, new):
        o = kwargs.pop(o, None)
        if o is not None:
            kwargs[n] = o


def _erase(kwargs, names):
    for u in names:
        kwargs.pop(u, None)


def create_optax_optim(name, learning_rate=None, momentum=0.9, weight_decay=0, **kwargs):
    """ Optimizer Factory

    Args:
        learning_rate (float): specify learning rate or leave up to scheduler / optim if None
        weight_decay (float): weight decay to apply to all params, not applied if 0
        **kwargs: optional / optimizer specific params that override defaults

    With regards to the kwargs, I've tried to keep the param naming incoming via kwargs from
    config file more consistent so there is less variation. Names of common args such as eps,
    beta1, beta2 etc will be remapped where possible (even if optimizer impl uses a diff name)
    and removed when not needed. A list of some common params to use in config files as named:
        eps (float): default stability / regularization epsilon value
        beta1 (float): moving average / momentum coefficient for gradient
        beta2 (float): moving average / momentum coefficient for gradient magnitude (squared grad)
    """
    name = name.lower()
    opt_args = dict(learning_rate=learning_rate, **kwargs)
    _rename(opt_args, ('beta1', 'beta2'), ('b1', 'b2'))
    if name == 'sgd' or name == 'momentum' or name == 'nesterov':
        _erase(opt_args, ('eps',))
        if name == 'momentum':
            optimizer = optax.sgd(momentum=momentum, **opt_args)
        elif name == 'nesterov':
            optimizer = optax.sgd(momentum=momentum, nesterov=True)
        else:
            assert name == 'sgd'
            optimizer = optax.sgd(momentum=0, **opt_args)
    elif name == 'adabelief':
        optimizer = optax.adabelief(**opt_args)
    elif name == 'adam' or name == 'adamw':
        if name == 'adamw':
            optimizer = optax.adamw(weight_decay=weight_decay, **opt_args)
        else:
            optimizer = optax.adam(**opt_args)
    elif name == 'lamb':
        optimizer = optax.lamb(weight_decay=weight_decay, **opt_args)
    elif name == 'lars':
        optimizer = lars(weight_decay=weight_decay, **opt_args)
    elif name == 'rmsprop':
        optimizer = optax.rmsprop(momentum=momentum, **opt_args)
    elif name == 'rmsproptf':
        optimizer = optax.rmsprop(momentum=momentum, initial_scale=1.0, **opt_args)
    else:
        assert False, f"Invalid optimizer name specified ({name})"

    return optimizer
