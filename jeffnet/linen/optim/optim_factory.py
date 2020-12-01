import flax.optim
from .rmsprop_tensorflow import RMSPropTensorflow


def _rename(kwargs, originals, new):
    for o, n in zip(originals, new):
        o = kwargs.pop(o, None)
        if o is not None:
            kwargs[n] = o


def _erase(kwargs, names):
    for u in names:
        kwargs.pop(u, None)


def create_optim(name, params, learning_rate=None, weight_decay=0, **kwargs):
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
    opt_args = dict(learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
    if name == 'sgd':
        _erase(opt_args, ('eps', 'beta2'))
        _rename(opt_args, ('beta1',), ('beta',))
        optimizer_def = flax.optim.GradientDescent(**opt_args)
    elif name == 'momentum':
        _erase(opt_args, ('eps', 'beta2'))
        _rename(opt_args, ('beta1',), ('beta',))
        optimizer_def = flax.optim.Momentum(**opt_args)
    elif name == 'nesterov':
        _erase(opt_args, ('eps', 'beta2'))
        _rename(opt_args, ('beta1',), ('beta',))
        optimizer_def = flax.optim.Momentum(**opt_args, nesterov=True)
    elif name == 'adafactor':
        _erase(opt_args, ('eps', 'beta2', 'weight_decay'))
        _rename(opt_args, ('eps1', 'eps2'), ('epsilon1', 'epsilon2'))
        # eps is not used, there is a separate epsilon1 and epsilon2 parameter
        # beta2 not used, decay_rate is a related param, should beta2 be accepted from config as decay_rate?
        optimizer_def = flax.optim.Adafactor(**opt_args)
    elif name == 'adagrad':
        _erase(opt_args, ('beta1', 'beta2', 'weight_decay'))
        optimizer_def = flax.optim.Adagrad(**opt_args)
    elif name == 'adam':
        optimizer_def = flax.optim.Adam(**opt_args)
    elif name == 'rmsprop':
        _erase(opt_args, ('beta1', 'weight_decay'))
        optimizer_def = flax.optim.RMSProp(**opt_args)
    elif name == 'rmsproptf':
        optimizer_def = RMSPropTensorflow(**opt_args)
    else:
        assert False, f"Invalid optimizer name specified ({name})"

    optimizer = optimizer_def.create(params)
    return optimizer
