""" EfficientNet-B3 for TPU v3-8 training
"""

from train_configs import default as default_lib


def get_config():
    config = default_lib.get_config()

    config.model = 'pt_efficientnet_b3'
    config.batch_size = 2048
    config.learning_rate = 0.1
    config.eval_batch_size = 1000
    config.ema_decay = .9999
    config.num_epochs = 550
    config.drop_rate = 0.3
    config.weight_decay = 0.

    config.opt = 'shampoo'
    #config.opt_eps = .001
    config.opt_beta1 = 0.9
    config.opt_beta2 = 0.999
    config.opt_weight_decay = 1e-5  # by default, weight decay not applied in opt, l2 penalty above is used

    return config
