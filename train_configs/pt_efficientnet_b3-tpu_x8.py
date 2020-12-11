""" EfficientNet-B3 for TPU v3-8 training
"""

from train_configs import default as default_lib


def get_config():
    config = default_lib.get_config()

    config.model = 'pt_efficientnet_b3'
    config.batch_size = 2048
    config.eval_batch_size = 1000
    config.ema_decay = .9999
    config.num_epochs = 550
    config.drop_rate = 0.3

    return config
