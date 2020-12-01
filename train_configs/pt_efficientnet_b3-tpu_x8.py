""" EfficientNet-B3 for TPU v3-8 training
"""

from .default import get_config as get_default


def get_config():
    config = get_default()

    config.model = 'pt_efficientnet_b3'
    config.batch_size = 2000
    config.ema_decay = .99993
    config.num_epochs = 550
    config.drop_rate = 0.3

    return config
