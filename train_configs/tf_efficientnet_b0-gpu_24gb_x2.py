""" EfficientNet-B0 for 2 x 24GB GPU training
"""

from train_configs import default as default_lib


def get_config():
    config = default_lib.get_config()

    config.batch_size = 500

    return config
