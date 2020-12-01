""" EfficientNet-B0 for 2 x 24GB GPU training
"""

from .default import get_config as get_default


def get_config():
    config = get_default()

    config.batch_size = 500

    return config
