# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # base output directory for experiments (checkpoints, summaries), './output' if not valid
    config.output_base_dir = ''

    config.data_dir = '/data/'  # use --config.data_dir arg to set without modifying config file
    config.dataset = 'imagenet2012:5.0.0'
    config.num_classes = 1000  # FIXME not currently used

    config.model = 'tf_efficientnet_b0'
    config.image_size = 0  # set from model defaults if 0
    config.batch_size = 224
    config.eval_batch_size = 100  # set to config.bach_size if 0
    config.lr = .8  #  0.016
    config.label_smoothing = 0.1
    config.weight_decay = 1e-5  # l2 weight penalty added to loss
    config.ema_decay = .99997

    #config.opt = 'adamw'
    #config.opt = 'lars'
    #config.opt_eps = 1e-6
    #config.opt_beta1 = 0.9
    #config.opt_beta2 = 0.999
    #config.opt_weight_decay = 0.00001  # by default, weight decay not applied in opt, l2 penalty above is used

    config.opt = 'rmsproptf'
    config.opt_eps = .001
    config.opt_momentum = 0.9
    config.opt_decay = 0.9
    config.opt_weight_decay = 0.  # by default, weight decay not applied in opt, l2 penalty above is used

    config.lr_schedule = 'step'
    config.lr_decay_rate = 0.97
    config.lr_decay_epochs = 2.4
    config.lr_warmup_epochs = 5.
    config.lr_minimum = 1e-6
    config.num_epochs = 450

    config.cache = False
    config.half_precision = True

    config.drop_rate = 0.2
    config.drop_path_rate = 0.2

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
