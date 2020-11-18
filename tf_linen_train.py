""" ImageNet Training

This example script has been sliced and diced from Flax Linen ImageNet examples at
https://github.com/google/flax/tree/1c7f06bbeb9d45f7a0fb5ce65cd532a28cf95d90/linen_examples/imagenet

Original copyrights below. Modifications by Ross Wightman
"""
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

import os
import functools
import time
from typing import Any

import ml_collections
from ml_collections import config_flags
from absl import app
from absl import flags
from absl import logging

import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import tensorflow as tf
import tensorflow_datasets as tfds

import jeffnet.data.tf_input_pipeline as input_pipeline
from jeffnet.common import acc_topk
from jeffnet.linen import create_model

# enable jax omnistaging
jax.config.enable_omnistaging()


def weighted_cross_entropy_loss(logits, targets, weights=None, label_smoothing=0.0):
    """Compute weighted cross entropy and entropy for log probs and targets.
    Args:
        logits: [batch, length, num_classes] float array.
        targets: categorical targets [batch, length] int array.
        weights: None or array of shape [batch, length].
        label_smoothing: label smoothing constant, used to determine the on and off values.
    Returns:
        Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                         (str(logits.shape), str(targets.shape)))
    num_classes = logits.shape[-1]
    off_value = label_smoothing / num_classes
    on_value = 1. - label_smoothing + off_value
    soft_targets = common_utils.onehot(targets, num_classes, on_value=on_value, off_value=off_value)
    loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def compute_metrics(logits, labels):
    loss = weighted_cross_entropy_loss(logits, labels, label_smoothing=0.1)
    top1, top5 = acc_topk(logits, labels, (1, 5))
    metrics = {
        'loss': loss,
        'top1': top1,
        'top5': top5,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def step_decay(lr, step, total_steps):
    ratio = jnp.maximum(0., step / total_steps)
    mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
    return mult * lr


def cosine_decay(lr, step, total_steps):
    ratio = jnp.maximum(0., step / total_steps)
    mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
    return mult * lr


def create_learning_rate_fn(base_learning_rate, steps_per_epoch, num_epochs):
    warmup_epochs = 5

    def step_fn(step):
        epoch = step / steps_per_epoch
        lr = cosine_decay(base_learning_rate, epoch - warmup_epochs, num_epochs - warmup_epochs)
        warmup = jnp.minimum(1., epoch / warmup_epochs)
        return lr * warmup

    return step_fn


def train_step(apply_fn, state, batch, learning_rate_fn, weight_decay=1e-4, dropout_rng=None):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        variables = {'params': params, **state.model_state}
        logits, new_model_state = apply_fn(
            variables, batch['image'], training=True, mutable=['batch_stats'], rngs={'dropout': dropout_rng})
        loss = weighted_cross_entropy_loss(logits, batch['label'], label_smoothing=0.1)
        weight_penalty_params = jax.tree_leaves(variables['params'])
        weight_penalty = 0.5 * weight_decay * sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
        loss = loss + weight_penalty
        return loss, (new_model_state, logits)

    step = state.step
    optimizer = state.optimizer
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grad = grad_fn(optimizer.target)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grad = grad_fn(optimizer.target)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grad = lax.pmean(grad, axis_name='batch')
    new_model_state, logits = aux[1]
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    metrics = compute_metrics(logits, batch['label'])
    metrics['learning_rate'] = lr

    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and the old optimizer
        # state should be restored.
        new_optimizer = jax.tree_multimap(functools.partial(jnp.where, is_fin), new_optimizer, optimizer)
        metrics['scale'] = dynamic_scale.scale

    new_state = state.replace(
        step=step + 1, optimizer=new_optimizer, model_state=new_model_state, dynamic_scale=dynamic_scale)
    return new_state, metrics


def eval_step(apply_fn, state, batch):
    params = state.optimizer.target
    variables = {'params': params, **state.model_state}
    logits = apply_fn(variables, batch['image'], training=False, mutable=False)
    return compute_metrics(logits, batch['label'])


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, train, image_size, half_precision, cache):
    ds = input_pipeline.create_split(
        dataset_builder, batch_size, train=train, image_size=image_size, half_precision=half_precision, cache=cache)
    it = map(prepare_tf_data, ds)
    it = flax.jax_utils.prefetch_to_device(it, 2)
    return it


# flax.struct.dataclass enables instances of this class to be passed into jax
# transformations like tree_map and pmap.
@flax.struct.dataclass
class TrainState:
    step: int
    optimizer: flax.optim.Optimizer
    model_state: Any
    dynamic_scale: flax.optim.DynamicScale


def restore_checkpoint(state, model_dir):
    return checkpoints.restore_checkpoint(model_dir, state)


def save_checkpoint(state, model_dir):
    if jax.host_id() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(model_dir, state, step, keep=3)


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = state.model_state.copy({'batch_stats': avg(state.model_state['batch_stats'])})
    return state.replace(model_state=new_model_state)


def create_train_state(config: ml_collections.ConfigDict, params, model_state):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = flax.optim.DynamicScale()

    # FIXME add optimizer factory and allow choice via config
    optimizer = flax.optim.Momentum(beta=config.momentum, nesterov=True).create(params)
    state = TrainState(step=0, optimizer=optimizer, model_state=model_state, dynamic_scale=dynamic_scale)
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, model_dir: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      model_dir: Directory where the tensorboard summaries are written to.
    """

    if jax.host_id() == 0:
        summary_writer = tensorboard.SummaryWriter(model_dir)
        summary_writer.hparams(dict(config))

    rng = random.PRNGKey(42)
    image_size = 224  # FIXME set from config / model

    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.host_count()

    platform = jax.local_devices()[0].platform
    half_prec = config.half_precision
    if half_prec:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32

    rng, model_create_rng = random.split(rng)
    model, variables = create_model(
        'tf_efficientnet_b0',  # FIXME from config
        dtype=model_dtype,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        rng=model_create_rng)
    model_state, params = variables.pop('params')

    dataset_builder = tfds.builder('imagenet2012:5.*.*', data_dir='/data/')

    train_iter = create_input_iter(
        dataset_builder, local_batch_size, train=True,
        image_size=image_size, half_precision=half_prec, cache=config.cache)

    eval_iter = create_input_iter(
        dataset_builder, local_batch_size, train=False,
        image_size=image_size, half_precision=half_prec, cache=config.cache)

    steps_per_epoch = dataset_builder.info.splits['train'].num_examples // config.batch_size

    if config.num_train_steps == -1:
        num_steps = steps_per_epoch * config.num_epochs
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits['validation'].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    base_learning_rate = config.learning_rate * config.batch_size / 256.

    state = create_train_state(config, params, model_state)
    state = restore_checkpoint(state, model_dir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = flax.jax_utils.replicate(state)

    learning_rate_fn = create_learning_rate_fn(base_learning_rate, steps_per_epoch, config.num_epochs)

    p_train_step = jax.pmap(
        functools.partial(train_step, model.apply, learning_rate_fn=learning_rate_fn), axis_name='batch')
    p_eval_step = jax.pmap(functools.partial(eval_step, model.apply), axis_name='batch')

    epoch_metrics = []
    t_loop_start = time.time()
    num_samples = 0
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        step_p1 = step + 1
        rng, step_rng = random.split(rng)
        sharded_rng = common_utils.shard_prng_key(step_rng)

        num_samples += config.batch_size
        state, metrics = p_train_step(state, batch, dropout_rng=sharded_rng)
        epoch_metrics.append(metrics)

        if step_p1 % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            epoch_metrics = common_utils.get_metrics(epoch_metrics)
            summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
            samples_per_sec = num_samples / (time.time() - t_loop_start)
            logging.info('train epoch: %d, loss: %.4f, img/sec %.2f, top1: %.2f, top5: %.3f',
                         epoch, summary['loss'], samples_per_sec, summary['top1'], summary['top5'])

            if jax.host_id() == 0:
                for key, vals in epoch_metrics.items():
                    tag = 'train_%s' % key
                    for i, val in enumerate(vals):
                        summary_writer.scalar(tag, val, step_p1 - len(vals) + i)
                summary_writer.scalar('samples per second', samples_per_sec, step)

            epoch_metrics = []
            eval_metrics = []
            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for step_eval in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)

            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, loss: %.4f, top1: %.2f, top5: %.3f',
                         epoch, summary['loss'], summary['top1'], summary['top5'])
            if jax.host_id() == 0:
                for key, val in eval_metrics.items():
                    tag = 'eval_%s' % key
                    summary_writer.scalar(tag, val.mean(), step)
                summary_writer.flush()
            t_loop_start = time.time()
            num_samples = 0

        elif step_p1 % 100 == 0:
            summary = jax.tree_map(lambda x: x.mean(), common_utils.get_metrics(epoch_metrics))
            samples_per_sec = num_samples / (time.time() - t_loop_start)
            logging.info(
                'train steps: %d, loss: %.4f, img/sec: %.2f', step_p1, summary['loss'], samples_per_sec)

        if step_p1 % steps_per_checkpoint == 0 or step_p1 == num_steps:
            state = sync_batch_stats(state)
            save_checkpoint(state, model_dir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default='./output',
    help=('Directory to store model data.'))

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'train_configs/default.py'),
    'File path to the Training hyperparameter configuration.')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)

    train_and_evaluate(model_dir=FLAGS.model_dir, config=FLAGS.config)


if __name__ == '__main__':
  app.run(main)