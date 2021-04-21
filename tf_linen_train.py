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
from datetime import datetime

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

import optax

import tensorflow_datasets as tfds

import jeffnet.data.tf_input_pipeline as input_pipeline
from jeffnet.common import acc_topk, create_lr_schedule_epochs, cross_entropy_loss, get_outdir
from jeffnet.common.optim import create_optax_optim
from jeffnet.linen import create_model, EmaState

# enable jax omnistaging
jax.config.enable_omnistaging()


# flax.struct.dataclass enables instances of this class to be passed into jax
# transformations like tree_map and pmap.
@flax.struct.dataclass
class TrainState:
    step: int
    variables: flax.core.FrozenDict[str, Any]
    dynamic_scale: flax.optim.DynamicScale
    opt_tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    ema: EmaState


def config_to_opt_args(config: ml_collections.ConfigDict):
    opt_kwargs = dict(
        eps=config.get('opt_eps'),
        decay=config.get('opt_decay'),
        momentum=config.get('opt_momentum'),
        beta1=config.get('opt_beta1'),
        beta2=config.get('opt_beta2'),
        weight_decay=config.get('opt_weight_decay', 0))
    opt_kwargs = {k: v for k, v in opt_kwargs.items() if v is not None}
    return opt_kwargs


def create_train_state(config: ml_collections.ConfigDict, variables, lr_fn):
    """Create initial training state."""
    params = variables['params']
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = flax.optim.DynamicScale()

    opt_tx = create_optax_optim(
        config.opt,
        learning_rate=lr_fn,
        **config_to_opt_args(config))
    opt_state = opt_tx.init(params)

    state = TrainState(
        step=0,
        variables=variables,
        opt_tx=opt_tx,
        opt_state=opt_state,
        dynamic_scale=dynamic_scale,
        ema=EmaState.create(config.ema_decay, variables))
    return state


def restore_checkpoint(state, model_dir):
    return checkpoints.restore_checkpoint(model_dir, state)


def save_checkpoint(state, model_dir):
    if jax.host_id() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(model_dir, state, step, keep=3)


def compute_metrics(logits, labels, label_smoothing=0.):
    loss = cross_entropy_loss(logits, labels, label_smoothing=label_smoothing)
    top1, top5 = acc_topk(logits, labels, (1, 5))
    metrics = {
        'loss': loss,
        'top1': top1,
        'top5': top5,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def train_step(apply_fn, state: TrainState, batch, lr_fn, label_smoothing=0.1, weight_decay=1e-4, dropout_rng=None):
    """Perform a single training step."""

    def loss_fn(_params):
        """loss function used for training."""
        _logits, _new_model_state = apply_fn(
            state.variables.copy({'params': _params}),
            batch['image'], training=True, mutable=['batch_stats'], rngs={'dropout': dropout_rng})
        loss = cross_entropy_loss(_logits, batch['label'], label_smoothing=label_smoothing)
        weight_penalty_params = jax.tree_leaves(_params)
        weight_penalty = 0.5 * weight_decay * sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
        loss = loss + weight_penalty
        return loss, (_new_model_state, _logits)

    step = state.step
    variables, params = state.variables.pop('params')
    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = state.dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grad = grad_fn(params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grad = grad_fn(params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grad = lax.pmean(grad, axis_name='batch')
    new_model_state, logits = aux[1]
    updates, new_opt_state = state.opt_tx.update(grad, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics = compute_metrics(logits, batch['label'], label_smoothing=label_smoothing)
    metrics['learning_rate'] = lr_fn(step)

    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and the old optimizer state should be restored.
        new_opt_state = jax.tree_multimap(functools.partial(jnp.where, is_fin), new_opt_state, state.opt_state)
        new_params = jax.tree_multimap(functools.partial(jnp.where, is_fin), new_params, params)
        metrics['scale'] = dynamic_scale.scale

    new_variables = state.variables.copy({'params': new_params, **new_model_state})
    new_ema = state.ema.update(new_variables) if state.ema is not None else None
    new_state = state.replace(
        step=step + 1, variables=new_variables, opt_state=new_opt_state, dynamic_scale=dynamic_scale, ema=new_ema)
    return new_state, metrics


def eval_step(apply_fn, state, batch):
    logits = apply_fn(state.variables, batch['image'], training=False, mutable=False)
    return compute_metrics(logits, batch['label'])


def eval_step_ema(apply_fn, state, batch):
    logits = apply_fn(state.ema.variables, batch['image'], training=False, mutable=False)
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


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_variables = state.variables.copy({'batch_stats': avg(state.variables['batch_stats'])})
    if state.ema is not None:
        new_ema_variables = state.ema.variables.copy({'batch_stats': avg(state.ema.variables['batch_stats'])})
        return state.replace(variables=new_variables, ema=state.ema.replace(variables=new_ema_variables))
    else:
        return state.replace(variables=new_variables)


def train_and_evaluate(config: ml_collections.ConfigDict, resume: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      resume: Resume from checkpoints at specified dir if set (TDDO: support specific checkpoint file/step)
    """
    rng = random.PRNGKey(42)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.host_count()
    config.eval_batch_size = config.eval_batch_size or config.batch_size
    if config.eval_batch_size % jax.device_count() > 0:
        raise ValueError('Validation batch size must be divisible by the number of devices')
    local_eval_batch_size = config.eval_batch_size // jax.host_count()

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
        config.model,
        dtype=model_dtype,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        rng=model_create_rng)
    image_size = config.image_size or model.default_cfg['input_size'][-1]

    dataset_builder = tfds.builder(config.dataset, data_dir=config.data_dir)

    train_iter = create_input_iter(
        dataset_builder, local_batch_size, train=True,
        image_size=image_size, half_precision=half_prec, cache=config.cache)

    eval_iter = create_input_iter(
        dataset_builder, local_eval_batch_size, train=False,
        image_size=image_size, half_precision=half_prec, cache=config.cache)

    steps_per_epoch = dataset_builder.info.splits['train'].num_examples // config.batch_size

    if config.num_train_steps == -1:
        num_steps = steps_per_epoch * config.num_epochs
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits['validation'].num_examples
        steps_per_eval = num_validation_examples // config.eval_batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 1

    base_lr = config.lr * config.batch_size / 256.
    lr_fn = create_lr_schedule_epochs(
        base_lr, config.lr_schedule, steps_per_epoch=steps_per_epoch, total_epochs=config.num_epochs,
        decay_rate=config.lr_decay_rate, decay_epochs=config.lr_decay_epochs, warmup_epochs=config.lr_warmup_epochs,
        min_lr=config.lr_minimum)

    state = create_train_state(config, variables, lr_fn)
    if resume:
        state = restore_checkpoint(state, resume)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = flax.jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            model.apply,
            lr_fn=lr_fn,
            label_smoothing=config.label_smoothing,
            weight_decay=config.weight_decay),
        axis_name='batch')
    p_eval_step = jax.pmap(functools.partial(eval_step, model.apply), axis_name='batch')
    p_eval_step_ema = None
    if config.ema_decay != 0.:
        p_eval_step_ema = jax.pmap(functools.partial(eval_step_ema, model.apply), axis_name='batch')

    if jax.host_id() == 0:
        if resume and step_offset > 0:
            output_dir = resume
        else:
            output_base = config.output_base_dir if config.output_base_dir else './output'
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                config.model
            ])
            output_dir = get_outdir(output_base, exp_name)
        summary_writer = tensorboard.SummaryWriter(output_dir)
        summary_writer.hparams(dict(config))

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
            state = sync_batch_stats(state)  # sync batch statistics across replicas

            eval_metrics = []
            for step_eval in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)

            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, loss: %.4f, top1: %.2f, top5: %.3f',
                         epoch, summary['loss'], summary['top1'], summary['top5'])

            if p_eval_step_ema is not None:
                # NOTE running both ema and non-ema eval while improving this script
                eval_metrics = []
                for step_eval in range(steps_per_eval):
                    eval_batch = next(eval_iter)
                    metrics = p_eval_step_ema(state, eval_batch)
                    eval_metrics.append(metrics)

                eval_metrics = common_utils.get_metrics(eval_metrics)
                summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
                logging.info('eval epoch ema: %d, loss: %.4f, top1: %.2f, top5: %.3f',
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
            save_checkpoint(state, output_dir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'resume', default='', help='Output path to resume session from (if set).')

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'train_configs/default.py'),
    'File path to the Training hyperparameter configuration.')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    print('JAX host: %d / %d' % (jax.host_id(), jax.host_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)

    train_and_evaluate(config=FLAGS.config, resume=FLAGS.resume)


if __name__ == '__main__':
  app.run(main)