import jax
import jax.numpy as jnp


def create_lr_schedule_epochs(
        base_lr,
        decay_type,
        steps_per_epoch,
        total_epochs,
        decay_rate=0.1,
        decay_epochs=0,
        warmup_epochs=5.,
        power=1.0,
        min_lr=1e-5):
    total_steps = int(total_epochs * steps_per_epoch)
    decay_steps = int(decay_epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    return create_lr_schedule(
        base_lr=base_lr,
        decay_type=decay_type,
        total_steps=total_steps,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        power=power,
        min_lr=min_lr,
    )


def create_lr_schedule(
        base_lr,
        decay_type,
        total_steps,
        decay_rate=0.1,
        decay_steps=0,
        warmup_steps=0,
        power=1.0,
        min_lr=1e-5):
    """Creates learning rate schedule.

    Currently only warmup + {linear,cosine} but will be a proper mini-language
    like preprocessing one in the future.

    Args:
        total_steps: The total number of steps to run.
        base_lr: The starting learning-rate (without warmup).
        decay_type: One of 'cosine', 'step', 'poly', 'exponential', 'constant'
        decay_rate: Decay fraction for step / exponential schedules
        decay_steps: Number of steps for each application of decay_rate
        warmup_steps: how many steps to warm up for.
        min_lr: Minimum learning rate.

    Returns:
        A function learning_rate(step): float -> {"learning_rate": float}.
    """

    def step_fn(step):
        """Step to learning rate function."""
        lr = base_lr
        step_mwu = jnp.maximum(0., step - warmup_steps)
        step_pct = jnp.clip(step_mwu / float(total_steps - warmup_steps), 0.0, 1.0)

        if decay_type == 'cosine':
            lr = min_lr + lr * 0.5 * (1. + jnp.cos(jnp.pi * step_pct))
        elif decay_type == 'step':
            assert decay_steps > 0
            lr = lr * decay_rate ** (step_mwu // decay_steps)
        elif decay_type.startswith('poly'):
            lr = min_lr + (lr - min_lr) * (1. - step_pct) ** power
        elif decay_type.startswith('exp'):
            assert decay_steps > 0
            lr = lr * decay_rate ** (step_mwu / decay_steps)
        elif not decay_type or decay_type.startswith('const'):
            lr = lr
        else:
            raise ValueError(f'Unknown lr type {decay_type}')

        lr = jnp.maximum(min_lr, lr)
        if warmup_steps:
            lr = lr * jnp.minimum(1., step / warmup_steps)

        return jnp.asarray(lr, dtype=jnp.float32)

    return step_fn
