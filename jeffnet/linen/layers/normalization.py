from typing import Any, Callable, Tuple, Optional

import flax.linen as nn
import flax.linen.initializers as initializers

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


def batchnorm2d(
        eps=1e-05,
        momentum=0.1,
        affine=True,
        training=True,
        name: Optional[str] = None,
        bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros,
        weight_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones,
):
    return nn.BatchNorm(
        use_running_average=not training,
        momentum=momentum,
        epsilon=eps,
        use_bias=affine,
        use_scale=affine,
        name=name,
        bias_init=bias_init,
        scale_init=weight_init,
    )
