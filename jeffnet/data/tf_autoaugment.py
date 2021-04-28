""" RandAudgment and AutoAugment for TF data pipeline.

This code is a mish mash of various RA and AA impl for TF, including bits and pieces from:
  * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
  * https://github.com/google-research/fixmatch/tree/master/imagenet/augment

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
import tensorflow as tf

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
from jeffnet.data.tf_image_ops import cutout, solarize, solarize_add, color, contrast, brightness, posterize, rotate, \
    translate_x, translate_y, shear_x, shear_y, autocontrast, sharpness, equalize, invert, autocontrast_or_tone

_MAX_LEVEL = 10.

IMAGENET_AUG_OPS = [
    'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
    'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
    'TranslateX', 'TranslateY', 'SolarizeAdd', 'Identity',
]

NAME_TO_FUNC = {
    'AutoContrast': autocontrast_or_tone,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return level,


def _shrink_level(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return 1.0,  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return level,


def _enhance_level(level):
    # NOTE original level range doesn't make sense
    # M0 -> all degenerate
    # M0-M5 -> interpolation
    # M5 -> no op
    # M5+ -> extrapolation
    #level = (level / _MAX_LEVEL) * 1.8 + 0.1

    # this will randomly flip between interpolate and extrapolate with increasing amount based on level
    # FIXME what to limit range to?  typically makes sense 0. - 2., but larger range possible
    level = (level / _MAX_LEVEL) * .9
    level = 1.0 + _randomly_negate_tensor(level)
    level = tf.clip_by_value(level, 0., 3.)
    return level,


def _shear_level(level):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return level,


def _translate_level(level, translate_const):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return level,


def _get_args_fn(hparams):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Rotate': lambda level: _rotate_level(level) + (hparams['fill_value'],),
        # FIXME fix posterize/solarize scale as per timm
        'Posterize': lambda level: (int((level / _MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110),),
        'Color': _enhance_level,
        'Contrast': _enhance_level,
        'Brightness': _enhance_level,
        'Sharpness': _enhance_level,
        'ShearX': lambda level: _shear_level(level) + (hparams['fill_value'],),
        'ShearY': lambda level: _shear_level(level) + (hparams['fill_value'],),
        # pylint:disable=g-long-lambda
        'TranslateX': lambda level: _translate_level(level, hparams['translate_const']) + (hparams['fill_value'],),
        'TranslateY': lambda level: _translate_level(level, hparams['translate_const']) + (hparams['fill_value'],),
        # FIXME relative translate as per timm
        # pylint:enable=g-long-lambda
        'Cutout': lambda level: (),
    }


class RandAugment:
    """Random augment with fixed magnitude.
    FIXME this is a class based impl or RA from fixmatch, it needs some changes before using
    """

    def __init__(self,
                 num_layers=2,
                 prob_to_apply=None,
                 magnitude=None,
                 num_levels=10,
                 ):
        """Initialized rand augment.
        Args:
            num_layers: number of augmentation layers, i.e. how many times to do augmentation.
          prob_to_apply: probability to apply on each layer. If None then always apply.
          magnitude: default magnitude in range [0, 1], if None then magnitude will be chosen randomly.
          num_levels: number of levels for quantization of the magnitude.
        """
        self.num_layers = num_layers
        self.prob_to_apply = float(prob_to_apply) if prob_to_apply is not None else None
        self.num_levels = int(num_levels) if num_levels else None
        self.level = float(magnitude) if magnitude is not None else None
        self.augmentation_hparams = dict(
            translate_rel=0.4,
            translate_const=100)

    def _get_level(self):
        if self.level is not None:
            return tf.convert_to_tensor(self.level)
        if self.num_levels is None:
            return tf.random.uniform(shape=[], dtype=tf.float32)
        else:
            level = tf.random.uniform(shape=[], maxval=self.num_levels + 1, dtype=tf.int32)
            return tf.cast(level, tf.float32) / self.num_levels

    def _apply_one_layer(self, image):
        """Applies one level of augmentation to the image."""
        level = self._get_level()
        branch_fns = []
        for augment_op_name in IMAGENET_AUG_OPS:
            augment_fn = NAME_TO_FUNC[augment_op_name]
            args_fn = _get_args_fn(self.augmentation_hparams)[augment_op_name]

            def _branch_fn(image=image, augment_fn=augment_fn, args_fn=args_fn):
                args = [image] + list(args_fn(level))
                return augment_fn(*args)

            branch_fns.append(_branch_fn)

        branch_index = tf.random.uniform(shape=[], maxval=len(branch_fns), dtype=tf.int32)
        aug_image = tf.switch_case(branch_index, branch_fns, default=lambda: image)
        if self.prob_to_apply is not None:
            return tf.cond(
                tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
                lambda: aug_image,
                lambda: image)
        else:
            return aug_image

    def __call__(self, image, aug_image_key='image'):
        output_dict = {}

        if aug_image_key is not None:
            aug_image = image
            for _ in range(self.num_layers):
                aug_image = self._apply_one_layer(aug_image)
            output_dict[aug_image_key] = aug_image

        if aug_image_key != 'image':
            output_dict['image'] = image

        return output_dict


def _parse_policy_info(name, prob, level, augmentation_hparams):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = _get_args_fn(augmentation_hparams)[name](level)

    return func, prob, args


def _apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, *args),
        lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies, image):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for i, policy in enumerate(policies):
        image = tf.cond(
            tf.equal(i, policy_to_select),
            lambda selected_policy=policy: selected_policy(image),
            lambda: image)
    return image


def distort_image_with_randaugment(image, num_layers, magnitude, fill_value=(128, 128, 128)):
    """Applies the RandAugment policy to `image`.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719,

    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        num_layers: Integer, the number of augmentation transformations to apply
            sequentially to an image. Represented as (N) in the paper. Usually best
            values will be in the range [1, 3].
        magnitude: Integer, shared magnitude across all augmentation operations.
            Represented as (M) in the paper. Usually best values are in the range [5, 30].

    Returns:
        The augmented version of `image`.
    """
    augmentation_hparams = dict(
        translate_rel=0.4,
        translate_const=100,
        fill_value=fill_value,
    )
    available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
        'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'SolarizeAdd']

    for layer_num in range(num_layers):
        op_to_select = tf.random.uniform([], maxval=len(available_ops), dtype=tf.int32)
        random_magnitude = float(magnitude)
        with tf.name_scope('randaug_layer_{}'.format(layer_num)):
            for (i, op_name) in enumerate(available_ops):
                prob = tf.random.uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
                func, _, args = _parse_policy_info(op_name, prob, random_magnitude, augmentation_hparams)
                image = tf.cond(
                    tf.equal(i, op_to_select),
                    # pylint:disable=g-long-lambda
                    lambda selected_func=func, selected_args=args: selected_func(image, *selected_args),
                    # pylint:enable=g-long-lambda
                    lambda: image)
    return image
