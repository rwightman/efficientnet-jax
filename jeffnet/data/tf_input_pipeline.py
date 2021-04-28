""" Tensorflow ImageNet Data Pipeline

ImageNet data pipeline adapted from Flax examples. This is mostly redundant wrt
tf_imagenet_data.py. However I wanted to get off the ground quickly with the
Flax training example scripts and there are a few changes needed to use one in
each use case.

Eventually there will be one Tensorflow image pipeline + dataset factor that will
support RandAug/AutoAug/AugMix and other datasets.

Original copyrights below. Modifications by Ross Wightman.
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

"""ImageNet input pipeline.
"""
from typing import Optional, Tuple

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

from .tf_autoaugment import distort_image_with_randaugment
from .tf_image_ops import to_float, to_uint8

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=100):
    """Generates cropped_image using one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image_bytes: `Tensor` of binary image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
    Returns:
      cropped image `Tensor`
    """
    shape = tf.io.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image


def resize(image, image_size, interpolation=tf.image.ResizeMethod.BICUBIC, antialias=True):
    return tf.image.resize([image], [image_size, image_size], method=interpolation, antialias=antialias)[0]


def at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def decode_and_random_crop(image_bytes, image_size, interpolation):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10)
    original_shape = tf.io.extract_jpeg_shape(image_bytes)
    bad = at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: decode_and_center_crop(image_bytes, image_size, interpolation),
        lambda: resize(image, image_size, interpolation))

    return image


def decode_and_center_crop(image_bytes, image_size, interpolation):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (image_size / (image_size + CROP_PADDING)) * tf.cast(tf.minimum(image_height, image_width), tf.float32),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width, padded_center_crop_size, padded_center_crop_size])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window)
    image = resize(image, image_size, interpolation)

    return image


def normalize_image(image, mean=MEAN_RGB, std=STDDEV_RGB):
    image -= tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    return image


def preprocess_for_train(
        image_bytes,
        dtype=tf.float32,
        image_size=IMAGE_SIZE,
        mean=MEAN_RGB,
        std=STDDEV_RGB,
        interpolation=tf.image.ResizeMethod.BICUBIC,
        augment_name=None,
        randaug_num_layers=None,
        randaug_magnitude=None,
):
    """Preprocesses the given image for training.

    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      dtype: data type of the image.
      image_size: image size.

    Returns:
      A preprocessed image `Tensor`.
    """
    image = decode_and_random_crop(image_bytes, image_size, interpolation)
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [image_size, image_size, 3])

    if augment_name:
        logging.info('Apply AutoAugment policy %s', augment_name)
        fill_value = [int(round(v)) for v in MEAN_RGB]
        # if augment_name == 'autoaugment':
        #     logging.info('Apply AutoAugment policy %s', augment_name)
        #     image = distort_image_with_autoaugment(image, 'v0')
        image = to_uint8(image, saturate=False)
        if augment_name == 'randaugment':
            image = distort_image_with_randaugment(
                image, randaug_num_layers, randaug_magnitude, fill_value=fill_value)
        else:
            raise ValueError('Invalid value for augment_name: %s' % augment_name)
        image = to_float(image)  # float32, [0., 255.)

    image = normalize_image(image, mean=mean, std=std)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(
        image_bytes,
        dtype=tf.float32,
        image_size=IMAGE_SIZE,
        mean=MEAN_RGB,
        std=STDDEV_RGB,
        interpolation=tf.image.ResizeMethod.BICUBIC,
):
    """Preprocesses the given image for evaluation.

    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      dtype: data type of the image.
      image_size: image size.

    Returns:
      A preprocessed image `Tensor`.
    """
    image = decode_and_center_crop(image_bytes, image_size, interpolation)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = normalize_image(image, mean=mean, std=std)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def create_split(
        dataset_builder: tfds.core.DatasetBuilder,
        batch_size: int,
        train: bool = True,
        half_precision: bool = False,
        image_size: int = IMAGE_SIZE,
        mean: Optional[Tuple[float]] = None,
        std: Optional[Tuple[float]] = None,
        interpolation: str = 'bicubic',
        augment_name: Optional[str] = None,
        randaug_num_layers: Optional[int] = None,
        randaug_magnitude: Optional[int] = None,
        cache: bool = False,
        no_repeat: bool = False,
):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.

    Args:
      dataset_builder: TFDS dataset builder for ImageNet.
      batch_size: the batch size returned by the data pipeline.
      train: Whether to load the train or evaluation split.
      half_precision: convert image datatype to half-precision
      image_size: The target size of the images (default: 224).
      mean: image dataset mean
      std: image dataset std-dev
      interpolation: interpolation method to use for image resize (default: 'bicubic')
      cache: Whether to cache the dataset (default: False).
      no_repeat: disable repeat iter for evaluation
    Returns:
      A `tf.data.Dataset`.
    """
    mean = mean or MEAN_RGB
    std = std or STDDEV_RGB
    interpolation = tf.image.ResizeMethod.BICUBIC if interpolation == 'bicubic' else tf.image.ResizeMethod.BILINEAR
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    if train:
        data_size = dataset_builder.info.splits['train'].num_examples
        split = 'train'
    else:
        data_size = dataset_builder.info.splits['validation'].num_examples
        split = 'validation'
    split_size = data_size // jax.host_count()
    start = jax.host_id() * split_size
    split = split + '[{}:{}]'.format(start, start + split_size)

    def _decode_example(example):
        if train:
            image = preprocess_for_train(
                example['image'], input_dtype, image_size, mean, std, interpolation,
                augment_name=augment_name,
                randaug_num_layers=randaug_num_layers,
                randaug_magnitude=randaug_magnitude)
        else:
            image = preprocess_for_eval(example['image'], input_dtype, image_size, mean, std, interpolation)
        return {'image': image, 'label': example['label']}

    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            'image': tfds.decode.SkipDecoding()
        }
    )
    ds.options().experimental_threading.private_threadpool_size = 16
    ds.options().experimental_threading.max_intra_op_parallelism = 1

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=0)

    ds = ds.map(_decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train and not no_repeat:
        ds = ds.repeat()

    ds = ds.prefetch(10)

    return ds
