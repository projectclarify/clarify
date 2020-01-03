# coding=utf-8
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
"""Augmentation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import hashlib
import math

from PIL import ImageEnhance
from PIL import Image


def random_temporal_subsample(video,
                              length=15,
                              max_frame_skips=1,
                              frame_skip_probability=1):

  t, x, y, c = video.shape

  if t <= length + max_frame_skips:
    msg = "Can't subsample target length %s > source %s" % (length +
                                                            max_frame_skips, t)
    raise ValueError(msg)

  num_frame_skips = np.random.randint(0, max_frame_skips)
  sample_size = length + num_frame_skips
  start = np.random.randint(0, t - sample_size)
  indices = np.asarray([i for i in range(length + num_frame_skips)])
  indices += start

  for _ in range(num_frame_skips):
    skip = np.random.randint(0, len(indices))
    indices = np.delete(indices, skip)

  return video[indices]


def random_xy_shift(video, max_xy_shift=5):
  t, x, y, c = video.shape
  x_shift = np.random.randint(0, max_xy_shift)
  y_shift = np.random.randint(0, max_xy_shift)
  pad = [[0, 0], [0, 0], [0, 0], [0, 0]]
  pad[1][np.random.randint(0, 2)] = x_shift
  pad[2][np.random.randint(0, 2)] = y_shift

  padded = np.pad(video, pad_width=pad, mode='constant', constant_values=0)

  return padded[:, pad[1][1]:(x + pad[1][1]), pad[2][1]:(y + pad[2][1]), :]


def random_flips(video):
  if np.random.randint(0, 2) > 0:
    video = np.flip(video, axis=1)
  if np.random.randint(0, 2) > 0:
    video = np.flip(video, axis=2)
  return video


def random_mask(video, num_mask_patches=5, max_xy_mask_fraction=0.3):
  t, x, y, c = video.shape

  modified = np.copy(video)

  for _ in range(num_mask_patches):

    mask_fraction = np.random.uniform(0, max_xy_mask_fraction)

    x_width = int(mask_fraction * x)
    x_start = np.random.randint(x - x_width)
    x_end = x_start + x_width

    y_width = int(mask_fraction * y)
    y_start = np.random.randint(y - y_width)
    y_end = y_start + y_width

    for i in range(t):
      modified[i][x_start:x_end, y_start:y_end, :] = 0

  return modified


def random_enhancements(video,
                        min_color=0,
                        max_color=1.0,
                        min_contrast=0,
                        max_contrast=1.0,
                        min_brightness=0,
                        max_brightness=1.0,
                        min_sharpness=0,
                        max_sharpness=2.0):
  """
  Expects uint8 video in [0,255].
  """

  if video.dtype != np.uint8:
    msg = "Random enhancer expects type uint8."
    raise ValueError(msg)

  modified = np.copy(video)
  t, x, y, c = modified.shape

  color = np.random.uniform(min_color, max_color)
  contrast = np.random.uniform(min_contrast, max_contrast)
  brightness = np.random.uniform(min_brightness, max_brightness)
  sharpness = np.random.uniform(min_sharpness, max_sharpness)

  for i in range(t):
    image = Image.fromarray((modified[i]))
    image = ImageEnhance.Color(image).enhance(color)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    modified[i] = np.array(image)

  return modified


def augment_video(video,
                  do_random_subsampling=True,
                  do_random_flips=True,
                  do_random_masking=True,
                  do_random_enhancement=True,
                  do_random_shift=True,
                  subsample_length=15,
                  subsample_max_frame_skips=1,
                  shift_max_xy=5,
                  mask_num_patches=10,
                  mask_max_patch_fraction=0.2,
                  enhance_min_color=0.0,
                  enhance_max_color=1.0,
                  enhance_min_contrast=0.0,
                  enhance_max_contrast=1.0,
                  enhance_min_brightness=0.0,
                  enhance_max_brightness=1.0,
                  enhance_min_sharpness=0.0,
                  enhance_max_sharpness=2.0):
  """Wrapper for various video augmentation operations."""

  if isinstance(video, list):
    video = np.asarray(video)

  mod = np.copy(video)

  if mod.max() <= 0.5 and mod.min() >= -0.5:
    mod *= 255.0

  # Convert range and type
  if mod.dtype != np.uint8:
    mod = mod.astype(np.uint8)

  if do_random_subsampling:
    mod = random_temporal_subsample(mod,
                                    length=subsample_length,
                                    max_frame_skips=subsample_max_frame_skips)

  if do_random_enhancement:
    mod = random_enhancements(mod,
                              min_color=enhance_min_color,
                              max_color=enhance_max_color,
                              min_contrast=enhance_min_contrast,
                              max_contrast=enhance_max_contrast,
                              min_brightness=enhance_min_brightness,
                              max_brightness=enhance_max_brightness,
                              min_sharpness=enhance_min_sharpness,
                              max_sharpness=enhance_max_sharpness)

  if do_random_flips:
    mod = random_flips(mod)

  if do_random_shift:
    mod = random_xy_shift(mod, max_xy_shift=shift_max_xy)

  if do_random_masking:
    mod = random_mask(mod)

  return mod


def augment_audio(audio,
                  do_random_shift=True,
                  do_add_gaussian_noise=True,
                  shift_reduction=0.05,
                  gaussian_snr=10,
                  data_range=[-0.5, 0.5]):
  """Wrapper for various audio augmentation operations."""

  if isinstance(audio, list):
    audio = np.asarray(audio)

  original_type = audio.dtype

  if original_type != np.float32:
    audio = audio.astype(np.float32)

  modified = np.copy(audio)

  if do_random_shift:
    max_shift = int(len(modified) * (shift_reduction))
    target_length = len(modified) - max_shift
    offset = np.random.randint(0, max_shift)
    end_index = offset + target_length
    audio = audio[offset:end_index]

  if do_add_gaussian_noise:

    assert audio.max() <= data_range[1]
    assert audio.min() >= data_range[0]

    noise = np.random.uniform(data_range[0] / float(gaussian_snr),
                              data_range[1] / float(gaussian_snr), len(audio))
    audio += noise

  audio = audio.astype(original_type)

  return audio
