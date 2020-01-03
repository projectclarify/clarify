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
"""Utilities supporting modality correspondence learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor import models

import numpy as np
import os

from tensor2tensor.models.transformer import transformer_prepare_decoder, transformer_prepare_encoder, features_to_nonpadding
from tensor2tensor.utils import t2t_model

from tensor2tensor.data_generators import problem

from tensor2tensor.models import transformer
from tensor2tensor.models import resnet
from tensor2tensor.models.research import universal_transformer

from tensor2tensor.layers import common_video
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_audio

import inspect


def compute_dense_reduction(input_size, target_size, reducing_factor=2):
  """Compute halving dense reduction sizes.
  
  E.g. given (256,1,4) return [128,64,32,16,8,4,2]
  """

  sizes = []
  current_size = input_size

  while current_size % reducing_factor == 0:

    current_size *= 1 / reducing_factor

    if current_size < target_size:
      return sizes

    sizes.append(int(current_size))

  tf.logging.info("Using dense reduction: {}".format(sizes))

  return sizes


def t2t_preprocess_waveforms(hparams,
                             inputs,
                             scope_name=None,
                             use_bfloat16=False):

  #if use_bfloat16:
  #  inputs = tf.cast(inputs, tf.bfloat16)
  #  hparams.audio_dither = tf.cast(hparams.audio_dither, tf.bfloat16)
  #else:
  inputs = tf.cast(inputs, tf.float32)

  p = hparams

  num_mel_bins = p.audio_num_mel_bins
  num_channels = 3 if p.audio_add_delta_deltas else 1

  with tf.variable_scope(scope_name):
    # Compute filterbanks
    #with tf.variable_scope("fbanks"):
    #waveforms = tf.squeeze(inputs, [2, 3])
    waveforms = inputs
    mel_fbanks = common_audio.compute_mel_filterbank_features(
        waveforms,
        sample_rate=p.audio_sample_rate,
        dither=p.audio_dither,
        preemphasis=p.audio_preemphasis,
        frame_length=p.audio_frame_length,
        frame_step=p.audio_frame_step,
        lower_edge_hertz=p.audio_lower_edge_hertz,
        upper_edge_hertz=p.audio_upper_edge_hertz,
        num_mel_bins=p.audio_num_mel_bins,
        apply_mask=True)
    if p.audio_add_delta_deltas:
      mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)
    x = tf.reshape(
        mel_fbanks,
        common_layers.shape_list(mel_fbanks)[:2] + [num_mel_bins, num_channels])

    nonpadding_mask = 1. - common_attention.embedding_to_padding(x)
    num_of_nonpadding_elements = tf.reduce_sum(
        nonpadding_mask) * num_mel_bins * num_channels

    # This replaces CMVN estimation on data
    var_epsilon = 1e-09
    mean = tf.reduce_sum(x, axis=[1],
                         keepdims=True) / num_of_nonpadding_elements
    variance = (num_of_nonpadding_elements * mean**2. - 2. * mean *
                tf.reduce_sum(x, axis=[1], keepdims=True) + tf.reduce_sum(
                    x**2, axis=[1], keepdims=True)) / num_of_nonpadding_elements
    x = (x - mean) * tf.rsqrt(variance + var_epsilon) * tf.expand_dims(
        nonpadding_mask, -1)

    if use_bfloat16:
      x = tf.cast(x, tf.bfloat16)

  return x


def resnet_wrapper(hp, inputs, use_bfloat16=False):

  block_fns = {
      "residual": resnet.residual_block,
      "bottleneck": resnet.bottleneck_block,
  }
  assert hp.block_fn in block_fns

  is_training = hp.mode == tf.estimator.ModeKeys.TRAIN

  data_format = "channels_last"
  if hp.use_nchw:
    # Convert from channels_last (NHWC) to channels_first (NCHW). This
    # provides a large performance boost on GPU.
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    data_format = "channels_first"

  inputs = resnet.conv2d_fixed_padding(inputs=inputs,
                                       filters=hp.filter_sizes[0],
                                       kernel_size=7,
                                       strides=1 if hp.is_cifar else 2,
                                       data_format=data_format)
  inputs = tf.identity(inputs, "initial_conv")
  inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

  if not hp.is_cifar:
    inputs = tf.layers.max_pooling2d(inputs=inputs,
                                     pool_size=3,
                                     strides=2,
                                     padding="SAME",
                                     data_format=data_format)
    inputs = tf.identity(inputs, "initial_max_pool")

  out = resnet.resnet_v2(inputs,
                         block_fns[hp.block_fn],
                         hp.layer_sizes,
                         hp.filter_sizes,
                         data_format,
                         is_training=is_training,
                         is_cifar=hp.is_cifar,
                         use_td=hp.use_td,
                         targeting_rate=hp.targeting_rate,
                         keep_prob=hp.keep_prob)

  return out


def resnet_waveforms_v2(hp, waveforms, scope_name, use_bfloat16=False):

  # HACK
  #waveforms = waveforms / tf.constant([256.0])

  with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

    # Preproess the waveform into an image-like tensor
    x = t2t_preprocess_waveforms(hp,
                                 waveforms,
                                 scope_name,
                                 use_bfloat16=use_bfloat16)

    # Apply ResNet, reducing the image-like tensor to a feature-vector-like tensor
    x = resnet_wrapper(hp, x, use_bfloat16=use_bfloat16)

    return x


def resnet_frame_stack_v2(hp,
                          stacked_frame_tensor,
                          scope=None,
                          use_bfloat16=False):

  if use_bfloat16:
    stacked_frame_tensor = tf.cast(stacked_frame_tensor, tf.bfloat16)
  else:
    stacked_frame_tensor = tf.cast(stacked_frame_tensor, tf.float32)

  unstacked = tf.unstack(stacked_frame_tensor, axis=1)

  encoded_frames = []

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    for frame_batch in unstacked:

      out = resnet_wrapper(hp, frame_batch)

      # HACK =====
      b, j, k, h = common_layers.shape_list(out)
      out = tf.reshape(out, (b, 1, 1, j * k * h))
      out = tf.layers.dense(out, h)
      # ==========

      encoded_frames.append(out)

  t = tf.stack(encoded_frames)

  # Transform tensor from [t, b, ...] to [b, t, ...]
  return tf.transpose(t, [1, 0, 2, 3, 4])


def conv_pool_3d_block(video,
                       conv_kernel_depth=5,
                       conv_kernel_xy=7,
                       conv_strides=[1, 2, 2],
                       conv_padding="SAME",
                       maxpool_padding="SAME",
                       maxpool_strides=[1, 2, 2]):
  """
    
    Notes:
    * With default settings returns shape (b, t, h/4, w/4, c).

    """

  b, t, x, y, c = common_layers.shape_list(video)
  dtype = video.dtype

  def _weight_variable(name, shape):
    return tf.get_variable(name, shape, dtype,
                           tf.truncated_normal_initializer(stddev=0.1))

  conv_kernel_shape = [conv_kernel_depth, conv_kernel_xy, conv_kernel_xy, c, c]
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  conv_kernel = tf.get_variable("conv3d", conv_kernel_shape, dtype, initializer)
  conv_strides = [1] + conv_strides + [1]

  # Conv 3D
  convolved_video = tf.nn.conv3d(video, conv_kernel, conv_strides, conv_padding)

  # MaxPool 3D
  maxpool_kernel_shape = [1, 1, 1, 1, 1]
  maxpool_strides = [1] + maxpool_strides + [1]
  convpooled_video = tf.nn.max_pool3d(convolved_video,
                                      ksize=maxpool_kernel_shape,
                                      strides=maxpool_strides,
                                      padding='SAME')

  return convpooled_video
