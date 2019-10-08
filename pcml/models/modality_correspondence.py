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

"""Modality correspondence learning models."""

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

from pcml.models import modality_correspondence_utils as mcl_utils


def extend_hparams_with_hparams(hp1, hp2):
  """Given one set of hparams add to that another."""
  for k,v in hp2.__dict__.items():
    setattr(hp1, k, v)
  return hp1


def extend_with_asr_hparams(hparams):
  # ASR hparams
  hparams.add_hparam("audio_preproc_in_bottom", True)
  hparams.add_hparam("audio_keep_example_waveforms", False)
  hparams.add_hparam("audio_sample_rate", 16000)
  hparams.add_hparam("audio_preemphasis", 0.97)
  hparams.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
  hparams.add_hparam("audio_frame_length", 25.0)
  hparams.add_hparam("audio_frame_step", 10.0)
  hparams.add_hparam("audio_lower_edge_hertz", 20.0)
  hparams.add_hparam("audio_upper_edge_hertz", 8000.0)
  hparams.add_hparam("audio_num_mel_bins", 32)
  hparams.add_hparam("audio_add_delta_deltas", False)
  hparams.add_hparam("num_zeropad_frames", 250)
  return hparams


def extend_with_resnet_hparams(hparams):

  hparams.add_hparam("layer_sizes", [3, 4, 6, 3])
  hparams.add_hparam("bottleneck_ratios", [4, 4, 4, 4])
  hparams.add_hparam("filter_sizes", [64, 64, 128, 256, 512])
  hparams.add_hparam("block_fn", "bottleneck")
  hparams.add_hparam("use_nchw", True)
  hparams.add_hparam("is_cifar", False)

  # Targeted dropout
  hparams.add_hparam("use_td", False)
  hparams.add_hparam("targeting_rate", None)
  hparams.add_hparam("keep_prob", None)

  hparams.use_nchw = False
    
  return hparams


@registry.register_hparams
def mcl_res_ut():
  """Base parameters for Universal Transformer."""
  hparams = transformer.transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  hparams = universal_transformer.update_hparams_for_universal_transformer(hparams)
  transformer.update_hparams_for_tpu(hparams)
  hparams.add_step_timing_signal = False

  hparams.recurrence_type = "basic"

  # ===
  # ASR hparams
  hparams.add_hparam("audio_preproc_in_bottom", True)
  hparams.add_hparam("audio_keep_example_waveforms", False)
  hparams.add_hparam("audio_sample_rate", 16000)
  hparams.add_hparam("audio_preemphasis", 0.97)
  hparams.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
  hparams.add_hparam("audio_frame_length", 25.0)
  hparams.add_hparam("audio_frame_step", 10.0)
  hparams.add_hparam("audio_lower_edge_hertz", 20.0)
  hparams.add_hparam("audio_upper_edge_hertz", 8000.0)
  hparams.add_hparam("audio_num_mel_bins", 32)
  hparams.add_hparam("audio_add_delta_deltas", False)
  hparams.add_hparam("num_zeropad_frames", 250)

  # =====
  
  
  # =====
  # Resnet hparams
  
  hparams.add_hparam("layer_sizes", [3, 4, 6, 3])
  hparams.add_hparam("bottleneck_ratios", [4, 4, 4, 4])
  hparams.add_hparam("filter_sizes", [64, 64, 128, 256, 512])
  hparams.add_hparam("block_fn", "bottleneck")
  hparams.add_hparam("use_nchw", True)
  hparams.add_hparam("is_cifar", False)

  # Targeted dropout
  hparams.add_hparam("use_td", False)
  hparams.add_hparam("targeting_rate", None)
  hparams.add_hparam("keep_prob", None)

  hparams.use_nchw = False
  
  # =====
  
  hparams.add_hparam("multiproblem_task_id", 0)

  hparams.batch_size = 24
  
  delattr(hparams, "batch_shuffle_size")
  hparams._hparam_types.pop("batch_shuffle_size", None)
  
  #hparams.activation_dtype = "bfloat16"
  #hparams.weight_dtype = "bfloat16"

  hparams.add_hparam("encoder_type", "dense_only")
  #hparams.add_hparam("video_consider_num_frames", 4) # DEBUG
  hparams.add_hparam("video_consider_num_frames", None)
  hparams.add_hparam("loss_type", "similarity_cosine_mse")
  hparams.add_hparam("decoder_type", "transformer")

  hparams.add_hparam("encode_video_use_conv3d", True)

  #hparams.layer_sizes = [3, 8, 36, 3]
  hparams.layer_sizes = [3, 4, 6, 3]
  #hparams.layer_sizes = [2, 3, 4, 3]

  return hparams


@registry.register_hparams
def mcl_res200():
  hparams = mcl_res_ut()
  hparams.layer_sizes = [3, 24, 36, 3]
  return hparams
 

@registry.register_hparams
def mcl_res_ut_tiny():
  hparams = mcl_res_ut()
  hparams.layer_sizes = [1,1,1,1]
  hparams.batch_size = 4
  hparams.hidden_size = 128
  hparams.filter_size = 1024
  hparams.num_heads = 8
  return hparams


def _hack_make_non_dynamic(tensor):
  shape = common_layers.shape_list(tensor)
  tensor = tf.reshape(tensor, shape)
  return tensor


def log_shape(var):
  callers_local_vars = inspect.currentframe().f_back.f_locals.items()
  variable_name = [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
  tf.logging.debug("%s shape: %s" % (variable_name, tf.shape(var)))


@registry.register_model
class ModalityCorrespondenceLearner(universal_transformer.UniversalTransformer):

  def top(self, body_output, _):  # pylint: disable=no-self-use
    return body_output

  def infer(self, features=None, **kwargs):
    del kwargs
    predictions, _ = self(features)
    return predictions

  @property
  def has_input(self):
    return True

  def encode_audio(self, features, hparams, target_space=1, use_bfloat16=False):

    if "audio" not in features:
      raise ValueError("'audio' feature not found in features")

    with tf.variable_scope("audio"):

      # Perform audio preprocessing and apply resnet to produce feature vecs.
      audio = features["audio"]
      log_shape(audio)
      audio = mcl_utils.resnet_waveforms_v2(hparams, audio, scope_name="audio", use_bfloat16=use_bfloat16)
      log_shape(audio)

      audio_shape = common_layers.shape_list(audio)
      audio_features = tf.reshape(audio, [audio_shape[0],
                                          audio_shape[1],
                                          1,
                                          audio_shape[-1]])
      audio_features = _hack_make_non_dynamic(audio_features)
      log_shape(audio_features)

      audio_features = tf.layers.dense(audio_features, hparams.hidden_size)
      log_shape(audio_features)

      (encoder_output,
       encoder_decoder_attention_bias,
       enc_extra_output) = self.encode(
         inputs=audio_features,
         target_space=target_space,
         hparams=hparams,
         losses=None)

      log_shape(encoder_output)

    encoder_output = tf.squeeze(encoder_output)
    log_shape(encoder_output)
    encoder_output_shape = common_layers.shape_list(encoder_output)

    if len(encoder_output_shape) == 3:

      last_dim = encoder_output_shape[1]*encoder_output_shape[-1]
      encoder_output = tf.reshape(encoder_output,
                                  (encoder_output_shape[0],
                                   last_dim))
      encoder_output = tf.layers.dense(encoder_output,
                                       hparams.hidden_size)

    return encoder_output, encoder_decoder_attention_bias

  def encode_video(self, features, hparams, target_space=1, use_bfloat16=False):

    # HACK
    target_space = None

    if "video" not in features:
      raise ValueError("'video' feature not found in features")

    b, t, x, y, c = common_layers.shape_list(features["video"])

    if hparams.encode_video_use_conv3d:
      features["video"] = mcl_utils.conv_pool_3d_block(features["video"])    

    batched_frames = tf.unstack(features["video"], axis=1)

    num_frames = hparams.video_consider_num_frames
    if num_frames is None:
      num_frames = len(batched_frames)

    video_fvs = []

    for i in range(num_frames):
      frame = batched_frames[i]
      frame = tf.expand_dims(frame, 1)
      video_features = mcl_utils.resnet_frame_stack_v2(
        hparams, frame, scope="visual", use_bfloat16=use_bfloat16)
      
      video_features = tf.layers.dense(video_features, hparams.hidden_size)
      video_fvs.append(video_features)

    video_features = tf.expand_dims(tf.squeeze(tf.concat(video_fvs, axis=3)), 2)
    log_shape(video_features)

    video_features = _hack_make_non_dynamic(video_features)

    with tf.variable_scope("video_encoder"):

      (video_features,
       encoder_decoder_attention_bias,
       enc_extra_output) = self.encode(
           inputs=video_features,
           target_space=target_space,
           hparams=hparams,
           losses=None)

      log_shape(video_features)

      # Map the encoder_output into a flat embedding vector
      b, t, h = common_layers.shape_list(video_features)
      video_features = tf.reshape(video_features, (b, t*h))
      video_features = tf.layers.dense(video_features, hparams.hidden_size)
      log_shape(video_features)

    return video_features, encoder_decoder_attention_bias

  def compute_similarity_loss(self, embedding_a, embedding_b, hparams,
                              targets):
    """Compute a loss that directly compares embedding vectors.

    Notes:
    * First, a matrix of similarities is computed between all pairs of
      embeddings according to whether hparams specify this be using
      cosine distance, euclidean distance, or something else.
    * Given such a similarity matrix and a vector of target values
      for the diagonal of that matrix, compute loss components for the
      diagonal and non-diagonal elements and sum (giving these two
      equal weight and avoiding relative distortion of these two as
      the batch size is changed).

    """

    def _raise_for_unrecognized_loss(loss_type):
      raise ValueError("Unrecognized loss type: %s" % loss_type)

    if hparams.loss_type.startswith("similarity_cosine"):

      norm_a = tf.nn.l2_normalize(embedding_a, 1)
      norm_b = tf.nn.l2_normalize(embedding_b, 1)
      similarity = tf.matmul(norm_a, norm_b, transpose_b=True)

    elif hparams.loss_type.startswith("similarity_euclidean"):

      # TODO: compute all-pairs similarity of embeddings.
      # For now just do these with the diagonal.
      c = (embedding_a - embedding_b)
      c = tf.math.multiply(c, c)
      c = tf.reduce_sum(c, axis=1)
      c = 0.5*tf.math.sqrt(c)

      # A sum of the distances we would want to be close to zero
      # (i.e. embedding pairs that do correspond).
      should_be_zero = tf.reduce_sum(tf.math.multiply(c, targets))
      should_be_zero = should_be_zero/tf.reduce_sum(targets)

      # A sum of the distances we would want to be far apart
      # (i.e. embedding pairs that do not correspond).
      should_be_one = tf.reduce_sum(tf.math.multiply(c, 1 - targets))
      should_be_one = should_be_one/tf.reduce_sum(1-targets)

      loss = (should_be_zero + (1 - should_be_one))/2
      
      return loss, tf.expand_dims(c, -1)

    else:

      _raise_for_unrecognized_loss(hparams.loss_type)

    targets = tf.squeeze(targets)

    # Compute diagonal component of loss
    diag = tf.linalg.tensor_diag_part(similarity)
    
    if hparams.loss_type.endswith("sigmoid_ce"):
      diag_loss = tf.losses.sigmoid_cross_entropy(targets, diag)
    elif hparams.loss_type.endswith("mse"):
      diag_loss = tf.losses.mean_squared_error(targets, diag)
    else:
      _raise_for_unrecognized_loss(hparams.loss_type)

    # Compute non-diagonal component of loss
    diag_mask = (1 - tf.eye(num_rows=tf.shape(similarity)[0]))
    diag_masked = similarity * diag_mask
    non_diag_target = tf.zeros_like(diag_masked)

    if hparams.loss_type.endswith("sigmoid_ce"):
      non_diag_loss = tf.losses.sigmoid_cross_entropy(
        non_diag_target, diag_masked)
    elif hparams.loss_type.endswith("mse"):
      non_diag_loss = tf.losses.mean_squared_error(
        non_diag_target, diag_masked)
    else:
      _raise_for_unrecognized_loss(hparams.loss_type)

    loss = (diag_loss + non_diag_loss) / 2
    
    return loss, tf.expand_dims(diag, -1)

  def decoder_dense(self, embeddings, hparams):
    """Map from concatenated features to target via dense layers."""

    fv = tf.layers.dense(embeddings, hparams.hidden_size)
    fv = _hack_make_non_dynamic(fv)
    
    sizes = mcl_utils.compute_dense_reduction(input_size=hparams.hidden_size,
                                              target_size=64)

    for size in sizes:
      fv = tf.layers.dense(fv, size)
      fv = _hack_make_non_dynamic(fv)

    target_length = 1
    prediction = tf.layers.dense(fv, target_length)
    prediction = _hack_make_non_dynamic(prediction)
    log_shape(prediction)
    
    return prediction
  
  def decoder_transformer(self, embeddings, hparams):
    """Map from concatenated features to target via transformer."""

    target_space = 1
    
    fv = tf.expand_dims(tf.expand_dims(embeddings, 1), 1)

    fv = _hack_make_non_dynamic(fv)
    log_shape(fv)

    # HACK
    hparams.hidden_size = 2*hparams.hidden_size
    
    with tf.variable_scope("decode"):

      (encoder_output,
       encoder_decoder_attention_bias,
       enc_extra_output) = self.encode(
         inputs=fv,
         target_space=target_space,
         hparams=hparams,
         losses=None)

      log_shape(encoder_output)

    prediction = tf.squeeze(tf.layers.dense(encoder_output, 1), 1)

    prediction = _hack_make_non_dynamic(prediction)
    log_shape(prediction)

    return prediction
  
  def body(self, features):

    hparams = self._hparams
    target_space = 1

    video_embedding, v_attn = self.encode_video(features, hparams)

    audio_embedding, a_attn = self.encode_audio(features, hparams)

    embeddings = tf.concat([tf.squeeze(video_embedding),
                            tf.squeeze(audio_embedding)],
                          axis=1)
    log_shape(embeddings)

    # Whether to compute a similarity loss, e.g.
    # similarity_cosine or similarity_euclidean
    if hparams.loss_type.startswith("similarity"):
      loss, prediction = self.compute_similarity_loss(
        video_embedding, audio_embedding, hparams,
        features["targets"])
      res = tf.concat([embeddings, prediction], axis=1)
      return res, {"training": loss}
  
    # DEBUG: In attempts to run on ctpu directly changing the decoder
    # type via the --extra_hparams flag we may not have been seeing 
    # any different hparams configuration - it might have wanted to see
    # --hparams instead of --extra_hparams or --hparams.
    if hparams.decoder_type == "dense":
      prediction = self.decoder_dense(embeddings, hparams)
    elif hparams.decoder_type == "transformer":
      prediction = self.decoder_transformer(embeddings, hparams)

    res = tf.concat([embeddings, prediction], axis=1)

    def _raise_for_unrecognized_loss(loss_type):
      raise ValueError("Unrecognized loss type: %s" % loss_type)

    # DEBUG: This is the loss that was used in the previous version
    # that ran without exceeding the gRPC deadline.
    if hparams.loss_type == "sigmoid_ce":
      loss = tf.losses.sigmoid_cross_entropy(
        tf.squeeze(features["targets"]), tf.squeeze(prediction))

    elif hparams.loss_type == "softmax_ce":
      loss = tf.losses.softmax_cross_entropy(
        tf.squeeze(features["targets"]), tf.squeeze(prediction))
    elif hparams.loss_type == "mse":
      loss = tf.losses.mean_squared_error(
        tf.squeeze(features["targets"]), tf.squeeze(prediction))
    elif hparams.loss_type == "huber":
      loss = tf.losses.huber_loss(
        tf.squeeze(features["targets"]), tf.squeeze(prediction))
    else:
      _raise_for_unrecognized_loss(hparams.loss_type)

    return res, {"training": loss}
