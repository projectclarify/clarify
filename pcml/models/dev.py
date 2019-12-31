from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

from tensor2tensor.models import resnet

from tensor2tensor.data_generators import problem

from tensor2tensor.layers import common_video
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_audio

from tensor2tensor.models import transformer
from tensor2tensor.models import resnet
from tensor2tensor import models
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.models.transformer import transformer_prepare_decoder, transformer_prepare_encoder, features_to_nonpadding

from pcml.models import modality_correspondence_utils as mcl_utils
from pcml.models import modality_correspondence as mcl

from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem

tf.logging.set_verbosity(tf.logging.DEBUG)

from tensor2tensor.data_generators.mnist import ImageMnist, mnist_generator

_MNIST_IMAGE_SIZE = 28


def layers():
    return common_layers.layers()


@registry.register_problem
class DevProblem(ImageMnist):

    def preprocess_example(self, example, mode, unused_hparams):
        image = example["inputs"]
        image.set_shape([_MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1])
        if not self._was_reversed:
            image = tf.image.per_image_standardization(image)
        example["inputs"] = image
        example["targets"] = tf.to_float(tf.equal(example["targets"], 1))
        return example

    @property
    def num_classes(self):
        return 1

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.modality = {
            "inputs": modalities.ModalityType.IMAGE,
            "targets": modalities.ModalityType.IDENTITY
        }
        p.vocab_size = {"inputs": 256, "targets": self.num_classes}
        p.batch_size_multiplier = 4 if self.is_small else 256
        p.loss_multiplier = 3.0 if self.is_small else 1.0
        if self._was_reversed:
            p.loss_multiplier = 1.0
        p.input_space_id = problem.SpaceID.IMAGE
        p.target_space_id = problem.SpaceID.IMAGE_LABEL


@registry.register_hparams
def mcl_res_ut_vtiny():
    hparams = mcl.mcl_res_ut()
    hparams.layer_sizes = [4, 4, 4, 4]
    hparams.batch_size = 4
    hparams.hidden_size = 64
    hparams.filter_size = 32
    hparams.num_heads = 4
    return hparams


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
    inputs = resnet.batch_norm_relu(inputs,
                                    is_training,
                                    data_format=data_format)

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
                           keep_prob=hp.keep_prob,
                           bottleneck_ratios=hp.bottleneck_ratios)

    if hp.use_nchw:
        out = tf.transpose(out, [0, 2, 3, 1])

    return out


@registry.register_model
class MCLDev(mcl.ModalityCorrespondenceLearner):

    def top(self, body_output, _):  # pylint: disable=no-self-use
        return body_output

    def eager_embed_single_image(self, image_data, **kwargs):
        del kwargs
        image_tensor = tf.convert_to_tensor(image_data)
        image_tensor = tf.expand_dims(image_tensor, 0)
        features = {"inputs": image_tensor}
        return tf.squeeze(self.embed_image(features)).numpy()

    @property
    def has_input(self):
        return True

    def embed_image(self, features):
        hparams = self.hparams
        embedding = resnet_wrapper(hparams, features["inputs"])
        return embedding

    def body(self, features):

        hparams = self._hparams

        features["inputs"] = tf.cast(features["inputs"], tf.float32)

        emb = self.embed_image(features)
        mcl.log_shape(emb)

        b, j, k, h = common_layers.shape_list(emb)
        shape = (b, j * k * h)
        emb = tf.reshape(emb, shape)
        mcl.log_shape(emb)
        out = tf.layers.dense(emb, 1)
        mcl.log_shape(out)

        if "targets" in features:
            loss = tf.losses.mean_squared_error(tf.squeeze(features["targets"]),
                                                tf.squeeze(out))
        else:
            loss = 0.0

        return out, {"training": loss}
        """
    From original t2t resnet code
    

    # =================================
    out = tf.reduce_mean(out, [1, 2])
    logits = layers().Dense(1, name="logits")(out)
    # =================================


    if is_training:
      loss = tf.losses.sparse_softmax_cross_entropy(
          labels=tf.squeeze(targets), logits=logits)
      loss = tf.reduce_mean(loss)

      losses = {"training": loss}

    logits = tf.reshape(logits, [-1, 1, 1, 1, logits.shape[1]])

    return logits, losses
    
    """


from pcml.datasets import vox_celeb_cbt
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem


@registry.register_problem
class VoxCelebImageAudio(vox_celeb_cbt.VoxCelebCbt):

    @property
    def examples_table_name(self):
        return "vox-celeb-cbt-ex"

    def preprocess_example(self, example, mode, hparams):

        # Reduce the audio data to the required audio shape if it isn't
        example["audio"] = tf.slice(example["audio"], (0,),
                                    (self.audio_shape[0],))

        # Just a single frame
        example["video"] = tf.slice(example["video"], (0, 0, 0, 0),
                                    (1, -1, -1, -1))
        example["video"] = tf.squeeze(example["video"])

        example["targets"] = tf.random.uniform((1,),
                                               minval=0,
                                               maxval=5,
                                               dtype=tf.int32)

        return example

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.modality = {
            "video": modalities.ModalityType.IDENTITY,
            "audio": modalities.ModalityType.IDENTITY,
            "targets": modalities.ModalityType.IDENTITY
        }

        p.vocab_size = {"video": 256, "audio": 256, "targets": self.num_classes}

        p.batch_size_multiplier = 4
        p.loss_multiplier = 3.0
        if self._was_reversed:
            p.loss_multiplier = 1.0
        p.input_space_id = problem.SpaceID.IMAGE
        p.target_space_id = problem.SpaceID.IMAGE_LABEL


@registry.register_model
class MCLDev2(mcl.ModalityCorrespondenceLearner):

    def top(self, body_output, _):  # pylint: disable=no-self-use
        return body_output

    def eager_embed_single_image(self, image_data, **kwargs):
        del kwargs
        image_tensor = tf.convert_to_tensor(image_data)
        image_tensor = tf.expand_dims(image_tensor, 0)
        features = {"inputs": image_tensor}
        return tf.squeeze(self.embed_image(features)).numpy()

    @property
    def has_input(self):
        return True

    def embed_image(self, batched_image_tensor):
        hparams = self.hparams
        embedding = resnet_wrapper(hparams, batched_image_tensor)
        return embedding

    def body(self, features):

        hparams = self._hparams

        with tf.contrib.tpu.bfloat16_scope():

            emb = self.embed_image(features["video"])
            b, j, k, h = common_layers.shape_list(emb)
            shape = (b, j * k * h)
            emb = tf.reshape(emb, shape)
            out = tf.layers.dense(emb, 1)

        out = tf.cast(out, dtype=tf.float32)

        if "targets" in features:
            loss = tf.losses.mean_squared_error(tf.squeeze(features["targets"]),
                                                tf.squeeze(out))
        else:
            loss = 0.0

        return out, {"training": loss}


def _hack_make_non_dynamic(tensor):
    shape = common_layers.shape_list(tensor)
    return tf.reshape(tensor, shape)


def dense_reduction(x, target_size, reducing_factor=2, flatten=True):

    if flatten:
        b, j, k, h = common_layers.shape_list(x)
        shape = (b, j * k * h)
        x = tf.reshape(x, shape)

    input_size = tf.cast(common_layers.shape_list(x)[-1], tf.int32)

    sizes = []
    current_size = input_size

    while current_size % reducing_factor == 0:

        current_size *= 1 / reducing_factor

        if current_size < target_size:
            break

        sizes.append(int(current_size))

    # If we didn't find any reduction steps, return the input potentially with a
    # single reduction step
    if not sizes:
        x = tf.reshape(x, [-1, current_size])
        x = tf.layers.dense(x, target_size, name="reduce_last_a")
        return x

    # Apply the reduction steps
    for i, size in enumerate(sizes):
        x = tf.reshape(x, [-1, current_size])
        x = tf.layers.dense(x, size, name="reduce_{}".format(size))
        current_size = size

    if sizes[-1] > target_size:
        x = tf.reshape(x, [-1, current_size])
        x = tf.layers.dense(x, target_size, name="reduce_last_b")

    return x


def t2t_preprocess_waveforms(inputs, hparams):

    p = hparams

    num_mel_bins = p.audio_num_mel_bins
    num_channels = 3 if p.audio_add_delta_deltas else 1

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

    return x


@registry.register_model
class MCLDev3(mcl.ModalityCorrespondenceLearner):

    def top(self, body_output, _):  # pylint: disable=no-self-use
        return body_output

    def eager_embed_single_image(self, image_data, **kwargs):
        del kwargs
        image_tensor = tf.convert_to_tensor(image_data)
        image_tensor = tf.expand_dims(image_tensor, 0)
        image_tensor = tf.cast(image_tensor, tf.uint8)
        return tf.squeeze(self.embed_image(image_tensor)).numpy()

    def eager_embed_single_audio(self, d, **kwargs):
        del kwargs
        t = tf.convert_to_tensor(d)
        t = tf.expand_dims(t, 0)
        features = {"inputs": t}
        return tf.squeeze(self.embed_audio(features)).numpy()

    @property
    def has_input(self):
        return True

    @property
    def modality_embedding_size(self):
        return 512

    def _embed(self, tensor, embedding_size, dense_reduction_factor, hparams):

        embedding = resnet_wrapper(hp=hparams, inputs=tensor)

        embedding = dense_reduction(embedding,
                                    target_size=embedding_size,
                                    reducing_factor=dense_reduction_factor)

        embedding = tf.reshape(embedding, [-1, embedding_size])

        return embedding

    def embed_audio(self,
                    batched_feature_tensor,
                    embedding_size=64,
                    dense_reduction_factor=2):

        with tf.variable_scope("audio", reuse=tf.AUTO_REUSE):

            processed_waveforms = t2t_preprocess_waveforms(
                batched_feature_tensor, hparams=self.hparams)

            embedding = self._embed(
                processed_waveforms,
                embedding_size=embedding_size,
                dense_reduction_factor=dense_reduction_factor,
                hparams=self.hparams)

            return embedding

    def embed_image(self,
                    batched_image_tensor,
                    embedding_size=64,
                    dense_reduction_factor=2):

        with tf.variable_scope("video", reuse=tf.AUTO_REUSE):

            embedding = self._embed(
                batched_image_tensor,
                embedding_size=embedding_size,
                dense_reduction_factor=dense_reduction_factor,
                hparams=self.hparams)

            return embedding

    def body(self, features):

        hparams = self._hparams
        num_classes = 1
        """

    The current use of self.modality_embedding_size may not be ideal.
    Especially in the context of inference. wip.

    """

        with tf.contrib.tpu.bfloat16_scope():
            #with tf.variable_scope("dummy", reuse=tf.AUTO_REUSE):

            video_embedding = self.embed_image(
                features["video"], embedding_size=self.modality_embedding_size)

            audio_embedding = self.embed_audio(
                features["audio"], embedding_size=self.modality_embedding_size)

            concatenated = tf.concat(
                [tf.squeeze(video_embedding),
                 tf.squeeze(audio_embedding)],
                axis=1)
            concatenated = tf.reshape(concatenated,
                                      [-1, 2 * self.modality_embedding_size])

            prediction = dense_reduction(concatenated,
                                         target_size=num_classes,
                                         reducing_factor=2,
                                         flatten=False)

        prediction = tf.cast(prediction, dtype=tf.float32)

        if "targets" in features:
            loss = tf.losses.mean_squared_error(tf.squeeze(features["targets"]),
                                                tf.squeeze(prediction))
        else:
            loss = 0.0

        return prediction, {"training": loss}
