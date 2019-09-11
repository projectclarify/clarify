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

"""Additional distributed datagen and augmentation problem defs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import tempfile
import math
import re

from tensor2tensor.data_generators.problem import default_model_hparams

from tensor2tensor.data_generators import problem

from tensor2tensor.utils import registry

import multiprocessing
from multiprocessing import Pool, TimeoutError
from tensor2tensor.data_generators import generator_utils
import numpy as np

from pcml.utils import augmentation_utils

from pcml.utils import cbt_utils
from pcml.utils import audio_utils

from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem

VOX_CELEB_ROOT = "gs://clarify-data/requires-eula/voxceleb2"

import os


def get_manifest_lookup(vox_celeb_root=VOX_CELEB_ROOT):
  return {
    "train": "{}/dev-paths.txt".format(vox_celeb_root),
    "eval": "{}/test-paths.txt".format(vox_celeb_root),
    "test": "{}/veri-paths.txt".format(vox_celeb_root)
  }


def standardize_audio_array(audio, audio_shape):  
  pad_size = audio_shape[0] - len(audio)
  if pad_size > 0:
    return np.concatenate([audio, np.random.uniform(-0.5, 0.5, (pad_size,))])
  else:
    return audio[0:audio_shape[0]]


def example_generator(raw_sampler,
                      video_shape,
                      audio_shape,
                      max_examples,
                      augmentation_hparams,
                      skip_subtypes=[]):

  ct = 0

  for raw_sample in raw_sampler:

    for sample_subtype in raw_sample:
      
      if sample_subtype in skip_subtypes:
        continue

      if isinstance(max_examples, int) and ct > max_examples:
        break

      video = np.reshape(raw_sample[sample_subtype].video,
                         video_shape)

      # Is this necessary? If so need to be able to look up the audio
      # shape which is with the current setup more of a side effect of
      # the video shape.
      #audio = np.reshape(raw_sample[sample_subtype].audio,
      #                   audio_shape)
      audio = raw_sample[sample_subtype].audio

      video = augmentation_utils.augment_video(
        video=video,
        **augmentation_hparams["video"]
      ).astype(np.float32)

      video = (video - 128.0)/255.0

      audio = augmentation_utils.augment_audio(
        audio=audio,
        **augmentation_hparams["audio"],
        data_range=[0,255]
      ).astype(np.float32)

      audio = (audio - 128.0)/255.0

      overlap = raw_sample[sample_subtype].labels["overlap"]
      #same_video = raw_sample[sample_subtype].labels["same_video"]

      audio = standardize_audio_array(
        audio=audio, audio_shape=audio_shape)

      yield {
        "audio": audio.tolist(),
        "video": video.flatten().tolist(),
        #"targets": [int(same_video), int(overlap)]
        "targets": [int(overlap)]
      }

      ct += 1


@registry.register_problem
class VoxCelebCbt(problem.Problem):

  @property
  def num_classes(self):
    # The length of the targets vector
    return 1

  @property
  def video_shape(self):
    return [20, 96, 96, 3]

  @property
  def vocab_size(self):
    return 256

  @property
  def audio_mel_bins(self):
    return 32
  
  @property
  def audio_shape(self):
    return [35757]

  @property
  def generate_max_examples(self):
    return None

  def sampling_generator(self, source_selection):

    sample_generator = source_selection.sample_av_correspondence_examples(
      frames_per_video=self.video_shape[0])

    generator = example_generator(
      raw_sampler=sample_generator,
      video_shape=self.video_shape,
      audio_shape=self.audio_shape,
      max_examples=self.max_examples,
      augmentation_hparams=self.augmentation_hparams
    )

    return generator

  @property
  def augmentation_hparams(self):
    """

    Notes:
    * Turned off subsampling operations given currently doing these at the
      raw sampling level and doing them here complicates things because it
      changes the feature shapes (i.e. their length).

    """

    return {
      "video": {
        "do_random_subsampling": False,
        "do_random_flips": True,
        "do_random_masking": True,
        "do_random_enhancement": True,
        "do_random_shift": True,
        "subsample_max_frame_skips": 1,
        "shift_max_xy": 5,
        "mask_num_patches": 10,
        "mask_max_patch_fraction": 0.2,
        "enhance_min_color": 0.25,
        "enhance_max_color": 1.0,
        "enhance_min_contrast": 0.25,
        "enhance_max_contrast": 1.0,
        "enhance_min_brightness": 0.25,
        "enhance_max_brightness": 1.0,
        "enhance_min_sharpness": 0.25,
        "enhance_max_sharpness": 1.75
      },
      "audio": {
        "do_random_shift": False,
        "do_add_gaussian_noise": True,
        "gaussian_snr": 10
      }
    }

  def preprocess_example(self, example, mode, hparams):
    # For whatever reason not having any ops in preprocess_example causes the example
    # prefetching to run forever (or at least longer than I waited around) without
    # proceeding to training; much longer than when this presumably trivial op is applied
    # given here we're slicing the audio to its current full length
    example["audio"] = tf.slice(example["audio"], (0,), (self.audio_shape[0],))
    return example

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=multiprocessing.cpu_count(),
              output_buffer_size=None,
              shuffle_files=False,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None,
              partition_id=0,
              num_partitions=1,
              shuffle_buffer_size=16,
              max_records=-1,
              mode_override=None):

    # HACK: Don't shuffle files
    shuffle_files = False

    # HACK
    # Model predicts NaN's in eval mode because the model is differently configured
    # in eval mode, not because it's operating on eval instead of training data. It's
    # important to know exactly what part of this model configuration has this effect
    # but until then inference can be performed in training mode with eval data by
    # specifying this `mode_override` parameter when instantiating the tf.data dataset.
    if mode_override is not None:
      mode = mode_override

    selection = self.dataset_selection(mode=mode)

    if not isinstance(selection, cbt_utils.TFExampleSelection):
      msg = "Expected a TFExampleSelection saw {}".format(type(selection))
      raise ValueError(msg)

    tf.logging.info("Validating selection: {}".format(
      selection.as_dict()
    ))

    # This not only checks that it's nonempty but verifies the trainer VM
    # has the credentials to access the table. Before we go any further into
    # building a graph and surfacing an error to those effects via traditional
    # python instead of via the bigquery client running inside of graph mode.
    if not selection.rows_at_least(10):
      msg = "Selection appears to be empty, please check configuration, {}".format(
        selection.as_dict()
      )
      raise ValueError(msg)
      # So at least in this case you don't just see the "no more retries"
      # error that is what is shown in this condition of we proceed to
      # working with this empty table by way of tf.contrib.cloud.BigTableClient

    # This shouldn't be necessary because if it's an unexpected mode then
    # the assert nonempty assertion will be false
    expected_modes = ["train", "eval", "test"]
    if mode not in expected_modes:
      raise ValueError("Expected mode in {}, saw {}.".format(
        mode, expected_modes
      ))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if hparams is None:
      hparams = default_model_hparams()

    if not hasattr(hparams, "data_dir"):
      hparams.add_hparam("data_dir", data_dir)
    if not hparams.data_dir:
      hparams.data_dir = data_dir
    # Construct the Problem's hparams so that items within it are accessible
    _ = self.get_hparams(hparams)

    with tf.device("CPU"):

      bigtable_client = tf.contrib.cloud.BigtableClient(
          project_id=selection.project,
          instance_id=selection.instance_name)

      table = bigtable_client.table(selection.table_name)

      dataset = table.parallel_scan_prefix(
          mode,
          columns=[(selection.column_family,
                    selection.column_qualifier)]
      )

      dataset = dataset.map(lambda index, data: data)

      dataset = dataset.map(self.decode_example,
                            num_parallel_calls=num_threads)

      if preprocess:
        dataset = self.preprocess(dataset, mode, hparams,
                                  interleave=shuffle_files)

      dataset = dataset.map(
          self.maybe_reverse_and_copy, num_parallel_calls=num_threads)

      dataset = dataset.take(max_records)

      # HACK
      output_buffer_size = 8
      if output_buffer_size:
        dataset = dataset.prefetch(output_buffer_size)

    return dataset

  @property
  def raw_table_name(self):
    # We'll want to leave the raw table name fixed most of the time and
    # change the examples table name every time we try a different problem
    # variant.
    return "vox-celeb-2-raw"

  @property
  def dataset_version_tag(self):
    return None

  @property
  def examples_table_name(self):
    base_name = re.sub("_", "-", self.name)
    if self.dataset_version_tag is not None:
      base_name +=  ("-" + self.dataset_version_tag)
    base_name += "-ex"
    return base_name

  def dataset_selection(self, mode):
    # HACK/TODO: look these up from environment variables.
    return cbt_utils.TFExampleSelection(
      project="clarify",
      instance="clarify",
      table=self.examples_table_name,
      prefix=mode
    )

  def example_reading_spec(self):
    
    data_fields = {
      "audio": tf.VarLenFeature(dtype=tf.float32),
      "video": tf.FixedLenFeature(self.video_shape, dtype=tf.float32),
      "targets": tf.FixedLenFeature([self.num_classes], dtype=tf.int64),
    }

    return data_fields, None

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "audio": "IdentityModality",
        "video": "IdentityModality",
        "targets": "IdentityModality",
    }
    p.vocab_size = {
        "audio": self.vocab_size,
        "video": self.vocab_size,
        "targets": self.vocab_size,
    }

  def eval_metrics(self):
    eval_metrics = [
       metrics.Metrics.ACC
    ]
    return eval_metrics

  def mode_to_manifest_lookup(self):
    return get_manifest_lookup()

  @property
  def max_examples(self):
    return None

  def dataset_filename(self):
    return "vox_celeb_problem"


@registry.register_problem
class VoxCelebSingleFrame(VoxCelebCbt):

  @property
  def video_shape(self):
    return [1, 96, 96, 3]

  def preprocess_example(self, example, mode, hparams):

    # Reduce the audio data to the required audio shape if it isn't
    example["audio"] = tf.slice(example["audio"], (0,), (self.audio_shape[0],))

    example["audio"].set_shape((self.audio_shape[0], ))
    example["video"].set_shape(self.video_shape)

    example["video"] = tf.squeeze(example["video"])

    return example

  def sampling_generator(self, source_selection):

    sample_generator = source_selection.sample_av_correspondence_examples(
      frames_per_video=self.video_shape[0])

    generator = example_generator(
      raw_sampler=sample_generator,
      video_shape=self.video_shape,
      audio_shape=self.audio_shape,
      max_examples=self.max_examples,
      augmentation_hparams=self.augmentation_hparams,
      skip_subtypes=["negative_different"]
    )

    return generator

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"video": modalities.ModalityType.IDENTITY,
                  "audio": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}

    p.vocab_size = {"video": 256,
                    "audio": 256,
                    "targets": self.num_classes}

    p.batch_size_multiplier = 4
    p.loss_multiplier = 3.0
    if self._was_reversed:
      p.loss_multiplier = 1.0
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE_LABEL
