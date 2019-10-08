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

"""DISFA problem definitions."""

import shutil
import tempfile
import os
import math
import json
import uuid
import pickle

import mne

import numpy as np
import pandas as pd

import tensorflow as tf

from tensor2tensor.data_generators import generator_utils

from pcml.datasets.utils import get_input_file_paths

from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

from tensor2tensor.layers import modalities


DEFAULT_DATA_ROOT = "gs://clarify-data/requires-eula/disfa/"

DISFA_AUS = ["1", "2", "4", "5", "6", "9", "12", "15", "20", "25", "26"]

DISFA_SUBJECT_IDS = [
  'SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008',
  'SN009', 'SN010', 'SN011', 'SN012', 'SN013', 'SN016', 'SN017', 'SN018',
  'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029',
  'SN030', 'SN031', 'SN032']


def maybe_get_disfa_data(tmp_dir,
                         remote_data_root,
                         is_training=True,
                         training_fraction=0.9):
  """Maybe download DISFA data.

  Args:
    tmp_dir(str): A local path where temporary files can
      be stored.
    remote_data_root(str): The root path to a directory
      containing raw data.

  """

  vleft_path_regex = os.path.join(DEFAULT_DATA_ROOT, "video_leftcamera/*")
  vleft_paths = get_input_file_paths(vleft_path_regex, is_training, training_fraction)
  vright_path_regex = os.path.join(DEFAULT_DATA_ROOT, "video_rightcamera/*")
  vright_paths = get_input_file_paths(vright_path_regex, is_training, training_fraction)

  for i, _ in enumerate(vleft_paths):
    vleft_path = vleft_paths[i]
    vright_path = vright_paths[i]
    subject_id = DISFA_SUBJECT_IDS[i]
    if subject_id not in vleft_path or subject_id not in vright_path:
      raise ValueError("Malformed remote data directories, use dir verifier to diagnose.")
    au_file_regex = os.path.join(remote_data_root, "au_labels/%s/*" % subject_id)
    au_file_paths = get_input_file_paths(au_file_regex, True, 1)

    vleft_fname = vleft_path.split("/")[-1]
    vleft_path_local = generator_utils.maybe_download(tmp_dir, vleft_fname, vleft_path)
    vright_fname = vright_path.split("/")[-1]
    vright_path_local = generator_utils.maybe_download(tmp_dir, vright_fname, vright_path)
    local_au_paths = []
    for au_file_path in au_file_paths:
      fname = au_file_path.split("/")[-1]
      local_au_path = generator_utils.maybe_download(tmp_dir, fname, au_file_path)
      local_au_paths.append(local_au_path)
    
    yield {"left_video": vleft_path_local,
           "right_video": vright_path_local,
           "au_labels": local_au_paths}


def _raw_data_verifier(remote_data_root):
  """Raw data verifier."""

  vleft_path_regex = os.path.join(remote_data_root, "video_leftcamera/*")
  vleft_paths = get_input_file_paths(vleft_path_regex, True, 1)
  vright_path_regex = os.path.join(remote_data_root, "video_rightcamera/*")
  vright_paths = get_input_file_paths(vright_path_regex, True, 1)
  if not len(vleft_paths) == 27 or not len(vright_paths) == 27:
    raise ValueError("Expected 27 of both left and right videos.")


def _load_disfa_aus(au_file_list):
  """Load DISFA AU's given a list of them on local disk."""

  au_data = []

  # Build an au to au file lookup
  for au_id in DISFA_AUS:
    suffix = "_au%s.txt" % au_id
    for file in au_file_list:
      data = []
      if file.endswith(suffix):
        with open(file, "r") as f:
          for line in f:
            au = line.strip().split(",")[1]
            data.append(int(au))
        au_data.append(data)

  return au_data


def avi_to_frame_array_iterator(avi_path):
  """Read an .avi video file as a numpy array of frames.
  
  Returns:
    A frame iterator (faster than reading all frames if we only
      want to fetch one, such as during development).

  """
  clip = VideoFileClip(avi_path)
  for frame in clip.iter_frames():
    yield frame
  #return [frame for frame in clip.iter_frames()]


def _tiled_subsample_example(example, subsample_width, subsample_step):

  if subsample_width <= 0 or subsample_step <= 0:
    raise ValueError("subsample width and step must be natural numbers")

  # TODO: Is there a way we can avoid reading the whole video in order
  # to be able to index into its frames using moviepy? Or should we just
  # do this? Currently the reader returns an iterator... Or perhaps we
  # could index into sub-regions of the clip using the moviepy syntax
  # for doing so. It's worth confirming frame numbers for the action unit
  # labels correspond to the number of frames in the videos.
    
  for i, _ in enumerate(example["labels/continuous/action_units"]):

    interval_start = i*subsample_step
    interval_end = interval_start + subsample_width
    subsampled_example = {}
    subsampled_labels = [thing[interval_start:interval_end] for thing in feature]
    subsampled_vleft = [thing for thing in feature][]
    subsampled_vright
    yield {
        "labels/continuous/action_units": subsampled_labels,
        "video/left": subsampled_vleft,
        "video/right": subsampled_vright
    }

    yield subsampled_example


def _generator(tmp_dir, subsample_width=100, subsample_step=10, how_many=None,
               is_training=True, training_fraction=0.9, vocab_size=256):
  """Generator for base training examples from the DISFA dataset.

  Notes:

    See: http://mohammadmahoor.com/disfa/ for more details.

  Args:
    tmp_dir(str): Directory to which to download raw data prior to
      processing into examples.
    subsample_width(int): Number of sensor measurements per sub-sample.
    subsample_step(int): Step size of sensor measurements between
      subsequent tiled sub-sampled, i.e. for width w and step s, we
      will index sub-intervals as follows: [(i-1)*s, (i-1)*s + w]
    how_many(int): How many examples should we generate?
    is_training(bool): Whether to generate examples from the
      `training_fraction` (or 1-`training_fraction`) portion of input
      training files.
    training_fraction(float): The fraction of input files to use for
      training.

  """
  ct = 0

  def _flatten(list_or_array):
    return np.asarray(list_or_array).flatten().tolist()

  data_path_iterator = maybe_get_disfa_data(
      tmp_dir, DEFAULT_DATA_ROOT, is_training=is_training,
      training_fraction=training_fraction)

  for subject_data_paths in data_path_iterator:
    vleft_path = subject_data_paths["left_video"]
    vright_path = subject_data_paths["right_video"]
    au_paths = subject_data_paths["au_labels"]
    au_data = _load_disfa_aus(au_paths)
    
    vleft_array_data = avi_to_frame_array_iterator(vleft_path)
    vright_array_data = avi_to_frame_array_iterator(vright_path)
    
    example = {
        "labels/continuous/action_units": au_data,
        "video/left": vleft_array_data,
        "video/right": vright_array_data
    }

    for subsampled in _tiled_subsample_example(
        example, subsample_width, subsample_step
      ):
        for key, value in subsampled.items():
          subsampled[key] = _flatten(value)
        subsampled["video/left/encoded"] = array2gif(subsampled["video/left"])
        subsampled.pop("video/left", None)
        subsampled["video/right/encoded"] = array2gif(subsampled["video/right"])
        subsampled.pop("video/right", None)
        yield subsampled
        ct += 1
        if isinstance(how_many, int) and ct >= how_many:
          return


@registry.register_problem
class DisfaProblemBase(problem.Problem):

  @property
  def problem_code(self):
    return 1234

  @property
  def trial_subsample_hparams(self):
    return {"width": 100, "step": 10}

  @property
  def training_fraction(self):
    return 0.9

  def generator(self, tmp_dir, max_nbr_cases, is_training):
    w, s = self.trial_subsample_hparams.values()
    return _generator(tmp_dir=tmp_dir,
                      subsample_width=w,
                      subsample_step=s,
                      how_many=max_nbr_cases,
                      is_training=is_training,
                      training_fraction=self.training_fraction)

  @property
  def train_size(self):
    return 100

  @property
  def dev_size(self):
    return 100

  @property
  def num_shards(self):
    return 1

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    generator_utils.generate_dataset_and_shuffle(
        self.generator(tmp_dir, self.train_size, is_training=True),
        self.training_filepaths(data_dir, self.num_shards, shuffled=True),
        self.generator(tmp_dir, self.dev_size, is_training=False),
        self.dev_filepaths(data_dir, 1, shuffled=True),
        shuffle=True)

  def feature_encoders(self, data_dir):
    del data_dir

    return {
      #"video": text_encoder.ImageEncoder(channels=self.video_shape[3]),
    }

  def example_reading_spec(self):

    video_shape = () # WIP
    num_channels = 3
    targets_shape = ()
    
    data_fields = {
      "video/left/encoded": tf.FixedLenFeature((), tf.string),
      "video/right/encoded": tf.FixedLenFeature((), tf.string),
      "frames/format": tf.FixedLenFeature((), tf.string),
      "targets": tf.FixedLenFeature(targets_shape, dtype=tf.int64),
    }

    data_items_to_decoders = {
      "video/left": tf.contrib.slim.tfexample_decoder.Image(
                image_key="video/left/encoded",
                format_key="frames/format",
                shape=video_shape,
                channels=num_channels),
      "video/right": tf.contrib.slim.tfexample_decoder.Image(
                image_key="video/right/encoded",
                format_key="frames/format",
                shape=video_shape,
                channels=num_channels),
      "targets": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="targets"),
    }

    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "video/left": modalities.IdentityModality,
        "video/right": modalities.IdentityModality,
        "targets": modalities.SymbolModality,
    }
    p.vocab_size = {
        "video/left": self.vocab_size,
        "video/right": self.vocab_size,
        "targets": self.vocab_size,
    }

  def preprocess_example(self, example, mode, hparams):
    return example
