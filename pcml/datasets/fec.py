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

"""FEC problem definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import json
import os

import numpy as np

import cv2

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils


def _load_fec_meta_post_download(download_root, shuffle=True, mode="train"):

  modes = tf.gfile.ListDirectory(download_root)
  if (mode + "/") not in modes:
    raise ValueError("Could not find expected mode in modes, {}".format(
      mode
    ))

  unsharded_meta = []

  mode_root = os.path.join(download_root, mode)
  mode_shards = tf.gfile.ListDirectory(mode_root)
  for mode_shard in mode_shards:
    mode_shard_dir = os.path.join(mode_root, mode_shard)
    mode_shard_files = tf.gfile.ListDirectory(mode_shard_dir)
    if "nonfailed.json" not in mode_shard_files:
      raise ValueError("Mode shard files should contain nonfailed.json")

    with tf.gfile.Open(os.path.join(mode_shard_dir, "nonfailed.json")) as f:
      shard_meta = json.loads(f.read())

    for shard_meta_elemet in shard_meta:

      shard_meta_elemet["a"]["full_remote_path"] = os.path.join(
        mode_shard_dir, shard_meta_elemet["a"]["cropped_filename"])

      shard_meta_elemet["b"]["full_remote_path"] = os.path.join(
        mode_shard_dir, shard_meta_elemet["b"]["cropped_filename"])

      shard_meta_elemet["c"]["full_remote_path"] = os.path.join(
        mode_shard_dir, shard_meta_elemet["c"]["cropped_filename"])

      unsharded_meta.append(shard_meta_elemet)

  random.shuffle(unsharded_meta)

  return unsharded_meta


def fec_generator(fec_data_root,
                  tmp_dir,
                  training,
                  how_many=None):

  mode = "train"
  if not training:
    mode = "eval"

  meta = _load_fec_meta_post_download(fec_data_root, mode=mode)

  # For the specified number of examples, loop over that metadata
  for i, entry in enumerate(meta):

    if how_many and i > how_many:
      break

    path_a = entry["a"]["full_remote_path"]
    path_b = entry["b"]["full_remote_path"]
    path_c = entry["c"]["full_remote_path"]

    def _maybe_download_and_read(path):
      fname = path.split("/")[-1]
      local_path_a = generator_utils.maybe_download(
        tmp_dir, fname, path)
      return cv2.imread(local_path_a).astype(np.int64).flatten().tolist()

    # Then constructing an example dictionary for the triplet and yielding it
    yield {
        "image/a": _maybe_download_and_read(path_a),
        "image/b": _maybe_download_and_read(path_b),
        "image/c": _maybe_download_and_read(path_c),
        "type": [entry["triplet_type"]],
        "rating/mean": [float(entry["mean_rating"])],
        "rating/mode": [int(entry["mode_rating"])],
        "rating/all": [json.dumps(entry["ratings"])]
      }


@registry.register_problem
class FacialExpressionCorrespondence(image_utils.Image2ClassProblem):
  """Facial expression correspondence problem"""

  @property
  def is_small(self):
    return True

  @property
  def num_classes(self):
    return 3

  @property
  def train_shards(self):
    return 10

  @property
  def fec_data_root(self):
    return "gs://clarify-data/fec/dev"

  @property
  def image_shape(self):
    return (64, 64, 3)

  def example_reading_spec(self):

    targets_shape = ()
    image_shape = self.image_shape

    data_fields = {
      "image/a": tf.FixedLenFeature(image_shape, dtype=tf.int64),
      "image/b": tf.FixedLenFeature(image_shape, dtype=tf.int64),
      "image/c": tf.FixedLenFeature(image_shape, dtype=tf.int64),
      "frames/format": tf.FixedLenFeature((), tf.string),
      "targets": tf.FixedLenFeature(targets_shape, dtype=tf.int64),
    }

    data_items_to_decoders = {
      "image/a": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/a"),
      "image/b": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/b"),
      "image/c": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/c"),
      "targets": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="targets"),
    }

    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):

    p = defaults
    p.modality = {"image/a": modalities.ModalityType.IDENTITY,
                  "image/b": modalities.ModalityType.IDENTITY,
                  "image/c": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}

    p.vocab_size = {"image/a": 256,
                    "image/b": 256,
                    "image/c": 256,
                    "targets": self.num_classes}

    p.batch_size_multiplier = 4
    p.loss_multiplier = 3.0
    if self._was_reversed:
      p.loss_multiplier = 1.0
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE_LABEL

  def preprocess_example(self, example, mode, unused_hparams):

    def _preproc(image):
      image.set_shape(self.image_shape)
      if not self._was_reversed:
        iamge = tf.image.per_image_standardization(image)
      return image

    example["image/a"] = _preproc(example["image/a"])
    example["image/b"] = _preproc(example["image/b"])
    example["image/c"] = _preproc(example["image/c"])

    return example

  def generator(self, data_dir, tmp_dir, is_training):
    del data_dir
    fec_data_root=self.fec_data_root
    if is_training:
      return fec_generator(
              fec_data_root=fec_data_root,
              tmp_dir=tmp_dir,
              training=is_training,
              how_many=5000
            )
    else:
      return fec_generator(
              fec_data_root=fec_data_root,
              tmp_dir=tmp_dir,
              training=is_training,
              how_many=5000
            )


@registry.register_problem
class FecTiny(FacialExpressionCorrespondence):
  pass
