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
import tempfile
import re
import multiprocessing
from tensor2tensor.data_generators.problem import default_model_hparams

import numpy as np

import cv2

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils

from pcml.utils import cbt_utils
from pcml.datasets.base import TripletImageProblem


def _load_fec_meta_post_download(download_root, shuffle=True, mode="train"):
    """Load metadata for downloaded FEC examples.
  
  * If a meta.json file is not present in `download_root`/`mode`, produce
    one, otherwise just read the one that's there.
  
  """

    modes = tf.gfile.ListDirectory(download_root)
    if (mode + "/") not in modes:
        raise ValueError(
            "Could not find expected mode in modes, {}".format(mode))

    unsharded_meta = []

    mode_root = os.path.join(download_root, mode)
    dataset_meta_path = os.path.join(mode_root, "meta.json")
    if tf.gfile.Exists(dataset_meta_path):
        # read it
        with tf.gfile.Open(dataset_meta_path, "r") as f:
            meta = json.loads(f.read())
        return meta

    # Otherwise read the meta for each individual shard and compile an single
    # meta.json for the whole mode subset (i.e. the train or test split of the
    # dataset).

    mode_shards = tf.gfile.ListDirectory(mode_root)
    for mode_shard in mode_shards:

        print(mode_shard)

        mode_shard_dir = os.path.join(mode_root, mode_shard)
        mode_shard_files = tf.gfile.ListDirectory(mode_shard_dir)
        if "nonfailed.json" not in mode_shard_files:
            raise ValueError("Mode shard files should contain nonfailed.json")

        with tf.gfile.Open(os.path.join(mode_shard_dir, "nonfailed.json")) as f:
            shard_meta = json.loads(f.read())

        for shard_meta_element in shard_meta:

            shard_meta_element["a"]["full_remote_path"] = os.path.join(
                mode_shard_dir, shard_meta_element["a"]["cropped_filename"])

            shard_meta_element["b"]["full_remote_path"] = os.path.join(
                mode_shard_dir, shard_meta_element["b"]["cropped_filename"])

            shard_meta_element["c"]["full_remote_path"] = os.path.join(
                mode_shard_dir, shard_meta_element["c"]["cropped_filename"])

            unsharded_meta.append(shard_meta_element)

    random.shuffle(unsharded_meta)

    with tf.gfile.Open(dataset_meta_path, "w") as f:
        f.write(json.dumps(unsharded_meta))

    return unsharded_meta


def _random_crop_square(image):

    x, y, c = image.shape

    x_crop_before = 0
    x_crop_after = 0
    y_crop_before = 0
    y_crop_after = 0

    if x > y:
        x_crop = x - y
        x_crop_before = np.random.randint(0, x_crop)
        x_crop_after = x_crop - x_crop_before
    elif y > x:
        y_crop = y - x
        y_crop_before = np.random.randint(0, y_crop)
        y_crop_after = y_crop - y_crop_before

    x_start = x_crop_before
    x_end = x - x_crop_after
    y_start = y_crop_before
    y_end = y - y_crop_after

    return image[x_start:x_end, y_start:y_end, :]


def _normalize_dimensions(image, target_shape):

    image = _random_crop_square(image)

    mn, mx = np.amin(image), np.amax(image)
    if mn >= 0 and mx <= 255:
        image = image / 255.0

    source_shape = image.shape
    scale_x_factor = target_shape[0] / source_shape[0]
    scale_y_factor = target_shape[1] / source_shape[1]
    scale_x_first = (scale_x_factor <= scale_y_factor)

    if scale_x_first:

        new_x = target_shape[0]
        new_y = int(source_shape[1] * scale_x_factor)
        resize_dim = (new_x, new_y)
        newimg = cv2.resize(image, resize_dim)
        pad_width = target_shape[1] - new_y
        if pad_width > 0:
            # Pad in Y direction
            newimg = np.pad(newimg, [(0, pad_width), (0, 0), (0, 0)],
                            mode="mean")

    else:

        new_y = target_shape[1]
        new_x = int(source_shape[0] * scale_y_factor)
        resize_dim = (new_x, new_y)
        newimg = cv2.resize(image, resize_dim)
        pad_width = target_shape[0] - new_x
        if pad_width > 0:
            # Pad in X direction
            newimg = np.pad(newimg, [(0, 0), (0, pad_width), (0, 0)],
                            mode="mean")

    newimg = (newimg * 255.0).astype(np.int64)

    return newimg


def sharded_subset_list(l, num_shards, shard_id):

    if num_shards <= 0 or shard_id <= 0:
        return l

    shard_size = int(len(l) / num_shards)
    shard_offset = shard_size * shard_id
    return l[shard_offset:(shard_offset + shard_size)]


def _read_image(image_path, image_shape, tmp_dir):

    fname = image_path.split("/")[-1]
    local_path = generator_utils.maybe_download(tmp_dir, fname, image_path)
    d = cv2.imread(local_path)
    d = _normalize_dimensions(d, image_shape)
    return d.flatten().tolist()


def fec_generator(fec_data_root,
                  tmp_dir,
                  mode,
                  image_shape,
                  how_many=None,
                  num_shards=-1,
                  shard_id=-1):

    meta = _load_fec_meta_post_download(fec_data_root, mode=mode)

    meta = sharded_subset_list(meta, num_shards, shard_id)

    # For the specified number of examples, loop over that metadata
    for i, entry in enumerate(meta):

        if how_many and i > how_many:
            break

        path_a = entry["a"]["full_remote_path"]
        path_b = entry["b"]["full_remote_path"]
        path_c = entry["c"]["full_remote_path"]

        def _maybe_download_and_read(path):
            fname = path.split("/")[-1]
            local_path_a = generator_utils.maybe_download(tmp_dir, fname, path)
            d = cv2.imread(local_path_a)
            d = _normalize_dimensions(d, image_shape)
            return d.flatten().tolist()

        def _type_to_code(t):
            if t == "ONE_CLASS_TRIPLET":
                return 1
            elif t == "TWO_CLASS_TRIPLET":
                return 2
            elif t == "THREE_CLASS_TRIPLET":
                return 3
            else:
                raise ValueError()

        try:
            # Then constructing an example dictionary for the triplet and yielding it
            ex = {
                "image/a": _maybe_download_and_read(path_a),
                "image/b": _maybe_download_and_read(path_b),
                "image/c": _maybe_download_and_read(path_c),
                "type": [_type_to_code(entry["triplet_type"])],
                "rating/mean": [float(entry["mean_rating"])],
                "rating/mode": [int(entry["mode_rating"])],
                "rating/all": [json.dumps(entry["ratings"])],
            }

            yield ex

        # HACK: Currently there is an error where occasionally it will look for a file on GCS that
        # is not but should be there...
        except:
            pass


@registry.register_problem
class FacialExpressionCorrespondence(TripletImageProblem):
    """Facial expression correspondence problem"""

    @property
    def image_statistics(self):
        # Mean and standard deviation per color channel
        return {"mean": [0.330, 0.537, -0.242], "sd": [0.220, 0.169, 1.156]}

    @property
    def data_root(self):
        return "gs://clarify-data/fec/dev"

    @property
    def image_shape(self):
        return (64, 64, 3)

    def _generator(self, data_root, tmp_dir, mode, how_many, image_shape,
                   num_shards, shard_id):
        return fec_generator(data_root, tmp_dir, mode, image_shape, how_many,
                             num_shards, shard_id)

    def example_reading_spec(self):

        targets_shape = ()
        image_shape = self.stored_image_shape

        data_fields = {
            "image/a": tf.FixedLenFeature(image_shape, dtype=tf.int64),
            "image/b": tf.FixedLenFeature(image_shape, dtype=tf.int64),
            "image/c": tf.FixedLenFeature(image_shape, dtype=tf.int64),
            "type": tf.FixedLenFeature((), dtype=tf.int64),
            "rating/mode": tf.FixedLenFeature(targets_shape, dtype=tf.int64),
        }

        data_items_to_decoders = {
            "image/a":
                tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/a"),
            "image/b":
                tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/b"),
            "image/c":
                tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/c"),
            "type":
                tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="type"),
            "triplet_code":
                tf.contrib.slim.tfexample_decoder.Tensor(
                    tensor_key="rating/mode"),
            "targets":
                tf.contrib.slim.tfexample_decoder.Tensor(
                    tensor_key="rating/mode"),
        }

        return data_fields, data_items_to_decoders

    def hparams(self, defaults, unused_model_hparams):

        p = defaults
        p.modality = {
            "image/a": modalities.ModalityType.IDENTITY,
            "image/b": modalities.ModalityType.IDENTITY,
            "image/c": modalities.ModalityType.IDENTITY,
            "type": modalities.ModalityType.IDENTITY,
            "triplet_code": modalities.ModalityType.IDENTITY,
            "targets": modalities.ModalityType.IDENTITY
        }

        p.vocab_size = {
            "image/a": 256,
            "image/b": 256,
            "image/c": 256,
            "type": 3,
            "triplet_code": self.num_classes,
            "targets": self.num_classes
        }

        p.batch_size_multiplier = 4
        p.loss_multiplier = 3.0
        if self._was_reversed:
            p.loss_multiplier = 1.0
        p.input_space_id = problem.SpaceID.IMAGE
        p.target_space_id = problem.SpaceID.IMAGE_LABEL

    @property
    def train_shards(self):
        return 1000

    @property
    def dev_shards(self):
        return 10

    @property
    def test_shards(self):
        return 10

    """
  # For computing dataset statistics
  @property
  def train_size(self):
    return 3200
  
  @property
  def normalize_image(self):
    return False
  """


@registry.register_problem
class FecTiny(FacialExpressionCorrespondence):

    @property
    def name_override(self):
        return "facial_expression_correspondence"

    @property
    def train_size(self):
        return 10

    @property
    def eval_size(self):
        return 10
