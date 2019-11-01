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

"""VGGFace2 problem.

TODO: What exactly is the learning problem?

"""

import os
import json

import tempfile

import cv2

import tensorflow as tf

from tensor2tensor.data_generators import generator_utils

from pcml.datasets.fec import _normalize_dimensions

import numpy as np

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils

from pcml.datasets.base import TripletImageProblem

from tensor2tensor.data_generators.problem import default_model_hparams

from pcml.datasets import image_aug

from tensor2tensor.data_generators import multi_problem_v2

from pcml.datasets.example_utils import ExampleTemplate
from pcml.datasets.example_utils import ExampleFieldTemplate
from pcml.datasets.utils import gen_dummy_schedule


def _load_vgg_meta(download_root, shuffle=True, mode="train"):

  if mode not in ["train", "test", "eval"]:
    raise ValueError("Unrecognized mode.")

  if mode == "eval":
    # Naming convention for this dataset. Test can be performed on a
    # completely separate dataset like LFW. Here we equate "test" and
    # "dev/eval".
    mode = "test"

  download_root = os.path.join(download_root, mode)
  meta_path = os.path.join(download_root, "meta.txt")

  tf.logging.info("Loaded metadata for {} from {}".format(
    mode, meta_path
  ))

  identity_to_paths = {}

  with tf.gfile.Open(meta_path, "r") as f:
    for line in f:
      line = line.strip()
      arr = line.split("/")
      identity = arr[0]
      if identity not in identity_to_paths:
        identity_to_paths[identity] = []
      identity_to_paths[identity].append(line)

  return identity_to_paths, download_root


def _vgg_sampling_iterator(meta):

  keys = list(meta.keys())
  keys_len = len(keys)

  while True:

    id1 = keys[np.random.randint(0,keys_len)]
    id2 = keys[np.random.randint(0,keys_len)]

    id1_images = meta[id1]
    id2_images = meta[id2]

    img1 = id1_images[np.random.randint(0,len(id1_images))]
    img2 = id1_images[np.random.randint(0,len(id1_images))]
    img3 = id2_images[np.random.randint(0,len(id2_images))]

    yield (img1, img2, img3)


def _read_image(image_path, image_shape, tmp_dir):

  fname = "-".join(image_path.split("/")[-2:])
  local_path = generator_utils.maybe_download(
    tmp_dir, fname, image_path)
  d = cv2.imread(local_path)
  d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
  d = _normalize_dimensions(d, image_shape)
  return d.flatten().tolist()


def _generator(data_root, tmp_dir, mode, how_many, image_shape,
               num_shards=-1, shard_id=-1):

  # Num shards and shard_id are not used because the dataset can be
  # sampled ad infinitum. The number of examples produced is simply
  # equal to `how_many` times the number of datagen jobs launched.

  meta, download_root = _load_vgg_meta(data_root, mode=mode)

  tf.logging.info("Geneating {} examples, shard_id {}, numshards {}".format(
    how_many, shard_id, num_shards
  ))

  for i, sample in enumerate(_vgg_sampling_iterator(meta)):

    if how_many and i > how_many:
      break

    def _mkpath(subpath):
      return os.path.join(download_root, subpath)

    paths = [_mkpath(subpath) for subpath in sample]

    ex = {

      "image/a": _read_image(paths[0], image_shape, tmp_dir),
      "image/b": _read_image(paths[1], image_shape, tmp_dir),
      "image/c": _read_image(paths[2], image_shape, tmp_dir),

      # For compatibility with FEC setup, maybe include a label
      # for rating/mode that is always 3 which according to the
      # scheme used in the FEC dataset corresponds to the first
      # two images in a pair belonging closer in embedding space
      # than those compared to the third.
      "triplet_code": [3],

      # These have identity and origin information in case this is needed
      # for debugging later.
      "path/a": paths[0],
      "path/b": paths[1],
      "path/c": paths[2]

    }
    
    yield ex


@registry.register_problem
class VggFace2(TripletImageProblem):

  @property
  def image_statistics(self):
    # Mean and standard deviation per color channel
    return {"mean": [0.508,0.578,0.233], "sd": [0.236,0.204,0.284]}

  @property
  def data_root(self):
    return "gs://clarify-data/requires-eula/vggface2"

  @property
  def image_shape(self):
    return (64, 64, 3)

  def _generator(self, data_root, tmp_dir, mode, how_many,
                 image_shape, num_shards, shard_id):
    return _generator(data_root, tmp_dir, mode, how_many,
                      image_shape, num_shards, shard_id)

  @property
  def train_size(self):
    # Here, the meaning of train_size is "the number of examples produced
    # each time data generation for this problem is run" which may be at high
    # multiplicity i.e. 1k per datagen replica. So if we want 1M examples we
    # just run launch 1k datagen replicas. In part because we are generating
    # examples by sampling there is no definitive total number of examples.
    return 1000

  @property
  def eval_size(self):
    return 1000

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
class VggFace2Tiny(VggFace2):

  @property
  def name_override(self):
    return "vgg_face2"

  @property
  def train_size(self):
    return 10

  @property
  def eval_size(self):
    return 10


@registry.register_problem
class VggFace2UdaFar(VggFace2):

  @property
  def name_override(self):
    return "vgg_face2"

  @property
  def uda_near_far(self):
    return "far"

  def preprocess_example(self, example, mode, unused_hparams):

    def _preproc(image):

      image = image_aug.preprocess_image(
        image, mode,
        resize_size=self.image_shape,
        normalize=self.normalize_image,
        image_statistics=self.image_statistics)

      image.set_shape(self.image_shape)

      return image

    img = example["image/a"]

    if self.uda_near_far == "far":
      example["image/c"] = _preproc(example["image/c"])
    elif self.uda_near_far == "near":
      example["image/c"] = _preproc(example["image/b"])
    else:
      raise ValueError("uda_near_far should be either near or far")

    example["image/a"] = _preproc(img)
    example["image/b"] = _preproc(img)

    example["triplet_code"] = tf.cast(example["triplet_code"], tf.int64)

    return example


@registry.register_problem
class VggFace2UdaNear(VggFace2UdaFar):

  @property
  def name_override(self):
    return "vgg_face2"

  @property
  def uda_near_far(self):
    return "near"

  
"""
class MultiProblemV3(multi_problem_v2.MultiProblemV2):

  def get_multi_dataset(datasets, pmf=None):
    pmf = tf.fill([len(datasets)], 1.0 / len(datasets)) if pmf is None else pmf
    samplers = []
    for d in datasets:
      d = d.repeat()
      sampler = tf.compat.v1.data.make_one_shot_iterator(d).get_next
      samplers.append(sampler)
    sample = lambda _: categorical_case(pmf, samplers)
    return tf.data.Dataset.from_tensors([]).repeat().map(sample)


@registry.register_problem
class VggFace2UdaMultiproblem(MultiProblemV3, VggFace2):

  def __init__(self, *args, **kwargs):

    problems = [
        VggFace2(),
        VggFace2UdaFar(),
        VggFace2UdaNear()
    ]

    schedule = gen_dummy_schedule(len(problems), num_steps=100)

    super(VggFace2UdaMultiproblem, self).__init__(
        problems=problems,
        schedule=schedule,
        *args, **kwargs)

  def normalize_example(self, example, _):
    return example

  def generate_data(self, *args, **kwargs):
    for i, p in enumerate(self.problems):
      p.generate_data(task_id=i, *args, **kwargs)
"""