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
"""Base problem for various PCML datasets."""

import re
import tempfile
import multiprocessing

import tensorflow as tf

from clarify.utils import cbt_utils

from tensor2tensor.data_generators import problem

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils

from tensor2tensor.data_generators.problem import default_model_hparams

from clarify.datasets.utils import image_aug


class PCMLProblem(problem.Problem):
  """Common base for PCML problems."""

  def dataset_selection(self):
    raise NotImplementedError()

  @property
  def name_override(self):
    # To be used when two problems should share the same CBT table.
    return None

  @property
  def examples_table_name(self):
    name = self.name
    if self.name_override:
      name = self.name_override
    base_name = re.sub("_", "-", name)
    if self.dataset_version_tag is not None:
      base_name += ("-" + self.dataset_version_tag)
    base_name += "-ex"
    return base_name

  # ----------------------------
  # HACK

  @property
  def dataset_version_tag(self):
    if not hasattr(self, '__dataset_version_tag'):
      self.__dataset_version_tag = None
    return self.__dataset_version_tag

  @dataset_version_tag.setter
  def dataset_version_tag(self, x):
    self.__dataset_version_tag = x

  # Probably make these settable
  @property
  def project(self):
    return "clarify"

  @property
  def cbt_instance(self):
    return "clarify"

  # ------------------------------

  def dataset_selection(self, mode):

    return cbt_utils.TFExampleSelection(project=self.project,
                                        instance=self.cbt_instance,
                                        table=self.examples_table_name,
                                        prefix=mode)

  def cbt_generate(self, project, instance, mode, tmp_dir=None, shard_id=-1):

    if not tmp_dir:
      tmp_dir = tempfile.mkdtemp()

    gen = self.generator(tmp_dir, shard_id=shard_id)

    target_selection = self.dataset_selection(mode=mode)

    target_selection.random_load_from_generator(generator=gen,
                                                log_every=100,
                                                prefix_tag_length=8)

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

    tf.logging.info("Validating selection: {}".format(selection.as_dict()))

    # This not only checks that it's nonempty but verifies the trainer VM
    # has the credentials to access the table. Before we go any further into
    # building a graph and surfacing an error to those effects via traditional
    # python instead of via the bigquery client running inside of graph mode.
    if not selection.rows_at_least(10):
      msg = "Selection appears to be empty, please check configuration, {}".format(
          selection.as_dict())
      raise ValueError(msg)
      # So at least in this case you don't just see the "no more retries"
      # error that is what is shown in this condition of we proceed to
      # working with this empty table by way of tf.contrib.cloud.BigTableClient

    # This shouldn't be necessary because if it's an unexpected mode then
    # the assert nonempty assertion will be false
    expected_modes = ["train", "eval", "test"]
    if mode not in expected_modes:
      raise ValueError("Expected mode in {}, saw {}.".format(
          mode, expected_modes))

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
          project_id=selection.project, instance_id=selection.instance_name)

      table = bigtable_client.table(selection.table_name)

      dataset = table.parallel_scan_prefix(mode,
                                           columns=[(selection.column_family,
                                                     selection.column_qualifier)
                                                   ])

      dataset = dataset.map(lambda index, data: data)

      dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)

      if preprocess:
        dataset = self.preprocess(dataset,
                                  mode,
                                  hparams,
                                  interleave=shuffle_files)

      dataset = dataset.map(self.maybe_reverse_and_copy,
                            num_parallel_calls=num_threads)

      dataset = dataset.take(max_records)

      # HACK
      output_buffer_size = 8
      if output_buffer_size:
        dataset = dataset.prefetch(output_buffer_size)

    return dataset


class TripletImageProblem(PCMLProblem):

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
  def data_root(self):
    raise NotImplementedError()

  @property
  def stored_image_shape(self):
    return (128, 128, 3)

  @property
  def identity_shape(self):
    return ()

  @property
  def train_size(self):
    return None

  @property
  def eval_size(self):
    return None

  def example_reading_spec(self):

    image_shape = self.stored_image_shape

    data_fields = {
        "image/a": tf.FixedLenFeature(image_shape, dtype=tf.int64),
        "image/b": tf.FixedLenFeature(image_shape, dtype=tf.int64),
        "image/c": tf.FixedLenFeature(image_shape, dtype=tf.int64),
        "triplet_code": tf.FixedLenFeature((), dtype=tf.int64),
    }

    data_items_to_decoders = {
        "image/a":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/a"),
        "image/b":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/b"),
        "image/c":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="image/c"),
        "triplet_code":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="triplet_code"),

        # Dummy
        "targets":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="triplet_code"),
    }

    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):

    p = defaults
    p.modality = {
        "image/a": modalities.ModalityType.IDENTITY,
        "image/b": modalities.ModalityType.IDENTITY,
        "image/c": modalities.ModalityType.IDENTITY,
        "triplet_code": modalities.ModalityType.IDENTITY,
        "targets": modalities.ModalityType.IDENTITY
    }

    p.vocab_size = {
        "image/a": 256,
        "image/b": 256,
        "image/c": 256,
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
  def image_shape(self):
    # Image shape to be used for training and eval; that which will be
    # produced from preprocess_example.
    raise NotImplementedError()

  @property
  def normalize_image(self):
    """Whether the image should be normalized in preprocessing."""
    return True

  def preprocess_example(self, example, mode, unused_hparams):

    def _preproc(image):

      image = image_aug.preprocess_image(image,
                                         mode,
                                         resize_size=self.image_shape,
                                         normalize=self.normalize_image,
                                         image_statistics=self.image_statistics)

      image.set_shape(self.image_shape)

      return image

    example["image/a"] = _preproc(example["image/a"])
    example["image/b"] = _preproc(example["image/b"])
    example["image/c"] = _preproc(example["image/c"])
    example["triplet_code"] = tf.cast(example["triplet_code"], tf.int64)

    return example

  @property
  def train_shards(self):
    return 1000

  @property
  def dev_shards(self):
    return 10

  @property
  def test_shards(self):
    return 10

  def _generator(self, data_root, tmp_dir, mode, how_many, image_shape,
                 num_shards, shard_id):
    raise NotImplementedError()

  def generator(self, tmp_dir, shard_id=-1):

    data_root = self.data_root

    how_many = self.train_size
    num_shards = self.train_shards
    if self.mode == "eval":
      how_many = self.eval_size
      num_shards = self.dev_shards

    tf.logging.info("Generating for mode {}".format(self.mode))

    return self._generator(data_root=data_root,
                           tmp_dir=tmp_dir,
                           mode=self.mode,
                           how_many=how_many,
                           image_shape=self.stored_image_shape,
                           num_shards=num_shards,
                           shard_id=shard_id)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    gen = self.generator(tmp_dir, shard_id=task_id)

    if self.mode == "train":
      paths = self.training_filepaths(data_dir,
                                      self.train_shards,
                                      shuffled=False)
      paths = sharded_subset_list(paths, self.train_shards, task_id)
    else:
      paths = self.test_filepaths(data_dir, self.test_shards, shuffled=False)
      paths = sharded_subset_list(paths, self.test_shards, task_id)

    generator_utils.generate_files(gen, paths)

    generator_utils.shuffle_dataset(paths)
