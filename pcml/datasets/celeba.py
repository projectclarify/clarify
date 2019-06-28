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

"""CelebA problem definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry
from tensor2tensor.layers import modalities

from tensor2tensor.data_generators.celeba import ImageCeleba
from tensor2tensor.data_generators import generator_utils


@registry.register_problem
class ImageCelebaPcml(ImageCeleba):

  @property
  def example_split_config(self):
    return {"train": (162770, 0),
            "dev": (19867, 162770),
            "test": (19962, 162770+19867)}

  @property
  def image_dim(self):
    return 64

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    esc = self.example_split_config
    train_gen = self.generator(tmp_dir, esc["train"][0], esc["train"][1])
    train_paths = self.training_filepaths(
        data_dir, self.train_shards, shuffled=False)
    generator_utils.generate_files(train_gen, train_paths)

    dev_gen = self.generator(tmp_dir, esc["dev"][0], esc["dev"][1])
    dev_paths = self.dev_filepaths(data_dir, self.dev_shards, shuffled=False)
    generator_utils.generate_files(dev_gen, dev_paths)

    test_gen = self.generator(tmp_dir, esc["test"][0], esc["test"][1])
    test_paths = self.test_filepaths(data_dir, self.test_shards, shuffled=False)
    generator_utils.generate_files(test_gen, test_paths)

    generator_utils.shuffle_dataset(train_paths + dev_paths + test_paths)

  def preprocess_example(self, example, mode, hparams):
    inputs = example["inputs"]
    example["image"] = image_utils.resize_by_area(inputs, self.image_dim)
    example.pop("inputs", None)
   
    example.pop("landmarks", None)
    example.pop("attributes", None)

    # I think there might be 10 instead of 12 landmarks?
    example["targets"] = tf.pad(example["targets"], tf.constant([[0,2]]))

    # HACK: RECENT ADDITION ===================
    example["image"] = (tf.cast(example["image"], tf.float32) - tf.constant([128.0])) / tf.constant([256.0])
    #example["image"] = (tf.cast(example["image"], tf.float32)) / tf.constant([256.0])
    example["targets"] = tf.cast(example["targets"], tf.float32) / tf.constant([256.0])
    # =========================================
    
    return example

  def example_reading_spec(self):
    data_fields, di2dec = super(ImageCelebaPcml, self).example_reading_spec()
    data_fields["landmarks"] = tf.FixedLenFeature((10,), dtype=tf.int64)
    di2dec["targets"] = tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="landmarks")
    return data_fields, di2dec

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"image": "IdentityModality", #modalities.ModalityType.IDENTITY,
                  "targets": "IdentityModality"} #modalities.ModalityType.IDENTITY}
    p.vocab_size = {"image": 256,
                    "targets": 256}
    p.batch_size_multiplier = 256
    p.input_space_id = 1
    p.target_space_id = 1


@registry.register_problem
class ImageCelebaTinyV2(ImageCelebaPcml):

  @property
  def example_split_config(self):
    return {"train": (100, 0),
            "dev": (100, 0),
            "test": (100, 0)}

  @property
  def image_dim(self):
    return 4


@registry.register_problem
class ImageCelebaPcmlDev(ImageCelebaPcml):

  @property
  def example_split_config(self):
    return {"train": (100, 0),
            "dev": (100, 0),
            "test": (100, 0)}

  @property
  def image_dim(self):
    return 64


@registry.register_problem
class ImageCelebaPcmlMedium(ImageCelebaPcml):

  @property
  def example_split_config(self):
    return {"train": (10000, 0),
            "dev": (1000, 10000),
            "test": (1000, 11000)}

  @property
  def image_dim(self):
    return 64


@registry.register_problem
class ImageCelebaAttributes(ImageCelebaPcml):

  @property
  def image_dim(self):
    return 64

  def dataset_filename(self):
    return "image_celeba_pcml"

  def example_reading_spec(self):
    data_fields, di2dec = super(ImageCelebaAttributes, self).example_reading_spec()
    data_fields["attributes"] = tf.FixedLenFeature((40,), dtype=tf.int64)
    di2dec["targets"] = tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="attributes")
    return data_fields, di2dec

  def preprocess_example(self, example, mode, hparams):
    inputs = example["inputs"]
    example["image"] = image_utils.resize_by_area(inputs, self.image_dim)
    example.pop("inputs", None)
    example.pop("landmarks", None)
    example.pop("attributes", None)
    example["targets"] = tf.slice(example["targets"], [0], [12])
    
    # HACK: Re-scale from [-1,1] to [0,1]
    example["image"] = (tf.cast(example["image"], tf.float32) - tf.constant([128.0])) / tf.constant([256.0])
    #example["image"] = (tf.cast(example["image"], tf.float32)) / tf.constant([256.0])
    example["targets"] = tf.cast(example["targets"], tf.float32)
    example["targets"] = (example["targets"] + tf.constant([1.0], dtype=tf.float32))
    example["targets"] = (example["targets"] / tf.constant([2.0], dtype=tf.float32))

    return example
  
  
# ========

from pcml.operations.tfrecord2bigtable import BigTableSelection
from tensor2tensor.data_generators.problem import default_model_hparams

@registry.register_problem
class CelebaBigTableDev(ImageCelebaPcml):

  @property
  def selection(self):
    return BigTableSelection(
        project="clarify",
        instance="clarify-cbt-instance",
        table="clarify-cbt-devtable",
        prefix="train",
        column_family="tfexample",
        column_qualifier="example"
    )

  def dataset_filename(self):
    return "image_celeba_pcml"

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None,
              partition_id=0,
              num_partitions=1,
              shuffle_buffer_size=1024,
              max_records=-1):

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    dataset_split = dataset_split or mode

    if hparams is None:
      hparams = default_model_hparams()

    selection = self.selection

    # Construct the Problem's hparams so that items within it are accessible
    _ = self.get_hparams(hparams)

    bigtable_client = tf.contrib.cloud.BigtableClient(
        project_id=selection.project,
        instance_id=selection.instance)

    table = bigtable_client.table(selection.table)

    dataset = table.parallel_scan_prefix(
        selection.prefix,
        columns=[(selection.column_family,
                  selection.column_qualifier)])

    dataset = dataset.map(lambda index, data: data)

    if preprocess:
      dataset = self.preprocess(dataset, mode, hparams,
                                interleave=shuffle_files)

    dataset = dataset.take(max_records)

    dataset = dataset.prefetch(output_buffer_size)

    return dataset

  def preprocess_example(self, example, mode, hparams):
    return example

  def serving_input_fn(self, hparams, decode_hparams=None, use_tpu=False):
    """Input fn for serving export, starting from serialized example."""
    
    raise ValueError()
    
    mode = tf.estimator.ModeKeys.PREDICT
    serialized_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="serialized_example")
    dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
    dataset = dataset.map(self.decode_example)
    dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
    dataset = dataset.map(data_reader.cast_ints_to_int32)

    if use_tpu:
      padded_shapes = data_reader.pad_for_tpu(dataset.output_shapes, hparams,
                                              hparams.max_length)
      batch_size = 1 if not decode_hparams else getattr(decode_hparams,
                                                        "batch_size", 1)
      dataset = dataset.padded_batch(
          batch_size, padded_shapes, drop_remainder=False)
      dataset = dataset.map(
          functools.partial(data_reader.pad_batch, batch_multiple=batch_size))
    else:
      dataset = dataset.padded_batch(
          tf.shape(serialized_example, out_type=tf.int64)[0],
          dataset.output_shapes)

    dataset = dataset.map(data_reader.standardize_shapes)
    features = tf.data.experimental.get_single_element(dataset)

    if self.has_inputs:
      features.pop("targets", None)

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=serialized_example)