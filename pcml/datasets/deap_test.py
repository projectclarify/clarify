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
"""Tests of DEAP problem definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from pcml.datasets import deap
from pcml.datasets.test_utils import TrivialModel
from pcml.utils.dev_utils import T2TDevHelper

TESTING_TMP = "/tmp/deap"  #HACK


class TestDeapUtils(tf.test.TestCase):

  def test_raw_data_verifier(self):
    deap._raw_data_verifier(deap.DEFAULT_DEAP_ROOT)

  def test_load_preprocessed(self):
    input_path = deap.maybe_get_deap_data(tmp_dir=TESTING_TMP,
                                          deap_root=deap.DEFAULT_DEAP_ROOT,
                                          is_training=True,
                                          training_fraction=1).__next__()
    rec, pos, raw, labels, chan = deap.load_preprocessed_deap_data(input_path)
    # TODO: Assert all data are floats in [0,1]

  def test_tiled_subsample_example(self):

    example = {
        "eeg/raw": [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
        "eeg/positions": [[1, 1], [1, 1]],
        "eeg/channels": ["Ch1", "Ch2"],
        "physio/hEOG": [1, 1, 1, 1, 1, 1, 1],
        "physio/vEOG": [1, 1, 1, 1, 1, 1, 1],
        "physio/zEMG": [1, 1, 1, 1, 1, 1, 1],
        "physio/tEMG": [1, 1, 1, 1, 1, 1, 1],
        "physio/GSR": [1, 1, 1, 1, 1, 1, 1],
        "physio/rAMP": [1, 1, 1, 1, 1, 1, 1],
        "physio/plethysmograph": [1, 1, 1, 1, 1, 1, 1],
        "physio/temp": [1, 1, 1, 1, 1, 1, 1],
        "affect/valence": [1],
        "affect/arousal": [1],
        "affect/dominance": [1],
        "affect/liking": [1]
    }

    subsampled = deap._tiled_subsample_example(example,
                                               subsample_width=2,
                                               subsample_step=1)
    for example in subsampled:
      self.assertEqual(len(example["affect/liking"]), 1)

  def test_generator(self):

    generated_examples = []

    num_samples = 100
    subsample_width = 100
    subsample_step = 10

    for example in deap._generator(tmp_dir=TESTING_TMP,
                                   subsample_width=subsample_width,
                                   subsample_step=subsample_step,
                                   how_many=num_samples,
                                   is_training=True,
                                   training_fraction=0.9):
      generated_examples.append(example)
    self.assertEqual(len(generated_examples), num_samples)

    expected_channels = deap.DEAP_EXPECTED_MAPPED_EEG_CHANNELS

    expected_keys_shapes = [
        ("eeg/raw", (len(expected_channels) * subsample_width,), int),
        ("eeg/positions", (len(expected_channels) * 4,), None),
        ("eeg/channels", (len(expected_channels),), None),
        ("physio/hEOG", (subsample_width,), None),
        ("physio/vEOG", (subsample_width,), None),
        ("physio/zEMG", (subsample_width,), None),
        ("physio/tEMG", (subsample_width,), None),
        ("physio/GSR", (subsample_width,), None),
        ("physio/rAMP", (subsample_width,), None),
        ("physio/plethysmograph", (subsample_width,), None),
        ("physio/temp", (subsample_width,), None),
        ("affect/trial_selfreport", (4,), None)
    ]

    for expected_key, expected_shape, expected_type in expected_keys_shapes:
      assert expected_key in generated_examples[0]
      field = generated_examples[0][expected_key]
      field = np.asarray(field)
      self.assertEqual(field.shape, expected_shape)
      if expected_type is not None:
        self.assertEqual(field.dtype, expected_type)

    self.assertTrue(
        np.array_equal(sorted(example["eeg/channels"]),
                       sorted(np.asarray(expected_channels))))

  def test_clip_and_standardize(self):

    cases = [{
        "min_val": 0,
        "max_val": 10,
        "voc": 256,
        "dt": np.int32
    }, {
        "min_val": -20,
        "max_val": 20,
        "voc": 256,
        "dt": np.int32
    }]

    for case in cases:
      min_val, max_val = case["min_val"], case["max_val"]
      a = np.random.randint(min_val, max_val, (100,))
      std = deap.clip_and_standardize(a, min_val, max_val, case["voc"],
                                      case["dt"])
      assert std.min() >= -1 * case["voc"] / 2
      assert std.max() <= case["voc"] / 2

  def test_deap_problem_generates(self):
    helper = T2TDevHelper("multi_modal_dev_model", "deap_problem_base",
                          "multi_modal_dev_model_tiny", None)
    helper.datagen()
    dataset = helper.problem.dataset("train", data_dir=helper.data_dir)
    example = dataset.make_one_shot_iterator().next()
    example = example["targets"]

    #vsize = helper.problem.vocab_size
    # In Eager mode by way of T2TDevHelper import?
    #self.assertTrue(tf.reduce_all([
    #    tf.reduce_all(tf.less(example, tf.zeros_like(example) + vsize)),
    #    tf.reduce_all(tf.greater_equal(example, tf.zeros_like(example)))]).numpy())

  def test_deap_problem_train(self):

    # TODO: Current error comes from using stock infer method of trivial
    # model which assumes the features has "inputs", easy to update, looks
    # like it trains up to the point of export in that case, worth giving
    # multi-task training a shot instead of modifying mmdm, perhaps in
    # notebook

    helper = T2TDevHelper("trivial_model", "deap_problem_base",
                          "multi_modal_dev_model_tiny", None)
    helper.run_e2e()


if __name__ == "__main__":
  tf.test.main()
