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

"""Tests of DISFA problem definitions and utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.datasets import disfa

TMP_TESTING_ROOT = "/tmp"


class TestDISFA(tf.test.TestCase):

  def test_walkthrough(self):
    paths = disfa.maybe_get_disfa_data(TMP_TESTING_ROOT,
                                      disfa.DEFAULT_DATA_ROOT).next()
    self.assertTrue(tf.gfile.Exists(paths["left_video"]))
    self.assertTrue(tf.gfile.Exists(paths["right_video"]))
    for au_label_path in paths["au_labels"]:
      self.assertTrue(tf.gfile.Exists(au_label_path))
    dat = disfa._load_disfa_aus(paths["au_labels"])
    self.assertEqual(len(dat), len(disfa.DISFA_AUS))
    example = {"labels/continuous/action_units": dat,
               "video/left": [],
               "video/right": []}
    subsampled = disfa._tiled_subsample_example(example, 100, 10).next()
    self.assertEqual(len(subsampled["labels/continuous/action_units"]), 11)
    self.assertEqual(len(subsampled["labels/continuous/action_units"][0]), 100)
    
  def test_raw_data_verifier(self):
    disfa._raw_data_verifier(disfa.DEFAULT_DATA_ROOT)

  def test_generator(self):
    
    # WIP: Currently working on how to best handle video when we want to
    # subsample regions of it; could do this by seconds with moviepy,
    # read all frames and index into the array by position, build a frame
    # hash using python, see if moviepy already does this, etc.
    
    example = disfa._generator(TMP_TESTING_ROOT).next()
    expected_keys = ["video/left/encoded", "video/right/encoded",
                     "labels/continuous/action_units"]
    for key in expected_keys:
      self.assertTrue(key in example.keys())


if __name__ == "__main__":
  tf.test.main()
