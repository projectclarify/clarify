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
"""Tests of common test utils related to data generators."""

import os

import tensorflow as tf

from pcml.datasets.test_utils import TemporaryDirectory
from pcml.datasets.test_utils import get_test_data_path


class TestTestUtils(tf.test.TestCase):
  """Tests of test utils."""

  def test_temporary_directory(self):

    with TemporaryDirectory() as tmp_dir:
      self.assertTrue(os.path.exists(tmp_dir))

  def test_get_test_data_path(self):
    self.assertEqual(get_test_data_path().split('/')[-1], "test_data")


if __name__ == "__main__":
  tf.test.main()
