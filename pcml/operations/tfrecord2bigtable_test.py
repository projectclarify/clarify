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

"""Tests of Cloud BigTable operation(s)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TestBigTableOperation(tf.test.TestCase):
  """Tests of Cloud BigTable operations."""
  
  def test_tfrecords_to_cbt(self):
    """Test of tfrecords_to_cbt utility."""
    # TODO: Create a small dummy TFRecords set using t2t and load
    # it into a temporary Cloud BigTable table, verifying is has
    # been so added.
    pass

  def test_e2e(self):
    """And end-to-end test of training on examples from cbt.
    
    1. Generate TFRecords.
    2. Upload them to Cloud BigTable.
    3. Perform a training run using those examples.
    
    """
    pass


if __name__ == "__main__":
  tf.test.main()
