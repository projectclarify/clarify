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

"""Test extraction trigger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf

from pcml.functions.extract.publish import trigger_extraction

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestExtractFn(tf.test.TestCase):

  def test_trigger_extraction(self):

    salt = str(uuid.uuid4())
    target_table_name = "ext-fn-test" + str(uuid.uuid4())
    problem_name = "vox_celeb_single_frame"

    trigger_extraction(
      problem_name=problem_name,
      project=TEST_CONFIG.get("project"),
      bigtable_instance=TEST_CONFIG.get("test_cbt_instance"),
      prefix="train",
      downsample_xy_dims=96,
      greyscale=True,
      resample_every=2,
      audio_block_size=1000,
      override_target_table_name=target_table_name,
      override_max_files_processed=1000)
    
    # For now only minimally tests that the messages can be
    # emitted not that when consumed they result in successful
    # processing.


if __name__ == "__main__":
  tf.test.main()
