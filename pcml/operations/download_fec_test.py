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
"""Test the FEC dataset downloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kubernetes
import datetime
import tempfile
import uuid
import os
import numpy as np

import tensorflow as tf

from pcml.operations import download_fec
from pcml.launcher.kube import wait_for_job
from pcml.launcher.kube_test import _testing_run_poll_and_check_job

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestDownloadFec(tf.test.TestCase):

  def setUp(self):
    self.test_run_tag = "clarify-test-{}-download-fec".format(
        str(uuid.uuid4())[0:8])
    self.staging = os.path.join(TEST_CONFIG.test_artifacts_root,
                                self.test_run_tag)

  def test_normalize_shape(self):

    cases = [(64, 64, 3), (16, 16, 3), (72, 72, 3), (128, 72, 3), (128, 72, 3)]

    target_shape = (64, 64, 3)

    for case in cases:

      cropped = np.random.randint(0, 255, case)
      cropped = download_fec._normalize_dimensions(cropped, target_shape)
      self.assertTrue(cropped.shape == target_shape)

  '''
  def test_sharded_download_fec_data(self):
    
    tmp_dir = tempfile.mktemp()

    nonfailed_meta = download_fec.sharded_download_fec_data(
      tmp_dir=tmp_dir, is_training=False, shard_id=0, num_shards=20000)

  def test_e2e(self):

    job = download_fec.DownloadFec(
      output_bucket=self.staging,
      is_training=0,
      staging_path=self.staging)

    create_responses = job.launch_shard_parallel_jobs(num_shards=20000,
                                                      max_num_jobs=1)

    _testing_run_poll_and_check_job(test_object=self,
                                    create_response=create_responses[0],
                                    expect_in_logs=download_fec._SUCCESS_MESSAGE)
  '''


if __name__ == "__main__":
  tf.test.main()
