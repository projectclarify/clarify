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
"""Tests of batch image embedding job."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kubernetes
import datetime
import os
import uuid

import tensorflow as tf

from pcml.operations import embed_images
from pcml.launcher.kube import wait_for_job
from pcml.launcher.kube_test import _testing_run_poll_and_check_job

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestEmbedImages(tf.test.TestCase):

  def setUp(self):
    self.test_run_tag = "clarify-test-{}-embed-images".format(
        str(uuid.uuid4())[0:8])
    self.staging = os.path.join(TEST_CONFIG.test_artifacts_root,
                                self.test_run_tag)
    self.model_name = "mcl_dev"
    self.problem_name = "dev_problem"
    self.hparams_set_name = "mcl_res_ut_vtiny"

  #def test_embed_images_fn(self):
  #  embed_images.run()

  def test_e2e(self):

    job = embed_images.EmbedImages(input_manifest="",
                                   target_csv="",
                                   ckpt_path="",
                                   problem_name=self.problem_name,
                                   model_name=self.model_name,
                                   hparams_set_name=self.hparams_set_name,
                                   staging_path=self.staging)

    create_response = job.stage_and_batch_run()

    _testing_run_poll_and_check_job(
        test_object=self,
        create_response=create_response,
        expect_in_logs=embed_images._SUCCESS_MESSAGE)


if __name__ == "__main__":
  tf.test.main()
