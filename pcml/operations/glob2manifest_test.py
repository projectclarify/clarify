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

"""Tests of glob2manifest file path writing Job."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kubernetes
import datetime

import tensorflow as tf

from pcml.operations import glob2manifest
from pcml.launcher.kube import wait_for_job
from pcml.launcher.kube_test import _testing_run_poll_and_check_job


class TestGlob2Manifest(tf.test.TestCase):

  def test_glob2manifest_fn(self):

    glob2manifest.run(
        source_glob="gs://clarify-dev/tmp/03312019/pcml-new/*.py",
        target_path="gs://clarify-dev/tmp/beam/devmanifest.txt")

  def test_e2e(self):

    job = glob2manifest.Glob2Manifest(
        source_glob="gs://clarify-dev/tmp/03312019/pcml-new/*.py",
        target_path="gs://clarify-dev/tmp/beam/devmanifest.txt",
        staging_path="gs://clarify-dev/dummy/staging")

    create_response = job.stage_and_batch_run()

    _testing_run_poll_and_check_job(test_object=self,
                                    create_response=create_response,
                                    expect_in_logs=None)


if __name__ == "__main__":
  tf.test.main()
