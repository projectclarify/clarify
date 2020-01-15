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
"""Tests of Cloud BigTable-centric T2T datagen."""

import uuid
import tempfile
import os

import tensorflow as tf

#from pcml.launcher.kube_test import _testing_run_poll_and_check_job

from clarify.datasets.core import cbt_datagen
from clarify.datasets.voxceleb2 import extract

from clarify.utils import cbt_utils

from clarify.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestCBTDatagenJob(tf.test.TestCase):

  def setUp(self):

    self.project = TEST_CONFIG.get("project")
    self.instance = TEST_CONFIG.get("test_cbt_instance")
    self.tmpdir = tempfile.mkdtemp()
    self.test_run_tag = "clarify-test-{}-cbt-datagen".format(
        str(uuid.uuid4())[0:8])
    self.prefix = "train"
    self.problem_name = "cbt_datagen_test_problem"
    self.staging = os.path.join(TEST_CONFIG.test_artifacts_root,
                                self.test_run_tag)
    self.manifest_path = "gs://clarify-dev/test/extract/manifest.csv"
    self.frames_per_video = 15

    self.source_table_name = self.test_run_tag + "s"

    # Populate a table with some raw data to sample
    selection = cbt_utils.RawVideoSelection(project=self.project,
                                            instance=self.instance,
                                            table=self.source_table_name,
                                            prefix=self.prefix)

    extract.extract_to_cbt(manifest_path=self.manifest_path,
                           shard_id=0,
                           num_shards=1,
                           project=self.project,
                           instance=self.instance,
                           table=self.source_table_name,
                           target_prefix=self.prefix,
                           tmp_dir=tempfile.mkdtemp())

  def test_fn(self):

    target_table_name = self.test_run_tag + "ft"

    cbt_datagen.cbt_generate_and_load_examples(
        project=self.project,
        bigtable_instance=self.instance,
        bigtable_source_table_name=self.source_table_name,
        bigtable_target_table_name=target_table_name,
        max_num_examples=100,
        prefix=self.prefix,
        problem_name=self.problem_name)

  def test_e2e(self):

    target_table_name = self.test_run_tag + "bt"

    job = cbt_datagen.CBTDatagenJob(
        problem_name=self.problem_name,
        project=self.project,
        bigtable_instance=self.instance,
        bigtable_source_table_name=self.source_table_name,
        bigtable_target_table_name=target_table_name,
        prefix=self.prefix,
        staging_path=self.staging,
        max_num_examples=100,
        node_selector={"type": "datagen-small"})

    create_responses = job.launch_shard_parallel_jobs(num_shards=1)

    for create_response in create_responses:
      _testing_run_poll_and_check_job(test_object=self,
                                      create_response=create_response,
                                      expect_in_logs="Completed datagen.")


if __name__ == "__main__":
  tf.test.main()
