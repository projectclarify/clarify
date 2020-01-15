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

import uuid
import os
import tensorflow as tf
import tempfile

from pcml.operations import extract
from pcml.launcher.kube_test import _testing_run_poll_and_check_job

from google.cloud.bigtable import row_filters

from pcml.utils import cbt_utils

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


def furnish_test_artifacts_path(test_artifacts_root, test_name="untitled"):
  uid = str(uuid.uuid4())
  return os.path.join(test_artifacts_root, uid, test_name)


class TestExtract(tf.test.TestCase):

  def setUp(self):

    self.artifacts_path = furnish_test_artifacts_path(
        test_artifacts_root=TEST_CONFIG.get("test_artifacts_root"),
        test_name="operations/extract")

    if hasattr(TEST_CONFIG, "test_video_manifest_path"):
      test_video_manifest_path = TEST_CONFIG.get("test_video_manifest_path")
      if isinstance(test_video_manifest_path, str):
        self.manifest_path = test_video_manifest_path

    if not hasattr(self, "manifest_path"):
      self.manifest_path = os.path.join(self.artifacts_path, "manifest.csv")

      vox_celeb_root = TEST_CONFIG.get("vox_celeb_data_root")
      test_video = os.path.join(vox_celeb_root, "dev/mp4/id00012/21Uxsk56VDQ",
                                "00001.mp4")

      with tf.gfile.Open(self.manifest_path, "w") as manifest:
        manifest.write(test_video)

    self.project = TEST_CONFIG.get("project")
    self.instance = TEST_CONFIG.get("test_cbt_instance")
    self.tmpdir = tempfile.mkdtemp()

    self.table = "clarify-test-{}-extract".format(str(uuid.uuid4())[0:8])

  def test_extract_given_sharding(self):

    array = [i for i in range(200)]
    num_shards = 4
    reconstructed = []

    for shard_id in range(num_shards):

      shard_subset = extract.subset_given_sharding(array, shard_id, num_shards)

      reconstructed.extend(shard_subset)

    for i, _ in enumerate(reconstructed):
      assert reconstructed[i] == array[i]

  def test_extract_fn(self):

    table_name = "{}-fn".format(self.table)
    target_prefix = "train"

    selection = cbt_utils.RawVideoSelection(project=self.project,
                                            instance=self.instance,
                                            table=table_name,
                                            prefix=target_prefix)

    self.assertTrue(not selection.rows_at_least(10))

    extract.extract_to_cbt(manifest_path=self.manifest_path,
                           shard_id=0,
                           num_shards=1,
                           project=self.project,
                           instance=self.instance,
                           table=table_name,
                           target_prefix=target_prefix,
                           tmp_dir=self.tmpdir)

    # TODO: The test should not only check that the job completes
    # without error but also that there is data in a new table that was
    # created for this test.
    self.assertTrue(selection.rows_at_least(10))

  def test_e2e(self):

    table_name = "{}-e2e".format(self.table)
    target_prefix = "train"

    job = extract.ExtractVideos(
        manifest_path=self.manifest_path,
        staging_path=TEST_CONFIG.get("test_artifacts_root"),
        project=self.project,
        instance=self.instance,
        table=table_name,
        tmp_dir=self.tmpdir,
        target_prefix=target_prefix,
        node_selector={"type": "datagen-small"})

    job.launch_shard_parallel_jobs(mock=True, num_shards=1)

    create_responses = job.launch_shard_parallel_jobs(num_shards=1)

    for create_response in create_responses:
      _testing_run_poll_and_check_job(
          test_object=self,
          create_response=create_response,
          expect_in_logs="Batch extraction complete.")

    selection = cbt_utils.RawVideoSelection(project=self.project,
                                            instance=self.instance,
                                            table=table_name,
                                            prefix="train")

    # TODO: The test should not only check that the job completes
    # without error but also that there is data in a new table that was
    # created for this test.
    self.assertTrue(selection.rows_at_least(10))

    shard_metadata = selection.lookup_shard_metadata()
    expected_shard_metadata = cbt_utils.VideoShardMeta(num_videos=1,
                                                       status="finished",
                                                       shard_id=0,
                                                       num_shards=1)

    self.assertTrue(isinstance(shard_metadata, dict))
    self.assertTrue("train_meta_0" in shard_metadata)

    self.assertEqual(shard_metadata["train_meta_0"].as_dict(),
                     expected_shard_metadata.as_dict())

    # Just to double-check we're correctly counting by prefix and not
    # just generally which would mean this would pass even when data
    # were written to the wrong prefix.
    selection.prefix = "eval"
    self.assertTrue(not selection.rows_at_least(1))


if __name__ == "__main__":
  tf.test.main()
