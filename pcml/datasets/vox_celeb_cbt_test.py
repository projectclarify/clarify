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
"""Tests of CBT-centric vox celeb problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid
import numpy as np

from tensor2tensor.utils import registry

import tensorflow as tf
import tempfile
from pcml.datasets import vox_celeb_cbt
from pcml.utils import cbt_utils

from pcml.operations import extract
from pcml.operations import cbt_datagen

from pcml.launcher.kube_test import _testing_run_poll_and_check_job

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestProblem(tf.test.TestCase):

    def setUp(self):

        self.project = TEST_CONFIG.get("project")
        self.instance = TEST_CONFIG.get("test_cbt_instance")
        self.tmpdir = tempfile.mkdtemp()
        self.test_run_tag = "clarify-test-{}-vox-cbt".format(
            str(uuid.uuid4())[0:8])
        self.prefix = "train"
        self.problem_name = "vox_celeb_cbt"
        self.staging = os.path.join(TEST_CONFIG.test_artifacts_root,
                                    self.test_run_tag)
        self.manifest_path = "gs://clarify-dev/test/extract/manifest.csv"
        self.source_table_name = self.test_run_tag + "s"

        # Populate a table with some raw data to sample
        self.source_selection = cbt_utils.RawVideoSelection(
            project=self.project,
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

    def test_lookup(self):

        problem = registry.problem(self.problem_name)

    def test_example_generator(self):

        problem = registry.problem(self.problem_name)

        sample_generator = self.source_selection.sample_av_correspondence_examples(
            frames_per_video=problem.video_shape[0], max_num_samples=10)

        example_generator = vox_celeb_cbt.example_generator(
            raw_sampler=sample_generator,
            video_shape=problem.video_shape,
            audio_shape=problem.audio_shape,
            max_examples=100,
            augmentation_hparams=problem.augmentation_hparams)

        example = example_generator.__next__()

        for key in ["audio", "video", "targets"]:
            self.assertTrue(key in example)

        #np.reshape(np.asarray(example["audio"]), audio_shape)
        np.reshape(np.asarray(example["video"]), problem.video_shape)

    def test_datagen_in_batch(self):

        problem = registry.problem(self.problem_name)

        target_table_name = self.test_run_tag + "bt"

        job = cbt_datagen.CBTDatagenJob(
            problem_name=self.problem_name,
            project=self.project,
            bigtable_instance=self.instance,
            bigtable_source_table_name=self.source_table_name,
            bigtable_target_table_name=target_table_name,
            prefix=self.prefix,
            staging_path=self.staging,
            node_selector={"type": "datagen-small"})

        create_responses = job.launch_shard_parallel_jobs(num_shards=1)

        for create_response in create_responses:
            _testing_run_poll_and_check_job(test_object=self,
                                            create_response=create_response,
                                            expect_in_logs="Completed datagen.")

        tfexample_selection = cbt_utils.TFExampleSelection(
            project=self.project,
            instance=self.instance,
            table=target_table_name,
            prefix=self.prefix)

        example_iterator = tfexample_selection.iterate_tfexamples()

        ex = example_iterator.__next__()

        recv_audio = ex.features.feature['audio'].float_list.value
        recv_video = ex.features.feature['video'].float_list.value
        recv_target = ex.features.feature['targets'].int64_list.value

        #_ = np.reshape(recv_audio, problem.audio_shape)
        _ = np.reshape(recv_video, problem.video_shape)
        _ = np.reshape(recv_target, (1))

    """
  def test_extract_generate_and_fetch_via_dataset(self):

    problem_obj = registry.problem(problem_name)

    manifest_lookup = vox_celeb_cbt.get_manifest_lookup()

    cbt_datagen.dev_extract_and_generate(
      mode_to_manifest_lookup=manifest_lookup,
      staging_path=self.staging,
      project=self.project,
      cbt_instance=self.instance,
      node_tmpdir="/mnt/ssd0",
      problem_name=self.problem_name,
      raw_table_name=None,
      examples_table_name=None)

  """


if __name__ == "__main__":
    tf.test.main()
