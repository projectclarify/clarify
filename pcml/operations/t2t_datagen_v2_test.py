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
# limitations under the License

"""Tests of dedicated datagen Job wrapper direct to CBT."""

import uuid

import tensorflow as tf

from pcml.launcher.kube_test import _testing_run_poll_and_check_job

from pcml.operations.t2t_datagen_v2 import T2TDatagenJobV2


import os



class TestT2TDatagenJobV2(tf.test.TestCase):

  def test_instantiate_and_mock(self):

    job = T2TDatagenJobV2(problem_name="vox_celeb_sharded_generator_dev",
                          data_dir="gs://clarify-dev/tmp/datagendev",
                          job_name_prefix="cbtdirect",
                          image="gcr.io/clarify/basic-runtime:0.0.4",
                          staging_path="gs://clarify-dev/tmp/datagendev",
                          node_selector={"type": "datagen-small"})

    job.launch_shard_parallel_jobs(mock=True)

  def test_e2e(self):

    job = T2TDatagenJobV2(problem_name="vox_celeb_sharded_generator_dev",
                          bigtable_instance="clarify-cbt-instance",
                          bigtable_table="clarify-cbt-devtable",
                          project="clarify",
                          data_dir="gs://clarify-dev/tmp/datagendev",
                          job_name_prefix="cbtdirect",
                          image="gcr.io/clarify/basic-runtime:0.0.4",
                          staging_path="gs://clarify-dev/tmp/datagendev",
                          node_selector={"type": "datagen-small"},
                          num_cpu=1,
                          memory="7Gi")

    create_responses = job.launch_shard_parallel_jobs(dev_max_num_jobs=1)

    for create_response in create_responses:
      _testing_run_poll_and_check_job(
        test_object=self, create_response=create_response,
        expect_in_logs="Completed datagen.")
    
    """

    These keep crashing because of memory usage. What's happening is each
    thread is accumulating a large number of samples (in memory) before
    writing these to CBT because the threads return instead of yield. If
    this was built to be single-threaded there wouldn't be as much of an
    issue.
    
    Trying to turn down the number of samples per video which will probably
    solve it but that's not a compromise I want to have to make; the
    solution mentioned above would be better.

    """

    
if __name__ == "__main__":
  tf.test.main()