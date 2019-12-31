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
"""Tests of dedicated datagen Job wrapper."""

import uuid

import tensorflow as tf

from pcml.operations.t2t_datagen import T2TDatagenJob


class TestT2TDatagenJob(tf.test.TestCase):

    def test_instantiate_and_mock(self):

        job = T2TDatagenJob(problem_name="vox_celeb_distributed_datagen",
                            data_dir="gs://clarify-dev/tmp/datagendev",
                            job_name_prefix="datagen-test",
                            image="gcr.io/clarify/basic-runtime:0.0.4",
                            staging_path="gs://clarify-dev/tmp/datagendev",
                            node_selector={"type": "tpu-host"})
        # HACK: Cause job to be allocated onto host that

        job.launch_shard_parallel_jobs(mock=True)

    def test_run_small_scale(self):

        tag = str(uuid.uuid4())

        job = T2TDatagenJob(problem_name="vox_celeb_distributed_datagen",
                            data_dir="gs://clarify-dev/tmp/datagendev-%s" % tag,
                            job_name_prefix="datagen-test",
                            image="gcr.io/clarify/basic-runtime:0.0.4",
                            staging_path="gs://clarify-dev/tmp/datagendev",
                            node_selector={"type": "tpu-host"})

        job.launch_shard_parallel_jobs(dev_max_num_jobs=10)

        # TODO: Return job ids.

        # TODO: Add a poll and check function that verifies whether the
        # launched jobs succeeded.


if __name__ == "__main__":
    tf.test.main()
