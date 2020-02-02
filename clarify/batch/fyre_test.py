#!/usr/bin/env python
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
"""Test batch run wrapper."""

import datetime
from absl.testing import absltest

import os

from clarify.batch import fyre
from clarify.batch import jobs
"""

TODO: Currently this test does not pass when run via Bazel as the
bazel run //:push_runtime command depends on the environment having
$HOME defined.

Requires CLARIFY_WORKSPACE_ROOT to be defined, such as when using
Bazel via --test_env=CLARIFY_WORKSPACE_ROOT=<path to repo root>.

Requires adding --test_env=HOME, added that to bazelrc.

"""


class FyreTest(absltest.TestCase):

  def setUp(self):

    self.workspace_root = os.environ["CLARIFY_WORKSPACE_ROOT"]

  def test_basic(self):

    f = fyre.Fyre(workspace_root=self.workspace_root,
                  command=["echo", "hello", "world"])
    """

    Currently fails in CB's notebook container, running out of disk.

    create_response = f.batch_run()

    status_response = f.wait_for_job(
      timeout=datetime.timedelta(seconds=20),
      polling_interval=datetime.timedelta(seconds=1)
    )

    """

  def test_trax_entrypoint(self):

    f = fyre.Fyre(workspace_root=self.workspace_root,
                  command=[
                      "/clarify/bin/train",
                      "--config_file=/clarify/configs/image_fec/mini_test.gin"
                  ])
    """
    create_response = f.batch_run()

    status_response = f.wait_for_job(
      timeout=datetime.timedelta(seconds=20),
      polling_interval=datetime.timedelta(seconds=1)
    )

    Successfully starts up but has trouble initializing large model,
    reduced size 100x, runs 1 step in 139 seconds, eventaully writes Finished
    training.

    """

  def test_trax_tpu(self):

    f = fyre.Fyre(workspace_root=self.workspace_root,
                  command=[
                      "/clarify/bin/train",
                      "--config_file=/clarify/configs/image_fec/mini_test.gin"
                  ],
                  job=jobs.TPUJob)
    """
    create_response = f.batch_run()

    status_response = f.wait_for_job(
      timeout=datetime.timedelta(seconds=20),
      polling_interval=datetime.timedelta(seconds=1)
    )
    """


if __name__ == '__main__':
  absltest.main()
