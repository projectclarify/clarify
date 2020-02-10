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
"""Test of Job subclasses."""

import datetime

from kubernetes import client, config, utils

from absl.testing import absltest

from clarify.batch import jobs


class TPUJobTest(absltest.TestCase):

  def test_tpu_job_basic(self):
    """
    
    TODO: Add one of allowed TPU designations to either
    limits or requests and provided cluster is TPU-enabld it
    should work.
    
    Since the additional configuration is minimal instead of
    having a separate TPUJob could just have a bunch of Resource
    template objects that can be added to a container by Fyre,
    leaving the V1Container -> Job core of this simple.
    
    """

    container = client.V1Container(
        name="clarify",
        image="gcr.io/clarify/runtime-base:v0.1.0-2370",
        command=["pwd"])

    j = jobs.TPUJob(container=container)

    run_request = j.batch_run()

    status = j.wait_for_job(
        timeout=datetime.timedelta(seconds=20),
        polling_interval=datetime.timedelta(seconds=1),
    )


if __name__ == '__main__':
  absltest.main()
