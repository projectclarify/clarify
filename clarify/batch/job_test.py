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

"""Test of base Job object."""

import time
import datetime

from kubernetes import client, config, utils

from absl.testing import absltest

from clarify.batch import job


class JobTest(absltest.TestCase):

  def test_job_basic(self):

    container=client.V1Container(
      name="clarify",
      image="gcr.io/clarify/runtime-base:v0.1.0-2370",
      command=["pwd"]
    )

    j = job.SimpleJob(container=container)

    run_request = j.batch_run()

    """

    Looks like jobs run indefinitely because istio pod continues running.

    Could run daemon to 

    kubectl exec <pod> -c istio-proxy -- curl -X POST localhost:15000/quitquitquit

    for any jobs 

    containerStatuses[*].state.terminated.reason=completed

    --

    Added envoy-preflight to rtbase 
    https://github.com/monzo/envoy-preflight
    requires env variables to be set in order to work, incl 127.0.0.1:15000

    Pods terminate with status complete albeit with delay but Job never shows
    status complete. Despite job pod showing status complete the istio container
    shows status running.

    """

    pods = []
    mx = 20
    ct = 0
    while True:
      pods = j.get_pods()
      time.sleep(1)
      ct += 2
      if ct >= mx:
        break

    self.assertTrue(pods)

    status = j.wait_for_job(
      timeout=datetime.timedelta(seconds=20),
      polling_interval=datetime.timedelta(seconds=1),    
    )
    
    """
    
    TODO: As described above, test fails with timeout.
    
    """


if __name__ == '__main__':
  absltest.main()
