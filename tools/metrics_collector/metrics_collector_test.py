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

"""Tests of utilities for constructing Katib StudyJob's."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from metrics_collector import MetricsCollector

# VERY HACK ==============
import katib_api_pb2 as api_pb2
# ======================

class TestMetricsCollector(tf.test.TestCase):

  def test_collect_from_gcs(self):
        
    manager_addr = "vizier-core"
    manager_port = 6789
    study_id = "ldb98ccb1aaad68b"
    worker_id = "o52d9dcf4fb7bfa5"
    log_dir = "gs://clarify-models-us-central1/experiments/example-scaleup5/test-e2e-gpu-ut-j0327-0031-5c24/la6aa0abad6c0a19"
    
    mc = MetricsCollector(manager_addr,
                          manager_port,
                          study_id,
                          worker_id,
                          log_dir)

    mlset = mc.run(report=False)

    self.assertTrue(isinstance(mlset, api_pb2.MetricsLogSet))
    self.assertTrue(len(mlset.metrics_logs) == 1)
    self.assertTrue(mlset.metrics_logs[0].name == "losses/training")
    self.assertTrue(len(mlset.metrics_logs[0].values) == 1)
    self.assertTrue(mlset.metrics_logs[0].values[0].time == "2019-03-27T00:51:46+00:00")
    self.assertTrue(mlset.metrics_logs[0].values[0].value == "4686.5615234375")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
