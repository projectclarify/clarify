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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pcml.launcher.kube import wait_for_job
from pcml.launcher.kube import get_job_pods
from pcml.operations.eval import EvalJob
from pcml.operations.eval import eval_for_ckpt_dir
import kubernetes
import datetime
import time

kubernetes.config.load_kube_config()
batch_v1beta1_api = kubernetes.client.BatchV1beta1Api()
batch_api = kubernetes.client.BatchV1Api()
core_api = kubernetes.client.apis.core_v1_api.CoreV1Api()


def get_job_names_with_prefix(prefix):
  jobs = batch_api.list_namespaced_job(namespace="kubeflow")
  names = []
  for job in jobs.items:
    name = job.metadata.name
    if name.startswith(prefix):
      names.append(name)
  return names


CREATE_TIMEOUT = datetime.timedelta(seconds=120)

def _testing_run_poll_and_check_cron_job(test_object, create_response,
                                         expect_in_logs=None,
                                         startup_timeout=CREATE_TIMEOUT):
  """Poll for completion of job then check status and log contents."""
  created_name = create_response.metadata.name
  created_namespace = create_response.metadata.namespace

  # Given the way we name cron jobs this should uniquely identify a
  # collection of jobs associated with the cron job of the same name.
  start_time = datetime.datetime.now()
  while True:
    current_time = datetime.datetime.now()
    if (current_time - start_time) > startup_timeout:
      raise Exception(
          "Timeout waiting for cron jobs to be created."
      )
    jobs = get_job_names_with_prefix(created_name)
    if jobs:
      break
    time.sleep(10)

  tf.logging.info("Found jobs: %s" % jobs)

  poll_response = wait_for_job(
      batch_api,
      namespace=created_namespace,
      name=jobs[0],
      polling_interval=datetime.timedelta(seconds=3)
  )

  test_object.assertEqual(poll_response.spec.completions, 1)
  test_object.assertEqual(poll_response.status.succeeded, 1)

  job_pods = get_job_pods(core_api,
                          namespace=created_namespace,
                          job_name=jobs[0])

  tf.logging.info("Job pods: %s" % job_pods)

  logs = core_api.read_namespaced_pod_log(
      name=job_pods[0],
      namespace=created_namespace)

  if expect_in_logs is not None:
    test_object.assertTrue(expect_in_logs in logs)


class TestEvalJob(tf.test.TestCase):

  def setUp(self):
    self.ckpt_path = (
      "gs://clarify-models-us-central1/experiments/cbtdev/b96vox-cel-cbt-j0625-0356-f671/output")
    self.staging_path = "gs://clarify-dev/tmp/evaldev"
    
  def test_eval_fn(self):
    """Local test of fn which pulls ckpts and writes summaries."""
    out = eval_for_ckpt_dir(self.ckpt_path,
                            delay_seconds=5, mock_acc=False,
                            mock_step=False, continuous=False,
                            eval_steps=5)

  def test_e2e(self):

    job = EvalJob(ckpt_dir=self.ckpt_path,
                  staging_path=self.staging_path,
                  # HACK: Pick a node that will have the necessary 
                  # credentials
                  selector_labels={"type": "tpu-host"})

    create_response = job.stage_and_batch_run()

    _testing_run_poll_and_check_cron_job(test_object=self,
                                    create_response=create_response,
                                    expect_in_logs=self.ckpt_path)

  def tearDown(self):
    """Delete all cron jobs created as part of the test."""
    pass

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
