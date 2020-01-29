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
"""Job base class."""

import os
import uuid
import datetime
import time
from absl import logging

import kubernetes.client
from kubernetes.client.rest import ApiException
from kubernetes import client, config, utils

config.load_kube_config()

configuration = kubernetes.client.Configuration()

core_api_instance = kubernetes.client.CoreV1Api(
    kubernetes.client.ApiClient(configuration))

batch_api_instance = kubernetes.client.BatchV1Api(
    kubernetes.client.ApiClient(configuration))


def gen_timestamped_uid():
  """Generate a timestamped UID of form MMDD-HHMM-UUUU"""
  now = datetime.datetime.now()
  return now.strftime("%m%d-%H%M") + "-" + uuid.uuid4().hex[0:4]


def str_if_bytes(maybe_bytes):
  if isinstance(maybe_bytes, bytes):
    maybe_bytes = maybe_bytes.decode()
  return maybe_bytes


def create_resource_spec(limits, requests):

  GKE_TPU_DESIGNATORS = [
      "cloud-tpus.google.com/v2", "cloud-tpus.google.com/preemptible-v2",
      "cloud-tpus.google.com/v3", "cloud-tpus.google.com/preemptible-v3"
  ]

  allowed_keys = ["cpu", "memory", "nvidia.com/gpu"]

  allowed_keys.extend(GKE_TPU_DESIGNATORS)

  # Define container resource requirements
  resources = client.V1ResourceRequirements()

  def _raise_if_disallowed(key):
    if key not in allowed_keys:
      raise ValueError("Saw resource request or limit key %s "
                       "which is not in allowed keys %s" % (key, allowed_keys))

  for key, value in limits.items():
    raise_if_disallowed_key(key)
    resources.limits[key] = value

  for key, value in requests.items():
    raise_if_disallowed_key(key)
    resources.requests[key] = value

  return resources


def is_running_in_k8s():
  return os.path.isdir('/var/run/secrets/kubernetes.io/')


def get_current_k8s_namespace():
  with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace',
            'r') as f:
    return f.readline()


def get_default_target_namespace():
  if not is_running_in_k8s():
    return 'default'
  return get_current_k8s_namespace()


def _create_job_request(job_name,
                        container,
                        namespace=get_default_target_namespace()):

  # =====
  # HACK
  ann = {"sidecar.istio.io/inject": "false"}
  # =====

  # Create the Job request body
  body = client.V1Job(
      metadata=client.V1ObjectMeta(namespace=namespace, name=job_name),
      spec=client.V1JobSpec(template=client.V1PodTemplateSpec(
          metadata=client.V1ObjectMeta(namespace=namespace, annotations=ann),
          spec=client.V1PodSpec(containers=[container],
                                restart_policy="Never"))))

  return body


def _run_job(request_body):

  api_response = batch_api_instance.create_namespaced_job(
      namespace=request_body.metadata.namespace, body=request_body, pretty=True)

  return api_response


class SimpleJob(object):

  def __init__(self,
               container,
               job_name_stem="unti",
               namespace=get_default_target_namespace()):

    job_name = gen_timestamped_uid()
    job_name = "{}-{}".format(job_name_stem, job_name)

    self.request = _create_job_request(job_name=job_name,
                                       container=container,
                                       namespace=namespace)

  def wait_for_job(self, *args, **kwargs):
    return _wait_for_job(job_request=self.request, *args, **kwargs)

  def batch_run(self):
    return _run_job(self.request)

  def get_pods(self):
    return _get_job_pods(self.request)


def _get_job_pods(job_request):
  """Obtain a list of pods associated with job named `job_name`."""

  job_name = job_request.metadata.name
  namespace = job_request.metadata.namespace

  hits = []

  job_name = str_if_bytes(job_name)

  namespace = str_if_bytes(namespace)

  pods = core_api_instance.list_namespaced_pod(namespace)

  for pod in pods.items:

    labels_dict = pod.metadata.labels

    if labels_dict is not None:
      if "job-name" in labels_dict:
        if labels_dict["job-name"] == job_name:
          hits.append(pod.metadata.name)

  return hits


def _wait_for_job(job_request,
                  timeout=datetime.timedelta(seconds=(24 * 60 * 60)),
                  polling_interval=datetime.timedelta(seconds=30),
                  return_after_num_completions=1,
                  max_failures=1):

  name = str_if_bytes(job_request.metadata.name)
  namespace = str_if_bytes(job_request.metadata.namespace)
  logging.debug("Waiting for job %s in namespace %s..." % (name, namespace))

  end_time = datetime.datetime.now() + timeout

  poll_count = 0
  while True:

    response = batch_api_instance.read_namespaced_job_status(name, namespace)

    if response.status.completion_time is not None:
      return response

    if response.status.failed is not None:
      if response.status.failed >= max_failures:
        return response

    if datetime.datetime.now() + polling_interval > end_time:
      raise Exception(
          "Timeout waiting for job {0} in namespace {1} to finish.".format(
              name, namespace))

    time.sleep(polling_interval.seconds)

    poll_count += 1

    logging.debug("Still waiting for job %s (poll_count=%s)" %
                  (name, poll_count))

  # Linter complains if we don't have a return statement even though
  # this code is unreachable.
  return None
