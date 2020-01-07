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
"""Kubernetes models and utils supporting templating Job's and TFJob's

TODO: Should consider making use of existing Kubernetes python client
object models.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import datetime
import kubernetes
import time
import uuid

from pcml.launcher.util import expect_type
from pcml.launcher.util import object_as_dict
from pcml.launcher.util import run_and_output
from pcml.launcher.util import dict_prune_private
from pcml.utils.fs_utils import get_pcml_root
from pcml.launcher.util import _compress_and_stage

# A common base image that extends kubeflow tensorflow_notebook workspace
# image with python dependencies needed for various examples.
# TODO: In the future include various additional deps in this base image
_COMMON_BASE = "gcr.io/kubeflow-rl/common-base:0.1.0"
_TF_JOB_GROUP = "kubeflow.org"
_TF_JOB_VERSION = "v1beta1"
_TF_JOB_PLURAL = "tfjobs"


def gen_timestamped_uid():
  """Generate a timestamped UID of form MMDD-HHMM-UUUU"""
  now = datetime.datetime.now()
  return now.strftime("%m%d-%H%M") + "-" + uuid.uuid4().hex[0:4]


def str_if_bytes(maybe_bytes):
  if isinstance(maybe_bytes, bytes):
    maybe_bytes = maybe_bytes.decode()
  return maybe_bytes


def build_command(base_command, **kwargs):
  """Build a command array extending a base command with -- form kwargs.

  E.g. [t2t-trainer, {key: value}] -> [t2t-trainer, --key=value]

  """

  expect_type(base_command, str)

  command = [base_command]

  for key, value in kwargs.items():

    expect_type(key, str)
    expect_type(value, str)

    command.append("--%s=%s" % (key, value))

  return command


class AttachedVolume(object):
  """Model of information needed to attach a Kubernetes Volume.

  Primarily manages correpondence of volume and volume_mount fields and
  expects objects recieving `AttachedVolume` as argument to know whether to
  access `volume` or `volume_mount` fields.

  """

  def __init__(self,
               claim_name,
               mount_path=None,
               volume_name=None,
               volume_type="persistentVolumeClaim"):

    if not isinstance(claim_name, str):
      raise ValueError("Expected string claim_name, saw %s" % claim_name)

    if mount_path is None:
      mount_path = "/mnt/%s" % claim_name

    if not isinstance(mount_path, str):
      raise ValueError("Expected string mount_path, saw %s" % claim_name)

    if not mount_path.startswith("/"):
      raise ValueError("Mount path should start with '/', saw %s" % mount_path)

    if volume_name is None:
      volume_name = claim_name

    if not isinstance(volume_name, str):
      raise ValueError("Expected string volume_name, saw %s" % volume_name)

    self.volume = {
        "name": volume_name,
        "persistentVolumeClaim": {
            "claimName": claim_name
        }
    }
    self.volume_mount = {"name": volume_name, "mountPath": mount_path}


class LocalSSD(object):

  def __init__(self, disk_id=0):

    self.volume = {
        "name": "ssd%s" % disk_id,
        "hostPath": {
            "path": "/mnt/disks/ssd%s" % disk_id
        }
    }

    self.volume_mount = {
        "name": "ssd%s" % disk_id,
        "mountPath": "/mnt/ssd%s" % disk_id
    }


GKE_TPU_DESIGNATORS = [
    "cloud-tpus.google.com/v2", "cloud-tpus.google.com/preemptible-v2",
    "cloud-tpus.google.com/v3", "cloud-tpus.google.com/preemptible-v3"
]


class Resources(object):
  """Model of Kuberentes Container resources"""

  def __init__(self, limits=None, requests=None):

    allowed_keys = ["cpu", "memory", "nvidia.com/gpu"]
    allowed_keys.extend(GKE_TPU_DESIGNATORS)

    def raise_if_disallowed_key(key):
      if key not in allowed_keys:
        raise ValueError("Saw resource request or limit key %s "
                         "which is not in allowed keys %s" %
                         (key, allowed_keys))

    if limits is not None:
      self.limits = {}
      for key, value in limits.items():
        raise_if_disallowed_key(key)
        self.limits[key] = value

    if requests is not None:
      self.requests = {}
      for key, value in requests.items():
        raise_if_disallowed_key(key)
        self.requests[key] = value


class Container(object):
  """Model of Kubernetes Container object."""

  def __init__(self,
               image,
               name=None,
               command_args=None,
               command=None,
               resources=None,
               volume_mounts=None,
               allow_nameless=False,
               ports=None):

    if command_args is not None:
      self.args = command_args

    if command is not None:
      self.command = command

    if ports is not None:

      if not isinstance(ports, list):
        raise ValueError("ports must be a list, saw %s" % ports)

      for port in ports:
        if not isinstance(port, dict):
          raise ValueError("ports must be a list of dict.'s, saw %s" % ports)

      self.ports = ports

    self.image = image

    if name is not None:
      self.name = name

    elif not allow_nameless:
      raise ValueError("The `name` argument must be specified "
                       "unless `allow_nameless` is True.")

    if resources is not None:

      if not isinstance(resources, Resources):
        raise ValueError("non-null resources expected to be of "
                         "type Resources, saw %s" % type(resources))

      self.resources = resources

    if volume_mounts is not None:

      if not isinstance(volume_mounts, list):
        raise ValueError("non-null volume_mounts expected to be of "
                         "type list, saw %s" % type(volume_mounts))

      # pylint: disable=invalid-name
      self.volumeMounts = volume_mounts


def job_status_callback(job_response):
  """A callback to use with wait_for_job."""

  tf.logging.info("Job %s in namespace %s; uid=%s; succeeded=%s" %
                  (job_response.metadata.name, job_response.metadata.namespace,
                   job_response.metadata.uid, job_response.status.succeeded))

  return job_response


def wait_for_job(batch_api,
                 namespace,
                 name,
                 timeout=datetime.timedelta(seconds=(24 * 60 * 60)),
                 polling_interval=datetime.timedelta(seconds=30),
                 return_after_num_completions=1,
                 max_failures=1):

  name = str_if_bytes(name)
  namespace = str_if_bytes(namespace)
  tf.logging.debug("Waiting for job %s in namespace %s..." % (name, namespace))

  end_time = datetime.datetime.now() + timeout

  poll_count = 0
  while True:

    response = batch_api.read_namespaced_job_status(name, namespace)

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

    tf.logging.debug("Still waiting for job %s (poll_count=%s)" %
                     (name, poll_count))

  # Linter complains if we don't have a return statement even though
  # this code is unreachable.
  return None


def get_job_pods(core_api, job_name, namespace):
  """Obtain a list of pods associated with job named `job_name`."""

  hits = []

  job_name = str_if_bytes(job_name)

  namespace = str_if_bytes(namespace)

  pods = core_api.list_namespaced_pod(namespace)

  for pod in pods.items:

    labels_dict = pod.metadata.labels

    if labels_dict is not None:
      if "job-name" in labels_dict:
        if labels_dict["job-name"] == job_name:
          hits.append(pod.metadata.name)

  return hits


class Job(object):
  """Python model of a Kubernetes Job object."""

  def __init__(self,
               job_name,
               command=None,
               command_args=None,
               image=_COMMON_BASE,
               restart_policy="Never",
               namespace="default",
               volume_claim_id=None,
               batch=True,
               no_wait=False,
               api_version=None,
               kind=None,
               metadata=None,
               spec=None,
               smoke=False,
               node_selector=None,
               additional_metadata=None,
               pod_affinity=None,
               num_local_ssd=0,
               resources=None,
               **kwargs):
    """Check args for and template a Job object.

    name (str): A unique string name for the job.
    image (str): The image within which to run the job command.
    restart_policy (str): The restart policy (e.g. Never, onFailure).
    namespace (str): The namespace within which to run the Job.
    volume_claim_id (str): The ID of a persistent volume to mount at
      /mnt/`volume_claim_id`.
    api_version (str): Allow an alternative API version to be
      specified.
    kind (str): Allow the job kind to be overridden by subclasses.
    spec (str): Allow the job spec to be specified explicitly.

    """

    # Private attributes (i.e. those with the _ prefix) will be ignored
    # when converting object to the dict that will be used as the
    # request body to the kubernetes API.
    self._command = command
    self._args = command_args
    self._batch = batch
    self._poll_and_check = True

    if not isinstance(smoke, bool):
      raise ValueError("The type of `smoke` should be boolean, "
                       "saw %s" % smoke)
    self._smoke = smoke

    tf.logging.info("smoke: %s" % self._smoke)

    if no_wait:
      self._poll_and_check = False

    container_kwargs = {
        "command": command,
        "command_args": command_args,
        "image": image,
        "name": "container",
        "resources": resources,
    }

    volumes = []

    if volume_claim_id is not None:
      volumes.append(AttachedVolume(volume_claim_id))

    if num_local_ssd > 0:
      volumes.append(LocalSSD())

    tf.logging.debug("volumes: %s" % volumes)

    volume_mounts_spec = [getattr(volume, "volume_mount") for volume in volumes]

    tf.logging.debug("volume mounts spec: %s" % volume_mounts_spec)

    if volume_mounts_spec:
      container_kwargs["volume_mounts"] = volume_mounts_spec

    container = Container(**container_kwargs)

    # pylint: disable=invalid-name
    self.apiVersion = (api_version if api_version is not None else "batch/v1")

    self.kind = kind if kind is not None else "Job"

    # Allow metadata to be passed in as a parameter, such as in the
    # construction of a TFJob subclass.
    if metadata is None:
      self.metadata = {"name": job_name, "namespace": namespace}
    else:
      self.metadata = metadata

    if additional_metadata is not None:
      if not isinstance(additional_metadata, dict):
        raise ValueError("additional_metadata must be of type dict, saw "
                         "%s" % additional_metadata)
      self.metadata.update(additional_metadata)

    self.spec = spec if spec is not None else {
        "template": {
            "spec": {
                "containers": [container],
                "restartPolicy": restart_policy
            }
        },
        "backoffLimit": 4
    }

    if spec is None:
      volumes_spec = [getattr(volume, "volume") for volume in volumes]
      if volumes_spec:
        self.spec["template"]["spec"]["volumes"] = volumes_spec

    self.set_node_selector(node_selector)

    self.set_pod_affinity(pod_affinity)

  def set_node_selector(self, node_selector):

    if node_selector is None:
      return

    if not isinstance(node_selector, dict):
      raise ValueError(("Non-None node_selector expected to have type dict, "
                        "saw %s" % node_selector))

    for key, value in node_selector.items():
      node_selector[key] = str(value)
    self.spec["template"]["spec"]["nodeSelector"] = node_selector

  def set_pod_affinity(self, pod_affinity):

    if pod_affinity is None:
      return

    if not isinstance(pod_affinity, dict):
      raise ValueError(("Non-None pod_affinity expected to have type dict, "
                        "saw %s" % pod_affinity))

    affinity_key = list(pod_affinity.keys())[0]
    affinity_values = pod_affinity[affinity_key]

    if not isinstance(affinity_values, list):
      raise ValueError(
          ("For now expecting that pod_affinity is a dict with a single "
           "key into a list of values, saw %s" % pod_affinity))

    self.spec["template"]["spec"]["affinity"] = {
        "podAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [{
                "labelSelector": {
                    "matchExpressions": [{
                        "key": affinity_key,
                        "operator": "In",
                        "values": affinity_values
                    }]
                }
            }, {
                "topologyKey": "kubernetes.io/hostname"
            }]
        }
    }

  def run(self):

    if self._smoke:
      cmd = ["echo"]
      cmd.extend(self._command)
      self._command = cmd

    if self._batch:
      self.batch_run()
    else:
      self.local_run()

  def as_dict(self):
    return dict_prune_private(object_as_dict(self))

  def batch_run(self):

    kubernetes.config.load_kube_config()

    job_client = kubernetes.client.BatchV1Api()

    job_dict = self.as_dict()

    tf.logging.info("Triggering batch run with job config: %s" % job_dict)

    create_response = job_client.create_namespaced_job(
        job_dict["metadata"]["namespace"], job_dict)

    return create_response

  def local_run(self, show=True):
    """Run the job command locally."""

    tf.logging.info("Triggering local run.")

    return run_and_output(self._command)


class CronJob(Job):

  def __init__(self, schedule="*/10 * * * *", *args, **kwargs):
    super(CronJob, self).__init__(*args, **kwargs)
    job_spec = self.spec
    self.kind = "CronJob"
    self.apiVersion = "batch/v1beta1"
    self.spec = {"schedule": schedule, "jobTemplate": {"spec": job_spec}}

  def batch_run(self):

    kubernetes.config.load_kube_config()

    job_client = kubernetes.client.BatchV1beta1Api()

    job_dict = self.as_dict()

    tf.logging.info("Triggering batch run with job config: %s" % job_dict)

    create_response = job_client.create_namespaced_cron_job(
        namespace=job_dict["metadata"]["namespace"], body=job_dict)

    return create_response
