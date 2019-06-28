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

"""Kubernetes models and utils supporting templating Job's and TFJob's"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import datetime
import kubernetes
import tensorflow as tf

from pcml.launcher.kube import TFJob
from pcml.launcher.kube import TFJobReplica
from pcml.launcher.kube import Resources
from pcml.launcher.kube import AttachedVolume
from pcml.launcher.kube import Job
from pcml.launcher.kube import build_command
from pcml.launcher.kube import Container
from pcml.launcher.kube import LocalSSD
from pcml.launcher.kube import wait_for_job
from pcml.launcher.kube import wait_for_tfjob
from pcml.launcher.kube import get_job_pods
from pcml.launcher.util import gen_timestamped_uid
from pcml.launcher.util import object_as_dict
from pcml.launcher.kube import get_tfjob_pods

kubernetes.config.load_kube_config()

# Instantiate various Kubernetes API client objects
batch_api = kubernetes.client.BatchV1Api()
core_api = kubernetes.client.apis.core_v1_api.CoreV1Api()
crd_api = kubernetes.client.CustomObjectsApi()


def str_if_bytes(maybe_bytes):
  if isinstance(maybe_bytes, bytes):
    maybe_bytes = maybe_bytes.decode()
  return maybe_bytes


def _testing_run_poll_and_check_job(test_object, create_response,
                                    expect_in_logs=None):
  """Poll for completion of job then check status and log contents."""
  created_name = create_response.metadata.name
  created_namespace = create_response.metadata.namespace
  poll_response = wait_for_job(
      batch_api,
      namespace=created_namespace,
      name=created_name,
      polling_interval=datetime.timedelta(seconds=3)
  )
  test_object.assertEqual(poll_response.spec.completions, 1)
  test_object.assertEqual(poll_response.status.succeeded, 1)
  job_pods = get_job_pods(core_api,
                          namespace=created_namespace,
                          job_name=created_name)
  logs = core_api.read_namespaced_pod_log(
      name=job_pods[0],
      namespace=created_namespace)
  if expect_in_logs is not None:
    test_object.assertTrue(expect_in_logs in logs)


def _testing_run_poll_and_check_tfjob(test_object, create_response,
                                      expect_in_logs=None):
  """Poll for completion of tfjob then check status and log contents."""
  created_name = create_response.get("metadata", {}).get("name", {})
  created_metadata = create_response.get("metadata", {})
  created_namespace = created_metadata.get("namespace", {})
  _, last_status = wait_for_tfjob(
      crd_api,
      namespace=created_namespace,
      name=created_name,
      polling_interval=datetime.timedelta(seconds=3),
      timeout=datetime.timedelta(minutes=20)
  )
  test_object.assertEqual(last_status, "Succeeded")

  job_pods = get_tfjob_pods(core_api,
                            namespace=created_namespace,
                            tfjob_name=created_name)

  logs = core_api.read_namespaced_pod_log(
      name=str_if_bytes(job_pods["master"][0]),
      namespace=created_namespace)
  if expect_in_logs is not None:
    for expected in expect_in_logs:
      test_object.assertTrue(expected in logs)


class TestContainer(tf.test.TestCase):

  def test_instantiate(self):

    config = {
        "model": "something"
    }

    container_args = build_command("t2t-trainer", **config)

    av = AttachedVolume("nfs-1")
    resources = Resources(requests={"cpu": 1,
                                    "memory": "1Gi",
                                    "nvidia.com/gpu": 1})
    image_tag = "gcr.io/kubeflow-rl/enhance:0321-2116-e45a"

    cases = [
        {
            "kwargs": {
                "command_args": container_args,
                "image": image_tag,
                "name": "tensorflow",
                "resources": resources,
                "volume_mounts": [av.volume_mount],
            },
            "expected": {
                'args': ['t2t-trainer', '--model=something'],
                'image': 'gcr.io/kubeflow-rl/enhance:0321-2116-e45a',
                'name': 'tensorflow',
                'resources': {'requests': {'cpu': 1,
                                           'memory': '1Gi',
                                           'nvidia.com/gpu': 1}},
                'volumeMounts': [{
                    'mountPath': '/mnt/nfs-1', 'name': 'nfs-1'
                }]
            }
        },
        {
            "kwargs": {
                "command": ["/bin/sh", "-c"],
                "command_args": container_args + ["&&", "echo", "hworld"],
                "image": image_tag,
                "name": "tensorflow",
                "resources": resources,
                "volume_mounts": [av.volume_mount],
            },
            "expected": {
                'command': ['/bin/sh', '-c'],
                'args': [
                    't2t-trainer', '--model=something', '&&',
                    'echo', 'hworld'
                ],
                'image': 'gcr.io/kubeflow-rl/enhance:0321-2116-e45a',
                'name': 'tensorflow',
                'resources': {'requests': {'cpu': 1,
                                           'memory': '1Gi',
                                           'nvidia.com/gpu': 1}},
                'volumeMounts': [{
                    'mountPath': '/mnt/nfs-1',
                    'name': 'nfs-1'
                }]
            }
        }

    ]

    for case in cases:
      self.assertEqual(object_as_dict(Container(**case["kwargs"])),
                       case["expected"])


  def test_expose_ports(self):

    kwargs = {
        "image": "gcr.io/kubeflow-rl/enhance:0321-2116-e45a",
        "command_args": ['t2t-trainer', '--model=something'],
        "name": "tensorflow",
        "ports": [{"containerPort": 80}]
    }
    expected = {
        'args': ['t2t-trainer', '--model=something'],
        'image': 'gcr.io/kubeflow-rl/enhance:0321-2116-e45a',
        'name': 'tensorflow',
        'ports': [{"containerPort": 80}]
    }

    self.assertEqual(object_as_dict(Container(**kwargs)), expected)


class TestAttachedVolume(tf.test.TestCase):

  def test_instantiate(self):

    cases = [
        {
            "kwargs": {"claim_name": "nfs-1"},
            "expected": {
                'volume': {
                    'name': 'nfs-1',
                    'persistentVolumeClaim': {'claimName': 'nfs-1'}
                },
                'volume_mount': {'mountPath': '/mnt/nfs-1', 'name': 'nfs-1'}
            }
        }
    ]

    for case in cases:
      av = AttachedVolume(**case["kwargs"])
      self.assertEqual(object_as_dict(av),
                       case["expected"])


class TestLocalSSD(tf.test.TestCase):

  def test_instantiate(self):

    cases = [
        {
            "kwargs": {},
            "expected": {
                'volume': {
                    'name': 'ssd0',
                    'hostPath': {
                        'path': '/mnt/disks/ssd0'
                    }
                },
                'volume_mount': {
                    'mountPath': '/mnt/ssd0',
                    'name': 'ssd0'
                }
            }
        }
    ]

    for case in cases:
      v = LocalSSD(**case["kwargs"])
      self.assertEqual(object_as_dict(v),
                       case["expected"])


class TestJob(tf.test.TestCase):

  def test_instantiates_job(self):
    """Test our ability to instantiate a job"""

    cases = [
        {
            # Without NFS
            "job_object_args": {
                "job_name": "kittens",
                "command": ["ls"],
                "image": "ubuntu"
            },
            "expected_dict": {
                'apiVersion': 'batch/v1',
                'kind': 'Job',
                'metadata': {
                    'name': 'kittens',
                    'namespace': 'default'
                },
                'spec': {
                    'backoffLimit': 4,
                    'template': {
                        'spec': {
                            'containers': [
                                {
                                    'command': ['ls'],
                                    'image': 'ubuntu',
                                    'name': 'container'
                                }
                            ],
                            'restartPolicy': 'Never'
                        }
                    }
                }
            }
        },
        {
            # With NFS
            "job_object_args": {
                "job_name": "puppies",
                "image": "ubuntu",
                "command": ["ls"],
                "namespace": "kubeflow",
                "volume_claim_id": "nfs-1"
            },
            "expected_dict": {
                'apiVersion': 'batch/v1',
                'kind': 'Job',
                'metadata': {
                    'name': 'puppies',
                    'namespace': 'kubeflow'
                },
                'spec': {
                    'backoffLimit': 4,
                    'template': {
                        'spec': {
                            'containers': [
                                {
                                    'command': ['ls'],
                                    'image': 'ubuntu',
                                    'name': 'container',
                                    'volumeMounts': [{
                                        'mountPath': '/mnt/nfs-1',
                                        'name': 'nfs-1'
                                    }]
                                }
                            ],
                            'restartPolicy': 'Never',
                            'volumes': [{
                                'name': 'nfs-1',
                                'persistentVolumeClaim': {
                                    'claimName': 'nfs-1'
                                }
                            }]
                        }
                    }
                }
            }
        }
    ]

    for case in cases:

      job = Job(**case["job_object_args"])

      pprint.pprint(object_as_dict(job))
      pprint.pprint(case["expected_dict"])

      self.assertEqual(job.as_dict(),
                       case["expected_dict"])

  def test_schedules_on_gpu_node_pool(self):
    """Test that we can schedule a job into a GPU node pool."""

    skip = True

    class ReportGPU(Job):

      def __init__(self, *args, **kwargs):
        """Reveals whether a GPU device is visible via nvidia libs."""
        command = ["/bin/sh", "-c"]
        command_args = [
            ("export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"
             "/usr/local/nvidia/lib64; echo ${LD_LIBRARY_PATH}; "
             "export PATH=$PATH:/usr/local/nvidia/bin/; nvidia-smi")
        ]
        super(ReportGPU, self).__init__(
            command=command,
            command_args=command_args,
            *args, **kwargs
        )

    jid = gen_timestamped_uid()
    job = ReportGPU(**{
        "job_name": jid,
        "image": "tensorflow/tensorflow:latest-gpu",
        "node_selector": {
            "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
        },
        "resources": Resources(limits={"nvidia.com/gpu": 1})
    })

    if not skip:
      create_response = job.batch_run()
      _testing_run_poll_and_check_job(test_object=self,
                                      create_response=create_response,
                                      expect_in_logs="Tesla K80")

  def test_local_ssd(self):
    """Test that num_local_ssd leads to local SSD mount."""

    skip = True

    command = ["ls", "/mnt"]

    jid = gen_timestamped_uid()
    job = Job(**{
        "job_name": jid,
        "image": "ubuntu",
        "num_local_ssd": 1,
        "command": command
    })

    if not skip:

      create_response = job.batch_run()
      _testing_run_poll_and_check_job(test_object=self,
                                      create_response=create_response,
                                      expect_in_logs="ssd0")

  def test_complex_command(self):

    skip = True

    command = ["/bin/sh", "-c"]
    command_args = ["echo hworld1; echo hworld2"]

    jid = gen_timestamped_uid()
    job = Job(**{
        "job_name": jid,
        "image": "ubuntu",
        "command": command,
        "command_args": command_args
    })

    if not skip:
      create_response = job.batch_run()
      _testing_run_poll_and_check_job(test_object=self,
                                      create_response=create_response)


class TestTFJob(tf.test.TestCase):

  def test_instantiate_tfjob_replica(self):
    """Test that a TFJobReplica model can be instantiated."""

    job_name = gen_timestamped_uid()

    _ = TFJobReplica(
        replica_type="MASTER",
        num_replicas=1,
        command_args="pwd",
        image="tensorflow/tensorflow:nightly-gpu",
        resources=Resources(
            requests={
                "cpu": 7,
                "memory": "16Gi",
                "nvidia.com/gpu": 1
            }),
        attached_volumes=[AttachedVolume("nfs-1"),
                          LocalSSD()],
        additional_metadata={
            "master_name": job_name
        },
        node_selector={
            "gpuspernode": 8,
            "highmem": "true",
            "preemptible": "true"
        },
        pod_affinity={"master_name": [job_name]}
    )

  def test_instantiate_tfjob(self):
    """Test that a local TFJob model can be instantiated."""

    skip = True

    image = "tensorflow/tensorflow:nightly-gpu"
    train_resources = Resources(
        limits={"nvidia.com/gpu": 1}
    )
    node_selector = {
        "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
    }

    command = ["/bin/sh", "-c"]
    command_args = [
        ("export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"
         "/usr/local/nvidia/lib64; echo ${LD_LIBRARY_PATH}; "
         "export PATH=$PATH:/usr/local/nvidia/bin/; nvidia-smi")
    ]

    replicas = [
        TFJobReplica(replica_type="MASTER",
                     num_replicas=1,
                     command_args=command_args,
                     command=command,
                     image=image,
                     resources=train_resources,
                     node_selector=node_selector)
    ]

    tfjob = TFJob(command=command,
                  job_name=gen_timestamped_uid(),
                  namespace="kubeflow",
                  replicas=replicas)

    if not skip:

      create_response, _ = tfjob.batch_run()

      _testing_run_poll_and_check_tfjob(
          test_object=self, create_response=create_response,
          expect_in_logs="Tesla K80")

  def test_can_use_tpu(self):
    """Test of whether we can run a job that uses a TPU."""

    skip = False

    image = "gcr.io/clarify/clarify-base:0.0.13"

    train_resources = Resources(limits={
        "cpu": 7,
        "memory": "28Gi",
        "cloud-tpus.google.com/v3": 8
    })

    node_selector = {
        "type": "tpu-host"
    }

    metadata_annotations = {
        "tf-version.cloud-tpus.google.com": "\"1.12\""
    }

    command = ["/bin/sh", "-c"]

    command_args = ["echo TPU_NAME=${TPU_NAME}"]

    replicas = [
        TFJobReplica(replica_type="MASTER",
                     num_replicas=1,
                     command_args=command_args,
                     command=command,
                     image=image,
                     resources=train_resources,
                     node_selector=node_selector)
    ]

    tfjob = TFJob(command=command,
                  job_name=gen_timestamped_uid(),
                  namespace="kubeflow",
                  replicas=replicas,
                  use_tpu=True,
                  annotations=metadata_annotations)

    if not skip:

      create_response, _ = tfjob.batch_run()

      # For not not expecting anything in the logs; checks that
      # training can be performed in this sort of environment is
      # left to experiment_batch_test.py (e.g. checking that logs
      # contain both "Loss for final step" and "Shutdown TPU system").
      _testing_run_poll_and_check_tfjob(
          test_object=self, create_response=create_response,
          expect_in_logs=[])


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()
