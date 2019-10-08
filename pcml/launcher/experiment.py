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

"""Extend t2t-trainer with startup tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import shutil

import tensorflow as tf

from tensor2tensor.bin.t2t_trainer import FLAGS
from tensor2tensor.bin.t2t_trainer import main as trainer_main

from pcml.launcher.kube import AttachedVolume
from pcml.launcher.kube import LocalSSD
from pcml.launcher.kube import Resources
from pcml.launcher.kube import TFJob
from pcml.launcher.kube import TFJobReplica
from pcml.launcher.util import hack_dict_to_cli_args
from pcml.launcher.util import generate_job_name
from pcml.launcher.util import _compress_and_stage

from multiprocessing import Pool, Process
import time

from pcml.operations.eval import trigger_eval


# HACK: To be able to read data from BigTable
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


def construct_job_command(remote_app_root, train_args, use_katib=False, 
                          vendor_t2t=False):
  """Construct a job command string.

  TODO: Consider refactoring as a jinja template.
  TODO: Separate core logging and operations that should always be part of
  pcml.launcher.experiment from the job command the user will construct.

  Args:
    remote_app_root(str): A path to a directory on GCS holding
      staged pcml.tgz and to which checkpoints will be written.
    train_args(dict): A dictionary of additional command-line
      arguments to convert to "--key=value,..." form and append
      to the python -m pcml.launcher.experiment command.
    use_katib(bool): Whether to add Katib-related template fields to
      job command (relevant in the event that one is using
      pcml.launcher.study.StudyJob for hyperparameter tuning.

  """

  cmd = []

  pcml_fname = "pcml-0.0.1.tar.gz"
  remote_path = "%s/%s" % (remote_app_root, pcml_fname)

  cmd.append("unset TF_CONFIG") # HACK
  # TODO: Understand why not unsetting TF_CONFIG causes jobs to hang;
  # likely having to do with our custom code that parses the TF_CONFIG
  # into t2t_trainer FLAGS.

  # HACK ========================================================
  # Datagen container needs ffmpeg and other deps that are there but
  # don't want to occupy a GPU thus need non-gpu tensorflow and don't
  # want to build another container at the moment.
  # cmd.append("pip uninstall -y tensorflow; pip install
  # tensorflow==1.13.1")
  # =============================================================

  # Install some dependencies
  # TODO: Move this to the base container
  cmd.append(
      ("apt-get update && apt-get install -y "
       "libsm6 libxext6 libxrender-dev libglib2.0-0 htop")
  )

  # Copy down pcml code
  cmd.append(
      ("python -c 'import tensorflow as tf; "
       "tf.gfile.Copy(\"%s\", \"/tmp/%s\", overwrite=True)'" % (
           remote_path, pcml_fname
       )
      )
  )

  # Decompress code tarball
  cmd.append("cd /tmp; tar -xzvf /tmp/%s" % pcml_fname)

  # Install vendored t2t if requested
  if vendor_t2t:
    cmd.append("cd /tmp/pcml-0.0.1; pip install -e vendor/tensor2tensor")

  # Install PCML
  cmd.append("cd /tmp/pcml-0.0.1; pip install -e .")

  # ====================
  # HACK
  # Experimental patch of tpu_estimator.py in exploration of
  # https://github.com/tensorflow/tensorflow/issues/30869
  cmd.append("python -m pcml.utils.patch_tpu_estimator")
  # ====================

  # Print TensorFlow version
  cmd.append("python -c 'import tensorflow as tf; print(tf.__version__)'")

  # Verify the GPU is visible by nvidia-smi as well as tf device_lib
  #cmd.append("nvidia-smi")
  cmd.append(
      ("python -c 'from tensorflow.python.client import device_lib; "
       "print(device_lib.list_local_devices())'")
  )

  # Log the TF_CONFIG
  cmd.append("echo ${TF_CONFIG}")

  # Construct the trainer command
  #train_cmd = ["python", "-m", "pcml.launcher.experiment --generate_data"]
  train_cmd = ["python", "-m", "pcml.launcher.experiment"]
  train_cmd.extend(hack_dict_to_cli_args(train_args))

  # Finalize the train_cmd for non-tuning usage
  train_cmd_str = " ".join(train_cmd)

  if use_katib:
    # Extend the train_cmd to template Katib-provided hyperparameters
    # as command-line arguments
    train_cmd_str += " --hparams='{{- with .HyperParameters}}{{- range .}}"
    train_cmd_str += "{{.Name}}={{.Value}},{{- end}}{{- end}}'"

  cmd.append(train_cmd_str)

  # Finalize the complete command
  cmd = "; ".join(cmd)

  return [cmd]


class T2TKubeExperiment(TFJob):
  """A Job that runs a training experiment."""

  def __init__(self,
               app_root,
               image,
               num_worker_replicas=0,
               num_ps_replicas=0,
               cpu=1,
               memory="1Gi",
               master_gpu=0,
               worker_gpu=0,
               ps_gpu=0,
               selector_labels={},
               num_local_ssd=0,
               remote_app_root=None,
               train_args=None,
               annotations=None,
               use_tpu=False,
               tpu_type="v3",
               num_tpu_cores=0,
               volumes=[],
               use_katib=False,
               vendor_t2t=False,
               *args, **kwargs):
    """Configure a T2TKubeExperiment object.

    Vars:
        app_root (str):
        image (str):
        num_worker_replicas (int):
        num_ps_replicas (int):
        cpu (int):
        memory (str):
        pvc_id (str): ID of a PVC to mount to the training volume.
        stage_data_dir_to_local_ssd (bool): HACK: Whether to mount local
            SSD and at the start of training stage contents
        selector_labels (dict):

    """

    # HACK: Should check type at arg parser level not here, this change
    # is part of a larger change in way additional args are loaded i.e.
    # parsing FLAGS for called apps instead of providing a dict template.
    if isinstance(num_worker_replicas, str):
      num_worker_replicas = int(num_worker_replicas)

    nwr_not_int = (not isinstance(num_worker_replicas, int))
    if nwr_not_int or num_worker_replicas < 0:
      raise ValueError("The number of worker replicas must be an "
                       "integer greater than or equal to zero.")

    if (not isinstance(num_ps_replicas, int) or
        num_ps_replicas < 0):
      raise ValueError("The number of ps replicas must be an "
                       "integer greater than or equal to zero.")

    cmd = construct_job_command(remote_app_root, train_args,
                                use_katib=use_katib,
                                vendor_t2t=vendor_t2t)

    self._remote_app_root = remote_app_root
    self._train_args = train_args

    # TODO: For now, just run all components with the same resources.

    limits = {
        "cpu": cpu,
        "memory": memory
    }

    if not use_tpu and master_gpu > 0:
      limits.update({"nvidia.com/gpu": master_gpu})

    elif use_tpu and num_tpu_cores > 0:
      request_type = "cloud-tpus.google.com/%s" % tpu_type
      limits.update({request_type: num_tpu_cores})

    resources = Resources(limits=limits)

    # HACK: For simplicity / development
    master_resources = resources
    worker_resources = resources
    ps_resources = resources

    for v in volumes:
      if not isinstance(v, AttachedVolume):
        raise ValueError(
            "volume arguments must be of type AttachedVolume, saw %s" % v)

    if num_local_ssd > 0:
      for i in range(num_local_ssd):
        volumes.append(LocalSSD(disk_id=i))

    if not volumes:
      volumes = None

    replicas = [
        TFJobReplica(replica_type="MASTER",
                     num_replicas=1,
                     command=["/bin/sh", "-c"],
                     command_args=cmd,
                     image=image,
                     resources=master_resources,
                     attached_volumes=volumes,
                     node_selector=selector_labels,
                     annotations=annotations)
    ]

    if num_ps_replicas > 0:
      replicas.append(
          TFJobReplica(replica_type="PS",
                       num_replicas=num_ps_replicas,
                       command=["/bin/sh", "-c"],
                       command_args=cmd,
                       image=image,
                       resources=ps_resources,
                       attached_volumes=volumes,
                       node_selector=selector_labels,
                       annotations=annotations)
      )

    if num_worker_replicas > 0:
      replicas.append(
          TFJobReplica(replica_type="WORKER",
                       num_replicas=num_worker_replicas,
                       command=["/bin/sh", "-c"],
                       command_args=cmd,
                       image=image,
                       resources=worker_resources,
                       attached_volumes=volumes,
                       node_selector=selector_labels,
                       annotations=annotations)
      )

    super(T2TKubeExperiment, self).__init__(command="",
                                            replicas=replicas,
                                            *args, **kwargs)


def tf_config_to_additional_flags():
  """Read TF_CONFIG and set relevant t2t FLAGS."""

  if "TF_CONFIG" not in os.environ:
    tf.logging.info("No TF_CONFIG present, returning dummy.")
    task_type = "master"
    tid = 0
    #FLAGS.master = None
    #FLAGS.ps_replicas = 0
    #FLAGS.worker_id = tid
    #FLAGS.worker_job = '/job:%s' % task_type
    #FLAGS.worker_gpu = 0
    #FLAGS.worker_replicas = 1
    #FLAGS.schedule = 'train'
    return task_type, 0

  tf_config = os.environ["TF_CONFIG"]

  tf_config = json.loads(tf_config)

  tf.logging.info("Loaded TF_CONFIG: %s" % tf_config)

  if "cluster" not in tf_config:
    raise ValueError("TF_CONFIG environment variable should always "
                     "have a 'cluster' field, saw %s" % tf_config)

  cluster_spec = tf_config["cluster"]

  if "master" not in cluster_spec or not cluster_spec["master"]:
    raise ValueError("Expected at least one master defined in "
                     "master field of cluster_spec.")

  masters = cluster_spec["master"]
  num_masters = len(masters)
  tf.logging.info("num_masters: %s" % num_masters)

  ps_tasks = cluster_spec.get("ps", [])
  num_ps = len(ps_tasks)
  tf.logging.info("num_ps: %s" % num_ps)

  worker_tasks = cluster_spec.get("worker", [])
  num_workers = len(worker_tasks)
  tf.logging.info("worker_tasks: %s" % num_workers)

  master_address = "grpc://%s" % masters[0]
  tf.logging.info("master address: %s" % master_address)

  tid = tf_config["task"]["index"]
  task_type = tf_config["task"]["type"]

  FLAGS.master = master_address
  FLAGS.ps_replicas = num_ps

  if task_type == "ps":
    FLAGS.schedule = "run_std_server"
    return task_type, tid

  FLAGS.worker_id = tid
  FLAGS.worker_job = '/job:%s' % task_type
  FLAGS.worker_gpu = 0
  FLAGS.worker_replicas = 1

  FLAGS.sync = True
  #FLAGS.schedule = 'train'

  return task_type, tid


def _stage(local_app_root, remote_app_root):
  """Stage data from `local_app_root` to `remote_app_root`.

  Args:
      local_app_root (str): Directory path on local FS.
      remote_app_root (str): Directory path on remote FS.
  """

  if not os.path.exists(local_app_root):
    raise ValueError("Can't stage from a non-existent source, "
                     "saw %s" % local_app_root)

  shutil.copytree(local_app_root, remote_app_root)


def _expect_non_negative_int(val):

  if not isinstance(val, int):
    raise ValueError("Expected integer value, saw %s" % val)

  if val < 0:
    raise ValueError("Expected non-negative value, saw %s" % val)


def build_accelerator_args(gpu_type="nvidia-tesla-k80",
                           num_gpu_per_worker=1,
                           use_tpu=False, num_tpu_cores=0,
                           tpu_tf_version="1.13",
                           num_gpu_per_ps=0, tpu_type="v3"):
  """Build internal configs for use of accelerators.

  Returns:
    (dict, dict): Dictionary updats to the train_args and experiment_args
      dicts of configure_experiment.

  """

  cfg = {
      "use_tpu": use_tpu,
      "num_tpu_cores": num_tpu_cores,
      "num_gpu_per_worker": num_gpu_per_worker
  }

  _expect_non_negative_int(num_tpu_cores)
  _expect_non_negative_int(num_gpu_per_worker)

  if (use_tpu or num_tpu_cores > 0) and num_gpu_per_worker > 0:
    raise ValueError(
        "Saw request for both GPU and TPU accelerators, %s" % cfg)

  if (not use_tpu or num_tpu_cores == 0) and (num_gpu_per_worker == 0):
    return {}, {} # Not using accelerators

  if use_tpu and num_tpu_cores == 0:
    raise ValueError("Saw conflicting use_tpu and num_tpu_cores, %s" % cfg)

  if use_tpu:

    extra_train_args = {
        "use_tpu": True
    }

    extra_experiment_args = {
        "annotations": {
            "tf-version.cloud-tpus.google.com": tpu_tf_version
        },
        "num_tpu_cores": num_tpu_cores,
        "use_tpu": True # Used to signal using tpu at Kubernetes-level
    }

    return extra_train_args, extra_experiment_args

  else:

    extra_train_args = {
        "worker_gpu": num_gpu_per_worker,
        "ps_gpu": num_gpu_per_ps,
        "worker_gpu_memory_fraction": 0.95
    }

    extra_experiment_args = {}

    return extra_train_args, extra_experiment_args


def configure_experiment(base_name,
                         problem,
                         model,
                         hparams_set,
                         remote_base,
                         num_train_steps,
                         num_gpu_per_worker=0,
                         num_eval_steps=100,
                         local_eval_frequency=90,
                         num_workers=0,
                         num_ps=0,
                         ps_gpu=0,
                         log_device_placement=False,
                         profile=False,
                         dbgprofile=False,
                         extra_hparams={},
                         trainer_memory="7Gi",
                         trainer_cpu=4,
                         app_root="/home/jovyan/work/pcml",
                         base_image="tensorflow/tensorflow:1.13.1-py3",
                         reuse_output_dir=None,
                         schedule="train",
                         data_dir="/mnt/disks/ssd0",
                         tmp_dir="/mnt/disks/ssd0",
                         gpu_type="nvidia-tesla-k80",
                         use_katib=False,
                         use_tpu=True,
                         num_tpu_cores=8,
                         tpu_tf_version="1.14",
                         selector_labels={"type": "tpu-host"},
                         save_checkpoints_secs=1800,
                         vendor_t2t=False,
                         stage_and_install=False,
                         **kwargs):
  """Wrapper to construct args object and produce job scripts.

  Args:
      base_name (str): The base name to be used to identify the
        experiment.
        
  TODO: It would be nice to convert this over to the PCMLJob style.

  """

  # TODO: Address linter complaints about dangerous-default-value for
  # those {} defaults above.

  # TODO: Address linter complaints about keyword-arg-before-vararg

  # Generate a unique job name that includes `base_name`
  job_name = generate_job_name(base_name)

  # Construct remote app root and checkpoint output dirs
  remote_app_root = "%s/%s" % (remote_base, job_name)

  checkpoint_output_dir = os.path.join(remote_app_root, "output")

  if isinstance(reuse_output_dir, str):
    # Apply output dir override
    checkpoint_output_dir = reuse_output_dir

  if use_katib:
    checkpoint_output_dir = os.path.join(remote_app_root,
                                         "{{.WorkerID}}")
    job_name = "{{.WorkerID}}"

  volumes = []

  # Translate extra_hparams into a comma-separated 'key=value,...'
  hparams = ""
  for k, v in extra_hparams.items():
    if hparams:
      hparams += ","
    hparams += "%s=%s" % (k, v)
  hparams_str = "'%s'" % hparams

  # Arguments for tensor2tensor t2t_trainer call
  train_args = {
      "problem": problem,
      "model": model,
      "hparams_set": hparams_set,
      "data_dir": data_dir,
      "output_dir": checkpoint_output_dir,
      "train_steps": num_train_steps,
      "eval_steps": num_eval_steps,
      "schedule": schedule,
      "profile": profile,
      "log_device_placement": log_device_placement,
      "worker_gpu": num_gpu_per_worker,
      "ps_gpu": ps_gpu,
      "save_checkpoints_secs": save_checkpoints_secs,
      "dbgprofile": dbgprofile,
      "ssd_mount_path": "/mnt/disks/ssd0",
      "tmp_dir": tmp_dir,
      "worker_gpu_memory_fraction": 0.95,
      # On the workers, the pcml code will reside at /tmp/pcml
      "t2t_usr_dir": "/tmp/pcml/pcml",
      "hparams": hparams_str,
      "local_eval_frequency": local_eval_frequency
  }

  extra_train_args, extra_experiment_args = build_accelerator_args(
      gpu_type=gpu_type,
      num_gpu_per_worker=num_gpu_per_worker,
      use_tpu=use_tpu,
      num_tpu_cores=num_tpu_cores,
      tpu_tf_version=tpu_tf_version
  )

  train_args.update(extra_train_args)

  _compress_and_stage(app_root, remote_app_root)

  experiment_args = {
      "job_name": job_name,
      "app_root": app_root,
      "namespace": "kubeflow",
      "image": base_image,
      "smoke": True,
      "train_args": train_args,
      "cpu": trainer_cpu,
      "memory": trainer_memory,
      "num_gpu": num_gpu_per_worker,
      "master_gpu": num_gpu_per_worker,
      "ps_gpu": ps_gpu,
      "worker_gpu": num_gpu_per_worker,
      "num_local_ssd": 1,
      "no_wait": True,
      "num_worker_replicas": num_workers,
      "num_ps_replicas": num_ps,
      "selector_labels": selector_labels,
      "remote_app_root": remote_app_root,
      "volumes": volumes,
      "use_katib": use_katib,
      "tpu_type": tpu_type,
      "vendor_t2t": vendor_t2t
  }

  experiment_args.update(extra_experiment_args)

  return T2TKubeExperiment(**experiment_args)


def periodic_eval_old(args):
  """Run eval by way of T2T Problem framework (graph mode).
  
  Doesn't currently work for pcml primary models which is
  the reason eval is being done with Eager (in a separate
  process) instead. For potential future use - this is a
  more concise.
  
  """

  FLAGS = args[0]
  time.sleep(10) # startup delay

  inter_eval_delay = 600

  while True:

    FLAGS.use_tpu = False # Eval on VM not TPU.
    FLAGS.schedule = "evaluate"

    batch_size = 360
    batch_size_arg = "batch_size=%s" % batch_size
    extra_hparams = FLAGS.hparams.split(",")

    def _maybe_set_batch_size():
      for i, hp in enumerate(extra_hparams):
        if hp.startswith("batch_size="):
          extra_hparams[i] = batch_size_arg
          return ",".join(extra_hparams)
      extra_hparams.append(batch_size_arg)
      return ",".join(extra_hparams)

    FLAGS.hparams = _maybe_set_batch_size()

    if FLAGS.hparams.startswith(","):
      FLAGS.hparams = FLAGS.hparams[1:]

    # Doesn't currently work
    trainer_main(None)

    time.sleep(inter_eval_delay)


def main(argv):
  """Configure, setup logging, and train."""

  # This modifies the FLAGS global which is used by
  # trainer_main.
  _, _ = tf_config_to_additional_flags()

  # HACK ==
  tf.gfile.MakeDirs(FLAGS.output_dir)
  #final_output_dir = FLAGS.output_dir
  #FLAGS.output_dir = "/mnt/disks/ssd0"
  # =======

  # Start evaluation running in Eager mode periodically
  # in a background system process.
  if FLAGS.use_tpu :
    p = Process(target=trigger_eval,
                args=(FLAGS.output_dir, FLAGS.data_dir, FLAGS.eval_steps))
    p.start()

  # Run training
  trainer_main(None)

  #trigger_eval((FLAGS.output_dir,))

  # Stop the eval thread
  if FLAGS.use_tpu:
    # This doesn't have to be completely clean because any
    # descendent processes will exit by virtue of the PID
    # 1 process (and thus the Job) terminating. But it could
    # be made to be using Popen instead of check_output.
    p.terminate()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()
