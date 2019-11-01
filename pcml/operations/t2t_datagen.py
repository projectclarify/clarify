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

"""Dedicated wrapper for datagen including sharded datagen."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensor2tensor.utils import registry

from pcml.launcher.kube import Resources
from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid

from pcml.utils.fs_utils import get_pcml_root
from pcml.launcher.util import _compress_and_stage


class T2TDatagenJob(PCMLJob):

  def __init__(self, problem_name, mode, data_dir,
               job_name_prefix="datagen",
               image="gcr.io/clarify/basic-runtime:0.0.4",
               num_cpu=7,
               memory="25Gi",
               *args, **kwargs):
    """Run T2T datagen optionally with one job per shard."""
    
    self.problem = registry.problem(problem_name)
    # Having attributes train_shards, dev_shards, test_shards

    cmd = "python -m pcml.operations.t2t_datagen "
    cmd += "--problem=%s " % problem_name
    cmd += "--mode=%s " % mode
    cmd += "--data_dir=%s " % data_dir
    cmd += "--tmp_dir=/mnt/ssd0 "

    command = ["/bin/sh", "-c"]
    command_args = [cmd]

    job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())
    # This is the job_name that will be used if we directly call
    # .batch_run without doing so by way of launch_shard_parallel_jobs.
    # In the latter case, new (informative) job names will be
    # constructed using this same prefix.
    self.job_name_prefix = job_name_prefix

    super(T2TDatagenJob, self).__init__(
      job_name=job_name,
      command=command,
      command_args=command_args,
      namespace="kubeflow",
      image=image,
      num_local_ssd=1,
      resources=Resources(limits={"cpu": num_cpu, "memory": memory}),
      *args, **kwargs)


def _maybe_please_specify_a(flag):
  if getattr(FLAGS, flag) is None:
    raise ValueError("Please specify a %s using --%s" % (flag, flag))


def main(_):

  for flag in ["data_dir", "num_shards", "shard_id", "problem", "tmp_dir",
               "mode"]:
    _maybe_please_specify_a(flag)

  problem = registry.problem(FLAGS.problem)

  local_tfrecords_filepath = problem.generate_data(
    data_dir=FLAGS.tmp_dir, tmp_dir=FLAGS.tmp_dir,
    task_id=FLAGS.shard_id)

  # Stage out TFRecords to GCS
  filename = os.path.split(local_tfrecords_filepath)[-1]
  remote_tfrecords_filepath = os.path.join(FLAGS.data_dir, filename)
  tf.gfile.Copy(local_tfrecords_filepath, remote_tfrecords_filepath)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string("data_dir", None, "Data directory for TFRecord files.")
  flags.DEFINE_string("tmp_dir", None, "Temporary storage directory.")
  flags.DEFINE_string("problem", None,
                      "The name of the problem for which to generate data.")
  flags.DEFINE_integer("num_shards", None, "Number of shards.")
  flags.DEFINE_integer("shard_id", None, "The shard for which to generate data.")
  flags.DEFINE_string("mode", None,
                       "The shard type in ['train', 'dev', 'test']")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
