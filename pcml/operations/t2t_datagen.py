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

"""Dedicated wrapper for datagen included sharded datagen."""

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

from pcml.operations.tfrecord2bigtable import tfrecord_files_to_cbt_table
from pcml.operations.tfrecord2bigtable import BigTableSelection
from pcml.operations.tfrecord2bigtable import materialize_bigtable_selection


class T2TDatagenJob(PCMLJob):

  def __init__(self, problem_name, data_dir,
               job_name_prefix="datagen",
               image="gcr.io/clarify/basic-runtime:0.0.4",
               num_cpu=7,
               memory="25Gi",
               stage_out_tfrecords=False,
               upload_to_bigtable=False,
               project=None,
               bigtable_instance=None,
               bigtable_table=None,
               column="example",
               column_family="tfexample",
               cbt_max_records=100000000,
               *args, **kwargs):
    """Run T2T datagen optionally with one job per shard."""
    
    self.problem = registry.problem(problem_name)
    # Having attributes train_shards, dev_shards, test_shards

    cmd = "python -m pcml.operations.t2t_datagen "
    cmd += "--problem=%s " % problem_name
    cmd += "--data_dir=%s " % data_dir
    cmd += "--tmp_dir=/mnt/ssd0 "

    # This as opposed to always staging the result out to GCS
    # because (1) we may not always want to keep the resulting
    # TFRecords, and (2) if we are to upload them to BigTable
    # we incur at least an extra 2N of network usage and 3N in
    # the case where we didn't want to keep the records in file
    # form. But - from a developer's perspective it might be a
    # bit simpler to consider these two distinct operations.
    if upload_to_bigtable:
      cmd += "--project=%s " % project
      cmd += "--bigtable_instance=%s " % bigtable_instance
      cmd += "--bigtable_table=%s " % bigtable_table
      cmd += "--column=%s " % column
      cmd += "--column_family=%s " % column_family
      cmd += "--cbt_max_records=%s " % cbt_max_records

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

  def launch_shard_parallel_jobs(self, mock=False, dev_max_num_jobs=None):
    """Launch a datagen job for each shard."""

    app_root = get_pcml_root()
    _compress_and_stage(app_root, self._remote_app_root)
    
    shard_type_lookup = {
      "train": "num_training_shards",
      "dev": "num_dev_shards",
      "test": "num_test_shards"
    }
    
    num_jobs_launched = 0

    uid = gen_timestamped_uid()
    
    self.base_command = self.spec["template"]["spec"]["containers"][0].args[0]

    for shard_type in ["train", "dev", "test"]:

      # Get the number of shards of this type
      shard_attr = shard_type_lookup[shard_type]
      num_shards = getattr(self.problem, shard_attr)
      tf.logging.info("Found %s shards for type %s" % (num_shards,
                                                       shard_type))

      for shard_id in range(num_shards):
        
        job_name = "%s-%s-%s-%s" % (
          self.job_name_prefix, shard_type, shard_id, uid
        )

        if isinstance(dev_max_num_jobs, int):
          if num_jobs_launched >= dev_max_num_jobs:
            return

        shard_command = self.base_command
        shard_command += "--num_shards=%s " % num_shards
        shard_command += "--shard_id=%s " % shard_id
        shard_command += "--shard_type=%s " % shard_type
        self.spec["template"]["spec"]["containers"][0].args[0] = shard_command
        
        # Stage and run
        if not mock:
          
          # TODO: Need to update the job name for each of these so they
          # are distinct.
          self.metadata["name"] = job_name
          
          self.batch_run()
          num_jobs_launched += 1


def _maybe_please_specify_a(flag):
  if getattr(FLAGS, flag) is None:
    raise ValueError("Please specify a %s using --%s" % (flag, flag))


def main(_):

  for flag in ["data_dir", "num_shards", "shard_id", "problem", "tmp_dir",
              "shard_type"]:
    _maybe_please_specify_a(flag)

  if FLAGS.upload_to_bigtable:
    for flag in ["bigtable_instance", "bigtable_table", "project"]:
      _maybe_please_specify_a(flag)

    cbt_selection = BigTableSelection(
      project=FLAGS.project,
      instance=FLAGS.bigtable_instance,
      table=FLAGS.bigtable_table,
      prefix=FLAGS.shard_type, # Attention: prefix is shard_type
      column_family=FLAGS.column_family,
      column_qualifier=FLAGS.column)

    (cbt_client,
     cbt_table,
     cbt_instance) = materialize_bigtable_selection(cbt_selection)
    
    # TODO: In the future we can write here from the problem generator
    # instead of going by way of disk. For now the above are not used
    # until after examples are written to disk.

  if FLAGS.stage_out_tfrecords:
    if not FLAGS.data_dir.startswith("gs://"):
      msg = ("If FLAGS.stage_out_tfrecords expecting GCS "
             "FLAGS.data_dir, saw %s" % FLAGS.data_dir)
      raise ValueError(msg)

  problem = registry.problem(FLAGS.problem)

  local_tfrecords_filepath = problem.distributed_generate_data(
    data_dir=FLAGS.tmp_dir,
    tmp_dir=FLAGS.tmp_dir, # Attention: TFRecords are written to tmp_dir
                           # which in this case is local SSD storage and
                           # only copied out to GCS if
                           # FLAGS.stage_out_tfrecords.
    shard_type=FLAGS.shard_type,
    shard_id=FLAGS.shard_id,
    num_shards=FLAGS.num_shards)

  # ======
  # Stage out TFRecords to GCS
  if FLAGS.stage_out_tfrecords:
    filename = os.path.split(local_tfrecords_filepath)[-1]
    remote_tfrecords_filepath = os.path.join(FLAGS.data_dir, filename)
    tf.gfile.Copy(local_tfrecords_filepath, remote_tfrecords_filepath)

  # ======
  # CBT upload 
  if FLAGS.upload_to_bigtable:
    tfrecord_files_to_cbt_table(glob=local_tfrecords_filepath,
                                # Attention: The TFRecord file we just
                                # generated.
                                table=cbt_table,
                                selection=cbt_selection,
                                max_records=FLAGS.cbt_max_records)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string("data_dir", None, "Data directory for TFRecord files.")
  flags.DEFINE_string("tmp_dir", None, "Temporary storage directory.")
  flags.DEFINE_string("problem", None,
                      "The name of the problem for which to generate data.")
  flags.DEFINE_integer("num_shards", None, "Number of shards.")
  flags.DEFINE_integer("shard_id", None, "The shard for which to generate data.")
  flags.DEFINE_string("shard_type", None,
                       "The shard type in ['train', 'dev', 'test']")
  flags.DEFINE_bool("stage_out_tfrecords", True,
                    "Whether to stage resulting tfrecords to remote storage.")
  flags.DEFINE_bool("upload_to_bigtable", True,
                    "Whether to load the resulting examples into BigTable.")

  flags.DEFINE_string('bigtable_instance', None, 'The Cloud Bigtable instance.')
  flags.DEFINE_string('bigtable_table', None, 'The table within the instance to '
                      'write to.')
  flags.DEFINE_string('project', None, 'The Project to use. (Optional if running '
                      'on a Compute Engine VM, as it can be auto-determined from '
                      'the metadata service.)')
  flags.DEFINE_integer(
      'cbt_max_records', 100000000, 'The approximate dataset size (used for padding '
      'the appropriate number of zeros when constructing row keys). It should '
      'not be smaller than the actual number of records.')
  flags.DEFINE_integer('num_parallel_reads', 1, 'The number of parallel reads '
                       'from the source file system.')
  flags.DEFINE_string('column_family', 'tfexample',
                      'The column family to write the data into.')
  flags.DEFINE_string('column', 'example',
                      'The column name (qualifier) to write the data into.')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
