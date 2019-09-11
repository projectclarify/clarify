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


class ExportJob(PCMLJob):

  def __init__(self,
               output_dir,
               staging_path,
               pcml_tgz_path,
               tmp_dir="/mnt/ssd0",
               job_name_prefix="export",
               num_cpu=1,
               memory="6Gi",
               image="gcr.io/clarify/basic-runtime:0.0.4",
               *args, **kwargs):

    cmd = "python -m pcml.operations.export "
    cmd += "--output_dir={}".format(output_dir)

    super(ExportJob, self).__init__(
      job_name="export-{}".format(gen_timestamped_uid()),
      command=["/bin/sh", "-c"],
      command_args=self.build_cmd(),
      namespace="kubeflow",
      image=image,
      staging_path=pcml_tgz_path,
      resources=Resources(
        limits={"cpu": num_cpu, "memory": memory}),
      *args, **kwargs)


def main(_):

  if FLAGS.batch:
    job = ExportJob(output_dir=FLAGS.output_dir,
                    staging_path=FLAGS.staging_path,
                    pcml_tgz_bundle=FLAGS.pcml_tgz_bundle)

    # Should be able to on the fly make a job object and build a command
    # that sets all of the local variables as parameters - whaterver command
    # you run if it includes --batch that exact command will be run as a PCMLJob
    # via some default staging path

    # I.e. without staging anything since we want to use what's already
    # there.
    return job.stage_and_batch_run()

  version_path = setup_pcml_version(FLAGS.staging_path)

  FLAGS.t2t_usr_dir = version_path #??

  # Make pmcl registered models available
  import pcml
  export.main(_)


if __name__ == "__main__":

  from tensor2tensor.serving.export import FLAGS

  tf.flags.DEFINE_boolean('batch', False,
                          'Whether to run in batch or locally.')

  tf.flags.DEFINE_string('staging_path', False,
                         'Path to which to stage.')

  tf.flags.DEFINE_string('pcml_tgz_bundle', False,
                         'Path to PCML bundle to use when running.')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
