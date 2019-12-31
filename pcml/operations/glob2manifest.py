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
"""A kubernetes Job to write file paths matching glob pattern to path manifest

E.g.
python -m pcml.operations.glob2manifest \
  --source_glob=gs://clarify-dev/tmp/03312019/pcml-new/*.py \
  --target_path=gs://clarify-dev/tmp/beam/devmanifest2.txt

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
from pcml.launcher.kube import Job
import tensorflow as tf

from pcml.launcher.kube import gen_timestamped_uid

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('source_glob', None, 'The GCS glob pattern to search.')
flags.DEFINE_string('target_path', None,
                    'The full output path for the manifest.')

from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid


class Glob2Manifest(PCMLJob):

    def __init__(self, source_glob, target_path, *args, **kwargs):

        cmd = []

        cmd.append("python -m pcml.operations.glob2manifest")
        cmd.append("--target_path=%s" % target_path)
        cmd.append("--source_glob=%s" % source_glob)

        cmd = " ".join(cmd)

        command = ["/bin/sh", "-c"]
        command_args = [cmd]

        job_name_prefix = "glob2manifest"
        job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())

        super(Glob2Manifest,
              self).__init__(job_name=job_name,
                             command=command,
                             command_args=command_args,
                             namespace="kubeflow",
                             image="gcr.io/clarify/basic-runtime:0.0.3",
                             *args,
                             **kwargs)


def run(source_glob, target_path):
    tf.logging.info("Searching for files matching glob string: %s" %
                    source_glob)
    files = tf.gfile.Glob(source_glob)
    tf.logging.info("Found %s files matching glob pattern." % len(files))
    with tf.gfile.Open(target_path, "w") as f:
        for filename in files:
            f.write(filename + "\n")
    tf.logging.info("Successfully wrote file paths to output: %s" % target_path)


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.source_glob is None:
        raise ValueError("Please provide a source glob with --source_glob.")

    if FLAGS.target_path is None:
        raise ValueError("Please provide a target path with --target_path.")

    run(FLAGS.source_glob, FLAGS.target_path)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
