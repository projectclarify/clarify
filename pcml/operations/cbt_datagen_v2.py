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
"""Cloud BigTable-centric T2T datagen, leaving particulars to Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

import os
import math
import json

import tempfile

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem

from pcml.utils import cbt_utils

from pcml.operations import extract

from pcml.launcher.kube import Resources
from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid

from pcml.utils.cmd_utils import run_and_output
from pcml.utils.fs_utils import get_pcml_root


class CBTDatagenJobV2(PCMLJob):

    def __init__(self,
                 problem_name,
                 project,
                 instance,
                 mode,
                 job_name_prefix="cbt-datagen",
                 image="gcr.io/clarify/basic-runtime:0.0.4",
                 num_cpu=1,
                 memory="6Gi",
                 *args,
                 **kwargs):

        cmd = "python -m pcml.operations.cbt_datagen_v2 "
        cmd += "--problem_name=%s " % problem_name
        cmd += "--project=%s " % project
        cmd += "--instance=%s " % instance
        cmd += "--mode=%s " % mode

        command = ["/bin/sh", "-c"]
        command_args = [cmd]

        job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())
        self.job_name_prefix = job_name_prefix

        super(CBTDatagenJobV2, self).__init__(job_name=job_name,
                                              command=command,
                                              command_args=command_args,
                                              namespace="kubeflow",
                                              image=image,
                                              num_local_ssd=1,
                                              resources=Resources(limits={
                                                  "cpu": num_cpu,
                                                  "memory": memory
                                              }),
                                              *args,
                                              **kwargs)


def log_flags(flags):
    for key in flags:
        tf.logging.info("%s: %s" % (key, getattr(flags, key)))


def main(_):

    log_flags(FLAGS)

    prob = registry.problem(FLAGS.problem_name)

    prob.mode = FLAGS.mode

    prob.cbt_generate(project=FLAGS.project,
                      instance=FLAGS.instance,
                      mode=FLAGS.mode,
                      shard_id=FLAGS.shard_id)


if __name__ == "__main__":

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('mode', None, 'One of train, eval, or test.')

    flags.DEFINE_string('project', None, 'A GCP project.')

    flags.DEFINE_string('instance', None, 'A Google Cloud BigTable instance.')

    flags.DEFINE_string('problem_name', None, 'A registered t2t problem name.')

    flags.DEFINE_integer('shard_id', -1, 'The shard ID.')

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
