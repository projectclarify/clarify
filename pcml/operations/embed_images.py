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
"""A kubernetes Job to embed a collection of images in tf.Eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
from pcml.launcher.kube import Job

import tensorflow as tf
tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name

from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid

_SUCCESS_MESSAGE = "Successfully completed batch image embedding."


class EmbedImages(PCMLJob):

  def __init__(self, input_manifest, target_csv, ckpt_path, problem_name,
               model_name, hparams_set_name, *args, **kwargs):

    cmd = []

    cmd.append("python -m pcml.operations.embed_images")
    cmd.append("--input_manifest=%s" % input_manifest)
    cmd.append("--target_csv=%s" % target_csv)
    cmd.append("--ckpt_path=%s" % ckpt_path)
    cmd.append("--problem_name=%s" % problem_name)
    cmd.append("--model_name=%s" % model_name)
    cmd.append("--hparams_set_name=%s" % hparams_set_name)

    cmd = " ".join(cmd)

    command = ["/bin/sh", "-c"]
    command_args = [cmd]

    job_name_prefix = "embed-images"
    job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())

    super(EmbedImages,
          self).__init__(job_name=job_name,
                         command=command,
                         command_args=command_args,
                         namespace="kubeflow",
                         image="gcr.io/clarify/basic-runtime:0.0.3",
                         *args,
                         **kwargs)


def run(input_manifest, target_csv, ckpt_path, problem_name, model_name,
        hparams_set_name):

  problem_obj = registry.problem(problem_name)
  hparams = registry.hparams_set(hparams_set_name)
  hparams.data_dir = "foo"
  p_hparams = problem_obj.get_hparams(hparams)
  mode = "predict"
  ModelObj = registry.model(model_name)

  with tfe.restore_variables_on_create(tf.train.latest_checkpoint(ckpt_path)):
    model = ModelObj(hparams, mode, p_hparams)
  """

  Question: Should the same problem that was used in training be used to iterate images
  at embedding time? Probably yes if we're going to need to apply the same kind of
  standardization.

  """

  tf.logging.info(locals())


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  run(FLAGS.input_manifest, FLAGS.target_csv, FLAGS.ckpt_path,
      FLAGS.problem_name, FLAGS.model_name, FLAGS.hparams_set_name)

  tf.logging.info(_SUCCESS_MESSAGE)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string('input_manifest', None,
                      'An csv of (image_path, labels_path).')

  flags.DEFINE_string('target_csv', None,
                      'CSV path to which to write embeddings and labels.')

  flags.DEFINE_string('ckpt_path', None, 'Path to model checkpoints to restore')

  flags.DEFINE_string('problem_name', None,
                      'T2T problem name (possibly dummy).')

  flags.DEFINE_string('model_name', None, 'T2T model name.')

  flags.DEFINE_string('hparams_set_name', None, 'T2T hparams set name.')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
