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
"""Cloud BigTable-centric T2T datagen."""

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
from pcml.utils.cbt_utils import VideoMeta


@registry.register_problem
class CBTDatagenTestProblem(problem.Problem):

    def sampling_generator(self, source_selection):
        """A trivial example generator that just yields sampled video segments."""

        sample_generator = source_selection.sample_av_correspondence_examples(
            frames_per_video=15, max_num_samples=10)

        for sample_set in sample_generator:

            for example_type in sample_set:

                yield {
                    "audio": sample_set[example_type].audio.tolist(),
                    "video": sample_set[example_type].video.flatten().tolist(),
                    "target": [sample_set[example_type].labels["overlap"]]
                }


class CBTDatagenJob(PCMLJob):

    def __init__(self,
                 problem_name,
                 project,
                 bigtable_instance,
                 prefix,
                 max_num_examples=-1,
                 bigtable_source_table_name="",
                 bigtable_target_table_name="",
                 job_name_prefix="cbt-datagen",
                 image="gcr.io/clarify/basic-runtime:0.0.4",
                 num_cpu=1,
                 memory="6Gi",
                 *args,
                 **kwargs):
        """Sample raw subvideos from source, augment, and write to target."""

        cmd = "python -m pcml.operations.cbt_datagen "
        cmd += "--problem_name=%s " % problem_name
        cmd += "--project=%s " % project
        cmd += "--bigtable_instance=%s " % bigtable_instance
        cmd += "--bigtable_source_table=%s " % bigtable_source_table_name
        cmd += "--bigtable_target_table=%s " % bigtable_target_table_name
        cmd += "--max_num_examples=%s " % max_num_examples
        cmd += "--prefix=%s " % prefix

        command = ["/bin/sh", "-c"]
        command_args = [cmd]

        job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())
        # This is the job_name that will be used if we directly call
        # .batch_run without doing so by way of launch_shard_parallel_jobs.
        # In the latter case, new (informative) job names will be
        # constructed using this same prefix.
        self.job_name_prefix = job_name_prefix

        super(CBTDatagenJob, self).__init__(job_name=job_name,
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


def cbt_generate_and_load_examples(project,
                                   bigtable_instance,
                                   bigtable_source_table_name,
                                   bigtable_target_table_name,
                                   prefix,
                                   problem_name,
                                   max_num_examples=None):

    problem = registry.problem(problem_name)

    if not hasattr(problem, "sampling_generator"):
        msg = ("Problem {} is not compatible, must "
               "implement an {} method.").format(problem_name, attr)
        raise ValueError(msg)

    source_selection = cbt_utils.RawVideoSelection(
        project=project,
        instance=bigtable_instance,
        table=bigtable_source_table_name,
        prefix=prefix)

    target_selection = cbt_utils.TFExampleSelection(
        project=project,
        instance=bigtable_instance,
        table=bigtable_target_table_name,
        prefix=prefix)

    generator = problem.sampling_generator(source_selection)

    target_selection.random_load_from_generator(
        generator=generator, max_num_examples=max_num_examples, log_every=100)

    tf.logging.info("Completed datagen.")


def log_flags(flags):
    for key in flags:
        tf.logging.info("%s: %s" % (key, getattr(flags, key)))


def cbt_generate_and_load_examples_experimental(project,
                                                problem_name,
                                                bigtable_instance,
                                                prefix,
                                                max_num_examples=10,
                                                audio_shard_size=1000):

    problem = registry.problem(problem_name)

    if not hasattr(problem, "sampling_generator"):
        msg = ("Problem {} is not compatible, must "
               "implement an {} method.").format(problem_name, attr)
        raise ValueError(msg)

    # For now the raw and example video shapes are the same. But if the raw
    # video shape were different we could e.g. add a .raw_video_shape attr.
    video_shape = problem.video_shape
    audio_shape = problem.audio_shape

    source_selection = cbt_utils.RawVideoSelection(project=project,
                                                   instance=bigtable_instance,
                                                   table=problem.raw_table_name,
                                                   prefix=prefix)

    tf.logging.debug("instantiated raw video table selection")

    generator = source_selection.sample_av_correspondence_examples(
        frames_per_video=video_shape[0],
        max_num_samples=int(math.floor(max_num_examples / 3)),
        keys_only=True)

    tf.logging.debug("instantiated generator")

    def _unpack(generator):
        for example_set in generator:
            for key, value in example_set.items():
                yield value.dump_keys()

    samples = json.dumps([data for data in _unpack(generator)])

    _cbt_datagen(project=project,
                 instance_name=bigtable_instance,
                 source_table_name=problem.raw_table_name,
                 target_table_name=problem.examples_table_name,
                 target_prefix=prefix,
                 samples=samples,
                 source_shape=video_shape[1:],
                 target_shape=video_shape[1:],
                 audio_shard_size=audio_shard_size)


def _cbt_datagen(project,
                 instance_name,
                 source_table_name,
                 target_table_name,
                 target_prefix,
                 source_shape,
                 target_shape,
                 samples,
                 target_key_size=4,
                 mode="compiled",
                 audio_shard_size=1000):

    pcml_root = get_pcml_root()

    main_go_path = os.path.join(pcml_root, "go", "main.go")
    main_compiled_path = os.path.join(pcml_root, "go", "main")

    gopath = "/usr/local/go/bin/go"

    go_command = gopath
    go_run_command = " ".join([gopath, "run"])

    if mode == "compiled":
        #command_prefix = " ".join([go_command, main_compiled_path])
        command_prefix = main_compiled_path
    else:
        command_prefix = " ".join([go_run_command, main_go_path])

    start = datetime.datetime.now()

    tempd = tempfile.mkdtemp()
    samples_path = os.path.join(tempd, "samples.json")
    with tf.gfile.Open(samples_path, "w") as f:
        f.write(samples)

    run_and_output([
        command_prefix, "--project", project, "--instance", instance_name,
        "--sourceTableName", source_table_name, "--targetTableName",
        target_table_name, "--targetPrefix", target_prefix, "--targetKeySize",
        str(target_key_size), "--sourceFrameX",
        str(source_shape[0]), "--sourceFrameY",
        str(source_shape[1]), "--sourceFrameC",
        str(source_shape[2]), "--targetFrameX",
        str(target_shape[0]), "--targetFrameY",
        str(target_shape[1]), "--targetFrameC",
        str(target_shape[2]), "--audioShardSize",
        str(audio_shard_size), "--samplesPath", samples_path
    ])

    end = datetime.datetime.now()

    print("go runtime: {}".format(str(end - start)))


def main(_):

    log_flags(FLAGS)

    if FLAGS.batch:

        # Run in batch
        job = CBTDatagenJob(
            problem_name=FLAGS.problem_name,
            project=FLAGS.project,
            bigtable_instance=FLAGS.bigtable_instance,
            bigtable_source_table_name=FLAGS.bigtable_source_table,
            bigtable_target_table_name=FLAGS.bigtable_target_table,
            max_num_examples=FLAGS.max_num_examples,
            staging_path=FLAGS.batch_staging_path,
            prefix=FLAGS.prefix,

            # HACK
            node_selector={"type": "datagen-small"})

        return job.launch_shard_parallel_jobs(num_shards=FLAGS.batch_num_shards)

    cbt_generate_and_load_examples_experimental(
        FLAGS.project,
        FLAGS.problem_name,
        FLAGS.bigtable_instance,
        FLAGS.prefix,
        max_num_examples=FLAGS.max_num_examples)

    # Run locally, e.g. inside VM running in batch.
    '''
  cbt_generate_and_load_examples(
    project=FLAGS.project,
    bigtable_instance=FLAGS.bigtable_instance,
    bigtable_source_table_name=FLAGS.bigtable_source_table,
    bigtable_target_table_name=FLAGS.bigtable_target_table,
    max_num_examples=FLAGS.max_num_examples,
    prefix=FLAGS.prefix,
    problem_name=FLAGS.problem_name)
  '''


if __name__ == "__main__":

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_integer('max_num_examples', 1000,
                         'Num examples to gen per job.')

    flags.DEFINE_boolean('batch', False, 'Whether to run in batch or locally.')

    flags.DEFINE_integer('batch_num_shards', 8,
                         'Num shards when running in batch.')

    flags.DEFINE_string('batch_staging_path', None,
                        'Path to which to stage when running in batch.')

    flags.DEFINE_string('prefix', None, 'One of train, eval, or test.')

    flags.DEFINE_string('project', None, 'A GCP project.')

    flags.DEFINE_string('bigtable_instance', None,
                        'A Google Cloud BigTable instance.')

    flags.DEFINE_string('bigtable_source_table', None,
                        'A Google Cloud BigTable source table of raw videos.')

    flags.DEFINE_string('bigtable_target_table', None,
                        'A Google Cloud BigTable target table for tfexamples.')

    flags.DEFINE_string('problem_name', None, 'A registered t2t problem name.')

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
