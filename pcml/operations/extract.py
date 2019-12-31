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
"""Load VoxCeleb2 videos into Cloud BigTable.

TODO: This slowly leaks memory at a rate of about ~10Mb/s or like
30Mb/video. Probably something like a subprocess not being closed when
finished.

TODO: I would like to do this a different way. Parameters only need be
defined once and there can be a general "run this in batch" operation
instead of the need to define it for each type of task.

This might be well-suited to be a cloud function that just reads work from
a pubsub topic.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import json
import datetime
import re

from scipy.signal import resample

import numpy as np

from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid
from pcml.launcher.kube import Resources

from pcml.utils import cbt_utils

from tensor2tensor.data_generators import generator_utils
from pcml.utils.audio_utils import mp4_to_1d_array

from pcml.utils.fs_utils import get_pcml_root
from pcml.launcher.util import _compress_and_stage

from pcml.utils.cbt_utils import random_key
from pcml.utils import video_utils


def subset_given_sharding(array, shard_id, num_shards):

    alen = len(array)
    elements_per_shard = math.floor(alen / num_shards)
    start_idx = (shard_id * elements_per_shard)
    end_idx = min(alen, (shard_id + 1) * elements_per_shard)
    return array[start_idx:end_idx]


class ExtractVideos(PCMLJob):

    def __init__(self,
                 manifest_path,
                 staging_path,
                 project,
                 instance,
                 table,
                 target_prefix,
                 log_verbosity="INFO",
                 tmp_dir="/mnt/ssd0",
                 job_name_prefix="extract",
                 num_cpu=1,
                 memory="6Gi",
                 image="gcr.io/clarify/basic-runtime:0.0.4",
                 *args,
                 **kwargs):
        """Extract videos from files to Cloud BigTable.

    Notes:
    * Because in this project the only consumers of these data
      expect row key prefixes of "train", "eval", or "test" this
      is enforced here.

    Args:
      manifest_path(str): A path to a manifest of video files to
        load into Cloud BigTable.
      target_prefix(str): The row key prefix to use when data is
        written to Cloud BigTable.

    """

        expected_prefixes = ["train", "eval", "test"]
        if target_prefix not in expected_prefixes:
            msg = "unexpected prefix {}, expected {}".format(
                target_prefix, expected_prefixes)
            raise ValueError(msg)

        self.job_name_prefix = "{}-{}".format(job_name_prefix, target_prefix)

        cmd = "python -m pcml.operations.extract "
        cmd += "--manifest_path={} ".format(manifest_path)
        cmd += "--tmp_dir={} ".format(tmp_dir)

        # CBT options
        cmd += "--project={} ".format(project)
        cmd += "--instance={} ".format(instance)
        cmd += "--table={} ".format(table)
        cmd += "--target_prefix={} ".format(target_prefix)

        command = ["/bin/sh", "-c"]
        command_args = [cmd]

        job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())

        super(ExtractVideos, self).__init__(job_name=job_name,
                                            command=command,
                                            command_args=command_args,
                                            namespace="kubeflow",
                                            image=image,
                                            staging_path=staging_path,
                                            resources=Resources(limits={
                                                "cpu": num_cpu,
                                                "memory": memory
                                            }),
                                            *args,
                                            **kwargs)


def video_file_to_cbt(remote_file_path,
                      selection,
                      tmp_dir,
                      shard_id,
                      num_shards,
                      video_id,
                      downsample_xy_dims=64,
                      greyscale=True,
                      resample_every=2,
                      audio_block_size=1000):
    """Extract from input path to target CBT selection."""

    tf.logging.info("Loading CBT table {}".format(selection.table_name))

    tf.logging.info("Processing file: {}".format(remote_file_path))

    filename = "-".join(remote_file_path.split("/")[-3:])

    local_file_path = generator_utils.maybe_download(tmp_dir, filename,
                                                     remote_file_path)

    audio_array = mp4_to_1d_array(local_file_path)

    # Re-sample every N steps (numpy slicing syntax)
    audio_array = audio_array[0::resample_every]

    audio_array = np.clip((audio_array + 0.5) * 255.0, a_min=0, a_max=255)

    # Read a frame iterable
    video = video_utils.Video()
    video.load_from_file(local_file_path,
                         downsample_size=(downsample_xy_dims,
                                          downsample_xy_dims),
                         greyscale=greyscale)

    selection.write_av(audio=audio_array,
                       frames=video,
                       shard_id=shard_id,
                       video_id=video_id,
                       audio_block_size=audio_block_size)


def _expect_type(obj, t):
    if not isinstance(obj, t):
        msg = "object {} expected to have type {}, saw {}".format(
            obj, t, type(obj))
        raise ValueError(msg)


def extract_to_cbt(manifest_path,
                   project,
                   instance,
                   table,
                   tmp_dir,
                   target_prefix,
                   shard_id=0,
                   num_shards=1,
                   downsample_xy_dims=64,
                   greyscale=True,
                   resample_every=2,
                   audio_block_size=1000):
    """Data-parallel extraction of input from file path manifest."""

    tf.logging.info("Processing manifest: %s" % manifest_path)

    tf.logging.info("Processing shard {shard_id} of {num_shards}".format(
        shard_id=shard_id, num_shards=num_shards))

    for obj in [shard_id, num_shards]:
        _expect_type(obj, int)

    #target_prefix = "{}_{}".format(target_prefix, str(shard_id))

    tf.logging.info("Writing to prefix {}.".format(target_prefix))

    file_paths = []

    with tf.gfile.Open(manifest_path) as f:

        for line in f:

            file_path = line.strip()
            file_paths.append(file_path)

    file_paths = subset_given_sharding(file_paths,
                                       shard_id=shard_id,
                                       num_shards=num_shards)

    selection = cbt_utils.RawVideoSelection(project=project,
                                            instance=instance,
                                            table=table,
                                            prefix=target_prefix)

    shard_meta = cbt_utils.VideoShardMeta(shard_id=shard_id,
                                          num_videos=0,
                                          status="started",
                                          num_shards=num_shards)
    #selection.set_shard_meta(shard_meta)

    ct = 0
    for video_id, remote_file_path in enumerate(file_paths):

        video_file_to_cbt(remote_file_path=remote_file_path,
                          selection=selection,
                          tmp_dir=tmp_dir,
                          shard_id=shard_id,
                          num_shards=num_shards,
                          video_id=video_id,
                          downsample_xy_dims=downsample_xy_dims,
                          greyscale=greyscale,
                          resample_every=resample_every,
                          audio_block_size=audio_block_size)
        ct += 1

    shard_meta.num_videos = ct
    shard_meta.status = "finished"

    selection.set_shard_meta(shard_meta)

    tf.logging.info("Batch extraction complete.")


def log_flags(flags):
    for key in flags:
        tf.logging.info("%s: %s" % (key, getattr(flags, key)))


def handle_log_verbosity(log_verbosity):
    if not log_verbosity:
        return
    if not hasattr(tf.logging, log_verbosity):
        msg = "Unrecognized log verbosity: {}".format(log_verbosity)
        raise ValueError(msg)
    tf.logging.set_verbosity(getattr(tf.logging, log_verbosity))


def main(_):

    log_flags(FLAGS)

    handle_log_verbosity(FLAGS.log_verbosity)

    if FLAGS.batch:

        #node_selector = FLAGS.node_selector
        #if isinstance(node_selector, str):
        #  node_selector = json.loads(node_selector)
        #  assert isinstance(node_selector, dict)

        # Run in batch
        job = ExtractVideos(
            manifest_path=FLAGS.manifest_path,
            staging_path=FLAGS.batch_staging_path,
            project=FLAGS.project,
            instance=FLAGS.instance,
            table=FLAGS.table,
            target_prefix=FLAGS.target_prefix,
            log_verbosity=FLAGS.log_verbosity,
            # HACK!
            node_selector={"type": "datagen-small"})

        return job.launch_shard_parallel_jobs(
            num_shards=FLAGS.batch_num_shards,
            max_num_jobs=FLAGS.batch_max_num_jobs)

    # Run locally, e.g. inside VM running in batch.
    extract_to_cbt(manifest_path=FLAGS.manifest_path,
                   shard_id=FLAGS.shard_id,
                   num_shards=FLAGS.num_shards,
                   project=FLAGS.project,
                   instance=FLAGS.instance,
                   table=FLAGS.table,
                   tmp_dir=FLAGS.tmp_dir,
                   target_prefix=FLAGS.target_prefix)


def flag_definitions(flags):

    flags.DEFINE_boolean('batch', False, 'Whether to run in batch or locally.')

    flags.DEFINE_integer('batch_num_shards', 8,
                         'Num shards when running in batch.')

    flags.DEFINE_string('batch_staging_path', None,
                        'Path to which to stage when running in batch.')

    flags.DEFINE_integer('batch_max_num_jobs', None,
                         'Max num jobs to launch in batch.')

    #flags.DEFINE_string('node_selector', None,
    #                    'Node pool tag to specify for batch jobs.')

    flags.DEFINE_string('manifest_path', None, 'Path to video file manifest.')

    flags.DEFINE_integer('shard_id', -1, 'The shard ID.')

    flags.DEFINE_integer('num_shards', 1, 'The number of shards.')

    flags.DEFINE_string('target_prefix', None,
                        'A row key prefix to use when writing data.')

    flags.DEFINE_string('project', None, 'A GCP project.')

    flags.DEFINE_string('instance', None, 'A Google Cloud BigTable instance.')

    flags.DEFINE_string('table', None, 'A Google Cloud BigTable instance.')

    flags.DEFINE_string('tmp_dir', None, 'Node directory to use for tmp files.')

    flags.DEFINE_string('log_verbosity', 'INFO', 'Log verbosity.')


if __name__ == "__main__":

    flags = tf.flags
    FLAGS = flags.FLAGS

    flag_definitions(flags)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
