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

"""Trigger extraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json

import tensorflow as tf

from google.cloud import pubsub_v1

from tensor2tensor.utils import registry

from pcml.utils import cbt_utils

from pcml.functions.extract.messages import ExtractTriggerMessage

from pcml.functions.extract.deploy import TRIGGER_TOPIC


def trigger_extraction(problem_name,
                       project,
                       bigtable_instance,
                       prefix,
                       greyscale=False,
                       downsample_xy_dims=96,
                       resample_every=2,
                       audio_block_size=1000,
                       override_target_table_name=None,
                       override_max_files_processed=None):

  # Lookup the registered table name for raw data assoc. with this problem
  problem = registry.problem(problem_name)
  target_table_name = problem.raw_table_name
  if override_target_table_name:
    target_table_name = override_target_table_name

  mode_to_manifest = problem.mode_to_manifest_lookup()
  manifest_path = mode_to_manifest[prefix]

  file_paths = []

  with tf.gfile.Open(manifest_path) as f:
    for line in f:
      file_path = line.strip()
      file_paths.append(file_path)

  # Create the target table
  selection = cbt_utils.RawVideoSelection(
    project=project,
    instance=bigtable_instance,
    table=target_table_name,
    prefix=prefix)

  shard_meta = cbt_utils.VideoShardMeta(shard_id=0,
                                        num_videos=0,
                                        status="started",
                                        num_shards=1)

  trigger_message = ExtractTriggerMessage(
    project=project,
    bigtable_instance=bigtable_instance,
    target_table_name=target_table_name,
    prefix=prefix,
    mp4_paths=[],
    video_ids=[],
    downsample_xy_dims=downsample_xy_dims,
    greyscale=greyscale,
    resample_every=resample_every,
    audio_block_size=audio_block_size)

  batch_settings = pubsub_v1.types.BatchSettings(
    max_bytes=1024,  # One kilobyte
    max_latency=1,   # One second
  )
  publisher_client = pubsub_v1.PublisherClient(batch_settings)

  topic_path = publisher_client.topic_path(
      project, TRIGGER_TOPIC)

  msg = "Publishing extraction triggers for {} files.".format(
    len(file_paths)
  )
  tf.logging.info(msg)

  futures = dict()

  def get_callback(f, data):
    def callback(f):
        try:
            res = f.result()
            futures.pop(data)
        except:  # noqa
            print('Please handle {} for {}.'.format(f.exception(), data))

    return callback

  ct = 0
  report_every = 10000

  qsize = 0
  qmax = 40

  def _publish(video_id, trigger_message):
    futures_key = str(video_id)
    futures.update({futures_key: None})
    data = json.dumps(trigger_message.__dict__).encode('utf-8')

    future = publisher_client.publish(topic_path, data=data)
    futures[futures_key] = future
    future.add_done_callback(get_callback(future, futures_key))

  for video_id, remote_file_path in enumerate(file_paths):

    if qsize >= qmax:
      _publish(video_id, trigger_message)
      trigger_message.video_ids = []
      trigger_message.mp4_paths = []
      qsize = 0

    trigger_message.video_ids.append(video_id)
    trigger_message.mp4_paths.append(remote_file_path)

    ct += 1
    qsize += 1

    if override_max_files_processed and ct >= override_max_files_processed:
      break

    if video_id % report_every == 0:
      msg = "Created {} extraction message futures...".format(
        video_id
      )
      tf.logging.info(msg)
      tf.logging.info("Futures outstanding: {}".format(len(futures.keys())))

  # Modulo qmax
  if len(trigger_message.video_ids) > 0: 
    _publish(ct, trigger_message)

  while futures:
    time.sleep(5)
    msg = "Created {} of {} messages...".format(
      ct - len(futures.keys()), ct)
    tf.logging.info(msg)

  shard_meta.num_videos = ct
  shard_meta.status = "finished"

  selection.set_shard_meta(shard_meta)
