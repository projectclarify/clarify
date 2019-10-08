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

"""Cloud Function for triggering extraction of videos to Cloud BigTable.

Note: Currently requires the extract message producer to assign a unique ID to
each video, otherwise extraction operations will overlap writes.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import json
import tempfile

import tensorflow as tf

from pcml.utils import cbt_utils
from pcml.operations.extract import video_file_to_cbt

from pcml.functions.extract.messages import ExtractTriggerMessage


def extract(event, context):

  if 'data' not in event:
    raise ValueError("Received event trigger without PubSub message data.")

  msg_data_raw = json.loads(base64.b64decode(event['data']).decode('utf-8'))
  msg_data = ExtractTriggerMessage(**msg_data_raw)

  tf.logging.info("Received request: {}".format(msg_data.__dict__))

  tmpdir = tempfile.mkdtemp()

  selection = cbt_utils.RawVideoSelection(
    project=msg_data.project,
    instance=msg_data.bigtable_instance,
    table=msg_data.target_table_name,
    prefix=msg_data.prefix)

  for i, mp4_path in enumerate(msg_data.mp4_paths):

    video_id = msg_data.video_ids[i]

    video_file_to_cbt(
      remote_file_path=mp4_path,
      selection=selection,
      tmp_dir=tmpdir,
      shard_id=0,
      num_shards=1,
      downsample_xy_dims=msg_data.downsample_xy_dims,
      greyscale=msg_data.greyscale,
      video_id=video_id)

  tf.logging.info("Finished function.")
