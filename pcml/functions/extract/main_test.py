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

import tensorflow as tf

import json
import base64
import uuid

from collections import UserDict
import mock

from pcml.functions.extract.messages import ExtractTriggerMessage

import main

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestExtractFn(tf.test.TestCase):

  def test_extract(self):

    function_name = "extract"
    project = TEST_CONFIG.get("project")
    sa = TEST_CONFIG.get("functions_testing_service_account")
    region = TEST_CONFIG.get("region")
    staging = TEST_CONFIG.get("test_artifacts_root")
    cbt_instance = TEST_CONFIG.get("test_cbt_instance")
    test_video_path = TEST_CONFIG.get("test_video_path")

    salt = str(uuid.uuid4())
    target_table_name = "ext-fn-test" + str(uuid.uuid4())

    mock_context = UserDict()
    mock_context = mock.Mock()
    mock_context.event_id = '617187464135194'
    mock_context.timestamp = '2019-07-15T22:09:03.761Z'

    mp4_path = "gs://clarify-data/requires-eula/voxceleb2/dev/mp4/id00012/21Uxsk56VDQ/00001.mp4"

    test_message = ExtractTriggerMessage(**{
      "project": project,
      "bigtable_instance": cbt_instance,
      "target_table_name": target_table_name,
      "mp4_paths": [test_video_path, test_video_path],
      "prefix": "train",
      "video_ids": [0,1],
      "downsample_xy_dims": 96,
      "greyscale": True,
      "resample_every": 2,
      "audio_block_size": 1000
    }).__dict__

    event = {
      "data": base64.b64encode(
        json.dumps(test_message).encode('utf-8'))}

    main.extract(event, mock_context)


if __name__ == "__main__":
  tf.test.main()
