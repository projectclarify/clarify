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

from collections import UserDict
import mock

from pcml.functions.extract.messages import ExtractTriggerMessage

import main


class TestExtractFn(tf.test.TestCase):

  def test_extract(self):

    mock_context = UserDict()
    mock_context = mock.Mock()
    mock_context.event_id = '617187464135194'
    mock_context.timestamp = '2019-07-15T22:09:03.761Z'

    mp4_path = "gs://clarify-data/requires-eula/voxceleb2/dev/mp4/id00012/21Uxsk56VDQ/00001.mp4"

    test_message = ExtractTriggerMessage(**{
      "problem_name": problem_name,
      "project": project,
      "bigtable_instance": cbt_instance,
      "target_table_name": target_table_name,
      "mp4_path": mp4_path,
      "video_id": 0
    }).__dict__

    event = {
      "data": base64.b64encode(
        json.dumps(test_message).encode('utf-8'))}

    main.extract(event, mock_context)


if __name__ == "__main__":
  tf.test.main()
