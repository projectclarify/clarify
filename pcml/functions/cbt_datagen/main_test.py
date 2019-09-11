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

import uuid
import json
import base64

import tensorflow as tf

from collections import UserDict
import mock

import main

from tensor2tensor.utils import registry

from pcml.functions.cbt_datagen.messages import CBTDatagenTriggerMessage

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestCBTDatagenFn(tf.test.TestCase):

  def test_cbt_datagen(self):

    project = TEST_CONFIG.get("project")
    cbt_instance = TEST_CONFIG.get("test_cbt_instance")
    num_examples = 100
    prefix = "train"

    problem_name = "vox_celeb_single_frame"
    problem = registry.problem(problem_name)
    source_table_name = problem.raw_table_name

    salt = str(uuid.uuid4())
    target_table_name = "dgen-fn-test" + str(uuid.uuid4())

    mock_context = UserDict()
    mock_context = mock.Mock()
    mock_context.event_id = '617187464135194'
    mock_context.timestamp = '2019-07-15T22:09:03.761Z'

    test_message = CBTDatagenTriggerMessage(**{
      "problem_name": "vox_celeb_single_frame",
      "project": project,
      "bigtable_instance": cbt_instance,
      "source_table_name": source_table_name,
      "target_table_name": target_table_name,
      "prefix": prefix
    }).__dict__

    event = {"data": base64.b64encode(json.dumps(test_message).encode('utf-8'))}

    main.cbt_datagen(event, mock_context)


if __name__ == "__main__":
  tf.test.main()
