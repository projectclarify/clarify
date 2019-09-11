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
import subprocess
import datetime
import json
import time

from google.cloud import pubsub_v1

import tensorflow as tf

from tensor2tensor.utils import registry

from pcml.functions.extract.deploy import TRIGGER_TOPIC
from pcml.functions.extract.deploy import _deploy

from pcml.functions.extract.messages import ExtractTriggerMessage
from pcml.functions.utils.test_utils import e2e_test_function

from pcml.utils.cmd_utils import run_and_output
from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestExtractFnE2E(tf.test.TestCase):

  def test_extract_e2e(self):

    function_name = "extract"
    project = TEST_CONFIG.get("project")
    sa = TEST_CONFIG.get("functions_testing_service_account")
    region = TEST_CONFIG.get("region")
    staging = TEST_CONFIG.get("test_artifacts_root")
    cbt_instance = TEST_CONFIG.get("test_cbt_instance")
    test_video_path = TEST_CONFIG.get("test_video_path")

    salt = str(uuid.uuid4())
    target_table_name = "ext-fn-test" + str(uuid.uuid4())

    test_message = ExtractTriggerMessage(**{
      "project": project,
      "bigtable_instance": cbt_instance,
      "target_table_name": target_table_name,
      "mp4_path": test_video_path,
      "video_id": 0
    })

    e2e_test_function(
      function_name=function_name,
      trigger_message=test_message,
      trigger_topic=TRIGGER_TOPIC,
      project=project,
      service_account=sa,
      region=region,
      staging=staging,
      deploy_fn=_deploy,
      expect_string="Finished function.")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()
