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

import uuid
import subprocess
import datetime
import json
import time

from google.cloud import pubsub_v1

import tensorflow as tf

from tensor2tensor.utils import registry

from pcml.functions.cbt_datagen.deploy import TRIGGER_TOPIC
from pcml.functions.cbt_datagen.deploy import _deploy

from pcml.functions.cbt_datagen.messages import CBTDatagenTriggerMessage
from pcml.functions.utils.test_utils import e2e_test_function

from pcml.utils.cmd_utils import run_and_output
from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestCBTDatagenFnE2E(tf.test.TestCase):

  def test_cbt_datagen_remote(self):

    function_name = "cbt_datagen"
    project = TEST_CONFIG.get("project")
    sa = TEST_CONFIG.get("functions_testing_service_account")
    region = TEST_CONFIG.get("region")
    staging = TEST_CONFIG.get("test_artifacts_root")
    cbt_instance = TEST_CONFIG.get("test_cbt_instance")
    num_examples = 100

    problem_name = "vox_celeb_single_frame"
    problem = registry.problem(problem_name)
    source_table_name = problem.raw_table_name

    salt = str(uuid.uuid4())
    target_table_name = "dgen-fn-test" + str(uuid.uuid4())

    prefix = "train"

    test_message = CBTDatagenTriggerMessage(**{
      "problem_name": problem_name,
      "project": project,
      "bigtable_instance": cbt_instance,
      "source_table_name": source_table_name,
      "target_table_name": target_table_name,
      "prefix": prefix
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
      expect_string="Generated 100 examples...")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()