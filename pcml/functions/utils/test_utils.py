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

import datetime

from google.cloud import pubsub_v1

import json
import time

from pcml.utils.cmd_utils import run_and_output

import tensorflow as tf


def e2e_test_function(function_name,
                      trigger_message,
                      trigger_topic,
                      project,
                      service_account,
                      region,
                      staging,
                      deploy_fn,
                      expect_string,
                      wait_seconds=60):

  deploy_fn(project_id=project,
            service_account=service_account,
            region=region,
            staging_root=staging)

  start_time = datetime.datetime.now().isoformat()
  publisher_client = pubsub_v1.PublisherClient()
  topic_path = publisher_client.topic_path(project, trigger_topic)

  data = json.dumps(trigger_message.__dict__).encode('utf-8')
  publisher_client.publish(topic_path, data=data).result()

  # Wait for function execution to complete including coldstart
  time.sleep(wait_seconds)

  # Check logs after a delay
  logs = run_and_output([
      'gcloud', 'alpha', 'functions', 'logs', 'read', function_name,
      '--start-time', start_time, '--project', project
  ])

  tf.logging.info("fn logs: {}".format(logs))

  assert expect_string in logs
