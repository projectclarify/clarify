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

"""Messages related to cloud-function-triggered cbt datagen.

E.g., a message might look like the following:

{
  "problem_name": "vox_celeb_single_frame",
  "project": "clarify",
  "bigtable_instance": "clarify",
  "source_table_name": "vox-celeb-2-raw",
  "target_table_name": "vox-celeb-single-frame-ex-dev",
  "prefix": "train",
}

which should be serialized, i.e.
{"problem_name": "vox_celeb_single_frame", "project": "clarify", "bigtable_instance": "clarify", "source_table_name": "vox-celeb-2-raw", "target_table_name": "vox-celeb-single-frame-ex-dev", "prefix": "train"}

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

from google.cloud import pubsub_v1

from tensor2tensor.utils import registry

from pcml.functions.cbt_datagen.deploy import TRIGGER_TOPIC
from pcml.functions.cbt_datagen.messages import CBTDatagenTriggerMessage


def trigger_datagen(problem_name, project, bigtable_instance, prefix,
                    num_invocations=1):

  problem = registry.problem(problem_name)

  trigger_message = CBTDatagenTriggerMessage(
    problem_name=problem_name,
    project=project,
    bigtable_instance=bigtable_instance,
    source_table_name=problem.raw_table_name,
    target_table_name=problem.examples_table_name,
    prefix=prefix)

  publisher_client = pubsub_v1.PublisherClient()

  topic_path = publisher_client.topic_path(
      project, TRIGGER_TOPIC)

  data = json.dumps(trigger_message.__dict__).encode('utf-8')

  for _ in range(num_invocations):
    publisher_client.publish(topic_path, data=data).result()