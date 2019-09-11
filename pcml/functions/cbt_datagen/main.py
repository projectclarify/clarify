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

"""PubSub-triggered cbt_datagen."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import base64
import json

from pcml.operations.cbt_datagen import cbt_generate_and_load_examples
from pcml.datasets import vox_celeb_cbt

# Import from messages because afaict other files have trouble
# importing from a file named main.py and would like to share this
# message with both producers and consumers (including tests).
from pcml.functions.cbt_datagen.messages import CBTDatagenTriggerMessage

FUNCTION_NAME = "cbt_datagen"

NUM_EXAMPLES_PER_TRIGER = 1000


def cbt_datagen(event, context):

  if 'data' not in event:
    raise ValueError("Received event trigger without PubSub message data.")

  msg_data_raw = json.loads(base64.b64decode(event['data']).decode('utf-8'))
  msg_data = CBTDatagenTriggerMessage(**msg_data_raw)

  print("Received datagen request: {}".format(msg_data.__dict__))

  cbt_generate_and_load_examples(
    project=msg_data.project,
    bigtable_instance=msg_data.bigtable_instance,
    bigtable_source_table_name=msg_data.source_table_name,
    bigtable_target_table_name=msg_data.target_table_name,
    prefix=msg_data.prefix,
    problem_name=msg_data.problem_name,
    max_num_examples=NUM_EXAMPLES_PER_TRIGER)
  
  print("Finished function.")
