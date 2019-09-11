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

import base64
from flask import Flask, request
import os
import sys
import json

import tensorflow as tf

from pcml.functions.cbt_datagen.messages import CBTDatagenTriggerMessage
from pcml.operations.cbt_datagen import cbt_generate_and_load_examples
from pcml.datasets import vox_celeb_cbt

import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()

app = Flask(__name__)

NUM_EXAMPLES_PER_TRIGER = 1000


@app.route('/', methods=['POST'])
def index():

    envelope = request.get_json()

    if not envelope:
        msg = 'no Pub/Sub message received'
        tf.logging.error(msg)
        return 'Bad Request: {}'.format(msg), 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        tf.logging.error(msg)
        return 'Bad Request: {}'.format(msg), 400

    event = envelope['message']

    msg_data_raw = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    msg_data = CBTDatagenTriggerMessage(**msg_data_raw)

    tf.logging.info("Received datagen request: {}".format(msg_data.__dict__))

    cbt_generate_and_load_examples(
      project=msg_data.project,
      bigtable_instance=msg_data.bigtable_instance,
      bigtable_source_table_name=msg_data.source_table_name,
      bigtable_target_table_name=msg_data.target_table_name,
      prefix=msg_data.prefix,
      problem_name=msg_data.problem_name,
      max_num_examples=NUM_EXAMPLES_PER_TRIGER)

    tf.logging.info("Finished function.")

    # Flush the stdout to avoid log buffering.
    sys.stdout.flush()

    return ('', 204)


if __name__ == '__main__':
    PORT = int(os.getenv('PORT')) if os.getenv('PORT') else 8080

    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host='127.0.0.1', port=PORT, debug=True)
