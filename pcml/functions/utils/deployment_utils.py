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

from pcml.utils.cmd_utils import run_and_output
import tensorflow as tf
from google.cloud import pubsub_v1


def _lookup_firestore_event_type(type_shorthand):
  type_lookup = {
    "create": "providers/cloud.firestore/eventTypes/document.create",
    "write": "providers/cloud.firestore/eventTypes/document.write",
    "delete": "providers/cloud.firestore/eventTypes/document.delete",
    "update": "providers/cloud.firestore/eventTypes/document.update"
  }
  if type_shorthand not in type_lookup:
    msg = "Unrecognized event type {}, expected {}".format(
      type_shorthand, type_lookup.keys()
    )
    raise ValueError(msg)
  return type_lookup[type_shorthand]


def _validate_runtime(runtime):
  allowed_runtimes = ["python37"]
  if runtime not in allowed_runtimes:
    raise ValueError("Runtime {} must be one of {}".format(
      runtime, allowed_runtimes
    ))


def deploy_firestore_responder(function_name,
                               event_type,
                               project_id,
                               collection,
                               document_path,
                               service_account=None,
                               source=None,
                               runtime="python37",
                               region="us-central1"):
  """Convenience wrapper for deployment of firestore responder fn.
  
  Notes:
  * Service account defaults to
    {project name}@appspot.gserviceaccount.com
  
  """

  _validate_runtime(runtime)

  event_type_longhand = _lookup_firestore_event_type(event_type)

  triggering_resource = "projects/{}/databases/default/".format(
        project_id)

  triggering_resource += "documents/{}/{}".format(
    collection, document_path
  )

  msg = "Function {} will trigger on {} ".format(
    function_name, event_type_longhand
  )

  msg += "in response to triggering resource {}.".format(
    triggering_resource
  )

  tf.logging.info(msg)

  cmd = [
    "gcloud", "functions", "deploy", function_name,
    "--trigger-event", event_type_longhand,
    "--trigger-resource", triggering_resource,
    "--runtime", runtime,
    "--source", source
  ]

  if service_account:
    cmd.extend(["--service-account", service_account])

  if region:
    cmd.extend(["--region", region])

  return run_and_output(cmd)


def _create_topic(project_id, topic_name):
  msg = "Creating topic {} in project {}".format(topic_name,
                                                 project_id)
  tf.logging.info(msg)
  publisher = pubsub_v1.PublisherClient()
  topic_path = publisher.topic_path(project_id, topic_name)
  topic = publisher.create_topic(topic_path)


def deploy_topic_responder(function_name,
                           trigger_topic,
                           project_id,
                           service_account=None,
                           source=None,
                           runtime="python37",
                           region="us-central1",
                           create_topic=True,
                           create_done_topic=True):

  _validate_runtime(runtime)

  msg = "Function {} will be triggered by topic {} ".format(
    function_name, trigger_topic
  )

  tf.logging.info(msg)

  if create_topic:
    _create_topic(project_id, trigger_topic)

  if create_done_topic:
    _create_topic(project_id, trigger_topic + "-done")

  cmd = [
    "gcloud", "functions", "deploy", function_name,
    "--trigger-topic", trigger_topic,
    "--runtime", runtime,
    "--source", source
  ]

  if service_account:
    cmd.extend(["--service-account", service_account])

  if region:
    cmd.extend(["--region", region])

  return run_and_output(cmd)
