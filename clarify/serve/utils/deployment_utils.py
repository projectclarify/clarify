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

import os
import datetime

import tensorflow as tf
from google.cloud import pubsub_v1

from clarify.utils.fs_utils import get_pcml_root
from clarify.utils.fs_utils import TemporaryDirectory
from clarify.utils.cmd_utils import run_and_output


def _lookup_firestore_event_type(type_shorthand):
  type_lookup = {
      "create": "providers/cloud.firestore/eventTypes/document.create",
      "write": "providers/cloud.firestore/eventTypes/document.write",
      "delete": "providers/cloud.firestore/eventTypes/document.delete",
      "update": "providers/cloud.firestore/eventTypes/document.update"
  }
  if type_shorthand not in type_lookup:
    msg = "Unrecognized event type {}, expected {}".format(
        type_shorthand, type_lookup.keys())
    raise ValueError(msg)
  return type_lookup[type_shorthand]


def _validate_runtime(runtime):
  allowed_runtimes = ["python37"]
  if runtime not in allowed_runtimes:
    raise ValueError("Runtime {} must be one of {}".format(
        runtime, allowed_runtimes))


def deploy_firestore_responder(function_name,
                               event_type,
                               project_id,
                               collection,
                               document_path,
                               service_account=None,
                               source=None,
                               runtime="python37",
                               region="us-central1",
                               memory="256MB",
                               timeout="60s"):
  """Convenience wrapper for deployment of firestore responder fn.
  
  Notes:
  * Service account defaults to
    {project name}@appspot.gserviceaccount.com
  
  """

  _validate_runtime(runtime)

  event_type_longhand = _lookup_firestore_event_type(event_type)

  triggering_resource = "projects/{}/databases/(default)/".format(project_id)

  triggering_resource += "documents/{}/{}".format(collection, document_path)

  msg = "Function {} will trigger on {} ".format(function_name,
                                                 event_type_longhand)

  msg += "in response to triggering resource {}.".format(triggering_resource)

  tf.logging.info(msg)

  cmd = [
      "gcloud", "functions", "deploy", function_name, "--trigger-event",
      event_type_longhand, "--trigger-resource", triggering_resource,
      "--runtime", runtime, "--source", source, "--memory", memory, "--timeout",
      timeout
  ]

  if service_account:
    cmd.extend(["--service-account", service_account])

  if region:
    cmd.extend(["--region", region])

  return run_and_output(cmd)


def _create_topic(project_id, topic_name):
  msg = "Creating topic {} in project {}".format(topic_name, project_id)
  tf.logging.info(msg)
  publisher = pubsub_v1.PublisherClient()
  topic_path = publisher.topic_path(project_id, topic_name)
  print(topic_path)

  project_path = publisher.project_path(project_id)

  topic_paths = [topic.name for topic in publisher.list_topics(project_path)]

  if topic_path not in topic_paths:
    topic = publisher.create_topic(topic_path)


def deploy_topic_responder(function_name,
                           trigger_topic,
                           project_id,
                           service_account=None,
                           source=None,
                           runtime="python37",
                           region="us-central1",
                           create_topic=True,
                           create_done_topic=True,
                           memory="256MB",
                           timeout="60s",
                           max_instances=1000):

  _validate_runtime(runtime)

  msg = "Function {} will be triggered by topic {} ".format(
      function_name, trigger_topic)

  tf.logging.info(msg)

  if create_topic:
    _create_topic(project_id, trigger_topic)

  if create_done_topic:
    _create_topic(project_id, trigger_topic + "-done")

  cmd = [
      "gcloud", "functions", "deploy", function_name, "--trigger-topic",
      trigger_topic, "--runtime", runtime, "--source", source, "--memory",
      memory, "--timeout", timeout, "--max-instances",
      str(max_instances)
  ]

  if service_account:
    cmd.extend(["--service-account", service_account])

  if region:
    cmd.extend(["--region", region])

  return run_and_output(cmd)


def _touch(path):
  with open(path, "w") as f:
    f.write("")


def _timestamp():
  now = datetime.datetime.now()
  epoch = datetime.datetime.utcfromtimestamp(0)
  ts = int((now - epoch).total_seconds() * 100000.0)
  return ts


def prepare_functions_bundle(function_code_path, tmpdir, pcml_lib_root):

  pcml_lib_root = os.path.join(get_pcml_root(), "pcml")

  # Recursive copy pcml_root/pcml into tmpdir
  tf.gfile.MakeDirs(os.path.join(tmpdir, "lib"))
  tmp_pcml_path = os.path.join(tmpdir, "lib", "pcml")

  run_and_output(["cp", "-r", pcml_lib_root, tmp_pcml_path])

  run_and_output(["ls", tmp_pcml_path])

  # Replace __init__.py with an empty one
  tmp_init = os.path.join(tmp_pcml_path, "__init__.py")
  with tf.gfile.Open(tmp_init, "w") as f:
    f.write("")

  # Copy in function code
  source_function_code_path = os.path.join(pcml_lib_root, function_code_path)
  tmp_lib_path = os.path.join(tmpdir, "lib")

  def _allow_filename(filename):
    if filename.endswith(".py") or filename.endswith(".txt"):
      return True
    if filename == "Dockerfile":
      return True
    return False

  for filename in tf.gfile.ListDirectory(source_function_code_path):
    if _allow_filename(filename):
      source_path = os.path.join(source_function_code_path, filename)
      target_path = os.path.join(tmp_lib_path, filename)
      tf.gfile.Copy(source_path, target_path)

  return tmp_lib_path


def stage_functions_bundle(gcs_staging_path, function_code_path):

  pcml_lib_root = os.path.join(get_pcml_root(), "pcml")

  with TemporaryDirectory() as tmpdir:

    tmp_lib_path = prepare_functions_bundle(function_code_path, tmpdir,
                                            pcml_lib_root)

    local_zip_filename = "bundle.zip"
    local_zip_path = os.path.join(tmpdir, local_zip_filename)

    remote_zip_path = os.path.join(
        gcs_staging_path, "{}-{}".format(_timestamp(), local_zip_filename))

    # Create zip
    os.chdir(tmp_lib_path)
    run_and_output(["zip", "-r", local_zip_path, "./"])

    tf.gfile.Copy(local_zip_path, remote_zip_path, overwrite=True)

  return remote_zip_path
