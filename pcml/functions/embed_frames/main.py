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

import json
import tensorflow as tf
from google.cloud import firestore
import numpy as np

client = firestore.Client()


# This function is configured to query a specific served model as
# hard-coded here; in the future this could be configured by setting
# an environment variable to specify the same.
SERVED_MODEL_CONFIG = {
    
}


def _construct_query_from_frames(frames):
  query = None
  return query


def _query_served_model(query):
  return None


def _prediction_for_frames(frames):
  """Given frames constructs query, sends, and parses result."""
  
  # Construct the query
  query = _construct_query_from_frames(frames)

  # Make the query against the served model.
  response = _query_served_model(query)

  # Parse the response
  # TODO

  return None


def embed_frames(data, context):
  """Glue between video in Firestore and queries to TFServing.

  Notes:
  * Function is triggered by a change to a Firestore document.
  * When called, function validates changed data and constructs
    a serving query.
  * Query is made against tensorflow model deployed on Kubernetes
    via TFServing.
  * The model's prediction is received and written to the
    Firebase database.

  Args:
    data (dict): The event payload.
    context (google.cloud.functions.Context): Metadata for the event.

  """

  path_parts = context.resource.split('/documents/')[1].split('/')
  collection_path = path_parts[0]
  document_path = '/'.join(path_parts[1:])
  uid = path_parts[1]

  affected_doc = client.collection(collection_path).document(document_path)

  video_frames_raw = data["value"]["fields"]["original"]["stringValue"]

  embedding = _prediction_for_frames(video_frames_raw)

  # =================
  # HACK: Mock an embedding vector
  embedding = np.random.random((128,)).tolist()
  # =================

  print("Received healthy model response.")

  target_doc_path = "{}/state_predictions/raw_embedding".format(uid)
  target_doc = client.collection(collection_path).document(target_doc_path)
  # The result is a reference to
  # /documents/users/{uid}/state_predictions/raw_embedding

  target_doc.set({
    u'value': embedding
  })
