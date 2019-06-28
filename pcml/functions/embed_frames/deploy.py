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

import tensorflow as tf
import os

from pcml.functions.utils import deployment_utils
from pcml.utils.fs_utils import get_pcml_root

FUNCTION_NAME = "embed_frames"

SOURCE_PATH = os.path.join(get_pcml_root(),
                           "pcml", "functions",
                           FUNCTION_NAME)


def _deploy(project_id, region, service_account=None,
            collection="users",
            document_path="{uid}/modalities/video"):

  return deployment_utils.deploy_firestore_responder(
        function_name=FUNCTION_NAME,
        event_type="update",
        project_id=project,
        collection=collection,
        document_path=document_path,
        service_account=service_account,
        source=SOURCE_PATH,
        runtime="python37",
        region=region)


def main(_):

  _deploy(project_id=FLAGS.project,
          service_account=FLAGS.service_account,
          region=FLAGS.gcp_region)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string("project", "clarify",
                      "A project ID.")

  flags.DEFINE_string("service_account", None,
                      "A service acct to allow GCF to r/w GCP resources.")

  flags.DEFINE_string("gcp_region", "us-central1",
                      "A GCP region where the function will be deployed.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()