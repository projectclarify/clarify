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

from pcml.functions.utils.deployment_utils import deploy_topic_responder
from pcml.functions.utils.deployment_utils import stage_functions_bundle

TRIGGER_TOPIC="extract-videos-dev"
FUNCTION_NAME="extract"

def _deploy(project_id,
            service_account,
            staging_root,
            region="us-central1"):

  source_path = stage_functions_bundle(
      staging_root, function_code_path="functions/extract")

  deploy_topic_responder(
    function_name=FUNCTION_NAME,
    trigger_topic=TRIGGER_TOPIC,
    project_id=project_id,
    service_account=service_account,
    source=source_path,
    runtime="python37",
    region=region,
    create_topic=True,
    create_done_topic=True,
    memory="1024MB",
    timeout="540s")


def main(_):

  _deploy(project_id=FLAGS.project,
          service_account=FLAGS.service_account,
          region=FLAGS.gcp_region,
          staging_root=FLAGS.staging_root)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string("project", "clarify",
                      "A project ID.")

  flags.DEFINE_string("service_account", "clarify@appspot.gserviceaccount.com",
                      "A service acct to allow GCF to r/w GCP resources.")

  flags.DEFINE_string("staging_root", "gs://clarify-dev/tmp/fnstaging",
                      "GCS bucket to use for staging function bundles.")

  flags.DEFINE_string("gcp_region", "us-central1",
                      "A GCP region where the function will be deployed.")

  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()