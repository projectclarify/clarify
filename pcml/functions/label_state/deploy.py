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

import os
import datetime
import tempfile
import subprocess

import tensorflow as tf

from pcml.functions.utils import deployment_utils
from pcml.utils.fs_utils import get_pcml_root
from pcml.utils.fs_utils import TemporaryDirectory


def _touch(path):
    with open(path, "w") as f:
        f.write("")


def _timestamp():
    now = datetime.datetime.now()
    epoch = datetime.datetime.utcfromtimestamp(0)
    ts = int((now - epoch).total_seconds() * 100000.0)
    return ts


def stage_functions_bundle(gcs_staging_path):

    pcml_root = get_pcml_root()

    with TemporaryDirectory() as tmpdir:

        # Mock a pcml module tree to be able to use pcml model code
        # without changing import paths (from way used in training/dev)
        fake_pcml_path = os.path.join(tmpdir, "lib", "pcml")
        models_path = os.path.join(fake_pcml_path, "models")
        tf.gfile.MakeDirs(models_path)
        _touch(os.path.join(fake_pcml_path, "__init__.py"))
        _touch(os.path.join(models_path, "__init__.py"))

        models_path_src = os.path.join(pcml_root, "pcml", "models")

        # For now the plan is to avoid staging in the full pcml src tree and related
        # depedencies and use models that have no PCML dependencies and problem stubs
        # that emulate but don't depend on problems that were used in training.
        for model_filename in [
                "dev.py", "modality_correspondence.py",
                "modality_correspondence_utils.py"
        ]:

            model_file_path_src = os.path.join(models_path_src, model_filename)

            model_file_path_tgt = os.path.join(models_path, model_filename)

            tf.gfile.Copy(model_file_path_src, model_file_path_tgt)

        # Copy in requirements
        reqs_source_path = os.path.join(pcml_root, "requirements.txt")
        reqs_target_path = os.path.join(tmpdir, "lib", "requirements.txt")
        tf.gfile.Copy(reqs_source_path, reqs_target_path)

        # Copy in function code
        main_src = os.path.join(pcml_root, "pcml", "functions", "label_state",
                                "main.py")
        main_tgt = os.path.join(tmpdir, "lib", "main.py")
        tf.gfile.Copy(main_src, main_tgt)

        local_zip_filename = "bundle.zip"
        local_zip_path = os.path.join(tmpdir, local_zip_filename)
        remote_zip_path = os.path.join(
            gcs_staging_path, "{}-{}".format(_timestamp(), local_zip_filename))

        # Create zip
        os.chdir(os.path.join(tmpdir, "lib"))
        subprocess.check_output(["zip", "-r", local_zip_path, "./"])

        tf.gfile.Copy(local_zip_path, remote_zip_path, overwrite=True)

    return remote_zip_path


def _deploy(project_id,
            region,
            staging_root,
            service_account=None,
            collection="users",
            document_path="{uid}/modalities/av"):

    source_path = stage_embed_frames_bundle(staging_root)

    return deployment_utils.deploy_firestore_responder(
        function_name="label_state",
        event_type="write",
        project_id=project_id,
        collection=collection,
        document_path=document_path,
        service_account=service_account,
        source=source_path,
        runtime="python37",
        memory="2048MB",
        timeout="540s",
        region=region)


def main(_):

    _deploy(project_id=FLAGS.project,
            service_account=FLAGS.service_account,
            region=FLAGS.gcp_region,
            staging_root=FLAGS.staging_root)


if __name__ == "__main__":

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string("project", "clarify", "A project ID.")

    flags.DEFINE_string("service_account",
                        "clarify@appspot.gserviceaccount.com",
                        "A service acct to allow GCF to r/w GCP resources.")

    flags.DEFINE_string("staging_root", "gs://clarify-dev/tmp/fnstaging",
                        "GCS bucket to use for staging function bundles.")

    flags.DEFINE_string("gcp_region", "us-central1",
                        "A GCP region where the function will be deployed.")

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
