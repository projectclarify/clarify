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
from pcml.utils.gcb_utils import generate_image_tag
from pcml.utils.gcb_utils import gcb_build_and_push
from pcml.utils.fs_utils import get_pcml_root
from pcml.functions.utils import deployment_utils
from pcml.utils.fs_utils import TemporaryDirectory

import datetime
import os
import yaml
import uuid
import re
import dateutil

import tensorflow as tf

from google.cloud import pubsub_v1
    
from google.oauth2 import service_account
import googleapiclient.discovery
from googleapiclient import _auth

credentials = _auth.default_credentials()

service = googleapiclient.discovery.build(
    'iam', 'v1', credentials=credentials)

crm_service = googleapiclient.discovery.build(
    'cloudresourcemanager', 'v1', credentials=credentials)

cloudbuild = googleapiclient.discovery.build(
    'cloudbuild', 'v1', credentials=credentials)


def latest_successful_build(image_uri, project_id):
  """Given an image URI get the most recent green cloudbuild."""

  builds = cloudbuild.projects().builds().list(
      projectId=project_id).execute()

  uri_prefix = image_uri.split(":")[0]

  latest_time = None
  latest = None

  for build in builds["builds"]:
    if build["status"] == "SUCCESS":
      images = build["images"]
      if len(images) == 1:
        if images[0].startswith(uri_prefix):
          finish_time = dateutil.parser.parse(build["finishTime"])
          if not latest:
            latest_time = finish_time
          if finish_time >= latest_time:
            latest = images[0]
            latest_time = finish_time

  if latest:
    tf.logging.info("Found a latest successful build: {}".format(
      latest
    ))

  return latest


def _build_cloud_run_image(
    function_name, project_id,
    default_cache_from="tensorflow/tensorflow:1.14.0"):

  function_code_path = "functions/{}".format(function_name)

  pcml_lib_root = os.path.join(get_pcml_root(), "pcml")

  with TemporaryDirectory() as tmpdir:

    tmp_lib_path = deployment_utils.prepare_functions_bundle(
        function_code_path=function_code_path,
        tmpdir=tmpdir,
        pcml_lib_root=pcml_lib_root)

    image_uri = generate_image_tag(project_id, function_name)

    cache_from = latest_successful_build(image_uri=image_uri,
                                         project_id=project_id)
    if not cache_from:
      cache_from = default_cache_from

    gcb_build_and_push(image_tag=image_uri,
                       build_dir=tmp_lib_path,
                       cache_from=cache_from)

  return image_uri


def update_service(function_name, image_uri, region, memory="2Gi",
                   concurrency=40, timeout="10m"):
  """Update a cloud run service given a container image."""

  run_and_output(["gcloud", "beta", "run", "deploy",
                  "--platform", "managed", 
                  "--region", region,
                  function_name,
                  "--image", image_uri,
                  "--memory", memory,
                  "--timeout", timeout,
                  "--concurrency", str(concurrency)])


def get_domain_for_cloudrun_service(service_name, region):

  out = run_and_output([
      "gcloud", "beta", "run", "services", "describe",
      service_name, "--platform", "managed",
      "--region", region])

  domain = None
  for line in out.split("\n"):
    if "domain" in line:
      domain = line.split("domain:")[1].split(" ")[1]

  return domain


def list_service_accounts(project_id):
    """Lists all service accounts for the current project."""

    service_accounts = service.projects().serviceAccounts().list(
        name='projects/' + project_id).execute()

    return service_accounts


def service_account_exists(service_account_email, project):

  service_accounts = list_service_accounts(project)

  for account in service_accounts["accounts"]:
    if service_account_email == account["email"]:
      return account

  return None


def maybe_create_service_account(service_account_name, project):

  service_account_email = "{}@{}.iam.gserviceaccount.com".format(
    service_account_name, project
  )

  if not service_account_exists(
      service_account_email=service_account_email,
      project=project):

    service_account = service.projects().serviceAccounts().create(
        name='projects/' + project,
        body={
            'accountId': service_account_name,
            'serviceAccount': {
                'displayName': service_account_name
            }
        }).execute()

    tf.logging.info('Created service account: ' + service_account['email'])

  return service_account_email


def configure_invoker_sa(service_name, project, region):

  service_account_name = "{}-invoker".format(service_name)

  service_account_email = maybe_create_service_account(
      service_account_name, project)

  member_arg = "--member=serviceAccount:{}".format(
    service_account_email
  )

  role_arg = "--role=roles/run.invoker"

  run_and_output(["gcloud", "beta", "run", "services",
                  "add-iam-policy-binding",
                  "--platform", "managed", 
                  "--region", region,
                  service_name,
                  member_arg, role_arg])

  return service_account_email


def get_project_number(project):

  project_number = None

  project_data = crm_service.projects().get(projectId=project).execute()

  if project_data:
    project_number = project_data["projectNumber"]

  return project_number


def maybe_add_pubsub_token_creator_policy(project_id):

  project_number = get_project_number(project_id)

  pubsub_sa = "service-{}".format(project_number)
  pubsub_sa += "@gcp-sa-pubsub.iam.gserviceaccount.com"

  member_arg = "--member=serviceAccount:{}".format(pubsub_sa)

  role_arg = "--role=roles/iam.serviceAccountTokenCreator"

  run_and_output(["gcloud", "projects", "add-iam-policy-binding",
                  project_number, member_arg, role_arg])

    
def list_subscriptions_in_project(project_id):
  """Lists all subscriptions in the current project."""

  subscriber = pubsub_v1.SubscriberClient()
  project_path = subscriber.project_path(project_id)

  subscriptions = subscriber.list_subscriptions(project_path)

  return [subscription.name for subscription in subscriptions]


def maybe_create_topic(project, topic_name):
  msg = "Creating topic {} in project {}".format(topic_name,
                                                 project)
  tf.logging.info(msg)
  publisher = pubsub_v1.PublisherClient()
  topic_path = publisher.topic_path(project, topic_name)

  project_path = publisher.project_path(project)

  topic_paths = [topic.name for topic in publisher.list_topics(project_path)]

  if topic_path not in topic_paths:
    topic = publisher.create_topic(topic_path)


def maybe_create_subscription_for_service(service_name,
                                          service_account_email,
                                          service_url,
                                          project,
                                          region,
                                          topic_name):

  subscriptions = list_subscriptions_in_project(project)

  # Create a unique subscription ID
  subscription_name = "{}-{}".format(
    service_name, topic_name
  )

  subscriber = pubsub_v1.SubscriberClient()
  subscription_path = subscriber.subscription_path(
    project, subscription_name)

  publisher = pubsub_v1.PublisherClient()
  topic_path = publisher.topic_path(project, topic_name)

  if subscription_path not in subscriptions:

    run_and_output(["gcloud", "beta", "pubsub", "subscriptions",
                    "create", subscription_path,
                    "--topic", topic_path,
                    "--push-endpoint={}".format(service_url),
                    "--push-auth-service-account={}".format(
                      service_account_email
                    )])


def deploy_cloud_run_topic_responder(project, region, function_name,
                                     memory="2Gi", concurrency=80,
                                     timeout="14m"):

  image_uri = _build_cloud_run_image(function_name, project)

  function_name = re.sub("_", "-", function_name)

  topic_name = "{}-trigger".format(function_name)

  update_service(function_name, image_uri, region=region,
                 memory=memory, concurrency=concurrency, timeout=timeout)

  domain = get_domain_for_cloudrun_service(
    service_name=function_name, region=region)

  service_account_email = configure_invoker_sa(
    service_name=function_name,
    project=project,
    region=region)

  maybe_add_pubsub_token_creator_policy(project)

  maybe_create_topic(project=project, topic_name=topic_name)

  maybe_create_subscription_for_service(
    service_name=function_name,
    service_account_email=service_account_email,
    service_url=domain,
    project=project,
    region=region,
    topic_name=topic_name
  )
