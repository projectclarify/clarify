#!/usr/bin/env python
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

"""Kubernetes admin operations.

Primary goal is to furnish a unique service account for each new user, copied from a
central template, then to update their profile with this service account and create a
role in their namespace that gives this service account jobs.create permissions in that
namespace.

Work in progress.

"""

import time

from pprint import pprint
import subprocess
import sys
import json

from absl import app
from absl import flags
from absl import logging

import kubernetes.client
from kubernetes.client.rest import ApiException
from kubernetes import client, config, utils

config.load_kube_config()

configuration = kubernetes.client.Configuration()

core_api_instance = kubernetes.client.CoreV1Api(
    kubernetes.client.ApiClient(configuration)
)

rbac_api_instance = kubernetes.client.RbacAuthorizationV1Api(
  kubernetes.client.ApiClient(configuration)
)

FLAGS = flags.FLAGS

flags.DEFINE_string('template_service_account', None,
                    'ID of a service account to use as a template.')



def run_and_output(command, cwd=None, env=None):
  
  process = subprocess.Popen(
    command, cwd=cwd, env=env,
    stdout=subprocess.PIPE
  )

  output = []

  for line in process.stdout:
    line = line.decode("utf-8")
    sys.stdout.write(line)
    output.append(line)

  return output


def profile_names_and_sa():
  """Not clear how to do this via Python API so wrapping CLI"""

  profile_info = run_and_output(["kubectl", "get", "profiles", "--output=json"])
  data = []
  raw = json.loads("\n".join(profile_info))
  if "items" in raw:
    for item in raw["items"]:

      profile_name = item["metadata"]["name"]
      owner = item["spec"]["owner"]["name"]

      wisa = None
      for plugin in item["spec"]["plugins"]:
        if plugin["kind"] == "WorkloadIdentity":
          wisa = plugin["spec"]["gcpServiceAccount"]

      data.append({
        "profile_name": profile_name,
        "owner_name": owner,
        "service_account": wisa
      })

  return data


def get_namespaces():
  return core_api_instance.list_namespace()


def list_roles(namespace):
  return rbac_api_instance.list_namespaced_role(namespace)


def role_exists_in_namespace(role_name, namespace):
  namespace_roles = list_roles(namespace)
  for item in namespace_roles.items:
    if item.metadata.name == role_name:
      return True
  return False


def list_rolebindings(namespace):
  return rbac_api_instance.list_namespaced_role_binding(namespace)


def role_binding_exists_in_namespace(role_binding_name, namespace):
  for item in list_rolebindings(namespace).items:
    if item.metadata.name == role_binding_name:
      return True
  return False


def _jobs_create_role_obj(namespace, name="jobs-create"):
  meta = kubernetes.client.V1ObjectMeta(
    name=name,
    namespace=namespace
  )
  role_body = kubernetes.client.V1Role(
    metadata=meta,
    rules=[
      kubernetes.client.V1PolicyRule(
          resources=["jobs"],
          verbs=["create", "delete"]
      )
    ]
  )

  return role_body


def namespace_exists(namespace):
  namespaces = get_namespaces()
  for item in namespaces.items:
    if item.metadata.name == namespace:
      return True
  return False


def service_account_for_user(user_name):
  """
  
  TODO: Generate the unique service account name that should exist given
  a user name.
  
  """
  return None


def maybe_create_jobs_create_role(namespace):

  jobs_create_role_body = _jobs_create_role_obj(
    namespace=namespace
  )

  role_exists = role_exists_in_namespace(
    role_name=jobs_create_role_body.metadata.name,
    namespace=jobs_create_role_body.meatadata.namespace
  )

  if not role_exists:
    create_role_response = api_instance.create_namespaced_role(
      namespace=namespace,
      body=jobs_create_role_body
    )
  else:

    """
    
    TODO: If the role does exist, update/replace it instead as the role should
    be uniquely as to have been created by us and additional resources may need
    to be added later (to existing roles).
    
    """

    pass

  return role_name=jobs_create_role_body


def _jobs_create_rolebinding_request(role_request, service_account_name):

  # Name rolebinding with name of role + -rb
  rb_name = role_request.metadata.name + "-rb"

  meta = kubernetes.client.V1ObjectMeta(
    name=rb_name,
    namespace=role_request.metadata.namespace
  )

  role_ref = kubernetes.client.V1RoleRef(
    name=role_request.metadata.name
  )

  role_binding_request = kubernetes.client.V1RoleBinding(
    metadata=meta, role_ref=role_ref
    subjects=[
      kubernetes.client.V1Subject(
        name=service_account_name,
        namespace=role_request.metadata.namespace
      )
    ]
  )

  return role_binding_request


def update_profile_and_rbac_configurations(template_service_account):

  profile_data = profile_names_and_sa()

  for profile_item in profile_data:

    expected_namespace = profile_item["profile_name"]

    if not namespace_exists(expected_namespace):
      raise ValueError("Unexpected condition, TODO")

    intended_service_account = service_account_for_user(
      profile_item["owner"]
    )
    
    """

    TODO: If the service account does not exist, create it, copying from template
    (i.e. template_service_account).

    """

    if profile_iem["service_account"] != intended_service_account:

      """

      TODO: If the service account set on the profile is not the intended service
      account, update that setting to the intended one.

      """
      pass
    
    jobs_create_role_request = maybe_create_jobs_create_role(namespace) 

    role_binding_request = _jobs_create_rolebinding_request(
      role_request=jobs_create_role_request,
      service_account_name=intended_service_account_name
    )

    # Use a role binding that has the same name as the role, for now assuming
    # only one per user and one user per namespace
    rb_exists = role_binding_exists_in_namespace(
      name=jobs_create_request.metadata.name,
      namespace=jobs_create_request.metadata.namespace
    )

    if not rb_exists:
      rb_create_response = api_instance.create_namespaced_role_binding(
        name=role_binding_request.metadata.name,
        namespace=role_binding_request.metadata.namespace,
        body=role_binding_request)
    else:
      # Otherwise, replace
      rb_replace_response = api_instance.replace_namespaced_role_binding(
        name=role_binding_request.metadata.name,
        namespace=role_binding_request.metadata.namespace,
        body=role_binding_request)


def main(_):
  update_profile_and_rbac_configurations(
    template_service_account=FLAGS.template_service_account
  )


if __name__ == '__main__':
  app.run(main)
