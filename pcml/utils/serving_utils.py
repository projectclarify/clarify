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

"""Utilities wrapping Kubeflow TFServing component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import os
import uuid
import re

import subprocess
from subprocess import Popen, PIPE

import tensorflow as tf


def check_for_ks(ks_path):
  try:
    subprocess.check_call([ks_path])
  except subprocess.CalledProcessError as e:
    tf.logging.error(
        "Can't find ksonnet binary 'ks', please install it.")
    raise e


def ks_init_tmp(ks_path):
  d = tempfile.mkdtemp()
  os.chdir(d)
  print(d)
  _ = subprocess.check_call([ks_path, "init", "deploy"])
  return os.path.join(d, "deploy")


def call(cmd, cwd=None):
  tf.logging.info("Calling command: %s" % cmd)
  p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, cwd=cwd)
  stdout, stderr = p.communicate()
  if p.returncode != 0:
    raise ValueError(stderr.decode())
  else:
    return stdout.decode()


def serve_model(ks_path, model_name, model_export_path, kf_app_path=None,
                kf_env="default"):
  check_for_ks(ks_path)
  #cwd = ks_init_tmp(ks_path)
  cwd = kf_app_path
  tag = str(uuid.uuid4())[-4:]
  model_name = re.sub("_", "-", model_name)
  service_component_name = "%s-service-%s" % (model_name, tag)
  service_component_name = re.sub("_", "-", service_component_name)
  model_component_name = "%s-%s" % (model_name, tag)
  model_component_name = re.sub("_", "-", model_component_name)

  # Generate and configure service
  call("%s generate tf-serving-service %s" % (
      ks_path, service_component_name
  ), cwd=cwd)

  call("%s param set %s modelName %s" % (
      ks_path, service_component_name, model_name
  ), cwd=cwd)

  call("%s param set %s trafficRule v1:100" % (
      ks_path, service_component_name
  ), cwd=cwd)

  call("%s param set %s serviceType ClusterIP" % (
      ks_path, service_component_name
  ), cwd=cwd)

  # Generate and configure serving deployment
  call("%s generate tf-serving-deployment-gcp %s" % (
      ks_path, model_component_name
  ), cwd=cwd)

  call("%s param set %s modelName %s" % (
      ks_path, model_component_name, model_name
  ), cwd=cwd)

  call("%s param set %s versionName v1" % (
      ks_path, model_component_name
  ), cwd=cwd)

  call("%s param set %s modelBasePath %s" % (
      ks_path, model_component_name, model_export_path
  ), cwd=cwd)

  call("%s param set %s gcpCredentialSecretName user-gcp-sa" % (
      ks_path, model_component_name
  ), cwd=cwd)

  call("%s param set %s injectIstio true" % (
      ks_path, model_component_name
  ), cwd=cwd)

  #call("%s param set %s numGpus 1" % (
  #    ks_path, model_component_name
  #), cwd=cwd)

  # Deploy it
  call("%s env set default" % (ks_path), cwd=cwd)

  call("%s apply %s -c %s" % (
      ks_path, kf_env, service_component_name
  ), cwd=cwd)

  call("%s apply %s -c %s" % (
      ks_path, kf_env, model_component_name
  ), cwd=cwd)
