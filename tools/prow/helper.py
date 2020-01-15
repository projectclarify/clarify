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

"""Prow deployment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import subprocess
import os
import shutil
import sys


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


def add_hook(test_infra_root, hook_secret_path, gh_token_path, address,
             repo_path="projectclarify/pcml"):
  os.chdir(test_infra_root)
  run_and_output([
    "bazel", "run", "//experiment/add-hook", "--",
    "--hmac-path={}".format(hook_secret_path),
    "--github-token-path={}".format(gh_token_path),
    "--hook-url", "http://{}/hook".format(address),
    "--repo", repo_path,
    "--confirm=true"])


def check_config(test_infra_root, plugins_path, config_path):
  os.chdir(test_infra_root)
  return run_and_output([
    "bazel", "run", "//prow/cmd/checkconfig", "--",
    "--plugin-config={}".format(plugins_path),
    "--config-path={}".format(config_path)])


def deploy_base(test_infra_root):

  os.chdir(test_infra_root)

  run_and_output(["kubectl", "apply", "-f",
                  "prow/cluster/starter.yaml"])


def _update_config(test_infra_root, path, name):

  os.chdir(test_infra_root)

  if name not in ["plugins", "config"]:
    raise ValueError("Unexpected config name")

  tmpdir = tempfile.mkdtemp()
  cfg_map = os.path.join(tmpdir, "configmap")

  with open(cfg_map, "w") as f:
    for line in run_and_output([
      "kubectl", "create", "configmap", name,
      "--from-file={}.yaml={}".format(name, path),
      "--dry-run", "-o", "yaml"]):
      f.write(line)

  run_and_output([
    "kubectl", "replace", "configmap", name, "-f",
    cfg_map])


def update_config(test_infra_root, plugins_path, config_path):

  os.chdir(test_infra_root)

  _update_config(
      test_infra_root=test_infra_root,
      path=plugins_path,
      name="plugins"
  )

  _update_config(
      test_infra_root=test_infra_root,
      path=config_path,
      name="config"
  )


def get_prow():
  tmpdir = tempfile.mkdtemp()
  os.chdir(tmpdir)
  run_and_output([
    "git", "clone",
    "https://github.com/kubernetes/test-infra.git"
  ])
  test_infra_root = os.path.join(tmpdir, "test-infra")
  return test_infra_root


def get_prow_address(namespace="default"):

  for i in range(30):
    address = run_and_output([
      "kubectl", "get", "ingress", "ing",
      "-n", namespace,
      "-o", "jsonpath='{.status.loadBalancer.ingress[0].ip}'"
    ])
    if address:
      break

    time.sleep(2)

  return address


def deploy(test_infra_root, gh_token_path, namespace="default"):

  if not os.path.exists(gh_token_path):
    raise ValueError("Please provide a GH token at path `gh_token_path`")

  hook_secret_path = tempfile.NamedTemporaryFile()

  os.system(" ".join(["openssl", "rand", "-hex", "20", ">", hook_secret_path]))

  run_and_output([
    "kubectl", "create", "secret", "generic", "hmac-token",
    "--from-file=hmac={}".format(hook_secret_path)
  ])

  run_and_output([
    "kubectl", "create", "secret", "generic", "oauth-token",
    "--from-file=oauth={}".format(gh_token_path)
  ])

  deploy_base(test_infra_root=test_infra_root)

  address = get_prow_address()

  add_hook(test_infra_root=test_infra_root,
           hook_secret_path=hook_secret_path,
           gh_token_path=gh_token_path,
           address=address,
           repo_path="projectclarify/pcml")

  return test_infra_root


def reconfigure(test_infra_root, namespace="default"):

  ctx_dir = os.path.dirname(os.path.abspath(__file__))
  plugins_path = os.path.join(ctx_dir, "plugins.yaml")
  config_path = os.path.join(ctx_dir, "config.yaml")

  check_output = check_config(test_infra_root=test_infra_root,
                              plugins_path=plugins_path,
                              config_path=config_path)

  # TODO: Parse check output to see if the check passed, otherwise raise an error

  update_config(test_infra_root=test_infra_root,
                plugins_path=plugins_path,
                config_path=config_path)


if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser(description='Deploy/reconfigure prow.')

  parser.add_argument('--hook_secret_path', type=str, default=None,
                      required=True,
                      help='Full path to a hook secret.')

  parser.add_argument('--gh_token_path', type=str, default=None,
                      required=True,
                      help='Full path to a GH token.')

  parser.add_argument('--test_infra_root', type=str, default=None,
                      required=False,
                      help='Path to root of kube test-infra.')

  parser.add_argument('--mode', type=str, default=None,
                      required=True,
                      help='Full deployment or config update modes.')

  args = parser.parse_args()

  if args.mode not in ["deploy", "update"]:
    raise ValueError("Unrecognized --mode {}".format(args.mode))

  test_infra_root = args.test_infra_root
  if not test_infra_root:
    test_infra_root = get_prow()

  if args.mode == "deploy":

    deploy(test_infra_root=test_infra_root,
           hmac_path=args.hook_secret_path,
           gh_token_path=args.gh_token_path)

  reconfigure(test_infra_root)
