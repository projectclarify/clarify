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
"""Deployment utility.

Would like to conditionally deploy only if something has changed
but for now this will just deploy everything.

"""

import subprocess

from collections import namedtuple

import random
import multiprocessing

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_boolean('all', False,
                     'Whether to deploy all of the core functions.')

flags.DEFINE_string('region', None, 'A GCP region.')

flags.DEFINE_string('iam_member', None, 'A GCP IAM member account ID.')

flags.DEFINE_string('iam_role', None, 'A comma-separated list of IAM roles.')

flags.DEFINE_string('which', '', 'A comma-sep. list of fn names.')


def _deploy_http_trigger_function(fname, region, runtime="python37",
                                  iam_member=None, iam_role=None):
  cmd = [
    "gcloud", "functions", "deploy", "--region", region, "--trigger-http",
    "--update-labels", "function-group=loader",
    "--runtime", runtime, fname
  ]
  print("Initiated deployment with command: {}".format(" ".join(cmd)))
  deploy_output = subprocess.check_output(cmd)
  print(deploy_output)

  iam_output = b""
  if iam_member and iam_role:
    iam_cmd = [
      "gcloud", "alpha", "functions", "add-iam-policy-binding",
      "--region", region, "--member", iam_member, "--role", iam_role
    ]
    print("Updating gcf iam with cmd: {}".format(" ".join(iam_cmd)))
    iam_output = subprocess.check_output(iam_cmd)
    print(iam_output)

  return [deploy_output, iam_output]


def _deploy_check_membership(region, iam_member=None, iam_role=None):
  _deploy_http_trigger_function(fname="check_membership",
                                region=region, iam_member=iam_member,
                                iam_role=iam_role)


def _deploy_mint_join_codes(region, iam_member=None, iam_role=None):
  _deploy_http_trigger_function(fname="mint_join_codes",
                                region=region, iam_member=iam_member,
                                iam_role=iam_role)


def _deploy_maybe_render(region, iam_member=None, iam_role=None):
  _deploy_http_trigger_function(fname="maybe_render",
                                region=region, iam_member=iam_member,
                                iam_role=iam_role)


def _deploy_save_registration(region, iam_member=None, iam_role=None):
  _deploy_http_trigger_function(fname="save_registration",
                                region=region, iam_member=iam_member,
                                iam_role=iam_role)


def _deploy_query_model_template(region, iam_member=None, iam_role=None):
  _deploy_http_trigger_function(fname="query_model_template",
                                region=region, iam_member=iam_member,
                                iam_role=iam_role)


def _deploy_rfn_study_1234(region, iam_member=None, iam_role=None):
  _deploy_http_trigger_function(fname="rfn_study_1234",
                                region=region, iam_member=iam_member,
                                iam_role=iam_role)


# Some functions may need different deploy functions so let's just specify
# what python deploy function to call for each function name. E.g. the
# model inference glue function needs a vpc while the others dont.

FUNCTIONS = {
  "check_membership": _deploy_check_membership,
  "mint_join_codes": _deploy_mint_join_codes,
  "maybe_render": _deploy_maybe_render,
  "save_registration": _deploy_save_registration,
  "query_model_template": _deploy_query_model_template,
  "rfn_study_1234": _deploy_rfn_study_1234
}


def main(argv):

  region = FLAGS.region
  if not region:
    import os
    if "GCP_REGION" not in os.environ:
      raise ValueError("--region unset and $GCP_REGION is unset.")
    region = os.environ["GCP_REGION"]

  which_functions = []

  if FLAGS.all:
    which_functions = list(FUNCTIONS.keys())
    print("Deploying EVERYTHING!")

  # Container for multiprocessing processes
  jobs = []

  exceptions = []

  process_args = namedtuple("pargs", ["region", "iam_member", "iam_role"])(
    region=region, iam_member=FLAGS.iam_member, iam_role=FLAGS.iam_role)

  for function_name in which_functions:
    if function_name not in FUNCTIONS:
      # Because users might provide function names that are not correct
      raise ValueError("unrecognized function {}".format(function_name))
    #FUNCTIONS[function_name](region)

    process = multiprocessing.Process(target=FUNCTIONS[function_name], 
                                      args=process_args)
    jobs.append(process)

  for j in jobs:
    j.start()

  for i, j in enumerate(jobs):
    j.join()
    print("Job {} completed.".format(i))


if __name__ == '__main__':
  app.run(main)
