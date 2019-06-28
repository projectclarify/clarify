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

"""Environment configuration utilities."""

from pcml.utils.fs_utils import get_pcml_root
import json
import os
import tensorflow as tf

DEFAULT_CONFIG_PATH = os.path.join(
  get_pcml_root(), "default_config.json"
)


class Config(object):

  def __init__(self,
               project=None,
               service_account_path=None,
               from_path=DEFAULT_CONFIG_PATH,
               test_artifacts_root=None):

    self.project = project
    self.service_account_path = service_account_path
    if isinstance(from_path, str):
      self.load_config_from_path(from_path)

    #if isinstance(self.service_account_path, str):
    #  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

  def load_config_from_path(self, path):

    with open(path, "r") as config_file:
      data = json.load(config_file)

    for key, value in data.items():
      setattr(self, key, value)

    if "service_account_path" in data:
      sa_path = data["service_account_path"]
      if isinstance(sa_path, str):

        with open(sa_path, "r") as sa_file:
          sa_data = json.load(sa_path)

        if "project" not in sa_data:
          raise ValueError("Expected project in sa key data.")
        self.project = sa_data["project"]

  def get(self, attr):
    if hasattr(self, attr):
      return getattr(self, attr)
    return None
