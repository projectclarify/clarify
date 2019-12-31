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

from pcml.test_config import PCML_CONFIG


class Config(object):

  def __init__(self,
               project=None,
               service_account_path=None,
               test_artifacts_root=None):

    self.project = project
    self.service_account_path = service_account_path

    for key, value in PCML_CONFIG.items():
      setattr(self, key, value)

    if "service_account_path" in PCML_CONFIG:
      sa_path = PCML_CONFIG["service_account_path"]
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
