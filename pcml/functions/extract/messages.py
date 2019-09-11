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

"""Messages related video extraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

prohibited_targets = [
  # Took a long time to generate, let's not accidentally write the wrong
  # data to it.
  "vox-celeb-2-raw"
]


class ExtractTriggerMessage(object):
  def __init__(self, project, bigtable_instance, target_table_name,
               prefix, mp4_path, video_id):
    self.project = project
    self.bigtable_instance = bigtable_instance

    msg = "Writing data to {} is prohibited".format(target_table_name)
    if target_table_name in prohibited_targets:
      raise ValueError(msg)
    self.target_table_name = target_table_name

    self.prefix = prefix
    self.mp4_path = mp4_path
    self.video_id
