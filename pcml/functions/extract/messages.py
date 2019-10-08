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

import inspect

prohibited_targets = [
  # Took a long time to generate, let's not accidentally write the wrong
  # data to it.
  #"vox-celeb-2-raw"
]


def _expect_type(obj, t):
    if not isinstance(obj, t):
      raise ValueError("variable {} should be of type {}, saw {}".format(
        obj, t, type(obj)
      ))


class ExtractTriggerMessage(object):
  def __init__(self, project, bigtable_instance, target_table_name,
               prefix, mp4_paths, video_ids, greyscale=False,
               downsample_xy_dims=96, resample_every=2,
               audio_block_size=1000):

    _expect_type(project, str)
    self.project = project

    _expect_type(bigtable_instance, str)
    self.bigtable_instance = bigtable_instance

    msg = "Writing data to {} is prohibited".format(target_table_name)
    if target_table_name in prohibited_targets:
      raise ValueError(msg)
    _expect_type(target_table_name, str)
    self.target_table_name = target_table_name

    _expect_type(prefix, str)
    self.prefix = prefix    

    _expect_type(mp4_paths, list)
    self.mp4_paths = mp4_paths

    _expect_type(video_ids, list)
    self.video_ids = video_ids

    if len(mp4_paths) != len(video_ids):
      raise ValueError("Length of paths and ids should match: {}, {}".format(
        len(mp4_paths), len(video_ids)
      ))

    _expect_type(greyscale, bool)
    self.greyscale = greyscale

    _expect_type(downsample_xy_dims, int)
    self.downsample_xy_dims = downsample_xy_dims

    _expect_type(resample_every, int)
    self.resample_every = resample_every

    _expect_type(audio_block_size, int)
    self.audio_block_size = audio_block_size
