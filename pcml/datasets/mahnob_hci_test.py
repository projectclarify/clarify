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
"""Tests of MAHNOB-HCI problem definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.datasets import mahnob_hci

TESTING_TMP = "/tmp"


class TestMahnobProblem(tf.test.TestCase):

    def test_data_verifier(self):

        mahnob_hci._raw_data_verifier(mahnob_hci.DEFAULT_DATA_ROOT)

    def test_walkthrough(self):

        paths = mahnob_hci.maybe_get_data(TESTING_TMP,
                                          mahnob_hci.DEFAULT_DATA_ROOT).next()
        expected_keys = [
            "video_path", "audio_path", "guide_cut_path", "all_data_path"
        ]
        for key in expected_keys:
            self.assertTrue(key in paths.keys())


if __name__ == "__main__":
    tf.test.main()
