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
"""Tests of file-system utilities."""

import tensorflow as tf
import tempfile

from pcml.utils import fs_utils


class TestFSUtils(tf.test.TestCase):

    def test_finds_workspace_root(self):
        """Test that we can run a command."""

        root = fs_utils.get_pcml_root()

        #self.assertEqual(root, "/home/jovyan/work/pcml")
        # TODO: Generalize or disable by default.

    def test_expect_path(self):
        tmpd = tempfile.mkdtemp()
        fs_utils.expect_path(tmpd)


if __name__ == "__main__":
    tf.test.main()
