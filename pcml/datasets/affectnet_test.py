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
"""AffectNet problem definition tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.datasets import affectnet
from tensor2tensor.utils import registry


class TestAffectnetProblem(tf.test.TestCase):

  def test_registry_lookups(self):

    problem_names = ["affectnet_base", "affectnet_tiny"]

    for problem_name in problem_names:
      _ = registry.problem(problem_name)

  def test_tiny_e2e(self):
    pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
