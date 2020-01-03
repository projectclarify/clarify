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
"""Tests of VGGFace2 problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tempfile
import uuid
import os

from pcml.datasets import vggface2
from tensor2tensor.utils import registry

from pcml.utils.dev_utils import T2TDevHelper

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestVGGFace2Problem(tf.test.TestCase):

  def setUp(self):

    self.project = TEST_CONFIG.get("project")
    self.instance = TEST_CONFIG.get("test_cbt_instance")
    self.tmpdir = tempfile.mkdtemp()
    self.salt = str(uuid.uuid4())[0:8]
    self.test_run_tag = "clarify-test-{}-vggface-cbt".format(self.salt)
    self.mode = "train"
    self.test_problem_name = "vgg_face2_tiny"
    self.staging = os.path.join(TEST_CONFIG.test_artifacts_root,
                                self.test_run_tag)

  def test_registry_lookups(self):

    problem_names = ["vgg_face2", "vgg_face2_tiny"]

    for problem_name in problem_names:
      _ = registry.problem(problem_name)

  def test_cbt_generate(self):

    prob = registry.problem(self.test_problem_name)
    prob.mode = self.mode
    prob.dataset_version_tag = self.salt
    prob.cbt_generate(self.project, self.instance, self.mode)

    selection = prob.dataset_selection(self.mode)
    example_iterator = selection.iterate_tfexamples()
    ex = example_iterator.__next__()
    self.assertTrue(ex is not None)

  def test_tiny_e2e(self):

    tmp = tempfile.mkdtemp()

    helper = T2TDevHelper(problem_name=self.test_problem_name,
                          model_name="percep_similarity_triplet_emb",
                          hparams_set="mcl_res_ut_vtiny",
                          data_dir=tmp,
                          tmp_dir=tmp,
                          queries=None,
                          mode="train")

    helper.problem.dataset_version_tag = self.salt
    helper.problem.cbt_generate(self.project, self.instance, "train")
    helper.problem.cbt_generate(self.project, self.instance, "eval")

    ex = helper.eager_get_example()

    helper.eager_train_one_step()

    # HACK
    def _mock_tfrecords_for_mode(mode):
      path = os.path.join(tmp, "{}-{}-01-of-01".format(self.test_problem_name,
                                                       mode))
      with open(path, "w") as f:
        f.write("")

    _mock_tfrecords_for_mode("train")
    _mock_tfrecords_for_mode("eval")

    helper.train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
