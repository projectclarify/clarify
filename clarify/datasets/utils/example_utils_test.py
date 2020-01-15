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
"""Example utils."""

import tensorflow as tf

import numpy as np

from tensor2tensor.data_generators import algorithmic
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import multi_problem_v2
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from pcml.utils.dev_utils import T2TDevHelper

from pcml.datasets.example_utils import ExampleFieldTemplate
from pcml.datasets.example_utils import ExampleTemplate
from pcml.datasets.utils import gen_dummy_schedule


class DevExampleTemplate(ExampleTemplate):

  def __init__(self, *args, **kwargs):
    super(DevExampleTemplate, self).__init__(
        fields={
            "modalitya":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=(40,),
                                     field_type="input",
                                     dtype=tf.int64),
            "modalityb":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=(40,),
                                     field_type="input",
                                     dtype=tf.int64),
            "modalityc":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=(40,),
                                     field_type="input",
                                     dtype=tf.int64),
            "targets":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_1,
                                     shape=(1,),
                                     field_type="target",
                                     dtype=tf.int64),
            "problem_id":
                ExampleFieldTemplate(
                    modality=None,
                    vocab_size=2,  #HACK
                    space_id=-1,
                    shape=(1,),
                    field_type=None,
                    dtype=tf.int64),
        },
        *args,
        **kwargs)


@registry.register_problem
class DummyProblem(algorithmic.AlgorithmicProblem):

  @property
  def num_symbols(self):
    return 2

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 40

  @property
  def train_size(self):
    return 100

  @property
  def dev_size(self):
    return 100

  @property
  def num_shards(self):
    return 1

  def generator(self, nbr_symbols, max_length, nbr_cases):
    """Generator for the identity (copy) task on sequences of symbols.
    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn uniformly at random from [0, nbr_symbols) until
    nbr_cases sequences have been produced.
    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list and target-list are the same.
    """
    for _ in range(nbr_cases):
      l = max_length + 1
      inputs = [np.random.randint(nbr_symbols) for _ in range(l)]
      yield {"inputs": inputs, "targets": inputs}


@registry.register_problem
class DummyProblemA(DummyProblem):

  def preprocess_example(self, example, mode, hparams):
    return {
        "modalitya": example["inputs"],
        "modalityb": example["inputs"],
        "targets": example["targets"]
    }


@registry.register_problem
class DummyProblemB(DummyProblem):

  def preprocess_example(self, example, mode, hparams):
    return {
        "modalitya": example["inputs"],
        "modalityc": example["inputs"],
        "targets": example["targets"]
    }


@registry.register_problem
class MultiModalTestMultiProblemDev(multi_problem_v2.MultiProblemV2,
                                    algorithmic.AlgorithmicIdentityBinary40):
  """Dataset scheduling for multiple text-to-text problems."""

  def __init__(self, **kwargs):
    problems = [DummyProblemA(), DummyProblemB()]
    super(MultiModalTestMultiProblemDev,
          self).__init__(problems=problems,
                         schedule=gen_dummy_schedule(len(problems)),
                         **kwargs)
    self.normalization_spec = DevExampleTemplate()

  def normalize_example(self, example, hparams):
    """Assumes that example contains both inputs and targets."""
    return self.normalization_spec.normalize(example)


@registry.register_model
class TrivialModel(t2t_model.T2TModel):

  def body(self, features):
    return features["targets"]


class TestExampleUtils(tf.test.TestCase):

  def test_e2e(self):
    """This test is broken."""

    helper = T2TDevHelper("trivial_model", "multi_modal_test_multi_problem_dev",
                          "transformer_tiny", [["1 1 0 1 0"]])

    helper.datagen()
    helper.train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
