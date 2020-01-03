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
"""Multi-modal neuro-imaging multi-problems."""

import tensorflow as tf

from tensor2tensor.data_generators import multi_problem_v2
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem

from pcml.datasets.example_utils import ExampleTemplate
from pcml.datasets.example_utils import ExampleFieldTemplate
from pcml.datasets.utils import gen_dummy_schedule

from pcml.datasets import celeba
from pcml.datasets import deap
from pcml.datasets import vox_celeb


class MultiModalImagingExampleSpecOld(ExampleTemplate):

  def __init__(self,
               video_shape=(4, 4, 4, 3),
               image_shape=(4, 4, 3),
               audio_shape=(100,),
               eeg_shape=(4, 4, 4, 3),
               target_shape=(12,),
               *args,
               **kwargs):
    super(MultiModalImagingExampleSpec, self).__init__(
        fields={
            "image":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=image_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "audio":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=audio_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "video":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=video_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "eeg":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=eeg_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "targets":
                ExampleFieldTemplate(modality=modalities.ModalityType.SYMBOL,
                                     vocab_size=256,
                                     space_id=problem.SpaceID.DIGIT_1,
                                     shape=target_shape,
                                     field_type="target",
                                     dtype=tf.float32),
            "problem_id":
                ExampleFieldTemplate(
                    modality=None,
                    vocab_size=2,  #HACK
                    space_id=-1,
                    shape=(1,),
                    field_type=None,
                    dtype=tf.float32),
        },
        *args,
        **kwargs)


class MultiModalImagingExampleSpec(ExampleTemplate):

  def __init__(self,
               video_shape=(4, 4, 4, 3),
               image_shape=(4, 4, 3),
               audio_shape=(100,),
               eeg_shape=(4, 4, 4, 3),
               target_shape=(12,),
               *args,
               **kwargs):
    super(MultiModalImagingExampleSpec, self).__init__(
        fields={
            "image":
                ExampleFieldTemplate(modality=None,
                                     vocab_size=1,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=image_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "audio":
                ExampleFieldTemplate(modality=None,
                                     vocab_size=1,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=audio_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "video":
                ExampleFieldTemplate(modality=None,
                                     vocab_size=1,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=video_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "eeg":
                ExampleFieldTemplate(modality=None,
                                     vocab_size=1,
                                     space_id=problem.SpaceID.DIGIT_0,
                                     shape=eeg_shape,
                                     field_type="input",
                                     dtype=tf.float32),
            "targets":
                ExampleFieldTemplate(modality=None,
                                     vocab_size=1,
                                     space_id=problem.SpaceID.DIGIT_1,
                                     shape=target_shape,
                                     field_type="target",
                                     dtype=tf.float32),
            "problem_id":
                ExampleFieldTemplate(
                    modality=None,
                    vocab_size=2,  #HACK
                    space_id=-1,
                    shape=(1,),
                    field_type=None,
                    dtype=tf.float32),
        },
        *args,
        **kwargs)


# HACK: This is temporary until ExampleTemplate pads targets to common len
@registry.register_problem
class DeapProblem(deap.DeapProblemBase):

  def preprocess_example(self, example, mode, hparams):
    example["targets"] = tf.pad(example["targets"], tf.constant([[0, 8]]))
    return example


# TODO: Consider setting only_eval_first_problem=True.
@registry.register_problem
class MultiModalImagingMultiProblemDev(multi_problem_v2.MultiProblemV2,
                                       vox_celeb.VoxCelebProblemDev):

  def __init__(self, *args, **kwargs):

    deap_problem = DeapProblem()
    vox_celeb_problem = vox_celeb.VoxCelebProblemDev()
    celeba_problem = celeba.ImageCelebaTiny()

    self.example_specification = MultiModalImagingExampleSpec(
        eeg_shape=(32 * 100,),  # HACK
        image_shape=(4, 4, 3)  # HACK
    )

    problems = [vox_celeb_problem, celeba_problem, deap_problem]

    schedule = gen_dummy_schedule(len(problems))

    super(MultiModalImagingMultiProblemDev, self).__init__(problems=problems,
                                                           schedule=schedule,
                                                           *args,
                                                           **kwargs)

  def normalize_example(self, example, _):
    return self.example_specification.normalize(example)

  def generate_data(self, *args, **kwargs):
    for i, p in enumerate(self.problems):
      p.generate_data(task_id=i, *args, **kwargs)

  def example_reading_spec(self):

    image_shape = self.example_specification.fields["image"].shape
    eeg_shape = self.example_specification.fields["eeg"].shape

    data_fields = {
        "audio": tf.FixedLenFeature(self.audio_shape, dtype=tf.int64),
        "video": tf.FixedLenFeature(self.video_shape, dtype=tf.int64),
        "image": tf.FixedLenFeature(image_shape, dtype=tf.int64),
        "eeg": tf.FixedLenFeature(eeg_shape, dtype=tf.int64),
        "targets": tf.FixedLenFeature([self.num_classes], dtype=tf.int64),
    }

    data_items_to_decoders = None

    return data_fields, data_items_to_decoders


@registry.register_problem
class MultiModalImagingMultiProblem(multi_problem_v2.MultiProblemV2,
                                    vox_celeb.VoxCelebProblemDim64):

  def __init__(self, *args, **kwargs):

    vox_celeb_problem = vox_celeb.VoxCelebProblemDim64()
    celeba_problem = celeba.ImageCelebaPcml()

    self.example_specification = MultiModalImagingExampleSpec(
        video_shape=(4, 64, 64, 3),
        image_shape=(64, 64, 3),
        audio_shape=(1926,),
        eeg_shape=(32 * 100,),  # HACK
        target_shape=(12,),
    )

    problems = [
        vox_celeb_problem,
        celeba_problem,
    ]

    # TODO: What happens when we run out of steps in our schedule but
    # still have training steps left over? Do we run off the end with
    # whatever problem we were using last or does it keep following
    # the same distribution? Very important to know.
    schedule = gen_dummy_schedule(len(problems), num_steps=6000)

    super(MultiModalImagingMultiProblem, self).__init__(problems=problems,
                                                        schedule=schedule,
                                                        *args,
                                                        **kwargs)

  def normalize_example(self, example, _):
    return self.example_specification.normalize(example)

  def generate_data(self, *args, **kwargs):
    for i, p in enumerate(self.problems):
      p.generate_data(task_id=i, *args, **kwargs)

  def example_reading_spec(self):

    image_shape = self.example_specification.fields["image"].shape
    eeg_shape = self.example_specification.fields["eeg"].shape

    data_fields = {
        "audio": tf.FixedLenFeature(self.audio_shape, dtype=tf.int64),
        "video": tf.FixedLenFeature(self.video_shape, dtype=tf.int64),
        "image": tf.FixedLenFeature(image_shape, dtype=tf.int64),
        "eeg": tf.FixedLenFeature(eeg_shape, dtype=tf.int64),
        "targets": tf.FixedLenFeature([self.num_classes], dtype=tf.int64),
    }

    data_items_to_decoders = None

    return data_fields, data_items_to_decoders


@registry.register_problem
class MultiModalImagingMultiProblemLarge(multi_problem_v2.MultiProblemV2,
                                         vox_celeb.VoxCelebProblemDim64):

  def __init__(self, *args, **kwargs):

    vox_celeb_problem = vox_celeb.VoxCelebProblemDim64Vlarge()
    celeba_landmark_problem = celeba.ImageCelebaPcml()
    celeba_attr_problem = celeba.ImageCelebaAttributes()

    self.example_specification = MultiModalImagingExampleSpec(
        video_shape=(4, 64, 64, 3),
        image_shape=(64, 64, 3),
        audio_shape=(1926,),
        eeg_shape=(32 * 100,),  # HACK
        target_shape=(12,),
    )

    problems = [vox_celeb_problem, celeba_landmark_problem, celeba_attr_problem]

    # TODO: What happens when we run out of steps in our schedule but
    # still have training steps left over? Do we run off the end with
    # whatever problem we were using last or does it keep following
    # the same distribution? Very important to know.
    schedule = gen_dummy_schedule(len(problems), num_steps=500000)

    super(MultiModalImagingMultiProblemLarge, self).__init__(problems=problems,
                                                             schedule=schedule,
                                                             *args,
                                                             **kwargs)

  def normalize_example(self, example, _):
    return self.example_specification.normalize(example)

  def generate_data(self, *args, **kwargs):
    for i, p in enumerate(self.problems):
      p.generate_data(task_id=i, *args, **kwargs)

  def example_reading_spec(self):

    image_shape = self.example_specification.fields["image"].shape
    eeg_shape = self.example_specification.fields["eeg"].shape

    data_fields = {
        "audio": tf.FixedLenFeature(self.audio_shape, dtype=tf.int64),
        "video": tf.FixedLenFeature(self.video_shape, dtype=tf.int64),
        "image": tf.FixedLenFeature(image_shape, dtype=tf.int64),
        "eeg": tf.FixedLenFeature(eeg_shape, dtype=tf.int64),
        "targets": tf.FixedLenFeature([self.num_classes], dtype=tf.int64),
    }

    data_items_to_decoders = None

    return data_fields, data_items_to_decoders


@registry.register_problem
class MultiModalImagingMultiProblemV2(multi_problem_v2.MultiProblemV2,
                                      vox_celeb.VoxCelebProblemV2):

  def __init__(self, *args, **kwargs):

    vox_celeb_problem = vox_celeb.VoxCelebProblemV2()
    celeba_landmark_problem = celeba.ImageCelebaPcml()
    celeba_attr_problem = celeba.ImageCelebaAttributes()

    self.example_specification = MultiModalImagingExampleSpec(
        video_shape=(4, 64, 64, 3),
        image_shape=(64, 64, 3),
        audio_shape=(1926,),
        eeg_shape=(32 * 100,),  # HACK
        target_shape=(12,),
    )

    problems = [vox_celeb_problem, celeba_landmark_problem, celeba_attr_problem]

    # TODO: What happens when we run out of steps in our schedule but
    # still have training steps left over? Do we run off the end with
    # whatever problem we were using last or does it keep following
    # the same distribution? Very important to know.
    schedule = gen_dummy_schedule(len(problems), num_steps=100)

    super(MultiModalImagingMultiProblemV2, self).__init__(problems=problems,
                                                          schedule=schedule,
                                                          *args,
                                                          **kwargs)

  def normalize_example(self, example, _):
    return self.example_specification.normalize(example)

  def generate_data(self, *args, **kwargs):
    for i, p in enumerate(self.problems):
      p.generate_data(task_id=i, *args, **kwargs)

  def example_reading_spec(self):

    image_shape = self.example_specification.fields["image"].shape
    eeg_shape = self.example_specification.fields["eeg"].shape

    data_fields = {
        "audio": tf.FixedLenFeature(self.audio_shape, dtype=tf.float32),
        "video": tf.FixedLenFeature(self.video_shape, dtype=tf.float32),
        "image": tf.FixedLenFeature(image_shape, dtype=tf.float32),
        "eeg": tf.FixedLenFeature(eeg_shape, dtype=tf.float32),
        "targets": tf.FixedLenFeature([self.num_classes], dtype=tf.float32),
    }

    data_items_to_decoders = None

    return data_fields, data_items_to_decoders
