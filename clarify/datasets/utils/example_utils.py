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


class ExampleFieldTemplate(object):
  """Model of an example field."""

  def __init__(self, modality, vocab_size, space_id, shape, field_type, dtype):
    self.modality = modality
    self.vocab_size = vocab_size
    self.space_id = space_id
    self.shape = shape
    self.field_type = field_type
    self.dtype = dtype

  @property
  def dtype(self):
    """A tf.dtype.
    
    # TODO: Verify this is a tf.dtype not a numpy dtype if there's not a good
      way to work with them both simultaneously.

    """
    return self.__dtype

  @dtype.setter
  def dtype(self, x):
    self.__dtype = x

  @property
  def modality(self):
    """E.g. a modalities.ModalityType."""
    return self.__modality

  @modality.setter
  def modality(self, x):
    self.__modality = x

  @property
  def vocab_size(self):
    return self.__vocab_size

  @vocab_size.setter
  def vocab_size(self, x):
    assert isinstance(x, int)
    self.__vocab_size = x

  @property
  def space_id(self):
    return self.__space_id

  @space_id.setter
  def space_id(self, x):
    """An integer."""
    assert isinstance(x, int)
    self.__space_id = x

  @property
  def shape(self):
    return self.__shape

  @shape.setter
  def shape(self, x):
    if not isinstance(x, tuple):
      raise ValueError("shape must be a tuple, saw %s" % x)
    self.__shape = x

  def as_dict(self):
    return {
        "dtype": self.dtype,
        "modality": self.modality,
        "vocab_size": self.vocab_size,
        "space_id": self.space_id,
        "shape": self.shape,
        "field_type": self.field_type
    }

  def mock_one(self, zeros=False):
    """Mock an instance that matches the specified field template.
    
    Notes:
      - TODO: Do we need to mock differently depending on graph vs. eager modes?
        Shoudl this be a tf.Feature instead of a tf.Example?

        Can we use tf data types with numpy? Do we need to be able to mock
        tf.Example's as well as dicts of python lists? The former can always
        just be converted to the latter in eager mode.
    
    Args:
      zeros(bool): Whether to mock a tensor of zeros (true) or default to the
        standard of sampling values from [0, self.vocab_size).
    
    """

    if zeros:
      return tf.cast(tf.zeros(shape=self.shape), dtype=self.dtype)

    return tf.random.uniform(shape=self.shape,
                             minval=0,
                             maxval=self.vocab_size,
                             dtype=self.dtype)

  def mock(self, zeros=False, batch_size=1):
    if batch_size is None:
      return self.mock_one(zeros)

    if not isinstance(batch_size, int):
      raise ValueError(
          "batch_size passed to .mock must be of type int, saw %s" % batch_size)
    return tf.convert_to_tensor(
        [self.mock_one(zeros) for _ in range(batch_size)], self.dtype)

  def matches(self, field):
    shape_matches = tf.reduce_all(
        tf.math.equal(tf.shape(field), tf.convert_to_tensor(self.shape)))
    return shape_matches
    #dtype_matches = tf.convert_to_tensor(field.dtype == self.dtype)
    #return tf.reduce_all(tf.concat(shape_matches, dtype_matches))

  def normalize(self, field):
    #HACK: This is causing issues in graph mode
    field = tf.reshape(field, self.shape)
    field = tf.cast(field, self.dtype)
    return field

    if self.matches(field):
      return field
    else:
      # TODO: Haven't yet implemented padding or clipping existing fields to field spec shape
      raise NotImplementedError()


class ExampleTemplate(object):
  """Model of an example."""

  def __init__(self, fields, *args, **kwargs):
    self.fields = fields

  def construct_from_problems(self, template_problems):
    """Construct a template from a collection of problems.

    TODO, if possible and useful... I almost prefer building problems around
    explicit example models instead of the other way around.

    """
    raise NotImplementedError()

  @property
  def fields(self):
    return self.__fields

  @fields.setter
  def fields(self, fields):
    assert isinstance(fields, dict)
    for field in fields.values():
      assert isinstance(field, ExampleFieldTemplate)
    self.__fields = fields

  def normalize(self, example, clip=True):
    """Normalize a single (un-batched) example to the shape of the spec."""
    observed = {}
    for key, value in example.items():
      # TODO: If there's an extra field should we keep it or delete it? Delete for now.
      if key not in self.fields.keys():
        del example[key]
      # If the field is already present confirm it matches by calling .normalize which
      # will pad to the target shape if the shape does not match the desired shape.
      example[key] = self.fields[key].normalize(value)
      observed[key] = True
    for key in self.fields.keys():
      if key not in observed.keys():
        # If a field is missing in the input example, mock it with zeros to the
        # shape and dtype specified in the corresponding ExampleFieldSpec.
        example[key] = self.fields[key].mock_one(zeros=False)
    return example

  def mock_one(self, zeros=False):
    """Mock a random unbatched example that matches the template."""
    return self.mock(batch_size=None, zeros=zeros)

  def mock(self, batch_size=1, zeros=False):
    """Mock a random batched example that matches the template."""
    example = {}
    for key, field_template in self.fields.items():
      example[key] = field_template.mock(batch_size=batch_size, zeros=zeros)
    return example

  def matches(self, example):
    """Check whether an example matches the template."""
    matches = isinstance(example, dict)
    for key, value in example.items():
      matches = matches and isinstance(value, tf.Tensor)
      matches = matches and key in self.fields.keys()
      matches = matches and self.fields[key].matches(value)
    return matches

  def as_dict(self):
    """Return the template as a dictionary of lists."""
    pass
    d = {}
    for key in self.fields:
      d[key] = self.fields[key].as_dict()
    return d


def update_hparams_given_example_spec(hparams, example_spec):
  """Updates 'modality' and 'vocab_size' attrs. of `hparams` given `example_spec`."""
  modality_update = {}
  vocab_size_update = {}
  for field_name, field in example_spec.fields.items():
    modality_update[field_name] = field.modality
    vocab_size_update[field_name] = field.vocab_size
  setattr(hparams, "modality", modality_update)
  setattr(hparams, "vocab_size", vocab_size_update)
  return hparams
