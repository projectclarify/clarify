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
"""Image Facial Expression Correspondence (image-FEC)."""

from trax import layers as tl
from trax.models.resnet import ConvBlock
from trax.models.resnet import IdentityBlock


# Forked from trax, just to help get set up
def ImageFEC(d_hidden=64,
             n_output_classes=1001,
             mode='train',
             norm=tl.BatchNorm,
             non_linearity=tl.Relu):
  """ResNet.
  Args:
    d_hidden: Dimensionality of the first hidden layer (multiplied later).
    n_output_classes: Number of distinct output classes.
    mode: Whether we are training or evaluating or doing inference.
    norm: `Layer` used for normalization, Ex: BatchNorm or
      FilterResponseNorm.
    non_linearity: `Layer` used as a non-linearity, Ex: If norm is
      BatchNorm then this is a Relu, otherwise for FilterResponseNorm this
      should be ThresholdedLinearUnit.
  Returns:
    The list of layers comprising a ResNet model with the given parameters.
  """

  # A ConvBlock configured with the given norm, non-linearity and mode.
  def Resnet50ConvBlock(filter_multiplier=1, strides=(2, 2)):
    filters = ([
        filter_multiplier * dim for dim in [d_hidden, d_hidden, 4 * d_hidden]
    ])
    return ConvBlock(3, filters, strides, norm, non_linearity, mode)

  # Same as above for IdentityBlock.
  def Resnet50IdentityBlock(filter_multiplier=1):
    filters = ([
        filter_multiplier * dim for dim in [d_hidden, d_hidden, 4 * d_hidden]
    ])
    return IdentityBlock(3, filters, norm, non_linearity, mode)

  return tl.Serial(
      tl.ToFloat(),
      tl.Conv(d_hidden, (7, 7), (2, 2), 'SAME'),
      norm(mode=mode),
      non_linearity(),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),
      Resnet50ConvBlock(strides=(1, 1)),
      [Resnet50IdentityBlock() for _ in range(2)],
      Resnet50ConvBlock(2),
      [Resnet50IdentityBlock(2) for _ in range(3)],
      Resnet50ConvBlock(4),
      [Resnet50IdentityBlock(4) for _ in range(5)],
      Resnet50ConvBlock(8),
      [Resnet50IdentityBlock(8) for _ in range(2)],
      tl.AvgPool(pool_size=(7, 7)),
      tl.Flatten(),
      tl.Dense(n_output_classes),
      tl.LogSoftmax(),
  )
