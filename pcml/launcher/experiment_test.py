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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from pcml.launcher.experiment import tf_config_to_additional_flags


class TestParseTFConfig(tf.test.TestCase):

    def test_parse_tf_config(self):
        """Simple test that tf_config_to_additional_flags can be run."""

        os.environ["TF_CONFIG"] = json.dumps({
            u'environment': u'cloud',
            u'cluster': {
                u'master': [u'enhance-0401-0010-882a-master-5sq4-0:2222']
            },
            u'task': {
                u'index': 0,
                u'type': u'master'
            }
        })

        flags = tf_config_to_additional_flags()
        tf.logging.info(flags)


if __name__ == "__main__":
    tf.test.main()
