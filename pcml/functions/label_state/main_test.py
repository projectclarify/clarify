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

from collections import UserDict
from datetime import datetime
import json
import uuid
import numpy as np

from mock import MagicMock, patch

import main


class Context(object):
    pass


@patch('main.client')
def test_label_state(firestore_mock, capsys):

    firestore_mock.collection = MagicMock(return_value=firestore_mock)
    firestore_mock.document = MagicMock(return_value=firestore_mock)
    firestore_mock.set = MagicMock(return_value=firestore_mock)

    user_id = str(uuid.uuid4())
    date_string = datetime.now().isoformat()
    email_string = '%s@%s.com' % (uuid.uuid4(), uuid.uuid4())
    video = np.random.randint(0, 255, (15, 96, 96, 3), dtype=np.uint8)
    video = bytes(video.flatten())
    audio = np.random.randint(0, 255, (1000,), dtype=np.uint8)
    audio = bytes(audio.flatten())

    data = {
        'uid': user_id,
        'metadata': {
            'createdAt': date_string
        },
        'email': email_string,
        'value': {
            'fields': {
                'audioData': {
                    'stringValue': audio
                },
                'videoData': {
                    'stringValue': video
                },
                'audioMeta': {
                    "mapValue": {
                        "fields": {}
                    }
                },
                'videoMeta': {
                    "mapValue": {
                        "fields": {}
                    }
                },
                'meta': {
                    "mapValue": {
                        "fields": {}
                    }
                }
            }
        }
    }

    context = UserDict()
    context.resource = '/documents/users/{uid}/modalities/av'

    main.label_state(data, context)

    out, _ = capsys.readouterr()

    #assert 'Received healthy model response.' in out
