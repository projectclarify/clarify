#!/bin/bash
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

# Provide the git commit to be tested as the first argument to the script.

# Usage: test_circle.sh ${CIRCLE_SHA1}

docker run -it gcr.io/clarify/runtime-base:v0.1.0-799d \
  -v ~/project:/home/jovyan/pcml \
  /bin/bash -c "source ~/.bashrc && pip install -r dev-requirements.txt --user && sh tools/testing/test_local.sh"
