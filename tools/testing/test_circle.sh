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

docker run -it gcr.io/clarify/runtime-base:v0.1.0-2370 \
    /bin/bash -c "source ~/.bashrc && pip install tensorflow==1.15.0 --user && mkdir -p ~/testing && cd ~/testing && git clone https://github.com/projectclarify/clarify.git && cd clarify && git checkout $1 && pip install -r dev-requirements.txt --user && sh tools/testing/test_local.sh"

