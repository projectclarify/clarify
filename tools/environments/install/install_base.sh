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

apt-get update

apt-get install -y pkg-config \
                   zip \
                   g++ \
                   zlib1g-dev \
                   unzip \
                   ffmpeg \
                   wget \
                   curl \
                   libsm6 \
                   libxext6 \
                   libxrender-dev \
                   libglib2.0-0 \
                   htop \
                   git \
                   vim

pip install --upgrade pip
