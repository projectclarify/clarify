#!/usr/bin/bash
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

# Depricated

sudo apt-get update
sudo apt-get install -y curl

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server

pip install --upgrade pip
pip3 install --upgrade pip3

#pip install tensorflow_serving_api==1.12.0
#pip uninstall tensorflow==1.14.0
pip install tensor2tensor==1.14.0
pip install tensorflow-probability==0.7.0

#python -c 'from tensorflow_serving.apis import predict_pb2'
#python -c 'from tensorflow_serving.apis import prediction_service_pb2_grpc'
