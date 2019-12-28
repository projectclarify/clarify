#!/usr/bin/bash

sudo apt-get update
sudo apt-get install -y curl

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server

pip install tensorflow_serving_api

python -c 'from tensorflow_serving.apis import predict_pb2'
python -c 'from tensorflow_serving.apis import prediction_service_pb2_grpc'