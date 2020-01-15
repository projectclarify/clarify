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

from kubernetes import client

from kfserving import KFServingClient
from kfserving import constants
from kfserving import utils
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2CustomSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kubernetes.client import V1ResourceRequirements

namespace = utils.get_default_target_namespace()

api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION

default_endpoint_spec = V1alpha2EndpointSpec(
    predictor=V1alpha2PredictorSpec(
      tensorflow=V1alpha2CustomSpec(
        storage_uri='gs://kfserving-samples/models/tensorflow/flowers',
        resources=V1ResourceRequirements(
          requests={'cpu':'100m','memory':'1Gi'},
          limits={'cpu':'100m', 'memory':'1Gi'}
        )
      )
    )
)

isvc = V1alpha2InferenceService(
    api_version=api_version,
    kind=constants.KFSERVING_KIND,
    metadata=client.V1ObjectMeta(
      name='flower-sample',
      namespace=namespace
    ),
    spec=V1alpha2InferenceServiceSpec(
        default=default_endpoint_spec
    )
)

KFServing = KFServingClient()

KFServing.create(isvc)

KFServing.get('flower-sample',
              namespace=namespace,
              watch=True,
              timeout_seconds=120)

# KFServing.delete('flower-sample', namespace=namespace)
