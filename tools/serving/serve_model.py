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
