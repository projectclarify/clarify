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
"""Utilities for constructing Katib StudyJob's."""

import datetime
import json
import logging
import time
import yaml
import tensorflow as tf
import multiprocessing

import kubernetes
from kubernetes import client as k8s_client
from kubernetes.client import rest

from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams

from pcml.launcher.util import dict_prune_private
from pcml.launcher.util import object_as_dict

STUDY_JOB_VERSION = "v1alpha1"
STUDY_JOB_GROUP = "kubeflow.org"
STUDY_JOB_PLURAL = "studyjobs"
STUDY_JOB_KIND = "StudyJob"
STUDY_JOB_API_VERSION = "%s/%s" % (STUDY_JOB_GROUP, STUDY_JOB_VERSION)

TIMEOUT = 120


# wait_for_condition, create_study_job, and delete_study_job borrowed
# from the katib codebase with only minor modification
def wait_for_condition(client,
                       namespace,
                       name,
                       expected_condition=[],
                       version="v1alpha1",
                       timeout=datetime.timedelta(minutes=10),
                       polling_interval=datetime.timedelta(seconds=30),
                       status_callback=None):
    """Waits until any of the specified conditions occur.

  Args:
    client: K8s api client.
    namespace: namespace for the studyjob.
    name: Name of the studyjob.
    expected_condition: A list of conditions. Function waits until any of
      the supplied conditions is reached.
    timeout: How long to wait for the job.
    polling_interval: How often to poll for the status of the job.
    status_callback: (Optional): Callable. If supplied this callable is
      invoked after we poll the job. Callable takes a single argument which
      is the job.

  """

    crd_api = k8s_client.CustomObjectsApi(client)
    end_time = datetime.datetime.now() + timeout

    while True:

        thread = crd_api.get_namespaced_custom_object(STUDY_JOB_GROUP,
                                                      version,
                                                      namespace,
                                                      STUDY_JOB_PLURAL,
                                                      name,
                                                      async_req=True)

        results = None
        try:
            results = thread.get(TIMEOUT)

        except multiprocessing.TimeoutError:
            logging.error(
                ("Timeout trying to get studyJob %s/%s." % (namespace, name)))

        except Exception as e:
            logging.error(("There was a problem waiting for StudyJob "
                           "%s/%s; Exception: %s" % (namespace, name, e)))
            raise

        if results:
            if status_callback:
                status_callback(results)

            condition = results.get("status", {}).get("condition")
            if condition in expected_condition:
                return results

        if datetime.datetime.now() + polling_interval > end_time:
            raise multiprocessing.TimeoutError(
                ("Timeout waiting for studyJob %s in namespace %s "
                 " to enter one of the conditions %s " %
                 (name, namespace, expected_condition)), results)

        time.sleep(polling_interval.seconds)


def create_study_job(client, spec, version="v1alpha1"):
    """Create a studyJob.

  Args:
    client: A K8s api client.
    spec: The spec for the job.
  """

    crd_api = k8s_client.CustomObjectsApi(client)

    try:

        namespace = spec["metadata"].get("namespace", "default")

        thread = crd_api.create_namespaced_custom_object(STUDY_JOB_GROUP,
                                                         version,
                                                         namespace,
                                                         STUDY_JOB_PLURAL,
                                                         spec,
                                                         async_req=True)

        api_response = thread.get(TIMEOUT)

        tf.logging.info(
            ("Created studyJob %s" % api_response["metadata"]["name"]))

        return api_response

    except rest.ApiException as e:

        message = ""

        if e.message:
            message = e.message

        if e.body:

            try:

                body = json.loads(e.body)

            except ValueError:

                # There was a problem parsing the body of the response as json.
                tf.logging.error(
                    ("Exception when calling DefaultApi->"
                     "apis_fqdn_v1_namespaces_namespace_resource_post. "
                     "body: %s" % e.body))

                raise

            message = body.get("message")

        tf.logging.error(("Exception when calling DefaultApi->"
                          "apis_fqdn_v1_namespaces_namespace_resource_post: "
                          "%s" % message))

        raise e


def delete_study_job(client, name, namespace, version="v1alpha1"):
    """Delete a StudyJob by `name` in `namespace`."""

    crd_api = k8s_client.CustomObjectsApi(client)

    try:

        body = {
            "propagationPolicy": "Foreground",
        }

        tf.logging.info("Deleting studyJob %s/%s", namespace, name)

        thread = crd_api.delete_namespaced_custom_object(STUDY_JOB_GROUP,
                                                         version,
                                                         namespace,
                                                         STUDY_JOB_PLURAL,
                                                         name,
                                                         body,
                                                         async_req=True)

        api_response = thread.get(TIMEOUT)

        tf.logging.info(("Deleting studyJob %s/%s returned: %s" %
                         (namespace, name, api_response)))

        return api_response

    except rest.ApiException as e:

        message = ""

        if e.message:
            message = e.message

        if e.body:

            try:
                body = json.loads(e.body)

            except ValueError:

                # There was a problem parsing the body of the response as json.
                logging.error(
                    ("Exception when calling DefaultApi->"
                     "apis_fqdn_v1_namespaces_namespace_resource_delete. "
                     "body: %s" % e.body))

                raise

            message = body.get("message")

        tf.logging.error(("Exception when calling DefaultApi->"
                          "apis_fqdn_v1_namespaces_namespace_resource_delete: "
                          "%s" % message))

        raise e


def parameter_configs_from_t2t_rhp(study_rhp):

    parameter_configs = []

    def _maybe_prefix(name):
        #if not name.startswith("--"):
        #name = "--hp_%s" % name
        return name

    def to_str(thing):
        if isinstance(thing, list):
            return [str(p) for p in thing]
        else:
            return str(thing)

    for name, parameter_config in study_rhp._categorical_params.items():
        parameter_configs.append({
            "name": _maybe_prefix(name),
            "parametertype": "categorical",
            "feasible": {
                "list": to_str(parameter_config[1])
            }
        })

    for name, parameter_config in study_rhp._discrete_params.items():
        parameter_configs.append({
            "name": _maybe_prefix(name),
            "parametertype": "discrete",
            "feasible": {
                "list": to_str(parameter_config[1])
            }
        })

    for name, parameter_config in study_rhp._float_params.items():
        parameter_configs.append({
            "name": _maybe_prefix(name),
            "parametertype": "double",
            "feasible": {
                "min": to_str(parameter_config[1]),
                "max": to_str(parameter_config[2])
            }
        })

    for name, parameter_config in study_rhp._int_params.items():
        parameter_configs.append({
            "name": _maybe_prefix(name),
            "parametertype": "int",
            "feasible": {
                "min": to_str(parameter_config[1]),
                "max": to_str(parameter_config[2])
            }
        })

    return parameter_configs


metrics_collector_template_old = """apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: "{{.WorkerID}}"
  namespace: kubeflow
spec:
  schedule: "*/1 * * * *"
  successfulJobsHistoryLimit: 0
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: "{{.WorkerID}}"
            image: gcr.io/kubeflow-ci/katib/tfevent-metrics-collector:v0.1.2-alpha-77-g9324cad
            args:
            - "python"
            - "main.py"
            - "-m"
            - "vizier-core"
            - "-s"
            - "{{.StudyID}}"
            - "-w"
            - "{{.WorkerID}}"
            - "-d"
            - "/train/{{.WorkerID}}"
            volumeMounts:
                - mountPath: "/train"
                  name: "train"
          volumes:
            - name: "train"
              persistentVolumeClaim:
                  claimName: "tfevent-volume"
          restartPolicy: Never
          serviceAccountName: metrics-collector
"""


def get_metrics_collector_template(base_logs_path):
    return """apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: "{{.WorkerID}}"
  namespace: kubeflow
spec:
  schedule: "*/1 * * * *"
  successfulJobsHistoryLimit: 0
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: "{{.WorkerID}}"
            image: gcr.io/clarify/metics-collector:0.0.3
            args:
            - "python"
            - "metrics_collector.py"
            - "--study_id"
            - "{{.StudyID}}"
            - "--worker_id"
            - "{{.WorkerID}}"
            - "--log_dir"
            - "%s/{{.WorkerID}}"
          restartPolicy: Never
          serviceAccountName: metrics-collector
""" % base_logs_path


class KatibBayesianOptParams(object):

    def __init__(self,
                 N="100",
                 model_type="gp",
                 max_features="auto",
                 length_scale="0.5",
                 noise="0.0005",
                 nu="1.5",
                 kernel_type="matern",
                 n_estimators="50",
                 mode="pi",
                 trade_off="0.01",
                 burn_in="10"):
        self.params = {
            "N": N,
            "model_type": model_type,
            "max_features": max_features,
            "length_scale": length_scale,
            "noise": noise,
            "nu": nu,
            "kernel_type": kernel_type,
            "n_estimators": n_estimators,
            "mode": mode,
            "trade_off": trade_off,
            "burn_in": burn_in
        }

    def render(self):

        res = []

        for key, value in self.params.items():
            res.append({"name": key, "value": str(value)})

        return res


class T2TKubeStudy(object):

    def __init__(self,
                 study_name,
                 study_ranged_hparams,
                 experiment,
                 owner="me",
                 optimization_type="maximize",
                 metrics_names=["accuracy"],
                 suggestion_algorithm="bayesianoptimization",
                 suggestion_parameters=KatibBayesianOptParams(),
                 suggestion_request_number=1,
                 api_version=STUDY_JOB_API_VERSION):

        algorithm_to_hparams = {
            "bayesianoptimization": KatibBayesianOptParams,
        }

        if suggestion_algorithm not in algorithm_to_hparams:
            raise ValueError("Unsupported algorithm: %s" % suggestion_algorithm)

        expected_type = algorithm_to_hparams[suggestion_algorithm]

        if not isinstance(suggestion_parameters,
                          algorithm_to_hparams[suggestion_algorithm]):

            raise ValueError(
                ("for algorithm %s expected suggestion_parameters "
                 "arg of type %s" % (suggestion_algorithm, expected_type)))

        # pylint: disable=invalid-name
        self.apiVersion = api_version

        self.kind = STUDY_JOB_KIND

        self.metadata = {
            "name": study_name,
            "labels": {
                "controller-tools.k8s.io": "1.0"
            },
            "namespace": "kubeflow"
        }

        study_rhp = common_hparams.RangedHParams()
        registry.ranged_hparams(study_ranged_hparams)(study_rhp)

        parameter_configs = parameter_configs_from_t2t_rhp(study_rhp)

        job_yaml = yaml.dump(experiment.as_dict(),
                             default_flow_style=False,
                             width=99999)

        mc_template = get_metrics_collector_template(
            experiment._remote_app_root)

        self.spec = {
            "studyName": study_name,
            "owner": owner,
            "optimizationtype": optimization_type,
            "metricsnames": metrics_names,
            "parameterconfigs": parameter_configs,
            "workerSpec": {
                "goTemplate": {
                    "rawTemplate": job_yaml
                }
            },
            "suggestionSpec": {
                "suggestionAlgorithm": suggestion_algorithm,
                "requestNumber": suggestion_request_number,
                "suggestionParameters": suggestion_parameters.render()
            },
            "metricsCollectorSpec": {
                "goTemplate": {
                    "rawTemplate": mc_template
                }
            }
        }

    def as_dict(self):
        return dict_prune_private(object_as_dict(self))

    def create(self):

        kubernetes.config.load_kube_config()

        crd_client = kubernetes.client.CustomObjectsApi()

        job_dict = self.as_dict()

        logging.debug("Running StudyJob with name %s..." %
                      job_dict["metadata"]["name"])

        response = crd_client.create_namespaced_custom_object(
            STUDY_JOB_GROUP,
            STUDY_JOB_VERSION,
            job_dict["metadata"]["namespace"],
            STUDY_JOB_PLURAL,
            body=job_dict)

        return response, job_dict

    def delete(self):

        kubernetes.config.load_kube_config()

        crd_client = kubernetes.client.CustomObjectsApi()

        name = self.metadata["name"]
        namespace = self.metadata["namespace"]

        body = kubernetes.client.V1DeleteOptions()

        response = crd_client.delete_namespaced_custom_object(STUDY_JOB_GROUP,
                                                              STUDY_JOB_VERSION,
                                                              namespace,
                                                              STUDY_JOB_PLURAL,
                                                              name,
                                                              body=body)

        return response
