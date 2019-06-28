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

"""Metrics collector for GCS, forked from Katib codebase."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import sys
from datetime import datetime
import rfc3339
import grpc
#import api_pb2
#import api_pb2_grpc


# VERY HACK =======================
import katib_api_pb2 as api_pb2
# =================================

"""
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("manager_addr", "vizier-core",
                    "The address of the Vizier core manager service.")

flags.DEFINE_integer("manager_port", 6789,
                     "The manager port with which to communicate.")

flags.DEFINE_string("study_id", None,
                    "The ID of the study for which to report metrics.")

flags.DEFINE_string("worker_id", None,
                    "The ID of the study worker for which to report metrics.")

flags.DEFINE_string("log_dir", None,
                    "The directory to search for events.")

flags.DEFINE_string("verbosity", "INFO",
                    "Logging verbosity, must be in ['INFO', 'DEBUG'].")

"""

class TFEventFileParser(object):
    def find_all_files(self, directory):

        for a,b,c in tf.gfile.Walk(directory):
          for filename in c:
            filename = str(filename)
            if filename.startswith("events.out.tfevents"):
              yield (os.path.join(str(a), filename))

    def parse_summary(self, tfefile, metrics):
        metrics_log = {}
        for m in metrics:
            if len(m) == 0:
              continue
            metrics_log[m] = api_pb2.MetricsLog(name=m, values=[])
        for summary in tf.train.summary_iterator(tfefile):
            for v in summary.summary.value:
                for m in metrics:
                    if len(m) == 0:
                        continue
                    if str(v.tag) == m:
                        mv = metrics_log[m].values.add()
                        mv.time=rfc3339.rfc3339(datetime.fromtimestamp(summary.wall_time))
                        mv.value=str(v.simple_value)
        return metrics_log

    
class MetricsCollector(object):

    def __init__(self, study_id, worker_id, log_dir,
                 manager_addr="vizier-core",
                 manager_port=6789):

        self.manager_port = int(manager_port)
        self.manager_addr = manager_addr
        self.study_id  = study_id
        self.worker_id = worker_id
        self.log_dir = log_dir

        channel = grpc.beta.implementations.insecure_channel(self.manager_addr,
                                                             self.manager_port)

        with api_pb2.beta_create_Manager_stub(channel) as client:
            gsrep = client.GetStudy(api_pb2.GetStudyRequest(study_id=self.study_id), 10)
            self.metrics = gsrep.study_config.metrics
        self.parser = TFEventFileParser()

    def parse_file(self, directory):
        mls = []
        for f in self.parser.find_all_files(directory):
            if os.path.isdir(f):
                continue
            try:
                tf.logging.info(f+" will be parsed.")
                ml = self.parser.parse_summary(f, self.metrics)
                for m in ml:
                    mls.append(ml[m])
            except:
                tf.logging.warning("Unexpected error:"+ str(sys.exc_info()[0]))
                continue
        return mls

    def report(self, mlset):

        channel = grpc.beta.implementations.insecure_channel(self.manager_addr,
                                                             self.manager_port)

        with api_pb2.beta_create_Manager_stub(channel) as client:
          tf.logging.info("In " + mlset.worker_id + " " + str(len(mlset.metrics_logs)) + " metrics will be reported.")

        client.ReportMetricsLogs(api_pb2.ReportMetricsLogsRequest(
            study_id=self.study_id,
            metrics_log_sets=[mlset]
            ), 10)

    def run(self, report=True):

        mlset = api_pb2.MetricsLogSet(worker_id=self.worker_id, metrics_logs=[])

        mls = self.parse_file(self.log_dir)

        for ml in mls:
          mla = mlset.metrics_logs.add()
          mla.name = ml.name
          for v in ml.values:
            va = mla.values.add()
            va.time = v.time
            va.value = v.value
            
        if report:
          self.report(mlset)
        
        return mlset


def main(_):
    
  verbosities = ["INFO", "DEBUG"]
  if not FLAGS.verbosity in verbosities:
    raise ValueError("Allowed logging verbosities %s, saw %s" % (
      verbosities, FLAGS.verbosity
    ))

  tf.logging.set_verbosity(getattr(tf.logging, FLAGS.verbosity))

  mc = MetricsCollector(manager_addr=FLAGS.manager_addr,
                        manager_port=FLAGS.manager_port,
                        study_id=FLAGS.study_id,
                        worker_id=FLAGS.worker_id,
                        log_dir=FLAGS.log_dir)

  mc.run()


if __name__ == "__main__":
  tf.app.run()
