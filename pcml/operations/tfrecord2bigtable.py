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
"""A kubernetes Job to load TFRecords into a Cloud BigTable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

from collections import namedtuple

import tensorflow as tf

from google.cloud import bigtable
from google.cloud.bigtable import Client
from google.cloud.bigtable import enums
from google.cloud.bigtable import column_family
from google.cloud.bigtable import row_filters

# For clarity
from google.cloud.bigtable import column_family as cbt_lib_column_family

from google.cloud.bigtable_admin_v2.proto import (
    bigtable_table_admin_pb2 as table_admin_messages_v2_pb2,)
from google.cloud.bigtable.column_family import ColumnFamily
from google.cloud._helpers import _to_bytes

from pcml.launcher.kube import Job, Resources, gen_timestamped_uid

from pcml.launcher.kube import PCMLJob

import hashlib


def create_cbt_cluster(instance_id="clarify-cbt-instance",
                       cluster_id="clarify-cbt-cluster",
                       zone="us-central1-a",
                       serve_nodes=1,
                       storage_type=enums.StorageType.SSD,
                       instance_type=enums.Instance.Type.DEVELOPMENT,
                       labels={}):
  """Create a Google Cloud BigTable cluster."""

  client = Client(admin=True)

  instance = client.instance(instance_id,
                             instance_type=instance_type,
                             labels=labels)

  cluster = instance.cluster(
      cluster_id,
      location_id=zone,
      serve_nodes=serve_nodes,
      default_storage_type=storage_type,
  )

  operation = instance.create(clusters=[cluster])

  result = operation.result(timeout=100)

  return result, operation, cluster, instance


# TODO: Previous version that used tf.Session failed with Segmentation Fault,
# unclear why.
class TFRecordsToCBT(PCMLJob):

  def __init__(self,
               source_glob,
               row_prefix,
               job_name_prefix="cbt-load",
               bigtable_instance="clarify-cbt-instance",
               bigtable_table="clarify-cbt-devtable",
               column="example",
               column_family="tfexample",
               num_parallel_reads=30,
               project="clarify",
               image="gcr.io/clarify/basic-runtime:0.0.4",
               max_records=1234,
               *args,
               **kwargs):
    """Load TFRecords from GCS to a Cloud BigTable table."""

    cmd = "python -m pcml.operations.tfrecord2bigtable "
    cmd += "--project=%s " % project
    cmd += "--source_glob=%s " % source_glob
    cmd += "--bigtable_instance=%s " % bigtable_instance
    cmd += "--bigtable_table=%s " % bigtable_table
    cmd += "--column_family=%s " % column_family
    cmd += "--column=%s " % column
    cmd += "--row_prefix=%s " % row_prefix
    cmd += "--num_parallel_reads=%s " % num_parallel_reads
    cmd += "--max_records=%s " % max_records

    command = ["/bin/sh", "-c"]
    command_args = [cmd]

    job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())

    super(TFRecordsToCBT, self).__init__(
        job_name=job_name,
        command=command,
        command_args=command_args,
        namespace="kubeflow",
        image=image,
        resources=Resources(limits={"cpu": num_parallel_reads}),
        *args,
        **kwargs)


def iterable_dataset_from_file(filename):
  dataset = tf.data.TFRecordDataset(filename)

  iterator = dataset.make_initializable_iterator()

  next_element = iterator.get_next()

  with tf.Session() as sess:

    sess.run(iterator.initializer)

    i = 0
    while True:
      try:
        if i % 1000 == 0:
          print("Processed %s examples..." % i)
        yield sess.run(next_element)
        i += 1
      except tf.errors.OutOfRangeError:
        print("Ran out of examples (processed %s), exiting..." % i)
        break


BigTableSelection = namedtuple('BigTableSelection', [
    'project',
    'instance',
    'table',
    'prefix',
    'column_family',
    'column_qualifier',
])


def materialize_bigtable_selection(selection, sa_key_path=None):

  if isinstance(sa_key_path, str):
    client = bigtable.Client.from_service_account_json(sa_key_path, admin=True)
  else:
    client = bigtable.Client(admin=True)

  instance_name = selection.instance
  instance = client.instance(instance_name)
  table_name = selection.table
  table = instance.table(table_name)

  while True:

    if table.exists():
      return (client, table, instance)

    # TODO: This will exceed the deadline for table creation if the table
    # does not already exist. But table.create does not accept a timeout
    # argument or kwargs to allow a timeout parameter to be passed to
    # table_client.create_table (which does).

    max_versions_rule = cbt_lib_column_family.MaxVersionsGCRule(2)
    column_family_id = selection.column_family
    column_families = {column_family_id: max_versions_rule}
    initial_split_keys = []
    table.create(column_families=column_families)

  return (client, table, instance)


def make_prefixed_idx(i, row_prefix, max_records):
  max_width = len('%d' % (max_records - 1))
  current_width = len(str(i))
  pad = "".join(str(thing) for thing in [0] * (max_width - current_width))
  idx = "_".join([row_prefix, "".join([str(i), pad])]).encode('utf-8')

  return idx


def tfrecord_files_to_cbt_table(glob,
                                table,
                                selection,
                                max_records=100000000,
                                mutation_batch_size=32):
  """
  
  Notes:
  * Mutate in limited-size batches to avoid a deadline-exceeded error.
  
  """

  mutation_index = 0

  def new_mutation_batch():
    return [None for _ in range(mutation_batch_size)]

  files = tf.gfile.Glob(glob)

  for file_path in files:

    row_mutation_batch = new_mutation_batch()

    for i, example in enumerate(iterable_dataset_from_file(file_path)):

      idx = hashlib.md5(example).hexdigest()

      # DEV: To check "shuffle" effect add the id suffix
      idx = "_".join([selection.prefix, idx, str(i)])

      row = table.row(idx)
      row.set_cell(column_family_id=selection.column_family,
                   column=selection.column_qualifier,
                   value=example,
                   timestamp=datetime.datetime.utcnow())

      row_mutation_batch[mutation_index] = row

      if mutation_index == (mutation_batch_size - 1):
        table.mutate_rows(row_mutation_batch)
        row_mutation_batch = new_mutation_batch()
        mutation_index = 0
      else:
        mutation_index += 1

    final_mutation = row_mutation_batch[:(mutation_index - 1)]
    if final_mutation:
      table.mutate_rows(final_mutation)


def iterate_cbt_examples(project="clarify",
                         instance="clarify-cbt-instance",
                         table="clarify-cbt-devtable",
                         prefix="cbt-load",
                         column_family="tfexample",
                         column_qualifier="example",
                         column_family_id='tfexample'):

  selection = BigTableSelection(project=project,
                                instance=instance,
                                table=table,
                                prefix=prefix,
                                column_family=column_family,
                                column_qualifier=column_qualifier)

  (client, table, instance) = materialize_bigtable_selection(selection)

  column = column_qualifier.encode()

  row_filter = row_filters.CellsColumnLimitFilter(1)
  partial_rows = table.read_rows(filter_=row_filter)

  for row in partial_rows:
    cell = row.cells[column_family_id][column][0]
    value = cell.value
    yield value


def main(_):

  if FLAGS.source_glob is None:
    raise ValueError("Please provide a source glob with --source_glob.")

  if FLAGS.bigtable_instance is None:
    raise ValueError(
        "Please provide a BigTable instance with --bigtable_instance.")

  if FLAGS.bigtable_table is None:
    raise ValueError("Please provide a BigTable table with --bigtable_table.")

  if FLAGS.project is None:
    raise ValueError("Please provide a project with --project.")

  if FLAGS.row_prefix is None:
    raise ValueError("Please provide a row prefix with --row_prefix.")

  selection = BigTableSelection(project=FLAGS.project,
                                instance=FLAGS.bigtable_instance,
                                table=FLAGS.bigtable_table,
                                prefix=FLAGS.row_prefix,
                                column_family=FLAGS.column_family,
                                column_qualifier=FLAGS.column)

  (client, table, instance) = materialize_bigtable_selection(selection)

  tfrecord_files_to_cbt_table(glob=FLAGS.source_glob,
                              table=table,
                              selection=selection,
                              max_records=FLAGS.max_records)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string('op', "LoadTFRecords", 'The PCML operation to perform.')
  flags.DEFINE_string(
      'source_glob', None, 'The source TFRecord files to read '
      'from and push into Cloud Bigtable.')
  flags.DEFINE_string('bigtable_instance', None, 'The Cloud Bigtable instance.')
  flags.DEFINE_string('bigtable_table', None,
                      'The table within the instance to '
                      'write to.')
  flags.DEFINE_string(
      'project', None, 'The Project to use. (Optional if running '
      'on a Compute Engine VM, as it can be auto-determined from '
      'the metadata service.)')
  flags.DEFINE_integer(
      'max_records', 100000000,
      'The approximate dataset size (used for padding '
      'the appropriate number of zeros when constructing row keys). It should '
      'not be smaller than the actual number of records.')
  flags.DEFINE_integer(
      'num_parallel_reads', 1, 'The number of parallel reads '
      'from the source file system.')
  flags.DEFINE_string('column_family', 'tfexample',
                      'The column family to write the data into.')
  flags.DEFINE_string('column', 'example',
                      'The column name (qualifier) to write the data into.')
  flags.DEFINE_string('row_prefix', 'train_', 'A prefix for each row key.')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
