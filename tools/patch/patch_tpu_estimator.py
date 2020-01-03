import os
import tempfile
import tensorflow as tf
import tensorflow_estimator

"""

HACK: in the exploration of 
https://github.com/tensorflow/tensorflow/issues/30869

"""

def _get_tpu_estimator_path():
  root = "/".join(tensorflow_estimator.__file__.split("/")[:-1])
  return os.path.join(root, "python/estimator/tpu/tpu_estimator.py")

_patch = """
def _check_add_preemption_hook(cluster):
  return (tpu_cluster_resolver.is_running_in_gce() and
          cluster and
          isinstance(cluster, tpu_cluster_resolver.TPUClusterResolver) and
          cluster._should_resolve)
"""

def _patch_tpu_estimator():

  tmpdir = tempfile.mkdtemp()
  tpu_estimator = _get_tpu_estimator_path()
  tpu_estimator_tmp = os.path.join(tmpdir, "tmp")

  replace_target = "if tpu_cluster_resolver.is_running_in_gce():"
  replacement = "          if _check_add_preemption_hook(self._config.cluster):\n"

  with tf.gfile.Open(tpu_estimator_tmp, "w") as target_file:
    with tf.gfile.Open(tpu_estimator) as source_file:
        for line in source_file:
            if replace_target in line:
              line = replacement
            target_file.write(line)

    target_file.write(_patch)

  tf.gfile.Copy(tpu_estimator_tmp, tpu_estimator, overwrite=True)


if __name__ == "__main__":
  _patch_tpu_estimator()
