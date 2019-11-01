
# FEC
#### Part 1: Datagen

NOTE: This document is a draft.

Here is documented the steps for obtaining and training on the Facial Expression Correspondence dataset described in Vermulapalli and Agarwala (2019; 1), see also [2].

1. Vemulapalli, Raviteja, and Aseem Agarwala. "A Compact Embedding for Facial Expression Similarity." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
2. https://ai.google/tools/datasets/google-facial-expression/

## Parallel download and pre-filtering

The FEC dataset is distributed as a large CSV file indicating the URLs of images available on the public internet and annotations assigned to triplets of those by human annotators. That is, the primary dataset distribution does not include any images - these must be obtained from their URLs. Naturally not all images will still be obtainable or contain the same content as they did at the time they were used for annotation. For this reason the download step also involves filtering both for successful downloads as well as for images that contain faces within the expected regions. Images containing faces in the expected places are then cropped to these regions (+15%). The presence of faces is determined via DNN face detection via the Faced (https://github.com/gordalina/faced) library.

#### Download job container

First you will need to build a container that encapsulates the necessary code for the download job. To do so you will need to specify a GCP project.

```python

from pcml.utils.gcb_utils import gcb_build_and_push
from pcml.utils.gcb_utils import generate_image_tag

from pcml.utils.fs_utils import get_pcml_root

project = "<your GCP project name>"

image_tag = generate_image_tag(project, "download_fec")

gcb_build_and_push(image_tag=image_tag,
                   build_dir=get_pcml_root(),
                   cache_from=None)

```

The above call will block until completion. You may monitor logs and status for the build in the Container Builder section of the GCP console.

#### Launching the job

To launch the job you will also need to specify bucket(s) to which to (1) stage job artifacts, and (2) write the downloaded image data and metadata. The method launch_shard_parallel_jobs takes two arguments:

1. num_shards: This specifies how many parts into which to divide the total list of units of work, and
2. max_num_jobs: The number of these to actually launch. If unset, all of them.

To confirm everything is working correctly you may first want to set the `num_shards` to a high number and the `max_num_jobs` to 1 and verify the job runs without error by inspecting Pod logs or by running `python -m pcml.operations.download_fec_test` which will effectively do the same.

To download the full dataset, simply do not specify `max_num_jobs` when calling launch_shard_parallel_jobs, below.

Please be advised that running download jobs at high multiplicity could disrupt the service of domains representing a high fraction of the source dataset. You are responsible for anticipating and mitigating such possibilities and the software is released with a disclaimer of warranty and liability including fitness for any purpose including this one, see also [LICENSE](https://github.com/projectclarify/pcml/blob/master/LICENSE). Multiplicity can be limited by setting `num_shards` to a low number or by placing jobs in a node pool (by specifying `node_selector`) that is limited in its maximum size.

```python

from pcml.operations.download_fec import DownloadFec

staging_bucket = "<GCS bucket for staging>"
output_bucket = "<GCS bucket for output>"


# Download images for the train split
job = DownloadFec(output_bucket=output_bucket,
                  is_training=1,
                  staging_path=staging_bucket,
                  image=image_tag,
                  node_selector={"type": "datagen-small-preemptible"})

job.launch_shard_parallel_jobs(num_shards=2000)


# Download images for the eval split
job = DownloadFec(output_bucket=output_bucket,
                  is_training=0,
                  staging_path=staging_bucket,
                  image=image_tag,
                  node_selector={"type": "datagen-small-preemptible"})

job.launch_shard_parallel_jobs(num_shards=200)


```

Where in the above the `node_selector` argument specifies a label that in turn specifies which pool of nodes in the GKE cluster should run the job; this notable being one that has permission to write to the specified GCS bucket.

This will produce output in the specified `output_bucket` with the following approximate structure:

```bash

output_bucket_root/
  train/
    0/
      cropped@a-b-c-d#original-url.jpg
      cropped@a-b-c-d#original-url.jpg
      ...
    1/
    ...
  eval/
    0/
    1/
    ...

```

## Generating examples

In all cases below, examples are generated according to a Tensor2Tensor Problem definition that has been registered within the PCML source tree. You can read more about it [in the main t2t readme](github.com/tensorflow/tensor2tensor) or in [the specific Problem documentation](https://tensorflow.github.io/tensor2tensor/new_problem.html).

#### Generating examples locally

Tests are included in the codebase that will verify the example generation code along with its congruence with model code. It's a good idea to run these tests before launching a long-running datagen job. This can be accomplished via `python -m pcml.datasets.fec_test` or via the following. Note that for local testing we us a subsampled version of the full FEC problem, `fec_tiny`, and a small version of the full model, `percep_similarity_triplet_emb_tiny`.

```python

from pcml.utils.dev_utils import T2TDevHelper

helper = T2TDevHelper(
  problem="fec_tiny",
  model="percep_similarity_triplet_emb",
  hparams_set="percep_similarity_triplet_emb_tiny",
  data_dir="/tmp/fec",
  tmp_dir="/tmp/fec"
)

helper.datagen()

helper.eager_train_one_step()

```

#### Generating examples in batch

Let's launch a batch job to generate a large number of examples (having verified the aforementioned).

##### Building the job container

If nothing relevant has changed in your `pcml` source tree since the previous build you can skip this step. Otherwise you'll want to build another container version as described above and substitute that as the `image_tag` in the following step.

##### Launch the datagen job

We can construct the job for FEC example generation in much the same way as that in the previous step except that the job is singular instead of sharded and requires the specification of a PCML-registered T2T Problem name (see also https://tensorflow.github.io/tensor2tensor/new_problem.html) and a GCS bucket to which to stage the resulting TFRecords.

Note the problem specified here is the full version of the FEC problem whereas in debug steps described above the 

```python

from pcml.operations.t2t_datagen import T2TDatagenJob

problem_name = "fec_base"
data_dir = "<GCS bucket to which to stage resulting TFRecord's>"

job = T2TDatagenJob(problem_name=problem_name,
                    data_dir=data_dir,
                    image=image_tag,
                    stage_out_tfrecords=True)

job.batch_run()

```
