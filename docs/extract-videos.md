
# Extract videos

Note: This document is a draft.

A walkthrough of the extraction of a collection of raw .mp4 videos into individual frame and audio rows in a Cloud BigTable table.

Here we'll be using Google Cloud PubSub to manage a queue of messages describing the work to be done and Google Cloud Functions to manage the invocation of functions performing this work.

### Disclaimer

Please be advised this work is covered by an Apache 2.0 license (see [LICENSE](https://github.com/projectclarify/pcml/blob/master/LICENSE)) which disclaims warranty for any purpose; among other things you are responsible for anticipating any costs associated with using the software which are this case are non-trivial.

### Preliminaries

You'll need a Google Cloud Platform in which the Cloud Functions, Cloud PubSub, and Cloud BigTable APIs are enabled. You'll also need to create a CBT instance (see https://console.cloud.google.com/bigtable/instances). Lastly you'll need a service account that has permissions to write to Cloud BigTable.

### Deploy

The video extraction function and message queue can be deployed as follows provided a variety of arguments:

```bash

python -m pcml.functions.extract.deploy \
  --project=< your project id > \ 
  --service_account=< your service account > \
  --staging_root=< gcs bucket for staging function bundle > \
  --gcp_region=< a gcp region, e.g. us-central1 >

```

### Trigger

We can enqueue work messages for the `vox_celeb_single_frame` problem (find it [here](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/vox_celeb_cbt.py#L389)) using the following (of course with the possibility of using difference choices for problem and prefix). Here we're limiting to only 100k of the ~1M .mp4 files in the VoxCeleb2 dataset using the `override_max_files_processed` argument because this is a demonstration.

```python

from pcml.functions.extract.publish import trigger_extraction

trigger_extraction(
  problem_name="vox_celeb_single_frame",
  project=< your project name >,
  bigtable_instance=< your cbt instance name >,
  prefix="train",
  # Optional:
  override_max_files_processed=100000)

```

The current design uses the specified prefix to obtain a csv of .mp4 file paths by way of the specified problem's `mode_to_manifest_lookup` method (it has to have one) which in this case simply calls the following:

```python

VOX_CELEB_ROOT = "gs://clarify-data/requires-eula/voxceleb2"

def get_manifest_lookup(vox_celeb_root=VOX_CELEB_ROOT):
  return {
    "train": "{}/dev-paths.txt".format(vox_celeb_root),
    "eval": "{}/test-paths.txt".format(vox_celeb_root),
    "test": "{}/veri-paths.txt".format(vox_celeb_root)
  }

```

Thus you will need to sub-class this particular problem to re-define the `mode_to_manifest_lookup` method to provide manifest paths you can access if you don't have access to these. See also, [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).

The names of raw video and example tables are also tracked at the Problem definition level by way of the following methods:

```python

  @property
  def raw_table_name(self):
    return "vox-celeb-2-raw"

  @property
  def dataset_version_tag(self):
    return None

  @property
  def examples_table_name(self):
    base_name = re.sub("_", "-", self.name)
    if self.dataset_version_tag is not None:
      base_name +=  ("-" + self.dataset_version_tag)
    base_name += "-ex"
    return base_name
  
```

If everything is configured propperly the above described call to `trigger_extraction` should enqueue 1,000 messages describing work to be performed (having the structure specified in [ExtractTriggerMessage](https://github.com/projectclarify/pcml/blob/master/pcml/functions/extract/messages.py)). 

### Monitoring

As your function runs you can view a burndown of the un-acknowledged message queue on the page of the corresponding subscription (Cloud PubSub scriptions can be found here: https://console.cloud.google.com/cloudpubsub/subscription/list) and will look like so:

![](https://github.com/projectclarify/pcml/blob/master/docs/images/ext-cf-num-unacked.png)

If nothing is changed from the default configuration this subscription will be named "gcf-extract-extract-videos-dev". Here we can see the whole queue only takes around 10min to process which isn't shabby considering that's 100,000 mp4 files; which would mean extraction of the whole dataset of 1M videos would take less than 2h.

Cloud PubSub scales the rate at which it pushes messages to push subscriptions by a factor of 2 either up or down each time there is an acknowledgement or non-acknowledgement of a message, respectively. So with some quick math for a function that runs for 100s it takes about 5min to scale up to PubSub pushing enough messages to drive 1k Cloud Function instances.

From the Cloud BigTable monitoring console for the target table (read more [here](https://cloud.google.com/bigtable/docs/monitoring-instance#console-monitoring)) we can see a corresponding scale-up in the number of rows written

![](https://github.com/projectclarify/pcml/blob/master/docs/images/ext-cf-rows-written.png)

And the total storage utilization

![](https://github.com/projectclarify/pcml/blob/master/docs/images/ext-cf-storage-utiliz.png)

### Next steps

Now that we've loaded raw data into a Cloud BigTable table it's time to generate training examples; check out the docs for that next step [here](https://github.com/projectclarify/pcml/blob/master/docs/generate-examples.md).