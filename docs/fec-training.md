
# FEC

#### Part 2: Training

## Training in batch

In the same way as elsewhere in the codebase a batch training run can be launched as follows. Here we specify the full problem, hparams set, in addition to the model as named above.

```python

from pcml.launcher.experiment import configure_experiment

problem_name = "fec_base"
model_name = "percep_similarity_triplet_emb"
hparams_set_name = "percep_similarity_triplet_emb"
data_dir = "gs://clarify-models-us-central1/data/fec" # Modify

experiment = configure_experiment(
    base_name="fec-train",
    problem=problem_name,
    model=model_name,
    hparams_set=hparams_set_name,
    num_train_steps=3000,
    base_image=image_tag,
)

create_response, _ = experiment.batch_run()

```

Launching a job in this way requires that your GKE cluster be configured in a variety of ways including having TPU support as well as a node pool with the label `tpu-host` capable of satisfying the specified CPU and memory requests. Please refer to the Kubeflow and GCP documentation for details on configuring your cluster in this way.