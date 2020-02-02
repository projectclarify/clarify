
# hc 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/projectclarify/clarify/blob/master/hc/docs.ipynb) [![Go Report Card](https://goreportcard.com/badge/github.com/projectclarify/clarify)](https://goreportcard.com/report/github.com/projectclarify/clarify)

Components for training models with high-content data (most notably HD-(f)MRI and HD cortical sensing).

Please refer to the Colab notebook linked above for additional design, usage, and demo information.

## Building & Testing

To build and test everything under the project using Bazel, run:

```bash
bazel test //hc/...
```

To update BUILD file rules, run:

```bash
bazel run //hc:gazelle -- update-repos
```

## Running the sampler locally

To run the sampler with a 

```bash
# TODO

```
