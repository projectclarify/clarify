# Project Clarify

[![Travis](https://img.shields.io/travis/projectclarify/pcml.svg)](https://travis-ci.org/projectclarify/pcml)

Towards clarifying our signals processing capabilities for various neural- and bio-sensing modalities (including for example de-noising, artifact removal, and feature identification); towards providing a mature foundation for transfer- and co-learning of predictors that provide substantial value to the medical community; towards enabling biofeedback gaming that enhances users' self-awareness and self-regulatory skills regarding cognitive and emotional states (e.g. akin to making it easier to learn mindfulness meditation).

### Problems / data generators

Below we summarize the various data generators and models being developed herein, all defined within the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) paradigms of [T2TModel](https://github.com/tensorflow/tensor2tensor/blob/master/docs/new_model.md), [T2TProblem](https://github.com/tensorflow/tensor2tensor/blob/master/docs/new_problem.md), and [MultiProblemV2](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem_v2.py#L67).

| Dataset | Modalities | Example problem | Status | Code link |
|---|---|---|---|---|
| [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Video, audio | Learn embeddings that are similar for co-occurring signals | Draft | [here](https://github.com/projectclarify/pcml/blob/master/pcml/data_generators/vox_celeb_cbt.py) 

### Models

| Model | Description | Status | Code link |
|---|---|---|---|
| modality_correspondence_learner | Learn to represent modalities that co-occur, in the style of [1]. | Draft | [here](https://github.com/projectclarify/pcml/blob/master/pcml/models/modality_correspondence.py) |

[1] Afros et al. "Deep Audio-Visual Speech Recognition." arXiv preprint arXiv:1409.1556 (2018).

### Citations and Acknowledgments

This project depends on several others, some of which are citeable; if you make use of this work please take a moment to review our [citations and acknowledgements](docs/acknowledgments.md) and take the opportunity to express your own appreciation.
