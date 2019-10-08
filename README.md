![](docs/images/logo.png)

<a href="https://kubeflow.org" target="_blank"><img src="https://img.shields.io/static/v1?label=Built%20with&message=Kubeflow&color=blue"></img></a>

Towards clarifying our signals processing capabilities for various neural- and bio-sensing modalities (including for example de-noising, artifact removal, and feature identification); towards providing a mature foundation for transfer- and co-learning of predictors that provide value to the medical community; towards enabling biofeedback gaming that enhances users' self-awareness and self-regulatory skills regarding cognitive and emotional states (e.g. akin to making it easier to learn mindfulness meditation).

Find an intro to our machine learning methods at <a href="https://ai4hi.org/machine-learning" target="_blank">ai4hi.org/machine-learning</a> and a more technical summary of our current research plan [here](https://github.com/projectclarify/pcml/blob/master/docs/research-plan.md). A legacy version of a whitepaper originating the project can be found [here](https://github.com/projectclarify/experiments/raw/25e5a3e8f7854dc58f54db28cfda99181eb43b9e/public/assets/docs/project_clarify_whitepaper.pdf). Somewhat blue-sky interface design prototypes can be found at <a href="https://ai4hi.org/interface-design" target="_blank">ai4hi.org/interface-design</a>.

The following provides an overview of the computational infrastructure (on the Google Cloud) that enables us to make productive use of large-scale, high-content datasets including feeding Cloud TPUs at the necessary rate.

![](docs/images/infra.png)

Various additional options are available for model deployment that offer lower-latency than the diagrammed option which is the simplest from a research prototyping perspective (enabling cross-platform support for all moderate-latency applications), see also [TFLite](https://www.tensorflow.org/lite), [tfjs](https://www.tensorflow.org/js), and [TFServing](https://www.kubeflow.org/docs/components/serving/tfserving_new/).


### Documentation outline (work in progress):

1. Audio/video correspondence learning
    1. [Extract raw .mp4's to GCS](docs/extract-videos.md)
2. Facial expression perceptual similarity
    1. [Preprocess and generate examples](docs/fec.md)


### Problem implementations

The tensor2tensor Problem object provides a way to encapsulate the steps and parameters involved in processing raw data into training examples (for a particular problem). These can be sub-classed or combined in the context of multi-problem training. More information is available [here](https://tensorflow.github.io/tensor2tensor/new_problem.html). The PCML codebase includes a growing number of Problem implementations in support of various sub-projects, enumerated here:

| Dataset | Visual | Audio | EEG | Annotations | Other modalities | Status | Code | Source | Citation |
|---|---|---|---|---|---|---|---|---|---|
| AffectNet | Image | - | - | Emo. (rater) | - | Planned | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/affectnet.py) | [link](http://mohammadmahoor.com/affectnet/) | 7 |
| DEAP | Video | Audio | 32ch. | Vid. rating | Resp., temp., GSR | Examples | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/deap.py) | [link](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) | 4 |
| FEC | Image | - | - | Percep. sim. | - | Examples | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/fec.py) | [link](https://ai.google/tools/datasets/google-facial-expression/) | 9 |
| MMIMP | Video | Audio | Various | Various | Various | Examples* | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/mmimp.py) | various | - |
| VoxCeleb 2 | Video | Audio | - | N/A | - | Examples | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/vox_celeb_cbt.py) | [link](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 3 |
| DISFA | Image | - | EEG | FACS (rater) | - | Examples | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/disfa.py) | [link](http://mohammadmahoor.com/disfa/) | 6 |
| MAHNOB | Video | Audio | 32ch. | Emo. (self) | Resp., temp., eye | Download | [link](https://github.com/projectclarify/pcml/blob/master/pcml/datasets/mahnob_hci.py) | [link](https://mahnob-db.eu/hci-tagging/) | 5 |


### Citations

This work obviously depends heavily on what has come before it and work that continues in parallel. Each of the datasets above are cited below as well as primary citations for the novel audio/visual correspondence (Arandjelovic and Zisserman, 2017) and triplet perceptual similarity tasks (Vemulapalli and Agarwala, 2019).

1. Abadi, Martín, et al. "Tensorflow: A system for large-scale machine learning." 12th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 16). 2016.
2. Arandjelovic, Relja, and Andrew Zisserman. "Look, listen and learn." Proceedings of the IEEE International Conference on Computer Vision. 2017.
3. Chung, Joon Son, Arsha Nagrani, and Andrew Zisserman. "Voxceleb2: Deep speaker recognition." arXiv preprint arXiv:1806.05622 (2018).
4. Koelstra, Sander, et al. "Deap: A database for emotion analysis; using physiological signals." IEEE transactions on affective computing 3.1 (2011): 18-31.
5. Lichtenauer, Jeroen, and Mohammad, Soleymani. "Mahnob-Hci-Tagging Database." (2011).
6. Mavadati, S. Mohammad, et al. "Disfa: A spontaneous facial action intensity database." IEEE Transactions on Affective Computing 4.2 (2013): 151-160.
7. Mollahosseini, Ali, Behzad Hasani, and Mohammad H. Mahoor. "Affectnet: A database for facial expression, valence, and arousal computing in the wild." IEEE Transactions on Affective Computing 10.1 (2017): 18-31.
8. Vaswani, Ashish, et al. "Tensor2tensor for neural machine translation." arXiv preprint arXiv:1803.07416 (2018).
9. Vemulapalli, Raviteja, and Aseem Agarwala. "A Compact Embedding for Facial Expression Similarity." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

We care very much about citing where appropriate so if you believe a reference has been omitted that should be added please file a GitHub Issue [here](https://github.com/projectclarify/pcml/issues).