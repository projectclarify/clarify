### Bringing AI to Bear on the Enhancement of Human Cognitive & Emotional Skills Through Predictive Modeling of User Mental States

#### Christopher W. Beitel, Ph.D.
##### Director of Machine Learning Research (Neuroscape)
##### Department of Neurology
##### University of California, San Francisco


### I. Summary

Can you tell by looking at the face of someone you know well how focused, clear, happy, calm, and generally productive they are at that moment? People vary in their ability to make these assessments and individuals vary in terms of what they exhibit. Variation also exists across cultures, contexts, and numerous other dimensions. Facial expressions are nevertheless important ways our cognitive and emotional experience is communicated.

Extensive research in the domains of contemplative neuroscience, neurofeedback training, and cognitive computer gaming has established our ability to make the mind more effective across a wide range of tasks - in all cases as a consequence of the product of practice and plasticity. But again in all cases, our ability to deliver the optimal task is limited by our lack of awareness of the user's cognitive and emotional state.

Here we seek to build systems that can aid in improving both users metacognitive awareness and control over their mental and emotional states.

Succeeding with such an endeavor is not only a problem of machine learning research (which we focus on in this proposal) but one of delivering a functional, reliable system based on this into production suitable to enable downstream usage (that will then specialize according to application, device/platform, etc.).

### II. Strategy

#### Background

Deep learning can effectively be simplified as being a process of learning the parameters of black boxes that can be used to map from inputs to outputs - in the process automating the identification of task-relevant features as well as automating the modeling of spatial, temporal, semantic, or other structure of the task domain. Here we seek to be able to take in perceptions of users and output representations of their state.

Early successes in the history of deep learning primarily involved labeled data. In contrast to this there are approaches that seek to learn from non-classically-labeled data - being so perhaps for reasons of economic or perceptual feasibility. Several  promising deep learning methods for deriving value from unlabeled data exist but here we focus our attention on the most straightforward approach - supervision by modality co-occurrence.

Modality co-occurrence methods stand to enable the automated modeling of domains wherein two or more co-occurring data streams are available. Understanding early work on learning from the correspondence of audio and visual streams extracted from the same videos is a useful analogy for understanding the more general paradigm (including how that can be applied in the context of this project). The earliest instance of this was in 2017 when Arandjelovic and Zisserman first introduced the audio-visual correspondence (AVC) self-supervision task wherein a network is trained to predict whether a provided pair of audio and video sequences were sampled from overlapping regions of the same video. They found this paradigm enabled the establishment of a new state-of-the-art on two sound classification benchmarks and further enabled the networks to localize objects within the observed modalities - without any object labeling having been provided at training time. The intuition behind this is that, for example, the object nature of an element of a visual scene is cross-indicated by its potential to be a source of sound. And in the same sense, objects in the audio modality are “visually indicated” by their co-occurring with the visual one. In short, hearing the sound of a guitar across numerous contexts unified only by the presence of guitar-shaped sptaio-temporal objects is an indicator of the objectness of something (in this case, an unlabeled notion of guitar).

Preceding the aforementioned, various work demonstrated the potential of using the same modality and embedding correspondence loss paradigm to transfer the learnings of one network (perhaps pre-trained on labeled data) to an un-trained network operating on a co-occurring modality. Especially relevant (but not proceeding) was the work of Albanie et al. (2018) who used a network pre-trained to predict emotions from facial video frame sequences (from the VoxCeleb dataset) to supervise the learning of a model for predicting the same emotion labels from co-occurring speech audio - in so doing transferring the value of the labels that were available in the visual domain to the audio domain. Our task here is to use this simple notion of co-occurrence as a form of supervision to aid in the learning of features where labels are harder to obtain such as in the labeling of nuanced emotional and cognitive states from the potentially richly elucidative modality of EEG by way of first doing so within the easier context of co-occurring video and audio.

#### Objectives and Key results

BHAG: Significantly enhance the effectiveness of both vocational and educational systems by building ML-enabled feedback systems that can sense and cue improvement in (1) user’s cognitive and affective states and (2) user’s awareness of and ability to self-regulate these without the assistance of cues.

Y1AG1: Develop the capability to build a user-optimizing system.
* Model provides improved predictive power for initially-trained tasks (cognitive effectiveness and emotional state), relative to baseline.
* Model provides an improvement in signal processing (e.g. denoising/artifact removal, automating source localization), as measured in part by the ability to reconstruct artificially ablated and distorted signals.
* Learned model provides a mature foundation for transfer learning as measured by improved performance when used to train predictor for initially untrained task.

Y1CG1: Publish a paper describing work.
