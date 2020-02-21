/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { css, html, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store, RootState } from '../store.js';

import { mouseIcon } from './my-icons.js';

import '/node_modules/@polymer/paper-button/paper-button.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

import { conventionalLabeling, embeddingSpace, knn, sslArchitecture, percepSimilarity } from './ml-images.js';

import { ViewArticleBase } from './view-article-base.js';


@customElement('view-article-machine-learning')
export class ViewArticleMachineLearning extends connect(store)(ViewArticleBase) {

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`
      
        .article-large-width-image > svg {
          margin: 10px;
          flex-shrink: 0;
          width: 80%;
          height: auto;
        }

        .section-spacer {
            height: 20px;
        }

        .article-hero-heading { 
            font-size: 40px;
        }
 
       .article-large-width-image > img {
            flex-shrink: 0;
            width: 50%;
            height: auto;
        }

        .article-section > p {
            font-size: 14px;
        }

     `
    ];
  }

  protected render() {
    return html`

      <section class="article-hero">
        <p class="article-hero-pre-heading">ARTICLE</p>
        <p class="article-hero-heading">State understanding (when labeling is hard)</p>
        <p class="article-hero-subheading">Learn about how we are using deep learning to build next-generation training experiences.</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">Scroll to read more</p>
          ${mouseIcon}
        </div>
      </section>

      <section class="article-section scrollspy-target" id="overview">
        <p class="article-section-pre-title">Overview</p>
        <h2 class="article-section-title">An overview of our program</h2>
        <p>Professional athletes have professional trainers. But what about professional knowledge workers?</p>
        <p>Project Clarify is based on the premise that it would be useful for a neuro- or bio-feedback system or game to have an appreciation of the emotional, cognitive, and/or physiological state of its user so as to better deliver this feedback.</p>
        <p>Here we discuss the machine learning methods we are pursuing to derive such capability, initially with a focus on the video modality (here a “modality” being a channel of perceptual information such as sight and sound but also by extension EEG, fMRI, and other channels of neuro- and bio-sensing information).</p>
        <p>The methods we discuss here are those designed to achieve the task in a financially-feasible manner - i.e. without the need to spend hundreds of thousands or millions of dollars on data annotation. We argue the conjunction of self-supervised learning, small-scale high-quality annotation, and nearest-neighbors search provides this opportunity. In this regard we discuss self-supervised methods that have been shown to learn high-quality representations without the supervision of human subjective annotations. We further argue that even were it to be feasible to exhaustively label all states of interest across all (global) demographic niches - human semantic labels may not optimally represent a more nuanced notion of human perceptual similarity.</p>
        <p>Lastly we’ll discuss in detail how the capability of being able to sense a user’s emotional and/or cognitive state would provide significant new value in the context of classic neurofeedback, adaptive neurofeedback gaming, and more generally in the clinical cognitive neurosciences.</p>
      </section>

      <section class="article-section scrollspy-target" id="label-issues">
        <p class="article-section-pre-title">Labeling</p>
        <h2 class="article-section-title">Good labels are hard to come by</h2>
        <p>Classic supervised machine learning methods involve training a model using a dataset of labeled data items to predict correct labels from unlabeled query data. For example, given a dataset of flower images labeled with flower type we might train a model to predict those flower type labels from the images. Then to evaluate the model we could use labeled images our model has never seen before, obtain its predictions for these, and compare those predictions to the reference labels.</p>
        <p>Obtaining a new labeled image or video dataset using crowd-sourced labor can be a very expensive proposition relative to the budgetary scale of an academic research group. For example, Google’s Data Labeling Service charges $35 per 1,000 (or $35,000 per 1M) images for a cloud worker to assign class labels to the images. A similar task through Amazon’s SageMaker labeling service can be priced by the requestor but suggested pricing is still high at $12,000 per 1M labeled images. Video classification on Google Cloud is priced at $86,000 per 1M 5s videos.</p>
        <p>While the above costs are high one might argue they are not untenable given what value might correspond. An important point to make is that in the abstract the price of a data labeler’s time should approximately correspond to the *quality* of the labels they assign and this quality is fundamentally important as it is what models are learning to emulate. As a comparison perhaps we would like to be able to hire a world’s expert in expression recognition (who we might reasonably expect to bill around $150/hr) to label images or video clips according to ten emotion classes (according to extent as opposed to binary labels). If it took them only two seconds per class it would cost $833,000 to label 1M images and over $2M to do so for 30s video clips.</p>
        <p>There are also circumstances where human labelers cannot feasibly assign accurate labels (independent of expense) such as when the to-be-labeled phenomenon is not perceivable by humans. For example, humans have not evolved to perceive EEG or fMRI signals; we cannot crowd-source labels for these modalities because when labelers are shown such data they can’t just look at the data and indicate the person’s emotional or cognitive state (including clinically relevant labels of disease progression or those that could be used for neurofeedback). Furthermore, even when a modality is perceivable one might wonder if the assignment of a limited set of language labels is a restrictive simplification of the complexity of the labeled phenomenon (in addition to being error-prone as discussed above).</p>
        <p>In the following sections we’ll discuss ways to address the label cost efficiency and obtainability problems starting from a discussion of conventional label prediction.</p>

        <div class="section-spacer"></div>

        <p class="article-section-pre-title">Label prediction</p>
        <h2 class="article-section-title">Conventional label prediction</h2>
        <p>When a large amount of conventionally-labeled images or video are available all we should need is a conventionally-architected convolutional neural network and a loss that simply compares the prediction to the ground truth (i.e. where “truth” is the human-assigned label).</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${conventionalLabeling}
            </div>
        </div>

        <p>Convolutional neural networks are composed of a series of layers that learn a hierarchy of progressively more complex composite features that can be used to produce (or “encode”, blue above) an embedding vector that represents the input relative to other labeled examples in an high-dimensional embedding space. Labels can be “decoded” (above, purple) with additional neural network layers that map points from the embedding space into label space (e.g. the real number interval [0, 1] for a single continuous attribute label).</p>
        <p>We’ll discuss in subsequent sections how we can make use of the structure of the embedding space itself to help us deal with the various label-related problems discussed above (notably cost efficiency and obtainability).</p>
      </section>

      <section class="article-section scrollspy-target" id="repr-space">
        <p class="article-section-pre-title">Representation</p>
        <h2 class="article-section-title">A toy example of a representation space</h2>
        <p>For the sake of discussion let’s discuss a toy example of a representation space. We’ll need an intuitive understanding of this in order to understand the discussion in subsequent sections. Suppose we are given a dataset of emojis and a label that specifies whether the pictured is “happy” or “not happy”. Here’s our dataset:</p>
        <p>{(😎, 1), (😄, 1), (😊, 1), (🤗, 1), (🙋, 1), (🤭, 1), (😒, 0), (😟, 0), (😞, 0), (😓, 0), (😢, 0)}</p>
        <p>Let’s further imagine that in the course of training a model our encoder learns to map from (emoji) inputs into a 2-dimensional embedding space (i.e. simply consisting of elements (p, q)) and that this cleanly maps inputs of the two types to disjoint neighborhoods in the embedding space. Pictured below as (i) and (ii):</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${embeddingSpace}            
            </div>
        </div>
        
        <p>In this example we are imagining inputs like 😄 and 🤗 are mapped by our encoder to positions like <0.2, 0.9> and <0.3, 0.8>, respectively, whereas inputs like 😒 and 😓 produce embeddings like <0.9, 0.3> and <0.85, 0.2>, respectively. We can equip our planar embedding space with a notion of distance - i.e. choose a formula for taking two points and computing how far apart they are. For example, for the sake of simplicity, we might choose to use the classic <a href="https://en.wikipedia.org/wiki/Euclidean_distance" target="_blank">Euclidean distance metric</a> that is a measure of the combined difference along each dimension.</p>

        <div class="section-spacer"></div>

        <p class="article-section-pre-title">Reverse lookup</p>
        <h2 class="article-section-title">Using kNN to obtain predictions</h2>
        <p>Let’s understand how we can use an encoder and embedding space like the one in our toy example above together with a *small* amount of high-quality labeled data to accomplish something similar to what we can with a large amount of labeled data. Then in subsequent sections we’ll talk about how to learn embedding spaces without human-assigned labels.</p>
        <p>In the following figure we can see a number of inputs mapped into embedding space - some of which were unlabeled inputs and some that were labeled. Here let’s imagine we mapped a large number of inputs into embedding space (circles) then sampled a subset of these to run though our labeling pipeline given some algorithm for spanning the structure of the space, resulting in a subset of (position, label) pairs (colored).</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${knn}            
            </div>
        </div>

        <p>Next, let’s imagine we have a new input that is mapped into the space (labeled “query” above). Given some neighborhood of this point (in terms of some chosen metric of distance in the space, such as the Euclidean distance metric) we can apply one of a variety of algorithms for looking up all (position, label) pairs that fall within such distance of the query point - these are our “results”. Given this set, a label can be assigned to the query point according to some consensus algorithm applied to the label data present in the result set. For example, if the label data in the query neighborhood were {1, 1, 1} and our consensus procedure were to average the results, our resulting label for the query would be 1. This and related methods are referred to loosely as “nearest neighbor methods”.</p>
        <p>The predictions of this method can be validated by passing *labeled* data points through the procedure of embedding followed by nearest neighbor consensus as if they were unlabeled queries and comparing the resulting predicted label with the “known” (or at least expert-assigned) label. Notably these being data points that have not previously been used in the process of training our embedding network.</p>
        <p>Thus to summarize, assuming we have learned an encoder network that maps inputs into a space where closeness in that space corresponds to similarity of label - we then only need to label as many points as necessary to span the space up to a desired level of density in regions of interest.</p>
      </section>

      <section class="article-section scrollspy-target" id="repr-learning">
        <p class="article-section-pre-title">Self-supervision</p>
        <h2 class="article-section-title">Self-supervised learning</h2>
        <p>Next let’s discuss some methods for learning encoder networks that can produce embedding spaces of the desired type without the need for human-assigned labels. These methods will involve making use of a property of the data itself to obtain a label (i.e. a signal to drive supervised learning) - hence the label “self-supervised learning”.</p>
        <p>For example, a labeled dataset of between- or within-modality correspondence could be continuously sampled/constructed from a large collection of unlabeled videos to produce a virtually infinite collection of labeled examples. A within-modality correspondence dataset might be the correspondence of images or video clips occurring within the parent dataset where e.g. in the former case examples have the form (frame_a, frame_b, proximity). Here as in the latter case proximity might be 1 when the media segments overlap, 0 when they come from different clips, and when they originate from the same clip but don’t overlap some non-linear (e.g. exponential) function of the temporal distance between their starting points. The same paradigm applies in the cases of (video_clip, audio_clip) and (audio_clip_a, audio_clip_b) etc.</p>
        <p>Let’s take a step back and digest that for a moment - we could *construct* a very large labeled dataset from an unlabeled one. Then we could use this constructed dataset to learn an embedding space with meaningful structure. Then we could obtain labels for only a small subset of examples and use these to assign consensus labels to queries using the nearest-neighbor methods discussed above.</a>
        <p>Let’s understand a bit about what kind of neural network architecture could enable us to do so (below). Then in a latter section discuss some problems in neuroimaging and biosensing that could be formulated in this way.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${sslArchitecture}            
            </div>
        </div>

        <p>Interestingly, one can simply concatenate embedding vectors and pass these to downstream decoder layers as if they were produced by a preceding one. Diagrammed above is an architecture wherein media (e.g. images) a and b are encoded to produce embedding vectors for each. These are in turn concatenated and passed to a decoder that produces a label prediction. Regarding concatenation, for example given the two-dimensional embeddings <0.2, 0.5>, <0.3, 0.1> the concatenation of these would be <0.2, 0.5, 0.3, 0.1>.</p>
        <p>With a network architected in this way learning proceeds as the network receives feedback on the accuracy of its prediction as to whether the two inputs correspond (and if proximity is continuous instead of binary the constructed notion of extent of this). And as these predictions improve our hope is that the embedding space is optimized such that example proximity corresponds increasingly to proximity in the embedding space.</p>
        <p>In pioneering the audio-visual correspondence learning task, Arandjelovic and Zisserman (2017) demonstrated networks trained in this way can learn to recognize semantic entities e.g. guitars, accordions, keyboards, … That is, that in the presence of these distinct entities different high-level spatial feature detection neurons fire to paint the abstract visual field with a signal of recognition of each class, see below. Notably, without having trained the networks with any explicit supervisory signal of the value of learning to do so:</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                <img src="/images/ml-figures/avc-representations.png"></img>
            </div>
        </div>

        <p>Above, a composite of <a href="https://arxiv.org/pdf/1705.08168.pdf" taget="_blank">Arandjelovic and Zisserman (2017)</a> Figures 3 and 4, shows the above described property of certain portions of the visual network learning to fire preferentially in the presence of sound-associated semantic features. Notably, for example, the area where a pins crash in a bowling alley or the keyboard of an accordion.</p>
        <p>One might wonder then about combining such a representation learning method with the above described methods of sparse annotation and nearest neighbor consensus. As well as perhaps wonder further about applications of this learning paradigm in the domains of applied cognitive neuroscience and neurofeedback gaming. For example, extending the above paradigm to the correspondence of video and audio, can tonal and other events in spoken audio demarcate visual-spatial features in video that are arranged in embedding space in a meaningful way in terms of the emotional or cognitive state of the speaker? That is, can the sound of happy speech provide a supervisory signal for learning what a happy facial expression looks like (through time)? And, excitingly, can we extend the metaphor of audio supervising the learning of features in video to other modality pairs such as video supervising the learning of features in EEG (or perhaps other neural sensing modalities)?</p>

        <div class="section-spacer"></div>

        <p class="article-section-pre-title">Perceptual labeling</p>
        <h2 class="article-section-title">Capturing language-free perceptual similarity</h2>
        <p>Extending our discussion on learning meaningful representation spaces as well as that of dealing with potentially inaccurate labels we’ll briefly discuss a fully-supervised method of learning representations of facial expressions that seeks to more accurately match human perceptual similarity (as well as in the course of this contrast language label similarity with perceptual similarity).</p>
        <p>In the learning setup introduced by Vemulapalli and Agarwala (2019), pictured below, three images are passed to encoder networks in parallel then concatenated and passed to the decoder (which produces a prediction) as described in a previous section. Here the input triplets consist of a query, q, and two comparisons, a and b, and the task is to predict the human-assigned label as to whether q is more visually similar to a or b - specifically in terms of the facial expression being exhibited.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${percepSimilarity}            
            </div>
        </div>

        <p>These authors demonstrate the resulting representations can be used with kNN to obtain a useful level of accuracy in predicting emotions despite not having been trained on this task. An interesting dimension to this is that nearest neighbor queries (below) produced more perceptually similar results when performed in the representation space learned with the triplet loss described above than one learned using conventional emotion labels - suggesting the former is more accurately representing the phenomenon despite producing poorer predictions of semantic labels.</p>
        
        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                <img src="/images/ml-figures/similarity-lookup.png"></img>
            </div>
        </div>
        
        <p>Pictured above is a selection from Figure 13 of Vemulapalli and Agarwala (2019) where a nearest-neighbor query (left) is shown with results (right), the latter being from spaces learned from perceived emotion similarity (top) vs. semantic emotion labels (bottom). This discussion should help understand one of the ways we can explore the usefulness of learned representations, how to evaluate these relative to another, and raise some questions in our mind about the need to learn more subtle representations of emotional and cognitive states than simply those for assigning a limited set of binary labels.</p>
      </section>

      <section class="article-section scrollspy-target" id="adv-opportunities">
        <p class="article-section-pre-title">Advantages & opportunities</p>
        <h2 class="article-section-title">Advantages and opportunities for clinical neuroscience</h2>
        <p>Why would success in these regards be of significant value for the field of neuroscience broadly as well as narrowly for the application of neurofeedback gaming? Here we argue our program has merit in both the short-term (3-6mo.) and long-term (6-12mo.), each discussed in detail below.</p>
        <p>In the short-term, the ability to sense emotional and cognitive state is a fundamentally useful capability for building emotional and cognitive training games that are used to boost the effectiveness of adult knowledge workers. Most simply, displaying cues for someone in the background of their work or meditation regarding when their expression has drifted beyond that correlating with their optimal state would help train users to themselves more quickly note and address this. Introspective emotional intelligence is well known to correlate with work performance as well as simple “noting” being a well known means of diffusing adverse or distracting emotions. Yet in contrast to the classic approach of slowly over time learning to automatically note through thousands of hours of meditative practice we anticipate the described feedback mechanism would shorten this process dramatically and facilitate its transfer into the context of one’s work - whether feedback is practiced in the background of either work or a sitting meditation.</p>
        <p>State-aware brain gaming - moving beyond simply providing direct feedback - is another significant short-term opportunity. Currently Neuroscape is in the process of developing several such games, none of which take into account whether the user may be experiencing either and adverse or enhanced state independent of the game content - something one should expect to have great relevance to understanding the effectiveness of the game itself as well as relevance to adapting the content of the game to optimally suit the user. With relevance to both, again, it is almost universally well understood that a person’s emotional and cognitive state affects their ability to learn and succeed in challenging tasks.</p>
        <p>Furthermore, notably given the described strategy above, we anticipate being able to deliver on these objectives without having to spend hundreds of thousands or millions of dollars on data annotation thus rendering it feasible financially given other non-personnel costs should total under $60,000/y.</p>
        <p>In the long-term we see the work as progress towards a fundamental advance in the field’s ability to model and make functional use of neural- and bio-sensing time-series data - both individually as well as when co-acquired - notably including EEG, fMRI, and peripheral physiological measures. The same advances in deep learning that have spawned the self-driving revolution - i.e. those enabling the understanding of visual time-series (video) - have yet to bear similar fruits in domains where labels are not so easily obtainable. Whereas in the visual domain one can pay a crowd worker to label pixels according to their class or object-ness - the same cannot be done in domains our perceptual systems have not evolved to perceive, such as EEG. Nevertheless, were we to have such labels for this domain it’s reasonable to explore the possibility that deep learning would be able to automate the process of identifying and modeling relationships between features there. Beginning with the metaphor of using audio to supervise the learning of features in the visual domain is a suitable step in a progression towards using video as a supervisory signal for the same in EEG, fMRI, and other time-series neural- and bio-sensing modalities.</p>
      </section>

      <section class="article-section scrollspy-target" id="references">
        <p class="article-section-pre-title">References</p>
        <ol type="1">
            <li>https://cloud.google.com/data-labeling/pricing</li>
            <li>https://aws.amazon.com/sagemaker/groundtruth/pricing</li>
            <li>Arandjelovic, Relja, and Andrew Zisserman. "Look, listen and learn." <em>Proceedings of the IEEE International Conference on Computer Vision.</em> 2017.</li>
            <li>Vemulapalli, Raviteja, and Aseem Agarwala. "A Compact Embedding for Facial Expression Similarity." <em>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.</em> 2019.</li>
        </ol>

      </section>

      <div class="article-bottom-spacer"></div>

      <div class="made-with-love"><p>Made with love in California.</p></div>

      <div id="scrollspy">
        ${this.renderScrollSpyItem("label-issues", "Label issues")}
        ${this.renderScrollSpyItem("repr-space", "Repr. space")}
        ${this.renderScrollSpyItem("repr-learning", "Repr. learning")}
        ${this.renderScrollSpyItem("adv-opportunities", "Opportunities")}
      </div>

    `;
  }


}


