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

import { infra } from './ml-images.js';


@customElement('view-article-scaling-tpu-input')
export class ViewArticleScalingTpuInput extends connect(store)(ViewArticleBase) {

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
        <p class="article-hero-pre-heading">TECHNICAL DISCUSSION</p>
        <p class="article-hero-heading">Scaling TPU input</p>
        <p class="article-hero-subheading">Scaling the random sampling of videos to maximize TPU throughput</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">Scroll to read more</p>
          ${mouseIcon}
        </div>
      </section>

      <section class="article-section scrollspy-target" id="problem">
        <p class="article-section-pre-title">Problem</p>
        <h2 class="article-section-title">An input pipeline bottleneck</h2>
        <p>Machine learning research itself is an iterative exploratory process (including of the development of exploratory processes themselves) where we seek to maximize insight derived from our time, money, compute, and other resources; being able to experiment faster, less expensively, and more flexibly is beneficial for obvious reasons.</p>
        <p>Cloud TPUs have the potential to enable this. One Cloud TPU v3 (pictured below) has compute capability up to 420 teraflops and carries 128 GB of high-bandwidth memory (HBM) between eight compute cores sharing the same accelerator card. That’s compared to about 15 teraflops and 16-32 GB of HBM on NVIDIA’s V100 GPU accelerator.</p>
        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                <img src="https://cloud.google.com/images/products/tpu/cloud-tpu-v3-img_2x.png"></img>
            </div>
        </div>
        <p>Cloud TPUs are available in “pod” format (pictured below) where individual TPUs are connected via a 2D toroidal mesh network aggregating 100+ petaflops and 32TB of HBM. Google compared the cost and time to target accuracy for training ResNet-50 between (1) a cloud VM with eight V100 GPUs and (2) one full Cloud TPU (v2) pod - finding the latter was 27 times faster and 38% less expensive. In the latter case taking only 7.9 minutes.</p>
        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                <img src="https://cloud.google.com/images/products/tpu/cloud-tpu-v3-pod-beta_2x.png"></img>
            </div>
        </div>
        <p>One of the primary challenges in working with TPUs is being able to feed examples at a rate that can keep up with their consumption of them, especially at the pod scale - there are 512 TPU cores in a full v2 pod and 2048 in the largest available v3 pod (with 256 v3 boards @ 8 cores/board). Serving examples to TPUs via Cloud BigTable in a pre-augmented ready-to-train form is a good way to address this challenge.</p>
        <p>Here we'll review some of the code that was written to enable this in the pursuit of the state understanding methods we describe in <a href="machine-learning">another article</a>.</p>
      </section>


      <section class="article-section scrollspy-target" id="overview">
        <p class="article-section-pre-title">Overview</p>
        <h2 class="article-section-title">Overview diagram of our infrastructure</h2>
        <p>The following diagrams our data preprocessing, training, and deployment infrastructure. Reading more in detail about <a href="/machine-learning" target="_blank">our machine learning methods</a> is a useful entrypoint to understanding the purpose of various components in the following.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${infra}
            </div>
        </div>

        <p>From left to right, videos in .mp4 format are read from Google Cloud Storage (GCS) by an Extract job running on either of Kubernetes (as a Job), Cloud Functions (CF), Cloud Run (CR), or Apache Beam (Beam). Extract operations read raw video data and write each frame and audio track individually to a Cloud Big Table (CBT) table. A CBTDatagen job, with similar options for runtime, randomly samples pairs of video and from these extracts random contiguous segments of frames (and corresponding audio); these samples are used to construct both positive and negative frame/audio samples for segments from the same or different video that either do or do not overlap. These examples are serialized in TFExample form and randomly written to another CBT table avoiding the need for large-scale data shuffling. TFJobs, launched either individually or by a Katib StudyJob, running on Google Kubernetes Engine (GKE) with network-attached TPU instances randomly sample examples from the CBT example table for use in training. Training instances periodically produce checkpoints and summary statistics that are stored on GCS. An embedding job loads model parameters from GCS and maps labeled data into embedding space, writing the conjunction of these embeddings with the original labels back to an embedding data file on GCS. This along with a model checkpoint is needed by the state labeling Cloud Function that is triggered automatically by the arrival of new video frames (produced by the front end) in a section of a Firebase database. This function applies the embedding portion of the trained model to the arriving frames then uses that embedding together with the previously computed embeddings for labeled data to form a set of consensus labels for the triggering query. The result is written back to a different section of a Firebase database that itself triggers and update to a front-end data visualization.</p>
      </section>




      <div class="article-bottom-spacer"></div>

      <div class="made-with-love"><p>Made with love in California.</p></div>

      <div id="scrollspy">
        ${this.renderScrollSpyItem("problem", "Problem")}

      </div>

    `;
  }


}


