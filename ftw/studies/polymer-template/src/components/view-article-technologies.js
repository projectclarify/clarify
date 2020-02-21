/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { css, html, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';
import { mouseIcon } from './my-icons.js';
// This element is connected to the Redux store.
import { store } from '../store.js';
import './exercise-card.js';
import { infra } from './ml-images.js';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
let ViewArticleTechnologies = class ViewArticleTechnologies extends connect(store)(PageViewElement) {
    static get styles() {
        return [
            SharedStyles,
            css `
        a {
          text-decoration: none;
        }

        a:visited {
          color: var(--app-primary-color);
        }

        .article-large-width-image > svg {
          margin: 10px;
          flex-shrink: 0;
          width: 80%;
          height: auto;
        }

        .article-section > p {
            font-size: 14px;
        }

      `
        ];
    }
    render() {
        return html `

      <section class="article-hero">
        <p class="article-hero-pre-heading">TECHNICAL ARTICLE</p>
        <p class="article-hero-heading">Technologies</p>
        <p class="article-hero-subheading">A high-level overview of the technologies we use to do our research</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">Scroll to read more</p>
          ${mouseIcon}
        </div>
      </section>



      <section class="article-section scrollspy-target" id="core-infra">
        <p class="article-section-pre-title">Core infrastructure</p>
        <h2 class="article-section-title">Clinical-grade container infrastructure on the Google Cloud</h2>
        <p>We make use of a system of containerized infrastructure components that run on the container orchestration system <a href="https://kubernetes.io/" target="_blank">Kubernetes</a> in turn on <a href="https://cloud.google.com/kubernetes-engine/" target="_blank">Google Kubernetes Engine</a>. This is a common starting point in the technology industry for building scalable and reliable web and otherwise computational applications.</p>

        <table class="article-figure-legend-table">

        <tr><td><b>Kubeflow: </b>We use Kubeflow to facilitate the deployment and operation of our complete ML infrastructure stack. Kubeflow includes components for <a href="https://www.kubeflow.org/docs/components/training/tftraining/" target="_blank">training</a>, <a href="https://www.kubeflow.org/docs/components/hyperparameter-tuning/hyperparameter/" target="_blank">tuning</a>, <a href="https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/" target="_blank">serving</a>, and automating complex <a href="https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/" target="_blank">workflows</a> as well as <a href="https://www.kubeflow.org/docs/notebooks/" target="_blank">deploying</a> a shared <a href="https://jupyterlab.readthedocs.io/en/stable/" target="_blank">JupyterLab</a> research environment.</td></tr>
        <tr><td><b>JupyterLab: </b>We use Kubeflow’s deployment of <a href="https://jupyterlab.readthedocs.io/en/stable/" target="_blank">JupyterLab</a> running on <a href="https://cloud.google.com/kubernetes-engine/" target="_blank">Google Kubernetes Engine</a> as our primary research work environment. This provides us all of the benefits of similar commercial alternatives while maintaining a suitable environment for working with protected health information and clinical data.</td></tr>
        <tr><td><b>Containers: </b>We use <a href="https://www.docker.com/resources/what-container" target="_blank">containers</a> to make our model training environment reproducible with a benefit to debugging, transfer learning / continued training, and reproducibility between research groups.</td></tr>
        <tr><td><a href="https://www.youtube.com/watch?v=4ht22ReBjno" target="_blank"><b>Kubernetes: </b></a><a href="https://www.youtube.com/watch?v=4ht22ReBjno" target="_blank">Kubernetes</a> allows us to automate the process of provisioning new machines (including autoscaling) and scheduling jobs onto physical nodes as well as automating a range of other cluster-level operations. Kubernetes provides a broad array of highly useful APIs for launching operations (such as batch or cron jobs) as well as instrumenting operation status (notably including in the context of integration testing).</td></tr>

        </table>

        <p>The above approach, among other benefits, gives us the ability to ensure our data are handled in a manner compliant with clinical and privacy regulations (necessary for us to be able to use the infrastructure for clinical and PHI research).</p>
      </section>

      <section class="article-section scrollspy-target" id="ml-infra">
        <p class="article-section-pre-title">ML Infrastructure</p>
        <h2 class="article-section-title">We use ML infrastructure that accelerates our research</h2>
        <p>Doing machine learning research that (1) leverages <a href="https://en.wikipedia.org/wiki/Automated_machine_learning" target="_blank">automated search</a> of portions of model parameter space and (2) is consistent with the usage of the result in production is non-trivial. The speed and expense (and thus feasibility) of the former is limited by the speed of training of each model variant which we accelerate by using the most performant deep learning accelerators and attached data infrastructure we have available to us, namely <a href="https://cloud.google.com/tpu/" target="_blank">Cloud TPUs</a> and <a href="https://cloud.google.com/bigtable/" target="_blank">Cloud BigTable</a>.</p>

        <table class="article-figure-legend-table">

        <tr><td><a href="https://cloud.google.com/tpu/" target="_blank"><b>TPUs: </b></a>We’re using the most advanced ML accelerators to speed up our training and enable ML research as a search of the space of possible models.</td></tr>
        <tr><td><b>Cloud BigTable: </b>We use GCP’s <a href="https://cloud.google.com/bigtable/" target="_blank">BigTable</a> product to store and serve trainer-ready video examples to TPU training jobs thereby eliminating that source of network bottleneck for training jobs.</td></tr>
        <tr><td><b>Parallel augmentation: </b>Examples are prepared (including augmented) in <a href="https://github.com/projectclarify/pcml/blob/0597301f7915114c3b134a3ad43c2727d32e239c/pcml/operations/cbt_datagen.py#L55" target="_blank">jobs</a> separate from training jobs removing the limit on the computational intensity of augmentation operations we can employ.</td></tr>
        <tr><td><b>Tuning: </b>We’re using Kubeflow’s <a href="https://www.usenix.org/conference/opml19/presentation/zhou" target="_blank">Katib</a> <a href="https://github.com/kubeflow/katib" target="_blank">library</a> to coordinate the search over our models parameter space, primarily with an emphasis on Bayesian Optimization but with an eye to more sophisticated schemes.</td></tr>
        <tr><td><a href="https://github.com/tensorflow/tensor2tensor/blob/master/README.md" target="_blank"><b>Tensor2Tensor: </b></a>We use <a href="https://github.com/tensorflow/tensor2tensor/blob/master/README.md" target="_blank">tensor2tensor</a> to accelerate our research by making our problem and model definitions reproducible, composable, and similarly structured. T2T also includes a wide variety of model and problem building blocks as well as a best-practices model training layer (including TPU compatibility).</td></tr>
        <tr><td><b>TFServing & tfjs: </b>We use a mixture of both <a href="https://github.com/tensorflow/serving/blob/master/README.md" target="_blank">TFServing</a> to serve models from the back end as well as <a href="https://www.tensorflow.org/js" target="_blank">TensorFlow JS</a> to do so from the front-end. Between these we find a tradeoff between network and model size issues as well as privacy (both data and model).</td></tr>

        </table>

        <p>The simple result of our prior ML infrastructure work is that researchers can conveniently launch and quickly compare the results from experiments helping them engage with the research process at a higher level of abstraction and proceed with greater velocity.</p>

      </section>

      <section class="article-section scrollspy-target" id="interface">
        <p class="article-section-pre-title">Interface</p>
        <h2 class="article-section-title">We use a modern set of tools for building interfaces from remixable components</h2>
        <p>Our interface development strategy centers around developing for the browser in a way that allows quick development of new trial components as well as remixing of these by our peers.</p>

        <table class="article-figure-legend-table">

        <tr><td><b>Browser-focused: </b>We see developing training applications for the browser, in contrast to native OS’s as beneficial due to it being cross platform, crowd-compatible, simplifying development, and due to our not needing functionalities only available on native devices; <a href="https://en.wikipedia.org/wiki/Progressive_web_applications" target="_blank">progressive web application</a> methods allow us to still deliver an app-like experience.</td></tr>
        <tr><td><a href="https://polymer-library.polymer-project.org/3.0/docs/devguide/feature-overview" target="_blank"><b>Polymer: </b></a>We use <a href="https://polymer-library.polymer-project.org/3.0/docs/devguide/feature-overview" target="_blank">Polymer 3.0</a> <a href="https://www.webcomponents.org/introduction" target="_blank">WebComponents</a> and <a href="https://www.typescriptlang.org/" target="_blank">TypeScript</a> to build our interface primarily because of the ratio of power and simplicity of this approach as well as secondarily for the interoperability of resulting components with other applications already invested in a framework such as <a href="https://reactjs.org/" target="_blank">React</a> or <a href="https://angular.io/" target="_blank">Angular</a>.</td></tr>
        <tr><td><b>Hot-start trials: </b>We believe firmly in the importance of reducing barriers to trying new ideas; our interface strategy is oriented towards providing a starting point, including UI components and data APIs, that allow prototyping new training experiences with perhaps 10+ times less effort than starting from scratch and building bespoke for each new study.</td></tr>
        <tr><td><b>Share & remix: </b>In the spirit of reproducible science we think both the whole web application as well as the individual components of our training exercises should be able to be re-used and re-mixed by our peers. This can be accomplished by simply forking and re-branding our open source application code or migrating individual (portable) <a href="https://www.webcomponents.org/introduction" target="_blank">WebComponents</a> into another application.</td></tr>
        <tr><td><a href="https://firebase.google.com/" target="_blank"><b>Firebase: </b></a>We use Firebase to serve our <a href="https://en.wikipedia.org/wiki/Progressive_web_applications" target="_blank">PWA</a> and <a href="https://firebase.google.com/docs/firestore" target="_blank">Firestore</a> for application data storage (and options like <a href="https://cloud.google.com/bigtable/" target="_blank">BigTable</a> and <a href="https://cloud.google.com/bigtable/" target="_blank">Google Cloud Storage</a> for non-application storage). This approach greatly simplifies our application infrastructure, accelerates development, and simplifies operation.</td></tr>

        </table>

        <p></p>

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



      <section class="article-section scrollspy-target" id="sensors-datasets">
        <p class="article-section-pre-title">Sensors & datasets</p>
        <h2 class="article-section-title">Special measurement equipment means special opportunities for applied ML</h2>

        <p>As part of <a href="http://neuroscape.ucsf.edu" target="_blank">Neuroscape</a> at the <a href="https://www.ucsfhealth.org/" target="_blank">University of California, San Francisco</a>, the machine learning research of Project Clarify is positioned with access to a variety of sophisticated neural sensing systems such as a <a href="https://www.siemens-healthineers.com/en-us/magnetic-resonance-imaging/3t-mri-scanner/magnetom-prisma" target="_blank">3T Siemens Prisma Fit</a> MRI with 64- and 20-channel H/N coils capable of SMS diffusion and <a href="https://en.wikipedia.org/wiki/Blood-oxygen-level-dependent_imaging" target="_blank">BOLD</a> imaging, 3D pCASL, SWI, and high-resolution anatomical imaging. In addition we have access to a number of best-in-class <a href="https://en.wikipedia.org/wiki/Electroencephalography" target="_blank">EEG</a> systems including 128- and 64-channel active electrode systems from <a href="https://www.biosemi.com/" target="_blank">BioSemi</a> (wired) and various mobile EEG systems from <a href="https://www.neuroelectrics.com/products/enobio/" target="_blank">EnoBio</a>.</p>

        <div class="article-large-width-image-container">

            <div class="article-large-width-image">

                <img src="https://www.som.com/FILE/18605/sandlersneuroscience_1400x800_cesarrubio_01new3.jpg"></img>
    
            </div>
    
        </div>

        <p>These and other systems available on our and surrounding UC campuses put Project Clarify in a position to innovate in ways that hybridize innovation across many domains, including infrastructure, user experience, machine learning methods, and data collection methods.</p>

        <p>If you're a developer we hope you'll join us for a <a href="/hack">developer event</a> and/or check out our <a href="https://github.com/projectclarify/pcml" target="_blank">codebase</a> on GitHub.

      </section>

      <div class="article-bottom-spacer"></div>

      <div class="made-with-love"><p>Made with love in California.</p></div>

    `;
    }
};
ViewArticleTechnologies = __decorate([
    customElement('view-article-technologies')
], ViewArticleTechnologies);
export { ViewArticleTechnologies };
