/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { css, html, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store } from '../store.js';

import { mouseIcon } from './my-icons.js';

import '/node_modules/@polymer/paper-button/paper-button.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

@customElement('view-article-residency-program')
export class ViewArticleResidencyProgram extends connect(store)(PageViewElement) {

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`
      .float-center-button-container{
          display: flex;
          flex-direction: row;
          width: 100%;
          justify-content: center;
          margin-top: 50px;
      }
      `
    ];
  }

  protected render() {
    return html`

      <section class="article-hero">
        <p class="article-hero-pre-heading">PROGRAMS</p>
        <p class="article-hero-heading">Residency Program</p>
        <p class="article-hero-subheading">Self-directed, interest-area projects on a quarterly basis</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">Scroll to read more</p>
          ${mouseIcon}
        </div>
      </section>

      <section class="article-section scrollspy-target" id="summary">
        <p class="article-section-pre-title">Summary</p>
        <h2 class="article-section-title">What is the residency program?</h2>
        <p>Neuroscape's Project Clarify Residency Program is a quarterly program where contributors explore and deliver on a concise project in an area of significant interest to them. Residency Program members will have the opportunity to benefit from the expertise and exposure both to each-other as well as senior engineers and researchers affiliated with the project. Program participants will come away from their quarter project with a tangible product of having expanded their skills and knowledge into a new area - both in name of haivng participated and in the form of code and digital assets contributed to the Project Clarify open source project.</p>
      </section>

      <section class="article-section scrollspy-target" id="project-opportunities">
        <p class="article-section-pre-title">Project Opportunities</p>
        <h2 class="article-section-title">Freedom to pursue your interests</h2>
        <p>Residency Program projects are designed at the discretion of the participant but here are some examples to inspire your imagination:</p>

        <p style="text-align: center">Data engineering</p>

        <table class="article-figure-legend-table">

          <tr><td><b>New data types: </b>Implement and test data processing steps of pipelines for ingest of interesting new data types (e.g. functional MRI, positron emission tomography).</td></tr>
          <tr><td><b>Simulation and learnability: </b>Build a progressively more accurate simulator of a category of learning problem to allow exploration of trainability of models for that problem as a function of simulation parameters.</td></tr>

        </table>

        <p style="text-align: center">ML engineering</p>

        <table class="article-figure-legend-table">

          <tr><td><b>Understand bias: </b>Apply existing or design new methods to characterize and mitigate bias (e.g. demographic) in state prediction models.</td></tr>
          <tr><td><b>Reinforcement learning: </b>Train and deploy a model-based reinforcement learning agent to optimize user experience (e.g. the difficulty of a game the user is playing)</td></tr>
          <tr><td><b>Improved understanding: </b>Design a new learning problem to enhance performance on state labeling or representation problems.</td></tr>

        </table>

        <p style="text-align: center">UX design and strategy</p>

        <table class="article-figure-legend-table">

          <tr><td><b>Design interactives: </b>Design elegant, interactive, real-time, ML- and data-driven experiences for the Project Clarify demonstration site that illustrate new opportunities and broaden our understand of the requirements of our core ML research program.</td></tr>
          <tr><td><b>Assessments: </b>Take stock of existing training methodologies and communicate opportunities for the application of machine learning in the form of concise whitepapers/briefs.</td></tr>

        </table>

        <p>The project ideas above are just a sample of what's possible and it's up to you to decide either how to make one of these your own or come up with something completely new.</p>

      </section>

      <section class="article-section scrollspy-target" id="apply">
        <p class="article-section-pre-title">APPLY</p>
        <h2 class="article-section-title">Apply to the Program</h2>
        <p>Applications to the Residency Program are reviewed on a rolling quarterly basis one month prior to the start of the program round (e.g. September 1st for Q4, December 1st for Q1). In the course of this application residents are asked to draft an inital project proposal and to provide information about their skills and interests. We are so excited to hear from you!</p>

        <div class="float-center-button-container">
            <paper-button @click="${this._navApply}">Apply Here</paper-button>
        </div>

      </section>

      <div class="article-bottom-spacer"></div>

      <div class="made-with-love"><p>Made with love in California.</p></div>

    `;
  }

  _navApply() {
    window.open('https://forms.gle/bbnN9JqmNdqzjUxs7', '_blank');
  }

}


