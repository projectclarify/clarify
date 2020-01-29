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

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { mouseIcon, scrollSpyIcon } from './my-icons.js';

import '/node_modules/@polymer/paper-button/paper-button.js';

import {scroll} from '@polymer/app-layout/helpers/helpers.js';
import '@polymer/iron-scroll-threshold/iron-scroll-threshold.js';

import {Debouncer} from '@polymer/polymer/lib/utils/debounce.js';
import {microTask} from '@polymer/polymer/lib/utils/async.js';

import { store } from '../store.js';
import { navigate } from '../actions/app.js';

import { updateProgressBarState } from '../actions/app.js';

@customElement('view-article-project-clarify')
export class ViewArticleProjectClarify extends PageViewElement {

  @property({ type: String })
  activeSection = '';

  @property({ type: Map})
  _scrollThresholds = new Map<String, number>();

  @property({type: Object})
  _debounceJob = "";

  @property({type: Number})
  _docHeight = 100;

  static get styles() {
    return [
      SharedStyles,
      css`
      
      #site-heading {
        font-family: Pacifico;
        color: var(--app-primary-color);
        font-size: 96px;
      }

      #site-subheading {
        margin-top: 50px;
      }

      `
    ];
  }

  protected render() {
    return html`
      <section class="article-hero">
        <p class="article-hero-pre-heading">NEUROSCAPE'S</p>
        <p class="article-hero-heading" id="site-heading">Project Clarify</p>
        <p class="article-hero-subheading" id="site-subheading">Applying machine learning to the improvement of emotional intelligence training</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">Scroll to read more about our goals and strategy.</p>
          ${mouseIcon}
        </div>
      </section>

      <section class="article-section scrollspy-target" id="big-idea">
        <p class="article-section-pre-title">Big Idea</p>
        <h2 class="article-section-title">Finding the right exercises to boost effectiveness and quality of life</h2>
        <p>The focus of Project Clarify is on boosting the <a href="https://en.wikipedia.org/wiki/Effectiveness" target="_blank">workplace effectiveness</a> and <a href="https://en.wikipedia.org/wiki/Quality_of_life" target="_blank">quality of life</a> of adult <a href="https://en.wikipedia.org/wiki/Knowledge_worker" target="_blank">knowledge workers</a> initially through machine learning enabled emotional intelligence training.</p>
        <p>It is well established that the brain is amenable to significant change even in adulthood (a phenomenon termed <a href="http://dbm.neuro.uni-jena.de/pdf-files/Draganski-Nature.pdf" target="_blank">neuroplasticity</a>, see <a href="https://en.wikipedia.org/wiki/Neuroplasticity" target="_blank">also</a>).
        Eastern contemplative traditions have pursued improvements in both emotional and cognitive awareness and regulation for millennia.
        Pioneering work in the contemplative neurosciences has begun to <a href="https://en.wikipedia.org/wiki/Research_on_meditation" target="_blank">establish</a> the effectiveness of some of these types of practices. </p>
        <p class="large-quote">“There is also growing evidence that mindfulness training improves adults’ ability to regulate attention and executive function, including orienting attention and monitoring conflict (<a href="https://link.springer.com/article/10.3758/CABN.7.2.109" target="_blank">Jha, Krompinger, & Baime, 2007</a>) and inhibiting emotionally charged but irrelevant information (<a href="https://link.springer.com/article/10.1007/s11031-007-9076-7" target="_blank">Ortner, Kilner, & Zelazo, 2007</a>) … In adults, meditation practice can also induce present-oriented forms of self-awareness (<a href="https://academic.oup.com/scan/article/2/4/313/1676557" target="_blank">Farb et al., 2007</a>) that likely enhance motivation and learning (<a href="https://www.tandfonline.com/doi/abs/10.1080/00461520902832376" target="_blank">Roeser & Peck, 2009</a>).” - <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1750-8606.2012.00240.x" target="_blank">Davidson, et al. 2012</a></p>
        <p>Classical contemplative training proceeds slowly, is challenging to measure, and challenging to engage with without expert instruction.</p>
        <p>Project Clarify aims to facilitate a modern extension of this classic tradition toward increasing measurability, accelerating progression, and reducing the need for in-person expert instruction. We aim to do this through the development of browser-based emotion state awareness games (see below) that are natural for users to integrate into their existing daily routine.</p>
        <p>We aim to achieve this through the application of modern computational, machine learning, and neuro-sensing technologies - chiefly by way of building training experiences that are state-aware. A technical discussion of the technologies we use to train models to make such state predictions from video and various neural sensing modalities can be found in the <a href="/models-and-infra">Technical Discussion</a> section.</p>
      </section>
   
      <section class="article-section scrollspy-target" id="training-games">
        <p class="article-section-pre-title">Training Games</p>
        <h2 class="article-section-title">The space of state-aware training games</h2>

        <p>Initially we seek to explore the effectiveness of state-aware training exercises within four primary categories: user labeling, vocation background, meditation background, and mirroring. These are described below.</p>

        <table class="article-figure-legend-table">

        <tr>
            <td><b>User labeling: </b>User is presented with images and asked to assign emotion labels across a range of emotions simultaneously, continuous values, and their predictions are matched to that of trained experts. No state awareness.</td>
        </tr>
        <tr>
            <td><b>Vocation background: </b>User engages in a vocational task (such as programming or writing) while machine learning system predicts emotional state from video and presents user with background audio cues of state including that relative to a state objective.</td>
        </tr>
        <tr>
            <td><b>Meditation background: </b>User engages in a classic meditation exercise (e.g. mindful breathing, open awareness, body scan, compassion visualization) while receiving the same sort of state feedback provided in the vocation background exercise.</td>
        </tr>
        <tr>
            <td><b>Mirroring: </b>User watches a video of a single speaker and attempts to closely match the emotional state of the speaker; state awareness models score and feed back a measure of state similarity.</td>
        </tr>

        </table>

      <p>Even though these games are our initial focus, as an open community project, we are always interested in additional ideas including new categories of training exercise. If you're interested in suggesting one feel free to simply <a href="https://github.com/projectclarify/pcml/issues" target="_blank">file an issue</a> in our GitHub <a href="https://github.com/projectclarify/pcml/issues" target="_blank">repository</a>.</p>

      </section>

      <section class="article-section scrollspy-target" id="trial-platform">
        <p class="article-section-pre-title">Trial platform</p>
        <h2 class="article-section-title">A professional-grade platform for trials of ML-enabled digital therapeutics</h2>
  
        <p>More generally we seek to establish a platform to enable the exploration and formal evaluation of the above to be shared with our colleagues across the research community, in the spirit of open, reproducible science and mutual gain through the sharing of effort and expertise.</p>
        
        <p>We see the following as some of the essential features of such a platform (discussed in greater detail in the <a href="/models-and-infra">Technical Discussion</a> section):</p>

        <ol type="1">
            <li><b>Validity: </b>Support for randomized, controlled designs</li>
            <li><b>Scale: </b>The capability to scale to support a sufficiently large number of users / samples</li>
            <li><b>Evaluation: </b>Support for diverse forms of pre-, mid-, and post-trial evaluations in terms of key predictive and response variables (e.g. stress level, subjective state, vocational effectiveness)</li>
            <li><b>Richness: </b>Support for collection of data of such diversity to approximately understand the mechanisms underlying successful interventions</li>
            <li><b>ML: </b>Supporting the real-time use of machine learning models in training exercises including models trained using the most modern machine learning hardware and methods</li>
            <li><b>Crowd: </b>Support for the integration of crowd-sourcing</li>
            <li><b>Professional-grade: </b>The system should achieve the performance, reliability, security, and otherwise quality of implementation considered standard and requisite within top-tier technology companies (exceeding the norms of academic projects)</li>
            <li><b>Lowers barrier: </b>Researchers with new ideas for training exercises to trial should experience minimal barrier to evaluating their idea thereby enabling a greater diversity of exercises to be evaluated.</li>
        </ol>

        <p>We see a significant opportunity to share in the development of common infrastructure allowing a shift in the allocation of resources from duplicated infrastructure efforts to a greater diversity of training exercise trials.</p>

        <p>If you’re a developer you may want to continue reading about the trial platform in our <a href="/technologies">Technical Discussion</a> section or check out our machine learning research <a href="https://github.com/projectclarify/pcml" target="_blank">codebase</a> on GitHub.</p>

      </section>



      <div class="article-bottom-spacer"></div>

      <div class="made-with-love"><p>Made with love in California.</p></div>

      <div id="scrollspy">
        <div class="scrollspy-item" scrollTarget="big-idea" @click="${this._scrollToScrollTarget}">
          <p class="scrollspy-item-text">Big Idea</p>
          <div class="scrollspy-item-icon">${scrollSpyIcon}</div>
        </div>
        <div class="scrollspy-item" scrollTarget="training-games" @click="${this._scrollToScrollTarget}">
          <p class="scrollspy-item-text">Training games</p>
          <div class="scrollspy-item-icon">${scrollSpyIcon}</div>
        </div>
        <div class="scrollspy-item" scrollTarget="trial-platform" @click="${this._scrollToScrollTarget}">
          <p class="scrollspy-item-text">Trial platform</p>
          <div class="scrollspy-item-icon">${scrollSpyIcon}</div>
        </div>
      </div>

    `;
  }

  _navigateToDemo() {
    store.dispatch(navigate("/interactive-perspective-shift"));
  }

  _updateScrollThresholdPositions() {

    let sections = this.shadowRoot!.querySelectorAll("section");

    let bodyRect = document.body.getBoundingClientRect();
    for (let section of sections) {
      let topOffset = section.getBoundingClientRect().top - bodyRect.top - 50;
      let name = section.id;
      if (name != "") {
        this._scrollThresholds[name] = topOffset;
      }
    }

    console.log(this._scrollThresholds);

  }

  _updateScrollSpyStyling() {

    console.log("updating scrollspy styling");

    // Get a reference to the scrollspy element by its id
    let scrollspyItems = this.shadowRoot!.querySelectorAll(".scrollspy-item");

    // For each item there within with class "scrollspy-item"
    for(let scrollspyItem of scrollspyItems){
      let scrollTarget = scrollspyItem.attributes.scrollTarget.nodeValue;
      if(scrollTarget != this.activeSection) {
        // This scrollspy item is not an active one
        scrollspyItem.setAttribute("selected", false);
      } else {
        // This scrollspy item *is* an active one.
        scrollspyItem.setAttribute("selected", true);
      }
    }

  }

  _scrollToScrollTarget(e: Event) {

    // Get the name of the scroll target from the `scrollTarget` attribute of the
    // .scrollspy-item container (including in the case where e.target is
    // scrollspy-item-text instead of scrollspy-item...)
    let target = e.target;

    if (target.className == "scrollspy-item-text") {
      target = e.composedPath()[1];
    }

    let scrollTargetID = target.attributes.scrollTarget.nodeValue;
    let scrollTarget = this.shadowRoot.querySelector("#" + scrollTargetID);

    let bodyRect = document.body.getBoundingClientRect();
    let elemRect = scrollTarget.getBoundingClientRect();
    let offset = elemRect.top - bodyRect.top - 50;

    // Scroll there
    scroll({top: offset, behavior: "smooth"});

  }

  _debounceHandleScroll() {
    this._debounceJob = Debouncer.debounce(this._debounceJob,
      microTask, () => this._handleScroll());
  }

  _handleScroll() {

    let scrollThresholds = this._scrollThresholds;

    if(Object.keys(scrollThresholds).length == 0){
      console.log("updating scroll thresholds");
      this._updateScrollThresholdPositions();
      scrollThresholds = this._scrollThresholds;
    }

    let currentPosition = window.pageYOffset;

    // What will hold what we determine to be the currently active section
    let sectionId = "";

    let thresholdFun = function(key: any) {
      let value = scrollThresholds[key];
      if(currentPosition > (value - 300)) {
        if(key !== ""){
          sectionId = key;
        };
      };
    };

    Object.keys(scrollThresholds).forEach(thresholdFun);

    if (sectionId != this.activeSection) {
      // Trigger an update to the scrollspy menu
      this.activeSection = sectionId;
      this._updateScrollSpyStyling();
    }

    console.log(this.activeSection);

    if (this._docHeight <= 0) {
      store.dispatch(updateProgressBarState(0));
    } else {
      store.dispatch(updateProgressBarState(currentPosition / this._docHeight));
    }

  }

  constructor() {
    super();
    this._boundListener = this._debounceHandleScroll.bind(this);
  }

  updateDocHeight() {
    this._docHeight = (document.height !== undefined) ? document.height : document.body.offsetHeight;
  }

  firstUpdated() {
    this._updateScrollThresholdPositions();
    this.updateDocHeight();
    //store.dispatch(updateLoadingAnimationState(false));
  }

  connectedCallback() {
    super.connectedCallback();
    window.addEventListener('scroll', this._boundListener);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('scroll', this._boundListener);
  }

}
