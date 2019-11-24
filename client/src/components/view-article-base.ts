/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { css, html, property } from 'lit-element';
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

import { updateProgressBarState } from '../actions/app.js';

export class ViewArticleBase extends PageViewElement {

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
      css``
    ];
  }

  renderSiteHeading(preHeading: string, heading: string, subHeading: string, scrollInfo: string) {
     return html`
      <section class="article-hero">
        <p class="article-hero-pre-heading">${preHeading}</p>
        <p class="article-hero-heading">${heading}</p>
        <p class="article-hero-subheading">${subHeading}</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">${scrollInfo}</p>
          ${mouseIcon}
        </div>
      </section>
      `
  }

  renderScrollSpyItem(itemTargetID: string, itemText: string) {
      return html`
        <div class="scrollspy-item" scrollTarget="${itemTargetID}" @click="${this._scrollToScrollTarget}">
          <p class="scrollspy-item-text">${itemText}</p>
          <div class="scrollspy-item-icon">${scrollSpyIcon}</div>
        </div>
      `
  }

  protected render() {
    return html`

      ${this.renderHeader("NEUROSCAPE'S",
                          "Project Clarify",
                          "Applying machine learning to the improvement of emotional intelligence training",
                          "Scroll to read more about our goals and strategy.")}

      <section class="article-section scrollspy-target" id="big-idea">
        <p class="article-section-pre-title"></p>
        <h2 class="article-section-title"></h2>
        <p></p>
      </section>

      <div id="scrollspy">
        ${this.renderScrollSpyItem("big-idea", "Big Idea")}
      </div>

    `;
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
