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

import '/node_modules/@polymer/paper-button/paper-button.js';
import '@polymer/iron-scroll-threshold/iron-scroll-threshold.js';

import { store } from '../store.js';
import { navigate } from '../actions/app.js';

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
      </section>

    `;
  }

  _navigateToDemo() {
    store.dispatch(navigate("/interactive-perspective-shift"));
  }

  _navLogin(){
      store.dispatch(navigate("/login"));
  }

  constructor() {
      super();
  }

  firstUpdated() {
    //store.dispatch(updateLoadingAnimationState(false));
  }

  connectedCallback() {
    super.connectedCallback();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
  }

}
