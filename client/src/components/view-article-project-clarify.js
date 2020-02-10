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
import { css, html, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import '/node_modules/@polymer/paper-button/paper-button.js';
import '@polymer/iron-scroll-threshold/iron-scroll-threshold.js';
import { store } from '../store.js';
import { navigate } from '../actions/app.js';
let ViewArticleProjectClarify = class ViewArticleProjectClarify extends PageViewElement {
    constructor() {
        super();
        this.activeSection = '';
        this._scrollThresholds = new Map();
        this._debounceJob = "";
        this._docHeight = 100;
    }
    static get styles() {
        return [
            SharedStyles,
            css `

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
    render() {
        return html `
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
    _navLogin() {
        store.dispatch(navigate("/login"));
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
};
__decorate([
    property({ type: String })
], ViewArticleProjectClarify.prototype, "activeSection", void 0);
__decorate([
    property({ type: Map })
], ViewArticleProjectClarify.prototype, "_scrollThresholds", void 0);
__decorate([
    property({ type: Object })
], ViewArticleProjectClarify.prototype, "_debounceJob", void 0);
__decorate([
    property({ type: Number })
], ViewArticleProjectClarify.prototype, "_docHeight", void 0);
ViewArticleProjectClarify = __decorate([
    customElement('view-article-project-clarify')
], ViewArticleProjectClarify);
export { ViewArticleProjectClarify };
