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
import { html, css, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';
// This element is connected to the Redux store.
import { store } from '../store.js';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
import "/node_modules/@polymer/iron-collapse/iron-collapse.js";
import './habit-data-viz.js';
import './subjectives-data-viz.js';
import './sensor-data-viz.js';
let DashViewData = class DashViewData extends connect(store)(PageViewElement) {
    constructor() {
        super(...arguments);
        this._navSection = 'habits';
        this.userData = null;
    }
    static get styles() {
        return [
            SharedStyles,
            ButtonSharedStyles,
            css `
      
      .data-category-view {
          width: 100%
          height: 100%;
      }

      .data-category-view[active="false"] {
        display: none
      }

      `
        ];
    }
    _handleNavClick(e) {
        this._navSection = e.path[0].getAttribute("navid");
    }
    render() {
        return html `

    <div class="top-spacer"></div>

    <div class="nav">
      <div class="nav-items">
        <div class="nav-item nav-title">DATA TYPES</div>
        <div class="nav-item" navid="habits" @click="${this._handleNavClick}" active="${this._navSection == 'habits'}">Habits</div>
        <div class="nav-item" navid="subjectives" @click="${this._handleNavClick}" active="${this._navSection == 'subjectives'}">Subjectives</div>
      </div>
    </div>

    <div class="data-category-view" active="${this._navSection == 'subjectives'}">
      <subjectives-data-viz ></subjectives-data-viz>
    </div>

    <div class="data-category-view" active="${this._navSection == 'habits'}">
      <habit-data-viz ></habit-data-viz>
    </div>

    `;
    }
    stateChanged(state) {
        this.userData = state.user.userData;
    }
};
__decorate([
    property({ type: String })
], DashViewData.prototype, "_navSection", void 0);
__decorate([
    property({ type: Object })
], DashViewData.prototype, "userData", void 0);
DashViewData = __decorate([
    customElement('dash-view-data')
], DashViewData);
export { DashViewData };
