
/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { html, css, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store, RootState } from '../store.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

import "/node_modules/@polymer/iron-collapse/iron-collapse.js";

import './habit-data-viz.js';
import './subjectives-data-viz.js';
import './sensor-data-viz.js';

@customElement('dash-view-data')
export class DashViewData extends connect(store)(PageViewElement) {

  @property({type: String})
  _navSection = 'habits';

  @property({type: Object})
  userData = null;

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`
      
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

  _handleNavClick(e: Event) {
    this._navSection = e.path[0].getAttribute("navid");
  }

  protected render() {
    return html`

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

  stateChanged(state: RootState) {
    this.userData = state.user!.userData;
  }

}

