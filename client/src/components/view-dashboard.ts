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

// These are the actions needed by this element.
import { navigate } from '../actions/app.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
import { moduleItemIcon } from './my-icons.js';

import "/node_modules/@polymer/iron-collapse/iron-collapse.js";

import "./dash-view-navigator.js";
import "./dash-view-exercises.js";
import "./dash-view-data.js";

@customElement('view-dashboard')
export class ViewDashboard extends connect(store)(PageViewElement) {

  @property({type: Object})
  _user = '';

  @property({type: String})
  _dashView = 'exercises';

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

        .dashboard-nav {
          display: flex;
          flex-direction: row;
          justify-content: center;
          width: 100%;
          position: fixed;
          top: 64px;
          background-color: white;
          border-bottom: 1px solid lightgrey;
        }

        .dashboard-nav-item {
          font-size: 11px;
          font-weight: 300;
          color: var(--app-secondary-color);
          padding-left: 10px;
          padding-right: 10px;
          padding-bottom: 5px;
          cursor: pointer;
        }

        .dashboard-nav-item[active="true"] {
          color: var(--app-primary-color);
          font-weight: 600;
          border-bottom: 2px solid var(--app-primary-color);
        }

        .page {
          display: none;
        }

        .page[active="true"] {
          display: block;
        }

        .top-spacer {
          height: 87.4px;
        }

      `
    ];
  }

  handleNavClick(e: Event) {
    this._dashView = e.path[0].getAttribute("navid");
  }

  protected render() {
    return html`

    <div class="dashboard-nav">

        <div class="dashboard-nav-item" active="${this._dashView == 'navigator'}" @click="${this.handleNavClick}" navid="navigator">Navigator</div>
        <div class="dashboard-nav-item" active="${this._dashView == 'exercises'}" @click="${this.handleNavClick}" navid="exercises">Exercises</div>
        <div class="dashboard-nav-item" active="${this._dashView == 'data'}" @click="${this.handleNavClick}" navid="data">Data</div>
 
    </div>

    <div class="top-spacer"></div>

      <div>
        <dash-view-navigator class="page" active="${this._dashView === 'navigator'}"></dash-view-navigator>
        <dash-view-exercises class="page" active="${this._dashView === 'exercises'}"></dash-view-exercises>
        <dash-view-data class="page" active="${this._dashView === 'data'}"></dash-view-data>
      </div>

    `;
  }

  stateChanged(state: RootState) {

  }

}
