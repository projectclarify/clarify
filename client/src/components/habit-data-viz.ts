import { LitElement, css, html, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store, RootState } from '../store.js';

// This element is *not* connected to the Redux store.
@customElement('habit-data-viz')
export class HabitDataViz extends connect(store)(PageViewElement) {

  @property({type: Object})
  _data = {};

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

      :host {
        width: 100%;
        height: 100%;
      }

      `];
  }

  _handleNavClick(e: Event) {
    this.habitToggle["testname"] = true;
  }

  protected render() {
    return html`

    <div class="nav right-nav">
        <div class="nav-items">
            <div class="nav-item nav-title">DATA TYPES</div>
            <div class="nav-item" navid="habits" @click="${this._handleNavClick}" active="${this._navSection == 'habits'}">AM Run</div>
            <div class="nav-item" navid="subjectives" @click="${this._handleNavClick}" active="${this._navSection == 'subjectives'}">Cold therapy</div>
            <div class="nav-item" navid="sensor-data" @click="${this._handleNavClick}" active="${this._navSection == 'sensor-data'}">Sensor data</div>
        </div>
    </div>

    `;
  }

  // This is called every time something is updated in the store.
  stateChanged(state: RootState) {
    //this._data = state.data!.habitData;
  }

}
