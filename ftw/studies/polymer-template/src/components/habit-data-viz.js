var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { css, html, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
import { connect } from 'pwa-helpers/connect-mixin.js';
// This element is connected to the Redux store.
import { store } from '../store.js';
// This element is *not* connected to the Redux store.
let HabitDataViz = class HabitDataViz extends connect(store)(PageViewElement) {
    // This element is *not* connected to the Redux store.
    constructor() {
        super(...arguments);
        this._data = {};
    }
    static get styles() {
        return [
            SharedStyles,
            ButtonSharedStyles,
            css `

      :host {
        width: 100%;
        height: 100%;
      }

      `
        ];
    }
    _handleNavClick(e) {
        this.habitToggle["testname"] = true;
    }
    render() {
        return html `

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
    stateChanged(state) {
        //this._data = state.data!.habitData;
    }
};
__decorate([
    property({ type: Object })
], HabitDataViz.prototype, "_data", void 0);
HabitDataViz = __decorate([
    customElement('habit-data-viz')
], HabitDataViz);
export { HabitDataViz };
