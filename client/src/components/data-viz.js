var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { LitElement, css, html, property, customElement } from 'lit-element';
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
import './data-viz.js';
// This element is *not* connected to the Redux store.
let DataVizBase = class DataVizBase extends LitElement {
    // This element is *not* connected to the Redux store.
    constructor() {
        super(...arguments);
        this.extraWidthY = 100;
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
    render() {
        return html `

        subjectives


    `;
    }
};
__decorate([
    property({ type: Number })
], DataVizBase.prototype, "extraWidthY", void 0);
DataVizBase = __decorate([
    customElement('subjectives-data-viz')
], DataVizBase);
export { DataVizBase };
