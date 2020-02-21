import { LitElement, css, html, property, customElement } from 'lit-element';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

import './data-viz.js';

// This element is *not* connected to the Redux store.
@customElement('subjectives-data-viz')
export class DataVizBase extends LitElement {

  @property({ type: Number })
  extraWidthY = 100;

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

  protected render() {
    return html`

        subjectives


    `;
  }
}
