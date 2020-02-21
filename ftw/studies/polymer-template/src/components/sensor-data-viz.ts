import { LitElement, css, html, property, customElement } from 'lit-element';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

// This element is *not* connected to the Redux store.
@customElement('sensor-data-viz')
export class SensorDataViz extends LitElement {

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

        sensor


    `;
  }
}
