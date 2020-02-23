import { html, css, LitElement } from 'lit-element';

import '../../../steps/activity-step-base/activity-step-base.js';

export class ViewActivityBase extends LitElement {
  static get styles() {
    return css`
      :host {
        --view-activity-base-text-color: #000;

        display: block;
        padding: 25px;
        color: var(--view-activity-base-text-color);
      }
    `;
  }

  static get properties() {
    return {
      title: { type: String },
      counter: { type: Number },
    };
  }

  constructor() {
    super();
    this.title = 'Hey there';
    this.counter = 5;
  }

  __increment() {
    this.counter += 1;
  }

  render() {
    return html`
      <h2>${this.title} Nr. ${this.counter}!</h2>
      <button @click=${this.__increment}>increment</button>
    `;
  }
}
