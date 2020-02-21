import { html, css, LitElement } from 'lit-element';

import '../../../activities/view-activity-base/view-activity-base.js';

export class ViewSessionBase extends LitElement {
  static get styles() {
    return css`
      :host {
        --view-session-base-text-color: #000;

        display: block;
        padding: 25px;
        color: var(--view-session-base-text-color);
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
