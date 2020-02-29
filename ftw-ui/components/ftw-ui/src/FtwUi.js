import { LitElement, html, css } from 'lit-element';
import { classMap } from 'lit-html/directives/class-map.js';

import '../../core/views/view-account/view-account.js';
import '../../core/views/view-login/view-login.js';
import '../../core/views/view-privacy/view-privacy.js';
import '../../core/views/view-sessions-base/view-sessions-base.js';
import '../../core/views/view-tos/view-tos.js';

export class FtwUi extends LitElement {
  static get properties() {
    return {
      title: { type: String },
      page: { type: String },
    };
  }

  static get styles() {
    return css`
      :host {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        font-size: calc(10px + 2vmin);
        color: #1a2b42;
        max-width: 960px;
        margin: 0 auto;
      }

      header {
        width: 100%;
        background: #fff;
        border-bottom: 1px solid #ccc;
      }

      header ul {
        display: flex;
        justify-content: space-around;
        min-width: 400px;
        margin: 0 auto;
        padding: 0;
      }

      header ul li {
        display: flex;
      }

      header ul li a {
        color: #5a5c5e;
        text-decoration: none;
        font-size: 18px;
        line-height: 36px;
      }

      header ul li a:hover,
      header ul li a.active {
        color: blue;
      }

      main {
        flex-grow: 1;
      }

      .app-footer {
        font-size: calc(12px + 0.5vmin);
        align-items: center;
      }

      .app-footer a {
        margin-left: 5px;
      }
    `;
  }

  constructor() {
    super();
    this.page = 'sessions';
  }

  render() {
    return html`
      <header>
        <ul>
          <li>
            <a href="#sessions" class=${this.__navClass('sessions')} @click=${this.__onNavClicked}>
              Sessions
            </a>
          </li>
          <li>
            <a href="#account" class=${this.__navClass('account')} @click=${this.__onNavClicked}>
              Account
            </a>
          </li>
        </ul>
      </header>

      <main>${this._renderPage()}</main>

      <p class="app-footer"></p>
    `;
  }

  _renderPage() {
    switch (this.page) {
      case 'sessions':
        return html`
          <view-sessions></view-sessions>
        `;
      case 'account':
        return html`
          <view-account></view-account>
        `;
      default:
        return html`
          <p>Page not found try going to <a href="#main">Main</a></p>
        `;
    }
  }

  __onNavClicked(ev) {
    ev.preventDefault();
    this.page = ev.target.hash.substring(1);
  }

  __navClass(page) {
    return classMap({ active: this.page === page });
  }
}
