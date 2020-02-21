
import { LitElement, css, html, property, customElement } from 'lit-element';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
import { navigate } from '../actions/app.js';
import { store } from '../store.js';

@customElement('access-demo')
export class AccessDemo extends (LitElement) {

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

        .article-large-width-image {
            position: relative;
        }

        .article-large-width-image > svg {
          margin: 10px;
          flex-shrink: 0;
          width: 80%;
          height: auto;
        }

        .access-demo-container {
            width: 100%;
            display: flex;
            flex-direction: row;
            justify-content: center;
        }

        .access-demo {
            height: 100%;
            display:flex;
            flex-direction: column;
        }

        .access-demo-title {
            font-size: 14px;
            font-weight: 100;
            text-align: center;
        }

        .access-demo-more-info {
            font-size: 10px;
            font-weight: 100;
            text-align: center;
        }

        a {
            color: var(--app-primary-color);
            cursor: pointer;
            text-decoration: underline;
        }

        .pew-inner > paper-button {
        }

      `];
  }

  protected render() {
    return html`

      <div class="access-demo-container">

        <div class="access-demo">

          <p class="access-demo-title">We're building a demo site!</p>
          <p class="access-demo-more-info"><a @click=${this.navDesign}>Read more</a> about the design.</p>
          <paper-button class="inverted-primary" @click="${this.navAccessDemo}">Request access</paper-button>

        </div>

      </div>

    `;
  }

  navDesign() {
    store.dispatch(navigate("/interface-design"));
  }

  navAccessDemo() {
    window.open('https://forms.gle/n5HZdeyYCNbGcY7T9', '_blank');
  }

}
