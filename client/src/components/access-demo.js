var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { LitElement, css, html, customElement } from 'lit-element';
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
import { navigate } from '../actions/app.js';
import { store } from '../store.js';
let AccessDemo = class AccessDemo extends (LitElement) {
    static get styles() {
        return [
            SharedStyles,
            ButtonSharedStyles,
            css `

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

      `
        ];
    }
    render() {
        return html `

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
};
AccessDemo = __decorate([
    customElement('access-demo')
], AccessDemo);
export { AccessDemo };
