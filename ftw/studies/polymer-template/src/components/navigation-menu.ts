import { LitElement, css, html, property, customElement } from 'lit-element';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

@customElement('navigation-menu')
export class NavigationMenu extends LitElement {

  @property({ type: String })
  category = "";

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

      `];
  }

  protected render() {
    return html`


    `;
  }
}
