
/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { html, css, property, customElement, LitElement } from 'lit-element';
import { connect } from 'pwa-helpers/connect-mixin.js';

import { store, RootState } from '../store.js';

@customElement('clarify-icon')
export class ClarifyIcon extends connect(store)(LitElement) {

  @property({ type: Boolean })
  _loading = false;

  @property({ type: Boolean })
  _toolbarLight = false;

  static get styles() {
    return [
      css`

        .icon-container{
            width: 42px;
            height: 42px;
            transform: rotate(45deg);
            transition: 1s;
            transition: transform .3s ease-in-out;
        }
        .icon-container svg circle {
            stroke: #cccccc;
            fill: none;
            stroke-width: 0.5rem;
            stroke-dasharray: 815;
            stroke-dashoffset: 700;
            transition: transform .3s ease-in-out;
        }

        .icon-container[lightbg="false"] svg circle {
          stroke: white;
        }

        .icon-container[loading="true"] {
          animation: container-loading 2s ease-in-out infinite;
        }

        .icon-container[loading="true"] svg circle {
          animation: circle-loading 2s ease-in-out infinite;
        }

        @keyframes container-loading {
        
        0% {
            transform: rotate(45deg);
        }
        
        49.9% {
            transform: rotate(405deg);
        }
        
        50% {
            transform: rotate(45deg);
        }

        99.9% {
            transform: rotate(405deg);
        }
        
        100% {
            transform: rotate(45deg);
        }

        }


        @keyframes circle-loading {
        
        0% {
            stroke-dasharray: 815;
        }

        50% {
            stroke-dasharray: 700;
        }

        100% {
            stroke-dasharray: 815;
        }

        }

      `
    ];
  }

  protected render() {
    return html`

      <div class="icon-container" loading="${this._loading}" lightbg="${this._toolbarLight}">
        <svg width="100%" height="100%" viewBox="-1 -1 103 103">
           <circle cx="51" cy="51" r="24"/>
        </svg>
      </div>

    `;
  }

  stateChanged(state: RootState) {
    this._loading = state.app!.loading;
    this._toolbarLight = state.app!.toolbarLight;
  }

  /*
    Options:

    1. use CSS, when the app loading state changes, loading="${this._loading}" changes
      along with the corresponding CSS-linked html element attribute.

      Potential issue: don't want to terminate an animation abruptly.

    2. Use a javascript function to trigger the animation to avoid the former issue.


    - bind this._loading to loading=

    - use css to repeat forever the animation

    - respond to a change in _toolbarLight with a change in the styling of the icon
      border.

  */

}

