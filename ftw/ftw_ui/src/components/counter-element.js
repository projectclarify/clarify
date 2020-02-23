/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { LitElement, html, css, property, customElement } from 'lit-element';
// These are the elements needed by this element.
import { plusIcon, minusIcon } from './my-icons';
// These are the shared styles needed by this element.
import { ButtonSharedStyles } from './button-shared-styles';
// This is a reusable element. It is not connected to the store. You can
// imagine that it could just as well be a third-party element that you
// got from someone else.
let CounterElement = class CounterElement extends LitElement {
    constructor() {
        super(...arguments);
        this.clicks = 0;
        this.value = 0;
    }
    static get styles() {
        return [
            ButtonSharedStyles,
            css `
        span {
          width: 20px;
          display: inline-block;
          text-align: center;
          font-weight: bold;
        }
      `
        ];
    }
    render() {
        return html `
      <div>
        <p>
          Clicked: <span>${this.clicks}</span> times.
          Value is <span>${this.value}</span>.
          <button @click="${this._onIncrement}" title="Add 1">${plusIcon}</button>
          <button @click="${this._onDecrement}" title="Minus 1">${minusIcon}</button>
        </p>
      </div>
    `;
    }
    _onIncrement() {
        this.value++;
        this.clicks++;
        this.dispatchEvent(new CustomEvent('counter-incremented'));
    }
    _onDecrement() {
        this.value--;
        this.clicks++;
        this.dispatchEvent(new CustomEvent('counter-decremented'));
    }
};
__decorate([
    property({ type: Number })
], CounterElement.prototype, "clicks", void 0);
__decorate([
    property({ type: Number })
], CounterElement.prototype, "value", void 0);
CounterElement = __decorate([
    customElement('counter-element')
], CounterElement);
export { CounterElement };
