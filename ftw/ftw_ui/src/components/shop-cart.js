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
import { connect } from 'pwa-helpers/connect-mixin.js';
// This element is connected to the Redux store.
import { store } from '../store';
// These are the elements needed by this element.
import { removeFromCartIcon } from './my-icons';
import './shop-item';
// These are the actions needed by this element.
import { removeFromCart } from '../actions/shop';
// These are the reducers needed by this element.
import { cartItemsSelector, cartTotalSelector } from '../reducers/shop';
// These are the shared styles needed by this element.
import { ButtonSharedStyles } from './button-shared-styles';
let ShopCart = class ShopCart extends connect(store)(LitElement) {
    constructor() {
        super(...arguments);
        this._items = [];
        this._total = 0;
    }
    static get styles() {
        return [
            ButtonSharedStyles,
            css `
        :host {
          display: block;
        }
      `
        ];
    }
    render() {
        return html `
      <p ?hidden="${this._items.length !== 0}">Please add some products to cart.</p>
      ${this._items.map((item) => html `
          <div>
            <shop-item .name="${item.title}" .amount="${item.amount}" .price="${item.price}"></shop-item>
            <button
                @click="${this._removeButtonClicked}"
                data-index="${item.id}"
                title="Remove from cart">
              ${removeFromCartIcon}
            </button>
          </div>
        `)}
      <p ?hidden="${!this._items.length}"><b>Total:</b> ${this._total}</p>
    `;
    }
    _removeButtonClicked(e) {
        store.dispatch(removeFromCart(e.currentTarget.dataset['index']));
    }
    // This is called every time something is updated in the store.
    stateChanged(state) {
        this._items = cartItemsSelector(state);
        this._total = cartTotalSelector(state);
    }
};
__decorate([
    property({ type: Array })
], ShopCart.prototype, "_items", void 0);
__decorate([
    property({ type: Number })
], ShopCart.prototype, "_total", void 0);
ShopCart = __decorate([
    customElement('shop-cart')
], ShopCart);
export { ShopCart };
