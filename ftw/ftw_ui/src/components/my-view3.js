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
import { html, css, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element';
import { connect } from 'pwa-helpers/connect-mixin.js';
// This element is connected to the Redux store.
import { store } from '../store';
// These are the actions needed by this element.
import { checkout } from '../actions/shop';
// We are lazy loading its reducer.
import shop, { cartQuantitySelector } from '../reducers/shop';
store.addReducers({
    shop
});
// These are the elements needed by this element.
import './shop-products';
import './shop-cart';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles';
import { ButtonSharedStyles } from './button-shared-styles';
import { addToCartIcon } from './my-icons';
let MyView3 = class MyView3 extends connect(store)(PageViewElement) {
    constructor() {
        super(...arguments);
        this._quantity = 0;
        this._error = '';
    }
    static get styles() {
        return [
            SharedStyles,
            ButtonSharedStyles,
            css `
        button {
          border: 2px solid var(--app-dark-text-color);
          border-radius: 3px;
          padding: 8px 16px;
        }

        button:hover {
          border-color: var(--app-primary-color);
          color: var(--app-primary-color);
        }

        .cart,
        .cart svg {
          fill: var(--app-primary-color);
          width: 64px;
          height: 64px;
        }

        .circle.small {
          margin-top: -72px;
          width: 28px;
          height: 28px;
          font-size: 16px;
          font-weight: bold;
          line-height: 30px;
        }
      `
        ];
    }
    render() {
        return html `
      <section>
        <h2>Redux example: shopping cart</h2>
        <div class="cart">${addToCartIcon}<div class="circle small">${this._quantity}</div></div>
        <p>This is a slightly more advanced Redux example, that simulates a
          shopping cart: getting the products, adding/removing items to the
          cart, and a checkout action, that can sometimes randomly fail (to
          simulate where you would add failure handling). </p>
        <p>This view, as well as its 2 child elements, <code>&lt;shop-products&gt;</code> and
        <code>&lt;shop-cart&gt;</code> are connected to the Redux store.</p>
      </section>
      <section>
        <h3>Products</h3>
        <shop-products></shop-products>

        <br>
        <h3>Your Cart</h3>
        <shop-cart></shop-cart>

        <div>${this._error}</div>
        <br>
        <p>
          <button ?hidden="${this._quantity == 0}" @click="${this._checkoutButtonClicked}">
            Checkout
          </button>
        </p>
      </section>
    `;
    }
    _checkoutButtonClicked() {
        store.dispatch(checkout());
    }
    // This is called every time something is updated in the store.
    stateChanged(state) {
        this._quantity = cartQuantitySelector(state);
        this._error = state.shop.error;
    }
};
__decorate([
    property({ type: Number })
], MyView3.prototype, "_quantity", void 0);
__decorate([
    property({ type: String })
], MyView3.prototype, "_error", void 0);
MyView3 = __decorate([
    customElement('my-view3')
], MyView3);
export { MyView3 };
