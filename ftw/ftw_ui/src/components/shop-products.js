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
import './shop-item';
// These are the actions needed by this element.
import { getAllProducts, addToCart } from '../actions/shop';
// These are the elements needed by this element.
import { addToCartIcon } from './my-icons';
// These are the shared styles needed by this element.
import { ButtonSharedStyles } from './button-shared-styles';
let ShopProducts = class ShopProducts extends connect(store)(LitElement) {
    constructor() {
        super(...arguments);
        this._products = {};
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
      ${Object.keys(this._products).map((key) => {
            const item = this._products[key];
            return html `
          <div>
            <shop-item name="${item.title}" amount="${item.inventory}" price="${item.price}"></shop-item>
            <button
                .disabled="${item.inventory === 0}"
                @click="${this._addButtonClicked}"
                data-index="${item.id}"
                title="${item.inventory === 0 ? 'Sold out' : 'Add to cart'}">
              ${item.inventory === 0 ? 'Sold out' : addToCartIcon}
            </button>
          </div>
        `;
        })}
    `;
    }
    firstUpdated() {
        store.dispatch(getAllProducts());
    }
    _addButtonClicked(e) {
        store.dispatch(addToCart(e.currentTarget.dataset['index']));
    }
    // This is called every time something is updated in the store.
    stateChanged(state) {
        this._products = state.shop.products;
    }
};
__decorate([
    property({ type: Object })
], ShopProducts.prototype, "_products", void 0);
ShopProducts = __decorate([
    customElement('shop-products')
], ShopProducts);
export { ShopProducts };
