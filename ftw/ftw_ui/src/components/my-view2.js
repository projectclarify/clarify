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
import { html, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element';
import { connect } from 'pwa-helpers/connect-mixin.js';
// This element is connected to the Redux store.
import { store } from '../store';
// These are the actions needed by this element.
import { increment, decrement } from '../actions/counter';
// We are lazy loading its reducer.
import counter from '../reducers/counter';
store.addReducers({
    counter
});
// These are the elements needed by this element.
import './counter-element';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles';
let MyView2 = class MyView2 extends connect(store)(PageViewElement) {
    constructor() {
        super(...arguments);
        this._clicks = 0;
        this._value = 0;
    }
    static get styles() {
        return [
            SharedStyles
        ];
    }
    render() {
        return html `
      <section>
        <h2>Redux example: simple counter</h2>
        <div class="circle">${this._value}</div>
        <p>This page contains a reusable <code>&lt;counter-element&gt;</code>. The
        element is not built in a Redux-y way (you can think of it as being a
        third-party element you got from someone else), but this page is connected to the
        Redux store. When the element updates its counter, this page updates the values
        in the Redux store, and you can see the current value of the counter reflected in
        the bubble above.</p>
        <br><br>
      </section>
      <section>
        <p>
          <counter-element
              value="${this._value}"
              clicks="${this._clicks}"
              @counter-incremented="${this._counterIncremented}"
              @counter-decremented="${this._counterDecremented}">
          </counter-element>
        </p>
      </section>
    `;
    }
    _counterIncremented() {
        store.dispatch(increment());
    }
    _counterDecremented() {
        store.dispatch(decrement());
    }
    // This is called every time something is updated in the store.
    stateChanged(state) {
        this._clicks = state.counter.clicks;
        this._value = state.counter.value;
    }
};
__decorate([
    property({ type: Number })
], MyView2.prototype, "_clicks", void 0);
__decorate([
    property({ type: Number })
], MyView2.prototype, "_value", void 0);
MyView2 = __decorate([
    customElement('my-view2')
], MyView2);
export { MyView2 };
