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
import { css, html, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
let MyView404 = class MyView404 extends PageViewElement {
    static get styles() {
        return [
            SharedStyles,
            css `
      .article-hero {
        background: url("https://images.unsplash.com/photo-1562873658-15d15ab42c13?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2550&q=80");
        background-repeat: no-repeat;
        background-size: 100% 100%;
      }

      .article-hero-heading {
          color: white
      }
      `
        ];
    }
    render() {
        return html `
      <section class="article-hero">
        <p class="article-hero-heading">"If we knew what it was we were doing, it would not be called research, would it?" -A.E.</p>
      </section>
    `;
    }
};
MyView404 = __decorate([
    customElement('my-view404')
], MyView404);
export { MyView404 };
