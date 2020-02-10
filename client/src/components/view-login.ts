/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { html, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store } from '../store.js';

import '/node_modules/@polymer/paper-input/paper-input.js';
import '/node_modules/@polymer/paper-button/paper-button.js';

// We are lazy loading its reducer.
import counter from '../reducers/counter.js';
store.addReducers({
  counter
});

import { firebase } from './firebase.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';

@customElement('view-login')
export class ViewLogin extends connect(store)(PageViewElement) {

  @property({type: Boolean})
  private _viewStateUserNotCrowd = true

  @property({type: Object})
  _user = '';

  static get styles() {
    return [
      SharedStyles
    ];
  }

  protected _handleCrowdLogin() {
      console.log("Crowd login button clicked.")
  }

  protected renderLoggedIn() {
    return html`
      <section class="article-hero">
        <p class="article-hero-heading">Welcome to the Demo!</p>
        <p class="article-hero-subheading">It's a work in progress but will be updated periodically. See "Dashboard" above to check it out or our <a href="/interface-design">interface design</a> article to see where we're headed.</p>
      </section>
    `;
  }

  protected renderLoginPage() {
    return html`
    ${this._viewStateUserNotCrowd ?
    html`

      <section class="article-hero">
        <p class="article-hero-pre-heading">ACCOUNT</p>
        <p class="article-hero-heading">User sign in</p>
        <p class="article-hero-subheading">Sign in to experience demos and participate in research</p>
        <paper-button class="primary-btn" @click="${this._handleSignInWithGoogle}">Sign in with Google</paper-button>
 
        <!--
        <paper-button class="secondary-btn secondary-login-btn" @click="${this._handleSignInWithGitHub}">Sign in with GitHub</paper-button>
        -->

        <p id="login-state-toggle" @click="${this._toggleLoginViewState}">Have a work unit ID? <a>Sign in</a> as a crowd worker.</p>
      </section>

      ` : html`

      <section class="article-hero">
        <p class="article-hero-pre-heading">ACCOUNT</p>
        <p class="article-hero-heading">Crowd worker sign in</p>
        <p class="article-hero-subheading">Sign in as a crowd worker with work unit ID</p>
        <paper-input always-float-label label="Please enter your work ID..."></paper-input>
        <paper-button class="primary-btn" @click="${this._handleCrowdLogin}">Sign in as Crowd</paper-button>
        <p id="login-state-toggle" @click="${this._toggleLoginViewState}"><a>Sign in</a> as a regular user.</p>
      </section>

    `
    }
    `;
  }

  protected render() {
    return html`${this._user ? html`${this.renderLoggedIn()}` : html`${this.renderLoginPage()}` }`;
  }

  private _toggleLoginViewState() {
    this._viewStateUserNotCrowd = !this._viewStateUserNotCrowd;
  }

  private _handleSignInWithGoogle() {
    var provider = new firebase.auth.GoogleAuthProvider();
    firebase.auth().signInWithPopup(provider);
  }

  private _handleSignInWithGitHub() {
    var provider = new firebase.auth.GithubAuthProvider();
    firebase.auth().signInWithPopup(provider);
  }

  /*
  // This is called every time something is updated in the store.
  stateChanged(state: RootState) {
    this._user = state.user!.currentUser;
  }
  */

  handleDemoAccess() {
      console.log("handling demo access")
  }

}
