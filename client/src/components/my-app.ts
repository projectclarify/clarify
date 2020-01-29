/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { LitElement, html, css, property, PropertyValues, customElement } from 'lit-element';
import { setPassiveTouchGestures } from '@polymer/polymer/lib/utils/settings.js';
import { connect } from 'pwa-helpers/connect-mixin.js';
import { installMediaQueryWatcher } from 'pwa-helpers/media-query.js';
import { installOfflineWatcher } from 'pwa-helpers/network.js';
import { installRouter } from 'pwa-helpers/router.js';
import { updateMetadata } from 'pwa-helpers/metadata.js';

// This element is connected to the Redux store.
import { store, RootState } from '../store.js';

import { ButtonSharedStyles } from './button-shared-styles.js';

// These are the actions needed by this element.
import {
    navigate,
    updateOffline,
    updateDrawerState,
    updateAccountDropdownState
} from '../actions/app.js';

// The following line imports the type only - it will be removed by tsc so
// another import for app-drawer.js is required below.
import { AppDrawerElement } from '@polymer/app-layout/app-drawer/app-drawer.js';

// These are the elements needed by this element.
import '@polymer/app-layout/app-drawer/app-drawer.js';
import '@polymer/app-layout/app-header/app-header.js';
import '@polymer/app-layout/app-scroll-effects/effects/waterfall.js';
import '@polymer/app-layout/app-toolbar/app-toolbar.js';

import './clarify-icon.js';

import '/node_modules/@polymer/paper-icon-button/paper-icon-button.js';

import { logoutIcon } from './my-icons.js';
import './snack-bar.js';

import { firebase } from './firebase.js';

@customElement('my-app')
export class MyApp extends connect(store)(LitElement) {
    @property({ type: String })
    appTitle = '';

    @property({ type: String })
    private _page = '';

    @property({ type: Boolean })
    private _drawerOpened = false;

    @property({ type: Boolean})
    private _accountDropdownOpened = false;

    @property({ type: Boolean })
    private _snackbarOpened = false;

    @property({ type: Boolean })
    private _offline = false;

    @property({type: Object})
    _user = '';

    @property({type: Boolean})
    _toolbarLight = true;

    @property({type: Boolean})
    _loading = true;

    @property({type: Number})
    _progress = 0;

    @property({type: String})
    _progressLabel = "";

    @property({type: Boolean})
    _dashboardActive = true;

    static get styles() {
        return [
            ButtonSharedStyles,
            css`
        :host {
          display: block;

          --app-drawer-width: 225px;

          --app-primary-color: #3c78d8;
          --app-secondary-color: dimgrey;
          --app-dark-text-color: var(--app-secondary-color);
          --app-light-text-color: white;
          --app-section-even-color: #f7f7f7;
          --app-section-odd-color: white;

          --app-header-background-color: white;
          --app-header-text-color: var(--app-dark-text-color);
          --app-header-selected-color: var(--app-primary-color);
          --app-header-logo-color: #666666;

          --app-drawer-background-color: transparent;
          --app-drawer-text-color: black;
          --app-drawer-selected-color: #6d9eeb;
          --app-drawer-scrim-background: transparent;

        }


        @media (max-width: 1048px) {
    
          :host {

            --app-header-background-color: white;

          }

        }

        app-header {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          text-align: center;
          background-color: var(--app-header-background-color);
          color: var(--app-header-text-color);
          border-bottom: 1px solid #fafafa;
        }

        app-header[inverted="true"] {
          border-bottom: none;
        }

        .toolbar-top {
          background-color: var(--app-header-background-color);
          text-align: left;
        }

        .progress-bar {
          background-color: var(--app-primary-color); //#3c78d8ff;
          height: 2.5px;
          position: fixed;
          top: 64px;
        }

        .progress-bar[inverted="true"]{
          background-color: white;
        }

        .progress-text {
           position: absolute;
           top: 0px;
           height: 64px;
           left: 50%;
           display: flex;
           flex-direction: column;
           justify-content: center;
           color: var(--app-primary-color);
        }

        .progress-text[inverted="true"] {
          color: white;
        }

        [main-title] {
          font-family: 'IBM Plex Sans',sans-serif;
          text-transform: none;
          font-size: 16px;
          /* In the narrow layout, the toolbar is offset by the width of the
          drawer button, and the text looks not centered. Add a padding to
          match that button */
          padding-right: 44px;
          padding-left: 42px;
        }

        /* When we are to display the toolbar in dark mode */
        .toolbar-top[light="false"] {
          background-color: #3c78d8ff;
          color: white;
        }

        .toolbar-list {
          display: none;
        }

        .toolbar-list > a {
          display: inline-block;
          color: var(--app-header-text-color);
          text-decoration: none;
          line-height: 30px;
          padding: 4px 24px;
        }

        .toolbar-list > a[selected] {
          color: var(--app-header-selected-color);
          border-bottom: 4px solid var(--app-header-selected-color);
        }

        .menu-btn {
          background: none;
          border: none;
          fill: var(--app-header-logo-color);
          cursor: pointer;
          height: 44px;
          width: 44px;
        }

        .drawer-list {
          box-sizing: border-box;
          width: 100%;
          height: 100%;
          padding: 0px;
          background: var(--app-drawer-background-color);
          position: relative;
          border-right: 1px solid #fafafa;
        }

        app-drawer {
          app-drawer-scrim-background: transparent;
          z-index:1000;
        }

        .drawer-list > a {
          display: block;
          text-decoration: none;
          color: #434343;
          line-height: 40px;
          padding: 0px 0px 0px 28px;
          font-size: 14px;
        }

        .drawer-list > a[selected] {
          color: var(--app-drawer-selected-color);
          font-weight: 600;
          border-left: 4px solid #0168FA;
        }

        .drawer-section-divider {
          text-transform: uppercase;
          font-weight: 600;
          font-size: 14px;
          padding: 24px 24px 12px 28px;
          color: #434343;
        }

        /* Workaround for IE11 displaying <main> as inline */
        main {
          display: block;
        }

        .main-content {
          padding-top: 64px;
          min-height: 100vh;
        }

        .page {
          display: none;
        }

        .page[active] {
          display: block;
        }

        .login-btn {
            background-color: #0168FA;
            border: none;
            color: #FFFFFF;
            padding: 6.25px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 11px;
            text-transform: uppercase;
            margin: 0px 0px 0px 0px;
            line-height: 1.773;
            font-weight: 700;
            border-radius: 3px;
        }

        .logout-btn {
            background-color: white;
            border: none;
            color: black;
            padding: 6.25px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 11px;
            text-transform: uppercase;
            margin: 0px 0px 0px 0px;
            line-height: 1.773;
            font-weight: 700;
            border-radius: 3px;
        }

        footer {
          position: fixed;
          bottom: 0px;
          padding: 12px;
          background: var(--app-drawer-background-color);
          color: var(--app-drawer-text-color);
          text-align: center;
          font-size: 10px;
          width: 100%;
        }

        .footer-link {
            font-weight: 500;
            color:#cccccc;
            padding-left: 4px;
            padding-right: 4px;
            text-decoration: none;
        }

        .footer-divider {
            color: #eeeeee;
        }

        .account-dropdown {
            position: fixed;
            top: 64px;
            right: 32px;
            padding: 0px;
            border: 1px solid #efefef;
            background-color: white;
        }

        .account-dropdown-item {
          display: flex;
          flex-direction: horizontal;
          cursor: pointer;
          padding: 10px;
        }

        .account-dropdown-item:hover {
          background-color: #dbf3ff
        }

        .account-dropdown-item > p {
          margin: 0px;
          font-size: 12px;
          margin-bottom: 4px;
          margin-left: 10px;
        }

        .account-dropdown-item > svg {
          height: 18px;
          width: 18px;
          fill: #545454
        }

        .account-dropdown-toggle {
          height: 40px;
          width: 40px;
          border-radius: 50%;
          border: 2.5px solid white;
          background-size: 100%;
        }

        .main-content {
          padding-top: 0px
        }

        /* Wide layout: when the viewport width is bigger than 460px, layout
        changes to a wide layout */
        @media (min-width: 99999px) {
          .toolbar-list {
            display: block;
          }

          .menu-btn {
            display: none;
          }

          .main-content {
            padding-top: 107px;
          }

          /* The drawer button isn't shown in the wide layout, so we don't
          need to offset the title */
          [main-title] {
            padding-right: 0px;
          }
        }
      `
        ];
    }

    protected render() {

        this.appTitle = this._parsePageToTitle(this._page);

        return html`
      <!-- Header -->
      <app-header fixed inverted="${!this._toolbarLight}">
        <app-toolbar class="toolbar-top" light="${this._toolbarLight}">
          <button class="menu-btn" title="Menu" @click="${this._menuButtonClicked}"><clarify-icon></clarify-icon></button>
          <div main-title>${this.appTitle}</div>
          ${this._user ?
            html`
            ${this._toolbarLight ? html`
              <button class="secondary-light-toolbar-button" title="Dashboard" @click="${this._navDashView}" inverted="${!this._toolbarLight}">Dashboard</button>
              `: html`
              <button class="secondary-light-toolbar-button" title="Dashboard" @click="${this._navDashView}" inverted="${!this._toolbarLight}">Exit session</button>
              `}
            <div class="account-dropdown-toggle" style="background-image: url('${this._user.photoURL}')" alt="account" title="account"  @click="${this._toggleAccountDropdown}" noink></div>
            <!--<button class="logout-btn" title="Logout" @click="${this._logoutButtonClicked}">Sign out</button>-->
            ` : html`
            <button class="login-btn" title="Login" @click="${this._navLogin}">Sign In</button>
            <!--<button class="login-btn" title="Login" @click="${this._loginButtonClicked}">Sign In</button>-->
            `
            }
        </app-toolbar>

      </app-header>

      <div class="progress-bar" inverted="${!this._toolbarLight}"></div>

      <!-- Account dropdown -->
      <div class="account-dropdown" ?hidden=${!this._accountDropdownOpened}>
        <div class="account-dropdown-item" @click="${this._logoutButtonClicked}">${logoutIcon}<p>Sign Out</p></div>
      </div>

      <!-- Drawer content -->
      <app-drawer
          .opened="${this._drawerOpened}"
          @opened-changed="${this._drawerOpenedChanged}">
        <nav class="drawer-list">

          <div class="drawer-section-divider">Articles</div>
          <a ?selected="${this._page === 'project-clarify'}" href="/project-clarify">Project Clarify</a>
          <a ?selected="${this._page === 'technologies'}" href="/technologies">Technologies</a>
          <a ?selected="${this._page === 'interface-design'}" href="/interface-design">Interface Design</a>
          <a ?selected="${this._page === 'machine-learning'}" href="/machine-learning">Machine Learning</a>
    
          <div class="drawer-section-divider">Contributor</div>
          <a ?selected="${this._page === 'residency-program'}" href="/residency-program">Residency Program</a>

        </nav>
      </app-drawer>

      <main role="main" class="main-content">

        <view-article-project-clarify class="page" ?active="${this._page === 'project-clarify'}"></view-article-project-clarify>
        <view-article-technologies class="page" ?active="${this._page === 'technologies'}"></view-article-technologies>
        <view-dashboard class="page" ?active="${this._page === 'dashboard'}"></view-dashboard>
        <view-login class="page" ?active="${this._page === 'login'}"></view-login>
        <view-privacy-policy class="page" ?active="${this._page === 'privacy-policy'}"></view-privacy-policy>
        <view-terms-of-service class="page" ?active="${this._page === 'terms-of-service'}"></view-terms-of-service>
        <view-article-hackathons class="page" ?active="${this._page === 'hack'}"></view-article-hackathons>
        <my-view404 class="page" ?active="${this._page === 'thats-a-404'}"></my-view404>
        <view-data class="page" ?active="${this._page === 'data'}"></view-data>
        <view-documentation class="page" ?active="${this._page === 'documentation'}"></view-documentation>
        <view-interactive class="page" ?active="${this._page === 'guided-meditation'}"></view-interactive>
        <view-article-interface-design class="page" ?active="${this._page === 'interface-design'}"></view-article-interface-design>
        <view-article-residency-program class="page" ?active="${this._page === 'residency-program'}"></view-article-residency-program>
        <view-article-machine-learning class="page" ?active="${this._page === 'machine-learning'}"></view-article-machine-learning>

        <!-- Technical articles -->
        <view-article-scaling-tpu-input class="page" ?active="${this._page === 'scaling-tpu-input'}"></view-article-scaling-tpu-input>

      </main>

      <footer>
        <a class="footer-link" href="/terms-of-service">terms</a>
        <a class="footer-divider">/</a>
        <a class="footer-link" href="/privacy-policy">privacy</a>
        <a class="footer-divider">/</a>
        <a class="footer-link" href="https://neuroscape.ucsf.edu/" target="_blank">neuroscape</a>
      </footer>

      <snack-bar ?active="${this._snackbarOpened}">
        You are now ${this._offline ? 'offline' : 'online'}.
      </snack-bar>

    `;
    }

    constructor() {
        super();
        setPassiveTouchGestures(true);
        this.addEventListener('click', this._outsideAccountDropdownClickDetector);
    }

    protected firstUpdated() {
        installRouter((location) => store.dispatch(navigate(decodeURIComponent(location.pathname))));
        installOfflineWatcher((offline) => store.dispatch(updateOffline(offline)));
        installMediaQueryWatcher(`(min-width: 460px)`,
            () => store.dispatch(updateDrawerState(false)));
    }

    protected updated(changedProps: PropertyValues) {
        if (changedProps.has('_page')) {
            const pageTitle = this.appTitle;  //+ ' - ' + this._page;
            updateMetadata({
                title: pageTitle,
                description: pageTitle
                // This object also takes an image property, that points to an img src.
            });
        }
    }

    private _menuButtonClicked() {
        store.dispatch(updateDrawerState(true));
    }

    private _drawerOpenedChanged(e: Event) {
        store.dispatch(updateDrawerState((e.target as AppDrawerElement).opened));
    }

    private _loginButtonClicked() {
      console.log(`Login button clicked.`);
      var provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth().signInWithPopup(provider);
    }

    private _parsePageToTitle(page: String) {
        let pageParts = page.split("-");
        let name = "";
        for (let part of pageParts) {
          if(name.length > 0){
            name = name + " ";
          }
          name = name + part.charAt(0).toUpperCase() + part.slice(1)
        }
        return name
    }

    _handle_progress_change(progress: number) {
        if (progress && progress != this._progress){
          let progressBar = this.shadowRoot!.querySelector(".progress-bar");
          let widthPct = progress * 100.0;
          progressBar.style.width = `${widthPct}%`;
        }

        this._progress = progress;

    }

    stateChanged(state: RootState) {
        this._page = state.app!.page;
        this._offline = state.app!.offline;
        this._snackbarOpened = state.app!.snackbarOpened;
        this._drawerOpened = state.app!.drawerOpened;
        this._user = state.user!.currentUser;
        this._accountDropdownOpened = state.app!.accountDropdownOpened;
        this._toolbarLight = state.app!.toolbarLight;
        this._loading = state.app!.loading;

        /* Should be moved to render() */
        this._handle_progress_change(state.app!.progress);

        this._progressLabel = state.app!.progressLabel;

        this._dashboardActive = state.app!.dashboardActive;


    }

    private _logoutButtonClicked() {
      store.dispatch(updateAccountDropdownState(false));
      store.dispatch(navigate("/"));
      firebase.auth().signOut();
    }

    _outsideAccountDropdownClickDetector(e: Event) {

        let sourceElementClass = e.path[0].className;
        if(sourceElementClass != "account-dropdown-toggle") {
          store.dispatch(updateAccountDropdownState(false));
        }

    }

    _navLogin(){
        store.dispatch(navigate("/login"));
    }

    _toggleAccountDropdown() {
        store.dispatch(updateAccountDropdownState(!this._accountDropdownOpened));
    }

    /*
    _navDashboard() {
        store.dispatch(navigate("/dashboard"));
    }*/

    _navDataView() {
      store.dispatch(navigate("/data"));
    }

    _navDashView() {
      store.dispatch(navigate("/dashboard"));
    }


/*

- when there is a change in this._progress, select .progress-bar and change its width %
  accordingly.

*/


}
