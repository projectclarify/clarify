/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/


import { html, css, property, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store, RootState } from '../store.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

import '/node_modules/@polymer/paper-slider/paper-slider.js';

import { navigate } from '../actions/app.js';

import {
    updateProgressBarState
} from '../actions/app.js';

import '/node_modules/@polymer/paper-button/paper-button.js';

import { navToggleIcon } from './my-icons.js';

import './feedback-radar.js'

@customElement('view-interactive')
export class ViewInteractive extends connect(store)(PageViewElement) {

  @property({type: String})
  _navSection = "pre-session";

  @property({type: Boolean})
  _leftToggleActive = true;

  @property({type: Boolean})
  _sessionHasStarted = false;

  @property({type: Boolean})
  _sessionInProgress = false;

  @property({type: Boolean})
  _sessionComplete = false;

  @property({type: Boolean})
  _postSessionComplete = false;

  @property({type: Array})
  _steps = ["pre-session", "session", "post-session", "congrats"];

  @property({type: Number})
  _stepId = 0;

  @property({type: Number})
  _sessionSecondsElapsed = 0;

  @property({type: Number})
  _sessionTotalSeconds = 0;

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

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

        .dark-background {
          background-color: var(--app-primary-color);
          height: 100%;
          width: 100%;
          position: fixed;
          top: 0px;
          left: 0px;
          padding: none;
          z-index: -999;
        }

        .main-content {
          height: 100%;
          width: 100%;
          position: relative;
          top: 0px;
          left: 0px;
          padding: 0px;
        }

        .left-nav {
            position: fixed;
            left: 0px;
            top: 125px;
            width: 200px;
            display: flex;
            flex-direction: row;
        }

        .left-nav-item {
            font-size: 12px;
            color: white;
            padding-left: 15px;
            padding-top: 7px;
            padding-bottom: 7px;
        }

        .left-nav-item[active="true"] {
            font-weight: 600;
            border-left: 3px solid white;
        }

        .left-nav-title {
          font-size: 12px;
          color: white;
          font-weight: 600;
          padding-bottom: 14px;
        }

        .left-nav-toggle {
          display: none;
        }

        .left-nav-items {

        }

        .left-nav-toggle {
          display: none;
        }

        .left-nav-toggle-inner {
          padding-top: 15px;
          padding-bottom: 15px;
          padding-left: 3px;
          padding-right: 3px;
          border: 1px solid white;
          border-radius: 3px;
          margin-left: 2px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }

        @media (max-width: 1048px) {
    
          .left-nav {
            left: -100px;
            transition-duration: 0.1s;
          }

          .left-nav[active="true"] {
            left: 0px;
          }

          .left-nav-toggle {
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding-left: 15px;
            transform: rotate(0deg);
            transition-duration: 0.1s;
          }

          .left-nav-toggle[active="true"] > .left-nav-toggle-inner > svg {
            transform: rotate(180deg);
            transition-duration: 0.1s;
          }

          .left-nav-toggle > .left-nav-toggle-inner > svg > path {
            fill: white;
          }

          .left-nav {
            display: none;
          }

          .assessment-form-item-title {
            font-size: 12px;
            display: flex;
            flex-direction: column;
            justify-content: center;
          }
            
        }

        .session-step {
          display: block;
          height: 100vh;
        }

        .session-step[active="false"] {
          display: none
        }

        .session-step-container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: row;
            justify-content: center;
            color: white;
        }

        .session-step-inner {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            width: 50vw;
            max-width: 800px
        }

        .session-step-inner-wide {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            width: 75vw;
            max-width: 1200px;
        }

        .session-step-center-top {
            width: 100%;
            text-align: center;
            margin-top: 80px;
        }

        .session-step-center-middle {
            width: 100%;
            text-align: center;
        }

        .session-step-center-bottom-container {
          width: 100%;
          height: 150px;
          display: flex;
          flex-direction: row;
          justify-content: center;
        }

        .session-step-center-bottom {
            text-align: center;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .session-step-pre-title {
            font-size: 16px;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom:10px;
        }

        .session-step-title {
            font-size: 30px;
        }

        .session-step-description {
            font-size: 18px;
            font-weight: 100;
            padding: 20px;
            height: 20px;
        }

        paper-slider {
          --paper-slider-active-color: white;
          --paper-slider-font-color: var(--app-primary-color);
          --paper-slider-knob-color: white;
          --paper-slider-pin-color: white;
          --paper-slider-height: 2px;
        }

        .assessment-form {
            width: 100%;
            padding-top: 50px;
        }

        .assessment-form-item {
          display: flex;
          flex-direction: row;
          justify-content: space-between
          padding-top: 40px;
        }

        .assessment-form-item-title {
          width: 100px;
          text-align: left;
        }

        .assessment-form-item-input {
          width: 100%;
        }



        .session-step-two-up-responsive-layout {

        }

        .session-step-inner {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            width: 50vw;
            max-width: 800px;
        }

        .session-right-container {
          margin-left: 40px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          height: 100%;
        }

        #congrats-inner > * > div {
        }

        .session-timer-container {
          display: flex;
          flex-direction: row;
          justify-content: center;
          width: 100%;
        }

        .session-timer-container > div {
          padding: 20px;
        }

        #session-timer-border {
          stroke-width: 2px;
          stroke: white;
        }

        #session-timer-interior {
          fill: var(--app-primary-color);
          stroke: white;
          stroke-width: 2px;
        }

        .session-step-center-middle-container {
          display: flex;
          flex-direction: row;
          justify-content: center;
          width: 100%;
        }

        feedback-radar {
          width: 400px;
          height: 400px;
        }

        .feedback-radar-wrapper {
          width: 100%;
          display: flex;
          flex-direction: row;
          justify-content: center;
        }

        /* Wide layout: when the viewport width is bigger than 460px, layout
        changes to a wide layout */
        @media (max-width: 1048px) {

          .session-step-center-middle-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
          }

          feedback-radar {
            width: 80vw;
          }

          .session-step-description {
            font-size: 14px;
            font-weight: 100;
            padding: 20px;
            height: 20px;
          }

        }

      `
    ];
  }

  protected renderPreSession() {
      return html`
      
        <div class="session-step-container">

            <div class="session-step-inner">

                <div class="session-step-center-top"></div>

                <div class="session-step-center-middle">
                
                  <div class="session-step-pre-title">Assessment</div>
                  <div class="session-step-title">Pre-session Assessment</div>
                  <div class="session-step-description">Let's do a quick check-in before your meditation session.</div>

                   <div class="assessment-form">

                    <div class="assessment-form-item"><div class="assessment-form-item-title">Happiness</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Focus</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Kindness</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Confidence</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Calm</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>

                   </div>

                </div>

              <div class="session-step-center-bottom-container">
                <div class="session-step-center-bottom">

                  <paper-button class="inverted-primary" raised @click="${this._nextStep}">Continue</paper-button>

                </div>
              </div>

            </div>

        </div>

      `
  }

  protected renderSessionInProgressButtons() {

      return html`${this._sessionHasStarted ? html`
        ${this._sessionInProgress ? html`
            <paper-button class="inverted-secondary" @click="${this._toggleSessionInProgress}">Pause</paper-button>
        ` : html`
            <paper-button class="inverted-primary" @click="${this._toggleSessionInProgress}" raised>Resume</paper-button>
        `}
      `: html`
        <paper-button class="inverted-primary" @click="${this._beginSession}" raised>Begin</paper-button>

      `}`

  }

  protected renderSessionButtons() {

    return html`${this._sessionComplete ? html`
        <paper-button class="inverted-primary" @click="${this._completeSession}" raised @click="${this._nextStep}">Continue</paper-button>
    `:html`
        ${this.renderSessionInProgressButtons()}
    `}`

  }

  protected renderSessionTimer() {
    return html`
        <div class="session-timer-container">

        <div class="session-timer-elapsed">00:00:00</div>
        <div class="session-timer-graphic">
            <svg width="24" height="24" viewbox="0 0 24 24">
            <circle cx="12" cy="12" r="12" fill="white" />
            <path id="session-timer-border" transform="translate(125, 125)"/>
            <path id="session-timer-interior" transform="translate(12, 12) scale(.9)" />
            </svg>
        </div>
        <div class="session-timer-total">02:00:00</div>

        </div>
    `
  }

  renderSessionLeftMiddle(){
    return html`
        <div class="session-step-center-middle">

            <div class="session-step-pre-title">SESSION</div>
            <div class="session-step-title">Meditation Session</div>

            ${this.renderSessionTimer()}

            <div class="session-step-description">
              Show documentation in new tab.
            </div>

        </div>
    `
  }

  renderSmallCircle() {
    return html`
      <svg width="8" height="8" viewbox="0 0 8 8">
        <circle cx="4" cy="4" r="4" fill="white" />
      </svg>
    `
  }

  renderInitialInstructions() {
      return html`
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque ut risus vel sapien porttitor aliquam. Duis non dolor urna. Vestibulum consectetur, sem nec volutpat vehicula, nibh est egestas tortor, a facilisis elit massa vitae ipsum. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nam vulputate nunc risus, a venenatis quam posuere in. Cras auctor urna non leo euismod, tincidunt fermentum mi bibendum.</p>
        `
  }

  protected renderSession() {
      return html`
      
        <div class="session-step-container session-step-two-up-responsive-layout">

            <div class="session-step-inner-wide">

                <div class="session-step-center-top"></div>

                <div class="session-step-center-middle-container">
                
                  ${this.renderSessionLeftMiddle()}

                  <div class="special-spacer" style="width: 20px"></div>

                  <div class="feedback-radar-wrapper">
                    
                    ${this.renderInitialInstructions()}

                  </div>

               </div>

              <div class="session-step-center-bottom-container">

                 <div class="session-step-center-bottom">

                 ${this.renderSessionButtons()}

                 </div>

               </div>
            
            </div>

        </div>

      `
  }

  protected renderPostSession() {
      return html`

        <div class="session-step-container">

            <div class="session-step-inner">

                <div class="session-step-center-top"></div>

                <div class="session-step-center-middle">
                
                  <div class="session-step-pre-title">Assessment</div>
                  <div class="session-step-title">Post-session Assessment</div>
                  <div class="session-step-description">Let's do a quick check-in to conclude your meditation session.</div>

                   <div class="assessment-form">

                    <div class="assessment-form-item"><div class="assessment-form-item-title">Productivity</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Happiness</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Flow</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Focus</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Kindness</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Confidence</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>
                    <div class="assessment-form-item"><div class="assessment-form-item-title">Calm</div><paper-slider value="5" max="10"  pin class="assessment-form-item-input"></paper-slider></div>

                   </div>

                </div>

              <div class="session-step-center-bottom-container">
                <div class="session-step-center-bottom">
                
                <paper-button class="inverted-primary" raised @click="${this._nextStep}">Continue</paper-button>

                </div>
              </div>

            </div>

        </div>

      `
  }

  protected renderCongrats() {
      return html`

        <div class="session-step-container">

            <div class="session-step-inner" id="congrats-inner">

                <div class="session-step-center-top"></div>

                <div class="session-step-center-middle">
                
                  <div class="session-step-pre-title">Session Complete</div>
                  <div class="session-step-title">Congratulations! You completed another work session!</div>
                  <div class="session-step-description">Keep it up, you're a star!</div>
                
                </div>

              <div class="session-step-center-bottom-container">
                <div class="session-step-center-bottom">
        
                    <paper-button class="inverted-primary" raised @click="${this._exitSession}">Return Home ></paper-button>

                </div>
              </div>

            </div>

        </div>

      `
  }

  protected renderNav() {
      return html`
        <div class="left-nav" active="${this._leftToggleActive}">
          <div class="left-nav-items">
            <div class="left-nav-item left-nav-title">OUTLINE</div>
            <div class="left-nav-item" navid="pre-session" @click="${this._handleNavClick}" active="${this._navSection == 'pre-session'}">Pre-session</div>
            <div class="left-nav-item" navid="session" @click="${this._handleNavClick}" active="${this._navSection == 'session'}">Session</div>
            <div class="left-nav-item" navid="post-session" @click="${this._handleNavClick}" active="${this._navSection == 'post-session'}">Post-session</div>
            <div class="left-nav-item" navid="congrats" @click="${this._handleNavClick}" active="${this._navSection == 'congrats'}">Done</div>
          </div>
          <div class="left-nav-toggle" @click="${this._toggleNav}" active="${this._leftToggleActive}">
            <div class="left-nav-toggle-inner">${navToggleIcon}</div>
          </div>
        </div>
      `
  }

  protected render() {
    return html`

      <div class="dark-background"></div>

      <div class="main-content">

        <div class="session-step" navid="pre-session" active="${this._navSection == 'pre-session'}">${this.renderPreSession()}</div>
        <div class="session-step" navid="session" active="${this._navSection == 'session'}">${this.renderSession()}</div>
        <div class="session-step" navid="post-session" active="${this._navSection == 'post-session'}">${this.renderPostSession()}</div>
        <div class="session-step" navid="congrats" active="${this._navSection == 'congrats'}">${this.renderCongrats()}</div>

        ${this.renderNav()}

      </div>

    `;
  }

  firstUpdated() {
    store.dispatch(updateProgressBarState(0));
  }

  _handleNavClick(e: Event) {
    this._navSection = e.path[0].getAttribute("navid");
    /* for the time being don't allow nav by way of nav clicks */
  }

  _toggleNav() {

      this._leftToggleActive = !this._leftToggleActive;
      console.log(this._leftToggleActive);
      /* update the variable that determines whether the nav is shown open or closed
         then preferably animate the transition between the two.
      */

  }

  _handleTestClick() {
    console.log("saw test click")
  }

  _toggleSessionInProgress() {
    this._sessionInProgress = !this._sessionInProgress;
  }

  _beginSession(){
    this._sessionHasStarted = true;
    this._toggleSessionInProgress();
    this._drawTimer();
    this.shadowRoot.querySelector(".feedback-radar-wrapper").innerHTML = "<feedback-radar></feedback-radar>";
  }

  _nextStep() {
    if (this._stepId < this._steps.length - 1)  {
      this._stepId = this._stepId + 1;
      this._navSection = this._steps[this._stepId];

      // Ah idk not really a fan.
      //store.dispatch(updateProgressBarState(this._stepId / this._steps.length));

    }
  }

  _exitSession() {
    this._navSection = "pre-session";
    this._stepId = 0;
    //store.dispatch(updateProgressBarState(0));
    store.dispatch(navigate("/dashboard"));
  }

  _drawTimer() {

    let interior = this.shadowRoot!.querySelector("#session-timer-interior");
    let border = this.shadowRoot!.querySelector("#session-timer-border");
    let alpha = 0;
    let pi = Math.PI;
    let t = 1000;
    let cc = 0;

    (function draw() {

      alpha += 6;
      cc += 6;
      alpha %= 360;

      let r = ( alpha * pi / 180 );
      let x = Math.sin( r ) * 12;
      let y = Math.cos( r ) * - 12;
      let mid = ( alpha > 180 ) ? 1 : 0;
      
      let anim = 'M 0 0 v -12 A 12 12 1 '
        + mid + ' 1 '
        +  x  + ' '
        +  y  + ' z';

      interior!.setAttribute( 'd', anim );
      border!.setAttribute( 'd', anim );
    
      if (cc < 360) {
        setTimeout(draw, t);
      }

    })();

  }


}

