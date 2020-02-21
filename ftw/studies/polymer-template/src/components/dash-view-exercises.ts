
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

import "/node_modules/@polymer/iron-collapse/iron-collapse.js";

import './exercise-card.js';
import './navigation-menu.js';

@customElement('dash-view-exercises')
export class DashViewExercises extends connect(store)(PageViewElement) {

  @property({type: String})
  _navSection = 'vocational';

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

      .exercise-category-container {
        width: 100%;
        display: flex;
        flex-direction: row;
        justify-content: center;
      }

      .exercise-category-container[active="false"] {
        display: none;
      }

      .exercise-category-container-inner {
        width: 70vw;
        max-width: 1400px;
      }

      .exercise-category-pre-title {
        width: 100%;
        text-align: center;
        text-transform: uppercase;
        font-size: 14px;
        font-weight: 400;
        padding-bottom: 15px;
      }

      .exercise-category-title {
        text-align: center;
        font-size: 32px;
        font-weight: 100;
        padding-bottom: 40px;
      }

      .exercise-category-description {
        text-align: center;
        font-size: 18px;
        font-weight: 100;
        padding-bottom: 60px;
      }

      .exercise-card-container {
        display: flex;
        flex-direction: row;
        justify-content: space-around;
        width: 100%;
        flex-wrap: wrap;
        padding-bottom: 100px;
      }

      exercise-card {
        margin: 20px;
      }

      `
    ];
  }

  _handleNavClick(e: Event) {
    this._navSection = e.path[0].getAttribute("navid");
  }

  protected render() {
    return html`

    <div class="top-spacer"></div>

    <div class="nav">
      <div class="nav-items">
        <div class="nav-item left-nav-title">EXERCISE TYPES</div>
        <div class="nav-item" navid="vocational" @click="${this._handleNavClick}" active="${this._navSection == 'vocational'}">Vocational</div>
        <div class="nav-item" navid="meditation" @click="${this._handleNavClick}" active="${this._navSection == 'meditation'}">Meditation</div>
        <div class="nav-item" navid="emotional" @click="${this._handleNavClick}" active="${this._navSection == 'emotional'}">Emotional</div>
        <div class="nav-item" navid="motivation" @click="${this._handleNavClick}" active="${this._navSection == 'motivation'}">Motivation</div>
      </div>
    </div>
    
    <div class="exercise-category-container">

      <div class="exercise-category-container-inner">

        <div class="exercise-category-pre-title">Vocational</div>
        <div class="exercise-category-title">Exercises to enhance your work</div>
        <div class="exercise-category-description">Struturing your work sessions with pre- and post-assessments as well as a timer and feedback can help you enhance your effectiveness - in an individual session as well as over time.</div>

        <div class="exercise-card-container">

            <exercise-card 
                category="vocational"
                iconText="Dw"
                subcategory="deep-work"
                title="Deep Work Session"
                description="Engage in a deep work session while being provided ML-enabled feedback">
                </exercise-card>

            <exercise-card 
                category="vocational"
                iconText="Lw"
                subcategory="light-work"
                title="Light Work Session"
                description="Quickly complete several small, undemanding tasks">
                </exercise-card>

        </div>

      </div>

    </div>

    <div class="exercise-category-container">

      <div class="exercise-category-container-inner">

        <div class="exercise-category-pre-title">Meditation</div>
        <div class="exercise-category-title">Exercises to strengthen your mind</div>
        <div class="exercise-category-description">Meditation exercises are generally grouped into three types: mindfulness, open awareness, and mind-body exercises. There are several exercise variants whose primary type is one of these but which include auxiliary content from one or more of the other types. If you want to try one out you can just tap the card and the exercise will launch.</div>

        <div class="exercise-card-container">

            <exercise-card 
                category="meditation"
                iconText="Mi"
                subcategory="mindfulness"
                title="Breath as gravity"
                description="Pay attention to breath and notice when you become distracted">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="Oa"
                subcategory="open-awareness"
                title="Stream of Thought"
                description="Maintain non-reactive open awareness of thought stream">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="Bm"
                subcategory="body-scan"
                title="Ball of light"
                description="Develop greater mind-body awareness by scanning yours mindfully">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="Bm"
                subcategory="body-scan"
                title="Whole body breathing"
                description="Pracitce mindfulness and body awareness at the same time.">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="Vi"
                subcategory="visualization"
                title="Gratitude visualization"
                description="Pratice culitvating and strengthening feelings of gratitude.">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="Vi"
                subcategory="visualization"
                title="Gratitude visualization"
                description="Pratice culitvating and strengthening feelings of gratitude.">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="Oa"
                subcategory="open-awareness"
                title="Noting"
                description="Pratice a state non-reactive open awareness through noting.">
                </exercise-card>

            <exercise-card 
                category="meditation"
                iconText="So"
                subcategory="social"
                title="Shift to connection"
                description="Practice the habit of offering kindness and seeing similarity.">
                </exercise-card>

          </div> <!-- card container -->
    
        </div> <!-- category inner -->

      </div> <!-- category container -->

    <div class="exercise-category-container">

      <div class="exercise-category-container-inner">

        <div class="exercise-category-pre-title">Emotional</div>
        <div class="exercise-category-title">Exercises to boost your emotional intelligence</div>
        <div class="exercise-category-description">The ability to perceive emotion states in others with fine-grained resolution is trainable. Try some of these exercises for yourself.</div>

        <div class="exercise-card-container">

            <exercise-card 
                category="emotional"
                iconText="La"
                subcategory="emotion-labeling"
                title="Label Game"
                description="Practice recognizing emotions by labeling short video clips.">
                </exercise-card>
        </div>

      </div>

    </div>

    <div class="exercise-category-container">

      <div class="exercise-category-container-inner">

        <div class="exercise-category-pre-title">Motivation</div>
        <div class="exercise-category-title">Boost your motivation via introspection</div>
        <div class="exercise-category-description">Motivation can be enhanced through exercises that help you identify deep values and connect these to your daily activities.</div>

        <div class="exercise-card-container">

            <exercise-card 
                category="motivation"
                iconText="Mo"
                subcategory="motivation-journaling"
                title="Best case scenario"
                description="Journal about your best case scenario to boost motivation">
                </exercise-card>
        </div>

      </div>

    </div>

    `;
  }

}

