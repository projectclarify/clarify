/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { css, html, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';

// This element is connected to the Redux store.
import { store } from '../store.js';

import { mouseIcon } from './my-icons.js';

// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';

import { navigator, exerciseBrowser, exerciseDocumentation, interactive, habitDataviz } from './ui-images.js';

@customElement('view-article-interface-design')
export class ViewArticleInterfaceDesign extends connect(store)(PageViewElement) {

  static get styles() {
    return [
      SharedStyles,
      css`
        a {
          text-decoration: none;
        }

        a:visited {
          color: var(--app-primary-color);
        }

        .article-large-width-image > svg {
          box-shadow: 0px 1.5px 4px rgba(0, 0, 0, .4);
          margin: 10px;
          flex-shrink: 0;
          width: 80%;
          height: auto;
        }

      `
    ];
  }

  protected render() {
    return html`

      <section class="article-hero">
        <p class="article-hero-pre-heading">ARTICLE</p>
        <p class="article-hero-heading">Interface design</p>
        <p class="article-hero-subheading">Understand our interface design strategy</p>
        <div class="article-hero-scroll-cue">
          <p class="article-hero-scroll-cue-text">Scroll to read more</p>
          ${mouseIcon}
        </div>
      </section>

      <section class="article-section scrollspy-target" id="navigator">
        <p class="article-section-pre-title">Navigator</p>
        <h2 class="article-section-title">Manage a diverse set of habits</h2>
        <p>It is willpower-intensive to start and maintain a long-term meditative or other brain-training regimen. Our desire to deliver training exercises is naturally coupled with a realistic means to enable users to engage in those exercises over the long-term.</p>

        <p>At a high level, for us, this has three parts: making it easy to (1) engage in, (2) keep track of engaging in, and (3) stay motivated to engage in these exercises.</p>

        <p>Below is shown the primary dashboard view - the "navigator". A radial re-imagination of the conventional day calendar enabling users both easy means of logging binary habits (perhaps fewer than pictured) as well as easy means of launching interactive exercises. In this design, training exercises are treated like any of the various other habits you likely aim to engage in throughout the day and can be launched from a central location with a click.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${navigator}
            </div>
        </div>

        <p>Here the color of the fill of the circle clearly indicates the completion of the habit as well as contributes to forming a high-level intuition of what your day was like (in terms of these habits).</p>

      </section>

      <section class="article-section scrollspy-target" id="exercise-browser">
        <p class="article-section-pre-title">Exercise Browser</p>
        <h2 class="article-section-title">A browsable library of exercises</h2>

        <p>A browsable library of the available exercises gives users a way to see the organization of exercises as well as an entrypoint to either the exercise itself or it's extended documentation (see next section). One could imagine this library growing to include a wide variety of training exercises produced by and initially piloted by contributors from the open source community.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${exerciseBrowser}
            </div>
        </div>

        <p>An open task is to improve on this design to make the structure of exercise types and sub-types even more intuitively apparent (such as replacing the current single-color text icons in the top right with variously-colored graphical ones).</p>

        <p>It's worth noting that although this section will allow people to launch exercises outside of their established habit structure - our goal with the navigator view is to prevent the need for people to search through the exercise library in order to initiate one.</p>

      </section>

      <section class="article-section scrollspy-target" id="exercise-documentation">
        <p class="article-section-pre-title">Exercise Documentation</p>
        <h2 class="article-section-title">Clear exercise explanations</h2>

        <p>The extended exercise documentation view is simple - it will give you the information to be able to perfor that particular exercise most effectively.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${exerciseDocumentation}
            </div>
        </div>

        <p>Deeply understanding what it is you're actually meant to be doing and why is a key requirement to performing, for example, meditative training exercises effectively</p>

      </section>

      <section class="article-section scrollspy-target" id="interactive">
        <p class="article-section-pre-title">Interactive Session</p>
        <h2 class="article-section-title">Exercises and pre-/post- subjectives</h2>

        <p>Interactive sessions are the place where training exercises are performed - between pre- and post-assessments. Below is pictured a vocation background feedback exercise where a user can receive state feedback that is (1) visual, in the form of a radar chart or (2) a background audio cue constructed from state predictions.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${interactive}
            </div>
        </div>

        <p>An initial set of such exercises will provide the layout, UI elements, and data patterns to allow us to experiment broadly with exercise types without the need to build one-off interfaces.</p>

      </section>

      <section class="article-section scrollspy-target" id="habit-visualization">
        <p class="article-section-pre-title">Data Visualization</p>
        <h2 class="article-section-title">Understanding habits, performance, and health</h2>

        <p>As noted above our goal is not to simply have our users perform training exercises over a limited duration - but rather to help in the establishment of these exercises as long-term habits (such as the habit of micro mindfulness meditations between work sessions). Toward that end we were inspired to emulate the enveloped habit acquisition trajectory design from <a href="https://www.beeminder.com/">BeeMinder</a> (which is worth trying if you haven't!). This approach to habit acquisition and maintenance treats a habit like a thing to be acquired gradually and recognizes there is natural day to day variation in adherence.</p>

        <div class="article-large-width-image-container">
            <div class="article-large-width-image">
                ${habitDataviz}
            </div>
        </div>

        <p>You may want to continue reading about our <a href="/machine-learning">machine learning methods</a>, read about the <a href="/technologies">technologies</a> we use, or check out our <a href="https://github.com/projectclarify/pcml" target="_blank">codebase</a> on GitHub.</p>

      </section>

      <div class="article-bottom-spacer"></div>

      <div class="made-with-love"><p>Made with love in California.</p></div>

    `;
  }

}
