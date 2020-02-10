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
import { html, css, customElement } from 'lit-element';
import { PageViewElement } from './page-view-element.js';
import { connect } from 'pwa-helpers/connect-mixin.js';
// This element is connected to the Redux store.
import { store } from '../store.js';
// These are the shared styles needed by this element.
import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';
let ViewTermsOfService = class ViewTermsOfService extends connect(store)(PageViewElement) {
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

      <section class="article-hero">
        <p class="article-hero-pre-heading">POLICY DOCUMENTS</p>
        <p class="article-hero-heading">Terms of service</p>
      </section>

      <section class="article-section scrollspy-target" id="overview">
        <p class="article-section-pre-title">Overview</p>
        <h2 class="article-section-title">Overview of the policy</h2>
        <p>Please read these Terms and Conditions ("Terms", "Terms and Conditions") carefully before using the https://www.ai4hi.org website (the "Service") operated by The Project Clarify Playground ("us", "we", or "our").</p>
        <p>Your access to and use of the Service is conditioned on your acceptance of and compliance with these Terms. These Terms apply to all visitors, users and others who access or use the Service.</p>
        <p>By accessing or using the Service you agree to be bound by these Terms. If you disagree with any part of the terms then you may not access the Service. The Terms and Conditions agreement for The Project Clarify Playground has been created with the help of TermsFeed.</p>
      </section>
      <section class="article-section scrollspy-target" id="account">
        <p class="article-section-pre-title">Account</p>
        <h2 class="article-section-title">Terms of your account</h2>
        <p>When you create an account with us, you must provide us information that is accurate, complete, and current at all times. Failure to do so constitutes a breach of the Terms, which may result in immediate termination of your account on our Service.</p>
        <p>You are responsible for safeguarding the password that you use to access the Service and for any activities or actions under your password, whether your password is with our Service or a third-party service.</p>
        <p>You agree not to disclose your password to any third party. You must notify us immediately upon becoming aware of any breach of security or unauthorized use of your account.</p>
      </section>
      <section class="article-section scrollspy-target" id="links">
        <p class="article-section-pre-title">Links</p>
        <h2 class="article-section-title">Links to other websites</h2>
        <p>Our Service may contain links to third-party web sites or services that are not owned or controlled by The Project Clarify Playground.</p>
        <p>The Project Clarify Playground has no control over, and assumes no responsibility for, the content, privacy policies, or practices of any third party web sites or services. You further acknowledge and agree that The Project Clarify Playground shall not be responsible or liable, directly or indirectly, for any damage or loss caused or alleged to be caused by or in connection with use of or reliance on any such content, goods or services available on or through any such web sites or services.</p>
        <p>We strongly advise you to read the terms and conditions and privacy policies of any third-party web sites or services that you visit.</p>
      </section>
      <section class="article-section scrollspy-target" id="termination">
        <p class="article-section-pre-title">Termination</p>
        <h2 class="article-section-title">Terms of termination</h2>
        <p>We may terminate or suspend access to our Service immediately, without prior notice or liability, for any reason whatsoever, including without limitation if you breach the Terms.</p>
        <p>All provisions of the Terms which by their nature should survive termination shall survive termination, including, without limitation, ownership provisions, warranty disclaimers, indemnity and limitations of liability.</p>
        <p>We may terminate or suspend your account immediately, without prior notice or liability, for any reason whatsoever, including without limitation if you breach the Terms.</p>
        <p>Upon termination, your right to use the Service will immediately cease. If you wish to terminate your account, you may simply discontinue using the Service.</p>
        <p>All provisions of the Terms which by their nature should survive termination shall survive termination, including, without limitation, ownership provisions, warranty disclaimers, indemnity and limitations of liability.</p>
      </section>
      <section class="article-section scrollspy-target" id="governing-law">
        <p class="article-section-pre-title">Governing Law</p>
        <h2 class="article-section-title">Terms of governing law</h2>
        <p>These Terms shall be governed and construed in accordance with the laws of California, United States, without regard to its conflict of law provisions.</p>
        <p>Our failure to enforce any right or provision of these Terms will not be considered a waiver of those rights. If any provision of these Terms is held to be invalid or unenforceable by a court, the remaining provisions of these Terms will remain in effect. These Terms constitute the entire agreement between us regarding our Service, and supersede and replace any prior agreements we might have between us regarding the Service.</p>
      </section>
      <section class="article-section scrollspy-target" id="changes">
        <p class="article-section-pre-title">Changes</p>
        <h2 class="article-section-title">Terms of changes to the terms</h2>
        <p>We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material we will try to provide at least 30 days notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.</p>
        <p>By continuing to access or use our Service after those revisions become effective, you agree to be bound by the revised terms. If you do not agree to the new terms, please stop using the Service.</p>
      </section>
      <section class="article-section scrollspy-target" id="contact">
        <p class="article-section-pre-title">Contact</p>
        <h2 class="article-section-title">Contacting us about the terms</h2>
        <p>If you have any questions about these Terms, please contact us.</p>
      </section>


    `;
    }
};
ViewTermsOfService = __decorate([
    customElement('view-terms-of-service')
], ViewTermsOfService);
export { ViewTermsOfService };
