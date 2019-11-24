/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { css } from 'lit-element';

export const SharedStyles = css`
  :host {
    display: block;
    box-sizing: border-box;
  }

  section {
    padding: 24px;
    background: var(--app-section-odd-color);
  }

  section > * {
    max-width: 600px;
    margin-right: auto;
    margin-left: auto;
  }

  h2 {
    font-size: 24px;
    text-align: center;
    color: var(--app-dark-text-color);
  }

  @media (min-width: 460px) {
    h2 {
      font-size: 36px;
    }
  }

  .circle {
    display: block;
    width: 64px;
    height: 64px;
    margin: 0 auto;
    text-align: center;
    border-radius: 50%;
    background: var(--app-primary-color);
    color: var(--app-light-text-color);
    font-size: 30px;
    line-height: 64px;
  }

  .article-hero {
    text-align: center;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center
  }

  .article-hero > * {
    max-width: 90%
  }

  .article-hero-pre-heading {
      font-size:14px;
      font-weight: 400;
  }

  .article-hero-heading {
      font-size: 55px;
      font-family: "IBM Plex Sans",sans-serif;
      font-weight: 300;
      color: #2b2b2b;
      margin-top: 0px;
      margin-bottom: 0px;
  }

  .article-hero-subheading {
      font-size: 20px;
      font-weight: 300;
  }

  .article-hero-side-note {
      font-size: 12px;
      margin-top: 36px;
      margin-bottom: 36px;
  }

  .article-hero-side-note > a:visited {
    color: #0000EE;
  }

  .primary-btn {
    background-color: #0168FA;
    border: none;
    color: #FFFFFF;
    text-decoration: none;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 500;
    border-radius: 6px;
    padding: 8px 17px;
    line-height: 2;
    font-size: 13.5px;
  }

  .secondary-btn {
    background-color: white;
    border: 1px solid #efefef;
    color: black;
    text-decoration: none;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 500;
    border-radius: 6px;
    padding: 8px 17px;
    line-height: 2;
    font-size: 13.5px;
  }

  paper-button:hover {
    box-shadow: 0px 0px 3px #a8a8a8;
    transition: 0.1s;
  }

  .article-hero-button {
    border-radius: 3px;
    border: 1px solid #a8a8a8;
    text-transform: uppercase;
    font-size: 10px;
    padding-left: 10px;
    padding-right: 10px;
    font-weight: 400;
  }

  .article-hero-button:hover {
      box-shadow: 0px 0px 3px #a8a8a8;
      transition: 0.1s;
  }

  .article-hero-scroll-cue {
      margin-top: 36px;
  }

  .article-hero-scroll-cue > p {
      color: #424242;
      font-family: "IBM Plex Sans",sans-serif;
      font-weight: 100;
      font-size: 14px;
  }

  .article-hero-scroll-cue > svg {
    fill: #5c5c5c;
  }

  .article-section {
      font-family: "IBM Plex Sans",sans-serif;
      font-weight: 100;
      font-size: 14px;
      padding-left: 0px;
      padding-right: 0px;
  }

  .article-section > p {
    font-size: 18px;
  }

  .article-section > p > a {
      color: var(--app-primary-color);
      font-style: normal;
      text-decoration: none;
  }

  .article-section > p > a:hover {
    text-decoration: underline;
    cursor: pointer;
  }

  .article-section > p > b {
    font-weight: 600;
    font-color: var(--app-primary-color);
  }

  .article-section > p.large-quote {
    font-size: 24px;
    max-width: 800px;
    text-justify: auto;
    padding: 40px;
    padding-top: 80px;
    padding-bottom: 80px;
  }

  .article-section > p.large-quote > a {
    font-weight: 300;
  }

  .article-section-pre-title {
    font-weight: 400;
    font-size: 15px;
    text-transform: uppercase;
    text-align: center;
  }

  .article-section-title {
      font-family: "IBM Plex Sans",sans-serif;
      font-weight: 300;
      font-size: 35px;
  }

  .article-full-width-image {
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    max-width: 100%;
    margin-top: 20px;
  }

  .article-full-width-image > img {
    flex-shrink: 0;
    width: 100%;
    height: auto;
  }

  .article-full-width-image-container {
    max-width: 100%;
  }

  .article-large-width-image {
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    max-width: 100%;
    margin-top: 80px;
    margin-bottom: 40px;

  }

  .article-large-width-image > img {
    flex-shrink: 0;
    width: 80%;
    height: auto;
  }

  .article-large-width-image-container {
    max-width: 100%;
  }

  .article-figure-legend-table > tbody > tr > td > b {
    color: var(--app-primary-color);
  }

  .image-legend {
    font-size: 12px;
    margin-left: 40px;
    margin-right: 40px;
    margin-top: 20px;
  }

  #scrollspy {
    position: fixed;
    right: 0px;
    top: 40%;
    padding: 20px;
    background-color: transparent;
  }

  .scrollspy-item {
    display: flex;
    flex-direction: horizontal;
    direction: ltr;
    justify-content: flex-end;
  }

  p.scrollspy-item-text {
    font-family: "IBM Plex Sans",sans-serif;
    font-weight: 300;
    font-size: 12px;
    text-align: right;
    color: #c2c2c2;
    cursor: default;
  }

  #scrollspy:hover > .scrollspy-item > .scrollspy-item-text {
    visibility: visible;
  }

  .scrollspy-item-icon {
    padding: 10px;
  }

  .scrollspy-item-icon > svg > circle {
    stroke: #efefef;
    stroke-width: 1;
  }

  .scrollspy-item:hover > p.scrollspy-item-text {
    font-weight: 600;
    visibility: visible;
  }

  .scrollspy-item:hover > .scrollspy-item-icon > svg > circle {
    stroke-width: 2;
  }

  .scrollspy-item[selected="true"] > .scrollspy-item-icon > svg > circle {
    stroke: #3c78d8ff;
    stroke-width: 3;
  }

  .scrollspy-item[selected="true"] > p.scrollspy-item-text {
    color: #3c78d8ff;
    font-family: "IBM Plex Sans",sans-serif;
    font-weight: 600;
  }

  .data-interactive-container {
    background-color: #efefef;
  }

  section > ol > li {

  }
  
  section > ol > li > b {
    color: var(--app-primary-color);
  }

  @media (max-width: 1048px) {
    p.scrollspy-item-text {
      visibility: hidden;
    }

    .article-hero-pre-heading {
        
    }

    .article-hero-heading {
        font-size: 2.5rem
    }

    .article-hero-subheading {
    }

    .article-hero-side-note {

    }

    .article-section-title {
        padding-left: 40px;
        padding-right: 40px;
    }

    .article-section > p {
        font-size: 14px;
        padding-left: 40px;
        padding-right: 40px;
    }

    #scrollspy {
        padding: 3px;
    }

    .article-section > table {
        font-size: 12px;
        padding-left: 40px;
        padding-right: 40px;
    }

    .article-section > ol {
        font-size: 12px;
        padding-left: 40px;
        padding-right: 40px;
    }

  }

  .login-section {
    text-align:center;
    
  }

  .login-title {

  }

  .login-description {

  }

  .login-state-toggle {
      
  }

  .secondary-login-btn {

  }

  .login-btn {

  }

  .article-bottom-spacer {
    height: 200px;
  }

  .made-with-love {
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 300px;
    background-color: var(--app-primary-color);
    box-shadow: inset 0 7px 9px -7px rgba(0,0,0,0.7);
  }

  .made-with-love > p {
    font-family: Pacifico;
    font-size: 3rem;
    color: white;
    text-align: center;
    padding-left: 10%;
    padding-right: 10%;
  }  

  b {
    color: var(--app-primary-color);
  }

    .top-spacer {
    height: 87.4px;
    }

    .nav {
    position: fixed;
    left: 0px;
    top: 125px;
    width: 200px;
    display: flex;
    flex-direction: row;
    }

    .right-nav {
    right: 0px;
    }

    .nav-item {
    font-size: 12px;
    padding-left: 15px;
    padding-top: 7px;
    padding-bottom: 7px;
    }

    .nav-item[active="true"] {
    font-weight: 600;
    border-left: 3px solid var(--app-primary-color);
    color: var(--app-primary-color);
    }

    .nav-title {
    font-size: 12px;
    font-weight: 600;
    padding-bottom: 14px;
    }

    .nav-items {

    }
    
    @media (max-width: 1048px) {
    .nav {
        display: none;
    }
    }

`;
