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

export const ButtonSharedStyles = css`
  button {
    font-size: inherit;
    vertical-align: middle;
    background: transparent;
    border: none;
    cursor: pointer;
  }

  button:hover svg {
    fill: var(--app-primary-color);
  }

  paper-button {
    background-color: var(--app-primary-color);
    border: none;
    color: #FFFFFF;
    text-decoration: none;
    font-size: 11px;
    text-transform: none;
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 17px;
    line-height: 2;
    font-size: 11px;
  }

  .inverted-primary {
    border: 1px solid var(--app-primary-color);
    color: var(--app-primary-color);
    background-color: white;
    padding: 8px 18px;
    font-size: 12px;
  }

  .inverted-secondary {
    border: 1px solid white;
    color: white;
    background-color: transparent;
    padding: 8px 18px;
    font-size: 12px;
    background-color: var(--app-primary-color);
  }

  .secondary-light-toolbar-button {
    color: var(--app-primary-color);
    border: 1px solid var(--app-primary-color);
    background-color: transparent;
    line-height: 1.25;
    margin-right: 15px;
    text-decoration: none;
    font-size: 11px;
    text-transform: none;
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 17px;
    font-size: 11px;
  }

  .secondary-light-toolbar-button[inverted="true"] {
    color: white;
    border: 1px solid white;
  }

`;
