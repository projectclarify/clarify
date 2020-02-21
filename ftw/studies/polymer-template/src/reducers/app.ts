/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { Reducer } from 'redux';
import {
  UPDATE_PAGE,
  UPDATE_OFFLINE,
  OPEN_SNACKBAR,
  CLOSE_SNACKBAR,
  UPDATE_DRAWER_STATE,
  UPDATE_ACCOUNT_DROPDOWN_STATE,
  UPDATE_TOOLBAR_LIGHT_DARK_STATE,
  UPDATE_LOADING_ANIMATION_STATE,
  UPDATE_PROGRESS_BAR_STATE,
  UPDATE_PROGRESS_LABEL,
  UPDATE_DASHBOARD_ACTIVE
} from '../actions/app.js';
import { RootAction } from '../store.js';

export interface AppState {
  page: string;
  offline: boolean;
  drawerOpened: boolean;
  snackbarOpened: boolean;
  accountDropdownOpened: boolean;
  toolbarLight: boolean;
  loading: boolean;
  progress: number;
  progressLabel: string;
  dashboardActive: boolean
}

const INITIAL_STATE: AppState = {
  page: '',
  offline: false,
  drawerOpened: false,
  snackbarOpened: false,
  accountDropdownOpened: false,
  toolbarLight: true,
  loading: true,
  progress: 0,
  progressLabel: "",
  dashboardActive: true
};

const app: Reducer<AppState, RootAction> = (state = INITIAL_STATE, action) => {
  switch (action.type) {
    case UPDATE_PAGE:
      return {
        ...state,
        page: action.page
      };
    case UPDATE_OFFLINE:
      return {
        ...state,
        offline: action.offline
      };
    case UPDATE_DRAWER_STATE:
      return {
        ...state,
        drawerOpened: action.opened
      };
    case UPDATE_ACCOUNT_DROPDOWN_STATE:
      return {
        ...state,
        accountDropdownOpened: action.opened
      };
    case OPEN_SNACKBAR:
      return {
        ...state,
        snackbarOpened: true
      };
    case CLOSE_SNACKBAR:
      return {
        ...state,
        snackbarOpened: false
      };
    case UPDATE_TOOLBAR_LIGHT_DARK_STATE:
      return {
        ...state,
        toolbarLight: action.light
      };
    case UPDATE_LOADING_ANIMATION_STATE:
      return {
        ...state,
        loading: action.loading
      };
    case UPDATE_PROGRESS_BAR_STATE:
      return {
        ...state,
        progress: action.progress
      };
    case UPDATE_PROGRESS_LABEL:
      return {
        ...state,
        progressLabel: action.progressLabel
      };
    case UPDATE_DASHBOARD_ACTIVE:
      return {
        ...state,
        dashboardActive: action.dashboardActive
      };
    default:
      return state;
  }
};

export default app;
