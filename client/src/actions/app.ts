/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import { Action, ActionCreator } from 'redux';
import { ThunkAction } from 'redux-thunk';
import { RootState } from '../store.js';

import {scroll} from '@polymer/app-layout/helpers/helpers.js';

export const UPDATE_PAGE = 'UPDATE_PAGE';
export const UPDATE_OFFLINE = 'UPDATE_OFFLINE';
export const UPDATE_DRAWER_STATE = 'UPDATE_DRAWER_STATE';
export const UPDATE_ACCOUNT_DROPDOWN_STATE = 'UPDATE_ACCOUNT_DROPDOWN_STATE';
export const OPEN_SNACKBAR = 'OPEN_SNACKBAR';
export const CLOSE_SNACKBAR = 'CLOSE_SNACKBAR';
export const UPDATE_TOOLBAR_LIGHT_DARK_STATE = 'UPDATE_TOOLBAR_LIGHT_DARK_STATE';
export const UPDATE_LOADING_ANIMATION_STATE = 'UPDATE_LOADING_ANIMATION_STATE';
export const UPDATE_PROGRESS_BAR_STATE = 'UPDATE_PROGRESS_BAR_STATE';
export const UPDATE_PROGRESS_LABEL = 'UPDATE_PROGRESS_LABEL';
export const UPDATE_DASHBOARD_ACTIVE = 'UPDATE_DASHBOARD_ACTIVE';

export interface AppActionUpdatePage extends Action<'UPDATE_PAGE'> {page: string};
export interface AppActionUpdateOffline extends Action<'UPDATE_OFFLINE'> {offline: boolean};
export interface AppActionUpdateDrawerState extends Action<'UPDATE_DRAWER_STATE'> {opened: boolean};
export interface AppActionOpenSnackbar extends Action<'OPEN_SNACKBAR'> {};
export interface AppActionCloseSnackbar extends Action<'CLOSE_SNACKBAR'> {};
export interface AppActionUpdateAccountDropdownState extends Action<'UPDATE_ACCOUNT_DROPDOWN_STATE'> {opened: boolean};
export interface AppActionUpdateToolbarLightDarkState extends Action<'UPDATE_TOOLBAR_LIGHT_DARK_STATE'> {light: boolean};
export interface AppActionUpdateLoadingAnimationState extends Action<'UPDATE_LOADING_ANIMATION_STATE'> {loading: boolean};
export interface AppActionUpdateProgressBarState extends Action<'UPDATE_PROGRESS_BAR_STATE'> {progress: number};
export interface AppActionUpdateProgressLabel extends Action<'UPDATE_PROGRESS_LABEL'> {progressLabel: string};
export interface AppActionUpdateDashboardActive extends Action<'UPDATE_DASHBOARD_ACTIVE'> {dashboardActive: boolean};

export type AppAction = AppActionUpdateProgressLabel | AppActionUpdateProgressBarState | AppActionUpdatePage | AppActionUpdateOffline | AppActionUpdateDrawerState | AppActionOpenSnackbar | AppActionCloseSnackbar | AppActionUpdateAccountDropdownState | AppActionUpdateToolbarLightDarkState | AppActionUpdateLoadingAnimationState | AppActionUpdateDashboardActive;
type ThunkResult = ThunkAction<void, RootState, undefined, AppAction>;

export const navigate: ActionCreator<ThunkResult> = (path: string) => (dispatch) => {
  // Extract the page name from path.
  const page = path === '/' ? 'project-clarify' : path.slice(1);

  dispatch(updateLoadingAnimationState(true));
  dispatch(updateProgressBarState(0));
  dispatch(updateProgressLabel(""));

  // Always make the toolbar light unless otherwise configured downstream
  dispatch(updateToolbarLightDarkState(true));

  // Any other info you might want to extract from the path (like page type),
  // you can do here
  dispatch(loadPage(page));

  // Close the drawer - in case the *path* change came from a link in the drawer.
  dispatch(updateDrawerState(false));

  // Close the account dropdown in case the path change came from a link in the acct dropdown.
  dispatch(updateAccountDropdownState(false));

  scroll({top: 0, left: 0});

};

const loadPage: ActionCreator<ThunkResult> = (page: string) => (dispatch) => {
  switch(page) {
    case 'project-clarify':
      import('../components/view-article-project-clarify.js').then((_module) => {
        // This is repetitive but not familiar enough with js to do more succinctly
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'technologies':
      import('../components/view-article-technologies.js').then((_module) => {
        // This is repetitive but not familiar enough with js to do more succinctly
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'dashboard':
      import('../components/view-dashboard.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(true));
      });
      break;
    case 'data':
      import('../components/view-data.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'login':
      import('../components/view-login.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
      });
      break;
    case 'privacy-policy':
      import('../components/view-privacy-policy.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'terms-of-service':
      import('../components/view-terms-of-service.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'hack':
      import('../components/view-article-hackathons.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'machine-learning':
      import('../components/view-article-machine-learning.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'guided-meditation':
      import('../components/view-interactive.js').then((_module) => {
        dispatch(updateToolbarLightDarkState(false));
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'interface-design':
      import('../components/view-article-interface-design.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'residency-program':
      import('../components/view-article-residency-program.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    case 'scaling-tpu-input':
      import('../components/view-article-scaling-tpu-input.js').then((_module) => {
        dispatch(updateLoadingAnimationState(false));
        dispatch(updateDashboardActive(false));
      });
      break;
    default:
      page = 'thats-a-404';
      import('../components/view404.js');
  }

  dispatch(updatePage(page));

};

const updatePage: ActionCreator<AppActionUpdatePage> = (page: string) => {
  return {
    type: UPDATE_PAGE,
    page
  };
};

let snackbarTimer: number;

export const showSnackbar: ActionCreator<ThunkResult> = () => (dispatch) => {
  dispatch({
    type: OPEN_SNACKBAR
  });
  window.clearTimeout(snackbarTimer);
  snackbarTimer = window.setTimeout(() =>
    dispatch({ type: CLOSE_SNACKBAR }), 3000);
};

export const updateOffline: ActionCreator<ThunkResult> = (offline: boolean) => (dispatch, getState) => {
  // Show the snackbar only if offline status changes.
  if (offline !== getState().app!.offline) {
    dispatch(showSnackbar());
  }
  dispatch({
    type: UPDATE_OFFLINE,
    offline
  });
};

export const updateDrawerState: ActionCreator<AppActionUpdateDrawerState> = (opened: boolean) => {
  return {
    type: UPDATE_DRAWER_STATE,
    opened
  };
};

export const updateAccountDropdownState: ActionCreator<AppActionUpdateAccountDropdownState> = (opened: boolean) => {
  return {
    type: UPDATE_ACCOUNT_DROPDOWN_STATE,
    opened
  };
};

export const updateToolbarLightDarkState: ActionCreator<AppActionUpdateToolbarLightDarkState> = (light: boolean) => {
  return {
    type: UPDATE_TOOLBAR_LIGHT_DARK_STATE,
    light
  };
};

export const updateLoadingAnimationState: ActionCreator<AppActionUpdateLoadingAnimationState> = (loading: boolean) => {
  return {
    type: UPDATE_LOADING_ANIMATION_STATE,
    loading
  };
};

export const updateProgressBarState: ActionCreator<AppActionUpdateProgressBarState> = (progress: number) => {
  return {
    type: UPDATE_PROGRESS_BAR_STATE,
    progress
  };
};

export const updateProgressLabel: ActionCreator<AppActionUpdateProgressLabel> = (progressLabel: string) => {
  return {
    type: UPDATE_PROGRESS_LABEL,
    progressLabel
  };
};



export const updateDashboardActive: ActionCreator<AppActionUpdateDashboardActive> = (dashboardActive: boolean) => {
  return {
    type: UPDATE_DASHBOARD_ACTIVE,
    dashboardActive
  };
};

