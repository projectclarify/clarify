/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/
import { UPDATE_PAGE, UPDATE_OFFLINE, OPEN_SNACKBAR, CLOSE_SNACKBAR, UPDATE_DRAWER_STATE, UPDATE_ACCOUNT_DROPDOWN_STATE, UPDATE_TOOLBAR_LIGHT_DARK_STATE, UPDATE_LOADING_ANIMATION_STATE, UPDATE_PROGRESS_BAR_STATE, UPDATE_PROGRESS_LABEL, UPDATE_DASHBOARD_ACTIVE } from '../actions/app.js';
const INITIAL_STATE = {
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
const app = (state = INITIAL_STATE, action) => {
    switch (action.type) {
        case UPDATE_PAGE:
            return Object.assign({}, state, { page: action.page });
        case UPDATE_OFFLINE:
            return Object.assign({}, state, { offline: action.offline });
        case UPDATE_DRAWER_STATE:
            return Object.assign({}, state, { drawerOpened: action.opened });
        case UPDATE_ACCOUNT_DROPDOWN_STATE:
            return Object.assign({}, state, { accountDropdownOpened: action.opened });
        case OPEN_SNACKBAR:
            return Object.assign({}, state, { snackbarOpened: true });
        case CLOSE_SNACKBAR:
            return Object.assign({}, state, { snackbarOpened: false });
        case UPDATE_TOOLBAR_LIGHT_DARK_STATE:
            return Object.assign({}, state, { toolbarLight: action.light });
        case UPDATE_LOADING_ANIMATION_STATE:
            return Object.assign({}, state, { loading: action.loading });
        case UPDATE_PROGRESS_BAR_STATE:
            return Object.assign({}, state, { progress: action.progress });
        case UPDATE_PROGRESS_LABEL:
            return Object.assign({}, state, { progressLabel: action.progressLabel });
        case UPDATE_DASHBOARD_ACTIVE:
            return Object.assign({}, state, { dashboardActive: action.dashboardActive });
        default:
            return state;
    }
};
export default app;
