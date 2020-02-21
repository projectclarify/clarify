/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/
import { SET_USER, SET_USER_DATA, SET_USER_ACTIVITIES, SET_USER_ACTIVITIES_LIBRARY, SET_USER_ACTIVITY_COMPLETIONS, SET_USER_MEASURES, SET_USER_MEASURE_LIBRARY, SET_USER_STRUCTURE, SET_USER_FIRESTORE_DB } from '../actions/user.js';
const user = (state = { currentUser: null, userData: null, userActivities: null, userActivitiesLibrary: null, userActivityCompletions: null, userMeasures: null, userMeasureLibrary: null, userStructure: null, userFirestoreDB: null }, action) => {
    switch (action.type) {
        case SET_USER:
            return Object.assign({}, state, { currentUser: action.user });
        case SET_USER_DATA:
            return Object.assign({}, state, { userData: action.userData });
        case SET_USER_ACTIVITIES:
            return Object.assign({}, state, { userActivities: action.userActivities });
        case SET_USER_ACTIVITIES_LIBRARY:
            return Object.assign({}, state, { userActivitiesLibrary: action.userActivitiesLibrary });
        case SET_USER_ACTIVITY_COMPLETIONS:
            return Object.assign({}, state, { userActivityCompletions: action.userActivityCompletions });
        case SET_USER_MEASURES:
            return Object.assign({}, state, { userMeasures: action.userMeasures });
        case SET_USER_MEASURE_LIBRARY:
            return Object.assign({}, state, { userMeasureLibrary: action.userMeasureLibrary });
        case SET_USER_STRUCTURE:
            return Object.assign({}, state, { userStructure: action.userStructure });
        case SET_USER_FIRESTORE_DB:
            return Object.assign({}, state, { userFirestoreDB: action.userFirestoreDB });
        default:
            return state;
    }
};
export default user;
