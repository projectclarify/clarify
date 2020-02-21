/**
@license
Copyright (c) 2018 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

import firebase from '@firebase/app';
import '@firebase/auth';
import '@firebase/firestore';

import { store } from '../store.js';

import { setUser,
         setUserData,
         setUserFirestoreDB } from '../actions/user.js';

import user from '../reducers/user.js';

store.addReducers({
  user
});

// Initialize Firebase
var config = {
  apiKey: "AIzaSyAU4BL4cX5zV-C5mQgjJY5Fv2NC0KFYM_Y",
  authDomain: "clarify-32fcf.firebaseapp.com",
  databaseURL: "https://clarify-32fcf.firebaseio.com",
  projectId: "clarify",
  storageBucket: "clarify.appspot.com",
  messagingSenderId: "340173104977",
  appId: "1:340173104977:web:31e17e60e5d0c648"
};
firebase.initializeApp(config);


let db = firebase.firestore!();
db.settings({
  cacheSizeBytes: firebase.firestore.CACHE_SIZE_UNLIMITED
});
db.enablePersistence()
  .catch(function(err) {
      if (err.code == 'failed-precondition') {
          // Multiple tabs open, persistence can only be enabled
          // in one tab at a a time.
          // ...
      } else if (err.code == 'unimplemented') {
          // The current browser does not support all of the
          // features required to enable persistence
          // ...
      }
  });


firebase.auth!().onAuthStateChanged(function(user) {
  console.log(user);

  if(user) {

    let usersFirestoreDocument = db.collection("users").doc(user.uid);

    store.dispatch(setUserFirestoreDB(usersFirestoreDocument));

    usersFirestoreDocument.onSnapshot(function (snapshot: Object){
      store.dispatch(setUserData(snapshot.data()));
    });

  }

  store.dispatch(setUser(user));

});

export { firebase };
