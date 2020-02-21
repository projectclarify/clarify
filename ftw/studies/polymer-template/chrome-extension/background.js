// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

/*
  Displays a notification with the current time. Requires "notifications"
  permission in the manifest file (or calling
  "Notification.requestPermission" beforehand).
*/

function notificationClicked(notificationId, buttonIndex) {
  window.open("https://ai4hi.org");
}

function show() {

     chrome.notifications.create("", {
        type:    "basic",
        iconUrl: "42623753.png",
        title:   "Project Clarify",
        message: "It's time for cognitive enhancement!",
        contextMessage: "Time to train!",
        buttons: [{
          "title": "Launch the dashboard"
        }]
    }, function(id) {
        myNotificationID = id;
    });
    
    chrome.notifications.onButtonClicked.addListener(
      notificationClicked
    );

}

// Conditionally initialize the options.
if (!localStorage.isInitialized) {
  localStorage.isActivated = true;   // The display activation.
  localStorage.frequency = 1;        // The display frequency, in minutes.
  localStorage.isInitialized = true; // The option initialization.
}

// Test for notification support.
if (window.Notification) {
  // While activated, show notifications at the display frequency.
  if (JSON.parse(localStorage.isActivated)) {show();}

  var interval = 0; // The display interval, in minutes.

  setInterval(function() {
    interval++;

    if (
      JSON.parse(localStorage.isActivated) &&
        localStorage.frequency <= interval
    ) {
      show();
      interval = 0;
    }
  }, 60000);
}
