
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import os

credentials_path = os.environ['FIREBASE_ADMIN_CREDENTIALS']

cred = credentials.Certificate(credentials_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

uid = u"2LbhP63ADQfo5XkmKeVVEtPWvAD2"

import json
with open("dev_data.json", "r") as f:
  data = json.loads(f.read().strip())

db.collection(u"users").document(uid).set(data)