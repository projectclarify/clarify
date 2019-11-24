import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pprint
import os

credentials_path = os.environ['FIREBASE_ADMIN_CREDENTIALS']

cred = credentials.Certificate(credentials_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

#creds = "/home/chris_w_beitel/creds/firebase-admin.json"

ref = db.collection('users').document('2LbhP63ADQfo5XkmKeVVEtPWvAD2')

# Read the data at the posts reference (this is a blocking operation)
print(ref.collection("activities").document("x6q6ol7TcOV8gZAzegjE").get().to_dict())

activities = '''am-1-deep-work-session
activity-id
"deep-work-session"
completed
false
scheduling
"am-1"
am-2-micro-meditation
activity-id
"micro-meditation"
completed
false
scheduling
"am-2"
am-3-deep-work-session
activity-id
"deep-work-session"
completed
false
scheduling
"am-3"
am-pm-nutritious-meal
activity-id
"nutritious-meal"
completed
false
scheduling
"am-pm"
am-pm-water
activity-id
"liter-water"
completed
false
scheduling
"am-pm"
eam-1-run
activity-id
"warmup-run"
completed
false
scheduling
"eam-1"
eam-1-water
activity-id
"liter-water"
completed
false
scheduling
"eam-1"
eam-2-water
activity-id
"liter-water"
completed
false
scheduling
"eam-2"
eam-3-nutritious-meal
activity-id
"nutritious-meal"
completed
false
scheduling
"eam-3"
eam-am-micro-meditation
activity-id
"micro-meditation"
completed
false
scheduling
"eam-am"
pm-1-deep-work-session
activity-id
"deep-work-session"
completed
false
scheduling
"pm-1"
pm-2-micro-meditation
activity-id
"micro-meditation"
completed
false
scheduling
"pm-2"
pm-2-micro-nap
activity-id
"micro-nap"
completed
false
scheduling
"pm-2"
pm-2-water
activity-id
"liter-water"
completed
false
scheduling
"pm-2"
pm-3-deep-work-session
activity-id
"deep-work-session"
completed
false
scheduling
"pm-3"
pm-lpm-micro-meditation
activity-id
"micro-meditation"
completed
false
scheduling
"pm-lpm"
pm-lpm-nutritious-meal
activity-id
"nutritious-meal"
completed
false
scheduling
"pm-lpm"'''
 
activities_lib='''deep-work-session
name
"Deep Work Session"
navigates
true
liter-water
name
"1L Water"
navigates
false
meditation
name
"Meditation"
navigates
true
micro-meditation
name
"Micro-meditation"
navigates
true
micro-nap
name
"Micro-nap"
navigates
false
nutritious-meal
name
"Nutritious Meal"
navigates
false
warmup-run
name
"Warm-up Run"
navigates
false'''

measure_library='''_categories
state-prediction
data-range-max
1
data-range-min
0
data-type
"float"
state-subjective
data-range-max
1
data-range-min
0
data-type
"float"
calm-predicted
category
"state-prediction"
calm-subjective
category
"state-subjective"
confidence-subjective
category
"state-subjective"
energy-predicted
category
"state-prediction"
energy-subjective
category
"state-subjective"
flow-predicted
category
"state-prediction"
flow-subjective
category
"state-subjective"
focus-predicted
category
"state-prediction"
focus-subjective
category
"state-subjective"
happiness-predicted
category
"state-prediction"
happiness-subjective
category
"state-subjective"
kindness-subjective
category
"state-subjective"
optimism-subjective
category
"state-subjective"
skill-confidence-subjective
category
"state-subjective"'''

measures='''calm-subjective
100001
0.5
100003
0.8
100008
0.9
confidence-subjective
100001
0.5
100002
0.85
100007
0.95
energy-subjective
100001
0.5
100003
0.65
100005
0.95
flow-subjective
100001
0.5
100003
0.75
100004
0.85
focus-subjective
100001
0.5
100003
0.95
100012
0.85
happiness-subjective
100001
0.5
100002
0.95
100005
0.85
optimism-subjective
100001
0.5
100003
0.85
100015
0.95'''

structure='''am-1
order
5
am-2
order
6
am-3
order
7
am-pm
order
8
eam-1
order
1
eam-2
order
2
eam-3
order
3
eam-am
order
4
lpm-1
order
13
lpm-2
order
14
lpm-3
order
15
pm-1
order
9
pm-2
order
10
pm-3
order
11
pm-lpm
order
12'''

# activities, activities_lib, measures, measure_library, structure

data = {}
key = None
data["structure"] = {}
data["activities"] = {}

for i, line in enumerate(structure.split("\n")):
  if (i % 3 == 0):
    key = line
    data["structure"][key] = {}
  elif data != "order":
      data["structure"][key]["order"] = int(line)

state = None
for i, line in enumerate(activities.split("\n")):
  
  if (i % 7 == 0):
      key = line
      data["activities"][key] = {}
  elif line in ["completed", "scheduling", "activity-id"]:
      state = line
  elif state == "completed":
      data["activities"][key][state] = (line == 'true')
  else:
      data["activities"][key][state] = line[1:-1]

state = 0
data["measures"] = {}
subkey = None
for i, line in enumerate(measures.split("\n")):
  if (i % 7 == 0):
    key = line
    data["measures"][key] = {}
  elif state == 0:
    subkey = line
    state = 1
  elif state == 1:
    data["measures"][key][subkey] = float(line)
    state = 0

data["measure_library"] = {
  "_categories": {
      "state-prediction": {
          "data-range-max": 1,
          "data-range-min": 0,
          "data-type": "float"},
      "state-subjective": {
          "data-range-max": 1,
          "data-range-min": 0,
          "data-type": "float"}
          },
  "calm-subjective": {"category": "state-subjective"},
  "calm-predicted": {"category": "state-predicted"},
  "confidence-subjective": {"category": "state-subjective"},
  "confidence-predicted": {"category": "state-predicted"},
  "energy-subjective": {"category": "state-subjective"},
  "energy-predicted": {"category": "state-predicted"},
  "flow-subjective": {"category": "state-subjective"},
  "flow-predicted": {"category": "state-predicted"},
  "focus-subjective": {"category": "state-subjective"},
  "focus-predicted": {"category": "state-predicted"},
  "happiness-subjective": {"category": "state-subjective"},
  "happiness-predicted": {"category": "state-predicted"},
  "kindness-subjective": {"category": "state-subjective"},
  "kindness-predicted": {"category": "state-predicted"},
  "optimism-subjective": {"category": "state-subjective"}
}

state = None
data["activities_lib"] = {}
for i, line in enumerate(activities_lib.split("\n")):
  if (i % 5 == 0):
    key = line
    data["activities_lib"][key] = {}
  elif line in ["navigates", "name"]:
    state = line
  elif state == "navigates":
      data["activities_lib"][key][state] = (line == "true")
  else:
    data["activities_lib"][key][state] = line[1:-1]
    
import json

with open("dev_data.json", "w") as f:
  f.write(json.dumps(data))