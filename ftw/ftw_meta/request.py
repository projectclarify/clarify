
import requests
import subprocess

def _get_url(zone, project, fname):
  return "https://{}-{}.cloudfunctions.net/{}".format(
    zone, project, fname
  )

url = _get_url("us-central1", "clarify", "check_membership")

tok = subprocess.check_output([
    "gcloud", "auth", "print-identity-token"
]).decode("utf-8").strip()

headers={'Content-Type':'application/json',
         'Authorization': 'Bearer {}'.format(tok)}


try:
  params = {}
  body = {'token': tok, 'returnSecureToken': True}
  response = requests.post(url, headers=headers)
  response = requests.request('post', url, params=params, json=body)
  response.raise_for_status()
  print(response)
except requests.exceptions.RequestException as e:
  print(e)
except Exception as err:
  print(err)
else:
  print("success")