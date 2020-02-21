# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Objects shared across multiple functions."""

from absl import logging

from firebase_admin import auth


def require_admin_auth(func):
  # TODO
  pass


class User(object):
  def __init__(self, uid=None, display_name=None, email=None,
               email_verified=False):
    self.uid = uid
    self.display_name = display_name
    self.email = email
    self.email_verified = email_verified


class Context(object):
  def __init__(self, response_headers={}, user=User()):
    self.response_headers = response_headers
    self.user = user


def require_user_auth(func):
    def wrapper_require_user_auth(request):

        import pprint
        pprint.pprint(request.headers)

        # The require_user_auth always wraps functions that
        # operate on request objects.
        ctx = Context()

        #allow_origin = "https://demo.cl4rify.org"
        allow_origin = "*"
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': allow_origin,
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Authorization',
                'Access-Control-Max-Age': '3600',
                'Access-Control-Allow-Credentials': 'true'
            }

            return ('', 204, headers)

        # Set CORS headers for main requests
        headers = {
            'Access-Control-Allow-Origin': allow_origin,
            'Access-Control-Allow-Credentials': 'true'
        }
        ctx.response_headers = headers

        decoded_token = {}

        if not hasattr(request, "headers"):
          msg = 'Not authorized. Also, the request did not have any headers??'
          return (msg, 403, {})

        if "Authorization" not in request.headers:
          # There should always be an auth code in the header so if this
          # happens something is wrong with the function or its deployment.
          logging.warning("Saw a request without an authorization token...")
          return ('Not authorized. Auth token was not provided.', 403, {})

        import firebase_admin
        default_app = firebase_admin.initialize_app()

        try:
          id_token = request.headers["Authorization"].split(" ")[1]
          decoded_token = auth.verify_id_token(id_token,
                                               check_revoked=True)
        except auth.RevokedIdTokenError:
          logging.warning('Access attempt with revoked token.')
          msg = "Not authorized. Attempted access with revoked token."
          return (msg, 403, headers)
        except auth.InvalidIdTokenError:
          logging.warning('Access attempt with invalid token.')
          msg = "Not authorized. Attempted access with invalid token."
          return (msg, 403, headers)

        ctx.user.uid = decoded_token['uid']
        user_info = auth.get_user(decoded_token['uid'])
        ctx.user.display_name = user_info["display_name"]
        ctx.user.email = user_info["email"]
        ctx.user.email_verified = user_info["email_verified"]

        return func(request, ctx)

    return wrapper_require_user_auth
