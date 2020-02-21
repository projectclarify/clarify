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
"""Return the right rendered page or trigger alt. (e.g. reg.)."""

from ftw_meta import common

def maybe_render(request, ctx=None):
  """Title.

    Args:
      request: An https request.

    Returns:
      A tuple (data, code, headers).
  """
  return ("Hello world", 200, {})
