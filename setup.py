# coding=utf-8
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

"""Install cl4rify for notebook usage."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='clarfiy',
    version='0.0.1',
    description='Project Clarify',
    url='http://github.com/projectclarify/clarfiy',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={},
    scripts=[],
    dependency_links=[
      "/usr/local/src/jax",
      "/usr/local/src/jaxlib"
    ],
    install_requires=[
        'numpy==1.16.2',
        'requests==2.21.0',
        'scipy==1.2.1',
        'six==1.12.0',
        'seaborn==0.9.0',
        'mne==0.17.1',
        'moviepy==1.0.0',
        'kubernetes==9.0.0',
        'rfc3339==6.0',
        'dill==0.2.9',
        'tensorflow_hub',
        'google-cloud-bigtable',
        'google-cloud-pubsub',
        'firebase-admin',
        'google-cloud-firestore==0.31.0',
        'google-auth',
        'requests',
        'requests-toolbelt',
        'scikit-learn>=0.21.3',
        'faced @ https://github.com/iitzco/faced/archive/31ef0d30e1567a06113f49ff4a1202760d952df2.zip',
        'tensor2tensor',
        'google-cloud-logging',
        'pathos',
        'opencv-python',
        'tensorflow-probability==0.7.0',
        'gTTS',
        'sympy',
        'absl-py',
        'funcsigs',
        'trax'
        ],
    extras_require={
        'jax': [
            'jax', 'jaxlib'
        ],
        'tensorflow': [
            'tensorflow==1.15.0'
        ],
        'tensorflow-gpu': [
            'tensorflow-gpu>=1.13.1'
        ],
        'tests': [
            'pylint',
            'pytest',
            'pytest-cache',
            'tensorflow-serving-api==1.12.0'
        ],
        'serving': [
            'requests_toolbelt'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License'
    ],
    keywords='clinical cognitive neuroscience',
)
