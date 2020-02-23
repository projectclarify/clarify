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
    name='ftw-meta',
    version='0.0.1',
    description='meta ui loader for project clarify ftw',
    url='http://github.com/projectclarify/clarfiy',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        'absl-py',
        'firebase-admin'
        ],
    extras_require={
        'tests': [
            'pylint',
            'pytest',
            'pytest-cache',
            'tensorflow-serving-api==1.12.0'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License'
    ]
)
