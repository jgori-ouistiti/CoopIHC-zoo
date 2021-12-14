# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['coopihczoo',
 'coopihczoo.examples',
 'coopihczoo.examples.basic',
 'coopihczoo.examples.rl',
 'coopihczoo.eye',
 'coopihczoo.pointing',
 'coopihczoo.pointing.test']

package_data = \
{'': ['*']}

install_requires = \
['PyYAML>=6.0,<7.0',
 'numpy>=1.21.4,<2.0.0',
 'stable-baselines3>=1.3.0,<2.0.0',
 'torch>=1.10.0,<2.0.0',
 'websockets>=10.1,<11.0']

setup_kwargs = {
    'name': 'coopihczoo',
    'version': '0.0.4',
    'description': 'CoopIHC-zoo: Collection of tasks and agents for CoopIHC',
    'long_description': None,
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.11',
}


setup(**setup_kwargs)
