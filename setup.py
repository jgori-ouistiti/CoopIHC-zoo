#!/usr/bin/env python

"""CoopIHC-zoo: Collection of tasks and agents for CoopIHC
"""

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="coopihc-zoo",
    version="0.0.1",
    description="Collection of tasks and agents for CoopIHC",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jgori-ouistiti/CoopIHC-zoo",
    author="Julien Gori",
    author_email="gori@isir.upmc.fr",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "coopihc",
        "matplotlib",
        "websockets",
    ],
)
