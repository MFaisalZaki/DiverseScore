#!/usr/bin/env python3

from setuptools import setup, find_packages  # type: ignore
import diversescore

# setup.py
setup(
    name="DiverseScore",
    version=diversescore.__version__,
    description="A python package the computes the diversity score using different diversity models and distance functions.",
    author="Mustafa Abdelwahed",
    author_email="ma342@st-andrews.ac.uk",
    packages=find_packages(),
    install_requires=[ ],
    include_package_data=True,
    py_modules=["diversescore"],
    entry_points={
        "console_scripts": [
            "diversescorecli = diversescore.cmd.diversescorecli:main",
        ],
    },
)
