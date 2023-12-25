#!/usr/bin/env python3

from setuptools import setup, find_packages  # type: ignore


import subprocess
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import os
import urllib
import shutil
import sys

import diversescore

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # Mark us as not a pure python package
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            # We don't link with python ABI, but require python3
            python, abi = 'py3', 'none'
            return python, abi, plat
except ImportError:
    bdist_wheel = None

IBM_DIVERSESCORE_NAME = 'ibm-diversescore'
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
IBM_DIVERSESCORE_DIR = os.path.join(CURRENT_DIR, IBM_DIVERSESCORE_NAME)

def clone_and_compile_ibm_diversescore():
    try:
        subprocess.run(['git', 'clone', 'https://github.com/IBM/diversescore.git', IBM_DIVERSESCORE_NAME], cwd=CURRENT_DIR)
        # Apply patch
        patches = []
        patches.append(os.path.join(CURRENT_DIR, "diversescore", "patches", "diverscore.1.patch"))
        for patch in patches:
            try:
                subprocess.check_call(['git','apply', patch], cwd=CURRENT_DIR)
            except:
                pass
    except:
        pass
    subprocess.run([sys.executable, 'build.py'], cwd=IBM_DIVERSESCORE_DIR)

class install_ibm_diverse_score(build_py):
    """Custom install command."""
    def run(self):
        clone_and_compile_ibm_diversescore()
        build_py.run(self)

class install_ibm_diverse_score_develop(develop):
    """Custom install command."""
    def run(self):
        clone_and_compile_ibm_diversescore()
        develop.run(self)

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
    package_data={
        "": [f'{IBM_DIVERSESCORE_DIR}/fast-downward.py',
            f'{IBM_DIVERSESCORE_DIR}/README.md', 
            f'{IBM_DIVERSESCORE_DIR}/LICENSE.md',
            f'{IBM_DIVERSESCORE_DIR}/builds/release/bin/*',
            f'{IBM_DIVERSESCORE_DIR}/builds/release/bin/translate/*',
            f'{IBM_DIVERSESCORE_DIR}/builds/release/bin/translate/pddl/*',
            f'{IBM_DIVERSESCORE_DIR}/builds/release/bin/translate/pddl_parser/*',
            f'{IBM_DIVERSESCORE_DIR}/driver/*', 
            f'{IBM_DIVERSESCORE_DIR}/driver/portfolios/*']
      },
    cmdclass={
        'bdist_wheel': bdist_wheel,
        'build_py': install_ibm_diverse_score,
        'develop': install_ibm_diverse_score_develop,
    },
    has_ext_modules=lambda: True
)
