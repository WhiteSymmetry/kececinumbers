# setup.py
import io
import os
import re
from setuptools import setup, find_packages
import sys

def get_version():
    with open('adnus/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="adnus",
    version=get_version(),
    description="A Python library for Advanced Number Systems (AdNuS), including Bicomplex, Neutrosophic, Hyperreal numbers, reals, Complex, Quaternion, Octonion, Sedenion, Pathion, Chingon, Routon, Voudon.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mehmet Keçeci",
    author_email="enfo@tuta.io",
    maintainer="Mehmet Keçeci",
    maintainer_email="enfo@tuta.io",
    url="https://github.com/WhiteSymmetry/adnus",
    package_dir={"": "adnus"},
    packages=find_packages(where="adnus"),
    license="AGPL-3.0-or-later",
    install_requires=[
        "numpy>=2.4.1",
    ],
    extras_require={
        'dev': ["pytest>=9.0.2"],
        'build': ["setuptools>=80.10.1", "cython>=3.2.4"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires='>=3.11',
)
