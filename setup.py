# -*- coding: utf-8 -*-
import io
import re
from setuptools import setup, find_packages
import sys
import os

# UTF-8 encoding sorunlarını çöz
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_version():
    with open('kececinumbers/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

def get_install_requires():
    """Kurulum bağımlılıklarını dinamik olarak belirle"""
    base_requires = [
        "numpy",
        "matplotlib",
        "scipy",
        "sklearn",
        "sympy",
    ]

setup(
    name="kececinumbers",
    version=get_version(),
    description="Keçeci Numbers: An Exploration of a Dynamic Sequence Across Diverse Number Sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mehmet Keçeci",
    maintainer="Mehmet Keçeci",
    author_email="mkececi@yaani.com",
    maintainer_email="mkececi@yaani.com",
    url="https://github.com/WhiteSymmetry/kececinumbers",
    packages=find_packages(),
    package_data={
        "kececinumbers": ["__init__.py", "_version.py", "*.pyi"]
    },
    install_requires=get_install_requires(),
    extras_require={
        'test': [
            "pytest",
            "pytest-cov",
        ],
        'dev': [
            "pytest",
            "pytest-cov",
            "twine",
            "wheel",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    python_requires='>=3.11',
    license="AGPL-3.0-or-later",
    keywords="mathematics numbers quaternion hypercomplex kececi",
    project_urls={
        "Documentation": "https://github.com/WhiteSymmetry/kececinumbers",
        "Source": "https://github.com/WhiteSymmetry/kececinumbers",
        "Tracker": "https://github.com/WhiteSymmetry/kececinumbers/issues",
    },
)
