# -*- coding: utf-8 -*-
import io
import re
from setuptools import setup, find_packages
import sys


# BU SATIRLAR SORUNUN KALICI ÇÖZÜMÜDÜR.
# Python'a, README.md dosyasını hangi işletim sisteminde olursa olsun
# her zaman UTF-8 kodlamasıyla okumasını söylüyoruz.
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

setup(
    name="kececinumbers",
    #version="0.4.1",
    description="Keçeci Numbers: An Exploration of a Dynamic Sequence Across Diverse Number Sets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mehmet Keçeci",
    maintainer="Mehmet Keçeci",
    author_email="mkececi@yaani.com",
    maintainer_email="mkececi@yaani.com",
    url="https://github.com/WhiteSymmetry/kececinumbers",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "numpy-quaternion",
        "sympy",
    ],
    extras_require={

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.10',
    license="MIT",
)
