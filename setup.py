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

with open("kececinumbers/kececinumbers.py", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("kececinumbers/__init__.py", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("kececinumbers/_version.py", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("tests/test_sample.py", "r", encoding="utf-8") as f:
    long_description = f.read()

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
    packages=find_packages(),  # 'kececinumbers' klasörünü otomatik bulur
    package_data={
        "kececinumbers": ["__init__.py", "_version.py"]  # Gerekli dosyaları dahil et
    },
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "sympy",
    ],
    extras_require={
        'quaternion': ["numpy-quaternion"],  # pip için
        'quaternion-conda': ["quaternion"],  # conda için
        'all': ["numpy-quaternion"],  # Varsayılan pip
        'test': [
            "pytest",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.10',
    license="MIT",
)
