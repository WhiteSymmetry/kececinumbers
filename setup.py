# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="kececinumbers",
    version="0.1.5",
    description="Keçeci Numbers An Exploration of a Dynamic Sequence Across Diverse Number Sets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mehmet Keçeci",
    maintainer="Mehmet Keçeci",
    author_email="bilginomi@yaani.com",
    maintainer_email="bilginomi@yaani.com",
    url="https://github.com/WhiteSymmetry/kececinumbers",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "numpy-quaternion"
    ],
    extras_require={

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    license="MIT",
)
