.. kececinumbers documentation master file, created by
   sphinx-quickstart on Tue Aug 05 11:00:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###########################################
Welcome to Keçeci Numbers's Documentation!
###########################################

.. image:: https://badge.fury.io/py/kececinumbers.svg
   :target: https://pypi.org/project/kececinumbers/
.. image:: https://anaconda.org/bilgi/kececinumbers/badges/version.svg
   :target: https://anaconda.org/bilgi/kececinumbers
.. image:: https://readthedocs.org/projects/kececinumbers/badge/?version=latest
   :target: https://kececinumbers.readthedocs.io/en/latest/
.. image:: https://github.com/WhiteSymmetry/kececinumbers/actions/workflows/python_ci.yml/badge.svg?branch=main
   :target: https://github.com/WhiteSymmetry/kececinumbers/actions/workflows/python_ci.yml
.. image:: https://codecov.io/gh/WhiteSymmetry/kececinumbers/graph/badge.svg?token=0X78S7TL0W
   :target: https://codecov.io/gh/WhiteSymmetry/kececinumbers
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15377659.svg
   :target: https://doi.org/10.5281/zenodo.15377659

**Keçeci Numbers** is a Python library for generating, analyzing, and visualizing dynamic sequences inspired by the Collatz Conjecture across diverse number systems.

This library provides a unified algorithm that operates on 11 different number types, from standard integers to complex algebraic structures like quaternions and neutrosophic numbers. It is designed as a tool for academic research and exploration in number theory.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   examples
   conjecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/kececinumbers

.. toctree::
   :maxdepth: 2
   :caption: Project Info

   citation
   contributing
   license


What are Keçeci Numbers?
========================

Keçeci Numbers are sequences generated from a specific starting value using a recursive rule. The process for each step is as follows:

1.  **Add:** An increment value is added to the current value. This "added value" is recorded in the sequence.
2.  **Divide:** An attempt is made to divide this "added value" by 3 or 2 (whichever was not used in the previous step).
3.  **ASK (Augment/Shrink then Check) Rule:** If the number is indivisible and **prime**, a type-specific unit value is added or subtracted. This "modified value" is recorded, and the division is re-attempted.
4.  **Carry Over:** If division fails, the number itself (either the "added value" or "modified value") becomes the next element in the sequence.


Key Features
============

*   **11 Different Number Types:** Supports integers, rationals, complex numbers, quaternions, neutrosophic numbers, and more.
*   **Unified Generator:** Uses a single, consistent ``unified_generator`` algorithm for all number types.
*   **Advanced Visualization:** Provides a multi-dimensional ``plot_numbers`` function tailored to the nature of each number system.
*   **Keçeci Prime Number (KPN) Analysis:** Identifies the most recurring prime representation in sequences to analyze their convergence behavior.
*   **Interactive and Programmatic Usage:** Allows for both interactive parameter input (``get_interactive``) and direct use in scripts (``get_with_params``).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
