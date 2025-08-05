.. _api_reference:

#############
API Reference
#############

This section provides auto-generated documentation for the public API of the ``kececinumbers`` module. It includes high-level functions, core components, custom classes, and constants.

For this to work correctly, your Python code should have well-written docstrings following a standard format (like Google or NumPy style).

High-Level Functions
====================
These are the main functions intended for direct user interaction.

.. autofunction:: kececinumbers.get_with_params
.. autofunction:: kececinumbers.get_interactive
.. autofunction:: kececinumbers.get_random_type


Core Generation & Analysis
==========================
These functions provide the core logic for sequence generation and mathematical analysis.

.. autofunction:: kececinumbers.unified_generator
.. autofunction:: kececinumbers.find_kececi_prime_number
.. autofunction:: kececinumbers.is_prime


Custom Number Classes
=====================
These classes define the custom algebraic structures supported by the library.

.. autoclass:: kececinumbers.NeutrosophicNumber
   :members:

.. autoclass:: kececinumbers.NeutrosophicComplexNumber
   :members:

.. autoclass:: kececinumbers.HyperrealNumber
   :members:

.. autoclass:: kececinumbers.BicomplexNumber
   :members:

.. autoclass:: kececinumbers.NeutrosophicBicomplexNumber
   :members:


Visualization & Reporting
=========================

.. autofunction:: kececinumbers.plot_numbers
.. autofunction:: kececinumbers.print_detailed_report


Module Constants
================
These constants are used to specify the number type in functions like ``get_with_params``.

.. rubric:: Keçeci Number Types

.. data:: kececinumbers.TYPE_POSITIVE_REAL
.. data:: kececinumbers.TYPE_NEGATIVE_REAL
.. data:: kececinumbers.TYPE_COMPLEX
.. data:: kececinumbers.TYPE_FLOAT
.. data:: kececinumbers.TYPE_RATIONAL
.. data:: kececinumbers.TYPE_QUATERNION
.. data:: kececinumbers.TYPE_NEUTROSOPHIC
.. data:: kececinumbers.TYPE_NEUTROSOPHIC_COMPLEX
.. data:: kececinumbers.TYPE_HYPERREAL
.. data:: kececinumbers.TYPE_BICOMPLEX
.. data:: kececinumbers.TYPE_NEUTROSOPHIC_BICOMPLEX
