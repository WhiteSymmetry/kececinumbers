.. _quickstart:

############
Quick Start
############

This guide will walk you through the process of generating and visualizing your first Keçeci Number sequence in just a few lines of code. We will use the ``get_with_params`` function to create a sequence of complex numbers.

Step 1: Import the Necessary Libraries
=======================================

First, we need to import the ``kececinumbers`` library itself and ``matplotlib.pyplot`` for displaying the final plot.

.. code-block:: python

   import matplotlib.pyplot as plt
   import kececinumbers as kn

Step 2: Generate the Keçeci Sequence
=====================================

Next, we will call the ``get_with_params`` function. This function is a high-level wrapper that handles the sequence generation based on the parameters you provide.

*   ``kececi_type_choice``: We will use ``kn.TYPE_COMPLEX`` to specify that we want to work with complex numbers.
*   ``iterations``: This sets the number of Keçeci steps the algorithm will perform. Let's use 60.
*   ``start_value_raw``: The initial value for the sequence. For complex numbers, this is a string like ``"1+2j"``.
*   ``add_value_base_scalar``: The scalar value used to construct the increment for each step.

.. code-block:: python

   # Define parameters and generate the sequence
   sequence = kn.get_with_params(
       kececi_type_choice=kn.TYPE_COMPLEX,
       iterations=60,
       start_value_raw="1+2j",
       add_value_base_scalar=3.0
   )

Step 3: Visualize the Sequence
===============================

The ``kececinumbers`` library includes a powerful ``plot_numbers`` function that automatically creates a detailed visualization tailored to the number type. For complex numbers, it will generate a plot showing the trajectory in the complex plane.

We must call ``plt.show()`` to display the plot window.

.. code-block:: python

   # If the sequence was generated successfully, plot it
   if sequence:
       kn.plot_numbers(sequence, title="My First Complex Keçeci Sequence")
       plt.show()

Full Code Example
=================

Here is the complete script:

.. code-block:: python

   import matplotlib.pyplot as plt
   import kececinumbers as kn

   # Generate a Keçeci sequence with specific parameters
   sequence = kn.get_with_params(
       kececi_type_choice=kn.TYPE_COMPLEX,
       iterations=60,
       start_value_raw="1+2j",
       add_value_base_scalar=3.0
   )

   # If the sequence was generated successfully, plot it
   if sequence:
       kn.plot_numbers(sequence, title="My First Complex Keçeci Sequence")
       plt.show()

       # Optionally, find and print the Keçeci Prime Number (KPN)
       kpn = kn.find_kececi_prime_number(sequence)
       if kpn:
           print(f"\nKeçeci Prime Number (KPN) found: {kpn}")


Expected Output
===============

Running the script will print a summary to your console and then display a plot.

**Console Output:**

.. code-block:: text

   --- Generating Sequence: Type 3, Steps 60 ---
   Start: '1+2j', Increment: 3.0
   Generated 181 numbers. Preview: [(1+2j), (4+5j), (7+8j)]...
   Keçeci Prime Number for this sequence: 7

**Plot:**

.. image:: https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-2.png?raw=true
   :alt: Example plot of a complex Keçeci Number sequence

Next Steps
==========

Congratulations! You've created your first Keçeci Number sequence. From here, you can:

*   Explore other number types by changing ``kececi_type_choice``.
*   Try the interactive mode with ``kn.get_interactive()``.
*   Dive into the :ref:`api_reference` to understand all available functions.
