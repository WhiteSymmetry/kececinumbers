.. _installation:

############
Installation
############

This guide provides instructions for installing the ``kececinumbers`` library. You can install the package from PyPI, Conda, or directly from the source code.

Prerequisites
=============

Before you begin, ensure you have the following installed:

*   **Python 3.8 or newer**.
*   **pip** (the Python package installer), which is usually included with modern Python installations.
*   (Optional) The **Conda** package manager, if you prefer installing through the Conda ecosystem.

Option 1: Install with pip (Recommended)
=========================================

This is the most common and recommended way to install ``kececinumbers`` for most users.

Open your terminal or command prompt and run the following command:

.. code-block:: bash

   pip install kececinumbers

This will download and install the latest stable version of the library from the Python Package Index (PyPI), along with all its required dependencies.

Option 2: Install with Conda
============================

If you use the Anaconda or Miniconda distribution, you can install the package from the ``bilgi`` channel on Anaconda Cloud.

.. code-block:: bash

   conda install -c bilgi kececinumbers

This command ensures that the package and its dependencies are managed within your Conda environment.

Option 3: Install from Source
=============================

If you want to get the very latest, unreleased features or wish to contribute to the project, you can install it directly from the source code on GitHub.

1.  **Clone the repository:**
    First, you need to clone the project's repository to your local machine using Git.

    .. code-block:: bash

       git clone https://github.com/WhiteSymmetry/kececinumbers.git

2.  **Navigate to the directory:**
    Change into the newly created project directory.

    .. code-block:: bash

       cd kececinumbers

3.  **Install the package:**
    Install the package in editable mode using pip. This is useful for development as any changes you make to the source code will be immediately effective.

    .. code-block:: bash

       pip install -e .

Verifying the Installation
==========================

To ensure that the library was installed correctly, you can open a Python interpreter and try to import it:

.. code-block:: python

   import kececinumbers as kn

   print(f"Successfully installed kececinumbers version: {kn.__version__}")

If this command runs without any errors, the installation was successful. You are now ready to use the library!
