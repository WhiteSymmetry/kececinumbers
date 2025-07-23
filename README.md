# Keçeci Numbers: Keçeci Sayıları

[![PyPI version](https://badge.fury.io/py/kececinumbers.svg)](https://badge.fury.io/py/kececinumbers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15377659.svg)](https://doi.org/10.5281/zenodo.15377659)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.14.1-blue)](https://doi.org/10.48546/workflowhub.datafile.14.1)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/version.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/platforms.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/license.svg)](https://anaconda.org/bilgi/kececinumbers)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececinumbers/badge/?0.2.0=main)](https://kececinumbers.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10536/badge)](https://www.bestpractices.dev/projects/10536)

[![Python CI](https://github.com/WhiteSymmetry/kececinumbers/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/kececinumbers/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/kececinumbers/graph/badge.svg?token=0X78S7TL0W)](https://codecov.io/gh/WhiteSymmetry/kececinumbers)
[![Documentation Status](https://readthedocs.org/projects/kececinumbers/badge/?version=latest)](https://kececinumbers.readthedocs.io/en/latest/)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececinumbers/HEAD)
[![PyPI version](https://badge.fury.io/py/kececinumbers.svg)](https://badge.fury.io/py/kececinumbers)
[![PyPI Downloads](https://static.pepy.tech/badge/kececinumbers)](https://pepy.tech/projects/kececinumbers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) 

---

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI</td>
            <td style="text-align: center;">
                <a href="https://pypi.org/project/kececinumbers/">
                    <img src="https://badge.fury.io/py/kececinumbers.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">Conda</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/bilgi/kececinumbers">
                    <img src="https://anaconda.org/bilgi/kececinumbers/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">DOI</td>
            <td style="text-align: center;">
                <a href="https://doi.org/10.5281/zenodo.15377659">
                    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15377659.svg" alt="DOI" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">License: MIT</td>
            <td style="text-align: center;">
                <a href="https://opensource.org/licenses/MIT">
                    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>

---

## Description / Açıklama

**Keçeci Numbers (Keçeci Sayıları)**: Keçeci Numbers; An Exploration of a Dynamic Sequence Across Diverse Number Sets: This work introduces a novel numerical sequence concept termed "Keçeci Numbers." Keçeci Numbers are a dynamic sequence generated through an iterative process, originating from a specific starting value and an increment value. In each iteration, the increment value is added to the current value, and this "added value" is recorded in the sequence. Subsequently, a division operation is attempted on this "added value," primarily using the divisors 2 and 3, with the choice of divisor depending on the one used in the previous step. If division is successful, the quotient becomes the next element in the sequence. If the division operation fails, the primality of the "added value" (or its real/scalar part for complex/quaternion numbers, or integer part for rational numbers) is checked. If it is prime, an "Augment/Shrink then Check" (ASK) rule is invoked: a type-specific unit value is added or subtracted (based on the previous ASK application), this "modified value" is recorded in the sequence, and the division operation is re-attempted on it. If division fails again, or if the number is not prime, the "added value" (or the "modified value" post-ASK) itself becomes the next element in the sequence. This mechanism is designed to be applicable across various number sets, including positive and negative real numbers, complex numbers, floating-point numbers, rational numbers, and quaternions. The increment value, ASK unit, and divisibility checks are appropriately adapted for each number type. This flexibility of Keçeci Numbers offers rich potential for studying their behavior in different numerical systems. The patterns exhibited by the sequences, their convergence/divergence properties, and potential for chaotic behavior may constitute interesting research avenues for advanced mathematical analysis and number theory applications. This study outlines the fundamental generation mechanism of Keçeci Numbers and their initial behaviors across diverse number sets.

---

## Installation / Kurulum

```bash
conda install bilgi::kececinumbers -y

pip install kececinumbers
```
https://anaconda.org/bilgi/kececinumbers

https://pypi.org/project/kececinumbers/

https://github.com/WhiteSymmetry/kececinumbers

https://zenodo.org/records/15377660

https://zenodo.org/records/

---

## Usage / Kullanım

### Example

```python
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from fractions import Fraction
import quaternion # pip install numpy numpy-quaternion
```
```python
import matplotlib.pyplot as plt
import kececinumbers as kn

# Matplotlib grafiklerinin notebook içinde gösterilmesini sağla
%matplotlib inline

print("Trying interactive mode (will prompt for input in the console/output area)...")
interactive_sequence = kn.get_interactive()
if interactive_sequence:
    kn.plot_numbers(interactive_sequence, title="Keçeci Numbers")

print("Done with examples.")
print("Keçeci Numbers Module Loaded.")
print("This module provides functions to generate and plot Keçeci Numbers.")
print("Example: Use 'import kececinumbers as kn' in your script/notebook.")
print("\nAvailable functions:")
print("- kn.get_interactive()")
print("- kn.get_with_params(kececi_type, iterations, ...)")
print("- kn.get_random_type(iterations, ...)")
print("- kn.plot_numbers(sequence, title)")
print("- kn.unified_generator(...) (low-level)")
print("\nAccess definitions with: kn.DEFINITIONS")
print("\nAccess type constants like: kn.TYPE_COMPLEX")
```
---
Trying interactive mode (will prompt for input in the console/output area)...

Keçeci Number Types:

1: Positive Real Numbers (Integer: e.g., 1)

2: Negative Real Numbers (Integer: e.g., -3)

3: Complex Numbers (e.g., 3+4j)

4: Floating-Point Numbers (e.g., 2.5)

5: Rational Numbers (e.g., 3/2, 5)

6: Quaternions (scalar start input becomes q(s,s,s,s): e.g.,  1 or 2.5)

Please select Keçeci Number Type (1-6):  1

Enter the starting number (e.g., 0 or 2.5, complex:3+4j, rational: 3/4, quaternions: 1)  :  0

Enter the base scalar value for increment (e.g., 9):  9

Enter the number of iterations (positive integer: e.g., 30):  30

---
![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-1.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-2.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-3.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-4.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-5.png?raw=true)

---
# Keçeci Prime Number

```python
import matplotlib.pyplot as plt
import kececinumbers as kn


print("--- Interactive Test ---")
seq_interactive = kn.get_interactive()
if seq_interactive:
    kn.plot_numbers(seq_interactive, "Keçeci Numbers")

print("\n--- Random Type Test (60 Keçeci Steps) ---")
# num_iterations burada Keçeci adımı sayısıdır
seq_random = kn.get_random_type(num_iterations=60) 
if seq_random:
    kn.plot_numbers(seq_random, "Random Type Keçeci Numbers")

print("\n--- Fixed Params Test (Complex, 60 Keçeci Steps) ---")
seq_fixed = kn.get_with_params(
    kececi_type_choice=kn.TYPE_COMPLEX, 
    iterations=60, 
    start_value_raw="1+2j", 
    add_value_base_scalar=3.0
)
if seq_fixed:
    kn.plot_numbers(seq_fixed, "Fixed Params (Complex) Keçeci Numbers")

# İsterseniz find_kececi_prime_number'ı ayrıca da çağırabilirsiniz:
if seq_fixed:
    kpn_direct = kn.find_kececi_prime_number(seq_fixed)
    if kpn_direct is not None:
        print(f"\nDirect call to find_kececi_prime_number for fixed numbers: {kpn_direct}")
```

Generated Keçeci Sequence (first 20 of 121): [4, 11, 12, 4, 11, 10, 5, 12, 4, 11, 12, 6, 13, 12, 4, 11, 12, 6, 13, 12]...
Keçeci Prime Number for this sequence: 11

--- Random Type Test (60 Keçeci Steps) ---

Randomly selected Keçeci Number Type: 1 (Positive Integer)

Generated Keçeci Sequence (using get_with_params, first 20 of 61): [0, 9, 3, 12, 6, 15, 5, 14, 7, 16, 8, 17, 18, 6, 15, 5, 14, 7, 16, 8]...
Keçeci Prime Number for this sequence: 17

---

## License / Lisans

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@misc{kececi_2025_15377659,
  author       = {Keçeci, Mehmet},
  title        = {kececinumbers},
  month        = may,
  year         = 2025,
  publisher    = {PyPI, Anaconda, Github, Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.15377659},
  url          = {https://doi.org/10.5281/zenodo.15377659},
}
```

### APA

```
Keçeci, M. (2025). Geometric Interpretations of Keçeci Numbers with Neutrosophic and Hyperreal Numbers. Zenodo. https://doi.org/10.5281/zenodo.16344232

Keçeci, M. (2025). Keçeci Sayılarının Nötrosofik ve Hipergerçek Sayılarla Geometrik Yorumlamaları. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.16343568

Keçeci, M. (2025). kececinumbers [Data set]. WorkflowHub. https://doi.org/10.48546/workflowhub.datafile.14.1

Keçeci, M. (2025). kececinumbers. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15377659

Keçeci, M. (2025). Keçeci Numbers and the Keçeci Prime Number: A Potential Number Theoretic Exploratory Tool. https://doi.org/10.5281/zenodo.15381698

Keçeci, M. (2025). Diversity of Keçeci Numbers and Their Application to Prešić-Type Fixed-Point Iterations: A Numerical Exploration. https://doi.org/10.5281/zenodo.15481711

Keçeci, M. (2025). Keçeci Numbers and the Keçeci Prime Number. Authorea. June 02, 2025. https://doi.org/10.22541/au.174890181.14730464/v1

Keçeci, M. (2025, May 11). Keçeci numbers and the Keçeci prime number: A potential number theoretic exploratory tool. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15381697
```

### Chicago
```
Keçeci, Mehmet. kececinumbers [Data set]. WorkflowHub, 2025. https://doi.org/10.48546/workflowhub.datafile.14.1

Keçeci, Mehmet. "kececinumbers". Open Science Articles (OSAs), Zenodo, 01 May 2025. https://doi.org/10.5281/zenodo.15377659

Keçeci, Mehmet. "Keçeci Numbers and the Keçeci Prime Number: A Potential Number Theoretic Exploratory Tool", 11 Mayıs 2025. https://doi.org/10.5281/zenodo.15381698

Keçeci, Mehmet. "Diversity of Keçeci Numbers and Their Application to Prešić-Type Fixed-Point Iterations: A Numerical Exploration". https://doi.org/10.5281/zenodo.15481711

Keçeci, Mehmet. "Keçeci Numbers and the Keçeci Prime Number". Authorea. June 02, 2025. https://doi.org/10.22541/au.174890181.14730464/v1

Keçeci, Mehmet. Keçeci numbers and the Keçeci prime number: A potential number theoretic exploratory tool. Open Science Articles (OSAs), Zenodo. 2025. https://doi.org/10.5281/zenodo.15381697
```
