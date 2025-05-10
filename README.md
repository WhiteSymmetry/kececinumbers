# Keçeci Numbers: Keçeci Sayıları

[![PyPI version](https://badge.fury.io/py/kececinumbers.svg)](https://badge.fury.io/py/kececinumbers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15377660.svg)](https://doi.org/10.5281/zenodo.15377660)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.*.svg)](https://doi.org/10.5281/zenodo.*)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/version.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/platforms.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/license.svg)](https://anaconda.org/bilgi/kececinumbers)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececinumbers/badge/?0.1.2=main)](https://kececinumbers.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10536/badge)](https://www.bestpractices.dev/projects/10536)

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

## License / Lisans

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@misc{kececi_2025_15313947,
  author       = {Keçeci, Mehmet},
  title        = {kececinumbers},
  month        = may,
  year         = 2025,
  publisher    = {PyPI, Anaconda, Github, Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.15377660},
  url          = {https://doi.org/10.5281/zenodo.15377660},
}

@misc{kececi_2025_15314329,
  author       = {Keçeci, Mehmet},
  title        = {Keçeci numbers},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.},
  url          = {https://doi.org/10.5281/zenodo.},
}
```

### APA

```
Keçeci, M. (2025). kececinumbers (0.1.0). PyPI, Anaconda, GitHub, Zenodo. https://doi.org/10.5281/zenodo.15377660

Keçeci, M. (2025). Keçeci Numbers. https://doi.org/10.5281/zenodo.
```

### Chicago
```
Keçeci, Mehmet. "kececinumbers". PyPI, Anaconda, GitHub, Zenodo, 01 May 2025. https://doi.org/10.5281/zenodo.15377660

Keçeci, Mehmet. "Keçeci Numbers", 01 May 2025. https://doi.org/10.5281/zenodo.
```
