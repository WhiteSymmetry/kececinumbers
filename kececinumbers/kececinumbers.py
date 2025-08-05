# kececinumbers.py (Ruff-Formatted and Refactored)

# -*- coding: utf-8 -*-
"""
Keçeci Numbers Module (kececinumbers.py)

This module provides a comprehensive framework for generating, analyzing, and
visualizing Keçeci Numbers across various number systems. It supports 11
distinct types, from standard integers and complex numbers to more exotic
constructs like neutrosophic and bicomplex numbers.

The core of the module is the `unified_generator`, which implements the
specific algorithm for creating Keçeci Number sequences. High-level functions
are available for easy interaction, parameter-based generation, and plotting.

Key Features:
- Generation of 11 types of Keçeci Numbers.
- A robust, unified algorithm for all number types.
- Helper functions for mathematical properties like primality and divisibility.
- Advanced plotting capabilities tailored to each number system.
- Functions for interactive use or programmatic integration.
"""

# --- Standard Library Imports ---
import collections
import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, List, Optional, Tuple

# --- Third-Party Imports ---
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from matplotlib.gridspec import GridSpec

# ==============================================================================
# --- MODULE CONSTANTS: KEÇECI NUMBER TYPES ---
# ==============================================================================
TYPE_POSITIVE_REAL = 1
TYPE_NEGATIVE_REAL = 2
TYPE_COMPLEX = 3
TYPE_FLOAT = 4
TYPE_RATIONAL = 5
TYPE_QUATERNION = 6
TYPE_NEUTROSOPHIC = 7
TYPE_NEUTROSOPHIC_COMPLEX = 8
TYPE_HYPERREAL = 9
TYPE_BICOMPLEX = 10
TYPE_NEUTROSOPHIC_BICOMPLEX = 11

# ==============================================================================
# --- CUSTOM NUMBER CLASS DEFINITIONS ---
# ==============================================================================

@dataclass
class NeutrosophicNumber:
    """Represents a neutrosophic number of the form a + bI, where I^2 = I."""
    a: float
    b: float

    def __add__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.a + other.a, self.b + other.b)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.a + other, self.b)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.a - other.a, self.b - other.b)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.a - other, self.b)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            # (a + bI)(c + dI) = ac + (ad + bc + bd)I
            return NeutrosophicNumber(
                self.a * other.a,
                self.a * other.b + self.b * other.a + self.b * other.b
            )
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.a * other, self.b * other)
        return NotImplemented

    def __truediv__(self, divisor: float) -> "NeutrosophicNumber":
        if isinstance(divisor, (int, float)):
            return NeutrosophicNumber(self.a / divisor, self.b / divisor)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        return f"{self.a} + {self.b}I"

@dataclass
class NeutrosophicComplexNumber:
    """
    Represents a neutrosophic-complex number (real + imag*j) + indeterminacy*I.
    Models a value with both a complex state and an associated uncertainty.
    """
    real: float = 0.0
    imag: float = 0.0
    indeterminacy: float = 0.0

    def __repr__(self) -> str:
        return f"NeutrosophicComplexNumber(real={self.real}, imag={self.imag}, indeterminacy={self.indeterminacy})"

    def __str__(self) -> str:
        return f"({self.real}{self.imag:+}j) + {self.indeterminacy}I"

    def __add__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy + other.indeterminacy,
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real + other, self.imag, self.indeterminacy)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy,
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real - other, self.imag, self.indeterminacy)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            new_indeterminacy = self.indeterminacy + other.indeterminacy + (self.magnitude_sq() * other.indeterminacy)
            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        if isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real * other, self.imag * other, self.indeterminacy * other)
        return NotImplemented

    def __truediv__(self, divisor: float) -> "NeutrosophicComplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicComplexNumber(
                self.real / divisor,
                self.imag / divisor,
                self.indeterminacy / divisor
            )
        raise TypeError("Only scalar division is supported.")

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def magnitude_sq(self) -> float:
        """Returns the squared magnitude of the deterministic (complex) part."""
        return self.real**2 + self.imag**2

@dataclass
class HyperrealNumber:
    """Represents a hyperreal number as a sequence of real numbers."""
    sequence: list

    def __add__(self, other: "HyperrealNumber") -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            return HyperrealNumber([a + b for a, b in zip(self.sequence, other.sequence)])
        return NotImplemented

    def __sub__(self, other: "HyperrealNumber") -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            return HyperrealNumber([a - b for a, b in zip(self.sequence, other.sequence)])
        return NotImplemented

    def __mul__(self, scalar: float) -> "HyperrealNumber":
        if isinstance(scalar, (int, float)):
            return HyperrealNumber([x * scalar for x in self.sequence])
        return NotImplemented

    def __rmul__(self, scalar: float) -> "HyperrealNumber":
        return self.__mul__(scalar)

    def __truediv__(self, divisor: float) -> "HyperrealNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Scalar division by zero.")
            return HyperrealNumber([x / divisor for x in self.sequence])
        raise TypeError("Only scalar division is supported.")

    def __mod__(self, divisor: float) -> List[float]:
        if isinstance(divisor, (int, float)):
            return [x % divisor for x in self.sequence]
        raise TypeError("Modulo only supported with a scalar divisor.")

    def __str__(self) -> str:
        return f"Hyperreal({self.sequence[:3]}...)"

@dataclass
class BicomplexNumber:
    """Represents a bicomplex number z1 + j*z2, where i^2 = j^2 = -1."""
    z1: complex
    z2: complex

    def __add__(self, other: "BicomplexNumber") -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        return NotImplemented

    def __sub__(self, other: "BicomplexNumber") -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        return NotImplemented

    def __mul__(self, other: "BicomplexNumber") -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            # (z1 + z2j)(w1 + w2j) = (z1w1 - z2w2) + (z1w2 + z2w1)j
            return BicomplexNumber(
                (self.z1 * other.z1) - (self.z2 * other.z2),
                (self.z1 * other.z2) + (self.z2 * other.z1),
            )
        return NotImplemented

    def __truediv__(self, scalar: float) -> "BicomplexNumber":
        if isinstance(scalar, (int, float)):
            return BicomplexNumber(self.z1 / scalar, self.z2 / scalar)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        return f"Bicomplex({self.z1}, {self.z2})"

@dataclass
class NeutrosophicBicomplexNumber:
    """Represents a simplified neutrosophic-bicomplex number for demonstration."""
    real: float
    imag: float
    neut_real: float
    neut_imag: float
    j_real: float
    j_imag: float
    j_neut_real: float
    j_neut_imag: float

    def __add__(self, other: "NeutrosophicBicomplexNumber") -> "NeutrosophicBicomplexNumber":
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(*(a + b for a, b in zip(self.__dict__.values(), other.__dict__.values())))
        return NotImplemented

    def __sub__(self, other: "NeutrosophicBicomplexNumber") -> "NeutrosophicBicomplexNumber":
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(*(a - b for a, b in zip(self.__dict__.values(), other.__dict__.values())))
        return NotImplemented

    def __truediv__(self, scalar: float) -> "NeutrosophicBicomplexNumber":
        if isinstance(scalar, (int, float)):
            return NeutrosophicBicomplexNumber(*(val / scalar for val in self.__dict__.values()))
        raise TypeError("Only scalar division supported.")

    def __str__(self) -> str:
        return f"NeutroBicomplex(r={self.real}, i={self.imag}, Ir={self.neut_real}, ...)"


# ==============================================================================
# --- INTERNAL HELPER FUNCTIONS ---
# ==============================================================================

def _get_integer_representation(n_input: Any) -> int:
    """Extracts the primary integer component from any supported number type."""
    val = 0
    if isinstance(n_input, (int, float, Fraction)): val = abs(int(n_input))
    elif isinstance(n_input, complex): val = abs(int(n_input.real))
    elif isinstance(n_input, np.quaternion): val = abs(int(n_input.w))
    elif isinstance(n_input, NeutrosophicNumber): val = abs(int(n_input.a))
    elif isinstance(n_input, NeutrosophicComplexNumber): val = abs(int(n_input.real))
    elif isinstance(n_input, HyperrealNumber): val = abs(int(n_input.sequence[0])) if n_input.sequence else 0
    elif isinstance(n_input, BicomplexNumber): val = abs(int(n_input.z1.real))
    elif isinstance(n_input, NeutrosophicBicomplexNumber): val = abs(int(n_input.real))
    else:
        try:
            val = abs(int(n_input))
        except (ValueError, TypeError):
            return 0  # Return a non-prime default
    return val

def _is_divisible(value: Any, divisor: int, kececi_type: int) -> bool:
    """Helper to check divisibility for different number types."""
    try:
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            return value % divisor == 0
        if kececi_type == TYPE_FLOAT:
            return math.isclose(value % divisor, 0)
        if kececi_type == TYPE_RATIONAL:
            return (value / divisor).denominator == 1
        if kececi_type == TYPE_COMPLEX:
            return math.isclose(value.real % divisor, 0) and math.isclose(value.imag % divisor, 0)
        if kececi_type == TYPE_QUATERNION:
            return all(math.isclose(c % divisor, 0) for c in [value.w, value.x, value.y, value.z])
        if kececi_type == TYPE_NEUTROSOPHIC:
            return math.isclose(value.a % divisor, 0) and math.isclose(value.b % divisor, 0)
        if kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            return all(math.isclose(c % divisor, 0) for c in [value.real, value.imag, value.indeterminacy])
        if kececi_type == TYPE_HYPERREAL:
            return all(math.isclose(x % divisor, 0) for x in value.sequence)
        if kececi_type == TYPE_BICOMPLEX:
            return _is_divisible(value.z1, divisor, TYPE_COMPLEX) and _is_divisible(value.z2, divisor, TYPE_COMPLEX)
        if kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            return all(math.isclose(c % divisor, 0) for c in value.__dict__.values())
    except (TypeError, ValueError):
        return False
    return False

def _parse_complex(s: str) -> complex:
    """Parses a string into a complex number. '3' becomes 3+3j."""
    s_clean = s.strip().lower()
    try:
        c = complex(s_clean)
        # If input was just a real number string, interpret as c + cj
        if c.imag == 0 and 'j' not in s_clean:
            return complex(c.real, c.real)
        return c
    except ValueError as e:
        raise ValueError(f"Invalid complex number format: '{s}'") from e

def _parse_neutrosophic(s: str) -> Tuple[float, float]:
    """Parses a neutrosophic string 'a+bI' into a tuple (a, b)."""
    s = s.strip().replace(" ", "").upper()
    if 'I' not in s:
        return float(s), 0.0
    
    a_part_str, _, b_part_str = s.partition('I')
    # This simplified regex-based approach handles '5+2I', '3I', '-I', '+I', etc.
    match = re.match(r"^(.*)([+-])(.*)$", a_part_str)
    if match:
        a_str = match.group(1)
        sign = match.group(2)
        b_str = sign + match.group(3)
    else:
        a_str = "0"
        b_str = a_part_str if a_part_str else "1" # Handle 'I' and '-I'
        if b_str == '+': b_str = '1'
        if b_str == '-': b_str = '-1'

    a = float(a_str) if a_str else 0.0
    b = float(b_str)
    return a, b

# ... Other parsing functions like _parse_hyperreal and _parse_quaternion remain similar ...

# ==============================================================================
# --- PUBLIC API FUNCTIONS ---
# ==============================================================================

def is_prime(n_input: Any) -> bool:
    """
    Checks if a given number (or its principal component) is prime.
    """
    value_to_check = _get_integer_representation(n_input)
    if value_to_check < 2: return False
    if value_to_check == 2: return True
    if value_to_check % 2 == 0: return False
    for i in range(3, int(math.sqrt(value_to_check)) + 1, 2):
        if value_to_check % i == 0:
            return False
    return True

def find_kececi_prime_number(kececi_numbers_list: List[Any]) -> Optional[int]:
    """
    Finds the Keçeci Prime Number (KPN) from a sequence.
    KPN is the most frequent prime integer representation, with ties
    broken by choosing the larger prime.
    """
    if not kececi_numbers_list:
        return None

    prime_reps = [_get_integer_representation(num) for num in kececi_numbers_list if is_prime(num)]
    if not prime_reps:
        return None

    counts = collections.Counter(prime_reps)
    repeating_primes = [(freq, prime) for prime, freq in counts.items() if freq > 1]
    if not repeating_primes:
        return None

    # Find the one with the highest frequency, using prime value as a tie-breaker
    _, best_prime = max(repeating_primes)
    return best_prime
