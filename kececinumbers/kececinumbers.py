### `kececinumbers.py`

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
from dataclasses import dataclass
from fractions import Fraction

# --- Third-Party Imports ---
import matplotlib.pyplot as plt
import numpy as np
import quaternion  # Requires: pip install numpy numpy-quaternion
from matplotlib.gridspec import GridSpec
import re

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
    """
    Represents a neutrosophic number of the form a + bI, where I is the
    indeterminate part and I^2 = I.
    
    Attributes:
        a (float): The determinate part.
        b (float): The indeterminate part.
    """
    a: float
    b: float

    def __add__(self, other):
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.a + other.a, self.b + other.b)
        return NeutrosophicNumber(self.a + other, self.b)

    def __sub__(self, other):
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.a - other.a, self.b - other.b)
        return NeutrosophicNumber(self.a - other, self.b)

    def __mul__(self, other):
        if isinstance(other, NeutrosophicNumber):
            # (a + bI)(c + dI) = ac + (ad + bc + bd)I
            return NeutrosophicNumber(
                self.a * other.a,
                self.a * other.b + self.b * other.a + self.b * other.b
            )
        return NeutrosophicNumber(self.a * other, self.b * other)

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            return NeutrosophicNumber(self.a / divisor, self.b / divisor)
        raise TypeError("Only scalar division is supported for NeutrosophicNumber.")

    def __str__(self):
        return f"{self.a} + {self.b}I"

@dataclass
class NeutrosophicComplexNumber:
    """
    Represents a neutrosophic-complex number, combining a standard complex number
    (real + imag*j) with an independent level of indeterminacy (I).
    
    This object models systems where a value has both a complex-valued state
    (like quantum amplitude) and an associated level of uncertainty or
    unreliability (like quantum decoherence).

    Attributes:
        real (float): The real part of the deterministic component.
        imag (float): The imaginary part of the deterministic component.
        indeterminacy (float): The coefficient of the indeterminate part, I.
    """

    def __init__(self, real: float = 0.0, imag: float = 0.0, indeterminacy: float = 0.0):
        """
        Initialises a NeutrosophicComplexNumber.
        
        Args:
            real (float): The initial real part. Defaults to 0.0.
            imag (float): The initial imaginary part. Defaults to 0.0.
            indeterminacy (float): The initial indeterminacy level. Defaults to 0.0.
        """
        self.real = float(real)
        self.imag = float(imag)
        self.indeterminacy = float(indeterminacy)

    def __repr__(self) -> str:
        """
        Returns an unambiguous, developer-friendly representation of the object.
        """
        return f"NeutrosophicComplexNumber(real={self.real}, imag={self.imag}, indeterminacy={self.indeterminacy})"

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the object.
        """
        # Shows a sign for the imaginary part for clarity (e.g., +1.0j, -2.0j)
        return f"({self.real}{self.imag:+}j) + {self.indeterminacy}I"

    # --- Mathematical Operations ---

    def __add__(self, other):
        """Adds another number to this one."""
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy + other.indeterminacy
            )
        # Allows adding a scalar (int/float) to the real part.
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real + other, self.imag, self.indeterminacy)
        return NotImplemented

    def __sub__(self, other):
        """Subtracts another number from this one."""
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy
            )
        elif isinstance(other, (int, float)):
             return NeutrosophicComplexNumber(self.real - other, self.imag, self.indeterminacy)
        return NotImplemented

    def __mul__(self, other):
        """
        Multiplies this number by another number (scalar, complex, or neutrosophic-complex).
        This is the most critical operation for complex dynamics.
        """
        if isinstance(other, NeutrosophicComplexNumber):
            # (a+bj)*(c+dj) = (ac-bd) + (ad+bc)j
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            
            # The indeterminacy grows based on both original indeterminacies and
            # the magnitude of the deterministic part, creating rich, non-linear behaviour.
            new_indeterminacy = self.indeterminacy + other.indeterminacy + (self.magnitude_sq() * other.indeterminacy)
            
            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        
        elif isinstance(other, complex):
            # Multiply by a standard Python complex number
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            # The indeterminacy is unaffected when multiplied by a purely deterministic complex number.
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
            
        elif isinstance(other, (int, float)):
            # Multiply by a scalar
            return NeutrosophicComplexNumber(
                self.real * other,
                self.imag * other,
                self.indeterminacy * other
            )
        return NotImplemented

    def __truediv__(self, divisor):
        """Divides this number by a scalar."""
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide a NeutrosophicComplexNumber by zero.")
            return NeutrosophicComplexNumber(
                self.real / divisor,
                self.imag / divisor,
                self.indeterminacy / divisor
            )
        raise TypeError("Only scalar division is supported for NeutrosophicComplexNumber.")

    # --- Reversed Mathematical Operations ---

    def __radd__(self, other):
        """Handles cases like `5 + my_number`."""
        return self.__add__(other)

    def __rsub__(self, other):
        """Handles cases like `5 - my_number`."""
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(other - self.real, -self.imag, -self.indeterminacy)
        return NotImplemented

    def __rmul__(self, other):
        """Handles cases like `5 * my_number`."""
        return self.__mul__(other)

    # --- Unary and Comparison Operations ---

    def __neg__(self):
        """Returns the negative of the number."""
        return NeutrosophicComplexNumber(-self.real, -self.imag, self.indeterminacy)

    def __eq__(self, other) -> bool:
        """Checks for equality between two numbers."""
        if not isinstance(other, NeutrosophicComplexNumber):
            return False
        return (self.real == other.real and
                self.imag == other.imag and
                self.indeterminacy == other.indeterminacy)

    # --- Helper Methods ---

    def magnitude_sq(self) -> float:
        """Returns the squared magnitude of the deterministic (complex) part."""
        return self.real**2 + self.imag**2

    def magnitude(self) -> float:
        """Returns the magnitude (modulus or absolute value) of the deterministic part."""
        return math.sqrt(self.magnitude_sq())

    def deterministic_part(self) -> complex:
        """Returns the deterministic part as a standard Python complex number."""
        return complex(self.real, self.imag)
        

@dataclass
class HyperrealNumber:
    """
    Represents a hyperreal number as a sequence of real numbers.
    Operations are performed element-wise on the sequences.
    
    Attributes:
        sequence (list[float]): The sequence representing the hyperreal.
    """
    sequence: list

    def __add__(self, other):
        if isinstance(other, HyperrealNumber):
            return HyperrealNumber([a + b for a, b in zip(self.sequence, other.sequence)])
        raise TypeError("Unsupported operand for +: HyperrealNumber and non-HyperrealNumber.")
    
    def __sub__(self, other):
        if isinstance(other, HyperrealNumber):
            return HyperrealNumber([a - b for a, b in zip(self.sequence, other.sequence)])
        raise TypeError("Unsupported operand for -: HyperrealNumber and non-HyperrealNumber.")

    # --- YENİ EKLENEN DÜZELTME ---
    # --- NEWLY ADDED FIX ---
    def __mul__(self, scalar):
        """Handles multiplication by a scalar (int or float)."""
        if isinstance(scalar, (int, float)):
            return HyperrealNumber([x * scalar for x in self.sequence])
        raise TypeError(f"Unsupported operand for *: HyperrealNumber and {type(scalar).__name__}")
    
    def __rmul__(self, scalar):
        """Handles the case where the scalar is on the left (e.g., float * HyperrealNumber)."""
        return self.__mul__(scalar)
    # --- DÜZELTME SONU ---
    # --- END OF FIX ---

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Scalar division by zero.")
            return HyperrealNumber([x / divisor for x in self.sequence])
        raise TypeError("Only scalar division is supported.")
    
    def __mod__(self, divisor):
        if isinstance(divisor, (int, float)):
            return [x % divisor for x in self.sequence]
        raise TypeError("Modulo operation only supported with a scalar divisor.")

    def __str__(self):
        return f"Hyperreal({self.sequence[:3]}...)"

@dataclass
class BicomplexNumber:
    """
    Represents a bicomplex number of the form z1 + j*z2, where z1 and z2
    are standard complex numbers, i^2 = -1, and j^2 = -1.
    
    Attributes:
        z1 (complex): The first complex component.
        z2 (complex): The second complex component (coefficient of j).
    """
    z1: complex
    z2: complex
    
    def __add__(self, other):
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        raise TypeError("Unsupported operand for +: BicomplexNumber and non-BicomplexNumber.")

    def __sub__(self, other):
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        raise TypeError("Unsupported operand for -: BicomplexNumber and non-BicomplexNumber.")

    def __mul__(self, other):
        if isinstance(other, BicomplexNumber):
            # (z1 + z2j)(w1 + w2j) = (z1w1 - z2w2) + (z1w2 + z2w1)j
            return BicomplexNumber(
                (self.z1 * other.z1) - (self.z2 * other.z2),
                (self.z1 * other.z2) + (self.z2 * other.z1)
            )
        raise TypeError("Unsupported operand for *: BicomplexNumber and non-BicomplexNumber.")
        
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return BicomplexNumber(self.z1 / scalar, self.z2 / scalar)
        raise TypeError("Only scalar division is supported.")

    def __str__(self):
        return f"Bicomplex({self.z1}, {self.z2})"

@dataclass
class NeutrosophicBicomplexNumber:
    """
    Represents a highly complex number with multiple components.
    NOTE: The multiplication implemented here is a simplified, element-wise
    operation for demonstrative purposes and is not mathematically rigorous.
    The true algebraic multiplication is exceedingly complex.
    """
    real: float
    imag: float
    neut_real: float
    neut_imag: float
    j_real: float
    j_imag: float
    j_neut_real: float
    j_neut_imag: float

    def __add__(self, other):
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(*(a + b for a, b in zip(self.__dict__.values(), other.__dict__.values())))
        raise TypeError("Unsupported operand for +.")
        
    def __sub__(self, other):
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(*(a - b for a, b in zip(self.__dict__.values(), other.__dict__.values())))
        raise TypeError("Unsupported operand for -.")
        
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return NeutrosophicBicomplexNumber(*(val / scalar for val in self.__dict__.values()))
        raise TypeError("Only scalar division supported.")

    def __str__(self):
        return f"NeutroBicomplex(r={self.real}, i={self.imag}, Ir={self.neut_real}, ...)"


# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def is_prime(n_input):
    """
    Checks if a given number (or its principal component) is prime.
    Extracts the relevant integer part from various number types for testing.
    """
    value_to_check = 0
    # Extract the integer part to check for primality based on type
    if isinstance(n_input, (int, float)):
        value_to_check = abs(int(n_input))
    elif isinstance(n_input, Fraction):
        value_to_check = abs(int(n_input))
    elif isinstance(n_input, complex):
        value_to_check = abs(int(n_input.real))
    elif isinstance(n_input, np.quaternion):
        value_to_check = abs(int(n_input.w))
    elif isinstance(n_input, NeutrosophicNumber):
        value_to_check = abs(int(n_input.a))
    elif isinstance(n_input, NeutrosophicComplexNumber):
        value_to_check = abs(int(n_input.real))
    elif isinstance(n_input, HyperrealNumber):
        value_to_check = abs(int(n_input.sequence[0])) if n_input.sequence else 0
    elif isinstance(n_input, BicomplexNumber):
        value_to_check = abs(int(n_input.z1.real))
    elif isinstance(n_input, NeutrosophicBicomplexNumber):
        value_to_check = abs(int(n_input.real))
    else:
        try:
            value_to_check = abs(int(n_input))
        except (ValueError, TypeError):
            return False

    # Standard primality test algorithm
    if value_to_check < 2:
        return False
    if value_to_check == 2:
        return True
    if value_to_check % 2 == 0:
        return False
    # Check only odd divisors up to the square root
    for i in range(3, int(math.sqrt(value_to_check)) + 1, 2):
        if value_to_check % i == 0:
            return False
    return True

def _is_divisible(value, divisor, kececi_type):
    """
    Helper to check divisibility for different number types.
    Returns True if a number is "perfectly divisible" by an integer divisor.
    """
    try:
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            return value % divisor == 0
        elif kececi_type == TYPE_FLOAT:
            return math.isclose(value % divisor, 0)
        elif kececi_type == TYPE_RATIONAL:
            return (value / divisor).denominator == 1
        elif kececi_type == TYPE_COMPLEX:
            return math.isclose(value.real % divisor, 0) and math.isclose(value.imag % divisor, 0)
        elif kececi_type == TYPE_QUATERNION:
            return all(math.isclose(c % divisor, 0) for c in [value.w, value.x, value.y, value.z])
        elif kececi_type == TYPE_NEUTROSOPHIC:
            return math.isclose(value.a % divisor, 0) and math.isclose(value.b % divisor, 0)
        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            return all(math.isclose(c % divisor, 0) for c in [value.real, value.imag, value.indeterminacy])
        elif kececi_type == TYPE_HYPERREAL:
            return all(math.isclose(x % divisor, 0) for x in value.sequence)
        elif kececi_type == TYPE_BICOMPLEX:
            return (_is_divisible(value.z1, divisor, TYPE_COMPLEX) and
                    _is_divisible(value.z2, divisor, TYPE_COMPLEX))
        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            return all(math.isclose(c % divisor, 0) for c in value.__dict__.values())
    except (TypeError, ValueError):
        return False
    return False

# ==============================================================================
# --- CORE GENERATOR ---
# ==============================================================================

def unified_generator(kececi_type, start_input_raw, add_input_base_scalar, iterations):
    """
    Herhangi bir desteklenen türde Keçeci Sayı dizileri üreten çekirdek motor.
    Bu sürüm, tüm tipler için sağlam tür dönüştürme ve özel format ayrıştırma içerir.
    """
    current_value = None
    add_value_typed = None
    ask_unit = None
    use_integer_division = False

    try:
        # --- Adım 1: Keçeci Türüne Göre Başlatma ---
        a_float = float(add_input_base_scalar)

        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            s_int = int(float(start_input_raw))
            current_value = s_int
            add_value_typed = int(a_float)
            ask_unit = 1
            use_integer_division = True
            
        elif kececi_type == TYPE_FLOAT:
            current_value = float(start_input_raw)
            add_value_typed = a_float
            ask_unit = 1.0
            
        elif kececi_type == TYPE_RATIONAL:
            current_value = Fraction(start_input_raw)
            add_value_typed = Fraction(add_input_base_scalar)
            ask_unit = Fraction(1)
            
        elif kececi_type == TYPE_COMPLEX:
            current_value = _parse_complex(start_input_raw)
            add_value_typed = complex(a_float, a_float)
            ask_unit = 1 + 1j

        elif kececi_type == TYPE_NEUTROSOPHIC:
            a, b = _parse_neutrosophic(start_input_raw)
            current_value = NeutrosophicNumber(a, b)
            add_value_typed = NeutrosophicNumber(a_float, 0)
            ask_unit = NeutrosophicNumber(1, 1)

        # --- YENİ EKLENEN/DÜZELTİLEN BLOKLAR ---

        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX: # HATA DÜZELTİLDİ
            s_complex = _parse_complex(start_input_raw)
            # Başlangıç indeterminacy değerini 0 olarak varsayalım
            current_value = NeutrosophicComplexNumber(s_complex.real, s_complex.imag, 0.0)
            # Artış, deterministik reel kısma etki eder
            add_value_typed = NeutrosophicComplexNumber(a_float, 0.0, 0.0)
            ask_unit = NeutrosophicComplexNumber(1, 1, 1)

        elif kececi_type == TYPE_HYPERREAL: # HATA DÜZELTİLDİ
            a, b = _parse_hyperreal(start_input_raw)
            # 'a' reel kısmı, 'b' ise sonsuz küçükleri ölçekler
            sequence_list = [a + b / n for n in range(1, 11)]
            current_value = HyperrealNumber(sequence_list)
            # Artış, sadece standart (ilk) reel kısma etki eder
            add_sequence = [a_float] + [0.0] * 9
            add_value_typed = HyperrealNumber(add_sequence)
            ask_unit = HyperrealNumber([1.0] * 10)

        elif kececi_type == TYPE_BICOMPLEX: # Mantık aynı, sadece ayrıştırıcıyı kullanıyor
            s_complex = _parse_complex(start_input_raw)
            a_complex = complex(a_float)
            current_value = BicomplexNumber(s_complex, s_complex / 2)
            add_value_typed = BicomplexNumber(a_complex, a_complex / 2)
            ask_unit = BicomplexNumber(complex(1, 1), complex(0.5, 0.5))

        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX: # HATA DÜZELTİLDİ
            s_complex = _parse_complex(start_input_raw)
            # Başlangıç değeri olarak kompleks sayıyı kullanıp diğer 6 bileşeni 0 yapalım
            current_value = NeutrosophicBicomplexNumber(s_complex.real, s_complex.imag, 0, 0, 0, 0, 0, 0)
            # Artış, sadece ana reel kısma etki eder
            add_value_typed = NeutrosophicBicomplexNumber(a_float, 0, 0, 0, 0, 0, 0, 0)
            ask_unit = NeutrosophicBicomplexNumber(*([1.0] * 8))

        # --- DİĞER TİPLER ---

        elif kececi_type == TYPE_QUATERNION:
            s_float = float(start_input_raw)
            current_value = np.quaternion(s_float, s_float, s_float, s_float)
            add_value_typed = np.quaternion(a_float, a_float, a_float, a_float)
            ask_unit = np.quaternion(1, 1, 1, 1)

        else:
            raise ValueError(f"Geçersiz veya desteklenmeyen Keçeci Sayı Tipi: {kececi_type}")

    except (ValueError, TypeError) as e:
        print(f"HATA: Tip {kececi_type} için '{start_input_raw}' girdisiyle başlatma başarısız: {e}")
        return []

    # --- Adım 2: İterasyon Döngüsü ---
    sequence = [current_value]
    last_divisor_used = None
    ask_counter = 0
    
    for _ in range(iterations):
        added_value = current_value + add_value_typed
        sequence.append(added_value)
        
        result_value = added_value
        divided_successfully = False

        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3
        
        for divisor in [primary_divisor, alternative_divisor]:
            if _is_divisible(added_value, divisor, kececi_type):
                result_value = added_value // divisor if use_integer_division else added_value / divisor
                last_divisor_used = divisor
                divided_successfully = True
                break
        
        if not divided_successfully and is_prime(added_value):
            modified_value = (added_value + ask_unit) if ask_counter == 0 else (added_value - ask_unit)
            ask_counter = 1 - ask_counter
            sequence.append(modified_value)
            
            result_value = modified_value 
            
            for divisor in [primary_divisor, alternative_divisor]:
                if _is_divisible(modified_value, divisor, kececi_type):
                    result_value = modified_value // divisor if use_integer_division else modified_value / divisor
                    last_divisor_used = divisor
                    break
        
        sequence.append(result_value)
        current_value = result_value
        
    return sequence

def _parse_complex(s: str) -> complex:
    """
    Bir string'i kompleks sayıya çevirir. 
    Eğer 'j' içermiyorsa, "3" girdisini 3+3j olarak yorumlar.
    """
    s_clean = s.strip().lower()
    try:
        # Doğrudan kompleks çevirmeyi dene (ör: "3+4j")
        c = complex(s_clean)
        # Eğer girdi "3" gibi sadece reel bir sayıysa ve 'j' içermiyorsa,
        # onu s_complex.real + s_complex.real*j yap.
        if c.imag == 0 and 'j' not in s_clean:
            return complex(c.real, c.real)
        return c
    except ValueError as e:
        raise ValueError(f"Geçersiz kompleks sayı formatı: '{s}'") from e

def _parse_neutrosophic(s: str) -> (float, float):
    """
    Parses a neutrosophic number string of the form 'a+bI' into a tuple (a, b).
    Handles cases like '5+2I', '3-I', '7', '4I', '-I'.
    """
    s = s.strip().replace(" ", "").upper()
    if not s:
        return 0.0, 0.0

    # Eğer 'I' yoksa, bu standart bir sayıdır (b=0)
    if 'I' not in s:
        try:
            return float(s), 0.0
        except ValueError:
            raise ValueError(f"Invalid number format for non-neutrosophic part: {s}")

    # 'I' varsa, a ve b kısımlarını ayır
    # 'b' kısmını bul
    i_pos = s.find('I')
    a_part_str = s[:i_pos]
    
    # Sadece 'I' veya '-I' gibi durumlar
    if not a_part_str or a_part_str == '+':
        b = 1.0
    elif a_part_str == '-':
        b = -1.0
    else:
        # 'a' ve 'b' kısımlarını ayırmak için sondaki işareti bul
        last_plus = a_part_str.rfind('+')
        last_minus = a_part_str.rfind('-')
        
        # Eğer başta eksi işareti varsa onu ayraç olarak sayma
        if last_minus == 0 and last_plus == -1:
             split_pos = -1
        else:
             split_pos = max(last_plus, last_minus)

        if split_pos == -1: # Örneğin '3I' durumu
            a = 0.0
            b = float(a_part_str)
        else: # Örneğin '5+3I' veya '-2-4I' durumu
            a = float(a_part_str[:split_pos])
            b_str = a_part_str[split_pos:]
            if b_str == '+':
                b = 1.0
            elif b_str == '-':
                b = -1.0
            else:
                b = float(b_str)
    
    return a, b

def _parse_hyperreal(s: str) -> (float, float):
    """
    'a+be' formatındaki bir hiperreel string'i (a, b) demetine ayrıştırır.
    '5+3e', '-2-e', '10', '4e', '-e' gibi formatları işler.
    """
    s = s.strip().replace(" ", "").lower()
    if not s:
        return 0.0, 0.0

    # Eğer 'e' yoksa, bu standart bir sayıdır (b=0)
    if 'e' not in s:
        return float(s), 0.0

    # 'a+be' formatını regex ile ayrıştır
    # Örnekler: 5+3e, -3.1-4.5e, +2e, e, -e
    match = re.match(r"^(?P<a>[+-]?\d+\.?\d*)?(?P<b>[+-]?\d*\.?\d*)e$", s)
    if not match:
        # Sadece +e veya -e durumları için
        match = re.match(r"^(?P<a>[+-]?\d+\.?\d*)?(?P<b>[+-])e$", s)

    if not match:
        raise ValueError(f"Geçersiz hiperreel format: {s}")

    parts = match.groupdict()
    a_part = parts.get('a')
    b_part = parts.get('b')

    a = float(a_part) if a_part else 0.0
    
    if b_part is None or b_part == '' or b_part == '+':
        b = 1.0
    elif b_part == '-':
        b = -1.0
    else:
        b = float(b_part)
        
    return a, b


def get_random_type(num_iterations, fixed_start_raw="0", fixed_add_base_scalar=9.0):
    """
    Generates Keçeci Numbers for a randomly selected type using fixed parameters.
    (Updated with the full list of 11 types and compatible with the new get_with_params signature)
    """
    # Rastgele sayı üretme aralığı 11 tipe göre güncellendi.
    random_type_choice = random.randint(1, 11)
    
    # Kullanıcının sağladığı tam liste eklendi.
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutro-Complex", "Hyperreal", 
        "Bicomplex", "Neutro-Bicomplex"
    ]
    
    # Listenin indeksi 0'dan başladığı için random_type_choice-1 kullanılır.
    print(f"\nRandomly selected Keçeci Number Type: {random_type_choice} ({type_names_list[random_type_choice-1]})")
    
    # get_with_params fonksiyonu sadeleştirilmiş haliyle çağrılıyor.
    # Beklenmeyen 'random_range_factor' veya 'fixed_params' argümanları yok.
    generated_sequence = get_with_params(
        kececi_type_choice=random_type_choice, 
        iterations=num_iterations,
        start_value_raw=fixed_start_raw,
        add_value_base_scalar=fixed_add_base_scalar
    )
    
    return generated_sequence

def print_detailed_report(sequence, params):
    """
    Generates and prints a detailed report of the Keçeci sequence results.
    
    Args:
        sequence (list): The generated Keçeci sequence.
        params (dict): A dictionary containing the generation parameters.
    """
    if not sequence:
        print("\n--- REPORT ---")
        print("Sequence could not be generated. No report to show.")
        return

    print("\n\n" + "="*50)
    print("--- DETAILED SEQUENCE REPORT ---")
    print("="*50)

    # Parametreleri yazdır
    print("\n[Parameters Used]")
    print(f"  - Keçeci Type:   {params['type_name']} ({params['type_choice']})")
    print(f"  - Start Value:   '{params['start_val']}'")
    print(f"  - Increment:     {params['add_val']}")
    print(f"  - Keçeci Steps:  {params['steps']}")

    # Dizi Özeti
    print("\n[Sequence Summary]")
    print(f"  - Total Numbers Generated: {len(sequence)}")
    
    # Keçeci Asal Sayısı (KPN) sonucunu bul ve yazdır
    kpn = find_kececi_prime_number(sequence)
    if kpn is not None:
        print(f"  - Keçeci Prime Number (KPN): {kpn}")
    else:
        print("  - Keçeci Prime Number (KPN): Not found in this sequence.")

    # Dizi Önizlemesi
    print("\n[Sequence Preview]")
    # Dizinin tamamını yazdırmadan önce bir önizleme sunalım
    preview_count = min(len(sequence), 5)
    print(f"  --- First {preview_count} Numbers ---")
    for i in range(preview_count):
        print(f"    {i}: {sequence[i]}")

    if len(sequence) > preview_count:
        print(f"\n  --- Last {preview_count} Numbers ---")
        for i in range(len(sequence) - preview_count, len(sequence)):
            print(f"    {i}: {sequence[i]}")
            
    print("\n" + "="*50)

    # Kullanıcıya tüm diziyi görmek isteyip istemediğini sor
    while True:
        show_all = input("Do you want to print the full sequence? (y/n): ").lower().strip()
        if show_all in ['y', 'n']:
            break
    
    if show_all == 'y':
        print("\n--- FULL SEQUENCE ---")
        for i, num in enumerate(sequence):
            print(f"{i}: {num}")
        print("="*50)

# ==============================================================================
# --- HIGH-LEVEL CONTROL FUNCTIONS ---
# ==============================================================================

def get_with_params(kececi_type_choice, iterations, start_value_raw="0", add_value_base_scalar=9.0):
    """
    Generates Keçeci Numbers with specified parameters.
    """
    print(f"\n--- Generating Sequence: Type {kececi_type_choice}, Steps {iterations} ---")
    print(f"Start: '{start_value_raw}', Increment: {add_value_base_scalar}")

    generated_sequence = unified_generator(
        kececi_type_choice, 
        start_value_raw, 
        add_value_base_scalar, 
        iterations
    )
    
    if generated_sequence:
        print(f"Generated {len(generated_sequence)} numbers. Preview: {generated_sequence[:3]}...")
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            print(f"Keçeci Prime Number for this sequence: {kpn}")
        else:
            print("No repeating Keçeci Prime Number found.")
    else:
        print("Sequence generation failed.")
        
    return generated_sequence

def get_interactive():
    """
    Interactively gets parameters from the user and generates Keçeci Numbers.
    This version only gets data and returns the sequence without plotting.
    """
    print("\n--- Keçeci Number Interactive Generator ---")
    print("  1: Positive Real    2: Negative Real     3: Complex")
    print("  4: Float            5: Rational          6: Quaternion")
    print("  7: Neutrosophic     8: Neutro-Complex   9: Hyperreal")
    print(" 10: Bicomplex        11: Neutro-Bicomplex")
    
    while True:
        try:
            type_choice = int(input(f"Select Keçeci Number Type (1-11): "))
            if 1 <= type_choice <= 11: break
            else: print("Invalid type. Please enter a number between 1 and 11.")
        except ValueError: print("Invalid input. Please enter a number.")
        
    # Her tip için özel başlangıç değeri istemleri (prompts)
    prompts = {
        1:  "Enter positive integer start (e.g., '10'): ",
        2:  "Enter negative integer start (e.g., '-5'): ",
        3:  "Enter complex start (e.g., '3+4j' or '3' for 3+3j): ",
        4:  "Enter float start (e.g., '3.14' or '-0.5'): ",
        5:  "Enter rational start (e.g., '7/2' or '5' for 5/1): ",
        6:  "Enter scalar for quaternion base (e.g., '2.5' for 2.5+2.5i+2.5j+2.5k): ",
        7:  "Enter neutrosophic start (e.g., '5+2I' or '7'): ",
        8:  "Enter complex base for neutro-complex (e.g., '1-2j'): ",
        9:  "Enter hyperreal start (e.g., '5+3e' or '10'): ",
        10: "Enter complex base for bicomplex (e.g., '2+1j'): ",
        11: "Enter complex base for neutro-bicomplex (e.g., '1+2j'): "
    }
    
    # Seçilen tipe göre doğru istemi al, yoksa genel bir istem kullan
    start_prompt = prompts.get(type_choice, "Enter starting value: ")

    start_input_val_raw = input(start_prompt)
    add_base_scalar_val = float(input("Enter base scalar increment (e.g., 9.0 for positive, -3.0 for negative): "))
    num_kececi_steps = int(input("Enter number of Keçeci steps (e.g., 15): "))
    
    # Diziyi oluştur ve çizdir
    sequence = get_with_params(type_choice, num_kececi_steps, start_input_val_raw, add_base_scalar_val)
    if sequence:
        # sequence adı altında bir isim çakışması olmaması için başlığı değiştiriyoruz.
        plot_title = f"Keçeci Type {type_choice} Sequence"
        #plot_numbers(sequence, plot_title)
        #plt.show() # Grafiği göstermek için
        
    # Fonksiyonun, ana kodun kullanabilmesi için tüm önemli bilgileri döndürmesini sağlıyoruz.
    return sequence, type_choice, start_input_val_raw, add_base_scalar_val, num_kececi_steps

# ==============================================================================
# --- ANALYSIS AND PLOTTING ---
# ==============================================================================

def find_kececi_prime_number(kececi_numbers_list):
    """
    Finds the Keçeci Prime Number from a generated sequence.
    
    The Keçeci Prime is the integer representation of the most frequent number
    in the sequence whose principal component is itself prime. Ties in frequency
    are broken by choosing the larger prime number.
    """
    if not kececi_numbers_list:
        return None

    # Extract integer representations of numbers that are prime
    integer_prime_reps = []
    for num in kececi_numbers_list:
        if is_prime(num):
            # This logic is duplicated from is_prime to get the value itself
            value = 0
            if isinstance(num, (int, float, Fraction)): value = abs(int(num))
            elif isinstance(num, complex): value = abs(int(num.real))
            elif isinstance(num, np.quaternion): value = abs(int(num.w))
            elif isinstance(num, NeutrosophicNumber): value = abs(int(num.a))
            elif isinstance(num, NeutrosophicComplexNumber): value = abs(int(num.real))
            elif isinstance(num, HyperrealNumber): value = abs(int(num.sequence[0])) if num.sequence else 0
            elif isinstance(num, BicomplexNumber): value = abs(int(num.z1.real))
            elif isinstance(num, NeutrosophicBicomplexNumber): value = abs(int(num.real))
            integer_prime_reps.append(value)

    if not integer_prime_reps:
        return None

    # Count frequencies of these prime integers
    counts = collections.Counter(integer_prime_reps)
    
    # Find primes that repeat
    repeating_primes = [(freq, prime) for prime, freq in counts.items() if freq > 1]

    if not repeating_primes:
        return None
    
    # Find the one with the highest frequency, using the prime value as a tie-breaker
    _, best_prime = max(repeating_primes)
    return best_prime

def plot_numbers(sequence, title="Keçeci Number Sequence Analysis"):
    """
    Plots the generated Keçeci Number sequence with appropriate visualizations
    for each number type.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not sequence:
        print("Sequence is empty, nothing to plot.")
        return

    fig = plt.figure(figsize=(15, 8))
    plt.suptitle(title, fontsize=16, y=0.98)
    first_elem = sequence[0]
    
    # --- Plotting logic per type ---
    
    # CORRECTED: Check for the actual Python types.
    # This correctly handles types 1, 2, and 4 (Positive/Negative Real, Float).
    if isinstance(first_elem, (int, float)):
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([float(x) for x in sequence], 'o-')
        ax.set_title("Value over Iterations")
        ax.set_xlabel("Index"); ax.set_ylabel("Value")

    elif isinstance(first_elem, Fraction):
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([float(x) for x in sequence], 'o-')
        ax.set_title("Value over Iterations (as float)")
        ax.set_xlabel("Index"); ax.set_ylabel("Value")

    elif isinstance(first_elem, complex):
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        real_parts = [c.real for c in sequence]
        imag_parts = [c.imag for c in sequence]
        
        ax1.plot(real_parts, 'o-', label='Real Part')
        ax1.set_title("Real Part"); ax1.legend()
        
        ax2.plot(imag_parts, 'o-', color='red', label='Imaginary Part')
        ax2.set_title("Imaginary Part"); ax2.legend()
        
        ax3.plot(real_parts, imag_parts, '.-', label='Trajectory')
        ax3.scatter(real_parts[0], imag_parts[0], c='g', s=100, label='Start', zorder=5)
        ax3.scatter(real_parts[-1], imag_parts[-1], c='r', s=100, label='End', zorder=5)
        ax3.set_title("Trajectory in Complex Plane"); ax3.set_xlabel("Real"); ax3.set_ylabel("Imaginary"); ax3.legend(); ax3.axis('equal')

    elif isinstance(first_elem, np.quaternion):
        gs = GridSpec(2, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        ax1.plot([q.w for q in sequence], 'o-', label='w (scalar)')
        ax1.plot([q.x for q in sequence], 's--', label='x')
        ax1.plot([q.y for q in sequence], '^--', label='y')
        ax1.plot([q.z for q in sequence], 'd--', label='z')
        ax1.set_title("Quaternion Components"); ax1.legend()

        magnitudes = [np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2) for q in sequence]
        ax2.plot(magnitudes, 'o-', color='purple', label='Magnitude')
        ax2.set_title("Quaternion Magnitude"); ax2.legend()

    elif isinstance(first_elem, BicomplexNumber):
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])
        
        z1r = [x.z1.real for x in sequence]; z1i = [x.z1.imag for x in sequence]
        z2r = [x.z2.real for x in sequence]; z2i = [x.z2.imag for x in sequence]
        
        ax1.plot(z1r, label='z1.real'); ax1.plot(z1i, label='z1.imag')
        ax1.set_title("Component z1"); ax1.legend()
        
        ax2.plot(z2r, label='z2.real'); ax2.plot(z2i, label='z2.imag')
        ax2.set_title("Component z2"); ax2.legend()
        
        ax3.plot(z1r, z1i, '.-'); ax3.set_title("z1 in Complex Plane")
        ax4.plot(z2r, z2i, '.-'); ax4.set_title("z2 in Complex Plane")
        
    elif isinstance(first_elem, NeutrosophicNumber):
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
        
        a = [x.a for x in sequence]; b = [x.b for x in sequence]
        ax1.plot(a, label='Determinate (a)'); ax1.plot(b, label='Indeterminate (b)')
        ax1.set_title("Components"); ax1.legend()
        
        sc = ax2.scatter(a, b, c=range(len(a)), cmap='viridis')
        ax2.set_title("Determinate vs. Indeterminate"); fig.colorbar(sc, ax=ax2)

    elif isinstance(first_elem, NeutrosophicComplexNumber):
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
        
        r = [x.real for x in sequence]; i = [x.imag for x in sequence]; ind = [x.indeterminacy for x in sequence]
        ax1.plot(r, label='Real'); ax1.plot(i, label='Imag'); ax1.plot(ind, label='Indeterminacy', linestyle=':')
        ax1.set_title("Components"); ax1.legend()
        
        sc = ax2.scatter(r, i, c=ind, cmap='magma')
        ax2.set_title("Complex Plane (colored by Indeterminacy)"); fig.colorbar(sc, ax=ax2, label='Indeterminacy')
    
    else: # Fallback for Hyperreal, Neutro-Bicomplex, and others
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f"Plotting for type '{type(first_elem).__name__}'\nis not specifically implemented.\nShowing string representation of first 3 elements:\n\n1. {sequence[0]}\n2. {sequence[1]}\n3. {sequence[2]}",
                ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightyellow'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("  Keçeci Numbers Module - Demonstration")
    print("="*60)
    print(f"This script demonstrates the generation of various Keçeci Number types.")
    
    # --- Example 1: Interactive Mode ---
    # To run interactive mode, uncomment the following line:
    # get_interactive()

    # --- Example 2: Programmatic Generation and Plotting ---
    # We will generate a sequence for each type to test the system.
    print("\nRunning programmatic tests for all 11 number types...")
    
    # Test parameters
    STEPS = 15
    START_VAL = "2.5"
    ADD_VAL = 3.0

    all_types = {
        "Positive Real": TYPE_POSITIVE_REAL,
        "Negative Real": TYPE_NEGATIVE_REAL,
        "Complex": TYPE_COMPLEX,
        "Float": TYPE_FLOAT,
        "Rational": TYPE_RATIONAL,
        "Quaternion": TYPE_QUATERNION,
        "Neutrosophic": TYPE_NEUTROSOPHIC,
        "Neutrosophic Complex": TYPE_NEUTROSOPHIC_COMPLEX,
        "Hyperreal": TYPE_HYPERREAL,
        "Bicomplex": TYPE_BICOMPLEX,
        "Neutrosophic Bicomplex": TYPE_NEUTROSOPHIC_BICOMPLEX
    }

    # Generate and plot for a few selected types
    types_to_plot = [
        "Complex", 
        "Quaternion", 
        "Bicomplex", 
        "Neutrosophic Complex"
    ]
    
    for name, type_id in all_types.items():
        # Adjust start/add values for specific types if needed
        start = "-5" if type_id == TYPE_NEGATIVE_REAL else "2+3j" if type_id in [TYPE_COMPLEX, TYPE_BICOMPLEX] else START_VAL
        
        seq = get_with_params(type_id, STEPS, start, ADD_VAL)
        
        if name in types_to_plot and seq:
            plot_numbers(seq, title=f"Demonstration: {name} Keçeci Numbers")

    print("\n\nDemonstration finished. Plots for selected types are shown.")
    plt.show()
