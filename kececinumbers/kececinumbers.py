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
    The core engine for generating Keçeci Number sequences of any supported type.
    This version includes robust type conversion to prevent initialization errors.
    """
    # --- Step 1: Initialization based on Keçeci Type ---
    current_value = None
    add_value_typed = None
    ask_unit = None
    use_integer_division = False

    try:
        # Convert the ADD value once, as it's always a scalar float.
        a_float = float(add_input_base_scalar)

        # Handle START value conversion properly within each type-specific block.
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            # Correctly handle float strings like "2.5" by converting to float first.
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
            # The Fraction constructor correctly handles strings like "2.5".
            current_value = Fraction(str(start_input_raw))
            add_value_typed = Fraction(str(add_input_base_scalar))
            ask_unit = Fraction(1, 1)
            
        elif kececi_type == TYPE_COMPLEX:
            s_complex = complex(start_input_raw)
            # If input was a plain number (e.g., "2.5"), interpret it as s+sj.
            if s_complex.imag == 0 and 'j' not in str(start_input_raw).lower():
                s_complex = complex(s_complex.real, s_complex.real)
            current_value = s_complex
            add_value_typed = complex(a_float, a_float)
            ask_unit = 1 + 1j

        elif kececi_type == TYPE_QUATERNION:
            # Explicitly convert the input string to a float before use.
            s_float = float(start_input_raw)
            current_value = np.quaternion(s_float, s_float, s_float, s_float)
            add_value_typed = np.quaternion(a_float, a_float, a_float, a_float)
            ask_unit = np.quaternion(1, 1, 1, 1)
            
        elif kececi_type == TYPE_NEUTROSOPHIC:
            s_float = float(start_input_raw)
            current_value = NeutrosophicNumber(s_float, s_float / 2)
            add_value_typed = NeutrosophicNumber(a_float, a_float / 2)
            ask_unit = NeutrosophicNumber(1, 1)
            
        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            s_float = float(start_input_raw)
            current_value = NeutrosophicComplexNumber(s_float, s_float / 2, s_float / 3)
            add_value_typed = NeutrosophicComplexNumber(a_float, a_float / 2, a_float / 3)
            ask_unit = NeutrosophicComplexNumber(1, 1, 1)
            
        elif kececi_type == TYPE_HYPERREAL:
            s_float = float(start_input_raw)
            current_value = HyperrealNumber([s_float / n for n in range(1, 11)])
            add_value_typed = HyperrealNumber([a_float / n for n in range(1, 11)])
            ask_unit = HyperrealNumber([1.0] * 10)
            
        elif kececi_type == TYPE_BICOMPLEX:
            s_complex = complex(start_input_raw)
            a_complex = complex(a_float)
            current_value = BicomplexNumber(s_complex, s_complex / 2)
            add_value_typed = BicomplexNumber(a_complex, a_complex / 2)
            ask_unit = BicomplexNumber(complex(1, 1), complex(0.5, 0.5))

        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            s_float = float(start_input_raw)
            parts = [s_float / (n + 1) for n in range(8)]
            add_parts = [a_float / (n + 1) for n in range(8)]
            ask_parts = [1.0 / (n + 1) for n in range(8)]
            current_value = NeutrosophicBicomplexNumber(*parts)
            add_value_typed = NeutrosophicBicomplexNumber(*add_parts)
            ask_unit = NeutrosophicBicomplexNumber(*ask_parts)
            
        else:
            raise ValueError(f"Invalid Keçeci Number Type: {kececi_type}")

    except (ValueError, TypeError) as e:
        print(f"Error initializing generator for type {kececi_type} with input '{start_input_raw}': {e}")
        return []

    # --- Step 2: Iteration Loop (This part remains unchanged) ---
    sequence = [current_value]
    last_divisor_used = None
    ask_counter = 0
    
    for _ in range(iterations):
        # Rule 1: Add the increment value
        added_value = current_value + add_value_typed
        sequence.append(added_value)
        
        result_value = added_value
        divided_successfully = False

        # Rule 2: Attempt Division
        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3
        
        for divisor in [primary_divisor, alternative_divisor]:
            if _is_divisible(added_value, divisor, kececi_type):
                result_value = added_value // divisor if use_integer_division else added_value / divisor
                last_divisor_used = divisor
                divided_successfully = True
                break
        
        # Rule 3: Apply ASK Rule if division failed and the number is prime
        if not divided_successfully and is_prime(added_value):
            # Augment or Shrink the value
            modified_value = (added_value + ask_unit) if ask_counter == 0 else (added_value - ask_unit)
            ask_counter = 1 - ask_counter  # Flip between 0 and 1
            sequence.append(modified_value)
            
            result_value = modified_value # Default to modified value if re-division fails
            
            # Re-attempt division on the modified value
            for divisor in [primary_divisor, alternative_divisor]:
                if _is_divisible(modified_value, divisor, kececi_type):
                    result_value = modified_value // divisor if use_integer_division else modified_value / divisor
                    last_divisor_used = divisor
                    break
        
        sequence.append(result_value)
        current_value = result_value
        
    return sequence

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
        
    start_prompt = "Enter starting value: "
    if type_choice == TYPE_COMPLEX: start_prompt = "Enter complex start (e.g., '3+4j' or '3' for 3+3j): "
    elif type_choice == TYPE_RATIONAL: start_prompt = "Enter rational start (e.g., '7/2' or '5'): "
    elif type_choice == TYPE_BICOMPLEX: start_prompt = "Enter bicomplex start (complex, e.g., '2+1j'): "

    start_input_val_raw = input(start_prompt)
    add_base_scalar_val = float(input("Enter base scalar increment (e.g., 9.0): "))
    num_kececi_steps = int(input("Enter number of Keçeci steps (e.g., 15): "))
    
    sequence = get_with_params(type_choice, num_kececi_steps, start_input_val_raw, add_base_scalar_val)
    plot_numbers(sequence, f"Keçeci Type {type_choice} Sequence")
    plt.show()

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
