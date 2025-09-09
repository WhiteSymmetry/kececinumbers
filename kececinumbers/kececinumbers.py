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
---

Keçeci Conjecture: Keçeci Varsayımı, Keçeci-Vermutung, Conjecture de Keçeci, Гипотеза Кечеджи, 凯杰西猜想, ケジェジ予想, Keçeci Huds, Keçeci Hudsiye, Keçeci Hudsia, كَچَه جِي ,حدس کچه جی, کچہ جی حدسیہ

Keçeci Varsayımı (Keçeci Conjecture) - Önerilen

Her Keçeci Sayı türü için, `unified_generator` fonksiyonu tarafından oluşturulan dizilerin, sonlu adımdan sonra periyodik bir yapıya veya tekrar eden bir asal temsiline (Keçeci Asal Sayısı, KPN) yakınsadığı sanılmaktadır. Bu davranış, Collatz Varsayımı'nın çoklu cebirsel sistemlere genişletilmiş bir hali olarak değerlendirilebilir.

Henüz kanıtlanmamıştır ve bu modül bu varsayımı test etmek için bir çerçeve sunar.
"""

# --- Standard Library Imports ---
import collections
from dataclasses import dataclass
from fractions import Fraction
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import random
import re
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
import sympy
from typing import Any, Dict, List, Optional, Tuple


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
    """Represents a neutrosophic number of the form a + bI where I^2 = I."""
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
            return NeutrosophicNumber(
                self.a * other.a,
                self.a * other.b + self.b * other.a + self.b * other.b,
            )
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.a * other, self.b * other)
        return NotImplemented

    def __truediv__(self, divisor: float) -> "NeutrosophicNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicNumber(self.a / divisor, self.b / divisor)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        return f"{self.a} + {self.b}I"

@dataclass
class NeutrosophicComplexNumber:
    """Represents a number with a complex part and an indeterminacy level."""
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
                self.real + other.real, self.imag + other.imag, self.indeterminacy + other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real + other, self.imag, self.indeterminacy)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real, self.imag - other.imag, self.indeterminacy - other.indeterminacy
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
                self.real / divisor, self.imag / divisor, self.indeterminacy / divisor
            )
        raise TypeError("Only scalar division is supported.")

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(other - self.real, -self.imag, -self.indeterminacy)
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def magnitude_sq(self) -> float:
        return self.real**2 + self.imag**2

@dataclass
class HyperrealNumber:
    """Represents a hyperreal number as a sequence of real numbers."""
    sequence: list

    def __add__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            return HyperrealNumber([a + b for a, b in zip(self.sequence, other.sequence)])
        return NotImplemented

    def __sub__(self, other: Any) -> "HyperrealNumber":
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
        return f"Hyperreal({self.sequence[:30]}...)"

@dataclass
class BicomplexNumber:
    """Represents a bicomplex number z1 + j*z2, where i^2 = j^2 = -1."""
    z1: complex
    z2: complex

    def __add__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        return NotImplemented

    def __sub__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        return NotImplemented

    def __mul__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(
                (self.z1 * other.z1) - (self.z2 * other.z2),
                (self.z1 * other.z2) + (self.z2 * other.z1),
            )
        return NotImplemented

    def __truediv__(self, scalar: float) -> "BicomplexNumber":
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return BicomplexNumber(self.z1 / scalar, self.z2 / scalar)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        return f"Bicomplex({self.z1}, {self.z2})"

@dataclass
class NeutrosophicBicomplexNumber:
    """Represents a simplified neutrosophic-bicomplex number."""
    real: float
    imag: float
    neut_real: float
    neut_imag: float
    j_real: float
    j_imag: float
    j_neut_real: float
    j_neut_imag: float

    def __add__(self, other: Any) -> "NeutrosophicBicomplexNumber":
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(*(a + b for a, b in zip(self.__dict__.values(), other.__dict__.values())))
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicBicomplexNumber":
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(*(a - b for a, b in zip(self.__dict__.values(), other.__dict__.values())))
        return NotImplemented

    def __truediv__(self, scalar: float) -> "NeutrosophicBicomplexNumber":
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicBicomplexNumber(*(val / scalar for val in self.__dict__.values()))
        raise TypeError("Only scalar division supported.")

    def __str__(self) -> str:
        return f"NeutroBicomplex(r={self.real}, i={self.imag}, Ir={self.neut_real}, ...)"

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def _get_integer_representation(n_input: Any) -> Optional[int]:
    """Extracts the primary integer component from any supported number type."""
    try:
        if isinstance(n_input, (int, float, Fraction)):
            return abs(int(n_input))
        if isinstance(n_input, complex):
            return abs(int(n_input.real))
        if isinstance(n_input, np.quaternion):
            return abs(int(n_input.w))
        if isinstance(n_input, NeutrosophicNumber):
            return abs(int(n_input.a))
        if isinstance(n_input, NeutrosophicComplexNumber):
            return abs(int(n_input.real))
        if isinstance(n_input, HyperrealNumber):
            return abs(int(n_input.sequence[0])) if n_input.sequence else 0
        if isinstance(n_input, BicomplexNumber):
            return abs(int(n_input.z1.real))
        if isinstance(n_input, NeutrosophicBicomplexNumber):
            return abs(int(n_input.real))
        return abs(int(n_input))
    except (ValueError, TypeError, IndexError):
        return None

def is_prime(n_input: Any) -> bool:
    """
    Checks if a given number (or its principal component) is prime
    using the robust sympy.isprime function.
    """
    # Adım 1: Karmaşık sayı türünden tamsayıyı çıkarma (Bu kısım aynı kalıyor)
    value_to_check = _get_integer_representation(n_input)

    # Adım 2: Tamsayı geçerli değilse False döndür
    if value_to_check is None:
        return False
    
    # Adım 3: Asallık testini sympy'ye bırak
    # sympy.isprime, 2'den küçük sayılar (1, 0, negatifler) için zaten False döndürür.
    return sympy.isprime(value_to_check)
    
"""
def is_prime(n_input: Any) -> bool:
    #Checks if a given number (or its principal component) is prime.
    value_to_check = _get_integer_representation(n_input)
    if value_to_check is None or value_to_check < 2:
        return False
    if value_to_check == 2:
        return True
    if value_to_check % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(value_to_check)) + 1, 2):
        if value_to_check % i == 0:
            return False
    return True
"""

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
        if c.imag == 0 and 'j' not in s_clean:
            return complex(c.real, c.real)
        return c
    except ValueError as e:
        raise ValueError(f"Invalid complex number format: '{s}'") from e

def _parse_neutrosophic(s: str) -> Tuple[float, float]:
    """Parses a neutrosophic string 'a+bI' into a tuple (a, b)."""
    s = s.strip().replace(" ", "").upper()
    if not s:
        return 0.0, 0.0

    if 'I' not in s:
        return float(s), 0.0

    pattern = re.compile(r"^(?P<a>[+-]?\d+\.?\d*)?(?P<b>[+-]?)I$")
    match = re.match(r"^(?P<val>[+-]?\d+\.?\d*)$", s)
    if match: # Just a number
        return float(match.group('val')), 0.0
        
    pattern = re.compile(r"^(?P<a>[+-]?\d+\.?\d*)?(?P<b>[+-]?\d*\.?\d*)I$")
    full_match = pattern.match(s)
    if not full_match:
        raise ValueError(f"Invalid neutrosophic format: {s}")

    parts = full_match.groupdict()
    a_part = parts.get('a') or "0"
    b_part = parts.get('b')

    if b_part in (None, "", "+"):
        b_val = 1.0
    elif b_part == "-":
        b_val = -1.0
    else:
        b_val = float(b_part)
        
    return float(a_part), b_val


def _parse_hyperreal(s: str) -> Tuple[float, float]:
    """Parses 'a+be' string into a tuple (a, b)."""
    s = s.strip().replace(" ", "").lower()
    if not s:
        return 0.0, 0.0
    if 'e' not in s:
        return float(s), 0.0

    pattern = re.compile(r"^(?P<a>[+-]?\d+\.?\d*)?(?P<b>[+-]?\d*\.?\d*)e$")
    match = pattern.match(s)
    if not match:
        raise ValueError(f"Invalid hyperreal format: {s}")

    parts = match.groupdict()
    a_part = parts.get('a') or "0"
    b_part = parts.get('b')
    
    if b_part in (None, "", "+"):
        b_val = 1.0
    elif b_part == "-":
        b_val = -1.0
    else:
        b_val = float(b_part)
        
    return float(a_part), b_val

def _parse_quaternion(s: str) -> np.quaternion:
    """Parses user string ('a+bi+cj+dk' or scalar) into a quaternion."""
    s_clean = s.replace(" ", "").lower()
    if not s_clean:
        raise ValueError("Input cannot be empty.")

    try:
        val = float(s_clean)
        return np.quaternion(val, val, val, val)
    except ValueError:
        pass
    
    s_temp = re.sub(r'([+-])([ijk])', r'\g<1>1\g<2>', s_clean)
    if s_temp.startswith(('i', 'j', 'k')):
        s_temp = '1' + s_temp
    
    pattern = re.compile(r'([+-]?\d*\.?\d*)([ijk])?')
    matches = pattern.findall(s_temp)
    
    parts = {'w': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    for value_str, component in matches:
        if not value_str:
            continue
        value = float(value_str)
        if component == 'i':
            parts['x'] += value
        elif component == 'j':
            parts['y'] += value
        elif component == 'k':
            parts['z'] += value
        else:
            parts['w'] += value
            
    return np.quaternion(parts['w'], parts['x'], parts['y'], parts['z'])

def get_random_type(num_iterations: int, fixed_start_raw: str = "0", fixed_add_base_scalar: float = 9.0) -> List[Any]:
    """Generates Keçeci Numbers for a randomly selected type."""
    random_type_choice = random.randint(1, 11)
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutro-Complex", "Hyperreal", 
        "Bicomplex", "Neutro-Bicomplex"
    ]
    print(f"\nRandomly selected Keçeci Number Type: {random_type_choice} ({type_names_list[random_type_choice-1]})")
    
    return get_with_params(
        kececi_type_choice=random_type_choice, 
        iterations=num_iterations,
        start_value_raw=fixed_start_raw,
        add_value_raw=fixed_add_base_scalar
    )

def generate_kececi_vectorial(q0_str, c_str, u_str, iterations):
    """
    Keçeci Haritası'nı tam vektörel toplama ile üreten geliştirilmiş fonksiyon.
    Bu, kütüphanenin ana üretim fonksiyonu olabilir.
    Tüm girdileri metin (string) olarak alarak esneklik sağlar.
    """
    try:
        # Girdi metinlerini kuaterniyon nesnelerine dönüştür
        w, x, y, z = map(float, q0_str.split(','))
        q0 = np.quaternion(w, x, y, z)
        
        cw, cx, cy, cz = map(float, c_str.split(','))
        c = np.quaternion(cw, cx, cy, cz)

        uw, ux, uy, uz = map(float, u_str.split(','))
        u = np.quaternion(uw, ux, uy, uz)

    except (ValueError, IndexError):
        raise ValueError("Girdi metinleri 'w,x,y,z' formatında olmalıdır.")

    trajectory = [q0]
    prime_events = []
    current_q = q0

    for i in range(iterations):
        y = current_q + c
        processing_val = y

        while True:
            scalar_int = int(processing_val.w)

            if scalar_int % 2 == 0:
                next_q = processing_val / 2.0
                break
            elif scalar_int % 3 == 0:
                next_q = processing_val / 3.0
                break
            elif is_prime(scalar_int):
                if processing_val == y:
                    prime_events.append((i, scalar_int))
                processing_val += u
                continue
            else:
                next_q = processing_val
                break
        
        trajectory.append(next_q)
        current_q = next_q
        
    return trajectory, prime_events

def _parse_quaternion_from_csv(s: str) -> np.quaternion:
    """Parses a comma-separated string 'w,x,y,z' into a quaternion."""
    try:
        parts = [float(p.strip()) for p in s.split(',')]
        if len(parts) != 4:
            raise ValueError("Girdi 4 bileşen içermelidir.")
        # *parts -> (parts[0], parts[1], parts[2], parts[3])
        return np.quaternion(*parts)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Geçersiz virgülle ayrılmış kuaterniyon formatı: '{s}'.") from e

def _load_zeta_zeros(filename="zeta.txt"):
    """
    Loads Riemann zeta zeros from a text file.
    Each line should contain one floating-point number representing the imaginary part of a zeta zero.
    Lines that are empty or start with '#' are ignored.
    Returns:
        numpy.ndarray: Array of zeta zeros, or empty array if file not found.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        zeta_zeros = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                zeta_zeros.append(float(line))
            except ValueError:
                print(f"Invalid line skipped: {line}")
        print(f"{len(zeta_zeros)} zeta zeros loaded.")
        return np.array(zeta_zeros)
    except FileNotFoundError:
        print(f"'{filename}' not found.")
        return np.array([])


def _compute_gue_similarity(sequence, tolerance=0.5):
    """
    Measures how closely the frequency spectrum of a Keçeci sequence matches the GUE (Gaussian Unitary Ensemble) statistics.
    Uses Kolmogorov-Smirnov test against Wigner-Dyson distribution.
    Args:
        sequence (list): The Keçeci number sequence.
        tolerance (float): Not used here; kept for interface consistency.
    Returns:
        tuple: (similarity_score, p_value)
    """
    from . import _get_integer_representation

    values = [val for z in sequence if (val := _get_integer_representation(z)) is not None]
    if len(values) < 10:
        return 0.0, 0.0

    values = np.array(values) - np.mean(values)
    N = len(values)
    powers = np.abs(fft(values))**2
    freqs = fftfreq(N)

    mask = (freqs > 0)
    freqs_pos = freqs[mask]
    powers_pos = powers[mask]

    if len(powers_pos) == 0:
        return 0.0, 0.0

    peaks, _ = find_peaks(powers_pos, height=np.max(powers_pos)*1e-7)
    strong_freqs = freqs_pos[peaks]

    if len(strong_freqs) < 2:
        return 0.0, 0.0

    # Scale so the strongest peak aligns with the first Riemann zeta zero
    peak_freq = strong_freqs[np.argmax(powers_pos[peaks])]
    scale_factor = 14.134725 / peak_freq
    scaled_freqs = np.sort(strong_freqs * scale_factor)

    # Compute level spacings
    if len(scaled_freqs) < 2:
        return 0.0, 0.0
    diffs = np.diff(scaled_freqs)
    if np.mean(diffs) == 0:
        return 0.0, 0.0
    diffs_norm = diffs / np.mean(diffs)

    # Generate GUE sample using Wigner-Dyson distribution
    def wigner_dyson(s):
        return (32 / np.pi) * s**2 * np.exp(-4 * s**2 / np.pi)

    s_gue = np.linspace(0.01, 3.0, 1000)
    p_gue = wigner_dyson(s_gue)
    p_gue = p_gue / np.sum(p_gue)
    sample_gue = np.random.choice(s_gue, size=1000, p=p_gue)

    # Perform KS test
    ks_stat, ks_p = ks_2samp(diffs_norm, sample_gue)
    similarity_score = 1.0 - ks_stat

    return similarity_score, ks_p


def _find_kececi_zeta_zeros(sequence, tolerance=0.5):
    """
    Estimates the zeros of the Keçeci Zeta Function from the spectral peaks of the sequence.
    Compares them to known Riemann zeta zeros.
    Args:
        sequence (list): The Keçeci number sequence.
        tolerance (float): Maximum distance for a match between Keçeci and Riemann zeros.
    Returns:
        tuple: (list of Keçeci zeta zeros, matching score)
    """
    from . import _get_integer_representation

    values = [val for z in sequence if (val := _get_integer_representation(z)) is not None]
    if len(values) < 10:
        return [], 0.0

    values = np.array(values) - np.mean(values)
    N = len(values)
    powers = np.abs(fft(values))**2
    freqs = fftfreq(N)

    mask = (freqs > 0)
    freqs_pos = freqs[mask]
    powers_pos = powers[mask]

    if len(powers_pos) == 0:
        return [], 0.0

    peaks, _ = find_peaks(powers_pos, height=np.max(powers_pos)*1e-7)
    strong_freqs = freqs_pos[peaks]

    if len(strong_freqs) < 2:
        return [], 0.0

    # Scale so the strongest peak aligns with the first Riemann zeta zero
    peak_freq = strong_freqs[np.argmax(powers_pos[peaks])]
    scale_factor = 14.134725 / peak_freq
    scaled_freqs = np.sort(strong_freqs * scale_factor)

    # Find candidate zeros by analyzing the Keçeci Zeta Function
    t_vals = np.linspace(0, 650, 10000)
    zeta_vals = np.array([sum((scaled_freqs + 1e-10)**(- (0.5 + 1j * t))) for t in t_vals])
    minima, _ = find_peaks(-np.abs(zeta_vals), height=-0.5*np.max(np.abs(zeta_vals)), distance=5)
    kececi_zeta_zeros = t_vals[minima]

    # Load Riemann zeta zeros for comparison
    zeta_zeros_imag = _load_zeta_zeros("zeta.txt")
    if len(zeta_zeros_imag) == 0:
        return kececi_zeta_zeros, 0.0

    # Calculate matching score
    close_matches = [kz for kz in kececi_zeta_zeros if min(abs(kz - zeta_zeros_imag)) < tolerance]
    score = len(close_matches) / len(kececi_zeta_zeros) if kececi_zeta_zeros.size > 0 else 0.0

    return kececi_zeta_zeros, score


def analyze_all_types(iterations=120):
    """
    Performs automated analysis on all 11 Keçeci number types.
    For each type, it tests multiple parameter sets, computes similarity to Riemann zeta zeros and GUE statistics,
    then reports and plots the results.
    Args:
        iterations (int): Number of Keçeci steps to generate for each sequence.
    Returns:
        tuple: (sorted_by_zeta, sorted_by_gue) - Lists of results sorted by performance.
    """
    from . import (
        get_with_params,
        TYPE_POSITIVE_REAL,
        TYPE_NEGATIVE_REAL,
        TYPE_COMPLEX,
        TYPE_FLOAT,
        TYPE_RATIONAL,
        TYPE_QUATERNION,
        TYPE_NEUTROSOPHIC,
        TYPE_NEUTROSOPHIC_COMPLEX,
        TYPE_HYPERREAL,
        TYPE_BICOMPLEX,
        TYPE_NEUTROSOPHIC_BICOMPLEX
    )

    print("Automated Analysis for 11 Keçeci Types")
    print("=" * 80)

    include_intermediate = True
    results = []

    # Parameter sets to test
    param_sets = [
        ('0.0', '9.0'),
        ('1.0', '7.0'),
        ('2.0', '5.0'),
        ('3.0', '11.0'),
        ('1+1j', '9.0'),
        ('0.0001412', '0.037')
    ]

    type_names = {
        1: "Positive Real",
        2: "Negative Real",
        3: "Complex",
        4: "Float",
        5: "Rational",
        6: "Quaternion",
        7: "Neutrosophic",
        8: "Neutro-Complex",
        9: "Hyperreal",
        10: "Bicomplex",
        11: "Neutro-Bicomplex"
    }

    for kececi_type in range(1, 12):
        name = type_names[kececi_type]
        best_zeta_score = 0.0
        best_gue_score = 0.0
        best_params = None

        print(f"Analyzing type {kececi_type} ({name})...")

        for start, add in param_sets:
            try:
                # Special formatting for complex types
                if kececi_type == 3 and '+' not in start:
                    start = f"{start}+{start}j"
                if kececi_type == 10 and '+' not in start:
                    start = f"{start}+{start}j"

                sequence = get_with_params(
                    kececi_type_choice=kececi_type,
                    iterations=iterations,
                    start_value_raw=start,
                    add_value_raw=add,
                    include_intermediate_steps=include_intermediate
                )

                if not sequence or len(sequence) < 50:
                    continue

                _, zeta_score = _find_kececi_zeta_zeros(sequence, tolerance=0.5)
                _, gue_score = _compute_gue_similarity(sequence)

                if zeta_score > best_zeta_score:
                    best_zeta_score = zeta_score
                    best_gue_score = gue_score
                    best_params = (start, add)

            except Exception as e:
                continue

        if best_params:
            results.append({
                'type': kececi_type,
                'name': name,
                'start': best_params[0],
                'add': best_params[1],
                'zeta_score': best_zeta_score,
                'gue_score': best_gue_score
            })

    # Sort results
    sorted_by_zeta = sorted(results, key=lambda x: x['zeta_score'], reverse=True)
    sorted_by_gue = sorted(results, key=lambda x: x['gue_score'], reverse=True)

    print("\n" + "=" * 100)
    print("HIGHEST RIEMANN ZETA MATCHING SCORES (TOP 11)")
    print("=" * 100)
    print(f"{'Type':<20} {'Score':<8} {'Start':<12} {'Increment':<12}")
    print("-" * 100)
    for r in sorted_by_zeta:
        print(f"{r['name']:<20} {r['zeta_score']:<8.3f} {r['start']:<12} {r['add']:<12}")

    print("\n" + "=" * 100)
    print("HIGHEST GUE SIMILARITY SCORES (TOP 11)")
    print("=" * 100)
    print(f"{'Type':<20} {'Score':<8} {'Start':<12} {'Increment':<12}")
    print("-" * 100)
    for r in sorted_by_gue:
        print(f"{r['name']:<20} {r['gue_score']:<8.3f} {r['start']:<12} {r['add']:<12}")

    # Plot results
    _plot_comparison(sorted_by_zeta, sorted_by_gue)

    return sorted_by_zeta, sorted_by_gue


def _plot_comparison(zeta_results, gue_results):
    """
    Creates bar charts comparing the performance of Keçeci types in matching Riemann zeta zeros and GUE statistics.
    Args:
        zeta_results (list): Results sorted by zeta matching score.
        gue_results (list): Results sorted by GUE similarity score.
    """
    # Riemann Zeta Matching Plot
    plt.figure(figsize=(14, 7))
    types = [r['name'] for r in zeta_results]
    scores = [r['zeta_score'] for r in zeta_results]
    colors = ['skyblue'] * len(scores)
    if scores:
        colors[0] = 'red'
    bars = plt.bar(types, scores, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Riemann Zeta Matching Score")
    plt.title("Keçeci Types vs Riemann Zeta Zeros")
    plt.grid(True, alpha=0.3)
    if bars:
        bars[0].set_edgecolor('darkred')
        bars[0].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()

    # GUE Similarity Plot
    plt.figure(figsize=(14, 7))
    types = [r['name'] for r in gue_results]
    scores = [r['gue_score'] for r in gue_results]
    colors = ['skyblue'] * len(scores)
    if scores:
        colors[0] = 'red'
    bars = plt.bar(types, scores, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("GUE Similarity Score")
    plt.title("Keçeci Types vs GUE Statistics")
    plt.grid(True, alpha=0.3)
    if bars:
        bars[0].set_edgecolor('darkred')
        bars[0].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()

def _pair_correlation(ordered_zeros, max_gap=3.0, bin_size=0.1):
    """
    Computes the pair correlation of a list of ordered zeros.
    This function calculates the normalized spacings between all pairs of zeros
    and returns a histogram of their distribution.
    Args:
        ordered_zeros (numpy.ndarray): Sorted array of zero locations (e.g., Keçeci or Riemann zeta zeros).
        max_gap (float): Maximum normalized gap to consider.
        bin_size (float): Size of bins for the histogram.
    Returns:
        tuple: (bin_centers, histogram) - The centers of the bins and the normalized histogram values.
    """
    n = len(ordered_zeros)
    if n < 2:
        return np.array([]), np.array([])

    # Compute average spacing for normalization
    avg_spacing = np.mean(np.diff(ordered_zeros))
    normalized_zeros = ordered_zeros / avg_spacing

    # Compute all pairwise gaps within max_gap
    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(normalized_zeros[j] - normalized_zeros[i])
            if gap <= max_gap:
                gaps.append(gap)

    # Create histogram
    bins = np.arange(0, max_gap + bin_size, bin_size)
    hist, _ = np.histogram(gaps, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, hist


def _gue_pair_correlation(s):
    """
    Theoretical pair correlation function for the Gaussian Unitary Ensemble (GUE).
    This function is used as a reference for comparing the statistical distribution
    of eigenvalues (or zeta zeros) in quantum chaotic systems.
    Args:
        s (numpy.ndarray or float): Normalized spacing(s).
    Returns:
        numpy.ndarray or float: The GUE pair correlation value(s) at s.
    """
    return 1 - np.sinc(s)**2


def analyze_pair_correlation(sequence, title="Pair Correlation of Keçeci Zeta Zeros"):
    """
    Analyzes and plots the pair correlation of Keçeci Zeta zeros derived from a Keçeci sequence.
    Compares the empirical pair correlation to the theoretical GUE prediction.
    Performs a Kolmogorov-Smirnov test to quantify the similarity.
    Args:
        sequence (list): A Keçeci number sequence.
        title (str): Title for the resulting plot.
    """
    from . import _get_integer_representation

    # Extract integer representations and remove DC component
    values = [val for z in sequence if (val := _get_integer_representation(z)) is not None]
    if len(values) < 10:
        print("Insufficient data.")
        return

    values = np.array(values) - np.mean(values)
    N = len(values)
    powers = np.abs(fft(values))**2
    freqs = fftfreq(N)

    # Filter positive frequencies
    mask = (freqs > 0)
    freqs_pos = freqs[mask]
    powers_pos = powers[mask]

    if len(powers_pos) == 0:
        print("No positive frequencies found.")
        return

    # Find spectral peaks
    peaks, _ = find_peaks(powers_pos, height=np.max(powers_pos)*1e-7)
    strong_freqs = freqs_pos[peaks]

    if len(strong_freqs) < 2:
        print("Insufficient frequency peaks.")
        return

    # Scale frequencies so the strongest peak aligns with the first Riemann zeta zero
    peak_freq = strong_freqs[np.argmax(powers_pos[peaks])]
    scale_factor = 14.134725 / peak_freq
    scaled_freqs = np.sort(strong_freqs * scale_factor)

    # Estimate Keçeci Zeta zeros by finding minima of |ζ_Kececi(0.5 + it)|
    t_vals = np.linspace(0, 650, 10000)
    zeta_vals = np.array([sum((scaled_freqs + 1e-10)**(- (0.5 + 1j * t))) for t in t_vals])
    minima, _ = find_peaks(-np.abs(zeta_vals), height=-0.5*np.max(np.abs(zeta_vals)), distance=5)
    kececi_zeta_zeros = t_vals[minima]

    if len(kececi_zeta_zeros) < 2:
        print("Insufficient Keçeci zeta zeros found.")
        return

    # Compute pair correlation
    bin_centers, hist = _pair_correlation(kececi_zeta_zeros, max_gap=3.0, bin_size=0.1)
    gue_corr = _gue_pair_correlation(bin_centers)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, hist, 'o-', label="Keçeci Zeta Zeros", linewidth=2)
    plt.plot(bin_centers, gue_corr, 'r-', label="GUE (Theoretical)", linewidth=2)
    plt.title(title)
    plt.xlabel("Normalized Spacing (s)")
    plt.ylabel("Pair Correlation Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_p = ks_2samp(hist, gue_corr)
    print(f"Pair Correlation KS Test: Statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

# ==============================================================================
# --- CORE GENERATOR ---
# ==============================================================================

def unified_generator(kececi_type: int, start_input_raw: str, add_input_raw: str, iterations: int, include_intermediate_steps: bool = False) -> List[Any]:
    """
    Core engine to generate Keçeci Number sequences.

    Bu nihai versiyon, tüm sayı tipleri için esnek girdi işleme, kuaterniyonlar
    için tam vektörel toplama desteği sunar ve isteğe bağlı olarak ara
    hesaplama adımlarını da, veri tekrarı olmadan doğru bir şekilde döndürür.

    Args:
        kececi_type (int): Keçeci Sayı türü (1-11).
        start_input_raw (str): Başlangıç değerini temsil eden metin.
        add_input_raw (str): Her adımda eklenecek sabiti temsil eden metin.
        iterations (int): Üretilecek Keçeci adımı sayısı.
        include_intermediate_steps (bool, optional): True ise, ara hesaplama
            değerlerini de son listeye ekler. Varsayılan: False.

    Returns:
        List[Any]: Oluşturulan Keçeci Sayıları dizisi.
    """
    
    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_NEUTROSOPHIC_BICOMPLEX):
        raise ValueError(f"Invalid Keçeci Number Type: {kececi_type}. Must be between {TYPE_POSITIVE_REAL} and {TYPE_NEUTROSOPHIC_BICOMPLEX}.")

    # --- 1. Değişkenlerin Başlatılması ---
    # Varsayılan bölme türünü ondalıklı olarak ayarla.
    # Tamsayı tipleri (POSITIVE/NEGATIVE_REAL) bu değeri kendi blokları içinde ezecektir.
    use_integer_division = False
    
    try:
        # Her sayı tipi, kendi `elif` bloğu içinde kendi girdisini işler.
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            current_value = int(float(start_input_raw)); 
            add_value_typed = int(float(add_input_raw)); 
            ask_unit = 1; 
            use_integer_division = True
        elif kececi_type == TYPE_FLOAT:
            current_value = float(start_input_raw); 
            add_value_typed = float(add_input_raw); 
            ask_unit = 1.0
        elif kececi_type == TYPE_RATIONAL:
            current_value = Fraction(start_input_raw); 
            add_value_typed = Fraction(add_input_raw); 
            ask_unit = Fraction(1)
        elif kececi_type == TYPE_COMPLEX:
            current_value = _parse_complex(start_input_raw); 
            a_float = float(add_input_raw); 
            add_value_typed = complex(a_float, a_float); 
            ask_unit = 1 + 1j
        elif kececi_type == TYPE_QUATERNION:
            current_value = _parse_quaternion_from_csv(start_input_raw); 
            add_value_typed = _parse_quaternion_from_csv(add_input_raw); 
            ask_unit = np.quaternion(1, 1, 1, 1)
        elif kececi_type == TYPE_NEUTROSOPHIC:
            a, b = _parse_neutrosophic(start_input_raw); 
            current_value = NeutrosophicNumber(a, b); 
            a_float = float(add_input_raw); 
            add_value_typed = NeutrosophicNumber(a_float, 0); 
            ask_unit = NeutrosophicNumber(1, 1)
        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            s_complex = _parse_complex(start_input_raw); 
            current_value = NeutrosophicComplexNumber(s_complex.real, s_complex.imag, 0.0); 
            a_float = float(add_input_raw); 
            add_value_typed = NeutrosophicComplexNumber(a_float, 0.0, 0.0); 
            ask_unit = NeutrosophicComplexNumber(1, 1, 1)
        elif kececi_type == TYPE_HYPERREAL:
            a, b = _parse_hyperreal(start_input_raw); 
            sequence_list = [a + b / n for n in range(1, 11)]; 
            current_value = HyperrealNumber(sequence_list); 
            a_float = float(add_input_raw); 
            add_sequence = [a_float] + [0.0] * 9; 
            add_value_typed = HyperrealNumber(add_sequence); 
            ask_unit = HyperrealNumber([1.0] * 10)
        elif kececi_type == TYPE_BICOMPLEX:
            s_complex = _parse_complex(start_input_raw); 
            a_float = float(add_input_raw); 
            a_complex = complex(a_float); 
            current_value = BicomplexNumber(s_complex, s_complex / 2); 
            add_value_typed = BicomplexNumber(a_complex, a_complex / 2); 
            ask_unit = BicomplexNumber(complex(1, 1), complex(0.5, 0.5))
        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            s_complex = _parse_complex(start_input_raw); 
            current_value = NeutrosophicBicomplexNumber(s_complex.real, s_complex.imag, 0, 0, 0, 0, 0, 0); 
            a_float = float(add_input_raw); 
            add_value_typed = NeutrosophicBicomplexNumber(a_float, 0, 0, 0, 0, 0, 0, 0); 
            ask_unit = NeutrosophicBicomplexNumber(*([1.0] * 8))
    except (ValueError, TypeError) as e:
        print(f"ERROR: Failed to initialize type {kececi_type} with start='{start_input_raw}' and increment='{add_input_raw}': {e}")
        return []

    # --- 2. Üreteç Döngüsü ---
    clean_trajectory = [current_value]
    full_log = [current_value]
    
    last_divisor_used = None
    ask_counter = 0
    
    for _ in range(iterations):
        # --- Bir Sonraki Adımın Değerini (next_q) Hesapla ---
        added_value = current_value + add_value_typed
        
        next_q = added_value
        divided_successfully = False
        modified_value = None

        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3
        
        for divisor in [primary_divisor, alternative_divisor]:
            if _is_divisible(added_value, divisor, kececi_type):
                next_q = added_value // divisor if use_integer_division else added_value / divisor
                last_divisor_used = divisor
                divided_successfully = True
                break
        
        if not divided_successfully and is_prime(added_value):
            modified_value = (added_value + ask_unit) if ask_counter == 0 else (added_value - ask_unit)
            ask_counter = 1 - ask_counter
            
            next_q = modified_value 
            
            for divisor in [primary_divisor, alternative_divisor]:
                if _is_divisible(modified_value, divisor, kececi_type):
                    next_q = modified_value // divisor if use_integer_division else modified_value / divisor
                    last_divisor_used = divisor
                    break
        
        # --- Sonuçları Ayrı ve Doğru Listelere Kaydet ---
        full_log.append(added_value)
        if modified_value is not None:
            full_log.append(modified_value)
        
        # Nihai sonucu, eğer bir önceki ara adımdan farklıysa log'a ekle.
        # Bu, `(12.3, ...), (12.3, ...)` tekrarını önler.
        if not full_log or next_q != full_log[-1]:
             full_log.append(next_q)
        
        clean_trajectory.append(next_q)
        
        # --- Durumu Güncelle ---
        current_value = next_q
        
    # --- 3. İsteğe Göre Doğru Listeyi Döndür ---
    if include_intermediate_steps:
        return full_log
    else:
        return clean_trajectory

def print_detailed_report(sequence: List[Any], params: Dict[str, Any]):
    """Generates and prints a detailed report of the sequence results."""
    if not sequence:
        print("\n--- REPORT ---\nSequence could not be generated.")
        return

    print("\n\n" + "="*50)
    print("--- DETAILED SEQUENCE REPORT ---")
    print("="*50)

    print("\n[Parameters Used]")
    print(f"  - Keçeci Type:   {params.get('type_name', 'N/A')} ({params['type_choice']})")
    print(f"  - Start Value:   '{params['start_val']}'")
    print(f"  - Increment:     {params['add_val']}")
    print(f"  - Keçeci Steps:  {params['steps']}")

    print("\n[Sequence Summary]")
    print(f"  - Total Numbers Generated: {len(sequence)}")
    
    kpn = find_kececi_prime_number(sequence)
    print(f"  - Keçeci Prime Number (KPN): {kpn if kpn is not None else 'Not found'}")

    print("\n[Sequence Preview]")
    preview_count = min(len(sequence), 30)
    print(f"  --- First {preview_count} Numbers ---")
    for i in range(preview_count):
        print(f"    {i}: {sequence[i]}")

    if len(sequence) > preview_count:
        print(f"\n  --- Last {preview_count} Numbers ---")
        for i in range(len(sequence) - preview_count, len(sequence)):
            print(f"    {i}: {sequence[i]}")
            
    print("\n" + "="*50)

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

def get_with_params(kececi_type_choice: int, iterations: int, start_value_raw: str, add_value_raw: str, include_intermediate_steps: bool = False) -> List[Any]:
    """Generates Keçeci Numbers with specified parameters, supporting full vectorial addition."""
    print(f"\n--- Generating Sequence: Type {kececi_type_choice}, Steps {iterations} ---")
    print(f"Start: '{start_value_raw}', Increment: '{add_value_raw}'")
    if include_intermediate_steps:
        print("Mode: Detailed (including intermediate steps)")

    generated_sequence = unified_generator(
        kececi_type_choice, 
        start_value_raw, 
        add_value_raw, 
        iterations,
        # Yeni parametreyi aktar
        include_intermediate_steps=include_intermediate_steps 
    )
    
    if generated_sequence:
        print(f"Generated {len(generated_sequence)} numbers. Preview: {generated_sequence[:30]}...")
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            print(f"Keçeci Prime Number for this sequence: {kpn}")
        else:
            print("No repeating Keçeci Prime Number found.")
    else:
        print("Sequence generation failed.")
        
    return generated_sequence

def get_interactive() -> Tuple[List[Any], Dict[str, Any]]:
    """
    Interactively gets parameters from the user to generate a Keçeci Numbers sequence.
    This version is improved for code clarity and compatibility with the module's
    current interface.
    """
    print("\n--- Keçeci Numbers Interactive Generator ---")
    print("  1: Positive Real    2: Negative Real     3: Complex")
    print("  4: Float            5: Rational          6: Quaternion")
    print("  7: Neutrosophic     8: Neutro-Complex   9: Hyperreal")
    print(" 10: Bicomplex        11: Neutro-Bicomplex")
    
    # Get a valid number type from the user
    while True:
        try:
            type_choice = int(input("Select a Keçeci Number Type (1-11): "))
            if 1 <= type_choice <= 11:
                break
            print("Invalid type. Please enter a number between 1 and 11.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        
    # User prompts for the starting value
    start_prompts = {
        1: "Enter positive integer start (e.g., '10.0'): ",
        2: "Enter negative integer start (e.g., '-5.0'): ",
        3: "Enter complex start (e.g., '3+4j'): ",
        4: "Enter float start (e.g., '3.14'): ",
        5: "Enter rational start (e.g., '7/2'): ",
        6: "Enter quaternion start (in 'w,x,y,z' format, e.g., '1.0,2.0,-3.0,1.0'): ",
        7: "Enter neutrosophic start (e.g., '5+2I'): ",
        8: "Enter complex base for neutro-complex (e.g., '1-2j'): ",
        9: "Enter hyperreal start (e.g., '5+3e'): ",
        10: "Enter complex base for bicomplex (e.g., '2+1j'): ",
        11: "Enter complex base for neutro-bicomplex (e.g., '1+2j'): "
    }
    
    # User prompts for the increment value
    add_prompts = {
        TYPE_QUATERNION: "Enter quaternion increment (in 'w,x,y,z' format, e.g., '1.3,-2.1,0.5,3.4'): "
    }
    default_add_prompt = "Enter increment value (e.g., '9.0'): "
    
    # Get inputs from the user
    start_input_val_raw = input(start_prompts.get(type_choice, "Enter starting value: "))
    add_input_val_raw = input(add_prompts.get(type_choice, default_add_prompt))
    num_kececi_steps = int(input("Enter number of Keçeci steps (e.g., 30): "))

    show_details_input = input("Do you want to include the intermediate calculation steps? (y/n): ").lower().strip()
    show_details = (show_details_input == 'y')
    
    # Generate the sequence with the correct parameter names and values
    sequence = get_with_params(
        kececi_type_choice=type_choice,
        iterations=num_kececi_steps,
        start_value_raw=start_input_val_raw,
        add_value_raw=add_input_val_raw,
        include_intermediate_steps=show_details
    )
    
    # Gather the parameters in a dictionary to return
    params = {
        "type_choice": type_choice,
        "start_val": start_input_val_raw,
        "add_val": add_input_val_raw,
        "steps": num_kececi_steps,
        "detailed_view": show_details
    }
    return sequence, params

# ==============================================================================
# --- ANALYSIS AND PLOTTING ---
# ==============================================================================

def find_period(sequence: List[Any], min_repeats: int = 3) -> Optional[List[Any]]:
    """
    Checks if the end of a sequence has a repeating cycle (period).

    Args:
        sequence: The list of numbers to check.
        min_repeats: How many times the cycle must repeat to be considered stable.

    Returns:
        The repeating cycle as a list if found, otherwise None.
    """
    if len(sequence) < 10:  # Çok kısa dizilerde periyot aramak anlamsız
        return None

    # Olası periyot uzunluklarını dizinin yarısına kadar kontrol et
    for p_len in range(1, len(sequence) // min_repeats):
        # Dizinin sonundan potansiyel döngüyü al
        candidate_cycle = sequence[-p_len:]
        
        # Döngünün en az `min_repeats` defa tekrar edip etmediğini kontrol et
        is_periodic = True
        for i in range(1, min_repeats):
            start_index = -(i + 1) * p_len
            end_index = -i * p_len
            
            # Dizinin o bölümünü al
            previous_block = sequence[start_index:end_index]

            # Eğer bloklar uyuşmuyorsa, bu periyot değildir
            if candidate_cycle != previous_block:
                is_periodic = False
                break
        
        # Eğer döngü tüm kontrollerden geçtiyse, periyodu bulduk demektir
        if is_periodic:
            return candidate_cycle

    # Hiçbir periyot bulunamadı
    return None

def find_kececi_prime_number(kececi_numbers_list: List[Any]) -> Optional[int]:
    """Finds the Keçeci Prime Number from a generated sequence."""
    if not kececi_numbers_list:
        return None

    integer_prime_reps = [
        rep for num in kececi_numbers_list
        if is_prime(num) and (rep := _get_integer_representation(num)) is not None
    ]

    if not integer_prime_reps:
        return None

    counts = collections.Counter(integer_prime_reps)
    repeating_primes = [(freq, prime) for prime, freq in counts.items() if freq > 1]
    if not repeating_primes:
        return None
    
    _, best_prime = max(repeating_primes)
    return best_prime

def plot_numbers(sequence: List[Any], title: str = "Keçeci Number Sequence Analysis"):
    """Plots the generated sequence with detailed visualizations for each type."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not sequence:
        print("Sequence is empty, nothing to plot.")
        return

    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(title, fontsize=16, y=0.98)
    first_elem = sequence[0]
    
    if isinstance(first_elem, (int, float, Fraction)):
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([float(x) for x in sequence], 'o-', label="Value")
        ax.set_title("Value over Iterations")
        ax.set_xlabel("Index"), ax.set_ylabel("Value"), ax.legend()

    elif isinstance(first_elem, complex):
        gs = GridSpec(2, 2, figure=fig)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        real_parts, imag_parts = [c.real for c in sequence], [c.imag for c in sequence]
        ax1.plot(real_parts, 'o-', label='Real Part'), ax1.set_title("Real Part"), ax1.legend()
        ax2.plot(imag_parts, 'o-', color='red', label='Imaginary Part'), ax2.set_title("Imaginary Part"), ax2.legend()
        ax3.plot(real_parts, imag_parts, '.-', label='Trajectory')
        ax3.scatter(real_parts[0], imag_parts[0], c='g', s=100, label='Start', zorder=5)
        ax3.scatter(real_parts[-1], imag_parts[-1], c='r', s=100, label='End', zorder=5)
        ax3.set_title("Trajectory in Complex Plane"), ax3.set_xlabel("Real"), ax3.set_ylabel("Imaginary"), ax3.legend(), ax3.axis('equal')

    elif isinstance(first_elem, np.quaternion):
        gs = GridSpec(2, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax1.plot([q.w for q in sequence], 'o-', label='w (scalar)'), ax1.plot([q.x for q in sequence], 's--', label='x')
        ax1.plot([q.y for q in sequence], '^--', label='y'), ax1.plot([q.z for q in sequence], 'd--', label='z')
        ax1.set_title("Quaternion Components"), ax1.legend()
        magnitudes = [abs(q) for q in sequence]
        ax2.plot(magnitudes, 'o-', color='purple', label='Magnitude'), ax2.set_title("Magnitude"), ax2.legend(), ax2.set_xlabel("Index")

    elif isinstance(first_elem, BicomplexNumber):
        gs = GridSpec(2, 2, figure=fig)
        ax1, ax2, ax3, ax4 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
        z1r, z1i = [x.z1.real for x in sequence], [x.z1.imag for x in sequence]
        z2r, z2i = [x.z2.real for x in sequence], [x.z2.imag for x in sequence]
        ax1.plot(z1r, label='z1.real'), ax1.plot(z1i, label='z1.imag'), ax1.set_title("Component z1"), ax1.legend()
        ax2.plot(z2r, label='z2.real'), ax2.plot(z2i, label='z2.imag'), ax2.set_title("Component z2"), ax2.legend()
        ax3.plot(z1r, z1i, '.-'), ax3.set_title("z1 Trajectory"), ax3.set_xlabel("Real"), ax3.set_ylabel("Imaginary")
        ax4.plot(z2r, z2i, '.-'), ax4.set_title("z2 Trajectory"), ax4.set_xlabel("Real"), ax4.set_ylabel("Imaginary")
        
    elif isinstance(first_elem, NeutrosophicNumber):
        gs = GridSpec(1, 2, figure=fig)
        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        a, b = [x.a for x in sequence], [x.b for x in sequence]
        ax1.plot(a, label='Determinate (a)'), ax1.plot(b, label='Indeterminate (b)'), ax1.set_title("Components"), ax1.legend()
        sc = ax2.scatter(a, b, c=range(len(a)), cmap='viridis')
        ax2.set_title("Trajectory"), ax2.set_xlabel("Determinate"), ax2.set_ylabel("Indeterminate"), fig.colorbar(sc, ax=ax2, label="Iteration")

    elif isinstance(first_elem, NeutrosophicComplexNumber):
        gs = GridSpec(2, 1, figure=fig)
        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        r, i, ind = [x.real for x in sequence], [x.imag for x in sequence], [x.indeterminacy for x in sequence]
        ax1.plot(r, label='Real'), ax1.plot(i, label='Imag'), ax1.plot(ind, label='Indeterminacy', linestyle=':')
        ax1.set_title("Components"), ax1.legend()
        sc = ax2.scatter(r, i, c=ind, cmap='magma', s=20)
        ax2.set_title("Trajectory (colored by Indeterminacy)"), ax2.set_xlabel("Real"), ax2.set_ylabel("Imaginary")
        fig.colorbar(sc, ax=ax2, label='Indeterminacy'), ax2.axis('equal')

    elif isinstance(first_elem, HyperrealNumber):
        gs = GridSpec(2, 1, figure=fig)
        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        num_components = min(len(first_elem.sequence), 4)
        for i in range(num_components):
            ax1.plot([h.sequence[i] for h in sequence], label=f'Comp {i}')
        ax1.set_title("Hyperreal Components"), ax1.legend()
        comp0, comp1 = [h.sequence[0] for h in sequence], [h.sequence[1] for h in sequence]
        sc = ax2.scatter(comp0, comp1, c=range(len(comp0)), cmap='plasma')
        ax2.set_title("Trajectory (C0 vs C1)"), ax2.set_xlabel("C0"), ax2.set_ylabel("C1"), fig.colorbar(sc, ax=ax2, label="Iteration")
        
    elif isinstance(first_elem, NeutrosophicBicomplexNumber):
        gs = GridSpec(2, 2, figure=fig)
        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        ax1.plot([n.real for n in sequence], [n.imag for n in sequence], '.-'), ax1.set_title("Primary Deterministic")
        ax2.plot([n.neut_real for n in sequence], [n.neut_imag for n in sequence], '.-'), ax2.set_title("Primary Neutrosophic")
        ax3.plot([n.j_real for n in sequence], [n.j_imag for n in sequence], '.-'), ax3.set_title("Secondary Deterministic")
        ax4.plot([n.j_neut_real for n in sequence], [n.j_neut_imag for n in sequence], '.-'), ax4.set_title("Secondary Neutrosophic")

    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f"Plotting for '{type(first_elem).__name__}' not implemented.", ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("  Keçeci Numbers Module - Demonstration")
    print("="*60)
    print("This script demonstrates the generation of various Keçeci Number types.")
    
    # --- Example 1: Interactive Mode ---
    # Uncomment the following lines to run in interactive mode:
    # seq, params = get_interactive()
    # if seq:
    #     plot_numbers(seq, title=f"Keçeci Type {params['type_choice']} Sequence")
    #     plt.show()

    # --- Example 2: Programmatic Generation and Plotting ---
    print("\nRunning programmatic tests for all 11 number types...")
    
    STEPS = 30
    START_VAL = "2.5"
    ADD_VAL = 3.0

    all_types = {
        "Positive Real": TYPE_POSITIVE_REAL, "Negative Real": TYPE_NEGATIVE_REAL,
        "Complex": TYPE_COMPLEX, "Float": TYPE_FLOAT, "Rational": TYPE_RATIONAL,
        "Quaternion": TYPE_QUATERNION, "Neutrosophic": TYPE_NEUTROSOPHIC,
        "Neutrosophic Complex": TYPE_NEUTROSOPHIC_COMPLEX, "Hyperreal": TYPE_HYPERREAL,
        "Bicomplex": TYPE_BICOMPLEX, "Neutrosophic Bicomplex": TYPE_NEUTROSOPHIC_BICOMPLEX
    }

    types_to_plot = [
        "Complex", "Quaternion", "Bicomplex", "Neutrosophic Complex", "Hyperreal"
    ]
    
    for name, type_id in all_types.items():
        start = "-5" if type_id == TYPE_NEGATIVE_REAL else "2+3j" if type_id in [TYPE_COMPLEX, TYPE_BICOMPLEX] else START_VAL
        
        seq = get_with_params(type_id, STEPS, start, ADD_VAL)
        
        if name in types_to_plot and seq:
            plot_numbers(seq, title=f"Demonstration: {name} Keçeci Numbers")

    print("\n\nDemonstration finished. Plots for selected types are shown.")
    plt.show()
