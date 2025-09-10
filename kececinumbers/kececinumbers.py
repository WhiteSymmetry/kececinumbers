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
- Generation of 16 types of Keçeci Numbers.
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
import numbers
#from numbers import Real
import numpy as np
import quaternion   # pip instal numpy-quaternion # conda install -c conda-forge quaternion
import random
import re
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
import sympy
from typing import Any, Dict, List, Optional, Tuple

# numpy ve diğer Keçeci tipleri için gerekli importlar
# Eğer bu kütüphaneler yüklü değilse, pip install numpy gibi komutlarla yüklemeniz gerekebilir.
try:
    import numpy as np
    # numpy.quaternion'ın eski adı np.quaternion. Yeni versiyonlarda quaternion modülü ayrı olabilir.
    # Eğer sorun yaşarsanız, 'pip install numpy-quaternion' ve 'import quaternion as np_quaternion' kullanabilirsiniz.
except ImportError:
    print("NumPy not found. Quaternion functionality might be limited.")
    class DummyQuaternion: # Placeholder for np.quaternion
        def __init__(self, w, x, y, z): self.w = w
        def __add__(self, other): return self
    np = type('module', (object,), {'quaternion': DummyQuaternion})



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
TYPE_OCTONION = 12
TYPE_SEDENION = 13
TYPE_CLIFFORD = 14
TYPE_DUAL = 15
TYPE_SPLIT_COMPLEX = 16



# ==============================================================================
# --- CUSTOM NUMBER CLASS DEFINITIONS ---
# ==============================================================================

@dataclass
class OctonionNumber:
    """
    Represents an octonion number with 8 components.
    Implements octonion multiplication rules.
    """
    def __init__(self, *args):
        # Varsayılan
        self.w = self.x = self.y = self.z = self.e = self.f = self.g = self.h = 0.0
        
        if len(args) == 8:
            self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = args
        elif len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                if len(args[0]) == 8:
                    self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = args[0]
                else:
                    components = list(args[0]) + [0.0] * (8 - len(args[0]))
                    self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = components
            elif isinstance(args[0], (int, float)):
                self.w = float(args[0])
        elif len(args) == 0:
            pass
        else:
            raise ValueError("Invalid arguments for OctonionNumber")

    @property
    def coeffs(self):
        """Octonion bileşenlerini liste olarak döner."""
        return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]

    def __add__(self, other):
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z,
                self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, OctonionNumber):
            # Mevcut octonion çarpımı (7-boyutlu çapraz çarpım kuralları)
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z \
                - self.e * other.e - self.f * other.f - self.g * other.g - self.h * other.h
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y \
                + self.e * other.f - self.f * other.e + self.g * other.h - self.h * other.g
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x \
                + self.e * other.g - self.g * other.e - self.f * other.h + self.h * other.f
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w \
                + self.e * other.h - self.h * other.e + self.f * other.g - self.g * other.f
            e = self.w * other.e - self.x * other.f - self.y * other.g - self.z * other.h \
                + self.e * other.w + self.f * other.x + self.g * other.y + self.h * other.z
            f = self.w * other.f + self.x * other.e - self.y * other.h + self.z * other.g \
                - self.e * other.x + self.f * other.w - self.g * other.z + self.h * other.y
            g = self.w * other.g + self.x * other.h + self.y * other.e - self.z * other.f \
                - self.e * other.y + self.f * other.z + self.g * other.w - self.h * other.x
            h = self.w * other.h - self.x * other.g + self.y * other.f + self.z * other.e \
                - self.e * other.z - self.f * other.y + self.g * other.x + self.h * other.w

            return OctonionNumber(w, x, y, z, e, f, g, h)

        elif isinstance(other, (int, float)):
            # Skaler çarpım: tüm bileşenler other ile çarpılır
            return OctonionNumber(
                self.w * other, self.x * other, self.y * other, self.z * other,
                self.e * other, self.f * other, self.g * other, self.h * other
            )

        return NotImplemented

    def __rmul__(self, other):
        # Skaler çarpımda değişme özelliği vardır: other * self == self * other
        if isinstance(other, (int, float)):
            return self.__mul__(other)  # self * other
        return NotImplemented


    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return OctonionNumber(
                self.w / scalar, self.x / scalar, self.y / scalar, self.z / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, OctonionNumber):
            return False
        tol = 1e-12
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol 
                   for attr in ['w', 'x', 'y', 'z', 'e', 'f', 'g', 'h'])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Octonion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.e:.3f}, {self.f:.3f}, {self.g:.3f}, {self.h:.3f})"

    def __repr__(self):
        return str(self)


@property
def coeffs(self):
    return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]

class Constants:
    """Oktonyon sabitleri."""
    ZERO = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
    ONE = OctonionNumber(1, 0, 0, 0, 0, 0, 0, 0)
    I = OctonionNumber(0, 1, 0, 0, 0, 0, 0, 0)
    J = OctonionNumber(0, 0, 1, 0, 0, 0, 0, 0)
    K = OctonionNumber(0, 0, 0, 1, 0, 0, 0, 0)
    E = OctonionNumber(0, 0, 0, 0, 1, 0, 0, 0)
    F = OctonionNumber(0, 0, 0, 0, 0, 1, 0, 0)
    G = OctonionNumber(0, 0, 0, 0, 0, 0, 1, 0)
    H = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 1)


@dataclass
class NeutrosophicNumber:
    """Represents a neutrosophic number of the form t + iI + fF."""
    t: float  # truth
    i: float  # indeterminacy
    f: float  # falsity

    def __init__(self, t: float, i: float, f: float = 0.0):
        self.t = t
        self.i = i
        self.f = f

    def __add__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.t + other.t, self.i + other.i, self.f + other.f)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t + other, self.i, self.f)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.t - other.t, self.i - other.i, self.f - other.f)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t - other, self.i, self.f)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(
                self.t * other.t,
                self.t * other.i + self.i * other.t + self.i * other.i,
                self.t * other.f + self.f * other.t + self.f * other.f
            )
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t * other, self.i * other, self.f * other)
        return NotImplemented

    def __truediv__(self, divisor: float) -> "NeutrosophicNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicNumber(self.t / divisor, self.i / divisor, self.f / divisor)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        parts = []
        if self.t != 0:
            parts.append(f"{self.t}")
        if self.i != 0:
            parts.append(f"{self.i}I")
        if self.f != 0:
            parts.append(f"{self.f}F")
        return " + ".join(parts) if parts else "0"

@dataclass
class NeutrosophicComplexNumber:
    """
    Represents a number with a complex part and an indeterminacy level.
    z = (a + bj) + cI, where I = indeterminacy.
    """

    def __init__(self, real: float = 0.0, imag: float = 0.0, indeterminacy: float = 0.0):
        self.real = float(real)
        self.imag = float(imag)
        self.indeterminacy = float(indeterminacy)

    def __repr__(self) -> str:
        return f"NeutrosophicComplexNumber(real={self.real}, imag={self.imag}, indeterminacy={self.indeterminacy})"

    def __str__(self) -> str:
        return f"({self.real}{self.imag:+}j) + {self.indeterminacy}I"

    def __add__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy + other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real + other, self.imag, self.indeterminacy)
        if isinstance(other, complex):
            return NeutrosophicComplexNumber(self.real + other.real, self.imag + other.imag, self.indeterminacy)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real - other, self.imag, self.indeterminacy)
        if isinstance(other, complex):
            return NeutrosophicComplexNumber(self.real - other.real, self.imag - other.imag, self.indeterminacy)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            # Indeterminacy: basitleştirilmiş model
            new_indeterminacy = (self.indeterminacy + other.indeterminacy +
                               self.magnitude_sq() * other.indeterminacy +
                               other.magnitude_sq() * self.indeterminacy)
            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        if isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real * other,
                self.imag * other,
                self.indeterminacy * other
            )
        return NotImplemented

    def __truediv__(self, divisor: Any) -> "NeutrosophicComplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicComplexNumber(
                self.real / divisor,
                self.imag / divisor,
                self.indeterminacy / divisor
            )
        return NotImplemented  # complex / NeutrosophicComplex desteklenmiyor

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                other - self.real,
                -self.imag,
                -self.indeterminacy
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def magnitude_sq(self) -> float:
        """Returns the squared magnitude of the complex part."""
        return self.real**2 + self.imag**2

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicComplexNumber):
            return (abs(self.real - other.real) < 1e-12 and
                    abs(self.imag - other.imag) < 1e-12 and
                    abs(self.indeterminacy - other.indeterminacy) < 1e-12)
        return False

@dataclass
class HyperrealNumber:
    """Represents a hyperreal number as a sequence of real numbers."""
    sequence: List[float]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.sequence = args[0]
        else:
            self.sequence = list(args)

    def __add__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            # Sequence'leri eşit uzunluğa getir
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a + b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] += other  # Sadece finite part'a ekle
            return HyperrealNumber(new_seq)
        return NotImplemented

    def __sub__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a - b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] -= other
            return HyperrealNumber(new_seq)
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

    def __mod__(self, divisor: float) -> "HyperrealNumber":
        if isinstance(divisor, (int, float)):
            return HyperrealNumber([x % divisor for x in self.sequence])
        raise TypeError("Modulo only supported with a scalar divisor.")

    def __str__(self) -> str:
        if len(self.sequence) <= 5:
            return f"Hyperreal{self.sequence}"
        return f"Hyperreal({self.sequence[:3]}...)" 

    @property
    def finite(self):
        """Returns the finite part (first component)"""
        return self.sequence[0] if self.sequence else 0.0

    @property
    def infinitesimal(self):
        """Returns the first infinitesimal part (second component)"""
        return self.sequence[1] if len(self.sequence) > 1 else 0.0

@dataclass
class BicomplexNumber:
    """Represents a bicomplex number with two complex components."""
    z1: complex  # First complex component
    z2: complex  # Second complex component

    def __add__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 + other, self.z2)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'BicomplexNumber' and '{type(other).__name__}'")

    def __sub__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 - other, self.z2)
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'BicomplexNumber' and '{type(other).__name__}'")

    def __mul__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            # Bicomplex multiplication: (z1 + z2j) * (w1 + w2j) = (z1*w1 - z2*w2) + (z1*w2 + z2*w1)j
            return BicomplexNumber(
                self.z1 * other.z1 - self.z2 * other.z2,
                self.z1 * other.z2 + self.z2 * other.z1
            )
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 * other, self.z2 * other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'BicomplexNumber' and '{type(other).__name__}'")

    def __truediv__(self, divisor: float) -> "BicomplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Division by zero")
            return BicomplexNumber(self.z1 / divisor, self.z2 / divisor)
        else:
            raise TypeError("Only scalar division is supported")

    def __str__(self) -> str:
        parts = []
        if self.z1 != 0j:
            parts.append(f"({self.z1.real}+{self.z1.imag}j)")
        if self.z2 != 0j:
            parts.append(f"({self.z2.real}+{self.z2.imag}j)j")
        return " + ".join(parts) if parts else "0"

    def __abs__(self) -> float:
        """Returns the magnitude of the bicomplex number."""
        return abs(self.z1) + abs(self.z2)  # Basit magnitude

def _parse_bicomplex(s: str) -> BicomplexNumber:
    """Parses a string into a BicomplexNumber."""
    s_clean = s.strip().replace(" ", "")
    
    # VİRGÜL formatı: z1_real,z1_imag,z2_real,z2_imag (4 parametre)
    if ',' in s_clean:
        parts = [float(p.strip()) for p in s_clean.split(',')]
        if len(parts) == 4:
            return BicomplexNumber(complex(parts[0], parts[1]), complex(parts[2], parts[3]))
        elif len(parts) == 2:
            # Sadece z1 verilmiş, z2 = 0
            return BicomplexNumber(complex(parts[0], parts[1]), complex(0, 0))
        elif len(parts) == 1:
            # Sadece real kısım
            return BicomplexNumber(complex(parts[0], 0), complex(0, 0))
    
    # Eğer virgül yoksa, complex olarak parsing dene
    try:
        # Önce complex olarak parse et
        c = _parse_complex(s_clean)
        return BicomplexNumber(c, complex(0, 0))
    except ValueError:
        # Son çare: sadece real kısım
        try:
            real_val = float(_extract_numeric_part(s_clean))
            return BicomplexNumber(complex(real_val, 0), complex(0, 0))
        except ValueError:
            return BicomplexNumber(complex(0, 0), complex(0, 0))  # Default

def _has_bicomplex_format(s: str) -> bool:
    """Checks if string has bicomplex format (comma-separated)."""
    return ',' in s and s.count(',') in [1, 3]  # 2 or 4 components

@dataclass
class NeutrosophicBicomplexNumber:
    def __init__(self, a, b, c, d, e, f, g, h):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)
        self.f = float(f)
        self.g = float(g)
        self.h = float(h)

    def __repr__(self):
        return f"NeutrosophicBicomplexNumber({self.a}, {self.b}, {self.c}, {self.d}, {self.e}, {self.f}, {self.g}, {self.h})"

    def __str__(self):
        return f"({self.a} + {self.b}i) + ({self.c} + {self.d}i)I + ({self.e} + {self.f}i)j + ({self.g} + {self.h}i)Ij"

    def __add__(self, other):
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(
                self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        return NotImplemented

    def __mul__(self, other):
        # Basitleştirilmiş çarpım (tam bicomplex kuralı karmaşık)
        if isinstance(other, (int, float)):
            return NeutrosophicBicomplexNumber(
                *(other * x for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h])
            )
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero")
            return NeutrosophicBicomplexNumber(
                self.a / scalar, self.b / scalar, self.c / scalar, self.d / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented

    def __eq__(self, other):
        """Equality with tolerance for float comparison."""
        if not isinstance(other, NeutrosophicBicomplexNumber):
            return False
        tol = 1e-12
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol 
                   for attr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    def __ne__(self, other):
        return not self.__eq__(other)

@dataclass
class SedenionNumber:
    def __init__(self, coeffs):
        if len(coeffs) != 16:
            raise ValueError("Sedenion must have 16 components")
        self.coeffs = list(map(float, coeffs))

    def __add__(self, other):
        if isinstance(other, SedenionNumber):
            return SedenionNumber([s + o for s, o in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return SedenionNumber([self.coeffs[0] + other] + self.coeffs[1:])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SedenionNumber):
            return SedenionNumber([s - o for s, o in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return SedenionNumber([self.coeffs[0] - other] + self.coeffs[1:])
        return NotImplemented

    def __mul__(self, other):
        # Sedenion çarpımı Cayley-Dickson yapısı ile tanımlanır ama oldukça karmaşıktır.
        # Basitleştirme: skaler çarpım veya element-wise çarpım DEĞİL.
        # Gerçek sedenion çarpımı 16x16 çarpım tablosu gerektirir.
        # Bu örnekte, yalnızca skaler çarpım ve Sedenion-Sedenion için çarpım tablosuz basitleştirilmiş hâli (örnek amaçlı) verilir.
        # Gerçek uygulama için sedenion multiplication table kullanılmalıdır.
        if isinstance(other, (int, float)):
            return SedenionNumber([c * other for c in self.coeffs])
        elif isinstance(other, SedenionNumber):
            # NOT: Gerçek sedenion çarpımı burada eksik (çok karmaşık). 
            # Element-wise çarpım matematiksel olarak doğru değildir ama örnek olsun diye konuldu.
            # Gerçek hâli için: https://en.wikipedia.org/wiki/Sedenion
            return NotImplemented  # Gerçek sedenion çarpımı oldukça karmaşıktır
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return SedenionNumber([c / other for c in self.coeffs])
        return NotImplemented

    def __str__(self):
        return "(" + ", ".join(f"{c:.2f}" for c in self.coeffs) + ")"

    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"

@dataclass
class CliffordNumber:

    def __init__(self, basis_dict):
        """
        CliffordNumber constructor - sadece basis_dict alır.
        Dimension property olarak hesaplanabilir.
        """
        self.basis = {k: float(v) for k, v in basis_dict.items()}
    
    @property
    def dimension(self):
        """Vector space dimension'ını otomatik hesaplar."""
        if not self.basis:
            return 0
        max_index = 0
        for key in self.basis.keys():
            if key:  # scalar değilse
                indices = [int(c) for c in key if c.isdigit()]
                if indices:
                    max_index = max(max_index, max(indices))
        return max_index

    def __add__(self, other):
        if isinstance(other, CliffordNumber):
            new_basis = self.basis.copy()
            for k, v in other.basis.items():
                new_basis[k] = new_basis.get(k, 0.0) + v
            return CliffordNumber(new_basis)
        elif isinstance(other, (int, float)):
            new_basis = self.basis.copy()
            new_basis[''] = new_basis.get('', 0.0) + other
            return CliffordNumber(new_basis)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, CliffordNumber):
            new_basis = self.basis.copy()
            for k, v in other.basis.items():
                new_basis[k] = new_basis.get(k, 0.0) - v
            return CliffordNumber(new_basis)
        elif isinstance(other, (int, float)):
            new_basis = self.basis.copy()
            new_basis[''] = new_basis.get('', 0.0) - other
            return CliffordNumber(new_basis)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return CliffordNumber({k: v * other for k, v in self.basis.items()})
        elif isinstance(other, CliffordNumber):
            # Clifford çarpımı: e_i * e_i = +1 veya -1 olabilir ama burada basitleştirme
            # Genel durum için çarpım kuralları tanımlanmalı (örneğin e1^2 = 1, e2^2 = -1 vs)
            # Burada sadece cebirsel çarpım desteklenmiyor çünkü kurallar eksik.
            return NotImplemented  # Gerçek Clifford çarpımı için kurallar gerekir
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return CliffordNumber({k: v / other for k, v in self.basis.items()})
        return NotImplemented

    def __str__(self):
        parts = []
        if '' in self.basis and self.basis[''] != 0.0:
            parts.append(f"{self.basis['']:.2f}")
        
        sorted_keys = sorted([k for k in self.basis if k != ''], key=lambda x: (len(x), x))
        for k in sorted_keys:
            v = self.basis[k]
            if abs(v) > 1e-10:  # sıfıra çok yakın olanları atla
                sign = '+' if v > 0 and parts else ''
                parts.append(f"{sign}{v:.2f}e{k}")
        
        result = "".join(parts).replace("+-", "-")
        return result if result else "0.0"

    def __repr__(self):
        parts = []
        if '' in self.basis and self.basis[''] != 0.0:
            parts.append(str(self.basis['']))
        
        sorted_keys = sorted([k for k in self.basis if k != ''], key=lambda x: (len(x), x))
        for k in sorted_keys:
            v = self.basis[k]
            if v != 0.0:
                sign = '+' if v > 0 and parts else ''
                parts.append(f"{sign}{v}e{k}")
        
        if not parts:
            return "0.0"
        return "".join(parts).replace("+-", "-")


@dataclass
class DualNumber:

    real: float
    dual: float

    def __init__(self, real, dual):
        self.real = float(real)
        self.dual = float(dual)
    def __add__(self, other):
        if isinstance(other, DualNumber): return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int, float)): return DualNumber(self.real + other, self.dual)
        raise TypeError
    def __sub__(self, other):
        if isinstance(other, DualNumber): return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int, float)): return DualNumber(self.real - other, self.dual)
        raise TypeError
    def __mul__(self, other):
        if isinstance(other, DualNumber): return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        elif isinstance(other, (int, float)): return DualNumber(self.real * other, self.dual * other)
        raise TypeError
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError
            return DualNumber(self.real / other, self.dual / other)
        elif isinstance(other, DualNumber):
            if other.real == 0: raise ZeroDivisionError
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
        raise TypeError
    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError
            return DualNumber(self.real // other, self.dual // other)
        raise TypeError
    def __eq__(self, other):
        if isinstance(other, DualNumber): return self.real == other.real and self.dual == other.dual
        elif isinstance(other, (int, float)): return self.real == other and self.dual == 0
        return False
    def __str__(self): return f"{self.real} + {self.dual}ε"
    def __repr__(self): return self.__str__() # __repr__ eklenmiş
    def __int__(self): return int(self.real) # int() dönüşümü eklenmiş
    def __radd__(self, other):
       return self.__add__(other)  # commutative
    def __rsub__(self, other):
       if isinstance(other, (int, float)):
           return DualNumber(other - self.real, -self.dual)
       return NotImplemented

    def __neg__(self):
       return DualNumber(-self.real, -self.dual)

    def __hash__(self):
       return hash((self.real, self.dual))


@dataclass
class SplitcomplexNumber:
    def __init__(self, real, split):
        self.real = float(real)
        self.split = float(split)

    def __add__(self, other):
        if isinstance(other, SplitcomplexNumber):
            return SplitcomplexNumber(self.real + other.real, self.split + other.split)
        elif isinstance(other, (int, float)):
            return SplitcomplexNumber(self.real + other, self.split)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SplitcomplexNumber):
            return SplitcomplexNumber(self.real - other.real, self.split - other.split)
        elif isinstance(other, (int, float)):
            return SplitcomplexNumber(self.real - other, self.split)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, SplitcomplexNumber):
            # (a + bj) * (c + dj) = (ac + bd) + (ad + bc)j, çünkü j² = +1
            real = self.real * other.real + self.split * other.split
            split = self.real * other.split + self.split * other.real
            return SplitcomplexNumber(real, split)
        elif isinstance(other, (int, float)):
            return SplitcomplexNumber(self.real * other, self.split * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return SplitcomplexNumber(self.real / other, self.split / other)
        elif isinstance(other, SplitcomplexNumber):
            # (a + bj) / (c + dj) = ?
            # Payda: (c + dj)(c - dj) = c² - d² (çünkü j² = 1)
            # Yani bölme yalnızca c² ≠ d² ise tanımlıdır.
            a, b = self.real, self.split
            c, d = other.real, other.split
            norm = c * c - d * d
            if abs(norm) < 1e-10:
                raise ZeroDivisionError("Split-complex division by zero (null divisor)")
            real = (a * c - b * d) / norm
            split = (b * c - a * d) / norm
            return SplitcomplexNumber(real, split)
        return NotImplemented

    def __str__(self):
        return f"{self.real:.2f} + {self.split:.2f}j'"

    def __repr__(self):
        return f"({self.real}, {self.split}j')"


# Yardımcı fonksiyonlar
def _extract_numeric_part(s: str) -> str:
    """Bir string'den sadece sayısal kısmı ayıklar.
    Bilimsel gösterimi ve karmaşık formatları destekler.
    """
    s = s.strip()
    
    # Bilimsel gösterim için pattern
    scientific_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    match = re.match(scientific_pattern, s)
    
    if match:
        return match.group(0)
    
    # Eğer bilimsel gösterim bulunamazsa, basit sayı ara
    simple_match = re.search(r"[-+]?\d*\.?\d+", s)
    if simple_match:
        return simple_match.group(0)
    
    # Hiç sayı bulunamazsa orijinal string'i döndür
    return s

def _parse_complex(s: str) -> complex:
    """Bir string'i complex sayıya dönüştürür.
    "real,imag", "real+imag(i/j)", "real", "imag(i/j)" formatlarını destekler.
    """
    s = s.strip().replace('J', 'j').replace('i', 'j') # Hem J hem i yerine j kullan

    # 1. Eğer "real,imag" formatındaysa
    if ',' in s:
        parts = s.split(',')
        if len(parts) == 2:
            try:
                return complex(float(parts[0]), float(parts[1]))
            except ValueError:
                pass # Devam et

    # 2. Python'ın kendi complex() dönüştürücüsünü kullanmayı dene (örn: "1+2j", "3j", "-5")
    try:
        return complex(s)
    except ValueError:
        # 3. Sadece real kısmı varsa (örn: "5")
        try:
            return complex(float(s), 0)
        except ValueError:
            # 4. Sadece sanal kısmı varsa (örn: "2j", "j")
            if s.endswith('j'):
                try:
                    imag_val = float(s[:-1]) if s[:-1] else 1.0 # "j" -> 1.0j
                    return complex(0, imag_val)
                except ValueError:
                    pass
            
            raise ValueError(f"Geçersiz kompleks sayı formatı: '{s}'")

def convert_to_float(value):
    """Convert various Keçeci number types to a float or raise an error if not possible."""
    if isinstance(value, (int, float)):
        return float(value)
    elif hasattr(value, 'real') and hasattr(value, 'imag'):
        return value.real  # For complex-like types
    elif hasattr(value, 'w'):  # For quaternions or similar
        return value.w  # Assuming w is the real part
    elif hasattr(value, 'coeffs') and isinstance(value.coeffs, list):  # For Sedenion or similar
        return value.coeffs[0]  # Assuming first coefficient is the real part
    elif hasattr(value, 'real_part'):  # For DualNumber or similar
        return value.real_part  # Assuming this is how you access the real part
    elif hasattr(value, 'value'):  # For CliffordNumber or similar
        return value.value  # Assuming this returns the appropriate value
    else:
        raise TypeError(f"Cannot convert {type(value).__name__} to float.")

def safe_add(added_value, ask_unit, direction):
    """
    Adds ±ask_unit to added_value using native algebraic operations.

    This function performs: `added_value + (ask_unit * direction)`
    It assumes that both operands support algebraic addition and scalar multiplication.

    Parameters
    ----------
    added_value : Any
        The base value (e.g., DualNumber, OctonionNumber, CliffordNumber).
    ask_unit : Same type as added_value
        The unit increment to add or subtract.
    direction : int
        Either +1 or -1, determining the sign of the increment.

    Returns
    -------
    Same type as added_value
        Result of `added_value + (ask_unit * direction)`.

    Raises
    ------
    TypeError
        If `ask_unit` does not support multiplication by an int,
        or if `added_value` does not support addition with `ask_unit`.
    """
    try:
        # Scale the unit: ask_unit * (+1 or -1)
        if not hasattr(ask_unit, '__mul__'):
            raise TypeError(f"Type '{type(ask_unit).__name__}' does not support scalar multiplication (missing __mul__).")
        scaled_unit = ask_unit * direction

        # Add to the current value
        if not hasattr(added_value, '__add__'):
            raise TypeError(f"Type '{type(added_value).__name__}' does not support addition (missing __add__).")
        result = added_value + scaled_unit

        return result

    except Exception as e:
        # Daha açıklayıcı hata mesajı
        msg = f"safe_add failed: Cannot compute {repr(added_value)} + ({direction} * {repr(ask_unit)})"
        raise TypeError(f"{msg} → {type(e).__name__}: {e}") from e


def _parse_neutrosophic(s: str) -> Tuple[float, float, float]:
    """Parses neutrosophic string into (t, i, f) tuple."""
    s_clean = s.strip().replace(" ", "").upper()
    
    # VİRGÜL formatı: t,i,f (3 parametre)
    if ',' in s_clean:
        parts = s_clean.split(',')
        try:
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]), float(parts[1]), 0.0
            elif len(parts) == 1:
                return float(parts[0]), 0.0, 0.0
        except ValueError:
            pass

    # Eski formatları destekle
    try:
        if 'I' in s_clean or 'F' in s_clean:
            # Basit parsing
            t_part = s_clean
            i_val, f_val = 0.0, 0.0
            
            if 'I' in s_clean:
                parts = s_clean.split('I')
                t_part = parts[0]
                i_val = float(parts[1]) if parts[1] and parts[1] not in ['', '+', '-'] else 1.0
            
            if 'F' in t_part:
                parts = t_part.split('F')
                t_val = float(parts[0]) if parts[0] and parts[0] not in ['', '+', '-'] else 0.0
                f_val = float(parts[1]) if len(parts) > 1 and parts[1] not in ['', '+', '-'] else 1.0
            else:
                t_val = float(t_part) if t_part not in ['', '+', '-'] else 0.0
            
            return t_val, i_val, f_val
        else:
            # Sadece sayısal değer
            return float(s_clean), 0.0, 0.0
    except ValueError:
        return 0.0, 0.0, 0.0  # Default

def _parse_hyperreal(s: str) -> Tuple[float, float]:
    """Parses hyperreal string into (finite, infinitesimal) tuple."""
    s_clean = s.strip().replace(" ", "")
    
    # VİRGÜL formatı: finite,infinitesimal
    if ',' in s_clean:
        parts = s_clean.split(',')
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        elif len(parts) == 1:
            try:
                return float(parts[0]), 0.0
            except ValueError:
                pass

    # Eski 'a+be' formatını destekle
    if 'e' in s_clean:
        try:
            parts = s_clean.split('e')
            finite = float(parts[0]) if parts[0] not in ['', '+', '-'] else 0.0
            infinitesimal = float(parts[1]) if len(parts) > 1 and parts[1] not in ['', '+', '-'] else 1.0
            return finite, infinitesimal
        except ValueError:
            pass
    
    # Sadece sayısal değer
    try:
        return float(s_clean), 0.0
    except ValueError:
        return 0.0, 0.0  # Default

def _parse_quaternion_from_csv(s: str) -> np.quaternion:
    """Virgülle ayrılmış string'i Quaternion'a dönüştürür.
    "w,x,y,z" veya sadece "w" (skaler) formatlarını destekler.
    """
    s = s.strip()
    parts_str = s.split(',')
    
    # Tüm parçaları float'a dönüştürmeyi dene
    try:
        parts_float = [float(p.strip()) for p in parts_str]
    except ValueError:
        raise ValueError(f"Quaternion bileşenleri sayı olmalı: '{s}'")

    if len(parts_float) == 4:
        return np.quaternion(*parts_float)
    elif len(parts_float) == 1: # Sadece skaler değer
        return np.quaternion(parts_float[0], 0, 0, 0)
    else:
        raise ValueError(f"Geçersiz quaternion formatı. 1 veya 4 bileşen bekleniyor: '{s}'")


def _has_comma_format(s: str) -> bool:
    """String'in virgül içerip içermediğini kontrol eder."""
    return ',' in s

def _parse_neutrosophic_bicomplex(s: str) -> NeutrosophicBicomplexNumber:
    """
    Parses string like "a,b,c,d,e,f,g,h" into 8 components.
    """
    try:
        parts = s.split(',')
        if len(parts) != 8:
            raise ValueError(f"Expected 8 components, got {len(parts)}")
        values = [float(part.strip()) for part in parts]
        return NeutrosophicBicomplexNumber(*values)
    except Exception as e:
        raise ValueError(f"Invalid NeutrosophicBicomplex format: '{s}' → {e}")

def _parse_octonion(s: str) -> OctonionNumber:
    """'w,x,y,z,e,f,g,h' formatındaki stringi OctonionNumber'a çevirir."""
    s_clean = s.strip()
    
    # Eğer virgül içermiyorsa, skaler olarak kabul et
    if ',' not in s_clean:
        try:
            scalar = float(s_clean)
            return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
        except ValueError:
            raise ValueError(f"Invalid octonion format: '{s}'")
    
    # Virgülle ayrılmışsa
    try:
        parts = [float(p.strip()) for p in s_clean.split(',')]
        if len(parts) == 8:
            return OctonionNumber(*parts)  # 8 parametre olarak gönder
        else:
            # Eksik veya fazla bileşen için default
            scalar = parts[0] if parts else 0.0
            return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
    except ValueError as e:
        raise ValueError(f"Invalid octonion format: '{s}'") from e


def _parse_sedenion(s: str) -> SedenionNumber:
    """String'i SedenionNumber'a dönüştürür."""
    s = s.strip()
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 16:
        try:
            return SedenionNumber(list(map(float, parts)))
        except ValueError as e:
            raise ValueError(f"Geçersiz sedenion bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1: # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return SedenionNumber([scalar_val] + [0.0] * 15)
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler sedenion değeri: '{s}' -> {e}") from e
            
    raise ValueError(f"Sedenion için 16 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_clifford(s: str) -> CliffordNumber:
    """Algebraik string'i CliffordNumber'a dönüştürür (ör: '1.0+2.0e1')."""
    s = s.strip().replace(' ', '')
    basis_dict = {'': 0.0}  # Skaler kısım
    
    terms_pattern = re.compile(r"([+-]?)(\d*\.?\d+(?:[eE][+-]?\d+)?)?(e\d*)?")
    remaining_s = s
    
    while remaining_s:
        match = terms_pattern.match(remaining_s)
        if not match or (match.group(2) is None and match.group(3) is None):
            if remaining_s.startswith('+') or remaining_s.startswith('-'):
                current_sign = remaining_s[0]
                remaining_s = remaining_s[1:]
                if not remaining_s:
                    coeff = 1.0 if current_sign == '+' else -1.0
                    basis_dict[''] = basis_dict.get('', 0.0) + coeff
                continue
            raise ValueError(f"Geçersiz Clifford terim formatı: '{remaining_s}'")

        sign_str, coeff_str, basis_str_with_e = match.groups()
        current_sign = 1.0
        if sign_str == '-': current_sign = -1.0
        
        coeff = 1.0
        if coeff_str: coeff = float(coeff_str)
        elif not sign_str: coeff = 1.0
        elif sign_str and not coeff_str: coeff = 1.0

        coeff *= current_sign

        if basis_str_with_e:
            basis_key = basis_str_with_e.lstrip('e')
            basis_dict[basis_key] = basis_dict.get(basis_key, 0.0) + coeff
        else:
            basis_dict[''] = basis_dict.get('', 0.0) + coeff
        
        remaining_s = remaining_s[match.end():]

    return CliffordNumber(basis_dict)

def _parse_dual(s: str) -> DualNumber:
    """Virgülle ayrılmış string'i DualNumber'a dönüştürür."""
    s = s.strip()
    parts = [p.strip() for p in s.split(',')]
    
    # Sadece ilk iki bileşeni al
    if len(parts) >= 2:
        try:
            return DualNumber(float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    elif len(parts) == 1: # Sadece real kısım verilmiş
        try:
            return DualNumber(float(parts[0]), 0.0)
        except ValueError:
            pass
            
    raise ValueError(f"Geçersiz Dual sayı formatı: '{s}' (Real, Dual veya sadece Real bekleniyor)")

def _parse_splitcomplex(s: str) -> SplitcomplexNumber:
    """Virgülle ayrılmış string'i SplitcomplexNumber'a dönüştürür."""
    s = s.strip()
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 2:
        try:
            return SplitcomplexNumber(float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    elif len(parts) == 1: # Sadece real kısım verilmiş
        try:
            return SplitcomplexNumber(float(parts[0]), 0.0)
        except ValueError:
            pass
            
    raise ValueError(f"Geçersiz Split-Complex sayı formatı: '{s}' (Real, Split veya sadece Real bekleniyor)")


class Constants:
    """Oktonyon sabitleri."""
    ZERO = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
    ONE = OctonionNumber(1, 0, 0, 0, 0, 0, 0, 0)
    I = OctonionNumber(0, 1, 0, 0, 0, 0, 0, 0)
    J = OctonionNumber(0, 0, 1, 0, 0, 0, 0, 0)
    K = OctonionNumber(0, 0, 0, 1, 0, 0, 0, 0)
    E = OctonionNumber(0, 0, 0, 0, 1, 0, 0, 0)
    F = OctonionNumber(0, 0, 0, 0, 0, 1, 0, 0)
    G = OctonionNumber(0, 0, 0, 0, 0, 0, 1, 0)
    H = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 1)



def generate_octonion(w, x, y, z, e, f, g, h):
    """8 bileşenden bir oktonyon oluşturur."""
    return OctonionNumber(w, x, y, z, e, f, g, h)

# Bazı önemli oktonyon sabitleri
ZERO = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
ONE = OctonionNumber(1, 0, 0, 0, 0, 0, 0, 0)
I = OctonionNumber(0, 1, 0, 0, 0, 0, 0, 0)
J = OctonionNumber(0, 0, 1, 0, 0, 0, 0, 0)
K = OctonionNumber(0, 0, 0, 1, 0, 0, 0, 0)
E = OctonionNumber(0, 0, 0, 0, 1, 0, 0, 0)
F = OctonionNumber(0, 0, 0, 0, 0, 1, 0, 0)
G = OctonionNumber(0, 0, 0, 0, 0, 0, 1, 0)
H = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 1)


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
    random_type_choice = random.randint(1, 16)
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutro-Complex", "Hyperreal", 
        "Bicomplex", "Neutro-Bicomplex", "Octonion", "Sedenion", "Clifford", "Dual", "Split-Complex",
    ]
    print(f"\nRandomly selected Keçeci Number Type: {random_type_choice} ({type_names_list[random_type_choice-1]})")
    
    return get_with_params(
        kececi_type_choice=random_type_choice, 
        iterations=num_iterations,
        start_value_raw=fixed_start_raw,
        add_value_raw=fixed_add_base_scalar
    )

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
    preview_count = min(len(sequence), 40)
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


def _is_divisible(value: Any, divisor: int, kececi_type: int) -> bool:
    TOLERANCE = 1e-12

    try:
        if kececi_type == TYPE_DUAL:
            if isinstance(value, DualNumber):
                return math.isclose(value.real % divisor, 0.0, abs_tol=TOLERANCE)
            return math.isclose(value % divisor, 0.0, abs_tol=TOLERANCE)
        elif kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            # Tam sayılar için, sadece int tipinde ve kalansız bölünebilir mi
            # float değerlerin int'e yuvarlanarak kontrol edilmesi gerekebilir
            if isinstance(value, (int, float)) and is_near_integer(value):
                return int(round(float(value))) % divisor == 0
            return False
        elif kececi_type == TYPE_FLOAT:
            mod = value % divisor
            return math.isclose(mod, 0.0, abs_tol=TOLERANCE)
        elif kececi_type == TYPE_RATIONAL:
            quotient = value / divisor
            return quotient.denominator == 1
        elif kececi_type == TYPE_COMPLEX:
            real_mod = value.real % divisor
            imag_mod = value.imag % divisor
            return (math.isclose(real_mod, 0.0, abs_tol=TOLERANCE) and
                    math.isclose(imag_mod, 0.0, abs_tol=TOLERANCE))
        elif kececi_type == TYPE_QUATERNION:
            components = [value.w, value.x, value.y, value.z]
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in components)
        elif kececi_type == TYPE_NEUTROSOPHIC:
            return (math.isclose(value.a % divisor, 0.0, abs_tol=TOLERANCE) and
                    math.isclose(value.b % divisor, 0.0, abs_tol=TOLERANCE))
        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            components = [value.real, value.imag, value.indeterminacy]
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in components)
        elif kececi_type == TYPE_HYPERREAL:
            return all(math.isclose(x % divisor, 0.0, abs_tol=TOLERANCE) for x in value.sequence)
        elif kececi_type == TYPE_BICOMPLEX:
            return (_is_divisible(value.z1, divisor, TYPE_COMPLEX) and
                    _is_divisible(value.z2, divisor, TYPE_COMPLEX))
        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            components = [
                value.a, value.b, value.c, value.d,
                value.e, value.f, value.g, value.h
            ]
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in components)
        elif kececi_type == TYPE_OCTONION:
            components = [
                value.w, value.x, value.y, value.z,
                value.e, value.f, value.g, value.h
            ]
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in components)
        elif kececi_type == TYPE_SEDENION:
            if hasattr(value, 'coeffs'):
                coeffs = value.coeffs
            else:
                coeffs = list(value)
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in coeffs)
        elif kececi_type == TYPE_CLIFFORD:
            if hasattr(value, 'basis') and isinstance(value.basis, dict):
                components = value.basis.values()
            else:
                components = []
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in components)
        elif kececi_type == TYPE_SPLIT_COMPLEX:
            real_mod = value.real % divisor
            split_mod = value.split % divisor
            return (math.isclose(real_mod, 0.0, abs_tol=TOLERANCE) and
                    math.isclose(split_mod, 0.0, abs_tol=TOLERANCE))
        else:
            return False
    except (TypeError, AttributeError, ValueError, ZeroDivisionError):
        return False

def _get_integer_representation(n_input: Any) -> Optional[int]:
    """Extracts the primary integer component from any supported number type."""
    try:
        if isinstance(n_input, (int, float, Fraction)):
            # is_near_integer kontrolü eklendi, çünkü float'tan int'e direkt dönüş hassasiyet sorunları yaratabilir
            if is_near_integer(n_input):
                return abs(int(round(float(n_input))))
            return None # Tam sayıya yakın değilse None döndür
        
        if isinstance(n_input, complex):
            if is_near_integer(n_input.real):
                return abs(int(round(n_input.real)))
            return None
        
        # NumPy Quaternion özel kontrolü
        if hasattr(np, 'quaternion') and isinstance(n_input, np.quaternion):
            if is_near_integer(n_input.w):
                return abs(int(round(n_input.w)))
            return None
        
        # Keçeci Sayı Tipleri
        if isinstance(n_input, NeutrosophicNumber):
            if is_near_integer(n_input.t):
                return abs(int(round(n_input.t)))
            return None
        
        if isinstance(n_input, NeutrosophicComplexNumber):
            if is_near_integer(n_input.real):
                return abs(int(round(n_input.real)))
            return None
        
        if isinstance(n_input, HyperrealNumber):
            if n_input.sequence and is_near_integer(n_input.sequence[0]):
                return abs(int(round(n_input.sequence[0])))
            return None
        
        if isinstance(n_input, BicomplexNumber):
            if is_near_integer(n_input.z1.real):
                return abs(int(round(n_input.z1.real)))
            return None
        
        if isinstance(n_input, NeutrosophicBicomplexNumber):
            if is_near_integer(n_input.real): # Varsayım: .real metodu var
                return abs(int(round(n_input.real)))
            return None
        
        if isinstance(n_input, OctonionNumber):
            if is_near_integer(n_input.w):
                return abs(int(round(n_input.w)))
            return None
        
        if isinstance(n_input, SedenionNumber):
            if n_input.coefficients and is_near_integer(n_input.coefficients[0]):
                return abs(int(round(n_input.coefficients[0])))
            return None
        
        if isinstance(n_input, CliffordNumber):
            scalar_part = n_input.basis.get('', 0)
            if is_near_integer(scalar_part):
                return abs(int(round(scalar_part)))
            return None
        
        if isinstance(n_input, DualNumber):
            if is_near_integer(n_input.real):
                return abs(int(round(n_input.real)))
            return None
        
        if isinstance(n_input, SplitcomplexNumber):
            if is_near_integer(n_input.real):
                return abs(int(round(n_input.real)))
            return None
        
        # Fallback (Bu satırın çoğu durumda ulaşılmaması beklenir)
        if is_near_integer(n_input):
            return abs(int(round(float(n_input))))
        return None
        
    except (ValueError, TypeError, IndexError, AttributeError):
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


def is_near_integer(x, tol=1e-12):
    """
    Checks if a number (or its real part) is close to an integer.
    Useful for float-based primality and divisibility checks.
    """
    try:
        if isinstance(x, complex):
            # Sadece gerçek kısım önemli, imajiner sıfıra yakın olmalı
            if abs(x.imag) > tol:
                return False
            x = x.real
        elif isinstance(x, (list, tuple)):
            return False  # Desteklenmeyen tip

        # Genel durum: float veya int
        x = float(x)
        return abs(x - round(x)) < tol
    except:
        return False

def is_prime_like(value, kececi_type: int) -> bool:
    """
    Checks if the value should be treated as a "prime" in Keçeci logic,
    by examining its principal (scalar/real) component.
    """
    try:
        if kececi_type == TYPE_DUAL:
            if isinstance(value, DualNumber):
                return is_prime(int(value.real))
            else:
                return is_prime(int(value))
        elif kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL, TYPE_FLOAT]:
            if is_near_integer(value):
                n = int(round(float(value)))
                return is_prime(n)
            return False
        elif kececi_type == TYPE_COMPLEX:
            if is_near_integer(value.real):
                return is_prime(value.real)
            return False
        elif kececi_type == TYPE_SPLIT_COMPLEX:
            if is_near_integer(value.real):
                return is_prime(value.real)
            return False
        elif kececi_type in [TYPE_QUATERNION, TYPE_OCTONION, TYPE_SEDENION]:
            scalar = getattr(value, 'w', None) # Quaternion için .w, diğerleri için de varsa
            if scalar is None and hasattr(value, 'coeffs') and value.coeffs: # Sedenion için
                scalar = value.coeffs[0]
            if scalar is None and hasattr(value, 'coefficients') and value.coefficients: # Alternatif Sedenion için
                scalar = value.coefficients[0]
            if scalar is not None and is_near_integer(scalar):
                return is_prime(scalar)
            return False
        elif kececi_type == TYPE_CLIFFORD:
            scalar = value.basis.get('', 0.0)
            if is_near_integer(scalar):
                return is_prime(scalar)
            return False
        else: # Diğer tipler için genel kural, önceki else bloğu yerine
            main_part = getattr(value, 'real', None)
            if main_part is None:
                main_part = getattr(value, 'a', None) # Neutrosophic için
            if main_part is None:
                main_part = getattr(value, 'w', None) # Diğer bazı tipler için
            if main_part is not None and is_near_integer(main_part):
                return is_prime(main_part)
            return False
    except Exception as e: # Geniş yakalama yerine spesifik yakalama önerilir
        # print(f"Error in is_prime_like: {e}") # Hata ayıklama için
        return False

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

def analyze_all_types(iterations=120, additional_params=None):
    """
    Performs automated analysis on all Keçeci number types.
    Args:
        iterations (int): Number of Keçeci steps to generate for each sequence.
        additional_params (list): List of tuples for additional parameter sets.
    Returns:
        tuple: (sorted_by_zeta, sorted_by_gue) - Lists of results sorted by performance.
    """
    
    from . import (
        NeutrosophicNumber,
        NeutrosophicComplexNumber,
        HyperrealNumber,
        BicomplexNumber,
        NeutrosophicBicomplexNumber,
        OctonionNumber,
        SedenionNumber,
        CliffordNumber,
        DualNumber,
        SplitcomplexNumber,
        analyze_all_types,
        get_with_params,
        _find_kececi_zeta_zeros,
        _compute_gue_similarity,
        _plot_comparison,
    )
    
    print("Automated Analysis for Keçeci Types")
    print("=" * 80)

    include_intermediate = True
    results = []

    # Default parameter sets
    # Parameter sets to test
    param_sets = [
        # 1. Positive Real - SADECE sayısal değerler
        ('5.0', '2.0'),
        ('10', '3.0'),
        ('0.0', '1.5'),
        ('3.14', '0.5'),
        
        # 2. Negative Real - SADECE sayısal değerler
        ('-3.0', '-1.0'),
        ('-5.0', '-2.0'),
        ('-1.5', '-0.5'),
        ('-2.718', '-1.0'),
        
        # 3. Complex - DOĞRU formatında
        ('1+2j', '0.5+0.5j'),
        ('3-4j', '1-2j'),
        ('0.0+0.0j', '1.0+2.0j'),
        ('-1.5+2.5j', '0.2-0.3j'),
        
        # Diğer complex örnekleri
        ('2j', '1j'),               # Sadece imaginary
        ('-3j', '-2j'),             # Negatif imaginary
        ('1.5', '0.5'),             # Sadece real (otomatik complex olur)
        ('0+1j', '0+0.5j'),         # Explicit format
        
        # 4. Float - SADECE sayısal değerler
        ('0.0001412', '0.037'),
        ('3.14159', '0.01'),
        ('2.71828', '0.1'),
        ('0.0', '0.001'),
        
        # 5. Rational - SADECE sayısal/rational değerler
        ('1/2', '1/4'),
        ('3/4', '1/8'),
        ('0', '2/3'),
        ('5/6', '1/12'),
        
        # 6. Quaternion - VİRGÜL formatında
        ('1.0,0.0,0.0,0.0', '0.1,0.0,0.0,0.0'),      # Scalar addition
        ('0.0,1.0,0.0,0.0', '0.0,0.1,0.0,0.0'),      # i component
        ('0.0,0.0,1.0,0.0', '0.0,0.0,0.1,0.0'),      # j component
        ('0.0,0.0,0.0,1.0', '0.0,0.0,0.0,0.1'),      # k component
        ('0.5,0.5,0.5,0.5', '0.05,0.05,0.05,0.05'),  # All components
        ('1.0', '0.5'),                              # Sadece skaler (otomatik 0,0,0 eklenir)
        ('2.0,3.0,4.0,5.0', '0.1,-0.2,0.3,-0.4'),    # Karışık değerler
        
        # 7. Neutrosophic - DOĞRU formatında (t, i, f)
        ('0.8,0.1,0.1', '0.0,0.05,0.0'),
        ('0.6,0.2,0.2', '0.1,0.0,0.0'),
        ('0.9,0.05,0.05', '0.0,0.0,0.02'),
        ('0.7,0.15,0.15', '0.05,0.0,0.0'),
        ('3.0', '0.5'), # Sadece T değeri
        ('1.0,0.0,0.0', '0.2'), # TIF ve sadece T değeri

        # 8. Neutro-Complex - Format: "complex_part_str,t,i"
        # complex_part_str, t, i
        ('1.0+2.0j,0.5,0.2', '0.1+0.1j,0.0,0.1'),
        ('0.0+1.0j,1.0,0.3', '0.1+0.0j,0.0,0.0'),
        ('2.0,0.0,0.1', '0.0+0.5j,0.5,0.0'), # complex_part olarak sadece real
        ('0.5+0.5j,0.5,0.4', '0.1+0.1j,0.1,0.05'),
        ('1+2j', '0.5+0.5j'), # Sadece complex kısım, diğerleri varsayılan 0
        
        # 9. Hyperreal - VİRGÜL formatında: finite,infinitesimal
        ('3.0,0.001', '0.1,0.0001'),
        ('0.0,0.0005', '1.0,0.0'),
        ('2.0', '0.5'), # Sadece finite kısmı
        ('1.0,0.0', '0.001'), # Finite, Infinitesimal ve sadece finite

        # 10. Bicomplex - Format: "z1_str,z2_str"
        ('1.0+0.5j,0.2+0.1j', '0.1+0.0j,0.0+0.0j'),
        ('2.0,3.0j', '0.5,0.1j'), # z1 sadece real, z2 sadece imag
        ('1+2j', '0.5'), # Sadece z1, z2 ve increment sadece real
        
        # 11. Neutrosophic Bicomplex - 8 PARÇALI VİRGÜL: r1,i1,r2,i2,T,I,F,G
        ('1.0,2.0,0.1,0.2,0.3,0.4,0.5,0.6', '0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0'),
        ('1.0', '0.5'), # Sadece skaler
        
        # 12. Octonion - 8 PARÇALI VİRGÜL
        ('1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0', '0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0'),
        ('1.0', '0.5'), # Sadece skaler
        ('2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0', '0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8'),
        
        # 13. Sedenion - 16 PARÇALI VİRGÜL
        ('1.0' + ',0.0'*15, '0.1' + ',0.0'*15),
        ('1.0', '0.1'), # Sadece skaler
        
        # 14. Clifford - ALGEBRAIC format: scalar+e1+e2+e12
        ('1.0+2.0e1', '0.1+0.2e1'),
        ('2.0e1+3.0e12', '0.1e1-0.2e12'),
        ('5.0', '0.5'), # Sadece skaler
        ('-1.0e1', '2.0e1'),
        
        # 15. Dual - VİRGÜL formatında: real,dual
        ('2.0,0.5', '0.1,0.0'),
        ('1.0', '0.5'), # Sadece real kısım
        ('3.0,1.0', '0.2'),
        
        # 16. Split-Complex - VİRGÜL formatında: real,split
        ('1.0,0.8', '0.1,0.0'),
        ('2.0', '0.3'), # Sadece real kısım
        ('4.0,1.0', '0.5'),
    ]

    # If additional parameters are provided, extend the default set
    if additional_params:
        param_sets.extend(additional_params)

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
        11: "Neutro-Bicomplex",
        12: "Octonion",
        13: "Sedenion",
        14: "Clifford",
        15: "Dual",
        16: "Split-Complex",
    }

    for kececi_type in range(1, 16):
        name = type_names.get(kececi_type, "Unknown Type")
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
                    print(f"Skipped type {kececi_type} with params {start}, {add}: insufficient sequence length")
                    continue

                _, zeta_score = _find_kececi_zeta_zeros(sequence, tolerance=0.5)
                _, gue_score = _compute_gue_similarity(sequence)

                if zeta_score > best_zeta_score:
                    best_zeta_score = zeta_score
                    best_gue_score = gue_score
                    best_params = (start, add)

            except Exception as e:
                print(f"Error analyzing type {kececi_type} with params {start}, {add}: {e}")
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

    # Sort and display results
    sorted_by_zeta = sorted(results, key=lambda x: x['zeta_score'], reverse=True)
    sorted_by_gue = sorted(results, key=lambda x: x['gue_score'], reverse=True)

    print("\nHIGHEST RIEMANN ZETA MATCHING SCORES (TOP 12)")
    print("=" * 80)
    for r in sorted_by_zeta[:12]:
        print(f"{r['name']:<20} {r['zeta_score']:<8.3f} {r['start']:<12} {r['add']:<12}")

    print("\nHIGHEST GUE SIMILARITY SCORES (TOP 12)")
    print("=" * 80)
    for r in sorted_by_gue[:12]:
        print(f"{r['name']:<20} {r['gue_score']:<8.3f} {r['start']:<12} {r['add']:<12}")

    # Plot results
    _plot_comparison(sorted_by_zeta, sorted_by_gue)

    return sorted_by_zeta, sorted_by_gue


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

def _extract_numeric_part(s: str) -> str:
    """String'den sadece sayısal kısmı çıkarır"""
    import re
    # Sayısal kısmı bul (negatif işaretli olabilir)
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        return match.group()
    return "0"  # Default

def _has_comma_format(s: str) -> bool:
    """String'in virgülle ayrılmış formatı olup olmadığını kontrol eder"""
    return ',' in s and not any(c in s for c in ['+', '-', 'j', 'i', 'ε'])

def _is_complex_like(s: str) -> bool:
    """String'in complex sayı formatına benzer olup olmadığını kontrol eder"""
    return any(c in s for c in ['j', 'i', '+', '-']) and ',' not in s

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

    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_SPLIT_COMPLEX):
        raise ValueError(f"Invalid Keçeci Number Type: {kececi_type}")

    use_integer_division = False
    current_value = None
    add_value_typed = None
    ask_unit = None

    try:
        # --- SAYI TİPİNE GÖRE PARSING ---
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            current_value = int(float(start_input_raw))
            add_value_typed = int(float(add_input_raw))
            ask_unit = 1
            use_integer_division = True

        elif kececi_type == TYPE_FLOAT:
            current_value = float(start_input_raw)
            add_value_typed = float(add_input_raw)
            ask_unit = 1.0

        elif kececi_type == TYPE_RATIONAL:
            current_value = Fraction(start_input_raw)
            add_value_typed = Fraction(add_input_raw)
            ask_unit = Fraction(1)

        elif kececi_type == TYPE_COMPLEX:
            current_value = _parse_complex(start_input_raw)
            add_value_typed = _parse_complex(add_input_raw)
            ask_unit = 1 + 1j

        elif kececi_type == TYPE_QUATERNION:
            current_value = _parse_quaternion_from_csv(start_input_raw)
            add_value_typed = _parse_quaternion_from_csv(add_input_raw)
            ask_unit = np.quaternion(1, 1, 1, 1)

        elif kececi_type == TYPE_NEUTROSOPHIC:
            t, i, f = _parse_neutrosophic(start_input_raw)
            current_value = NeutrosophicNumber(t, i, f)
            t_inc, i_inc, f_inc = _parse_neutrosophic(add_input_raw)
            add_value_typed = NeutrosophicNumber(t_inc, i_inc, f_inc)
            ask_unit = NeutrosophicNumber(1, 1, 1)

        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            s_complex = _parse_complex(start_input_raw)
            current_value = NeutrosophicComplexNumber(s_complex.real, s_complex.imag, 0.0)
            a_complex = _parse_complex(add_input_raw)
            add_value_typed = NeutrosophicComplexNumber(a_complex.real, a_complex.imag, 0.0)
            ask_unit = NeutrosophicComplexNumber(1, 1, 1)

        elif kececi_type == TYPE_HYPERREAL:
            finite, infinitesimal = _parse_hyperreal(start_input_raw)
            current_value = HyperrealNumber([finite, infinitesimal])
            
            finite_inc, infinitesimal_inc = _parse_hyperreal(add_input_raw)
            add_value_typed = HyperrealNumber([finite_inc, infinitesimal_inc])
            
            ask_unit = HyperrealNumber([1.0, 1.0])

        elif kececi_type == TYPE_BICOMPLEX:
            current_value = _parse_bicomplex(start_input_raw)
            add_value_typed = _parse_bicomplex(add_input_raw)
            ask_unit = BicomplexNumber(complex(1, 1), complex(1, 1))

        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            current_value = _parse_neutrosophic_bicomplex(start_input_raw)
            add_value_typed = _parse_neutrosophic_bicomplex(add_input_raw)
            ask_unit = NeutrosophicBicomplexNumber(1, 1, 1, 1, 1, 1, 1, 1)

        elif kececi_type == TYPE_OCTONION:
            current_value = _parse_octonion(start_input_raw)
            add_value_typed = _parse_octonion(add_input_raw)
            ask_unit = OctonionNumber(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        elif kececi_type == TYPE_SEDENION:
            current_value = _parse_sedenion(start_input_raw)
            add_value_typed = _parse_sedenion(add_input_raw)
            ask_unit = SedenionNumber([1.0] * 16)

        elif kececi_type == TYPE_CLIFFORD:
            current_value = _parse_clifford(start_input_raw)
            add_value_typed = _parse_clifford(add_input_raw)
            ask_unit = CliffordNumber({'': 1.0})

        elif kececi_type == TYPE_DUAL:
            current_value = _parse_dual(start_input_raw)
            add_value_typed = _parse_dual(add_input_raw)
            ask_unit = DualNumber(1.0, 1.0)

        elif kececi_type == TYPE_SPLIT_COMPLEX:
            current_value = _parse_splitcomplex(start_input_raw)
            add_value_typed = _parse_splitcomplex(add_input_raw)
            ask_unit = SplitcomplexNumber(1.0, 1.0)

        else:
            raise ValueError(f"Unsupported Keçeci type: {kececi_type}")

    except (ValueError, TypeError) as e:
        print(f"ERROR: Failed to initialize type {kececi_type} with start='{start_input_raw}' and increment='{add_input_raw}': {e}")
        return []

    # Log ve temizleme listelerini başlatın
    clean_trajectory = [current_value]
    full_log = [current_value]
    last_divisor_used = None
    ask_counter = 0  # 0: +ask_unit, 1: -ask_unit

    for step in range(iterations):
        # 1. Topla
        added_value = current_value + add_value_typed
        next_q = added_value
        divided_successfully = False
        modified_value = None

        # primary_divisor ve alternative_divisor hesaplama
        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3

        # 2. Bölme dene
        for divisor in [primary_divisor, alternative_divisor]:
            if _is_divisible(added_value, divisor, kececi_type):
                next_q = added_value // divisor if use_integer_division else added_value / divisor
                last_divisor_used = divisor
                divided_successfully = True
                break

        # 3. Eğer bölünemediyse ve "asalsa", ask_unit ile değiştir
        if not divided_successfully and is_prime_like(added_value, kececi_type):
            direction = 1 if ask_counter == 0 else -1
            
            try:
                modified_value = safe_add(added_value, ask_unit, direction)
            except TypeError as e:
                print(f"Error converting ask_unit: {e}")
                continue  # veya uygun bir hata yönetimi yapabilirsiniz

            ask_counter = 1 - ask_counter
            next_q = modified_value

            # Bölme denemelerini burada yap
            for divisor in [primary_divisor, alternative_divisor]:
                if _is_divisible(modified_value, divisor, kececi_type):
                    next_q = modified_value // divisor if use_integer_division else modified_value / divisor
                    last_divisor_used = divisor
                    break

        # 4. Loglara ekle
        full_log.append(added_value)
        if modified_value is not None:
            full_log.append(modified_value)
        if not full_log or next_q != full_log[-1]:  # next_q zaten full_log'a eklenmişse tekrar ekleme
            full_log.append(next_q)

        clean_trajectory.append(next_q)
        current_value = next_q

    # --- SONUÇ ---
    return full_log if include_intermediate_steps else clean_trajectory

def get_with_params(
    kececi_type_choice: int,
    iterations: int,
    start_value_raw: str,
    add_value_raw: str,
    include_intermediate_steps: bool = False
) -> List[Any]:
    """
    Tüm 16 sayı sistemi için ortak arayüz.
    Keçeci mantığı (ask, bölme, asallık) unified_generator ile uygulanır.
    Sadece toplama değil, koşullu değişimler de yapılır.
    """
    print(f"\n--- Generating Sequence: Type {kececi_type_choice}, Steps {iterations} ---")
    print(f"Start: '{start_value_raw}', Addition: '{add_value_raw}'")

    # --- unified_generator'ı doğrudan çağır ---
    # (İlk başta basit toplama döngüsü yapma!)


    try:
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_value_raw,
            add_input_raw=add_value_raw,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps
        )

        if generated_sequence:
            print(f"Generated {len(generated_sequence)} numbers.")
            preview = [str(x) for x in generated_sequence[:40]]  # okunabilirlik için
            print(f"Preview: {preview}...")
            
            # Keçeci Prime Number kontrolü (varsa)
            kpn = find_kececi_prime_number(generated_sequence)
            if kpn is not None:
                print(f"Keçeci Prime Number found: {kpn}")
            else:
                print("No Keçeci Prime Number found.")
        else:
            print("Sequence generation failed or returned empty.")

        return generated_sequence

    except Exception as e:
        print(f"ERROR during sequence generation: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_interactive() -> Tuple[List[Any], Dict[str, Any]]:
    """
    Interactively gets parameters from the user to generate a Keçeci Numbers sequence.
    This version is improved for code clarity and compatibility with the module's
    current interface.
    """

    # Tip sabitleri
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
    TYPE_OCTONION = 12
    TYPE_SEDENION = 13
    TYPE_CLIFFORD = 14
    TYPE_DUAL = 15
    TYPE_SPLIT_COMPLEX = 16

    
    print("\n--- Keçeci Numbers Interactive Generator ---")
    print("  1: Positive Real    2: Negative Real      3: Complex")
    print("  4: Float            5: Rational           6: Quaternion")
    print("  7: Neutrosophic     8: Neutro-Complex     9: Hyperreal")
    print(" 10: Bicomplex       11: Neutro-Bicomplex  12: Octonion")
    print(" 13: Sedenion        14: Clifford          15: Dual")
    print(" 16: Split-Complex")
    
    # Get a valid number type from the user
    while True:
        try:
            type_choice = int(input("Select a Keçeci Number Type (1-16): "))
            if 1 <= type_choice <= 16:
                break
            print("Invalid type. Please enter a number between 1 and 16.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        
    # User prompts for the starting value
    start_prompts = {
        1: "Enter Positive Real start (e.g., '10.0', '5', '3.14'): ",
        2: "Enter Negative Real start (e.g., '-5.0', '-2', '-1.5'): ",
        3: "Enter Complex start (e.g., '1.0+1.0j', '3+4j'): ",
        4: "Enter Float start (e.g., '0.0001412', '3.14', '2.718'): ",
        5: "Enter Rational start (e.g., '0.5', '7/2', '3/4'): ",
        6: "Enter Quaternion start (in 'w,x,y,z' format, e.g., '1.0,2.0,-3.0,1.0'): ",
        7: "Enter Neutrosophic start (in 't,i,f' format, e.g., '0.1,0.0,0.0'): ",
        8: "Enter Neutro-Complex start (in 'real,imag,neut' format, e.g., '2+3j'): ",
        9: "Enter Hyperreal start (in 'finite,infinitesimal' format, e.g., '0.0,0.0001'): ",
        10: "Enter Bicomplex start (in 'z1_real,z1_imag,z2_real,z2_imag' format, e.g., '1.0,2.0,3.0,4.0', '1.0,2.0', '3.0', '1.0+1.0j'): ",
        11: "Enter Neutro-Bicomplex start (in 'r,i,nr,ni,jr,ji,jnr,jni' format, e.g., '1,2,0.1,0.2,0.3,0.4,0.5,0.6', '0.1'): ",
        12: "Enter Octonion start (in 'e0,e1,e2,e3,e4,e5,e6,e7' format, e.g., '1.0,0.5,-0.2,0.3,0.1,-0.4,0.2,0.0'): ",
        13: "Enter Sedenion start (in 'e0,e1,...,e15' format, e.g., '1.0', '0.0'): ",
        14: "Enter Clifford start (in 'scalar,e1,e2,e12,...' format, e.g., '0.1+0.2e1', '1.0+2.0e1+3.0e12'): ",
        15: "Enter Dual start (in 'real,dual' format, e.g., '2.0,0.5'): ",
        16: "Enter Split-Complex start (in 'real,split' format, e.g., '1.0,0.8'): ",
    }
    
    # User prompts for the increment value
    add_prompts = {
        1: "Enter Positive Real increment (e.g., '2.0', '1.5'): ",
        2: "Enter Negative Real increment (e.g., '-1.0', '-0.5'): ",
        3: "Enter Complex increment (e.g., '0.5+0.5j', '3+4j'): ",
        4: "Enter Float increment (e.g., '0.1', '2.5'): ",
        5: "Enter Rational increment (e.g., '1/3', '0.25'): ",
        6: "Enter Quaternion increment (in 'w,x,y,z' format, e.g., '0.1,-0.2,0.3,0.0'): ",
        7: "Enter Neutrosophic increment (in 't,i,f' format, e.g., '0.1,0.0,0.0'): ",
        8: "Enter Neutro-Complex increment (in 'real,imag,neut' format, e.g., '2+3j', 5): ",
        9: "Enter Hyperreal increment (in 'finite,infinitesimal' format, e.g., '0.0,0.0001'): ",
        10: "Enter Bicomplex increment (in 'z1_real,z1_imag,z2_real,z2_imag' format, e.g., '1.0,2.0,3.0,4.0', '1.0,2.0', '3.0', '1.0+1.0j'): ",
        11: "Enter Neutro-Bicomplex increment (in 'r,i,nr,ni,jr,ji,jnr,jni' format, e.g., '1,2,0.1,0.2,0.3,0.4,0.5,0.6', '0.1')): ",
        12: "Enter Octonion increment (in 'e0,e1,e2,e3,e4,e5,e6,e7' format, e.g., '0.1', '0.0'): ",
        13: "Enter Sedenion increment (in 'e0,e1,...,e15' format, e.g., '0.1', '0.0'): ",
        14: "Enter Clifford increment (in 'scalar,e1,e2,e12,...' format, e.g., '0.1+0.2e1', '1.0+2.0e1+3.0e12'): ",
        15: "Enter Dual increment (in 'real,dual' format, e.g., 1.0, '0.1,0.0'): ",
        16: "Enter Split-Complex increment (in 'real,split' format, e.g., '0.1,0.0'): ",
    }
    
    # Get inputs from the user
    start_input_val_raw = input(start_prompts.get(type_choice, "Enter starting value: "))
    add_input_val_raw = input(add_prompts.get(type_choice, "Enter increment value: "))
    
    while True:
        try:
            num_kececi_steps = int(input("Enter number of Keçeci steps (e.g., 40, 60): "))
            if num_kececi_steps > 0:
                break
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    show_details_input = input("Do you want to include the intermediate calculation steps? (y/n): ").lower().strip()
    show_details = (show_details_input == 'y' or show_details_input == 'yes')
    
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
    if len(sequence) < 4:  # Çok kısa dizilerde periyot aramak anlamsız
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

def is_quaternion_like(obj):
    if isinstance(obj, np.quaternion):
        return True
    if hasattr(obj, 'components'):
        comp = np.array(obj.components)
        return comp.size == 4
    if all(hasattr(obj, attr) for attr in ['w', 'x', 'y', 'z']):
        return True
    if hasattr(obj, 'scalar') and hasattr(obj, 'vector') and isinstance(obj.vector, (list, np.ndarray)) and len(obj.vector) == 3:
        return True
    return False

def is_neutrosophic_like(obj):
    """NeutrosophicNumber gibi görünen objeleri tanır (t,i,f veya a,b vs.)"""
    return (hasattr(obj, 't') and hasattr(obj, 'i') and hasattr(obj, 'f')) or \
           (hasattr(obj, 'a') and hasattr(obj, 'b')) or \
           (hasattr(obj, 'value') and hasattr(obj, 'indeterminacy')) or \
           (hasattr(obj, 'determinate') and hasattr(obj, 'indeterminate'))

def plot_numbers(sequence: List[Any], title: str = "Keçeci Number Sequence Analysis"):
    """
    Tüm 16 Keçeci Sayı türü için detaylı görselleştirme sağlar.
    """
    if not sequence:
        print("Sequence is empty. Nothing to plot.")
        return

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    first_elem = sequence[0]

    # --- 1. Fraction (Rational)
    if isinstance(first_elem, Fraction):
        float_vals = [float(x) for x in sequence]
        numerators = [x.numerator for x in sequence]
        denominators = [x.denominator for x in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(float_vals, 'o-', color='tab:blue')
        ax1.set_title("Fraction as Float")
        ax1.set_ylabel("Value")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(numerators, 's-', label='Numerator', color='tab:orange')
        ax2.plot(denominators, '^-', label='Denominator', color='tab:green')
        ax2.set_title("Numerator & Denominator")
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot([n/d for n, d in zip(numerators, denominators)], 'o-', color='tab:purple')
        ax3.set_title("Computed Ratio")
        ax3.set_ylabel("n/d")

        ax4 = fig.add_subplot(gs[1, 1])
        sc = ax4.scatter(numerators, denominators, c=range(len(sequence)), cmap='plasma', s=30)
        ax4.set_title("Num vs Den Trajectory")
        ax4.set_xlabel("Numerator")
        ax4.set_ylabel("Denominator")
        plt.colorbar(sc, ax=ax4, label="Step")

    # --- 2. int, float (Positive/Negative Real, Float)
    elif isinstance(first_elem, (int, float)):
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([float(x) for x in sequence], 'o-', color='tab:blue', markersize=5)
        ax.set_title("Real Number Sequence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    # --- 3. Complex
    elif isinstance(first_elem, complex):
        real_parts = [z.real for z in sequence]
        imag_parts = [z.imag for z in sequence]
        magnitudes = [abs(z) for z in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_parts, 'o-', color='tab:blue')
        ax1.set_title("Real Part")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(imag_parts, 'o-', color='tab:red')
        ax2.set_title("Imaginary Part")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:purple')
        ax3.set_title("Magnitude |z|")

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(real_parts, imag_parts, '.-', alpha=0.7)
        ax4.scatter(real_parts[0], imag_parts[0], c='g', s=100, label='Start')
        ax4.scatter(real_parts[-1], imag_parts[-1], c='r', s=100, label='End')
        ax4.set_title("Complex Plane")
        ax4.set_xlabel("Re(z)")
        ax4.set_ylabel("Im(z)")
        ax4.legend()
        ax4.axis('equal')
        ax4.grid(True, alpha=0.3)

    # --- 4. Quaternion
    elif hasattr(first_elem, 'components') and len(getattr(first_elem, 'components', [])) == 4:
        try:
            comp = np.array([getattr(q, 'components', [q.w, q.x, q.y, q.z]) for q in sequence])
            w, x, y, z = comp.T
            magnitudes = np.linalg.norm(comp, axis=1)
            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            labels = ['w', 'x', 'y', 'z']
            for i, label in enumerate(labels):
                ax1.plot(comp[:, i], label=label, alpha=0.8)
            ax1.set_title("Quaternion Components")
            ax1.legend()

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(magnitudes, 'o-', color='tab:purple')
            ax2.set_title("Magnitude |q|")

            ax3 = fig.add_subplot(gs[1, :], projection='3d')
            ax3.plot(x, y, z, alpha=0.7)
            ax3.scatter(x[0], y[0], z[0], c='g', s=100, label='Start')
            ax3.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='End')
            ax3.set_title("3D Trajectory (x,y,z)")
            ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
            ax3.legend()
        except Exception as e:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='red')

    # --- 5. OctonionNumber
    elif hasattr(first_elem, 'coeffs') and len(first_elem.coeffs) == 8:
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(4):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.7)
        ax1.set_title("e0-e3 Components")
        ax1.legend(ncol=2)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(4, 8):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.7)
        ax2.set_title("e4-e7 Components")
        ax2.legend(ncol=2)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:purple')
        ax3.set_title("Magnitude |o|")

        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        ax4.plot(coeffs[:, 1], coeffs[:, 2], coeffs[:, 3], alpha=0.7)
        ax4.set_title("3D (e1,e2,e3)")
        ax4.set_xlabel("e1"); ax4.set_ylabel("e2"); ax4.set_zlabel("e3")

    # --- 6. SedenionNumber
    elif hasattr(first_elem, 'coeffs') and len(first_elem.coeffs) == 16:
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(8):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax1.set_title("Sedenion e0-e7")
        ax1.legend(ncol=2, fontsize=6)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(8, 16):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax2.set_title("e8-e15")
        ax2.legend(ncol=2, fontsize=6)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:purple')
        ax3.set_title("Magnitude |s|")

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            if len(sequence) > 2:
                proj = pca.fit_transform(coeffs)
                ax4 = fig.add_subplot(gs[1, 1])
                sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                ax4.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                plt.colorbar(sc, ax=ax4, label="Iteration")
        except ImportError:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 7. CliffordNumber
    elif hasattr(first_elem, 'basis') and isinstance(first_elem.basis, dict):
        all_keys = sorted(first_elem.basis.keys(), key=lambda x: (len(x), x))
        values = {k: [elem.basis.get(k, 0.0) for elem in sequence] for k in all_keys}
        scalar = values.get('', [0]*len(sequence))
        vector_keys = [k for k in all_keys if len(k) == 1]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(scalar, 'o-', label='Scalar', color='black')
        for k in vector_keys[:3]:
            ax1.plot(values[k], label=f'Vec {k}', alpha=0.7)
        ax1.set_title("Scalar & Vectors")
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        bivector_mags = [sum(v**2 for k, v in elem.basis.items() if len(k) == 2)**0.5 for elem in sequence]
        ax2.plot(bivector_mags, 'o-', color='tab:green')
        ax2.set_title("Bivector Magnitude")

        try:
            from sklearn.decomposition import PCA
            matrix = np.array([list(v.values()) for v in values.values()]).T
            pca = PCA(n_components=2)
            if len(sequence) > 1:
                proj = pca.fit_transform(matrix)
                ax3 = fig.add_subplot(gs[1, :])
                sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='plasma')
                ax3.set_title("Clifford PCA")
                plt.colorbar(sc, ax=ax3, label="Component")
        except:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.text(0.5, 0.5, "Install sklearn for PCA", ha='center')

    # --- 8. DualNumber
    elif hasattr(first_elem, 'real') and hasattr(first_elem, 'dual'):
        real_vals = [x.real for x in sequence]
        dual_vals = [x.dual for x in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_vals, 'o-', color='tab:blue')
        ax1.set_title("Real Part")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(dual_vals, 'o-', color='tab:orange')
        ax2.set_title("Dual Part (ε)")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(real_vals, dual_vals, '.-')
        ax3.set_title("Real vs Dual")
        ax3.set_xlabel("Real")
        ax3.set_ylabel("Dual")

        ax4 = fig.add_subplot(gs[1, 1])
        ratios = [d/r if r != 0 else 0 for r, d in zip(real_vals, dual_vals)]
        ax4.plot(ratios, 'o-', color='tab:purple')
        ax4.set_title("Dual/Real Ratio")

    # --- 9. SplitcomplexNumber
    elif hasattr(first_elem, 'real') and hasattr(first_elem, 'split'):
        real_vals = [x.real for x in sequence]
        split_vals = [x.split for x in sequence]
        u_vals = [r + s for r, s in zip(real_vals, split_vals)]
        v_vals = [r - s for r, s in zip(real_vals, split_vals)]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_vals, 'o-', color='tab:green')
        ax1.set_title("Real Part")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(split_vals, 'o-', color='tab:brown')
        ax2.set_title("Split Part (j)")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(real_vals, split_vals, '.-')
        ax3.set_title("Trajectory (Real vs Split)")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(u_vals, label='u = r+j')
        ax4.plot(v_vals, label='v = r-j')
        ax4.set_title("Light-Cone Coordinates")
        ax4.legend()

    # --- 10. NeutrosophicNumber
    elif is_neutrosophic_like(first_elem):
        # t, i, f değerlerini al
        if hasattr(first_elem, 't'):
            t_vals = [x.t for x in sequence]
            i_vals = [x.i for x in sequence]
            f_vals = [x.f for x in sequence]
        elif hasattr(first_elem, 'a'):
            t_vals = [x.a for x in sequence]
            i_vals = [x.b for x in sequence]
            f_vals = [0] * len(sequence)  # f yoksa sıfır
        elif hasattr(first_elem, 'value'):
            t_vals = [x.value for x in sequence]
            i_vals = [x.indeterminacy for x in sequence]
            f_vals = [0] * len(sequence)
        else:
            t_vals = i_vals = f_vals = []

        gs = GridSpec(2, 2, figure=fig)

        # 1. t, i, f zaman içinde
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_vals, 'o-', label='Truth (t)', color='tab:blue')
        ax1.plot(i_vals, 's-', label='Indeterminacy (i)', color='tab:orange')
        ax1.plot(f_vals, '^-', label='Falsity (f)', color='tab:red')
        ax1.set_title("Neutrosophic Components")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. t vs i
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(t_vals, i_vals, c=range(len(t_vals)), cmap='viridis', s=30)
        ax2.set_title("t vs i Trajectory")
        ax2.set_xlabel("Truth (t)")
        ax2.set_ylabel("Indeterminacy (i)")
        plt.colorbar(ax2.collections[0], ax=ax2, label="Step")

        # 3. t vs f
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(t_vals, f_vals, c=range(len(t_vals)), cmap='plasma', s=30)
        ax3.set_title("t vs f Trajectory")
        ax3.set_xlabel("Truth (t)")
        ax3.set_ylabel("Falsity (f)")
        plt.colorbar(ax3.collections[0], ax=ax3, label="Step")

        # 4. Magnitude (t² + i² + f²)
        magnitudes = [np.sqrt(t**2 + i**2 + f**2) for t, i, f in zip(t_vals, i_vals, f_vals)]
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(magnitudes, 'o-', color='tab:purple')
        ax4.set_title("Magnitude √(t²+i²+f²)")
        ax4.set_ylabel("|n|")

    # --- 11. NeutrosophiccomplexNumber
    elif (hasattr(first_elem, 'real') and hasattr(first_elem, 'imag') and 
          hasattr(first_elem, 'indeterminacy')):
        
        real_parts = [x.real for x in sequence]
        imag_parts = [x.imag for x in sequence]
        indeter_parts = [x.indeterminacy for x in sequence]
        magnitudes_z = [abs(complex(x.real, x.imag)) for x in sequence]

        gs = GridSpec(2, 2, figure=fig)

        # 1. Complex Plane
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_parts, imag_parts, '.-', alpha=0.7)
        ax1.scatter(real_parts[0], imag_parts[0], c='g', s=100, label='Start')
        ax1.scatter(real_parts[-1], imag_parts[-1], c='r', s=100, label='End')
        ax1.set_title("Complex Plane")
        ax1.set_xlabel("Re(z)")
        ax1.set_ylabel("Im(z)")
        ax1.legend()
        ax1.axis('equal')

        # 2. Indeterminacy over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(indeter_parts, 'o-', color='tab:purple')
        ax2.set_title("Indeterminacy Level")
        ax2.set_ylabel("I")

        # 3. |z| vs Indeterminacy
        ax3 = fig.add_subplot(gs[1, 0])
        sc = ax3.scatter(magnitudes_z, indeter_parts, c=range(len(magnitudes_z)), cmap='viridis', s=30)
        ax3.set_title("Magnitude vs Indeterminacy")
        ax3.set_xlabel("|z|")
        ax3.set_ylabel("I")
        plt.colorbar(sc, ax=ax3, label="Step")

        # 4. Real vs Imag colored by I
        ax4 = fig.add_subplot(gs[1, 1])
        sc2 = ax4.scatter(real_parts, imag_parts, c=indeter_parts, cmap='plasma', s=40)
        ax4.set_title("Real vs Imag (colored by I)")
        ax4.set_xlabel("Re(z)")
        ax4.set_ylabel("Im(z)")
        plt.colorbar(sc2, ax=ax4, label="Indeterminacy")

    # --- 12. HyperrealNumber
    elif hasattr(first_elem, 'sequence') and isinstance(first_elem.sequence, (list, np.ndarray)):
        seq_len = min(len(first_elem.sequence), 5)  # İlk 5 bileşen
        data = np.array([x.sequence[:seq_len] for x in sequence])
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(seq_len):
            ax1.plot(data[:, i], label=f'ε^{i}', alpha=0.8)
        ax1.set_title("Hyperreal Components")
        ax1.legend(ncol=2)

        ax2 = fig.add_subplot(gs[0, 1])
        magnitudes = np.linalg.norm(data, axis=1)
        ax2.plot(magnitudes, 'o-', color='tab:purple')
        ax2.set_title("Magnitude")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(data[:, 0], 'o-', label='Standard Part')
        ax3.set_title("Standard Part (ε⁰)")
        ax3.legend()

        ax4 = fig.add_subplot(gs[1, 1])
        sc = ax4.scatter(data[:, 0], data[:, 1], c=range(len(data)), cmap='viridis')
        ax4.set_title("Standard vs Infinitesimal")
        ax4.set_xlabel("Standard")
        ax4.set_ylabel("ε¹")
        plt.colorbar(sc, ax=ax4, label="Step")

    # --- 13. BicomplexNumber
    elif hasattr(first_elem, 'z1') and hasattr(first_elem, 'z2'):
        z1_real = [x.z1.real for x in sequence]
        z1_imag = [x.z1.imag for x in sequence]
        z2_real = [x.z2.real for x in sequence]
        z2_imag = [x.z2.imag for x in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(z1_real, label='Re(z1)')
        ax1.plot(z1_imag, label='Im(z1)')
        ax1.set_title("Bicomplex z1")
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(z2_real, label='Re(z2)')
        ax2.plot(z2_imag, label='Im(z2)')
        ax2.set_title("Bicomplex z2")
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(z1_real, z1_imag, '.-')
        ax3.set_title("z1 Trajectory")
        ax3.set_xlabel("Re(z1)")
        ax3.set_ylabel("Im(z1)")

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(z2_real, z2_imag, '.-')
        ax4.set_title("z2 Trajectory")
        ax4.set_xlabel("Re(z2)")
        ax4.set_ylabel("Im(z2)")

    # --- 13. NeutrosophicBicomplexNumber ---
    elif hasattr(first_elem, 'real') and hasattr(first_elem, 'neut_real'):
        try:
            # Güvenli float dönüşümü
            comps = np.array([
                [float(getattr(x, attr))
                 for attr in ['real', 'imag', 'neut_real', 'neut_imag',
                             'j_real', 'j_imag', 'j_neut_real', 'j_neut_imag']]
                for x in sequence
            ])
            magnitudes = np.linalg.norm(comps, axis=1)
            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            for i, label in enumerate(['a', 'b', 'c', 'd']):
                ax1.plot(comps[:, i], label=label, alpha=0.7)
            ax1.set_title("First 4 Components")
            ax1.legend()

            ax2 = fig.add_subplot(gs[0, 1])
            for i, label in enumerate(['e', 'f', 'g', 'h']):
                ax2.plot(comps[:, i + 4], label=label, alpha=0.7)
            ax2.set_title("Last 4 Components")
            ax2.legend()

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(magnitudes, 'o-', color='tab:purple')
            ax3.set_title("Magnitude")

            ax4 = fig.add_subplot(gs[1, 1])
            sc = ax4.scatter(comps[:, 0], comps[:, 1], c=range(len(comps)), cmap='plasma')
            ax4.set_title("a vs b Trajectory")
            ax4.set_xlabel("a")
            ax4.set_ylabel("b")
            plt.colorbar(sc, ax=ax4, label="Step")

        except Exception as e:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Plot error: {e}", ha='center', va='center', color='red')
            ax.set_xticks([])
            ax.set_yticks([])

    # --- 15. Bilinmeyen tip
    else:
        ax = fig.add_subplot(1, 1, 1)
        type_name = type(first_elem).__name__
        ax.text(0.5, 0.5, f"Plotting not implemented\nfor '{type_name}'",
                ha='center', va='center', fontsize=14, fontweight='bold', color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


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
    print("\nRunning programmatic tests for all 16 number types...")
    
    STEPS = 40
    START_VAL = "2.5"
    ADD_VAL = 3.0

    all_types = {
        "Positive Real": TYPE_POSITIVE_REAL, "Negative Real": TYPE_NEGATIVE_REAL,
        "Complex": TYPE_COMPLEX, "Float": TYPE_FLOAT, "Rational": TYPE_RATIONAL,
        "Quaternion": TYPE_QUATERNION, "Neutrosophic": TYPE_NEUTROSOPHIC,
        "Neutrosophic Complex": TYPE_NEUTROSOPHIC_COMPLEX, "Hyperreal": TYPE_HYPERREAL,
        "Bicomplex": TYPE_BICOMPLEX, "Neutrosophic Bicomplex": TYPE_NEUTROSOPHIC_BICOMPLEX,
        "Octonion": TYPE_OCTONION, "Sedenion": TYPE_SEDENION, "Clifford": TYPE_CLIFFORD, 
        "Dual": TYPE_DUAL, "Splitcomplex": TYPE_SPLIT_COMPLEX,
    }

    types_to_plot = [
        "Complex", "Quaternion", "Bicomplex", "Neutrosophic Complex", "Hyperreal", "Octonion", "Sedenion", "Clifford", "Dual", "Splitcompllex",
    ]
    
    for name, type_id in all_types.items():
        start = "-5" if type_id == TYPE_NEGATIVE_REAL else "2+3j" if type_id in [TYPE_COMPLEX, TYPE_BICOMPLEX] else START_VAL
        
        seq = get_with_params(type_id, STEPS, start, ADD_VAL)
        
        if name in types_to_plot and seq:
            plot_numbers(seq, title=f"Demonstration: {name} Keçeci Numbers")

    print("\n\nDemonstration finished. Plots for selected types are shown.")
    plt.show()
