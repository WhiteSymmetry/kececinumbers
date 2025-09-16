# -*- coding: utf-8 -*-
"""
Keçeci Numbers Module (kececinumbers.py)

This module provides a comprehensive framework for generating, analyzing, and
visualizing Keçeci Numbers across various number systems. It supports 20
distinct types, from standard integers and complex numbers to more exotic
constructs like neutrosophic and bicomplex numbers.

The core of the module is the `unified_generator`, which implements the
specific algorithm for creating Keçeci Number sequences. High-level functions
are available for easy interaction, parameter-based generation, and plotting.

Key Features:
- Generation of 20 types of Keçeci Numbers.
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
from quaternion import quaternion  
# conda install -c conda-forge quaternion # pip install numpy-quaternion
import random
import re
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
import sympy
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# --- MODULE CONSTANTS: Keçeci NUMBER TYPES ---
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
TYPE_Pathion = 17
TYPE_Chingon = 18
TYPE_Routon = 19
TYPE_Voudon = 20


# ==============================================================================
# --- CUSTOM NUMBER CLASS DEFINITIONS ---
# ==============================================================================

class PathionNumber:
    """32-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 32:
            coeffs = list(coeffs) + [0.0] * (32 - len(coeffs))
            if len(coeffs) > 32:
                coeffs = coeffs[:32]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self):
        """Gerçek kısım (ilk bileşen)"""
        return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"PathionNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, PathionNumber):
            return PathionNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return PathionNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, PathionNumber):
            return PathionNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return PathionNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, PathionNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return PathionNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return PathionNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return PathionNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if isinstance(other, PathionNumber):
            return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return PathionNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'PathionNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return PathionNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'PathionNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / PathionNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return PathionNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'PathionNumber'")


class ChingonNumber:
    """64-bileşenli Chingon sayısı"""  # Açıklama düzeltildi
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 64:
            coeffs = list(coeffs) + [0.0] * (64 - len(coeffs))
            if len(coeffs) > 64:
                coeffs = coeffs[:64]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self):
        """Gerçek kısım (ilk bileşen)"""
        return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"ChingonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"ChingonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return ChingonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return ChingonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, ChingonNumber):
            # Basitçe bileşen bazlı çarpma
            return ChingonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])  # ChingonNumber döndür
        else:
            # Skaler çarpma
            return ChingonNumber([c * float(other) for c in self.coeffs])  # ChingonNumber döndür
    
    def __mod__(self, divisor):
        return ChingonNumber([c % divisor for c in self.coeffs])  # ChingonNumber döndür
    
    def __eq__(self, other):
        if isinstance(other, ChingonNumber):  # ChingonNumber ile karşılaştır
            return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return ChingonNumber([c / other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return ChingonNumber([c // other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / ChingonNumber"""
        if isinstance(other, (int, float)):
            return ChingonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'ChingonNumber'")  # ChingonNumber


class RoutonNumber:
    """128-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 128:
            coeffs = list(coeffs) + [0.0] * (128 - len(coeffs))
            if len(coeffs) > 128:
                coeffs = coeffs[:128]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self):
        """Gerçek kısım (ilk bileşen)"""
        return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"RoutonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"RoutonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, RoutonNumber):
            return RoutonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return RoutonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, RoutonNumber):
            return RoutonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return RoutonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, RoutonNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return RoutonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return RoutonNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return RoutonNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if isinstance(other, RoutonNumber):
            return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return RoutonNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'RoutonNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return RoutonNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'RoutonNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / RoutonNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return RoutonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'RoutonNumber'")


class VoudonNumber:
    """256-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 256:
            coeffs = list(coeffs) + [0.0] * (256 - len(coeffs))
            if len(coeffs) > 256:
                coeffs = coeffs[:256]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self):
        """Gerçek kısım (ilk bileşen)"""
        return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"VoudonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"VoudonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, VoudonNumber):
            return VoudonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return VoudonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, VoudonNumber):
            return VoudonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return VoudonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, VoudonNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return VoudonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return VoudonNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return VoudonNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if isinstance(other, VoudonNumber):
            return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return VoudonNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'VoudonNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return VoudonNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'VoudonNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / VoudonNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return VoudonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'VoudonNumber'")

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
            parts.append(f"({self.z2.real}+{self.z2.imag}j)e")
        return " + ".join(parts) if parts else "0"

def _parse_bicomplex(s: str) -> BicomplexNumber:
    """
    Kececi kütüphanesinin beklediği format: Tek parametre ile
    Sadece bicomplex parsing yapar
    """
    s_clean = s.strip().replace(" ", "")
    
    try:
        # Format 1: Comma-separated "z1_real,z1_imag,z2_real,z2_imag"
        if ',' in s_clean:
            parts = [float(p) for p in s_clean.split(',')]
            if len(parts) == 4:
                return BicomplexNumber(complex(parts[0], parts[1]), 
                                      complex(parts[2], parts[3]))
            elif len(parts) == 2:
                return BicomplexNumber(complex(parts[0], parts[1]), 
                                      complex(0, 0))
            elif len(parts) == 1:
                return BicomplexNumber(complex(parts[0], 0), 
                                      complex(0, 0))
        
        # Format 2: Explicit "(a+bj)+(c+dj)e"
        if 'e' in s_clean and '(' in s_clean:
            pattern = r'\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)\s*\+\s*\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)e'
            match = re.search(pattern, s_clean)
            
            if match:
                z1_real = float(match.group(1))
                z1_imag_sign = -1 if match.group(2) == '-' else 1
                z1_imag_val = float(match.group(3) or '1')
                z1_imag = z1_imag_sign * z1_imag_val
                
                z2_real = float(match.group(4))
                z2_imag_sign = -1 if match.group(5) == '-' else 1
                z2_imag_val = float(match.group(6) or '1')
                z2_imag = z2_imag_sign * z2_imag_val
                
                return BicomplexNumber(complex(z1_real, z1_imag), 
                                      complex(z2_real, z2_imag))
        
        # Format 3: Simple values
        try:
            if 'j' in s_clean:
                # Complex number parsing
                if 'j' not in s_clean:
                    return BicomplexNumber(complex(float(s_clean), 0), complex(0, 0))
                
                pattern = r'^([+-]?\d*\.?\d*)([+-]?\d*\.?\d*)j$'
                match = re.match(pattern, s_clean)
                if match:
                    real_part = match.group(1)
                    imag_part = match.group(2)
                    
                    if real_part in ['', '+', '-']:
                        real_part = real_part + '1' if real_part else '0'
                    if imag_part in ['', '+', '-']:
                        imag_part = imag_part + '1' if imag_part else '0'
                    
                    return BicomplexNumber(complex(float(real_part or 0), float(imag_part or 0)), 
                                          complex(0, 0))
            
            # Real number
            real_val = float(s_clean)
            return BicomplexNumber(complex(real_val, 0), complex(0, 0))
            
        except:
            pass
            
    except Exception as e:
        print(f"Bicomplex parsing error for '{s}': {e}")
    
    # Default fallback
    return BicomplexNumber(complex(0, 0), complex(0, 0))

def _parse_universal(s: str, target_type: str) -> Any:
    """
    Universal parser - Sizin diğer yerlerde kullandığınız iki parametreli versiyon
    """
    if target_type == "real":
        try:
            return float(s.strip())
        except ValueError:
            return 0.0
    
    elif target_type == "complex":
        try:
            s_clean = s.strip().replace(" ", "")
            if 'j' not in s_clean:
                return complex(float(s_clean), 0.0)
            
            pattern = r'^([+-]?\d*\.?\d*)([+-]?\d*\.?\d*)j$'
            match = re.match(pattern, s_clean)
            if match:
                real_part = match.group(1)
                imag_part = match.group(2)
                
                if real_part in ['', '+', '-']:
                    real_part = real_part + '1' if real_part else '0'
                if imag_part in ['', '+', '-']:
                    imag_part = imag_part + '1' if imag_part else '0'
                
                return complex(float(real_part or 0), float(imag_part or 0))
            
            return complex(s_clean)
        except:
            return complex(0, 0)
    
    elif target_type == "bicomplex":
        # _parse_bicomplex'i çağır (tek parametreli)
        return _parse_bicomplex(s)
    
    return None

def kececi_bicomplex_algorithm(start: BicomplexNumber, add_val: BicomplexNumber, iterations: int, include_intermediate: bool = True) -> list:
    """
    Gerçek Keçeci algoritmasının bikompleks versiyonunu uygular
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Toplama işlemi
        current = current + add_val
        
        # 2. Keçeci algoritmasının özelliği: Mod alma ve asal sayı kontrolü
        # Gerçek algoritmada belirli bir modüle göre işlem yapılır
        mod_value = 100  # Bu değer algoritmaya göre ayarlanabilir
        
        # z1 ve z2 için mod alma (gerçek ve sanal kısımlar ayrı ayrı)
        current = BicomplexNumber(
            complex(current.z1.real % mod_value, current.z1.imag % mod_value),
            complex(current.z2.real % mod_value, current.z2.imag % mod_value)
        )
        
        # 3. Ara adımları ekle (Keçeci algoritmasının karakteristik özelliği)
        if include_intermediate:
            # Ara değerler için özel işlemler (örneğin: çarpma, bölme, vs.)
            intermediate = current * BicomplexNumber(complex(0.5, 0), complex(0, 0))
            sequence.append(intermediate)
        
        sequence.append(current)
        
        # 4. Asal sayı kontrolü (Keçeci algoritmasının önemli bir parçası)
        # Bu kısım algoritmanın detayına göre özelleştirilebilir
        magnitude = abs(current.z1) + abs(current.z2)
        if magnitude > 1 and all(magnitude % i != 0 for i in range(2, int(magnitude**0.5) + 1)):
            print(f"Keçeci Prime found at step {i}: {magnitude:.2f}")
        
        # 5. Özel durum: Belirli değerlere ulaşıldığında resetleme
        if abs(current.z1) < 1e-10 and abs(current.z2) < 1e-10:
            current = start  # Başa dön
    
    return sequence

# --- DAHA GERÇEKÇİ BİR VERSİYON ---
def kececi_bicomplex_advanced(start: BicomplexNumber, add_val: BicomplexNumber, 
                            iterations: int, include_intermediate: bool = True) -> list:
    """
    Gelişmiş Keçeci algoritması - daha karmaşık matematiksel işlemler içerir
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Temel toplama
        current = current + add_val
        
        # 2. Doğrusal olmayan dönüşümler (Keçeci algoritmasının özelliği)
        # Kök alma ve kuvvet alma işlemleri
        current = BicomplexNumber(
            complex(current.z1.real**0.5, current.z1.imag**0.5),
            complex(current.z2.real**0.5, current.z2.imag**0.5)
        )
        
        # 3. Modüler aritmetik
        mod_real = 50
        mod_imag = 50
        current = BicomplexNumber(
            complex(current.z1.real % mod_real, current.z1.imag % mod_imag),
            complex(current.z2.real % mod_real, current.z2.imag % mod_imag)
        )
        
        # 4. Ara adımlar
        if include_intermediate:
            # Çapraz çarpım ara değerleri
            cross_product = BicomplexNumber(
                complex(current.z1.real * current.z2.imag, 0),
                complex(0, current.z1.imag * current.z2.real)
            )
            sequence.append(cross_product)
        
        sequence.append(current)
        
        # 5. Dinamik sistem davranışı için feedback
        if i % 10 == 0 and i > 0:
            # Her 10 adımda bir küçük bir perturbasyon ekle
            perturbation = BicomplexNumber(
                complex(0.1 * np.sin(i), 0.1 * np.cos(i)),
                complex(0.05 * np.sin(i*0.5), 0.05 * np.cos(i*0.5))
            )
            current = current + perturbation
    
    return sequence

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
    def __init__(self, basis_dict: Dict[str, float]):
        """CliffordNumber constructor."""
        # Sadece sıfır olmayan değerleri sakla
        self.basis = {k: float(v) for k, v in basis_dict.items() if abs(float(v)) > 1e-10}
    
    @property
    def dimension(self) -> int:
        """Vector space dimension'ını otomatik hesaplar."""
        max_index = 0
        for key in self.basis.keys():
            if key:  # scalar değilse
                # '12', '123' gibi string'lerden maksimum rakamı bul
                if key.isdigit():
                    max_index = max(max_index, max(int(c) for c in key))
        return max_index

    def __add__(self, other):
        if isinstance(other, CliffordNumber):
            new_basis = self.basis.copy()
            for k, v in other.basis.items():
                new_basis[k] = new_basis.get(k, 0.0) + v
                # Sıfıra yakın değerleri temizle
                if abs(new_basis[k]) < 1e-10:
                    del new_basis[k]
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
                if abs(new_basis[k]) < 1e-10:
                    del new_basis[k]
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
            # Basit Clifford çarpımı (e_i^2 = +1 varsayımıyla)
            new_basis = {}
            
            for k1, v1 in self.basis.items():
                for k2, v2 in other.basis.items():
                    # Skaler çarpım
                    if k1 == '':
                        product_key = k2
                        sign = 1.0
                    elif k2 == '':
                        product_key = k1
                        sign = 1.0
                    else:
                        # Vektör çarpımı: e_i * e_j
                        combined = sorted(k1 + k2)
                        product_key = ''.join(combined)
                        
                        # Basitleştirilmiş: e_i^2 = +1, anti-commutative
                        sign = 1.0
                        # Burada gerçek Clifford cebir kuralları uygulanmalı
                    
                    new_basis[product_key] = new_basis.get(product_key, 0.0) + sign * v1 * v2
            
            return CliffordNumber(new_basis)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return CliffordNumber({k: v / other for k, v in self.basis.items()})
        return NotImplemented

    def __str__(self):
        parts = []
        if '' in self.basis and abs(self.basis['']) > 1e-10:
            parts.append(f"{self.basis['']:.2f}")
        
        sorted_keys = sorted([k for k in self.basis if k != ''], key=lambda x: (len(x), x))
        for k in sorted_keys:
            v = self.basis[k]
            if abs(v) > 1e-10:
                sign = '+' if v > 0 and parts else ''
                parts.append(f"{sign}{v:.2f}e{k}")
        
        result = "".join(parts).replace("+-", "-")
        return result if result else "0.0"

    @classmethod
    def parse(cls, s) -> 'CliffordNumber':
        """Class method olarak parse metodu"""
        return _parse_clifford(s)

    def __repr__(self):
        return self.__str__()


@dataclass
class DualNumber:

    real: float
    dual: float

    def __init__(self, real, dual):
        self.real = float(real)
        self.dual = float(dual)
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real + other, self.dual)
        raise TypeError
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real - other, self.dual)
        raise TypeError
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real * other, self.dual * other)
        raise TypeError
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            return DualNumber(self.real / other, self.dual / other)
        elif isinstance(other, DualNumber):
            if other.real == 0:
                raise ZeroDivisionError
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
        raise TypeError
    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            return DualNumber(self.real // other, self.dual // other)
        raise TypeError
    def __eq__(self, other):
        if isinstance(other, DualNumber):
            return self.real == other.real and self.dual == other.dual
        elif isinstance(other, (int, float)):
            return self.real == other and self.dual == 0
        return False
    def __str__(self):
        return f"{self.real} + {self.dual}ε"
    def __repr__(self):
        return self.__str__() # __repr__ eklenmiş
    def __int__(self):
        return int(self.real) # int() dönüşümü eklenmiş
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

def _parse_complex(s) -> complex:
    """Bir string'i veya sayıyı complex sayıya dönüştürür.
    "real,imag", "real+imag(i/j)", "real", "imag(i/j)" formatlarını destekler.
    Float ve int tiplerini de doğrudan kabul eder.
    """
    # Eğer zaten complex sayıysa doğrudan döndür
    if isinstance(s, complex):
        return s
    
    # Eğer float veya int ise doğrudan complex'e dönüştür
    if isinstance(s, (float, int)):
        return complex(s)
    
    # String işlemleri için önce string'e dönüştür
    if isinstance(s, str):
        s = s.strip().replace('J', 'j').replace('i', 'j') # Hem J hem i yerine j kullan
    else:
        s = str(s).strip().replace('J', 'j').replace('i', 'j')
    
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


def _parse_neutrosophic(s) -> Tuple[float, float, float]:
    """Parses neutrosophic string into (t, i, f) tuple."""
    # Eğer zaten tuple ise doğrudan döndür
    if isinstance(s, (tuple, list)) and len(s) >= 3:
        return float(s[0]), float(s[1]), float(s[2])
    
    # Sayısal tipse sadece t değeri olarak işle
    if isinstance(s, (float, int, complex)):
        return float(s), 0.0, 0.0
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
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

def _parse_hyperreal(s) -> Tuple[float, float]:
    """Parses hyperreal string into (finite, infinitesimal) tuple."""
    # Eğer zaten tuple ise doğrudan döndür
    if isinstance(s, (tuple, list)) and len(s) >= 2:
        return float(s[0]), float(s[1])
    
    # Sayısal tipse sadece finite değeri olarak işle
    if isinstance(s, (float, int, complex)):
        return float(s), 0.0
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
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

def _parse_quaternion_from_csv(s) -> quaternion:
    """Virgülle ayrılmış string'i veya sayıyı Quaternion'a dönüştürür."""
    # Eğer zaten quaternion ise doğrudan döndür
    if isinstance(s, quaternion):
        return s
    
    # Sayısal tipse skaler quaternion olarak işle
    if isinstance(s, (float, int, complex)):
        return quaternion(float(s), 0, 0, 0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    parts_str = s.split(',')
    
    # Tüm parçaları float'a dönüştürmeyi dene
    try:
        parts_float = [float(p.strip()) for p in parts_str]
    except ValueError:
        raise ValueError(f"Quaternion bileşenleri sayı olmalı: '{s}'")

    if len(parts_float) == 4:
        return quaternion(*parts_float)
    elif len(parts_float) == 1: # Sadece skaler değer
        return quaternion(parts_float[0], 0, 0, 0)
    else:
        raise ValueError(f"Geçersiz quaternion formatı. 1 veya 4 bileşen bekleniyor: '{s}'")

def _has_comma_format(s) -> bool:
    """String'in virgül içerip içermediğini kontrol eder."""
    if not isinstance(s, str):
        s = str(s)
    return ',' in s

def _parse_neutrosophic_bicomplex(s) -> NeutrosophicBicomplexNumber:
    """
    Parses string or numbers into NeutrosophicBicomplexNumber.
    """
    # Eğer zaten NeutrosophicBicomplexNumber ise doğrudan döndür
    if isinstance(s, NeutrosophicBicomplexNumber):
        return s
    
    # Sayısal tipse tüm bileşenler 0, sadece ilk bileşen değerli
    if isinstance(s, (float, int, complex)):
        values = [float(s)] + [0.0] * 7
        return NeutrosophicBicomplexNumber(*values)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    try:
        parts = s.split(',')
        if len(parts) != 8:
            raise ValueError(f"Expected 8 components, got {len(parts)}")
        values = [float(part.strip()) for part in parts]
        return NeutrosophicBicomplexNumber(*values)
    except Exception as e:
        raise ValueError(f"Invalid NeutrosophicBicomplex format: '{s}' → {e}")


def _parse_octonion(s) -> OctonionNumber:
    """String'i veya sayıyı OctonionNumber'a dönüştürür.
    w,x,y,z,e,f,g,h:e0,e1,e2,e3,e4,e5,e6,e7
    """
    # Eğer zaten OctonionNumber ise doğrudan döndür
    if isinstance(s, OctonionNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) skaler olarak işle
    if isinstance(s, (float, int, complex)):
        scalar = float(s)
        return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
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


def _parse_sedenion(s) -> SedenionNumber:
    """String'i veya sayıyı SedenionNumber'a dönüştürür."""
    # Eğer zaten SedenionNumber ise doğrudan döndür
    if isinstance(s, SedenionNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) skaler olarak işle
    if isinstance(s, (float, int, complex)):
        scalar_val = float(s)
        return SedenionNumber([scalar_val] + [0.0] * 15)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
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

def _parse_pathion(s) -> PathionNumber:
    """String'i veya sayıyı PathionNumber'a dönüştürür."""
    if isinstance(s, PathionNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return PathionNumber(float(s), *[0.0] * 31)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return PathionNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 32:  # Pathion 32 bileşenli olmalı
        try:
            return PathionNumber(*map(float, parts))  # 32 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz pathion bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return PathionNumber(scalar_val, *[0.0] * 31)  # 32 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler pathion değeri: '{s}' -> {e}") from e

    raise ValueError(f"Pathion için 32 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_chingon(s) -> ChingonNumber:
    """String'i veya sayıyı ChingonNumber'a dönüştürür."""
    if isinstance(s, ChingonNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return ChingonNumber(float(s), *[0.0] * 63)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return ChingonNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 64:  # Pathion 32 bileşenli olmalı
        try:
            return ChingonNumber(*map(float, parts))  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz chingon bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return ChingonNumber(scalar_val, *[0.0] * 63)  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler Chingon değeri: '{s}' -> {e}") from e

    raise ValueError(f"Chingon için 64 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_routon(s) -> RoutonNumber:
    """String'i veya sayıyı RoutonNumber'a dönüştürür."""
    if isinstance(s, RoutonNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return RoutonNumber(float(s), *[0.0] * 127)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return RoutonNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 128:  # Pathion 32 bileşenli olmalı
        try:
            return RoutonNumber(*map(float, parts))  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz routon bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return RoutonNumber(scalar_val, *[0.0] * 127)  # 128 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler routon değeri: '{s}' -> {e}") from e

    raise ValueError(f"Routon için 64 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_voudon(s) -> VoudonNumber:
    """String'i veya sayıyı VoudonNumber'a dönüştürür."""
    if isinstance(s, VoudonNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return VoudonNumber(float(s), *[0.0] * 255)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return VoudonNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 256:  # Pathion 32 bileşenli olmalı
        try:
            return VoudonNumber(*map(float, parts))  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz voudon bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return VoudonNumber(scalar_val, *[0.0] * 255)  # 256 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler voudon değeri: '{s}' -> {e}") from e

    raise ValueError(f"Voudon için 64 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")


def _parse_clifford(s) -> CliffordNumber:
    """Algebraik string'i CliffordNumber'a dönüştürür (ör: '1.0+2.0e1')."""
    if isinstance(s, CliffordNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return CliffordNumber({'': float(s)})
    
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip().replace(' ', '').replace('^', '')  # ^ işaretini kaldır
    basis_dict = {}
    
    # Daha iyi regex pattern: +-1.23e12 formatını yakala
    pattern = r'([+-]?)(\d*\.?\d+)(?:e(\d+))?|([+-]?)(?:e(\d+))'
    matches = re.findall(pattern, s)
    
    for match in matches:
        sign_str, coeff_str, basis1, sign_str2, basis2 = match
        
        # Hangi grup match oldu?
        if coeff_str or basis1:
            sign = -1.0 if sign_str == '-' else 1.0
            coeff = float(coeff_str) if coeff_str else 1.0
            basis_key = basis1 if basis1 else ''
        else:
            sign = -1.0 if sign_str2 == '-' else 1.0
            coeff = 1.0
            basis_key = basis2
        
        value = sign * coeff
        basis_dict[basis_key] = basis_dict.get(basis_key, 0.0) + value
    
    # Ayrıca +e1, -e2 gibi ifadeleri yakala
    pattern2 = r'([+-])e(\d+)'
    matches2 = re.findall(pattern2, s)
    
    for sign_str, basis_key in matches2:
        sign = -1.0 if sign_str == '-' else 1.0
        basis_dict[basis_key] = basis_dict.get(basis_key, 0.0) + sign

    return CliffordNumber(basis_dict)


def _parse_dual(s) -> DualNumber:
    """String'i veya sayıyı DualNumber'a dönüştürür."""
    # Eğer zaten DualNumber ise doğrudan döndür
    if isinstance(s, DualNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) real kısım olarak işle
    if isinstance(s, (float, int, complex)):
        return DualNumber(float(s), 0.0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
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


def _parse_splitcomplex(s) -> SplitcomplexNumber:
    """String'i veya sayıyı SplitcomplexNumber'a dönüştürür."""
    # Eğer zaten SplitcomplexNumber ise doğrudan döndür
    if isinstance(s, SplitcomplexNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) real kısım olarak işle
    if isinstance(s, (float, int, complex)):
        return SplitcomplexNumber(float(s), 0.0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
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


def _parse_quaternion(s: str) -> quaternion:
    """Parses user string ('a+bi+cj+dk' or scalar) into a quaternion."""
    s_clean = s.replace(" ", "").lower()
    if not s_clean:
        raise ValueError("Input cannot be empty.")

    try:
        val = float(s_clean)
        return quaternion(val, val, val, val)
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
            
    return quaternion(parts['w'], parts['x'], parts['y'], parts['z'])

def get_random_type(num_iterations: int, fixed_start_raw: str = "0", fixed_add_base_scalar: float = 9.0) -> List[Any]:
    """Generates Keçeci Numbers for a randomly selected type."""
    random_type_choice = random.randint(1, 20)
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutro-Complex", "Hyperreal", 
        "Bicomplex", "Neutro-Bicomplex", "Octonion", "Sedenion", "Clifford", "Dual", "Split-Complex",
        "Pathion", "Chingon", "Routon", "Voudon",
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
        elif kececi_type in [TYPE_Pathion, TYPE_Chingon, TYPE_Routon, TYPE_Voudon, TYPE_SEDENION]:
            # Hypercomplex tipler için ortak çözüm
            try:
                if hasattr(value, 'coeffs'):
                    coeffs = value.coeffs
                else:
                    # Hypercomplex objesini float listesine dönüştürmeye çalış
                    coeffs = [float(getattr(value, f'e{i}', 0.0)) for i in range(32)]  # Varsayılan uzunluk
            except (AttributeError, TypeError):
                # Başarısız olursa, iterable olup olmadığını kontrol et
                try:
                    coeffs = list(value)
                except TypeError:
                    coeffs = [float(value)]  # Skaler değer
            
            return all(math.isclose(c % divisor, 0.0, abs_tol=TOLERANCE) for c in coeffs)
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
        if hasattr(np, 'quaternion') and isinstance(n_input, quaternion):
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

        elif kececi_type == TYPE_QUATERNION:
            if not all(is_near_integer(c) for c in [value.w, value.x, value.y, value.z]):
                return False
            return is_prime(round(value.w))
        
        elif kececi_type == TYPE_OCTONION:
            if hasattr(value, 'coeffs'):
                components = value.coeffs
            else:
                components = list(value)
            if not all(is_near_integer(c) for c in components):
                return False
            return is_prime(round(components[0]))
        
        elif kececi_type == TYPE_SEDENION:
            if hasattr(value, 'coeffs'):
                components = value.coeffs
            else:
                components = list(value)
            if not all(is_near_integer(c) for c in components):
                return False
            return is_prime(round(components[0]))
        
        elif kececi_type in [TYPE_Pathion, TYPE_Chingon, TYPE_Routon, TYPE_Voudon]:
            if hasattr(value, 'coeffs'):
                components = value.coeffs
            else:
                components = list(value)
            if not all(is_near_integer(c) for c in components):
                return False
            return is_prime(round(components[0]))

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
        q0 = quaternion(w, x, y, z)
        
        cw, cx, cy, cz = map(float, c_str.split(','))
        c = quaternion(cw, cx, cy, cz)

        uw, ux, uy, uz = map(float, u_str.split(','))
        u = quaternion(uw, ux, uy, uz)

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
        # Classes
        NeutrosophicNumber,
        NeutrosophicComplexNumber,
        HyperrealNumber,
        BicomplexNumber,
        NeutrosophicBicomplexNumber,
        OctonionNumber,
        Constants,
        SedenionNumber,
        CliffordNumber,
        DualNumber,
        SplitcomplexNumber,
        
    
        # Functions
        get_with_params,
        get_interactive,
        get_random_type,
        _get_integer_representation,
        _parse_quaternion,
        _parse_quaternion_from_csv,
        _parse_complex,
        _parse_bicomplex,
        _parse_universal,
        _parse_octonion,
        _parse_sedenion,
        _parse_neutrosophic,
        _parse_neutrosophic_bicomplex,
        _parse_hyperreal,
        _parse_clifford,
        _parse_dual,
        _parse_splitcomplex,
        kececi_bicomplex_algorithm,
        kececi_bicomplex_advanced,
        generate_kececi_vectorial,
        unified_generator,
        is_prime,
        find_period,
        find_kececi_prime_number,
        plot_numbers,
        print_detailed_report,
        _plot_comparison,
        _find_kececi_zeta_zeros,
        _compute_gue_similarity,
        _load_zeta_zeros,
        analyze_all_types,
        analyze_pair_correlation,
        _gue_pair_correlation,
        _pair_correlation,
        generate_octonion,
        is_quaternion_like,
        is_neutrosophic_like,
        _has_bicomplex_format,
        coeffs,
        convert_to_float,
        safe_add,
        ZERO,
        ONE,
        I,
        J,
        K,
        E,
        F,
        G,
        H,
        _extract_numeric_part,
        _has_comma_format,
        _is_complex_like,
        is_prime_like,
        is_near_integer,
        _plot_component_distribution,
        _parse_pathion,
        _parse_chingon,
        _parse_routon,
        _parse_voudon,
         
    
        # Constants
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
        TYPE_NEUTROSOPHIC_BICOMPLEX,
        TYPE_OCTONION,
        TYPE_SEDENION,
        TYPE_CLIFFORD,
        TYPE_DUAL,
        TYPE_SPLIT_COMPLEX,
        TYPE_Pathion,
        TYPE_Chingon,
        TYPE_Routon,
        TYPE_Voudon,
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

        # 17. Pathion - 32 PARÇALI VİRGÜL
        ('1.0' + ',0.0'*31, '0.1' + ',0.0'*31),
        ('1.0', '0.1'), # Sadece skaler

        # 13. Chingon - 64 PARÇALI VİRGÜL
        ('1.0' + ',0.0'*63, '0.1' + ',0.0'*63),
        ('1.0', '0.1'), # Sadece skaler

        # 13. Routon - 128 PARÇALI VİRGÜL
        ('1.0' + ',0.0'*127, '0.1' + ',0.0'*127),
        ('1.0', '0.1'), # Sadece skaler

        # 13. Voudon - 256 PARÇALI VİRGÜL
        ('1.0' + ',0.0'*255, '0.1' + ',0.0'*255),
        ('1.0', '0.1'), # Sadece skaler
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
        17: "Pathion",
        18: "Chingon",
        19: "Routon",
        20: "Voudon",
    }

    for kececi_type in range(1, 20):
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
    Generates bar charts comparing the performance of Keçeci types in matching Riemann zeta zeros and GUE statistics.
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

    # Generate histogram
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

    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_Voudon):
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
            ask_unit = quaternion(1, 1, 1, 1)

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
            ask_unit = OctonionNumber([1.0] + [0.0] * 7)

        elif kececi_type == TYPE_SEDENION:
            current_value = _parse_sedenion(start_input_raw)
            add_value_typed = _parse_sedenion(add_input_raw)
            ask_unit = SedenionNumber([1.0] + [0.0] * 15)

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

        elif kececi_type == TYPE_Pathion:
            current_value = _parse_pathion(start_input_raw)
            add_value_typed = _parse_pathion(add_input_raw)
            ask_unit = PathionNumber([1.0] + [0.0] * 31)  # Sadece ilk bileşen 1.0

        elif kececi_type == TYPE_Chingon:
            current_value = _parse_chingon(start_input_raw)
            add_value_typed = _parse_chingon(add_input_raw)
            ask_unit = ChingonNumber([1.0] + [0.0] * 63)  # Sadece ilk bileşen 1.0

        elif kececi_type == TYPE_Routon:
            current_value = _parse_routon(start_input_raw)
            add_value_typed = _parse_routon(add_input_raw)
            ask_unit = RoutonNumber([1.0] + [0.0] * 127)  # Sadece ilk bileşen 1.0

        elif kececi_type == TYPE_Voudon:
            current_value = _parse_voudon(start_input_raw)
            add_value_typed = _parse_voudon(add_input_raw)
            ask_unit = VoudonNumber([1.0] + [0.0] * 255)  # Sadece ilk bileşen 1.0

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
    Tüm 20 sayı sistemi için ortak arayüz.
    Keçeci mantığı (ask, bölme, asallık) unified_generator ile uygulanır.
    Sadece toplama değil, koşullu değişimler de yapılır.
    """
    print(f"\n--- Generating Keçeci Sequence: Type {kececi_type_choice}, Steps {iterations} ---")
    print(f"Start: '{start_value_raw}', Addition: '{add_value_raw}'")
    print(f"Include intermediate steps: {include_intermediate_steps}")

    try:
        # unified_generator'ı doğru şekilde çağır
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_value_raw,
            add_input_raw=add_value_raw,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps
        )

        if not generated_sequence:
            print("Sequence generation failed or returned empty.")
            return []

        print(f"Successfully generated {len(generated_sequence)} numbers.")
        
        # Önizleme için ilk 5 ve son 5 elemanı göster
        preview_start = [str(x) for x in generated_sequence[:5]]
        preview_end = [str(x) for x in generated_sequence[-5:]] if len(generated_sequence) > 5 else []
        
        print(f"First 5: {preview_start}")
        if preview_end:
            print(f"Last 5: {preview_end}")
        
        # Keçeci Prime Number kontrolü - ÇOK DAHA GÜVENLİ VERSİYON
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            # Önce kpn'nin doğru tip ve değerde olduğundan emin ol
            try:
                # String karşılaştırması yap - daha güvenli
                kpn_str = str(kpn)
                found_index = None
                
                for i, item in enumerate(generated_sequence):
                    if str(item) == kpn_str:
                        found_index = i
                        break
                
                if found_index is not None:
                    print(f"Keçeci Prime Number found at step {found_index}: {kpn}")
                else:
                    print(f"Keçeci Prime Number found: {kpn} (but not in the main sequence)")
                    
            except Exception as inner_e:
                print(f"Keçeci Prime Number found: {kpn} (error locating index: {inner_e})")
        else:
            print("No Keçeci Prime Number found in the sequence.")
        
        # İstatistikler (opsiyonel)
        if len(generated_sequence) > 0:
            try:
                # Sayısal değerler için basit istatistik
                first_elem = generated_sequence[0]
                if hasattr(first_elem, 'real'):
                    real_parts = [x.real for x in generated_sequence if hasattr(x, 'real')]
                    if real_parts:
                        print(f"Real parts range: {min(real_parts):.3f} to {max(real_parts):.3f}")
                
                # Quaternion özel istatistikler
                if hasattr(first_elem, 'w') and hasattr(first_elem, 'x'):
                    norms = [np.sqrt(x.w**2 + x.x**2 + x.y**2 + x.z**2) for x in generated_sequence 
                           if hasattr(x, 'w') and hasattr(x, 'x')]
                    if norms:
                        print(f"Quaternion norms range: {min(norms):.3f} to {max(norms):.3f}")
                        
            except Exception as stats_e:
                print(f"Statistics calculation skipped: {stats_e}")

        return generated_sequence

    except Exception as e:
        print(f"ERROR during sequence generation: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_interactive() -> Tuple[List[Any], Dict[str, Any]]:
    """
    Interactively gets parameters from the user to generate a Keçeci Numbers sequence.
    This version includes default values for all inputs.
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
    TYPE_Pathion = 17
    TYPE_Chingon = 18
    TYPE_Routon = 19
    TYPE_Voudon = 20

    
    print("\n--- Keçeci Numbers Interactive Generator ---")
    print("  1: Positive Real    2: Negative Real      3: Complex")
    print("  4: Float            5: Rational           6: Quaternion")
    print("  7: Neutrosophic     8: Neutro-Complex     9: Hyperreal")
    print(" 10: Bicomplex       11: Neutro-Bicomplex  12: Octonion")
    print(" 13: Sedenion        14: Clifford          15: Dual")
    print(" 16: Split-Complex   17: Pathion           18: Chingon")
    print(" 19: Routon          20: Voudon")

    
    # Varsayılan değerler
    DEFAULT_TYPE = 3  # Complex
    DEFAULT_STEPS = 40
    DEFAULT_SHOW_DETAILS = "yes"
    
    # Tip seçimi için varsayılan değerler
    default_start_values = {
        1: "2.5",      # Positive Real
        2: "-5.0",     # Negative Real
        3: "1+1j",     # Complex
        4: "3.14",      # Float
        5: "3.5",      # Rational
        6: "1.0,0.0,0.0,0.0",  # Quaternion
        7: "0.6,0.2,0.1",  # Neutrosophic
        8: "1+1j",     # Neutro-Complex
        9: "0.0,0.001",  # Hyperreal
        10: "1.0,0.5,0.3,0.2",  # Bicomplex
        11: "1.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0",  # Neutro-Bicomplex
        12: "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",  # Octonion
        13: "1.0" + ",0.0" * 15,  # Sedenion
        14: "1.0+2.0e1+3.0e12",     # Clifford
        15: "1.0,0.1", # Dual
        16: "1.0,0.5", # Split-Complex
        17: "1.0" + ",0.0" * 31,  # Pathion
        18: "1.0" + ",0.0" * 63,  # Chingon
        19: "1.0" + ",0.0" * 127,  # Routon
        20: "1.0" + ",0.0" * 255,  # Voudon
    }
    
    default_add_values = {
        1: "0.5",      # Positive Real
        2: "-0.5",     # Negative Real
        3: "0.1+0.1j", # Complex
        4: "0.1",      # Float
        5: "0.1",      # Rational
        6: "0.1,0.0,0.0,0.0",  # Quaternion
        7: "0.1,0.0,0.0",  # Neutrosophic
        8: "0.1+0.1j", # Neutro-Complex
        9: "0.0,0.001",  # Hyperreal
        10: "0.1,0.0,0.0,0.0",  # Bicomplex
        11: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0",  # Neutro-Bicomplex
        12: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0",  # Octonion
        13: "0.1" + ",0.0" * 15,  # Sedenion
        14: "1.0+2.0e1+3.0e12",     # Clifford
        15: "0.1,0.0", # Dual
        16: "0.1,0.0",  # Split-Complex
        17: "1.0" + ",0.0" * 31,  # Pathion
        18: "1.0" + ",0.0" * 63,  # Chingon
        19: "1.0" + ",0.0" * 127,  # Routon
        20: "1.0" + ",0.0" * 255,  # Voudon
    }
    
    # Get a valid number type from the user with default
    while True:
        try:
            type_input = input(f"Select a Keçeci Number Type (1-20) [default: {DEFAULT_TYPE}]: ").strip()
            if type_input == "":
                type_choice = DEFAULT_TYPE
                break
            type_choice = int(type_input)
            if 1 <= type_choice <= 20:
                break
            print("Invalid type. Please enter a number between 1 and 20.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # User prompts for the starting value with default
    start_prompts = {
        1: f"Enter Positive Real start [default: '{default_start_values[1]}']: ",
        2: f"Enter Negative Real start [default: '{default_start_values[2]}']: ",
        3: f"Enter Complex start [default: '{default_start_values[3]}']: ",
        4: f"Enter Float start [default: '{default_start_values[4]}']: ",
        5: f"Enter Rational start [default: '{default_start_values[5]}']: ",
        6: f"Enter Quaternion start [default: '{default_start_values[6]}']: ",
        7: f"Enter Neutrosophic start [default: '{default_start_values[7]}']: ",
        8: f"Enter Neutro-Complex start [default: '{default_start_values[8]}']: ",
        9: f"Enter Hyperreal start [default: '{default_start_values[9]}']: ",
        10: f"Enter Bicomplex start [default: '{default_start_values[10]}']: ",
        11: f"Enter Neutro-Bicomplex start [default: '{default_start_values[11]}']: ",
        12: f"Enter Octonion start [default: '{default_start_values[12]}']: ",
        13: f"Enter Sedenion start [default: '{default_start_values[13]}']: ",
        14: f"Enter Clifford start [default: '{default_start_values[14]}']: ",
        15: f"Enter Dual start [default: '{default_start_values[15]}']: ",
        16: f"Enter Split-Complex start [default: '{default_start_values[16]}']: ",
        17: f"Enter Pathion start [default: '{default_start_values[17]}']: ",
        18: f"Enter Chingon start [default: '{default_start_values[18]}']: ",
        19: f"Enter Routon start [default: '{default_start_values[19]}']: ",
        20: f"Enter Voudon start [default: '{default_start_values[20]}']: ",
    }
    
    # User prompts for the increment value with default
    add_prompts = {
        1: f"Enter Positive Real increment [default: '{default_add_values[1]}']: ",
        2: f"Enter Negative Real increment [default: '{default_add_values[2]}']: ",
        3: f"Enter Complex increment [default: '{default_add_values[3]}']: ",
        4: f"Enter Float increment [default: '{default_add_values[4]}']: ",
        5: f"Enter Rational increment [default: '{default_add_values[5]}']: ",
        6: f"Enter Quaternion increment [default: '{default_add_values[6]}']: ",
        7: f"Enter Neutrosophic increment [default: '{default_add_values[7]}']: ",
        8: f"Enter Neutro-Complex increment [default: '{default_add_values[8]}']: ",
        9: f"Enter Hyperreal increment [default: '{default_add_values[9]}']: ",
        10: f"Enter Bicomplex increment [default: '{default_add_values[10]}']: ",
        11: f"Enter Neutro-Bicomplex increment [default: '{default_add_values[11]}']: ",
        12: f"Enter Octonion increment [default: '{default_add_values[12]}']: ",
        13: f"Enter Sedenion increment [default: '{default_add_values[13]}']: ",
        14: f"Enter Clifford increment [default: '{default_add_values[14]}']: ",
        15: f"Enter Dual increment [default: '{default_add_values[15]}']: ",
        16: f"Enter Split-Complex increment [default: '{default_add_values[16]}']: ",
        17: f"Enter Pathion start [default: '{default_start_values[17]}']: ",
        18: f"Enter Chingon start [default: '{default_start_values[18]}']: ",
        19: f"Enter Routon start [default: '{default_start_values[19]}']: ",
        20: f"Enter Voudon start [default: '{default_start_values[20]}']: ",
    }
    
    # Get inputs from the user with defaults
    start_input = input(start_prompts.get(type_choice, "Enter starting value: ")).strip()
    start_input_val_raw = start_input if start_input else default_start_values[type_choice]
    
    add_input = input(add_prompts.get(type_choice, "Enter increment value: ")).strip()
    add_input_val_raw = add_input if add_input else default_add_values[type_choice]
    
    # Steps with default
    steps_input = input(f"Enter number of Keçeci steps [default: {DEFAULT_STEPS}]: ").strip()
    if steps_input:
        try:
            num_kececi_steps = int(steps_input)
            if num_kececi_steps <= 0:
                print("Please enter a positive integer. Using default.")
                num_kececi_steps = DEFAULT_STEPS
        except ValueError:
            print("Invalid input. Using default.")
            num_kececi_steps = DEFAULT_STEPS
    else:
        num_kececi_steps = DEFAULT_STEPS
    
    # Show details with default
    show_details_input = input(f"Do you want to include the intermediate calculation steps? (y/n) [default: {DEFAULT_SHOW_DETAILS}]: ").lower().strip()
    if not show_details_input:
        show_details = (DEFAULT_SHOW_DETAILS == 'y' or DEFAULT_SHOW_DETAILS == 'yes')
    else:
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
    
    print(f"\nUsing parameters: Type={type_choice}, Start='{start_input_val_raw}', Add='{add_input_val_raw}', Steps={num_kececi_steps}, Details={show_details}")
    
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
    if isinstance(obj, quaternion):
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

# Yardımcı fonksiyon: Bileşen dağılımı grafiği
def _plot_component_distribution(ax, elem, all_keys, seq_length=1):
    """Bileşen dağılımını gösterir"""
    if seq_length == 1:
        # Tek veri noktası için bileşen değerleri
        components = []
        values = []
        
        for key in all_keys:
            if key == '':
                components.append('Scalar')
            else:
                components.append(f'e{key}')
            values.append(elem.basis.get(key, 0.0))
        
        bars = ax.bar(components, values, alpha=0.7, color='tab:blue')
        ax.set_title("Component Values")
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
    else:
        # Çoklu veri ama PCA yapılamıyor
        ax.text(0.5, 0.5, f"Need ≥2 data points and ≥2 features\n(Current: {seq_length} points, {len(all_keys)} features)", 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title("Insufficient for PCA")

def plot_numbers(sequence: List[Any], title: str = "Keçeci Number Sequence Analysis"):
    """
    Tüm 20 Keçeci Sayı türü için detaylı görselleştirme sağlar.
    """

    if not sequence:
        print("Sequence is empty. Nothing to plot.")
        return

    # Ensure numpy is available for plotting functions
    try:
        import numpy as np
    except ImportError:
        print("Numpy not installed. Cannot plot effectively.")
        return

    try:
        from sklearn.decomposition import PCA
        use_pca = True
    except ImportError:
        use_pca = False
        print("scikit-learn kurulu değil. PCA olmadan çizim yapılıyor...")


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
    # Check for numpy-quaternion's quaternion type, or a custom one with 'components' or 'w,x,y,z'
    elif isinstance(first_elem, quaternion) or (hasattr(first_elem, 'components') and len(getattr(first_elem, 'components', [])) == 4) or \
         (hasattr(first_elem, 'w') and hasattr(first_elem, 'x') and hasattr(first_elem, 'y') and hasattr(first_elem, 'z')):
        try:
            comp = np.array([
                (q.w, q.x, q.y, q.z) if hasattr(q, 'w') else q.components
                for q in sequence
            ])
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
            ax3.set_xlabel("x");
            ax3.set_ylabel("y");
            ax3.set_zlabel("z")
            ax3.legend()

        except Exception as e:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='red')


    # --- 5. OctonionNumber
    elif isinstance(first_elem, OctonionNumber):
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
        ax4.set_xlabel("e1");
        ax4.set_ylabel("e2");
        ax4.set_zlabel("e3")

    # --- 6. SedenionNumber
    elif isinstance(first_elem, SedenionNumber):
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

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    ax4.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 7. CliffordNumber
    elif isinstance(first_elem, CliffordNumber):
        all_keys = sorted(first_elem.basis.keys(), key=lambda x: (len(x), x))
        values = {k: [elem.basis.get(k, 0.0) for elem in sequence] for k in all_keys}
        scalar = values.get('', [0]*len(sequence))
        vector_keys = [k for k in all_keys if len(k) == 1]

        # GERÇEK özellik sayısını hesapla (sıfır olmayan bileşenler)
        non_zero_features = 0
        for key in all_keys:
            if any(abs(elem.basis.get(key, 0.0)) > 1e-10 for elem in sequence):
                non_zero_features += 1

        # Her zaman 2x2 grid kullan
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        # 1. Grafik: Skaler ve Vektör Bileşenleri
        ax1.plot(scalar, 'o-', label='Scalar', color='black', linewidth=2)

        # Sadece sıfır olmayan vektör bileşenlerini göster
        visible_vectors = 0
        for k in vector_keys:
            if any(abs(v) > 1e-10 for v in values[k]):
                ax1.plot(values[k], 'o-', label=f'Vec {k}', alpha=0.7, linewidth=1.5)
                visible_vectors += 1
            if visible_vectors >= 3:
                break

        ax1.set_title("Scalar & Vector Components Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Grafik: Bivector Magnitude
        bivector_mags = [sum(v**2 for k, v in elem.basis.items() if len(k) == 2)**0.5 for elem in sequence]
        ax2.plot(bivector_mags, 'o-', color='tab:green', linewidth=2, label='Bivector Magnitude')
        ax2.set_title("Bivector Magnitude Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Grafik: PCA - ARTIK PCA GÖSTERİYORUZ
        if use_pca and len(sequence) >= 2 and non_zero_features >= 2:
            try:
                # Tüm bileşenleri içeren matris oluştur
                matrix_data = []
                for elem in sequence:
                    row = []
                    for key in all_keys:
                        row.append(elem.basis.get(key, 0.0))
                    matrix_data.append(row)

                matrix = np.array(matrix_data)

                # PCA uygula
                pca = PCA(n_components=min(2, matrix.shape[1]))
                proj = pca.fit_transform(matrix)

                sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)),
                               cmap='plasma', s=50, alpha=0.8)
                ax3.set_title(f"PCA Projection ({non_zero_features} features)\nVariance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

                cbar = plt.colorbar(sc, ax=ax3)
                cbar.set_label("Time Step")

                ax3.plot(proj[:, 0], proj[:, 1], 'gray', linestyle='--', alpha=0.5)
                ax3.grid(True, alpha=0.3)

            except Exception as e:
                ax3.text(0.5, 0.5, f"PCA Error: {str(e)[:30]}",
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            # PCA yapılamazsa bilgi göster
            ax3.text(0.5, 0.5, f"Need ≥2 data points and ≥2 features\n(Current: {len(sequence)} points, {non_zero_features} features)",
                   ha='center', va='center', transform=ax3.transAxes)
            if not use_pca:
                ax3.text(0.5, 0.65, "Install sklearn for PCA",
                        ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Insufficient for PCA")


    # --- 8. DualNumber
    elif isinstance(first_elem, DualNumber):
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
    elif isinstance(first_elem, SplitcomplexNumber):
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
    elif isinstance(first_elem, NeutrosophicNumber):
        # NeutrosophicNumber sınıfının arayüzünü biliyoruz, hasattr gerekmez
        # Sınıfın public attribute'larına doğrudan erişim
        try:
            t_vals = [x.t for x in sequence]
            i_vals = [x.i for x in sequence]
            f_vals = [x.f for x in sequence]
        except AttributeError:
            # Eğer attribute yoksa, alternatif arayüzleri deneyebiliriz
            # Veya hata fırlatabiliriz
            try:
                t_vals = [x.a for x in sequence]
                i_vals = [x.b for x in sequence]
                f_vals = [0] * len(sequence)  # f yoksa sıfır
            except AttributeError:
                try:
                    t_vals = [x.value for x in sequence]
                    i_vals = [x.indeterminacy for x in sequence]
                    f_vals = [0] * len(sequence)
                except AttributeError:
                    # Hiçbiri yoksa boş liste
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

    # --- 11. NeutrosophicComplexNumber
    elif isinstance(first_elem, NeutrosophicComplexNumber):
        # Sınıfın arayüzünü biliyoruz
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
    elif isinstance(first_elem, HyperrealNumber):
        # Sınıfın arayüzünü biliyoruz
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
    elif isinstance(first_elem, BicomplexNumber):
        # Sınıfın arayüzünü biliyoruz
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

    # --- 14. NeutrosophicBicomplexNumber ---
    elif isinstance(first_elem, NeutrosophicBicomplexNumber):
        # Sınıfın - a, b, c, d, e, f, g, h attribute'ları var
        try:
            # Doğru attribute isimlerini kullanıyoruz
            comps = np.array([
                [float(getattr(x, attr))
                 for attr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
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

    # --- 15. Pathion
    elif isinstance(first_elem, PathionNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(8):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax1.set_title("PathionNumber e0-e7")
        ax1.legend(ncol=2, fontsize=6)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(8, 16):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax2.set_title("e8-e15")
        ax2.legend(ncol=2, fontsize=6)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:red')
        ax3.set_title("Magnitude |p|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    ax4.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 16. Chingon
    elif isinstance(first_elem, ChingonNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(16):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.5)
        ax1.set_title("ChingonNumber e0-e15")
        ax1.legend(ncol=4, fontsize=4)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(16, 32):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.5)
        ax2.set_title("e16-e31")
        ax2.legend(ncol=4, fontsize=4)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:green')
        ax3.set_title("Magnitude |c|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    ax4.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


    # --- 17. Routon
    elif isinstance(first_elem, RoutonNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(32):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.3)
        ax1.set_title("RoutonNumber e0-e31")
        ax1.legend(ncol=4, fontsize=3)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(32, 64):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.3)
        ax2.set_title("e32-e63")
        ax2.legend(ncol=4, fontsize=3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:blue')
        ax3.set_title("Magnitude |r|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    ax4.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 18. Voudon
    elif isinstance(first_elem, VoudonNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(64):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.2)
        ax1.set_title("VoudonNumber e0-e63")
        ax1.legend(ncol=4, fontsize=2)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(64, 128):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.2)
        ax2.set_title("e64-e127")
        ax2.legend(ncol=4, fontsize=2)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:orange')
        ax3.set_title("Magnitude |v|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    ax4.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


    # --- 19. Bilinmeyen tip
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
    print("\nRunning programmatic tests for all 20 number types...")
    
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
        "Dual": TYPE_DUAL, "Splitcomplex": TYPE_SPLIT_COMPLEX, "Pathion": TYPE_Pathion,
        "Chingon": TYPE_Chingon, "Routon": TYPE_Routon, "Voudon": TYPE_Voudon,
    }

    types_to_plot = [
        "Complex", "Quaternion", "Bicomplex", "Neutrosophic Complex", "Hyperreal", "Octonion", "Sedenion", "Clifford", "Dual", "Splitcompllex", 
        "Pathion", "Chingon", "Routon", "Voudon",
    ]
    
    for name, type_id in all_types.items():
        start = "-5" if type_id == TYPE_NEGATIVE_REAL else "2+3j" if type_id in [TYPE_COMPLEX, TYPE_BICOMPLEX] else START_VAL
        
        seq = get_with_params(type_id, STEPS, start, ADD_VAL)
        
        if name in types_to_plot and seq:
            plot_numbers(seq, title=f"Demonstration: {name} Keçeci Numbers")

    print("\n\nDemonstration finished. Plots for selected types are shown.")
    plt.show()
