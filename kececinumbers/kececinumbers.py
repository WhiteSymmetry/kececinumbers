# -*- coding: utf-8 -*-
"""
Keçeci Numbers Module (kececinumbers.py)

This module provides a comprehensive framework for generating, analyzing, and
visualizing Keçeci Numbers across various number systems. It supports 22
distinct types, from standard integers and complex numbers to more exotic
constructs like neutrosophic and bicomplex numbers.

The core of the module is the `unified_generator`, which implements the
specific algorithm for generating Keçeci Number sequences. High-level functions
are available for easy interaction, parameter-based generation, and plotting.

Key Features:
- Generation of 22 types of Keçeci Numbers.
- A robust, unified algorithm for all number types.
- Helper functions for mathematical properties like primality and divisibility.
- Advanced plotting capabilities tailored to each number system.
- Functions for interactive use or programmatic integration.
---

Keçeci Conjecture: Keçeci Varsayımı, Keçeci-Vermutung, Conjecture de Keçeci, Гипотеза Кечеджи, 凯杰西猜想, ケジェジ予想, Keçeci Huds, Keçeci Hudsiye, Keçeci Hudsia, [...]

Keçeci Varsayımı (Keçeci Conjecture) - Önerilen

Her Keçeci Sayı türü için, `unified_generator` fonksiyonu tarafından oluşturulan dizilerin, sonlu adımdan sonra periyodik bir yapıya veya tekrar eden bir asal temsiline (Keçeci Asal Sayısı[...]

Henüz kanıtlanmamıştır ve bu modül bu varsayımı test etmek için bir çerçeve sunar.
"""

# --- Standard Library Imports ---
from __future__ import annotations
from abc import ABC, abstractmethod
import collections
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numbers
from numbers import Number
#from numbers import Real
import numpy as np 
import random
import re
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sympy
from typing import Any, Callable, cast, ClassVar, Dict, Generator, Iterable, List, Optional, overload, Sequence, Tuple, TypeVar, TYPE_CHECKING, Union
import logging
import warnings

# Module logger — library code should not configure logging handlers.
logger = logging.getLogger(__name__)
"""
try:
    # numpy-quaternion kütüphanesinin sınıfını yüklemeye çalış. Artık bu modüle ihtiyaç kalmadı
    # conda install -c conda-forge quaternion # pip install numpy-quaternion
    from quaternion import quaternion as quaternion  # type: ignore
except Exception:
    # Eğer yoksa `quaternion` isimli sembolü None yap, kodun diğer yerleri bunu kontrol edebilir
    quaternion = None
    logger.warning("numpy-quaternion paketine ulaşılamadı — quaternion tip desteği devre dışı bırakıldı.")
"""

# Better type definition
Numeric = Union[int, float, complex]
# Number = Union[int, float, complex]
_T = TypeVar("_T", bound="BaseNumber")

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
TYPE_PATHION = 17
TYPE_CHINGON = 18
TYPE_ROUTON = 19
TYPE_VOUDON = 20
TYPE_SUPERREAL = 21
TYPE_TERNARY = 22
TYPE_HYPERCOMPLEX = 23


# ==============================================================================
# --- CUSTOM NUMBER CLASS DEFINITIONS ---
# ==============================================================================

"""
@property
def coeffs(self):
    #Dörtlü sayının tüm katsayılarını liste olarak döndürür.
    return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]
"""
@dataclass
class BaseNumber(ABC):
    """Tüm Keçeci sayı tipleri için ortak arayüz."""
    
    # Using init=False since we set it in __init__
    _value: Numeric = field(init=False)
    
    def __init__(self, value: Numeric) -> None:
        self._value = self._coerce(value)
    
    @staticmethod
    def _coerce(v: Numeric) -> Numeric:
        """Convert input to valid numeric type."""
        if isinstance(v, (int, float, complex)):
            return v
        raise TypeError(f"Geçersiz sayı tipi: {type(v)}")
    
    @property
    @abstractmethod
    def coeffs(self) -> np.ndarray:
        """NumPy array katsayıları."""
        pass
    
    @property
    def value(self) -> Numeric:
        return self._value
    
    # ------------------------------------------------------------------ #
    # Helper method to extract numeric value
    # ------------------------------------------------------------------ #
    def _extract_numeric(self, other: BaseNumber | Numeric) -> Numeric:
        """Extract numeric value from either BaseNumber or numeric type."""
        if isinstance(other, BaseNumber):
            return other._value
        elif isinstance(other, (int, float, complex)):
            return other
        else:
            raise TypeError(f"Unsupported type: {type(other)}")
    
    # ------------------------------------------------------------------ #
    # Matematiksel operator overload'ları
    # ------------------------------------------------------------------ #
    
    def __add__(self: _T, other: BaseNumber | Numeric) -> _T:
        other_val = self._extract_numeric(other)
        # Type narrowing for arithmetic operations
        result: Numeric
        if isinstance(self._value, complex) or isinstance(other_val, complex):
            result = self._value + other_val
        else:
            # For int/float combinations
            result = self._value + other_val
        return self.__class__(result)
    
    def __radd__(self: _T, other: Numeric) -> _T:
        return self.__add__(other)
    
    def __sub__(self: _T, other: BaseNumber | Numeric) -> _T:
        other_val = self._extract_numeric(other)
        result: Numeric
        if isinstance(self._value, complex) or isinstance(other_val, complex):
            result = self._value - other_val
        else:
            result = self._value - other_val
        return self.__class__(result)
    
    def __rsub__(self: _T, other: Numeric) -> _T:
        other_val = other
        result: Numeric
        if isinstance(self._value, complex) or isinstance(other_val, complex):
            result = other_val - self._value
        else:
            result = other_val - self._value
        return self.__class__(result)
    
    def __mul__(self: _T, other: BaseNumber | Numeric) -> _T:
        other_val = self._extract_numeric(other)
        result: Numeric
        if isinstance(self._value, complex) or isinstance(other_val, complex):
            result = self._value * other_val
        else:
            result = self._value * other_val
        return self.__class__(result)
    
    def __rmul__(self: _T, other: Numeric) -> _T:
        return self.__mul__(other)
    
    def __truediv__(self: _T, other: BaseNumber | Numeric) -> _T:
        other_val = self._extract_numeric(other)
        if other_val == 0:
            raise ZeroDivisionError("division by zero")
        result: Numeric
        if isinstance(self._value, complex) or isinstance(other_val, complex):
            result = self._value / other_val
        else:
            result = self._value / other_val
        return self.__class__(result)
    
    def __rtruediv__(self: _T, other: Numeric) -> _T:
        if self._value == 0:
            raise ZeroDivisionError("division by zero")
        result: Numeric
        if isinstance(self._value, complex) or isinstance(other, complex):
            result = other / self._value
        else:
            result = other / self._value
        return self.__class__(result)
    
    def __mod__(self: _T, divisor: Numeric) -> _T:
        if isinstance(self._value, complex) or isinstance(divisor, complex):
            raise TypeError("Modulo not supported for complex numbers")
        # For int/float
        if isinstance(self._value, (int, float)) and isinstance(divisor, (int, float)):
            result: Numeric = self._value % divisor
        else:
            raise TypeError(f"Unsupported types for modulo: {type(self._value)}, {type(divisor)}")
        return self.__class__(result)
    
    # ------------------------------------------------------------------ #
    # Karşılaştırmalar
    # ------------------------------------------------------------------ #
    
    def _to_real(self) -> float:
        """Sadece gerçek sayılara dönüştürür."""
        if isinstance(self._value, complex):
            if abs(self._value.imag) > 1e-12:
                raise ValueError("Cannot compare complex with non-zero imaginary part")
            return self._value.real
        # Convert int/float to float
        return float(self._value)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseNumber):
            return NotImplemented  # Return the value, not type annotation
        
        try:
            return math.isclose(
                self._to_real(), 
                other._to_real(), 
                rel_tol=1e-12
            )
        except (ValueError, TypeError):
            return NotImplemented  # Return the value, not type annotation
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"
    
    # ------------------------------------------------------------------ #
    # Alt sınıfların doldurması gereken soyut metodlar
    # ------------------------------------------------------------------ #
    
    def components(self) -> list[Numeric]:
        """Bileşen listesini (Python list) döndürür."""
        coeffs = self.coeffs
        if isinstance(coeffs, np.ndarray):
            return coeffs.tolist()
        try:
            return list(coeffs)  # type: ignore[arg-type]
        except Exception:
            return [coeffs]  # type: ignore[list-item]
    
    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy'nin `linalg.norm` fonksiyonu C-hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))
    
    def __hash__(self) -> int:
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))
    
    def phase(self) -> float:
        """
        Güvenli phase hesaplayıcı:
        - Eğer value complex ise imag/real üzerinden phase hesaplanır.
        - Eğer coeffs varsa, ilk bileşenin complex olması durumunda phase döner.
        - Diğer durumlarda 0.0 döndürür (tanımsız phase için güvenli fallback).
        """
        try:
            # If underlying value is complex
            if isinstance(self._value, complex):
                return math.atan2(self._value.imag, self._value.real)
            # If there's coeffs, use the first coefficient
            coeffs = self.coeffs
            if isinstance(coeffs, (list, tuple, np.ndarray)) and len(coeffs) > 0:
                first = coeffs[0]
                if isinstance(first, (complex, np.complex128, np.complex64)):
                    return math.atan2(first.imag, first.real)
            # If value has real/imag attributes (like some custom complex types)
            if hasattr(self._value, 'real') and hasattr(self._value, 'imag'):
                return math.atan2(self._value.imag, self._value.real)  # type: ignore[attr-defined]
        except Exception:
            pass
        return 0.0

class ComplexNumber:
    """Complex number implementation."""
    
    def __init__(self, real: float, imag: float = 0.0):
        self._real = float(real)
        self._imag = float(imag)
    
    @property
    def real(self) -> float:
        return self._real
    
    @property
    def imag(self) -> float:
        return self._imag
    
    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real + float(other), self.imag)
        elif isinstance(other, complex):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real - float(other), self.imag)
        elif isinstance(other, complex):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return ComplexNumber(float(other) - self.real, -self.imag)
        elif isinstance(other, complex):
            return ComplexNumber(other.real - self.real, other.imag - self.imag)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return ComplexNumber(real, imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real * float(other), self.imag * float(other))
        elif isinstance(other, complex):
            return self * ComplexNumber(other.real, other.imag)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            denominator = other.norm() ** 2
            if denominator == 0:
                raise ZeroDivisionError("Division by zero")
            conj = other.conjugate()
            result = self * conj
            return ComplexNumber(result.real / denominator, result.imag / denominator)
        elif isinstance(other, (int, float)):
            if float(other) == 0:
                raise ZeroDivisionError("Division by zero")
            return ComplexNumber(self.real / float(other), self.imag / float(other))
        elif isinstance(other, complex):
            return self / ComplexNumber(other.real, other.imag)
        return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return ComplexNumber(float(other), 0) / self
        elif isinstance(other, complex):
            return ComplexNumber(other.real, other.imag) / self
        return NotImplemented
    
    def __neg__(self):
        return ComplexNumber(-self.real, -self.imag)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        return self.norm()
    
    def __eq__(self, other):
        if isinstance(other, ComplexNumber):
            return math.isclose(self.real, other.real) and math.isclose(self.imag, other.imag)
        elif isinstance(other, (int, float)):
            return math.isclose(self.real, float(other)) and math.isclose(self.imag, 0)
        elif isinstance(other, complex):
            return math.isclose(self.real, other.real) and math.isclose(self.imag, other.imag)
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((round(self.real, 12), round(self.imag, 12)))
    
    def __repr__(self):
        return f"ComplexNumber({self.real}, {self.imag})"
    
    def __str__(self):
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {-self.imag}i"
    
    def norm(self) -> float:
        return math.sqrt(self.real**2 + self.imag**2)
    
    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)
    
    def to_complex(self) -> complex:
        return complex(self.real, self.imag)
    
    def to_hypercomplex(self) -> 'HypercomplexNumber':
        """Convert to HypercomplexNumber."""
        return HypercomplexNumber(self.real, self.imag, dimension=2)

# =============================================
# Unified Number System: HypercomplexNumber
# =============================================

class HypercomplexNumber:
    """
    Unified hypercomplex number implementation for all dimensions.
    Supports: Real (1), Complex (2), Quaternion (4), Octonion (8), etc.
    """
    
    # Dimension names
    DIM_NAMES = {
        1: "Real",
        2: "Complex",
        4: "Quaternion",
        8: "Octonion",
        16: "Sedenion",
        32: "Pathion",
        64: "Chingon",
        128: "Routon",
        256: "Voudon"
    }
    
    def __init__(self, *components: float, dimension: Optional[int] = None):
        """
        Initialize hypercomplex number.
        
        Args:
            *components: Number components
            dimension: Dimension (power of 2). If None, inferred from components.
        """
        if dimension is None:
            # Find smallest power of 2 >= len(components)
            n = len(components)
            dimension = 1
            while dimension < n and dimension < 256:
                dimension <<= 1
        
        if dimension & (dimension - 1) != 0 or dimension < 1:
            raise ValueError(f"Dimension must be power of 2 (1-256), got {dimension}")
        
        self.dimension = dimension
        
        # Pad or truncate components
        if len(components) < dimension:
            coeffs_tuple = components + (0.0,) * (dimension - len(components))
        elif len(components) > dimension:
            coeffs_tuple = components[:dimension]
        else:
            coeffs_tuple = components
        
        self.coeffs: Tuple[float, ...] = coeffs_tuple
        
        # Type name
        self.type_name = self.DIM_NAMES.get(dimension, f"Hypercomplex{dimension}")
    
    @classmethod
    def from_real(cls, value: float) -> 'HypercomplexNumber':
        """Create from a real number."""
        return cls(value, dimension=1)
    
    @classmethod
    def from_complex(cls, real: float, imag: float) -> 'HypercomplexNumber':
        """Create from complex components."""
        return cls(real, imag, dimension=2)
    
    @classmethod
    def from_quaternion(cls, w: float, x: float, y: float, z: float) -> 'HypercomplexNumber':
        """Create from quaternion components."""
        return cls(w, x, y, z, dimension=4)
    
    @classmethod
    def from_octonion(cls, *coeffs: float) -> 'HypercomplexNumber':
        """Create from octonion components."""
        if len(coeffs) != 8:
            coeffs = coeffs + (0.0,) * (8 - len(coeffs))
        return cls(*coeffs, dimension=8)
    
    @classmethod
    def from_any(cls, value: Any) -> 'HypercomplexNumber':
        """Create from any numeric type."""
        if isinstance(value, HypercomplexNumber):
            return value
        elif isinstance(value, (int, float)):
            return cls.from_real(float(value))
        elif isinstance(value, complex):
            return cls.from_complex(value.real, value.imag)
        elif isinstance(value, ComplexNumber):
            return cls.from_complex(value.real, value.imag)
        elif isinstance(value, (list, tuple)):
            return cls(*value)
        else:
            try:
                return cls.from_real(float(value))
            except:
                raise ValueError(f"Cannot convert {type(value)} to HypercomplexNumber")
    
    @property
    def real(self) -> float:
        """Real part (first component)."""
        return self.coeffs[0]
    
    @property
    def imag(self) -> float:
        """Imaginary part (for complex numbers)."""
        if self.dimension >= 2:
            return self.coeffs[1]
        return 0.0
    
    def __len__(self):
        return self.dimension
    
    def __getitem__(self, idx):
        return self.coeffs[idx]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __add__(self, other):
        # Convert other to HypercomplexNumber if needed
        if not isinstance(other, HypercomplexNumber):
            try:
                other = self.from_any(other)
            except:
                return NotImplemented
        
        # Handle different dimensions
        if self.dimension != other.dimension:
            common_dim = max(self.dimension, other.dimension)
            self_padded = self.pad_to_dimension(common_dim)
            other_padded = other.pad_to_dimension(common_dim)
            return self_padded + other_padded
        
        new_coeffs = tuple(a + b for a, b in zip(self.coeffs, other.coeffs))
        return HypercomplexNumber(*new_coeffs, dimension=self.dimension)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        # Convert other to HypercomplexNumber if needed
        if not isinstance(other, HypercomplexNumber):
            try:
                other = self.from_any(other)
            except:
                return NotImplemented
        
        # Handle different dimensions
        if self.dimension != other.dimension:
            common_dim = max(self.dimension, other.dimension)
            self_padded = self.pad_to_dimension(common_dim)
            other_padded = other.pad_to_dimension(common_dim)
            return self_padded - other_padded
        
        new_coeffs = tuple(a - b for a, b in zip(self.coeffs, other.coeffs))
        return HypercomplexNumber(*new_coeffs, dimension=self.dimension)
    
    def __rsub__(self, other):
        # Convert other to HypercomplexNumber
        try:
            other_num = self.from_any(other)
            return other_num - self
        except:
            return NotImplemented
    
    def _multiply_complex(self, other: 'HypercomplexNumber') -> 'HypercomplexNumber':
        """Complex multiplication (dimension 2)."""
        a, b = self.coeffs
        c, d = other.coeffs
        real = a*c - b*d
        imag = a*d + b*c
        return HypercomplexNumber(real, imag, dimension=2)
    
    def _multiply_quaternion(self, other: 'HypercomplexNumber') -> 'HypercomplexNumber':
        """Quaternion multiplication (dimension 4)."""
        w1, x1, y1, z1 = self.coeffs
        w2, x2, y2, z2 = other.coeffs
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return HypercomplexNumber(w, x, y, z, dimension=4)
    
    def __mul__(self, other):
        # Convert other to HypercomplexNumber if needed
        if not isinstance(other, HypercomplexNumber):
            try:
                other = self.from_any(other)
            except:
                return NotImplemented
        
        # Handle different dimensions
        if self.dimension != other.dimension:
            common_dim = max(self.dimension, other.dimension)
            self_padded = self.pad_to_dimension(common_dim)
            other_padded = other.pad_to_dimension(common_dim)
            return self_padded * other_padded
        
        # Different multiplication rules based on dimension
        if self.dimension == 1:
            # Real multiplication
            return HypercomplexNumber(self.coeffs[0] * other.coeffs[0], dimension=1)
        
        elif self.dimension == 2:
            # Complex multiplication
            return self._multiply_complex(other)
        
        elif self.dimension == 4:
            # Quaternion multiplication
            return self._multiply_quaternion(other)
        
        else:
            # For higher dimensions, use component-wise as fallback
            warnings.warn(f"Using component-wise multiplication for {self.type_name}", RuntimeWarning)
            new_coeffs = tuple(a * b for a, b in zip(self.coeffs, other.coeffs))
            return HypercomplexNumber(*new_coeffs, dimension=self.dimension)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        # Convert other to HypercomplexNumber if needed
        if not isinstance(other, HypercomplexNumber):
            try:
                other = self.from_any(other)
            except:
                return NotImplemented
        
        # Use inverse for division
        return self * other.inverse()
    
    def __rtruediv__(self, other):
        # Convert other to HypercomplexNumber
        try:
            other_num = self.from_any(other)
            return other_num / self
        except:
            return NotImplemented
    
    def __neg__(self):
        new_coeffs = tuple(-c for c in self.coeffs)
        return HypercomplexNumber(*new_coeffs, dimension=self.dimension)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        return self.norm()
    
    def __eq__(self, other):
        if not isinstance(other, HypercomplexNumber):
            try:
                other = self.from_any(other)
            except:
                return False
        
        if self.dimension != other.dimension:
            return False
        
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self.coeffs, other.coeffs))
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(tuple(round(c, 12) for c in self.coeffs))
    
    def __repr__(self):
        return f"HypercomplexNumber({', '.join(map(str, self.coeffs))}, dimension={self.dimension})"
    
    def __str__(self):
        if self.dimension == 1:
            return f"{self.type_name}({self.coeffs[0]})"
        elif self.dimension == 2:
            a, b = self.coeffs
            if b >= 0:
                return f"{self.type_name}({a} + {b}i)"
            else:
                return f"{self.type_name}({a} - {-b}i)"
        elif self.dimension <= 8:
            non_zero = [(i, c) for i, c in enumerate(self.coeffs) if abs(c) > 1e-10]
            if not non_zero:
                return f"{self.type_name}(0)"
            
            parts = []
            for i, c in non_zero:
                if i == 0:
                    parts.append(f"{c:.4f}")
                else:
                    sign = "+" if c >= 0 else "-"
                    parts.append(f"{sign} {abs(c):.4f}e{i}")
            return f"{self.type_name}({' '.join(parts)})"
        else:
            return f"{self.type_name}[real={self.real:.4f}, norm={self.norm():.4f}, dim={self.dimension}]"
    
    def norm(self) -> float:
        """Euclidean norm."""
        return math.sqrt(sum(c**2 for c in self.coeffs))
    
    def conjugate(self):
        """Conjugate (negate all imaginary parts)."""
        if self.dimension == 1:
            return self
        
        new_coeffs = list(self.coeffs)
        for i in range(1, self.dimension):
            new_coeffs[i] = -new_coeffs[i]
        
        return HypercomplexNumber(*new_coeffs, dimension=self.dimension)
    
    def inverse(self):
        """Multiplicative inverse."""
        norm_sq = self.norm() ** 2
        if norm_sq == 0:
            raise ZeroDivisionError("Cannot invert zero element")
        
        conj = self.conjugate()
        new_coeffs = tuple(c / norm_sq for c in conj.coeffs)
        return HypercomplexNumber(*new_coeffs, dimension=self.dimension)
    
    def pad_to_dimension(self, new_dim: int) -> 'HypercomplexNumber':
        """Pad to higher dimension with zeros."""
        if new_dim < self.dimension:
            raise ValueError(f"Cannot pad to smaller dimension: {new_dim} < {self.dimension}")
        
        if new_dim == self.dimension:
            return self
        
        new_coeffs = self.coeffs + (0.0,) * (new_dim - self.dimension)
        return HypercomplexNumber(*new_coeffs, dimension=new_dim)
    
    def truncate_to_dimension(self, new_dim: int) -> 'HypercomplexNumber':
        """Truncate to smaller dimension."""
        if new_dim > self.dimension:
            raise ValueError(f"Cannot truncate to larger dimension: {new_dim} > {self.dimension}")
        
        if new_dim == self.dimension:
            return self
        
        new_coeffs = self.coeffs[:new_dim]
        return HypercomplexNumber(*new_coeffs, dimension=new_dim)
    
    def to_list(self) -> List[float]:
        """Convert to Python list."""
        return list(self.coeffs)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.coeffs, dtype=np.float64)
    
    def copy(self) -> 'HypercomplexNumber':
        """Generate a copy."""
        return HypercomplexNumber(*self.coeffs, dimension=self.dimension)
    
    def __float__(self):
        """Convert to float (returns real part)."""
        return float(self.real)
    
    def to_complex(self) -> complex:
        """Convert to Python complex if possible."""
        if self.dimension >= 2:
            return complex(self.coeffs[0], self.coeffs[1])
        return complex(self.coeffs[0], 0.0)
    
    def to_hypercomplex(self) -> 'HypercomplexNumber':
        """Convert to HypercomplexNumber (identity for this class)."""
        return self


class quaternion(BaseNumber):
    """
    BaseNumber'dan türetilmiş Kuaterniyon sınıfı
    """
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        # BaseNumber'ı initialize et - value olarak skalar kısmı kullan
        super().__init__(self.w)
    
    @property
    def coeffs(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])
    
    # BaseNumber'dan gelen metodlar otomatik olarak çalışır:
    # - magnitude() -> norm() ile aynı işi yapar
    # - phase() -> quaternion için anlamlı olmayabilir, override edilebilir
    # - components() -> [w, x, y, z] döndürür

    # ------------------------------------------------------------------ #
    # Helper method to extract quaternion from various types
    # ------------------------------------------------------------------ #
    def _extract_quat(self, other: Any) -> 'quaternion':
        """Extract quaternion value from various input types."""
        if isinstance(other, quaternion):
            return other
        elif isinstance(other, (int, float)):
            # Scalar -> quaternion with only real part
            return quaternion(float(other), 0.0, 0.0, 0.0)
        elif isinstance(other, BaseNumber):
            # Try to convert BaseNumber to quaternion
            try:
                # Get the scalar value from BaseNumber
                scalar = other._extract_numeric(other)
                if isinstance(scalar, (int, float)):
                    return quaternion(float(scalar), 0.0, 0.0, 0.0)
                elif isinstance(scalar, complex):
                    return quaternion(float(scalar.real), float(scalar.imag), 0.0, 0.0)
            except (AttributeError, TypeError):
                pass
        raise TypeError(f"Cannot convert {type(other)} to quaternion")
    
    
    @classmethod
    def from_axis_angle(cls, axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float) -> 'quaternion':
        """Eksen-açı gösteriminden kuaterniyon oluşturur."""
        axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm == 0:
            raise ValueError("Eksen vektörü sıfır olamaz")
        
        axis = axis / axis_norm
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        
        return cls(
            w=math.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
        )
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float, 
                   order: str = 'zyx') -> 'quaternion':
        """
        Euler açılarından kuaterniyon oluşturur.
        
        Args:
            roll: X ekseni etrafında dönme (radyan)
            pitch: Y ekseni etrafında dönme (radyan)
            yaw: Z ekseni etrafında dönme (radyan)
            order: Dönme sırası ('zyx', 'xyz', 'yxz', vb.)
        
        Returns:
            quaternion: Kuaterniyon nesnesi
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        if order == 'zyx':  # Yaw, Pitch, Roll
            w = cy * cp * cr + sy * sp * sr
            x = cy * cp * sr - sy * sp * cr
            y = sy * cp * sr + cy * sp * cr
            z = sy * cp * cr - cy * sp * sr
        elif order == 'xyz':  # Roll, Pitch, Yaw
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
        else:
            raise ValueError(f"Desteklenmeyen dönme sırası: {order}")
        
        return cls(w, x, y, z)
    
    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> 'quaternion':
        """
        Dönüşüm matrisinden kuaterniyon oluşturur.
        
        Args:
            R: 3x3 dönüşüm matrisi
        
        Returns:
            quaternion: Kuaterniyon nesnesi
        """
        if R.shape != (3, 3):
            raise ValueError("Matris 3x3 boyutunda olmalıdır")
        
        trace = np.trace(R)
        
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return cls(w, x, y, z).normalized()
    
    # ------------------------------------------------------------------ #
    # Quaternion-specific methods (keep all your existing methods)
    # ------------------------------------------------------------------ #
    
    def conjugate(self) -> 'quaternion':
        """Kuaterniyonun eşleniğini döndürür."""
        return quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        """Kuaterniyonun normunu döndürür."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def magnitude(self) -> float:
        """BaseNumber compatibility - same as norm()."""
        return self.norm()
    
    def normalized(self) -> 'quaternion':
        """Normalize edilmiş kuaterniyonu döndürür."""
        n = self.norm()
        if n == 0:
            return quaternion(1, 0, 0, 0)
        return quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def inverse(self) -> 'quaternion':
        """Kuaterniyonun tersini döndürür."""
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq == 0:
            return quaternion(1, 0, 0, 0)
        conj = self.conjugate()
        return quaternion(conj.w/norm_sq, conj.x/norm_sq, conj.y/norm_sq, conj.z/norm_sq)
    
    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Kuaterniyonu eksen-açı gösterimine dönüştürür.
        
        Returns:
            Tuple[np.ndarray, float]: (eksen, açı)
        """
        if abs(self.w) > 1:
            q = self.normalized()
        else:
            q = self
        
        angle = 2 * math.acos(q.w)
        
        if abs(angle) < 1e-10:
            return np.array([1.0, 0.0, 0.0]), 0.0
        
        s = math.sqrt(1 - q.w**2)
        if s < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([q.x/s, q.y/s, q.z/s])
        
        return axis, angle
    
    def to_euler(self, order: str = 'zyx') -> Tuple[float, float, float]:
        """
        Kuaterniyonu Euler açılarına dönüştürür.
        
        Args:
            order: Dönme sırası
        
        Returns:
            Tuple[float, float, float]: (roll, pitch, yaw)
        """
        q = self.normalized()
        
        if order == 'zyx':  # Yaw, Pitch, Roll
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
        else:
            raise ValueError(f"Desteklenmeyen dönme sırası: {order}")
    
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Kuaterniyonu dönüşüm matrisine dönüştürür.
        
        Returns:
            np.ndarray: 3x3 dönüşüm matrisi
        """
        q = self.normalized()
        
        # 3x3 dönüşüm matrisi
        R = np.zeros((3, 3))
        
        # Matris elemanlarını hesapla
        R[0, 0] = 1 - 2*(q.y**2 + q.z**2)
        R[0, 1] = 2*(q.x*q.y - q.w*q.z)
        R[0, 2] = 2*(q.x*q.z + q.w*q.y)
        
        R[1, 0] = 2*(q.x*q.y + q.w*q.z)
        R[1, 1] = 1 - 2*(q.x**2 + q.z**2)
        R[1, 2] = 2*(q.y*q.z - q.w*q.x)
        
        R[2, 0] = 2*(q.x*q.z - q.w*q.y)
        R[2, 1] = 2*(q.y*q.z + q.w*q.x)
        R[2, 2] = 1 - 2*(q.x**2 + q.y**2)
        
        return R
    
    def rotate_vector(self, v: Union[List[float], Tuple[float, float, float], np.ndarray]) -> np.ndarray:
        """
        Vektörü kuaterniyon ile döndürür.
        
        Args:
            v: Döndürülecek 3 boyutlu vektör
        
        Returns:
            np.ndarray: Döndürülmüş vektör
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (3,):
            raise ValueError("Vektör 3 boyutlu olmalıdır")
        
        q = self.normalized()
        q_vec = np.array([q.x, q.y, q.z])
        q_w = q.w
        
        # Kuaterniyon çarpımı ile döndürme
        v_rot = v + 2 * np.cross(q_vec, np.cross(q_vec, v) + q_w * v)
        return v_rot
    
    def slerp(self, other: 'quaternion', t: float) -> 'quaternion':
        """
        Küresel lineer interpolasyon (SLERP) yapar.
        
        Args:
            other: Hedef kuaterniyon
            t: İnterpolasyon parametresi [0, 1]
        
        Returns:
            quaternion: İnterpole edilmiş kuaterniyon
        """
        if t <= 0:
            return self.normalized()
        if t >= 1:
            return other.normalized()
        
        q1 = self.normalized()
        q2 = other.normalized()
        
        # Nokta çarpım
        cos_half_theta = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
        
        # Eğer q1 ve q2 aynı yöndeyse
        if abs(cos_half_theta) >= 1.0:
            return q1
        
        # Eğer negatif nokta çarpım, kuaterniyonları ters çevir
        if cos_half_theta < 0:
            q2 = quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            cos_half_theta = -cos_half_theta
        
        half_theta = math.acos(cos_half_theta)
        sin_half_theta = math.sqrt(1.0 - cos_half_theta**2)
        
        if abs(sin_half_theta) < 1e-10:
            return quaternion(
                q1.w * 0.5 + q2.w * 0.5,
                q1.x * 0.5 + q2.x * 0.5,
                q1.y * 0.5 + q2.y * 0.5,
                q1.z * 0.5 + q2.z * 0.5
            ).normalized()
        
        ratio_a = math.sin((1 - t) * half_theta) / sin_half_theta
        ratio_b = math.sin(t * half_theta) / sin_half_theta
        
        return quaternion(
            q1.w * ratio_a + q2.w * ratio_b,
            q1.x * ratio_a + q2.x * ratio_b,
            q1.y * ratio_a + q2.y * ratio_b,
            q1.z * ratio_a + q2.z * ratio_b
        ).normalized()
    
    # ------------------------------------------------------------------ #
    # Operator overloads - Liskov compliant
    # ------------------------------------------------------------------ #
    
    def __add__(self, other: Union['quaternion', BaseNumber, int, float, complex]) -> 'quaternion':
        """Quaternion addition - Liskov compliant."""
        try:
            q_other = self._extract_quat(other)
            return quaternion(
                self.w + q_other.w,
                self.x + q_other.x,
                self.y + q_other.y,
                self.z + q_other.z
            )
        except TypeError:
            return NotImplemented
    
    def __radd__(self, other: Union[BaseNumber, int, float, complex]) -> 'quaternion':
        """Right addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['quaternion', BaseNumber, int, float, complex]) -> 'quaternion':
        """Quaternion subtraction - Liskov compliant."""
        try:
            q_other = self._extract_quat(other)
            return quaternion(
                self.w - q_other.w,
                self.x - q_other.x,
                self.y - q_other.y,
                self.z - q_other.z
            )
        except TypeError:
            return NotImplemented
    
    def __rsub__(self, other: Union[BaseNumber, int, float, complex]) -> 'quaternion':
        """Right subtraction."""
        try:
            q_other = self._extract_quat(other)
            return quaternion(
                q_other.w - self.w,
                q_other.x - self.x,
                q_other.y - self.y,
                q_other.z - self.z
            )
        except TypeError:
            return NotImplemented
    
    def __mul__(self, other: Union['quaternion', BaseNumber, int, float, complex]) -> 'quaternion':
        """Quaternion multiplication - Liskov compliant."""
        try:
            q_other = self._extract_quat(other)
            
            # Hamilton product for quaternion-quaternion multiplication
            w = self.w * q_other.w - self.x * q_other.x - self.y * q_other.y - self.z * q_other.z
            x = self.w * q_other.x + self.x * q_other.w + self.y * q_other.z - self.z * q_other.y
            y = self.w * q_other.y - self.x * q_other.z + self.y * q_other.w + self.z * q_other.x
            z = self.w * q_other.z + self.x * q_other.y - self.y * q_other.x + self.z * q_other.w
            
            return quaternion(w, x, y, z)
        except TypeError:
            return NotImplemented
    
    def __rmul__(self, other: Union[BaseNumber, int, float, complex]) -> 'quaternion':
        """Right multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['quaternion', BaseNumber, int, float, complex]) -> 'quaternion':
        """Quaternion division - Liskov compliant."""
        try:
            q_other = self._extract_quat(other)
            
            # Division as multiplication by inverse
            norm_sq = q_other.w**2 + q_other.x**2 + q_other.y**2 + q_other.z**2
            if norm_sq == 0:
                raise ZeroDivisionError("division by zero quaternion")
            
            # Compute inverse of other quaternion
            inv_w = q_other.w / norm_sq
            inv_x = -q_other.x / norm_sq
            inv_y = -q_other.y / norm_sq
            inv_z = -q_other.z / norm_sq
            
            # Multiply self by inverse
            return self * quaternion(inv_w, inv_x, inv_y, inv_z)
        except TypeError:
            return NotImplemented
    
    def __rtruediv__(self, other: Union[BaseNumber, int, float, complex]) -> 'quaternion':
        """Right division."""
        try:
            q_other = self._extract_quat(other)
            return q_other / self
        except TypeError:
            return NotImplemented
    
    def __neg__(self) -> 'quaternion':
        """Negate quaternion."""
        return quaternion(-self.w, -self.x, -self.y, -self.z)
    
    # ------------------------------------------------------------------ #
    # Comparison operators - Liskov compliant
    # ------------------------------------------------------------------ #
    
    def __eq__(self, other: object) -> bool:
        """Equality check - Liskov compliant."""
        if not isinstance(other, quaternion):
            return False
        
        return (math.isclose(self.w, other.w) and 
                math.isclose(self.x, other.x) and 
                math.isclose(self.y, other.y) and 
                math.isclose(self.z, other.z))
    
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """String representation."""
        return f"quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"
    
    def __str__(self) -> str:
        """String format."""
        return f"{self.w:.6f} + {self.x:.6f}i + {self.y:.6f}j + {self.z:.6f}k"
    
    def to_array(self) -> np.ndarray:
        """Kuaterniyonu numpy array'e dönüştürür."""
        return np.array([self.w, self.x, self.y, self.z])
    
    def to_list(self) -> List[float]:
        """Kuaterniyonu listeye dönüştürür."""
        return [self.w, self.x, self.y, self.z]
    
    @classmethod
    def identity(cls) -> 'quaternion':
        """Birim kuaterniyon döndürür."""
        return cls(1.0, 0.0, 0.0, 0.0)
    
    def is_identity(self, tolerance: float = 1e-10) -> bool:
        """Birim kuaterniyon olup olmadığını kontrol eder."""
        return (abs(self.w - 1.0) < tolerance and 
                abs(self.x) < tolerance and 
                abs(self.y) < tolerance and 
                abs(self.z) < tolerance)
    
    @classmethod
    def parse(cls, s) -> 'quaternion':
        """Çeşitli formatlardan quaternion oluşturur.
        
        Args:
            s: Dönüştürülecek değer
            
        Returns:
            quaternion: Dönüştürülmüş kuaterniyon
        """
        return _parse_quaternion_from_csv(s)
    
    @classmethod
    def from_csv_string(cls, s: str) -> 'quaternion':
        """CSV string'inden quaternion oluşturur.
        
        Args:
            s: Virgülle ayrılmış string ("w,x,y,z" veya "scalar")
            
        Returns:
            quaternion: Dönüştürülmüş kuaterniyon
        """
        return _parse_quaternion_from_csv(s)
    
    @classmethod
    def from_complex(cls, c: complex) -> 'quaternion':
        """Complex sayıdan quaternion oluşturur (sadece gerçek kısım kullanılır).
        
        Args:
            c: Complex sayı
            
        Returns:
            quaternion: Dönüştürülmüş kuaterniyon
        """
        return quaternion(float(c.real), 0, 0, 0)

@dataclass
class TernaryNumber:
    def __init__(self, digits: list):
        """
        Üçlü sayıyı oluşturur. Verilen değer bir liste olmalıdır.

        :param digits: Üçlü sayının rakamlarını temsil eden liste.
        """
        self.digits = digits

    @classmethod
    def from_ternary_string(cls, ternary_str: str) -> 'TernaryNumber':
        """Üçlü sayı sistemindeki stringi TernaryNumber'a dönüştürür."""
        ternary_str = ternary_str.strip()
        if not all(c in '012' for c in ternary_str):
            raise ValueError("Üçlü sayı sadece 0, 1 ve 2 rakamlarından oluşabilir.")
        digits = [int(c) for c in ternary_str]
        return cls(digits)

    @classmethod
    def from_decimal(cls, decimal: int) -> 'TernaryNumber':
        """Ondalık sayıyı üçlü sayı sistemine dönüştürür."""
        if decimal == 0:
            return cls([0])
        digits = []
        while decimal > 0:
            digits.append(decimal % 3)
            decimal = decimal // 3
        return cls(digits[::-1] if digits else [0])

    def to_decimal(self):
        """Üçlü sayının ondalık karşılığını döndürür."""
        decimal_value = 0
        for i, digit in enumerate(reversed(self.digits)):
            decimal_value += digit * (3 ** i)
        return decimal_value

    def __repr__(self):
        """Nesnenin yazdırılabilir temsilini döndürür."""
        return f"TernaryNumber({self.digits})"

    def __str__(self):
        """Nesnenin string temsilini döndürür."""
        return ''.join(map(str, self.digits))

    def __add__(self, other):
        """Toplama işlemini destekler."""
        if isinstance(other, TernaryNumber):
            result_decimal = self.to_decimal() + other.to_decimal()
        elif isinstance(other, (int, str)):
            result_decimal = self.to_decimal() + int(other)
        else:
            raise TypeError("TernaryNumber'ın başka bir sayıya veya TernaryNumber'e eklenebilir.")
        return TernaryNumber.from_decimal(result_decimal)

    def __radd__(self, other):
        """Toplama işleminin sağ taraf desteklenmesini sağlar."""
        return self.__add__(other)

    def __sub__(self, other):
        """Çıkarma işlemini destekler."""
        if isinstance(other, TernaryNumber):
            result_decimal = self.to_decimal() - other.to_decimal()
        elif isinstance(other, (int, str)):
            result_decimal = self.to_decimal() - int(other)
        else:
            raise TypeError("TernaryNumber'dan başka bir sayıya veya başka bir TernaryNumber çıkartılabilir.")
        if result_decimal < 0:
            raise ValueError("Bir üçlü sayıdan daha büyük bir sayı çıkaramazsınız.")
        return TernaryNumber.from_decimal(result_decimal)

    def __rsub__(self, other):
        """Çıkarma işleminin sağ taraf desteklenmesini sağlar."""
        if isinstance(other, (int, str)):
            result_decimal = int(other) - self.to_decimal()
        else:
            raise TypeError("TernaryNumber'dan bir sayı çıkartılabilir.")
        if result_decimal < 0:
            raise ValueError("Bir üçlü sayıdan daha büyük bir sayı çıkaramazsınız.")
        return TernaryNumber.from_decimal(result_decimal)

    def __mul__(self, scalar):
        """Skaler çarpım işlemini destekler."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("TernaryNumber sadece skaler ile çarpılabilir.")
        result_decimal = self.to_decimal() * scalar
        return TernaryNumber.from_decimal(int(result_decimal))

    def __rmul__(self, other):
        """ Çarpma işleminin sağ taraf desteklenmesini sağlar. """
        return self.__mul__(other)

    # Üçlü sayı sisteminde bölme işlemi, ondalık karşılığa dönüştürülerek yapılmalıdır.
    def __truediv__(self, other):
        """ Bölme işlemini destekler. """
        if isinstance(other, TernaryNumber):
            other_decimal = other.to_decimal()
            if other_decimal == 0:
                raise ZeroDivisionError("Bir TernaryNumber sıfırla bölünemez.")
            result_decimal = self.to_decimal() / other_decimal
            return TernaryNumber.from_decimal(int(round(result_decimal)))
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Sıfırla bölme hatası.")
            result_decimal = self.to_decimal() / other
            return TernaryNumber.from_decimal(int(round(result_decimal)))
        else:
            raise TypeError("TernaryNumber'i bir sayı veya başka bir TernaryNumber ile bölebilirsiniz.")

    # üçlü sayı sisteminde bölme işlemi, ondalık karşılığa dönüştürülerek yapılmalıdır.
    def __rtruediv__(self, other):
        """ Bölme işleminin sağ taraf desteklenmesini sağlar. """
        if isinstance(other, (int, float)):
            self_decimal = self.to_decimal()
            if self_decimal == 0:
                raise ZeroDivisionError("Sıfırla bölme hatası.")
            result_decimal = other / self_decimal
            return TernaryNumber.from_decimal(int(round(result_decimal)))
        else:
            raise TypeError("TernaryNumber ile bir sayı bölünebilir.")

    def __eq__(self, other):
        """Eşitlik kontrolü yapar."""
        if isinstance(other, TernaryNumber):
            return self.digits == other.digits
        elif isinstance(other, (int, str)):
            return self.to_decimal() == int(other)
        else:
            return False

    def __ne__(self, other):
        """Eşitsizlik kontrolü yapar."""
        return not self.__eq__(other)

# Superreal Sayılar
@dataclass
class SuperrealNumber:
    #def __init__(self, real_part=0.0):
    def __init__(self, real: float, split: float = 0.0):
        """
        SuperrealNumber nesnesini oluşturur.
        
        :param real_part: Gerçek sayı bileşeni (float).
        """
        #self.real = real_part
        self.real = real
        self.split = split

    def __repr__(self):
        """ Nesnenin yazdırılabilir temsilini döndürür. """
        return f"SuperrealNumber({self.real})"

    def __str__(self):
        """ Nesnenin string temsilini döndürür. """
        return str(self.real)

    def __add__(self, other):
        """ Toplama işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            return SuperrealNumber(self.real + other.real)
        elif isinstance(other, (int, float)):
            return SuperrealNumber(self.real + other)
        else:
            raise TypeError("SuperrealNumber'e bir sayı veya başka bir SuperrealNumber eklenebilir.")

    def __radd__(self, other):
        """ Toplama işleminin sağ taraf desteklenmesini sağlar. """
        return self.__add__(other)

    def __sub__(self, other):
        """ Çıkarma işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            return SuperrealNumber(self.real - other.real)
        elif isinstance(other, (int, float)):
            return SuperrealNumber(self.real - other)
        else:
            raise TypeError("SuperrealNumber'dan bir sayı veya başka bir SuperrealNumber çıkarılabilir.")

    def __rsub__(self, other):
        """ Çıkarma işleminin sağ taraf desteklenmesini sağlar. """
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        """ Çarpma işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            return SuperrealNumber(self.real * other.real)
        elif isinstance(other, (int, float)):
            return SuperrealNumber(self.real * other)
        else:
            raise TypeError("SuperrealNumber ile bir sayı veya başka bir SuperrealNumber çarpılabilir.")

    def __rmul__(self, other):
        """ Çarpma işleminin sağ taraf desteklenmesini sağlar. """
        return self.__mul__(other)

    def __truediv__(self, other):
        """ Bölme işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            if other.real == 0:
                raise ZeroDivisionError("Bir SuperrealNumber sıfırla bölünemez.")
            return SuperrealNumber(self.real / other.real)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Sıfırla bölme hatası.")
            return SuperrealNumber(self.real / other)
        else:
            raise TypeError("SuperrealNumber'i bir sayı veya başka bir SuperrealNumber ile bölebilirsiniz.")

    def __rtruediv__(self, other):
        """ Bölme işleminin sağ taraf desteklenmesini sağlar. """
        if self.real == 0:
            raise ZeroDivisionError("Sıfırla bölme hatası.")
        return SuperrealNumber(other / self.real)

    def __neg__(self):
        """ Negatif değeri döndürür. """
        return SuperrealNumber(-self.real)

    def __eq__(self, other):
        """ Eşitlik kontrolü yapar. """
        if isinstance(other, SuperrealNumber):
            return self.real == other.real
        elif isinstance(other, (int, float)):
            return self.real == other
        else:
            return False

    def __ne__(self, other):
        """ Eşitsizlik kontrolü yapar. """
        return not self.__eq__(other)

    def __lt__(self, other):
        """ Küçük olma kontrolü yapar. """
        if isinstance(other, SuperrealNumber):
            return self.real < other.real
        elif isinstance(other, (int, float)):
            return self.real < other
        else:
            raise TypeError("SuperrealNumber ile karşılaştırılabilir.")

    def __le__(self, other):
        """ Küçük veya eşit kontrolü yapar. """
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        """ Büyük olma kontrolü yapar. """
        return not self.__le__(other)

    def __ge__(self, other):
        """ Büyük veya eşit kontrolü yapar. """
        return not self.__lt__(other)

@dataclass
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
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"

    def __repr__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"
        #return f"PathionNumber({self.coeffs})"
    
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
        if not isinstance(other, PathionNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, PathionNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

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

    # ------------------------------------------------------------------
    # Yeni eklenen yardımcı metodlar
    # ------------------------------------------------------------------
    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # Güvenli phase: ilk bileşene bak, eğer complex ise angle döndür, değilse 0.0
        try:
            first = self.coeffs[0] if self.coeffs else 0.0
            if isinstance(first, complex):
                return math.atan2(first.imag, first.real)
        except Exception:
            pass
        return 0.0

from dataclasses import dataclass, field
from typing import List, Union, Any, Optional, Iterable, Tuple, Dict, Generator, Callable, overload, cast
import numpy as np
import math
import re
import warnings

@dataclass
class ChingonNumber:
    """
    64-bileşenli gelişmiş sayı sistemi.
    
    Chingon sayıları, 64 farklı bileşenden oluşan hiper kompleks sayıları temsil eder.
    Her bileşen bir boyutu/ekseni temsil eder ve çok boyutlu işlemler için kullanılır.
    """
    
    coeffs: List[float] = field(default_factory=lambda: [0.0] * 64)
    
    def __post_init__(self) -> None:
        """Başlatma sonrası işlemler: tip dönüşümleri ve boyut kontrolü"""
        # NumPy array'i liste olarak dönüştür
        if isinstance(self.coeffs, np.ndarray):
            self.coeffs = self.coeffs.tolist()
        elif isinstance(self.coeffs, tuple):
            self.coeffs = list(self.coeffs)
        elif not isinstance(self.coeffs, list):
            # Diğer tipleri listeye çevirmeye çalış
            try:
                self.coeffs = list(self.coeffs)
            except Exception as e:
                raise TypeError(f"Cannot convert {type(self.coeffs)} to list for ChingonNumber: {e}")
        
        # Uzunluk kontrolü ve ayarlama
        if len(self.coeffs) != 64:
            if len(self.coeffs) < 64:
                # Eksik bileşenleri 0 ile tamamla
                self.coeffs = self.coeffs + [0.0] * (64 - len(self.coeffs))
            else:
                # Fazla bileşenleri kes
                warnings.warn(f"ChingonNumber expects 64 components, got {len(self.coeffs)}. Truncating.", 
                             RuntimeWarning, stacklevel=2)
                self.coeffs = self.coeffs[:64]
        
        # Tüm bileşenleri float'a çevir
        processed_coeffs: List[float] = []
        for i, c in enumerate(self.coeffs):
            try:
                if isinstance(c, complex):
                    # Complex için real kısmı al
                    processed_coeffs.append(float(c.real))
                elif isinstance(c, (int, float, np.number)):
                    # Sayısal tipler için float'a çevir
                    processed_coeffs.append(float(c))
                else:
                    # Diğer tipler için string üzerinden çevir
                    processed_coeffs.append(float(str(c)))
            except (ValueError, TypeError) as e:
                warnings.warn(f"Component {i} of ChingonNumber cannot be converted to float: {c}. Using 0.0.", 
                             RuntimeWarning, stacklevel=2)
                processed_coeffs.append(0.0)
        
        self.coeffs = processed_coeffs
        
        # Numpy array önbelleği
        self._np_array: Optional[np.ndarray] = None
    
    @property
    def n(self) -> np.ndarray:
        """NumPy array özelliği (önbelleklenmiş)"""
        if self._np_array is None:
            self._np_array = np.array(self.coeffs, dtype=np.float64)
        return self._np_array
    
    @classmethod
    def zeros(cls) -> 'ChingonNumber':
        """Tüm bileşenleri sıfır olan ChingonNumber"""
        return cls([0.0] * 64)
    
    @classmethod
    def ones(cls) -> 'ChingonNumber':
        """Tüm bileşenleri bir olan ChingonNumber"""
        return cls([1.0] * 64)
    
    @classmethod
    def eye(cls, index: int) -> 'ChingonNumber':
        """Belirli bir indekste 1, diğerlerinde 0 olan birim vektör"""
        if index < 0 or index >= 64:
            raise IndexError(f"Index must be between 0 and 63, got {index}")
        coeffs = [0.0] * 64
        coeffs[index] = 1.0
        return cls(coeffs)
    
    @classmethod
    def from_scalar(cls, value: Union[int, float, complex]) -> 'ChingonNumber':
        """Skaler değerden ChingonNumber oluştur"""
        scalar = float(value.real) if isinstance(value, complex) else float(value)
        coeffs = [scalar] + [0.0] * 63
        return cls(coeffs)
    
    @classmethod
    def from_complex(cls, real: float, imag: float) -> 'ChingonNumber':
        """Complex sayıdan ChingonNumber oluştur"""
        coeffs = [float(real), float(imag)] + [0.0] * 62
        return cls(coeffs)
    
    @classmethod
    def from_string(cls, s: str) -> 'ChingonNumber':
        """String'den ChingonNumber oluştur"""
        # Basit string parsing
        s = s.strip()
        
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            coeffs: List[float] = []
            for p in parts:
                try:
                    coeffs.append(float(p))
                except ValueError:
                    coeffs.append(0.0)
            return cls(coeffs)
        else:
            try:
                scalar = float(s)
                return cls.from_scalar(scalar)
            except ValueError:
                # Sembolik form: "1.0e0 + 2.0e1 + ..."
                coeffs = [0.0] * 64
                for i in range(64):
                    pattern = rf'([+-]?\d*\.?\d*)\s*[eE]{i}'
                    match = re.search(pattern, s)
                    if match:
                        coeff_str = match.group(1)
                        if coeff_str in ['', '+']:
                            coeffs[i] = 1.0
                        elif coeff_str == '-':
                            coeffs[i] = -1.0
                        else:
                            coeffs[i] = float(coeff_str)
                return cls(coeffs)
    
    @classmethod
    def from_iterable(cls, iterable: Iterable) -> 'ChingonNumber':
        """Iterable'dan ChingonNumber oluştur"""
        return cls(list(iterable))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'ChingonNumber':
        """NumPy array'den ChingonNumber oluştur"""
        return cls(array.tolist())
    
    @classmethod
    def random(cls, low: float = -1.0, high: float = 1.0, seed: Optional[int] = None) -> 'ChingonNumber':
        """Rastgele ChingonNumber oluştur"""
        if seed is not None:
            np.random.seed(seed)
        coeffs = np.random.uniform(low, high, 64).tolist()
        return cls(coeffs)
    
    @property
    def real(self) -> float:
        """İlk bileşen – gerçek kısım"""
        return self.coeffs[0]
    
    @real.setter
    def real(self, value: Union[int, float, complex]) -> None:
        """Gerçek kısımı ayarla"""
        self.coeffs[0] = float(value.real) if isinstance(value, complex) else float(value)
        self._clear_cache()
    
    @property
    def imag(self) -> float:
        """İkinci bileşen – sanal kısım"""
        return self.coeffs[1] if len(self.coeffs) > 1 else 0.0
    
    @imag.setter
    def imag(self, value: Union[int, float, complex]) -> None:
        """Sanal kısmı ayarla"""
        if len(self.coeffs) > 1:
            self.coeffs[1] = float(value.real) if isinstance(value, complex) else float(value)
            self._clear_cache()
    
    def _clear_cache(self) -> None:
        """Önbelleği temizle"""
        self._np_array = None
    
    # ===== TEMEL İŞLEMLER =====
    
    def __iter__(self) -> Generator[float, None, None]:
        """İterasyon desteği"""
        return (c for c in self.coeffs)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[float, List[float]]:
        """İndeksleme desteği"""
        if isinstance(index, slice):
            return self.coeffs[index]
        elif isinstance(index, int):
            if index < 0:
                index = 64 + index
            if 0 <= index < 64:
                return self.coeffs[index]
            else:
                raise IndexError(f"Index {index} out of range for ChingonNumber (0-63)")
        else:
            raise TypeError(f"ChingonNumber indices must be integers or slices, not {type(index)}")
    
    def __setitem__(self, index: int, value: Union[int, float, complex]) -> None:
        """Bileşen atama desteği"""
        if index < 0:
            index = 64 + index
        if 0 <= index < 64:
            self.coeffs[index] = float(value.real) if isinstance(value, complex) else float(value)
            self._clear_cache()
        else:
            raise IndexError(f"Index {index} out of range for ChingonNumber (0-63)")
    
    def __len__(self) -> int:
        """Uzunluk (her zaman 64)"""
        return 64
    
    def __str__(self) -> str:
        """Kullanıcı dostu string gösterimi"""
        # Sıfır olmayan bileşenleri bul
        non_zero = [(i, c) for i, c in enumerate(self.coeffs) if abs(c) > 1e-10]
        
        if len(non_zero) == 0:
            return "ChingonNumber(0)"
        elif len(non_zero) <= 4:
            # Az sayıda sıfır olmayan bileşen varsa hepsini göster
            parts = [f"c{i}={c:.6g}" for i, c in non_zero]
            return f"ChingonNumber({', '.join(parts)})"
        else:
            # Çok sayıda bileşen varsa özet göster
            first = non_zero[:2]
            last = non_zero[-1]
            parts = [f"c{i}={c:.3g}" for i, c in first]
            return f"ChingonNumber({', '.join(parts)}, ..., c{last[0]}={last[1]:.3g})"
    
    def __repr__(self) -> str:
        """Teknik string gösterimi"""
        return f"ChingonNumber({self.coeffs})"
    
    def __format__(self, format_spec: str) -> str:
        """Formatlama desteği"""
        if not format_spec:
            return str(self)
        
        if format_spec == 'short':
            # Kısa format
            return f"CN[{self.coeffs[0]:.3g}, ..., {self.coeffs[-1]:.3g}]"
        elif format_spec == 'full':
            # Tam format
            return f"ChingonNumber({', '.join(f'{c:.6g}' for c in self.coeffs)})"
        elif format_spec.startswith('prec'):
            # Hassasiyet belirtilen format
            try:
                prec = int(format_spec[4:])
                return f"CN[{', '.join(f'{c:.{prec}f}' for c in self.coeffs[:3])}, ...]"
            except:
                return str(self)
        else:
            return str(self)
    
    # ===== ARİTMETİK İŞLEMLER =====
    # Tüm overload'ları birlikte tanımla
    @overload
    def __add__(self, other: 'ChingonNumber') -> 'ChingonNumber': ...
    
    @overload
    def __add__(self, other: int) -> 'ChingonNumber': ...
    
    @overload
    def __add__(self, other: float) -> 'ChingonNumber': ...
    
    @overload
    def __add__(self, other: complex) -> 'ChingonNumber': ...
    
    # Gerçek implementasyon (tüm overload'lardan sonra)
    def __add__(self, other: Any) -> Any:
        """Toplama: self + other"""
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, complex):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other.real)
            if len(new_coeffs) > 1:
                new_coeffs[1] += float(other.imag)
            return ChingonNumber(new_coeffs)
        return NotImplemented
    
    @overload
    def __radd__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    def __radd__(self, other: Any) -> Any:
        """Sağdan toplama: other + self"""
        return self.__add__(other)
    
    @overload
    def __sub__(self, other: 'ChingonNumber') -> 'ChingonNumber': ...
    
    @overload
    def __sub__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    def __sub__(self, other: Any) -> Any:
        """Çıkarma: self - other"""
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, complex):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other.real)
            if len(new_coeffs) > 1:
                new_coeffs[1] -= float(other.imag)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, np.ndarray):
            if other.shape == (64,):
                return ChingonNumber((self.n - other).tolist())
        return NotImplemented
    
    @overload
    def __rsub__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    def __rsub__(self, other: Any) -> Any:
        """Sağdan çıkarma: other - self"""
        if isinstance(other, (int, float)):
            new_coeffs = [-c for c in self.coeffs]
            new_coeffs[0] += float(other)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, complex):
            new_coeffs = [-c for c in self.coeffs]
            new_coeffs[0] += float(other.real)
            if len(new_coeffs) > 1:
                new_coeffs[1] += float(other.imag)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, np.ndarray):
            if other.shape == (64,):
                return ChingonNumber((other - self.n).tolist())
        return NotImplemented
    
    @overload
    def __mul__(self, other: 'ChingonNumber') -> 'ChingonNumber': ...
    
    @overload
    def __mul__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    def __mul__(self, other: Any) -> Any:
        """Çarpma: self * other"""
        if isinstance(other, ChingonNumber):
            # Hadamard product (element-wise multiplication)
            return ChingonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return ChingonNumber([c * float(other) for c in self.coeffs])
        elif isinstance(other, complex):
            # Complex çarpma: sadece real kısmı al
            return ChingonNumber([c * float(other.real) for c in self.coeffs])
        elif isinstance(other, np.ndarray):
            if other.shape == (64,):
                return ChingonNumber((self.n * other).tolist())
        return NotImplemented
    
    @overload
    def __rmul__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    def __rmul__(self, other: Any) -> Any:
        """Sağdan çarpma: other * self"""
        return self.__mul__(other)
    
    @overload
    def __truediv__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    @overload
    def __truediv__(self, other: 'ChingonNumber') -> 'ChingonNumber': ...
    
    def __truediv__(self, other: Any) -> Any:
        """Bölme: self / other"""
        if isinstance(other, (int, float)):
            if abs(float(other)) < 1e-12:
                raise ZeroDivisionError("Division by zero")
            return ChingonNumber([c / float(other) for c in self.coeffs])
        elif isinstance(other, ChingonNumber):
            # Element-wise division
            new_coeffs: List[float] = []
            for a, b in zip(self.coeffs, other.coeffs):
                if abs(b) < 1e-12:
                    # Sıfıra bölme
                    if abs(a) < 1e-12:
                        new_coeffs.append(float('nan'))  # 0/0
                    else:
                        new_coeffs.append(float('inf') if a >= 0 else float('-inf'))
                else:
                    new_coeffs.append(a / b)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, complex):
            if abs(other) < 1e-12:
                raise ZeroDivisionError("Division by complex zero")
            return ChingonNumber([c / abs(other) for c in self.coeffs])
        elif isinstance(other, np.ndarray):
            if other.shape == (64,):
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = self.n / other
                    result[np.isinf(result)] = float('inf')
                    result[np.isnan(result)] = float('nan')
                return ChingonNumber(result.tolist())
        return NotImplemented
    
    @overload
    def __rtruediv__(self, other: Union[int, float]) -> 'ChingonNumber': ...
    
    def __rtruediv__(self, other: Any) -> Any:
        """Sağdan bölme: other / self"""
        if isinstance(other, (int, float)):
            new_coeffs: List[float] = []
            other_float = float(other)
            for c in self.coeffs:
                if abs(c) < 1e-12:
                    if abs(other_float) < 1e-12:
                        new_coeffs.append(float('nan'))
                    else:
                        new_coeffs.append(float('inf') if other_float >= 0 else float('-inf'))
                else:
                    new_coeffs.append(other_float / c)
            return ChingonNumber(new_coeffs)
        elif isinstance(other, complex):
            new_coeffs = []
            other_complex = complex(other)
            for c in self.coeffs:
                if abs(c) < 1e-12:
                    if abs(other_complex) < 1e-12:
                        new_coeffs.append(float('nan'))
                    else:
                        new_coeffs.append(float('inf'))
                else:
                    new_coeffs.append(abs(other_complex) / abs(c))
            return ChingonNumber(new_coeffs)
        elif isinstance(other, np.ndarray):
            if other.shape == (64,):
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = other / self.n
                    result[np.isinf(result)] = float('inf')
                    result[np.isnan(result)] = float('nan')
                return ChingonNumber(result.tolist())
        return NotImplemented
    
    def __floordiv__(self, other: Any) -> 'ChingonNumber':
        """Tam sayı bölme: self // other"""
        if isinstance(other, (int, float)):
            if abs(float(other)) < 1e-12:
                raise ZeroDivisionError("Integer division by zero")
            return ChingonNumber([c // float(other) for c in self.coeffs])
        elif isinstance(other, complex):
            if abs(other) < 1e-12:
                raise ZeroDivisionError("Integer division by complex zero")
            return ChingonNumber([c // abs(other) for c in self.coeffs])
        return NotImplemented
    
    def __mod__(self, other: Any) -> 'ChingonNumber':
        """Modülüs: self % other"""
        if isinstance(other, (int, float)):
            if abs(float(other)) < 1e-12:
                raise ZeroDivisionError("Modulo by zero")
            return ChingonNumber([c % float(other) for c in self.coeffs])
        elif isinstance(other, complex):
            if abs(other) < 1e-12:
                raise ZeroDivisionError("Modulo by complex zero")
            return ChingonNumber([c % abs(other) for c in self.coeffs])
        return NotImplemented
    
    def __pow__(self, exponent: Any) -> 'ChingonNumber':
        """Üs alma: self ** exponent"""
        if isinstance(exponent, (int, float)):
            return ChingonNumber([c ** float(exponent) for c in self.coeffs])
        elif isinstance(exponent, complex):
            return ChingonNumber([c ** abs(exponent) for c in self.coeffs])
        return NotImplemented
    
    def __neg__(self) -> 'ChingonNumber':
        """Negatif: -self"""
        return ChingonNumber([-c for c in self.coeffs])
    
    def __pos__(self) -> 'ChingonNumber':
        """Pozitif: +self"""
        return self
    
    def __abs__(self) -> float:
        """Mutlak değer (büyüklük/norm)"""
        return self.norm()
    
    # ===== KARŞILAŞTIRMA İŞLEMLERİ =====
    
    def __eq__(self, other: Any) -> bool:
        """Eşitlik: self == other"""
        if isinstance(other, ChingonNumber):
            return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        elif isinstance(other, (int, float)):
            return np.allclose(self.coeffs, [float(other)] + [0.0] * 63, atol=1e-10)
        elif isinstance(other, complex):
            return (abs(self.coeffs[0] - other.real) < 1e-10 and 
                    abs(self.coeffs[1] - other.imag) < 1e-10 and
                    np.allclose(self.coeffs[2:], [0.0] * 62, atol=1e-10))
        return NotImplemented
    
    def __ne__(self, other: Any) -> bool:
        """Eşit değil: self != other"""
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result
    
    def __lt__(self, other: Any) -> bool:
        """Küçük: self < other"""
        if isinstance(other, (int, float)):
            return self.norm() < float(other)
        elif isinstance(other, ChingonNumber):
            return self.norm() < other.norm()
        return NotImplemented
    
    def __le__(self, other: Any) -> bool:
        """Küçük veya eşit: self <= other"""
        if isinstance(other, (int, float)):
            return self.norm() <= float(other)
        elif isinstance(other, ChingonNumber):
            return self.norm() <= other.norm()
        return NotImplemented
    
    def __gt__(self, other: Any) -> bool:
        """Büyük: self > other"""
        if isinstance(other, (int, float)):
            return self.norm() > float(other)
        elif isinstance(other, ChingonNumber):
            return self.norm() > other.norm()
        return NotImplemented
    
    def __ge__(self, other: Any) -> bool:
        """Büyük veya eşit: self >= other"""
        if isinstance(other, (int, float)):
            return self.norm() >= float(other)
        elif isinstance(other, ChingonNumber):
            return self.norm() >= other.norm()
        return NotImplemented
    
    # ===== MATEMATİKSEL FONKSİYONLAR =====
    
    def norm(self) -> float:
        """Euclidean norm (büyüklük)"""
        return float(np.linalg.norm(self.coeffs))
    
    def norm_squared(self) -> float:
        """Norm karesi"""
        return float(np.dot(self.coeffs, self.coeffs))
    
    def magnitude(self) -> float:
        """Büyüklük (norm ile aynı)"""
        return self.norm()
    
    def normalize(self) -> 'ChingonNumber':
        """Birim norma normalize et"""
        mag = self.norm()
        if mag < 1e-12:
            return ChingonNumber.zeros()
        return ChingonNumber([c / mag for c in self.coeffs])
    
    def conjugate(self) -> 'ChingonNumber':
        """Karmaşık eşlenik benzeri"""
        return ChingonNumber([-c for c in self.coeffs])
    
    def dot(self, other: 'ChingonNumber') -> float:
        """İç çarpım"""
        if not isinstance(other, ChingonNumber):
            raise TypeError(f"Dot product requires ChingonNumber, got {type(other)}")
        return float(np.dot(self.coeffs, other.coeffs))
    
    def cross(self, other: 'ChingonNumber') -> 'ChingonNumber':
        """Çapraz çarpım (ilk 3 bileşen için)"""
        if not isinstance(other, ChingonNumber):
            raise TypeError(f"Cross product requires ChingonNumber, got {type(other)}")
        
        # İlk 3 bileşen için 3D cross product
        a1, a2, a3 = self.coeffs[0], self.coeffs[1], self.coeffs[2]
        b1, b2, b3 = other.coeffs[0], other.coeffs[1], other.coeffs[2]
        
        result = [
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ] + [0.0] * 61
        
        return ChingonNumber(result)
    
    def phase(self) -> float:
        """Faz açısı (gerçek kısım için)"""
        if self.norm() < 1e-12:
            return 0.0
        return math.atan2(self.coeffs[1], self.coeffs[0]) if len(self.coeffs) > 1 else 0.0
    
    def to_polar(self) -> Tuple[float, float]:
        """Kutupsal koordinatlara dönüşüm"""
        return self.norm(), self.phase()
    
    # ===== DÖNÜŞÜMLER =====
    
    def to_list(self) -> List[float]:
        """Liste olarak döndür"""
        return self.coeffs.copy()
    
    def to_tuple(self) -> Tuple[float, ...]:
        """Tuple olarak döndür"""
        return tuple(self.coeffs)
    
    def to_numpy(self) -> np.ndarray:
        """NumPy array olarak döndür"""
        return np.array(self.coeffs, dtype=np.float64)
    
    def to_complex(self) -> complex:
        """Complex sayıya dönüştür (ilk 2 bileşen)"""
        return complex(self.coeffs[0], self.coeffs[1] if len(self.coeffs) > 1 else 0.0)
    
    def to_dict(self) -> Dict[str, float]:
        """Sözlük olarak döndür"""
        return {f'c{i}': c for i, c in enumerate(self.coeffs)}
    
    def to_string(self, precision: int = 6, fmt: str = 'comma') -> str:
        """String'e dönüştür"""
        if fmt == 'comma':
            return ','.join(f'{c:.{precision}f}' for c in self.coeffs)
        elif fmt == 'bracket':
            return f"[{', '.join(f'{c:.{precision}f}' for c in self.coeffs)}]"
        elif fmt == 'symbolic':
            parts = []
            for i, c in enumerate(self.coeffs):
                if abs(c) > 1e-10:
                    if i == 0:
                        parts.append(f"{c:.{precision}f}")
                    else:
                        parts.append(f"{c:.{precision}f}e{i}")
            return ' + '.join(parts) if parts else '0'
        else:
            return str(self)
    
    # ===== YARDIMCI METODLAR =====
    
    def copy(self) -> 'ChingonNumber':
        """Kopyasını oluştur"""
        return ChingonNumber(self.coeffs.copy())
    
    def is_zero(self, tolerance: float = 1e-12) -> bool:
        """Sıfıra yakın mı kontrol et"""
        return self.norm() < tolerance
    
    def is_unit(self, tolerance: float = 1e-12) -> bool:
        """Birim vektör mü kontrol et"""
        return abs(self.norm() - 1.0) < tolerance
    
    def max_component(self) -> Tuple[int, float]:
        """En büyük bileşeni ve değerini döndür"""
        idx = int(np.argmax(np.abs(self.coeffs)))  # numpy int'i Python int'e çevir
        return idx, self.coeffs[idx]
    
    def min_component(self) -> Tuple[int, float]:
        """En küçük bileşeni ve değerini döndür"""
        idx = int(np.argmin(np.abs(self.coeffs)))  # numpy int'i Python int'e çevir
        return idx, self.coeffs[idx]
    
    def apply(self, func: Callable[[float], float]) -> 'ChingonNumber':
        """Fonksiyonu tüm bileşenlere uygula"""
        return ChingonNumber([func(c) for c in self.coeffs])
    
    def sum(self) -> float:
        """Tüm bileşenlerin toplamı"""
        return sum(self.coeffs)
    
    def mean(self) -> float:
        """Bileşenlerin ortalaması"""
        return sum(self.coeffs) / 64
    
    def std(self) -> float:
        """Bileşenlerin standart sapması"""
        mean = self.mean()
        variance = sum((c - mean) ** 2 for c in self.coeffs) / 64
        return math.sqrt(variance)
    
    def abs(self) -> 'ChingonNumber':
        """Her bileşenin mutlak değeri"""
        return ChingonNumber([abs(c) for c in self.coeffs])
    
    def sign(self) -> 'ChingonNumber':
        """Her bileşenin işareti"""
        return ChingonNumber([1.0 if c >= 0 else -1.0 for c in self.coeffs])
    
    def floor(self) -> 'ChingonNumber':
        """Her bileşenin taban değeri"""
        return ChingonNumber([math.floor(c) for c in self.coeffs])
    
    def ceil(self) -> 'ChingonNumber':
        """Her bileşenin tavan değeri"""
        return ChingonNumber([math.ceil(c) for c in self.coeffs])
    
    def round(self, decimals: int = 0) -> 'ChingonNumber':
        """Her bileşeni yuvarla"""
        return ChingonNumber([round(c, decimals) for c in self.coeffs])
    
    def clip(self, min_val: float = -float('inf'), max_val: float = float('inf')) -> 'ChingonNumber':
        """Bileşenleri sınırla"""
        return ChingonNumber([max(min_val, min(max_val, c)) for c in self.coeffs])
    
    # ===== OPERATÖRLER =====
    
    def __hash__(self) -> int:
        """Hash değeri"""
        # Yuvarlama ile hassasiyet sorunlarını önle
        rounded = tuple(round(c, 12) for c in self.coeffs)
        return hash(rounded)
    
    def __bool__(self) -> bool:
        """Boolean dönüşümü"""
        return not self.is_zero()
    
    def __complex__(self) -> complex:
        """Complex dönüşümü"""
        return self.to_complex()
    
    def __float__(self) -> float:
        """Float dönüşümü (ilk bileşen)"""
        return float(self.coeffs[0])
    
    def __int__(self) -> int:
        """Integer dönüşümü (ilk bileşen)"""
        return int(self.coeffs[0])
    
    def __contains__(self, value: float) -> bool:
        """İçerik kontrolü"""
        return any(abs(c - value) < 1e-10 for c in self.coeffs)
    
    def __reversed__(self) -> Generator[float, None, None]:
        """Tersine iterasyon"""
        return (c for c in reversed(self.coeffs))
    
    def __sizeof__(self) -> int:
        """Bellek boyutu"""
        import sys
        return sys.getsizeof(self.coeffs) + sys.getsizeof(self._np_array or 0)

# ===== GLOBAL FONKSİYONLAR =====

def chingon_zeros() -> ChingonNumber:
    """Sıfır ChingonNumber"""
    return ChingonNumber.zeros()

def chingon_ones() -> ChingonNumber:
    """Birler ChingonNumber"""
    return ChingonNumber.ones()

def chingon_eye(index: int) -> ChingonNumber:
    """Birim vektör"""
    return ChingonNumber.eye(index)

def chingon_random(low: float = -1.0, high: float = 1.0, seed: Optional[int] = None) -> ChingonNumber:
    """Rastgele ChingonNumber"""
    return ChingonNumber.random(low, high, seed)

def chingon_linspace(start: Union[float, ChingonNumber], 
                     end: Union[float, ChingonNumber], 
                     num: int = 64) -> List[ChingonNumber]:
    """Doğrusal uzay oluştur"""
    if not isinstance(start, ChingonNumber):
        start = ChingonNumber.from_scalar(start)
    if not isinstance(end, ChingonNumber):
        end = ChingonNumber.from_scalar(end)
    
    result: List[ChingonNumber] = []
    for i in range(num):
        t = i / (num - 1) if num > 1 else 0
        result.append((1 - t) * start + t * end)
    
    return result

def chingon_dot(a: ChingonNumber, b: ChingonNumber) -> float:
    """İki ChingonNumber'ın iç çarpımı"""
    return a.dot(b)

def chingon_cross(a: ChingonNumber, b: ChingonNumber) -> ChingonNumber:
    """İki ChingonNumber'ın çapraz çarpımı"""
    return a.cross(b)

def chingon_norm(cn: ChingonNumber) -> float:
    """ChingonNumber'ın normu"""
    return cn.norm()

def chingon_normalize(cn: ChingonNumber) -> ChingonNumber:
    """ChingonNumber'ı normalize et"""
    return cn.normalize()

def chingon_unit_vector(index: int) -> ChingonNumber:
    """Belirtilen indekste 1, diğerlerinde 0 olan birim vektör"""
    if index < 0 or index >= 64:
        raise IndexError(f"Index {index} out of range for 64-component ChingonNumber")
    coeffs = [0.0] * 64
    coeffs[index] = 1.0
    return ChingonNumber(coeffs)

@dataclass
class RoutonNumber:
    """
    128-dimensional hypercomplex number (Routon).
    
    Routon numbers extend the Cayley-Dickson construction.
    They have 128 components and are high-dimensional algebraic structures.
    
    Note: True Routon multiplication is extremely complex (128x128 multiplication table).
    This implementation uses simplified operations for practical use.
    """
    
    coeffs: List[float] = field(default_factory=lambda: [0.0] * 128)
    
    def __post_init__(self):
        """Validate and normalize coefficients after initialization."""
        if len(self.coeffs) != 128:
            # Pad or truncate to exactly 128 components
            if len(self.coeffs) < 128:
                self.coeffs = list(self.coeffs) + [0.0] * (128 - len(self.coeffs))
            else:
                self.coeffs = self.coeffs[:128]
        
        # Ensure all are floats
        self.coeffs = [float(c) for c in self.coeffs]
    
    @classmethod
    def from_scalar(cls, value: float) -> 'RoutonNumber':
        """Create a Routon number from a scalar (real number)."""
        coeffs = [0.0] * 128
        coeffs[0] = float(value)
        return cls(coeffs)
    
    @classmethod
    def from_list(cls, values: List[float]) -> 'RoutonNumber':
        """Create from a list of up to 128 values."""
        if len(values) > 128:
            raise ValueError(f"List too long ({len(values)}), maximum 128 elements")
        coeffs = list(values) + [0.0] * (128 - len(values))
        return cls(coeffs)
    
    @classmethod
    def from_iterable(cls, values: Any) -> 'RoutonNumber':
        """Create from any iterable."""
        return cls.from_list(list(values))
    
    @classmethod
    def basis_element(cls, index: int) -> 'RoutonNumber':
        """Create a basis Routon (1 at position index, 0 elsewhere)."""
        if not 0 <= index < 128:
            raise ValueError(f"Index must be between 0 and 127, got {index}")
        coeffs = [0.0] * 128
        coeffs[index] = 1.0
        return cls(coeffs)
    
    @property
    def real(self) -> float:
        """Get the real part (first component)."""
        return self.coeffs[0]
    
    @real.setter
    def real(self, value: float):
        """Set the real part."""
        self.coeffs[0] = float(value)
    
    @property
    def imag(self) -> List[float]:
        """Get the imaginary parts (all except real)."""
        return self.coeffs[1:]
    
    def __getitem__(self, index: int) -> float:
        """Get component by index."""
        if not 0 <= index < 128:
            raise IndexError(f"Index {index} out of range for Routon")
        return self.coeffs[index]
    
    def __setitem__(self, index: int, value: float):
        """Set component by index."""
        if not 0 <= index < 128:
            raise IndexError(f"Index {index} out of range for Routon")
        self.coeffs[index] = float(value)
    
    def __len__(self) -> int:
        """Return number of components (always 128)."""
        return 128
    
    def __iter__(self):
        """Iterate over components."""
        return iter(self.coeffs)
    
    def __add__(self, other: Union['RoutonNumber', float, int]) -> 'RoutonNumber':
        """Add two Routon numbers or Routon and scalar."""
        if isinstance(other, RoutonNumber):
            new_coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
            return RoutonNumber(new_coeffs)
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __radd__(self, other: Union[float, int]) -> 'RoutonNumber':
        """Right addition: scalar + Routon."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['RoutonNumber', float, int]) -> 'RoutonNumber':
        """Subtract two Routon numbers or Routon and scalar."""
        if isinstance(other, RoutonNumber):
            new_coeffs = [a - b for a, b in zip(self.coeffs, other.coeffs)]
            return RoutonNumber(new_coeffs)
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __rsub__(self, other: Union[float, int]) -> 'RoutonNumber':
        """Right subtraction: scalar - Routon."""
        if isinstance(other, (int, float)):
            new_coeffs = [-c for c in self.coeffs]
            new_coeffs[0] += float(other)
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __mul__(self, other: Union['RoutonNumber', float, int]) -> 'RoutonNumber':
        """
        Multiply Routon by scalar or another Routon (simplified).
        
        Note: True Routon multiplication would require a 128x128 multiplication table.
        This implementation uses element-wise multiplication for Routon x Routon,
        which is mathematically incorrect but practical for many applications.
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            new_coeffs = [c * float(other) for c in self.coeffs]
            return RoutonNumber(new_coeffs)
        elif isinstance(other, RoutonNumber):
            # Simplified element-wise multiplication
            # WARNING: This is NOT true Routon multiplication!
            new_coeffs = [a * b for a, b in zip(self.coeffs, other.coeffs)]
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __rmul__(self, other: Union[float, int]) -> 'RoutonNumber':
        """Right multiplication: scalar * Routon."""
        return self.__mul__(other)
    
    def __truediv__(self, scalar: Union[float, int]) -> 'RoutonNumber':
        """Divide Routon by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide Routon by zero")
            new_coeffs = [c / float(scalar) for c in self.coeffs]
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __floordiv__(self, scalar: Union[float, int]) -> 'RoutonNumber':
        """Floor divide Routon by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide Routon by zero")
            new_coeffs = [c // float(scalar) for c in self.coeffs]
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __mod__(self, divisor: Union[float, int]) -> 'RoutonNumber':
        """Modulo operation on Routon components."""
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot take modulo by zero")
            new_coeffs = [c % float(divisor) for c in self.coeffs]
            return RoutonNumber(new_coeffs)
        return NotImplemented
    
    def __neg__(self) -> 'RoutonNumber':
        """Negate the Routon number."""
        return RoutonNumber([-c for c in self.coeffs])
    
    def __pos__(self) -> 'RoutonNumber':
        """Unary plus."""
        return self
    
    def __abs__(self) -> float:
        """Absolute value (magnitude)."""
        return self.magnitude()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another Routon."""
        if not isinstance(other, RoutonNumber):
            return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self.coeffs, other.coeffs))
    
    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        """Hash based on rounded components to avoid floating-point issues."""
        return hash(tuple(round(c, 12) for c in self.coeffs))
    
    def magnitude(self) -> float:
        """
        Calculate Euclidean norm (magnitude) of the Routon.
        
        Returns:
            float: sqrt(Σ_i coeff_i²)
        """
        return float(np.linalg.norm(self.coeffs))
    
    def norm(self) -> float:
        """Alias for magnitude."""
        return self.magnitude()
    
    def conjugate(self) -> 'RoutonNumber':
        """Return the conjugate (negate all imaginary parts)."""
        new_coeffs = self.coeffs.copy()
        for i in range(1, 128):
            new_coeffs[i] = -new_coeffs[i]
        return RoutonNumber(new_coeffs)
    
    def dot(self, other: 'RoutonNumber') -> float:
        """Dot product with another Routon."""
        if not isinstance(other, RoutonNumber):
            raise TypeError("Dot product requires another RoutonNumber")
        return sum(a * b for a, b in zip(self.coeffs, other.coeffs))
    
    def normalize(self) -> 'RoutonNumber':
        """Return a normalized (unit) version."""
        mag = self.magnitude()
        if mag == 0:
            raise ZeroDivisionError("Cannot normalize zero Routon")
        return RoutonNumber([c / mag for c in self.coeffs])
    
    def to_list(self) -> List[float]:
        """Convert to Python list."""
        return self.coeffs.copy()
    
    def to_tuple(self) -> Tuple[float, ...]:
        """Convert to tuple."""
        return tuple(self.coeffs)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.coeffs, dtype=np.float64)
    
    def copy(self) -> 'RoutonNumber':
        """Create a copy."""
        return RoutonNumber(self.coeffs.copy())
    
    def components(self) -> List[float]:
        """Get components as list (alias for to_list)."""
        return self.to_list()
    
    def phase(self) -> float:
        """
        Compute phase (angle) of the Routon number.
        
        For high-dimensional numbers, phase is not uniquely defined.
        This implementation returns the angle of the projection onto
        the real-imaginary plane.
        """
        if self.magnitude() == 0:
            return 0.0
        
        # Compute magnitude of imaginary parts
        imag_magnitude = math.sqrt(sum(i**2 for i in self.imag))
        if imag_magnitude == 0:
            return 0.0
        
        # Angle between real part and imaginary vector
        return math.atan2(imag_magnitude, self.real)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        # For 128 dimensions, show a summary
        non_zero = [(i, c) for i, c in enumerate(self.coeffs) if abs(c) > 1e-10]
        
        if not non_zero:
            return "Routon(0)"
        
        if len(non_zero) <= 5:
            # Show all non-zero components
            parts = []
            for i, c in non_zero:
                if i == 0:
                    parts.append(f"{c:.6f}")
                else:
                    sign = "+" if c >= 0 else "-"
                    parts.append(f"{sign} {abs(c):.6f}e{i}")
            return f"Routon({' '.join(parts)})"
        else:
            # Show summary
            return f"Routon[{len(non_zero)} non-zero components, real={self.real:.6f}, mag={self.magnitude():.6f}]"
    
    def __repr__(self) -> str:
        """Detailed representation showing first few components."""
        if len(self.coeffs) <= 8:
            return f"RoutonNumber({self.coeffs})"
        else:
            first_five = self.coeffs[:5]
            last_five = self.coeffs[-5:]
            return f"RoutonNumber({first_five} ... {last_five})"
    
    def summary(self) -> str:
        """Return a summary of the Routon number."""
        non_zero = sum(1 for c in self.coeffs if abs(c) > 1e-10)
        max_coeff = max(abs(c) for c in self.coeffs)
        min_coeff = min(abs(c) for c in self.coeffs if abs(c) > 0)
        
        return (f"RoutonNumber Summary:\n"
                f"  Dimensions: 128\n"
                f"  Non-zero components: {non_zero}\n"
                f"  Real part: {self.real:.6f}\n"
                f"  Magnitude: {self.magnitude():.6f}\n"
                f"  Phase: {self.phase():.6f} rad\n"
                f"  Max component: {max_coeff:.6f}\n"
                f"  Min non-zero: {min_coeff:.6f if non_zero > 0 else 0}")

@dataclass
class VoudonNumber:
    """
    256-dimensional hypercomplex number (Voudon).
    
    Voudon numbers extend the Cayley-Dickson construction beyond sedenions.
    They have 256 components and are extremely high-dimensional algebraic structures.
    
    Note: True Voudon multiplication is extremely complex (256x256 multiplication table).
    This implementation uses simplified operations for practical use.
    """
    
    coeffs: List[float] = field(default_factory=lambda: [0.0] * 256)
    
    def __post_init__(self):
        """Validate and normalize coefficients after initialization."""
        if len(self.coeffs) != 256:
            # Pad or truncate to exactly 256 components
            if len(self.coeffs) < 256:
                self.coeffs = list(self.coeffs) + [0.0] * (256 - len(self.coeffs))
            else:
                self.coeffs = self.coeffs[:256]
        
        # Ensure all are floats
        self.coeffs = [float(c) for c in self.coeffs]
    
    @classmethod
    def from_scalar(cls, value: float) -> 'VoudonNumber':
        """Create a Voudon number from a scalar (real number)."""
        coeffs = [0.0] * 256
        coeffs[0] = float(value)
        return cls(coeffs)
    
    @classmethod
    def from_list(cls, values: List[float]) -> 'VoudonNumber':
        """Create from a list of up to 256 values."""
        if len(values) > 256:
            raise ValueError(f"List too long ({len(values)}), maximum 256 elements")
        coeffs = list(values) + [0.0] * (256 - len(values))
        return cls(coeffs)
    
    @classmethod
    def from_iterable(cls, values: Any) -> 'VoudonNumber':
        """Create from any iterable."""
        return cls.from_list(list(values))
    
    @classmethod
    def basis_element(cls, index: int) -> 'VoudonNumber':
        """Create a basis Voudon (1 at position index, 0 elsewhere)."""
        if not 0 <= index < 256:
            raise ValueError(f"Index must be between 0 and 255, got {index}")
        coeffs = [0.0] * 256
        coeffs[index] = 1.0
        return cls(coeffs)
    
    @property
    def real(self) -> float:
        """Get the real part (first component)."""
        return self.coeffs[0]
    
    @real.setter
    def real(self, value: float):
        """Set the real part."""
        self.coeffs[0] = float(value)
    
    @property
    def imag(self) -> List[float]:
        """Get the imaginary parts (all except real)."""
        return self.coeffs[1:]
    
    def __getitem__(self, index: int) -> float:
        """Get component by index."""
        if not 0 <= index < 256:
            raise IndexError(f"Index {index} out of range for Voudon")
        return self.coeffs[index]
    
    def __setitem__(self, index: int, value: float):
        """Set component by index."""
        if not 0 <= index < 256:
            raise IndexError(f"Index {index} out of range for Voudon")
        self.coeffs[index] = float(value)
    
    def __len__(self) -> int:
        """Return number of components (always 256)."""
        return 256
    
    def __iter__(self):
        """Iterate over components."""
        return iter(self.coeffs)
    
    def __add__(self, other: Union['VoudonNumber', float, int]) -> 'VoudonNumber':
        """Add two Voudon numbers or Voudon and scalar."""
        if isinstance(other, VoudonNumber):
            new_coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
            return VoudonNumber(new_coeffs)
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __radd__(self, other: Union[float, int]) -> 'VoudonNumber':
        """Right addition: scalar + Voudon."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['VoudonNumber', float, int]) -> 'VoudonNumber':
        """Subtract two Voudon numbers or Voudon and scalar."""
        if isinstance(other, VoudonNumber):
            new_coeffs = [a - b for a, b in zip(self.coeffs, other.coeffs)]
            return VoudonNumber(new_coeffs)
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __rsub__(self, other: Union[float, int]) -> 'VoudonNumber':
        """Right subtraction: scalar - Voudon."""
        if isinstance(other, (int, float)):
            new_coeffs = [-c for c in self.coeffs]
            new_coeffs[0] += float(other)
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __mul__(self, other: Union['VoudonNumber', float, int]) -> 'VoudonNumber':
        """
        Multiply Voudon by scalar or another Voudon (simplified).
        
        Note: True Voudon multiplication would require a 256x256 multiplication table.
        This implementation uses element-wise multiplication for Voudon x Voudon,
        which is mathematically incorrect but practical for many applications.
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            new_coeffs = [c * float(other) for c in self.coeffs]
            return VoudonNumber(new_coeffs)
        elif isinstance(other, VoudonNumber):
            # Simplified element-wise multiplication
            # WARNING: This is NOT true Voudon multiplication!
            new_coeffs = [a * b for a, b in zip(self.coeffs, other.coeffs)]
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __rmul__(self, other: Union[float, int]) -> 'VoudonNumber':
        """Right multiplication: scalar * Voudon."""
        return self.__mul__(other)
    
    def __truediv__(self, scalar: Union[float, int]) -> 'VoudonNumber':
        """Divide Voudon by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide Voudon by zero")
            new_coeffs = [c / float(scalar) for c in self.coeffs]
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __floordiv__(self, scalar: Union[float, int]) -> 'VoudonNumber':
        """Floor divide Voudon by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide Voudon by zero")
            new_coeffs = [c // float(scalar) for c in self.coeffs]
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __mod__(self, divisor: Union[float, int]) -> 'VoudonNumber':
        """Modulo operation on Voudon components."""
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot take modulo by zero")
            new_coeffs = [c % float(divisor) for c in self.coeffs]
            return VoudonNumber(new_coeffs)
        return NotImplemented
    
    def __neg__(self) -> 'VoudonNumber':
        """Negate the Voudon number."""
        return VoudonNumber([-c for c in self.coeffs])
    
    def __pos__(self) -> 'VoudonNumber':
        """Unary plus."""
        return self
    
    def __abs__(self) -> float:
        """Absolute value (magnitude)."""
        return self.magnitude()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another Voudon."""
        if not isinstance(other, VoudonNumber):
            return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self.coeffs, other.coeffs))
    
    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        """Hash based on rounded components to avoid floating-point issues."""
        return hash(tuple(round(c, 12) for c in self.coeffs))
    
    def magnitude(self) -> float:
        """
        Calculate Euclidean norm (magnitude) of the Voudon.
        
        Returns:
            float: sqrt(Σ_i coeff_i²)
        """
        return float(np.linalg.norm(self.coeffs))
    
    def norm(self) -> float:
        """Alias for magnitude."""
        return self.magnitude()
    
    def conjugate(self) -> 'VoudonNumber':
        """Return the conjugate (negate all imaginary parts)."""
        new_coeffs = self.coeffs.copy()
        for i in range(1, 256):
            new_coeffs[i] = -new_coeffs[i]
        return VoudonNumber(new_coeffs)
    
    def dot(self, other: 'VoudonNumber') -> float:
        """Dot product with another Voudon."""
        if not isinstance(other, VoudonNumber):
            raise TypeError("Dot product requires another VoudonNumber")
        return sum(a * b for a, b in zip(self.coeffs, other.coeffs))
    
    def normalize(self) -> 'VoudonNumber':
        """Return a normalized (unit) version."""
        mag = self.magnitude()
        if mag == 0:
            raise ZeroDivisionError("Cannot normalize zero Voudon")
        return VoudonNumber([c / mag for c in self.coeffs])
    
    def to_list(self) -> List[float]:
        """Convert to Python list."""
        return self.coeffs.copy()
    
    def to_tuple(self) -> Tuple[float, ...]:
        """Convert to tuple."""
        return tuple(self.coeffs)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.coeffs, dtype=np.float64)
    
    def copy(self) -> 'VoudonNumber':
        """Create a copy."""
        return VoudonNumber(self.coeffs.copy())
    
    def components(self) -> List[float]:
        """Get components as list (alias for to_list)."""
        return self.to_list()
    
    def phase(self) -> float:
        """
        Compute phase (angle) of the Voudon number.
        
        For high-dimensional numbers, phase is not uniquely defined.
        This implementation returns the angle of the projection onto
        the real-imaginary plane.
        """
        if self.magnitude() == 0:
            return 0.0
        
        # Compute magnitude of imaginary parts
        imag_magnitude = math.sqrt(sum(i**2 for i in self.imag))
        if imag_magnitude == 0:
            return 0.0
        
        # Angle between real part and imaginary vector
        return math.atan2(imag_magnitude, self.real)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        # For 256 dimensions, show a summary
        non_zero = [(i, c) for i, c in enumerate(self.coeffs) if abs(c) > 1e-10]
        
        if not non_zero:
            return "Voudon(0)"
        
        if len(non_zero) <= 5:
            # Show all non-zero components
            parts = []
            for i, c in non_zero:
                if i == 0:
                    parts.append(f"{c:.6f}")
                else:
                    sign = "+" if c >= 0 else "-"
                    parts.append(f"{sign} {abs(c):.6f}e{i}")
            return f"Voudon({' '.join(parts)})"
        else:
            # Show summary
            return f"Voudon[{len(non_zero)} non-zero components, real={self.real:.6f}, mag={self.magnitude():.6f}]"
    
    def __repr__(self) -> str:
        """Detailed representation showing first few components."""
        if len(self.coeffs) <= 8:
            return f"VoudonNumber({self.coeffs})"
        else:
            first_five = self.coeffs[:5]
            last_five = self.coeffs[-5:]
            return f"VoudonNumber({first_five} ... {last_five})"
    
    def summary(self) -> str:
        """Return a summary of the Voudon number."""
        non_zero = sum(1 for c in self.coeffs if abs(c) > 1e-10)
        max_coeff = max(abs(c) for c in self.coeffs)
        min_coeff = min(abs(c) for c in self.coeffs if abs(c) > 0)
        
        return (f"VoudonNumber Summary:\n"
                f"  Dimensions: 256\n"
                f"  Non-zero components: {non_zero}\n"
                f"  Real part: {self.real:.6f}\n"
                f"  Magnitude: {self.magnitude():.6f}\n"
                f"  Phase: {self.phase():.6f} rad\n"
                f"  Max component: {max_coeff:.6f}\n"
                f"  Min non-zero: {min_coeff:.6f if non_zero > 0 else 0}")

@dataclass
class OctonionNumber:
    """
    Represents an octonion number with 8 components.
    Implements octonion multiplication rules (non-commutative, non-associative).
    
    Octonions are 8-dimensional hypercomplex numbers that extend quaternions.
    They have applications in string theory, quantum mechanics, and geometry.
    
    Attributes:
    ----------
    w, x, y, z, e, f, g, h : float
        The 8 components of the octonion
    """
    w: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    
    # Private field for phase computation
    _phase: float = field(init=False, default=0.0)
    
    def __post_init__(self):
        """Initialize phase after the object is created."""
        self._compute_phase()
    
    @classmethod
    def from_list(cls, components: List[float]) -> 'OctonionNumber':
        """
        Create OctonionNumber from a list of components.
        
        Args:
            components: List of 1-8 float values
        
        Returns:
            OctonionNumber instance
        """
        if len(components) == 8:
            return cls(*components)
        elif len(components) < 8:
            # Pad with zeros if less than 8 components
            padded = list(components) + [0.0] * (8 - len(components))
            return cls(*padded)
        else:
            # Truncate if more than 8 components
            return cls(*components[:8])
    
    @classmethod
    def from_scalar(cls, scalar: float) -> 'OctonionNumber':
        """
        Create OctonionNumber from a scalar (real number).
        
        Args:
            scalar: Real number to convert to octonion
        
        Returns:
            OctonionNumber with scalar as real part, others zero
        """
        return cls(w=float(scalar))
    
    @classmethod
    def from_complex(cls, z: complex) -> 'OctonionNumber':
        """
        Create OctonionNumber from a complex number.
        
        Args:
            z: Complex number to convert to octonion
        
        Returns:
            OctonionNumber with complex as first two components
        """
        return cls(w=z.real, x=z.imag)
    
    @property
    def coeffs(self) -> List[float]:
        """Get all components as a list."""
        return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]
    
    @property
    def real(self) -> float:
        """Get the real part (first component)."""
        return self.w
    
    @real.setter
    def real(self, value: float):
        """Set the real part."""
        self.w = float(value)
        self._compute_phase()
    
    @property
    def imag(self) -> List[float]:
        """Get the imaginary parts (all except real)."""
        return [self.x, self.y, self.z, self.e, self.f, self.g, self.h]
    
    def _compute_phase(self) -> None:
        """Compute and store the phase (angle) of the octonion."""
        magnitude = self.magnitude()
        if magnitude == 0:
            self._phase = 0.0
        else:
            # For octonions, phase is not uniquely defined.
            # We use the angle of the projection onto the real-imaginary plane
            imag_magnitude = np.sqrt(sum(i**2 for i in self.imag))
            if imag_magnitude == 0:
                self._phase = 0.0
            else:
                # Angle between real part and imaginary vector
                self._phase = np.arctan2(imag_magnitude, self.real)
    
    def components(self) -> List[float]:
        """Get components as list (alias for coeffs)."""
        return self.coeffs
    
    def magnitude(self) -> float:
        """
        Calculate the Euclidean norm (magnitude) of the octonion.
        
        Returns:
            float: sqrt(w² + x² + y² + z² + e² + f² + g² + h²)
        """
        return float(np.linalg.norm(self.coeffs))
    
    def norm(self) -> float:
        """Alias for magnitude."""
        return self.magnitude()
    
    def conjugate(self) -> 'OctonionNumber':
        """
        Return the conjugate of the octonion.
        
        Returns:
            OctonionNumber with signs of imaginary parts flipped
        """
        return OctonionNumber(
            self.w, -self.x, -self.y, -self.z,
            -self.e, -self.f, -self.g, -self.h
        )
    
    def inverse(self) -> 'OctonionNumber':
        """
        Return the multiplicative inverse.
        
        Returns:
            OctonionNumber: o⁻¹ such that o * o⁻¹ = o⁻¹ * o = 1
        
        Raises:
            ZeroDivisionError: If magnitude is zero
        """
        mag_sq = self.magnitude() ** 2
        if mag_sq == 0:
            raise ZeroDivisionError("Cannot invert zero octonion")
        conj = self.conjugate()
        return OctonionNumber(
            conj.w / mag_sq, conj.x / mag_sq, conj.y / mag_sq, conj.z / mag_sq,
            conj.e / mag_sq, conj.f / mag_sq, conj.g / mag_sq, conj.h / mag_sq
        )
    
    def dot(self, other: 'OctonionNumber') -> float:
        """
        Compute the dot product with another octonion.
        
        Args:
            other: Another OctonionNumber
        
        Returns:
            float: Dot product (sum of component-wise products)
        """
        if not isinstance(other, OctonionNumber):
            raise TypeError("Dot product requires another OctonionNumber")
        
        return sum(a * b for a, b in zip(self.coeffs, other.coeffs))
    
    def normalize(self) -> 'OctonionNumber':
        """
        Return a normalized (unit) version of this octonion.
        
        Returns:
            OctonionNumber with magnitude 1
        
        Raises:
            ZeroDivisionError: If magnitude is zero
        """
        mag = self.magnitude()
        if mag == 0:
            raise ZeroDivisionError("Cannot normalize zero octonion")
        
        return OctonionNumber(
            self.w / mag, self.x / mag, self.y / mag, self.z / mag,
            self.e / mag, self.f / mag, self.g / mag, self.h / mag
        )
    
    def phase(self) -> float:
        """
        Get the phase (angle) of the octonion.
        
        Returns:
            float: Phase angle in radians
        """
        return self._phase
    
    # Operator overloads
    def __add__(self, other: Union['OctonionNumber', float, int]) -> 'OctonionNumber':
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        elif isinstance(other, (int, float)):
            return OctonionNumber(self.w + other, self.x, self.y, self.z,
                                 self.e, self.f, self.g, self.h)
        return NotImplemented
    
    def __radd__(self, other: Union[float, int]) -> 'OctonionNumber':
        return self.__add__(other)
    
    def __sub__(self, other: Union['OctonionNumber', float, int]) -> 'OctonionNumber':
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z,
                self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h
            )
        elif isinstance(other, (int, float)):
            return OctonionNumber(self.w - other, self.x, self.y, self.z,
                                 self.e, self.f, self.g, self.h)
        return NotImplemented
    
    def __rsub__(self, other: Union[float, int]) -> 'OctonionNumber':
        if isinstance(other, (int, float)):
            return OctonionNumber(
                other - self.w, -self.x, -self.y, -self.z,
                -self.e, -self.f, -self.g, -self.h
            )
        return NotImplemented
    
    def __mul__(self, other: Union['OctonionNumber', float, int]) -> 'OctonionNumber':
        if isinstance(other, OctonionNumber):
            # Octonion multiplication (non-commutative, non-associative)
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
            # Scalar multiplication
            return OctonionNumber(
                self.w * other, self.x * other, self.y * other, self.z * other,
                self.e * other, self.f * other, self.g * other, self.h * other
            )
        
        return NotImplemented
    
    def __rmul__(self, other: Union[float, int]) -> 'OctonionNumber':
        return self.__mul__(other)
    
    def __truediv__(self, scalar: Union[float, int]) -> 'OctonionNumber':
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide octonion by zero")
            return OctonionNumber(
                self.w / scalar, self.x / scalar, self.y / scalar, self.z / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented
    
    def __neg__(self) -> 'OctonionNumber':
        return OctonionNumber(
            -self.w, -self.x, -self.y, -self.z,
            -self.e, -self.f, -self.g, -self.h
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OctonionNumber):
            return False
        
        tol = 1e-12
        return all(abs(a - b) < tol for a, b in zip(self.coeffs, other.coeffs))
    
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        # Round to avoid floating-point precision issues
        return hash(tuple(round(c, 10) for c in self.coeffs))
    
    def __str__(self) -> str:
        return f"Octonion({self.w:.6f}, {self.x:.6f}, {self.y:.6f}, {self.z:.6f}, " \
               f"{self.e:.6f}, {self.f:.6f}, {self.g:.6f}, {self.h:.6f})"
    
    def __repr__(self) -> str:
        return f"OctonionNumber({self.w}, {self.x}, {self.y}, {self.z}, " \
               f"{self.e}, {self.f}, {self.g}, {self.h})"
    
    def to_tuple(self) -> Tuple[float, ...]:
        """Convert to tuple."""
        return tuple(self.coeffs)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.coeffs, dtype=np.float64)
    
    def copy(self) -> 'OctonionNumber':
        """Create a copy of this octonion."""
        return OctonionNumber(*self.coeffs)

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

class Constants:
    """Oktonyon sabitleri (alias'lar)."""
    ZERO = ZERO
    ONE = ONE
    I = I
    J = J
    K = K
    E = E
    F = F
    G = G
    H = H

from dataclasses import dataclass, field
from typing import Union, List, Any, Tuple, Optional
import math

# Ana nötrözofik sayı sınıfı
@dataclass
class NeutrosophicNumber:
    """
    Represents a neutrosophic number of the form t + iI + fF.
    t = truth value
    i = indeterminacy value
    f = falsity value
    """
    t: float = 0.0
    i: float = 0.0
    f: float = 0.0
    
    def __post_init__(self):
        """Değerleri normalize et ve kontrol et"""
        self.t = float(self.t)
        self.i = float(self.i)
        self.f = float(self.f)
        
        # Normalizasyon (isteğe bağlı)
        # total = abs(self.t) + abs(self.i) + abs(self.f)
        # if total > 0:
        #     self.t /= total
        #     self.i /= total
        #     self.f /= total
    
    # Temel operatörler
    def __add__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(
                self.t + other.t,
                self.i + other.i,
                self.f + other.f
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t + other, self.i, self.f)
        return NotImplemented
    
    def __radd__(self, other: Any) -> "NeutrosophicNumber":
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(
                self.t - other.t,
                self.i - other.i,
                self.f - other.f
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t - other, self.i, self.f)
        return NotImplemented
    
    def __rsub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(
                other - self.t,
                -self.i,
                -self.f
            )
        return NotImplemented
    
    def __mul__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            # Nötrözofik çarpma: (t1 + i1I + f1F) * (t2 + i2I + f2F)
            return NeutrosophicNumber(
                t=self.t * other.t,
                i=self.t * other.i + self.i * other.t + self.i * other.i,
                f=self.t * other.f + self.f * other.t + self.f * other.f
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicNumber(
                self.t * other,
                self.i * other,
                self.f * other
            )
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "NeutrosophicNumber":
        return self.__mul__(other)
    
    def __truediv__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicNumber(
                self.t / other,
                self.i / other,
                self.f / other
            )
        return NotImplemented
    
    def __rtruediv__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(
                other / self.t if self.t != 0 else float('inf'),
                other / self.i if self.i != 0 else float('inf'),
                other / self.f if self.f != 0 else float('inf')
            )
        return NotImplemented
    
    def __floordiv__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicNumber(
                self.t // other,
                self.i // other,
                self.f // other
            )
        return NotImplemented
    
    def __mod__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot modulo by zero")
            return NeutrosophicNumber(
                self.t % other,
                self.i % other,
                self.f % other
            )
        return NotImplemented
    
    def __pow__(self, exponent: Any) -> "NeutrosophicNumber":
        if isinstance(exponent, (int, float)):
            return NeutrosophicNumber(
                self.t ** exponent,
                self.i ** exponent,
                self.f ** exponent
            )
        return NotImplemented
    
    def __neg__(self) -> "NeutrosophicNumber":
        return NeutrosophicNumber(-self.t, -self.i, -self.f)
    
    def __pos__(self) -> "NeutrosophicNumber":
        return self
    
    def __abs__(self) -> "NeutrosophicNumber":
        return NeutrosophicNumber(abs(self.t), abs(self.i), abs(self.f))
    
    # Karşılaştırma operatörleri
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NeutrosophicNumber):
            return NotImplemented
        return (
            math.isclose(self.t, other.t, abs_tol=1e-12) and
            math.isclose(self.i, other.i, abs_tol=1e-12) and
            math.isclose(self.f, other.f, abs_tol=1e-12)
        )
    
    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)
    
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicNumber):
            # Nötrözofik sıralama (gerçek kısım üzerinden)
            return self.t < other.t
        return NotImplemented
    
    def __le__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicNumber):
            return self.t <= other.t
        return NotImplemented
    
    def __gt__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicNumber):
            return self.t > other.t
        return NotImplemented
    
    def __ge__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicNumber):
            return self.t >= other.t
        return NotImplemented
    
    # String temsilleri
    def __str__(self) -> str:
        parts = []
        if abs(self.t) > 1e-12:
            parts.append(f"{self.t:.6g}")
        if abs(self.i) > 1e-12:
            parts.append(f"{self.i:.6g}I")
        if abs(self.f) > 1e-12:
            parts.append(f"{self.f:.6g}F")
        return " + ".join(parts) if parts else "0"
    
    def __repr__(self) -> str:
        return f"NeutrosophicNumber(t={self.t}, i={self.i}, f={self.f})"
    
    # Yardımcı metodlar
    def conjugate(self) -> "NeutrosophicNumber":
        """Nötrözofik eşlenik (işaret değişimi)"""
        return NeutrosophicNumber(self.t, -self.i, -self.f)
    
    def magnitude(self) -> float:
        """Büyüklük (Euclidean norm)"""
        return math.sqrt(self.t**2 + self.i**2 + self.f**2)
    
    def normalized(self) -> "NeutrosophicNumber":
        """Birim büyüklüğe normalize edilmiş nötrözofik sayı"""
        mag = self.magnitude()
        if mag == 0:
            return NeutrosophicNumber(0, 0, 0)
        return self / mag
    
    def score(self) -> float:
        """Net skor: t - f"""
        return self.t - self.f
    
    def uncertainty(self) -> float:
        """Belirsizlik seviyesi"""
        return self.i
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Tuple temsili"""
        return (self.t, self.i, self.f)
    
    @classmethod
    def from_tuple(cls, tpl: Tuple[float, float, float]) -> "NeutrosophicNumber":
        """Tuple'dan oluştur"""
        return cls(*tpl)
    
    @classmethod
    def truth(cls, value: float) -> "NeutrosophicNumber":
        """Sadece gerçek değer içeren nötrözofik sayı"""
        return cls(t=value, i=0.0, f=0.0)
    
    @classmethod
    def indeterminacy(cls, value: float) -> "NeutrosophicNumber":
        """Sadece belirsizlik içeren nötrözofik sayı"""
        return cls(t=0.0, i=value, f=0.0)
    
    @classmethod
    def falsity(cls, value: float) -> "NeutrosophicNumber":
        """Sadece yanlışlık içeren nötrözofik sayı"""
        return cls(t=0.0, i=0.0, f=value)


# Nötrözofik Karmaşık Sayı Sınıfı
@dataclass
class NeutrosophicComplexNumber:
    """
    Represents a neutrosophic complex number: (a + bj) + cI
    where I is the indeterminacy unit
    """
    real: float = 0.0
    imag: float = 0.0
    indeterminacy: float = 0.0
    
    def __post_init__(self):
        self.real = float(self.real)
        self.imag = float(self.imag)
        self.indeterminacy = float(self.indeterminacy)
    
    @property
    def complex_part(self) -> complex:
        """Karmaşık kısmı döndür"""
        return complex(self.real, self.imag)
    
    # Operatörler
    def __add__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy + other.indeterminacy
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real + other,
                self.imag,
                self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy
            )
        return NotImplemented
    
    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real - other,
                self.imag,
                self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy
            )
        return NotImplemented
    
    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                other - self.real,
                -self.imag,
                -self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                other.real - self.real,
                other.imag - self.imag,
                -self.indeterminacy
            )
        return NotImplemented
    
    def __mul__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            # Karmaşık çarpma + belirsizlik yayılımı
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            
            # Belirsizlik yayılımı (basitleştirilmiş model)
            mag_sq_self = self.real**2 + self.imag**2
            mag_sq_other = other.real**2 + other.imag**2
            new_indeterminacy = (
                self.indeterminacy + other.indeterminacy +
                mag_sq_self * other.indeterminacy +
                mag_sq_other * self.indeterminacy
            )
            
            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        elif isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real * other,
                self.imag * other,
                self.indeterminacy * other
            )
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)
    
    def __truediv__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicComplexNumber(
                self.real / other,
                self.imag / other,
                self.indeterminacy / other
            )
        return NotImplemented
    
    def __neg__(self) -> "NeutrosophicComplexNumber":
        return NeutrosophicComplexNumber(-self.real, -self.imag, -self.indeterminacy)
    
    def __abs__(self) -> float:
        """Büyüklük (karmaşık norm + belirsizlik)"""
        complex_mag = math.sqrt(self.real**2 + self.imag**2)
        return math.sqrt(complex_mag**2 + self.indeterminacy**2)
    
    # Karşılaştırma
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NeutrosophicComplexNumber):
            return NotImplemented
        return (
            math.isclose(self.real, other.real, abs_tol=1e-12) and
            math.isclose(self.imag, other.imag, abs_tol=1e-12) and
            math.isclose(self.indeterminacy, other.indeterminacy, abs_tol=1e-12)
        )
    
    # String temsilleri
    def __str__(self) -> str:
        parts = []
        if abs(self.real) > 1e-12 or abs(self.imag) > 1e-12:
            if abs(self.imag) < 1e-12:
                parts.append(f"{self.real:.6g}")
            else:
                parts.append(f"({self.real:.6g}{self.imag:+.6g}j)")
        if abs(self.indeterminacy) > 1e-12:
            parts.append(f"{self.indeterminacy:.6g}I")
        return " + ".join(parts) if parts else "0"
    
    def __repr__(self) -> str:
        return f"NeutrosophicComplexNumber(real={self.real}, imag={self.imag}, indeterminacy={self.indeterminacy})"
    
    # Yardımcı metodlar
    def conjugate(self) -> "NeutrosophicComplexNumber":
        """Karmaşık eşlenik alır, belirsizlik değişmez"""
        return NeutrosophicComplexNumber(self.real, -self.imag, self.indeterminacy)
    
    def magnitude_sq(self) -> float:
        """Karmaşık kısmın büyüklüğünün karesi"""
        return self.real**2 + self.imag**2
    
    def phase(self) -> float:
        """Faz açısı"""
        if abs(self.real) < 1e-12 and abs(self.imag) < 1e-12:
            return 0.0
        return math.atan2(self.imag, self.real)
    
    def to_polar(self) -> Tuple[float, float, float]:
        """Kutupsal koordinatlara dönüşüm"""
        r = math.sqrt(self.real**2 + self.imag**2)
        theta = self.phase()
        return (r, theta, self.indeterminacy)
    
    @classmethod
    def from_polar(cls, r: float, theta: float, indeterminacy: float = 0.0) -> "NeutrosophicComplexNumber":
        """Kutupsal koordinatlardan oluştur"""
        return cls(r * math.cos(theta), r * math.sin(theta), indeterminacy)


# Hiperreel Sayı Sınıfı
@dataclass
class HyperrealNumber:
    """
    Represents a hyperreal number as a sequence.
    Provides infinitesimal and infinite number support.
    """
    sequence: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.sequence:
            self.sequence = [0.0]
        self.sequence = [float(x) for x in self.sequence]
    
    @property
    def finite(self) -> float:
        """Sonlu kısım"""
        return self.sequence[0] if self.sequence else 0.0
    
    @property
    def infinitesimal(self) -> float:
        """Birinci sonsuz küçük kısım"""
        return self.sequence[1] if len(self.sequence) > 1 else 0.0
    
    @property
    def second_infinitesimal(self) -> float:
        """İkinci sonsuz küçük kısım"""
        return self.sequence[2] if len(self.sequence) > 2 else 0.0
    
    # Operatörler
    def __add__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a + b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] += other
            return HyperrealNumber(new_seq)
        return NotImplemented
    
    def __radd__(self, other: Any) -> "HyperrealNumber":
        return self.__add__(other)
    
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
    
    def __rsub__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, (int, float)):
            new_seq = [-x for x in self.sequence]
            new_seq[0] += other
            return HyperrealNumber(new_seq)
        return NotImplemented
    
    def __mul__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            # Hiperreel çarpma: sadece ilk birkaç terim için
            result = [0.0] * (len(self.sequence) + len(other.sequence) - 1)
            for i, a in enumerate(self.sequence):
                for j, b in enumerate(other.sequence):
                    result[i + j] += a * b
            return HyperrealNumber(result)
        elif isinstance(other, (int, float)):
            return HyperrealNumber([x * other for x in self.sequence])
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "HyperrealNumber":
        return self.__mul__(other)
    
    def __truediv__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return HyperrealNumber([x / other for x in self.sequence])
        return NotImplemented
    
    def __str__(self) -> str:
        if len(self.sequence) <= 3:
            return f"HyperrealNumber{self.sequence}"
        else:
            truncated = self.sequence[:3]
            return f"HyperrealNumber{truncated}..."
    
    def __repr__(self) -> str:
        return f"HyperrealNumber(sequence={self.sequence})"
    
    @classmethod
    def infinitesimal_number(cls, value: float = 1.0) -> "HyperrealNumber":
        """Sonsuz küçük sayı oluştur"""
        return cls([0.0, value])
    
    @classmethod
    def infinite_number(cls, value: float = 1.0) -> "HyperrealNumber":
        """Sonsuz büyük sayı oluştur"""
        return cls([value, 0.0, 0.0, 1.0])  # Son terim sonsuzluk için
    
    def is_infinitesimal(self) -> bool:
        """Sonsuz küçük mü?"""
        return abs(self.finite) < 1e-12 and any(abs(x) > 1e-12 for x in self.sequence[1:])
    
    def is_infinite(self) -> bool:
        """Sonsuz büyük mü?"""
        return abs(self.finite) > 1e8  # Basit bir eşik değeri


# Bikompleks Sayı Sınıfı
@dataclass
class BicomplexNumber:
    """
    Represents a bicomplex number: z1 + z2 * e
    where e is a new imaginary unit with e² = -1
    """
    z1: complex = 0j
    z2: complex = 0j
    
    def __post_init__(self):
        self.z1 = complex(self.z1)
        self.z2 = complex(self.z2)
    
    @property
    def real(self) -> float:
        """Gerçek kısım (z1'in gerçek kısmı)"""
        return self.z1.real
    
    @property
    def imag1(self) -> float:
        """Birinci sanal kısım (z1'in sanal kısmı)"""
        return self.z1.imag
    
    @property
    def imag2(self) -> float:
        """İkinci sanal kısım (z2'nin gerçek kısmı)"""
        return self.z2.real
    
    @property
    def imag3(self) -> float:
        """Üçüncü sanal kısım (z2'nin sanal kısmı)"""
        return self.z2.imag
    
    # Operatörler
    def __add__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        elif isinstance(other, (int, float)):
            return BicomplexNumber(self.z1 + other, self.z2)
        elif isinstance(other, complex):
            return BicomplexNumber(self.z1 + other, self.z2)
        return NotImplemented
    
    def __radd__(self, other: Any) -> "BicomplexNumber":
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        elif isinstance(other, (int, float)):
            return BicomplexNumber(self.z1 - other, self.z2)
        elif isinstance(other, complex):
            return BicomplexNumber(self.z1 - other, self.z2)
        return NotImplemented
    
    def __rsub__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, (int, float)):
            return BicomplexNumber(other - self.z1, -self.z2)
        elif isinstance(other, complex):
            return BicomplexNumber(other - self.z1, -self.z2)
        return NotImplemented
    
    def __mul__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            # Bikompleks çarpma
            return BicomplexNumber(
                self.z1 * other.z1 - self.z2 * other.z2,
                self.z1 * other.z2 + self.z2 * other.z1
            )
        elif isinstance(other, (int, float)):
            return BicomplexNumber(self.z1 * other, self.z2 * other)
        elif isinstance(other, complex):
            return BicomplexNumber(self.z1 * other, self.z2 * other)
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "BicomplexNumber":
        return self.__mul__(other)
    
    def __truediv__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return BicomplexNumber(self.z1 / other, self.z2 / other)
        return NotImplemented
    
    def __neg__(self) -> "BicomplexNumber":
        return BicomplexNumber(-self.z1, -self.z2)
    
    def __abs__(self) -> float:
        """Büyüklük"""
        return math.sqrt(abs(self.z1)**2 + abs(self.z2)**2)
    
    # String temsilleri
    def __str__(self) -> str:
        parts = []
        if self.z1 != 0j:
            parts.append(f"({self.z1.real:.6g}{self.z1.imag:+.6g}j)")
        if self.z2 != 0j:
            parts.append(f"({self.z2.real:.6g}{self.z2.imag:+.6g}j)e")
        return " + ".join(parts) if parts else "0"
    
    def __repr__(self) -> str:
        return f"BicomplexNumber(z1={self.z1}, z2={self.z2})"
    
    # Yardımcı metodlar
    def conjugate1(self) -> "BicomplexNumber":
        """Birinci eşlenik (z1'in eşleniği)"""
        return BicomplexNumber(self.z1.conjugate(), self.z2)
    
    def conjugate2(self) -> "BicomplexNumber":
        """İkinci eşlenik (z2'nin eşleniği)"""
        return BicomplexNumber(self.z1, self.z2.conjugate())
    
    def full_conjugate(self) -> "BicomplexNumber":
        """Tam eşlenik (her iki parçanın eşleniği)"""
        return BicomplexNumber(self.z1.conjugate(), self.z2.conjugate())
    
    def norm_sq(self) -> float:
        """Normun karesi"""
        return abs(self.z1)**2 + abs(self.z2)**2
    
    def inverse(self) -> "BicomplexNumber":
        """Ters eleman (eğer varsa)"""
        norm_sq = self.norm_sq()
        if norm_sq == 0:
            raise ZeroDivisionError("BicomplexNumber number has no inverse")
        return BicomplexNumber(self.z1.conjugate() / norm_sq, -self.z2 / norm_sq)


# Fabrika fonksiyonları
def neutrosophic_zero() -> NeutrosophicNumber:
    """Sıfır nötrözofik sayı"""
    return NeutrosophicNumber(0, 0, 0)

def neutrosophic_one() -> NeutrosophicNumber:
    """Bir nötrözofik sayı"""
    return NeutrosophicNumber(1, 0, 0)

def neutrosophic_i() -> NeutrosophicNumber:
    """Belirsizlik birimi"""
    return NeutrosophicNumber(0, 1, 0)

def neutrosophic_f() -> NeutrosophicNumber:
    """Yanlışlık birimi"""
    return NeutrosophicNumber(0, 0, 1)

def _parse_bicomplex(s: Any) -> BicomplexNumber:
    """
    Universally parse input into a BicomplexNumber.
    
    Features from both versions combined:
    1. Type checking and direct returns for BicomplexNumber
    2. Handles numeric types (int, float, numpy) -> z1 = num, z2 = 0
    3. Handles complex numbers -> z1 = complex, z2 = 0
    4. Handles iterables (list, tuple) of 1, 2, or 4 numbers
    5. String parsing with multiple formats:
       - Comma-separated "a,b,c,d" or "a,b" or "a"
       - Explicit "(a+bj)+(c+dj)e" format
       - Complex strings like "1+2j", "3j", "4"
       - Fallback to complex() parsing
    6. Robust error handling with logging
    7. Final fallback to zero bicomplex
    
    Args:
        s: Input to parse (any type)
    
    Returns:
        Parsed BicomplexNumber or BicomplexNumber(0,0) on failure
    """
    try:
        # Feature 1: Direct return if already BicomplexNumber
        if isinstance(s, BicomplexNumber):
            return s

        # Feature 2: Handle numeric scalars
        if isinstance(s, (int, float, np.floating, np.integer)):
            return BicomplexNumber(complex(float(s), 0.0), complex(0.0, 0.0))

        # Feature 3: Handle complex numbers
        if isinstance(s, complex):
            return BicomplexNumber(s, complex(0.0, 0.0))

        # Feature 4: Handle iterables (non-string)
        if hasattr(s, '__iter__') and not isinstance(s, str):
            parts = list(s)
            if len(parts) == 4:
                # Four numbers: (z1_real, z1_imag, z2_real, z2_imag)
                return BicomplexNumber(
                    complex(float(parts[0]), float(parts[1])),
                    complex(float(parts[2]), float(parts[3]))
                )
            elif len(parts) == 2:
                # Two numbers: (real, imag) for z1
                return BicomplexNumber(
                    complex(float(parts[0]), float(parts[1])),
                    complex(0.0, 0.0)
                )
            elif len(parts) == 1:
                # Single number: real part of z1
                return BicomplexNumber(
                    complex(float(parts[0]), 0.0),
                    complex(0.0, 0.0)
                )

        # Convert to string for further parsing
        if not isinstance(s, str):
            s = str(s)

        s_clean = s.strip().replace(" ", "")

        # Feature 5.1: Comma-separated numeric list
        if ',' in s_clean:
            parts = [p for p in s_clean.split(',') if p != '']
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 4:
                    return BicomplexNumber(
                        complex(nums[0], nums[1]),
                        complex(nums[2], nums[3])
                    )
                elif len(nums) == 2:
                    return BicomplexNumber(
                        complex(nums[0], nums[1]),
                        complex(0.0, 0.0)
                    )
                elif len(nums) == 1:
                    return BicomplexNumber(
                        complex(nums[0], 0.0),
                        complex(0.0, 0.0)
                    )
            except ValueError:
                # Not purely numeric, continue to other formats
                pass

        # Feature 5.2: Explicit "(a+bj)+(c+dj)e" format
        if 'e' in s_clean and '(' in s_clean:
            # Try both patterns from both versions
            patterns = [
                # From first version
                r'\(\s*([+-]?\d*\.?\d+)\s*([+-])\s*([\d\.]*)j\s*\)\s*(?:\+)\s*\(\s*([+-]?\d*\.?\d+)\s*([+-])\s*([\d\.]*)j\s*\)e',
                # From second version
                r'\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)\s*\+\s*\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)e'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, s_clean)
                if match:
                    try:
                        # Parse groups (adapt based on pattern)
                        groups = match.groups()
                        if len(groups) == 6:
                            if pattern == patterns[0]:
                                # First version pattern
                                z1_real = float(groups[0])
                                z1_imag_sign = -1.0 if groups[1] == '-' else 1.0
                                z1_imag_val = float(groups[2]) if groups[2] not in ['', None] else 1.0
                                z1_imag = z1_imag_sign * z1_imag_val
                                
                                z2_real = float(groups[3])
                                z2_imag_sign = -1.0 if groups[4] == '-' else 1.0
                                z2_imag_val = float(groups[5]) if groups[5] not in ['', None] else 1.0
                                z2_imag = z2_imag_sign * z2_imag_val
                            else:
                                # Second version pattern
                                z1_real = float(groups[0])
                                z1_imag_sign = -1 if groups[1] == '-' else 1
                                z1_imag_val = float(groups[2] or '1')
                                z1_imag = z1_imag_sign * z1_imag_val
                                
                                z2_real = float(groups[3])
                                z2_imag_sign = -1 if groups[4] == '-' else 1
                                z2_imag_val = float(groups[5] or '1')
                                z2_imag = z2_imag_sign * z2_imag_val
                            
                            return BicomplexNumber(
                                complex(z1_real, z1_imag),
                                complex(z2_real, z2_imag)
                            )
                    except Exception:
                        continue

        # Feature 5.3: Complex number parsing (common patterns)
        if 'j' in s_clean:
            # If string contains 'j', try to parse as complex
            try:
                # Try direct complex() parsing first
                c = complex(s_clean)
                return BicomplexNumber(c, complex(0.0, 0.0))
            except ValueError:
                # Try regex-based parsing for malformed complex numbers
                pattern = r'^([+-]?\d*\.?\d*)([+-]?\d*\.?\d*)j$'
                match = re.match(pattern, s_clean)
                if match:
                    real_part = match.group(1)
                    imag_part = match.group(2)
                    
                    # Handle edge cases
                    if real_part in ['', '+', '-']:
                        real_part = real_part + '1' if real_part else '0'
                    if imag_part in ['', '+', '-']:
                        imag_part = imag_part + '1' if imag_part else '0'
                    
                    return BicomplexNumber(
                        complex(float(real_part or 0), float(imag_part or 0)),
                        complex(0.0, 0.0)
                    )
        
        # Feature 5.4: Simple real number
        try:
            real_val = float(s_clean)
            return BicomplexNumber(complex(real_val, 0.0), complex(0.0, 0.0))
        except ValueError:
            pass
        
        # Feature 6: Fallback - try to extract any numeric part
        try:
            num_token = _extract_numeric_part(s_clean)
            if num_token:
                return BicomplexNumber(
                    complex(float(num_token), 0.0),
                    complex(0.0, 0.0)
                )
        except Exception:
            pass

    except Exception as e:
        # Feature 7: Logging on error
        if 'logger' in globals():
            logger.warning(f"Bicomplex parsing failed for {repr(s)}: {e}")
        else:
            print(f"Bicomplex parsing error for '{s}': {e}")
    
    # Final fallback: return zero bicomplex
    return BicomplexNumber(complex(0.0, 0.0), complex(0.0, 0.0))

# =============================================
# Utility Functions
# =============================================

def generate_cd_chain(max_level: int = 8) -> list:
    """
    Generate chain of Cayley-Dickson algebras.
    
    Args:
        max_level: Maximum level to generate
    
    Returns:
        List of cebr classes
    """
    return [cayley_dickson_cebr(i) for i in range(max_level + 1)]


def cd_number_from_components(level: int, *components) -> object:
    """
    generate a Cayley-Dickson number from components.
    
    Args:
        level: cebr level
        *components: Number components
    
    Returns:
        Cayley-Dickson number instance
    """
    cebr = cayley_dickson_cebr(level)
    return cebr(*components)

# Mevcut _parse_complex fonksiyonunuzu aynen koruyoruz
def _parse_complex(s) -> complex:
    """Bir string'i veya sayıyı complex sayıya dönüştürür.
    "real,imag", "real+imag(i/j)", "real", "imag(i/j)" formatlarını destekler.
    Float ve int tiplerini de doğrudan kabul eder.
    """
    # Eğer zaten complex sayıysa doğrudan döndür
    if isinstance(s, complex):
        return s
    
    # Eğer HypercomplexNumber ise, ilk iki bileşeni kullan
    if isinstance(s, HypercomplexNumber):
        if s.dimension >= 2:
            return complex(s[0], s[1])
        else:
            return complex(s.real, 0.0)
    
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
            
            # 5. Fallback: varsayılan kompleks sayı
            warnings.warn(f"Geçersiz kompleks sayı formatı: '{s}', 0+0j döndürülüyor", RuntimeWarning)
            return complex(0, 0)

def _get_default_hypercomplex(dimension: int) -> HypercomplexNumber:
    """Get default HypercomplexNumber for dimension."""
    coeffs = [0.0] * dimension
    return HypercomplexNumber(*coeffs, dimension=dimension)

def _parse_to_hypercomplex(s: Any, dimension: int) -> HypercomplexNumber:
    """Parse input to HypercomplexNumber with specific dimension."""
    try:
        # Eğer zaten HypercomplexNumber ise
        if isinstance(s, HypercomplexNumber):
            if s.dimension == dimension:
                return s
            elif s.dimension < dimension:
                return s.pad_to_dimension(dimension)
            else:
                return s.truncate_to_dimension(dimension)
        
        # Sayısal tipler
        if isinstance(s, (int, float)):
            coeffs = [float(s)] + [0.0] * (dimension - 1)
            return HypercomplexNumber(*coeffs, dimension=dimension)
        
        if isinstance(s, complex):
            coeffs = [s.real, s.imag] + [0.0] * (dimension - 2)
            return HypercomplexNumber(*coeffs, dimension=dimension)
        
        # İterable tipler (list, tuple, numpy array, vs.)
        if hasattr(s, '__iter__') and not isinstance(s, str):
            coeffs = list(s)
            if len(coeffs) < dimension:
                coeffs = coeffs + [0.0] * (dimension - len(coeffs))
            elif len(coeffs) > dimension:
                coeffs = coeffs[:dimension]
            return HypercomplexNumber(*coeffs, dimension=dimension)
        
        # String parsing
        if not isinstance(s, str):
            s = str(s)
        
        s = s.strip()
        
        # Parantezleri kaldır
        s = s.strip('[]{}()')
        
        # Boş string kontrolü
        if not s:
            coeffs = [0.0] * dimension
            return HypercomplexNumber(*coeffs, dimension=dimension)
        
        # Virgülle ayrılmış liste
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            
            if not parts:
                coeffs = [0.0] * dimension
                return HypercomplexNumber(*coeffs, dimension=dimension)
            
            try:
                coeffs = [float(p) for p in parts]
                if len(coeffs) < dimension:
                    coeffs = coeffs + [0.0] * (dimension - len(coeffs))
                elif len(coeffs) > dimension:
                    coeffs = coeffs[:dimension]
                return HypercomplexNumber(*coeffs, dimension=dimension)
            except ValueError as e:
                raise ValueError(f"Invalid numeric value in string: '{s}' -> {e}")
        
        # Tek sayı olarak dene
        try:
            coeffs = [float(s)] + [0.0] * (dimension - 1)
            return HypercomplexNumber(*coeffs, dimension=dimension)
        except ValueError:
            pass
        
        # Kompleks string olarak dene
        try:
            c = complex(s)
            coeffs = [c.real, c.imag] + [0.0] * (dimension - 2)
            return HypercomplexNumber(*coeffs, dimension=dimension)
        except ValueError:
            pass
        
        # Fallback: sıfır
        coeffs = [0.0] * dimension
        return HypercomplexNumber(*coeffs, dimension=dimension)
    
    except Exception as e:
        warnings.warn(f"Hypercomplex parse error (dim={dimension}) for input {repr(s)}: {e}", RuntimeWarning)
        return _get_default_hypercomplex(dimension)

def _parse_real(s: Any) -> float:
    """Parse input as real number (float)."""
    try:
        if isinstance(s, (int, float)):
            return float(s)
        
        if isinstance(s, complex):
            return float(s.real)
        
        if isinstance(s, HypercomplexNumber):
            return float(s.real)
        
        if not isinstance(s, str):
            s = str(s)
        
        s = s.strip()
        return float(s)
    
    except Exception as e:
        warnings.warn(f"Real parse error: {e}", RuntimeWarning)
        return 0.0

def _parse_super_real(s: Any) -> float:
    """
    Parse input as super real/hyperreal number with extended support.
    
    Supports:
    - Standard real numbers: 3.14, -2.5, etc.
    - Infinity representations: ∞, inf, infinity
    - Infinitesimals: ε, epsilon, dx, dt
    - Scientific notation: 1.23e-4, 5.67E+8
    - Engineering notation: 1.5k, 2.3M, 4.7m (k=1e3, M=1e6, m=1e-3, etc.)
    - Fractions: 1/2, 3/4, etc.
    - Mixed numbers: 1 1/2, 2 3/4
    - Percentage: 50%, 12.5%
    - Special constants: π, pi, e, φ, phi
    - Hypercomplex numbers (extract real part)
    
    Returns:
        float: Parsed real number (always float, never int)
    """
    import re
    import math
    import warnings
    
    try:
        # 1. Direkt sayısal tipler
        if isinstance(s, (int, float)):
            return float(s)
        
        # 2. Kompleks sayılar (reel kısmı al)
        if isinstance(s, complex):
            return float(s.real)
        
        # 3. HypercomplexNumber tipini kontrol et
        if hasattr(s, '__class__') and s.__class__.__name__ == 'HypercomplexNumber':
            try:
                return float(s.real)
            except AttributeError:
                pass
        
        # 4. String'e dönüştür
        if not isinstance(s, str):
            s = str(s)
        
        s_original = s  # Orijinal string'i sakla
        s = s.strip().lower()
        
        # 5. Özel durumlar/boş giriş
        if s in ["", "nan", "null", "none", "undefined"]:
            return 0.0
        
        # 6. Sonsuzluk değerleri
        infinity_patterns = {
            "∞": float('inf'),
            "inf": float('inf'),
            "infinity": float('inf'),
            "+∞": float('inf'),
            "+inf": float('inf'),
            "+infinity": float('inf'),
            "-∞": float('-inf'),
            "-inf": float('-inf'),
            "-infinity": float('-inf')
        }
        
        if s in infinity_patterns:
            return infinity_patterns[s]
        
        # 7. Bilimsel sabitler
        constants = {
            "π": math.pi,
            "pi": math.pi,
            "e": math.e,
            "φ": (1 + math.sqrt(5)) / 2,  # Altın oran
            "phi": (1 + math.sqrt(5)) / 2,
            "tau": 2 * math.pi,
            "γ": 0.5772156649015329,  # Euler-Mascheroni sabiti
        }
        
        if s in constants:
            return constants[s]
        
        # 8. Mühendislik notasyonu (k, M, G, m, μ, n, p, etc.)
        engineering_units = {
            'k': 1e3,   # kilo
            'm': 1e-3,  # milli (küçük m)
            'meg': 1e6, # mega
            'g': 1e9,   # giga
            't': 1e12,  # tera
            'μ': 1e-6,  # mikro
            'u': 1e-6,  # mikro (alternatif)
            'n': 1e-9,  # nano
            'p': 1e-12, # piko
            'f': 1e-15, # femto
            'a': 1e-18, # atto
        }
        
        # Mühendislik notasyonu regex'i (case-insensitive)
        eng_match = re.match(r'^\s*([+-]?\d*\.?\d+)\s*([a-zA-Zμ]+)\s*$', s_original)
        if eng_match:
            try:
                value = float(eng_match.group(1))
                unit = eng_match.group(2).lower()
                
                if unit in engineering_units:
                    return value * engineering_units[unit]
                elif unit == 'mil':  # bin (thousand)
                    return value * 1000
            except (ValueError, KeyError):
                pass
        
        # 9. Yüzde notasyonu
        if s.endswith('%'):
            try:
                # Orijinal string'den % işaretini kaldır (büyük/küçük harf fark etmez)
                value_str = s_original.rstrip('%').strip()
                value = float(value_str)
                return value / 100.0
            except ValueError:
                pass
        
        # 10. Kesirler ve karışık sayılar
        # Karışık sayı: "1 1/2"
        mixed_match = re.match(r'^\s*(\d+)\s+(\d+)/(\d+)\s*$', s)
        if mixed_match:
            try:
                whole = int(mixed_match.group(1))
                num = int(mixed_match.group(2))
                den = int(mixed_match.group(3))
                return float(whole) + (float(num) / float(den))
            except (ValueError, ZeroDivisionError):
                pass
        
        # Basit kesir: "3/4"
        if '/' in s and ' ' not in s:
            try:
                parts = s.split('/')
                if len(parts) == 2:
                    num = float(parts[0])
                    den = float(parts[1])
                    if den != 0:
                        result = num / den
                        return float(result)  # Açıkça float
            except (ValueError, ZeroDivisionError) as e:
                warnings.warn(f"Fraction parse failed: {e}", RuntimeWarning, stacklevel=2)
                return 0.0
        
        # 11. Infinitesimal notasyonu (ε, epsilon, dx, etc.)
        infinitesimals = {
            'ε': 1e-10,
            'epsilon': 1e-10,
            'δ': 1e-10,
            'delta': 1e-10,
            'dx': 1e-10,
            'dt': 1e-10,
            'dh': 1e-10,
            'infinitesimal': 1e-15,
        }
        
        if s in infinitesimals:
            return infinitesimals[s]
        
        # 12. Parantez içindeki ifadeler
        if '(' in s and ')' in s:
            # İçeriği al ve tekrar dene
            inner_start = s.find('(') + 1
            inner_end = s.find(')')
            if inner_start < inner_end:
                inner = s[inner_start:inner_end].strip()
                if inner:
                    try:
                        return _parse_super_real(inner)
                    except:
                        pass
        
        # 13. Standart float dönüşümü (son çare)
        try:
            # Bilimsel notasyon desteği
            return float(s)
        except ValueError:
            # Romawi rakamları
            roman_numerals = {
                'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
                'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
            }
            if s in roman_numerals:
                return float(roman_numerals[s])
    
    except Exception as e:
        warnings.warn(f"Super real parse error for '{s}': {e}", RuntimeWarning, stacklevel=2)
    
    # 14. Hiçbir şey işe yaramazsa
    return 0.0

def is_super_real_expression(expr: str) -> bool:
    """Check if string looks like a super real expression."""
    super_real_indicators = [
        '∞', 'inf', 'epsilon', 'ε', 'δ', 'dx', 'dt',
        'pi', 'π', 'e', 'phi', 'φ', 'tau', 'γ',
        'k', 'm', 'meg', 'g', 't', 'μ', 'n', 'p',
        '%', '/'
    ]
    
    expr_lower = expr.lower()
    return any(indicator in expr_lower for indicator in super_real_indicators)


def normalize_super_real(value: float) -> float:
    """Normalize super real values (e.g., replace very small numbers with 0)."""
    EPSILON = 1e-15
    
    if abs(value) < EPSILON:
        return 0.0
    elif math.isinf(value):
        return float('inf') if value > 0 else float('-inf')
    else:
        return value

def _parse_universal(s: Union[str, Any], target_type: str) -> Any:
    """
    Universal parser - Çeşitli sayı türlerini string'den veya diğer tiplerden parse eder
    
    Args:
        s: Parse edilecek input (string, sayı, liste, vs.)
        target_type: Hedef tür ("real", "complex", "quaternion", "octonion", 
                   "sedenion", "pathion", "chingon", "routon", "voudon", "bicomplex")
    
    Returns:
        Parse edilmiş değer veya hata durumunda varsayılan değer
        
    Özellikler:
        - "real": Float'a çevirir, hata durumunda 0.0 döner
        - "complex": _parse_complex fonksiyonunu çağırır (mevcut mantık korunur)
        - "quaternion": 4 bileşenli hiperkompleks sayı
        - "octonion": 8 bileşenli hiperkompleks sayı
        - "sedenion": 16 bileşenli hiperkompleks sayı
        - "pathion": 32 bileşenli hiperkompleks sayı
        - "chingon": 64 bileşenli hiperkompleks sayı
        - "routon": 128 bileşenli hiperkompleks sayı
        - "voudon": 256 bileşenli hiperkompleks sayı
        - "bicomplex": _parse_bicomplex fonksiyonunu çağırır (özel durum)
    """
    try:
        # Bicomplex özel durumu (mevcut implementasyonu koru)
        if target_type == "bicomplex":
            return _parse_bicomplex(s)
        
        # Complex özel durumu (mevcut implementasyonu koru)
        if target_type == "complex":
            return _parse_complex(s)
        
        # Real özel durumu
        if target_type == "real":
            try:
                if isinstance(s, (int, float)):
                    return float(s)
                
                if isinstance(s, complex):
                    return float(s.real)
                
                if isinstance(s, HypercomplexNumber):
                    return float(s.real)
                
                # Mevcut complex parser'ı kullan, sonra real kısmını al
                c = _parse_complex(s)
                return float(c.real)
            except Exception as e:
                warnings.warn(f"Real parse error: {e}", RuntimeWarning)
                return 0.0
        
        # HypercomplexNumber kullanacak tipler için mapping
        hypercomplex_map = {
            "quaternion": 4,
            "octonion": 8,
            "sedenion": 16,
            "pathion": 32,
            "chingon": 64,
            "routon": 128,
            "voudon": 256
        }
        
        if target_type in hypercomplex_map:
            dimension = hypercomplex_map[target_type]
            return _parse_to_hypercomplex(s, dimension)
        
        # Eğer target_type tanınmıyorsa None döndür
        warnings.warn(f"Unknown target_type: {target_type}", RuntimeWarning)
        return None
    
    except Exception as e:
        warnings.warn(f"Universal parser error for {target_type}: {e}", RuntimeWarning)
        # Hata durumunda varsayılan değer döndür
        return _get_default_value(target_type)


def _get_default_value(target_type: str) -> Any:
    """Get default value for target type."""
    defaults = {
        "real": 0.0,
        "complex": complex(0, 0),
        "quaternion": HypercomplexNumber(0, 0, 0, 0, dimension=4),
        "octonion": HypercomplexNumber(*([0.0] * 8), dimension=8),
        "sedenion": HypercomplexNumber(*([0.0] * 16), dimension=16),
        "pathion": HypercomplexNumber(*([0.0] * 32), dimension=32),
        "chingon": HypercomplexNumber(*([0.0] * 64), dimension=64),
        "routon": HypercomplexNumber(*([0.0] * 128), dimension=128),
        "voudon": HypercomplexNumber(*([0.0] * 256), dimension=256),
        "bicomplex": _parse_bicomplex("0") if '_parse_bicomplex' in globals() else None
    }
    
    return defaults.get(target_type, None)

# =============================================
# Mathematical Sequences and Functions
# =============================================

def oresme_sequence(n_terms: int) -> List[float]:
    """Generate Oresme sequence: n / 2^n."""
    return [n / (2 ** n) for n in range(1, n_terms + 1)]


def harmonic_numbers(n_terms: int) -> Generator[Fraction, None, None]:
    """Generate harmonic numbers: H_n = 1 + 1/2 + ... + 1/n."""
    current = Fraction(0)
    for n in range(1, n_terms + 1):
        current += Fraction(1, n)
        yield current


def binet_formula(n: int) -> float:
    """Calculate nth Fibonacci number using Binet's formula."""
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return (phi**n - psi**n) / sqrt5


def generate_cd_chain_names(max_level: int = 8) -> List[str]:
    """Generate names of Cayley-Dickson algebras up to given level."""
    names = ["Real", "Complex", "Quaternion", "Octonion", "Sedenion",
             "Pathion", "Chingon", "Routon", "Voudon"]
    return names[:max_level + 1]

# =============================================
# Cayley-Dickson Implementation
# =============================================

def cayley_dickson_process(cebr: type, base_type=float) -> type:
    """
    Apply the Cayley-Dickson construction to generate an algebra of twice the dimension.
    """
    
    class CayleyDicksonCebr:
        """Hypercomplex algebra generated via Cayley-Dickson construction."""
        
        dimensions: Optional[int] = None
        base = base_type
        
        def __init__(self, *args, pair=False):
            if pair and len(args) == 2:
                # (a, b) pair format
                self.a = self._ensure_cebr(args[0], cebr)
                self.b = self._ensure_cebr(args[1], cebr)
            else:
                # Handle various input formats
                if len(args) == 1:
                    arg = args[0]
                    # Handle complex numbers
                    if isinstance(arg, complex):
                        # Convert complex to pair (real, imag)
                        self.a = cebr(arg.real)
                        self.b = cebr(arg.imag)
                        return
                    # Handle strings
                    elif isinstance(arg, str):
                        # Try to parse as complex
                        try:
                            c = complex(arg)
                            self.a = cebr(c.real)
                            self.b = cebr(c.imag)
                            return
                        except ValueError:
                            pass
                    # Handle iterables
                    elif hasattr(arg, '__iter__'):
                        components = list(arg)
                    else:
                        components = [arg]
                else:
                    components = list(args)
                
                # Ensure even number of components
                if len(components) % 2 != 0:
                    components.append(base_type(0))
                
                half = len(components) // 2
                self.a = cebr(*components[:half])
                self.b = cebr(*components[half:])
        
        @staticmethod
        def _ensure_cebr(value, cebr_class):
            """Convert value to cebr instance if needed."""
            if isinstance(value, cebr_class):
                return value
            # Handle complex numbers
            elif isinstance(value, complex):
                return cebr_class(value.real, value.imag)
            # Handle single values
            else:
                return cebr_class(value)
        
        @classmethod
        def from_complex(cls, c: complex):
            """Generate from a complex number."""
            return cls(c.real, c.imag)
        
        @classmethod
        def from_pair(cls, a, b):
            """Generate from a pair (a, b)."""
            return cls(a, b, pair=True)
        
        @property
        def real(self) -> float:
            """Real part."""
            if hasattr(self.a, 'real'):
                return float(self.a.real)
            elif hasattr(self.a, 'value'):
                return float(self.a.value)
            else:
                try:
                    return float(self.a)
                except:
                    return 0.0
        
        def coefficients(self):
            """Get all coefficients as a tuple."""
            if hasattr(self.a, 'coefficients'):
                a_coeffs = self.a.coefficients()
            elif hasattr(self.a, 'value'):
                a_coeffs = (float(self.a.value),)
            else:
                try:
                    a_coeffs = (float(self.a),)
                except:
                    a_coeffs = (0.0,)
            
            if hasattr(self.b, 'coefficients'):
                b_coeffs = self.b.coefficients()
            elif hasattr(self.b, 'value'):
                b_coeffs = (float(self.b.value),)
            else:
                try:
                    b_coeffs = (float(self.b),)
                except:
                    b_coeffs = (0.0,)
            
            return a_coeffs + b_coeffs
        
        def __add__(self, other):
            if isinstance(other, CayleyDicksonCebr):
                return CayleyDicksonCebr(
                    self.a + other.a,
                    self.b + other.b,
                    pair=True
                )
            
            # Try to convert to this cebr
            try:
                other_cd = CayleyDicksonCebr(other)
                return self + other_cd
            except:
                return NotImplemented
        
        def __radd__(self, other):
            return self.__add__(other)
        
        def __sub__(self, other):
            if isinstance(other, CayleyDicksonCebr):
                return CayleyDicksonCebr(
                    self.a - other.a,
                    self.b - other.b,
                    pair=True
                )
            
            try:
                other_cd = CayleyDicksonCebr(other)
                return self - other_cd
            except:
                return NotImplemented
        
        def __rsub__(self, other):
            try:
                other_cd = CayleyDicksonCebr(other)
                return other_cd - self
            except:
                return NotImplemented
        
        def __mul__(self, other):
            if isinstance(other, CayleyDicksonCebr):
                # Cayley-Dickson multiplication
                a = self.a * other.a - other.b * self._conj_b()
                b = self._conj_a() * other.b + other.a * self.b
                return CayleyDicksonCebr(a, b, pair=True)
            
            # Scalar multiplication
            try:
                other_cd = CayleyDicksonCebr(other)
                return self * other_cd
            except:
                return NotImplemented
        
        def __rmul__(self, other):
            return self.__mul__(other)
        
        def __truediv__(self, other):
            if isinstance(other, CayleyDicksonCebr):
                return self * other.inverse()
            
            try:
                other_cd = CayleyDicksonCebr(other)
                return self / other_cd
            except:
                return NotImplemented
        
        def __rtruediv__(self, other):
            try:
                other_cd = CayleyDicksonCebr(other)
                return other_cd / self
            except:
                return NotImplemented
        
        def __neg__(self):
            return CayleyDicksonCebr(-self.a, -self.b, pair=True)
        
        def __pos__(self):
            return self
        
        def __abs__(self):
            return self.norm()
        
        def __eq__(self, other):
            if isinstance(other, CayleyDicksonCebr):
                return self.a == other.a and self.b == other.b
            
            try:
                other_cd = CayleyDicksonCebr(other)
                return self == other_cd
            except:
                return False
        
        def __ne__(self, other):
            return not self.__eq__(other)
        
        def __hash__(self):
            return hash((self.a, self.b))
        
        def __str__(self):
            coeffs = self.coefficients()
            if len(coeffs) <= 8:
                return f"({', '.join(f'{c:.4f}' for c in coeffs)})"
            else:
                return f"CD[{len(coeffs)}]({coeffs[0]:.4f}, ..., {coeffs[-1]:.4f})"
        
        def __repr__(self):
            coeffs = self.coefficients()
            return f"{self.__class__.__name__}({', '.join(map(str, coeffs))})"
        
        def _conj_a(self):
            """Conjugate of a."""
            if hasattr(self.a, 'conjugate'):
                return self.a.conjugate()
            elif hasattr(self.a, 'value'):
                return self.a  # Real numbers are self-conjugate
            return self.a
        
        def _conj_b(self):
            """Conjugate of b."""
            if hasattr(self.b, 'conjugate'):
                return self.b.conjugate()
            elif hasattr(self.b, 'value'):
                return self.b  # Real numbers are self-conjugate
            return self.b
        
        def conjugate(self):
            """Conjugate: conj(a, b) = (conj(a), -b)."""
            return CayleyDicksonCebr(
                self._conj_a(),
                -self.b,
                pair=True
            )
        
        def norm(self) -> float:
            """Euclidean norm."""
            
            def get_norm_squared(x):
                if hasattr(x, 'norm_squared'):
                    return float(x.norm_squared())
                elif hasattr(x, 'norm'):
                    n = float(x.norm())
                    return n * n
                elif hasattr(x, 'value'):
                    val = float(x.value)
                    return val * val
                else:
                    try:
                        val = float(x)
                        return val * val
                    except:
                        return 0.0
            
            norm_sq = get_norm_squared(self.a) + get_norm_squared(self.b)
            return math.sqrt(norm_sq)
        
        def norm_squared(self):
            """Square of the norm."""
            def get_norm_squared(x):
                if hasattr(x, 'norm_squared'):
                    return x.norm_squared()
                elif hasattr(x, 'norm'):
                    n = x.norm()
                    return n * n
                elif hasattr(x, 'value'):
                    return x.value * x.value
                else:
                    try:
                        return x * x
                    except:
                        return 0
            
            return get_norm_squared(self.a) + get_norm_squared(self.b)
        
        def inverse(self):
            """Multiplicative inverse."""
            norm_sq = self.norm_squared()
            if float(norm_sq) == 0:
                raise ZeroDivisionError("Cannot invert zero element")
            
            conj = self.conjugate()
            return CayleyDicksonCebr(
                conj.a / norm_sq,
                conj.b / norm_sq,
                pair=True
            )
    
    # Set class attributes
    if hasattr(cebr, 'dimensions'):
        CayleyDicksonCebr.dimensions = cebr.dimensions * 2
    
    # Use the base class name properly
    base_name = cebr.__name__ if hasattr(cebr, '__name__') else str(cebr)
    CayleyDicksonCebr.__name__ = f"CD{base_name}"
    
    return CayleyDicksonCebr


def cayley_dickson_cebr(level: int, base_type=float) -> type:
    """Generate Cayley-Dickson cebr of given level."""
    if not isinstance(level, int) or level < 0:
        raise ValueError(f"Level must be non-negative integer, got {level}")
    
    # Start with real numbers
    if level == 0:
        class RealAlgebra:
            dimensions = 1
            base = base_type
            
            def __init__(self, value):
                self.value = base_type(value)
            
            def __add__(self, other):
                if isinstance(other, RealAlgebra):
                    return RealAlgebra(self.value + other.value)
                return RealAlgebra(self.value + base_type(other))
            
            def __sub__(self, other):
                if isinstance(other, RealAlgebra):
                    return RealAlgebra(self.value - other.value)
                return RealAlgebra(self.value - base_type(other))
            
            def __mul__(self, other):
                if isinstance(other, RealAlgebra):
                    return RealAlgebra(self.value * other.value)
                return RealAlgebra(self.value * base_type(other))
            
            def __truediv__(self, other):
                if isinstance(other, RealAlgebra):
                    if other.value == 0:
                        raise ZeroDivisionError("Division by zero")
                    return RealAlgebra(self.value / other.value)
                if base_type(other) == 0:
                    raise ZeroDivisionError("Division by zero")
                return RealAlgebra(self.value / base_type(other))
            
            def __neg__(self):
                return RealAlgebra(-self.value)
            
            def __pos__(self):
                return self
            
            def __abs__(self):
                return RealAlgebra(abs(self.value))
            
            def __eq__(self, other):
                if isinstance(other, RealAlgebra):
                    return math.isclose(self.value, other.value)
                return math.isclose(self.value, base_type(other))
            
            def __repr__(self):
                return f"RealAlgebra({self.value})"
            
            def __str__(self):
                return f"{self.value:.4f}"
            
            def __float__(self):
                return float(self.value)
            
            @property
            def real(self):
                return float(self.value)
            
            def conjugate(self):
                return self
            
            def norm(self):
                return abs(float(self.value))
            
            def norm_squared(self):
                return self.value * self.value
            
            def coefficients(self):
                return (float(self.value),)
        
        return RealAlgebra
    
    # Apply construction level times
    current_cebr = cayley_dickson_cebr(0, base_type)
    for i in range(level):
        current_cebr = cayley_dickson_process(current_cebr, base_type)
    
    # Set name
    if level == 0:
        current_cebr.__name__ = "RealAlgebra"
    elif level == 1:
        current_cebr.__name__ = "Complex"
    elif level == 2:
        current_cebr.__name__ = "Quaternion"
    elif level == 3:
        current_cebr.__name__ = "Octonion"
    elif level == 4:
        current_cebr.__name__ = "Sedenion"
    elif level == 5:
        current_cebr.__name__ = "Pathion"
    elif level == 6:
        current_cebr.__name__ = "Chingon"
    elif level == 7:
        current_cebr.__name__ = "Routon"
    elif level == 8:
        current_cebr.__name__ = "Voudon"
    else:
        current_cebr.__name__ = f"CD{level}"
    
    return current_cebr

def kececi_bicomplex_algorithm(
    start: BicomplexNumber, 
    add_val: BicomplexNumber, 
    iterations: int, 
    include_intermediate: bool = True,
    mod_value: float = 100.0
) -> list:
    """
    Gerçek Keçeci algoritmasının bikompleks versiyonunu uygular.
    
    Bu algoritma orijinal Keçeci sayı üretecini bikompleks sayılara genişletir.
    
    Parametreler:
    ------------
    start : BicomplexNumber
        Algoritmanın başlangıç değeri
    add_val : BicomplexNumber
        Her iterasyonda eklenen değer
    iterations : int
        İterasyon sayısı
    include_intermediate : bool, varsayılan=True
        Ara adımları dizie ekleme
    mod_value : float, varsayılan=100.0
        Mod işlemi için kullanılacak değer
    
    Döndürür:
    --------
    list[BicomplexNumber]
        Üretilen Keçeci bikompleks dizisi
    
    Özellikler:
    ----------
    1. Toplama işlemi
    2. Mod alma işlemi (Keçeci algoritmasının karakteristik özelliği)
    3. Ara adımların eklenmesi (isteğe bağlı)
    4. Asal sayı kontrolü
    5. Sıfır değerinde resetleme
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Toplama işlemi
        current = current + add_val
        
        # 2. Keçeci algoritmasının özelliği: Mod alma
        # z1 ve z2 için mod alma (gerçek ve sanal kısımlar ayrı ayrı)
        current = BicomplexNumber(
            complex(current.z1.real % mod_value, current.z1.imag % mod_value),
            complex(current.z2.real % mod_value, current.z2.imag % mod_value)
        )
        
        # 3. Ara adımları ekle (Keçeci algoritmasının karakteristik özelliği)
        if include_intermediate:
            # Ara değerler için özel işlemler
            intermediate = current * BicomplexNumber(complex(0.5, 0), complex(0, 0))
            sequence.append(intermediate)
        
        sequence.append(current)
        
        # 4. Asal sayı kontrolü (Keçeci algoritmasının önemli bir parçası)
        # Bu kısım algoritmanın detayına göre özelleştirilebilir
        magnitude = abs(current.z1) + abs(current.z2)
        if magnitude > 1:
            # Basit asallık testi (büyük sayılar için verimsiz)
            is_prime = True
            sqrt_mag = int(magnitude**0.5) + 1
            for j in range(2, sqrt_mag):
                if magnitude % j == 0:
                    is_prime = False
                    break
            
            if is_prime:
                print(f"Keçeci Prime bulundu - adım {i}: büyüklük = {magnitude:.2f}")
        
        # 5. Özel durum: Belirli değerlere ulaşıldığında resetleme
        if abs(current.z1) < 1e-10 and abs(current.z2) < 1e-10:
            current = start  # Başa dön
    
    return sequence


def kececi_bicomplex_advanced(
    start: BicomplexNumber, 
    add_val: BicomplexNumber, 
    iterations: int, 
    include_intermediate: bool = True,
    mod_real: float = 50.0,
    mod_imag: float = 50.0,
    feedback_interval: int = 10
) -> list:
    """
    Gelişmiş Keçeci algoritması - daha karmaşık matematiksel işlemler içerir.
    
    Bu algoritma standart Keçeci algoritmasını daha gelişmiş matematiksel
    işlemlerle genişletir: doğrusal olmayan dönüşümler, modüler aritmetik,
    çapraz çarpımlar ve dinamik feedback mekanizmaları.
    
    Parametreler:
    ------------
    start : BicomplexNumber
        Algoritmanın başlangıç değeri
    add_val : BicomplexNumber
        Her iterasyonda eklenen değer
    iterations : int
        İterasyon sayısı
    include_intermediate : bool, varsayılan=True
        Ara adımları (çapraz çarpımları) dizie ekleme
    mod_real : float, varsayılan=50.0
        Gerçel kısımlar için mod değeri
    mod_imag : float, varsayılan=50.0
        Sanal kısımlar için mod değeri
    feedback_interval : int, varsayılan=10
        Feedback perturbasyonlarının uygulanma aralığı
    
    Döndürür:
    --------
    list[BicomplexNumber]
        Üretilen gelişmiş Keçeci bikompleks dizisi
    
    Özellikler:
    ----------
    1. Temel toplama işlemi
    2. Doğrusal olmayan dönüşümler (karekök)
    3. Modüler aritmetik
    4. Çapraz çarpım ara değerleri
    5. Dinamik feedback perturbasyonları
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Temel toplama
        current = current + add_val
        
        # 2. Doğrusal olmayan dönüşümler (Keçeci algoritmasının özelliği)
        # Karekök alma işlemleri - negatif değerler için güvenli hale getirildi
        try:
            z1_real_sqrt = math.sqrt(abs(current.z1.real)) * (1 if current.z1.real >= 0 else 1j)
            z1_imag_sqrt = math.sqrt(abs(current.z1.imag)) * (1 if current.z1.imag >= 0 else 1j)
            z2_real_sqrt = math.sqrt(abs(current.z2.real)) * (1 if current.z2.real >= 0 else 1j)
            z2_imag_sqrt = math.sqrt(abs(current.z2.imag)) * (1 if current.z2.imag >= 0 else 1j)
            
            current = BicomplexNumber(
                complex(z1_real_sqrt.real if isinstance(z1_real_sqrt, complex) else z1_real_sqrt,
                       z1_imag_sqrt.real if isinstance(z1_imag_sqrt, complex) else z1_imag_sqrt),
                complex(z2_real_sqrt.real if isinstance(z2_real_sqrt, complex) else z2_real_sqrt,
                       z2_imag_sqrt.real if isinstance(z2_imag_sqrt, complex) else z2_imag_sqrt)
            )
        except (ValueError, TypeError):
            # Karekök hatası durumunda alternatif yaklaşım
            current = BicomplexNumber(
                complex(np.sqrt(abs(current.z1.real)), np.sqrt(abs(current.z1.imag))),
                complex(np.sqrt(abs(current.z2.real)), np.sqrt(abs(current.z2.imag)))
            )
        
        # 3. Modüler aritmetik
        current = BicomplexNumber(
            complex(current.z1.real % mod_real, current.z1.imag % mod_imag),
            complex(current.z2.real % mod_real, current.z2.imag % mod_imag)
        )
        
        # 4. Ara adımlar (çapraz çarpımlar)
        if include_intermediate:
            # Çapraz çarpım ara değerleri
            cross_product = BicomplexNumber(
                complex(current.z1.real * current.z2.imag, 0),
                complex(0, current.z1.imag * current.z2.real)
            )
            sequence.append(cross_product)
        
        sequence.append(current)
        
        # 5. Dinamik sistem davranışı için feedback
        if feedback_interval > 0 and i % feedback_interval == 0 and i > 0:
            # Periyodik perturbasyon ekle (kaotik davranışı artırmak için)
            perturbation = BicomplexNumber(
                complex(0.1 * math.sin(i), 0.1 * math.cos(i)),
                complex(0.05 * math.sin(i*0.5), 0.05 * math.cos(i*0.5))
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
    """
    Sedenion (16-dimensional hypercomplex number) implementation.
    
    Sedenions are 16-dimensional numbers that extend octonions via the 
    Cayley-Dickson construction. They are non-commutative, non-associative,
    and not even alternative.
    
    Components: e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    where e0 is the real part.
    """
    coeffs: List[float] = field(default_factory=lambda: [0.0] * 16)
    
    def __post_init__(self):
        """Validate and ensure coefficients are correct length."""
        if len(self.coeffs) != 16:
            raise ValueError(f"Sedenion must have exactly 16 components, got {len(self.coeffs)}")
        # Ensure all are floats
        self.coeffs = [float(c) for c in self.coeffs]
    
    @classmethod
    def from_scalar(cls, value: float) -> 'SedenionNumber':
        """Create a sedenion from a scalar (real number)."""
        coeffs = [0.0] * 16
        coeffs[0] = float(value)
        return cls(coeffs)
    
    @classmethod
    def from_list(cls, values: List[float]) -> 'SedenionNumber':
        """Create a sedenion from a list of up to 16 values."""
        if len(values) > 16:
            raise ValueError(f"List too long ({len(values)}), maximum 16 elements")
        coeffs = list(values) + [0.0] * (16 - len(values))
        return cls(coeffs)
    
    @classmethod
    def basis_element(cls, index: int) -> 'SedenionNumber':
        """Create a basis sedenion (1 at position index, 0 elsewhere)."""
        if not 0 <= index < 16:
            raise ValueError(f"Index must be between 0 and 15, got {index}")
        coeffs = [0.0] * 16
        coeffs[index] = 1.0
        return cls(coeffs)
    
    @property
    def real(self) -> float:
        """Get the real part (first component)."""
        return self.coeffs[0]
    
    @real.setter
    def real(self, value: float):
        """Set the real part."""
        self.coeffs[0] = float(value)
    
    @property
    def imag(self) -> List[float]:
        """Get the imaginary parts (all except real)."""
        return self.coeffs[1:]
    
    def __getitem__(self, index: int) -> float:
        """Get component by index."""
        if not 0 <= index < 16:
            raise IndexError(f"Index {index} out of range for sedenion")
        return self.coeffs[index]
    
    def __setitem__(self, index: int, value: float):
        """Set component by index."""
        if not 0 <= index < 16:
            raise IndexError(f"Index {index} out of range for sedenion")
        self.coeffs[index] = float(value)
    
    def __len__(self) -> int:
        """Return number of components (always 16)."""
        return 16
    
    def __iter__(self):
        """Iterate over components."""
        return iter(self.coeffs)
    
    def __add__(self, other: Union['SedenionNumber', float, int]) -> 'SedenionNumber':
        """Add two sedenions or sedenion and scalar."""
        if isinstance(other, SedenionNumber):
            new_coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
            return SedenionNumber(new_coeffs)
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __radd__(self, other: Union[float, int]) -> 'SedenionNumber':
        """Right addition: scalar + sedenion."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['SedenionNumber', float, int]) -> 'SedenionNumber':
        """Subtract two sedenions or sedenion and scalar."""
        if isinstance(other, SedenionNumber):
            new_coeffs = [a - b for a, b in zip(self.coeffs, other.coeffs)]
            return SedenionNumber(new_coeffs)
        elif isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __rsub__(self, other: Union[float, int]) -> 'SedenionNumber':
        """Right subtraction: scalar - sedenion."""
        if isinstance(other, (int, float)):
            new_coeffs = [-c for c in self.coeffs]
            new_coeffs[0] += float(other)
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __mul__(self, other: Union['SedenionNumber', float, int]) -> 'SedenionNumber':
        """Multiply sedenion by scalar or another sedenion (simplified)."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            new_coeffs = [c * float(other) for c in self.coeffs]
            return SedenionNumber(new_coeffs)
        elif isinstance(other, SedenionNumber):
            # NOTE: This is NOT the true sedenion multiplication!
            # True sedenion multiplication requires a 16x16 multiplication table.
            # This is a simplified element-wise multiplication for demonstration.
            # For real applications, implement proper sedenion multiplication.
            new_coeffs = [a * b for a, b in zip(self.coeffs, other.coeffs)]
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __rmul__(self, other: Union[float, int]) -> 'SedenionNumber':
        """Right multiplication: scalar * sedenion."""
        return self.__mul__(other)
    
    def __truediv__(self, scalar: Union[float, int]) -> 'SedenionNumber':
        """Divide sedenion by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide sedenion by zero")
            new_coeffs = [c / float(scalar) for c in self.coeffs]
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __floordiv__(self, scalar: Union[float, int]) -> 'SedenionNumber':
        """Floor divide sedenion by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide sedenion by zero")
            new_coeffs = [c // float(scalar) for c in self.coeffs]
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __mod__(self, divisor: Union[float, int]) -> 'SedenionNumber':
        """Modulo operation on sedenion components."""
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot take modulo by zero")
            new_coeffs = [c % float(divisor) for c in self.coeffs]
            return SedenionNumber(new_coeffs)
        return NotImplemented
    
    def __neg__(self) -> 'SedenionNumber':
        """Negate the sedenion."""
        return SedenionNumber([-c for c in self.coeffs])
    
    def __pos__(self) -> 'SedenionNumber':
        """Unary plus."""
        return self
    
    def __abs__(self) -> float:
        """Absolute value (magnitude)."""
        return self.magnitude()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another sedenion."""
        if not isinstance(other, SedenionNumber):
            return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self.coeffs, other.coeffs))
    
    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        """Hash based on rounded components to avoid floating-point issues."""
        return hash(tuple(round(c, 10) for c in self.coeffs))
    
    def magnitude(self) -> float:
        """
        Calculate Euclidean norm (magnitude) of the sedenion.
        
        Returns:
            float: sqrt(Σ_i coeff_i²)
        """
        return float(np.linalg.norm(self.coeffs))
    
    def norm(self) -> float:
        """Alias for magnitude."""
        return self.magnitude()
    
    def conjugate(self) -> 'SedenionNumber':
        """Return the conjugate (negate all imaginary parts)."""
        new_coeffs = self.coeffs.copy()
        for i in range(1, 16):
            new_coeffs[i] = -new_coeffs[i]
        return SedenionNumber(new_coeffs)
    
    def dot(self, other: 'SedenionNumber') -> float:
        """Dot product with another sedenion."""
        if not isinstance(other, SedenionNumber):
            raise TypeError("Dot product requires another SedenionNumber")
        return sum(a * b for a, b in zip(self.coeffs, other.coeffs))
    
    def normalize(self) -> 'SedenionNumber':
        """Return a normalized (unit) version."""
        mag = self.magnitude()
        if mag == 0:
            raise ZeroDivisionError("Cannot normalize zero sedenion")
        return SedenionNumber([c / mag for c in self.coeffs])
    
    def to_list(self) -> List[float]:
        """Convert to Python list."""
        return self.coeffs.copy()
    
    def to_tuple(self) -> Tuple[float, ...]:
        """Convert to tuple."""
        return tuple(self.coeffs)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.coeffs, dtype=np.float64)
    
    def copy(self) -> 'SedenionNumber':
        """Create a copy."""
        return SedenionNumber(self.coeffs.copy())
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        # Show only non-zero components for clarity
        non_zero = [(i, c) for i, c in enumerate(self.coeffs) if abs(c) > 1e-10]
        if not non_zero:
            return "Sedenion(0)"
        
        parts = []
        for i, c in non_zero:
            if i == 0:
                parts.append(f"{c:.6f}")
            else:
                sign = "+" if c >= 0 else "-"
                parts.append(f"{sign} {abs(c):.6f}e{i}")
        
        return f"Sedenion({' '.join(parts)})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"SedenionNumber({self.coeffs})"

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
def _extract_numeric_part(s: Any) -> str:
    """
    Return the first numeric token found in s as string (supports scientific notation).
    Robust for None and non-string inputs.
    """
    if s is None:
        return "0"
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # match optional sign, digits, optional decimal, optional exponent
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return m.group(0) if m else "0"

def convert_to_float(value: Any) -> float:
    """
    Convert various Keçeci number types to a float (best-effort).
    Raises TypeError if conversion is not possible.
    Rules:
      - int/float -> float
      - complex -> real part (float)
      - numpy-quaternion or objects with attribute 'w' -> float(w)
      - objects with 'real' attribute -> float(real)
      - objects with 'coeffs' iterable -> float(first coeff)
      - objects with 'sequence' iterable -> float(first element)
    """
    # Direct numeric types
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)

    if isinstance(value, complex):
        return float(value.real)

    # quaternion-like
    try:
        #if isinstance(value, quaternion):
        #    return float(value.w)
        if quaternion is not None and isinstance(value, quaternion):
            comps = [value.w, value.x, value.y, value.z]
            if not all(is_near_integer(c) for c in comps):
                return False
            return sympy.isprime(int(round(float(comps[0]))))
    except Exception:
        pass

    # Generic attributes
    if hasattr(value, 'real'):
        try:
            return float(getattr(value, 'real'))
        except Exception:
            pass

    if hasattr(value, 'w'):
        try:
            return float(getattr(value, 'w'))
        except Exception:
            pass

    if hasattr(value, 'coeffs'):
        try:
            coeffs = getattr(value, 'coeffs')
            if isinstance(coeffs, np.ndarray):
                if coeffs.size > 0:
                    return float(coeffs.flatten()[0])
            else:
                # list/iterable
                it = list(coeffs)
                if it:
                    return float(it[0])
        except Exception:
            pass

    if hasattr(value, 'sequence'):
        try:
            seq = getattr(value, 'sequence')
            if seq and len(seq) > 0:
                return float(seq[0])
        except Exception:
            pass

    # TernaryNumber: digits -> decimal
    if hasattr(value, 'digits'):
        try:
            digits = list(value.digits)
            dec = 0
            for i, d in enumerate(reversed(digits)):
                dec += int(d) * (3 ** i)
            return float(dec)
        except Exception:
            pass

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

def _parse_neutrosophic(s: Any) -> Tuple[float, float, float]:
    """
    Parses various neutrosophic representations into (t, i, f) tuple.
    
    Supports:
    - Tuple/list: (t, i, f) or [t, i, f]
    - Numeric: 5.0 -> (5.0, 0.0, 0.0)
    - Complex: 3+4j -> (3.0, 4.0, 0.0)  # real -> t, imag -> i
    - String formats:
        * Comma-separated: "1.5,0.3,0.2"
        * Symbolic: "1.5 + 0.3I + 0.2F"
        * Mixed: "1.5I" or "0.2F"
    """
    # Eğer zaten tuple/list ise doğrudan döndür
    if isinstance(s, (tuple, list)):
        if len(s) >= 3:
            try:
                return float(s[0]), float(s[1]), float(s[2])
            except (ValueError, TypeError):
                pass
        elif len(s) == 2:
            try:
                return float(s[0]), float(s[1]), 0.0
            except (ValueError, TypeError):
                pass
        elif len(s) == 1:
            try:
                return float(s[0]), 0.0, 0.0
            except (ValueError, TypeError):
                pass
        return 0.0, 0.0, 0.0
    
    # Sayısal tipler için
    if isinstance(s, (float, int)):
        return float(s), 0.0, 0.0
    elif isinstance(s, complex):
        # Karmaşık sayı: real -> t, imag -> i
        return float(s.real), float(s.imag), 0.0
    
    # Eğer NeutrosophicNumber instance ise
    if hasattr(s, '__class__'):
        class_name = s.__class__.__name__
        if class_name == 'NeutrosophicNumber':
            try:
                return float(s.t), float(s.i), float(s.f)
            except (AttributeError, ValueError, TypeError):
                pass
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return 0.0, 0.0, 0.0
    
    s_clean = s.strip()
    if s_clean == "":
        return 0.0, 0.0, 0.0
    
    # Büyük harfe çevir ve boşlukları kaldır (sembol arama için)
    s_upper = s_clean.upper().replace(" ", "")
    
    # Özel durumlar
    if s_upper in ["NAN", "NULL", "NONE"]:
        return 0.0, 0.0, 0.0
    
    # 1. VİRGÜL formatı: t,i,f (3 parametre) - en basit ve güvenilir
    if ',' in s_clean and '(' not in s_clean and ')' not in s_clean:
        parts = [p.strip() for p in s_clean.split(',')]
        try:
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]), float(parts[1]), 0.0
            elif len(parts) == 1:
                return float(parts[0]), 0.0, 0.0
        except ValueError:
            # Bileşenlerden biri boş olabilir
            try:
                t_val = float(parts[0]) if parts[0] else 0.0
                i_val = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                f_val = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
                return t_val, i_val, f_val
            except (ValueError, IndexError):
                pass
    
    # 2. Regular expression ile daha güçlü parsing
    # Formatlar: "1.5", "1.5I", "1.5F", "1.5 + 0.3I", "1.5 + 0.3I + 0.2F"
    # İşaretleri ve birimleri doğru şekilde yakalamak için daha kapsamlı regex
    pattern = r"""
        ^\s*                                 # Başlangıç
        ([+-]?(?:\d+\.?\d*|\.\d+))?          # t değeri (opsiyonel)
        ([IF]?)                              # t birimi (opsiyonel)
        (?:                                  # İkinci terim (opsiyonel)
            \s*\+\s*                         # + işareti
            ([+-]?(?:\d+\.?\d*|\.\d+))?      # i/f değeri
            ([IF]?)                          # i/f birimi
        )?
        (?:                                  # Üçüncü terim (opsiyonel)
            \s*\+\s*                         # + işareti
            ([+-]?(?:\d+\.?\d*|\.\d+))?      # i/f değeri
            ([IF]?)                          # i/f birimi
        )?
        \s*$                                 # Son
    """
    
    match = re.match(pattern, s_clean, re.VERBOSE | re.IGNORECASE)
    
    if match:
        # Grupları al - bunlar string veya None olacak
        groups = match.groups()
        t_val_str, t_unit_str, i_val_str, i_unit_str, f_val_str, f_unit_str = groups
        
        # Debug için
        # print(f"Parsed groups: {groups}")
        
        # Başlangıç değerleri
        t, i, f = 0.0, 0.0, 0.0
        
        def parse_value(value_str: Optional[str], default: float = 0.0) -> float:
            """String değeri float'a çevir"""
            if not value_str:
                return default
            try:
                return float(value_str)
            except (ValueError, TypeError):
                # Özel durumlar: "+", "-", boş string
                if value_str == '+':
                    return 1.0
                elif value_str == '-':
                    return -1.0
                return default
        
        # İlk terim
        if t_val_str is not None:
            val = parse_value(t_val_str)
            if t_unit_str and t_unit_str.upper() == 'I':
                i = val
            elif t_unit_str and t_unit_str.upper() == 'F':
                f = val
            else:
                t = val
        
        # İkinci terim
        if i_val_str is not None:
            val = parse_value(i_val_str)
            if i_unit_str and i_unit_str.upper() == 'I':
                i = val
            elif i_unit_str and i_unit_str.upper() == 'F':
                f = val
            else:
                # Birim yoksa, hangi birime ait olduğunu belirle
                if t_unit_str and t_unit_str.upper() == 'I':
                    i += val
                elif t_unit_str and t_unit_str.upper() == 'F':
                    f += val
                else:
                    # t birimsizse, i'ye ekle (default I)
                    i = val
        
        # Üçüncü terim
        if f_val_str is not None:
            val = parse_value(f_val_str)
            if f_unit_str and f_unit_str.upper() == 'I':
                i = val
            elif f_unit_str and f_unit_str.upper() == 'F':
                f = val
            else:
                # Birim yoksa, hangi birime ait olduğunu belirle
                if i_unit_str and i_unit_str.upper() == 'I':
                    i += val
                elif i_unit_str and i_unit_str.upper() == 'F':
                    f += val
                elif t_unit_str and t_unit_str.upper() == 'I':
                    i += val
                elif t_unit_str and t_unit_str.upper() == 'F':
                    f += val
                else:
                    # Hiçbir birim yoksa, f'ye ekle (default F)
                    f = val
        
        return t, i, f
    
    # 3. Basit manuel parsing (regex başarısız olursa)
    # String'i büyük harfe çevir ve sembolleri ara
    s_upper = s_clean.upper().replace(" ", "")
    
    # Varsayılan değerler
    t, i, f = 0.0, 0.0, 0.0
    
    # "I" sembolünü ara
    if 'I' in s_upper:
        parts = s_upper.split('I', 1)
        before_i = parts[0]
        after_i = parts[1] if len(parts) > 1 else ''
        
        # I'dan önceki kısmı parse et
        if before_i:
            # Sayısal kısmı ayır
            num_match = re.search(r'([+-]?\d*\.?\d+)$', before_i)
            if num_match:
                t = float(num_match.group(1))
            elif before_i in ['+', '-']:
                t = 1.0 if before_i == '+' else -1.0
            elif before_i:
                # Sadece sayı olabilir
                try:
                    t = float(before_i)
                except ValueError:
                    pass
        
        # I'dan sonraki kısmı parse et (indeterminacy değeri)
        if after_i:
            try:
                i = float(after_i) if after_i not in ['', '+', '-'] else 1.0
                if after_i == '-':
                    i = -1.0
            except ValueError:
                i = 1.0  # Sadece "I" varsa
        else:
            i = 1.0  # Sadece "I" varsa
    
    # "F" sembolünü ara (I'dan bağımsız)
    if 'F' in s_upper:
        # I içeriyorsa, F'den önceki kısmı al
        if 'I' in s_upper:
            # "I...F" formatı
            i_match = re.search(r'I([^F]*)F', s_upper)
            if i_match:
                i_str = i_match.group(1)
                if i_str:
                    try:
                        i = float(i_str)
                    except ValueError:
                        if i_str in ['+', '-']:
                            i = 1.0 if i_str == '+' else -1.0
        else:
            # Sadece F içeriyor
            parts = s_upper.split('F', 1)
            before_f = parts[0]
            after_f = parts[1] if len(parts) > 1 else ''
            
            # F'dan önceki kısmı parse et
            if before_f:
                try:
                    t = float(before_f) if before_f not in ['', '+', '-'] else 0.0
                    if before_f == '+':
                        t = 1.0
                    elif before_f == '-':
                        t = -1.0
                except ValueError:
                    pass
            
            # F'dan sonraki kısmı parse et (falsity değeri)
            if after_f:
                try:
                    f = float(after_f) if after_f not in ['', '+', '-'] else 1.0
                    if after_f == '-':
                        f = -1.0
                except ValueError:
                    f = 1.0  # Sadece "F" varsa
            else:
                f = 1.0  # Sadece "F" varsa
    
    # 4. Hiçbir sembol yoksa, sadece sayı olabilir
    if not ('I' in s_upper or 'F' in s_upper):
        try:
            t = float(s_clean)
        except ValueError:
            # Parantez içinde olabilir
            if '(' in s_clean and ')' in s_clean:
                content = s_clean[s_clean.find('(')+1:s_clean.find(')')]
                try:
                    t = float(content)
                except ValueError:
                    pass
    
    return t, i, f

def _parse_neutrosophic_complex(s: Any) -> Tuple[float, float, float]:
    """
    Parses neutrosophic complex numbers into (t, i, f) tuple.
    
    Supports complex numbers where:
    - Real part represents truth value (t)
    - Imaginary part represents indeterminacy value (i)
    - Falsity value (f) is derived or set to 0
    
    Examples:
    - 3+4j -> (3.0, 4.0, 0.0)
    - (2+3j) -> (2.0, 3.0, 0.0)
    - complex(1.5, 2.5) -> (1.5, 2.5, 0.0)
    """
    import re
    
    # Eğer zaten kompleks sayı ise
    if isinstance(s, complex):
        return float(s.real), float(s.imag), 0.0
    
    # Eğer tuple/list ise ve kompleks sayı içeriyorsa
    if isinstance(s, (tuple, list)):
        if len(s) >= 1:
            # İlk eleman kompleks sayı olabilir
            if isinstance(s[0], complex):
                return float(s[0].real), float(s[0].imag), 0.0
            # Ya da 2 elemanlı (real, imag) olabilir
            elif len(s) >= 2:
                try:
                    real = float(s[0])
                    imag = float(s[1])
                    return real, imag, 0.0
                except (ValueError, TypeError):
                    pass
    
    # String işlemleri
    if isinstance(s, str):
        s_clean = s.strip()
        
        # 1. Kompleks sayı formatı: "a+bj" veya "a-bj"
        # Python'da kompleks sayı formatı
        complex_pattern = r"""
            ^\s*                                      # Başlangıç
            ([+-]?\d*\.?\d+)                          # Real kısım
            \s*                                       # Boşluk
            ([+-])\s*                                 # İşaret
            \s*                                       # Boşluk
            (\d*\.?\d+)\s*j\s*$                       # Imag kısım + j
        """
        
        match = re.match(complex_pattern, s_clean, re.VERBOSE | re.IGNORECASE)
        if match:
            try:
                real = float(match.group(1))
                sign = match.group(2)
                imag_str = match.group(3)
                
                imag = float(imag_str)
                if sign == '-':
                    imag = -imag
                
                return real, imag, 0.0
            except ValueError:
                pass
        
        # 2. Parantez içinde kompleks sayı: "(a+bj)"
        if '(' in s_clean and ')' in s_clean and 'j' in s_clean.lower():
            # Parantez içeriğini al
            content = s_clean[s_clean.find('(')+1:s_clean.find(')')].strip()
            try:
                # Python'ın kompleks sayı parser'ını kullan
                c = complex(content)
                return float(c.real), float(c.imag), 0.0
            except ValueError:
                pass
        
        # 3. "complex(a, b)" formatı
        if s_clean.lower().startswith('complex'):
            # "complex(1.5, 2.5)" veya "complex(1.5,2.5)" formatı
            match = re.match(r'complex\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', s_clean, re.IGNORECASE)
            if match:
                try:
                    real = float(match.group(1))
                    imag = float(match.group(2))
                    return real, imag, 0.0
                except ValueError:
                    pass
    
    # 4. Diğer formatlar için _parse_neutrosophic'i dene
    # (Bu, önceki fonksiyonunuz)
    try:
        t, i, f = _parse_neutrosophic(s)
        # Eğer i değeri varsa ve t ile f 0 ise, bu kompleks sayı olabilir
        if i != 0.0 and t == 0.0 and f == 0.0:
            return 0.0, i, 0.0
        return t, i, f
    except NameError:
        # _parse_neutrosophic fonksiyonu tanımlı değilse
        pass
    
    # 5. Sayısal dönüşüm dene
    try:
        # Float'a çevirmeyi dene
        val = float(s)
        return val, 0.0, 0.0
    except (ValueError, TypeError):
        pass
    
    # 6. Hiçbir şey çalışmazsa varsayılan değer
    return 0.0, 0.0, 0.0


def _parse_hyperreal(s: Any) -> Tuple[float, float, List[float]]:
    """
    Parses hyperreal representations into (finite, infinitesimal, sequence) tuple.
    
    Supports extended hyperreal formats including:
    
    BASIC FORMATS:
    - Tuple/list: [1.0, 0.5] or (1.0, 0.5, 0.1)
    - Numeric: 5.0 -> (5.0, 0.0, [5.0])
    - Complex: 3+4j -> (3.0, 4.0, [3.0, 4.0])
    
    STRING FORMATS:
    - Comma-separated: "1.5,0.3" -> finite=1.5, infinitesimal=0.3
    - Exponential: "1.5ε0.3" or "1.5e0.3"
    - Sequence: "[1.0, 0.5, 0.1]"
    - Standard: "1.5 + 0.3ε" or "2.0 - 0.5ε"
    
    EXTENDED FORMATS:
    - Infinities: "∞", "inf", "-infinity"
    - Infinitesimals: "ε", "dx", "dt", "dh"
    - Engineering: "1.5kε0.3" (k=1e3 multiplier)
    - Scientific: "1.23e-4ε2.5e-6"
    - Mixed: "π + 0.001ε" or "e - 0.0001ε"
    
    Returns:
        Tuple[float, float, List[float]]: 
            - finite part (standard real component)
            - infinitesimal part (ε coefficient)
            - full sequence representation
    """
    import re
    import math
    import warnings
    
    # 1. Eğer zaten Hyperreal instance ise
    if hasattr(s, '__class__'):
        class_name = s.__class__.__name__
        if class_name in ['Hyperreal', 'HyperReal']:
            try:
                if hasattr(s, 'finite') and hasattr(s, 'infinitesimal'):
                    finite = float(s.finite)
                    infinitesimal = float(s.infinitesimal)
                    seq = getattr(s, 'sequence', [finite, infinitesimal])
                    return finite, infinitesimal, seq
                elif hasattr(s, 'real') and hasattr(s, 'epsilon'):
                    finite = float(s.real)
                    infinitesimal = float(s.epsilon)
                    seq = getattr(s, 'sequence', [finite, infinitesimal])
                    return finite, infinitesimal, seq
            except (AttributeError, ValueError, TypeError):
                pass
    
    # 2. Tuple/list için
    if isinstance(s, (tuple, list)):
        try:
            seq = []
            for item in s:
                # Özel değerleri kontrol et
                if isinstance(item, str):
                    item_str = item.strip().lower()
                    if item_str in ['inf', 'infinity', '∞']:
                        seq.append(float('inf'))
                    elif item_str in ['-inf', '-infinity', '-∞']:
                        seq.append(float('-inf'))
                    elif item_str in ['nan', 'null']:
                        seq.append(float('nan'))
                    elif 'ε' in item_str or 'epsilon' in item_str:
                        # ε içeriyorsa, infinitesimal bileşen olarak işle
                        num = re.sub(r'[εepsilon]', '', item_str, flags=re.IGNORECASE)
                        if num in ['', '+']:
                            seq.append(1.0)
                        elif num == '-':
                            seq.append(-1.0)
                        else:
                            seq.append(float(num))
                    else:
                        seq.append(float(item))
                else:
                    seq.append(float(item))
            
            finite = seq[0] if seq else 0.0
            infinitesimal = seq[1] if len(seq) > 1 else 0.0
            return finite, infinitesimal, seq
        except (ValueError, IndexError, TypeError) as e:
            warnings.warn(f"Hyperreal tuple/list parse error: {e}", RuntimeWarning, stacklevel=2)
    
    # 3. Sayısal tipler için
    if isinstance(s, (float, int)):
        return float(s), 0.0, [float(s)]
    elif isinstance(s, complex):
        # Karmaşık sayı: real -> finite, imag -> infinitesimal
        return float(s.real), float(s.imag), [float(s.real), float(s.imag)]
    
    # 4. String işlemleri için
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            warnings.warn(f"Hyperreal conversion to string failed: {e}", RuntimeWarning, stacklevel=2)
            return 0.0, 0.0, [0.0]
    
    s_clean = s.strip()
    
    # 5. Özel durumlar
    if s_clean == "":
        return 0.0, 0.0, [0.0]
    
    # Sonsuzluk değerleri
    infinity_map = {
        '∞': float('inf'), 'inf': float('inf'), 'infinity': float('inf'),
        '+∞': float('inf'), '+inf': float('inf'), '+infinity': float('inf'),
        '-∞': float('-inf'), '-inf': float('-inf'), '-infinity': float('-inf')
    }
    
    s_lower = s_clean.lower()
    if s_lower in infinity_map:
        value = infinity_map[s_lower]
        return value, 0.0, [value]
    
    # NaN değerleri
    if s_lower in ['nan', 'null', 'none', 'undefined']:
        return float('nan'), 0.0, [float('nan')]
    
    # 6. Köşeli parantez içinde sequence (JSON benzeri)
    if s_clean.startswith('[') and s_clean.endswith(']'):
        try:
            content = s_clean[1:-1].strip()
            if content:
                parts = [p.strip() for p in re.split(r',|;', content)]
                seq = []
                for p in parts:
                    if p:
                        try:
                            # Özel sembolleri kontrol et
                            if p.lower() in infinity_map:
                                seq.append(infinity_map[p.lower()])
                            elif p.lower() == 'nan':
                                seq.append(float('nan'))
                            else:
                                seq.append(float(p))
                        except ValueError:
                            # Mühendislik notasyonu olabilir
                            try:
                                # 1.5k, 2.3m gibi
                                val = _parse_engineering_notation(p)
                                seq.append(val)
                            except:
                                seq.append(0.0)
                
                finite = seq[0] if seq else 0.0
                infinitesimal = seq[1] if len(seq) > 1 else 0.0
                return finite, infinitesimal, seq
        except Exception as e:
            warnings.warn(f"Hyperreal sequence parse error: {e}", RuntimeWarning, stacklevel=2)
    
    # 7. Virgülle ayrılmış format: a,b,c
    if ',' in s_clean and not s_clean.startswith('(') and not s_clean.endswith(')'):
        try:
            parts = [p.strip() for p in s_clean.split(',')]
            seq = []
            for p in parts:
                if p:
                    try:
                        seq.append(float(p))
                    except ValueError:
                        # Özel değerleri kontrol et
                        if p.lower() in infinity_map:
                            seq.append(infinity_map[p.lower()])
                        elif p.lower() == 'nan':
                            seq.append(float('nan'))
                        else:
                            seq.append(0.0)
            
            finite = seq[0] if seq else 0.0
            infinitesimal = seq[1] if len(seq) > 1 else 0.0
            return finite, infinitesimal, seq
        except Exception as e:
            warnings.warn(f"Hyperreal comma-separated parse error: {e}", RuntimeWarning, stacklevel=2)
    
    # 8. GELİŞMİŞ: Matematiksel ifadeler (π, e, φ gibi sabitler)
    constants = {
        'π': math.pi, 'pi': math.pi,
        'e': math.e,
        'φ': (1 + math.sqrt(5)) / 2, 'phi': (1 + math.sqrt(5)) / 2,
    }
    
    # Sabit içerip içermediğini kontrol et
    for const_name, const_value in constants.items():
        if const_name.lower() in s_lower:
            # Sabitin değerini al
            const_val = const_value
            # ε ile kombinasyonu kontrol et
            if 'ε' in s_clean or 'epsilon' in s_lower:
                # "π + 0.1ε" formatı
                match = re.search(r'([+-]?\s*\d*\.?\d+)\s*[εε]', s_clean, re.IGNORECASE)
                if match:
                    eps_val = float(match.group(1).replace(' ', '')) if match.group(1).strip() not in ['', '+', '-'] else 1.0
                    if match.group(1).strip() == '-':
                        eps_val = -1.0
                    return const_val, eps_val, [const_val, eps_val]
                else:
                    return const_val, 0.0, [const_val]
            else:
                return const_val, 0.0, [const_val]
    
    # 9. Exponential/epsilon formatları
    # "aεb", "a e b", "a + bε", "a - bε"
    epsilon_patterns = [
        r'^\s*([+-]?\d*\.?\d+)\s*[εε]\s*([+-]?\d*\.?\d+)\s*$',  # aεb
        r'^\s*([+-]?\d*\.?\d+)\s*e\s*([+-]?\d*\.?\d+)\s*$',     # a e b (hyperreal)
        r'^\s*([+-]?\d*\.?\d+)\s*\+\s*([+-]?\d*\.?\d+)\s*[εε]\s*$',  # a + bε
        r'^\s*([+-]?\d*\.?\d+)\s*\-\s*([+-]?\d*\.?\d+)\s*[εε]\s*$',  # a - bε
    ]
    
    for pattern in epsilon_patterns:
        match = re.match(pattern, s_clean, re.IGNORECASE)
        if match:
            try:
                finite_val = float(match.group(1))
                eps_val = float(match.group(2))
                return finite_val, eps_val, [finite_val, eps_val]
            except ValueError:
                continue
    
    # 10. Mühendislik notasyonu ile hyperreal
    # "1.5kε0.3m" gibi
    eng_pattern = r'^\s*([+-]?\d*\.?\d+)([kKmMgGtTμμunpf]?)\s*[εε]\s*([+-]?\d*\.?\d+)([kKmMgGtTμμunpf]?)\s*$'
    match = re.match(eng_pattern, s_clean, re.IGNORECASE)
    if match:
        try:
            finite_num = float(match.group(1))
            finite_unit = match.group(2).lower()
            eps_num = float(match.group(3))
            eps_unit = match.group(4).lower()
            
            # Mühendislik çarpanları
            multipliers = {
                'k': 1e3, 'm': 1e-3, 'meg': 1e6, 'g': 1e9,
                't': 1e12, 'μ': 1e-6, 'u': 1e-6, 'n': 1e-9,
                'p': 1e-12, 'f': 1e-15
            }
            
            finite = finite_num * multipliers.get(finite_unit, 1.0)
            infinitesimal = eps_num * multipliers.get(eps_unit, 1.0)
            return finite, infinitesimal, [finite, infinitesimal]
        except (ValueError, KeyError):
            pass
    
    # 11. Sadece epsilon (infinitesimal) formatı: "ε", "0.5ε", "-ε"
    epsilon_only = re.match(r'^\s*([+-]?\d*\.?\d*)\s*[εε]\s*$', s_clean, re.IGNORECASE)
    if epsilon_only:
        try:
            eps_str = epsilon_only.group(1)
            if eps_str in ['', '+']:
                infinitesimal = 1.0
            elif eps_str == '-':
                infinitesimal = -1.0
            else:
                infinitesimal = float(eps_str)
            return 0.0, infinitesimal, [0.0, infinitesimal]
        except ValueError:
            pass
    
    # 12. Bilimsel gösterim (hyperreal olmayan)
    sci_pattern = r'^[+-]?\d*\.?\d+[eE][+-]?\d+$'
    if re.match(sci_pattern, s_clean):
        try:
            value = float(s_clean)
            return value, 0.0, [value]
        except ValueError:
            pass
    
    # 13. Sadece sayı
    try:
        # Mühendislik notasyonu olabilir
        value = _parse_engineering_notation(s_clean)
        return value, 0.0, [value]
    except (ValueError, TypeError):
        pass
    
    # 14. Varsayılan
    warnings.warn(f"Could not parse hyperreal: '{s}'", RuntimeWarning, stacklevel=2)
    return 0.0, 0.0, [0.0]

# Yardımcı fonksiyon
def _parse_engineering_notation(s: str) -> float:
    """Parse engineering notation (1.5k, 2.3m, etc.)"""
    import re
    
    s = s.strip().lower()
    
    # Mühendislik çarpanları
    multipliers = {
        'k': 1e3, 'm': 1e-3, 'meg': 1e6, 'g': 1e9,
        't': 1e12, 'μ': 1e-6, 'u': 1e-6, 'n': 1e-9,
        'p': 1e-12, 'f': 1e-15, 'a': 1e-18,
        'mil': 1000  # thousand
    }
    
    # Regex pattern
    pattern = r'^([+-]?\d*\.?\d+)\s*([a-zμ]+)?$'
    match = re.match(pattern, s)
    
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2) or ''
            
            if unit in multipliers:
                return value * multipliers[unit]
            elif unit == '':
                return value
            
            # Özel birimler
            if unit.startswith('e'):
                # 1.5e-3 gibi
                return float(s)
        except (ValueError, KeyError):
            pass
    
    # Standart float dönüşümü
    return float(s)

# Yardımcı fonksiyonlar
def parse_to_neutrosophic(s: Any) -> 'NeutrosophicNumber':
    """Parse to NeutrosophicNumber object directly"""
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .kececinumbers import NeutrosophicNumber
    
    t, i, f = _parse_neutrosophic(s)
    return NeutrosophicNumber(t, i, f)


def parse_to_hyperreal(s: Any) -> 'HyperrealNumber':
    """Parse to Hyperreal object directly"""
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .kececinumbers import HyperrealNumber
    
    finite, infinitesimal, seq = _parse_hyperreal(s)
    return HyperrealNumber(sequence=seq)

def _parse_quaternion_from_csv(s) -> quaternion:
    """Virgülle ayrılmış string'i veya sayıyı quaternion'a dönüştürür.
    
    Args:
        s: Dönüştürülecek değer. Şu formatları destekler:
            - quaternion nesnesi (doğrudan döndürülür)
            - float, int, complex sayılar (skaler quaternion)
            - String ("w,x,y,z" veya "scalar" formatında)
            - Diğer tipler (string'e dönüştürülerek işlenir)
    
    Returns:
        quaternion: Dönüştürülmüş kuaterniyon
    
    Raises:
        ValueError: Geçersiz format veya sayısal olmayan bileşenler durumunda
    """
    # Eğer zaten quaternion ise doğrudan döndür
    if isinstance(s, quaternion):
        return s
    
    # Sayısal tipse skaler quaternion olarak işle
    if isinstance(s, (float, int)):
        return quaternion(float(s), 0, 0, 0)
    
    # Complex sayı için özel işlem
    if isinstance(s, complex):
        # Complex sayının sadece gerçek kısmını al
        return quaternion(float(s.real), 0, 0, 0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    
    # Boş string kontrolü
    if not s:
        raise ValueError(f"Boş string quaternion'a dönüştürülemez")
    
    # String'i virgülle ayır
    parts_str = s.split(',')
    
    # Tüm parçaları float'a dönüştürmeyi dene
    parts_float = []
    for p in parts_str:
        p = p.strip()
        if not p:
            raise ValueError(f"Boş bileşen bulundu: '{s}'")
        
        try:
            # Önce normal float olarak dene
            parts_float.append(float(p))
        except ValueError:
            # Float olarak parse edilemezse complex olarak dene
            try:
                # 'i' karakterini 'j' ile değiştir (complex fonksiyonu 'j' bekler)
                complex_str = p.replace('i', 'j').replace('I', 'J')
                # Eğer 'j' yoksa ve sayı değilse hata ver
                if 'j' not in complex_str.lower():
                    raise ValueError(f"Geçersiz sayı formatı: '{p}'")
                
                c = complex(complex_str)
                parts_float.append(float(c.real))
            except ValueError:
                raise ValueError(f"quaternion bileşeni sayı olmalı: '{p}' (string: '{s}')")

    if len(parts_float) == 4:
        return quaternion(*parts_float)
    elif len(parts_float) == 1:  # Sadece skaler değer
        return quaternion(parts_float[0], 0, 0, 0)
    else:
        raise ValueError(f"Geçersiz quaternion formatı. 1 veya 4 bileşen bekleniyor, {len(parts_float)} alındı: '{s}'")

def _has_comma_format(s: Any) -> bool:
    """
    True if value is a string and contains a comma (CSV-like format).
    Guard against non-strings.
    """
    if s is None:
        return False
    if not isinstance(s, str):
        s = str(s)
    # Consider comma-format only when there's at least one digit and a comma
    return ',' in s and bool(re.search(r'\d', s))

def _is_complex_like(s: Any) -> bool:
    """
    Check if s looks like a complex literal (contains 'j'/'i' or +-/ with j).
    Accepts non-strings by attempting to str() them.
    """
    if s is None:
        return False
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # quick checks
    if 'j' in s or 'i' in s:
        return True
    # pattern like "a+bi" or "a-bi"
    if re.search(r'[+-]\d', s) and ('+' in s or '-' in s):
        # avoid classifying comma-separated lists as complex
        if ',' in s:
            return False
        return True
    return False

def _safe_float_convert(value: Any) -> float:
    """
    Güvenli float dönüşümü.
    
    Args:
        value: Dönüştürülecek değer
        
    Returns:
        Float değeri veya 0.0
    """
    if isinstance(value, (float, int)):
        return float(value)
    elif isinstance(value, complex):
        return float(value.real)  # veya abs(value) seçeneği
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            # Özel semboller
            value_upper = value.upper().strip()
            if value_upper in ['', 'NAN', 'NULL', 'NONE']:
                return 0.0
            elif value_upper == 'INF' or value_upper == 'INFINITY':
                return float('inf')
            elif value_upper == '-INF' or value_upper == '-INFINITY':
                return float('-inf')
            # '+' veya '-' işaretleri
            elif value == '+':
                return 1.0
            elif value == '-':
                return -1.0
            else:
                try:
                    # Karmaşık sayı string'i olabilir
                    if 'j' in value or 'J' in value:
                        c = complex(value)
                        return float(c.real)
                except ValueError:
                    pass
                return 0.0
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


def _parse_complex_like_string(s: str) -> List[float]:
    """
    Karmaşık sayı benzeri string'i float listesine çevirir.
    Örnek: "1+2i-3j+4k" -> [1.0, 2.0, -3.0, 4.0, ...]
    """
    if not s:
        return [0.0]
    
    # Normalize et
    s = s.replace(' ', '').replace('J', 'j').replace('I', 'j').upper()
    
    # Tüm imajiner birimleri normalize et
    units = ['J', 'I', 'K', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    
    # İlk bileşen (reel kısım)
    result = [0.0] * (len(units) + 1)
    
    # Reel kısmı bul
    pattern = r'^([+-]?\d*\.?\d*)(?![' + ''.join(units) + '])'
    match = re.match(pattern, s)
    if match and match.group(1):
        result[0] = _safe_float_convert(match.group(1))
    
    # Her bir imajiner birim için
    for i, unit in enumerate(units, 1):
        pattern = r'([+-]?\d*\.?\d*)' + re.escape(unit)
        matches = re.findall(pattern, s)
        if matches:
            # Son eşleşmeyi al (tekrarlanmışsa)
            last_match = matches[-1]
            result[i] = _safe_float_convert(last_match)
    
    return result


def _parse_neutrosophic_bicomplex(s: Any) -> NeutrosophicBicomplexNumber:
    """
    Parses string or numbers into NeutrosophicBicomplexNumber.
    
    Supports:
    - NeutrosophicBicomplexNumber instance
    - Numeric types (float, int, complex)
    - Comma-separated string: "1,2,3,4,5,6,7,8"
    - List/tuple of 8 values
    """
    # Eğer zaten NeutrosophicBicomplexNumber ise doğrudan döndür
    if isinstance(s, NeutrosophicBicomplexNumber):
        return s
    
    # List/tuple ise
    if isinstance(s, (list, tuple)):
        if len(s) == 8:
            try:
                values = [_safe_float_convert(v) for v in s]
                return NeutrosophicBicomplexNumber(*values)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid component values: {s}") from e
        else:
            raise ValueError(f"Expected 8 components, got {len(s)}")
    
    # Sayısal tipse tüm bileşenler 0, sadece ilk bileşen değerli
    if isinstance(s, (float, int)):
        values = [_safe_float_convert(s)] + [0.0] * 7
        return NeutrosophicBicomplexNumber(*values)
    elif isinstance(s, complex):
        values = [_safe_float_convert(s.real), _safe_float_convert(s.imag)] + [0.0] * 6
        return NeutrosophicBicomplexNumber(*values)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            raise ValueError(f"Cannot convert to string: {s}") from e
    
    s = s.strip()
    if not s:
        return NeutrosophicBicomplexNumber(0, 0, 0, 0, 0, 0, 0, 0)
    
    # Virgülle ayrılmış format
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 8:
            try:
                values = [_safe_float_convert(p) for p in parts]
                return NeutrosophicBicomplexNumber(*values)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid component values in: '{s}'") from e
        else:
            # Virgül var ama 8 değil
            if len(parts) < 8:
                # Eksik değerleri 0 ile tamamla
                values = [_safe_float_convert(p) for p in parts] + [0.0] * (8 - len(parts))
                return NeutrosophicBicomplexNumber(*values)
            else:
                # Fazla değer varsa ilk 8'ini al
                values = [_safe_float_convert(p) for p in parts[:8]]
                return NeutrosophicBicomplexNumber(*values)
    
    # Karmaşık sayı formatı deneyelim
    try:
        # "1+2i+3j+4k+..." formatı
        values = _parse_complex_like_string(s)
        if len(values) >= 8:
            return NeutrosophicBicomplexNumber(*values[:8])
        else:
            values = values + [0.0] * (8 - len(values))
            return NeutrosophicBicomplexNumber(*values)
    except Exception:
        pass
    
    # Sadece sayı olabilir
    try:
        scalar = _safe_float_convert(s)
        values = [scalar] + [0.0] * 7
        return NeutrosophicBicomplexNumber(*values)
    except ValueError as e:
        raise ValueError(f"Invalid NeutrosophicBicomplex format: '{s}'") from e


def _parse_octonion(s: Any) -> OctonionNumber:
    """
    String'i veya sayıyı OctonionNumber'a dönüştürür.
    
    Supports:
    - OctonionNumber instance
    - Numeric types (float, int, complex)
    - Comma-separated string: "1,2,3,4,5,6,7,8"
    - Symbolic string: "1 + 2i + 3j + 4k + 5e + 6f + 7g + 8h"
    """
    # Eğer zaten OctonionNumber ise doğrudan döndür
    if isinstance(s, OctonionNumber):
        return s
    
    # List/tuple ise
    if isinstance(s, (list, tuple)):
        if len(s) == 8:
            try:
                return OctonionNumber(*[_safe_float_convert(v) for v in s])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid octonion component values: {s}") from e
        elif len(s) == 1:
            scalar = _safe_float_convert(s[0])
            return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
        else:
            # Eksik veya fazla bileşen
            if len(s) < 8:
                values = [_safe_float_convert(v) for v in s] + [0.0] * (8 - len(s))
                return OctonionNumber(*values)
            else:
                values = [_safe_float_convert(v) for v in s[:8]]
                return OctonionNumber(*values)
    
    # Sayısal tipse skaler olarak işle
    if isinstance(s, (float, int)):
        scalar = float(s)
        return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
    elif isinstance(s, complex):
        return OctonionNumber(s.real, s.imag, 0, 0, 0, 0, 0, 0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            raise ValueError(f"Cannot convert to string: {s}") from e
    
    s_clean = s.strip()
    if not s_clean:
        return OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
    
    # Virgülle ayrılmış format
    if ',' in s_clean:
        try:
            parts = [_safe_float_convert(p.strip()) for p in s_clean.split(',')]
            if len(parts) == 8:
                return OctonionNumber(*parts)
            elif len(parts) < 8:
                parts = parts + [0.0] * (8 - len(parts))
                return OctonionNumber(*parts)
            else:
                return OctonionNumber(*parts[:8])
        except ValueError as e:
            raise ValueError(f"Invalid octonion format: '{s}'") from e
    
    # Sembolik format (1 + 2i + 3j + ...)
    # Octonion birimleri: 1, i, j, k, e, f, g, h
    units = ['I', 'J', 'K', 'E', 'F', 'G', 'H']
    
    # Normalize et
    s_norm = s_clean.replace(' ', '').upper()
    
    # Reel kısmı bul
    real_part = 0.0
    unit_values = {unit: 0.0 for unit in units}
    
    # Regex pattern: sayı + birim (opsiyonel)
    # Önce reel kısmı bul
    real_match = re.match(r'^([+-]?\d*\.?\d*)(?![IJKEFGH])', s_norm)
    if real_match and real_match.group(1) and real_match.group(1) not in ['', '+', '-']:
        real_part = _safe_float_convert(real_match.group(1))
    
    # Her birim için değerleri bul
    for unit in units:
        pattern = r'([+-]?\d*\.?\d*)' + re.escape(unit)
        matches = re.findall(pattern, s_norm)
        if matches:
            last_match = matches[-1]
            if last_match in ['', '+']:
                unit_values[unit] = 1.0
            elif last_match == '-':
                unit_values[unit] = -1.0
            else:
                unit_values[unit] = _safe_float_convert(last_match)
    
    # Octonion oluştur
    return OctonionNumber(
        real_part,
        unit_values.get('I', 0.0),
        unit_values.get('J', 0.0),
        unit_values.get('K', 0.0),
        unit_values.get('E', 0.0),
        unit_values.get('F', 0.0),
        unit_values.get('G', 0.0),
        unit_values.get('H', 0.0)
    )


def _parse_sedenion(s: Any) -> SedenionNumber:
    """
    String'i veya sayıyı SedenionNumber'a dönüştürür.
    
    Supports:
    - SedenionNumber instance
    - Numeric types
    - Comma-separated string (16 values)
    - List/tuple of 16 values
    """
    # Eğer zaten SedenionNumber ise doğrudan döndür
    if isinstance(s, SedenionNumber):
        return s
    
    # List/tuple ise
    if isinstance(s, (list, tuple)):
        if len(s) == 16:
            try:
                values = [_safe_float_convert(v) for v in s]
                return SedenionNumber(values)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid sedenion component values: {s}") from e
        elif len(s) == 1:
            scalar = _safe_float_convert(s[0])
            return SedenionNumber([scalar] + [0.0] * 15)
        else:
            # Eksik veya fazla bileşen
            if len(s) < 16:
                values = [_safe_float_convert(v) for v in s] + [0.0] * (16 - len(s))
                return SedenionNumber(values)
            else:
                values = [_safe_float_convert(v) for v in s[:16]]
                return SedenionNumber(values)
    
    # Sayısal tipse skaler olarak işle
    if isinstance(s, (float, int)):
        scalar = float(s)
        return SedenionNumber([scalar] + [0.0] * 15)
    elif isinstance(s, complex):
        return SedenionNumber([s.real, s.imag] + [0.0] * 14)
    
    # String işlemleri
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            raise ValueError(f"Cannot convert to string: {s}") from e
    
    s = s.strip()
    if not s:
        return SedenionNumber([0.0] * 16)
    
    # Köşeli parantezleri kaldır
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    
    # Virgülle ayrılmışsa
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        
        if len(parts) == 16:
            try:
                values = [_safe_float_convert(p) for p in parts]
                return SedenionNumber(values)
            except ValueError as e:
                raise ValueError(f"Invalid sedenion component values: '{s}'") from e
        elif len(parts) == 1:
            try:
                scalar = _safe_float_convert(parts[0])
                return SedenionNumber([scalar] + [0.0] * 15)
            except ValueError as e:
                raise ValueError(f"Invalid scalar sedenion value: '{s}'") from e
        else:
            # Eksik veya fazla
            if len(parts) < 16:
                values = [_safe_float_convert(p) for p in parts] + [0.0] * (16 - len(parts))
                return SedenionNumber(values)
            else:
                values = [_safe_float_convert(p) for p in parts[:16]]
                return SedenionNumber(values)
    
    # Sadece sayı
    try:
        scalar = _safe_float_convert(s)
        return SedenionNumber([scalar] + [0.0] * 15)
    except ValueError as e:
        raise ValueError(f"Invalid sedenion format: '{s}'") from e


def _parse_pathion(s: Any) -> PathionNumber:
    """
    String'i veya sayıyı PathionNumber'a dönüştürür.
    
    Supports:
    - PathionNumber instance
    - Numeric types
    - Comma-separated string (32 values)
    - List/tuple of 32 values
    """
    # Eğer zaten PathionNumber ise doğrudan döndür
    if isinstance(s, PathionNumber):
        return s
    
    # Iterable ise (list, tuple, etc.)
    if hasattr(s, '__iter__') and not isinstance(s, str):
        try:
            values = list(s)
            if len(values) == 32:
                return PathionNumber([_safe_float_convert(v) for v in values])
            elif len(values) == 1:
                scalar = _safe_float_convert(values[0])
                return PathionNumber([scalar] + [0.0] * 31)
            else:
                if len(values) < 32:
                    values = [_safe_float_convert(v) for v in values] + [0.0] * (32 - len(values))
                else:
                    values = [_safe_float_convert(v) for v in values[:32]]
                return PathionNumber(values)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid pathion component values: {s}") from e
    
    # Sayısal tipse skaler olarak işle
    if isinstance(s, (float, int)):
        scalar = float(s)
        return PathionNumber([scalar] + [0.0] * 31)
    elif isinstance(s, complex):
        return PathionNumber([s.real, s.imag] + [0.0] * 30)
    
    # String işlemleri
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            raise ValueError(f"Cannot convert to string: {s}") from e
    
    s = s.strip()
    if not s:
        return PathionNumber([0.0] * 32)
    
    # Köşeli parantezleri kaldır
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    
    # Virgülle ayrılmışsa
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        
        if len(parts) == 32:
            try:
                values = [_safe_float_convert(p) for p in parts]
                return PathionNumber(values)
            except ValueError as e:
                raise ValueError(f"Invalid pathion component values: '{s}'") from e
        elif len(parts) == 1:
            try:
                scalar = _safe_float_convert(parts[0])
                return PathionNumber([scalar] + [0.0] * 31)
            except ValueError as e:
                raise ValueError(f"Invalid scalar pathion value: '{s}'") from e
        else:
            # Eksik veya fazla
            if len(parts) < 32:
                values = [_safe_float_convert(p) for p in parts] + [0.0] * (32 - len(parts))
                return PathionNumber(values)
            else:
                values = [_safe_float_convert(p) for p in parts[:32]]
                return PathionNumber(values)
    
    # Sadece sayı
    try:
        scalar = _safe_float_convert(s)
        return PathionNumber([scalar] + [0.0] * 31)
    except ValueError as e:
        raise ValueError(f"Invalid pathion format: '{s}'") from e

def _parse_chingon(s: Any) -> 'ChingonNumber':
    """
    String'i veya sayıyı ChingonNumber'a dönüştürür.
    
    ChingonNumber: 64 bileşenli sayı sistemi
    
    Desteklenen formatlar:
    1. ChingonNumber instance: doğrudan döndür
    2. Sayısal tipler (float, int, complex): skaler olarak işlenir
    3. List/tuple/iterable: 64 bileşenli dizi
    4. String formatları:
       - Virgülle ayrılmış: "1.0,2.0,3.0,..."
       - Köşeli parantez: "[1.0,2.0,3.0,...]"
       - Sembolik: "1 + 2e1 + 3e2 + ..." (e0-e63 birimleri)
       - Kısmi bileşen: "1,2,3" (kalanlar 0 olur)
    
    Args:
        s: Dönüştürülecek girdi
        
    Returns:
        ChingonNumber instance
    """
    from .kececinumbers import ChingonNumber
    
    # 1. Eğer zaten ChingonNumber ise doğrudan döndür
    if isinstance(s, ChingonNumber):
        return s
    
    # 2. Sayısal tipler için skaler dönüşüm
    if isinstance(s, (float, int)):
        try:
            scalar = float(s)
            return ChingonNumber.from_scalar(scalar)
        except Exception as e:
            raise ValueError(f"Cannot convert scalar {s} to ChingonNumber: {e}") from e
    
    elif isinstance(s, complex):
        try:
            # Karmaşık sayı: real -> ilk bileşen, imag -> ikinci bileşen
            components = [float(s.real), float(s.imag)] + [0.0] * 62
            return ChingonNumber(components)
        except Exception as e:
            raise ValueError(f"Cannot convert complex {s} to ChingonNumber: {e}") from e
    
    # 3. Iterable tipler için
    if hasattr(s, '__iter__') and not isinstance(s, str):
        try:
            # Iterable'ı listeye çevir
            values = list(s)
            
            # Özel durum: tek elemanlı iterable
            if len(values) == 1:
                try:
                    scalar = float(values[0])
                    return ChingonNumber.from_scalar(scalar)
                except (ValueError, TypeError):
                    pass
            
            # 64 bileşene tamamla veya kes
            if len(values) != 64:
                if len(values) < 64:
                    values = values + [0.0] * (64 - len(values))
                else:
                    values = values[:64]
            
            # Float'a çevir
            float_values = []
            for v in values:
                try:
                    float_values.append(float(v))
                except (ValueError, TypeError):
                    float_values.append(0.0)
            
            # Doğrudan liste ile ChingonNumber oluştur
            return ChingonNumber(float_values)
            
        except Exception as e:
            raise ValueError(f"Cannot convert iterable {type(s).__name__} to ChingonNumber: {e}") from e
    
    # 4. String işlemleri
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            raise ValueError(f"Cannot convert {type(s).__name__} to string: {e}") from e
    
    s = s.strip()
    if not s:
        # Boş string için sıfır ChingonNumber
        return ChingonNumber()
    
    # Özel durumlar
    s_upper = s.upper()
    if s_upper in ["NAN", "NULL", "NONE", ""]:
        return ChingonNumber()
    elif s_upper in ["INF", "INFINITY"]:
        values = [float('inf')] + [0.0] * 63
        return ChingonNumber(values)
    elif s_upper in ["-INF", "-INFINITY"]:
        values = [float('-inf')] + [0.0] * 63
        return ChingonNumber(values)
    
    # 5. Köşeli parantezleri kaldır
    s_clean = s.strip()
    if s_clean.startswith('[') and s_clean.endswith(']'):
        s_clean = s_clean[1:-1].strip()
    elif s_clean.startswith('(') and s_clean.endswith(')'):
        s_clean = s_clean[1:-1].strip()
    
    # 6. Virgülle ayrılmış format
    if ',' in s_clean:
        try:
            parts = [p.strip() for p in s_clean.split(',')]
            
            if len(parts) == 64:
                # Tam 64 bileşen
                try:
                    values = [float(p) for p in parts]
                    return ChingonNumber(values)
                except ValueError as e:
                    raise ValueError(f"Invalid Chingon component value in: '{s}'") from e
                    
            elif len(parts) == 1:
                # Sadece skaler değer
                try:
                    scalar_val = float(parts[0])
                    return ChingonNumber.from_scalar(scalar_val)
                except ValueError as e:
                    raise ValueError(f"Invalid scalar Chingon value: '{s}'") from e
                    
            else:
                # Kısmi bileşenler
                values = []
                for p in parts:
                    try:
                        values.append(float(p))
                    except ValueError:
                        values.append(0.0)
                
                # 64 bileşene tamamla
                if len(values) < 64:
                    values = values + [0.0] * (64 - len(values))
                else:
                    values = values[:64]
                
                return ChingonNumber(values)
                
        except Exception as e:
            raise ValueError(f"Error parsing Chingon string: '{s}'") from e
    
    # 7. Sembolik format
    if any(f'e{i}' in s_clean.lower() for i in range(64)):
        try:
            # Normalize et
            s_norm = s_clean.replace(' ', '').lower()
            
            # Varsayılan değerler
            values = [0.0] * 64
            
            # e0 (skaler/reel kısım) için
            real_match = re.search(r'^([+-]?\d*\.?\d*)(?=e|$)', s_norm)
            if real_match and real_match.group(1) and real_match.group(1) not in ['', '+', '-']:
                try:
                    values[0] = float(real_match.group(1))
                except ValueError:
                    pass
            
            # Her e birimi için
            for i in range(64):
                unit = f'e{i}'
                pattern = r'([+-]?\d*\.?\d*)' + re.escape(unit)
                match = re.search(pattern, s_norm)
                if match:
                    coeff_str = match.group(1)
                    if coeff_str in ['', '+']:
                        values[i] = 1.0
                    elif coeff_str == '-':
                        values[i] = -1.0
                    else:
                        try:
                            values[i] = float(coeff_str)
                        except ValueError:
                            values[i] = 0.0
            
            return ChingonNumber(values)
            
        except Exception as e:
            # Sembolik parsing başarısız olursa, normal sayı olarak dene
            pass
    
    # 8. Sadece sayı
    try:
        scalar_val = float(s_clean)
        return ChingonNumber.from_scalar(scalar_val)
    except ValueError:
        pass
    
    # 9. Bilimsel gösterim
    if 'e' in s_clean.lower() and not s_clean.lower().startswith('e'):
        try:
            if re.match(r'^[+-]?\d*\.?\d+[eE][+-]?\d+$', s_clean):
                scalar_val = float(s_clean)
                return ChingonNumber.from_scalar(scalar_val)
        except ValueError:
            pass
    
    # 10. Hiçbir format uymuyorsa hata ver
    raise ValueError(
        f"Invalid Chingon format: '{s}'. "
        f"Expected: comma-separated 64 values, scalar, or symbolic form (e0-e63)."
    )

def _parse_routon(s: Any) -> RoutonNumber:
    """
    Parse input into a RoutonNumber (128-dimensional hypercomplex number).
    
    Supports:
      - RoutonNumber instance (returned as-is)
      - Numeric scalars (int, float) -> real part, others zero
      - Complex numbers -> real and imag in first two components
      - Lists/tuples of numbers (up to 128)
      - Strings: comma-separated list or single number
    
    Args:
        s: Input to parse
    
    Returns:
        RoutonNumber instance
    
    Raises:
        ValueError: If parsing fails
    """
    try:
        # If already RoutonNumber, return as-is
        if isinstance(s, RoutonNumber):
            return s
        
        # Handle numeric types
        if isinstance(s, (int, float)):
            return RoutonNumber.from_scalar(float(s))
        
        # Handle complex numbers
        if isinstance(s, complex):
            coeffs = [0.0] * 128
            coeffs[0] = s.real
            coeffs[1] = s.imag
            return RoutonNumber(coeffs)
        
        # Handle iterables (non-string)
        if hasattr(s, '__iter__') and not isinstance(s, str):
            # Convert to list and ensure it's exactly 128 elements
            coeffs = list(s)
            if len(coeffs) < 128:
                coeffs = coeffs + [0.0] * (128 - len(coeffs))
            elif len(coeffs) > 128:
                coeffs = coeffs[:128]
            return RoutonNumber(coeffs)
        
        # Convert to string for parsing
        if not isinstance(s, str):
            s = str(s)
        
        s = s.strip()
        
        # Remove brackets if present
        s = s.strip('[]{}()')
        
        # Check if empty
        if not s:
            return RoutonNumber.from_scalar(0.0)
        
        # Try to parse as comma-separated list
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            parts = [p for p in parts if p]  # Filter empty
            
            if not parts:
                return RoutonNumber.from_scalar(0.0)
            
            try:
                # Parse all parts as floats
                float_parts = [float(p) for p in parts]
                
                # Ensure exactly 128 components
                if len(float_parts) == 128:
                    return RoutonNumber(float_parts)
                elif len(float_parts) < 128:
                    padded = float_parts + [0.0] * (128 - len(float_parts))
                    return RoutonNumber(padded)
                else:  # len(float_parts) > 128
                    import warnings
                    warnings.warn(f"Routon input has {len(float_parts)} components, truncating to 128", RuntimeWarning)
                    return RoutonNumber(float_parts[:128])
            
            except ValueError as e:
                raise ValueError(f"Invalid numeric value in Routon string: '{s}' -> {e}")
        
        # Try to parse as single number
        try:
            return RoutonNumber.from_scalar(float(s))
        except ValueError:
            pass
        
        # Try to parse as complex number string
        try:
            c = complex(s)
            coeffs = [0.0] * 128
            coeffs[0] = c.real
            coeffs[1] = c.imag
            return RoutonNumber(coeffs)
        except ValueError:
            pass
        
        # Try to extract any numeric content
        try:
            # Use regex to find first number
            import re
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if match:
                scalar_val = float(match.group())
                return RoutonNumber.from_scalar(scalar_val)
        except Exception:
            pass
        
        # If all else fails
        raise ValueError(f"Cannot parse Routon from input: {repr(s)}")
    
    except Exception as e:
        # Log the error if logger is available
        if 'logger' in globals():
            logger.warning(f"Routon parsing failed for {repr(s)}: {e}")
        else:
            import warnings
            warnings.warn(f"Routon parsing failed for {repr(s)}: {e}", RuntimeWarning)
        
        # Return zero Routon as fallback
        return RoutonNumber.from_scalar(0.0)

def _parse_voudon(s: Any) -> VoudonNumber:
    """
    Parse input into a VoudonNumber (256-dimensional hypercomplex number).
    
    Supports:
      - VoudonNumber instance (returned as-is)
      - Numeric scalars (int, float) -> real part, others zero
      - Complex numbers -> real and imag in first two components
      - Lists/tuples of numbers (up to 256)
      - Strings: comma-separated list or single number
    
    Args:
        s: Input to parse
    
    Returns:
        VoudonNumber instance
    
    Raises:
        ValueError: If parsing fails
    """
    try:
        # If already VoudonNumber, return as-is
        if isinstance(s, VoudonNumber):
            return s
        
        # Handle numeric types
        if isinstance(s, (int, float)):
            return VoudonNumber.from_scalar(float(s))
        
        # Handle complex numbers
        if isinstance(s, complex):
            coeffs = [0.0] * 256
            coeffs[0] = s.real
            coeffs[1] = s.imag
            return VoudonNumber(coeffs)
        
        # Handle iterables (non-string)
        if hasattr(s, '__iter__') and not isinstance(s, str):
            return VoudonNumber.from_iterable(s)
        
        # Convert to string for parsing
        if not isinstance(s, str):
            s = str(s)
        
        s = s.strip()
        
        # Remove brackets if present
        s = s.strip('[]{}()')
        
        # Check if empty
        if not s:
            return VoudonNumber.from_scalar(0.0)
        
        # Try to parse as comma-separated list
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            
            # Filter out empty parts
            parts = [p for p in parts if p]
            
            if not parts:
                return VoudonNumber.from_scalar(0.0)
            
            try:
                # Parse all parts as floats
                float_parts = [float(p) for p in parts]
                
                # If we have exactly 256 components
                if len(float_parts) == 256:
                    return VoudonNumber(float_parts)
                
                # If we have fewer than 256, pad with zeros
                elif len(float_parts) < 256:
                    padded = float_parts + [0.0] * (256 - len(float_parts))
                    return VoudonNumber(padded)
                
                # If we have more than 256, truncate
                else:
                    warnings.warn(f"Voudon input has {len(float_parts)} components, "
                                f"truncating to first 256", RuntimeWarning)
                    return VoudonNumber(float_parts[:256])
            
            except ValueError as e:
                raise ValueError(f"Invalid numeric value in Voudon string: '{s}' -> {e}")
        
        # Try to parse as single number
        try:
            scalar_val = float(s)
            return VoudonNumber.from_scalar(scalar_val)
        except ValueError:
            pass
        
        # Try to parse as complex number string
        try:
            c = complex(s)
            coeffs = [0.0] * 256
            coeffs[0] = c.real
            coeffs[1] = c.imag
            return VoudonNumber(coeffs)
        except ValueError:
            pass
        
        # Try to extract any numeric content
        try:
            # Use regex to find first number
            import re
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if match:
                scalar_val = float(match.group())
                return VoudonNumber.from_scalar(scalar_val)
        except Exception:
            pass
        
        # If all else fails
        raise ValueError(f"Cannot parse Voudon from input: {repr(s)}")
    
    except Exception as e:
        # Log the error if logger is available
        if 'logger' in globals():
            logger.warning(f"Voudon parsing failed for {repr(s)}: {e}")
        else:
            warnings.warn(f"Voudon parsing failed for {repr(s)}: {e}", RuntimeWarning)
        
        # Return zero Voudon as fallback
        return VoudonNumber.from_scalar(0.0)


def _safe_float(value: Any) -> float:
    """
    Güvenli float dönüşümü.
    
    Args:
        value: Dönüştürülecek değer
        
    Returns:
        Float değeri
    """
    if isinstance(value, (float, int)):
        return float(value)
    elif isinstance(value, complex):
        return float(value.real)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            value_upper = value.upper().strip()
            if value_upper in ['', 'NAN', 'NULL', 'NONE']:
                return 0.0
            elif value_upper in ['INF', 'INFINITY']:
                return float('inf')
            elif value_upper in ['-INF', '-INFINITY']:
                return float('-inf')
            elif value == '+':
                return 1.0
            elif value == '-':
                return -1.0
            return 0.0
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


def _parse_clifford(s: Any) -> CliffordNumber:
    """
    Algebraik string'i CliffordNumber'a dönüştürür.
    
    Desteklenen formatlar:
    - "1.0 + 2.0e1 - 3.5e12"
    - "e1 + 2e2"
    - "3.14" (skaler)
    - "1,2,3,4" (virgülle ayrılmış, basis sırasına göre)
    
    Args:
        s: Girdi (string, sayı, veya CliffordNumber)
        
    Returns:
        CliffordNumber instance
        
    Raises:
        ValueError: Geçersiz format
    """
    # 1. Eğer zaten CliffordNumber ise doğrudan döndür
    if isinstance(s, CliffordNumber):
        return s
    
    # 2. Sayısal tipler için skaler dönüşüm
    if isinstance(s, (float, int)):
        try:
            scalar = float(s)
            return CliffordNumber({'': scalar})  # Skaler kısım için boş string key
        except Exception as e:
            raise ValueError(f"Cannot convert scalar {s} to CliffordNumber: {e}") from e
    
    elif isinstance(s, complex):
        try:
            # Karmaşık sayı: real -> skaler, imag -> e1
            return CliffordNumber({'': float(s.real), '1': float(s.imag)})
        except Exception as e:
            raise ValueError(f"Cannot convert complex {s} to CliffordNumber: {e}") from e
    
    # 3. String işlemleri
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception as e:
            raise ValueError(f"Cannot convert to string: {s}") from e
    
    s = s.strip()
    if not s:
        return CliffordNumber({})
    
    # 4. Virgülle ayrılmış format (basit alternatif)
    if ',' in s and 'e' not in s.lower():
        try:
            parts = [p.strip() for p in s.split(',')]
            basis_dict = {}
            for i, part in enumerate(parts):
                if part:
                    try:
                        value = float(part)
                        if i == 0:
                            basis_dict[''] = value  # Skaler kısım
                        else:
                            basis_dict[str(i)] = value  # Basis kısımları
                    except ValueError:
                        pass
            return CliffordNumber(basis_dict)
        except Exception:
            pass  # Diğer formatları dene
    
    # 5. Algebraik format parsing
    s_clean = s.replace(' ', '').replace('^', '').upper()
    
    # Basis dictionary
    basis_dict = {}
    
    # Pattern 1: ±coefficient e basis (ör: +1.23e12, -2.5e1)
    # Bu pattern bilimsel gösterim ile karışabilir, dikkatli ol
    pattern1 = r'([+-]?)(\d*\.?\d+)(?:E(\d+))'
    
    # Pattern 2: ±coefficient (skaler) (ör: +1.23, -2.5)
    pattern2 = r'([+-]?)(\d*\.?\d+)(?![E\d])'
    
    # Pattern 3: ±e basis (coefficient yok, 1 kabul et) (ör: +e1, -e12)
    pattern3 = r'([+-]?)E(\d+)'
    
    # Önce tüm eşleşmeleri bul
    matches1 = re.findall(pattern1, s_clean)
    matches2 = re.findall(pattern2, s_clean)
    matches3 = re.findall(pattern3, s_clean)
    
    # 1. Pattern işleme: coefficient e basis
    for sign_str, coeff_str, basis_key in matches1:
        if basis_key:  # Basis key boş değilse
            sign = -1.0 if sign_str == '-' else 1.0
            try:
                coeff = float(coeff_str) if coeff_str else 1.0
                value = sign * coeff
                
                # Basis key'i sırala (ör: '12' yerine '12' ama '21' değil)
                sorted_basis = ''.join(sorted(basis_key))
                basis_dict[sorted_basis] = basis_dict.get(sorted_basis, 0.0) + value
            except (ValueError, TypeError):
                pass
    
    # 2. Pattern işleme: skaler kısımlar
    for sign_str, coeff_str in matches2:
        # Bu pattern'in başka bir pattern'in parçası olmadığından emin ol
        context = re.search(r'[+-]?' + re.escape(coeff_str) + r'(?![E\d])', s_clean)
        if context:
            sign = -1.0 if sign_str == '-' else 1.0
            try:
                coeff = float(coeff_str) if coeff_str else 1.0
                value = sign * coeff
                basis_dict[''] = basis_dict.get('', 0.0) + value
            except (ValueError, TypeError):
                pass
    
    # 3. Pattern işleme: e basis (coefficient yok)
    for sign_str, basis_key in matches3:
        if basis_key:
            sign = -1.0 if sign_str == '-' else 1.0
            value = sign * 1.0
            
            # Basis key'i sırala
            sorted_basis = ''.join(sorted(basis_key))
            basis_dict[sorted_basis] = basis_dict.get(sorted_basis, 0.0) + value
    
    # 6. Alternatif parsing: daha basit regex
    if not basis_dict:
        # Daha basit pattern: terimleri ayır
        # Terimler: ±(coefficient)(e basis)?
        terms = re.findall(r'([+-]?\d*\.?\d*)(?:E(\d+))?', s_clean)
        
        for coeff_str, basis_key in terms:
            if coeff_str or basis_key:
                # İşareti belirle
                if coeff_str.startswith(('+', '-')):
                    sign_str = coeff_str[0]
                    coeff_str = coeff_str[1:]
                else:
                    sign_str = '+'
                
                sign = -1.0 if sign_str == '-' else 1.0
                
                # Coefficient
                if coeff_str:
                    try:
                        coeff = float(coeff_str)
                    except ValueError:
                        coeff = 1.0 if coeff_str != '' else 1.0
                else:
                    coeff = 1.0
                
                value = sign * coeff
                
                # Basis key
                if basis_key:
                    sorted_basis = ''.join(sorted(basis_key))
                    basis_dict[sorted_basis] = basis_dict.get(sorted_basis, 0.0) + value
                else:
                    # Skaler kısım
                    basis_dict[''] = basis_dict.get('', 0.0) + value
    
    # 7. Eğer hala boşsa, string'i direkt sayı olarak dene
    if not basis_dict:
        try:
            scalar = float(s)
            return CliffordNumber({'': scalar})
        except ValueError:
            # Boş CliffordNumber döndür
            return CliffordNumber({})
    
    return CliffordNumber(basis_dict)


def _parse_dual(s: Any) -> DualNumber:
    """
    String'i veya sayıyı DualNumber'a dönüştürür.
    
    Dual number format: a + bε, where ε² = 0
    
    Desteklenen formatlar:
    - "1.5,2.3" (virgülle ayrılmış)
    - "1.5 + 2.3ε" (sembolik)
    - "3.14" (skaler)
    - DualNumber instance
    
    Args:
        s: Girdi
        
    Returns:
        DualNumber instance
    """
    # 1. Eğer zaten DualNumber ise doğrudan döndür
    if isinstance(s, DualNumber):
        return s
    
    # 2. Sayısal tipler
    if isinstance(s, (float, int)):
        return DualNumber(_safe_float(s), 0.0)
    elif isinstance(s, complex):
        return DualNumber(_safe_float(s.real), _safe_float(s.imag))
    
    # 3. List/tuple
    if isinstance(s, (list, tuple)):
        if len(s) >= 2:
            return DualNumber(_safe_float(s[0]), _safe_float(s[1]))
        elif len(s) == 1:
            return DualNumber(_safe_float(s[0]), 0.0)
        else:
            return DualNumber(0.0, 0.0)
    
    # 4. String işlemleri
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return DualNumber(0.0, 0.0)
    
    s = s.strip()
    if not s:
        return DualNumber(0.0, 0.0)
    
    # 5. Virgülle ayrılmış format
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) >= 2:
            try:
                return DualNumber(_safe_float(parts[0]), _safe_float(parts[1]))
            except (ValueError, TypeError):
                pass
        elif len(parts) == 1:
            try:
                return DualNumber(_safe_float(parts[0]), 0.0)
            except (ValueError, TypeError):
                pass
    
    # 6. Sembolik format: a + bε
    # ε sembolü için farklı olasılıklar: ε, e, E, epsilon
    s_lower = s.lower().replace(' ', '')
    
    # Pattern: (±?number)(±?number)epsilon
    patterns = [
        r'([+-]?\d*\.?\d*)(?:([+-]?\d*\.?\d*)[εe]|epsilon)',
        r'([+-]?\d*\.?\d*)[εe]',
        r'([+-]?\d*\.?\d*)epsilon',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, s_lower)
        if match:
            groups = match.groups()
            if len(groups) >= 1:
                real_part = groups[0]
                dual_part = groups[1] if len(groups) > 1 else '0'
                
                real = _safe_float(real_part)
                dual = _safe_float(dual_part)
                return DualNumber(real, dual)
    
    # 7. Sadece sayı
    try:
        scalar = _safe_float(s)
        return DualNumber(scalar, 0.0)
    except (ValueError, TypeError):
        return DualNumber(0.0, 0.0)


def _parse_splitcomplex(s: Any) -> SplitcomplexNumber:
    """
    String'i veya sayıyı SplitcomplexNumber'a dönüştürür.
    
    Split-complex format: a + bj, where j² = 1
    
    Desteklenen formatlar:
    - "1.5,2.3" (virgülle ayrılmış)
    - "1.5 + 2.3j" (sembolik)
    - "3.14" (skaler)
    - SplitcomplexNumber instance
    
    Args:
        s: Girdi
        
    Returns:
        SplitcomplexNumber instance
    """
    # 1. Eğer zaten SplitcomplexNumber ise doğrudan döndür
    if isinstance(s, SplitcomplexNumber):
        return s
    
    # 2. Sayısal tipler
    if isinstance(s, (float, int)):
        return SplitcomplexNumber(_safe_float(s), 0.0)
    elif isinstance(s, complex):
        # Complex ile aynı formatta ama farklı algebra
        return SplitcomplexNumber(_safe_float(s.real), _safe_float(s.imag))
    
    # 3. List/tuple
    if isinstance(s, (list, tuple)):
        if len(s) >= 2:
            return SplitcomplexNumber(_safe_float(s[0]), _safe_float(s[1]))
        elif len(s) == 1:
            return SplitcomplexNumber(_safe_float(s[0]), 0.0)
        else:
            return SplitcomplexNumber(0.0, 0.0)
    
    # 4. String işlemleri
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return SplitcomplexNumber(0.0, 0.0)
    
    s = s.strip()
    if not s:
        return SplitcomplexNumber(0.0, 0.0)
    
    # 5. Virgülle ayrılmış format
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) >= 2:
            try:
                return SplitcomplexNumber(_safe_float(parts[0]), _safe_float(parts[1]))
            except (ValueError, TypeError):
                pass
        elif len(parts) == 1:
            try:
                return SplitcomplexNumber(_safe_float(parts[0]), 0.0)
            except (ValueError, TypeError):
                pass
    
    # 6. Sembolik format: a + bj
    # j sembolü için: j, J, split-j
    s_lower = s.lower().replace(' ', '')
    
    # Karmaşık sayı formatını kontrol et (j ile)
    if 'j' in s_lower:
        # Karmaşık sayı parsing'i
        try:
            # j'yi standart karmaşık j'ye çevir
            s_complex = s_lower.replace('j', 'j')
            # Python complex() ile parse et
            c = complex(s_complex)
            return SplitcomplexNumber(c.real, c.imag)
        except (ValueError, TypeError):
            pass
    
    # 7. Özel split-complex format: a + bj veya a + b*j
    pattern = r'([+-]?\d*\.?\d*)(?:([+-]?\d*\.?\d*)[Jj]|\*[Jj])'
    match = re.match(pattern, s_lower)
    if match:
        groups = match.groups()
        if groups[0]:
            real = _safe_float(groups[0])
            imag = _safe_float(groups[1]) if groups[1] else 0.0
            return SplitcomplexNumber(real, imag)
    
    # 8. Sadece sayı
    try:
        scalar = _safe_float(s)
        return SplitcomplexNumber(scalar, 0.0)
    except (ValueError, TypeError):
        return SplitcomplexNumber(0.0, 0.0)

def generate_octonion(w, x, y, z, e, f, g, h):
    """8 bileşenden bir oktonyon oluşturur."""
    return OctonionNumber(w, x, y, z, e, f, g, h)


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

def _parse_superreal(s) -> SuperrealNumber:
    """String'i veya sayıyı SuperrealNumber'a dönüştürür."""
    if isinstance(s, SuperrealNumber):
        return s

    if isinstance(s, (float, int)):
        return SuperrealNumber(float(s), 0.0)

    if isinstance(s, complex):
        return SuperrealNumber(s.real, s.imag)

    if hasattr(s, '__iter__') and not isinstance(s, str):
        if len(s) == 2:
            return SuperrealNumber(float(s[0]), float(s[1]))
        else:
            raise ValueError("SuperrealNumber için 2 bileşen gereklidir.")

    # String işlemleri
    if not isinstance(s, str):
        s = str(s)

    s = s.strip().strip('()[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 2:
        try:
            real = float(parts[0])
            split = float(parts[1])
            return SuperrealNumber(real, split)
        except ValueError as e:
            raise ValueError(f"Geçersiz SuperrealNumber bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:
        try:
            real = float(parts[0])
            return SuperrealNumber(real, 0.0)
        except ValueError as e:
            raise ValueError(f"Geçersiz SuperrealNumber skaler değeri: '{s}' -> {e}") from e
    else:
        raise ValueError("SuperrealNumber için 1 veya 2 bileşen gereklidir.")

def _parse_ternary(s) -> TernaryNumber:
    """String'i veya sayıyı TernaryNumber'a dönüştürür."""
    if isinstance(s, TernaryNumber):
        return s

    if isinstance(s, (float, int)):
        # Sayıyı üçlü sayı sistemine dönüştür (örneğin, 11 -> "102")
        # Burada basitçe skaler bir değer olarak işleniyor
        return TernaryNumber.from_decimal(int(s))

    if hasattr(s, '__iter__') and not isinstance(s, str):
        return TernaryNumber(list(s))

    # String işlemleri
    if not isinstance(s, str):
        s = str(s)

    s = s.strip().strip('()[]')
    # Üçlü sayı sistemindeki geçersiz karakterleri kontrol et
    if not all(c in '012' for c in s):
        raise ValueError(f"Geçersiz üçlü sayı formatı: '{s}'")

    return TernaryNumber.from_ternary_string(s)

def get_random_type(
    num_iterations: int = 10,
    fixed_start_raw: Union[str, float, int] = "0",
    fixed_add_base_scalar: Union[str, float, int] = 9.0,
    exclude_types: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> List[Any]:
    """
    Generates Keçeci Numbers for a randomly selected type.
    
    Args:
        num_iterations: Number of iterations to generate
        fixed_start_raw: Starting value (can be string, float, or int)
        fixed_add_base_scalar: Value to add each iteration (can be string, float, or int)
        exclude_types: List of type numbers to exclude from random selection
        seed: Random seed for reproducible results
        
    Returns:
        List of generated Keçeci numbers
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Type definitions
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutrosophic Complex", "Hyperreal", 
        "Bicomplex", "Neutrosophic Bicomplex", "Octonion", "Sedenion", 
        "Clifford", "Dual", "Split-Complex", "Pathion", "Chingon", 
        "Routon", "Voudon", "Super Real", "Ternary"
    ]
    
    # Define available types (1-based indexing)
    available_types = list(range(1, len(type_names_list) + 1))
    
    # Exclude specified types
    if exclude_types:
        available_types = [t for t in available_types if t not in exclude_types]
    
    if not available_types:
        raise ValueError("No available types after exclusions")
    
    # Randomly select a type
    random_type_choice = random.choice(available_types)
    
    # Log the selection
    logger.info(
        "Randomly selected Keçeci Number Type: %d (%s)", 
        random_type_choice, 
        type_names_list[random_type_choice - 1]
    )
    
    # Ensure parameters are strings for get_with_params
    start_value_str = str(fixed_start_raw) if not isinstance(fixed_start_raw, str) else fixed_start_raw
    add_value_str = str(fixed_add_base_scalar) if not isinstance(fixed_add_base_scalar, str) else fixed_add_base_scalar
    
    # Call the generator function
    return get_with_params(
        kececi_type_choice=random_type_choice, 
        iterations=num_iterations,
        start_value_raw=start_value_str,
        add_value_raw=add_value_str
    )


def get_with_params(
    kececi_type_choice: int,
    iterations: int = 10,
    start_value_raw: Union[str, float, int] = "0",
    add_value_raw: Union[str, float, int] = "1.0",
    operation: str = "add",
    include_intermediate_steps: bool = False,
    custom_parser: Optional[Callable] = None
) -> List[Any]:
    """
    Unified entry point for generating Keçeci numbers based on specified parameters.
    
    Args:
        kececi_type_choice: Type of number to generate (1-22)
        iterations: Number of iterations to generate
        start_value_raw: Starting value (string, float, or int)
        add_value_raw: Value to add/multiply each iteration (string, float, or int)
        operation: Operation to perform ('add', 'multiply', 'subtract', 'divide', 'mod', 'power')
        include_intermediate_steps: Whether to include intermediate calculation steps
        custom_parser: Optional custom parser function
        
    Returns:
        List of generated Keçeci numbers
        
    Raises:
        ValueError: Invalid type choice or operation
        TypeError: Invalid input types
    """
    # Log the start of generation
    logger.info("Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations)
    logger.debug("Start: %r, Operation: %r with value: %r, Include intermediate: %s", 
                start_value_raw, operation, add_value_raw, include_intermediate_steps)
    
    # Basic input sanitation and type conversion
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        add_value_raw = "1" if operation == "add" else "2" if operation == "multiply" else "1"
    
    # Convert to strings if they're not already
    if not isinstance(start_value_raw, str):
        start_value_raw = str(start_value_raw)
    if not isinstance(add_value_raw, str):
        add_value_raw = str(add_value_raw)
    
    # Validate operation
    valid_operations = ['add', 'multiply', 'subtract', 'divide', 'mod', 'power']
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")
    
    # Validate iterations
    if iterations < 1:
        logger.warning(f"Invalid iterations value: {iterations}, using default 10")
        iterations = 10
    
    try:
        # Import parsers (lazy import to avoid circular imports)
        from .kececinumbers import (
            _parse_complex, _parse_neutrosophic, _parse_hyperreal,
            _parse_quaternion, _parse_octonion, _parse_sedenion,
            _parse_clifford, _parse_dual, _parse_splitcomplex,
            _parse_pathion, _parse_chingon, _parse_bicomplex,
            _parse_neutrosophic_complex, _parse_neutrosophic_bicomplex,
            _parse_routon, _parse_voudon, _parse_super_real, _parse_ternary
        )
        
        # Map type choices to parsers and their operations
        type_to_parser = {
            1: {'parser': lambda s: float(s), 'name': 'Positive Real'},
            2: {'parser': lambda s: -float(s), 'name': 'Negative Real'},
            3: {'parser': _parse_complex, 'name': 'Complex'},
            4: {'parser': lambda s: float(s), 'name': 'Float'},
            5: {'parser': lambda s: float(s), 'name': 'Rational'},
            6: {'parser': _parse_quaternion, 'name': 'Quaternion'},
            7: {'parser': _parse_neutrosophic, 'name': 'Neutrosophic'},
            8: {'parser': _parse_neutrosophic_complex, 'name': 'Neutrosophic Complex'},
            9: {'parser': _parse_hyperreal, 'name': 'Hyperreal'},
            10: {'parser': _parse_bicomplex, 'name': 'Bicomplex'},
            11: {'parser': _parse_neutrosophic_bicomplex, 'name': 'Neutrosophic Bicomplex'},
            12: {'parser': _parse_octonion, 'name': 'Octonion'},
            13: {'parser': _parse_sedenion, 'name': 'Sedenion'},
            14: {'parser': _parse_clifford, 'name': 'Clifford'},
            15: {'parser': _parse_dual, 'name': 'Dual'},
            16: {'parser': _parse_splitcomplex, 'name': 'Split-Complex'},
            17: {'parser': _parse_pathion, 'name': 'Pathion'},
            18: {'parser': _parse_chingon, 'name': 'Chingon'},
            19: {'parser': _parse_routon, 'name': 'Routon'},
            20: {'parser': _parse_voudon, 'name': 'Voudon'},
            21: {'parser': _parse_super_real, 'name': 'Super Real'},
            22: {'parser': _parse_ternary, 'name': 'Ternary'},
        }
        
        # Add custom parser if provided
        if custom_parser is not None:
            logger.debug("Using custom parser provided by user")
            parser_func = cast(Callable[[Any], Any], custom_parser)
            type_name = "Custom"
        else:
            if kececi_type_choice not in type_to_parser:
                raise ValueError(f"Invalid type choice: {kececi_type_choice}. Must be 1-22")
            
            type_info = type_to_parser[kececi_type_choice]
            parser_func = cast(Callable[[Any], Any], type_info['parser'])
            type_name = cast(str, type_info['name'])
        
        logger.info(f"Generating {type_name} numbers (type {kececi_type_choice})")
        
        # Parse start and add values
        try:
            start_value = parser_func(start_value_raw)
            add_value = parser_func(add_value_raw)
            
            # Log parsed values
            logger.debug(f"Parsed start value: {repr(start_value)}")
            logger.debug(f"Parsed operation value: {repr(add_value)}")
            
        except Exception as e:
            raise ValueError(f"Failed to parse values. Start: '{start_value_raw}', Add: '{add_value_raw}'. Error: {e}")
        
        # Generate the sequence based on operation
        result = _generate_sequence(
            start_value=start_value,
            add_value=add_value,
            iterations=iterations,
            operation=operation,
            include_intermediate_steps=include_intermediate_steps
        )
        
        if not result:
            logger.warning("Sequence generation failed or returned empty")
            return []
        
        # Log generation results
        logger.info(f"Generated {len(result)} numbers for type {type_name}")
        
        # Preview first and last few elements
        preview_size = min(3, len(result))
        if preview_size > 0:
            preview_start = [str(x)[:50] + "..." if len(str(x)) > 50 else str(x) for x in result[:preview_size]]
            logger.debug(f"First {preview_size}: {preview_start}")
            
            if len(result) > preview_size * 2:
                preview_end = [str(x)[:50] + "..." if len(str(x)) > 50 else str(x) for x in result[-preview_size:]]
                logger.debug(f"Last {preview_size}: {preview_end}")
        
        # Optional: Keçeci Prime Number check (if applicable)
        try:
            kpn = _find_kececi_prime_number(result)
            if kpn is not None:
                logger.info(f"Keçeci Prime Number (KPN) found: {kpn}")
            else:
                logger.debug("No Keçeci Prime Number found in the sequence.")
        except Exception as e:
            logger.debug(f"KPN check skipped or failed: {e}")
        
        return result
        
    except Exception as e:
        logger.exception(f"ERROR during sequence generation: {e}")
        raise  # Re-raise the exception for caller handling


def _generate_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False
) -> List[Any]:
    """
    Generate a sequence based on the operation.
    
    Args:
        start_value: Starting value
        add_value: Value to use in operation
        iterations: Number of iterations
        operation: Operation to perform
        include_intermediate_steps: Whether to include intermediate steps
        
    Returns:
        List of generated values. If include_intermediate_steps=True, 
        returns a list of dictionaries with step information.
    """
    from typing import Dict, Union
    
    if include_intermediate_steps:
        # Detaylı log için dictionary listesi
        detailed_result: List[Dict[str, Any]] = []
        
        # Başlangıç değerini ekle
        detailed_result.append({
            'step': 0,
            'operation': 'start',
            'value': start_value,
            'description': f"Start: {start_value}"
        })
        
        current = start_value
        
        for i in range(1, iterations):
            try:
                previous = current
                
                # İşlemi gerçekleştir
                if operation == "add":
                    current = current + add_value
                    op_symbol = "+"
                elif operation == "multiply":
                    current = current * add_value
                    op_symbol = "×"
                elif operation == "subtract":
                    current = current - add_value
                    op_symbol = "-"
                elif operation == "divide":
                    current = _safe_divide(current, add_value)
                    op_symbol = "/"
                elif operation == "mod":
                    current = _safe_mod(current, add_value)
                    op_symbol = "%"
                elif operation == "power":
                    current = _safe_power(current, add_value)
                    op_symbol = "^"
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                # Detaylı log ekle
                detailed_result.append({
                    'step': i,
                    'operation': operation,
                    'previous': previous,
                    'value': current,
                    'description': f"Step {i}: {previous} {op_symbol} {add_value} = {current}"
                })
                
            except Exception as e:
                logger.warning(f"Error at iteration {i}: {e}")
                
                # Hata durumunda default değer
                try:
                    if hasattr(type(current), '__call__'):
                        default_val = type(current)()
                    elif isinstance(current, (int, float)):
                        default_val = 0
                    elif isinstance(current, complex):
                        default_val = complex(0, 0)
                    else:
                        default_val = 0
                    
                    detailed_result.append({
                        'step': i,
                        'operation': operation,
                        'error': str(e),
                        'value': default_val,
                        'description': f"Step {i}: ERROR - {e}"
                    })
                    
                    current = default_val
                except Exception as e2:
                    logger.error(f"Cannot continue after error at iteration {i}: {e2}")
                    break
        
        return detailed_result  # Dictionary listesi döndür
        
    else:
        # Basit liste (sadece değerler)
        simple_result: List[Any] = [start_value]
        current = start_value
        
        for i in range(1, iterations):
            try:
                # İşlemi gerçekleştir
                if operation == "add":
                    current = current + add_value
                elif operation == "multiply":
                    current = current * add_value
                elif operation == "subtract":
                    current = current - add_value
                elif operation == "divide":
                    current = _safe_divide(current, add_value)
                elif operation == "mod":
                    current = _safe_mod(current, add_value)
                elif operation == "power":
                    current = _safe_power(current, add_value)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                simple_result.append(current)
                
            except Exception as e:
                logger.warning(f"Error at iteration {i}: {e}")
                
                # Hata durumunda default değer
                try:
                    if hasattr(type(current), '__call__'):
                        default_val = type(current)()
                    elif isinstance(current, (int, float)):
                        default_val = 0
                    elif isinstance(current, complex):
                        default_val = complex(0, 0)
                    else:
                        default_val = 0
                    
                    simple_result.append(default_val)
                    current = default_val
                except Exception as e2:
                    logger.error(f"Cannot continue after error at iteration {i}: {e2}")
                    break
        
        return simple_result  # Basit liste döndür


# Daha basit ve güvenli versiyon (alternatif)
def generate_sequence_safe(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False
) -> Union[List[Any], List[Dict[str, Any]]]:
    """
    Safer version with separate return types.
    """
    if include_intermediate_steps:
        return _generate_detailed_sequence(start_value, add_value, iterations, operation)
    else:
        return _generate_simple_sequence(start_value, add_value, iterations, operation)


def _generate_detailed_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str
) -> List[Dict[str, Any]]:
    """Generate detailed sequence with step information."""
    result: List[Dict[str, Any]] = []
    
    result.append({
        'step': 0,
        'operation': 'start',
        'value': start_value,
        'description': f"Start: {start_value}"
    })
    
    current = start_value
    
    for i in range(1, iterations):
        try:
            previous = current
            
            if operation == "add":
                current = current + add_value
                op_symbol = "+"
            elif operation == "multiply":
                current = current * add_value
                op_symbol = "×"
            elif operation == "subtract":
                current = current - add_value
                op_symbol = "-"
            elif operation == "divide":
                current = _safe_divide(current, add_value)
                op_symbol = "/"
            elif operation == "mod":
                current = _safe_mod(current, add_value)
                op_symbol = "%"
            elif operation == "power":
                current = _safe_power(current, add_value)
                op_symbol = "^"
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            result.append({
                'step': i,
                'operation': operation,
                'previous': previous,
                'value': current,
                'description': f"Step {i}: {previous} {op_symbol} {add_value} = {current}"
            })
            
        except Exception as e:
            logger.warning(f"Error at iteration {i}: {e}")
            
            # Create appropriate default value
            default_val = _generate_default_value(current)
            
            result.append({
                'step': i,
                'operation': operation,
                'error': str(e),
                'value': default_val,
                'description': f"Step {i}: ERROR - {e}"
            })
            
            current = default_val
    
    return result


def _generate_simple_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str
) -> List[Any]:
    """Generate simple sequence of values."""
    result: List[Any] = [start_value]
    current = start_value
    
    for i in range(1, iterations):
        try:
            if operation == "add":
                current = current + add_value
            elif operation == "multiply":
                current = current * add_value
            elif operation == "subtract":
                current = current - add_value
            elif operation == "divide":
                current = _safe_divide(current, add_value)
            elif operation == "mod":
                current = _safe_mod(current, add_value)
            elif operation == "power":
                current = _safe_power(current, add_value)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            result.append(current)
            
        except Exception as e:
            logger.warning(f"Error at iteration {i}: {e}")
            
            # Create appropriate default value
            default_val = _generate_default_value(current)
            result.append(default_val)
            current = default_val
    
    return result


def _generate_default_value(current_value: Any) -> Any:
    """Create appropriate default value based on current value type."""
    try:
        if hasattr(type(current_value), '__call__'):
            return type(current_value)()
        elif isinstance(current_value, (int, float)):
            return 0
        elif isinstance(current_value, complex):
            return complex(0, 0)
        elif isinstance(current_value, str):
            return ""
        elif isinstance(current_value, (list, tuple)):
            return type(current_value)()
        else:
            # Try to get real part if exists
            try:
                if hasattr(current_value, 'real'):
                    return type(current_value)(0)
            except:
                pass
            
            return 0
    except Exception:
        return 0


# Grafik çizimi için yardımcı fonksiyon
def extract_values_for_plotting(sequence: List[Any]) -> List[float]:
    """
    Extract numeric values from sequence for plotting.
    
    Args:
        sequence: Sequence generated by _generate_sequence
        
    Returns:
        List of float values suitable for plotting
    """
    values: List[float] = []
    
    for item in sequence:
        try:
            if isinstance(item, dict):
                # Dictionary'den 'value' anahtarını al
                value = item.get('value', 0)
            else:
                value = item
            
            # Float'a çevirmeye çalış
            if isinstance(value, (int, float, complex)):
                if isinstance(value, complex):
                    # Complex için magnitude
                    values.append(abs(value))
                else:
                    values.append(float(value))
            elif hasattr(value, 'real'):
                # real attribute'u olan nesneler için
                values.append(float(value.real))
            else:
                # String veya diğer tipler için
                try:
                    values.append(float(str(value)))
                except (ValueError, TypeError):
                    values.append(0.0)
                    
        except Exception as e:
            logger.debug(f"Error extracting value for plotting: {e}")
            values.append(0.0)
    
    return values

def _safe_divide(a: Any, b: Any) -> Any:
    """Safe division with zero handling."""
    try:
        # Check if b is effectively zero
        if hasattr(b, '__abs__'):
            if abs(b) < 1e-12:  # Near zero threshold
                # Return infinity or handle based on type
                if hasattr(a, '__mul__'):
                    try:
                        return a * float('inf')
                    except:
                        pass
                return type(a)(float('inf')) if hasattr(type(a), '__call__') else float('inf')
        
        return a / b
    except ZeroDivisionError:
        return type(a)(float('inf')) if hasattr(type(a), '__call__') else float('inf')
    except Exception as e:
        logger.warning(f"Division error: {e}")
        return a  # Return original value on error


def _safe_mod(a: Any, b: Any) -> Any:
    """Safe modulo operation."""
    try:
        # Check if b is effectively zero
        if hasattr(b, '__abs__') and abs(b) < 1e-12:
            logger.warning("Modulo by near-zero value")
            return a  # Return original
        
        return a % b
    except Exception as e:
        logger.warning(f"Modulo error: {e}")
        return a  # Return original value on error


def _safe_power(a: Any, b: Any) -> Any:
    """Safe power operation."""
    try:
        # For complex cases or special number types
        if hasattr(a, '__pow__'):
            return a ** b
        else:
            # Fallback for basic types
            return math.pow(float(a), float(b))
    except Exception as e:
        logger.warning(f"Power operation error: {e}")
        return a  # Return original value on error


def _find_kececi_prime_number(sequence: List[Any]) -> Optional[Any]:
    """
    Find a Keçeci Prime Number in the sequence.
    This is a placeholder implementation - customize based on your definition.
    
    Args:
        sequence: List of generated numbers
        
    Returns:
        The first Keçeci Prime Number found, or None
    """
    if not sequence:
        return None
    
    # Placeholder: Look for numbers with special properties
    # This should be customized based on your specific definition of KPN
    for num in sequence:
        try:
            # Example: Check if magnitude is prime (for complex-like numbers)
            if hasattr(num, 'magnitude'):
                mag = float(num.magnitude())
                if mag > 1 and all(mag % i != 0 for i in range(2, int(mag**0.5) + 1)):
                    return num
            
            # Example: Check real part for floats
            elif isinstance(num, (int, float)):
                if num > 1 and all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                    return num
            
            # Add more checks for other number types...
            
        except Exception:
            continue
    
    return None


def get_random_types_batch(
    num_types: int = 5,
    iterations_per_type: int = 5,
    start_value_raw: Union[str, float, int] = "0",
    add_value_raw: Union[str, float, int] = "1.0",
    seed: Optional[int] = None
) -> dict:
    """
    Generate multiple random types in one batch.
    
    Args:
        num_types: Number of different types to generate
        iterations_per_type: Iterations per type
        start_value_raw: Starting value
        add_value_raw: Value to add
        seed: Random seed
        
    Returns:
        Dictionary with type names as keys and lists as values
    """
    if seed is not None:
        random.seed(seed)
    
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutrosophic Complex", "Hyperreal", 
        "Bicomplex", "Neutrosophic Bicomplex", "Octonion", "Sedenion", 
        "Clifford", "Dual", "Split-Complex", "Pathion", "Chingon", 
        "Routon", "Voudon", "Super Real", "Ternary"
    ]
    
    # Select random types without replacement
    available_types = list(range(1, len(type_names_list) + 1))
    selected_types = random.sample(available_types, min(num_types, len(available_types)))
    
    results = {}
    
    for type_choice in selected_types:
        type_name = type_names_list[type_choice - 1]
        try:
            numbers = get_with_params(
                kececi_type_choice=type_choice,
                iterations=iterations_per_type,
                start_value_raw=str(start_value_raw),
                add_value_raw=str(add_value_raw)
            )
            results[type_name] = numbers
        except Exception as e:
            logger.error(f"Failed to generate type {type_name}: {e}")
            results[type_name] = []
    
    return results

def _get_integer_representation(n_input: Any) -> Optional[int]:
    """
    Extracts the primary integer component from supported Keçeci number types.

    Returns:
        absolute integer value (int) when a meaningful integer representation exists,
        otherwise None.
    """
    try:
        # None early exit
        if n_input is None:
            return None

        # Direct ints (including numpy ints)
        if isinstance(n_input, (int, np.integer)):
            return abs(int(n_input))

        # Fractions: only return if it's an integer fraction (denominator == 1)
        if isinstance(n_input, Fraction):
            if n_input.denominator == 1:
                return abs(int(n_input.numerator))
            return None

        # Floats: accept only if near integer
        if isinstance(n_input, (float, np.floating)):
            if is_near_integer(n_input):
                return abs(int(round(float(n_input))))
            return None

        # Complex: require imag ≈ 0 and real near-integer
        if isinstance(n_input, complex):
            if abs(n_input.imag) < 1e-12 and is_near_integer(n_input.real):
                return abs(int(round(n_input.real)))
            return None

        # numpy-quaternion or other quaternion types where 'w' is scalar part
        try:
            # `quaternion` type from numpy-quaternion has attribute 'w'
            #if isinstance(n_input, quaternion):
            #    w = getattr(n_input, 'w', None)
            #    if w is not None and is_near_integer(w):
            #        return abs(int(round(float(w)))
            if quaternion is not None and isinstance(n_input, quaternion):
                if is_near_integer(n_input.w):
                    return abs(int(round(n_input.w)))
                return None
        except Exception:
            # If quaternion type is not available or isinstance check fails, continue
            pass

        # If object exposes 'coeffs' (list/np.array), use first component
        if hasattr(n_input, 'coeffs'):
            coeffs = getattr(n_input, 'coeffs')
            # numpy array
            if isinstance(coeffs, np.ndarray):
                if coeffs.size > 0 and is_near_integer(coeffs.flatten()[0]):
                    return abs(int(round(float(coeffs.flatten()[0]))))
                return None
            # list/tuple-like
            try:
                # convert to list (works for many iterables)
                c0 = list(coeffs)[0]
                if is_near_integer(c0):
                    return abs(int(round(float(c0))))
                return None
            except Exception:
                # can't iterate coeffs reliably
                pass

        # Some classes expose 'coefficients' name instead
        if hasattr(n_input, 'coefficients'):
            try:
                c0 = list(getattr(n_input, 'coefficients'))[0]
                if is_near_integer(c0):
                    return abs(int(round(float(c0))))
            except Exception:
                pass

        # Try common scalar attributes in order of likelihood
        for attr in ('w', 'real', 't', 'a', 'value'):
            if hasattr(n_input, attr):
                val = getattr(n_input, attr)
                # If this is complex-like, use real part
                if isinstance(val, complex):
                    if abs(val.imag) < 1e-12 and is_near_integer(val.real):
                        return abs(int(round(val.real)))
                else:
                    try:
                        if is_near_integer(val):
                            return abs(int(round(float(val))))
                    except Exception:
                        pass

        # CliffordNumber: check basis dict scalar part ''
        if hasattr(n_input, 'basis') and isinstance(getattr(n_input, 'basis'), dict):
            scalar = n_input.basis.get('', 0)
            try:
                if is_near_integer(scalar):
                    return abs(int(round(float(scalar))))
            except Exception:
                pass

        # DualNumber / Superreal / others: if they expose .real attribute (and it's numeric)
        if hasattr(n_input, 'real') and not isinstance(n_input, (complex, float, int, np.floating, np.integer)):
            try:
                real_val = getattr(n_input, 'real')
                if is_near_integer(real_val):
                    return abs(int(round(float(real_val))))
            except Exception:
                pass

        # TernaryNumber: convert digits to decimal
        if hasattr(n_input, 'digits'):
            try:
                digits = list(n_input.digits)
                decimal_value = 0
                for i, d in enumerate(reversed(digits)):
                    decimal_value += int(d) * (3 ** i)
                return abs(int(decimal_value))
            except Exception:
                pass

        # HyperrealNumber: use finite part (sequence[0]) if present
        if hasattr(n_input, 'sequence') and isinstance(getattr(n_input, 'sequence'), (list, tuple)):
            seq = getattr(n_input, 'sequence')
            if seq:
                try:
                    if is_near_integer(seq[0]):
                        return abs(int(round(float(seq[0]))))
                except Exception:
                    pass

        # Fallback: try numeric coercion + is_near_integer
        try:
            if is_near_integer(n_input):
                return abs(int(round(float(n_input))))
        except Exception:
            pass

        # If nothing matched, return None
        return None

    except Exception:
        # On any unexpected failure, return None rather than raising
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

def print_detailed_report(sequence: List[Any], params: Dict[str, Any], show_all: bool = False) -> None:
    """Generates and logs a detailed report of the sequence results.

    Args:
        sequence: generated sequence (list)
        params: dict of parameters used to generate the sequence
        show_all: if True, include full sequence in the log; otherwise only preview
    """
    if not sequence:
        logger.info("--- REPORT ---\nSequence could not be generated.")
        return

    logger.info("\n" + "="*50)
    logger.info("--- DETAILED SEQUENCE REPORT ---")
    logger.info("="*50)

    logger.info("[Parameters Used]")
    logger.info("  - Keçeci Type:   %s (%s)", params.get('type_name', 'N/A'), params.get('type_choice'))
    logger.info("  - Start Value:   %r", params.get('start_val'))
    logger.info("  - Increment:     %r", params.get('add_val'))
    logger.info("  - Keçeci Steps:  %s", params.get('steps'))

    logger.info("[Sequence Summary]")
    logger.info("  - Total Numbers Generated: %d", len(sequence))

    kpn = find_kececi_prime_number(sequence)
    logger.info("  - Keçeci Prime Number (KPN): %s", kpn if kpn is not None else "Not found")

    preview_count = min(len(sequence), 40)
    logger.info("  --- First %d Numbers ---", preview_count)
    for i in range(preview_count):
        logger.info("    %d: %s", i, sequence[i])

    if len(sequence) > preview_count:
        logger.info("  --- Last %d Numbers ---", preview_count)
        for i in range(len(sequence) - preview_count, len(sequence)):
            logger.info("    %d: %s", i, sequence[i])

    if show_all:
        logger.info("--- FULL SEQUENCE ---")
        for i, num in enumerate(sequence):
            logger.info("%d: %s", i, num)
        logger.info("="*50)

def _is_divisible(value: Any, divisor: int, kececi_type: int) -> bool:
    """
    Robust divisibility check across Keçeci types.

    Returns True if value is divisible by divisor according to the semantics
    of the given kececi_type, otherwise False.

    Args:
        value: The value to check for divisibility
        divisor: The divisor (must be non-zero)
        kececi_type: Type identifier (1-22)

    Returns:
        bool: True if divisible, False otherwise
    """
    TOLERANCE = 1e-12

    if divisor == 0:
        warnings.warn("Divisor cannot be zero", RuntimeWarning)
        return False

    def _float_mod_zero(x: Any, divisor_arg: int, tol: float = TOLERANCE) -> bool:
        """
        Returns True if float(x) % divisor ≈ 0 within tolerance.
        Safe against exceptions (returns False on error).
        """
        try:
            # Convert to float
            float_val = float(x)
            # Calculate remainder
            remainder = float_val % divisor_arg
            # Check if remainder is close to 0 or close to divisor (due to floating point)
            return (math.isclose(remainder, 0.0, abs_tol=tol) or 
                    math.isclose(remainder, float(divisor_arg), abs_tol=tol))
        except Exception:
            return False

    def _int_mod_zero(x: Union[int, float], divisor_arg: int) -> bool:
        """Check if integer is divisible by divisor."""
        try:
            # Convert numpy integer to Python int
            if hasattr(x, 'item'):  # numpy scalar
                x = x.item()
            # Round to nearest integer
            int_val = int(round(float(x)))
            return int_val % divisor_arg == 0
        except Exception:
            return False

    def _fraction_mod_zero(fr: Fraction, divisor_arg: int) -> bool:
        """Check if fraction is divisible by divisor."""
        try:
            # fr = n/d, divisible by divisor iff n % (d * divisor) == 0
            return fr.numerator % (fr.denominator * divisor_arg) == 0
        except Exception:
            return False

    def _complex_mod_zero(c: complex, divisor_arg: int) -> bool:
        """Check if complex number is divisible by divisor."""
        # Both real and imaginary parts must be divisible
        return (_float_mod_zero(c.real, divisor_arg) and 
                _float_mod_zero(c.imag, divisor_arg))

    def _iterable_mod_zero(iterable: Any, divisor_arg: int) -> bool:
        """Check if all elements in iterable are divisible."""
        try:
            for item in iterable:
                # Handle different types
                if isinstance(item, Fraction):
                    if not _fraction_mod_zero(item, divisor_arg):
                        return False
                elif isinstance(item, complex):
                    if not _complex_mod_zero(item, divisor_arg):
                        return False
                elif isinstance(item, (int, np.integer)):
                    # Convert numpy integer to int for _int_mod_zero
                    if hasattr(item, 'item'):  # numpy scalar
                        item = item.item()
                    if not _int_mod_zero(int(item), divisor_arg):
                        return False
                else:
                    # Try float conversion
                    if not _float_mod_zero(item, divisor_arg):
                        return False
            return True
        except Exception:
            return False

    def is_near_integer(x: float, tol: float = TOLERANCE) -> bool:
        """Check if float is close to an integer."""
        return math.isclose(x, round(x), abs_tol=tol)

    # Helper for complex parsing (placeholder - implement as needed)
    def _parse_complex(s: Any) -> complex:
        """Parse complex number from various formats."""
        if isinstance(s, complex):
            return s
        elif isinstance(s, (int, float)):
            return complex(s, 0)
        elif isinstance(s, str):
            try:
                return complex(s.replace('i', 'j').replace('I', 'j'))
            except ValueError:
                return complex(0, 0)
        else:
            try:
                return complex(s)
            except:
                return complex(0, 0)

    try:
        # ===== DUAL NUMBER =====
        if kececi_type == TYPE_DUAL:
            if hasattr(value, 'real') and hasattr(value, 'dual'):
                return (_float_mod_zero(value.real, divisor) and 
                        _float_mod_zero(value.dual, divisor))
            # Fallback: element-wise if iterable
            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return _iterable_mod_zero(value, divisor)
            return False

        # ===== POSITIVE/NEGATIVE REAL =====
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            # Only accept near-integers
            if isinstance(value, (int, np.integer)):
                # Convert numpy integer to int
                if hasattr(value, 'item'):  # numpy scalar
                    value = value.item()
                return _int_mod_zero(int(value), divisor)
            if isinstance(value, Fraction):
                return _fraction_mod_zero(value, divisor)
            try:
                val = float(value)
                if is_near_integer(val):
                    return _int_mod_zero(val, divisor)
            except Exception:
                pass
            return False

        # ===== FLOAT =====
        if kececi_type == TYPE_FLOAT:
            try:
                return _float_mod_zero(float(value), divisor)
            except Exception:
                return False

        # ===== RATIONAL (FRACTION) =====
        if kececi_type == TYPE_RATIONAL:
            if isinstance(value, Fraction):
                return _fraction_mod_zero(value, divisor)
            # Try to coerce to Fraction
            try:
                fr = Fraction(str(value))
                return _fraction_mod_zero(fr, divisor)
            except Exception:
                return False

        # ===== COMPLEX =====
        if kececi_type == TYPE_COMPLEX:
            try:
                if isinstance(value, complex):
                    c = value
                else:
                    c = _parse_complex(value)
                return _complex_mod_zero(c, divisor)
            except Exception:
                return False

        # ===== QUATERNION =====
        if kececi_type == TYPE_QUATERNION:
            try:
                # Try numpy quaternion
                try:
                    from .kececinumbers import quaternion
                    if isinstance(value, quaternion.quaternion):
                        comps = [value.w, value.x, value.y, value.z]
                        return all(_float_mod_zero(c, divisor) for c in comps)
                except ImportError:
                    pass
                
                # Check for quaternion attributes
                if hasattr(value, 'w') and hasattr(value, 'x'):
                    components = [getattr(value, attr, 0.0) 
                                for attr in ('w', 'x', 'y', 'z')]
                    return all(_float_mod_zero(c, divisor) for c in components)
                
                # Fallback: iterable
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    return _iterable_mod_zero(value, divisor)
                    
            except Exception:
                return False
            return False

        # ===== NEUTROSOPHIC =====
        if kececi_type == TYPE_NEUTROSOPHIC:
            try:
                # Check for t,i,f attributes
                if hasattr(value, 't') and hasattr(value, 'i') and hasattr(value, 'f'):
                    return (_float_mod_zero(value.t, divisor) and 
                            _float_mod_zero(value.i, divisor) and 
                            _float_mod_zero(value.f, divisor))
                # Alternative attribute names
                if hasattr(value, 'a') and hasattr(value, 'b'):
                    return (_float_mod_zero(value.a, divisor) and 
                            _float_mod_zero(value.b, divisor))
            except Exception:
                return False
            return False

        # ===== NEUTROSOPHIC COMPLEX =====
        if kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            try:
                comps = []
                if hasattr(value, 'real'):
                    comps.append(value.real)
                if hasattr(value, 'imag'):
                    comps.append(value.imag)
                if hasattr(value, 'indeterminacy'):
                    comps.append(value.indeterminacy)
                
                return (all(_float_mod_zero(c, divisor) for c in comps) 
                        if comps else False)
            except Exception:
                return False

        # ===== HYPERREAL =====
        if kececi_type == TYPE_HYPERREAL:
            if hasattr(value, 'sequence') and isinstance(getattr(value, 'sequence'), (list, tuple)):
                return all(_float_mod_zero(x, divisor) for x in value.sequence)
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return _iterable_mod_zero(value, divisor)
            return False

        # ===== BICOMPLEX =====
        if kececi_type == TYPE_BICOMPLEX:
            try:
                if hasattr(value, 'z1') and hasattr(value, 'z2'):
                    return (_complex_mod_zero(value.z1, divisor) and 
                            _complex_mod_zero(value.z2, divisor))
            except Exception:
                return False
            return False

        # ===== NEUTROSOPHIC BICOMPLEX =====
        if kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            try:
                # Try common attribute names
                attrs_to_check = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                comps = [getattr(value, attr) for attr in attrs_to_check 
                        if hasattr(value, attr)]
                return (all(_float_mod_zero(c, divisor) for c in comps) 
                        if comps else False)
            except Exception:
                return False

        # ===== HYPERCOMPLEX TYPES (Octonion, Sedenion, etc.) =====
        hypercomplex_types = [
            TYPE_OCTONION, TYPE_SEDENION, TYPE_PATHION, 
            TYPE_CHINGON, TYPE_ROUTON, TYPE_VOUDON
        ]
        
        if kececi_type in hypercomplex_types:
            try:
                # Check for coeffs attribute
                if hasattr(value, 'coeffs'):
                    coeffs = getattr(value, 'coeffs')
                    if hasattr(coeffs, '__iter__'):
                        return _iterable_mod_zero(coeffs, divisor)
                
                # Fallback: check if iterable
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    return _iterable_mod_zero(value, divisor)
                
                # Single scalar fallback
                return _float_mod_zero(value, divisor)
                    
            except Exception:
                return False

        # ===== CLIFFORD =====
        if kececi_type == TYPE_CLIFFORD:
            try:
                # Check for basis dictionary
                if hasattr(value, 'basis_dict') and isinstance(getattr(value, 'basis_dict'), dict):
                    return all(_float_mod_zero(v, divisor) 
                            for v in value.basis_dict.values())
                elif hasattr(value, 'basis') and isinstance(getattr(value, 'basis'), dict):
                    return all(_float_mod_zero(v, divisor) 
                            for v in value.basis.values())
            except Exception:
                return False
            return False

        # ===== SPLIT-COMPLEX =====
        if kececi_type == TYPE_SPLIT_COMPLEX:
            try:
                if hasattr(value, 'real') and hasattr(value, 'imag'):
                    return (_float_mod_zero(value.real, divisor) and 
                            _float_mod_zero(value.imag, divisor))
                elif hasattr(value, 'real') and hasattr(value, 'split'):
                    return (_float_mod_zero(value.real, divisor) and 
                            _float_mod_zero(value.split, divisor))
            except Exception:
                return False
            return False

        # ===== SUPERREAL =====
        if kececi_type == TYPE_SUPERREAL:
            try:
                if hasattr(value, 'real') and hasattr(value, 'super'):
                    return (_float_mod_zero(value.real, divisor) and 
                            _float_mod_zero(value.super, divisor))
                elif hasattr(value, 'real') and hasattr(value, 'split'):
                    return (_float_mod_zero(value.real, divisor) and 
                            _float_mod_zero(value.split, divisor))
            except Exception:
                return False
            return False

        # ===== TERNARY =====
        if kececi_type == TYPE_TERNARY:
            try:
                # Convert ternary to decimal for divisibility check
                decimal_value = 0
                
                # Check for conversion method
                if hasattr(value, 'to_decimal'):
                    decimal_value = value.to_decimal()
                # Check for digits attribute
                elif hasattr(value, 'digits'):
                    digits = list(value.digits)
                    for i, digit in enumerate(reversed(digits)):
                        decimal_value += int(digit) * (3 ** i)
                # Check if iterable
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    digits = list(value)
                    for i, digit in enumerate(reversed(digits)):
                        decimal_value += int(digit) * (3 ** i)
                else:
                    # Try to convert directly
                    decimal_value = int(str(value))
                
                return _int_mod_zero(decimal_value, divisor)
                
            except Exception:
                return False

        # ===== FALLBACK STRATEGIES =====
        # Try common patterns before final numeric conversion
        
        # 1. Check for coeffs attribute
        if hasattr(value, 'coeffs'):
            coeffs = getattr(value, 'coeffs')
            if hasattr(coeffs, '__iter__'):
                return _iterable_mod_zero(coeffs, divisor)
        
        # 2. Check if iterable
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            return _iterable_mod_zero(value, divisor)
        
        # 3. Check for real/imag attributes (common for many number types)
        if hasattr(value, 'real') and hasattr(value, 'imag'):
            return (_float_mod_zero(value.real, divisor) and 
                    _float_mod_zero(value.imag, divisor))
        
        # 4. Check for scalar attributes
        scalar_attrs = ['value', 'scalar', 'magnitude', 'norm']
        for attr in scalar_attrs:
            if hasattr(value, attr):
                try:
                    return _float_mod_zero(getattr(value, attr), divisor)
                except Exception:
                    continue
        
        # ===== FINAL ATTEMPT: NUMERIC CONVERSION =====
        try:
            return _float_mod_zero(float(value), divisor)
        except Exception:
            return False

    except (TypeError, AttributeError, ValueError, ZeroDivisionError) as e:
        warnings.warn(f"Error in divisibility check: {e}", RuntimeWarning)
        return False

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

def _float_mod_zero(x: Any, divisor: int, tol: float = 1e-12) -> bool:
    """
    Check if float value is divisible by divisor within tolerance.
    
    Args:
        x: Value to check
        divisor: Divisor
        tol: Tolerance
        
    Returns:
        True if divisible, False otherwise
    """
    try:
        # Convert to float
        float_val = float(x)
        # Calculate remainder
        remainder = float_val % divisor
        # Check if remainder is close to 0 or close to divisor
        return (math.isclose(remainder, 0.0, abs_tol=tol) or 
                math.isclose(remainder, float(divisor), abs_tol=tol))
    except Exception:
        return False

def is_prime_like(value: Any, kececi_type: int) -> bool:
    """
    Heuristic to check whether `value` should be treated as a "prime candidate"
    under Keçeci logic for the given `kececi_type`.

    Strategy:
    - Prefer using _get_integer_representation when possible.
    - For quaternion/hypercomplex types require that component(s) are near-integers.
    - For ternary convert to decimal first.
    
    Args:
        value: Value to check
        kececi_type: Type identifier
        
    Returns:
        True if prime-like, False otherwise
    """
    try:
        # First, try a general integer extraction
        n = _get_integer_representation(value)
        if n is not None:
            # Check if it's a positive integer > 1
            if n <= 1:
                return False
            return sympy.isprime(int(n))

        # Handle quaternion specifically: require all components near-integer then test scalar
        if kececi_type == TYPE_QUATERNION:
            try:
                comps = []
                
                # Try numpy quaternion
                try:
                    from .kececinumbers import quaternion
                    if isinstance(value, quaternion.quaternion):
                        comps = [value.w, value.x, value.y, value.z]
                except ImportError:
                    pass
                
                # If not numpy quaternion, try common attribute patterns
                if not comps:
                    if hasattr(value, 'w') and hasattr(value, 'x'):
                        comps = [getattr(value, attr, 0.0) 
                                for attr in ('w', 'x', 'y', 'z')]
                    elif hasattr(value, 'coeffs'):
                        coeffs = getattr(value, 'coeffs')
                        if hasattr(coeffs, '__iter__'):
                            comps = list(coeffs)[:4]  # Take first 4 for quaternion
                
                if not comps:
                    return False
                
                # Check all components are near-integer
                if not all(is_near_integer(float(c)) for c in comps):
                    return False
                
                # Test the first (scalar) component for primality
                first_comp = float(comps[0])
                if not is_near_integer(first_comp):
                    return False
                    
                int_val = int(round(first_comp))
                if int_val <= 1:
                    return False
                    
                return sympy.isprime(int_val)
                
            except Exception:
                return False

        # Hypercomplex families: check coeffs exist and are integer-like, test first (scalar) component
        hypercomplex_types = [
            TYPE_OCTONION, TYPE_SEDENION, TYPE_PATHION, 
            TYPE_CHINGON, TYPE_ROUTON, TYPE_VOUDON
        ]
        
        if kececi_type in hypercomplex_types:
            try:
                coeffs = []
                
                if hasattr(value, 'coeffs'):
                    coeffs_attr = getattr(value, 'coeffs')
                    if hasattr(coeffs_attr, '__iter__'):
                        coeffs = list(coeffs_attr)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    coeffs = list(value)
                
                if not coeffs:
                    return False
                
                # Check all coefficients are near-integer
                if not all(is_near_integer(float(c)) for c in coeffs):
                    return False
                
                # Test the first (scalar) component for primality
                first_coeff = float(coeffs[0])
                if not is_near_integer(first_coeff):
                    return False
                    
                int_val = int(round(first_coeff))
                if int_val <= 1:
                    return False
                    
                return sympy.isprime(int_val)
                
            except Exception:
                return False

        # Ternary number system (base-3)
        if kececi_type == TYPE_TERNARY:
            try:
                decimal_value = 0
                
                # Check for conversion method
                if hasattr(value, 'to_decimal'):
                    decimal_value = int(value.to_decimal())
                
                # Check for digits attribute
                elif hasattr(value, 'digits'):
                    digits = list(value.digits)
                    # Reverse the digits for correct positional value
                    for i, digit in enumerate(reversed(digits)):
                        try:
                            digit_int = int(digit)
                            if digit_int not in [0, 1, 2]:
                                return False  # Invalid ternary digit
                            decimal_value += digit_int * (3 ** i)
                        except (ValueError, TypeError):
                            return False
                
                # Check if iterable of digits
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    digits = list(value)
                    # Reverse the digits for correct positional value
                    for i, digit in enumerate(reversed(digits)):
                        try:
                            digit_int = int(digit)
                            if digit_int not in [0, 1, 2]:
                                return False  # Invalid ternary digit
                            decimal_value += digit_int * (3 ** i)
                        except (ValueError, TypeError):
                            return False
                
                else:
                    # Try to convert directly from string representation
                    try:
                        # Check if it looks like ternary (only digits 0,1,2)
                        str_val = str(value)
                        if all(c in '012' for c in str_val):
                            # Convert from base-3
                            for i, digit_char in enumerate(reversed(str_val)):
                                digit_int = int(digit_char)
                                decimal_value += digit_int * (3 ** i)
                        else:
                            # Try regular integer conversion
                            decimal_value = int(str_val)
                    except:
                        return False
                
                # Check primality of decimal representation
                if decimal_value <= 1:
                    return False
                    
                return sympy.isprime(decimal_value)
                
            except Exception:
                return False

        # Clifford algebra: check scalar basis part
        if kececi_type == TYPE_CLIFFORD:
            try:
                scalar = 0
                
                # Try basis dictionary
                if hasattr(value, 'basis_dict') and isinstance(getattr(value, 'basis_dict'), dict):
                    scalar = value.basis_dict.get('', 0)
                elif hasattr(value, 'basis') and isinstance(getattr(value, 'basis'), dict):
                    scalar = value.basis.get('', 0)
                
                # Check if scalar is near-integer
                if not is_near_integer(float(scalar)):
                    return False
                
                int_val = int(round(float(scalar)))
                if int_val <= 1:
                    return False
                    
                return sympy.isprime(int_val)
                
            except Exception:
                return False

        # Superreal numbers
        if kececi_type == TYPE_SUPERREAL:
            try:
                real_part = 0
                
                # Try to get real part
                if hasattr(value, 'real'):
                    real_part = getattr(value, 'real')
                elif hasattr(value, 'value'):
                    real_part = getattr(value, 'value')
                
                # Check if real part is near-integer
                if not is_near_integer(float(real_part)):
                    return False
                
                int_val = int(round(float(real_part)))
                if int_val <= 1:
                    return False
                    
                return sympy.isprime(int_val)
                
            except Exception:
                return False

        # Additional number types could be added here...
        
        # Fallback conservative behavior: not prime-like
        return False

    except Exception as e:
        warnings.warn(f"Error in is_prime_like: {e}", RuntimeWarning)
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
    - Uses module-level helpers (_find_kececi_zeta_zeros, _compute_gue_similarity, get_with_params, _plot_comparison).
    - Avoids heavy imports at module import time by importing lazily where needed.
    - Iterates over 1..TYPE_TERNARY (inclusive).
    Returns:
        (sorted_by_zeta, sorted_by_gue)
    """
    print("Automated Analysis for Keçeci Types")
    print("=" * 80)

    include_intermediate = True
    results = []

    # Default parameter sets (keçeçi testleri için örnekler)
    param_sets = [
        ('2.0', '3.0'),
        ('1+1j', '0.5+0.5j'),
        ('1.0,0.0,0.0,0.0', '0.1,0.0,0.0,0.0'),
        ('0.8,0.1,0.1', '0.0,0.05,0.0'),
        ('1.0', '0.1'),
        ('102', '1'),
    ]

    if additional_params:
        param_sets.extend(additional_params)

    type_names = {
        1: "Positive Real", 2: "Negative Real", 3: "Complex", 4: "Float", 5: "Rational",
        6: "Quaternion", 7: "Neutrosophic", 8: "Neutro-Complex", 9: "Hyperreal", 10: "Bicomplex",
        11: "Neutro-Bicomplex", 12: "Octonion", 13: "Sedenion", 14: "Clifford", 15: "Dual",
        16: "Split-Complex", 17: "Pathion", 18: "Chingon", 19: "Routon", 20: "Voudon",
        21: "Super Real", 22: "Ternary", 23: "HyperComplex",
    }

    # Iterate all defined types (inclusive)
    for kececi_type in range(TYPE_POSITIVE_REAL, TYPE_TERNARY + 1):
        name = type_names.get(kececi_type, f"Type {kececi_type}")
        best_zeta_score = 0.0
        best_gue_score = 0.0
        best_params = None

        print(f"\nAnalyzing type {kececi_type} ({name})...")

        for start, add in param_sets:
            try:
                # generate sequence (get_with_params is defined in this module)
                sequence = get_with_params(
                    kececi_type_choice=kececi_type,
                    iterations=iterations,
                    start_value_raw=start,
                    add_value_raw=add,
                    include_intermediate_steps=include_intermediate
                )

                if not sequence or len(sequence) < 20:
                    # Skip too-short sequences
                    # (analysis routines expect some minimal data)
                    print(f"  Skipped (insufficient length): params {start}, {add}")
                    continue

                # Lazy import heavy helper functions (they exist in-module)
                try:
                    zzeros, zeta_score = _find_kececi_zeta_zeros(sequence, tolerance=0.5)
                except Exception as zz_err:
                    zzeros, zeta_score = [], 0.0
                    print(f"  Warning: _find_kececi_zeta_zeros failed for {name} with params {start},{add}: {zz_err}")

                try:
                    gue_score, gue_p = _compute_gue_similarity(sequence)
                except Exception as gue_err:
                    gue_score, gue_p = 0.0, 0.0
                    print(f"  Warning: _compute_gue_similarity failed for {name} with params {start},{add}: {gue_err}")

                if zeta_score > best_zeta_score or (zeta_score == best_zeta_score and gue_score > best_gue_score):
                    best_zeta_score = zeta_score
                    best_gue_score = gue_score
                    best_params = (start, add)

            except Exception as e:
                print(f"  Error analyzing params ({start}, {add}) for type {kececi_type}: {e}")
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
        else:
            print(f"  No successful parameter set found for {name}.")

    # Sort and display results
    sorted_by_zeta = sorted(results, key=lambda x: x['zeta_score'], reverse=True)
    sorted_by_gue = sorted(results, key=lambda x: x['gue_score'], reverse=True)

    # Plot comparison if there are results (lazy-plot)
    try:
        if sorted_by_zeta or sorted_by_gue:
            _plot_comparison(sorted_by_zeta, sorted_by_gue)
    except Exception as plot_err:
        print(f"Plotting failed: {plot_err}")

    return sorted_by_zeta, sorted_by_gue

def _load_zeta_zeros(filename="zeta.txt"):
    """
    Loads Riemann zeta zeros from a text file.
    Each line should contain one floating-point number representing the imaginary part of a zeta zero.
    Lines that are empty or start with '#' are ignored.
    Returns: numpy.ndarray of zeros, or empty array if file not found / invalid.
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
                logger.warning("Invalid line skipped in %s: %r", filename, line)
        logger.info("%d zeta zeros loaded from %s.", len(zeta_zeros), filename)
        return np.array(zeta_zeros)
    except FileNotFoundError:
        logger.warning("Zeta zeros file '%s' not found.", filename)
        return np.array([])
    except Exception as e:
        logger.exception("Error while loading zeta zeros from %s: %s", filename, e)
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
    from .kececinumbers import _get_integer_representation

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
    from .kececinumbers import _get_integer_representation

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

class ValueProcessor:
    """Farklı tiplerdeki değerler için işlemleri yönetir"""
    
    @staticmethod
    def add(value1, value2):
        """İki değeri toplar"""
        if isinstance(value1, (tuple, list)) and isinstance(value2, (tuple, list)):
            # Element-wise addition
            if len(value1) != len(value2):
                # Uzunluklar farklıysa, kısa olanı sıfırlarla genişlet
                max_len = max(len(value1), len(value2))
                v1 = list(value1) + [0] * (max_len - len(value1))
                v2 = list(value2) + [0] * (max_len - len(value2))
                return tuple(a + b for a, b in zip(v1, v2))
            return tuple(a + b for a, b in zip(value1, value2))
        elif isinstance(value1, (tuple, list)):
            # Skaler toplama
            try:
                scalar = float(value2)
                return tuple(a + scalar for a in value1)
            except (ValueError, TypeError):
                return value1
        elif isinstance(value2, (tuple, list)):
            # Skaler toplama
            try:
                scalar = float(value1)
                return tuple(scalar + b for b in value2)
            except (ValueError, TypeError):
                return value2
        else:
            # Normal toplama
            return value1 + value2
    
    @staticmethod
    def divide(value, divisor, integer_division=False):
        """Değeri böler"""
        if isinstance(value, (tuple, list)):
            # Element-wise division
            if integer_division:
                return tuple(a // divisor for a in value)
            else:
                return tuple(a / divisor for a in value)
        else:
            # Normal division
            if integer_division:
                return value // divisor
            else:
                return value / divisor
    
    @staticmethod
    def multiply_add(value, multiplier, constant):
        """value * multiplier + constant işlemi"""
        if isinstance(value, (tuple, list)):
            return tuple(a * multiplier + constant for a in value)
        else:
            return value * multiplier + constant

# ==============================================================================
# --- CORE GENERATOR ---
# ==============================================================================
def unified_generator(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str,
    iterations: int,
    include_intermediate_steps: bool = False
    ) -> List[Any]:
    """
    Unified generator with input validation and robust division/ask handling.
    
    Args:
        kececi_type: Type identifier (1-22)
        start_input_raw: Starting value as string
        add_input_raw: Value to add each iteration as string
        iterations: Number of iterations to generate
        include_intermediate_steps: Whether to include intermediate calculation steps
        
    Returns:
        List of generated values
        
    Raises:
        ValueError: Invalid type or parsing error
    """
    # Type validation
    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_TERNARY):
        raise ValueError(f"Invalid Keçeci Number Type: {kececi_type}")
    
    # Sanitize raw inputs
    if start_input_raw is None or str(start_input_raw).strip() == "":
        start_input_raw = "0"
    if add_input_raw is None or str(add_input_raw).strip() == "":
        add_input_raw = "1"
    
    # Initialize variables with proper types
    current_value: Optional[Any] = None
    add_value_typed: Optional[Any] = None
    ask_unit: Optional[Any] = None
    use_integer_division = False

    def safe_add(a: Any, b: Any, direction: Optional[int] = None) -> Any:
        """
        Type-safe addition that handles various number types.
        
        Args:
            a: First value
            b: Second value or unit
            direction: Optional direction multiplier (1 for +, -1 for -)
        
        Returns:
            Result of safe addition
        """
        try:
            # Apply direction if specified
            if direction is not None:
                b = b * direction if hasattr(b, '__mul__') else b
            
            # If both are same type or compatible types
            if type(a) == type(b):
                try:
                    return a + b
                except Exception:
                    pass
            
            # Convert to compatible types if possible
            # Handle Fraction with other types
            if isinstance(a, Fraction):
                if isinstance(b, (int, float)):
                    return a + Fraction(b)
                elif isinstance(b, Fraction):
                    return a + b
                else:
                    # For other types, convert Fraction to float
                    return float(a) + b
            elif isinstance(b, Fraction):
                if isinstance(a, (int, float)):
                    return Fraction(a) + b
                else:
                    return a + float(b)
            
            # Handle complex numbers
            if isinstance(a, complex) or isinstance(b, complex):
                # Convert both to complex
                try:
                    a_complex = complex(a) if not isinstance(a, complex) else a
                    b_complex = complex(b) if not isinstance(b, complex) else b
                    return a_complex + b_complex
                except Exception:
                    pass
            
            # Handle tuples/lists
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                # Element-wise addition
                max_len = max(len(a), len(b))
                result = []
                for i in range(max_len):
                    val_a = a[i] if i < len(a) else 0
                    val_b = b[i] if i < len(b) else 0
                    result.append(val_a + val_b)
                return tuple(result) if isinstance(a, tuple) and isinstance(b, tuple) else result
            
            # Scalar addition to tuple/list
            if isinstance(a, (tuple, list)) and isinstance(b, (int, float)):
                result = [x + b for x in a]
                return tuple(result) if isinstance(a, tuple) else result
            elif isinstance(b, (tuple, list)) and isinstance(a, (int, float)):
                result = [a + x for x in b]
                return tuple(result) if isinstance(b, tuple) else result
            
            # Default: try normal addition
            return a + b
            
        except Exception as e:
            logger.debug(f"safe_add failed: {e}")
            # Fallback for direction operations
            if direction is not None:
                try:
                    if direction > 0:
                        return a + b
                    else:
                        return a - b
                except Exception as e2:
                    logger.debug(f"Fallback add/sub failed: {e2}")
            
            # Return the first operand as fallback
            return a

    def safe_divide(val: Any, divisor: Union[int, float], integer_mode: bool = False) -> Any:
        """
        Safe division with appropriate operator, handling various number types.
        
        Args:
            val: Value to divide
            divisor: Divisor
            integer_mode: Whether to use integer division
        
        Returns:
            Result of division
        """
        try:
            if integer_mode:
                # Prefer __floordiv__ if available
                if hasattr(val, '__floordiv__'):
                    return val // divisor
                else:
                    # Fallback for tuples/lists
                    if isinstance(val, (tuple, list)):
                        return tuple(x // divisor for x in val)
                    # Fallback numeric division
                    try:
                        return type(val)(int(val) // int(divisor))
                    except (ValueError, TypeError):
                        return type(val)(float(val) // float(divisor))
            else:
                # True division
                if hasattr(val, '__truediv__'):
                    return val / divisor
                else:
                    # Fallback for tuples/lists
                    if isinstance(val, (tuple, list)):
                        return tuple(x / divisor for x in val)
                    # Fallback numeric
                    return type(val)(float(val) / float(divisor))
                    
        except Exception as e:
            logger.debug(f"Division failed for {val!r} by {divisor}: {e}")
            # Raise the exception for upstream handling
            raise

    def safe_mul_add(value: Any, multiplier: Any, constant: Any) -> Any:
        """
        Type-safe value * multiplier + constant operation.
        
        Args:
            value: Base value
            multiplier: Multiplier
            constant: Constant to add
        
        Returns:
            Result of value * multiplier + constant
        """
        try:
            # First try multiplication
            if hasattr(value, '__mul__'):
                try:
                    multiplied = value * multiplier
                except Exception:
                    # Fallback for special types
                    if isinstance(value, (tuple, list)):
                        multiplied = tuple(x * multiplier for x in value)
                    else:
                        multiplied = value * multiplier
            elif isinstance(value, (tuple, list)):
                multiplied = tuple(x * multiplier for x in value)
            else:
                multiplied = value * multiplier
            
            # Then try addition with safe_add
            return safe_add(multiplied, constant)
            
        except Exception as e:
            logger.debug(f"safe_mul_add failed: {e}")
            # Return original value as fallback
            return value

    def format_fraction(frac: Any) -> Any:
        """
        Format fractions for output.
        
        Args:
            frac: Value to format
        
        Returns:
            Formatted value
        """
        if isinstance(frac, Fraction):
            if frac.denominator == 1:
                return int(frac.numerator)
            return frac
        return frac

    # Alias for backward compatibility
    _safe_divide = safe_divide
    
    # --- Type-specific parsing with defensive error reporting ---
    try:
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            try:
                # For real types, parse as float first then convert to appropriate type
                start_float = float(start_input_raw)
                add_float = float(add_input_raw)
                
                if kececi_type == TYPE_POSITIVE_REAL:
                    current_value = abs(start_float)
                    add_value_typed = abs(add_float)
                else:  # TYPE_NEGATIVE_REAL
                    current_value = -abs(start_float)
                    add_value_typed = -abs(add_float)
                    
            except Exception as e:
                logger.error("Failed to parse real inputs: start=%r add=%r -> %s", 
                            start_input_raw, add_input_raw, e)
                raise
            ask_unit = 1 if kececi_type == TYPE_POSITIVE_REAL else -1
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
            # Import parser locally to avoid circular imports
            from .kececinumbers import _parse_complex
            current_value = _parse_complex(start_input_raw)
            add_value_typed = _parse_complex(add_input_raw)
            ask_unit = complex(1, 1)
        
        elif kececi_type == TYPE_QUATERNION:
            from .kececinumbers import _parse_quaternion
            current_value = _parse_quaternion(start_input_raw)
            add_value_typed = _parse_quaternion(add_input_raw)
            # Import quaternion class if needed
            try:
                import quaternion
                ask_unit = quaternion.quaternion(1, 1, 1, 1)
            except ImportError:
                # Use a mock quaternion if not available
                ask_unit = current_value.__class__(1, 1, 1, 1) if hasattr(current_value, '__class__') else None
        
        elif kececi_type == TYPE_NEUTROSOPHIC:
            from .kececinumbers import _parse_neutrosophic
            from .kececinumbers import NeutrosophicNumber
            
            t, i, f = _parse_neutrosophic(start_input_raw)
            current_value = NeutrosophicNumber(t, i, f)
            
            t_inc, i_inc, f_inc = _parse_neutrosophic(add_input_raw)
            add_value_typed = NeutrosophicNumber(t_inc, i_inc, f_inc)
            ask_unit = NeutrosophicNumber(1, 1, 1)
        
        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            from .kececinumbers import _parse_complex, _parse_neutrosophic_complex
            from .kececinumbers import NeutrosophicComplexNumber
            
            # Use dedicated parser if available
            try:
                current_value = _parse_neutrosophic_complex(start_input_raw)
                add_value_typed = _parse_neutrosophic_complex(add_input_raw)
            except (ImportError, AttributeError):
                # Fallback to complex parsing
                s_complex = _parse_complex(start_input_raw)
                current_value = NeutrosophicComplexNumber(s_complex.real, s_complex.imag, 0.0)
                a_complex = _parse_complex(add_input_raw)
                add_value_typed = NeutrosophicComplexNumber(a_complex.real, a_complex.imag, 0.0)
            
            ask_unit = NeutrosophicComplexNumber(1, 1, 1)
        
        elif kececi_type == TYPE_HYPERREAL:
            from .kececinumbers import _parse_hyperreal
            from .kececinumbers import HyperrealNumber
            
            finite, infinitesimal, _ = _parse_hyperreal(start_input_raw)
            current_value = HyperrealNumber([finite, infinitesimal])
            
            finite_inc, infinitesimal_inc, _ = _parse_hyperreal(add_input_raw)
            add_value_typed = HyperrealNumber([finite_inc, infinitesimal_inc])
            ask_unit = HyperrealNumber([1.0, 1.0])
        
        elif kececi_type == TYPE_BICOMPLEX:
            from .kececinumbers import _parse_bicomplex
            from .kececinumbers import BicomplexNumber
            
            current_value = _parse_bicomplex(start_input_raw)
            add_value_typed = _parse_bicomplex(add_input_raw)
            ask_unit = BicomplexNumber(complex(1, 1), complex(1, 1))
        
        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            from .kececinumbers import _parse_neutrosophic_bicomplex
            from .kececinumbers import NeutrosophicBicomplexNumber
            
            current_value = _parse_neutrosophic_bicomplex(start_input_raw)
            add_value_typed = _parse_neutrosophic_bicomplex(add_input_raw)
            ask_unit = NeutrosophicBicomplexNumber(1, 1, 1, 1, 1, 1, 1, 1)
        
        elif kececi_type == TYPE_OCTONION:
            from .kececinumbers import _parse_octonion
            from .kececinumbers import OctonionNumber
            
            current_value = _parse_octonion(start_input_raw)
            add_value_typed = _parse_octonion(add_input_raw)
            ask_unit = OctonionNumber(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        elif kececi_type == TYPE_SEDENION:
            from .kececinumbers import _parse_sedenion
            from .kececinumbers import SedenionNumber
            
            current_value = _parse_sedenion(start_input_raw)
            add_value_typed = _parse_sedenion(add_input_raw)
            ask_unit = SedenionNumber([1.0] + [0.0] * 15)
        
        elif kececi_type == TYPE_CLIFFORD:
            from .kececinumbers import _parse_clifford
            from .kececinumbers import CliffordNumber
            
            current_value = _parse_clifford(start_input_raw)
            add_value_typed = _parse_clifford(add_input_raw)
            ask_unit = CliffordNumber({'': 1.0})
        
        elif kececi_type == TYPE_DUAL:
            from .kececinumbers import _parse_dual
            from .kececinumbers import DualNumber
            
            current_value = _parse_dual(start_input_raw)
            add_value_typed = _parse_dual(add_input_raw)
            ask_unit = DualNumber(1.0, 1.0)
        
        elif kececi_type == TYPE_SPLIT_COMPLEX:
            from .kececinumbers import _parse_splitcomplex
            from .kececinumbers import SplitcomplexNumber
            
            current_value = _parse_splitcomplex(start_input_raw)
            add_value_typed = _parse_splitcomplex(add_input_raw)
            ask_unit = SplitcomplexNumber(1.0, 1.0)
        
        elif kececi_type == TYPE_PATHION:
            from .kececinumbers import _parse_pathion
            from .kececinumbers import PathionNumber
            
            current_value = _parse_pathion(start_input_raw)
            add_value_typed = _parse_pathion(add_input_raw)
            ask_unit = PathionNumber([1.0] + [0.0] * 31)
        
        elif kececi_type == TYPE_CHINGON:
            from .kececinumbers import _parse_chingon
            from .kececinumbers import ChingonNumber
            
            current_value = _parse_chingon(start_input_raw)
            add_value_typed = _parse_chingon(add_input_raw)
            ask_unit = ChingonNumber([1.0] + [0.0] * 63)
        
        elif kececi_type == TYPE_ROUTON:
            from .kececinumbers import _parse_routon
            from .kececinumbers import RoutonNumber
            
            current_value = _parse_routon(start_input_raw)
            add_value_typed = _parse_routon(add_input_raw)
            ask_unit = RoutonNumber([1.0] + [0.0] * 127)
        
        elif kececi_type == TYPE_VOUDON:
            from .kececinumbers import _parse_voudon
            from .kececinumbers import VoudonNumber
            
            current_value = _parse_voudon(start_input_raw)
            add_value_typed = _parse_voudon(add_input_raw)
            ask_unit = VoudonNumber([1.0] + [0.0] * 255)
        
        elif kececi_type == TYPE_SUPERREAL:
            from .kececinumbers import _parse_superreal
            from .kececinumbers import SuperrealNumber
            
            current_value = _parse_superreal(start_input_raw)
            add_value_typed = _parse_superreal(add_input_raw)
            ask_unit = SuperrealNumber(1.0, 0.0)
        
        elif kececi_type == TYPE_TERNARY:
            from .kececinumbers import _parse_ternary
            from .kececinumbers import TernaryNumber
            
            current_value = _parse_ternary(start_input_raw)
            add_value_typed = _parse_ternary(add_input_raw)
            ask_unit = TernaryNumber([1])
        
        else:
            raise ValueError(f"Unsupported Keçeci type: {kececi_type}")
        
        # Validate that parsing was successful
        if current_value is None or add_value_typed is None:
            raise ValueError(f"Failed to parse values for type {kececi_type}")
        
    except (ValueError, TypeError, ImportError) as e:
        logger.exception("Failed to initialize type %s with start=%r add=%r: %s", 
                        kececi_type, start_input_raw, add_input_raw, e)
        return []
    
    from .kececinumbers import (
        _parse_neutrosophic, _parse_real, _parse_super_real, _parse_hyperreal,
        _parse_chingon, _is_divisible, is_prime_like
    )
    
    # Helper function for fraction formatting
    def format_fraction_local(f: Fraction) -> str:
        """Format a Fraction for display."""
        if f.denominator == 1:
            return str(f.numerator)
        else:
            return f"{f.numerator}/{f.denominator}"
    
    # Select appropriate parser based on kececi_type
    parser_map = {
        1: {'parser': lambda s: float(s), 'name': 'Positive Real'},
        2: {'parser': lambda s: -float(s), 'name': 'Negative Real'},
        3: {'parser': _parse_complex, 'name': 'Complex'},
        4: {'parser': lambda s: float(s), 'name': 'Float'},
        5: {'parser': lambda s: float(s), 'name': 'Rational'},
        6: {'parser': _parse_quaternion, 'name': 'Quaternion'},
        7: {'parser': _parse_neutrosophic, 'name': 'Neutrosophic'},
        8: {'parser': _parse_neutrosophic_complex, 'name': 'Neutrosophic Complex'},
        9: {'parser': _parse_hyperreal, 'name': 'Hyperreal'},
        10: {'parser': _parse_bicomplex, 'name': 'Bicomplex'},
        11: {'parser': _parse_neutrosophic_bicomplex, 'name': 'Neutrosophic Bicomplex'},
        12: {'parser': _parse_octonion, 'name': 'Octonion'},
        13: {'parser': _parse_sedenion, 'name': 'Sedenion'},
        14: {'parser': _parse_clifford, 'name': 'Clifford'},
        15: {'parser': _parse_dual, 'name': 'Dual'},
        16: {'parser': _parse_splitcomplex, 'name': 'Split-Complex'},
        17: {'parser': _parse_pathion, 'name': 'Pathion'},
        18: {'parser': _parse_chingon, 'name': 'Chingon'},
        19: {'parser': _parse_routon, 'name': 'Routon'},
        20: {'parser': _parse_voudon, 'name': 'Voudon'},
        21: {'parser': _parse_super_real, 'name': 'Super Real'},
        22: {'parser': _parse_ternary, 'name': 'Ternary'},
    }
    
    # Get parser function
    parser_func = parser_map.get(kececi_type)
    if parser_func is None:
        raise ValueError(f"Unsupported kececi_type: {kececi_type}")
    
    # Parse input values
    try:
        start_value = parser_func(start_input_raw)
        add_value = parser_func(add_input_raw)
    except Exception as e:
        logger.error(f"Failed to parse values: {e}")
        raise ValueError(f"Invalid input values: {e}") from e
    
    # Main generation loop
    clean_trajectory: List[Any] = [start_value]
    full_log: Optional[List[Any]] = [start_value] if include_intermediate_steps else None
    last_divisor_used: Optional[int] = None
    ask_counter = 0  # 0: +ask_unit, 1: -ask_unit
    use_integer_division = False  # Default to float division
    ask_unit = None  # Default no ask_unit
    
    for step in range(1, iterations):
        # 1. Add current value with add_value (type-safe)
        current_value = clean_trajectory[-1]
        try:
            added_value = safe_add(current_value, add_value)
        except Exception as e:
            logger.exception(f"Addition failed at step {step}: {e}")
            logger.debug(f"current_value type: {type(current_value)}, value: {current_value}")
            logger.debug(f"add_value type: {type(add_value)}, value: {add_value}")
            added_value = current_value
        
        next_q = added_value
        divided_successfully = False
        modified_value: Optional[Any] = None
        
        # Choose which divisor to try first
        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3
        
        # 2. Try divisions defensively (type-safe)
        for divisor in (primary_divisor, alternative_divisor):
            try:
                # Note: kececi_type is int, not str
                if _is_divisible(added_value, divisor, kececi_type):
                    try:
                        # Use type-safe division
                        next_q = safe_divide(added_value, divisor, use_integer_division)
                        last_divisor_used = divisor
                        divided_successfully = True
                        break
                    except Exception as e:
                        logger.debug(f"Safe division failed for {added_value!r} by {divisor}: {e}")
                        
                        # Special handling for tuples/lists
                        if isinstance(added_value, (tuple, list)):
                            try:
                                next_q = safe_divide(added_value, divisor, use_integer_division)
                                last_divisor_used = divisor
                                divided_successfully = True
                                break
                            except Exception as e2:
                                logger.debug(f"Element-wise division failed: {e2}")
                                
                                # Alternative: divide only first component
                                try:
                                    if len(added_value) > 0:
                                        if use_integer_division:
                                            first = added_value[0] // divisor
                                        else:
                                            first = added_value[0] / divisor
                                        # Create new tuple/list with divided first element
                                        if isinstance(added_value, tuple):
                                            next_q = (first,) + added_value[1:]
                                        else:
                                            next_q = [first] + added_value[1:]
                                        last_divisor_used = divisor
                                        divided_successfully = True
                                        break
                                except Exception as e3:
                                    logger.debug(f"First component division failed: {e3}")
                else:
                    logger.debug(f"Not divisible: {added_value!r} by {divisor}")
                    
            except Exception as e:
                logger.debug(f"Divisibility check failed for {added_value!r} by {divisor}: {e}")
                continue
        
        # 3. If division failed, check prime-like and apply modification
        if not divided_successfully:
            # First check prime-like
            try:
                # Note: kececi_type is int, not str
                if is_prime_like(added_value, kececi_type):
                    logger.debug(f"Prime-like number detected at step {step}: {added_value!r}")
                    # If prime-like, modify with ask_unit
                    if ask_unit is not None:
                        try:
                            direction = 1 if ask_counter == 0 else -1
                            modified_value = safe_add(added_value, ask_unit * direction)
                            ask_counter = 1 - ask_counter  # Toggle counter
                            
                            # Try division again with modified value
                            for divisor in (primary_divisor, alternative_divisor):
                                try:
                                    if _is_divisible(modified_value, divisor, kececi_type):
                                        try:
                                            next_q = safe_divide(modified_value, divisor, use_integer_division)
                                            last_divisor_used = divisor
                                            divided_successfully = True
                                            logger.debug("Division successful after prime-like modification")
                                            break
                                        except Exception as e2:
                                            logger.debug(f"Division on modified_value failed: {e2}")
                                except Exception as e:
                                    logger.debug(f"Divisibility check on modified_value failed: {e}")
                                    continue
                        except Exception as e:
                            logger.debug(f"safe_add failed for prime-like number: {e}")
                            modified_value = None
            except Exception as e:
                logger.debug(f"Prime-like check failed: {e}")
            
            # If still not divided, apply Collatz-like modification
            if not divided_successfully:
                try:
                    if ask_counter == 0:
                        # +ask_unit (3*x + 1)
                        modified_value = safe_mul_add(added_value, 3, 1)
                        ask_counter = 1
                    else:
                        # -ask_unit (3*x - 1)
                        modified_value = safe_mul_add(added_value, 3, -1)
                        ask_counter = 0
                    
                    next_q = modified_value
                    logger.debug(f"Applied Collatz-like modification: {added_value!r} -> {next_q!r}")
                    
                except Exception as e:
                    logger.exception(f"Collatz modification failed at step {step}: {e}")
                    next_q = added_value
        
        # 4. Update trajectory
        clean_trajectory.append(next_q)
        
        # 5. Store result with detailed logging if requested
        if include_intermediate_steps:
            if full_log is None:
                full_log = []
            
            # Detailed log entry
            log_entry: Dict[str, Any] = {
                'step': step,
                'added': added_value,
                'divided_by': last_divisor_used if divided_successfully else None,
                'modified': modified_value if not divided_successfully else None,
                'result': next_q
            }
            full_log.append(log_entry)
            
            # Also add intermediate values
            if added_value != next_q:
                full_log.append(added_value)
            if modified_value is not None and modified_value != added_value and modified_value != next_q:
                full_log.append(modified_value)
        
        # 6. Prime-like check for current value
        try:
            # Note: kececi_type is int, not str
            if is_prime_like(next_q, kececi_type):
                logger.debug(f"Prime-like number found at step {step}: {next_q!r}")
        except Exception as e:
            logger.debug(f"Prime-like check failed for current value: {e}")
    
    # 7. Format and return the sequence
    if include_intermediate_steps:
        result_sequence = full_log if full_log is not None else clean_trajectory
    else:
        result_sequence = clean_trajectory
    
    # Format fractions if present - SADECE SAYISAL DEĞERLERİ FORMATLA
    formatted_sequence: List[Any] = []
    for item in result_sequence:
        try:
            if isinstance(item, dict):
                # Log entry dict ise, içindeki sayısal değerleri formatla
                formatted_item: Dict[str, Any] = {}
                for key, value in item.items():
                    if isinstance(value, Fraction):
                        formatted_item[key] = format_fraction_local(value)
                    elif isinstance(value, (int, float, complex)):
                        # Sayısal değerleri olduğu gibi bırak
                        formatted_item[key] = value
                    else:
                        # String veya diğer tipleri olduğu gibi bırak
                        formatted_item[key] = value
                formatted_sequence.append(formatted_item)
            elif isinstance(item, Fraction):
                formatted_sequence.append(format_fraction_local(item))
            elif isinstance(item, (int, float, complex)):
                # Sayısal değerleri olduğu gibi bırak
                formatted_sequence.append(item)
            else:
                # String veya diğer tipler için plotting hatası olmaması için
                # Sayısal değere çevirmeye çalış
                try:
                    numeric_value = float(item)
                    formatted_sequence.append(numeric_value)
                except (ValueError, TypeError):
                    # Çevrilemezse olduğu gibi bırak (ama plotting çalışmayabilir)
                    formatted_sequence.append(item)
        except Exception as e:
            logger.debug(f"Error formatting item {item!r}: {e}")
            formatted_sequence.append(item)
    
    return formatted_sequence

def get_interactive(auto_values: Optional[Dict[str, str]] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Interactively (or programmatically via auto_values) gets parameters to generate a Keçeci sequence.

    If auto_values is provided, keys can include:
        'type_choice' (int or str), 'start_val' (str), 'add_val' (str),
        'steps' (int or str), 'show_details' ('y'/'n').

    If auto_values is None, function behaves interactively and prints a type menu.
    """
    # Local prompt function: use auto_values if present otherwise input()
    def _ask(key: str, prompt: str, default: str) -> str:
        if auto_values and key in auto_values:
            return str(auto_values[key])
        try:
            return input(prompt).strip() or default
        except Exception:
            # In non-interactive contexts where input is not available, use default
            logger.debug("input() failed for prompt %r — using default %r", prompt, default)
            return default

    interactive_mode = (auto_values is None)
    logger.info("Keçeci Numbers Interactive Generator (interactive=%s)", interactive_mode)

    # If interactive, present the full menu of type options so users see 1-22 choices
    if interactive_mode:
        menu_lines = [
            "  1: Positive Real    2: Negative Real      3: Complex",
            "  4: Float            5: Rational           6: Quaternion",
            "  7: Neutrosophic     8: Neutro-Complex     9: Hyperreal",
            " 10: Bicomplex       11: Neutro-Bicomplex  12: Octonion",
            " 13: Sedenion        14: Clifford          15: Dual",
            " 16: Split-Complex   17: Pathion           18: Chingon",
            " 19: Routon          20: Voudon            21: SuperReal",
            " 22: Ternary"
        ]
        logger.info("Available Keçeci Number Types:")
        for line in menu_lines:
            logger.info(line)

    # Defaults
    DEFAULT_TYPE = 3
    DEFAULT_STEPS = 40
    DEFAULT_SHOW_DETAILS = "no"

    default_start_values = {
        1: "2.5", 2: "-5.0", 3: "1+1j", 4: "3.14", 5: "3.5",
        6: "1.0,0.0,0.0,0.0", 7: "0.6,0.2,0.1", 8: "1+1j", 9: "0.0,0.001",
        10: "1.0,0.5,0.3,0.2", 11: "1.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0",
        12: "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0", 13: "1.0" + ",0.0"*15,
        14: "1.0+2.0e1+3.0e12", 15: "1.0,0.1", 16: "1.0,0.5",
        17: "1.0" + ",0.0"*31, 18: "1.0" + ",0.0"*63, 19: "1.0" + ",0.0"*127,
        20: "1.0" + ",0.0"*255, 21: "3.0,0.5", 22: "12",
    }

    default_add_values = {
        1: "0.5", 2: "-0.5", 3: "0.1+0.1j", 4: "0.1", 5: "0.1",
        6: "0.1,0.0,0.0,0.0", 7: "0.1,0.0,0.0", 8: "0.1+0.1j", 9: "0.0,0.001",
        10: "0.1,0.0,0.0,0.0", 11: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        12: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0", 13: "0.1" + ",0.0"*15,
        14: "0.1+0.2e1", 15: "0.1,0.0", 16: "0.1,0.0",
        17: "1.0" + ",0.0"*31, 18: "1.0" + ",0.0"*63, 19: "1.0" + ",0.0"*127,
        20: "1.0" + ",0.0"*255, 21: "1.5,2.7", 22: "1",
    }

    # Ask for inputs (uses _ask which respects auto_values when provided)
    type_input_raw = _ask('type_choice', f"Select Keçeci Number Type (1-22) [default: {DEFAULT_TYPE}]: ", str(DEFAULT_TYPE))
    try:
        type_choice = int(type_input_raw)
        if not (1 <= type_choice <= 22):
            logger.warning("Invalid type_choice %r, using default %s", type_choice, DEFAULT_TYPE)
            type_choice = DEFAULT_TYPE
    except Exception:
        logger.warning("Could not parse type_choice %r, using default %s", type_input_raw, DEFAULT_TYPE)
        type_choice = DEFAULT_TYPE

    start_prompt = _ask('start_val', f"Enter start value [default: {default_start_values[type_choice]}]: ", default_start_values[type_choice])
    add_prompt = _ask('add_val', f"Enter increment value [default: {default_add_values[type_choice]}]: ", default_add_values[type_choice])
    steps_raw = _ask('steps', f"Enter number of Keçeci steps [default: {DEFAULT_STEPS}]: ", str(DEFAULT_STEPS))
    try:
        num_kececi_steps = int(steps_raw)
        if num_kececi_steps <= 0:
            logger.warning("Non-positive steps %r, using default %d", num_kececi_steps, DEFAULT_STEPS)
            num_kececi_steps = DEFAULT_STEPS
    except Exception:
        logger.warning("Could not parse steps %r, using default %d", steps_raw, DEFAULT_STEPS)
        num_kececi_steps = DEFAULT_STEPS

    show_detail_raw = _ask('show_details', f"Include intermediate steps? (y/n) [default: {DEFAULT_SHOW_DETAILS}]: ", DEFAULT_SHOW_DETAILS)
    show_details = str(show_detail_raw).strip().lower() in ['y', 'yes']

    sequence = get_with_params(
        kececi_type_choice=type_choice,
        iterations=num_kececi_steps,
        start_value_raw=start_prompt,
        add_value_raw=add_prompt,
        include_intermediate_steps=show_details
    )

    params = {
        "type_choice": type_choice,
        "start_val": start_prompt,
        "add_val": add_prompt,
        "steps": num_kececi_steps,
        "detailed_view": show_details
    }
    logger.info("Using parameters: Type=%s, Start=%r, Add=%r, Steps=%s, Details=%s", type_choice, start_prompt, add_prompt, num_kececi_steps, show_details)
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

def _pca_var_sum(pca) -> float:
    """
    Safely return sum of PCA explained variance ratio.
    Returns 0.0 if pca has no explained_variance_ratio_ or values are NaN/invalid.
    """
    try:
        arr = getattr(pca, "explained_variance_ratio_", None)
        if arr is None:
            return 0.0
        s = float(np.nansum(arr))
        if not np.isfinite(s):
            return 0.0
        return s
    except Exception:
        return 0.0

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

def plot_octonion_3d(octonion_sequence, title="3D Octonion Trajectory"):
    """
    Plots the trajectory of octonion numbers in 3D space using the first three imaginary components (x, y, z).
    Args:
        octonion_sequence (list): List of OctonionNumber objects.
        title (str): Title of the plot.
    """
    if not octonion_sequence:
        print("Empty sequence. Nothing to plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Octonion bileşenlerini ayıkla (w: gerçek, x/y/z: ilk üç sanal bileşen)
    x = [o.x for o in octonion_sequence]
    y = [o.y for o in octonion_sequence]
    z = [o.z for o in octonion_sequence]

    # 3D uzayda çiz
    ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(x[0], y[0], z[0], c='g', s=100, label='Start', depthshade=True)
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='End', depthshade=True)

    # Eksen etiketleri ve başlık
    ax.set_xlabel('X (i)')
    ax.set_ylabel('Y (j)')
    ax.set_zlabel('Z (k)')
    ax.set_title(title)

    # Legend ve grid
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Otomatik Periyot Tespiti ve Keçeci Asal Analizi
def analyze_kececi_sequence(sequence, kececi_type):
    """
    Analyzes a Keçeci sequence for periodicity and Keçeci Prime Numbers (KPN).
    Args:
        sequence (list): List of Keçeci numbers.
        kececi_type (int): Type of Keçeci number (e.g., TYPE_OCTONION).
    Returns:
        dict: Analysis results including periodicity and KPNs.
    """
    results = {
        "periodicity": None,
        "kececi_primes": [],
        "prime_indices": []
    }

    # Periyot tespiti
    for window in range(2, len(sequence) // 2):
        is_periodic = True
        for i in range(len(sequence) - window):
            if sequence[i] != sequence[i + window]:
                is_periodic = False
                break
        if is_periodic:
            results["periodicity"] = window
            break

    # Keçeci Asal sayıları tespit et
    for idx, num in enumerate(sequence):
        if is_prime_like(num, kececi_type):
            integer_rep = _get_integer_representation(num)
            if integer_rep is not None and sympy.isprime(integer_rep):
                results["kececi_primes"].append(integer_rep)
                results["prime_indices"].append(idx)

    return results

# Makine Öğrenimi Entegrasyonu: PCA ve Kümelenme Analizi
def apply_pca_clustering(sequence, n_components=2):
    """
    Applies PCA and clustering to a Keçeci sequence for dimensionality reduction and pattern discovery.
    Args:
        sequence (list): List of Keçeci numbers.
        n_components (int): Number of PCA components.
    Returns:
        tuple: (pca_result, clusters) - PCA-transformed data and cluster labels.
    """
    # Sayıları sayısal vektörlere dönüştür
    vectors = []
    for num in sequence:
        if isinstance(num, OctonionNumber):
            vectors.append(num.coeffs)
        elif isinstance(num, Fraction):
            vectors.append([float(num)])
        else:
            vectors.append([float(num)])

    # PCA uygula
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(vectors)

    # Kümelenme (K-Means)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    return pca_result, clusters

# Etkileşimli Görselleştirme (Plotly DASH)
def generate_interactive_plot(sequence, kececi_type):
    """
    Generates an interactive 3D plot using Plotly for Keçeci sequences.
    Args:
        sequence (list): List of Keçeci numbers.
        kececi_type (int): Type of Keçeci number.
    """
    import plotly.graph_objects as go

    if kececi_type == TYPE_OCTONION:
        x = [num.x for num in sequence]
        y = [num.y for num in sequence]
        z = [num.z for num in sequence]
    elif kececi_type == TYPE_COMPLEX:
        x = [num.real for num in sequence]
        y = [num.imag for num in sequence]
        z = [0] * len(sequence)
    else:
        x = range(len(sequence))
        y = [float(num) for num in sequence]
        z = [0] * len(sequence)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=5, color=z, colorscale='Viridis'),
        line=dict(width=2)
    )])

    fig.update_layout(
        title=f"Interactive 3D Plot: Keçeci Type {kececi_type}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()

# Keçeci Varsayımı Test Aracı
def test_kececi_conjecture(sequence: List[Any], add_value: Any, kececi_type: Optional[int] = None, max_steps: int = 1000) -> bool:
    """
    Tests the Keçeci Conjecture for a given starting `sequence`.
    - sequence: initial list-like of Keçeci numbers (will be copied).
    - add_value: typed increment (must be of compatible type with elements).
    - kececi_type: optional type constant (used by is_prime_like); if None, fallback to is_prime.
    - max_steps: maximum additional steps to try.
    Returns True if a Keçeci-prime is reached within max_steps, otherwise False.
    """
    traj = list(sequence)
    if not traj:
        raise ValueError("sequence must contain at least one element")

    for step in range(max_steps):
        last = traj[-1]
        # Check prime-like condition
        try:
            if kececi_type is not None:
                if is_prime_like(last, kececi_type):
                    return True
            else:
                # fallback: try is_prime on integer rep
                if is_prime(last):
                    return True
        except Exception:
            # If prime test fails, continue attempts
            pass

        # Compute next element: prefer safe_add, else try native addition
        next_val = None
        try:
            next_val = safe_add(last, add_value, +1)
        except Exception:
            try:
                next_val = last + add_value
            except Exception:
                # cannot add -> abort
                return False

        traj.append(next_val)

    return False

def format_fraction(value):
    """Fraction nesnelerini güvenli bir şekilde formatlar."""
    if isinstance(value, Fraction):
        return float(value)  # veya str(value)
    return value

# Yardımcı fonksiyon: Sequence'i temizle
def clean_sequence_for_plotting(sequence: List[Any]) -> List[Any]:
    """
    Her türlü sequence'i plot fonksiyonu için temizler.
    """
    if not sequence:
        return []
    
    # Eğer dictionary listesi ise
    if isinstance(sequence[0], dict):
        cleaned = []
        for item in sequence:
            if isinstance(item, dict):
                # Önce 'value' anahtarını ara
                for key in ['value', 'result', 'numeric_value', 'added', 'modified']:
                    if key in item:
                        cleaned.append(item[key])
                        break
                else:
                    cleaned.append(0)
            else:
                cleaned.append(item)
        sequence = cleaned
    
    # String, tuple, list içeriyorsa temizle
    cleaned_sequence = []
    for item in sequence:
        cleaned_sequence.append(extract_numeric_value(item))
    
    return cleaned_sequence


def extract_numeric_value(item: Any) -> float:
    """
    Her türlü değerden sayısal değer çıkar.
    Tüm dönüşümler float tipinde olacak.
    """
    # 1. Doğrudan sayısal tipler
    if isinstance(item, (int, float)):
        return float(item)
    
    # 2. Fraction tipi
    if isinstance(item, Fraction):
        return float(item)
    
    # 3. Decimal tipi
    if isinstance(item, Decimal):
        return float(item)
    
    # 4. Complex sayılar (sadece gerçek kısmı)
    if isinstance(item, complex):
        return float(item.real)
    
    # 5. String işleme
    if isinstance(item, str):
        item = item.strip()
        if not item:
            return 0.0
        
        # Kesir kontrolü
        if '/' in item:
            try:
                # Örnek: "3/4", "1 1/2"
                if ' ' in item:  # Karışık sayı: "1 1/2"
                    whole, fraction = item.split(' ', 1)
                    num, den = fraction.split('/')
                    return float(whole) + (float(num) / float(den))
                else:  # Basit kesir: "3/4"
                    num, den = item.split('/')
                    return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass
        
        # Normal sayısal dize
        try:
            # Bilimsel gösterim ve diğer formatları da destekle
            return float(item)
        except (ValueError, TypeError):
            return 0.0
    
    # 6. Dizi/Iterable tipler
    if isinstance(item, (tuple, list, set)):
        for element in item:
            value = extract_numeric_value(element)
            if value != 0:
                return value
        return 0.0
    
    # 7. Diğer tipler için deneme
    try:
        return float(item)
    except (ValueError, TypeError, AttributeError):
        return 0.0

def extract_numeric_values(sequence: List[Any], 
                          strict: bool = False) -> List[float]:
    """
    Her türlü değerden sayısal değerleri çıkar.
    
    Args:
        sequence: İşlenecek dizi
        strict: True ise, dönüştürülemeyen değerler için ValueError fırlatır
    
    Returns:
        Sayısal değerler listesi
    """
    result: List[float] = []
    
    for item in sequence:
        try:
            value = extract_numeric_value(item)
            result.append(value)
        except Exception as e:
            if strict:
                raise ValueError(f"Failed to extract numeric value from {item!r}") from e
            result.append(0.0)
    
    return result


# Ek yardımcı fonksiyonlar
def extract_clean_numbers(sequence: List[Any], 
                         remove_zeros: bool = False) -> List[float]:
    """
    Temiz sayısal değerleri çıkar ve opsiyonel olarak sıfırları kaldır.
    """
    values = extract_numeric_values(sequence)
    if remove_zeros:
        values = [v for v in values if v != 0]
    return values

def find_first_numeric(sequence: List[Any]) -> Optional[float]:
    """
    Dizideki ilk geçerli sayısal değeri bulur.
    """
    for item in sequence:
        value = extract_numeric_value(item)
        if value != 0:
            return value
    return None

def extract_fraction_values(sequence: List[Any]) -> tuple[List[float], List[int], List[int]]:
    """Safely extract values from Fraction sequence."""
    float_vals: List[float] = []
    numerators: List[int] = []
    denominators: List[int] = []
    
    for item in sequence:
        if isinstance(item, Fraction):
            float_vals.append(float(item))
            numerators.append(item.numerator)
            denominators.append(item.denominator)
        else:
            # Diğer tipler için fallback
            try:
                float_vals.append(float(item))
                # Fraction olmadığı için pay/payda uydur
                if isinstance(item, (int, float)):
                    numerators.append(int(item))
                    denominators.append(1)
                else:
                    numerators.append(0)
                    denominators.append(1)
            except (ValueError, TypeError):
                float_vals.append(0.0)
                numerators.append(0)
                denominators.append(1)
    
    return float_vals, numerators, denominators


def extract_complex_values(sequence: List[Any]) -> tuple[List[float], List[float], List[float]]:
    """Safely extract complex values."""
    real_parts: List[float] = []
    imag_parts: List[float] = []
    magnitudes: List[float] = []
    
    for item in sequence:
        if isinstance(item, complex):
            real_parts.append(float(item.real))
            imag_parts.append(float(item.imag))
            magnitudes.append(float(abs(item)))
        else:
            # Complex değilse sıfır ekle
            real_parts.append(0.0)
            imag_parts.append(0.0)
            magnitudes.append(0.0)
    
    return real_parts, imag_parts, magnitudes

def plot_numbers(sequence: List[Any], title: str = "Keçeci Number Sequence Analysis"):
    """
    Tüm 22 Keçeci Sayı türü için detaylı görselleştirme sağlar.
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

    # ÖNEMLİ: Eğer sequence dictionary listesi ise, sadece 'value' değerlerini al
    if sequence and isinstance(sequence[0], dict):
        print("Detected dictionary sequence. Extracting 'value' fields for plotting...")
        # Dictionary'den sadece 'value' değerlerini çıkar
        extracted_sequence = []
        for item in sequence:
            if isinstance(item, dict) and 'value' in item:
                extracted_sequence.append(item['value'])
            elif isinstance(item, dict) and 'result' in item:
                extracted_sequence.append(item['result'])
            else:
                # Diğer durumlarda sıfır ekle
                extracted_sequence.append(0)
        sequence = extracted_sequence
    
    # Ayrıca, eğer sequence string veya tuple içeriyorsa, sayısal değerlere çevir
    cleaned_sequence = []
    for item in sequence:
        if isinstance(item, (int, float, complex, Fraction)):
            cleaned_sequence.append(item)
        elif isinstance(item, dict):
            # Dictionary içinden sayısal değer bulmaya çalış
            for key in ['value', 'result', 'numeric_value']:
                if key in item and isinstance(item[key], (int, float, complex, Fraction)):
                    cleaned_sequence.append(item[key])
                    break
            else:
                cleaned_sequence.append(0)
        elif isinstance(item, str):
            # String'den sayı çıkarmaya çalış
            try:
                # Kesir formatı: "3/4"
                if '/' in item:
                    parts = item.split('/')
                    if len(parts) == 2:
                        num = float(parts[0])
                        den = float(parts[1])
                        if den != 0:
                            cleaned_sequence.append(num / den)
                            continue
                # Normal sayı
                cleaned_sequence.append(float(item))
            except (ValueError, TypeError):
                cleaned_sequence.append(0)
        elif isinstance(item, (tuple, list)):
            # Tuple/list içinden ilk sayısal değeri al
            for element in item:
                if isinstance(element, (int, float, complex, Fraction)):
                    cleaned_sequence.append(element)
                    break
            else:
                cleaned_sequence.append(0)
        elif hasattr(item, 'real'):
            # real attribute'u olan nesneler
            cleaned_sequence.append(float(item.real))
        else:
            cleaned_sequence.append(0)
    
    sequence = cleaned_sequence

    # Temizlenmiş sequence boş mu kontrol et
    if not sequence:
        print("Sequence is empty after cleaning. Nothing to plot.")
        return

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # `sequence` is the iterable you want to visualise
    first_elem = sequence[0]

    # --- 1. Fraction (Rational)
    if isinstance(first_elem, Fraction):
        # Helper fonksiyonları kullan
        float_vals, numerators, denominators = extract_fraction_values(sequence)
        
        # GridSpec ile 4 alt grafik oluştur
        gs = GridSpec(2, 2, figure=fig)

        # 1. Grafik: Float değerleri
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(float_vals, 'o-', color='tab:blue')
        ax1.set_title("Fraction as Float")
        ax1.set_ylabel("Value")

        # 2. Grafik: Pay ve payda değerleri
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(numerators, 's-', label='Numerator', color='tab:orange')
        ax2.plot(denominators, '^-', label='Denominator', color='tab:green')
        ax2.set_title("Numerator & Denominator")
        ax2.legend()

        # 3. Grafik: Pay/Payda oranı
        ax3 = fig.add_subplot(gs[1, 0])
        ratios = [float(n) / float(d) if d != 0 else 0.0 for n, d in zip(numerators, denominators)]
        ax3.plot(ratios, 'o-', color='tab:purple')
        ax3.set_title("Numerator/Denominator Ratio")
        ax3.set_ylabel("n/d")

        # 4. Grafik: Pay vs Payda dağılımı
        ax4 = fig.add_subplot(gs[1, 1])
        # numpy array'e çevirerek tip güvenliği sağla
        numerators_arr = np.array(numerators, dtype=np.float64)
        denominators_arr = np.array(denominators, dtype=np.float64)
        sc = ax4.scatter(numerators_arr, denominators_arr, 
                         c=range(len(sequence)), cmap='plasma', s=30)
        ax4.set_title("Numerator vs Denominator Trajectory")
        ax4.set_xlabel("Numerator")
        ax4.set_ylabel("Denominator")
        plt.colorbar(sc, ax=ax4, label="Step")

    # --- 2. int, float (Positive/Negative Real, Float)
    elif isinstance(first_elem, (int, float)):
        numeric_vals = extract_numeric_values(sequence)
        
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(numeric_vals, 'o-', color='tab:blue', markersize=5)
        ax.set_title("Real Number Sequence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    # --- 3. Complex
    elif isinstance(first_elem, complex):
        real_parts, imag_parts, magnitudes = extract_complex_values(sequence)
        
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
        # numpy array kullan
        real_arr = np.array(real_parts, dtype=np.float64)
        imag_arr = np.array(imag_parts, dtype=np.float64)
        ax4.plot(real_arr, imag_arr, '.-', alpha=0.7)
        ax4.scatter(real_arr[0], imag_arr[0], c='g', s=100, label='Start')
        ax4.scatter(real_arr[-1], imag_arr[-1], c='r', s=100, label='End')
        ax4.set_title("Complex Plane")
        ax4.set_xlabel("Re(z)")
        ax4.set_ylabel("Im(z)")
        ax4.legend()
        ax4.axis('equal')
        ax4.grid(True, alpha=0.3)

    # --- 4. quaternion
    elif (isinstance(first_elem, quaternion) or 
          (hasattr(first_elem, 'components') and len(getattr(first_elem, 'components', [])) == 4) or
          (hasattr(first_elem, 'w') and hasattr(first_elem, 'x') and 
           hasattr(first_elem, 'y') and hasattr(first_elem, 'z'))):

        try:
            comp = np.array([
                (q.w, q.x, q.y, q.z) if hasattr(q, 'w') else 
                (getattr(q, 'components', [0,0,0,0]) if hasattr(q, 'components') else [0,0,0,0])
                for q in sequence
            ])

            w, x, y, z = comp.T
            magnitudes = np.linalg.norm(comp, axis=1)
            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(2, 2, figure=fig)

            # Component time‑series
            ax1 = fig.add_subplot(gs[0, 0])
            labels = ['w', 'x', 'y', 'z']
            for i, label in enumerate(labels):
                ax1.plot(comp[:, i], label=label, alpha=0.8)
            ax1.set_title("Quaternion Components")
            ax1.legend()

            # Magnitude plot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(magnitudes, 'o-', color='tab:purple')
            ax2.set_title("Magnitude |q|")

            # 3‑D trajectory of the vector part (x, y, z)
            ax3 = fig.add_subplot(gs[1, :], projection='3d')
            ax3.plot(x, y, z, alpha=0.7)
            ax3.scatter(x[0], y[0], z[0], c='g', s=100, label='Start')
            ax3.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='End')
            ax3.set_title("3D Trajectory (x,y,z)")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
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
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
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
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
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
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
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
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
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
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


    # --- 21. Super Real
    elif isinstance(first_elem, SuperrealNumber):
        # real ve split bileşenlerini ayır
        reals = np.array([x.real for x in sequence])
        splits = np.array([x.split for x in sequence])

        gs = GridSpec(2, 2, figure=fig)  # 2 satır, 2 sütun
        #gs = GridSpec(2, 1, figure=fig)

        # Real bileşenini çizdir
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(reals, 'o-', color='tab:blue', label='Real')
        ax1.set_title("Real Component")
        ax1.legend()

        # Split bileşenini çizdir
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(splits, 'o-', color='tab:red', label='Split')
        ax2.set_title("Split Component")
        ax2.legend()

        if use_pca and len(sequence) > 2:
            try:
                # PCA için veriyi hazırla
                data = np.column_stack((reals, splits))
                # Minimum gereksinimler: en az 3 örnek ve en az 2 değişken (burada 2 var)
                if data.shape[0] >= 3:
                    # finite ve non-NaN satırları seç
                    mask = np.all(np.isfinite(data), axis=1)
                    data_clean = data[mask]
                    if data_clean.shape[0] >= 3:
                        try:
                            pca = PCA(n_components=2)
                            proj = pca.fit_transform(data_clean)

                            # Güvenli varyans toplamı helper'ı kullanın
                            var_sum = _pca_var_sum(pca)

                            # 2D çizim (projisyon iki bileşenli olduğu için daha uygun)
                            ax3 = fig.add_subplot(gs[:, 1])
                            sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                            ax3.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                            ax3.set_xlabel("PC1")
                            ax3.set_ylabel("PC2")
                            plt.colorbar(sc, ax=ax3, label="Iteration")

                            # Eğer 3D görmek isterseniz, alternatif:
                            # ax3 = fig.add_subplot(gs[:, 1], projection='3d')
                            # ax3.scatter(proj[:,0], proj[:,1], np.zeros_like(proj[:,0]), c=range(len(proj)), cmap='viridis', s=25)
                            # ax3.set_title(f"PCA Projection (Var: {var_sum:.3f})")

                        except Exception as e:
                            logger.exception("PCA failed for Superreal data: %s", e)
                            ax3 = fig.add_subplot(gs[:, 1])
                            ax3.text(0.5, 0.5, f"PCA Error: {str(e)[:80]}", ha='center', va='center', fontsize=10)
                            ax3.set_title("PCA Projection (Error)")
                    else:
                        ax3 = fig.add_subplot(gs[:, 1])
                        ax3.text(0.5, 0.5, "Insufficient finite data for PCA", ha='center', va='center')
                        ax3.set_title("PCA Projection (Insufficient data)")
                else:
                    ax3 = fig.add_subplot(gs[:, 1])
                    ax3.text(0.5, 0.5, "Need ≥3 samples for PCA", ha='center', va='center')
                    ax3.set_title("PCA Projection (Not enough samples)")
            except Exception as e:
                ax3 = fig.add_subplot(gs[:, 1])
                ax3.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax3 = fig.add_subplot(gs[:, 1])
            ax3.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 22. Ternary
    elif isinstance(first_elem, TernaryNumber):
        # Tüm TernaryNumber nesnelerinin digits uzunluğunu belirle
        max_length = max(len(x.digits) for x in sequence)

        # Her bir TernaryNumber nesnesinin digits listesini max_length uzunluğuna tamamla
        padded_digits = []
        for x in sequence:
            padded_digit = x.digits + [0] * (max_length - len(x.digits))
            padded_digits.append(padded_digit)

        # NumPy dizisine dönüştür
        digits = np.array(padded_digits)

        gs = GridSpec(2, 2, figure=fig)  # 2 satır, 2 sütun

        # Her bir rakamın dağılımını çizdir
        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(digits.shape[1]):
            ax1.plot(digits[:, i], 'o-', alpha=0.6, label=f'digit {i}')
        ax1.set_title("Ternary Digits")
        ax1.legend(ncol=4, fontsize=6)

        # Üçlü sayı sistemindeki değerleri ondalık sisteme çevirip çizdir
        decimal_values = np.array([x.to_decimal() for x in sequence])
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(decimal_values, 'o-', color='tab:green')
        ax2.set_title("Decimal Values")

        if use_pca and len(sequence) > 2:
            try:
                # PCA için veriyi hazırla
                pca = PCA(n_components=2)
                proj = pca.fit_transform(digits)

                # PCA projeksiyonunu çizdir
                ax3 = fig.add_subplot(gs[1, :])  # 2. satırın tamamını kullan
                sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                ax3.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                plt.colorbar(sc, ax=ax3, label="Iteration")
            except Exception as e:
                ax3 = fig.add_subplot(gs[1, :])
                ax3.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


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
    # If user runs module directly, configure basic logging to console for demonstration.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Keçeci Numbers Module - Demonstration")
    logger.info("This script demonstrates the generation of various Keçeci Number types.")

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
        "Dual": TYPE_DUAL, "Splitcomplex": TYPE_SPLIT_COMPLEX, "Pathion": TYPE_PATHION,
        "Chingon": TYPE_CHINGON, "Routon": TYPE_ROUTON, "Voudon": TYPE_VOUDON,
        "Super Real": TYPE_SUPERREAL, "Ternary": TYPE_TERNARY, "Hypercomplex": TYPE_HYPERCOMPLEX,
    }

    for name, type_id in all_types.items():
        start = "-5" if type_id == TYPE_NEGATIVE_REAL else "2+3j" if type_id in [TYPE_COMPLEX, TYPE_BICOMPLEX] else START_VAL
        try:
            seq = get_with_params(type_id, STEPS, start, ADD_VAL)
            if seq:
                logger.info("Generated sequence for %s (len=%d).", name, len(seq))
                # Optional: plot for a few selected types to avoid overloading user's environment
        except Exception as e:
            logger.exception("Demo generation failed for type %s: %s", name, e)

    logger.info("Demonstration finished.")

def Real(x: float) -> HypercomplexNumber:
    """Generate a real number (1D hypercomplex)."""
    return HypercomplexNumber.from_real(x)

def Complex(real: float, imag: float) -> HypercomplexNumber:
    """Generate a complex number (2D hypercomplex)."""
    return HypercomplexNumber.from_complex(real, imag)

def Quaternion(w: float, x: float, y: float, z: float) -> HypercomplexNumber:
    """Generate a quaternion (4D hypercomplex)."""
    return HypercomplexNumber.from_quaternion(w, x, y, z)

def Octonion(*coeffs: float) -> HypercomplexNumber:
    """Generate an octonion (8D hypercomplex)."""
    return HypercomplexNumber.from_octonion(*coeffs)
"""
def Bicomplex(z1_real: float, z1_imag: float, z2_real: float, z2_imag: float) -> BicomplexNumber:
    Generate a bicomplex number.
Argument 1,2 to "BicomplexNumber" has incompatible type "HypercomplexNumber"; expected "complex"  [arg-type]
    z1 = HypercomplexNumber(z1_real, z1_imag, dimension=2)
    z2 = HypercomplexNumber(z2_real, z2_imag, dimension=2)
    return BicomplexNumber(z1, z2)
"""
def Bicomplex(z1_real: float, z1_imag: float, 
              z2_real: float, z2_imag: float) -> BicomplexNumber:
    """Generate a bicomplex number from real/imag parts."""
    # Doğrudan complex sayılar oluştur
    z1 = complex(z1_real, z1_imag)
    z2 = complex(z2_real, z2_imag)
    return BicomplexNumber(z1, z2)

def Neutrosophic(determinate: float, indeterminate: float) -> NeutrosophicNumber:
    """Generate a neutrosophic number."""
    return NeutrosophicNumber(determinate, indeterminate)

def Sedenion(*coeffs) -> HypercomplexNumber:
    """Generate a sedenion."""
    coeffs_tuple = tuple(coeffs)
    if len(coeffs_tuple) != 16:
        coeffs_tuple = coeffs_tuple + (0.0,) * (16 - len(coeffs_tuple))
    return HypercomplexNumber(*coeffs_tuple, dimension=16)

def Pathion(*coeffs) -> HypercomplexNumber:
    """Generate a pathion."""
    coeffs_tuple = tuple(coeffs)
    if len(coeffs_tuple) != 32:
        coeffs_tuple = coeffs_tuple + (0.0,) * (32 - len(coeffs_tuple))
    return HypercomplexNumber(*coeffs_tuple, dimension=32)

def Chingon(*coeffs) -> HypercomplexNumber:
    """Generate a chingon."""
    coeffs_tuple = tuple(coeffs)
    if len(coeffs_tuple) != 64:
        coeffs_tuple = coeffs_tuple + (0.0,) * (64 - len(coeffs_tuple))
    return HypercomplexNumber(*coeffs_tuple, dimension=64)

def Routon(*coeffs) -> HypercomplexNumber:
    """Generate a routon."""
    coeffs_tuple = tuple(coeffs)
    if len(coeffs_tuple) != 128:
        coeffs_tuple = coeffs_tuple + (0.0,) * (128 - len(coeffs_tuple))
    return HypercomplexNumber(*coeffs_tuple, dimension=128)

def Voudon(*coeffs) -> HypercomplexNumber:
    """Generate a voudon."""
    coeffs_tuple = tuple(coeffs)
    if len(coeffs_tuple) != 256:
        coeffs_tuple = coeffs_tuple + (0.0,) * (256 - len(coeffs_tuple))
    return HypercomplexNumber(*coeffs_tuple, dimension=256)
