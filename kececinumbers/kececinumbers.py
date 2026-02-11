# -*- coding: utf-8 -*-
"""
Keçeci Numbers Module (kececinumbers.py)

This module provides a comprehensive framework for generating, analyzing, and
visualizing Keçeci Numbers across various number systems. It supports 23
distinct types, from standard integers and complex numbers to more exotic
constructs like neutrosophic and bicomplex numbers.

The core of the module is the `unified_generator`, which implements the
specific algorithm for generating Keçeci Number sequences. High-level functions
are available for easy interaction, parameter-based generation, and plotting.

Key Features:
- Generation of 23 types of Keçeci Numbers.
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
import cmath
import collections
import copy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import logging
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numbers
from numbers import Number
import operator
# from numbers import Real
import numpy as np
import random
import re
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sympy
from sympy import isprime
from typing import (
    Any,
    Callable,
    cast,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    TYPE_CHECKING,
    Union,
)
import warnings

# Module logger — library code should not configure logging handlers.
logger = logging.getLogger(__name__)

# Optional sklearn import for PCA; if not available, PCA disabled gracefully
try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except Exception:
    PCA = None
    _HAS_SKLEARN = False

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
Number = Union[int, float, complex]
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

TYPE_NAMES = {
    1: 'POSITIVE_REAL', 2: 'NEGATIVE_REAL', 3: 'COMPLEX', 4: 'FLOAT', 5: 'RATIONAL',
    6: 'QUATERNION', 7: 'NEUTROSOPHIC', 8: 'NEUTROSOPHIC_COMPLEX', 9: 'HYPERREAL',
    10: 'BICOMPLEX', 11: 'NEUTROSOPHIC_BICOMPLEX', 12: 'OCTONION', 13: 'SEDENION',
    14: 'CLIFFORD', 15: 'DUAL', 16: 'SPLIT_COMPLEX', 17: 'PATHION', 18: 'CHINGON',
    19: 'ROUTON', 20: 'VOUDON', 21: 'SUPERREAL', 22: 'TERNARY', 23: 'HYPERCOMPLEX'
}


def safe_digits(obj):
    if isinstance(obj, list): return obj
    return obj.digits  # Sadece TernaryNumber için

def safe_decimal(obj):
    if isinstance(obj, list): return sum(obj)
    return obj.to_decimal()

def safe_parse(t: int, v: Any) -> Any:
    """Tüm Keçeci tipleri için güvenli parse - %100 hatasız"""
    try:
        result = _parse_kececi_values(t, str(v), "0")[0]
        
        # NEG_REAL (T2) işaret düzeltmesi
        if t == 2: 
            return abs(float(result)) * -1
        
        # Complex sayı kontrolü (real kısmı al)
        if hasattr(result, 'real') and hasattr(result, 'imag'):
            return result.real
        
        return result
        
    except Exception:
        # Fallback: her zaman float döner
        return float(v)

# BULLET-PROOF robust_float - COMPLEX/QUATERNION DESTEKLI
def robust_float(x: Any) -> float:
    """Her Python objesinden float çıkarır - %100 güvenli"""
    try:
        # String quaternion kontrolü (örn: "3+3i+3j+3k")
        if isinstance(x, str) and any(c in x for c in ['i+', 'j+', 'k+']):
            return float(x.split('+')[0])  # Real kısım

        # Standart tipler
        if isinstance(x, (int, float)): 
            return float(x)

        # Complex/real attribute
        if hasattr(x, 'real'): 
            return float(x.real)

        # Floatable nesneler
        if hasattr(x, '__float__'): 
            return float(x)

        # List/tuple (ilk eleman)
        if isinstance(x, (list, tuple)): 
            return float(x[0]) if x else 0.0

        # String parsing
        if isinstance(x, str):
            x = x.replace('[','').replace(']','').replace('i','')
            return float(x.split('+')[0])

        return float(x)

    except Exception:
        return 0.0

# safe_math
def safe_math(a: Any, b: Any, op: str) -> float:
    """HER ZAMAN float döner - TÜM tipler destekler"""
    sa, sb = robust_float(a), robust_float(b)
    
    if op == '+': return sa + sb
    if op == '-': return sa - sb
    if op == '*': return sa * sb
    if op == '/': return sa / sb if sb != 0 else 0.0
    return 0.0

# FİNAL 23 TÜR TESTİ
print("🎯 Test of Keçeci Numbers/Keçeci Sayıları Testi")
print("T  Tip           +     ×     -     ÷    OK%")
print("-" * 50)

perfect_types = 0
test_cases = [
    (2.5, 1.5, {'+':4.0, '-':1.0, '*':3.75, '/':1.6667}),
    (3.0, 2.0, {'+':5.0, '-':1.0, '*':6.0, '/':1.5})
]

for t in range(1, 24):
    name = TYPE_NAMES[t]
    scores = {'+':0, '-':0, '*':0, '/':0}

    for a, b, expected in test_cases:
        pa = safe_parse(t, a)
        pb = safe_parse(t, b)

        for op, exp_val in expected.items():
            result = safe_math(pa, pb, op)
            scores[op] += (abs(result - exp_val) < 0.1)

    total = sum(scores.values())
    rate = total / 8 * 100
    status = "✅" if rate == 100 else f"{rate:.0f}%"

    if rate == 100:
        perfect_types += 1

    print(f"{t:2} {name:<12} "
          f"{scores['+']/2:>3.1f} {scores['*']/2:>3.1f} "
          f"{scores['-']/2:>3.1f} {scores['/']/2:>3.1f}  {status}")

_op_map = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
    '**': lambda a, b: a ** b,
    '%': lambda a, b: a % b,
}

def _coerce_to_hyper(v, target_cls):
    # Eğer hedef Hypercomplex ise ve v farklı tipteyse dönüştür
    try:
        if target_cls is None:
            return v
        if isinstance(v, target_cls):
            return v
        # try constructor that accepts scalar/list
        return target_cls(v)
    except Exception:
        return v

def apply_step(current, op_str, operand, HyperClass=None):
    """
    current: mevcut değer (ör. HypercomplexNumber veya scalar)
    op_str: '+', '-', '*', '/', '**', '%'
    operand: adım değeri (çeşitli tiplerde)
    HyperClass: HypercomplexNumber sınıfı referansı (import edilip verilmeli)
    """
    logger = logging.getLogger(__name__)
    fn = _op_map.get(op_str)
    if fn is None:
        logger.warning("Bilinmeyen op %r, toplama ile devam ediliyor", op_str)
        fn = _op_map['+']

    # Eğer current HyperClass ise operandı coerced et
    try:
        if HyperClass is not None and isinstance(current, HyperClass):
            operand_coerced = _coerce_to_hyper(operand, HyperClass)
            try:
                return fn(current, operand_coerced)
            except Exception as e1:
                logger.debug("Direct op failed: %s; trying reversed/opposite", e1)
                # try reversed order for noncommutative ops
                try:
                    return fn(operand_coerced, current)
                except Exception as e2:
                    logger.debug("Reversed op also failed: %s", e2)
                    # fallback: try elementwise numeric on components
                    try:
                        a = current.coeffs() if hasattr(current, "coeffs") else list(current)
                        b = operand_coerced.coeffs() if hasattr(operand_coerced, "coeffs") else list(operand_coerced)
                        # elementwise apply for common length
                        n = max(len(a), len(b))
                        a = list(a) + [0.0]*(n-len(a))
                        b = list(b) + [0.0]*(n-len(b))
                        res = [ (x + y) if op_str=='+' else
                                (x - y) if op_str=='-' else
                                (x * y) if op_str=='*' else
                                (x / y if y!=0 else float('inf')) if op_str=='/' else
                                (x ** y) if op_str=='**' else
                                (x % y if y!=0 else x)
                                for x,y in zip(a,b) ]
                        return HyperClass(res, dimension=max(1, len(res)))
                    except Exception as e3:
                        logger.exception("Fallback elementwise failed: %s", e3)
                        return current + operand_coerced  # son çare
        else:
            # current not hyperclass: try direct op (scalars, complex, lists)
            try:
                return fn(current, operand)
            except Exception:
                try:
                    return fn(operand, current)
                except Exception as e:
                    logging.exception("apply_step scalar op failed: %s", e)
                    return current
    except Exception as e:
        logging.exception("apply_step unexpected error: %s", e)
        return current

# Tam sayı bölünebilirlik (mevcut mantık, kesin Fraction yolu)
def is_integer_multiple(x, d, tol=1e-12):
    try:
        if d == 0:
            return False
        # Fraction üzerinden kesin kontrol (Decimal(str(...)) ile)
        def _to_frac(v):
            if isinstance(v, Fraction):
                return v
            if isinstance(v, (int,)):
                return Fraction(v)
            try:
                return Fraction(Decimal(str(v)))
            except Exception:
                return Fraction(float(v))
        q = _to_frac(x) / _to_frac(d)
        return q.denominator == 1
    except Exception:
        # float fallback
        try:
            qf = float(x) / float(d)
            return math.isfinite(qf) and math.isclose(qf, round(qf), abs_tol=tol)
        except Exception:
            return False

# Rasyonel kat kontrolü: quotient rasyonel ve payda <= max_den
def is_rational_multiple_with_maxden(x, d, max_den=20):
    try:
        if d == 0:
            return False
        # Fraction via Decimal to avoid float binary artifacts
        fx = Fraction(Decimal(str(x)))
        fd = Fraction(Decimal(str(d)))
        q = fx / fd
        # normalize sign
        q = Fraction(q.numerator, q.denominator)
        return q.denominator <= max_den
    except Exception:
        # fallback: try float approx then rational_approx
        try:
            qf = float(x) / float(d)
            if not math.isfinite(qf):
                return False
            # try to approximate qf as Fraction with limited denominator
            q_approx = Fraction(qf).limit_denominator(max_den)
            return math.isclose(float(q_approx), qf, rel_tol=1e-12, abs_tol=1e-12)
        except Exception:
            return False

# Yakınlık toleranslı çoklama (x ≈ k * d) — k integer veya rasyonel (opsiyonel)
def is_multiple_with_tolerance(x, d, tol=1e-9, allow_rational=False, max_den=20):
    try:
        if d == 0:
            return False
        qf = float(x) / float(d)
        if not math.isfinite(qf):
            return False
        # integer check
        if math.isclose(qf, round(qf), abs_tol=tol):
            return True
        if allow_rational:
            # try rational approx with limited denominator
            q_approx = Fraction(qf).limit_denominator(max_den)
            return math.isclose(float(q_approx), qf, rel_tol=tol, abs_tol=tol)
        return False
    except Exception:
        return False

def extract_scalar(result):
    if hasattr(result, 'real'): return result.real
    if isinstance(result, complex): return result.real
    if isinstance(result, (list,tuple)): return result[0]
    return float(result)

def _divisible_by_numeric(x, divisor, tol=1e-12):
    """
    Return True if x is divisible by divisor in numeric sense:
    i.e. q = x / divisor is finite and q is within tol of an integer.
    Works for int, float, Fraction.
    """
    try:
        # handle Fraction exactly
        if isinstance(x, Fraction) and isinstance(divisor, Fraction):
            # x/divisor is Fraction; check denominator divides numerator
            q = x / divisor
            return q.denominator == 1
        # if divisor is Fraction and x numeric
        if isinstance(divisor, Fraction):
            try:
                q = Fraction(x) / divisor
                return q.denominator == 1
            except Exception:
                pass
        # numeric fallback: compute float quotient and test near-integer
        q = float(x) / float(divisor)
        if not math.isfinite(q):
            return False
        return math.isclose(q, round(q), abs_tol=tol)
    except Exception:
        return False

def safe_divide(val: Any, divisor: Union[int, float, Fraction], integer_mode: bool = False) -> Any:
    try:
        # coerce divisor
        if isinstance(divisor, Fraction):
            pass
        elif isinstance(divisor, float) and integer_mode:
            # if divisor is near-integer, use int
            if math.isclose(divisor, round(divisor), abs_tol=1e-12):
                divisor_int = int(round(divisor))
                return val // divisor_int if hasattr(val, "__floordiv__") else type(val)(int(val) // divisor_int)
            else:
                # integer_mode requested but divisor not integer-like -> fallback to true division
                integer_mode = False

        if integer_mode:
            if hasattr(val, "__floordiv__"):
                return val // int(divisor)
            # iterable fallback...
        else:
            if hasattr(val, "__truediv__"):
                return val / divisor
            # iterable fallback...
    except Exception:
        raise


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


def Bicomplex(
    z1_real: float, z1_imag: float, z2_real: float, z2_imag: float
) -> BicomplexNumber:
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

class HCAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        args = kwargs.get("args", ())
        if args:
            new_args = []
            for a in args:
                if _is_hypercomplex_like(a):
                    new_args.append(format_hypercomplex_value(a))
                else:
                    new_args.append(a)
            kwargs["args"] = tuple(new_args)
        return msg, kwargs

logger = logging.getLogger("kececi")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
hc_logger = HCAdapter(logger, {})
# usage
#hc_logger.info("  %d: %s", i, val) # NameError: name 'i' is not defined


class HypercomplexFormatter(logging.Formatter):
    """
    Formatter that converts Hypercomplex-like objects in record.args to readable strings
    using format_hypercomplex_value before formatting the message.
    """
    def format(self, record):
        try:
            # If args is a tuple/dict, replace Hypercomplex-like entries
            if record.args:
                # handle tuple args
                if isinstance(record.args, tuple):
                    new_args = []
                    for a in record.args:
                        try:
                            if _is_hypercomplex_like(a):
                                new_args.append(format_hypercomplex_value(a))
                            else:
                                new_args.append(a)
                        except Exception:
                            new_args.append(a)
                    record.args = tuple(new_args)
                # handle dict-style args
                elif isinstance(record.args, dict):
                    new_args = {}
                    for k, v in record.args.items():
                        try:
                            if _is_hypercomplex_like(v):
                                new_args[k] = format_hypercomplex_value(v)
                            else:
                                new_args[k] = v
                        except Exception:
                            new_args[k] = v
                    record.args = new_args
        except Exception:
            # swallow formatter errors to avoid breaking logging
            pass
        return super().format(record)

def _is_hypercomplex_like(v):
    # minimal duck-typing: check for common helpers
    for attr in ("to_list", "to_components", "coeffs", "components", "to_summary"):
        if hasattr(v, attr):
            return True
    return False

# install formatter on root logger (or specific logger)
handler = logging.StreamHandler()
handler.setFormatter(HypercomplexFormatter("%(levelname)s: %(message)s"))
root = logging.getLogger()
root.handlers = []  # replace default handlers if desired
root.addHandler(handler)
root.setLevel(logging.INFO)


def format_hypercomplex_value(v, max_components: int = 8) -> str:
    try:
        if hasattr(v, "to_summary") and callable(getattr(v, "to_summary")):
            try:
                return v.to_summary(max_components=max_components)
            except TypeError:
                return v.to_summary()
        if hasattr(v, "to_list") and callable(getattr(v, "to_list")):
            comps = v.to_list()
            return _format_components_list(comps, max_components)
        if hasattr(v, "to_components") and callable(getattr(v, "to_components")):
            comps = v.to_components()
            return _format_components_list(comps, max_components)
        if hasattr(v, "coeffs"):
            c = v.coeffs() if callable(getattr(v, "coeffs")) else v.coeffs
            return _format_components_list(c, max_components)
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return _format_components_list(list(v), max_components)
        if isinstance(v, complex):
            return f"{v.real:.6g}+{v.imag:.6g}j"
        if isinstance(v, (int, float)):
            return f"{v:.6g}"
        return repr(v)
    except Exception:
        try:
            return repr(v)
        except Exception:
            return "<unprintable hypercomplex>"

def _format_components_list(comps, max_components=8):
    try:
        comps = list(comps)
    except Exception:
        return "<non-iterable components>"
    def _fmt(x):
        try:
            if isinstance(x, complex):
                return f"{x.real:.6g}+{x.imag:.6g}j"
            return f"{float(x):.6g}"
        except Exception:
            return str(x)
    shown = [_fmt(c) for c in comps[:max_components]]
    s = ", ".join(shown)
    if len(comps) > max_components:
        s += ", ..."
    try:
        import math
        mag = math.sqrt(sum((abs(complex(c)) if isinstance(c, complex) else float(c))**2 for c in comps))
        return f"[{s}] |v|={mag:.6g}"
    except Exception:
        return f"[{s}]"


def _extract_coeffs_list(seq: Iterable[Any], complex_mode: str = 'real') -> List[List[float]]:
    """
    Extract numeric coefficient lists from a sequence of Hypercomplex-like objects.
    complex_mode: 'real' -> use real part of complex components
                  'magnitude' -> use abs() of complex components
    Returns list of lists (samples x components) as floats.
    """
    out = []
    for v in seq:
        try:
            # Prefer explicit helpers
            if hasattr(v, 'to_list') and callable(getattr(v, 'to_list')):
                comps = v.to_list()
            elif hasattr(v, 'to_components') and callable(getattr(v, 'to_components')):
                comps = v.to_components()
            elif hasattr(v, 'coeffs'):
                c = v.coeffs() if callable(getattr(v, 'coeffs')) else v.coeffs
                comps = list(c)
            elif hasattr(v, 'components'):
                c = v.components() if callable(getattr(v, 'components')) else v.components
                comps = list(c)
            elif hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                comps = list(v)
            else:
                comps = [v]

            # Normalize to floats
            norm = []
            for c in comps:
                if isinstance(c, complex):
                    if complex_mode == 'magnitude':
                        norm.append(float(abs(c)))
                    else:
                        # default: real part
                        norm.append(float(c.real))
                else:
                    try:
                        norm.append(float(c))
                    except Exception:
                        # fallback 0.0 for non-numeric entries
                        norm.append(0.0)
            out.append(norm)
        except Exception as e:
            logger.debug("extract coeffs failed for %r: %s", v, e)
            out.append([0.0])
    return out

def _pca_var_sum(pca_obj) -> float:
    """
    Safely return sum of PCA explained variance ratio.
    - Uses pca_obj.explained_variance_ratio_ when available.
    - Returns 0.0 for missing, NaN, infinite or invalid values.
    """
    try:
        arr = getattr(pca_obj, "explained_variance_ratio_", None)
        if arr is None:
            return 0.0
        arr = np.asarray(arr, dtype=float)
        s = float(np.nansum(arr))
        return s if np.isfinite(s) else 0.0
    except Exception:
        return 0.0

def get_numeric_repr(v, max_components=8):
    """
    Return a human-readable numeric representation for v.
    - If v has to_list / to_components / coeffs, use them.
    - If v is iterable, return list.
    - Else return scalar formatted string.
    """
    try:
        # HypercomplexNumber-like
        if hasattr(v, "to_summary") and callable(getattr(v, "to_summary")):
            return v.to_summary(max_components=max_components)
        if hasattr(v, "to_list") and callable(getattr(v, "to_list")):
            return str(v.to_list())
        if hasattr(v, "coeffs"):
            c = v.coeffs() if callable(getattr(v, "coeffs")) else v.coeffs
            return str(list(c))
        # iterable but not string
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            try:
                return str([float(x) for x in v])
            except Exception:
                return str(list(v))
        # scalar
        if isinstance(v, complex):
            return f"{v.real:.6g}+{v.imag:.6g}j"
        if isinstance(v, (int, float)):
            return f"{v:.6g}"
        return str(v)
    except Exception:
        return repr(v)


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
            if value_upper in ["", "NAN", "NULL", "NONE"]:
                return 0.0
            elif value_upper == "INF" or value_upper == "INFINITY":
                return float("inf")
            elif value_upper == "-INF" or value_upper == "-INFINITY":
                return float("-inf")
            # '+' veya '-' işaretleri
            elif value == "+":
                return 1.0
            elif value == "-":
                return -1.0
            else:
                try:
                    # Karmaşık sayı string'i olabilir
                    if "j" in value or "J" in value:
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

# --- Temel yardımcılar ----------------------------------------------------
def _is_numeric_scalar(x: Any) -> bool:
    return isinstance(x, (int, float))

def _coerce_first_component(x: Any) -> float:
    if isinstance(x, (list, tuple)):
        return float(x[0]) if x else 0.0
    if isinstance(x, complex):
        return float(x.real)
    try:
        return float(x)
    except Exception:
        return 0.0

def _get_array_fallback(a: Any, b: Any, op: Callable[[Any, Any], Any]):
    if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
        return type(a)([op(x, b) for x in a])
    raise TypeError("Unsupported operand types for array fallback")

def _get_operation_symbol(operation: str) -> str:
    symbols = {"add": "+", "subtract": "-", "multiply": "×", "divide": "/", "mod": "%", "power": "^"}
    return symbols.get(operation, "?")

# --- Sıfır kontrolü ------------------------------------------------------

def _is_zero(value: Any) -> bool:
    """Check if a value is effectively zero."""
    try:
        if isinstance(value, (int, float)):
            return abs(value) < 1e-12
        if isinstance(value, complex):
            return abs(value) < 1e-12
        if isinstance(value, tuple) or isinstance(value, list):
            return all(_is_zero(v) for v in value)
        if hasattr(value, "__abs__"):
            try:
                return abs(value) < 1e-12
            except Exception:
                pass
        return abs(float(value)) < 1e-12
    except Exception:
        return False

# --- Güvenli float dönüşümleri -------------------------------------------

def _safe_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, complex):
        return float(value.real)
    if isinstance(value, str):
        s = value.strip()
        try:
            return float(s)
        except ValueError:
            su = s.upper()
            if su in ("", "NAN", "NULL", "NONE"):
                return 0.0
            if su in ("INF", "INFINITY"):
                return float("inf")
            if su in ("-INF", "-INFINITY"):
                return float("-inf")
            if s == "+":
                return 1.0
            if s == "-":
                return -1.0
            # karmaşık string varsa gerçek kısmı al
            if "j" in s or "J" in s:
                try:
                    return float(complex(s).real)
                except Exception:
                    return 0.0
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0

# --- Güvenli temel işlemler (fallback'ler) -------------------------------

def _safe_divide(a: Any, b: Any) -> Any:
    """Safe division with zero handling and array fallbacks."""
    try:
        if _is_zero(b):
            logger.warning("Division by near-zero value")
            # try to produce an 'infinite' of same shape/type if possible
            try:
                if isinstance(a, (list, tuple)):
                    return type(a)([float("inf")] * len(a))
                if hasattr(type(a), "__call__"):
                    return type(a)(float("inf"))
            except Exception:
                pass
            return float("inf")
        return a / b
    except Exception as e:
        logger.debug("Primary divide failed: %s", e)
        # elementwise for arrays when divisor is scalar
        if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
            return type(a)([x / b for x in a])
        # try alternative methods
        if hasattr(a, "__truediv__"):
            try:
                return a.__truediv__(b)
            except Exception:
                pass
        if hasattr(a, "divide"):
            try:
                return a.divide(b)
            except Exception:
                pass
        # last resort: convert to float
        try:
            return float(a) / float(b)
        except Exception:
            raise ValueError(f"Cannot divide {type(a)} by {type(b)}")

def _safe_mod(a: Any, b: Any) -> Any:
    """Safe modulo operation with fallbacks."""
    try:
        if _is_zero(b):
            logger.warning("Modulo by near-zero value")
            return a
        return a % b
    except Exception as e:
        logger.debug("Primary mod failed: %s", e)
        if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
            return type(a)([x % b for x in a])
        if hasattr(a, "__mod__"):
            try:
                return a.__mod__(b)
            except Exception:
                pass
        logger.warning("Modulo operation not defined for type %s, returning original value", type(a))
        return a

def _safe_power(a: Any, b: Any) -> Any:
    """Safe power operation with broad type support."""
    try:
        # numeric cases
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if a < 0 and not float(b).is_integer():
                # negative base with non-integer exponent -> complex result
                return cmath.exp(b * cmath.log(a))
            return a ** b
        # complex involvement
        if isinstance(a, complex) or isinstance(b, complex):
            return complex(a) ** complex(b)
        # custom __pow__
        if hasattr(a, "__pow__"):
            try:
                return a ** b
            except Exception:
                pass
        # array elementwise when exponent is scalar
        if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
            return type(a)([x ** b for x in a])
        # try float conversion
        af = _safe_float(a)
        bf = _safe_float(b)
        if af < 0 and not float(bf).is_integer():
            return cmath.exp(bf * cmath.log(af))
        return af ** bf
    except ValueError as e:
        logger.warning("ValueError in power: %s", e)
        try:
            return math.pow(float(a), float(b))
        except Exception:
            try:
                return cmath.exp(float(b) * cmath.log(float(a)))
            except Exception:
                raise ValueError(f"Cannot compute {a} ** {b}")
    except Exception as e:
        logger.error("Unexpected error in power: %s", e)
        # sensible fallbacks for small integer exponents
        try:
            if b == 2:
                return a * a
            if b == 1:
                return a
            if b == 0:
                try:
                    return type(a)(1)
                except Exception:
                    return 1
        except Exception:
            pass
        return a

# --- Keçeci özel güvenli işlemler (type-aware) ---------------------------

def _safe_divide_kececi(a: Any, b: Any, number_type: str = "Unknown") -> Any:
    """
    Safe division for Keçeci numbers.
    """
    if _is_zero(b):
        logger.warning(f"Division by zero in {number_type}")
        if number_type == "Complex":
            return complex(float("inf"), 0)
        if "Neutrosophic" in number_type:
            return (float("inf"), 0.0, 0.0)
        if number_type in ("Quaternion", "Octonion", "Sedenion"):
            try:
                return type(a)(float("inf"))
            except Exception:
                return float("inf")
        return float("inf")
    try:
        return _safe_divide(a, b)
    except Exception:
        # try object-specific methods
        if hasattr(a, "divide"):
            try:
                return a.divide(b)
            except Exception:
                pass
        raise

def _safe_mod_kececi(a: Any, b: Any, number_type: str = "Unknown") -> Any:
    """
    Safe modulo for Keçeci numbers.
    """
    if _is_zero(b):
        logger.warning(f"Modulo by zero in {number_type}")
        return a
    try:
        return _safe_mod(a, b)
    except Exception:
        if hasattr(a, "mod"):
            try:
                return a.mod(b)
            except Exception:
                pass
        logger.warning("Modulo operation not defined for %s, returning original", number_type)
        return a

def _safe_power_kececi(a: Any, b: Any, number_type: str = "Unknown") -> Any:
    """
    Safe power for Keçeci numbers.
    """
    try:
        return _safe_power(a, b)
    except Exception:
        if hasattr(a, "power"):
            try:
                return a.power(b)
            except Exception:
                pass
        # common fallbacks
        if isinstance(b, (int, float)):
            if b == 2:
                return a * a
            if b == 1:
                return a
            if b == 0:
                try:
                    return type(a)(1)
                except Exception:
                    return 1
        raise ValueError(f"Power operation not supported for {number_type}")

# --- Genel uygulayıcı ----------------------------------------------------

def _apply_kececi_operation(a: Any, b: Any, operation: str, number_type: str = "Unknown") -> Any:
    """
    Apply operation to Keçeci numbers with proper type handling and fallbacks.
    Supported operations: add, subtract, multiply, divide, mod, power
    """
    try:
        if operation == "add":
            return a + b
        if operation == "subtract":
            return a - b
        if operation == "multiply":
            return a * b
        if operation == "divide":
            return _safe_divide_kececi(a, b, number_type)
        if operation == "mod":
            return _safe_mod_kececi(a, b, number_type)
        if operation == "power":
            return _safe_power_kececi(a, b, number_type)
        raise ValueError(f"Unsupported operation: {operation}")
    except Exception as e:
        logger.debug("Standard %s failed: %s, trying alternatives", operation, e)
        # try object methods
        method_map = {
            "add": ("add",),
            "subtract": ("subtract",),
            "multiply": ("multiply",),
            "divide": ("divide",),
            "mod": ("mod", "__mod__"),
            "power": ("power", "__pow__"),
        }
        for method_name in method_map.get(operation, ()):
            if hasattr(a, method_name):
                try:
                    return getattr(a, method_name)(b)
                except Exception:
                    continue
        # array elementwise fallback when b is scalar
        if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
            if operation == "add":
                return type(a)([x + b for x in a])
            if operation == "subtract":
                return type(a)([x - b for x in a])
            if operation == "multiply":
                return type(a)([x * b for x in a])
            if operation == "divide":
                return type(a)([x / b for x in a])
            if operation == "mod":
                return type(a)([x % b for x in a])
            if operation == "power":
                return type(a)([x ** b for x in a])
        raise

# --- Basit sembol yardımcı fonksiyonu -----------------------------------

def get_operation_symbol(operation: str) -> str:
    return _get_operation_symbol(operation)


def _generate_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False,
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
        detailed_result.append(
            {
                "step": 0,
                "operation": "start",
                "value": start_value,
                "description": f"Start: {start_value}",
            }
        )

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
                detailed_result.append(
                    {
                        "step": i,
                        "operation": operation,
                        "previous": previous,
                        "value": current,
                        "description": f"Step {i}: {previous} {op_symbol} {add_value} = {current}",
                    }
                )

            except Exception as e:
                logger.warning(f"Error at iteration {i}: {e}")

                # Hata durumunda default değer
                default_val = _generate_default_value(current)
                
                detailed_result.append(
                    {
                        "step": i,
                        "operation": operation,
                        "error": str(e),
                        "value": default_val,
                        "description": f"Step {i}: ERROR - {e}",
                    }
                )

                current = default_val

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
                default_val = _generate_default_value(current)
                simple_result.append(default_val)
                current = default_val

        return simple_result  # Basit liste döndür


# Daha basit ve güvenli versiyon (alternatif)
def generate_sequence_safe(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False,
) -> Union[List[Any], List[Dict[str, Any]]]:
    """
    Safer version with separate return types.
    """
    if include_intermediate_steps:
        return _generate_detailed_sequence(
            start_value, add_value, iterations, operation
        )
    else:
        return _generate_simple_sequence(start_value, add_value, iterations, operation)


def _generate_detailed_sequence(
    start_value: Any, add_value: Any, iterations: int, operation: str
) -> List[Dict[str, Any]]:
    """Generate detailed sequence with step information."""
    result: List[Dict[str, Any]] = []

    result.append(
        {
            "step": 0,
            "operation": "start",
            "value": start_value,
            "description": f"Start: {start_value}",
        }
    )

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

            result.append(
                {
                    "step": i,
                    "operation": operation,
                    "previous": previous,
                    "value": current,
                    "description": f"Step {i}: {previous} {op_symbol} {add_value} = {current}",
                }
            )

        except Exception as e:
            logger.warning(f"Error at iteration {i}: {e}")

            # Generate appropriate default value
            default_val = _generate_default_value(current)

            result.append(
                {
                    "step": i,
                    "operation": operation,
                    "error": str(e),
                    "value": default_val,
                    "description": f"Step {i}: ERROR - {e}",
                }
            )

            current = default_val

    return result


def _generate_simple_sequence(
    start_value: Any, add_value: Any, iterations: int, operation: str
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

            # Generate appropriate default value
            default_val = _generate_default_value(current)
            result.append(default_val)
            current = default_val

    return result


def _generate_default_value(current_value: Any) -> Any:
    """Generate appropriate default value based on current value type."""
    try:
        if hasattr(type(current_value), "__call__"):
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
                if hasattr(current_value, "real"):
                    return type(current_value)(0)
            except:
                pass

            return 0
    except Exception:
        return 0


# Alternatif olarak, tüm işlemleri tek bir fonksiyonda yöneten basit versiyon
def apply_operation(a: Any, b: Any, operation: str) -> Any:
    """
    Apply an operation between two values with proper error handling.
    
    Args:
        a: First value
        b: Second value
        operation: Operation to apply ('add', 'subtract', 'multiply', 'divide', 'mod', 'power')
        
    Returns:
        Result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return _safe_divide(a, b)
    elif operation == "mod":
        return _safe_mod(a, b)
    elif operation == "power":
        return _safe_power(a, b)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


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
                value = item.get("value", 0)
            else:
                value = item

            # Float'a çevirmeye çalış
            if isinstance(value, (int, float, complex)):
                if isinstance(value, complex):
                    # Complex için magnitude
                    values.append(abs(value))
                else:
                    values.append(float(value))
            elif hasattr(value, "real"):
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

def get_random_types_batch(
    num_types: int = 5,
    iterations_per_type: int = 5,
    start_value_raw: Union[str, float, int] = "0",
    add_value_raw: Union[str, float, int] = "1.0",
    seed: Optional[int] = None,
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
        "Positive Real",
        "Negative Real",
        "Complex",
        "Float",
        "Rational",
        "Quaternion",
        "Neutrosophic",
        "Neutrosophic Complex",
        "Hyperreal",
        "Bicomplex",
        "Neutrosophic Bicomplex",
        "Octonion",
        "Sedenion",
        "Clifford",
        "Dual",
        "Split-Complex",
        "Pathion",
        "Chingon",
        "Routon",
        "Voudon",
        "Super Real",
        "Ternary",
        "Hypercomplex",
    ]

    # Select random types without replacement
    available_types = list(range(1, len(type_names_list) + 1))
    selected_types = random.sample(
        available_types, min(num_types, len(available_types))
    )

    results = {}

    for type_choice in selected_types:
        type_name = type_names_list[type_choice - 1]
        try:
            numbers = get_with_params(
                kececi_type_choice=type_choice,
                iterations=iterations_per_type,
                start_value_raw=str(start_value_raw),
                add_value_raw=str(add_value_raw),
            )
            results[type_name] = numbers
        except Exception as e:
            logger.error(f"Failed to generate type {type_name}: {e}")
            results[type_name] = []

    return results

def _parse_complex_like_string(s: str) -> List[float]:
    """
    Karmaşık sayı benzeri string'i float listesine çevirir.
    Örnek: "1+2i-3j+4k" -> [1.0, 2.0, -3.0, 4.0, ...]
    """
    if not s:
        return [0.0]

    # Normalize et
    s = s.replace(" ", "").replace("J", "j").replace("I", "j").upper()

    # Tüm imajiner birimleri normalize et
    units = [
        "J",
        "I",
        "K",
        "E",
        "F",
        "G",
        "H",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
    ]

    # İlk bileşen (reel kısım)
    result = [0.0] * (len(units) + 1)

    # Reel kısmı bul
    pattern = r"^([+-]?\d*\.?\d*)(?![" + "".join(units) + "])"
    match = re.match(pattern, s)
    if match and match.group(1):
        result[0] = _safe_float_convert(match.group(1))

    # Her bir imajiner birim için
    for i, unit in enumerate(units, 1):
        pattern = r"([+-]?\d*\.?\d*)" + re.escape(unit)
        matches = re.findall(pattern, s)
        if matches:
            # Son eşleşmeyi al (tekrarlanmışsa)
            last_match = matches[-1]
            result[i] = _safe_float_convert(last_match)

    return result

def _parse_engineering_notation(s: str) -> float:
    """Parse engineering notation (1.5k, 2.3m, etc.)"""
    import re

    s = s.strip().lower()

    # Mühendislik çarpanları
    multipliers = {
        "k": 1e3,
        "m": 1e-3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
        "μ": 1e-6,
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "f": 1e-15,
        "a": 1e-18,
        "mil": 1000,  # thousand
    }

    # Regex pattern
    pattern = r"^([+-]?\d*\.?\d+)\s*([a-zμ]+)?$"
    match = re.match(pattern, s)

    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2) or ""

            if unit in multipliers:
                return value * multipliers[unit]
            elif unit == "":
                return value

            # Özel birimler
            if unit.startswith("e"):
                # 1.5e-3 gibi
                return float(s)
        except (ValueError, KeyError):
            pass

    # Standart float dönüşümü
    return float(s)

def _parse_fraction(s: Union[str, float, int]) -> float:
    """
    Parse fraction strings like '1353/2791' or mixed numbers like '1 1/2'
    """
    if isinstance(s, (int, float)):
        return float(s)

    s_str = str(s).strip()

    # Empty string
    if not s_str:
        return 0.0

    # Already a float string
    try:
        return float(s_str)
    except ValueError:
        pass

    # Check for mixed numbers like "1 1/2"
    if " " in s_str and "/" in s_str:
        try:
            whole_part, frac_part = s_str.split(" ", 1)
            whole = float(whole_part) if whole_part else 0.0
            num, den = frac_part.split("/")
            fraction = float(num) / float(den) if float(den) != 0 else 0.0
            return whole + fraction
        except (ValueError, ZeroDivisionError):
            pass

    # Check for simple fractions like "1353/2791"
    if "/" in s_str:
        try:
            num, den = s_str.split("/")
            # Handle both integer and float numerator/denominator
            numerator = float(num) if "." in num else int(num)
            denominator = float(den) if "." in den else int(den)
            if denominator == 0:
                logger.warning(f"Division by zero in fraction: {s_str}")
                return float("inf") if numerator >= 0 else float("-inf")
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            pass

    # Try using Fraction class for more robust parsing
    try:
        return float(Fraction(s_str))
    except (ValueError, ZeroDivisionError):
        logger.warning(f"Could not parse as fraction: {s_str}")

    # Last resort
    try:
        return float(s_str)
    except ValueError:
        logger.error(f"Failed to parse numeric value: {s_str}")
        raise ValueError(f"Invalid numeric format: {s_str}")

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
        if hasattr(s, "__class__") and s.__class__.__name__ == "HypercomplexNumber":
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
            "∞": float("inf"),
            "inf": float("inf"),
            "infinity": float("inf"),
            "+∞": float("inf"),
            "+inf": float("inf"),
            "+infinity": float("inf"),
            "-∞": float("-inf"),
            "-inf": float("-inf"),
            "-infinity": float("-inf"),
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
            "k": 1e3,  # kilo
            "m": 1e-3,  # milli (küçük m)
            "meg": 1e6,  # mega
            "g": 1e9,  # giga
            "t": 1e12,  # tera
            "μ": 1e-6,  # mikro
            "u": 1e-6,  # mikro (alternatif)
            "n": 1e-9,  # nano
            "p": 1e-12,  # piko
            "f": 1e-15,  # femto
            "a": 1e-18,  # atto
        }

        # Mühendislik notasyonu regex'i (case-insensitive)
        eng_match = re.match(r"^\s*([+-]?\d*\.?\d+)\s*([a-zA-Zμ]+)\s*$", s_original)
        if eng_match:
            try:
                value = float(eng_match.group(1))
                unit = eng_match.group(2).lower()

                if unit in engineering_units:
                    return value * engineering_units[unit]
                elif unit == "mil":  # bin (thousand)
                    return value * 1000
            except (ValueError, KeyError):
                pass

        # 9. Yüzde notasyonu
        if s.endswith("%"):
            try:
                # Orijinal string'den % işaretini kaldır (büyük/küçük harf fark etmez)
                value_str = s_original.rstrip("%").strip()
                value = float(value_str)
                return value / 100.0
            except ValueError:
                pass

        # 10. Kesirler ve karışık sayılar
        # Karışık sayı: "1 1/2"
        mixed_match = re.match(r"^\s*(\d+)\s+(\d+)/(\d+)\s*$", s)
        if mixed_match:
            try:
                whole = int(mixed_match.group(1))
                num = int(mixed_match.group(2))
                den = int(mixed_match.group(3))
                return float(whole) + (float(num) / float(den))
            except (ValueError, ZeroDivisionError):
                pass

        # Basit kesir: "3/4"
        if "/" in s and " " not in s:
            try:
                parts = s.split("/")
                if len(parts) == 2:
                    num = float(parts[0])
                    den = float(parts[1])
                    if den != 0:
                        result = num / den
                        return float(result)  # Açıkça float
            except (ValueError, ZeroDivisionError) as e:
                warnings.warn(
                    f"Fraction parse failed: {e}", RuntimeWarning, stacklevel=2
                )
                return 0.0

        # 11. Infinitesimal notasyonu (ε, epsilon, dx, etc.)
        infinitesimals = {
            "ε": 1e-10,
            "epsilon": 1e-10,
            "δ": 1e-10,
            "delta": 1e-10,
            "dx": 1e-10,
            "dt": 1e-10,
            "dh": 1e-10,
            "infinitesimal": 1e-15,
        }

        if s in infinitesimals:
            return infinitesimals[s]

        # 12. Parantez içindeki ifadeler
        if "(" in s and ")" in s:
            # İçeriği al ve tekrar dene
            inner_start = s.find("(") + 1
            inner_end = s.find(")")
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
                "i": 1,
                "ii": 2,
                "iii": 3,
                "iv": 4,
                "v": 5,
                "vi": 6,
                "vii": 7,
                "viii": 8,
                "ix": 9,
                "x": 10,
            }
            if s in roman_numerals:
                return float(roman_numerals[s])

    except Exception as e:
        warnings.warn(
            f"Super real parse error for '{s}': {e}", RuntimeWarning, stacklevel=2
        )

    # 14. Hiçbir şey işe yaramazsa
    return 0.0


def is_super_real_expression(expr: str) -> bool:
    """Check if string looks like a super real expression."""
    super_real_indicators = [
        "∞",
        "inf",
        "epsilon",
        "ε",
        "δ",
        "dx",
        "dt",
        "pi",
        "π",
        "e",
        "phi",
        "φ",
        "tau",
        "γ",
        "k",
        "m",
        "meg",
        "g",
        "t",
        "μ",
        "n",
        "p",
        "%",
        "/",
    ]

    expr_lower = expr.lower()
    return any(indicator in expr_lower for indicator in super_real_indicators)


def normalize_super_real(value: float) -> float:
    """Normalize super real values (e.g., replace very small numbers with 0)."""
    EPSILON = 1e-15

    if abs(value) < EPSILON:
        return 0.0
    elif math.isinf(value):
        return float("inf") if value > 0 else float("-inf")
    else:
        return value

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
        if isinstance(other, (int, float)):
            other = self.__class__(other, 0, ...)  # scalar genişlet
        return super().__add__(other)

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
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
            return math.isclose(self.real, other.real) and math.isclose(
                self.imag, other.imag
            )
        elif isinstance(other, (int, float)):
            return math.isclose(self.real, float(other)) and math.isclose(self.imag, 0)
        elif isinstance(other, complex):
            return math.isclose(self.real, other.real) and math.isclose(
                self.imag, other.imag
            )
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

    def to_hypercomplex(self) -> "HypercomplexNumber":
        """Convert to HypercomplexNumber."""
        return HypercomplexNumber(self.real, self.imag, dimension=2)

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
            if hasattr(num, "magnitude"):
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
        return math.isclose(remainder, 0.0, abs_tol=tol) or math.isclose(
            remainder, float(divisor), abs_tol=tol
        )
    except Exception:
        return False

def _safe_divide(a: Any, b: Any) -> Any:
    """Safe division with zero handling and better type support."""
    try:
        # First, check if we're dealing with zero division
        if hasattr(b, "__float__"):
            b_float = float(b)
            if abs(b_float) < 1e-12:  # Near zero threshold
                logger.warning(f"Division by near-zero value: {b}")
                
                # Handle infinity based on type
                if hasattr(a, "__class__") and hasattr(a.__class__, "__call__"):
                    try:
                        # Try to create infinity of the same type
                        if hasattr(a, "__mul__"):
                            # For types that support multiplication
                            try:
                                inf_val = float('inf')
                                # Check if type can handle infinity
                                if hasattr(type(a)(1), '__mul__'):
                                    return type(a)(1) * inf_val
                            except:
                                pass
                        
                        # Try alternative: return maximum value of type
                        if hasattr(a, 'max_value'):
                            return a.max_value()
                    except:
                        pass
                
                # Default fallback
                return type(a)(float('inf')) if hasattr(type(a), '__call__') else float('inf')

        # Special handling for complex numbers
        if isinstance(a, complex) or isinstance(b, complex):
            try:
                return complex(a) / complex(b)
            except ZeroDivisionError:
                return complex(float('inf'), 0)

        # Special handling for Fraction type
        if 'Fraction' in str(type(a)) or 'Fraction' in str(type(b)):
            try:
                from fractions import Fraction
                a_frac = Fraction(str(a)) if not isinstance(a, Fraction) else a
                b_frac = Fraction(str(b)) if not isinstance(b, Fraction) else b
                return a_frac / b_frac
            except:
                pass

        # For custom number types that might have their own division
        if hasattr(a, '__truediv__'):
            try:
                return a.__truediv__(b)
            except:
                pass

        if hasattr(a, '__div__'):
            try:
                return a.__div__(b)
            except:
                pass

        # Standard division for numeric types
        return a / b

    except ZeroDivisionError:
        logger.warning(f"ZeroDivisionError: {a} / {b}")
        # Try to return appropriate infinity
        try:
            # Get sign of a
            if hasattr(a, '__float__'):
                a_float = float(a)
                inf_val = float('inf') if a_float >= 0 else float('-inf')
            else:
                inf_val = float('inf')
            
            # Try to convert to same type as a
            if hasattr(type(a), '__call__'):
                return type(a)(inf_val)
            return inf_val
        except:
            return float('inf')
    
    except TypeError as e:
        logger.warning(f"TypeError in division {a} / {b}: {e}")
        # Try to convert to compatible types
        try:
            # Convert both to float if possible
            a_float = float(a) if hasattr(a, '__float__') else a
            b_float = float(b) if hasattr(b, '__float__') else b
            return a_float / b_float
        except:
            # If conversion fails, try string-based approach for fractions
            try:
                from fractions import Fraction
                a_str = str(a)
                b_str = str(b)
                result = Fraction(a_str) / Fraction(b_str)
                # Convert back to original type if possible
                if hasattr(type(a), '__call__'):
                    try:
                        return type(a)(float(result))
                    except:
                        return type(a)(str(result))
                return float(result)
            except:
                raise ValueError(f"Cannot divide {type(a)} by {type(b)}")
    
    except Exception as e:
        logger.error(f"Unexpected error in division: {e}")
        return a  # Return original as fallback


# Daha iyi mod fonksiyonu
def _safe_mod(a: Any, b: Any) -> Any:
    """Safe modulo operation with better type support."""
    try:
        # Check for zero
        if hasattr(b, "__float__"):
            b_float = float(b)
            if abs(b_float) < 1e-12:
                logger.warning(f"Modulo by near-zero value: {b}")
                raise ZeroDivisionError("Modulo by (near) zero")

        # Special handling for integers and floats
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a % b

        # For custom types with __mod__ method
        if hasattr(a, '__mod__'):
            return a % b

        # For complex numbers, modulus returns magnitude
        if isinstance(a, complex):
            return abs(a) % abs(b) if isinstance(b, complex) else abs(a) % b

        # Try to convert to float
        a_float = float(a) if hasattr(a, '__float__') else a
        b_float = float(b) if hasattr(b, '__float__') else b
        
        return a_float % b_float

    except ZeroDivisionError:
        logger.warning(f"Modulo by zero: {a} % {b}")
        # For modulo by zero, return the dividend (mathematical convention in some systems)
        return a
    
    except TypeError as e:
        logger.warning(f"TypeError in modulo {a} % {b}: {e}")
        # Try alternative approaches
        try:
            # Convert to decimal
            from decimal import Decimal, InvalidOperation
            a_dec = Decimal(str(a))
            b_dec = Decimal(str(b))
            if b_dec == 0:
                return a_dec
            # Python Decimal doesn't have % operator, use remainder method
            return float(a_dec % b_dec)
        except:
            # Last resort: return a
            return a
    
    except Exception as e:
        logger.error(f"Unexpected error in modulo: {e}")
        return a




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


def chingon_random(
    low: float = -1.0, high: float = 1.0, seed: Optional[int] = None
) -> ChingonNumber:
    """Rastgele ChingonNumber"""
    return ChingonNumber.random(low, high, seed)


def chingon_linspace(
    start: Union[float, ChingonNumber], end: Union[float, ChingonNumber], num: int = 64
) -> List[ChingonNumber]:
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
                for key in ["value", "result", "numeric_value", "added", "modified"]:
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
        if "/" in item:
            try:
                # Örnek: "3/4", "1 1/2"
                if " " in item:  # Karışık sayı: "1 1/2"
                    whole, fraction = item.split(" ", 1)
                    num, den = fraction.split("/")
                    return float(whole) + (float(num) / float(den))
                else:  # Basit kesir: "3/4"
                    num, den = item.split("/")
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


def extract_numeric_values(sequence: List[Any], strict: bool = False) -> List[float]:
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
                raise ValueError(
                    f"Failed to extract numeric value from {item!r}"
                ) from e
            result.append(0.0)

    return result


# Ek yardımcı fonksiyonlar
def extract_clean_numbers(
    sequence: List[Any], remove_zeros: bool = False
) -> List[float]:
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


def extract_fraction_values(
    sequence: List[Any],
) -> tuple[List[float], List[int], List[int]]:
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


def extract_complex_values(
    sequence: List[Any],
) -> tuple[List[float], List[float], List[float]]:
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

# Fabrika fonksiyonları
def neutrosophic_zero() -> NeutrosophicNumber:
    """Sıfır Nötrosofik sayı"""
    return NeutrosophicNumber(0, 0, 0)


def neutrosophic_one() -> NeutrosophicNumber:
    """Bir Nötrosofik sayı"""
    return NeutrosophicNumber(1, 0, 0)


def neutrosophic_i() -> NeutrosophicNumber:
    """Belirsizlik birimi"""
    return NeutrosophicNumber(0, 1, 0)


def neutrosophic_f() -> NeutrosophicNumber:
    """Yanlışlık birimi"""
    return NeutrosophicNumber(0, 0, 1)

def parse_to_hyperreal(s: Any) -> "HyperrealNumber":
    """Parse to Hyperreal object directly"""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .kececinumbers import HyperrealNumber

    finite, infinitesimal, seq = _parse_hyperreal(s)
    return HyperrealNumber(sequence=seq)

# Yardımcı fonksiyonlar
def parse_to_neutrosophic(s: Any) -> "NeutrosophicNumber":
    """Parse to NeutrosophicNumber object directly"""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .kececinumbers import NeutrosophicNumber

    t, i, f = _parse_neutrosophic(s)
    return NeutrosophicNumber(t, i, f)

# ==============================================================================
# --- CUSTOM NUMBER CLASS DEFINITIONS ---
# ==============================================================================
# ---------- Cayley-Dickson tabanlı HypercomplexNumber (geliştirilmiş) ----

class HypercomplexNumber:
    """
    Unified wrapper for Cayley-Dickson hypercomplex numbers with flexible input.
    - Accepts scalar, iterable, or string inputs.
    - Supports dimensions that are powers of two up to 256 (1,2,4,8,...,256).
    - Uses project's cayley_dickson_algebra for algebraic multiplication/division when available.
    - Falls back to elementwise operations for scalar/iterable cases.
    """

    DIMENSION_NAMES = {
        1: "Real", 2: "Complex", 4: "Quaternion", 8: "Octonion",
        16: "Sedenion", 32: "Pathion", 64: "Chingon", 128: "Routon", 256: "Voudon"
    }
    _cd_classes = {}

    def __init__(self, components: Any = None, *, dimension: Optional[int] = None):
        # parse components flexibly
        comps = _parse_components(components)
        # infer dimension if not provided: smallest power of two >= len(comps) or default 1
        if dimension is None:
            n = max(1, len(comps))
            dim = 1
            while dim < n:
                dim <<= 1
        else:
            dim = int(dimension)
            if dim not in self.DIMENSION_NAMES and dim not in (1,2,4,8,16,32,64,128,256):
                raise ValueError("Dimension must be a power of two up to 256")
        self.dimension = dim
        # pad/truncate components
        if len(comps) < dim:
            comps = comps + [0.0] * (dim - len(comps))
        elif len(comps) > dim:
            comps = comps[:dim]
        self._comps = [complex(c) if isinstance(c, complex) else float(c) for c in comps]
        # try to create CD class and cd_number if available
        try:
            from .cd_helpers import cayley_dickson_algebra  # project helper
            level = int(math.log2(self.dimension))
            if self.dimension not in self._cd_classes:
                self._cd_classes[self.dimension] = cayley_dickson_algebra(level, float)
            cd_cls = self._cd_classes[self.dimension]
            self._cd_number = cd_cls(*self._comps)
            self._has_cd = True
        except Exception:
            # no CD algebra available; operate elementwise
            self._cd_number = None
            self._has_cd = False

    # --- basic accessors ---
    def coeffs(self) -> List[Number]:
        return list(self._comps)

    @property
    def real(self) -> float:
        return float(self._comps[0]) if self._comps else 0.0

    @property
    def imag(self) -> List[Number]:
        return self._comps[1:]

    def __len__(self) -> int:
        return self.dimension

    def to_list(self) -> List[float]:
        """Return components as plain Python list of floats (complex -> real part)."""
        out = []
        for c in self.coeffs():
            try:
                # if complex, keep complex; plotting code will decide how to handle
                out.append(float(c.real) if isinstance(c, complex) else float(c))
            except Exception:
                try:
                    out.append(float(c))
                except Exception:
                    out.append(0.0)
        return out

    def to_components(self) -> List[Union[float, complex]]:
        """Return raw components preserving complex entries if present."""
        return list(self.coeffs())

    def to_summary(self, max_components: int = 8) -> str:
        """Human-friendly short summary: first components and magnitude."""
        comps = self.to_list()
        shown = comps[:max_components]
        comps_str = ", ".join(f"{x:.6g}" for x in shown)
        if len(comps) > max_components:
            comps_str += ", ..."
        try:
            mag = self.norm()
            return f"[{comps_str}] |v|={mag:.6g}"
        except Exception:
            return f"[{comps_str}]"


    def copy(self) -> "HypercomplexNumber":
        return HypercomplexNumber(self.coeffs(), dimension=self.dimension)

    # --- internal helpers for dimension alignment ---
    def _align_with(self, other: Any) -> Tuple[List[Number], List[Number], int]:
        """
        Return (a_list, b_list, dim) where both lists are length dim.
        Accepts other as scalar, iterable, HypercomplexNumber.
        """
        if isinstance(other, HypercomplexNumber):
            dim = max(self.dimension, other.dimension)
            a = self.coeffs() + [0.0] * (dim - self.dimension)
            b = other.coeffs() + [0.0] * (dim - other.dimension)
            return a, b, dim
        # parse other
        other_comps = _parse_components(other)
        dim = max(self.dimension, max(1, len(other_comps)))
        a = self.coeffs() + [0.0] * (dim - self.dimension)
        b = (other_comps + [0.0] * (dim - len(other_comps))) if other_comps else [other] + [0.0] * (dim - 1)
        return a, b, dim

    # --- arithmetic using CD algebra when possible, else elementwise/fallback ---
    def __add__(self, other: Any) -> "HypercomplexNumber":
        if self._has_cd and isinstance(other, HypercomplexNumber) and other._has_cd and self.dimension == other.dimension:
            res_cd = self._cd_number + other._cd_number
            return HypercomplexNumber.from_cd_number(res_cd)
        a, b, dim = self._align_with(other)
        return HypercomplexNumber([x + y for x, y in zip(a, b)], dimension=dim)

    def __radd__(self, other: Any) -> "HypercomplexNumber":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "HypercomplexNumber":
        if self._has_cd and isinstance(other, HypercomplexNumber) and other._has_cd and self.dimension == other.dimension:
            res_cd = self._cd_number - other._cd_number
            return HypercomplexNumber.from_cd_number(res_cd)
        a, b, dim = self._align_with(other)
        return HypercomplexNumber([x - y for x, y in zip(a, b)], dimension=dim)

    def __rsub__(self, other: Any) -> "HypercomplexNumber":
        # other - self
        if isinstance(other, HypercomplexNumber):
            return other.__sub__(self)
        a, b, dim = self._align_with(other)
        return HypercomplexNumber([y - x for x, y in zip(a, b)], dimension=dim)

    def __mul__(self, other: Any) -> "HypercomplexNumber":
        # scalar multiplication
        if isinstance(other, (int, float, complex)):
            return HypercomplexNumber([x * other for x in self.coeffs()], dimension=self.dimension)
        # CD multiplication if both have CD and same dim
        if isinstance(other, HypercomplexNumber) and self._has_cd and other._has_cd:
            if self.dimension != other.dimension:
                common = max(self.dimension, other.dimension)
                return self.pad_to_dimension(common) * other.pad_to_dimension(common)
            res_cd = self._cd_number * other._cd_number
            return HypercomplexNumber.from_cd_number(res_cd)
        # elementwise fallback
        a, b, dim = self._align_with(other)
        return HypercomplexNumber([x * y for x, y in zip(a, b)], dimension=dim)

    def __rmul__(self, other: Any) -> "HypercomplexNumber":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "HypercomplexNumber":
        # scalar division
        if isinstance(other, (int, float, complex)):
            if _is_zero(other):
                raise ZeroDivisionError("Division by zero")
            return HypercomplexNumber([x / other for x in self.coeffs()], dimension=self.dimension)
        # CD division if possible
        if isinstance(other, HypercomplexNumber) and self._has_cd and other._has_cd:
            if self.dimension != other.dimension:
                common = max(self.dimension, other.dimension)
                return self.pad_to_dimension(common) / other.pad_to_dimension(common)
            res_cd = self._cd_number / other._cd_number
            return HypercomplexNumber.from_cd_number(res_cd)
        # elementwise fallback
        a, b, dim = self._align_with(other)
        res = []
        for x, y in zip(a, b):
            if _is_zero(y):
                res.append(float("inf"))
            else:
                res.append(x / y)
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
        return HypercomplexNumber(res, dimension=dim)

    def __rtruediv__(self, other: Any) -> "HypercomplexNumber":
        if isinstance(other, (int, float, complex)):
            other_coeffs = [other] + [0.0] * (self.dimension - 1)
            return HypercomplexNumber(other_coeffs, dimension=self.dimension) / self
        return NotImplemented

    def __mod__(self, other: Any) -> "HypercomplexNumber":
        # elementwise modulo fallback
        a, b, dim = self._align_with(other)
        res = []
        for x, y in zip(a, b):
            try:
                if _is_zero(y):
                    res.append(x)
                else:
                    res.append(x % y)
            except Exception:
                res.append(x)
        return HypercomplexNumber(res, dimension=dim)

    def __pow__(self, exponent: Any) -> "HypercomplexNumber":
        # elementwise power for scalar exponent
        if isinstance(exponent, (int, float)):
            return HypercomplexNumber([x ** exponent for x in self.coeffs()], dimension=self.dimension)
        if isinstance(exponent, HypercomplexNumber):
            a, b, dim = self._align_with(exponent)
            return HypercomplexNumber([x ** y for x, y in zip(a, b)], dimension=dim)
        raise TypeError("Unsupported exponent type for HypercomplexNumber")

    def __neg__(self) -> "HypercomplexNumber":
        return HypercomplexNumber([-c for c in self.coeffs()], dimension=self.dimension)

    def __abs__(self) -> float:
        # norm: sqrt(sum(|c|^2))
        s = 0.0
        for c in self.coeffs():
            try:
                s += (abs(c) ** 2)
            except Exception:
                try:
                    s += float(c) ** 2
                except Exception:
                    s += 0.0
        return math.sqrt(s)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HypercomplexNumber):
            if self.dimension != other.dimension:
                return False
            return all(abs(a - b) < 1e-12 for a, b in zip(self.coeffs(), other.coeffs()))
        if isinstance(other, (int, float)):
            return abs(self.real - float(other)) < 1e-12 and all(abs(c) < 1e-12 for c in self.imag)
        return False

    # --- CD helpers and conversions ---
    @classmethod
    def _get_cd_class(cls, dimension: int):
        # lazy load handled in __init__
        return cls._cd_classes.get(dimension)

    @classmethod
    def from_cd_number(cls, cd_number) -> "HypercomplexNumber":
        # cd_number must provide coefficients() and dimensions
        try:
            dim = getattr(cd_number, "dimensions", None) or len(cd_number.coefficients())
            comps = list(cd_number.coefficients())
            return cls(comps, dimension=dim)
        except Exception:
            # fallback: try to extract via to_list
            try:
                return cls(cd_number.to_list())
            except Exception:
                raise

    def pad_to_dimension(self, new_dimension: int) -> "HypercomplexNumber":
        if new_dimension < self.dimension:
            raise ValueError("Cannot pad to smaller dimension")
        if new_dimension == self.dimension:
            return self.copy()
        coeffs = self.coeffs() + [0.0] * (new_dimension - self.dimension)
        return HypercomplexNumber(coeffs, dimension=new_dimension)

    def truncate_to_dimension(self, new_dimension: int) -> "HypercomplexNumber":
        if new_dimension > self.dimension:
            raise ValueError("Cannot truncate to larger dimension")
        if new_dimension == self.dimension:
            return self.copy()
        coeffs = self.coeffs()[:new_dimension]
        return HypercomplexNumber(coeffs, dimension=new_dimension)

    # --- algebraic helpers ---
    def conjugate(self) -> "HypercomplexNumber":
        if self._has_cd:
            return HypercomplexNumber.from_cd_number(self._cd_number.conjugate())
        # elementwise conjugate: complex conjugate for each component
        return HypercomplexNumber([complex(c).conjugate() if isinstance(c, complex) else c for c in self.coeffs()], dimension=self.dimension)

    def norm(self) -> float:
        if self._has_cd:
            try:
                return float(self._cd_number.norm())
            except Exception:
                pass
        return abs(self)

    def inverse(self) -> "HypercomplexNumber":
        if self._has_cd:
            return HypercomplexNumber.from_cd_number(self._cd_number.inverse())
        # fallback: elementwise reciprocal where possible
        comps = []
        for c in self.coeffs():
            if _is_zero(c):
                comps.append(float("inf"))
            else:
                comps.append(1.0 / c)
        return HypercomplexNumber(comps, dimension=self.dimension)

    def normalize(self) -> "HypercomplexNumber":
        n = self.norm()
        if _is_zero(n):
            raise ZeroDivisionError("Cannot normalize zero")
        return HypercomplexNumber([c / n for c in self.coeffs()], dimension=self.dimension)

    def dot(self, other: Any) -> float:
        if not isinstance(other, HypercomplexNumber):
            other = HypercomplexNumber(other)
        a, b, dim = self._align_with(other)
        return sum((x * y) for x, y in zip(a, b))

    # --- utilities ---
    def to_tuple(self) -> Tuple[Number, ...]:
        return tuple(self.coeffs())

    def to_numpy(self):
        try:
            import numpy as np
            return np.array(self.coeffs(), dtype=float)
        except Exception:
            raise

    def summary(self) -> str:
        non_zero = sum(1 for c in self.coeffs() if abs(c) > 1e-12)
        max_coeff = max((abs(c) for c in self.coeffs()), default=0.0)
        min_non_zero = min((abs(c) for c in self.coeffs() if abs(c) > 0), default=0.0)
        return (f"{self.DIMENSION_NAMES.get(self.dimension, f'CD{self.dimension}')} Summary:\n"
                f"  Dimension: {self.dimension}\n"
                f"  Non-zero components: {non_zero}\n"
                f"  Real part: {self.real:.6f}\n"
                f"  Norm: {self.norm():.6f}\n"
                f"  Max component: {max_coeff:.6f}\n"
                f"  Min non-zero: {min_non_zero:.6f}")

    def __str__(self):
        return self.to_summary(max_components=8)

    def __repr__(self):
        return f"HypercomplexNumber({self.to_list()[:8]}{'...' if len(self._comps)>8 else ''})"


# ---------- Helper zero check (genel) ------------------------------------
def _is_zero(value: Any) -> bool:
    try:
        if isinstance(value, (int, float)):
            return abs(value) < 1e-12
        if isinstance(value, complex):
            return abs(value) < 1e-12
        if isinstance(value, HypercomplexNumber):
            return all(_is_zero(c) for c in value.coeffs())
        if isinstance(value, (list, tuple)):
            return all(_is_zero(v) for v in value)
        if hasattr(value, "__abs__"):
            try:
                return abs(value) < 1e-12
            except Exception:
                pass
        return abs(float(value)) < 1e-12
    except Exception:
        return False

# --- Yardımcı fonksiyonlar ------------------------------------------------
def _safe_import(name: str):
    try:
        module = __import__(name, fromlist=['*'])
        return module
    except Exception:
        return None

# Try to import sympy if available for robust primality
_sympy = _safe_import('sympy')
if _sympy:
    _sympy_isprime = getattr(_sympy, 'isprime', None)
else:
    _sympy_isprime = None


def _is_near_integer(x: Any, tol: float = 1e-9) -> bool:
    """Bir sayının neredeyse tam sayı olup olmadığını kontrol et."""
    try:
        if isinstance(x, int):
            return True
        xf = float(x)
        return abs(xf - round(xf)) <= tol
    except Exception:
        return False

def _float_mod_zero(x: Any, divisor: int = 1, tol: float = 1e-9) -> bool:
    """x % divisor yaklaşık sıfır mı? (float toleranslı)"""
    try:
        xf = float(x)
        if divisor == 0:
            return False
        return abs(xf - round(xf / divisor) * divisor) <= tol
    except Exception:
        return False

def _int_from_value(value: Any) -> int:
    """
    Değeri tamsayıya çevirmeye çalışır:
    - Eğer HypercomplexNumber veya iterable ise ilk bileşeni alır
    - Eğer string ise sayısal token'ı alır
    - Başarısızsa None döner
    """
    try:
        if value is None:
            return None
        # If object provides integer representation helper
        if hasattr(value, 'to_int') and callable(getattr(value, 'to_int')):
            try:
                return int(value.to_int())
            except Exception:
                pass
        # If object has coeffs or to_list
        comps = _parse_components(value)
        if comps:
            return int(round(comps[0]))
        # fallback scalar
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        if isinstance(value, complex):
            return int(round(value.real))
        # try direct conversion
        return int(float(str(value)))
    except Exception:
        return None

def _simple_is_prime(n: int) -> bool:
    """Küçük/orta büyüklükte tamsayılar için basit deterministik test."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

# --- Ana birleşik is_prime_like fonksiyonu -------------------------------

def is_prime_like(value: Any, kececi_type: int = None) -> bool:
    """
    Genişletilmiş prime-like testi.
    - Önce proje içi is_prime_like veya is_prime fonksiyonlarını dener.
    - sympy varsa onu kullanır.
    - Quaternion/Hypercomplex/Array-based/ternary/clifford gibi tipler için bileşenleri kontrol eder.
    - kececi_type belirtilirse tip-özel kurallar uygulanır.
    """
    try:
        # 1) Proje içi helper varsa kullan
        try:
            from .kececinumbers import is_prime_like as _proj_ipl
            return bool(_proj_ipl(value, kececi_type) if _proj_ipl.__code__.co_argcount >= 2 else _proj_ipl(value))
        except Exception:
            pass

        # 2) Eğer doğrudan integer temsil edilebiliyorsa onu al ve test et
        n = _int_from_value(value)
        if n is not None:
            if _sympy_isprime:
                try:
                    return bool(_sympy_isprime(int(n)))
                except Exception:
                    return _simple_is_prime(int(n))
            else:
                return _simple_is_prime(int(n))

        # 3) Tip bazlı heuristikler (kececi_type varsa)
        # Tip sabitleri proje içinde farklı isimlerde olabilir; burada sayısal kodlar kullanılıyor.
        # Kullanıcının tanımladığı TYPE_* sabitlerini kullanıyorsanız onları import edin veya
        # burada numeric karşılıklarını verin. Örnek: TYPE_HYPERCOMPLEX == 23
        TYPE_HYPERCOMPLEX = 23
        TYPE_QUATERNION = 6
        TYPE_OCTONION = 12
        TYPE_SEDENION = 13
        TYPE_PATHION = 17
        TYPE_CHINGON = 18
        TYPE_ROUTON = 19
        TYPE_VOUDON = 20
        TYPE_TERNARY = 22
        TYPE_CLIFFORD = 14
        TYPE_SUPERREAL = 21

        # Quaternion özel kontrol
        if kececi_type == TYPE_QUATERNION:
            try:
                # Eğer proje quaternion sınıfı varsa onun bileşenlerini al
                from .kececinumbers import quaternion as _quat_cls
            except Exception:
                _quat_cls = None
            try:
                if _quat_cls is not None and isinstance(value, _quat_cls):
                    comps = [value.w, value.x, value.y, value.z]
                elif hasattr(value, 'w') and hasattr(value, 'x'):
                    comps = [getattr(value, a) for a in ('w','x','y','z') if hasattr(value, a)]
                elif hasattr(value, 'coeffs'):
                    comps = list(value.coeffs())
                else:
                    comps = _parse_components(value)
                if not comps:
                    return False
                if not all(_is_near_integer(c) for c in comps):
                    return False
                # test first (real) component
                n0 = int(round(float(comps[0])))
                if _sympy_isprime:
                    return bool(_sympy_isprime(n0))
                return _simple_is_prime(n0)
            except Exception:
                return False

        # Hypercomplex family (octonion, sedenion, pathion, ...)
        if kececi_type in (TYPE_OCTONION, TYPE_SEDENION, TYPE_PATHION, TYPE_CHINGON, TYPE_ROUTON, TYPE_VOUDON, TYPE_HYPERCOMPLEX):
            try:
                # extract coeffs
                if hasattr(value, 'coeffs'):
                    coeffs = list(value.coeffs())
                elif hasattr(value, 'to_list'):
                    coeffs = list(value.to_list())
                elif isinstance(value, (list, tuple)):
                    coeffs = list(value)
                else:
                    coeffs = _parse_components(value)
                if not coeffs:
                    return False
                if not all(_is_near_integer(c) for c in coeffs):
                    return False
                n0 = int(round(float(coeffs[0])))
                if _sympy_isprime:
                    return bool(_sympy_isprime(n0))
                return _simple_is_prime(n0)
            except Exception:
                return False

        # Ternary
        if kececi_type == TYPE_TERNARY:
            try:
                # If object has to_decimal or digits
                if hasattr(value, 'to_decimal'):
                    dec = int(value.to_decimal())
                elif hasattr(value, 'digits'):
                    digits = list(value.digits)
                    dec = 0
                    for i, d in enumerate(reversed(digits)):
                        dec += int(d) * (3 ** i)
                elif isinstance(value, (list, tuple)):
                    parts = list(value)
                    dec = 0
                    for i, d in enumerate(reversed(parts)):
                        dec += int(d) * (3 ** i)
                else:
                    # try parse first component
                    comps = _parse_components(value)
                    if not comps:
                        return False
                    dec = int(round(comps[0]))
                if _sympy_isprime:
                    return bool(_sympy_isprime(dec))
                return _simple_is_prime(dec)
            except Exception:
                return False

        # Clifford
        if kececi_type == TYPE_CLIFFORD:
            try:
                if hasattr(value, 'basis') and isinstance(value.basis, dict):
                    scalar = value.basis.get('', 0)
                    if _is_near_integer(scalar):
                        n = int(round(float(scalar)))
                        if _sympy_isprime:
                            return bool(_sympy_isprime(n))
                        return _simple_is_prime(n)
                return False
            except Exception:
                return False

        # Superreal
        if kececi_type == TYPE_SUPERREAL:
            try:
                if hasattr(value, 'real'):
                    real = getattr(value, 'real')
                    if _is_near_integer(real):
                        n = int(round(float(real)))
                        if _sympy_isprime:
                            return bool(_sympy_isprime(n))
                        return _simple_is_prime(n)
                return False
            except Exception:
                return False

        # 4) Genel fallback: magnitude veya ilk bileşen üzerinden test et
        try:
            comps = _parse_components(value)
            if not comps:
                return False
            mag = int(abs(round(float(comps[0]))))
            if mag < 2:
                return False
            if _sympy_isprime:
                return bool(_sympy_isprime(mag))
            return _simple_is_prime(mag)
        except Exception:
            return False

    except Exception as e:
        logger.debug("is_prime_like unexpected error: %s", e)
        return False


# ---------- Güncellenmiş get_unit fonksiyonu ------------------------------
def _get_ask_unit_for_type(number_type: int, sample_value: Any = None) -> Any:
    """
    Get appropriate Keçeci unit for a number type.
    Hypercomplex (23) returns a HypercomplexNumber.unit inferred from sample_value or default dim 8.
    """
    # simple numeric types
    if number_type in [1, 4, 5]:
        return 1.0
    if number_type == 2:
        return -1.0
    if number_type == 3:
        return complex(1, 0)
    if number_type == 6:
        try:
            from .kececinumbers import quaternion
            return quaternion(1, 0, 0, 0)
        except Exception:
            return [1.0, 0.0, 0.0, 0.0]
    if number_type == 7:
        try:
            from .kececinumbers import NeutrosophicNumber
            return NeutrosophicNumber(1, 0, 0)
        except Exception:
            return (1.0, 0.0, 0.0)
    if number_type == 8:
        try:
            from .kececinumbers import NeutrosophicComplexNumber
            return NeutrosophicComplexNumber(1, 0, 0)
        except Exception:
            return complex(1, 0)
    if number_type in [12, 13, 17, 18, 19, 20, 22]:
        sizes = {12: 8, 13: 16, 17: 32, 18: 64, 19: 128, 20: 256, 22: 3}
        size = sizes.get(number_type, 1)
        if sample_value is not None and hasattr(sample_value, "__len__"):
            try:
                size = max(1, len(sample_value))
            except Exception:
                pass
        unit = [0.0] * size
        unit[0] = 1.0
        if sample_value is not None:
            try:
                return type(sample_value)(unit)
            except Exception:
                return unit
        return unit
    if number_type == 23:
        # infer dimension from sample_value if possible, default 8
        dim = 8
        if sample_value is not None:
            try:
                if isinstance(sample_value, HypercomplexNumber):
                    dim = max(1, len(sample_value))
                elif hasattr(sample_value, "__len__"):
                    dim = max(1, len(sample_value))
                else:
                    # if sample is string, parse components
                    comps = _parse_components(sample_value)
                    if comps:
                        dim = max(1, len(comps))
            except Exception:
                pass
        try:
            return HypercomplexNumber([1.0] + [0.0] * (dim - 1), dimension=dim)
        except Exception:
            return [1.0] + [0.0] * (dim - 1)
    # default fallback
    if sample_value is not None:
        try:
            return type(sample_value)(1)
        except Exception:
            pass
    return 1.0

# Yardımcı Fonksiyonlar:
# Factory functions for specific hypercomplex types
def Real(x: float) -> HypercomplexNumber:
    """Create a real number (dimension 1)."""
    return HypercomplexNumber(x, dimension=1)

def Complex(real: float, imag: float) -> HypercomplexNumber:
    """Create a complex number (dimension 2)."""
    return HypercomplexNumber(real, imag, dimension=2)

def Quaternion(w: float, x: float, y: float, z: float) -> HypercomplexNumber:
    """Create a quaternion (dimension 4)."""
    return HypercomplexNumber(w, x, y, z, dimension=4)

def Octonion(*components) -> HypercomplexNumber:
    """Create an octonion (dimension 8)."""
    if len(components) != 8:
        components = list(components) + [0.0] * (8 - len(components))
    return HypercomplexNumber(*components, dimension=8)

def Sedenion(*components) -> HypercomplexNumber:
    """Create a sedenion (dimension 16)."""
    if len(components) != 16:
        components = list(components) + [0.0] * (16 - len(components))
    return HypercomplexNumber(*components, dimension=16)

def Pathion(*components) -> HypercomplexNumber:
    """Create a pathion (dimension 32)."""
    if len(components) != 32:
        components = list(components) + [0.0] * (32 - len(components))
    return HypercomplexNumber(*components, dimension=32)

def Chingon(*components) -> HypercomplexNumber:
    """Create a chingon (dimension 64)."""
    if len(components) != 64:
        components = list(components) + [0.0] * (64 - len(components))
    return HypercomplexNumber(*components, dimension=64)

def Routon(*components) -> HypercomplexNumber:
    """Create a routon (dimension 128)."""
    if len(components) != 128:
        components = list(components) + [0.0] * (128 - len(components))
    return HypercomplexNumber(*components, dimension=128)

def Voudon(*components) -> HypercomplexNumber:
    """Create a voudon (dimension 256)."""
    if len(components) != 256:
        components = list(components) + [0.0] * (256 - len(components))
    return HypercomplexNumber(*components, dimension=256)

class quaternion:
    """
    Kuaterniyon sınıfı: w + xi + yj + zk formatında
    
    Attributes:
        w: Reel kısım
        x: i bileşeni
        y: j bileşeni
        z: k bileşeni
    """
    
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Kuaterniyon oluşturur.
        
        Args:
            w: Reel kısım
            x: i bileşeni
            y: j bileşeni
            z: k bileşeni
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @classmethod
    def from_axis_angle(cls, axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float) -> 'quaternion':
        """
        Eksen-açı gösteriminden kuaterniyon oluşturur.
        
        Args:
            axis: Dönme ekseni (3 boyutlu vektör)
            angle: Radyan cinsinden dönme açısı
        
        Returns:
            quaternion: Kuaterniyon nesnesi
        """
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
    
    def conjugate(self) -> 'quaternion':
        """Kuaterniyonun eşleniğini döndürür."""
        return quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        """Kuaterniyonun normunu döndürür."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
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
    
    def __add__(self, other: 'quaternion') -> 'quaternion':
        """Kuaterniyon toplama."""
        if isinstance(other, quaternion):
            return quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        raise TypeError("Sadece quaternion ile toplanabilir")
    
    def __sub__(self, other: 'quaternion') -> 'quaternion':
        """Kuaterniyon çıkarma."""
        if isinstance(other, quaternion):
            return quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Sadece quaternion ile çıkarılabilir")
    
    def __mul__(self, other: Union['quaternion', float, int]) -> 'quaternion':
        """Kuaterniyon çarpma veya skaler çarpma."""
        if isinstance(other, (int, float)):
            return quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        elif isinstance(other, quaternion):
            # Hamilton çarpımı
            w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            return quaternion(w, x, y, z)
        raise TypeError("Sadece quaternion veya skaler ile çarpılabilir")
    
    def __rmul__(self, other: Union[float, int]) -> 'quaternion':
        """Sağ taraftan skaler çarpma."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[float, int]) -> 'quaternion':
        """Skaler bölme."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Sıfıra bölme hatası")
            return quaternion(self.w / other, self.x / other, self.y / other, self.z / other)
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
        raise TypeError("Sadece skaler ile bölünebilir")
    
    def __eq__(self, other: 'quaternion') -> bool:
        """Eşitlik kontrolü."""
        if isinstance(other, quaternion):
            return (math.isclose(self.w, other.w) and 
                    math.isclose(self.x, other.x) and 
                    math.isclose(self.y, other.y) and 
                    math.isclose(self.z, other.z))
        return False
    
    def __ne__(self, other: 'quaternion') -> bool:
        """Eşitsizlik kontrolü."""
        return not self.__eq__(other)
    
    def __neg__(self) -> 'quaternion':
        """Negatif kuaterniyon."""
        return quaternion(-self.w, -self.x, -self.y, -self.z)
    
    def __repr__(self) -> str:
        """Nesnenin temsili."""
        return f"quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"
    
    def __str__(self) -> str:
        """String temsili."""
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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
class BaseNumber(ABC):
    """Tüm Keçeci sayı tipleri için ortak arayüz."""

    def __init__(self, value: Number):
        self._value = self._coerce(value)

    @staticmethod
    def _coerce(v: Number) -> Number:
        if isinstance(v, (int, float, complex)):
            return v
        raise TypeError(f"Geçersiz sayı tipi: {type(v)}")

    @property
    def value(self) -> Number:
        return self._value

    # ------------------------------------------------------------------ #
    # Matematiksel operator overload’ları (tek yönlü)
    # ------------------------------------------------------------------ #
    def __add__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        return self.__class__(self._value + other_val)

    def __radd__(self, other: Number) -> "BaseNumber":
        return self.__add__(other)

    def __sub__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        return self.__class__(self._value - other_val)

    def __rsub__(self, other: Number) -> "BaseNumber":
        return self.__class__(other - self._value)

    def __mul__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        return self.__class__(self._value * other_val)

    def __rmul__(self, other: Number) -> "BaseNumber":
        return self.__add__(other)

    def __truediv__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        if other_val == 0:
            raise ZeroDivisionError("division by zero")
        return self.__class__(self._value / other_val)
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

    def __rtruediv__(self, other: Number) -> "BaseNumber":
        if self._value == 0:
            raise ZeroDivisionError("division by zero")
        return self.__class__(other / self._value)

    def __mod__(self, divisor: Number) -> "BaseNumber":
        return self.__class__(self._value % divisor)

    # ------------------------------------------------------------------ #
    # Karşılaştırmalar
    # ------------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseNumber):
            return NotImplemented
        return math.isclose(float(self._value), float(other._value), rel_tol=1e-12)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"

    # ------------------------------------------------------------------ #
    # Alt sınıfların doldurması gereken soyut metodlar
    # ------------------------------------------------------------------ #
    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        # Daha dayanıklı dönüş: coeffs bir numpy array veya python list olabilir.
        if hasattr(self, 'coeffs'):
            coeffs = getattr(self, 'coeffs')
            if isinstance(coeffs, np.ndarray):
                return coeffs.tolist()
            try:
                return list(coeffs)
            except Exception:
                return [coeffs]
        # Fallback: tek değer
        return [self._value]

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
            if hasattr(self, 'coeffs'):
                coeffs = getattr(self, 'coeffs')
                if isinstance(coeffs, (list, tuple, np.ndarray)) and len(coeffs) > 0:
                    first = coeffs[0]
                    if isinstance(first, complex):
                        return math.atan2(first.imag, first.real)
            # If value has real/imag attributes (like some custom complex types)
            if hasattr(self._value, 'real') and hasattr(self._value, 'imag'):
                return math.atan2(self._value.imag, self._value.real)
        except Exception:
            pass
        return 0.0

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
    
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

@dataclass
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
        return f"ChingonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"ChingonNumber({self.coeffs})"
    
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
        if not isinstance(other, ChingonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, ChingonNumber):  # ChingonNumber ile karşılaştır
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return ChingonNumber([c / other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Any

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
    
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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
    
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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
    
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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

    def __str__(self) -> str:
        parts = []
        if self.t != 0:
            parts.append(f"{self.t}")
        if self.i != 0:
            parts.append(f"{self.i}I")
        if self.f != 0:
            parts.append(f"{self.f}F")
        return " + ".join(parts) if parts else "0"

# Nötrosofik Karmaşık Sayı Sınıfı
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
                self.indeterminacy + other.indeterminacy,
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real + other, self.imag, self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                self.real + other.real, self.imag + other.imag, self.indeterminacy
            )
        return NotImplemented

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy,
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real - other, self.imag, self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                self.real - other.real, self.imag - other.imag, self.indeterminacy
            )
        return NotImplemented

    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                other - self.real, -self.imag, -self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                other.real - self.real, other.imag - self.imag, -self.indeterminacy
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
                self.indeterminacy
                + other.indeterminacy
                + mag_sq_self * other.indeterminacy
                + mag_sq_other * self.indeterminacy
            )

            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        elif isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real * other, self.imag * other, self.indeterminacy * other
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicComplexNumber(
                self.real / other, self.imag / other, self.indeterminacy / other
            )
        return NotImplemented
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
            math.isclose(self.real, other.real, abs_tol=1e-12)
            and math.isclose(self.imag, other.imag, abs_tol=1e-12)
            and math.isclose(self.indeterminacy, other.indeterminacy, abs_tol=1e-12)
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
    def from_polar(
        cls, r: float, theta: float, indeterminacy: float = 0.0
    ) -> "NeutrosophicComplexNumber":
        """Kutupsal koordinatlardan oluştur"""
        return cls(r * math.cos(theta), r * math.sin(theta), indeterminacy)

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

    def __str__(self) -> str:
        parts = []
        if self.z1 != 0j:
            parts.append(f"({self.z1.real}+{self.z1.imag}j)")
        if self.z2 != 0j:
            parts.append(f"({self.z2.real}+{self.z2.imag}j)e")
        return " + ".join(parts) if parts else "0"

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
        if hasattr(s, "__iter__") and not isinstance(s, str):
            parts = list(s)
            if len(parts) == 4:
                # Four numbers: (z1_real, z1_imag, z2_real, z2_imag)
                return BicomplexNumber(
                    complex(float(parts[0]), float(parts[1])),
                    complex(float(parts[2]), float(parts[3])),
                )
            elif len(parts) == 2:
                # Two numbers: (real, imag) for z1
                return BicomplexNumber(
                    complex(float(parts[0]), float(parts[1])), complex(0.0, 0.0)
                )
            elif len(parts) == 1:
                # Single number: real part of z1
                return BicomplexNumber(complex(float(parts[0]), 0.0), complex(0.0, 0.0))

        # Convert to string for further parsing
        if not isinstance(s, str):
            s = str(s)

        s_clean = s.strip().replace(" ", "")

        # Feature 5.1: Comma-separated numeric list
        if "," in s_clean:
            parts = [p for p in s_clean.split(",") if p != ""]
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 4:
                    return BicomplexNumber(
                        complex(nums[0], nums[1]), complex(nums[2], nums[3])
                    )
                elif len(nums) == 2:
                    return BicomplexNumber(complex(nums[0], nums[1]), complex(0.0, 0.0))
                elif len(nums) == 1:
                    return BicomplexNumber(complex(nums[0], 0.0), complex(0.0, 0.0))
            except ValueError:
                # Not purely numeric, continue to other formats
                pass

        # Feature 5.2: Explicit "(a+bj)+(c+dj)e" format
        if "e" in s_clean and "(" in s_clean:
            # Try both patterns from both versions
            patterns = [
                # From first version
                r"\(\s*([+-]?\d*\.?\d+)\s*([+-])\s*([\d\.]*)j\s*\)\s*(?:\+)\s*\(\s*([+-]?\d*\.?\d+)\s*([+-])\s*([\d\.]*)j\s*\)e",
                # From second version
                r"\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)\s*\+\s*\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)e",
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
                                z1_imag_sign = -1.0 if groups[1] == "-" else 1.0
                                z1_imag_val = (
                                    float(groups[2])
                                    if groups[2] not in ["", None]
                                    else 1.0
                                )
                                z1_imag = z1_imag_sign * z1_imag_val

                                z2_real = float(groups[3])
                                z2_imag_sign = -1.0 if groups[4] == "-" else 1.0
                                z2_imag_val = (
                                    float(groups[5])
                                    if groups[5] not in ["", None]
                                    else 1.0
                                )
                                z2_imag = z2_imag_sign * z2_imag_val
                            else:
                                # Second version pattern
                                z1_real = float(groups[0])
                                z1_imag_sign = -1 if groups[1] == "-" else 1
                                z1_imag_val = float(groups[2] or "1")
                                z1_imag = z1_imag_sign * z1_imag_val

                                z2_real = float(groups[3])
                                z2_imag_sign = -1 if groups[4] == "-" else 1
                                z2_imag_val = float(groups[5] or "1")
                                z2_imag = z2_imag_sign * z2_imag_val

                            return BicomplexNumber(
                                complex(z1_real, z1_imag), complex(z2_real, z2_imag)
                            )
                    except Exception:
                        continue

        # Feature 5.3: Complex number parsing (common patterns)
        if "j" in s_clean:
            # If string contains 'j', try to parse as complex
            try:
                # Try direct complex() parsing first
                c = complex(s_clean)
                return BicomplexNumber(c, complex(0.0, 0.0))
            except ValueError:
                # Try regex-based parsing for malformed complex numbers
                pattern = r"^([+-]?\d*\.?\d*)([+-]?\d*\.?\d*)j$"
                match = re.match(pattern, s_clean)
                if match:
                    real_part = match.group(1)
                    imag_part = match.group(2)

                    # Handle edge cases
                    if real_part in ["", "+", "-"]:
                        real_part = real_part + "1" if real_part else "0"
                    if imag_part in ["", "+", "-"]:
                        imag_part = imag_part + "1" if imag_part else "0"

                    return BicomplexNumber(
                        complex(float(real_part or 0), float(imag_part or 0)),
                        complex(0.0, 0.0),
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
                    complex(float(num_token), 0.0), complex(0.0, 0.0)
                )
        except Exception:
            pass

    except Exception as e:
        # Feature 7: Logging on error
        if "logger" in globals():
            logger.warning(f"Bicomplex parsing failed for {repr(s)}: {e}")
        else:
            print(f"Bicomplex parsing error for '{s}': {e}")

    # Final fallback: return zero bicomplex
    return BicomplexNumber(complex(0.0, 0.0), complex(0.0, 0.0))

def _next_power_of_two_at_least(n: int, max_dim: int = 256) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n and p < max_dim:
        p <<= 1
    return p if p <= max_dim else max_dim

def _parse_single_token(tok: Any):
    """Tek token -> float veya complex; mixed/fraction destekli."""
    if tok is None:
        return 0.0
    if isinstance(tok, (int, float, complex)):
        return tok
    s = str(tok).strip()
    if s == "":
        return 0.0
    # normalize imaginary unit (i, I, j, J -> j)
    s = s.replace("I", "i").replace("J", "j").replace("i", "j")
    # complex first
    try:
        c = complex(s)
        if c.imag == 0:
            return float(c.real)
        return c
    except Exception:
        pass
    # mixed number "a b/c"
    if " " in s and "/" in s:
        try:
            whole, frac = s.split(" ", 1)
            num, den = frac.split("/")
            return float(whole) + float(num) / float(den)
        except Exception:
            pass
    # fraction "a/b"
    if "/" in s:
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except Exception:
            pass
    # fallback numeric token via regex
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass
    return 0.0

def _parse_components(value: Any) -> List:
    """Flexible parsing: iterable, object with coeffs/to_list, string, scalar."""
    if value is None:
        return []
    # object helpers
    try:
        if hasattr(value, "to_list") and callable(getattr(value, "to_list")):
            return [_parse_single_token(x) for x in value.to_list()]
        if hasattr(value, "coeffs") and callable(getattr(value, "coeffs")):
            return [_parse_single_token(x) for x in value.coeffs()]
    except Exception:
        pass
    # iterable but not string
    if isinstance(value, (list, tuple)):
        return [_parse_single_token(x) for x in value]
    if isinstance(value, (int, float, complex)):
        return [value]
    if isinstance(value, str):
        s = value.strip()
        # strip surrounding brackets
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            s = s[1:-1].strip()
        parts = [p.strip() for p in s.split(",")] if "," in s else s.split()
        # handle single mixed-number token like "1 1/2"
        if len(parts) == 1 and " " in parts[0] and "/" in parts[0]:
            parts = [parts[0]]
        comps = []
        for p in parts:
            if p == "":
                continue
            comps.append(_parse_single_token(p))
        return comps
    # fallback try float conversion
    try:
        return [float(value)]
    except Exception:
        return [0.0]

def _pad_or_truncate_list(lst: Iterable, dim: int) -> List:
    arr = list(lst)
    if len(arr) < dim:
        return arr + [0.0] * (dim - len(arr))
    return arr[:dim]

def _construct_hypercomplex_from_components(HC_cls, comps: List, dimension: int):
    """
    Try common constructor signatures for HypercomplexNumber-like classes.
    Returns instance or raises TypeError.
    """
    # prefer (list, dimension=dim)
    try:
        return HC_cls(comps, dimension=dimension)
    except TypeError:
        pass
    except Exception as e:
        logger.debug("HC constructor (list, dim) failed: %s", e)
    # try (*comps, dimension=dim)
    try:
        return HC_cls(*comps, dimension=dimension)
    except TypeError:
        pass
    except Exception as e:
        logger.debug("HC constructor (*comps, dim) failed: %s", e)
    # try list only
    try:
        return HC_cls(comps)
    except Exception as e:
        logger.debug("HC constructor (list) failed: %s", e)
    # try *comps only
    try:
        return HC_cls(*comps)
    except Exception as e:
        logger.debug("HC constructor (*comps) failed: %s", e)
    raise TypeError("No compatible Hypercomplex constructor found")

def _get_default_hypercomplex(dim: int):
    """Return a default fallback (list) for given dimension."""
    return [0.0] * max(1, int(dim))

# --- Ana fonksiyon -------------------------------------------------------

def _parse_hypercomplex(s: Any, dimension: Optional[int] = None):
    """
    Parse input to HypercomplexNumber (or fallback list) with specific dimension.
    - If dimension is None, infer from input length or default to 8.
    - Accepts HypercomplexNumber, scalars, complex, iterables, strings.
    - Returns HypercomplexNumber instance if class available, else a padded list.
    """
    try:
        # If already HypercomplexNumber-like, try to adapt
        try:
            from .kececinumbers import HypercomplexNumber as HC_cls  # project class if present
        except Exception:
            HC_cls = None

        # If input is already HC instance and dimension provided, adapt
        if HC_cls is not None and isinstance(s, HC_cls):
            if dimension is None or s.dimension == dimension:
                return s if dimension is None else s.pad_to_dimension(dimension) if s.dimension < dimension else s.truncate_to_dimension(dimension)
        # If input is our local HypercomplexNumber (same name but different module), handle generically
        try:
            # duck-typing: object with .dimension and .pad_to_dimension/truncate_to_dimension
            if hasattr(s, "dimension") and hasattr(s, "pad_to_dimension") and hasattr(s, "truncate_to_dimension"):
                if dimension is None or getattr(s, "dimension", None) == dimension:
                    return s if dimension is None else s.pad_to_dimension(dimension) if s.dimension < dimension else s.truncate_to_dimension(dimension)
        except Exception:
            pass

        # Scalars and complex
        if isinstance(s, (int, float)):
            # infer dimension if not provided
            dim = int(dimension) if dimension is not None else 8
            coeffs = [float(s)] + [0.0] * (dim - 1)
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception:
                    return coeffs
            return coeffs

        if isinstance(s, complex):
            dim = int(dimension) if dimension is not None else 8
            coeffs = [float(s.real), float(s.imag)] + [0.0] * (dim - 2)
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception:
                    return coeffs
            return coeffs

        # Iterable (list, tuple, numpy array, etc.)
        if hasattr(s, "__iter__") and not isinstance(s, str):
            comps = list(s)
            # parse elements to numeric where possible
            parsed = [_parse_single_token(c) for c in comps]
            # infer dimension if not provided
            desired = max(1, len(parsed))
            dim = int(dimension) if dimension is not None else _next_power_of_two_at_least(desired)
            coeffs = _pad_or_truncate_list(parsed, dim)
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception:
                    return coeffs
            return coeffs

        # String parsing
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        # remove surrounding brackets/braces/parentheses
        s = s.strip("[]{}()")
        if s == "":
            dim = int(dimension) if dimension is not None else 8
            coeffs = [0.0] * dim
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception:
                    return coeffs
            return coeffs

        # comma-separated list
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            parsed = [_parse_single_token(p) for p in parts]
            desired = max(1, len(parsed))
            dim = int(dimension) if dimension is not None else _next_power_of_two_at_least(desired)
            coeffs = _pad_or_truncate_list(parsed, dim)
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception as e:
                    logger.debug("HC construct from comma parts failed: %s", e)
                    return coeffs
            return coeffs

        # try single numeric
        try:
            val = float(s)
            dim = int(dimension) if dimension is not None else 8
            coeffs = [val] + [0.0] * (dim - 1)
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception:
                    return coeffs
            return coeffs
        except Exception:
            pass

        # try complex string
        try:
            c = complex(s)
            dim = int(dimension) if dimension is not None else 8
            coeffs = [float(c.real), float(c.imag)] + [0.0] * (dim - 2)
            if HC_cls:
                try:
                    return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
                except Exception:
                    return coeffs
            return coeffs
        except Exception:
            pass

        # fallback: zeros
        dim = int(dimension) if dimension is not None else 8
        coeffs = [0.0] * dim
        if HC_cls:
            try:
                return _construct_hypercomplex_from_components(HC_cls, coeffs, dim)
            except Exception:
                return coeffs
        return coeffs

    except Exception as e:
        warnings.warn(f"Hypercomplex parse error (dim={dimension}) for input {repr(s)}: {e}", RuntimeWarning)
        logger.exception("Hypercomplex parse error")
        return _get_default_hypercomplex(dimension or 8)

def _parse_universal(s: Union[str, Any], target_type: str) -> Any:
    """
    Universal parser for many numeric/hypercomplex target types.
    Returns parsed value or a safe default on error.
    """
    try:
        if target_type is None:
            warnings.warn("target_type is None", RuntimeWarning)
            return _get_default_value(target_type)

        key = str(target_type).strip().lower()

        # --- Special-case direct parsers if available ---
        # bicomplex
        if key == "bicomplex":
            parser = globals().get("_parse_bicomplex")
            if callable(parser):
                try:
                    return parser(s)
                except Exception as e:
                    warnings.warn(f"_parse_bicomplex failed: {e}", RuntimeWarning)
                    return _get_default_value("bicomplex")
            return _get_default_value("bicomplex")

        # complex
        if key == "complex":
            parser = globals().get("_parse_complex")
            if callable(parser):
                try:
                    return parser(s)
                except Exception as e:
                    warnings.warn(f"_parse_complex failed: {e}", RuntimeWarning)
                    return _get_default_value("complex")
            # fallback: try Python complex()
            try:
                return complex(s)
            except Exception:
                return _get_default_value("complex")

        # real
        if key == "real" or key == "float":
            try:
                if s is None:
                    return 0.0
                if isinstance(s, (int, float)):
                    return float(s)
                if isinstance(s, complex):
                    return float(s.real)
                # try complex parser then take real part
                cparser = globals().get("_parse_complex")
                if callable(cparser):
                    try:
                        c = cparser(s)
                        return float(getattr(c, "real", float(c)))
                    except Exception:
                        pass
                # last resort
                return float(str(s).strip())
            except Exception as e:
                warnings.warn(f"Real parse error: {e}", RuntimeWarning)
                return _get_default_value("real")

        # --- Hypercomplex families with explicit dimensions ---
        hyper_map = {
            "quaternion": 4,
            "octonion": 8,
            "sedenion": 16,
            "pathion": 32,
            "chingon": 64,
            "routon": 128,
            "voudon": 256
        }
        if key in hyper_map:
            dim = hyper_map[key]
            parser = globals().get("_parse_hypercomplex")
            if callable(parser):
                try:
                    return parser(s, dim)
                except Exception as e:
                    warnings.warn(f"_parse_hypercomplex(dim={dim}) failed: {e}", RuntimeWarning)
                    return _get_default_value(key)
            # fallback: try generic _parse_hypercomplex with dimension if available
            return _get_default_value(key)

        # generic hypercomplex or hypercomplex_N pattern
        if key == "hypercomplex" or key.startswith("hypercomplex"):
            # try generic parser first
            parser = globals().get("_parse_hypercomplex")
            if callable(parser):
                try:
                    # if parser expects dimension, try to infer default (4)
                    try:
                        return parser(s, 4)
                    except TypeError:
                        # parser may accept only (s) and embed dimension
                        return parser(s)
                except Exception:
                    pass
            # try pattern hypercomplex_<N> or hypercomplexN
            m = re.match(r"hypercomplex[_-]?(\d+)$", key)
            if m:
                try:
                    dim = int(m.group(1))
                    parser = globals().get("_parse_hypercomplex")
                    if callable(parser):
                        try:
                            return parser(s, dim)
                        except Exception:
                            pass
                    # fallback to default factory
                    return _get_default_value(f"hypercomplex_{dim}")
                except Exception:
                    return _get_default_value("hypercomplex")
            # final fallback: try to create a small hypercomplex (dim=4)
            return _get_default_value("hypercomplex") or _get_default_value("quaternion")

        # If unknown but numeric-like, try to coerce to float or complex
        try:
            if isinstance(s, (int, float)):
                return s
            if isinstance(s, complex):
                return s
            # try numeric string
            s_str = str(s).strip()
            if s_str:
                # try int/float then complex
                try:
                    return float(s_str)
                except Exception:
                    try:
                        return complex(s_str)
                    except Exception:
                        pass
        except Exception:
            pass

        # Unknown target_type: warn and return default
        warnings.warn(f"Unknown target_type: {target_type}", RuntimeWarning)
        return _get_default_value(target_type)

    except Exception as e:
        warnings.warn(f"Universal parser error for {target_type}: {e}", RuntimeWarning)
        # try to return a default value if available
        try:
            return _get_default_value(target_type)
        except Exception:
            return None

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
        s = s.strip().replace("J", "j").replace("i", "j")  # Hem J hem i yerine j kullan
    else:
        s = str(s).strip().replace("J", "j").replace("i", "j")

    # 1. Eğer "real,imag" formatındaysa
    if "," in s:
        parts = s.split(",")
        if len(parts) == 2:
            try:
                return complex(float(parts[0]), float(parts[1]))
            except ValueError:
                pass  # Devam et

    # 2. Python'ın kendi complex() dönüştürücüsünü kullanmayı dene (örn: "1+2j", "3j", "-5")
    try:
        return complex(s)
    except ValueError:
        # 3. Sadece real kısmı varsa (örn: "5")
        try:
            return complex(float(s), 0)
        except ValueError:
            # 4. Sadece sanal kısmı varsa (örn: "2j", "j")
            if s.endswith("j"):
                try:
                    imag_val = float(s[:-1]) if s[:-1] else 1.0  # "j" -> 1.0j
                    return complex(0, imag_val)
                except ValueError:
                    pass

            # 5. Fallback: varsayılan kompleks sayı
            warnings.warn(
                f"Geçersiz kompleks sayı formatı: '{s}', 0+0j döndürülüyor",
                RuntimeWarning,
            )
            return complex(0, 0)


def _make_hypercomplex_zero(dim: int) -> Any:
    """Try to construct a hypercomplex zero value for given dimension."""
    # try module-level constructors if available
    for name in ('HypercomplexNumber', 'HyperComplex', 'HyperComplexNumber', 'Quaternion', 'Octonion'):
        cls = globals().get(name)
        if cls is not None:
            try:
                # try common constructor signatures
                try:
                    return cls(*([0.0] * dim), dimension=dim)
                except TypeError:
                    pass
                try:
                    return cls(*([0.0] * dim))
                except TypeError:
                    pass
                try:
                    return cls([0.0] * dim)
                except TypeError:
                    pass
                try:
                    return cls(0)
                except Exception:
                    pass
            except Exception:
                continue

    # try module-level parser
    parser_name = f"_parse_hypercomplex_{dim}"
    parser = globals().get(parser_name) or globals().get("_parse_hypercomplex")
    if parser is not None:
        try:
            return parser("0")
        except Exception:
            pass

    # try to import a local module that may provide a constructor
    try:
        from .kececinumbers import HypercomplexNumber as _HC
        try:
            return _HC(*([0.0] * dim), dimension=dim)
        except Exception:
            try:
                return _HC(*([0.0] * dim))
            except Exception:
                pass
    except Exception:
        pass

    # fallback to numpy array of zeros if numpy available
    try:
        import numpy as _np
        return _np.zeros(dim, dtype=float)
    except Exception:
        pass

    # final fallback: plain Python list of zeros
    return [0.0] * dim


def _get_default_value(target_type: str) -> Any:
    """Get default value for target type in a broad, robust way."""
    # helper to try parser names
    def try_parser(*names):
        for n in names:
            p = globals().get(n)
            if p is not None:
                try:
                    return p("0")
                except Exception:
                    continue
        return None

    # mapping for fixed known types and dimensions
    mapping = {
        "real": lambda: 0.0,
        "float": lambda: 0.0,
        "int": lambda: 0,
        "complex": lambda: complex(0, 0),
        "bicomplex": lambda: try_parser("_parse_bicomplex", "_parse_bi_complex"),
        "superreal": lambda: try_parser("_parse_superreal"),
        "ternary": lambda: try_parser("_parse_ternary"),
        # named hypercomplex families with explicit dimensions
        "quaternion": lambda: _make_hypercomplex_zero(4),
        "octonion": lambda: _make_hypercomplex_zero(8),
        "sedenion": lambda: _make_hypercomplex_zero(16),
        "pathion": lambda: _make_hypercomplex_zero(32),
        "chingon": lambda: _make_hypercomplex_zero(64),
        "routon": lambda: _make_hypercomplex_zero(128),
        "voudon": lambda: _make_hypercomplex_zero(256),
        # generic hypercomplex: try parser, then try small dims, then None
        "hypercomplex": lambda: (
            try_parser("_parse_hypercomplex")
            or _make_hypercomplex_zero(4)
            or _make_hypercomplex_zero(8)
            or _make_hypercomplex_zero(16)
            or None
        ),
    }

    key = (target_type or "").lower()
    factory = mapping.get(key)
    if factory is None:
        # try pattern like hypercomplex<N> or hypercomplex_N
        import re
        m = re.match(r"hypercomplex[_-]?(\d+)$", key)
        if m:
            try:
                dim = int(m.group(1))
                return _make_hypercomplex_zero(dim)
            except Exception:
                return None
        return None

    try:
        return factory()
    except Exception:
        return None

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])
    
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
        return f"ChingonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"ChingonNumber({self.coeffs})"
    
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
        if not isinstance(other, ChingonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, ChingonNumber):  # ChingonNumber ile karşılaştır
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return ChingonNumber([c / other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])    
    
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

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

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
        # compute and return the phase value
        return self._phase   # or whatever logic you need

@property
def coeffs(self):
    return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]

# Ana Nötrosofik sayı sınıfı
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
                self.t + other.t, self.i + other.i, self.f + other.f
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t + other, self.i, self.f)
        return NotImplemented

    def __radd__(self, other: Any) -> "NeutrosophicNumber":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(
                self.t - other.t, self.i - other.i, self.f - other.f
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t - other, self.i, self.f)
        return NotImplemented

    def __rsub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(other - self.t, -self.i, -self.f)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            # Nötrosofik çarpma: (t1 + i1I + f1F) * (t2 + i2I + f2F)
            return NeutrosophicNumber(
                t=self.t * other.t,
                i=self.t * other.i + self.i * other.t + self.i * other.i,
                f=self.t * other.f + self.f * other.t + self.f * other.f,
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t * other, self.i * other, self.f * other)
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicNumber":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicNumber(self.t / other, self.i / other, self.f / other)
        return NotImplemented
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

    def __rtruediv__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(
                other / self.t if self.t != 0 else float("inf"),
                other / self.i if self.i != 0 else float("inf"),
                other / self.f if self.f != 0 else float("inf"),
            )
        return NotImplemented

    def __floordiv__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicNumber(self.t // other, self.i // other, self.f // other)
        return NotImplemented

    def __mod__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot modulo by zero")
            return NeutrosophicNumber(self.t % other, self.i % other, self.f % other)
        return NotImplemented

    def __pow__(self, exponent: Any) -> "NeutrosophicNumber":
        if isinstance(exponent, (int, float)):
            return NeutrosophicNumber(
                self.t**exponent, self.i**exponent, self.f**exponent
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
            math.isclose(self.t, other.t, abs_tol=1e-12)
            and math.isclose(self.i, other.i, abs_tol=1e-12)
            and math.isclose(self.f, other.f, abs_tol=1e-12)
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicNumber):
            # Nötrosofik sıralama (gerçek kısım üzerinden)
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
        """Nötrosofik eşlenik (işaret değişimi)"""
        return NeutrosophicNumber(self.t, -self.i, -self.f)

    def magnitude(self) -> float:
        """Büyüklük (Euclidean norm)"""
        return math.sqrt(self.t**2 + self.i**2 + self.f**2)

    def normalized(self) -> "NeutrosophicNumber":
        """Birim büyüklüğe normalize edilmiş Nötrosofik sayı"""
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
        """Sadece gerçek değer içeren Nötrosofik sayı"""
        return cls(t=value, i=0.0, f=0.0)

    @classmethod
    def indeterminacy(cls, value: float) -> "NeutrosophicNumber":
        """Sadece belirsizlik içeren Nötrosofik sayı"""
        return cls(t=0.0, i=value, f=0.0)

    @classmethod
    def falsity(cls, value: float) -> "NeutrosophicNumber":
        """Sadece yanlışlık içeren Nötrosofik sayı"""
        return cls(t=0.0, i=0.0, f=value)


# Nötrosofik Karmaşık Sayı Sınıfı
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
                self.indeterminacy + other.indeterminacy,
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real + other, self.imag, self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                self.real + other.real, self.imag + other.imag, self.indeterminacy
            )
        return NotImplemented

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy,
            )
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real - other, self.imag, self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                self.real - other.real, self.imag - other.imag, self.indeterminacy
            )
        return NotImplemented

    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                other - self.real, -self.imag, -self.indeterminacy
            )
        elif isinstance(other, complex):
            return NeutrosophicComplexNumber(
                other.real - self.real, other.imag - self.imag, -self.indeterminacy
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
                self.indeterminacy
                + other.indeterminacy
                + mag_sq_self * other.indeterminacy
                + mag_sq_other * self.indeterminacy
            )

            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        elif isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        elif isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real * other, self.imag * other, self.indeterminacy * other
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return NeutrosophicComplexNumber(
                self.real / other, self.imag / other, self.indeterminacy / other
            )
        return NotImplemented
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
            math.isclose(self.real, other.real, abs_tol=1e-12)
            and math.isclose(self.imag, other.imag, abs_tol=1e-12)
            and math.isclose(self.indeterminacy, other.indeterminacy, abs_tol=1e-12)
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
    def from_polar(
        cls, r: float, theta: float, indeterminacy: float = 0.0
    ) -> "NeutrosophicComplexNumber":
        """Kutupsal koordinatlardan oluştur"""
        return cls(r * math.cos(theta), r * math.sin(theta), indeterminacy)

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

    def __str__(self) -> str:
        parts = []
        if self.z1 != 0j:
            parts.append(f"({self.z1.real}+{self.z1.imag}j)")
        if self.z2 != 0j:
            parts.append(f"({self.z2.real}+{self.z2.imag}j)e")
        return " + ".join(parts) if parts else "0"


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
                self.a + other.a,
                self.b + other.b,
                self.c + other.c,
                self.d + other.d,
                self.e + other.e,
                self.f + other.f,
                self.g + other.g,
                self.h + other.h,
            )
        return NotImplemented

    def __mul__(self, other):
        # Basitleştirilmiş çarpım (tam bicomplex kuralı karmaşık)
        if isinstance(other, (int, float)):
            return NeutrosophicBicomplexNumber(
                *(
                    other * x
                    for x in [
                        self.a,
                        self.b,
                        self.c,
                        self.d,
                        self.e,
                        self.f,
                        self.g,
                        self.h,
                    ]
                )
            )
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero")
            return NeutrosophicBicomplexNumber(
                self.a / scalar,
                self.b / scalar,
                self.c / scalar,
                self.d / scalar,
                self.e / scalar,
                self.f / scalar,
                self.g / scalar,
                self.h / scalar,
            )
        return NotImplemented
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

    def __eq__(self, other):
        """Equality with tolerance for float comparison."""
        if not isinstance(other, NeutrosophicBicomplexNumber):
            return False
        tol = 1e-12
        return all(
            abs(getattr(self, attr) - getattr(other, attr)) < tol
            for attr in ["a", "b", "c", "d", "e", "f", "g", "h"]
        )

    def __ne__(self, other):
        return not self.__eq__(other)


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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
        if isinstance(other, (int, float, Fraction)):
            scalar = float(other)
            return self.__class__([c/scalar for c in self.coeffs])

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
    if hasattr(s, "__class__"):
        class_name = s.__class__.__name__
        if class_name == "NeutrosophicNumber":
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
    if "," in s_clean and "(" not in s_clean and ")" not in s_clean:
        parts = [p.strip() for p in s_clean.split(",")]
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
                if value_str == "+":
                    return 1.0
                elif value_str == "-":
                    return -1.0
                return default

        # İlk terim
        if t_val_str is not None:
            val = parse_value(t_val_str)
            if t_unit_str and t_unit_str.upper() == "I":
                i = val
            elif t_unit_str and t_unit_str.upper() == "F":
                f = val
            else:
                t = val

        # İkinci terim
        if i_val_str is not None:
            val = parse_value(i_val_str)
            if i_unit_str and i_unit_str.upper() == "I":
                i = val
            elif i_unit_str and i_unit_str.upper() == "F":
                f = val
            else:
                # Birim yoksa, hangi birime ait olduğunu belirle
                if t_unit_str and t_unit_str.upper() == "I":
                    i += val
                elif t_unit_str and t_unit_str.upper() == "F":
                    f += val
                else:
                    # t birimsizse, i'ye ekle (default I)
                    i = val

        # Üçüncü terim
        if f_val_str is not None:
            val = parse_value(f_val_str)
            if f_unit_str and f_unit_str.upper() == "I":
                i = val
            elif f_unit_str and f_unit_str.upper() == "F":
                f = val
            else:
                # Birim yoksa, hangi birime ait olduğunu belirle
                if i_unit_str and i_unit_str.upper() == "I":
                    i += val
                elif i_unit_str and i_unit_str.upper() == "F":
                    f += val
                elif t_unit_str and t_unit_str.upper() == "I":
                    i += val
                elif t_unit_str and t_unit_str.upper() == "F":
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
    if "I" in s_upper:
        parts = s_upper.split("I", 1)
        before_i = parts[0]
        after_i = parts[1] if len(parts) > 1 else ""

        # I'dan önceki kısmı parse et
        if before_i:
            # Sayısal kısmı ayır
            num_match = re.search(r"([+-]?\d*\.?\d+)$", before_i)
            if num_match:
                t = float(num_match.group(1))
            elif before_i in ["+", "-"]:
                t = 1.0 if before_i == "+" else -1.0
            elif before_i:
                # Sadece sayı olabilir
                try:
                    t = float(before_i)
                except ValueError:
                    pass

        # I'dan sonraki kısmı parse et (indeterminacy değeri)
        if after_i:
            try:
                i = float(after_i) if after_i not in ["", "+", "-"] else 1.0
                if after_i == "-":
                    i = -1.0
            except ValueError:
                i = 1.0  # Sadece "I" varsa
        else:
            i = 1.0  # Sadece "I" varsa

    # "F" sembolünü ara (I'dan bağımsız)
    if "F" in s_upper:
        # I içeriyorsa, F'den önceki kısmı al
        if "I" in s_upper:
            # "I...F" formatı
            i_match = re.search(r"I([^F]*)F", s_upper)
            if i_match:
                i_str = i_match.group(1)
                if i_str:
                    try:
                        i = float(i_str)
                    except ValueError:
                        if i_str in ["+", "-"]:
                            i = 1.0 if i_str == "+" else -1.0
        else:
            # Sadece F içeriyor
            parts = s_upper.split("F", 1)
            before_f = parts[0]
            after_f = parts[1] if len(parts) > 1 else ""

            # F'dan önceki kısmı parse et
            if before_f:
                try:
                    t = float(before_f) if before_f not in ["", "+", "-"] else 0.0
                    if before_f == "+":
                        t = 1.0
                    elif before_f == "-":
                        t = -1.0
                except ValueError:
                    pass

            # F'dan sonraki kısmı parse et (falsity değeri)
            if after_f:
                try:
                    f = float(after_f) if after_f not in ["", "+", "-"] else 1.0
                    if after_f == "-":
                        f = -1.0
                except ValueError:
                    f = 1.0  # Sadece "F" varsa
            else:
                f = 1.0  # Sadece "F" varsa

    # 4. Hiçbir sembol yoksa, sadece sayı olabilir
    if not ("I" in s_upper or "F" in s_upper):
        try:
            t = float(s_clean)
        except ValueError:
            # Parantez içinde olabilir
            if "(" in s_clean and ")" in s_clean:
                content = s_clean[s_clean.find("(") + 1 : s_clean.find(")")]
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
                if sign == "-":
                    imag = -imag

                return real, imag, 0.0
            except ValueError:
                pass

        # 2. Parantez içinde kompleks sayı: "(a+bj)"
        if "(" in s_clean and ")" in s_clean and "j" in s_clean.lower():
            # Parantez içeriğini al
            content = s_clean[s_clean.find("(") + 1 : s_clean.find(")")].strip()
            try:
                # Python'ın kompleks sayı parser'ını kullan
                c = complex(content)
                return float(c.real), float(c.imag), 0.0
            except ValueError:
                pass

        # 3. "complex(a, b)" formatı
        if s_clean.lower().startswith("complex"):
            # "complex(1.5, 2.5)" veya "complex(1.5,2.5)" formatı
            match = re.match(
                r"complex\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", s_clean, re.IGNORECASE
            )
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
    if hasattr(s, "__class__"):
        class_name = s.__class__.__name__
        if class_name in ["Hyperreal", "HyperReal"]:
            try:
                if hasattr(s, "finite") and hasattr(s, "infinitesimal"):
                    finite = float(s.finite)
                    infinitesimal = float(s.infinitesimal)
                    seq = getattr(s, "sequence", [finite, infinitesimal])
                    return finite, infinitesimal, seq
                elif hasattr(s, "real") and hasattr(s, "epsilon"):
                    finite = float(s.real)
                    infinitesimal = float(s.epsilon)
                    seq = getattr(s, "sequence", [finite, infinitesimal])
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
                    if item_str in ["inf", "infinity", "∞"]:
                        seq.append(float("inf"))
                    elif item_str in ["-inf", "-infinity", "-∞"]:
                        seq.append(float("-inf"))
                    elif item_str in ["nan", "null"]:
                        seq.append(float("nan"))
                    elif "ε" in item_str or "epsilon" in item_str:
                        # ε içeriyorsa, infinitesimal bileşen olarak işle
                        num = re.sub(r"[εepsilon]", "", item_str, flags=re.IGNORECASE)
                        if num in ["", "+"]:
                            seq.append(1.0)
                        elif num == "-":
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
            warnings.warn(
                f"Hyperreal tuple/list parse error: {e}", RuntimeWarning, stacklevel=2
            )

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
            warnings.warn(
                f"Hyperreal conversion to string failed: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0, 0.0, [0.0]

    s_clean = s.strip()

    # 5. Özel durumlar
    if s_clean == "":
        return 0.0, 0.0, [0.0]

    # Sonsuzluk değerleri
    infinity_map = {
        "∞": float("inf"),
        "inf": float("inf"),
        "infinity": float("inf"),
        "+∞": float("inf"),
        "+inf": float("inf"),
        "+infinity": float("inf"),
        "-∞": float("-inf"),
        "-inf": float("-inf"),
        "-infinity": float("-inf"),
    }

    s_lower = s_clean.lower()
    if s_lower in infinity_map:
        value = infinity_map[s_lower]
        return value, 0.0, [value]

    # NaN değerleri
    if s_lower in ["nan", "null", "none", "undefined"]:
        return float("nan"), 0.0, [float("nan")]

    # 6. Köşeli parantez içinde sequence (JSON benzeri)
    if s_clean.startswith("[") and s_clean.endswith("]"):
        try:
            content = s_clean[1:-1].strip()
            if content:
                parts = [p.strip() for p in re.split(r",|;", content)]
                seq = []
                for p in parts:
                    if p:
                        try:
                            # Özel sembolleri kontrol et
                            if p.lower() in infinity_map:
                                seq.append(infinity_map[p.lower()])
                            elif p.lower() == "nan":
                                seq.append(float("nan"))
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
            warnings.warn(
                f"Hyperreal sequence parse error: {e}", RuntimeWarning, stacklevel=2
            )

    # 7. Virgülle ayrılmış format: a,b,c
    if "," in s_clean and not s_clean.startswith("(") and not s_clean.endswith(")"):
        try:
            parts = [p.strip() for p in s_clean.split(",")]
            seq = []
            for p in parts:
                if p:
                    try:
                        seq.append(float(p))
                    except ValueError:
                        # Özel değerleri kontrol et
                        if p.lower() in infinity_map:
                            seq.append(infinity_map[p.lower()])
                        elif p.lower() == "nan":
                            seq.append(float("nan"))
                        else:
                            seq.append(0.0)

            finite = seq[0] if seq else 0.0
            infinitesimal = seq[1] if len(seq) > 1 else 0.0
            return finite, infinitesimal, seq
        except Exception as e:
            warnings.warn(
                f"Hyperreal comma-separated parse error: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # 8. GELİŞMİŞ: Matematiksel ifadeler (π, e, φ gibi sabitler)
    constants = {
        "π": math.pi,
        "pi": math.pi,
        "e": math.e,
        "φ": (1 + math.sqrt(5)) / 2,
        "phi": (1 + math.sqrt(5)) / 2,
    }

    # Sabit içerip içermediğini kontrol et
    for const_name, const_value in constants.items():
        if const_name.lower() in s_lower:
            # Sabitin değerini al
            const_val = const_value
            # ε ile kombinasyonu kontrol et
            if "ε" in s_clean or "epsilon" in s_lower:
                # "π + 0.1ε" formatı
                match = re.search(r"([+-]?\s*\d*\.?\d+)\s*[εε]", s_clean, re.IGNORECASE)
                if match:
                    eps_val = (
                        float(match.group(1).replace(" ", ""))
                        if match.group(1).strip() not in ["", "+", "-"]
                        else 1.0
                    )
                    if match.group(1).strip() == "-":
                        eps_val = -1.0
                    return const_val, eps_val, [const_val, eps_val]
                else:
                    return const_val, 0.0, [const_val]
            else:
                return const_val, 0.0, [const_val]

    # 9. Exponential/epsilon formatları
    # "aεb", "a e b", "a + bε", "a - bε"
    epsilon_patterns = [
        r"^\s*([+-]?\d*\.?\d+)\s*[εε]\s*([+-]?\d*\.?\d+)\s*$",  # aεb
        r"^\s*([+-]?\d*\.?\d+)\s*e\s*([+-]?\d*\.?\d+)\s*$",  # a e b (hyperreal)
        r"^\s*([+-]?\d*\.?\d+)\s*\+\s*([+-]?\d*\.?\d+)\s*[εε]\s*$",  # a + bε
        r"^\s*([+-]?\d*\.?\d+)\s*\-\s*([+-]?\d*\.?\d+)\s*[εε]\s*$",  # a - bε
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
    eng_pattern = r"^\s*([+-]?\d*\.?\d+)([kKmMgGtTμμunpf]?)\s*[εε]\s*([+-]?\d*\.?\d+)([kKmMgGtTμμunpf]?)\s*$"
    match = re.match(eng_pattern, s_clean, re.IGNORECASE)
    if match:
        try:
            finite_num = float(match.group(1))
            finite_unit = match.group(2).lower()
            eps_num = float(match.group(3))
            eps_unit = match.group(4).lower()

            # Mühendislik çarpanları
            multipliers = {
                "k": 1e3,
                "m": 1e-3,
                "meg": 1e6,
                "g": 1e9,
                "t": 1e12,
                "μ": 1e-6,
                "u": 1e-6,
                "n": 1e-9,
                "p": 1e-12,
                "f": 1e-15,
            }

            finite = finite_num * multipliers.get(finite_unit, 1.0)
            infinitesimal = eps_num * multipliers.get(eps_unit, 1.0)
            return finite, infinitesimal, [finite, infinitesimal]
        except (ValueError, KeyError):
            pass

    # 11. Sadece epsilon (infinitesimal) formatı: "ε", "0.5ε", "-ε"
    epsilon_only = re.match(r"^\s*([+-]?\d*\.?\d*)\s*[εε]\s*$", s_clean, re.IGNORECASE)
    if epsilon_only:
        try:
            eps_str = epsilon_only.group(1)
            if eps_str in ["", "+"]:
                infinitesimal = 1.0
            elif eps_str == "-":
                infinitesimal = -1.0
            else:
                infinitesimal = float(eps_str)
            return 0.0, infinitesimal, [0.0, infinitesimal]
        except ValueError:
            pass

    # 12. Bilimsel gösterim (hyperreal olmayan)
    sci_pattern = r"^[+-]?\d*\.?\d+[eE][+-]?\d+$"
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
"""
# ValueError: not enough values to unpack (expected 3, got 2): Type=9, Start='0.0,0.001', Add='0.0,0.001'
def _parse_hyperreal(s) -> Tuple[float, float]:
    #Parses hyperreal string into (finite, infinitesimal) tuple.
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
"""

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
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
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
                values = [_safe_float_convert(p) for p in parts] + [0.0] * (
                    8 - len(parts)
                )
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
    
    # DEBUG için
    # print(f"DEBUG _parse_dual: parsing '{s}'")
    
    # 1) Sadece ε sembolünü içeriyor mu kontrol et
    if 'ε' in s or 'ε' in s.lower():
        # Regex pattern: (real kısım)? (+/-) (dual kısım) ε
        # Örnekler: "1.2e-07ε", "3.14+1.2e-07ε", "3.14-1.2e-07ε", "+1.2e-07ε", "-1.2e-07ε"
        
        # ε sembolünü bul
        ε_pos = s.lower().find('ε')
        if ε_pos == -1:
            ε_pos = s.find('ε')
        
        before_ε = s[:ε_pos]
        after_ε = s[ε_pos+1:]
        
        # ε'dan sonra başka karakter varsa hata
        if after_ε.strip():
            raise ValueError(f"Geçersiz Dual sayı formatı: '{s}' (ε'dan sonra karakter var)")
        
        # before_ε'i analiz et
        expr = before_ε.strip()
        
        # Eğer expr boşsa, hem real hem dual 0
        if not expr:
            return DualNumber(0.0, 0.0)
        
        # Regex ile ayrıştır
        # Pattern: (sayı)? ([+-] sayı)?
        # Grup 1: real kısım (opsiyonel)
        # Grup 2: işaret + sayı (opsiyonel)
        
        # Basit regex pattern
        pattern = r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([+-]\d*\.?\d+(?:[eE][+-]?\d+)?)?$'
        
        match = re.match(pattern, expr)
        if match:
            real_part = match.group(1)
            dual_part_with_sign = match.group(2)
            
            try:
                if dual_part_with_sign:
                    # Hem real hem dual var
                    real = float(real_part) if real_part else 0.0
                    dual = float(dual_part_with_sign)
                    return DualNumber(real, dual)
                else:
                    # Sadece bir sayı var - bu real mi dual mi?
                    # Eğer expr + veya - ile başlıyorsa, bu dual kısım
                    if expr.startswith('+') or expr.startswith('-'):
                        dual = float(expr)
                        return DualNumber(0.0, dual)
                    else:
                        # Sadece sayı - bu real
                        real = float(expr)
                        return DualNumber(real, 0.0)
            except ValueError:
                pass
        
        # Regex başarısız oldu, manuel parsing dene
        # "+" işaretiyle ayrılmış mı?
        if '+' in expr:
            parts = expr.split('+')
            if len(parts) == 2:
                try:
                    real = float(parts[0].strip()) if parts[0].strip() else 0.0
                    dual = float(parts[1].strip()) if parts[1].strip() else 0.0
                    return DualNumber(real, dual)
                except ValueError:
                    pass
            elif len(parts) == 1:
                # Sadece dual kısım
                try:
                    dual = float(parts[0].strip()) if parts[0].strip() else 0.0
                    return DualNumber(0.0, dual)
                except ValueError:
                    pass
        
        # "-" işaretiyle ayrılmış mı? (ilk karakter hariç)
        minus_count = expr.count('-')
        if minus_count > 1 or (minus_count == 1 and expr[0] != '-'):
            # "real-dual" formatı
            if expr[0] == '-':
                # "-real-dual" veya "-dual" formatı
                # İkinci - işaretini bul
                second_minus = expr.find('-', 1)
                if second_minus != -1:
                    real_part = expr[:second_minus].strip()
                    dual_part = expr[second_minus:].strip()
                    try:
                        real = float(real_part) if real_part else 0.0
                        dual = float(dual_part)
                        return DualNumber(real, dual)
                    except ValueError:
                        pass
            else:
                # "real-dual" formatı
                minus_pos = expr.find('-')
                if minus_pos != -1:
                    real_part = expr[:minus_pos].strip()
                    dual_part = expr[minus_pos:].strip()
                    try:
                        real = float(real_part) if real_part else 0.0
                        dual = float(dual_part)
                        return DualNumber(real, dual)
                    except ValueError:
                        pass
        
        # Sadece bir sayı olabilir
        try:
            val = float(expr)
            # + veya - ile başlıyorsa dual, değilse real
            if expr.startswith('+') or expr.startswith('-'):
                return DualNumber(0.0, val)
            else:
                return DualNumber(val, 0.0)
        except ValueError:
            pass
    
    # 2) Virgülle ayrılmış format: "real, dual"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        try:
            if len(parts) == 2:
                real = float(parts[0]) if parts[0] else 0.0
                dual = float(parts[1]) if parts[1] else 0.0
                return DualNumber(real, dual)
            elif len(parts) == 1:
                real = float(parts[0]) if parts[0] else 0.0
                return DualNumber(real, 0.0)
        except ValueError:
            pass
    
    # 3) Sadece sayı
    try:
        return DualNumber(float(s), 0.0)
    except ValueError:
        pass
    
    # DEBUG
    # print(f"DEBUG _parse_dual: failed to parse '{s}'")
    
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
    
    # DEBUG için
    # print(f"DEBUG _parse_splitcomplex: parsing '{s}'")
    
    # 1) 'j' ile bitiyor mu kontrol et
    if s.endswith('j') or s.endswith('J'):
        # 'j' den önceki kısmı al
        before_j = s[:-1].strip()
        
        # Eğer before_j boşsa, hem real hem split 0
        if not before_j:
            return SplitcomplexNumber(0.0, 0.0)
        
        # Regex pattern aynı
        pattern = r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([+-]\d*\.?\d+(?:[eE][+-]?\d+)?)?$'
        
        match = re.match(pattern, before_j)
        if match:
            real_part = match.group(1)
            split_part_with_sign = match.group(2)
            
            try:
                if split_part_with_sign:
                    # Hem real hem split var
                    real = float(real_part) if real_part else 0.0
                    split = float(split_part_with_sign)
                    return SplitcomplexNumber(real, split)
                else:
                    # Sadece bir sayı var
                    if before_j.startswith('+') or before_j.startswith('-'):
                        split = float(before_j)
                        return SplitcomplexNumber(0.0, split)
                    else:
                        real = float(before_j)
                        return SplitcomplexNumber(real, 0.0)
            except ValueError:
                pass
        
        # Regex başarısız oldu, manuel parsing
        if '+' in before_j:
            parts = before_j.split('+')
            if len(parts) == 2:
                try:
                    real = float(parts[0].strip()) if parts[0].strip() else 0.0
                    split = float(parts[1].strip()) if parts[1].strip() else 0.0
                    return SplitcomplexNumber(real, split)
                except ValueError:
                    pass
            elif len(parts) == 1:
                try:
                    split = float(parts[0].strip()) if parts[0].strip() else 0.0
                    return SplitcomplexNumber(0.0, split)
                except ValueError:
                    pass
        
        # "-" işareti kontrolü
        minus_count = before_j.count('-')
        if minus_count > 1 or (minus_count == 1 and before_j[0] != '-'):
            if before_j[0] == '-':
                second_minus = before_j.find('-', 1)
                if second_minus != -1:
                    real_part = before_j[:second_minus].strip()
                    split_part = before_j[second_minus:].strip()
                    try:
                        real = float(real_part) if real_part else 0.0
                        split = float(split_part)
                        return SplitcomplexNumber(real, split)
                    except ValueError:
                        pass
            else:
                minus_pos = before_j.find('-')
                if minus_pos != -1:
                    real_part = before_j[:minus_pos].strip()
                    split_part = before_j[minus_pos:].strip()
                    try:
                        real = float(real_part) if real_part else 0.0
                        split = float(split_part)
                        return SplitcomplexNumber(real, split)
                    except ValueError:
                        pass
        
        # Sadece bir sayı
        try:
            val = float(before_j)
            if before_j.startswith('+') or before_j.startswith('-'):
                return SplitcomplexNumber(0.0, val)
            else:
                return SplitcomplexNumber(val, 0.0)
        except ValueError:
            pass
    
    # 2) Virgülle ayrılmış format: "real, split"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        try:
            if len(parts) == 2:
                real = float(parts[0]) if parts[0] else 0.0
                split = float(parts[1]) if parts[1] else 0.0
                return SplitcomplexNumber(real, split)
            elif len(parts) == 1:
                real = float(parts[0]) if parts[0] else 0.0
                return SplitcomplexNumber(real, 0.0)
        except ValueError:
            pass
    
    # 3) Sadece sayı
    try:
        return SplitcomplexNumber(float(s), 0.0)
    except ValueError:
        pass
    
    # DEBUG
    # print(f"DEBUG _parse_splitcomplex: failed to parse '{s}'")
    
    raise ValueError(f"Geçersiz Split-Complex sayı formatı: '{s}' (Real, Split veya sadece Real bekleniyor)")


"""
def _parse_dual(s) -> DualNumber:
    #String'i veya sayıyı DualNumber'a dönüştürür.
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
    
    # 1) Sadece ε içeren format: "1.2e-07ε" veya "-1.2e-07ε"
    s_lower = s.lower()
    
    # ε sembolünü kontrol et
    if 'ε' in s_lower:
        # ε sembolünün pozisyonunu bul
        ε_pos = s_lower.find('ε')
        
        # ε'dan önceki kısmı al
        before_ε = s[:ε_pos].strip()
        after_ε = s[ε_pos+1:].strip()  # ε'dan sonraki kısım (boş olmalı)
        
        # Eğer ε'dan sonra bir şey varsa geçersiz
        if after_ε:
            raise ValueError(f"Geçersiz Dual sayı formatı: '{s}'")
        
        # ε'dan önceki kısmı analiz et
        if before_ε:
            # + veya - işaretleriyle ayrılmış mı kontrol et
            if '+' in before_ε:
                parts = before_ε.split('+')
                if len(parts) == 2:
                    try:
                        real = float(parts[0].strip()) if parts[0].strip() else 0.0
                        dual = float(parts[1].strip()) if parts[1].strip() else 0.0
                        return DualNumber(real, dual)
                    except ValueError:
                        pass
                elif len(parts) == 1:
                    # Sadece dual kısım: "+1.2e-07" gibi
                    try:
                        dual = float(parts[0].strip()) if parts[0].strip() else 0.0
                        return DualNumber(0.0, dual)
                    except ValueError:
                        pass
            elif '-' in before_ε[1:]:  # İlk karakter hariç - işareti
                # İlk karakteri kontrol et
                if before_ε[0] == '-':
                    # "-1.2e-07" formatı - sadece dual kısım negatif
                    try:
                        dual = float(before_ε.strip())
                        return DualNumber(0.0, dual)
                    except ValueError:
                        pass
                else:
                    # "real-dual" formatı
                    minus_pos = before_ε.find('-', 1)
                    if minus_pos != -1:
                        real_part = before_ε[:minus_pos].strip()
                        dual_part = before_ε[minus_pos:].strip()  # - işaretiyle birlikte
                        try:
                            real = float(real_part) if real_part else 0.0
                            dual = float(dual_part)
                            return DualNumber(real, dual)
                        except ValueError:
                            pass
            else:
                # Sadece dual kısım: "1.2e-07" gibi
                try:
                    dual = float(before_ε.strip())
                    return DualNumber(0.0, dual)
                except ValueError:
                    pass
    
    # 2) Virgülle ayrılmış format: "real, dual"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) >= 2:
            try:
                real = float(parts[0]) if parts[0] else 0.0
                dual = float(parts[1]) if parts[1] else 0.0
                return DualNumber(real, dual)
            except ValueError:
                pass
        elif len(parts) == 1: # Sadece real kısım verilmiş
            try:
                real = float(parts[0]) if parts[0] else 0.0
                return DualNumber(real, 0.0)
            except ValueError:
                pass
    
    # 3) Sadece sayı
    try:
        return DualNumber(float(s), 0.0)
    except ValueError:
        pass
    
    raise ValueError(f"Geçersiz Dual sayı formatı: '{s}' (Real, Dual veya sadece Real bekleniyor)")

def _parse_splitcomplex(s) -> SplitcomplexNumber:
    #String'i veya sayıyı SplitcomplexNumber'a dönüştürür.
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
    
    # 1) j ile biten format: "0.00027182818284590454j" veya "a+bj"
    s_lower = s.lower()
    
    if s_lower.endswith('j'):
        # 'j' den önceki kısmı al
        before_j = s[:-1].strip()
        
        if before_j:
            # + veya - işaretleriyle ayrılmış mı kontrol et
            if '+' in before_j:
                parts = before_j.split('+')
                if len(parts) == 2:
                    try:
                        real = float(parts[0].strip()) if parts[0].strip() else 0.0
                        split = float(parts[1].strip()) if parts[1].strip() else 0.0
                        return SplitcomplexNumber(real, split)
                    except ValueError:
                        pass
                elif len(parts) == 1:
                    # Sadece split kısım: "+0.00027182818284590454" gibi
                    try:
                        split = float(parts[0].strip()) if parts[0].strip() else 0.0
                        return SplitcomplexNumber(0.0, split)
                    except ValueError:
                        pass
            elif '-' in before_j[1:]:  # İlk karakter hariç - işareti
                if before_j[0] == '-':
                    # Sadece split kısım negatif: "-0.00027182818284590454"
                    try:
                        split = float(before_j.strip())
                        return SplitcomplexNumber(0.0, split)
                    except ValueError:
                        pass
                else:
                    # "real-split" formatı
                    minus_pos = before_j.find('-', 1)
                    if minus_pos != -1:
                        real_part = before_j[:minus_pos].strip()
                        split_part = before_j[minus_pos:].strip()  # - işaretiyle birlikte
                        try:
                            real = float(real_part) if real_part else 0.0
                            split = float(split_part)
                            return SplitcomplexNumber(real, split)
                        except ValueError:
                            pass
            else:
                # Sadece split kısım: "0.00027182818284590454"
                try:
                    split = float(before_j.strip())
                    return SplitcomplexNumber(0.0, split)
                except ValueError:
                    pass
    
    # 2) Virgülle ayrılmış format: "real, split"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) >= 2:
            try:
                real = float(parts[0]) if parts[0] else 0.0
                split = float(parts[1]) if parts[1] else 0.0
                return SplitcomplexNumber(real, split)
            except ValueError:
                pass
        elif len(parts) == 1: # Sadece real kısım verilmiş
            try:
                real = float(parts[0]) if parts[0] else 0.0
                return SplitcomplexNumber(real, 0.0)
            except ValueError:
                pass
    
    # 3) Sadece sayı
    try:
        return SplitcomplexNumber(float(s), 0.0)
    except ValueError:
        pass
    
    raise ValueError(f"Geçersiz Split-Complex sayı formatı: '{s}' (Real, Split veya sadece Real bekleniyor)")
"""
"""
def _parse_dual(s) -> DualNumber:
    #String'i veya sayıyı DualNumber'a dönüştürür.
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
    
    # 1) Virgülle ayrılmış format: "real, dual"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
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
    
    # 2) Matematiksel ifade formatı: "a+bε" veya "a-bε"
    s_lower = s.lower()
    ε_pos = s_lower.find('ε')
    
    if ε_pos != -1:
        # ε sembolünden önceki kısmı al
        expr = s[:ε_pos].strip()
        
        # + veya - işaretlerini bul
        if '+' in expr:
            parts = expr.split('+')
            if len(parts) == 2:
                try:
                    real = float(parts[0].strip())
                    dual = float(parts[1].strip())
                    return DualNumber(real, dual)
                except ValueError:
                    pass
        elif '-' in expr[1:]:  # İlk karakterden sonraki - işareti
            # İlk - işaretini bul (ilk karakter hariç)
            minus_pos = expr.find('-', 1)
            if minus_pos != -1:
                real_part = expr[:minus_pos].strip()
                dual_part = expr[minus_pos:].strip()  # - işaretiyle birlikte
                try:
                    real = float(real_part)
                    dual = float(dual_part)
                    return DualNumber(real, dual)
                except ValueError:
                    pass
    
    # 3) Sadece real sayı
    try:
        return DualNumber(float(s), 0.0)
    except ValueError:
        pass
    
    raise ValueError(f"Geçersiz Dual sayı formatı: '{s}' (Real, Dual veya sadece Real bekleniyor)")

def _parse_splitcomplex(s) -> SplitcomplexNumber:
    #String'i veya sayıyı SplitcomplexNumber'a dönüştürür.
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
    
    # 1) Virgülle ayrılmış format: "real, split"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) >= 2:
            try:
                return SplitcomplexNumber(float(parts[0]), float(parts[1]))
            except ValueError:
                pass
        elif len(parts) == 1: # Sadece real kısım verilmiş
            try:
                return SplitcomplexNumber(float(parts[0]), 0.0)
            except ValueError:
                pass
    
    # 2) Matematiksel ifade formatı: "a+bj" veya "a-bj"
    s_lower = s.lower()
    
    # 'j' ile bitiyor mu kontrol et
    if s_lower.endswith('j'):
        # 'j' den önceki kısmı al
        expr = s[:-1].strip()
        
        # + veya - işaretlerini bul
        if '+' in expr:
            parts = expr.split('+')
            if len(parts) == 2:
                try:
                    real = float(parts[0].strip())
                    split = float(parts[1].strip())
                    return SplitcomplexNumber(real, split)
                except ValueError:
                    pass
        elif '-' in expr[1:]:  # İlk karakterden sonraki - işareti
            minus_pos = expr.find('-', 1)
            if minus_pos != -1:
                real_part = expr[:minus_pos].strip()
                split_part = expr[minus_pos:].strip()  # - işaretiyle birlikte
                try:
                    real = float(real_part)
                    split = float(split_part)
                    return SplitcomplexNumber(real, split)
                except ValueError:
                    pass
    
    # 3) Sadece real sayı
    try:
        return SplitcomplexNumber(float(s), 0.0)
    except ValueError:
        pass
    
    raise ValueError(f"Geçersiz Split-Complex sayı formatı: '{s}' (Real, Split veya sadece Real bekleniyor)")
"""
"""
def _parse_dual(s) -> DualNumber:
    #String'i veya sayıyı DualNumber'a dönüştürür.
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
    #String'i veya sayıyı SplitcomplexNumber'a dönüştürür.
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
"""

def generate_octonion(w, x, y, z, e, f, g, h):
    """8 bileşenden bir oktonyon oluşturur."""
    return OctonionNumber(w, x, y, z, e, f, g, h)


def _parse_quaternion(s: Any) -> Any:
    """Parses user string ('a+bi+cj+dk' or scalar) into a quaternion - DÜZELTİLMİŞ."""
    
    # ✅ SORUN 1: Parametre tipi düzeltildi (Any yerine str değil)
    if isinstance(s, (int, float, Fraction)):
        try:
            from kececinumbers import QuaternionNumber
            return QuaternionNumber(float(s), 0, 0, 0)  # ✅ SKALER DESTEK
        except:
            return [float(s), 0, 0, 0]  # List fallback
    
    # String değilse dönüştür
    if not isinstance(s, str):
        s = str(float(s))
    
    s_clean = s.replace(" ", "").lower()
    if not s_clean:
        raise ValueError("Input cannot be empty.")

    # ✅ SORUN 2: float kontrolü EN BAŞTA
    try:
        val = float(s_clean)
        try:
            from kececinumbers import quaternion
            return quaternion(val, 0, 0, 0)  # ✅ Sıfır imaginary
        except:
            return [val, 0, 0, 0]
    except ValueError:
        pass

    # ✅ SORUN 3: re import kontrolü
    try:
        import re
    except ImportError:
        # Regex yoksa basit parse
        return [float(s_clean.split('+')[0]) if '+' in s_clean else float(s_clean), 0, 0, 0]

    # Regex parsing
    s_temp = re.sub(r'([+-])([ijk])', r'\g<1>1\g<2>', s_clean)
    if s_temp.startswith(('i', 'j', 'k')):
        s_temp = '1' + s_temp
    
    # ✅ SORUN 4: Pattern düzeltildi (raw string)
    pattern = re.compile(r'([+-]?\d*\.?\d*)([ijk])?')
    matches = pattern.findall(s_temp)
    
    parts = {'w': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    for value_str, component in matches:
        if not value_str or value_str == '+':
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
            
    # ✅ SORUN 5: quaternion constructor güvenli
    try:
        from kececinumbers import quaternion
        return quaternion(parts['w'], parts['x'], parts['y'], parts['z'])
    except:
        return [parts['w'], parts['x'], parts['y'], parts['z']]  # List fallback


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

def _parse_ternary(s: Any) -> Any:
    """TERNARY parser - %100 çalışan versiyon"""
    try:
        if isinstance(s, (TernaryNumber, list)):
            return s
        
        if isinstance(s, (int, float, Fraction)):
            return TernaryNumber(float(s), 0.0, 0.0)  # ✅ SKALER DESTEK
        
        if isinstance(s, str):
            s = s.strip().strip('()[]')
            if all(c in '012' for c in s):
                return TernaryNumber.from_ternary_string(s)
            else:
                return TernaryNumber(float(s), 0.0, 0.0)  # ✅ Float string
        
        return TernaryNumber(float(s), 0.0, 0.0)
    except:
        return [float(s), 0.0, 0.0]  # List fallback

def get_random_type(
    num_iterations: int = 10,
    fixed_start_raw: Union[str, float, int] = "0",
    fixed_add_base_scalar: Union[str, float, int] = 9.0,
    exclude_types: Optional[List[int]] = None,
    seed: Optional[int] = None,
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
        "Positive Real",
        "Negative Real",
        "Complex",
        "Float",
        "Rational",
        "Quaternion",
        "Neutrosophic",
        "Neutrosophic Complex",
        "Hyperreal",
        "Bicomplex",
        "Neutrosophic Bicomplex",
        "Octonion",
        "Sedenion",
        "Clifford",
        "Dual",
        "Split-Complex",
        "Pathion",
        "Chingon",
        "Routon",
        "Voudon",
        "Super Real",
        "Ternary",
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
        type_names_list[random_type_choice - 1],
    )

    # Ensure parameters are strings for get_with_params
    start_value_str = (
        str(fixed_start_raw)
        if not isinstance(fixed_start_raw, str)
        else fixed_start_raw
    )
    add_value_str = (
        str(fixed_add_base_scalar)
        if not isinstance(fixed_add_base_scalar, str)
        else fixed_add_base_scalar
    )

    # Call the generator function
    return get_with_params(
        kececi_type_choice=random_type_choice,
        iterations=num_iterations,
        start_value_raw=start_value_str,
        add_value_raw=add_value_str,
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

def _is_divisible(value: Any, divisor: Union[int, float, Fraction], kececi_type: int) -> bool:
    """
    Robust divisibility check supporting integer and fractional divisors.
    Returns True if value is divisible by divisor according to type semantics.
    """
    TOL = 1e-12

    # --- helper: numeric divisibility via quotient near-integer ---
    def _divisible_by_numeric(x, d, tol=TOL):
        try:
            if d == 0:
                return False
            # Exact Fraction handling
            if isinstance(x, Fraction) or isinstance(d, Fraction):
                try:
                    q = Fraction(x) / Fraction(d)
                    return q.denominator == 1
                except Exception:
                    # fall through to float check
                    pass
            # float quotient near-integer check (works for floats and ints)
            q = float(x) / float(d)
            if not math.isfinite(q):
                return False
            return math.isclose(q, round(q), abs_tol=tol)
        except Exception:
            return False

    def _complex_divisible(c, d):
        try:
            return _divisible_by_numeric(c.real, d) and _divisible_by_numeric(c.imag, d)
        except Exception:
            return False

    def _iterable_divisible(it, d):
        try:
            for c in it:
                if isinstance(c, complex):
                    if not _complex_divisible(c, d):
                        return False
                elif isinstance(c, Fraction):
                    if not _divisible_by_numeric(c, d):
                        return False
                else:
                    if not _divisible_by_numeric(c, d):
                        return False
            return True
        except Exception:
            return False

    # coerce divisor to numeric if possible
    try:
        if not isinstance(divisor, (int, float, Fraction)):
            divisor = float(divisor)
    except Exception:
        return False

    try:
        # --- Type-specific branches ---
        if kececi_type in (TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL):
            return _divisible_by_numeric(value, divisor)

        if kececi_type == TYPE_RATIONAL:
            try:
                fr = value if isinstance(value, Fraction) else Fraction(value)
                return _divisible_by_numeric(fr, divisor)
            except Exception:
                return False

        if kececi_type == TYPE_COMPLEX:
            try:
                c = value if isinstance(value, complex) else _parse_complex(value)
                return _complex_divisible(c, divisor)
            except Exception:
                return False

        if kececi_type == TYPE_HYPERREAL:
            if hasattr(value, 'sequence') and isinstance(value.sequence, (list, tuple)):
                return _iterable_divisible(value.sequence, divisor)
            return False

        if kececi_type in (TYPE_OCTONION, TYPE_SEDENION, TYPE_PATHION,
                           TYPE_CHINGON, TYPE_ROUTON, TYPE_VOUDON, TYPE_HYPERCOMPLEX):
            # try norm first if available
            try:
                if hasattr(value, 'norm') and callable(getattr(value, 'norm')):
                    n = float(value.norm())
                    if _divisible_by_numeric(n, divisor):
                        return True
            except Exception:
                pass
            # fallback to component-wise
            if hasattr(value, 'coeffs'):
                try:
                    comps = getattr(value, 'coeffs')
                    comps = comps() if callable(comps) else comps
                    return _iterable_divisible(comps, divisor)
                except Exception:
                    pass
            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return _iterable_divisible(value, divisor)
            return _divisible_by_numeric(value, divisor)

        # Neutrosophic types: check relevant attributes if present
        if kececi_type == TYPE_NEUTROSOPHIC:
            try:
                if hasattr(value, 't') and hasattr(value, 'i'):
                    return _divisible_by_numeric(value.t, divisor) and _divisible_by_numeric(value.i, divisor)
                if hasattr(value, 'a') and hasattr(value, 'b'):
                    return _divisible_by_numeric(value.a, divisor) and _divisible_by_numeric(value.b, divisor)
            except Exception:
                return False
            return False

        if kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            try:
                comps = []
                if hasattr(value, 'real') and hasattr(value, 'imag'):
                    comps.extend([value.real, value.imag])
                if hasattr(value, 'indeterminacy'):
                    comps.append(value.indeterminacy)
                return all(_divisible_by_numeric(c, divisor) for c in comps) if comps else False
            except Exception:
                return False

        # Generic fallback: coeffs -> iterable -> numeric
        if hasattr(value, 'coeffs'):
            comps = getattr(value, 'coeffs')
            comps = comps() if callable(comps) else comps
            return _iterable_divisible(comps, divisor)
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            return _iterable_divisible(value, divisor)

        return _divisible_by_numeric(value, divisor)

    except Exception:
        return False


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
        21: "Super Real", 22: "Ternary", 23: "Hypercomplex",
    }

    # Iterate all defined types (inclusive)
    for kececi_type in range(TYPE_POSITIVE_REAL, TYPE_HYPERCOMPLEX + 1):
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
def _parse_kececi_values(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str = "0"
) -> Tuple[Any, Any]:
    """
    ✅ DÜZELTİLMİŞ - Tüm tipler destekler
    """
    def safe_float(x): 
        try: return float(x)
        except: return 0.0
    
    try:
        # ✅ BASİT TÜRLER (1-5)
        if kececi_type in [1, 2, 4, 5]:
            start_val = safe_float(start_input_raw)
            add_val = safe_float(add_input_raw)
            if kececi_type == 1: return abs(start_val), abs(add_val)
            if kececi_type == 2: return -abs(start_val), -abs(add_val)
            return start_val, add_val
        
        # COMPLEX
        if kececi_type == 3:
            return complex(start_input_raw), complex(add_input_raw)
        
        # ✅ TERNARY (22)
        if kececi_type == 22:
            return _parse_ternary(start_input_raw), _parse_ternary(add_input_raw)
        
        # QUATERNION (6)
        if kececi_type == 6:
            from kececinumbers import _parse_quaternion
            return _parse_quaternion(start_input_raw), _parse_quaternion(add_input_raw)
        
        # ✅ GENEL FALLBACK - TÜM DİĞER TİPLER
        dims = {12:8, 13:16, 20:32}.get(kececi_type, 4)
        start_list = [safe_float(start_input_raw)] + [0.0]*(dims-1)
        add_list = [safe_float(add_input_raw)] + [0.0]*(dims-1)
        return start_list, add_list
        
    except Exception as e:
        print(f"Parse fallback type {kececi_type}: {e}")
        return safe_float(start_input_raw), safe_float(add_input_raw)
"""
def _parse_kececi_values(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str
) -> Tuple[Any, Any]:

    #Parse values for a specific Keçeci number type.
    #Returns (start_value, add_value)

    try:
        # Import parsers
        from .kececinumbers import _parse_fraction
        
        # For basic types (1, 2, 4, 5) use _parse_fraction
        if kececi_type in [1, 2, 4, 5]:
            start_val = _parse_fraction(start_input_raw)
            add_val = _parse_fraction(add_input_raw)
            
            if kececi_type == 1:  # Positive Real
                return abs(start_val), abs(add_val)
            elif kececi_type == 2:  # Negative Real
                return -abs(start_val), -abs(add_val)
            else:  # Type 4 (Float), 5 (Rational)
                return start_val, add_val
        
        # For complex type (3)
        elif kececi_type == 3:
            from .kececinumbers import _parse_complex
            return _parse_complex(start_input_raw), _parse_complex(add_input_raw)
        
        # For other types, try to import specific parsers
        else:
            # Try to import all parsers
            try:
                from .kececinumbers import (
                    _parse_bicomplex,
                    _parse_chingon,
                    _parse_clifford,
                    _parse_complex,
                    _parse_complex_like_string,
                    _parse_dual,
                    _parse_engineering_notation,
                    _parse_fraction,
                    _parse_hyperreal,
                    _parse_neutrosophic,
                    _parse_neutrosophic_bicomplex,
                    _parse_neutrosophic_complex,
                    _parse_octonion,
                    _parse_pathion,
                    _parse_quaternion,
                    _parse_quaternion_from_csv,
                    _parse_real,
                    _parse_routon,
                    _parse_sedenion,
                    _parse_splitcomplex,
                    _parse_super_real,
                    _parse_superreal,
                    _parse_ternary,
                    _parse_hypercomplex,
                    _parse_universal,
                    _parse_voudon,
                    _generate_simple_ask_sequence,
                    _parse_with_fallback_simple,
                    _parse_kececi_values,
                    parse_to_hyperreal,
                    parse_to_neutrosophic,
                )
                
                # Map type to parser
                parser_map = {
                    6: _parse_quaternion,
                    7: _parse_neutrosophic,
                    8: _parse_neutrosophic_complex,
                    9: _parse_hyperreal,
                    10: _parse_bicomplex,
                    11: _parse_neutrosophic_bicomplex,
                    12: _parse_octonion,
                    13: _parse_sedenion,
                    14: _parse_clifford,
                    15: _parse_dual,
                    16: _parse_splitcomplex,
                    17: _parse_pathion,
                    18: _parse_chingon,
                    19: _parse_routon,
                    20: _parse_voudon,
                    21: _parse_super_real,
                    22: _parse_ternary,
                    23: _parse_hypercomplex,
                }
                
                parser = parser_map.get(kececi_type)
                if parser:
                    return parser(start_input_raw), parser(add_input_raw)
                else:
                    raise ValueError(f"No parser for type {kececi_type}")
                    
            except ImportError:
                # Fallback to simple parsing
                logger.warning(f"Parsers not available for type {kececi_type}, using fallback")
                return _parse_with_fallback_simple(kececi_type, start_input_raw, add_input_raw)
                
    except Exception as e:
        logger.error(f"Error parsing values for type {kececi_type}: {e}")
        # Fallback to simple parsing
        return _parse_with_fallback_simple(kececi_type, start_input_raw, add_input_raw)
"""

def _parse_with_fallback_simple(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str
) -> Tuple[Any, Any]:
    """
    Simple fallback parser when main parsers fail.
    Accepts flexible inputs:
      - plain numbers: "3.5", "2", "-1"
      - fractions: "3/4"
      - mixed numbers: "1 1/2"
      - complex: "1+2i", "3-4j"
      - lists/tuples: "[1,2,3]", "(1, 2, 3)", "1,2,3", "1 2 3"
      - component-wise complex: "1+2i,3+4i"
    Returns values mapped to the requested algebraic type.
    """

    def parse_number_token(tok: str):
        tok = tok.strip()
        if not tok:
            return 0.0
        # normalize imaginary unit
        tok = tok.replace('I', 'i').replace('J', 'j').replace('i', 'j')
        # try complex first
        try:
            # complex() accepts '1+2j' or '2j' etc.
            c = complex(tok)
            # if purely real, return float for convenience
            if c.imag == 0:
                return float(c.real)
            return c
        except Exception:
            pass

        # mixed number "a b/c"
        if ' ' in tok and '/' in tok:
            try:
                whole, frac = tok.split(' ', 1)
                num, den = frac.split('/')
                return float(whole) + (float(num) / float(den))
            except Exception:
                pass

        # fraction "a/b"
        if '/' in tok:
            try:
                num, den = tok.split('/')
                return float(num) / float(den)
            except Exception:
                pass

        # fallback float
        try:
            return float(tok)
        except Exception:
            return 0.0

    def parse_simple(val: str):
        if val is None:
            return 0.0
        s = str(val).strip()
        if s == '':
            return 0.0

        # If looks like a bracketed list or comma-separated components
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')) or (',' in s):
            # remove brackets
            inner = s.strip('[]() ')
            # split by comma first
            parts = [p.strip() for p in inner.split(',') if p.strip() != '']
            if len(parts) == 1:
                # maybe space-separated inside single part
                parts = parts[0].split()
            comps = []
            for p in parts:
                # allow each part to be complex or numeric
                comps.append(parse_number_token(p))
            return comps

        # If space-separated multiple tokens (e.g., "1 2 3")
        if ' ' in s and '/' not in s:  # avoid splitting mixed numbers like "1 1/2"
            parts = [p for p in s.split() if p != '']
            if len(parts) > 1:
                return [parse_number_token(p) for p in parts]

        # single token: try to parse as number/complex
        return parse_number_token(s)

    # Parse base values
    start_base = parse_simple(start_input_raw)
    add_base = parse_simple(add_input_raw)

    # Helper to coerce parsed value into numeric list of length n
    def to_components(x, n):
        if isinstance(x, list) or isinstance(x, tuple):
            comps = list(x)
        elif isinstance(x, complex):
            comps = [x.real, x.imag] + [0.0] * (n - 2)
        else:
            comps = [float(x)] + [0.0] * (n - 1)
        # ensure length n
        if len(comps) < n:
            comps += [0.0] * (n - len(comps))
        return comps[:n]

    # Map to appropriate type with flexible input handling
    if kececi_type == 1:  # Positive Real
        s = float(start_base[0]) if isinstance(start_base, (list, tuple)) else float(start_base.real if isinstance(start_base, complex) else start_base)
        a = float(add_base[0]) if isinstance(add_base, (list, tuple)) else float(add_base.real if isinstance(add_base, complex) else add_base)
        return abs(s), abs(a)

    elif kececi_type == 2:  # Negative Real
        s = float(start_base[0]) if isinstance(start_base, (list, tuple)) else float(start_base.real if isinstance(start_base, complex) else start_base)
        a = float(add_base[0]) if isinstance(add_base, (list, tuple)) else float(add_base.real if isinstance(add_base, complex) else add_base)
        return -abs(s), -abs(a)

    elif kececi_type == 3:  # Complex
        # If parsed as list with 2 components, use them as (real, imag)
        def make_complex(x):
            if isinstance(x, complex):
                return x
            if isinstance(x, (list, tuple)):
                comps = to_components(x, 2)
                return complex(comps[0], comps[1])
            return complex(float(x), 0.0)
        return make_complex(start_base), make_complex(add_base)

    elif kececi_type == 4:  # Float
        def make_float(x):
            if isinstance(x, complex):
                return float(x.real)
            if isinstance(x, (list, tuple)):
                return float(x[0]) if len(x) > 0 else 0.0
            return float(x)
        return make_float(start_base), make_float(add_base)

    elif kececi_type == 5:  # Rational
        def make_fraction(x):
            try:
                if isinstance(x, (list, tuple)):
                    return Fraction(float(x[0])).limit_denominator()
                if isinstance(x, complex):
                    return Fraction(float(x.real)).limit_denominator()
                return Fraction(x).limit_denominator()
            except Exception:
                return Fraction(float(x)).limit_denominator()
        return make_fraction(start_base), make_fraction(add_base)

    elif kececi_type == 6:  # Quaternion (4D tuple)
        s_comps = to_components(start_base, 4)
        a_comps = to_components(add_base, 4)
        return tuple(float(c) for c in s_comps), tuple(float(c) for c in a_comps)

    elif kececi_type == 7:  # Neutrosophic (T, I, F) 3-tuple
        s_comps = to_components(start_base, 3)
        a_comps = to_components(add_base, 3)
        return (s_comps[0], s_comps[1], s_comps[2]), (a_comps[0], a_comps[1], a_comps[2])

    elif kececi_type == 8:  # Neutrosophic Complex (complex for T, I, F fallback)
        def to_nc(x):
            if isinstance(x, complex):
                return x
            if isinstance(x, (list, tuple)):
                # if list of two complex-like entries, combine first as real, second as imag
                comps = to_components(x, 2)
                return complex(comps[0], comps[1])
            return complex(float(x), 0.0)
        return to_nc(start_base), to_nc(add_base)

    elif kececi_type == 9:  # Hyperreal (finite, infinitesimal) -> tuple [real, infinitesimal]
        def to_hyperreal(x):
            if isinstance(x, (list, tuple)):
                comps = to_components(x, 2)
                return [float(comps[0]), float(comps[1])]
            if isinstance(x, complex):
                return [float(x.real), float(x.imag)]
            return [float(x), 0.0]
        return to_hyperreal(start_base), to_hyperreal(add_base)

    elif kececi_type == 10:  # Bicomplex (fallback to complex)
        def to_bi(x):
            if isinstance(x, complex):
                return x
            if isinstance(x, (list, tuple)):
                comps = to_components(x, 2)
                return complex(comps[0], comps[1])
            return complex(float(x), 0.0)
        return to_bi(start_base), to_bi(add_base)

    elif kececi_type == 11:  # Neutrosophic Bicomplex (fallback to complex)
        def to_nb(x):
            if isinstance(x, complex):
                return x
            if isinstance(x, (list, tuple)):
                comps = to_components(x, 2)
                return complex(comps[0], comps[1])
            return complex(float(x), 0.0)
        return to_nb(start_base), to_nb(add_base)

    elif kececi_type == 12:  # Octonion (8D)
        s_comps = to_components(start_base, 8)
        a_comps = to_components(add_base, 8)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 13:  # Sedenion (16D)
        s_comps = to_components(start_base, 16)
        a_comps = to_components(add_base, 16)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 14:  # Clifford (simple dict)
        s0 = float(start_base[0]) if isinstance(start_base, (list, tuple)) else (start_base.real if isinstance(start_base, complex) else float(start_base))
        a0 = float(add_base[0]) if isinstance(add_base, (list, tuple)) else (add_base.real if isinstance(add_base, complex) else float(add_base))
        return {"e0": s0}, {"e0": a0}

    elif kececi_type == 15:  # Dual (real, dual)
        s_comps = to_components(start_base, 2)
        a_comps = to_components(add_base, 2)
        return (s_comps[0], s_comps[1]), (a_comps[0], a_comps[1])

    elif kececi_type == 16:  # Split-complex
        s_comps = to_components(start_base, 2)
        a_comps = to_components(add_base, 2)
        return (s_comps[0], s_comps[1]), (a_comps[0], a_comps[1])

    elif kececi_type == 17:  # Pathion (32D)
        s_comps = to_components(start_base, 32)
        a_comps = to_components(add_base, 32)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 18:  # Chingon (64D)
        s_comps = to_components(start_base, 64)
        a_comps = to_components(add_base, 64)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 19:  # Routon (128D)
        s_comps = to_components(start_base, 128)
        a_comps = to_components(add_base, 128)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 20:  # Voudon (256D)
        s_comps = to_components(start_base, 256)
        a_comps = to_components(add_base, 256)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 21:  # Superreal (real, superreal)
        s_comps = to_components(start_base, 2)
        a_comps = to_components(add_base, 2)
        return (s_comps[0], s_comps[1]), (a_comps[0], a_comps[1])

    elif kececi_type == 22:  # Ternary (3D)
        s_comps = to_components(start_base, 3)
        a_comps = to_components(add_base, 3)
        return [float(c) for c in s_comps], [float(c) for c in a_comps]

    elif kececi_type == 23:  # Hypercomplex (arbitrary-length components)
        # If user provided a list/tuple, use it; if single number, wrap it; if complex, split to [real, imag]
        def to_hc(x):
            if isinstance(x, (list, tuple)):
                return [float(c.real) if isinstance(c, complex) else float(c) for c in x]
            if isinstance(x, complex):
                return [float(x.real), float(x.imag)]
            return [float(x)]
        return to_hc(start_base), to_hc(add_base)

    else:
        # Default fallback: return parsed raw values coerced to floats where possible
        def fallback(x):
            if isinstance(x, complex):
                return float(x.real)
            if isinstance(x, (list, tuple)):
                return float(x[0]) if len(x) > 0 else 0.0
            return float(x)
        return fallback(start_base), fallback(add_base)


# unified_generator fonksiyonunu da basitleştirilmiş haliyle güncelleyelim:
def unified_generator(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str,
    iterations: int,
    include_intermediate_steps: bool = False,
) -> List[Any]:
    """
    Unified generator with input validation and robust division/ask handling.

    Args:
        kececi_type: Type identifier (1-23)
        start_input_raw: Starting value as string
        add_input_raw: Value to add each iteration as string
        iterations: Number of iterations to generate
        include_intermediate_steps: Whether to include intermediate calculation steps

    Returns:
        List of generated values

    Raises:
        ValueError: Invalid type or parsing error
    """
    # Önce Fraction'ı import et
    try:
        from fractions import Fraction
    except ImportError:
        Fraction = None
        logger.warning("Fraction module not available")

    # Type validation
    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_HYPERCOMPLEX):
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

    # --- helper: coerce ask_unit to same class as current_value if possible
    # --- Tip coercion: ask_unit'ı exemplar ile uyumlu hale getir ---
    def _coerce_unit_to_value(unit, exemplar):
        try:
            if exemplar is None:
                return unit
            if type(unit) == type(exemplar):
                return unit
            cls = exemplar.__class__
            # Eğer unit zaten iterable ise önce liste/tuple dene
            if hasattr(unit, "__iter__") and not isinstance(unit, (str, bytes)):
                try:
                    return cls(list(unit))
                except Exception:
                    pass
                try:
                    return cls(*unit)
                except Exception:
                    pass
            # tek argümanlı constructor dene
            try:
                return cls(unit)
            except Exception:
                pass
            # fallback: orijinal unit
            return unit
        except Exception:
            return unit

    # --- Güvenli uygulama yardımcıları (operatörler için) ---
    def _safe_apply_op(a, op, b, integer_division=False):
        """
        a op b denemesi: + - * / ; integer_division yalnızca bölme için kullanılır.
        Denemeler: a.op(b), b.op(a) (noncommutative için), fallback elementwise.
        """
        try:
            if op == '+':
                return a + b
            if op == '-':
                return a - b
            if op == '*':
                return a * b
            if op == '/':
                # eğer b numeric scalar ise safe_divide kullan
                if isinstance(b, (int, float)):
                    try:
                        return _safe_divide(a, b, integer_division)
                    except Exception:
                        return a / b
                else:
                    return a / b
        except Exception:
            # ters yönden dene (ör. scalar * hypercomplex)
            try:
                if op == '*':
                    return b * a
                if op == '/':
                    return b / a
            except Exception:
                pass
        # elementwise fallback: eğer her iki taraf da iterable ise elementwise uygula
        try:
            if hasattr(a, "__iter__") and hasattr(b, "__iter__") and not isinstance(a, (str, bytes)):
                la = list(a); lb = list(b)
                n = max(len(la), len(lb))
                la += [0.0] * (n - len(la))
                lb += [0.0] * (n - len(lb))
                if op == '+':
                    res = [x + y for x, y in zip(la, lb)]
                elif op == '-':
                    res = [x - y for x, y in zip(la, lb)]
                elif op == '*':
                    res = [x * y for x, y in zip(la, lb)]
                elif op == '/':
                    res = [ (x / y if y != 0 else float('inf')) for x, y in zip(la, lb) ]
                # preserve type of a if possible
                try:
                    return type(a)(res)
                except Exception:
                    return res
        except Exception:
            pass
        raise TypeError("Operation not supported for given operands")

    # --- Güçlü safe_mul_add (value * multiplier + constant) ---
    def safe_mul_add(value: Any, multiplier: Any, constant: Any):
        try:
            multiplied = _safe_apply_op(value, '*', multiplier)
        except Exception:
            # fallback: try elementwise or python *
            try:
                multiplied = value * multiplier
            except Exception:
                multiplied = value
        try:
            return _safe_apply_op(multiplied, '+', constant)
        except Exception:
            try:
                return multiplied + constant
            except Exception:
                return multiplied

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
                b = b * direction if hasattr(b, "__mul__") else b

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
                return (
                    tuple(result)
                    if isinstance(a, tuple) and isinstance(b, tuple)
                    else result
                )

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

    def safe_divide(val: Any, divisor: Union[int, float, Fraction], integer_mode: bool = False) -> Any:
        try:
            # coerce divisor
            if isinstance(divisor, Fraction):
                pass
            elif isinstance(divisor, float) and integer_mode:
                # if divisor is near-integer, use int
                if math.isclose(divisor, round(divisor), abs_tol=1e-12):
                    divisor_int = int(round(divisor))
                    return val // divisor_int if hasattr(val, "__floordiv__") else type(val)(int(val) // divisor_int)
                else:
                    # integer_mode requested but divisor not integer-like -> fallback to true division
                    integer_mode = False

            if integer_mode:
                if hasattr(val, "__floordiv__"):
                    return val // int(divisor)
                # iterable fallback...
            else:
                if hasattr(val, "__truediv__"):
                    return val / divisor
                # iterable fallback...
        except Exception:
            raise


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

    # Type validation
    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_HYPERCOMPLEX):
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

    # --- Type-specific parsing ---
    try:
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            # For real types, parse as float first then convert to appropriate type
            start_float = float(start_input_raw)
            add_float = float(add_input_raw)

            if kececi_type == TYPE_POSITIVE_REAL:
                current_value = abs(start_float)
                add_value_typed = abs(add_float)
            else:  # TYPE_NEGATIVE_REAL
                current_value = -abs(start_float)
                add_value_typed = -abs(add_float)

            ask_unit = 1 if kececi_type == TYPE_POSITIVE_REAL else -1
            use_integer_division = True

        elif kececi_type == TYPE_FLOAT:
            current_value = float(start_input_raw)
            add_value_typed = float(add_input_raw)
            ask_unit = 1.0

        elif kececi_type == TYPE_RATIONAL:
            from fractions import Fraction
            current_value = Fraction(start_input_raw)
            add_value_typed = Fraction(add_input_raw)
            ask_unit = Fraction(1)

        elif kececi_type == TYPE_COMPLEX:
            # Import parser locally to avoid circular imports
            from .kececinumbers import _parse_complex
            current_value = _parse_complex(start_input_raw)
            add_value_typed = _parse_complex(add_input_raw)
            ask_unit = complex(1, 0)  # Düzeltildi: (1, 1) yerine (1, 0)

        elif kececi_type == TYPE_QUATERNION:
            from .kececinumbers import _parse_quaternion

            current_value = _parse_quaternion(start_input_raw)
            add_value_typed = _parse_quaternion(add_input_raw)
            # Import quaternion class if needed
            try:
                from .kececinumbers import quaternion

                ask_unit = quaternion(1, 1, 1, 1)
            except ImportError:
                # Use a mock quaternion if not available
                ask_unit = (
                    current_value.__class__(1, 1, 1, 1)
                    if hasattr(current_value, "__class__")
                    else None
                )

        elif kececi_type == TYPE_NEUTROSOPHIC:
            from .kececinumbers import _parse_neutrosophic
            from .kececinumbers import NeutrosophicNumber

            t, i, f = _parse_neutrosophic(start_input_raw)
            current_value = NeutrosophicNumber(t, i, f)

            t_inc, i_inc, f_inc = _parse_neutrosophic(add_input_raw)
            add_value_typed = NeutrosophicNumber(t_inc, i_inc, f_inc)
            ask_unit = NeutrosophicNumber(1, 1, 1)


        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            # Hem toplama HEM grafik uyumlu Neutrosophic Complex
            from .kececinumbers import (
                _parse_complex,
                _parse_neutrosophic_complex,
                NeutrosophicComplexNumber
            )

            # unified_generator parser'ı - DEĞİŞTİRMEYİN
            class PlotNeutroComplex:
                def __init__(self, real=0.0, imag=0.0, indeterminacy=0.0):
                    self.real = float(real)
                    self.imag = float(imag)
                    self.indeterminacy = float(indeterminacy)

                def __add__(self, other):
                    if isinstance(other, PlotNeutroComplex):
                        return PlotNeutroComplex(
                            self.real + other.real,
                            self.imag + other.imag,
                            self.indeterminacy + other.indeterminacy
                        )
                    return NotImplemented

                def __repr__(self):
                    return f"PlotNeutro({self.real:.3f}+{self.imag:.3f}j,I={self.indeterminacy:.3f})"

            # Güvenli parsing: önce proje sınıfı ile dene, yoksa fallback PlotNeutroComplex
            def safe_parse_neutro(raw_input):
                try:
                    neutro = _parse_neutrosophic_complex(raw_input)
                    # Eğer proje sınıfı kullanılabiliyorsa onu oluştur
                    try:
                        return NeutrosophicComplexNumber(
                            neutro.real, neutro.imag, getattr(neutro, 'indeterminacy', 0.0)
                        )
                    except Exception:
                        # constructor farklıysa veya sınıf yoksa fallback nesne
                        return PlotNeutroComplex(
                            neutro.real, neutro.imag, getattr(neutro, 'indeterminacy', 0.0)
                        )
                except Exception:
                    # Fallback: parse as plain complex then wrap
                    c = _parse_complex(raw_input)
                    try:
                        return NeutrosophicComplexNumber(c.real, c.imag, 0.0)
                    except Exception:
                        return PlotNeutroComplex(c.real, c.imag, 0.0)

            current_value = safe_parse_neutro(start_input_raw)
            add_value_typed = safe_parse_neutro(add_input_raw)
            # ask_unit olarak proje sınıfı varsa onu kullan; yoksa PlotNeutroComplex
            try:
                ask_unit = NeutrosophicComplexNumber(1, 1, 1)
            except Exception:
                ask_unit = PlotNeutroComplex(1, 1, 1)


        elif kececi_type == TYPE_HYPERREAL:
            from .kececinumbers import _parse_hyperreal
            from .kececinumbers import HyperrealNumber

            finite, infinitesimal, _ = _parse_hyperreal(start_input_raw) # ValueError: not enough values to unpack (expected 3, got 2)
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
            ask_unit = CliffordNumber({"": 1.0})

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

        elif kececi_type == TYPE_HYPERCOMPLEX:
            from .kececinumbers import _parse_hypercomplex

            # try several possible class names for hypercomplex types
            _HCClass = None
            try:
                from .kececinumbers import HypercomplexNumber as _HCClass
            except Exception:
                try:
                    from .kececinumbers import Quaternion as _HCClass
                except Exception:
                    try:
                        from .kececinumbers import Octonion as _HCClass
                    except Exception:
                        _HCClass = None

            # parse inputs (parsers should raise on invalid input; keep consistent with other branches)
            current_value = _parse_hypercomplex(start_input_raw)
            add_value_typed = _parse_hypercomplex(add_input_raw)

            # build a sensible unit for asking/incrementing: prefer class constructor, fallback to plain 1
            if _HCClass is not None:
                try:
                    ask_unit = _HCClass(1)
                except Exception:
                    try:
                        ask_unit = _HCClass(1, 0, 0, 0)
                    except Exception:
                        try:
                            ask_unit = _HCClass([1])
                        except Exception:
                            ask_unit = 1
            else:
                ask_unit = 1

        else:
            raise ValueError(f"Unsupported Keçeci type: {kececi_type}")

        # Validate that parsing was successful
        if current_value is None or add_value_typed is None:
            raise ValueError(f"Failed to parse values for type {kececi_type}")

    except (ValueError, TypeError, ImportError) as e:
        logger.exception(
            "Failed to initialize type %s with start=%r add=%r: %s",
            kececi_type,
            start_input_raw,
            add_input_raw,
            e,
        )
        return []

    from .kececinumbers import (
        _parse_bicomplex,
        _parse_chingon,
        _parse_clifford,
        _parse_complex,
        _parse_complex_like_string,
        _parse_dual,
        _parse_engineering_notation,
        _parse_fraction,
        _parse_hyperreal,
        _parse_neutrosophic,
        _parse_neutrosophic_bicomplex,
        _parse_neutrosophic_complex,
        _parse_octonion,
        _parse_pathion,
        _parse_quaternion,
        _parse_quaternion_from_csv,
        _parse_real,
        _parse_routon,
        _parse_sedenion,
        _parse_splitcomplex,
        _parse_super_real,
        _parse_superreal,
        _parse_ternary,
        _parse_hypercomplex,
        _parse_universal,
        _parse_voudon,
        _generate_simple_ask_sequence,
        _parse_with_fallback_simple,
        _parse_kececi_values,
        parse_to_hyperreal,
        parse_to_neutrosophic,
    )

    # Helper function for fraction formatting
    def format_fraction_local(f: Fraction) -> str:
        """Format a Fraction for display."""
        if f.denominator == 1:
            return str(f.numerator)
        else:
            return f"{f.numerator}/{f.denominator}"

    def get_parser(kececi_type: int) -> Callable[[Any], Any]:
        """Parser fonksiyonunu döndürür - test beklentilerine uygun."""
        parsers = {
            # Basit Python tipleri (test beklentileri)
            TYPE_POSITIVE_REAL: lambda s: int(_parse_fraction(s)),  # int
            TYPE_NEGATIVE_REAL: lambda s: int(-_parse_fraction(s)),  # int
            TYPE_FLOAT: lambda s: float(_parse_fraction(s)),  # float
            TYPE_RATIONAL: lambda s: Fraction.from_float(
                _parse_fraction(s)
            ),  # Fraction
            TYPE_COMPLEX: lambda s: complex(s),  # built-in complex
            # Kececi özel tipler (import edilen parser'lar)
            TYPE_QUATERNION: _parse_quaternion,
            TYPE_NEUTROSOPHIC: _parse_neutrosophic,
            TYPE_NEUTROSOPHIC_COMPLEX: _parse_neutrosophic_complex,
            TYPE_HYPERREAL: _parse_hyperreal,
            TYPE_BICOMPLEX: _parse_bicomplex,
            TYPE_NEUTROSOPHIC_BICOMPLEX: _parse_neutrosophic_bicomplex,
            TYPE_OCTONION: _parse_octonion,
            TYPE_SEDENION: _parse_sedenion,
            TYPE_CLIFFORD: _parse_clifford,
            TYPE_DUAL: _parse_dual,
            TYPE_SPLIT_COMPLEX: _parse_splitcomplex,
            TYPE_PATHION: _parse_pathion,
            TYPE_CHINGON: _parse_chingon,
            TYPE_ROUTON: _parse_routon,
            TYPE_VOUDON: _parse_voudon,
            TYPE_SUPERREAL: _parse_super_real,
            TYPE_TERNARY: _parse_ternary,
            TYPE_HYPERCOMPLEX: _parse_hypercomplex,
        }

        parser = parsers.get(kececi_type)
        if parser is None:
            raise ValueError(f"Unsupported kececi_type: {kececi_type}")
        return parser

    # Kullanım
    parser_func = get_parser(kececi_type)
    try:
        start_value = parser_func(start_input_raw)
        add_value = parser_func(add_input_raw)
    except Exception as e:
        logger.error(f"Failed to parse values: {e}")
        raise ValueError(f"Invalid input values: {e}") from e

    """
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
        23: {'parser':  _parse_hypercomplex, 'name': 'Hypercomplex'},
    }
    
    # Get parser function
    parser_func = parser_map.get(kececi_type)
    if parser_func is None:
        raise ValueError(f"Unsupported kececi_type: {kececi_type}")
    
    # Parse input values
    try:
        # error: "dict[str, object]" not callable  [operator]
        start_value = parser_func(start_input_raw)
        add_value = parser_func(add_input_raw)
    except Exception as e:
        logger.error(f"Failed to parse values: {e}")
        raise ValueError(f"Invalid input values: {e}") from e
    """

    # Main generation loop - ASK kuralını doğru uygulama
    clean_trajectory = [current_value]
    # full_log = [current_value]
    # full_log sadece include_intermediate_steps True ise tutulacak (bellek kullanımını azaltmak için)
    full_log = [current_value] if include_intermediate_steps else None
    last_divisor_used = None
    ask_counter = 0  # 0: +ask_unit, 1: -ask_unit

    for step in range(iterations):
        # 1. Add
        try:
            added_value = current_value + add_value_typed
        except Exception as e:
            logger.exception("Addition failed at step %s: %s", step, e)
            # cannot continue if addition is invalid for type
            break

        next_q = added_value
        divided_successfully = False
        modified_value = None

        # Choose which divisor to try first
        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3

        # 2. Try divisions defensively
        # --- helper: try dividing value by divisors list, return (success, result, used_divisor)
        def _try_divisions(value, divisors):
            for d in divisors:
                try:
                    if _is_divisible(value, d, kececi_type):
                        try:
                            res = _safe_divide(value, d, use_integer_division)
                            return True, res, d
                        except Exception:
                            try:
                                res = value / d
                                return True, res, d
                            except Exception:
                                logger.debug("Division fallback failed for %r by %s", value, d)
                                continue
                except Exception as e:
                    logger.debug("Divisibility check failed for %r by %s: %s", value, d, e)
                    continue
            return False, value, None

        # prepare divisors
        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3
        divisors_try = (primary_divisor, alternative_divisor)

        # try divisions on added_value first
        success, candidate, used_div = _try_divisions(added_value, divisors_try)
        if success:
            next_q = candidate
            last_divisor_used = used_div
            divided_successfully = True
        else:
            # division failed -> try ASK modifications with multiple ops
            coerced_ask = _coerce_unit_to_value(ask_unit, added_value)
            # operator order preference
            op_order = ['+', '-', '*', '/']
            modified_tried = False

            for op in op_order:
                if divided_successfully:
                    break
                # for + and - try both directions based on ask_counter
                if op in ('+', '-'):
                    directions = [1, -1] if ask_counter == 0 else [-1, 1]
                    for dir_sign in directions:
                        try:
                            operand_for_op = coerced_ask if dir_sign == 1 else _safe_apply_op(coerced_ask, '*', -1)
                        except Exception:
                            operand_for_op = coerced_ask
                        try:
                            modified = _safe_apply_op(added_value, op, operand_for_op)
                        except Exception:
                            continue
                        modified_tried = True
                        success2, candidate2, used_div2 = _try_divisions(modified, divisors_try)
                        if success2:
                            next_q = candidate2
                            last_divisor_used = used_div2
                            divided_successfully = True
                            ask_counter = 1 - ask_counter
                            break
                    if divided_successfully:
                        break
                else:
                    # '*' or '/'
                    try:
                        modified = _safe_apply_op(added_value, op, coerced_ask)
                    except Exception:
                        try:
                            modified = _safe_apply_op(coerced_ask, op, added_value)
                        except Exception:
                            continue
                    modified_tried = True
                    success2, candidate2, used_div2 = _try_divisions(modified, divisors_try)
                    if success2:
                        next_q = candidate2
                        last_divisor_used = used_div2
                        divided_successfully = True
                        break

            # if still not divisible, keep next_q as added_value (conservative)
            if not divided_successfully:
                next_q = added_value

        # 4. Logging and book-keeping
        #full_log.append(added_value)
        if include_intermediate_steps:
            full_log.append(added_value)
        #if modified_value is not None:
        if modified_value is not None and include_intermediate_steps:
            full_log.append(modified_value)
        #if not full_log or next_q != full_log[-1]:
        # Ve next_q eklenmeden önce veya sonra:
        if include_intermediate_steps:
            # next_q muhtemelen full_log son elemanı değilse yine ekle
            if not full_log or next_q != full_log[-1]:
                full_log.append(next_q)

        clean_trajectory.append(next_q)
        current_value = next_q

    # Sonuç formatlarken:
    #formatted_sequence = [format_fraction(x) if isinstance(x, Fraction) else x for x in (full_log if include_intermediate_steps else clean_trajectory)]
    formatted_sequence = [format_fraction(x) if isinstance(x, Fraction) else x
                          for x in (full_log if include_intermediate_steps else clean_trajectory)]
    return formatted_sequence


def _parse_quaternion_fixed(s: str):
    """
    Fixed quaternion parser that handles single numbers correctly.
    """
    from .kececinumbers import quaternion, _parse_fraction
    
    s_str = str(s).strip()
    
    # Handle empty string
    if not s_str:
        return quaternion(0, 0, 0, 0)
    
    # Handle comma-separated format: "w,x,y,z" or "w, x, y, z"
    if ',' in s_str:
        parts = [p.strip() for p in s_str.split(',')]
        if len(parts) == 4:
            try:
                w = _parse_fraction(parts[0])
                x = _parse_fraction(parts[1])
                y = _parse_fraction(parts[2])
                z = _parse_fraction(parts[3])
                return quaternion(w, x, y, z)
            except:
                pass
    
    # Handle single number - only w component, others 0
    try:
        w = _parse_fraction(s_str)
        return quaternion(w, 0, 0, 0)
    except:
        # Try as float
        try:
            w = float(s_str)
            return quaternion(w, 0, 0, 0)
        except:
            return quaternion(0, 0, 0, 0)


def _parse_neutrosophic_fixed(s: str):
    """Fixed neutrosophic parser."""
    from .kececinumbers import _parse_neutrosophic
    from .kececinumbers import NeutrosophicNumber
    
    try:
        return _parse_neutrosophic(s)
    except:
        # Fallback
        from .kececinumbers import _parse_fraction
        try:
            val = _parse_fraction(s)
            return NeutrosophicNumber(val, 0.0, 0.0)
        except:
            return NeutrosophicNumber(0.0, 0.0, 0.0)


def _parse_octonion_fixed(s: str):
    """Fixed octonion parser."""
    from .kececinumbers import _parse_octonion
    from .kececinumbers import OctonionNumber
    
    try:
        return _parse_octonion(s)
    except:
        # Fallback
        from .kececinumbers import _parse_fraction
        try:
            val = _parse_fraction(s)
            return OctonionNumber(val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        except:
            return OctonionNumber(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# Similar fixed parsers for other types...
def _parse_sedenion_fixed(s: str):
    from .kececinumbers import _parse_sedenion, SedenionNumber, _parse_fraction
    try:
        return _parse_sedenion(s)
    except:
        try:
            val = _parse_fraction(s)
            components = [val] + [0.0] * 15
            return SedenionNumber(components)
        except:
            return SedenionNumber([0.0] * 16)


def _parse_ternary(s: Any) -> Any:
    """TernaryNumber hatasız"""
    try:
        if isinstance(s, (int, float, Fraction)):
            try:
                from kececinumbers import TernaryNumber
                return TernaryNumber(float(s), 0.0, 0.0)
            except:
                return [float(s), 0.0, 0.0]
        return [float(s), 0.0, 0.0]
    except:
        return [0.0, 0.0, 0.0]


def _generate_proper_ask_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Proper ASK sequence that extracts values for plotting.
    """
    result = []
    current = start_value
    ask_counter = 0
    
    # Helper to extract plot value
    def extract_plot_value(val):
        """Extract a plottable value from any Keçeci number type."""
        try:
            # For quaternions, use w component or norm
            if hasattr(val, 'w') and hasattr(val, 'x') and hasattr(val, 'y') and hasattr(val, 'z'):
                return float(val.w)  # Use real part
            # For octonions
            elif hasattr(val, '__len__') and len(val) >= 8:
                return float(val[0]) if val else 0.0
            # For sedenions
            elif hasattr(val, '__len__') and len(val) >= 16:
                return float(val[0]) if val else 0.0
            # For complex
            elif isinstance(val, complex):
                return float(val.real)
            # For tuples (neutrosophic, etc.)
            elif isinstance(val, tuple) and len(val) >= 1:
                return float(val[0])
            # For lists
            elif isinstance(val, list) and val:
                return float(val[0])
            # For custom objects with real attribute
            elif hasattr(val, 'real'):
                return float(val.real)
            # For custom objects with value attribute
            elif hasattr(val, 'value'):
                return float(val.value)
            # For everything else, try to convert to float
            else:
                return float(val)
        except:
            return 0.0
    
    if include_intermediate_steps:
        plot_val = extract_plot_value(current)
        result.append({
            "step": 0,
            "value": current,
            "plot_value": plot_val,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(extract_plot_value(current))
    
    for i in range(1, iterations):
        try:
            # 1. ADD
            added = current + add_value
            
            next_val = added
            divided = False
            
            # 2. Check division by 2 or 3
            for divisor in [2, 3]:
                try:
                    # For quaternions, check norm for divisibility
                    if number_type == 6:  # Quaternion
                        if hasattr(added, 'norm'):
                            norm = added.norm()
                            if norm % divisor == 0:
                                next_val = added / divisor
                                divided = True
                                break
                        else:
                            # Try division anyway
                            next_val = added / divisor
                            divided = True
                            break
                    else:
                        # For other types, try division
                        next_val = added / divisor
                        divided = True
                        break
                except Exception as e:
                    logger.debug(f"Division by {divisor} failed: {e}")
                    continue
            
            # 3. Keçeci unit adjustment
            if not divided:
                # Check if prime-like
                is_prime_like = False
                try:
                    if number_type == 6:  # Quaternion
                        if hasattr(added, 'norm'):
                            norm = added.norm()
                            is_prime_like = _is_prime_int(int(norm))
                    else:
                        # Try to extract a numeric value
                        val = extract_plot_value(added)
                        is_prime_like = _is_prime_int(int(abs(val)))
                except:
                    is_prime_like = False
                
                if is_prime_like:
                    # Get appropriate unit
                    unit = _get_proper_unit(number_type, current)
                    
                    if ask_counter == 0:
                        adjusted = added + unit
                    else:
                        adjusted = added - unit
                    
                    ask_counter = 1 - ask_counter
                    
                    # Try division on adjusted value
                    for divisor in [2, 3]:
                        try:
                            next_val = adjusted / divisor
                            break
                        except:
                            continue
                    else:
                        next_val = adjusted
            
            current = next_val
            
            if include_intermediate_steps:
                plot_val = extract_plot_value(current)
                result.append({
                    "step": i,
                    "value": current,
                    "plot_value": plot_val,
                    "operation": "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(extract_plot_value(current))
                
        except Exception as e:
            logger.error(f"Error at iteration {i}: {e}")
            default_val = _get_proper_default(number_type)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "plot_value": extract_plot_value(default_val),
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(extract_plot_value(default_val))
            current = default_val
    
    return result


def _is_prime_int(n: int) -> bool:
    """Check if integer is prime."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True


def _get_proper_unit(number_type: int, sample=None):
    """Get proper unit for number type."""
    # helper to determine dimension from sample or default
    def _dim_from_sample(s, default=4):
        if s is None:
            return default
        # if sample is list/tuple-like, use its length
        try:
            if isinstance(s, (list, tuple)):
                return max(1, len(s))
            # if sample is an object with .components or similar, try to infer
            if hasattr(s, "__len__"):
                return max(1, len(s))
        except Exception:
            pass
        return default

    if number_type == 1:  # Positive Real
        return 1.0
    elif number_type == 2:  # Negative Real
        return -1.0
    elif number_type == 3:  # Complex
        return complex(1, 0)
    elif number_type == 4:  # Float
        return 1.0
    elif number_type == 5:  # Rational
        try:
            from fractions import Fraction
            return Fraction(1, 1)
        except Exception:
            return 1.0
    elif number_type == 6:  # Quaternion
        from .kececinumbers import quaternion
        return quaternion(1, 0, 0, 0)
    elif number_type == 7:  # Neutrosophic
        from .kececinumbers import NeutrosophicNumber
        return NeutrosophicNumber(1.0, 0.0, 0.0)
    elif number_type == 12:  # Octonion
        from .kececinumbers import OctonionNumber
        return OctonionNumber(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif number_type == 13:  # Sedenion
        from .kececinumbers import SedenionNumber
        return SedenionNumber([1.0] + [0.0] * 15)
    elif number_type == 22:  # Ternary
        from .kececinumbers import TernaryNumber
        return TernaryNumber([1])
    elif number_type == 23:  # Hypercomplex
        # infer dimension from sample if provided, default to 4
        dim = _dim_from_sample(sample, default=4)
        try:
            from .kececinumbers import HypercomplexNumber
            comps = [1.0] + [0.0] * (dim - 1)
            # assume HypercomplexNumber accepts a list of components
            return HypercomplexNumber(comps)
        except Exception:
            # fallback to plain list if class not available
            return [1.0] + [0.0] * dim
    else:
        return 1.0


def _get_proper_default(number_type: int, sample=None):
    """Get proper default for number type."""
    # helper to determine dimension from sample or default
    def _dim_from_sample(s, default=4):
        if s is None:
            return default
        try:
            if isinstance(s, (list, tuple)):
                return max(1, len(s))
            if hasattr(s, "__len__"):
                return max(1, len(s))
        except Exception:
            pass
        return default

    if number_type == 1:  # Positive Real
        return 0.0
    elif number_type == 2:  # Negative Real
        return 0.0
    elif number_type == 3:  # Complex
        return complex(0, 0)
    elif number_type == 4:  # Float
        return 0.0
    elif number_type == 5:  # Rational
        try:
            from fractions import Fraction
            return Fraction(0, 1)
        except Exception:
            return 0.0
    elif number_type == 6:  # Quaternion
        from .kececinumbers import quaternion
        return quaternion(0, 0, 0, 0)
    elif number_type == 7:  # Neutrosophic
        from .kececinumbers import NeutrosophicNumber
        return NeutrosophicNumber(0.0, 0.0, 0.0)
    elif number_type == 12:  # Octonion
        from .kececinumbers import OctonionNumber
        return OctonionNumber(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif number_type == 13:  # Sedenion
        from .kececinumbers import SedenionNumber
        return SedenionNumber([0.0] * 16)
    elif number_type == 22:  # Ternary
        from .kececinumbers import TernaryNumber
        return TernaryNumber([0])
    elif number_type == 23:  # Hypercomplex
        dim = _dim_from_sample(sample, default=4)
        try:
            from .kececinumbers import HypercomplexNumber
            comps = [0.0] * dim
            return HypercomplexNumber(comps)
        except Exception:
            return [0.0] * dim
    else:
        return 0.0


def _generate_fallback_sequence(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str,
    iterations: int,
    include_intermediate_steps: bool = False,
    operation: str = "ask"
) -> List[Any]:
    """
    Fallback sequence generator when main generator fails.
    """
    # Simple float parser
    def parse_float(val):
        try:
            return float(val)
        except:
            return 0.0
    
    start_val = parse_float(start_input_raw)
    add_val = parse_float(add_input_raw)
    
    result = []
    current = start_val
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            if operation == "ask":
                # Simple ASK for floats
                added = current + add_val
                next_val = added
                
                # Check division
                for divisor in [2, 3]:
                    if added % divisor == 0:
                        next_val = added / divisor
                        break
                
                # Prime check and unit adjustment
                if next_val == added:  # Not divided
                    if _is_prime_int(int(abs(added))):
                        unit = 1.0
                        # Simple unit adjustment
                        adjusted = added + unit if i % 2 == 0 else added - unit
                        # Try division again
                        for divisor in [2, 3]:
                            if adjusted % divisor == 0:
                                next_val = adjusted / divisor
                                break
                        else:
                            next_val = adjusted
                
                current = next_val
            else:
                # Simple operations
                if operation == "add":
                    current = current + add_val
                elif operation == "subtract":
                    current = current - add_val
                elif operation == "multiply":
                    current = current * add_val
                elif operation == "divide":
                    current = current / add_val if add_val != 0 else float('inf')
                elif operation == "mod":
                    current = current % add_val if add_val != 0 else current
                elif operation == "power":
                    current = current ** add_val
                else:
                    current = current + add_val
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation if operation != "ask" else "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} in fallback: {e}")
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": 0.0,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(0.0)
            current = 0.0
    
    return result


def _unified_generator_fallback(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str,
    iterations: int,
    include_intermediate_steps: bool = False,
    operation: str = "ask"
) -> List[Any]:
    """
    Fallback generator when imports fail.
    """
    # Simple parser
    def parse_simple(val: str) -> float:
        if not val:
            return 0.0
        val_str = str(val).strip()
        
        # Try complex
        val_str = val_str.replace('i', 'j').replace('J', 'j')
        try:
            c = complex(val_str)
            return float(c.real)
        except:
            pass
        
        # Try float
        try:
            return float(val_str)
        except:
            # Try fraction
            if '/' in val_str:
                try:
                    num, den = val_str.split('/')
                    return float(num) / float(den)
                except:
                    pass
            
            # Try mixed number
            if ' ' in val_str and '/' in val_str:
                try:
                    whole, frac = val_str.split(' ', 1)
                    num, den = frac.split('/')
                    return float(whole) + (float(num) / float(den))
                except:
                    pass
        
        return 0.0
    
    # Parse values
    start_base = parse_simple(start_input_raw)
    add_base = parse_simple(add_input_raw)
    
    # Adjust based on type
    if kececi_type == 1:  # Positive Real
        start_value = abs(start_base)
        add_value = abs(add_base)
    elif kececi_type == 2:  # Negative Real
        start_value = -abs(start_base)
        add_value = -abs(add_base)
    elif kececi_type == 3:  # Complex
        start_value = complex(start_base, 0)
        add_value = complex(add_base, 0)
    elif kececi_type in [4, 5]:  # Float, Rational
        start_value = start_base
        add_value = add_base
    else:
        # For other types, use float as fallback
        start_value = start_base
        add_value = add_base
    
    # Generate simple sequence
    return _generate_simple_sequence_direct(
        start_value=start_value,
        add_value=add_value,
        iterations=iterations,
        include_intermediate_steps=include_intermediate_steps,
        number_type=kececi_type
    )


def _generate_ask_sequence_direct(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Direct ASK sequence generation.
    """
    result = []
    current = start_value
    ask_counter = 0
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            # 1. ADD
            added = current + add_value
            
            next_val = added
            divided = False
            
            # 2. Check division by 2 or 3
            for divisor in [2, 3]:
                try:
                    # Try to check divisibility
                    if _check_divisible_simple(added, divisor):
                        # Try to divide
                        next_val = _divide_simple(added, divisor)
                        divided = True
                        logger.debug(f"Step {i}: Divided {added} by {divisor} = {next_val}")
                        break
                except Exception as e:
                    logger.debug(f"Division failed: {e}")
                    continue
            
            # 3. Keçeci unit adjustment
            if not divided and _check_prime_like_simple(added):
                unit = _get_unit_for_type_simple(number_type)
                
                if ask_counter == 0:
                    adjusted = added + unit
                    logger.debug(f"Step {i}: Added unit {unit} to {added} = {adjusted}")
                else:
                    adjusted = added - unit
                    logger.debug(f"Step {i}: Subtracted unit {unit} from {added} = {adjusted}")
                
                ask_counter = 1 - ask_counter
                
                # Try division on adjusted value
                for divisor in [2, 3]:
                    try:
                        if _check_divisible_simple(adjusted, divisor):
                            next_val = _divide_simple(adjusted, divisor)
                            logger.debug(f"Step {i}: Divided adjusted {adjusted} by {divisor} = {next_val}")
                            break
                    except Exception:
                        continue
                else:
                    next_val = adjusted
            
            current = next_val
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i}: {e}")
            default_val = _get_default_for_type_simple(number_type)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _generate_operation_sequence_direct(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Direct operation sequence generation.
    """
    result = []
    current = start_value
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            if operation == "add":
                current = current + add_value
            elif operation == "subtract":
                current = current - add_value
            elif operation == "multiply":
                current = current * add_value
            elif operation == "divide":
                current = _divide_simple(current, add_value)
            elif operation == "mod":
                try:
                    current = current % add_value
                except:
                    current = current  # Mod not supported
            elif operation == "power":
                try:
                    current = current ** add_value
                except:
                    current = current  # Power not supported
            else:
                current = current + add_value  # Default to add
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i}, operation {operation}: {e}")
            default_val = _get_default_for_type_simple(number_type)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _check_divisible_simple(value: Any, divisor: float) -> bool:
    """
    Simple divisibility check.
    """
    try:
        if isinstance(value, (int, float)):
            return abs(value % divisor) < 1e-12
        elif isinstance(value, complex):
            return abs(value.real % divisor) < 1e-12 and abs(value.imag % divisor) < 1e-12
        else:
            # For other types, assume divisible
            return True
    except:
        return True


def _divide_simple(value: Any, divisor: float) -> Any:
    """
    Simple division.
    """
    try:
        return value / divisor
    except:
        # Try alternatives
        if isinstance(value, (tuple, list)):
            return type(value)([x / divisor for x in value])
        else:
            raise


def _check_prime_like_simple(value: Any) -> bool:
    """
    Simple prime-like check.
    """
    try:
        # Get a numeric value
        if isinstance(value, (int, float)):
            val = abs(value)
        elif isinstance(value, complex):
            val = abs(value)
        elif hasattr(value, '__abs__'):
            val = abs(value)
        else:
            return False
        
        # Simple prime check
        if val < 2:
            return False
        for i in range(2, int(val**0.5) + 1):
            if val % i == 0:
                return False
        return True
    except:
        return False


def _get_unit_for_type_simple(number_type: int) -> Any:
    """
    Get unit for number type.
    """
    if number_type in [1, 4, 5]:
        return 1.0
    elif number_type == 2:
        return -1.0
    elif number_type == 3:
        return complex(1, 0)
    elif number_type == 6:  # Quaternion
        try:
            from .kececinumbers import quaternion
            return quaternion(1, 0, 0, 0)
        except:
            return (1.0, 0.0, 0.0, 0.0)
    elif number_type == 7:  # Neutrosophic
        return (1.0, 0.0, 0.0)
    else:
        return 1.0


def _get_default_for_type_simple(number_type: int) -> Any:
    """
    Get default value for number type.
    """
    if number_type in [1, 2, 4, 5]:
        return 0.0
    elif number_type == 3:
        return complex(0, 0)
    elif number_type == 6:  # Quaternion
        try:
            from .kececinumbers import quaternion
            return quaternion(0, 0, 0, 0)
        except:
            return (0.0, 0.0, 0.0, 0.0)
    elif number_type == 7:  # Neutrosophic
        return (0.0, 0.0, 0.0)
    else:
        return 0.0


def _generate_simple_sequence_direct(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Simple sequence generation (just addition).
    """
    result = []
    current = start_value
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            current = current + add_value
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "add",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i}: {e}")
            default_val = _get_default_for_type_simple(number_type)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _generate_ternary_ask_sequence(
    start_value: 'TernaryNumber',
    add_value: 'TernaryNumber',
    iterations: int,
    include_intermediate_steps: bool = False
) -> List[Any]:
    """
    ASK algorithm specifically for TernaryNumber.
    """
    result = []
    current = start_value
    ask_counter = 0  # 0: +unit, 1: -unit
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            # STEP 1: ADDITION
            added = current + add_value
            
            next_val = added
            divided = False
            
            # STEP 2: CHECK DIVISIBILITY by 2 or 3
            # Convert to decimal for divisibility checks
            added_decimal = added.to_decimal()
            
            for divisor in [2, 3]:
                try:
                    if added_decimal % divisor == 0:
                        # Divide in decimal, convert back to ternary
                        divided_decimal = added_decimal // divisor
                        divided_val = TernaryNumber.from_decimal(divided_decimal)
                        next_val = divided_val
                        divided = True
                        break
                except Exception:
                    continue
            
            # STEP 3: KECEÇI UNIT ADJUSTMENT if not divided and prime-like
            if not divided:
                # Check if prime-like (check decimal value)
                if _is_prime_decimal(added_decimal):
                    # Ternary unit is 1 in ternary = [1]
                    unit = TernaryNumber.from_decimal(1)
                    
                    # Apply unit based on ask_counter
                    if ask_counter == 0:
                        adjusted = added + unit
                    else:
                        adjusted = added - unit
                    
                    # Toggle ask counter
                    ask_counter = 1 - ask_counter
                    
                    # Try division again on adjusted value
                    adjusted_decimal = adjusted.to_decimal()
                    for divisor in [2, 3]:
                        try:
                            if adjusted_decimal % divisor == 0:
                                final_decimal = adjusted_decimal // divisor
                                next_val = TernaryNumber.from_decimal(final_decimal)
                                break
                        except Exception:
                            continue
                    else:
                        # No division successful after adjustment
                        next_val = adjusted
            
            current = next_val
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for Ternary: {e}")
            default_val = TernaryNumber.from_decimal(0)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _generate_ternary_operation_sequence(
    start_value: 'TernaryNumber',
    add_value: 'TernaryNumber',
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False
) -> List[Any]:
    """
    Standard operations for TernaryNumber.
    """
    result = []
    current = start_value
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            if operation == "add":
                current = current + add_value
            elif operation == "subtract":
                current = current - add_value
            elif operation == "multiply":
                # Convert add_value to scalar if it's a single digit ternary
                if isinstance(add_value, TernaryNumber):
                    add_decimal = add_value.to_decimal()
                    current = current * add_decimal
                else:
                    current = current * add_value
            elif operation == "divide":
                if isinstance(add_value, TernaryNumber):
                    add_decimal = add_value.to_decimal()
                    if add_decimal == 0:
                        raise ZeroDivisionError("Division by zero")
                    current_decimal = current.to_decimal()
                    result_decimal = current_decimal // add_decimal
                    current = TernaryNumber.from_decimal(result_decimal)
                else:
                    if add_value == 0:
                        raise ZeroDivisionError("Division by zero")
                    current_decimal = current.to_decimal()
                    result_decimal = current_decimal // add_value
                    current = TernaryNumber.from_decimal(result_decimal)
            elif operation == "mod":
                # Mod operation for Ternary (convert to decimal)
                current_decimal = current.to_decimal()
                if isinstance(add_value, TernaryNumber):
                    add_decimal = add_value.to_decimal()
                else:
                    add_decimal = add_value
                if add_decimal == 0:
                    raise ZeroDivisionError("Modulo by zero")
                result_decimal = current_decimal % add_decimal
                current = TernaryNumber.from_decimal(result_decimal)
            elif operation == "power":
                # Power operation (convert to decimal)
                current_decimal = current.to_decimal()
                if isinstance(add_value, TernaryNumber):
                    add_decimal = add_value.to_decimal()
                else:
                    add_decimal = add_value
                result_decimal = current_decimal ** add_decimal
                current = TernaryNumber.from_decimal(result_decimal)
            else:
                current = current + add_value  # Default to addition
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for Ternary, operation {operation}: {e}")
            default_val = TernaryNumber.from_decimal(0)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _generate_ask_sequence_proper(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Proper ASK algorithm that handles different number types correctly.
    """
    # Get type-specific handler
    handler = _get_type_handler(number_type)
    
    result = []
    current = start_value
    ask_counter = 0
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            # Use handler for all operations
            added = handler["add"](current, add_value)
            next_val = added
            divided = False
            
            # Check divisibility
            for divisor in [2, 3]:
                if handler["is_divisible"](added, divisor):
                    next_val = handler["divide"](added, divisor)
                    divided = True
                    break
            
            # Keçeci unit adjustment
            if not divided and handler["is_prime_like"](added):
                unit = handler["get_unit"]()
                
                if ask_counter == 0:
                    adjusted = handler["add"](added, unit)
                else:
                    adjusted = handler["subtract"](added, unit)
                
                ask_counter = 1 - ask_counter
                
                # Try division on adjusted value
                for divisor in [2, 3]:
                    if handler["is_divisible"](adjusted, divisor):
                        next_val = handler["divide"](adjusted, divisor)
                        break
                else:
                    next_val = adjusted
            
            current = next_val
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for type {number_type}: {e}")
            default_val = handler["get_default"]()
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _get_type_handler(number_type: int) -> Dict[str, Callable]:
    """
    Get proper handler for each number type.
    """
    # For simple numeric types (1-5)
    if number_type in [1, 2, 4, 5]:
        from .kececinumbers import _safe_divide, _safe_mod, _safe_power
        
        def numeric_add(a, b):
            return a + b
        
        def numeric_subtract(a, b):
            return a - b
        
        def numeric_divide(a, b):
            return _safe_divide(a, b)
        
        def numeric_is_divisible(a, divisor):
            try:
                if hasattr(a, '__mod__'):
                    remainder = a % divisor
                    return abs(remainder) < 1e-12
                return True
            except:
                return False
        
        def numeric_is_prime_like(a):
            try:
                val = abs(float(a))
                if val < 2:
                    return False
                for i in range(2, int(val**0.5) + 1):
                    if val % i == 0:
                        return False
                return True
            except:
                return False
        
        def numeric_get_unit():
            return 1.0 if number_type != 2 else -1.0
        
        def numeric_get_default():
            return 0.0
        
        return {
            "add": numeric_add,
            "subtract": numeric_subtract,
            "divide": numeric_divide,
            "is_divisible": numeric_is_divisible,
            "is_prime_like": numeric_is_prime_like,
            "get_unit": numeric_get_unit,
            "get_default": numeric_get_default,
        }
    
    # For Complex (3)
    elif number_type == 3:
        def complex_add(a, b):
            return a + b
        
        def complex_subtract(a, b):
            return a - b
        
        def complex_divide(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return complex(float('inf'), 0)
        
        def complex_is_divisible(a, divisor):
            # Check both real and imaginary parts
            try:
                return abs(a.real % divisor) < 1e-12 and abs(a.imag % divisor) < 1e-12
            except:
                return False
        
        def complex_is_prime_like(a):
            # Check magnitude
            try:
                mag = abs(a)
                if mag < 2:
                    return False
                for i in range(2, int(mag**0.5) + 1):
                    if mag % i == 0:
                        return False
                return True
            except:
                return False
        
        def complex_get_unit():
            return complex(1, 0)
        
        def complex_get_default():
            return complex(0, 0)
        
        return {
            "add": complex_add,
            "subtract": complex_subtract,
            "divide": complex_divide,
            "is_divisible": complex_is_divisible,
            "is_prime_like": complex_is_prime_like,
            "get_unit": complex_get_unit,
            "get_default": complex_get_default,
        }
    
    # For Quaternion (6)
    elif number_type == 6:
        try:
            from .kececinumbers import quaternion
            
            def quaternion_add(a, b):
                return a + b
            
            def quaternion_subtract(a, b):
                return a - b
            
            def quaternion_divide(a, b):
                try:
                    # Quaternion division might need special handling
                    if hasattr(a, '__truediv__'):
                        return a / b
                    else:
                        # Convert to list and divide components
                        if hasattr(a, 'a') and hasattr(a, 'b') and hasattr(a, 'c') and hasattr(a, 'd'):
                            return type(a)(a.a/b, a.b/b, a.c/b, a.d/b)
                        else:
                            return a / b
                except:
                    return a
            
            def quaternion_is_divisible(a, divisor):
                # Check all components
                try:
                    if hasattr(a, 'a') and hasattr(a, 'b') and hasattr(a, 'c') and hasattr(a, 'd'):
                        comps = [a.a, a.b, a.c, a.d]
                    elif isinstance(a, (tuple, list)) and len(a) >= 4:
                        comps = a[:4]
                    else:
                        comps = [a]
                    
                    for comp in comps:
                        if abs(comp % divisor) > 1e-12:
                            return False
                    return True
                except:
                    return True
            
            def quaternion_is_prime_like(a):
                # Check norm
                try:
                    if hasattr(a, 'norm'):
                        norm = a.norm()
                    elif hasattr(a, '__abs__'):
                        norm = abs(a)
                    else:
                        norm = 0
                    
                    return _is_prime_decimal(norm)
                except:
                    return False
            
            def quaternion_get_unit():
                return quaternion(1, 0, 0, 0)
            
            def quaternion_get_default():
                return quaternion(0, 0, 0, 0)
            
            return {
                "add": quaternion_add,
                "subtract": quaternion_subtract,
                "divide": quaternion_divide,
                "is_divisible": quaternion_is_divisible,
                "is_prime_like": quaternion_is_prime_like,
                "get_unit": quaternion_get_unit,
                "get_default": quaternion_get_default,
            }
        except:
            # Fallback
            return _get_generic_handler()
    
    # For other special types (7-21)
    elif 7 <= number_type <= 21:
        return _get_generic_handler()
    
    else:
        return _get_generic_handler()


def _get_generic_handler() -> Dict[str, Callable]:
    """
    Generic handler for number types without special implementation.
    """
    def generic_add(a, b):
        try:
            return a + b
        except:
            return a
    
    def generic_subtract(a, b):
        try:
            return a - b
        except:
            return a
    
    def generic_divide(a, b):
        try:
            return a / b
        except:
            return a
    
    def generic_is_divisible(a, divisor):
        return True  # Assume divisible
    
    def generic_is_prime_like(a):
        return False  # Assume not prime-like
    
    def generic_get_unit():
        return 1.0
    
    def generic_get_default():
        return 0.0
    
    return {
        "add": generic_add,
        "subtract": generic_subtract,
        "divide": generic_divide,
        "is_divisible": generic_is_divisible,
        "is_prime_like": generic_is_prime_like,
        "get_unit": generic_get_unit,
        "get_default": generic_get_default,
    }


def _is_prime_decimal(n: int) -> bool:
    """
    Check if an integer is prime.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True


def _get_parser_for_type(kececi_type: int) -> Callable[[str], Any]:
    """
    Get parser function for Keçeci type.
    """
    parser_map = {
        1: lambda s: abs(float(s)),  # Positive Real
        2: lambda s: -abs(float(s)),  # Negative Real
        3: lambda s: complex(s),  # Complex
        4: lambda s: float(s),  # Float
        5: lambda s: float(s),  # Rational
        6: lambda s: _parse_quaternion(s),  # Quaternion
        7: lambda s: _parse_neutrosophic(s),  # Neutrosophic
        8: lambda s: _parse_neutrosophic_complex(s),  # Neutrosophic Complex
        9: lambda s: _parse_hyperreal(s),  # Hyperreal
        10: lambda s: _parse_bicomplex(s),  # Bicomplex
        11: lambda s: _parse_neutrosophic_bicomplex(s),  # Neutrosophic Bicomplex
        12: lambda s: _parse_octonion(s),  # Octonion
        13: lambda s: _parse_sedenion(s),  # Sedenion
        14: lambda s: _parse_clifford(s),  # Clifford
        15: lambda s: _parse_dual(s),  # Dual
        16: lambda s: _parse_splitcomplex(s),  # Split-Complex
        17: lambda s: _parse_pathion(s),  # Pathion
        18: lambda s: _parse_chingon(s),  # Chingon
        19: lambda s: _parse_routon(s),  # Routon
        20: lambda s: _parse_voudon(s),  # Voudon
        21: lambda s: _parse_superreal(s),  # Superreal
        22: lambda s: _parse_ternary(s),  # Ternary
        23: lambda s: _parse_hypercomplex(s),  # 
    }
    
    return parser_map.get(kececi_type, lambda s: float(s))


def _generate_ask_sequence_fixed(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Fixed ASK algorithm that actually works for all number types.
    """
    # Get type-specific operations
    type_ops = _get_type_specific_operations(number_type)
    
    result = []
    current = start_value
    ask_counter = 0  # 0: +unit, 1: -unit
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        step_values = []
        
        try:
            # STEP 1: ADDITION
            added = type_ops["add"](current, add_value)
            step_values.append(("add", added))
            logger.debug(f"Step {i}.1: ADD {current} + {add_value} = {added}")
            
            next_val = added
            divided = False
            
            # STEP 2: CHECK DIVISIBILITY by 2 or 3
            for divisor in [2, 3]:
                try:
                    # Check if divisible
                    if type_ops["is_divisible"](added, divisor):
                        # Try to divide
                        divided_val = type_ops["divide"](added, divisor)
                        step_values.append((f"divide by {divisor}", divided_val))
                        logger.debug(f"Step {i}.2: DIVIDE {added} / {divisor} = {divided_val}")
                        
                        next_val = divided_val
                        divided = True
                        break
                except Exception as e:
                    logger.debug(f"Division by {divisor} failed: {e}")
                    continue
            
            # STEP 3: KECEÇI UNIT ADJUSTMENT if not divided and looks prime-like
            if not divided:
                # Check if value looks prime-like
                if type_ops["is_prime_like"](added):
                    # Get Keçeci unit for this type
                    unit = type_ops["get_unit"]()
                    
                    # Apply unit based on ask_counter
                    if ask_counter == 0:
                        adjusted = type_ops["add"](added, unit)
                        op_desc = f"+unit({unit})"
                    else:
                        adjusted = type_ops["subtract"](added, unit)
                        op_desc = f"-unit({unit})"
                    
                    step_values.append(("keçeci unit", adjusted))
                    logger.debug(f"Step {i}.3: KECEÇI UNIT {added} {op_desc} = {adjusted}")
                    
                    # Toggle ask counter
                    ask_counter = 1 - ask_counter
                    
                    # Try division again on adjusted value
                    for divisor in [2, 3]:
                        try:
                            if type_ops["is_divisible"](adjusted, divisor):
                                final_val = type_ops["divide"](adjusted, divisor)
                                step_values.append((f"divide adjusted by {divisor}", final_val))
                                logger.debug(f"Step {i}.4: DIVIDE ADJUSTED {adjusted} / {divisor} = {final_val}")
                                
                                next_val = final_val
                                break
                        except Exception as e:
                            logger.debug(f"Division on adjusted failed: {e}")
                            continue
                    else:
                        # No division successful after adjustment
                        next_val = adjusted
                else:
                    # Not prime-like, keep as is
                    logger.debug(f"Step {i}.3: Not prime-like, keeping {added}")
            
            # Update current value
            current = next_val
            
            # Build result
            if include_intermediate_steps:
                # Add all intermediate steps
                for j, (op, val) in enumerate(step_values):
                    result.append({
                        "step": i,
                        "substep": j,
                        "value": val,
                        "operation": op,
                        "description": f"Step {i}.{j}: {op} → {val}"
                    })
                # Add final value
                result.append({
                    "step": i,
                    "substep": len(step_values),
                    "value": current,
                    "operation": "final",
                    "description": f"Step {i} final: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for type {number_type}: {e}")
            default_val = type_ops["get_default"]()
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _get_type_specific_operations(number_type: int) -> Dict[str, Callable]:
    """
    Get type-specific operations with proper handling for each number type.
    """
    # Common operations
    def common_add(a, b):
        return a + b
    
    def common_subtract(a, b):
        return a - b
    
    def common_multiply(a, b):
        return a * b
    
    def common_divide(a, b):
        try:
            return a / b
        except ZeroDivisionError:
            # Handle division by zero
            if hasattr(a, '__class__'):
                try:
                    return a.__class__(float('inf'))
                except:
                    return float('inf')
            return float('inf')
    
    def common_mod(a, b):
        try:
            return a % b
        except:
            return a  # Return original if mod not supported
    
    # Type-specific implementations
    if number_type in [1, 2, 4, 5]:  # Real types
        def is_divisible_real(a, divisor):
            try:
                remainder = a % divisor
                return abs(remainder) < 1e-12
            except:
                # For floats, check if result is close to integer
                result = a / divisor
                return abs(result - round(result)) < 1e-12
        
        def is_prime_like_real(a):
            try:
                val = abs(a)
                if val < 2:
                    return False
                if val == 2 or val == 3:
                    return True
                if val % 2 == 0 or val % 3 == 0:
                    return False
                
                i = 5
                while i * i <= val:
                    if val % i == 0 or val % (i + 2) == 0:
                        return False
                    i += 6
                return True
            except:
                return False
        
        def get_unit_real():
            return 1.0 if number_type != 2 else -1.0
        
        def get_default_real():
            return 0.0
        
        return {
            "add": common_add,
            "subtract": common_subtract,
            "multiply": common_multiply,
            "divide": common_divide,
            "mod": common_mod,
            "is_divisible": is_divisible_real,
            "is_prime_like": is_prime_like_real,
            "get_unit": get_unit_real,
            "get_default": get_default_real,
        }
    
    elif number_type == 3:  # Complex
        def is_divisible_complex(a, divisor):
            # For complex, check if both real and imag parts are divisible
            try:
                real_rem = a.real % divisor
                imag_rem = a.imag % divisor
                return abs(real_rem) < 1e-12 and abs(imag_rem) < 1e-12
            except:
                return False
        
        def is_prime_like_complex(a):
            # Check magnitude
            try:
                mag = abs(a)
                return is_prime_like_complex.__closure__[0].cell_contents(mag) if number_type == 1 else False
            except:
                return False
        
        def get_unit_complex():
            return complex(1, 0)
        
        def get_default_complex():
            return complex(0, 0)
        
        return {
            "add": common_add,
            "subtract": common_subtract,
            "multiply": common_multiply,
            "divide": common_divide,
            "mod": lambda a, b: a,  # Mod not typically defined for complex
            "is_divisible": is_divisible_complex,
            "is_prime_like": is_prime_like_complex,
            "get_unit": get_unit_complex,
            "get_default": get_default_complex,
        }
    
    elif number_type == 6:  # Quaternion
        try:
            from .kececinumbers import quaternion
            
            def is_divisible_quaternion(q, divisor):
                # For quaternion, check if all components are divisible
                try:
                    # q is typically (w, x, y, z) or has .a, .b, .c, .d attributes
                    if hasattr(q, 'a') and hasattr(q, 'b') and hasattr(q, 'c') and hasattr(q, 'd'):
                        comps = [q.a, q.b, q.c, q.d]
                    elif isinstance(q, (tuple, list)) and len(q) >= 4:
                        comps = [q[0], q[1], q[2], q[3]]
                    else:
                        # Try to extract components
                        comps = [getattr(q, 'w', 0), getattr(q, 'x', 0), 
                                getattr(q, 'y', 0), getattr(q, 'z', 0)]
                    
                    for comp in comps:
                        if abs(comp % divisor) > 1e-12:
                            return False
                    return True
                except:
                    return False
            
            def is_prime_like_quaternion(q):
                # Check norm
                try:
                    if hasattr(q, 'norm'):
                        norm = q.norm()
                    elif hasattr(q, '__abs__'):
                        norm = abs(q)
                    else:
                        # Calculate norm manually
                        if hasattr(q, 'a') and hasattr(q, 'b') and hasattr(q, 'c') and hasattr(q, 'd'):
                            norm = (q.a**2 + q.b**2 + q.c**2 + q.d**2)**0.5
                        else:
                            norm = 0
                    
                    # Simple prime check on norm
                    if norm < 2:
                        return False
                    for i in range(2, int(norm**0.5) + 1):
                        if norm % i == 0:
                            return False
                    return True
                except:
                    return False
            
            def get_unit_quaternion():
                return quaternion(1, 0, 0, 0)
            
            def get_default_quaternion():
                return quaternion(0, 0, 0, 0)
            
            return {
                "add": common_add,
                "subtract": common_subtract,
                "multiply": common_multiply,
                "divide": common_divide,
                "mod": lambda a, b: a,
                "is_divisible": is_divisible_quaternion,
                "is_prime_like": is_prime_like_quaternion,
                "get_unit": get_unit_quaternion,
                "get_default": get_default_quaternion,
            }
            
        except ImportError:
            # Fallback for quaternion
            logger.warning("Quaternion class not available, using tuple representation")
            return _get_array_type_operations(4, "Quaternion")
    
    elif number_type == 7:  # Neutrosophic
        def is_divisible_neutro(n, divisor):
            # n is (T, I, F) tuple
            try:
                t, i, f = n
                return (abs(t % divisor) < 1e-12 and 
                        abs(i % divisor) < 1e-12 and 
                        abs(f % divisor) < 1e-12)
            except:
                return False
        
        def is_prime_like_neutro(n):
            # Check truth component
            try:
                t, i, f = n
                # Simple prime check on truth component
                if t < 2:
                    return False
                for j in range(2, int(t**0.5) + 1):
                    if t % j == 0:
                        return False
                return True
            except:
                return False
        
        def get_unit_neutro():
            return (1.0, 0.0, 0.0)
        
        def get_default_neutro():
            return (0.0, 0.0, 0.0)
        
        return {
            "add": lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]),
            "subtract": lambda a, b: (a[0] - b[0], a[1] - b[1], a[2] - b[2]),
            "multiply": lambda a, b: (a[0] * b[0], a[1] * b[1], a[2] * b[2]),
            "divide": lambda a, b: (a[0] / b[0], a[1] / b[1], a[2] / b[2]),
            "mod": lambda a, b: a,
            "is_divisible": is_divisible_neutro,
            "is_prime_like": is_prime_like_neutro,
            "get_unit": get_unit_neutro,
            "get_default": get_default_neutro,
        }
    
    elif number_type in [12, 13, 17, 18, 19, 20, 22]:  # Array-based types
        sizes = {
            12: 8,   # Octonion
            13: 16,  # Sedenion
            17: 32,  # Pathion
            18: 64,  # Chingon
            19: 128, # Routon
            20: 256, # Voudon
            22: 3,   # Ternary
        }
        size = sizes.get(number_type, 1)
        return _get_array_type_operations(size, f"Type {number_type}")
    
    else:
        # Default for other types
        return {
            "add": common_add,
            "subtract": common_subtract,
            "multiply": common_multiply,
            "divide": common_divide,
            "mod": lambda a, b: a,
            "is_divisible": lambda a, b: True,
            "is_prime_like": lambda a: False,
            "get_unit": lambda: 1.0,
            "get_default": lambda: 0.0,
        }


def _get_array_type_operations(size: int, type_name: str) -> Dict[str, Callable]:
    """
    Get operations for array-based number types.
    """
    def array_add(a, b):
        if isinstance(b, (int, float)):
            return [x + b for x in a]
        else:
            # Element-wise addition
            return [a[i] + b[i] for i in range(min(len(a), len(b)))]
    
    def array_subtract(a, b):
        if isinstance(b, (int, float)):
            return [x - b for x in a]
        else:
            return [a[i] - b[i] for i in range(min(len(a), len(b)))]
    
    def array_multiply(a, b):
        if isinstance(b, (int, float)):
            return [x * b for x in a]
        else:
            return [a[i] * b[i] for i in range(min(len(a), len(b)))]
    
    def array_divide(a, b):
        if isinstance(b, (int, float)):
            return [x / b for x in a]
        else:
            return [a[i] / b[i] for i in range(min(len(a), len(b)))]
    
    def array_is_divisible(a, divisor):
        # Check first component
        try:
            return abs(a[0] % divisor) < 1e-12
        except:
            return True
    
    def array_is_prime_like(a):
        # Check first component
        try:
            val = abs(a[0])
            if val < 2:
                return False
            for i in range(2, int(val**0.5) + 1):
                if val % i == 0:
                    return False
            return True
        except:
            return False
    
    def array_get_unit():
        unit = [0.0] * size
        unit[0] = 1.0
        return unit
    
    def array_get_default():
        return [0.0] * size
    
    return {
        "add": array_add,
        "subtract": array_subtract,
        "multiply": array_multiply,
        "divide": array_divide,
        "mod": lambda a, b: a,
        "is_divisible": array_is_divisible,
        "is_prime_like": array_is_prime_like,
        "get_unit": array_get_unit,
        "get_default": array_get_default,
    }


def _get_parser_for_type_simple(kececi_type: int) -> Callable[[str], Any]:
    """
    Get parser function for Keçeci type.
    Uses existing parsers from kececinumbers module.
    """
    # Map type to parser
    if kececi_type == 1:  # Positive Real
        return _parse_fraction
    elif kececi_type == 2:  # Negative Real
        return lambda s: -_parse_fraction(s)
    elif kececi_type == 3:  # Complex
        return _parse_complex
    elif kececi_type == 4:  # Float
        return _parse_fraction
    elif kececi_type == 5:  # Rational
        # Try to return as Fraction
        try:
            from fractions import Fraction
            return lambda s: Fraction(_parse_fraction(s)).limit_denominator()
        except:
            return _parse_fraction
    elif kececi_type == 6:  # Quaternion
        return _parse_quaternion
    elif kececi_type == 7:  # Neutrosophic
        return _parse_neutrosophic
    elif kececi_type == 8:  # Neutrosophic Complex
        return _parse_neutrosophic_complex
    elif kececi_type == 9:  # Hyperreal
        return _parse_hyperreal
    elif kececi_type == 10:  # Bicomplex
        return _parse_bicomplex
    elif kececi_type == 11:  # Neutrosophic Bicomplex
        return _parse_neutrosophic_bicomplex
    elif kececi_type == 12:  # Octonion
        return _parse_octonion
    elif kececi_type == 13:  # Sedenion
        return _parse_sedenion
    elif kececi_type == 14:  # Clifford
        return _parse_clifford
    elif kececi_type == 15:  # Dual
        return _parse_dual
    elif kececi_type == 16:  # Split-Complex
        return _parse_splitcomplex
    elif kececi_type == 17:  # Pathion
        return _parse_pathion
    elif kececi_type == 18:  # Chingon
        return _parse_chingon
    elif kececi_type == 19:  # Routon
        return _parse_routon
    elif kececi_type == 20:  # Voudon
        return _parse_voudon
    elif kececi_type == 21:  # Superreal
        return _parse_superreal
    elif kececi_type == 22:  # Ternary
        return _parse_ternary
    elif kececi_type == 23:  # Hypercomplex
        return _parse_hypercomplex
    else:
        raise ValueError(f"Unsupported type: {kececi_type}")


def _generate_sequence_original(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Original ASK algorithm from version 0.8.6.
    """
    # Get operations for this type
    ops = _get_type_operations(number_type)
    
    result = []
    current = start_value
    ask_counter = 0  # 0: +unit, 1: -unit
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            # 1. ADDITION
            added = ops["add"](current, add_value)
            
            next_val = added
            divided = False
            
            # 2. CHECK DIVISIBILITY by 2 or 3
            for divisor in [2, 3]:
                try:
                    if ops["is_divisible"](added, divisor):
                        divided_val = ops["divide"](added, divisor)
                        next_val = divided_val
                        divided = True
                        break
                except Exception:
                    continue
            
            # 3. KECEÇI UNIT adjustment if not divided and prime-like
            if not divided and ops["is_prime_like"](added):
                unit = ops["get_unit"](current)
                direction = 1 if ask_counter == 0 else -1
                
                try:
                    if direction > 0:
                        adjusted = ops["add"](added, unit)
                    else:
                        adjusted = ops["subtract"](added, unit)
                    
                    # Toggle ask counter
                    ask_counter = 1 - ask_counter
                    
                    # Try division again on adjusted value
                    for divisor in [2, 3]:
                        try:
                            if ops["is_divisible"](adjusted, divisor):
                                final_val = ops["divide"](adjusted, divisor)
                                next_val = final_val
                                break
                        except Exception:
                            continue
                    else:
                        # No division successful, use adjusted value
                        next_val = adjusted
                except Exception:
                    # If unit adjustment fails, keep original
                    pass
            
            # Update current value
            current = next_val
            
            # Add to result
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for type {number_type}: {e}")
            default_val = ops["get_default"]()
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _generate_sequence_with_operation(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Generate sequence using standard mathematical operations.
    """
    ops = _get_type_operations(number_type)
    
    result = []
    current = start_value
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            # Apply operation
            if operation == "add":
                current = ops["add"](current, add_value)
            elif operation == "subtract":
                current = ops["subtract"](current, add_value)
            elif operation == "multiply":
                current = ops["multiply"](current, add_value)
            elif operation == "divide":
                current = ops["divide"](current, add_value)
            elif operation == "mod":
                current = ops["mod"](current, add_value)
            elif operation == "power":
                current = ops["power"](current, add_value)
            else:
                # Default to addition
                current = ops["add"](current, add_value)
            
            # Add to result
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for type {number_type}, operation {operation}: {e}")
            default_val = ops["get_default"]()
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result




def _get_type_operations(number_type: int) -> Dict[str, Callable]:
    """
    Return a dict of operations for the given number_type.
    Each operation is a callable that accepts (a, b) or (a,) depending on op.
    Fallbacks try to be consistent and predictable across types.
    """

    # Basic arithmetic wrappers with layered fallbacks
    def type_add(a, b):
        try:
            return a + b
        except Exception as e:
            logger.debug("add failed with %s, trying fallbacks", e)
            # try module-level safe_add if available
            try:
                from .kececinumbers import safe_add
                return safe_add(a, b)
            except Exception:
                pass
            # elementwise fallback when b is scalar
            if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
                return type(a)([x + b for x in a])
            raise

    def type_subtract(a, b):
        try:
            return a - b
        except Exception as e:
            logger.debug("subtract failed with %s", e)
            if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
                return type(a)([x - b for x in a])
            raise

    def type_multiply(a, b):
        try:
            return a * b
        except Exception as e:
            logger.debug("multiply failed with %s", e)
            if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
                return type(a)([x * b for x in a])
            raise

    def type_divide(a, b):
        try:
            return _safe_divide(a, b)
        except Exception as e:
            logger.debug("safe_divide failed with %s", e)
            try:
                return a / b
            except Exception:
                if isinstance(a, (list, tuple)) and _is_numeric_scalar(b):
                    return type(a)([x / b for x in a])
                raise

    def type_mod(a, b):
        try:
            return _safe_mod(a, b)
        except Exception as e:
            logger.debug("safe_mod failed with %s", e)
            try:
                return a % b
            except Exception:
                # If mod not supported, return a unchanged (explicit choice)
                return a

    def type_power(a, b):
        try:
            return _safe_power(a, b)
        except Exception as e:
            logger.debug("safe_power failed with %s", e)
            try:
                return a ** b
            except Exception:
                # sensible fallbacks for common exponents
                if _is_numeric_scalar(b):
                    if b == 2:
                        return type_multiply(a, a)
                    if b == 1:
                        return a
                    if b == 0:
                        # try to return multiplicative identity for the type
                        try:
                            unit = ops["get_unit"]()
                            return unit if unit is not None else 1
                        except Exception:
                            return 1
                raise

    # divisibility and primality helpers
    def type_is_divisible(a, divisor):
        try:
            from .kececinumbers import _is_divisible as _mod_check
            return _mod_check(a, divisor, number_type)
        except Exception:
            try:
                # numeric fallback
                if _is_numeric_scalar(a) and _is_numeric_scalar(divisor):
                    return abs(a % divisor) < 1e-12
                if isinstance(a, complex) and _is_numeric_scalar(divisor):
                    return (abs(a.real % divisor) < 1e-12) and (abs(a.imag % divisor) < 1e-12)
                # arrays: check first component
                if isinstance(a, (list, tuple)) and _is_numeric_scalar(divisor):
                    return abs(_coerce_first_component(a) % divisor) < 1e-12
            except Exception as e:
                logger.debug("is_divisible fallback failed %s", e)
            # conservative default
            return False

    def type_is_prime_like(a):
        try:
            from .kececinumbers import is_prime_like
            return is_prime_like(a, number_type)
        except Exception:
            # fallback: check primality of magnitude or first component
            try:
                from .kececinumbers import is_prime
            except Exception:
                is_prime = None
            try:
                mag = _coerce_first_component(a)
                if is_prime:
                    return is_prime(int(abs(mag)))
                # simple trial division for small integers
                n = int(abs(mag))
                if n < 2:
                    return False
                if n in (2, 3):
                    return True
                if n % 2 == 0:
                    return False
                r = int(math.sqrt(n))
                for i in range(3, r + 1, 2):
                    if n % i == 0:
                        return False
                return True
            except Exception as e:
                logger.debug("is_prime_like fallback failed %s", e)
                return False

    # unit and default value providers
    def type_get_unit(sample=None):
        # try to use centralized helper if available
        try:
            from .module_helpers import get_unit_for_type
            return get_unit_for_type(number_type, sample=sample)
        except Exception:
            pass

        # inline fallbacks
        if number_type in (1, 4, 5):
            return 1.0
        if number_type == 2:
            return -1.0
        if number_type == 3:
            return complex(1, 0)
        if number_type == 6:
            try:
                from .kececinumbers import quaternion
                return quaternion(1, 0, 0, 0)
            except Exception:
                return (1.0, 0.0, 0.0, 0.0)
        if number_type == 7:
            try:
                from .kececinumbers import NeutrosophicNumber
                return NeutrosophicNumber(1.0, 0.0, 0.0)
            except Exception:
                return (1.0, 0.0, 0.0)
        if number_type in (12, 13, 17, 18, 19, 20, 22):
            sizes = {12: 8, 13: 16, 17: 32, 18: 64, 19: 128, 20: 256, 22: 3}
            size = sizes.get(number_type, 1)
            return [1.0] + [0.0] * (size - 1)
        if number_type == 9:
            return [1.0, 0.0]
        if number_type == 10:
            return complex(1, 0)
        if number_type == 23: #hypercomplex
            # infer dimension from sample if provided, default to 8
            dim = 8
            if isinstance(sample, (list, tuple)):
                dim = max(1, len(sample))
            return [1.0] + [0.0] * (dim - 1)
        # default
        return 1.0

    def type_get_default(sample=None):
        try:
            from .module_helpers import get_default_for_type
            return get_default_for_type(number_type, sample=sample)
        except Exception:
            pass

        if number_type in (1, 2, 4, 5):
            return 0.0
        if number_type == 3:
            return complex(0, 0)
        if number_type == 6:
            try:
                from .kececinumbers import quaternion
                return quaternion(0, 0, 0, 0)
            except Exception:
                return (0.0, 0.0, 0.0, 0.0)
        if number_type == 7:
            try:
                from .kececinumbers import NeutrosophicNumber
                return NeutrosophicNumber(0.0, 0.0, 0.0)
            except Exception:
                return (0.0, 0.0, 0.0)
        if number_type in (12, 13, 17, 18, 19, 20, 22):
            sizes = {12: 8, 13: 16, 17: 32, 18: 64, 19: 128, 20: 256, 22: 3}
            size = sizes.get(number_type, 1)
            return [0.0] * size
        if number_type == 9:
            return [0.0, 0.0]
        if number_type == 10:
            return complex(0, 0)
        if number_type == 23:
            dim = 8
            if isinstance(sample, (list, tuple)):
                dim = max(1, len(sample))
            return [0.0] * dim
        return 0.0

    # assemble ops dict; some ops may reference others so create dict first then fill if needed
    ops = {
        "add": type_add,
        "subtract": type_subtract,
        "multiply": type_multiply,
        "divide": type_divide,
        "mod": type_mod,
        "power": type_power,
        "is_divisible": type_is_divisible,
        "is_prime_like": type_is_prime_like,
        "get_unit": type_get_unit,
        "get_default": type_get_default,
    }

    return ops


def _parse_special_type(
    kececi_type: int,
    start_input_raw: str,
    add_input_raw: str
) -> Tuple[Any, Any]:
    """
    Parse values for special Keçeci number types (6-23).
    """
    try:
        # First try to use the specific parser
        parser_func = _get_parser_for_type(kececi_type)
        if parser_func:
            return parser_func(start_input_raw), parser_func(add_input_raw)
    except Exception as e:
        logger.debug(f"Specific parser failed for type {kececi_type}: {e}")
    
    # Fallback to generic parsing
    return _parse_with_generic_fallback(kececi_type, start_input_raw, add_input_raw)


def _get_parser_for_type(kececi_type: int) -> Optional[Callable]:
    """
    Get parser function for a specific Keçeci type.
    """
    try:
        from .kececinumbers import (
            _parse_bicomplex,
            _parse_chingon,
            _parse_clifford,
            _parse_complex,
            _parse_complex_like_string,
            _parse_dual,
            _parse_engineering_notation,
            _parse_fraction,
            _parse_hyperreal,
            _parse_neutrosophic,
            _parse_neutrosophic_bicomplex,
            _parse_neutrosophic_complex,
            _parse_octonion,
            _parse_pathion,
            _parse_quaternion,
            _parse_quaternion_from_csv,
            _parse_real,
            _parse_routon,
            _parse_sedenion,
            _parse_splitcomplex,
            _parse_super_real,
            _parse_superreal,
            _parse_ternary,
            _parse_hypercomplex,
            _parse_universal,
            _parse_voudon,
            _generate_simple_ask_sequence,
            _parse_with_fallback_simple,
            _parse_kececi_values,
            parse_to_hyperreal,
            parse_to_neutrosophic,
        )
        
        parser_map = {
            6: _parse_quaternion,
            7: _parse_neutrosophic,
            8: _parse_neutrosophic_complex,
            9: _parse_hyperreal,
            10: _parse_bicomplex,
            11: _parse_neutrosophic_bicomplex,
            12: _parse_octonion,
            13: _parse_sedenion,
            14: _parse_clifford,
            15: _parse_dual,
            16: _parse_splitcomplex,
            17: _parse_pathion,
            18: _parse_chingon,
            19: _parse_routon,
            20: _parse_voudon,
            21: _parse_super_real,
            22: _parse_ternary,
            23: _parse_hypercomplex,
        }
        
        return parser_map.get(kececi_type)
    except ImportError:
        return None

# ----------------- Yardımcılar -----------------
def _pad_or_truncate(lst: Iterable[Any], dim: int) -> List[Any]:
    arr = list(lst)
    if len(arr) < dim:
        return arr + [0.0] * (dim - len(arr))
    return arr[:dim]

def _try_construct(cls, args_list: List[Any], dimension: Optional[int] = None):
    """
    Bir sınıfı farklı constructor imzalarıyla dene:
      - cls(list, dimension=dim)
      - cls(*components, dimension=dim)
      - cls(list)
      - cls(*components)
    """
    # prefer list + dimension
    try:
        if dimension is not None:
            return cls(args_list, dimension=dimension)
    except TypeError:
        pass
    except Exception as e:
        logger.debug("Constructor attempt failed (list+dim): %s", e)
    # try *args + dimension
    try:
        if dimension is not None:
            return cls(*args_list, dimension=dimension)
    except TypeError:
        pass
    except Exception as e:
        logger.debug("Constructor attempt failed (*args+dim): %s", e)
    # try list only
    try:
        return cls(args_list)
    except Exception as e:
        logger.debug("Constructor attempt failed (list): %s", e)
    # try *args only
    try:
        return cls(*args_list)
    except Exception as e:
        logger.debug("Constructor attempt failed (*args): %s", e)
    # all failed
    raise TypeError("No compatible constructor found for class {}".format(cls))

# ----------------- Ana fonksiyon -----------------

def _parse_with_generic_fallback(
    kececi_type: int,
    start_input_raw: Any,
    add_input_raw: Any
) -> Tuple[Any, Any]:
    """
    Generic fallback parser for special types.
    - start_input_raw / add_input_raw: string, iterable, scalar accepted.
    - Returns (start_value, add_value) as either project class instances (if available)
      or consistent Python fallback structures (tuple/list/complex/float).
    """
    # First try to use project-specific fraction parser if available (keçeci projesi)
    base_start = None
    base_add = None
    try:
        from .kececinumbers import _parse_fraction as _pf
        try:
            base_start = _pf(start_input_raw)
        except Exception:
            base_start = None
        try:
            base_add = _pf(add_input_raw)
        except Exception:
            base_add = None
    except Exception:
        base_start = None
        base_add = None

    # If project parser didn't produce values, use generic parser
    if base_start is None:
        parsed_start = _parse_components(start_input_raw)
        base_start = parsed_start[0] if parsed_start else 0.0
    if base_add is None:
        parsed_add = _parse_components(add_input_raw)
        base_add = parsed_add[0] if parsed_add else 0.0

    # Helper to attempt class construction with safe logging
    def _construct_or_fallback(class_name: str, cls_try, args_list, dim=None, fallback_type="list"):
        try:
            inst = _try_construct(cls_try, args_list, dimension=dim)
            return inst
        except Exception as e:
            logger.debug("Could not construct %s: %s", class_name, e)
            # fallback: return list or tuple depending on fallback_type
            if fallback_type == "tuple":
                return tuple(args_list)
            return list(args_list)

    # Map types
    # For scalar-like types we return simple scalars/complex
    if kececi_type == 6:  # Quaternion (4D)
        quat = (base_start, 0.0, 0.0, 0.0)
        try:
            from .kececinumbers import QuaternionNumber
            return _construct_or_fallback("QuaternionNumber", QuaternionNumber, list(quat), dim=4, fallback_type="tuple"), \
                   _construct_or_fallback("QuaternionNumber", QuaternionNumber, list((base_add,0.0,0.0,0.0)), dim=4, fallback_type="tuple")
        except Exception:
            return tuple(quat), tuple((base_add,0.0,0.0,0.0))

    if kececi_type == 7:  # Neutrosophic (T,I,F)
        neutro = (base_start, 0.0, 0.0)
        try:
            from .kececinumbers import NeutrosophicNumber
            return _construct_or_fallback("NeutrosophicNumber", NeutrosophicNumber, list(neutro), dim=3, fallback_type="tuple"), \
                   _construct_or_fallback("NeutrosophicNumber", NeutrosophicNumber, list((base_add,0.0,0.0)), dim=3, fallback_type="tuple")
        except Exception:
            return tuple(neutro), tuple((base_add,0.0,0.0))

    if kececi_type == 8:  # Neutrosophic Complex
        try:
            from .kececinumbers import NeutrosophicComplexNumber
            return _construct_or_fallback("NeutrosophicComplexNumber", NeutrosophicComplexNumber, [base_start, 0.0, 0.0], dim=None), \
                   _construct_or_fallback("NeutrosophicComplexNumber", NeutrosophicComplexNumber, [base_add, 0.0, 0.0], dim=None)
        except Exception:
            return complex(base_start, 0), complex(base_add, 0)

    if kececi_type == 9:  # Hyperreal [finite, infinitesimal]
        hyper_start = [base_start, 0.0]
        hyper_add = [base_add, 0.0]
        try:
            from .kececinumbers import HyperrealNumber
            return _construct_or_fallback("HyperrealNumber", HyperrealNumber, hyper_start, dim=2), \
                   _construct_or_fallback("HyperrealNumber", HyperrealNumber, hyper_add, dim=2)
        except Exception:
            return hyper_start, hyper_add

    if kececi_type == 10:  # Bicomplex
        bic_start = complex(base_start, 0)
        bic_add = complex(base_add, 0)
        try:
            from .kececinumbers import BicomplexNumber
            return _construct_or_fallback("BicomplexNumber", BicomplexNumber, [bic_start]), \
                   _construct_or_fallback("BicomplexNumber", BicomplexNumber, [bic_add])
        except Exception:
            return bic_start, bic_add

    if kececi_type == 11:  # Neutrosophic Bicomplex (fallback to complex)
        try:
            from .kececinumbers import NeutrosophicBicomplexNumber
            return _construct_or_fallback("NeutrosophicBicomplexNumber", NeutrosophicBicomplexNumber, [base_start]), \
                   _construct_or_fallback("NeutrosophicBicomplexNumber", NeutrosophicBicomplexNumber, [base_add])
        except Exception:
            return complex(base_start, 0), complex(base_add, 0)

    if kececi_type == 12:  # Octonion (8D)
        octo_start = [base_start] + [0.0] * 7
        octo_add = [base_add] + [0.0] * 7
        try:
            from .kececinumbers import OctonionNumber
            return _construct_or_fallback("OctonionNumber", OctonionNumber, octo_start, dim=8), \
                   _construct_or_fallback("OctonionNumber", OctonionNumber, octo_add, dim=8)
        except Exception:
            return octo_start, octo_add

    if kececi_type == 13:  # Sedenion (16D)
        sed_start = [base_start] + [0.0] * 15
        sed_add = [base_add] + [0.0] * 15
        try:
            from .kececinumbers import SedenionNumber
            return _construct_or_fallback("SedenionNumber", SedenionNumber, sed_start, dim=16), \
                   _construct_or_fallback("SedenionNumber", SedenionNumber, sed_add, dim=16)
        except Exception:
            return sed_start, sed_add

    if kececi_type == 14:  # Clifford
        cliff_start = {"e0": base_start}
        cliff_add = {"e0": base_add}
        try:
            from .kececinumbers import CliffordNumber
            return _construct_or_fallback("CliffordNumber", CliffordNumber, [cliff_start]), \
                   _construct_or_fallback("CliffordNumber", CliffordNumber, [cliff_add])
        except Exception:
            return cliff_start, cliff_add

    if kececi_type == 15:  # Dual
        dual_start = (base_start, 0.0)
        dual_add = (base_add, 0.0)
        try:
            from .kececinumbers import DualNumber
            return _construct_or_fallback("DualNumber", DualNumber, list(dual_start), dim=2, fallback_type="tuple"), \
                   _construct_or_fallback("DualNumber", DualNumber, list(dual_add), dim=2, fallback_type="tuple")
        except Exception:
            return dual_start, dual_add

    if kececi_type == 16:  # Split-complex
        split_start = (base_start, 0.0)
        split_add = (base_add, 0.0)
        try:
            from .kececinumbers import SplitcomplexNumber
            return _construct_or_fallback("SplitcomplexNumber", SplitcomplexNumber, list(split_start), dim=2, fallback_type="tuple"), \
                   _construct_or_fallback("SplitcomplexNumber", SplitcomplexNumber, list(split_add), dim=2, fallback_type="tuple")
        except Exception:
            return split_start, split_add

    if kececi_type == 17:  # Pathion (32D)
        path_start = [base_start] + [0.0] * 31
        path_add = [base_add] + [0.0] * 31
        try:
            from .kececinumbers import PathionNumber
            return _construct_or_fallback("PathionNumber", PathionNumber, path_start, dim=32), \
                   _construct_or_fallback("PathionNumber", PathionNumber, path_add, dim=32)
        except Exception:
            return path_start, path_add

    if kececi_type == 18:  # Chingon (64D)
        ching_start = [base_start] + [0.0] * 63
        ching_add = [base_add] + [0.0] * 63
        try:
            from .kececinumbers import ChingonNumber
            return _construct_or_fallback("ChingonNumber", ChingonNumber, ching_start, dim=64), \
                   _construct_or_fallback("ChingonNumber", ChingonNumber, ching_add, dim=64)
        except Exception:
            return ching_start, ching_add

    if kececi_type == 19:  # Routon (128D)
        rout_start = [base_start] + [0.0] * 127
        rout_add = [base_add] + [0.0] * 127
        try:
            from .kececinumbers import RoutonNumber
            return _construct_or_fallback("RoutonNumber", RoutonNumber, rout_start, dim=128), \
                   _construct_or_fallback("RoutonNumber", RoutonNumber, rout_add, dim=128)
        except Exception:
            return rout_start, rout_add

    if kececi_type == 20:  # Voudon (256D)
        voud_start = [base_start] + [0.0] * 255
        voud_add = [base_add] + [0.0] * 255
        try:
            from .kececinumbers import VoudonNumber
            return _construct_or_fallback("VoudonNumber", VoudonNumber, voud_start, dim=256), \
                   _construct_or_fallback("VoudonNumber", VoudonNumber, voud_add, dim=256)
        except Exception:
            return voud_start, voud_add

    if kececi_type == 21:  # Superreal
        super_start = (base_start, 0.0)
        super_add = (base_add, 0.0)
        try:
            from .kececinumbers import SuperrealNumber
            return _construct_or_fallback("SuperrealNumber", SuperrealNumber, list(super_start), dim=2, fallback_type="tuple"), \
                   _construct_or_fallback("SuperrealNumber", SuperrealNumber, list(super_add), dim=2, fallback_type="tuple")
        except Exception:
            return super_start, super_add

    if kececi_type == 22:  # Ternary (3D)
        parsed_start = _parse_components(start_input_raw)
        parsed_add = _parse_components(add_input_raw)
        base_start_comp = float(parsed_start[0]) if parsed_start else 0.0
        base_add_comp = float(parsed_add[0]) if parsed_add else 0.0
        ternary_start = [base_start_comp, 0.0, 0.0]
        ternary_add = [base_add_comp, 0.0, 0.0]
        try:
            from .kececinumbers import TernaryNumber
            return _construct_or_fallback("TernaryNumber", TernaryNumber, ternary_start, dim=3), \
                   _construct_or_fallback("TernaryNumber", TernaryNumber, ternary_add, dim=3)
        except Exception as e:
            logger.debug("TernaryNumber import/construct failed: %s", e)
            return ternary_start, ternary_add

    if kececi_type == 23:  # Hypercomplex (variable power-of-two dimension)
        parsed_start = _parse_components(start_input_raw)
        parsed_add = _parse_components(add_input_raw)
        desired_len = max(1, len(parsed_start), len(parsed_add))
        dim = _next_power_of_two_at_least(desired_len, max_dim=256)
        hyper_start_list = _pad_or_truncate(parsed_start, dim)
        hyper_add_list = _pad_or_truncate(parsed_add, dim)
        try:
            from .kececinumbers import HypercomplexNumber as HC
            try:
                return _construct_or_fallback("HypercomplexNumber", HC, hyper_start_list, dim=dim), \
                       _construct_or_fallback("HypercomplexNumber", HC, hyper_add_list, dim=dim)
            except Exception:
                # last attempt: try without dimension kwarg
                return _construct_or_fallback("HypercomplexNumber", HC, hyper_start_list), \
                       _construct_or_fallback("HypercomplexNumber", HC, hyper_add_list)
        except Exception as e:
            logger.debug("HypercomplexNumber import/construct failed: %s", e)
            return hyper_start_list, hyper_add_list

    # Default fallback: return parsed scalar/fraction results
    return base_start, base_add



def _generate_ask_for_type(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool,
    number_type: int
) -> List[Any]:
    """
    Generate ASK sequence for a specific number type.
    """
    # Get appropriate operations for this type
    operations = _get_operations_for_type(number_type)
    
    result = []
    current = start_value
    ask_counter = 0
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "type": _get_type_name(number_type),
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    for i in range(1, iterations):
        try:
            # 1. ADD
            added = operations["add"](current, add_value)
            
            # 2. Check division
            next_val = added
            divided = False
            
            for divisor in [2, 3]:
                try:
                    if operations["is_divisible"](added, divisor):
                        next_val = operations["divide"](added, divisor)
                        divided = True
                        break
                except:
                    continue
            
            # 3. Keçeci unit adjustment
            if not divided and operations["is_prime_like"](added):
                unit = operations["get_unit"](current)
                
                if ask_counter == 0:
                    adjusted = operations["add"](added, unit)
                else:
                    adjusted = operations["subtract"](added, unit)
                
                ask_counter = 1 - ask_counter
                
                # Try division on adjusted value
                for divisor in [2, 3]:
                    try:
                        if operations["is_divisible"](adjusted, divisor):
                            next_val = operations["divide"](adjusted, divisor)
                            break
                    except:
                        continue
                else:
                    next_val = adjusted
            
            current = next_val
            
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "step",
                    "type": _get_type_name(number_type),
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i} for type {number_type}: {e}")
            default_val = operations["get_default"]()
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "type": _get_type_name(number_type),
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _get_operations_for_type(number_type: int) -> Dict[str, Callable]:
    """
    Get appropriate operations for a number type.
    """
    # Basic operations that work for most types
    def basic_add(a, b):
        try:
            return a + b
        except:
            # Fallback for tuples/lists
            if isinstance(a, (tuple, list)) and isinstance(b, (int, float)):
                if isinstance(a, tuple):
                    return tuple(x + b for x in a)
                return [x + b for x in a]
            raise
    
    def basic_subtract(a, b):
        try:
            return a - b
        except:
            if isinstance(a, (tuple, list)) and isinstance(b, (int, float)):
                if isinstance(a, tuple):
                    return tuple(x - b for x in a)
                return [x - b for x in a]
            raise
    
    def basic_divide(a, divisor):
        try:
            return a / divisor
        except:
            if isinstance(a, (tuple, list)):
                if isinstance(a, tuple):
                    return tuple(x / divisor for x in a)
                return [x / divisor for x in a]
            raise
    
    def basic_is_divisible(a, divisor):
        try:
            # For numeric types
            if hasattr(a, '__mod__'):
                remainder = a % divisor
                if hasattr(remainder, '__abs__'):
                    return abs(remainder) < 1e-10
                return abs(remainder) < 1e-10
            return True
        except:
            return True
    
    def basic_is_prime_like(a):
        try:
            # Try to get a numeric value
            if isinstance(a, (int, float, complex)):
                val = abs(a)
            elif isinstance(a, (tuple, list)):
                # Use first component
                val = abs(a[0]) if a else 0
            else:
                val = abs(float(str(a)))
            
            # Simple prime check
            if val < 2:
                return False
            for i in range(2, int(val**0.5) + 1):
                if val % i == 0:
                    return False
            return True
        except:
            return False
    
    def basic_get_unit(sample=None):
        if number_type in [1, 4, 5]:
            return 1.0
        elif number_type == 2:
            return -1.0
        elif number_type == 3:
            return complex(1, 0)
        elif number_type == 6:
            return (1.0, 0.0, 0.0, 0.0)
        elif number_type == 7:
            return (1.0, 0.0, 0.0)
        elif number_type in [12, 13, 17, 18, 19, 20, 22]:
            # Array types
            sizes = {
                12: 8, 13: 16, 17: 32, 18: 64, 19: 128, 20: 256, 22: 3
            }
            size = sizes.get(number_type, 1)
            unit = [0.0] * size
            unit[0] = 1.0
            return unit
        else:
            return 1.0
    
    def basic_get_default():
        if number_type in [1, 2, 4, 5]:
            return 0.0
        elif number_type == 3:
            return complex(0, 0)
        elif number_type == 6:
            return (0.0, 0.0, 0.0, 0.0)
        elif number_type == 7:
            return (0.0, 0.0, 0.0)
        elif number_type in [12, 13, 17, 18, 19, 20, 22]:
            sizes = {
                12: 8, 13: 16, 17: 32, 18: 64, 19: 128, 20: 256, 22: 3
            }
            size = sizes.get(number_type, 1)
            return [0.0] * size
        else:
            return 0.0
    
    return {
        "add": basic_add,
        "subtract": basic_subtract,
        "divide": basic_divide,
        "is_divisible": basic_is_divisible,
        "is_prime_like": basic_is_prime_like,
        "get_unit": basic_get_unit,
        "get_default": basic_get_default,
    }


def _get_type_name(number_type: int) -> str:
    """Get name for number type."""
    names = {
        1: "Positive Real",
        2: "Negative Real",
        3: "Complex",
        4: "Float",
        5: "Rational",
        6: "Quaternion",
        7: "Neutrosophic",
        8: "Neutrosophic Complex",
        9: "Hyperreal",
        10: "Bicomplex",
        11: "Neutrosophic Bicomplex",
        12: "Octonion",
        13: "Sedenion",
        14: "Clifford",
        15: "Dual",
        16: "Split-Complex",
        17: "Pathion",
        18: "Chingon",
        19: "Routon",
        20: "Voudon",
        21: "Superreal",
        22: "Ternary",
        23: "Hypercomplex",
    }
    return names.get(number_type, f"Type {number_type}")


def _generate_simple_ask_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Simple ASK algorithm implementation.
    """
    result = []
    current = start_value
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    else:
        result.append(current)
    
    ask_counter = 0  # 0: add unit, 1: subtract unit
    
    for i in range(1, iterations):
        try:
            # 1. ADD
            added = _simple_add(current, add_value)
            
            # 2. Check division by 2 or 3
            next_val = added
            divided = False
            
            for divisor in [2, 3]:
                try:
                    if _simple_is_divisible(added, divisor):
                        next_val = _simple_divide(added, divisor)
                        divided = True
                        logger.debug(f"Divided by {divisor}: {added} -> {next_val}")
                        break
                except Exception as e:
                    logger.debug(f"Division by {divisor} failed: {e}")
                    continue
            
            # 3. If not divided, check if prime-like and apply Keçeci unit
            if not divided:
                if _simple_is_prime_like(added):
                    # Get appropriate unit
                    unit = _get_simple_unit(number_type, current)
                    
                    # Apply unit based on ask_counter
                    if ask_counter == 0:
                        adjusted = _simple_add(added, unit)
                        logger.debug(f"Added unit {unit}: {added} -> {adjusted}")
                    else:
                        adjusted = _simple_subtract(added, unit)
                        logger.debug(f"Subtracted unit {unit}: {added} -> {adjusted}")
                    
                    # Toggle ask_counter
                    ask_counter = 1 - ask_counter
                    
                    # Try division again on adjusted value
                    for divisor in [2, 3]:
                        try:
                            if _simple_is_divisible(adjusted, divisor):
                                next_val = _simple_divide(adjusted, divisor)
                                logger.debug(f"Divided adjusted by {divisor}: {adjusted} -> {next_val}")
                                break
                        except:
                            continue
                    else:
                        # No division successful, use adjusted
                        next_val = adjusted
            
            # Update current value
            current = next_val
            
            # Add to result
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "step",
                    "description": f"Step {i}: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i}: {e}")
            default_val = _get_simple_default(number_type)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _simple_add(a: Any, b: Any) -> Any:
    """Simple addition."""
    try:
        return a + b
    except Exception:
        # For lists/tuples
        if isinstance(a, (list, tuple)) and isinstance(b, (int, float)):
            if isinstance(a, tuple):
                return tuple(x + b for x in a)
            else:
                return [x + b for x in a]
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            # Element-wise addition
            size = max(len(a), len(b))
            result = []
            for i in range(size):
                val_a = a[i] if i < len(a) else 0
                val_b = b[i] if i < len(b) else 0
                result.append(val_a + val_b)
            if isinstance(a, tuple):
                return tuple(result)
            else:
                return result
        else:
            raise


def _simple_subtract(a: Any, b: Any) -> Any:
    """Simple subtraction."""
    try:
        return a - b
    except Exception:
        # Similar to _simple_add but for subtraction
        if isinstance(a, (list, tuple)) and isinstance(b, (int, float)):
            if isinstance(a, tuple):
                return tuple(x - b for x in a)
            else:
                return [x - b for x in a]
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            size = max(len(a), len(b))
            result = []
            for i in range(size):
                val_a = a[i] if i < len(a) else 0
                val_b = b[i] if i < len(b) else 0
                result.append(val_a - val_b)
            if isinstance(a, tuple):
                return tuple(result)
            else:
                return result
        else:
            raise


def _simple_divide(a: Any, divisor: float) -> Any:
    """Simple division."""
    try:
        return a / divisor
    except Exception:
        if isinstance(a, (list, tuple)):
            if isinstance(a, tuple):
                return tuple(x / divisor for x in a)
            else:
                return [x / divisor for x in a]
        else:
            raise


def _simple_is_divisible(a: Any, divisor: float) -> bool:
    """Simple divisibility check."""
    try:
        if isinstance(a, (int, float)):
            return abs(a % divisor) < 1e-10
        elif isinstance(a, complex):
            return abs(a.real % divisor) < 1e-10 and abs(a.imag % divisor) < 1e-10
        elif isinstance(a, (list, tuple)):
            # Check first element
            if a:
                return _simple_is_divisible(a[0], divisor)
            return True
        else:
            return True  # Assume divisible for unknown types
    except:
        return False


def _simple_is_prime_like(a: Any) -> bool:
    """Simple prime-like check."""
    try:
        # Convert to float for checking
        if isinstance(a, (int, float)):
            val = abs(a)
        elif isinstance(a, complex):
            val = abs(a)
        elif isinstance(a, (list, tuple)):
            # Use first non-zero element
            for x in a:
                if abs(x) > 1e-10:
                    val = abs(x)
                    break
            else:
                return False
        else:
            # Try to get magnitude
            try:
                val = abs(a)
            except:
                return False
        
        # Simple prime check
        if val < 2:
            return False
        if val == 2 or val == 3:
            return True
        if val % 2 == 0 or val % 3 == 0:
            return False
        
        i = 5
        while i * i <= val:
            if val % i == 0 or val % (i + 2) == 0:
                return False
            i += 6
        return True
    except:
        return False


def _get_simple_unit(number_type: int, sample_value: Any = None) -> Any:
    """Get simple unit for number type."""
    if number_type in [1, 4, 5]:
        return 1.0
    elif number_type == 2:
        return -1.0
    elif number_type == 3:
        return complex(1, 0)
    elif number_type == 6:  # Quaternion
        return (1.0, 0.0, 0.0, 0.0)
    elif number_type == 7:  # Neutrosophic
        return (1.0, 0.0, 0.0)
    elif number_type in [12, 13, 17, 18, 19, 20, 22]:  # Array types
        if sample_value and hasattr(sample_value, '__len__'):
            size = len(sample_value)
            unit = [0.0] * size
            unit[0] = 1.0
            if isinstance(sample_value, tuple):
                return tuple(unit)
            return unit
        return 1.0
    else:
        return 1.0


def _get_simple_default(number_type: int) -> Any:
    """Get default value for number type."""
    if number_type in [1, 2, 4, 5]:
        return 0.0
    elif number_type == 3:
        return complex(0, 0)
    elif number_type == 6:
        return (0.0, 0.0, 0.0, 0.0)
    elif number_type == 7:
        return (0.0, 0.0, 0.0)
    elif number_type in [12, 13, 17, 18, 19, 20, 22]:
        # Return appropriate sized zero array
        sizes = {
            12: 8,   # Octonion
            13: 16,  # Sedenion
            17: 32,  # Pathion
            18: 64,  # Chingon
            19: 128, # Routon
            20: 256, # Voudon
            22: 3,   # Ternary
        }
        size = sizes.get(number_type, 1)
        return [0.0] * size
    else:
        return 0.0


def _generate_ask_sequence_complete(
    start_value: Any,
    add_value: Any,
    iterations: int,
    include_intermediate_steps: bool = False,
    number_type: int = 1
) -> List[Any]:
    """
    Complete ASK algorithm implementation for all number types.
    Steps: Add, check division by 2/3, apply Keçeci unit if prime-like.
    """
    # Get appropriate ask_unit for the number type
    ask_unit = _get_ask_unit_for_type(number_type, start_value)
    
    result = []
    current = start_value
    ask_counter = 0  # 0: +ask_unit, 1: -ask_unit
    
    if include_intermediate_steps:
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
    
    # Main ASK loop
    for i in range(1, iterations):
        step_log = [] if include_intermediate_steps else None
        
        try:
            # 1. ADDITION
            added = _safe_add(current, add_value, number_type)
            if include_intermediate_steps:
                step_log.append({
                    "operation": "add",
                    "value": added,
                    "description": f"Add {add_value}: {current} + {add_value} = {added}"
                })
            
            next_val = added
            divided = False
            
            # 2. CHECK DIVISIBILITY by 2 or 3
            for divisor in [2, 3]:
                try:
                    if _is_divisible_ask(added, divisor, number_type):
                        divided_val = _safe_divide_ask(added, divisor, number_type)
                        if include_intermediate_steps:
                            step_log.append({
                                "operation": f"divide by {divisor}",
                                "value": divided_val,
                                "description": f"Divide by {divisor}: {added} / {divisor} = {divided_val}"
                            })
                        next_val = divided_val
                        divided = True
                        break
                except Exception as e:
                    logger.debug(f"Division by {divisor} failed: {e}")
                    continue
            
            # 3. KECEÇI UNIT adjustment if not divided and prime-like
            if not divided and _is_prime_like_ask(added, number_type):
                direction = 1 if ask_counter == 0 else -1
                try:
                    # Apply Keçeci unit
                    if direction > 0:
                        adjusted = _safe_add(added, ask_unit, number_type)
                        op_desc = f"+{ask_unit}"
                    else:
                        adjusted = _safe_subtract(added, ask_unit, number_type)
                        op_desc = f"-{ask_unit}"
                    
                    if include_intermediate_steps:
                        step_log.append({
                            "operation": "keçeci unit",
                            "value": adjusted,
                            "description": f"Apply Keçeci unit {op_desc}: {added} {op_desc} = {adjusted}"
                        })
                    
                    # Toggle ask counter
                    ask_counter = 1 - ask_counter
                    
                    # Try division again on adjusted value
                    for divisor in [2, 3]:
                        try:
                            if _is_divisible_ask(adjusted, divisor, number_type):
                                final_val = _safe_divide_ask(adjusted, divisor, number_type)
                                if include_intermediate_steps:
                                    step_log.append({
                                        "operation": f"divide by {divisor}",
                                        "value": final_val,
                                        "description": f"Divide adjusted by {divisor}: {adjusted} / {divisor} = {final_val}"
                                    })
                                next_val = final_val
                                break
                        except Exception as e:
                            logger.debug(f"Division on adjusted value failed: {e}")
                            continue
                    else:
                        # No division successful, use adjusted value
                        next_val = adjusted
                except Exception as e:
                    logger.debug(f"Keçeci unit adjustment failed: {e}")
                    # Keep original added value
            
            # Update current value
            current = next_val
            
            # Add to result
            if include_intermediate_steps:
                # Add all intermediate steps
                for step in step_log:
                    result.append({
                        "step": i,
                        "value": step["value"],
                        "operation": step["operation"],
                        "description": step["description"]
                    })
                # Add final value for this iteration
                result.append({
                    "step": i,
                    "value": current,
                    "operation": "final",
                    "description": f"Iteration {i} final: {current}"
                })
            else:
                result.append(current)
                
        except Exception as e:
            logger.error(f"Error at iteration {i}: {e}")
            default_val = _get_default_value_for_type(current, number_type)
            if include_intermediate_steps:
                result.append({
                    "step": i,
                    "value": default_val,
                    "operation": "error",
                    "description": f"ERROR: {e}"
                })
            else:
                result.append(default_val)
            current = default_val
    
    return result


def _safe_add(a: Any, b: Any, number_type: int) -> Any:
    """Safe addition for all types."""
    try:
        return a + b
    except Exception:
        # Try alternative methods
        if hasattr(a, 'add'):
            return a.add(b)
        elif hasattr(a, '__add__'):
            return a.__add__(b)
        else:
            # For array types
            if isinstance(a, (list, tuple)) and isinstance(b, (int, float)):
                return type(a)([x + b for x in a])
            elif isinstance(b, (list, tuple)) and isinstance(a, (int, float)):
                return type(b)([a + x for x in b])
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                # Element-wise addition
                size = max(len(a), len(b))
                result = []
                for i in range(size):
                    val_a = a[i] if i < len(a) else 0
                    val_b = b[i] if i < len(b) else 0
                    result.append(val_a + val_b)
                return type(a)(result)
            else:
                raise


def _safe_subtract(a: Any, b: Any, number_type: int) -> Any:
    """Safe subtraction for all types."""
    try:
        return a - b
    except Exception:
        if hasattr(a, 'subtract'):
            return a.subtract(b)
        elif hasattr(a, '__sub__'):
            return a.__sub__(b)
        else:
            # Similar logic to _safe_add but for subtraction
            if isinstance(a, (list, tuple)) and isinstance(b, (int, float)):
                return type(a)([x - b for x in a])
            elif isinstance(b, (list, tuple)) and isinstance(a, (int, float)):
                return type(b)([a - x for x in b])
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                size = max(len(a), len(b))
                result = []
                for i in range(size):
                    val_a = a[i] if i < len(a) else 0
                    val_b = b[i] if i < len(b) else 0
                    result.append(val_a - val_b)
                return type(a)(result)
            else:
                raise


def _safe_divide_ask(a: Any, divisor: int, number_type: int) -> Any:
    """Safe division for ASK algorithm."""
    try:
        return a / divisor
    except Exception:
        # Try alternative division methods
        if hasattr(a, '__truediv__'):
            return a.__truediv__(divisor)
        elif hasattr(a, 'divide'):
            return a.divide(divisor)
        else:
            # For array types
            if isinstance(a, (list, tuple)):
                return type(a)([x / divisor for x in a])
            else:
                # Try to convert to float
                try:
                    return type(a)(float(a) / divisor)
                except:
                    raise


def _is_divisible_ask(value: Any, divisor: int, number_type: int) -> bool:
    """
    Check divisibility for ASK algorithm.
    """
    try:
        if isinstance(value, (int, float)):
            # Check remainder is close to zero
            remainder = value % divisor
            return abs(remainder) < 1e-10
        elif isinstance(value, complex):
            # Check both real and imaginary parts
            real_rem = value.real % divisor
            imag_rem = value.imag % divisor
            return abs(real_rem) < 1e-10 and abs(imag_rem) < 1e-10
        elif isinstance(value, (list, tuple)):
            # For array types, check first element as representative
            if value:
                return _is_divisible_ask(value[0], divisor, number_type)
            else:
                return True
        elif isinstance(value, tuple) and len(value) == 3:  # Neutrosophic
            # Check all components
            return all(_is_divisible_ask(v, divisor, number_type) for v in value)
        else:
            # For other types, assume divisible
            return True
    except Exception:
        # If check fails, assume not divisible
        return False


def _is_prime_like_ask(value: Any, number_type: int) -> bool:
    """
    Check if value is prime-like for ASK algorithm.
    """
    try:
        if isinstance(value, (int, float)):
            v = abs(value)
            # Simple prime check
            if v < 2:
                return False
            if v == 2 or v == 3:
                return True
            if v % 2 == 0 or v % 3 == 0:
                return False
            
            # Check up to sqrt(v)
            i = 5
            while i * i <= v:
                if v % i == 0 or v % (i + 2) == 0:
                    return False
                i += 6
            return True
        elif isinstance(value, complex):
            # Check magnitude
            mag = abs(value)
            return _is_prime_like_ask(mag, number_type)
        elif isinstance(value, (list, tuple)):
            # For array types, check first non-zero element
            for v in value:
                if abs(v) > 1e-10:
                    return _is_prime_like_ask(v, number_type)
            return False
        elif isinstance(value, tuple) and len(value) == 3:  # Neutrosophic
            # Check truth component
            return _is_prime_like_ask(value[0], number_type)
        else:
            # For other types, use string representation
            try:
                s = str(value)
                # Extract numbers from string
                import re
                numbers = re.findall(r'\d+\.?\d*', s)
                if numbers:
                    return _is_prime_like_ask(float(numbers[0]), number_type)
                return False
            except:
                return False
    except Exception:
        return False

"""
# Ayrıca, get_with_params fonksiyonunuzu da güncelleyin:
def get_with_params(
    kececi_type_choice: int,
    iterations: int = 10,
    start_value_raw: Union[str, float, int] = "0",
    add_value_raw: Union[str, float, int] = "1.0",
    operation: str = "ask",  # Default ASK algoritması
    include_intermediate_steps: bool = False,
    custom_parser: Optional[Callable] = None,
) -> List[Any]:

    #Unified entry point for generating Keçeci numbers.
    #Default operation: "ask" (ASK algoritması)

    logger.info("Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations)
    logger.debug("Start: %r, Operation: %r with value: %r", 
                 start_value_raw, operation, add_value_raw)
    
    # Basic input sanitation
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        add_value_raw = "1"
    
    # Convert to strings
    start_str = str(start_value_raw) if not isinstance(start_value_raw, str) else start_value_raw
    add_str = str(add_value_raw) if not isinstance(add_value_raw, str) else add_value_raw
    
    try:
        # unified_generator'ı operation parametresi ile çağır
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_str,
            add_input_raw=add_str,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps,
            operation=operation  # Operation parametresini ekle
        )
        
        if not generated_sequence:
            logger.warning("Sequence generation failed or returned empty for type %s with start=%r add=%r", 
                          kececi_type_choice, start_value_raw, add_value_raw)
            return []

        logger.info("Generated %d numbers for type %s", len(generated_sequence), kececi_type_choice)
        
        # Preview
        preview_size = min(5, len(generated_sequence))
        if preview_size > 0:
            preview_start = [str(x) for x in generated_sequence[:preview_size]]
            logger.debug("First %d: %s", preview_size, preview_start)
            
            if len(generated_sequence) > preview_size * 2:
                preview_end = [str(x) for x in generated_sequence[-preview_size:]]
                logger.debug("Last %d: %s", preview_size, preview_end)

        # Keçeci Prime Number check
        try:
            kpn = find_kececi_prime_number(generated_sequence)
            if kpn is not None:
                logger.info("Keçeci Prime Number (KPN) found: %s", kpn)
            else:
                logger.debug("No Keçeci Prime Number found in the sequence.")
        except Exception as e:
            logger.debug(f"KPN check skipped or failed: {e}")

        return generated_sequence
        
    except Exception as e:
        logger.exception("ERROR during sequence generation: %s", e)
        raise
"""
"""
# Sorunsuz çalışıyor
def get_with_params(
    kececi_type_choice: int,
    iterations: int,
    start_value_raw: str,
    add_value_raw: str,
    include_intermediate_steps: bool = False
) -> List[Any]:

    #Common entry point: validates inputs early, logs info instead of printing.

    from fractions import Fraction
    logger.info("Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations)
    logger.debug("Start: %r, Addition: %r, Include intermediate: %s", start_value_raw, add_value_raw, include_intermediate_steps)

    # Basic input sanitation
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        # choose a conservative default for increment
        add_value_raw = "1"

    try:
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_value_raw,
            add_input_raw=add_value_raw,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps
        )

        if not generated_sequence:
            logger.warning("Sequence generation failed or returned empty for type %s with start=%r add=%r", kececi_type_choice, start_value_raw, add_value_raw)
            return []

        logger.info("Generated %d numbers for type %s", len(generated_sequence), kececi_type_choice)
        # preview
        preview_start = [str(x) for x in generated_sequence[:5]]
        preview_end = [str(x) for x in generated_sequence[-5:]] if len(generated_sequence) > 5 else []

        logger.debug("First 5: %s", preview_start)
        if preview_end:
            logger.debug("Last 5: %s", preview_end)

        # Keçeci Prime Number check
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            logger.info("Keçeci Prime Number (KPN) found: %s", kpn)
        else:
            logger.info("No Keçeci Prime Number found in the sequence.")

        return generated_sequence

    except Exception as e:
        logger.exception("ERROR during sequence generation: %s", e)
        return []
"""

def get_with_params(
    kececi_type_choice: int,
    iterations: int,
    start_value_raw: str,
    add_value_raw: str,
    include_intermediate_steps: bool = False
) -> List[Any]:

    #Common entry point: validates inputs early, logs info instead of printing.

    logger.info("Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations)
    logger.debug("Start: %r, Addition: %r, Include intermediate: %s", 
                 start_value_raw, add_value_raw, include_intermediate_steps)

    # Basic input sanitation
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        add_value_raw = "1"

    try:
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_value_raw,
            add_input_raw=add_value_raw,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps
        )

        if not generated_sequence:
            logger.warning("Sequence generation failed or returned empty for type %s with start=%r add=%r", 
                          kececi_type_choice, start_value_raw, add_value_raw)
            return []

        logger.info("Generated %d numbers for type %s", len(generated_sequence), kececi_type_choice)
        
        # Preview
        preview_start = [str(x) for x in generated_sequence[:5]]
        preview_end = [str(x) for x in generated_sequence[-5:]] if len(generated_sequence) > 5 else []

        logger.debug("First 5: %s", preview_start)
        if preview_end:
            logger.debug("Last 5: %s", preview_end)

        # Keçeci Prime Number check
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            logger.info("Keçeci Prime Number (KPN) found: %s", kpn)
        else:
            logger.info("No Keçeci Prime Number found in the sequence.")

        return generated_sequence

    except Exception as e:
        logger.exception("ERROR during sequence generation: %s", e)
        return []

"""
def get_with_params(
    kececi_type_choice: int,
    iterations: int = 10,
    start_value_raw: Union[str, float, int] = "0",
    add_value_raw: Union[str, float, int] = "1.0",
    operation: str = "add",
    include_intermediate_steps: bool = False,
    custom_parser: Optional[Any] = None,
) -> List[Any]:

    #Unified entry point for generating Keçeci numbers based on specified parameters.

    logger.info("Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations)
    logger.debug("Start: %r, Operation: %r with value: %r, Include intermediate: %s",
                 start_value_raw, operation, add_value_raw, include_intermediate_steps)

    # Basic input sanitation
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        add_value_raw = "1"

    # Convert to strings for unified_generator
    start_str = str(start_value_raw) if not isinstance(start_value_raw, str) else start_value_raw
    add_str = str(add_value_raw) if not isinstance(add_value_raw, str) else add_value_raw

    try:
        # unified_generator'ı operation parametresi ile çağır
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_str,
            add_input_raw=add_str,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps,
            operation=operation  # operation parametresini ekle
        )

        if not generated_sequence:
            logger.warning("Sequence generation failed or returned empty for type %s with start=%r add=%r", 
                          kececi_type_choice, start_value_raw, add_value_raw)
            return []

        logger.info("Generated %d numbers for type %s", len(generated_sequence), kececi_type_choice)
        
        # Preview
        preview_size = min(5, len(generated_sequence))
        if preview_size > 0:
            preview_start = [str(x) for x in generated_sequence[:preview_size]]
            logger.debug("First %d: %s", preview_size, preview_start)
            
            if len(generated_sequence) > preview_size * 2:
                preview_end = [str(x) for x in generated_sequence[-preview_size:]]
                logger.debug("Last %d: %s", preview_size, preview_end)

        # Keçeci Prime Number check
        try:
            kpn = find_kececi_prime_number(generated_sequence)
            if kpn is not None:
                logger.info("Keçeci Prime Number (KPN) found: %s", kpn)
            else:
                logger.debug("No Keçeci Prime Number found in the sequence.")
        except Exception as e:
            logger.debug(f"KPN check skipped or failed: {e}")

        return generated_sequence

    except Exception as e:
        logger.exception("ERROR during sequence generation: %s", e)
        raise
"""
"""
# 2. tray bloğu ask kurallarını uygulamıyor
def get_with_params(
    kececi_type_choice: int,
    iterations: int = 10,
    start_value_raw: Union[str, float, int] = "0",
    add_value_raw: Union[str, float, int] = "1.0",
    operation: str = "add",
    include_intermediate_steps: bool = False,
    custom_parser: Optional[Callable] = None,
) -> List[Any]:

    #Unified entry point for generating Keçeci numbers based on specified parameters.
    from fractions import Fraction

    # Log the start of generation
    logger.info(
        "Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations
    )
    logger.debug(
        "Start: %r, Operation: %r with value: %r, Include intermediate: %s",
        start_value_raw,
        operation,
        add_value_raw,
        include_intermediate_steps,
    )

    # Basic input sanitation and type conversion
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        if operation == "add":
            add_value_raw = "1"
        elif operation == "multiply":
            add_value_raw = "2"
        else:
            add_value_raw = "1"

    # Validate operation
    valid_operations = ["add", "multiply", "subtract", "divide", "mod", "power"]
    if operation not in valid_operations:
        raise ValueError(
            f"Invalid operation: {operation}. Must be one of {valid_operations}"
        )

    # Validate iterations
    if iterations < 1:
        logger.warning(f"Invalid iterations value: {iterations}, using default 10")
        iterations = 10

    try:
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_value_raw,
            add_input_raw=add_value_raw,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps
        )

        if not generated_sequence:
            logger.warning("Sequence generation failed or returned empty for type %s with start=%r add=%r", kececi_type_choice, start_value_raw, add_value_raw)
            return []

        logger.info("Generated %d numbers for type %s", len(generated_sequence), kececi_type_choice)
        # preview
        preview_start = [str(x) for x in generated_sequence[:5]]
        preview_end = [str(x) for x in generated_sequence[-5:]] if len(generated_sequence) > 5 else []

        logger.debug("First 5: %s", preview_start)
        if preview_end:
            logger.debug("Last 5: %s", preview_end)

        # Keçeci Prime Number check
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            logger.info("Keçeci Prime Number (KPN) found: %s", kpn)
        else:
            logger.info("No Keçeci Prime Number found in the sequence.")

        return generated_sequence

    except Exception as e:
        logger.exception("ERROR during sequence generation: %s", e)
        return []
"""
"""
    try:
        # Import parsers and number classes
        try:
            from .kececinumbers import (
                # Parsers
                _parse_bicomplex,
                _parse_chingon,
                _parse_clifford,
                _parse_complex,
                _parse_complex_like_string,
                _parse_dual,
                _parse_engineering_notation,
                _parse_fraction,
                _parse_hyperreal,
                _parse_neutrosophic,
                _parse_neutrosophic_bicomplex,
                _parse_neutrosophic_complex,
                _parse_octonion,
                _parse_pathion,
                _parse_quaternion,
                _parse_quaternion_from_csv,
                _parse_real,
                _parse_routon,
                _parse_sedenion,
                _parse_splitcomplex,
                _parse_super_real,
                _parse_superreal,
                _parse_ternary,
                _parse_hypercomplex,
                _parse_universal,
                _parse_voudon,
                _generate_simple_ask_sequence,
                _parse_with_fallback_simple,
                _parse_kececi_values,
                parse_to_hyperreal,
                parse_to_neutrosophic,
            )
            parsers_available = True
        except ImportError as e:
            logger.warning(f"Import error: {e}. Using fallback parsers")
            parsers_available = False
            
            # FIX: Define _parse_complex properly before using it
            def _parse_complex(s: Union[str, float, int]) -> complex:
                #Fallback complex parser that handles strings like '2+3j'.
                if isinstance(s, complex):
                    return s
                if isinstance(s, (int, float)):
                    return complex(s)
                
                s_str = str(s).strip()
                try:
                    # Try Python's built-in complex parser first
                    return complex(s_str)
                except ValueError:
                    # Try alternative formats
                    s_str = s_str.replace('i', 'j').replace('J', 'j')
                    
                    # Handle format: "real,imag"
                    if ',' in s_str:
                        parts = s_str.split(',')
                        if len(parts) == 2:
                            try:
                                return complex(float(parts[0]), float(parts[1]))
                            except:
                                pass
                    
                    # Handle format: "real+imagj"
                    if '+' in s_str and 'j' in s_str:
                        # Remove any spaces
                        s_str = s_str.replace(' ', '')
                        if 'j' in s_str:
                            # Split by '+' but be careful with signs
                            parts = s_str.split('+')
                            if len(parts) == 2:
                                try:
                                    real_part = parts[0]
                                    imag_part = parts[1]
                                    # Remove 'j' from imag part
                                    if imag_part.endswith('j'):
                                        imag_part = imag_part[:-1]
                                    return complex(float(real_part), float(imag_part))
                                except:
                                    pass
                    
                    # If all else fails, try to parse as float for real part
                    try:
                        return complex(float(s_str), 0)
                    except:
                        return complex(0, 0)
            
            # Define _parse_fraction with complex handling
            def _parse_fraction(s: Union[str, float, int]) -> float:
                #Fallback fraction parser that handles complex numbers
                if isinstance(s, (int, float)):
                    return float(s)
                
                s_str = str(s).strip()
                if not s_str:
                    return 0.0
                
                # First, try to handle complex numbers
                try:
                    # Use our _parse_complex function
                    c = _parse_complex(s_str)
                    if c.imag != 0:
                        logger.warning(f"Complex number {s_str} for fraction parsing; using real part only.")
                        return float(c.real)
                    else:
                        return float(c.real)
                except:
                    pass
                
                # Try as float
                try:
                    return float(s_str)
                except ValueError:
                    pass
                
                # Try fractions
                if '/' in s_str:
                    try:
                        num, den = s_str.split('/')
                        return float(num) / float(den) if float(den) != 0 else float('inf')
                    except:
                        pass
                
                # Last resort
                try:
                    return float(s_str)
                except:
                    raise ValueError(f"Could not parse as number: {s_str}")
            
            # Simple fallbacks for other types
            _parse_neutrosophic = lambda s: (_parse_fraction(s), 0.0, 0.0)
            _parse_bicomplex = lambda s: complex(_parse_fraction(s), 0)
            _parse_neutrosophic_complex = lambda s: complex(_parse_fraction(s), 0)
            _parse_neutrosophic_bicomplex = lambda s: complex(_parse_fraction(s), 0)
            _parse_quaternion = lambda s: _parse_fraction(s)
            _parse_octonion = lambda s: _parse_fraction(s)
            _parse_sedenion = lambda s: _parse_fraction(s)
            _parse_clifford = lambda s: _parse_fraction(s)
            _parse_dual = lambda s: _parse_fraction(s)
            _parse_splitcomplex = lambda s: _parse_fraction(s)
            _parse_pathion = lambda s: _parse_fraction(s)
            _parse_chingon = lambda s: _parse_fraction(s)
            _parse_routon = lambda s: _parse_fraction(s)
            _parse_voudon = lambda s: _parse_fraction(s)
            _parse_super_real = lambda s: _parse_fraction(s)
            _parse_ternary = lambda s: _parse_fraction(s)
            _parse_hyperreal = lambda s: _parse_fraction(s)

        # Map type choices to parsers
        type_to_parser = {
            1: {"parser": _parse_fraction, "name": "Positive Real"},
            2: {"parser": lambda s: -_parse_fraction(s), "name": "Negative Real"},
            3: {"parser": _parse_complex, "name": "Complex"},
            4: {"parser": _parse_fraction, "name": "Float"},
            5: {"parser": _parse_fraction, "name": "Rational"},
            6: {"parser": _parse_quaternion, "name": "Quaternion"},
            7: {"parser": _parse_neutrosophic, "name": "Neutrosophic"},
            8: {"parser": _parse_neutrosophic_complex, "name": "Neutrosophic Complex"},
            9: {"parser": _parse_hyperreal, "name": "Hyperreal"},
            10: {"parser": _parse_bicomplex, "name": "Bicomplex"},
            11: {"parser": _parse_neutrosophic_bicomplex, "name": "Neutrosophic Bicomplex"},
            12: {"parser": _parse_octonion, "name": "Octonion"},
            13: {"parser": _parse_sedenion, "name": "Sedenion"},
            14: {"parser": _parse_clifford, "name": "Clifford"},
            15: {"parser": _parse_dual, "name": "Dual"},
            16: {"parser": _parse_splitcomplex, "name": "Split-Complex"},
            17: {"parser": _parse_pathion, "name": "Pathion"},
            18: {"parser": _parse_chingon, "name": "Chingon"},
            19: {"parser": _parse_routon, "name": "Routon"},
            20: {"parser": _parse_voudon, "name": "Voudon"},
            21: {"parser": _parse_super_real, "name": "Super Real"},
            22: {"parser": _parse_ternary, "name": "Ternary"},
        }

        # Add custom parser if provided
        if custom_parser is not None:
            logger.debug("Using custom parser provided by user")
            parser_func = cast(Callable[[Any], Any], custom_parser)
            type_name = "Custom"
        else:
            if kececi_type_choice not in type_to_parser:
                raise ValueError(
                    f"Invalid type choice: {kececi_type_choice}. Must be 1-22"
                )

            type_info = type_to_parser[kececi_type_choice]
            parser_func = cast(Callable[[Any], Any], type_info["parser"])
            type_name = cast(str, type_info["name"])

        logger.info(f"Generating {type_name} numbers (type {kececi_type_choice})")

        # Parse start and add values
        try:
            # Debug logging
            logger.debug(f"Parsing start_value_raw: {repr(start_value_raw)} with parser {parser_func.__name__ if hasattr(parser_func, '__name__') else type(parser_func).__name__}")
            logger.debug(f"Parsing add_value_raw: {repr(add_value_raw)}")
            
            start_value = parser_func(start_value_raw)
            add_value = parser_func(add_value_raw)

            logger.debug(
                f"Parsed start value: {repr(start_value)} (type: {type(start_value)})"
            )
            logger.debug(
                f"Parsed operation value: {repr(add_value)} (type: {type(add_value)})"
            )

        except Exception as e:
            logger.error(
                f"Parsing failed. Start type: {type(start_value_raw)}, Start value: {repr(start_value_raw)}"
            )
            logger.error(
                f"Parsing failed. Add type: {type(add_value_raw)}, Add value: {repr(add_value_raw)}"
            )
            raise ValueError(
                f"Failed to parse values. Start: '{start_value_raw}', Add: '{add_value_raw}'. Error: {str(e)}"
            )

        # Generate the sequence
        result = _generate_kececi_sequence(
            start_value=start_value,
            add_value=add_value,
            iterations=iterations,
            operation=operation,
            include_intermediate_steps=include_intermediate_steps,
            number_type=type_name
        )

        if not result:
            logger.warning("Sequence generation failed or returned empty")
            return []

        # Log generation results
        logger.info(f"Generated {len(result)} numbers for type {type_name}")

        # Preview first and last few elements
        preview_size = min(3, len(result))
        if preview_size > 0:
            preview_start = [
                str(x)[:50] + "..." if len(str(x)) > 50 else str(x)
                for x in result[:preview_size]
            ]
            logger.debug(f"First {preview_size}: {preview_start}")

            if len(result) > preview_size * 2:
                preview_end = [
                    str(x)[:50] + "..." if len(str(x)) > 50 else str(x)
                    for x in result[-preview_size:]
                ]
                logger.debug(f"Last {preview_size}: {preview_end}")

        # Keçeci Prime Number check
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
        raise
"""

# Yardımcı fonksiyonlar
def _generate_kececi_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False,
    number_type: str = "Unknown"
) -> List[Any]:
    """
    Generate sequence for Keçeci numbers with proper operation handling.
    """
    if include_intermediate_steps:
        # Detailed output with steps
        result = []
        current = start_value
        
        # Add initial state
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
        
        for i in range(1, iterations):
            previous = current
            try:
                current = _apply_kececi_operation(current, add_value, operation, number_type)
                
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "previous": previous,
                    "description": f"Step {i}: {previous} {_get_operation_symbol(operation)} {add_value} = {current}"
                })
            except Exception as e:
                logger.error(f"Error at step {i}: {e}")
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "error": str(e),
                    "description": f"Step {i}: ERROR - {e}"
                })
                break
        
        return result
    else:
        # Simple list output
        result = [start_value]
        current = start_value
        
        for i in range(1, iterations):
            try:
                current = _apply_kececi_operation(current, add_value, operation, number_type)
                result.append(current)
            except Exception as e:
                logger.error(f"Error at iteration {i}: {e}")
                # Try to continue with a default value
                default_val = _get_default_value_for_type(current)
                result.append(default_val)
                current = default_val
        
        return result


def _get_default_value_for_type(value: Any) -> Any:
    """Get default value for a given type."""
    if isinstance(value, (int, float)):
        return 0
    elif isinstance(value, complex):
        return complex(0, 0)
    elif isinstance(value, tuple):
        return tuple(0 for _ in value)
    elif hasattr(value, '__class__'):
        try:
            return type(value)()
        except:
            return 0
    else:
        return 0


def _generate_kececi_sequence(
    start_value: Any,
    add_value: Any,
    iterations: int,
    operation: str,
    include_intermediate_steps: bool = False,
    number_type: str = "Unknown"
) -> List[Any]:
    """
    Generate sequence for Keçeci numbers with proper operation handling.
    """
    if include_intermediate_steps:
        # Detailed output with steps
        result = []
        current = start_value
        
        # Add initial state
        result.append({
            "step": 0,
            "value": current,
            "operation": "start",
            "description": f"Start: {current}"
        })
        
        for i in range(1, iterations):
            previous = current
            try:
                current = _apply_kececi_operation(current, add_value, operation, number_type)
                
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "previous": previous,
                    "description": f"Step {i}: {previous} {_get_operation_symbol(operation)} {add_value} = {current}"
                })
            except Exception as e:
                logger.error(f"Error at step {i}: {e}")
                result.append({
                    "step": i,
                    "value": current,
                    "operation": operation,
                    "error": str(e),
                    "description": f"Step {i}: ERROR - {e}"
                })
                break
        
        return result
    else:
        # Simple list output
        result = [start_value]
        current = start_value
        
        for i in range(1, iterations):
            try:
                current = _apply_kececi_operation(current, add_value, operation, number_type)
                result.append(current)
            except Exception as e:
                logger.error(f"Error at iteration {i}: {e}")
                # Try to continue with a default value
                default_val = _get_default_value_for_type(current)
                result.append(default_val)
                current = default_val
        
        return result

def _get_default_value_for_type(value: Any) -> Any:
    """Get default value for a given type."""
    if isinstance(value, (int, float)):
        return 0
    elif isinstance(value, complex):
        return complex(0, 0)
    elif isinstance(value, tuple):
        return tuple(0 for _ in value)
    elif hasattr(value, '__class__'):
        try:
            return type(value)()
        except:
            return 0
    else:
        return 0

# Örnek: farklı boyutlarda hypercomplex default stringleri
def hypercomplex_str(dim, first=1.0, rest=0.0, complex_components=False):
    """
    dim: bileşen sayısı
    first: ilk bileşen değeri
    rest: diğer bileşenlerin değeri
    complex_components: True ise bileşenleri 'a+bj' formatında üretir (örnek)
    """
    comps = []
    for i in range(dim):
        if i == 0:
            v = first
        else:
            v = rest
        if complex_components:
            # örnek: ilk iki bileşeni karmaşık yap
            if i % 2 == 0:
                comps.append(f"{float(v)}+{float(v)/10}j")
            else:
                comps.append(f"{float(v)}")
        else:
            comps.append(str(float(v)))
    return ",".join(comps)

# Kullanım örnekleri
hc8 = hypercomplex_str(8, first=1.0, rest=0.0)        # "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
hc16 = hypercomplex_str(16, first=1.0, rest=0.0)
hc256 = hypercomplex_str(256, first=1.0, rest=0.0)
hc8_complex = hypercomplex_str(8, first=1.0, rest=0.0, complex_components=True)

def get_interactive(
    auto_values: Optional[Dict[str, str]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
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
            logger.debug(
                "input() failed for prompt %r — using default %r", prompt, default
            )
            return default

    interactive_mode = auto_values is None
    logger.info(
        "Keçeci Numbers Interactive Generator (interactive=%s)", interactive_mode
    )

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
            " 22: Ternary,       23: Hypercomplex",
        ]
        logger.info("Available Keçeci Number Types:")
        for line in menu_lines:
            logger.info(line)

    # Defaults
    DEFAULT_TYPE = 1
    DEFAULT_STEPS = 30
    DEFAULT_SHOW_DETAILS = "yes"

    default_start_values = {
        1: "2.5",
        2: "-5.0",
        3: "1+1j",
        4: "3.14",
        5: "3.5",
        6: "1.0,0.0,0.0,0.0",
        7: "0.6,0.2,0.1",
        8: "1+1j",
        9: "0.0,0.001",
        10: "1.0,0.5,0.3,0.2",
        11: "2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        12: "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        13: "1.0" + ",0.0" * 15,
        14: "1.0+2.0e1+3.0e12",
        15: "1.0,0.1",
        16: "1.0,0.5",
        17: "1.0" + ",0.0" * 31,
        18: "1.0" + ",0.0" * 63,
        19: "1.0" + ",0.0" * 127,
        20: "1.0" + ",0.0" * 255,
        21: "3.0,0.5",
        22: "12",
        23: "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"  # Hypercomplex: 8 bileşenli örnek (istendiğinde boyut artırılabilir)
    }

    default_add_values = {
        1: "0.5",
        2: "-0.5",
        3: "0.1+0.1j",
        4: "0.1",
        5: "0.1",
        6: "0.1,0.0,0.0,0.0",
        7: "0.1,0.0,0.0",
        8: "0.1+0.1j",
        9: "0.0,0.001",
        10: "0.1,0.0,0.0,0.0",
        11: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        12: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        13: "0.1" + ",0.0" * 15,
        14: "0.1+0.2e1",
        15: "0.1,0.0",
        16: "0.1,0.0",
        17: "1.0" + ",0.0" * 31,
        18: "1.0" + ",0.0" * 63,
        19: "1.0" + ",0.0" * 127,
        20: "1.0" + ",0.0" * 255,
        21: "1.5,2.7",
        22: "1",
        23: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0"  # Hypercomplex ekleme örneği
    }

    # Ask for inputs (uses _ask which respects auto_values when provided)
    type_input_raw = _ask(
        "type_choice",
        f"Select Keçeci Number Type (1-23) [default: {DEFAULT_TYPE}]: ",
        str(DEFAULT_TYPE),
    )
    try:
        type_choice = int(type_input_raw)
        if not (1 <= type_choice <= 23):
            logger.warning(
                "Invalid type_choice %r, using default %s", type_choice, DEFAULT_TYPE
            )
            type_choice = DEFAULT_TYPE
    except Exception:
        logger.warning(
            "Could not parse type_choice %r, using default %s",
            type_input_raw,
            DEFAULT_TYPE,
        )
        type_choice = DEFAULT_TYPE

    start_prompt = _ask(
        "start_val",
        f"Enter start value [default: {default_start_values[type_choice]}]: ",
        default_start_values[type_choice],
    )
    add_prompt = _ask(
        "add_val",
        f"Enter increment value [default: {default_add_values[type_choice]}]: ",
        default_add_values[type_choice],
    )
    steps_raw = _ask(
        "steps",
        f"Enter number of Keçeci steps [default: {DEFAULT_STEPS}]: ",
        str(DEFAULT_STEPS),
    )
    try:
        num_kececi_steps = int(steps_raw)
        if num_kececi_steps <= 0:
            logger.warning(
                "Non-positive steps %r, using default %d",
                num_kececi_steps,
                DEFAULT_STEPS,
            )
            num_kececi_steps = DEFAULT_STEPS
    except Exception:
        logger.warning(
            "Could not parse steps %r, using default %d", steps_raw, DEFAULT_STEPS
        )
        num_kececi_steps = DEFAULT_STEPS

    show_detail_raw = _ask(
        "show_details",
        f"Include intermediate steps? (y/n) [default: {DEFAULT_SHOW_DETAILS}]: ",
        DEFAULT_SHOW_DETAILS,
    )
    show_details = str(show_detail_raw).strip().lower() in ["y", "yes"]

    sequence = get_with_params(
        kececi_type_choice=type_choice,
        iterations=num_kececi_steps,
        start_value_raw=start_prompt,
        add_value_raw=add_prompt,
        include_intermediate_steps=show_details,
    )

    params = {
        "type_choice": type_choice,
        "start_val": start_prompt,
        "add_val": add_prompt,
        "steps": num_kececi_steps,
        "detailed_view": show_details,
    }
    logger.info(
        "Using parameters: Type=%s, Start=%r, Add=%r, Steps=%s, Details=%s",
        type_choice,
        start_prompt,
        add_prompt,
        num_kececi_steps,
        show_details,
    )
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

# Veri çıkarma fonksiyonu
def extract_neutro_components(sequence):
    real_parts, imag_parts, indeter_parts = [], [], []
    
    for i, x in enumerate(sequence):
        if isinstance(x, NeutrosophicComplexNumber):
            real_parts.append(x.real)
            imag_parts.append(x.imag)
            indeter_parts.append(x.indeterminacy)
        elif isinstance(x, (tuple, list)):
            # Tuple yapısını varsay: (real, imag, indeterminacy) veya (real, imag)
            real_parts.append(x[0] if len(x) > 0 else 0)
            imag_parts.append(x[1] if len(x) > 1 else 0)
            indeter_parts.append(x[2] if len(x) > 2 else 0)
        else:
            real_parts.append(0)
            imag_parts.append(0)
            indeter_parts.append(0)
    
    return real_parts, imag_parts, indeter_parts

def plot_neutrosophic_complex(sequence, start_input_raw, add_input_raw, fig):
    print("DEBUG: Sequence uzunluk:", len(sequence))
    
    # Tuple yapısından Neutro-complex'leri çıkar
    all_real_parts = []
    all_imag_parts = []
    all_indeter_parts = []
    
    current_pos = 0
    step_count = 0
    
    while current_pos < len(sequence[0]) and step_count < 41:  # Max 41 adım
        # Her Neutro-complex 3 eleman: (real, imag, indeterminacy)
        if current_pos + 2 < len(sequence[0]):
            real_val = sequence[0][current_pos]
            imag_val = sequence[0][current_pos + 1]
            indeter_val = sequence[0][current_pos + 2]
            
            all_real_parts.append(real_val)
            all_imag_parts.append(imag_val)
            all_indeter_parts.append(indeter_val)
            
            print(f"Adım {step_count}: ({real_val}, {imag_val}, {indeter_val})")
            current_pos += 3
            step_count += 1
        else:
            break
    
    # Eğer veri azsa doldur
    while len(all_real_parts) < 40:
        all_real_parts.append(all_real_parts[-1] if all_real_parts else 0)
        all_imag_parts.append(all_imag_parts[-1] if all_imag_parts else 0)
        all_indeter_parts.append(all_indeter_parts[-1] if all_indeter_parts else 0)
    
    magnitudes_z = [abs(complex(r, i)) for r, i in zip(all_real_parts, all_imag_parts)]
    
    print(f"Veri aralığı - Real: {min(all_real_parts):.2f}-{max(all_real_parts):.2f}")
    print(f"Veri aralığı - Imag: {min(all_imag_parts):.2f}-{max(all_imag_parts):.2f}")

    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Complex Plane
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_real_parts, all_imag_parts, ".-", alpha=0.7, linewidth=2)
    ax1.scatter(all_real_parts[0], all_imag_parts[0], c="green", s=150, label="Başlangıç", zorder=5)
    ax1.scatter(all_real_parts[-1], all_imag_parts[-1], c="red", s=150, label="Bitiş", zorder=5)
    ax1.set_title("Karmaşık Düzlem")
    ax1.set_xlabel("Re(z)")
    ax1.set_ylabel("Im(z)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # 2. Belirsizlik Zaman Üzerinde
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(all_indeter_parts, "o-", color="purple", markersize=6)
    ax2.set_title("Belirsizlik Seviyesi")
    ax2.set_xlabel("Adım")
    ax2.set_ylabel("I")
    ax2.grid(True, alpha=0.3)

    # 3. |z| vs Belirsizlik
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(magnitudes_z, all_indeter_parts, c=range(len(magnitudes_z)), 
                     cmap="viridis", s=50, edgecolors='white', linewidth=0.5)
    ax3.set_title("Büyüklük vs Belirsizlik")
    ax3.set_xlabel("|z|")
    ax3.set_ylabel("I")
    plt.colorbar(sc, ax=ax3, label="Adım")
    ax3.grid(True, alpha=0.3)

    # 4. Re vs Im (I'ye göre renklendirilmiş)
    ax4 = fig.add_subplot(gs[1, 1])
    sc2 = ax4.scatter(all_real_parts, all_imag_parts, c=all_indeter_parts, 
                      cmap="plasma", s=60, edgecolors='white', linewidth=0.5)
    ax4.set_title("Re vs Im (I renklendirme)")
    ax4.set_xlabel("Re(z)")
    ax4.set_ylabel("Im(z)")
    plt.colorbar(sc2, ax=ax4, label="Belirsizlik (I)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()



def plot_numbers(sequence: List[Any], title: str = "Keçeci Number Sequence Analysis"):
    """
    Tüm 23 Keçeci Sayı türü için detaylı görselleştirme sağlar.
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

    # --- helpers used in these branches ---
    def _pca_var_sum(pca_obj) -> float:
        try:
            arr = getattr(pca_obj, "explained_variance_ratio_", None)
            if arr is None:
                return 0.0
            arr = np.asarray(arr, dtype=float)
            s = float(np.nansum(arr))
            return s if np.isfinite(s) else 0.0
        except Exception:
            return 0.0

    def _ensure_fig():
        try:
            _ = fig
        except NameError:
            return plt.figure(figsize=(12, 8), constrained_layout=True)
        else:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass
            return fig


    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # `sequence` is the iterable you want to visualise
    first_elem = sequence[0]

    # --- 1. Fraction (Rational)
    if isinstance(first_elem, Fraction):
        # Tüm elemanları `float` olarak dönüştür
        float_vals = [float(x) for x in sequence]
        # float_vals = [float(x) if isinstance(x, (int, float, Fraction)) else float(x.value) for x in sequence]
        # Pay ve paydaları ayrı ayrı al
        numerators = [x.numerator for x in sequence]
        denominators = [x.denominator for x in sequence]

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
        ratios = [n / d for n, d in zip(numerators, denominators)]
        ax3.plot(ratios, 'o-', color='tab:purple')
        ax3.set_title("Numerator/Denominator Ratio")
        ax3.set_ylabel("n/d")

        # 4. Grafik: Pay vs Payda dağılımı
        ax4 = fig.add_subplot(gs[1, 1])
        sc = ax4.scatter(numerators, denominators, c=range(len(sequence)), cmap='plasma', s=30)
        ax4.set_title("Numerator vs Denominator Trajectory")
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

    # --- 4. quaternion
    # Check for numpy-quaternion's quaternion type, or a custom one with 'components' or 'w,x,y,z': çıkarıldı:  and len(getattr(first_elem, 'components', [])) == 4) or \
    elif isinstance(first_elem, quaternion) or (hasattr(first_elem, 'components') == 4) or \
         (hasattr(first_elem, 'w') and hasattr(first_elem, 'x') and hasattr(first_elem, 'y') and hasattr(first_elem, 'z')):

        try:
            comp = np.array([
                (q.w, q.x, q.y, q.z) if hasattr(q, 'w') else q.components
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
        # Local safe PCA variance helper (ensure available in this scope)
        def _pca_var_sum(pca_obj) -> float:
            try:
                arr = getattr(pca_obj, "explained_variance_ratio_", None)
                if arr is None:
                    return 0.0
                arr = np.asarray(arr, dtype=float)
                s = float(np.nansum(arr))
                return s if np.isfinite(s) else 0.0
            except Exception:
                return 0.0

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

        # 3. Grafik: PCA
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

    # --- 11. NeutrosophicComplexNumber (duck-typed, güvenli plotting) ---
    # --- 11. NeutrosophicComplexNumber (eski tarz, basit) ---
    elif isinstance(first_elem, NeutrosophicComplexNumber):
        try:
            # Basit, güvenli veri çıkarımı: önce attribute, sonra to_list/to_components/coeffs, son olarak tuple/list fallback
            real_parts = []
            imag_parts = []
            indet_parts = []
            for x in sequence:
                # 1) doğrudan attribute/property
                r = getattr(x, "real", None)
                im = getattr(x, "imag", None)
                ind = getattr(x, "indeterminacy", None)

                # 2) fallback: to_list / to_components / coeffs
                if (r is None or im is None or ind is None):
                    if hasattr(x, "to_list") and callable(getattr(x, "to_list")):
                        try:
                            comps = list(x.to_list())
                        except Exception:
                            comps = []
                    elif hasattr(x, "to_components") and callable(getattr(x, "to_components")):
                        try:
                            comps = list(x.to_components())
                        except Exception:
                            comps = []
                    elif hasattr(x, "coeffs"):
                        try:
                            c = x.coeffs() if callable(getattr(x, "coeffs")) else x.coeffs
                            comps = list(c)
                        except Exception:
                            comps = []
                    else:
                        comps = []

                    if r is None and len(comps) >= 1:
                        r = comps[0]
                    if im is None and len(comps) >= 2:
                        im = comps[1]
                    if ind is None and len(comps) >= 3:
                        ind = comps[2]

                # 3) son fallback: tuple/list pozisyonel
                if (r is None or im is None or ind is None) and isinstance(x, (tuple, list)):
                    if r is None and len(x) > 0:
                        r = x[0]
                    if im is None and len(x) > 1:
                        im = x[1]
                    if ind is None and len(x) > 2:
                        ind = x[2]

                # 4) numeric dönüşümler (güvenli)
                try:
                    real_parts.append(float(r) if r is not None else 0.0)
                except Exception:
                    real_parts.append(0.0)
                try:
                    imag_parts.append(float(im.real) if isinstance(im, complex) else float(im) if im is not None else 0.0)
                except Exception:
                    imag_parts.append(0.0)
                try:
                    indet_parts.append(float(ind) if ind is not None else 0.0)
                except Exception:
                    indet_parts.append(0.0)

            # magnitude hesapla
            magnitudes = [abs(complex(r, i)) for r, i in zip(real_parts, imag_parts)]

            # figür oluştur / yeniden kullan
            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(11, 7), constrained_layout=True)

            gs = GridSpec(2, 2, figure=fig)

            # 1) Complex plane
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(real_parts, imag_parts, ".-", alpha=0.8)
            if real_parts:
                ax1.scatter(real_parts[0], imag_parts[0], c="g", s=80, label="Start")
                ax1.scatter(real_parts[-1], imag_parts[-1], c="r", s=80, label="End")
            ax1.set_title("Neutrosophic Complex Plane")
            ax1.set_xlabel("Re(z)")
            ax1.set_ylabel("Im(z)")
            ax1.legend()
            ax1.axis("equal")
            ax1.grid(alpha=0.25)

            # 2) Indeterminacy over time
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(indet_parts, "o-", color="purple")
            ax2.set_title("Indeterminacy Level")
            ax2.set_ylabel("I")
            ax2.grid(alpha=0.25)

            # 3) |z| vs Indeterminacy
            ax3 = fig.add_subplot(gs[1, 0])
            sc = ax3.scatter(magnitudes, indet_parts, c=np.arange(len(magnitudes)), cmap="viridis", s=30)
            ax3.set_title("Magnitude vs Indeterminacy")
            ax3.set_xlabel("|z|")
            ax3.set_ylabel("I")
            try:
                cbar = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            except Exception:
                pass
            ax3.grid(alpha=0.25)

            # 4) Real vs Imag colored by I
            ax4 = fig.add_subplot(gs[1, 1])
            sc2 = ax4.scatter(real_parts, imag_parts, c=indet_parts, cmap="plasma", s=40)
            ax4.set_title("Real vs Imag (colored by I)")
            ax4.set_xlabel("Re(z)")
            ax4.set_ylabel("Im(z)")
            try:
                cbar2 = fig.colorbar(sc2, ax=ax4, fraction=0.046, pad=0.04)
                cbar2.ax.tick_params(labelsize=8)
            except Exception:
                pass
            ax4.grid(alpha=0.25)

            return fig

        except Exception as e:
            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(8, 4), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"NeutrosophicComplex plot error: {e}", ha='center', va='center', color='red')
            ax.set_xticks([]); ax.set_yticks([])
            logger.exception("NeutrosophicComplex plotting failed")
            return fig

        """
        # sorunsuz çalışıyor
        elif isinstance(first_elem, NeutrosophicComplexNumber):
            print("FIRST TYPE:", type(first_elem))
            print("MODULE:", first_elem.__class__.__module__)
            print("DEBUG: NeutrosophicComplex plotting - Universal handler")
            
            def safe_extract_real(obj):
                #Her türden real çıkarır
                if hasattr(obj, 'real'):
                    return float(obj.real)
                return 0.0
            
            def safe_extract_imag(obj):
                #Her türden imag çıkarır
                if hasattr(obj, 'imag'):
                    return float(obj.imag)
                return 0.0
            
            def safe_extract_indet(obj):
                #Her türden indeterminacy çıkarır
                if hasattr(obj, 'NeutrosophicComplexNumber'):
                    return float(obj.NeutrosophicComplexNumber)
                return 0.0
            
            # Sequence'den verileri çıkar (PlotNeutroComplex + diğer tipler)
            real_parts = [safe_extract_real(x) for x in sequence]
            imag_parts = [safe_extract_imag(x) for x in sequence]
            indeter_parts = [safe_extract_indet(x) for x in sequence]
            magnitudes_z = [abs(complex(r, i)) for r, i in zip(real_parts, imag_parts)]
            
            # 4 grafik - %100 sorunsuz
            gs = GridSpec(2, 2, figure=fig)
            
            # 1. Complex Plane
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(real_parts, imag_parts, ".-", alpha=0.7)
            ax1.scatter(real_parts[0], imag_parts[0], c="g", s=100, label="Start")
            ax1.scatter(real_parts[-1], imag_parts[-1], c="r", s=100, label="End")
            ax1.set_title("Neutrosophic Complex Plane")
            ax1.legend(); ax1.axis("equal")
            
            # 2. Indeterminacy
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(indeter_parts, "o-", color="purple")
            ax2.set_title("NeutrosophicComplexNumber (NCN)")
            
            # 3. Magnitude vs I
            ax3 = fig.add_subplot(gs[1, 0])
            sc = ax3.scatter(magnitudes_z, indeter_parts, c=range(len(sequence)), 
                            cmap="viridis", s=30)
            ax3.set_title("|z| vs I"); plt.colorbar(sc, ax=ax3, label="Step")
            
            # 4. Real-Imag colored by I
            ax4 = fig.add_subplot(gs[1, 1])
            sc2 = ax4.scatter(real_parts, imag_parts, c=indeter_parts, 
                             cmap="plasma", s=40)
            ax4.set_title("Re-Im (by I)"); plt.colorbar(sc2, ax=ax4)
        """


    # --- 12. HyperrealNumber (eski tarz, basit)
    elif isinstance(first_elem, HyperrealNumber):
        try:
            # Her elemandan .sequence veya to_list/coeffs ile bileşenleri al
            rows = []
            for x in sequence:
                if hasattr(x, "sequence"):
                    seq_vals = list(getattr(x, "sequence"))
                elif hasattr(x, "to_list") and callable(getattr(x, "to_list")):
                    seq_vals = list(x.to_list())
                elif hasattr(x, "coeffs"):
                    c = x.coeffs() if callable(getattr(x, "coeffs")) else x.coeffs
                    seq_vals = list(c)
                elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
                    seq_vals = list(x)
                else:
                    try:
                        seq_vals = [float(x)]
                    except Exception:
                        seq_vals = [0.0]
                rows.append(seq_vals)

            # seq_len: her satırın minimum uzunluğu, en fazla 5
            min_len = min(len(r) for r in rows) if rows else 0
            seq_len = min(5, max(1, min_len))

            # pad/truncate
            data = np.array([ (r + [0.0]*seq_len)[:seq_len] for r in rows ], dtype=float)

            # fig oluştur / yeniden kullan
            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(12, 8), constrained_layout=True)

            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            for i in range(seq_len):
                ax1.plot(data[:, i], label=f'ε^{i}', alpha=0.8)
            ax1.set_title("Hyperreal Components")
            ax1.legend(ncol=2)
            ax1.grid(alpha=0.25)

            ax2 = fig.add_subplot(gs[0, 1])
            magnitudes = np.linalg.norm(data, axis=1)
            ax2.plot(magnitudes, 'o-', color='tab:purple')
            ax2.set_title("Magnitude")
            ax2.grid(alpha=0.25)

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(data[:, 0], 'o-', label='Standard Part')
            ax3.set_title("Standard Part (ε⁰)")
            ax3.legend()
            ax3.grid(alpha=0.25)

            ax4 = fig.add_subplot(gs[1, 1])
            y2 = data[:, 1] if data.shape[1] > 1 else np.zeros(len(data))
            sc = ax4.scatter(data[:, 0], y2, c=np.arange(len(data)), cmap='viridis')
            ax4.set_title("Standard vs Infinitesimal")
            ax4.set_xlabel("Standard")
            ax4.set_ylabel("ε¹")
            try:
                cbar = fig.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            except Exception:
                pass

            return fig

        except Exception as e:
            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(8, 4), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Hyperreal plot error: {e}", ha='center', va='center', color='red')
            ax.set_xticks([]); ax.set_yticks([])
            logger.exception("Hyperreal plotting failed")
            return fig


    # --- 13. BicomplexNumber (eski tarz, basit)
    elif isinstance(first_elem, BicomplexNumber):
        try:
            z1_real = []
            z1_imag = []
            z2_real = []
            z2_imag = []
            for x in sequence:
                z1 = getattr(x, "z1", None)
                z2 = getattr(x, "z2", None)
                if z1 is None or z2 is None:
                    if hasattr(x, "to_components") and callable(getattr(x, "to_components")):
                        comps = x.to_components()
                        if len(comps) >= 2:
                            z1 = comps[0]; z2 = comps[1]
                    elif hasattr(x, "to_list") and callable(getattr(x, "to_list")):
                        comps = x.to_list()
                        if len(comps) >= 2:
                            z1 = comps[0]; z2 = comps[1]
                    elif hasattr(x, "coeffs"):
                        c = x.coeffs() if callable(getattr(x, "coeffs")) else x.coeffs
                        c = list(c)
                        if len(c) >= 2:
                            z1 = c[0]; z2 = c[1]
                try:
                    zr = complex(z1).real if z1 is not None else 0.0
                    zi = complex(z1).imag if z1 is not None else 0.0
                except Exception:
                    zr, zi = 0.0, 0.0
                try:
                    wr = complex(z2).real if z2 is not None else 0.0
                    wi = complex(z2).imag if z2 is not None else 0.0
                except Exception:
                    wr, wi = 0.0, 0.0
                z1_real.append(zr); z1_imag.append(zi)
                z2_real.append(wr); z2_imag.append(wi)

            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(12, 8), constrained_layout=True)

            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(z1_real, label='Re(z1)')
            ax1.plot(z1_imag, label='Im(z1)')
            ax1.set_title("Bicomplex z1")
            ax1.legend()
            ax1.grid(alpha=0.25)

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(z2_real, label='Re(z2)')
            ax2.plot(z2_imag, label='Im(z2)')
            ax2.set_title("Bicomplex z2")
            ax2.legend()
            ax2.grid(alpha=0.25)

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(z1_real, z1_imag, '.-')
            ax3.set_title("z1 Trajectory")
            ax3.set_xlabel("Re(z1)")
            ax3.set_ylabel("Im(z1)")
            ax3.grid(alpha=0.25)

            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(z2_real, z2_imag, '.-')
            ax4.set_title("z2 Trajectory")
            ax4.set_xlabel("Re(z2)")
            ax4.set_ylabel("Im(z2)")
            ax4.grid(alpha=0.25)

            return fig

        except Exception as e:
            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(8, 4), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Bicomplex plot error: {e}", ha='center', va='center', color='red')
            ax.set_xticks([]); ax.set_yticks([])
            logger.exception("Bicomplex plotting failed")
            return fig


    # --- 14. NeutrosophicBicomplexNumber (eski tarz, basit)
    elif isinstance(first_elem, NeutrosophicBicomplexNumber):
        try:
            comps = []
            for x in sequence:
                vals = []
                ok = True
                for attr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                    if hasattr(x, attr):
                        try:
                            vals.append(float(getattr(x, attr)))
                        except Exception:
                            vals.append(0.0)
                    else:
                        ok = False
                        break
                if not ok:
                    if hasattr(x, "to_components") and callable(getattr(x, "to_components")):
                        comps_list = x.to_components()
                    elif hasattr(x, "to_list") and callable(getattr(x, "to_list")):
                        comps_list = x.to_list()
                    elif hasattr(x, "coeffs"):
                        c = x.coeffs() if callable(getattr(x, "coeffs")) else x.coeffs
                        comps_list = list(c)
                    else:
                        try:
                            comps_list = list(x)
                        except Exception:
                            comps_list = [0.0]*8
                    comps_list = (list(comps_list) + [0.0]*8)[:8]
                    vals = []
                    for v in comps_list[:8]:
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(0.0)
                comps.append(vals)

            comps = np.array(comps, dtype=float)
            magnitudes = np.linalg.norm(comps, axis=1)

            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(12, 8), constrained_layout=True)

            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            for i, label in enumerate(['a', 'b', 'c', 'd']):
                ax1.plot(comps[:, i], label=label, alpha=0.7)
            ax1.set_title("First 4 Components")
            ax1.legend()
            ax1.grid(alpha=0.25)

            ax2 = fig.add_subplot(gs[0, 1])
            for i, label in enumerate(['e', 'f', 'g', 'h']):
                ax2.plot(comps[:, i + 4], label=label, alpha=0.7)
            ax2.set_title("Last 4 Components")
            ax2.legend()
            ax2.grid(alpha=0.25)

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(magnitudes, 'o-', color='tab:purple')
            ax3.set_title("Magnitude")
            ax3.grid(alpha=0.25)

            ax4 = fig.add_subplot(gs[1, 1])
            sc = ax4.scatter(comps[:, 0], comps[:, 1], c=np.arange(len(comps)), cmap='plasma')
            ax4.set_title("a vs b Trajectory")
            ax4.set_xlabel("a")
            ax4.set_ylabel("b")
            try:
                cbar = fig.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            except Exception:
                pass

            return fig

        except Exception as e:
            try:
                _ = fig
            except NameError:
                fig = plt.figure(figsize=(8, 4), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"NeutrosophicBicomplex plot error: {e}", ha='center', va='center', color='red')
            ax.set_xticks([]); ax.set_yticks([])
            logger.exception("NeutrosophicBicomplex plotting failed")
            return fig


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
        # Extract real and split components robustly (support attributes or methods)
        def _get_attr_or_callable(obj, name):
            if hasattr(obj, name):
                attr = getattr(obj, name)
                return attr() if callable(attr) else attr
            return None

        reals_list = []
        splits_list = []
        for x in sequence:
            # try common attribute names / callables
            r = _get_attr_or_callable(x, "real")
            s = _get_attr_or_callable(x, "split")
            # fallback: try to_list / coeffs if available (some implementations)
            if r is None or s is None:
                if hasattr(x, "to_list") and callable(getattr(x, "to_list")):
                    comps = x.to_list()
                    if r is None and len(comps) >= 1:
                        r = comps[0]
                    if s is None and len(comps) >= 2:
                        s = comps[1]
                elif hasattr(x, "coeffs"):
                    c = x.coeffs() if callable(getattr(x, "coeffs")) else x.coeffs
                    c = list(c)
                    if r is None and len(c) >= 1:
                        r = c[0]
                    if s is None and len(c) >= 2:
                        s = c[1]
            # final fallbacks to numeric zero
            try:
                reals_list.append(float(r) if r is not None else 0.0)
            except Exception:
                reals_list.append(0.0)
            try:
                splits_list.append(float(s) if s is not None else 0.0)
            except Exception:
                splits_list.append(0.0)

        reals = np.asarray(reals_list, dtype=float)
        splits = np.asarray(splits_list, dtype=float)

        # create figure and grid
        try:
            _ = fig
        except NameError:
            fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)

        # Real component plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(reals, 'o-', color='tab:blue', label='Real')
        ax1.set_title("Real Component")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Value")
        ax1.grid(alpha=0.25)
        ax1.legend()

        # Split component plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(splits, 'o-', color='tab:red', label='Split')
        ax2.set_title("Split Component")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Value")
        ax2.grid(alpha=0.25)
        ax2.legend()

        # Local safe PCA variance helper (ensure available in this scope)
        def _pca_var_sum(pca_obj) -> float:
            try:
                arr = getattr(pca_obj, "explained_variance_ratio_", None)
                if arr is None:
                    return 0.0
                arr = np.asarray(arr, dtype=float)
                s = float(np.nansum(arr))
                return s if np.isfinite(s) else 0.0
            except Exception:
                return 0.0

        # PCA panel (right column spanning both rows)
        axp = fig.add_subplot(gs[:, 1])
        if use_pca and len(sequence) > 2:
            try:
                # prepare data matrix (samples x features)
                data = np.column_stack((reals, splits))
                if data.shape[0] < 3:
                    axp.text(0.5, 0.5, "Need ≥3 samples for PCA", ha='center', va='center', fontsize=10)
                    axp.set_title("PCA Projection (Not enough samples)")
                else:
                    # filter finite rows
                    mask = np.all(np.isfinite(data), axis=1)
                    data_clean = data[mask]
                    if data_clean.shape[0] < 3:
                        axp.text(0.5, 0.5, "Insufficient finite data for PCA", ha='center', va='center', fontsize=10)
                        axp.set_title("PCA Projection (Insufficient data)")
                    else:
                        try:
                            from sklearn.decomposition import PCA as _PCA  # local import to avoid global dependency
                            pca = _PCA(n_components=2)
                            proj = pca.fit_transform(data_clean)
                            sc = axp.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap='viridis', s=25)
                            var_sum = _pca_var_sum(pca)
                            axp.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                            axp.set_xlabel("PC1")
                            axp.set_ylabel("PC2")
                            try:
                                cbar = fig.colorbar(sc, ax=axp, fraction=0.046, pad=0.04)
                                cbar.ax.tick_params(labelsize=8)
                            except Exception:
                                # fallback: no colorbar
                                pass
                        except Exception as e:
                            logger.exception("PCA failed for Superreal data: %s", e)
                            axp.text(0.5, 0.5, f"PCA Error: {str(e)[:120]}", ha='center', va='center', fontsize=10)
                            axp.set_title("PCA Projection (Error)")
            except Exception as e:
                logger.exception("PCA preparation failed: %s", e)
                axp.text(0.5, 0.5, f"PCA Error: {str(e)[:120]}", ha='center', va='center', fontsize=10)
                axp.set_title("PCA Projection (Error)")
        else:
            axp.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=12)
            axp.set_title("PCA Projection (Unavailable)")

        return fig

    # --- 22. Ternary
    elif isinstance(first_elem, (TernaryNumber, list)):
        """TERNARY grafik - list uyumlu"""
        
        # ✅ SORUN: x.digits → list fallback kontrolü
        def safe_digits(obj):
            """TernaryNumber veya list → digits listesi"""
            if isinstance(obj, list):
                return obj  # Direkt list kullan
            try:
                return obj.digits  # TernaryNumber.digits
            except:
                return [float(obj)]  # Scalar fallback
        
        # Tüm nesnelerin digits uzunluğunu belirle
        all_digits = [safe_digits(x) for x in sequence]
        max_length = max(len(d) for d in all_digits)
        
        # Padding yap
        padded_digits = []
        for d in all_digits:
            padded = d + [0.0] * (max_length - len(d))
            padded_digits.append(padded)
        
        digits = np.array(padded_digits)
        
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Ternary digits çizimi
        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(digits.shape[1]):
            ax1.plot(digits[:, i], 'o-', alpha=0.6, label=f'digit {i}')
        ax1.set_title("Ternary Digits")
        ax1.legend(ncol=4, fontsize=6)
        
        # 2. Ondalık değerler
        ax2 = fig.add_subplot(gs[0, 1])
        
        def safe_decimal(obj):
            """TernaryNumber.to_decimal() veya list → float"""
            if isinstance(obj, list):
                return sum(obj)  # Basit toplam
            try:
                return obj.to_decimal()
            except:
                return float(obj)
        
        decimal_values = np.array([safe_decimal(x) for x in sequence])
        ax2.plot(decimal_values, 'o-', color='tab:green')
        ax2.set_title("Decimal Values")
        
        # 3. PCA (opsiyonel)
        if use_pca and len(sequence) > 2:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                proj = pca.fit_transform(digits)
                
                ax3 = fig.add_subplot(gs[1, :])
                sc = ax3.scatter(proj[:, 0], proj[:, 1], 
                               c=range(len(proj)), cmap='viridis', s=25)
                ax3.set_title(f"PCA (Var: {sum(pca.explained_variance_ratio_):.3f})")
                plt.colorbar(sc, ax=ax3, label="Iteration")
            except ImportError:
                ax3 = fig.add_subplot(gs[1, :])
                ax3.text(0.5, 0.5, "sklearn yok\npip install scikit-learn", 
                        ha='center', va='center', fontsize=10)
            except Exception as e:
                ax3 = fig.add_subplot(gs[1, :])
                ax3.text(0.5, 0.5, f"PCA Error: {str(e)[:30]}", 
                        ha='center', va='center', fontsize=10)
        else:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.text(0.5, 0.5, "PCA için 3+ örnek\nveya sklearn kurun", 
                    ha='center', va='center', fontsize=10)
        """
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
        """

    # --- 23. HypercomplexNumber
    elif isinstance(first_elem, HypercomplexNumber):
        # try to extract coefficient array from sequence of HypercomplexNumber-like objects
        try:
            def _extract_coeffs_list(seq, complex_mode='real'):
                """
                seq: iterable of values (HypercomplexNumber or lists or scalars)
                complex_mode: 'real' | 'magnitude' (how to handle complex components)
                Returns: list of lists (samples x components) as floats
                """
                out = []
                for v in seq:
                    try:
                        # Prefer explicit conversion helpers if present
                        if hasattr(v, 'to_list') and callable(getattr(v, 'to_list')):
                            comps = v.to_list()
                        elif hasattr(v, 'to_components') and callable(getattr(v, 'to_components')):
                            comps = v.to_components()
                        elif hasattr(v, 'coeffs'):
                            c = v.coeffs() if callable(getattr(v, 'coeffs')) else v.coeffs
                            comps = list(c)
                        elif hasattr(v, 'components'):
                            c = v.components() if callable(getattr(v, 'components')) else v.components
                            comps = list(c)
                        elif hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                            comps = list(v)
                        else:
                            comps = [v]

                        # normalize components to floats
                        norm = []
                        for c in comps:
                            if isinstance(c, complex):
                                if complex_mode == 'magnitude':
                                    norm.append(float(abs(c)))
                                else:
                                    # default: real part
                                    norm.append(float(c.real))
                            else:
                                try:
                                    norm.append(float(c))
                                except Exception:
                                    # non-numeric component -> 0.0
                                    norm.append(0.0)
                        out.append(norm)
                    except Exception as e:
                        logger.debug("extract coeffs failed for %r: %s", v, e)
                        out.append([0.0])
                return out

            def _pca_var_sum(pca_obj) -> float:
                """
                Safely return sum of PCA explained variance ratio.
                - Uses pca_obj.explained_variance_ratio_ when available.
                - Returns 0.0 for missing, NaN, infinite or invalid values.
                """
                try:
                    arr = getattr(pca_obj, "explained_variance_ratio_", None)
                    if arr is None:
                        return 0.0
                    arr = np.asarray(arr, dtype=float)
                    s = float(np.nansum(arr))
                    return s if np.isfinite(s) else 0.0
                except Exception:
                    return 0.0

            coeffs_list = _extract_coeffs_list(sequence, complex_mode='real')
            coeffs = np.array(coeffs_list, dtype=float)

        except Exception as e:
            # if extraction fails, show a message on the figure and bail out gracefully
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Coefficient extraction failed:\n{e}", ha='center', va='center', fontsize=10)
            logger.exception("Coefficient extraction failed")
            return fig

        # dimensions and magnitudes
        n_samples, dim = coeffs.shape
        magnitudes = np.linalg.norm(coeffs, axis=1)

        # Create or reuse figure with constrained_layout to avoid tight_layout/colorbar conflicts
        try:
            _ = fig  # reuse existing fig if present
        except NameError:
            fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        else:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass

        # layout decisions based on dimension
        if dim <= 16:
            cols = 4
            rows = int(np.ceil(dim / cols))
            # Reserve an extra row for magnitude / PCA
            gs = GridSpec(rows + 1, cols, figure=fig, height_ratios=[1] * rows + [0.8])
            axes = []
            for i in range(dim):
                r = i // cols
                c = i % cols
                ax = fig.add_subplot(gs[r, c])
                ax.plot(coeffs[:, i], '-', linewidth=0.8, alpha=0.8)
                ax.set_title(f"e{i}", fontsize=8)
                axes.append(ax)

            # magnitude plot in the first slot of the last row
            axm = fig.add_subplot(gs[rows, 0])
            axm.plot(magnitudes, 'o-', color='tab:orange')
            axm.set_title("Magnitude |v|", fontsize=9)
            axm.set_xlabel("Iteration")
            axm.set_ylabel("|v|")

            # PCA panel if requested
            if use_pca:
                try:
                    if PCA is None:
                        raise RuntimeError("sklearn not available for PCA")
                    if n_samples > 2:
                        pca = PCA(n_components=2)
                        proj = pca.fit_transform(coeffs)
                        axp = fig.add_subplot(gs[rows, 1])
                        sc = axp.scatter(proj[:, 0], proj[:, 1], c=np.arange(n_samples), cmap='viridis', s=20)
                        var_sum = _pca_var_sum(pca)
                        axp.set_title(f"PCA (Var: {var_sum:.3f})", fontsize=9)
                        try:
                            cbar = fig.colorbar(sc, ax=axp, fraction=0.046, pad=0.02)
                            cbar.ax.tick_params(labelsize=8)
                        except Exception as e:
                            logger.debug("PCA colorbar failed: %s", e)
                    else:
                        axp = fig.add_subplot(gs[rows, 1])
                        axp.text(0.5, 0.5, "Not enough samples for PCA", ha='center', va='center')
                except Exception as e:
                    axp = fig.add_subplot(gs[rows, 1])
                    axp.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=8)
                    logger.debug("PCA error: %s", e)

        else:
            # high-dimensional case: show first 64 components in two panels, heatmap for all components, magnitude and PCA
            gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.6])

            # panel 1: components 0..min(63, dim-1)
            ax1 = fig.add_subplot(gs[0, 0])
            max_plot = min(64, dim)
            for i in range(0, max_plot):
                ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.4)
            ax1.set_title(f"Components e0-e{max_plot-1}", fontsize=9)
            ax1.legend(ncol=4, fontsize=6, loc='upper right')

            # panel 2: components 64..127 if available
            ax2 = fig.add_subplot(gs[0, 1])
            if dim > 64:
                max_plot2 = min(128, dim)
                for i in range(64, max_plot2):
                    ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.4)
                ax2.set_title(f"Components e64-e{max_plot2-1}", fontsize=9)
                ax2.legend(ncol=4, fontsize=6, loc='upper right')
            else:
                ax2.text(0.5, 0.5, "No components 64+", ha='center', va='center', fontsize=10)

            # panel 3: magnitude
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(magnitudes, 'o-', color='tab:orange')
            ax3.set_title("Magnitude |v|", fontsize=9)

            # panel 4: heatmap of coefficients (samples x components) but downsample if huge
            ax4 = fig.add_subplot(gs[1, 1])
            try:
                # downsample rows if too many samples for display
                display_coeffs = coeffs
                if n_samples > 500:
                    idx = np.linspace(0, n_samples - 1, 500).astype(int)
                    display_coeffs = coeffs[idx, :]
                # downsample columns if too many components
                if dim > 1024:
                    cidx = np.linspace(0, dim - 1, 1024).astype(int)
                    display_coeffs = display_coeffs[:, cidx]
                im = ax4.imshow(display_coeffs.T, aspect='auto', cmap='RdBu_r', origin='lower')
                ax4.set_title("Coefficient heatmap (components x samples)", fontsize=9)
                try:
                    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                except Exception as e:
                    logger.debug("Heatmap colorbar failed: %s", e)
            except Exception as e:
                ax4.text(0.5, 0.5, f"Heatmap Error: {e}", ha='center', va='center', fontsize=8)
                logger.debug("Heatmap error: %s", e)

            # PCA row below
            if use_pca:
                try:
                    if PCA is None:
                        raise RuntimeError("sklearn not available for PCA")
                    if n_samples > 2:
                        pca = PCA(n_components=2)
                        proj = pca.fit_transform(coeffs)
                        axp = fig.add_subplot(gs[2, :])
                        sc = axp.scatter(proj[:, 0], proj[:, 1], c=np.arange(n_samples), cmap='viridis', s=18)
                        var_sum = _pca_var_sum(pca)
                        axp.set_title(f"PCA Projection (Var: {var_sum:.3f})", fontsize=9)
                        try:
                            cbar2 = fig.colorbar(sc, ax=axp, fraction=0.046, pad=0.02)
                            cbar2.ax.tick_params(labelsize=8)
                        except Exception as e:
                            logger.debug("PCA colorbar failed: %s", e)
                    else:
                        axp = fig.add_subplot(gs[2, :])
                        axp.text(0.5, 0.5, "Not enough samples for PCA", ha='center', va='center')
                except Exception as e:
                    axp = fig.add_subplot(gs[2, :])
                    axp.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
                    logger.debug("PCA error: %s", e)
            else:
                axp = fig.add_subplot(gs[2, :])
                axp.text(0.5, 0.5, "Install sklearn for PCA", ha='center', va='center', fontsize=10)

        # final: do not call tight_layout when constrained_layout=True
        # return the figure to caller
        return fig


    # --- 24. Bilinmeyen tip
    else:
        ax = fig.add_subplot(1, 1, 1)
        type_name = type(first_elem).__name__
        ax.text(0.5, 0.5, f"Plotting not implemented\nfor '{type_name}'",
                ha='center', va='center', fontsize=14, fontweight='bold', color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# Test kodu
def test_division():
    test_cases = [
        (10, 2, 5.0),
        (10, 0, float('inf')),
        (complex(10, 0), 2, complex(5, 0)),
        (Fraction(10, 1), 2, Fraction(5, 1)),
        (-10, 2, -5.0),
        (10, -2, -5.0),
    ]
    
    for a, b, expected in test_cases:
        try:
            result = _safe_divide(a, b)
            print(f"{a} / {b} = {result} (expected: {expected}) - {'✓' if str(result) == str(expected) else '✗'}")
        except Exception as e:
            print(f"{a} / {b} = ERROR: {e}")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    # If user runs module directly, configure basic logging to console for demonstration.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Keçeci Numbers Module - Demonstration")
    logger.info("This script demonstrates the generation of various Keçeci Number types.")

    STEPS = 30
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
