# test_helpers.py

import math
import numpy as np
import pytest
from fractions import Fraction
import kececinumbers as kn
import logging
# Module logger — library code should not configure logging handlers.
logger = logging.getLogger(__name__)

try:
    # numpy-quaternion kütüphanesinin sınıfını yüklemeye çalış
    # conda install -c conda-forge quaternion # pip install numpy-quaternion
    from quaternion import quaternion as quaternion  # type: ignore
except Exception:
    # Eğer yoksa `quaternion` isimli sembolü None yap, kodun diğer yerleri bunu kontrol edebilir
    quaternion = None
    logger.warning("numpy-quaternion paketine ulaşılamadı — quaternion tip desteği devre dışı bırakıldı.")

def test_get_integer_representation_basic_int():
    assert kn._get_integer_representation(5) == 5
    assert kn._get_integer_representation(-7) == 7

def test_get_integer_representation_float_near_int():
    assert kn._get_integer_representation(3.0000000000001) == 3
    assert kn._get_integer_representation(3.14) is None

def test_get_integer_representation_fraction():
    assert kn._get_integer_representation(Fraction(4,1)) == 4
    assert kn._get_integer_representation(Fraction(3,2)) is None

def test_get_integer_representation_complex():
    assert kn._get_integer_representation(6+0j) == 6
    assert kn._get_integer_representation(6+1e-13j) == 6
    assert kn._get_integer_representation(3+2j) is None

def test_convert_to_float_basic():
    assert math.isclose(kn.convert_to_float(5), 5.0)
    assert math.isclose(kn.convert_to_float(3.5), 3.5)
    assert math.isclose(kn.convert_to_float(3+0j), 3.0)

def test_convert_to_float_coeffs():
    # PathionNumber coerces first coeff as float
    p = kn.PathionNumber(11, *([0.0]*31))
    assert math.isclose(kn.convert_to_float(p), 11.0)

def test_is_divisible_simple():
    assert kn._is_divisible(6, 3, kn.TYPE_POSITIVE_REAL)
    assert not kn._is_divisible(7, 3, kn.TYPE_POSITIVE_REAL)
    assert kn._is_divisible(3+0j, 3, kn.TYPE_COMPLEX)
    assert not kn._is_divisible(3+1j, 3, kn.TYPE_COMPLEX)

def test_is_prime_like_basic():
    assert kn.is_prime_like(7, kn.TYPE_POSITIVE_REAL)
    assert not kn.is_prime_like(8, kn.TYPE_POSITIVE_REAL)

def test_ternary_conversion_and_prime():
    t = kn.TernaryNumber.from_ternary_string("102")  # decimal 11
    assert kn._get_integer_representation(t) == 11
    assert kn.is_prime_like(t, kn.TYPE_TERNARY)
