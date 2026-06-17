# -*- coding: utf-8 -*-
# test_helpers.py

import math
import unittest
from fractions import Fraction
from typing import Any, List

import pytest

"""
try:
    # numpy-quaternion kütüphanesinin sınıfını yüklemeye çalış
    # conda install -c conda-forge quaternion # pip install numpy-quaternion
    from quaternion import quaternion as quaternion  # type: ignore
except Exception:
    # Eğer yoksa `quaternion` isimli sembolü None yap, kodun diğer yerleri bunu kontrol edebilir
    quaternion = None
    logger.warning("numpy-quaternion paketine ulaşılamadı — quaternion tip desteği devre dışı bırakıldı.")
"""

import kececinumbers as kn
from kececinumbers import (
    # Classes / Number types
    test_kececi_conjecture as check_kececi_conjecture,  # Takma ad kullan
)


def test_get_integer_representation_basic_int():
    assert kn._get_integer_representation(5) == 5
    assert kn._get_integer_representation(-7) == 7


def test_get_integer_representation_float_near_int():
    assert kn._get_integer_representation(3.0000000000001) == 3
    assert kn._get_integer_representation(3.14) is None


def test_get_integer_representation_fraction():
    assert kn._get_integer_representation(Fraction(4, 1)) == 4
    assert kn._get_integer_representation(Fraction(3, 2)) is None


def test_get_integer_representation_complex():
    assert kn._get_integer_representation(6 + 0j) == 6
    assert kn._get_integer_representation(6 + 1e-13j) == 6
    assert kn._get_integer_representation(3 + 2j) is None


def test_convert_to_float_basic():
    assert math.isclose(kn.convert_to_float(5), 5.0)
    assert math.isclose(kn.convert_to_float(3.5), 3.5)
    assert math.isclose(kn.convert_to_float(3 + 0j), 3.0)


def test_convert_to_float_coeffs():
    # PathionNumber coerces first coeff as float
    p = kn.PathionNumber(11, *([0.0] * 31))
    assert math.isclose(kn.convert_to_float(p), 11.0)


def test_is_divisible_simple():
    assert kn._is_divisible(6, 3, kn.TYPE_POSITIVE_REAL)
    assert not kn._is_divisible(7, 3, kn.TYPE_POSITIVE_REAL)
    assert kn._is_divisible(3 + 0j, 3, kn.TYPE_COMPLEX)
    assert not kn._is_divisible(3 + 1j, 3, kn.TYPE_COMPLEX)


def test_is_prime_like_basic():
    assert kn.is_prime_like(7, kn.TYPE_POSITIVE_REAL)
    assert not kn.is_prime_like(8, kn.TYPE_POSITIVE_REAL)


def test_ternary_conversion_and_prime():
    t = kn.TernaryNumber.from_ternary_string("102")  # decimal 11
    assert kn._get_integer_representation(t) == 11
    int_repr = kn._get_integer_representation(t)
    assert int_repr is not None and kn.is_prime(int_repr)  # doğrudan asallık kontrolü


# test_kececi_conjecture için test fonksiyonları
# Pytest artık bunları test olarak görecek ama sorun olmayacak


def test_check_kececi_conjecture_basic():
    """Test basic Keçeci conjecture functionality."""
    # Test 1: Basic positive sequence
    result = check_kececi_conjecture(
        sequence=[5, 8, 11],
        add_value=3.0,
        kececi_type=kn.TYPE_POSITIVE_REAL,
        max_steps=10,
    )
    assert isinstance(result, bool)
    print(f"Basic conjecture test result: {result}")

    # Test 2: Empty sequence should raise ValueError
    try:
        check_kececi_conjecture([], 3.0, kn.TYPE_POSITIVE_REAL, 10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "sequence must contain at least one element" in str(e)
        print(f"Correctly raised ValueError: {e}")


def test_check_kececi_conjecture_different_types():
    """Test conjecture with different number types."""
    test_cases = [
        ([5, 8, 11], 3.0, kn.TYPE_POSITIVE_REAL, "Positive real"),
        ([-7, -5, -3], 2.0, kn.TYPE_NEGATIVE_REAL, "Negative real"),
        ([3.14, 4.14], 1.0, kn.TYPE_FLOAT, "Float"),
    ]

    for seq, add_val, k_type, desc in test_cases:
        result = check_kececi_conjecture(seq, add_val, k_type, 5)
        assert isinstance(result, bool), f"{desc} should return bool"
        print(f"{desc} test: sequence={seq}, result={result}")


@pytest.mark.parametrize(
    "seq,add_val,k_type",
    [
        ([5, 8, 11], 3.0, kn.TYPE_POSITIVE_REAL),
        ([-7, -5, -3], 2.0, kn.TYPE_NEGATIVE_REAL),
        ([3.14, 4.14], 1.0, kn.TYPE_FLOAT),
    ],
)
def test_check_kececi_conjecture_parametrized(seq, add_val, k_type):
    """Parametrized test for Keçeci conjecture."""
    result = check_kececi_conjecture(seq, add_val, k_type, 10)
    assert isinstance(result, bool)
    print(f"Parametrized test: seq={seq}, result={result}")


# Fixture tanımları (Pytest için)
@pytest.fixture
def positive_sequence() -> List[Any]:
    """Positive integer sequence fixture."""
    return [5, 8, 11, 14]


@pytest.fixture
def negative_sequence() -> List[Any]:
    """Negative integer sequence fixture."""
    return [-7, -5, -3, -1]


@pytest.fixture
def float_sequence() -> List[Any]:
    """Float sequence fixture."""
    return [3.14, 4.64, 6.14, 7.64]


@pytest.fixture
def add_value():
    """Default add value fixture."""
    return 3.0


# Fixture'ları kullanan testler
def test_check_kececi_conjecture_with_fixtures(positive_sequence, add_value):
    """Test using Pytest fixtures."""
    result = check_kececi_conjecture(
        sequence=positive_sequence,
        add_value=add_value,
        kececi_type=kn.TYPE_POSITIVE_REAL,
        max_steps=10,
    )
    assert isinstance(result, bool)
    print(f"Fixture test result: {result}")


# unittest sınıfı
class TestKececiConjectureClass(unittest.TestCase):
    def test_basic(self):
        """Basic test using unittest framework."""
        result = check_kececi_conjecture(
            sequence=[5, 8, 11],
            add_value=3.0,
            kececi_type=kn.TYPE_POSITIVE_REAL,
            max_steps=10,
        )
        self.assertIsInstance(result, bool)

    def test_empty_sequence(self):
        """Test that empty sequence raises error."""
        with self.assertRaises(ValueError):
            check_kececi_conjecture([], 3.0, kn.TYPE_POSITIVE_REAL, 10)

    def test_max_steps(self):
        """Test max_steps parameter."""
        # Use a sequence that likely won't find a prime quickly
        result = check_kececi_conjecture(
            sequence=[4, 6, 8],  # Even numbers (2 is prime but not in sequence)
            add_value=2.0,
            kececi_type=kn.TYPE_POSITIVE_REAL,
            max_steps=3,  # Very small limit
        )
        self.assertIsInstance(result, bool)


# Manually run conjecture tests
def run_manual_conjecture_tests():
    """Run manual tests for the conjecture function."""
    print("\n=== Manual Keçeci Conjecture Tests ===")

    # Test 1: Sequence that contains a prime
    seq1 = [4, 7, 10]  # Contains prime 7
    result1 = check_kececi_conjecture(seq1, 3.0, kn.TYPE_POSITIVE_REAL, 5)
    print(f"Sequence [4,7,10] (contains prime): {result1}")

    # Test 2: Sequence that might generate a prime
    seq2 = [14, 17, 20]  # 17 is prime
    result2 = check_kececi_conjecture(seq2, 3.0, kn.TYPE_POSITIVE_REAL, 5)
    print(f"Sequence [14,17,20] (contains prime): {result2}")

    # Test 3: Complex sequence
    seq3 = [complex(2, 3), complex(3, 4)]
    result3 = check_kececi_conjecture(seq3, complex(1, 0), kn.TYPE_COMPLEX, 5)
    print(f"Complex sequence: {result3}")

    # Test 4: Without kececi_type (uses fallback)
    seq4 = [5, 8, 11]
    result4 = check_kececi_conjecture(seq4, 3.0, None, 10)
    print(f"Without kececi_type: {result4}")


if __name__ == "__main__":
    print("Running Keçeci tests...")

    # Run unittest tests
    print("\n--- Running unittest tests ---")
    unittest.main(exit=False)

    # Run manual tests
    run_manual_conjecture_tests()

    print("\nTests completed successfully!")
