# -*- coding: utf-8 -*-
# test_helpers.py

from fractions import Fraction
import logging
import numpy as np
import math
import pytest
from typing import Callable, Any, List
import unittest

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

from kececinumbers import (
    # Classes / Number types
    _parse_engineering_notation,
    get_interactive,
    TernaryNumber,
    SuperrealNumber,
    BaseNumber,
    PathionNumber,
    ChingonNumber,
    RoutonNumber,
    VoudonNumber,
    OctonionNumber,
    generate_octonion,
    Constants,
    NeutrosophicNumber,
    NeutrosophicComplexNumber,
    HyperrealNumber,
    BicomplexNumber,
    NeutrosophicBicomplexNumber,
    SedenionNumber,
    CliffordNumber,
    DualNumber,
    SplitcomplexNumber,
    quaternion,
    # coeffs,
    find_period,
    ComplexNumber,
    HypercomplexNumber,
    Real,
    Complex,
    Quaternion,
    Octonion,
    Sedenion,
    Pathion,
    Chingon,
    Routon,
    Voudon,
    _parse_real,
    _parse_super_real,
    is_super_real_expression,
    normalize_super_real,
    _get_default_hypercomplex,
    _get_default_value,
    _parse_to_hypercomplex,
    chingon_zeros,
    chingon_ones,
    chingon_unit_vector,
    chingon_eye,
    chingon_random,
    chingon_linspace,
    chingon_dot,
    chingon_cross,
    chingon_norm,
    chingon_normalize,
    neutrosophic_zero,
    neutrosophic_one,
    neutrosophic_i,
    neutrosophic_f,
    parse_to_neutrosophic,
    parse_to_hyperreal,
    _safe_float_convert,
    _parse_complex_like_string,
    _safe_float,
    get_random_types_batch,
    _find_kececi_prime_number,
    _safe_power,
    _safe_mod,
    _safe_divide,
    _float_mod_zero,
    ValueProcessor,
    _generate_sequence,
    extract_values_for_plotting,
    _generate_simple_sequence,
    _generate_detailed_sequence,
    _generate_default_value,
    clean_sequence_for_plotting,
    extract_numeric_value,
    extract_numeric_values,
    extract_complex_values,
    extract_fraction_values,
    find_first_numeric,
    extract_clean_numbers,
    # Core generator / API
    unified_generator,
    get_with_params,
    get_random_type,
    generate_kececi_vectorial,
    kececi_bicomplex_algorithm,
    kececi_bicomplex_advanced,
    # Analysis / utilities
    find_kececi_prime_number,
    _get_integer_representation,
    _is_divisible,
    is_prime,
    is_prime_like,
    is_near_integer,
    is_neutrosophic_like,
    is_quaternion_like,
    # DİKKAT: Bu fonksiyonun adı değiştirilmeli veya kullanım şekli değiştirilmeli
    test_kececi_conjecture as check_kececi_conjecture,  # Takma ad kullan
    analyze_kececi_sequence,
    analyze_all_types,
    analyze_pair_correlation,
    _compute_gue_similarity,
    _find_kececi_zeta_zeros,
    _load_zeta_zeros,
    _pair_correlation,
    # Plotting / visualization
    print_detailed_report,
    plot_numbers,
    plot_octonion_3d,
    generate_interactive_plot,
    apply_pca_clustering,
    _plot_comparison,
    _plot_component_distribution,
    # Parsers (if you want them public)
    _parse_complex,
    _parse_bicomplex,
    _parse_octonion,
    _parse_sedenion,
    _parse_pathion,
    _parse_chingon,
    _parse_routon,
    _parse_universal,
    _parse_voudon,
    _parse_clifford,
    _parse_dual,
    _parse_splitcomplex,
    _parse_ternary,
    _parse_superreal,
    _parse_hyperreal,
    _parse_neutrosophic,
    _parse_neutrosophic_complex,
    _parse_neutrosophic_bicomplex,
    _parse_quaternion,
    _parse_quaternion_from_csv,
    _pca_var_sum,
    # Helpers / small utilities recently added
    convert_to_float,
    safe_add,
    format_fraction,
    _has_bicomplex_format,
    _extract_numeric_part,
    _has_comma_format,
    _gue_pair_correlation,
    _is_complex_like,
    logger,
    # TYPE constants
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
    TYPE_PATHION,
    TYPE_CHINGON,
    TYPE_ROUTON,
    TYPE_VOUDON,
    TYPE_SUPERREAL,
    TYPE_TERNARY,
    TYPE_HYPERCOMPLEX,
)

import kececinumbers as kn


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
    assert kn.is_prime_like(t, kn.TYPE_TERNARY)


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
