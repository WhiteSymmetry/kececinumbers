# test_helpers.py

import math
import numpy as np
import pytest
from fractions import Fraction
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

from kececinumbers import (
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
        BaseNumber,
        TernaryNumber,
        SuperrealNumber,
        
    
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
        is_prime,
        is_prime_like,
        is_near_integer,
        _plot_component_distribution,
        _parse_pathion,
        _parse_chingon,
        _parse_routon,
        _parse_voudon,
        format_fraction,
        test_kececi_conjecture,
        generate_interactive_plot,
        apply_pca_clustering,
        analyze_kececi_sequence,
        plot_octonion_3d,
        _parse_ternary,
        _parse_superreal,
         
    
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
        TYPE_PATHION,
        TYPE_CHINGON,
        TYPE_ROUTON,
        TYPE_VOUDON,
        TYPE_SUPERREAL,
        TYPE_TERNARY,
)

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
