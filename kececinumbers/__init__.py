# -*- coding: utf-8 -*-
# __init__.py

"""
Keçeci Numbers: A Comprehensive Framework for Number Sequence Analysis.

This package provides tools for generating, analyzing, and visualizing
16 different types of Keçeci Number sequences, from standard integers
to complex algebraic structures like quaternions and neutrosophic numbers.

Bu dosya paketin başlangıç noktası olarak çalışır.
Alt modülleri yükler, sürüm bilgileri tanımlar ve geriye dönük uyumluluk için uyarılar sağlar.
"""

from __future__ import annotations
import inspect
import importlib
import os
import warnings

# if os.getenv("DEVELOPMENT") == "true":
    # importlib.reload(kececinumbers) # F821 undefined name 'kececinumbers'

# Paket sürüm numarası
__version__ = "0.8.0"
__author__ = "Mehmet Keçeci"
__email__ = "mkececi@yaani.com"

# Public API exposed to users of the 'kececinumbers' package.
# Public API exposed to users of the 'kececinumbers' package.
__all__ = [
    # --- Custom Number Classes ---
    'NeutrosophicNumber',
    'NeutrosophicComplexNumber',
    'HyperrealNumber',
    'BicomplexNumber',
    'NeutrosophicBicomplexNumber',
    'OctonionNumber',
    'Constants',
    'SedenionNumber',
    'CliffordNumber',
    'DualNumber',
    'SplitcomplexNumber',

    # --- High-Level Functions ---
    'get_with_params',
    'get_interactive',
    'get_random_type',
    '_get_integer_representation',
    '_parse_quaternion',
    '_parse_quaternion_from_csv',
    '_parse_complex',
    '_parse_bicomplex',
    '_parse_universal',
    '_parse_octonion',
    '_parse_sedenion',
    '_parse_neutrosophic',
    '_parse_neutrosophic_bicomplex',
    '_parse_hyperreal',
    '_parse_clifford',
    '_parse_dual',
    '_parse_splitcomplex',
    'kececi_bicomplex_algorithm',
    'kececi_bicomplex_advanced',
    'generate_kececi_vectorial',
    '_plot_comparison', 
    '_find_kececi_zeta_zeros',
    '_compute_gue_similarity',
    '_load_zeta_zeros',
    'analyze_all_types',
    'analyze_pair_correlation',
    'print_detailed_report',
    '_gue_pair_correlation',
    '_pair_correlation',
    'generate_octonion',
    'OctonionNumber',
    'is_quaternion_like',
    'is_neutrosophic_like',
    '_has_bicomplex_format',
    'coeffs',
    'convert_to_float',
    'safe_add',
    'ZERO',
    'ONE',
    'I',
    'J',
    'K',
    'E',
    'F',
    'G',
    'H',
    '_extract_numeric_part',
    '_has_comma_format',
    '_is_complex_like',
    'is_prime_like',
    'is_near_integer',
    '_plot_component_distribution',
    '_parse_pathion',
    '_parse_chingon',
    '_parse_routon',
    '_parse_voudon',
    

    # --- Core Generation and Analysis ---
    'unified_generator',
    'is_prime',
    'find_period',
    'find_kececi_prime_number',

    # --- Visualization and Reporting ---
    'plot_numbers',

    # --- Type Constants ---
    'TYPE_POSITIVE_REAL',
    'TYPE_NEGATIVE_REAL',
    'TYPE_COMPLEX',
    'TYPE_FLOAT',
    'TYPE_RATIONAL',
    'TYPE_QUATERNION',
    'TYPE_NEUTROSOPHIC',
    'TYPE_NEUTROSOPHIC_COMPLEX',
    'TYPE_HYPERREAL',
    'TYPE_BICOMPLEX',
    'TYPE_NEUTROSOPHIC_BICOMPLEX',
    'TYPE_OCTONION',
    'TYPE_SEDENION',
    'TYPE_CLIFFORD',
    'TYPE_DUAL',
    'TYPE_SPLIT_COMPLEX',
    'TYPE_Pathion',
    'TYPE_Chingon',
    'TYPE_Routon',
    'TYPE_Voudon',
]

# Göreli modül içe aktarmaları
# F401 hatasını önlemek için sadece kullanacağınız şeyleri dışa aktarın
# Aksi halde linter'lar "imported but unused" uyarısı verir
from .kececinumbers import *
try:
    #from .kececinumbers import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    #from . import kececinumbers  # Modülün kendisine doğrudan erişim isteniyorsa
    # Import the public API into the package's namespace.
    from .kececinumbers import (
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
except ImportError as e:
    warnings.warn(f"Gerekli modül yüklenemedi: {e}", ImportWarning)

# Eski bir fonksiyonun yer tutucusu - gelecekte kaldırılacak
def eski_fonksiyon():
    """
    Kaldırılması planlanan eski bir fonksiyondur.
    Lütfen alternatif fonksiyonları kullanın.
    """
    warnings.warn(
        "eski_fonksiyon() artık kullanılmamaktadır ve gelecekte kaldırılacaktır. "
        "Lütfen yeni alternatif fonksiyonları kullanın. "
        "Keçeci numbers; Python 3.10-3.13 sürümlerinde sorunsuz çalışmalıdır.",
        category=DeprecationWarning,
        stacklevel=2
    )
