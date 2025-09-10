# -*- coding: utf-8 -*-
# __init__.py

"""
Keçeci Numbers: A Comprehensive Framework for Number Sequence Analysis.

This package provides tools for generating, analyzing, and visualizing
11 different types of Keçeci Number sequences, from standard integers
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
__version__ = "0.6.8"
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
    '_parse_quaternion_from_csv',
    'generate_kececi_vectorial',
    '_plot_comparison',  # DÜZELTİLDİ: '_plot_comparsion' -> '_plot_comparison'
    '_find_kececi_zeta_zeros',
    '_compute_gue_similarity',
    '_load_zeta_zeros',
    'analyze_all_types',
    'analyze_pair_correlation',
    'print_detailed_report',
    '_gue_pair_correlation',
    '_pair_correlation',
    '_parse_octonion',
    'generate_octonion',
    'OctonionNumber',
    'is_quaternion_like',
    'is_neutrosophic_like',
    '_has_bicomplex_format',
    '_parse_bicomplex',
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
        _parse_quaternion_from_csv,
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
        _parse_octonion,
        generate_octonion,
        is_quaternion_like,
        is_neutrosophic_like,
        _has_bicomplex_format,
        _parse_bicomplex,
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
        "Keçeci numbers; Python 3.7-3.14 sürümlerinde sorunsuz çalışmalıdır.",
        category=DeprecationWarning,
        stacklevel=2
    )
"""
# Explicit import for critical functions to avoid cache issues
try:
    from .kececinumbers import (
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
        _parse_quaternion_from_csv,
        generate_kececi_vectorial,
        unified_generator,
        is_prime,
        find_period,
        find_kececi_prime_number,
        plot_numbers,
        _plot_comparison,
        _find_kececi_zeta_zeros,
        _compute_gue_similarity,
        _load_zeta_zeros,
        analyze_all_types,
        analyze_pair_correlation,
        _gue_pair_correlation,
        _pair_correlation,
        _parse_octonion,
        generate_octonion,
        ZERO,
        ONE,
        I,
        J,
        K,
        E,
        F,
        G,
        H,
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
        TYPE_SPLIT_COMPLEX
    )
    
    # Explicit olarak global namespace'e ekle
    globals().update({
        'analyze_all_types': analyze_all_types,
        '_plot_comparison': _plot_comparison,
        'TYPE_OCTONION': TYPE_OCTONION,
        'TYPE_FLOAT': TYPE_FLOAT
    })
    
except ImportError as e:
    warnings.warn(f"Kritik fonksiyonlar import edilemedi: {e}", ImportWarning)
"""
"""
def __getattr__(name):
    #Dynamic attribute access for better compatibility
    if name in ['analyze_all_types', '_plot_comparison']:
        try:
            from .kececinumbers import analyze_all_types, _plot_comparison, TYPE_OCTONION, TYPE_FLOAT
            if name == 'analyze_all_types':
                return analyze_all_types
            elif name == '_plot_comparison':
                return _plot_comparison
            elif name == 'TYPE_OCTONION':
                return TYPE_OCTONION
            elif name == 'TYPE_FLOAT':
                return TYPE_FLOAT
        except ImportError:
            pass
    
    raise AttributeError(f"module 'kececinumbers' has no attribute '{name}'")
"""
