# -*- coding: utf-8 -*-
# __init__.py

"""
Keçeci Numbers: A Comprehensive Framework for Number Sequence Analysis.

This package provides tools for generating, analyzing, and visualizing
16 different types of Keçeci Number sequences, from standard integers
to complex algebraic structures like quaternions and neutrosophic numbers.

Bu dosya paketin başlangıç noktası olarak çalışır.
Alt modülleri yükler, sürüm bilgileri tanımlar ve geriye dönük uyumluluk için uyarılar sağlar.

:author: Mehmet Keçeci
:license: AGPL-3.0-or-later
:copyright: Copyright 2025-2026 Mehmet Keçeci
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, List

# Paket sürüm numarası
# __version__ = "1.0.5"
# __author__ = "Mehmet Keçeci"
# __email__ = "mkececi@yaani.com"
__description__ = (
    "Keçeci Numbers: An Exploration of a Dynamic Sequence Across Diverse Number Sets."
)

# ======================================================================
# METADATA & VERSIONING (Modern Approach)
# ======================================================================

# Try to read metadata from the installed package (pyproject.toml / setup.py)
try:
    from importlib.metadata import metadata as _pkg_metadata
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("kececinumbers")
    _meta = _pkg_metadata("kececinumbers")
    __author__ = _meta.get("Author-email", "Mehmet Keçeci <mkececi@yaani.com>")
    __license__ = _meta.get("License", "AGPL-3.0-or-later")
except Exception:
    # Fallback for development or if metadata is not available
    __version__ = "1.0.5"
    __author__ = "Mehmet Keçeci"
    __license__ = "AGPL-3.0-or-later"

__copyright__ = "Copyright 2025-2026 Mehmet Keçeci"
__email__ = "mkececi@yaani.com"
__certificate__ = "kececinumbers-PA-2025-001"

_log = logging.getLogger(__name__)

# BibTeX citation for academic use
__bibtex__ = r"""@misc{kececi_2025_15589625,
  author       = {Keçeci, Mehmet},
  title        = {Keçeci Numbers and the Keçeci Prime Number: A
                   Potential Number Theoretic Exploratory Tool
                  },
  month        = May,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15381697},
  url          = {https://doi.org/10.5281/zenodo.15381697},
}"""

# ======================================================================
# TYPE CHECKING (Only for IDEs and type checkers like mypy)
# ======================================================================
if TYPE_CHECKING:
    from typing import Literal

    LogLevel = int | str

# ======================================================================
# PUBLIC API IMPORTS
# ======================================================================

# Ana Analiz Fonksiyonları
# Parser Fonksiyonları
# Sayı Sınıfları
# Cebirsel Tipler
# Yardımcı Fonksiyonlar
# Tip Sabitleri
from .kececinumbers import (
    TYPE_BICOMPLEX,
    TYPE_CHINGON,
    TYPE_CLIFFORD,
    TYPE_COMPLEX,
    TYPE_DUAL,
    TYPE_FLOAT,
    TYPE_HYPERCOMPLEX,
    TYPE_HYPERREAL,
    TYPE_NAMES,
    TYPE_NEGATIVE_REAL,
    TYPE_NEUTROSOPHIC,
    TYPE_NEUTROSOPHIC_BICOMPLEX,
    TYPE_NEUTROSOPHIC_COMPLEX,
    TYPE_OCTONION,
    TYPE_PATHION,
    TYPE_POSITIVE_REAL,
    TYPE_QUATERNION,
    TYPE_RATIONAL,
    TYPE_ROUTON,
    TYPE_SEDENION,
    TYPE_SPLIT_COMPLEX,
    TYPE_SUPERREAL,
    TYPE_TERNARY,
    TYPE_VOUDON,
    BaseNumber,
    BicomplexNumber,
    Chingon,
    ChingonNumber,
    CliffordNumber,
    Complex,
    ComplexNumber,
    DualNumber,
    HypercomplexNumber,
    HyperrealNumber,
    KececiAnalyzer,
    NeutrosophicBicomplexNumber,
    NeutrosophicComplexNumber,
    NeutrosophicNumber,
    Octonion,
    OctonionNumber,
    Pathion,
    PathionNumber,
    Quaternion,
    Real,
    Routon,
    RoutonNumber,
    Sedenion,
    SedenionNumber,
    SplitcomplexNumber,
    SuperrealNumber,
    TernaryNumber,
    Voudon,
    VoudonNumber,
    _apply_kececi_operation,
    _compute_gue_similarity,
    _divisible_by_numeric,
    _extract_numeric_part,
    _fallback_random_numbers,
    _find_kececi_prime_number,
    _find_kececi_zeta_zeros,
    _first_component_as_int,
    _float_mod_zero,
    _generate_ask_sequence_complete,
    _generate_default_value,
    _generate_detailed_sequence,
    _generate_kececi_sequence,
    _generate_sequence,
    _generate_simple_ask_sequence,
    _generate_simple_sequence,
    _generate_ternary_ask_sequence,
    _generate_ternary_operation_sequence,
    _get_default_hypercomplex,
    _get_default_value,
    _get_integer_representation,
    _gue_pair_correlation,
    _has_bicomplex_format,
    _has_comma_format,
    _is_complex_like,
    _is_divisible,
    _is_prime_decimal,
    _load_zeta_zeros,
    _make_hypercomplex_zero,
    _pair_correlation,
    _parse_bicomplex,
    _parse_chingon,
    _parse_clifford,
    _parse_complex,
    _parse_complex_like_string,
    _parse_components,
    _parse_dual,
    _parse_engineering_notation,
    _parse_fraction,
    _parse_hypercomplex,
    _parse_hyperreal,
    _parse_kececi_values,
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
    _parse_universal,
    _parse_voudon,
    _parse_with_fallback_simple,
    _parse_with_generic_fallback,
    _pca_var_sum,
    _plot_comparison,
    _plot_component_distribution,
    _safe_divide,
    _safe_float,
    _safe_float_convert,
    _safe_mod,
    _safe_power,
    analyze_all_types,
    analyze_kececi_primes,
    analyze_kececi_sequence,
    analyze_pair_correlation,
    apply_operation,
    apply_pca_clustering,
    chingon_cross,
    chingon_dot,
    chingon_eye,
    chingon_linspace,
    chingon_norm,
    chingon_normalize,
    chingon_ones,
    chingon_random,
    chingon_unit_vector,
    chingon_zeros,
    clean_sequence_for_plotting,
    comprehensive_cramer_test,
    convert_to_float,
    convert_to_plot_value,
    extract_clean_numbers,
    extract_complex_values,
    extract_fraction_values,
    extract_numeric_value,
    extract_numeric_values,
    extract_numericval,
    extract_values_for_plotting,
    find_cycle_with_earliest_start,
    find_first_numeric,
    find_first_occurrence,
    find_kececi_prime_number,
    find_kpn,
    find_period,
    find_repeating_pattern,
    find_stable_period,
    flatten_sequence,
    format_fraction,
    generate_geometric_kececi_art,
    generate_interactive_plot,
    generate_kececi_sequence,
    generate_kececi_sequence0,
    generate_kececi_vectorial,
    generate_octonion,
    get_interactive,
    get_operation_symbol,
    get_parser,
    get_quantum_random_numbers,
    get_quantum_random_numbers_with_retry,
    get_random_type,
    get_random_types_batch,
    get_with_params,
    hypercomplex_str,
    is_integer_multiple,
    is_multiple_with_tolerance,
    is_near_integer,
    is_neutrosophic_like,
    is_prime,
    is_prime_like,
    is_prime_value,
    is_quaternion_like,
    is_rational_multiple_with_maxden,
    is_super_real_expression,
    kececi_bicomplex_advanced,
    kececi_bicomplex_algorithm,
    kececi_numbers_complex,
    kececi_to_color,
    logger,
    make_unit,
    neutrosophic_f,
    neutrosophic_i,
    neutrosophic_one,
    neutrosophic_zero,
    normalize_super_real,
    parse_to_hyperreal,
    parse_to_neutrosophic,
    plot_kececi_with_pattern,
    plot_neutrosophic_complex,
    plot_numbers,
    plot_octonion_3d,
    plot_results,
    print_detailed_report,
    print_results,
    quaternion,
    random_start_add,
    run_comprehensive_analysis,
    run_cramer_test,
    run_test,
    run_variation_test,
    safe_add,
    safe_find_kpn,
    safe_parse,
    safe_plot_numbers,
    shorten_string,
    srandom_start_add,
    ternary_to_decimal,
    test_kececi_conjecture,
    test_variations,
    to_ternary,
    unified_generator,
)

# Public API - from * import ile erişilebilenler
__all__ = [
    # Ana fonksiyonlar
    "safe_find_kpn",
    "KececiAnalyzer",
    "analyze_all_types",
    "analyze_kececi_sequence",
    "analyze_pair_correlation",
    "apply_pca_clustering",
    "find_kececi_prime_number",
    "test_kececi_conjecture",
    # Sınıflar
    "ComplexNumber",
    "Quaternion",
    "OctonionNumber",
    "BicomplexNumber",
    "HyperrealNumber",
    "NeutrosophicNumber",
    "ChingonNumber",
    # Parser
    "parse_to_hyperreal",
    "parse_to_neutrosophic",
    # Yardımcılar
    "is_prime",
    "plot_numbers",
    "generate_interactive_plot",
    "get_interactive",
    "generate_kececi_sequence",
    "_generate_kececi_sequence",
    "_generate_ask_sequence_complete",
    #'generate_ask_sequence_complete',
    # Tip sabitleri
    "TYPE_COMPLEX",
    "TYPE_QUATERNION",
    "TYPE_OCTONION",
    # Metadata
    "__version__",
    "__author__",
    "__description__",
    "plot_kececi_with_pattern",
    "extract_numericval",
    "ternary_to_decimal",
    "shorten_string",
    "to_ternary",
    "find_repeating_pattern",
    "_fallback_random_numbers",
    "get_quantum_random_numbers",
    "get_quantum_random_numbers_with_retry",
    "generate_geometric_kececi_art",
    "kececi_to_color",
    "kececi_numbers_complex",
]

# Ensure metadata is explicitly included in __all__
__all__.extend(
    [
        "__version__",
        "__author__",
        "__license__",
        "__copyright__",
        "__email__",
        "__certificate__",
        "__bibtex__",
    ]
)

# ======================================================================
# DEPRECATION UTILITIES
# ======================================================================


def deprecated(reason: str) -> Callable:
    """Decorator to mark functions as deprecated."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__}() is deprecated and will be removed in a future version. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ======================================================================
# DEPRECATED FUNCTIONS
# ======================================================================


@deprecated(
    "Please use the alternative functions provided in the public API. KHA-256 is compatible with Python 3.11-3.15."
)
def legacy_function() -> None:
    """Legacy function scheduled for removal."""
    pass


__all__.append("legacy_function")


# Paket yüklendiğinde kısa bilgi
def _show_welcome():
    print(f"🌟 Kececi Numbers v{__version__} yüklendi")
    # print("Hiperkompleks sayılar ve Kececi varsayımları hazır!")


# _show_welcome()  # İlk importta çalıştırmak için (isteğe bağlı)
