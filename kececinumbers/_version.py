# -*- coding: utf-8 -*-
# _version.py

from typing import Dict, List, Tuple

# Package version following Semantic Versioning (SemVer)
# Format: MAJOR.MINOR.PATCH
# - MAJOR: Incompatible API changes
# - MINOR: Backwards-compatible functionality additions
# - PATCH: Backwards-compatible bug fixes
__version__: str = "1.0.4"

# License information
__license__: str = "AGPL-3.0-or-later"
__license_name__: str = "GNU Affero General Public License v3.0 or later"
__license_url__: str = "https://www.gnu.org/licenses/agpl-3.0.html"

# Package description
__description__: str = (
    "Keçeci Numbers: An Exploration of a Dynamic Sequence Across Diverse Number Sets."
)
__summary__: str = (
    "A cryptographic hash algorithm maximizing security with performance trade-offs"
)
__keywords__: List[str] = [
    "cryptography",
    "hash",
    "security",
    "hashing-algorithm",
    "crypto",
    "kececinumbers",
    "kececi",
]

# Author information
__author__: str = "Mehmet Keçeci"
__author_email__: str = "mkececi@yaani.com"
__maintainer__: str = "Mehmet Keçeci"
__maintainer_email__: str = "mkececi@yaani.com"

# Project URLs
__url__: str = "https://github.com/WhiteSymmetry/kececinumbers"
__docs__: str = "https://github.com/WhiteSymmetry/kececinumbers"  # Documentation URL
__source__: str = "https://github.com/WhiteSymmetry/kececinumbers"
__tracker__: str = "https://github.com/WhiteSymmetry/kececinumbers/issues"
__download_url__: str = "https://pypi.org/project/kececinumbers/"

# Package requirements
__python_requires__: str = ">=3.11"
__dependencies__: List[str] = [
    "python>=3.11",
    # Add runtime dependencies here if any
    # Example: "cryptography>=3.4.0",
]
__extras_require__: Dict[str, List[str]] = {
    "dev": [
        "pytest>=9.0.3",
        "pytest-cov>=7.1.0",
        "black>=26.3.1",
        "isort>=8.0.1",
        "mypy>=1.20.2",
        "flake8>=7.3.0",
    ],
    "docs": [
        "sphinx>=9.1.0",
        "sphinx-rtd-theme>3.1.0",
    ],
    "test": [
        "pytest>=9.0.3",
        "pytest-benchmark>=5.2.3",
    ],
}

# Classification information for PyPI
__classifiers__: List[str] = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Programming Language :: Python :: 3.15",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Security",
    "Topic :: Security :: Cryptography",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Minimum and maximum supported Python versions
__min_python_version__: Tuple[int, int] = (3, 11)
__max_python_version__: Tuple[int, int] = (3, 15)

# Package status
__status__: str = "Beta"
__release_date__: str = "2025"  # Initial release year

# Security information
__security_audit_status__: str = "Pending"
__security_audit_date__: Optional[str] = None

# Build and distribution information
__build__: int = 1  # Increment for each build of the same version


def get_version_info() -> Dict[str, str]:
    """Return a dictionary with all version information.

    Returns:
        Dict[str, str]: Dictionary containing version metadata.
    """
    return {
        "version": __version__,
        "license": __license__,
        "description": __description__,
        "author": __author__,
        "author_email": __author_email__,
        "url": __url__,
        "docs": __docs__,
        "python_requires": __python_requires__,
    }


def get_dependency_info() -> Dict[str, List[str]]:
    """Return dependency information.

    Returns:
        Dict[str, List[str]]: Dictionary containing dependency information.
    """
    return {
        "install": __dependencies__,
        "extras": __extras_require__,
    }


def is_python_version_supported(major: int, minor: int) -> bool:
    """Check if a given Python version is supported.

    Args:
        major: Python major version (e.g., 3)
        minor: Python minor version (e.g., 11)

    Returns:
        bool: True if the version is supported, False otherwise.
    """
    min_supported = __min_python_version__
    max_supported = __max_python_version__

    version = (major, minor)
    return min_supported <= version <= max_supported


def get_supported_python_versions() -> List[str]:
    """Get a list of supported Python versions as strings.

    Returns:
        List[str]: List of supported Python versions.
    """
    versions = []
    for minor in range(__min_python_version__[1], __max_python_version__[1] + 1):
        versions.append(f"3.{minor}")
    return versions


# Version history for changelog purposes
__version_history__: List[Dict[str, str]] = [
    {
        "version": "1.0.1",
        "date": "2026-05-26",
        "changes": [
            "minor",
        ],
    },
]

# Export all public variables
__all__ = [
    "__version__",
    "__license__",
    "__license_name__",
    "__license_url__",
    "__description__",
    "__summary__",
    "__keywords__",
    "__author__",
    "__author_email__",
    "__maintainer__",
    "__maintainer_email__",
    "__url__",
    "__docs__",
    "__source__",
    "__tracker__",
    "__download_url__",
    "__python_requires__",
    "__dependencies__",
    "__extras_require__",
    "__classifiers__",
    "__min_python_version__",
    "__max_python_version__",
    "__status__",
    "__release_date__",
    "__security_audit_status__",
    "__security_audit_date__",
    "__build__",
    "__version_history__",
    "get_version_info",
    "get_dependency_info",
    "is_python_version_supported",
    "get_supported_python_versions",
]
