# Project Information
project = 'kececinumbers'
author = 'Mehmet Keçeci'
copyright = f"{datetime.now().year}, {author}"

# Version Management
# from setuptools_scm import get_version
# version = get_version(root='..', relative_to=__file__)
# Sürüm Bilgisi (setuptools_scm kullanmıyorsanız sabit olarak tanımlayın)
# Gerçek sürümü modülden al (eğer mümkünse)

# ======================================================================
# VERSION DETECTION (3 Aşamalı Güvenli Yöntem)
# ======================================================================
def get_version():
    """Versiyonu 3 farklı yöntemle almaya çalışır."""
    
    # 1. YÖNTEM: Paket kuruluysa importlib.metadata'den oku
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("kececinumbers")
    except Exception:
        pass
    
    # 2. YÖNTEM: __init__.py dosyasını regex ile oku (import etmeden!)
    try:
        init_path = os.path.join(os.path.abspath('../..'), 'kececinumbers', '__init__.py')
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
    except Exception:
        pass
    
    # 3. YÖNTEM: Fallback - Sabit değer
    return "1.0.2"

# Versiyonu al ve ata
release = get_version()
version = release  # version = kısa versiyon (örn: "0.3"), release = tam versiyon

print(f"📦 kececinumbers Dokümantasyon Sürümü: {release}")
"""
version = None
release = None

try:
    from kececinumbers import __version__ as pkg_version
    version = pkg_version
    release = pkg_version
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import __version__ from kececinumbers: {e}")
"""
"""    
try:
    from kececinumbers import __version__
    version = __version__
    release = __version__
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import __version__ from kececinumbers: {e}")
"""
    # Varsayılan değerler korunur
#version = '1.0.2'  # Replace with your actual version number
#release = version

# General Configuration
master_doc = 'index'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML Output Configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'  # Optional: Add your project logo
html_favicon = '_static/favicon.ico'  # Optional: Add a favicon
