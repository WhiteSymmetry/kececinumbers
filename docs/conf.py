# Project Information
project = 'kececinumbers'
author = 'Mehmet Keçeci'
copyright = '2025, Mehmet Keçeci'

# Version Management
# from setuptools_scm import get_version
# version = get_version(root='..', relative_to=__file__)
# Sürüm Bilgisi (setuptools_scm kullanmıyorsanız sabit olarak tanımlayın)
# Gerçek sürümü modülden al (eğer mümkünse)
try:
    from kececinumbers import __version__
    version = __version__
    release = __version__
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import __version__ from kececinumbers: {e}")
    # Varsayılan değerler korunur
#version = '0.4.1'  # Replace with your actual version number
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
