# noxfile.py

import nox
import os

# Test edilecek Python versiyonlarını listele
PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]

# Varsayılan olarak çalıştırılacak oturumları belirle
# (Terminalde sadece 'nox' yazıldığında bu oturumlar çalışır)
nox.options.sessions = ["lint", "tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """
    Run the test suite for all specified Python versions.
    Bu oturum 'nox -s tests-3.11' gibi komutlarla tetiklenir.
    """
    # 1. Gerekli bağımlılıkları kur
    #    'pytest', 'pytest-mock' ve projenin kendisini kur ('-e .' ile).
    # pyproject.toml yerine bağımlılıkları burada manuel olarak belirt
    session.install(
        "pytest",
        "pytest-cov",
        "pytest-mock",
        # ... diğer tüm test bağımlılıkları
    )
    session.install("-e", ".[test]") # pyproject.toml'daki [project.optional-dependencies] test grubunu kurar
    
    # Artık tüm kütüphaneler kurulu olduğu için pytest çalışabilir.
    # 2. Pytest'i çalıştır
    #    --cov ile kod kapsamı (code coverage) raporu oluştur.
    session.run("pytest", "--cov=kececinumbers", "--cov-report=xml")

@nox.session(python="3.11") # Linting genellikle tek bir versiyonda yapılır
def lint(session):
    """
    Run linters to check code style and quality.
    """
    # Linting araçlarını kur
    session.install("ruff")
    
    # Kodu kontrol et
    session.run("ruff", "check", "kececinumbers", "tests")
    session.run("ruff", "format", "--check", "kececinumbers", "tests")


# --- Proje Yapınıza Göre Eklemeler ---
# Eğer 'pyproject.toml' dosyanızda test bağımlılıkları tanımlı değilse,
# tests oturumundaki install satırını şu şekilde değiştirebilirsiniz:
#
# @nox.session(python=PYTHON_VERSIONS)
# def tests(session):
#     # Bağımlılıkları manuel olarak kur
#     session.install("pytest", "pytest-mock", "numpy", "matplotlib")
#     # Projeyi düzenlenebilir modda kur
#     session.install("-e", ".")
#     session.run("pytest", "--cov=kececinumbers", "--cov-report=xml")
