# Keçeci Numbers: Keçeci Sayıları

[![PyPI version](https://badge.fury.io/py/kececinumbers.svg)](https://badge.fury.io/py/kececinumbers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15377659.svg)](https://doi.org/10.5281/zenodo.15377659)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.14.1-blue)](https://doi.org/10.48546/workflowhub.datafile.14.1)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/version.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/platforms.svg)](https://anaconda.org/bilgi/kececinumbers)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececinumbers/badges/license.svg)](https://anaconda.org/bilgi/kececinumbers)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececinumbers/badge/?0.2.0=main)](https://kececinumbers.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10536/badge)](https://www.bestpractices.dev/projects/10536)

[![Python CI](https://github.com/WhiteSymmetry/kececinumbers/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/kececinumbers/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/kececinumbers/graph/badge.svg?token=0X78S7TL0W)](https://codecov.io/gh/WhiteSymmetry/kececinumbers)
[![Documentation Status](https://readthedocs.org/projects/kececinumbers/badge/?version=latest)](https://kececinumbers.readthedocs.io/en/latest/)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececinumbers/HEAD)
[![PyPI version](https://badge.fury.io/py/kececinumbers.svg)](https://badge.fury.io/py/kececinumbers)
[![PyPI Downloads](https://static.pepy.tech/badge/kececinumbers)](https://pepy.tech/projects/kececinumbers)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) 

---

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI</td>
            <td style="text-align: center;">
                <a href="https://pypi.org/project/kececinumbers/">
                    <img src="https://badge.fury.io/py/kececinumbers.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">Conda</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/bilgi/kececinumbers">
                    <img src="https://anaconda.org/bilgi/kececinumbers/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">DOI</td>
            <td style="text-align: center;">
                <a href="https://doi.org/10.5281/zenodo.15377659">
                    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15377659.svg" alt="DOI" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">License: MIT</td>
            <td style="text-align: center;">
                <a href="https://opensource.org/licenses/MIT">
                    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>

---

## Description / Açıklama

**Keçeci Numbers (Keçeci Sayıları)**: Keçeci Numbers; An Exploration of a Dynamic Sequence Across Diverse Number Sets: This work introduces a novel numerical sequence concept termed "Keçeci Numbers." Keçeci Numbers are a dynamic sequence generated through an iterative process, originating from a specific starting value and an increment value. In each iteration, the increment value is added to the current value, and this "added value" is recorded in the sequence. Subsequently, a division operation is attempted on this "added value," primarily using the divisors 2 and 3, with the choice of divisor depending on the one used in the previous step. If division is successful, the quotient becomes the next element in the sequence. If the division operation fails, the primality of the "added value" (or its real/scalar part for complex/quaternion numbers, or integer part for rational numbers) is checked. If it is prime, an "Augment/Shrink then Check" (ASK) rule is invoked: a type-specific unit value is added or subtracted (based on the previous ASK application), this "modified value" is recorded in the sequence, and the division operation is re-attempted on it. If division fails again, or if the number is not prime, the "added value" (or the "modified value" post-ASK) itself becomes the next element in the sequence. This mechanism is designed to be applicable across various number sets, including positive and negative real numbers, complex numbers, floating-point numbers, rational numbers, and quaternions. The increment value, ASK unit, and divisibility checks are appropriately adapted for each number type. This flexibility of Keçeci Numbers offers rich potential for studying their behavior in different numerical systems. The patterns exhibited by the sequences, their convergence/divergence properties, and potential for chaotic behavior may constitute interesting research avenues for advanced mathematical analysis and number theory applications. This study outlines the fundamental generation mechanism of Keçeci Numbers and their initial behaviors across diverse number sets.

---

## Installation / Kurulum

```bash
conda install bilgi::kececinumbers -y

pip install kececinumbers
```
https://anaconda.org/bilgi/kececinumbers

https://pypi.org/project/kececinumbers/

https://github.com/WhiteSymmetry/kececinumbers

https://zenodo.org/records/15377660

https://zenodo.org/records/

---

## Usage / Kullanım

### Example

```python
import matplotlib.pyplot as plt
import kececinumbers as kn

print("--- Interactive Test ---")

# Adım 1: get_interactive'ten tüm verileri al
# Not: Fonksiyon artık birden fazla değer döndürüyor.
interactive_results = kn.get_interactive()

# Fonksiyon bir dizi döndürdüyse (başarılıysa) devam et
if interactive_results and interactive_results[0]:
    # Dönen değerleri değişkenlere ata
    seq_interactive, type_choice, start_val, add_val, steps = interactive_results
    
    # Tip numarasını isme çevirelim
    type_names = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutro-Complex", "Hyperreal", 
        "Bicomplex", "Neutro-Bicomplex"
    ]
    type_name = type_names[type_choice - 1]

    # Adım 2: Ayrıntılı raporu yazdır
    params = {
        'type_choice': type_choice,
        'type_name': type_name,
        'start_val': start_val,
        'add_val': add_val,
        'steps': steps
    }
    kn.print_detailed_report(seq_interactive, params)
    
    # Adım 3: Grafiği SADECE BİR KERE çizdir
    print("\nDisplaying plot...")
    plot_title = f"Interactive Keçeci Sequence ({type_name})"
    kn.plot_numbers(seq_interactive, plot_title)
    plt.show()

else:
    print("Sequence generation was cancelled or failed.")
```

```python
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from fractions import Fraction
import quaternion # pip install numpy numpy-quaternion
import kececinumbers as kn

# Matplotlib grafiklerinin notebook içinde gösterilmesini sağla
%matplotlib inline

print("Trying interactive mode (will prompt for input in the console/output area)...")
interactive_sequence = kn.get_interactive()
if interactive_sequence:
    kn.plot_numbers(interactive_sequence, title="Keçeci Numbers")

print("Done with examples.")
print("Keçeci Numbers Module Loaded.")
print("This module provides functions to generate and plot Keçeci Numbers.")
print("Example: Use 'import kececinumbers as kn' in your script/notebook.")
print("\nAvailable functions:")
print("- kn.get_interactive()")
print("- kn.get_with_params(kececi_type, iterations, ...)")
print("- kn.get_random_type(iterations, ...)")
print("- kn.plot_numbers(sequence, title)")
print("- kn.unified_generator(...) (low-level)")
print("\nAccess definitions with: kn.DEFINITIONS")
print("\nAccess type constants like: kn.TYPE_COMPLEX")
```
---
Trying interactive mode (will prompt for input in the console/output area)...

Keçeci Number Types:

1: Positive Real Numbers (Integer: e.g., 1)

2: Negative Real Numbers (Integer: e.g., -3)

3: Complex Numbers (e.g., 3+4j)

4: Floating-Point Numbers (e.g., 2.5)

5: Rational Numbers (e.g., 3/2, 5)

6: Quaternions (scalar start input becomes q(s,s,s,s): e.g.,  1 or 2.5)

7: Neutrosophic     

8: Neutro-Complex   

9: Hyperreal
 
10: Bicomplex        

11: Neutro-Bicomplex

Please select Keçeci Number Type (1-11):  1

Enter the starting number (e.g., 0 or 2.5, complex:3+4j, rational: 3/4, quaternions: 1)  :  0

Enter the base scalar value for increment (e.g., 9):  9

Enter the number of iterations (positive integer: e.g., 30):  30

---
![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-1.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-2.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-3.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-4.png?raw=true)

![Keçeci Numbers Example](https://github.com/WhiteSymmetry/kececinumbers/blob/main/examples/kn-5.png?raw=true)

---
# Keçeci Prime Number

```python
import matplotlib.pyplot as plt
import kececinumbers as kn


print("--- Interactive Test ---")
seq_interactive = kn.get_interactive()
if seq_interactive:
    kn.plot_numbers(seq_interactive, "Keçeci Numbers")

print("\n--- Random Type Test (60 Keçeci Steps) ---")
# num_iterations burada Keçeci adımı sayısıdır
seq_random = kn.get_random_type(num_iterations=60) 
if seq_random:
    kn.plot_numbers(seq_random, "Random Type Keçeci Numbers")

print("\n--- Fixed Params Test (Complex, 60 Keçeci Steps) ---")
seq_fixed = kn.get_with_params(
    kececi_type_choice=kn.TYPE_COMPLEX, 
    iterations=60, 
    start_value_raw="1+2j", 
    add_value_base_scalar=3.0
)
if seq_fixed:
    kn.plot_numbers(seq_fixed, "Fixed Params (Complex) Keçeci Numbers")

# İsterseniz find_kececi_prime_number'ı ayrıca da çağırabilirsiniz:
if seq_fixed:
    kpn_direct = kn.find_kececi_prime_number(seq_fixed)
    if kpn_direct is not None:
        print(f"\nDirect call to find_kececi_prime_number for fixed numbers: {kpn_direct}")
```

Generated Keçeci Sequence (first 20 of 121): [4, 11, 12, 4, 11, 10, 5, 12, 4, 11, 12, 6, 13, 12, 4, 11, 12, 6, 13, 12]...
Keçeci Prime Number for this sequence: 11

--- Random Type Test (60 Keçeci Steps) ---

Randomly selected Keçeci Number Type: 1 (Positive Integer)

Generated Keçeci Sequence (using get_with_params, first 20 of 61): [0, 9, 3, 12, 6, 15, 5, 14, 7, 16, 8, 17, 18, 6, 15, 5, 14, 7, 16, 8]...
Keçeci Prime Number for this sequence: 17

---

## License / Lisans

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@misc{kececi_2025_15377659,
  author       = {Keçeci, Mehmet},
  title        = {kececinumbers},
  month        = may,
  year         = 2025,
  publisher    = {PyPI, Anaconda, Github, Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.15377659},
  url          = {https://doi.org/10.5281/zenodo.15377659},
}
```

### APA

```
Keçeci, M. (2025). Geometric Interpretations of Keçeci Numbers with Neutrosophic and Hyperreal Numbers. Zenodo. https://doi.org/10.5281/zenodo.16344232

Keçeci, M. (2025). Keçeci Sayılarının Nötrosofik ve Hipergerçek Sayılarla Geometrik Yorumlamaları. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.16343568

Keçeci, M. (2025). kececinumbers [Data set]. WorkflowHub. https://doi.org/10.48546/workflowhub.datafile.14.1

Keçeci, M. (2025). kececinumbers. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15377659

Keçeci, M. (2025). Keçeci Numbers and the Keçeci Prime Number: A Potential Number Theoretic Exploratory Tool. https://doi.org/10.5281/zenodo.15381698

Keçeci, M. (2025). Diversity of Keçeci Numbers and Their Application to Prešić-Type Fixed-Point Iterations: A Numerical Exploration. https://doi.org/10.5281/zenodo.15481711

Keçeci, M. (2025). Keçeci Numbers and the Keçeci Prime Number. Authorea. June 02, 2025. https://doi.org/10.22541/au.174890181.14730464/v1

Keçeci, M. (2025, May 11). Keçeci numbers and the Keçeci prime number: A potential number theoretic exploratory tool. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15381697
```

### Chicago
```
Keçeci, Mehmet. kececinumbers [Data set]. WorkflowHub, 2025. https://doi.org/10.48546/workflowhub.datafile.14.1

Keçeci, Mehmet. "kececinumbers". Open Science Articles (OSAs), Zenodo, 01 May 2025. https://doi.org/10.5281/zenodo.15377659

Keçeci, Mehmet. "Keçeci Numbers and the Keçeci Prime Number: A Potential Number Theoretic Exploratory Tool", 11 Mayıs 2025. https://doi.org/10.5281/zenodo.15381698

Keçeci, Mehmet. "Diversity of Keçeci Numbers and Their Application to Prešić-Type Fixed-Point Iterations: A Numerical Exploration". https://doi.org/10.5281/zenodo.15481711

Keçeci, Mehmet. "Keçeci Numbers and the Keçeci Prime Number". Authorea. June 02, 2025. https://doi.org/10.22541/au.174890181.14730464/v1

Keçeci, Mehmet. Keçeci numbers and the Keçeci prime number: A potential number theoretic exploratory tool. Open Science Articles (OSAs), Zenodo. 2025. https://doi.org/10.5281/zenodo.15381697
```

---

# Keçeci Conjecture: Keçeci Varsayımı, Keçeci-Vermutung, Conjecture de Keçeci, Гипотеза Кечеджи, Keçeci Hipoteza, 凯杰西猜想, Keçeci Xiǎngcāng, ケジェジ予想, Keçeci Yosō, Keçeci Huds, Keçeci Hudsiye, Keçeci Hudsia, حدس كَچَه جِي ,حدس کچه جی, کچہ جی حدسیہ
---

### 🇹🇷 **Türkçe**  
```text
## Keçeci Varsayımı (Keçeci Conjecture) - Önerilen

Her Keçeci Sayı türü için, `unified_generator` fonksiyonu tarafından oluşturulan dizilerin, sonlu adımdan sonra periyodik bir yapıya veya tekrar eden bir asal temsiline (Keçeci Asal Sayısı, KPN) yakınsadığı sanılmaktadır. Bu davranış, Collatz Varsayımı'nın çoklu cebirsel sistemlere genişletilmiş bir hali olarak değerlendirilebilir.

Henüz kanıtlanmamıştır ve bu modül bu varsayımı test etmek için bir çerçeve sunar.
```

---

### 🇬🇧 **İngilizce (English)**  
```text
## Keçeci Conjecture - Proposed

For every Keçeci Number type, sequences generated by the `unified_generator` function are conjectured to converge to a periodic structure or a recurring prime representation (Keçeci Prime Number, KPN) in finitely many steps. This behavior can be viewed as a generalization of the Collatz Conjecture to multiple algebraic systems.

It remains unproven, and this module provides a framework for testing the conjecture.
```

---

### 🇩🇪 **Almanca (Deutsch)**  
```text
## Keçeci-Vermutung – Vorgeschlagen

Es wird vermutet, dass die vom `unified_generator` erzeugten Sequenzen für jeden Keçeci-Zahl-Typ nach endlich vielen Schritten gegen eine periodische Struktur oder eine wiederkehrende Primdarstellung (Keçeci-Primzahl, KPN) konvergieren. Dieses Verhalten kann als eine Erweiterung der Collatz-Vermutung auf mehrere algebraische Systeme betrachtet werden.

Die Vermutung ist bisher unbewiesen, und dieses Modul bietet einen Rahmen, um sie zu untersuchen.
```

---

### 🇫🇷 **Fransızca (Français)**  
```text
## Conjecture de Keçeci – Proposée

On conjecture que, pour chaque type de nombre Keçeci, les suites générées par la fonction `unified_generator` convergent, en un nombre fini d'étapes, vers une structure périodique ou une représentation première récurrente (Nombre Premier Keçeci, KPN). Ce comportement peut être vu comme une généralisation de la conjecture de Collatz à divers systèmes algébriques.

Elle n'est pas encore démontrée, et ce module fournit un cadre pour la tester.
```


---

### 🇷🇺 **Rusça (Русский)**  
```text
## Гипотеза Кечеджи — Предложенная

Предполагается, что последовательности, генерируемые функцией `unified_generator` для каждого типа чисел Кечеджи, сходятся к периодической структуре или повторяющемуся простому представлению (Простое число Кечеджи, KPN) за конечное число шагов. Это поведение можно рассматривать как обобщение гипотезы Коллатца на многомерные алгебраические системы.

Гипотеза пока не доказана, и данный модуль предоставляет среду для её проверки.
```

---

### 🇨🇳 **Çince (中文 - Basitleştirilmiş)**  
```text
## 凯杰西猜想（Keçeci Conjecture）— 提出

据推测，对于每一种凯杰西数类型，由 `unified_generator` 函数生成的序列将在有限步内收敛到周期性结构或重复的素数表示（凯杰西素数，KPN）。这种行为可视为科拉茨猜想在多种代数系统中的推广。

该猜想尚未被证明，本模块提供了一个用于测试该猜想的框架。
```

---

### 🇯🇵 **Japonca (日本語)**  
```text
## ケジェジ予想（Keçeci Conjecture）― 提案

すべてのケジェジ数型に対して、`unified_generator` 関数によって生成される数列は、有限回のステップ後に周期的な構造または繰り返し現れる素数表現（ケジェジ素数、KPN）に収束すると考えられている。この振る舞いは、コラッツ予想を複数の代数系へと拡張したものと見なせる。

この予想は未だ証明されておらず、本モジュールはその検証のための枠組みを提供する。
```

---

### 🇸🇦 **Arapça (العربية) – Güncellenmiş: "كَچَه جِي"**
```text
## حدس كَچَه جِي (Keçeci Conjecture) — مقترح

يُفترض أن المتتاليات التي يولدها الدالة `unified_generator` لكل نوع من أعداد كَچَه جِي تتقارب، بعد عدد محدود من الخطوات، إلى بنية دورية أو إلى تمثيل أولي متكرر (العدد الأولي لكَچَه جِي، KPN). يمكن اعتبار هذا السلوك تعميمًا لحدس كولاتز على نظم جبرية متعددة.

ما زال هذا الحدس غير مثبت، ويقدم هذا الوحدة إطارًا لاختباره.
```

---

### 🇮🇷 **Farsça (فارسی) – Güncellenmiş: "کچه جی"**
```text
## حدس کچه جی (Keçeci Conjecture) — پیشنهادی

گمان می‌رود که دنباله‌های تولید شده توسط تابع `unified_generator` برای هر نوع از اعداد کچه جی، پس از تعداد محدودی گام، به یک ساختار تناوبی یا نمایش اول تکراری (عدد اول کچه جی، KPN) همگرا شوند. این رفتار را می‌توان تعمیمی از حدس کولاتز به سیستم‌های جبری چندگانه دانست.

این حدس هنوز اثبات نشده است و این ماژول چارچوبی برای آزمودن آن فراهم می‌کند.
```

---

### 🇵🇰 **Urduca (اردو) – Güncellenmiş: "کچہ جی"**
```text
## کچہ جی حدسیہ (Keçeci Conjecture) — تجویز شدہ

ہر قسم کے کچہ جی نمبر کے لیے، یہ تجویز کیا جاتا ہے کہ `unified_generator` فنکشن کے ذریعے تیار کردہ ترادف محدود مراحل کے بعد ایک دوری ساخت یا دہرائے گئے مفرد نمائندگی (کچہ جی مفرد نمبر، KPN) کی طرف مائل ہوتا ہے۔ اس رویے کو کولاتز حدسیہ کی متعدد الجبری نظاموں تک توسیع کے طور پر دیکھا جا سکتا ہے۔

ابھی تک یہ ثابت نہیں ہوا ہے، اور یہ ماڈیول اس حدسیہ کی جانچ کے لیے ایک فریم ورک فراہم کرتا ہے۔
```


