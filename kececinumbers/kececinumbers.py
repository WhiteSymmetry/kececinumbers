# -*- coding: utf-8 -*-
"""
Keçeci Numbers Module (kececinumbers.py)

This module provides a comprehensive framework for generating, analyzing, and
visualizing Keçeci Numbers across various number systems. It supports 22
distinct types, from standard integers and complex numbers to more exotic
constructs like neutrosophic and bicomplex numbers.

The core of the module is the `unified_generator`, which implements the
specific algorithm for generating Keçeci Number sequences. High-level functions
are available for easy interaction, parameter-based generation, and plotting.

Key Features:
- Generation of 22 types of Keçeci Numbers.
- A robust, unified algorithm for all number types.
- Helper functions for mathematical properties like primality and divisibility.
- Advanced plotting capabilities tailored to each number system.
- Functions for interactive use or programmatic integration.
---

Keçeci Conjecture: Keçeci Varsayımı, Keçeci-Vermutung, Conjecture de Keçeci, Гипотеза Кечеджи, 凯杰西猜想, ケジェジ予想, Keçeci Huds, Keçeci Hudsiye, Keçeci Hudsia, [...]

Keçeci Varsayımı (Keçeci Conjecture) - Önerilen

Her Keçeci Sayı türü için, `unified_generator` fonksiyonu tarafından oluşturulan dizilerin, sonlu adımdan sonra periyodik bir yapıya veya tekrar eden bir asal temsiline (Keçeci Asal Sayısı[...]

Henüz kanıtlanmamıştır ve bu modül bu varsayımı test etmek için bir çerçeve sunar.
"""

# --- Standard Library Imports ---
from __future__ import annotations
from abc import ABC, abstractmethod
import collections
from dataclasses import dataclass
from fractions import Fraction
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numbers
#from numbers import Real
import numpy as np 
import random
import re
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sympy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence, Union
import logging

# Module logger — library code should not configure logging handlers.
logger = logging.getLogger(__name__)
"""
try:
    # numpy-quaternion kütüphanesinin sınıfını yüklemeye çalış. Artık bu modüle ihtiyaç kalmadı
    # conda install -c conda-forge quaternion # pip install numpy-quaternion
    from quaternion import quaternion as quaternion  # type: ignore
except Exception:
    # Eğer yoksa `quaternion` isimli sembolü None yap, kodun diğer yerleri bunu kontrol edebilir
    quaternion = None
    logger.warning("numpy-quaternion paketine ulaşılamadı — quaternion tip desteği devre dışı bırakıldı.")
"""

# ==============================================================================
# --- MODULE CONSTANTS: Keçeci NUMBER TYPES ---
# ==============================================================================
TYPE_POSITIVE_REAL = 1
TYPE_NEGATIVE_REAL = 2
TYPE_COMPLEX = 3
TYPE_FLOAT = 4
TYPE_RATIONAL = 5
TYPE_QUATERNION = 6
TYPE_NEUTROSOPHIC = 7
TYPE_NEUTROSOPHIC_COMPLEX = 8
TYPE_HYPERREAL = 9
TYPE_BICOMPLEX = 10
TYPE_NEUTROSOPHIC_BICOMPLEX = 11
TYPE_OCTONION = 12
TYPE_SEDENION = 13
TYPE_CLIFFORD = 14
TYPE_DUAL = 15
TYPE_SPLIT_COMPLEX = 16
TYPE_PATHION = 17
TYPE_CHINGON = 18
TYPE_ROUTON = 19
TYPE_VOUDON = 20
TYPE_SUPERREAL = 21
TYPE_TERNARY = 22

Number = Union[int, float, complex]


# ==============================================================================
# --- CUSTOM NUMBER CLASS DEFINITIONS ---
# ==============================================================================
class quaternion:
    """
    Kuaterniyon sınıfı: w + xi + yj + zk formatında
    
    Attributes:
        w: Reel kısım
        x: i bileşeni
        y: j bileşeni
        z: k bileşeni
    """
    
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Kuaterniyon oluşturur.
        
        Args:
            w: Reel kısım
            x: i bileşeni
            y: j bileşeni
            z: k bileşeni
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @classmethod
    def from_axis_angle(cls, axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float) -> 'quaternion':
        """
        Eksen-açı gösteriminden kuaterniyon oluşturur.
        
        Args:
            axis: Dönme ekseni (3 boyutlu vektör)
            angle: Radyan cinsinden dönme açısı
        
        Returns:
            quaternion: Kuaterniyon nesnesi
        """
        axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm == 0:
            raise ValueError("Eksen vektörü sıfır olamaz")
        
        axis = axis / axis_norm
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        
        return cls(
            w=math.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
        )
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float, 
                   order: str = 'zyx') -> 'quaternion':
        """
        Euler açılarından kuaterniyon oluşturur.
        
        Args:
            roll: X ekseni etrafında dönme (radyan)
            pitch: Y ekseni etrafında dönme (radyan)
            yaw: Z ekseni etrafında dönme (radyan)
            order: Dönme sırası ('zyx', 'xyz', 'yxz', vb.)
        
        Returns:
            quaternion: Kuaterniyon nesnesi
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        if order == 'zyx':  # Yaw, Pitch, Roll
            w = cy * cp * cr + sy * sp * sr
            x = cy * cp * sr - sy * sp * cr
            y = sy * cp * sr + cy * sp * cr
            z = sy * cp * cr - cy * sp * sr
        elif order == 'xyz':  # Roll, Pitch, Yaw
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
        else:
            raise ValueError(f"Desteklenmeyen dönme sırası: {order}")
        
        return cls(w, x, y, z)
    
    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> 'quaternion':
        """
        Dönüşüm matrisinden kuaterniyon oluşturur.
        
        Args:
            R: 3x3 dönüşüm matrisi
        
        Returns:
            quaternion: Kuaterniyon nesnesi
        """
        if R.shape != (3, 3):
            raise ValueError("Matris 3x3 boyutunda olmalıdır")
        
        trace = np.trace(R)
        
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return cls(w, x, y, z).normalized()
    
    def conjugate(self) -> 'quaternion':
        """Kuaterniyonun eşleniğini döndürür."""
        return quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        """Kuaterniyonun normunu döndürür."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> 'quaternion':
        """Normalize edilmiş kuaterniyonu döndürür."""
        n = self.norm()
        if n == 0:
            return quaternion(1, 0, 0, 0)
        return quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def inverse(self) -> 'quaternion':
        """Kuaterniyonun tersini döndürür."""
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq == 0:
            return quaternion(1, 0, 0, 0)
        conj = self.conjugate()
        return quaternion(conj.w/norm_sq, conj.x/norm_sq, conj.y/norm_sq, conj.z/norm_sq)
    
    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Kuaterniyonu eksen-açı gösterimine dönüştürür.
        
        Returns:
            Tuple[np.ndarray, float]: (eksen, açı)
        """
        if abs(self.w) > 1:
            q = self.normalized()
        else:
            q = self
        
        angle = 2 * math.acos(q.w)
        
        if abs(angle) < 1e-10:
            return np.array([1.0, 0.0, 0.0]), 0.0
        
        s = math.sqrt(1 - q.w**2)
        if s < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([q.x/s, q.y/s, q.z/s])
        
        return axis, angle
    
    def to_euler(self, order: str = 'zyx') -> Tuple[float, float, float]:
        """
        Kuaterniyonu Euler açılarına dönüştürür.
        
        Args:
            order: Dönme sırası
        
        Returns:
            Tuple[float, float, float]: (roll, pitch, yaw)
        """
        q = self.normalized()
        
        if order == 'zyx':  # Yaw, Pitch, Roll
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
        else:
            raise ValueError(f"Desteklenmeyen dönme sırası: {order}")
    
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Kuaterniyonu dönüşüm matrisine dönüştürür.
        
        Returns:
            np.ndarray: 3x3 dönüşüm matrisi
        """
        q = self.normalized()
        
        # 3x3 dönüşüm matrisi
        R = np.zeros((3, 3))
        
        # Matris elemanlarını hesapla
        R[0, 0] = 1 - 2*(q.y**2 + q.z**2)
        R[0, 1] = 2*(q.x*q.y - q.w*q.z)
        R[0, 2] = 2*(q.x*q.z + q.w*q.y)
        
        R[1, 0] = 2*(q.x*q.y + q.w*q.z)
        R[1, 1] = 1 - 2*(q.x**2 + q.z**2)
        R[1, 2] = 2*(q.y*q.z - q.w*q.x)
        
        R[2, 0] = 2*(q.x*q.z - q.w*q.y)
        R[2, 1] = 2*(q.y*q.z + q.w*q.x)
        R[2, 2] = 1 - 2*(q.x**2 + q.y**2)
        
        return R
    
    def rotate_vector(self, v: Union[List[float], Tuple[float, float, float], np.ndarray]) -> np.ndarray:
        """
        Vektörü kuaterniyon ile döndürür.
        
        Args:
            v: Döndürülecek 3 boyutlu vektör
        
        Returns:
            np.ndarray: Döndürülmüş vektör
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (3,):
            raise ValueError("Vektör 3 boyutlu olmalıdır")
        
        q = self.normalized()
        q_vec = np.array([q.x, q.y, q.z])
        q_w = q.w
        
        # Kuaterniyon çarpımı ile döndürme
        v_rot = v + 2 * np.cross(q_vec, np.cross(q_vec, v) + q_w * v)
        return v_rot
    
    def slerp(self, other: 'quaternion', t: float) -> 'quaternion':
        """
        Küresel lineer interpolasyon (SLERP) yapar.
        
        Args:
            other: Hedef kuaterniyon
            t: İnterpolasyon parametresi [0, 1]
        
        Returns:
            quaternion: İnterpole edilmiş kuaterniyon
        """
        if t <= 0:
            return self.normalized()
        if t >= 1:
            return other.normalized()
        
        q1 = self.normalized()
        q2 = other.normalized()
        
        # Nokta çarpım
        cos_half_theta = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
        
        # Eğer q1 ve q2 aynı yöndeyse
        if abs(cos_half_theta) >= 1.0:
            return q1
        
        # Eğer negatif nokta çarpım, kuaterniyonları ters çevir
        if cos_half_theta < 0:
            q2 = quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            cos_half_theta = -cos_half_theta
        
        half_theta = math.acos(cos_half_theta)
        sin_half_theta = math.sqrt(1.0 - cos_half_theta**2)
        
        if abs(sin_half_theta) < 1e-10:
            return quaternion(
                q1.w * 0.5 + q2.w * 0.5,
                q1.x * 0.5 + q2.x * 0.5,
                q1.y * 0.5 + q2.y * 0.5,
                q1.z * 0.5 + q2.z * 0.5
            ).normalized()
        
        ratio_a = math.sin((1 - t) * half_theta) / sin_half_theta
        ratio_b = math.sin(t * half_theta) / sin_half_theta
        
        return quaternion(
            q1.w * ratio_a + q2.w * ratio_b,
            q1.x * ratio_a + q2.x * ratio_b,
            q1.y * ratio_a + q2.y * ratio_b,
            q1.z * ratio_a + q2.z * ratio_b
        ).normalized()
    
    def __add__(self, other: 'quaternion') -> 'quaternion':
        """Kuaterniyon toplama."""
        if isinstance(other, quaternion):
            return quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        raise TypeError("Sadece quaternion ile toplanabilir")
    
    def __sub__(self, other: 'quaternion') -> 'quaternion':
        """Kuaterniyon çıkarma."""
        if isinstance(other, quaternion):
            return quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Sadece quaternion ile çıkarılabilir")
    
    def __mul__(self, other: Union['quaternion', float, int]) -> 'quaternion':
        """Kuaterniyon çarpma veya skaler çarpma."""
        if isinstance(other, (int, float)):
            return quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        elif isinstance(other, quaternion):
            # Hamilton çarpımı
            w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            return quaternion(w, x, y, z)
        raise TypeError("Sadece quaternion veya skaler ile çarpılabilir")
    
    def __rmul__(self, other: Union[float, int]) -> 'quaternion':
        """Sağ taraftan skaler çarpma."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[float, int]) -> 'quaternion':
        """Skaler bölme."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Sıfıra bölme hatası")
            return quaternion(self.w / other, self.x / other, self.y / other, self.z / other)
        raise TypeError("Sadece skaler ile bölünebilir")
    
    def __eq__(self, other: 'quaternion') -> bool:
        """Eşitlik kontrolü."""
        if isinstance(other, quaternion):
            return (math.isclose(self.w, other.w) and 
                    math.isclose(self.x, other.x) and 
                    math.isclose(self.y, other.y) and 
                    math.isclose(self.z, other.z))
        return False
    
    def __ne__(self, other: 'quaternion') -> bool:
        """Eşitsizlik kontrolü."""
        return not self.__eq__(other)
    
    def __neg__(self) -> 'quaternion':
        """Negatif kuaterniyon."""
        return quaternion(-self.w, -self.x, -self.y, -self.z)
    
    def __repr__(self) -> str:
        """Nesnenin temsili."""
        return f"quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"
    
    def __str__(self) -> str:
        """String temsili."""
        return f"{self.w:.6f} + {self.x:.6f}i + {self.y:.6f}j + {self.z:.6f}k"
    
    def to_array(self) -> np.ndarray:
        """Kuaterniyonu numpy array'e dönüştürür."""
        return np.array([self.w, self.x, self.y, self.z])
    
    def to_list(self) -> List[float]:
        """Kuaterniyonu listeye dönüştürür."""
        return [self.w, self.x, self.y, self.z]
    
    @classmethod
    def identity(cls) -> 'quaternion':
        """Birim kuaterniyon döndürür."""
        return cls(1.0, 0.0, 0.0, 0.0)
    
    def is_identity(self, tolerance: float = 1e-10) -> bool:
        """Birim kuaterniyon olup olmadığını kontrol eder."""
        return (abs(self.w - 1.0) < tolerance and 
                abs(self.x) < tolerance and 
                abs(self.y) < tolerance and 
                abs(self.z) < tolerance)
    
    @classmethod
    def parse(cls, s) -> 'quaternion':
        """Çeşitli formatlardan quaternion oluşturur.
        
        Args:
            s: Dönüştürülecek değer
            
        Returns:
            quaternion: Dönüştürülmüş kuaterniyon
        """
        return _parse_quaternion_from_csv(s)
    
    @classmethod
    def from_csv_string(cls, s: str) -> 'quaternion':
        """CSV string'inden quaternion oluşturur.
        
        Args:
            s: Virgülle ayrılmış string ("w,x,y,z" veya "scalar")
            
        Returns:
            quaternion: Dönüştürülmüş kuaterniyon
        """
        return _parse_quaternion_from_csv(s)
    
    @classmethod
    def from_complex(cls, c: complex) -> 'quaternion':
        """Complex sayıdan quaternion oluşturur (sadece gerçek kısım kullanılır).
        
        Args:
            c: Complex sayı
            
        Returns:
            quaternion: Dönüştürülmüş kuaterniyon
        """
        return quaternion(float(c.real), 0, 0, 0)

@dataclass
class TernaryNumber:
    def __init__(self, digits: list):
        """
        Üçlü sayıyı oluşturur. Verilen değer bir liste olmalıdır.

        :param digits: Üçlü sayının rakamlarını temsil eden liste.
        """
        self.digits = digits

    @classmethod
    def from_ternary_string(cls, ternary_str: str) -> 'TernaryNumber':
        """Üçlü sayı sistemindeki stringi TernaryNumber'a dönüştürür."""
        ternary_str = ternary_str.strip()
        if not all(c in '012' for c in ternary_str):
            raise ValueError("Üçlü sayı sadece 0, 1 ve 2 rakamlarından oluşabilir.")
        digits = [int(c) for c in ternary_str]
        return cls(digits)

    @classmethod
    def from_decimal(cls, decimal: int) -> 'TernaryNumber':
        """Ondalık sayıyı üçlü sayı sistemine dönüştürür."""
        if decimal == 0:
            return cls([0])
        digits = []
        while decimal > 0:
            digits.append(decimal % 3)
            decimal = decimal // 3
        return cls(digits[::-1] if digits else [0])

    def to_decimal(self):
        """Üçlü sayının ondalık karşılığını döndürür."""
        decimal_value = 0
        for i, digit in enumerate(reversed(self.digits)):
            decimal_value += digit * (3 ** i)
        return decimal_value

    def __repr__(self):
        """Nesnenin yazdırılabilir temsilini döndürür."""
        return f"TernaryNumber({self.digits})"

    def __str__(self):
        """Nesnenin string temsilini döndürür."""
        return ''.join(map(str, self.digits))

    def __add__(self, other):
        """Toplama işlemini destekler."""
        if isinstance(other, TernaryNumber):
            result_decimal = self.to_decimal() + other.to_decimal()
        elif isinstance(other, (int, str)):
            result_decimal = self.to_decimal() + int(other)
        else:
            raise TypeError("TernaryNumber'ın başka bir sayıya veya TernaryNumber'e eklenebilir.")
        return TernaryNumber.from_decimal(result_decimal)

    def __radd__(self, other):
        """Toplama işleminin sağ taraf desteklenmesini sağlar."""
        return self.__add__(other)

    def __sub__(self, other):
        """Çıkarma işlemini destekler."""
        if isinstance(other, TernaryNumber):
            result_decimal = self.to_decimal() - other.to_decimal()
        elif isinstance(other, (int, str)):
            result_decimal = self.to_decimal() - int(other)
        else:
            raise TypeError("TernaryNumber'dan başka bir sayıya veya başka bir TernaryNumber çıkartılabilir.")
        if result_decimal < 0:
            raise ValueError("Bir üçlü sayıdan daha büyük bir sayı çıkaramazsınız.")
        return TernaryNumber.from_decimal(result_decimal)

    def __rsub__(self, other):
        """Çıkarma işleminin sağ taraf desteklenmesini sağlar."""
        if isinstance(other, (int, str)):
            result_decimal = int(other) - self.to_decimal()
        else:
            raise TypeError("TernaryNumber'dan bir sayı çıkartılabilir.")
        if result_decimal < 0:
            raise ValueError("Bir üçlü sayıdan daha büyük bir sayı çıkaramazsınız.")
        return TernaryNumber.from_decimal(result_decimal)

    def __mul__(self, scalar):
        """Skaler çarpım işlemini destekler."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("TernaryNumber sadece skaler ile çarpılabilir.")
        result_decimal = self.to_decimal() * scalar
        return TernaryNumber.from_decimal(int(result_decimal))

    def __rmul__(self, other):
        """ Çarpma işleminin sağ taraf desteklenmesini sağlar. """
        return self.__mul__(other)

    # Üçlü sayı sisteminde bölme işlemi, ondalık karşılığa dönüştürülerek yapılmalıdır.
    def __truediv__(self, other):
        """ Bölme işlemini destekler. """
        if isinstance(other, TernaryNumber):
            other_decimal = other.to_decimal()
            if other_decimal == 0:
                raise ZeroDivisionError("Bir TernaryNumber sıfırla bölünemez.")
            result_decimal = self.to_decimal() / other_decimal
            return TernaryNumber.from_decimal(int(round(result_decimal)))
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Sıfırla bölme hatası.")
            result_decimal = self.to_decimal() / other
            return TernaryNumber.from_decimal(int(round(result_decimal)))
        else:
            raise TypeError("TernaryNumber'i bir sayı veya başka bir TernaryNumber ile bölebilirsiniz.")

    # üçlü sayı sisteminde bölme işlemi, ondalık karşılığa dönüştürülerek yapılmalıdır.
    def __rtruediv__(self, other):
        """ Bölme işleminin sağ taraf desteklenmesini sağlar. """
        if isinstance(other, (int, float)):
            self_decimal = self.to_decimal()
            if self_decimal == 0:
                raise ZeroDivisionError("Sıfırla bölme hatası.")
            result_decimal = other / self_decimal
            return TernaryNumber.from_decimal(int(round(result_decimal)))
        else:
            raise TypeError("TernaryNumber ile bir sayı bölünebilir.")

    def __eq__(self, other):
        """Eşitlik kontrolü yapar."""
        if isinstance(other, TernaryNumber):
            return self.digits == other.digits
        elif isinstance(other, (int, str)):
            return self.to_decimal() == int(other)
        else:
            return False

    def __ne__(self, other):
        """Eşitsizlik kontrolü yapar."""
        return not self.__eq__(other)

# Superreal Sayılar
@dataclass
class SuperrealNumber:
    #def __init__(self, real_part=0.0):
    def __init__(self, real: float, split: float = 0.0):
        """
        SuperrealNumber nesnesini oluşturur.
        
        :param real_part: Gerçek sayı bileşeni (float).
        """
        #self.real = real_part
        self.real = real
        self.split = split

    def __repr__(self):
        """ Nesnenin yazdırılabilir temsilini döndürür. """
        return f"SuperrealNumber({self.real})"

    def __str__(self):
        """ Nesnenin string temsilini döndürür. """
        return str(self.real)

    def __add__(self, other):
        """ Toplama işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            return SuperrealNumber(self.real + other.real)
        elif isinstance(other, (int, float)):
            return SuperrealNumber(self.real + other)
        else:
            raise TypeError("SuperrealNumber'e bir sayı veya başka bir SuperrealNumber eklenebilir.")

    def __radd__(self, other):
        """ Toplama işleminin sağ taraf desteklenmesini sağlar. """
        return self.__add__(other)

    def __sub__(self, other):
        """ Çıkarma işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            return SuperrealNumber(self.real - other.real)
        elif isinstance(other, (int, float)):
            return SuperrealNumber(self.real - other)
        else:
            raise TypeError("SuperrealNumber'dan bir sayı veya başka bir SuperrealNumber çıkarılabilir.")

    def __rsub__(self, other):
        """ Çıkarma işleminin sağ taraf desteklenmesini sağlar. """
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        """ Çarpma işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            return SuperrealNumber(self.real * other.real)
        elif isinstance(other, (int, float)):
            return SuperrealNumber(self.real * other)
        else:
            raise TypeError("SuperrealNumber ile bir sayı veya başka bir SuperrealNumber çarpılabilir.")

    def __rmul__(self, other):
        """ Çarpma işleminin sağ taraf desteklenmesini sağlar. """
        return self.__mul__(other)

    def __truediv__(self, other):
        """ Bölme işlemini destekler. """
        if isinstance(other, SuperrealNumber):
            if other.real == 0:
                raise ZeroDivisionError("Bir SuperrealNumber sıfırla bölünemez.")
            return SuperrealNumber(self.real / other.real)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Sıfırla bölme hatası.")
            return SuperrealNumber(self.real / other)
        else:
            raise TypeError("SuperrealNumber'i bir sayı veya başka bir SuperrealNumber ile bölebilirsiniz.")

    def __rtruediv__(self, other):
        """ Bölme işleminin sağ taraf desteklenmesini sağlar. """
        if self.real == 0:
            raise ZeroDivisionError("Sıfırla bölme hatası.")
        return SuperrealNumber(other / self.real)

    def __neg__(self):
        """ Negatif değeri döndürür. """
        return SuperrealNumber(-self.real)

    def __eq__(self, other):
        """ Eşitlik kontrolü yapar. """
        if isinstance(other, SuperrealNumber):
            return self.real == other.real
        elif isinstance(other, (int, float)):
            return self.real == other
        else:
            return False

    def __ne__(self, other):
        """ Eşitsizlik kontrolü yapar. """
        return not self.__eq__(other)

    def __lt__(self, other):
        """ Küçük olma kontrolü yapar. """
        if isinstance(other, SuperrealNumber):
            return self.real < other.real
        elif isinstance(other, (int, float)):
            return self.real < other
        else:
            raise TypeError("SuperrealNumber ile karşılaştırılabilir.")

    def __le__(self, other):
        """ Küçük veya eşit kontrolü yapar. """
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        """ Büyük olma kontrolü yapar. """
        return not self.__le__(other)

    def __ge__(self, other):
        """ Büyük veya eşit kontrolü yapar. """
        return not self.__lt__(other)

@dataclass
class BaseNumber(ABC):
    """Tüm Keçeci sayı tipleri için ortak arayüz."""

    def __init__(self, value: Number):
        self._value = self._coerce(value)

    @staticmethod
    def _coerce(v: Number) -> Number:
        if isinstance(v, (int, float, complex)):
            return v
        raise TypeError(f"Geçersiz sayı tipi: {type(v)}")

    @property
    def value(self) -> Number:
        return self._value

    # ------------------------------------------------------------------ #
    # Matematiksel operator overload’ları (tek yönlü)
    # ------------------------------------------------------------------ #
    def __add__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        return self.__class__(self._value + other_val)

    def __radd__(self, other: Number) -> "BaseNumber":
        return self.__add__(other)

    def __sub__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        return self.__class__(self._value - other_val)

    def __rsub__(self, other: Number) -> "BaseNumber":
        return self.__class__(other - self._value)

    def __mul__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        return self.__class__(self._value * other_val)

    def __rmul__(self, other: Number) -> "BaseNumber":
        return self.__add__(other)

    def __truediv__(self, other: Union["BaseNumber", Number]) -> "BaseNumber":
        other_val = other.value if isinstance(other, BaseNumber) else other
        if other_val == 0:
            raise ZeroDivisionError("division by zero")
        return self.__class__(self._value / other_val)

    def __rtruediv__(self, other: Number) -> "BaseNumber":
        if self._value == 0:
            raise ZeroDivisionError("division by zero")
        return self.__class__(other / self._value)

    def __mod__(self, divisor: Number) -> "BaseNumber":
        return self.__class__(self._value % divisor)

    # ------------------------------------------------------------------ #
    # Karşılaştırmalar
    # ------------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseNumber):
            return NotImplemented
        return math.isclose(float(self._value), float(other._value), rel_tol=1e-12)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"

    # ------------------------------------------------------------------ #
    # Alt sınıfların doldurması gereken soyut metodlar
    # ------------------------------------------------------------------ #
    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        # Daha dayanıklı dönüş: coeffs bir numpy array veya python list olabilir.
        if hasattr(self, 'coeffs'):
            coeffs = getattr(self, 'coeffs')
            if isinstance(coeffs, np.ndarray):
                return coeffs.tolist()
            try:
                return list(coeffs)
            except Exception:
                return [coeffs]
        # Fallback: tek değer
        return [self._value]

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        """
        Güvenli phase hesaplayıcı:
        - Eğer value complex ise imag/real üzerinden phase hesaplanır.
        - Eğer coeffs varsa, ilk bileşenin complex olması durumunda phase döner.
        - Diğer durumlarda 0.0 döndürür (tanımsız phase için güvenli fallback).
        """
        try:
            # If underlying value is complex
            if isinstance(self._value, complex):
                return math.atan2(self._value.imag, self._value.real)
            # If there's coeffs, use the first coefficient
            if hasattr(self, 'coeffs'):
                coeffs = getattr(self, 'coeffs')
                if isinstance(coeffs, (list, tuple, np.ndarray)) and len(coeffs) > 0:
                    first = coeffs[0]
                    if isinstance(first, complex):
                        return math.atan2(first.imag, first.real)
            # If value has real/imag attributes (like some custom complex types)
            if hasattr(self._value, 'real') and hasattr(self._value, 'imag'):
                return math.atan2(self._value.imag, self._value.real)
        except Exception:
            pass
        return 0.0

@dataclass
class PathionNumber:
    """32-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 32:
            coeffs = list(coeffs) + [0.0] * (32 - len(coeffs))
            if len(coeffs) > 32:
                coeffs = coeffs[:32]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"

    def __repr__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"
        #return f"PathionNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, PathionNumber):
            return PathionNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return PathionNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, PathionNumber):
            return PathionNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return PathionNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, PathionNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return PathionNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return PathionNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return PathionNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if not isinstance(other, PathionNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, PathionNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return PathionNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'PathionNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return PathionNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'PathionNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / PathionNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return PathionNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'PathionNumber'")

    # ------------------------------------------------------------------
    # Yeni eklenen yardımcı metodlar
    # ------------------------------------------------------------------
    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # Güvenli phase: ilk bileşene bak, eğer complex ise angle döndür, değilse 0.0
        try:
            first = self.coeffs[0] if self.coeffs else 0.0
            if isinstance(first, complex):
                return math.atan2(first.imag, first.real)
        except Exception:
            pass
        return 0.0

@dataclass
class ChingonNumber:
    """64-bileşenli Chingon sayısı"""  # Açıklama düzeltildi
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 64:
            coeffs = list(coeffs) + [0.0] * (64 - len(coeffs))
            if len(coeffs) > 64:
                coeffs = coeffs[:64]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"ChingonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"ChingonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return ChingonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return ChingonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, ChingonNumber):
            # Basitçe bileşen bazlı çarpma
            return ChingonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])  # ChingonNumber döndür
        else:
            # Skaler çarpma
            return ChingonNumber([c * float(other) for c in self.coeffs])  # ChingonNumber döndür
    
    def __mod__(self, divisor):
        return ChingonNumber([c % divisor for c in self.coeffs])  # ChingonNumber döndür
    
    def __eq__(self, other):
        if not isinstance(other, ChingonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, ChingonNumber):  # ChingonNumber ile karşılaştır
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return ChingonNumber([c / other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return ChingonNumber([c // other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / ChingonNumber"""
        if isinstance(other, (int, float)):
            return ChingonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'ChingonNumber'")  # ChingonNumber

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # Güvenli phase: ilk bileşene bak, eğer complex ise angle döndür, değilse 0.0
        try:
            first = self.coeffs[0] if self.coeffs else 0.0
            if isinstance(first, complex):
                return math.atan2(first.imag, first.real)
        except Exception:
            pass
        return 0.0

@dataclass
class RoutonNumber:
    """128-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 128:
            coeffs = list(coeffs) + [0.0] * (128 - len(coeffs))
            if len(coeffs) > 128:
                coeffs = coeffs[:128]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"RoutonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"RoutonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, RoutonNumber):
            return RoutonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return RoutonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, RoutonNumber):
            return RoutonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return RoutonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, RoutonNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return RoutonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return RoutonNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return RoutonNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if not isinstance(other, RoutonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, RoutonNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return RoutonNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'RoutonNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return RoutonNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'RoutonNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / RoutonNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return RoutonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'RoutonNumber'")

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # Güvenli phase: ilk bileşene bak, eğer complex ise angle döndür, değilse 0.0
        try:
            first = self.coeffs[0] if self.coeffs else 0.0
            if isinstance(first, complex):
                return math.atan2(first.imag, first.real)
        except Exception:
            pass
        return 0.0

@dataclass
class VoudonNumber:
    """256-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 256:
            coeffs = list(coeffs) + [0.0] * (256 - len(coeffs))
            if len(coeffs) > 256:
                coeffs = coeffs[:256]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"VoudonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"VoudonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, VoudonNumber):
            return VoudonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return VoudonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, VoudonNumber):
            return VoudonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return VoudonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, VoudonNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return VoudonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return VoudonNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return VoudonNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if not isinstance(other, VoudonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, VoudonNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return VoudonNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'VoudonNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return VoudonNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'VoudonNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / VoudonNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return VoudonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'VoudonNumber'")

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # Güvenli phase: ilk bileşene bak, eğer complex ise angle döndür, değilse 0.0
        try:
            first = self.coeffs[0] if self.coeffs else 0.0
            if isinstance(first, complex):
                return math.atan2(first.imag, first.real)
        except Exception:
            pass
        return 0.0

@dataclass
class OctonionNumber:
    """
    Represents an octonion number with 8 components.
    Implements octonion multiplication rules.
    """
    def __init__(self, *args):
        # Varsayılan
        self.w = self.x = self.y = self.z = self.e = self.f = self.g = self.h = 0.0
        
        if len(args) == 8:
            self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = args
        elif len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                if len(args[0]) == 8:
                    self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = args[0]
                else:
                    components = list(args[0]) + [0.0] * (8 - len(args[0]))
                    self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = components
            elif isinstance(args[0], (int, float)):
                self.w = float(args[0])
        elif len(args) == 0:
            pass
        else:
            raise ValueError("Invalid arguments for OctonionNumber")

    @property
    def coeffs(self):
        """Octonion bileşenlerini liste olarak döner."""
        return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]

    def __add__(self, other):
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z,
                self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, OctonionNumber):
            # Mevcut octonion çarpımı (7-boyutlu çapraz çarpım kuralları)
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z \
                - self.e * other.e - self.f * other.f - self.g * other.g - self.h * other.h
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y \
                + self.e * other.f - self.f * other.e + self.g * other.h - self.h * other.g
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x \
                + self.e * other.g - self.g * other.e - self.f * other.h + self.h * other.f
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w \
                + self.e * other.h - self.h * other.e + self.f * other.g - self.g * other.f
            e = self.w * other.e - self.x * other.f - self.y * other.g - self.z * other.h \
                + self.e * other.w + self.f * other.x + self.g * other.y + self.h * other.z
            f = self.w * other.f + self.x * other.e - self.y * other.h + self.z * other.g \
                - self.e * other.x + self.f * other.w - self.g * other.z + self.h * other.y
            g = self.w * other.g + self.x * other.h + self.y * other.e - self.z * other.f \
                - self.e * other.y + self.f * other.z + self.g * other.w - self.h * other.x
            h = self.w * other.h - self.x * other.g + self.y * other.f + self.z * other.e \
                - self.e * other.z - self.f * other.y + self.g * other.x + self.h * other.w

            return OctonionNumber(w, x, y, z, e, f, g, h)

        elif isinstance(other, (int, float)):
            # Skaler çarpım: tüm bileşenler other ile çarpılır
            return OctonionNumber(
                self.w * other, self.x * other, self.y * other, self.z * other,
                self.e * other, self.f * other, self.g * other, self.h * other
            )

        return NotImplemented

    def __rmul__(self, other):
        # Skaler çarpımda değişme özelliği vardır: other * self == self * other
        if isinstance(other, (int, float)):
            return self.__mul__(other)  # self * other
        return NotImplemented


    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return OctonionNumber(
                self.w / scalar, self.x / scalar, self.y / scalar, self.z / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, OctonionNumber):
            return False
        tol = 1e-12
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol 
                   for attr in ['w', 'x', 'y', 'z', 'e', 'f', 'g', 'h'])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Octonion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.e:.3f}, {self.f:.3f}, {self.g:.3f}, {self.h:.3f})"

    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return str(self)

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]

    def phase(self):
        # Güvenli phase: octonionlarda özel tanım yok — fallback 0.0
        return 0.0


class Constants:
    """Oktonyon sabitleri."""
    ZERO = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
    ONE = OctonionNumber(1, 0, 0, 0, 0, 0, 0, 0)
    I = OctonionNumber(0, 1, 0, 0, 0, 0, 0, 0)
    J = OctonionNumber(0, 0, 1, 0, 0, 0, 0, 0)
    K = OctonionNumber(0, 0, 0, 1, 0, 0, 0, 0)
    E = OctonionNumber(0, 0, 0, 0, 1, 0, 0, 0)
    F = OctonionNumber(0, 0, 0, 0, 0, 1, 0, 0)
    G = OctonionNumber(0, 0, 0, 0, 0, 0, 1, 0)
    H = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 1)


@dataclass
class NeutrosophicNumber:
    """Represents a neutrosophic number of the form t + iI + fF."""
    t: float  # truth
    i: float  # indeterminacy
    f: float  # falsity

    def __init__(self, t: float, i: float, f: float = 0.0):
        self.t = t
        self.i = i
        self.f = f

    def __add__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.t + other.t, self.i + other.i, self.f + other.f)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t + other, self.i, self.f)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.t - other.t, self.i - other.i, self.f - other.f)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t - other, self.i, self.f)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(
                self.t * other.t,
                self.t * other.i + self.i * other.t + self.i * other.i,
                self.t * other.f + self.f * other.t + self.f * other.f
            )
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t * other, self.i * other, self.f * other)
        return NotImplemented

    def __truediv__(self, divisor: float) -> "NeutrosophicNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicNumber(self.t / divisor, self.i / divisor, self.f / divisor)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        parts = []
        if self.t != 0:
            parts.append(f"{self.t}")
        if self.i != 0:
            parts.append(f"{self.i}I")
        if self.f != 0:
            parts.append(f"{self.f}F")
        return " + ".join(parts) if parts else "0"

@dataclass
class NeutrosophicComplexNumber:
    """
    Represents a number with a complex part and an indeterminacy level.
    z = (a + bj) + cI, where I = indeterminacy.
    """

    def __init__(self, real: float = 0.0, imag: float = 0.0, indeterminacy: float = 0.0):
        self.real = float(real)
        self.imag = float(imag)
        self.indeterminacy = float(indeterminacy)

    def __repr__(self) -> str:
        return f"NeutrosophicComplexNumber(real={self.real}, imag={self.imag}, indeterminacy={self.indeterminacy})"

    def __str__(self) -> str:
        return f"({self.real}{self.imag:+}j) + {self.indeterminacy}I"

    def __add__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy + other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real + other, self.imag, self.indeterminacy)
        if isinstance(other, complex):
            return NeutrosophicComplexNumber(self.real + other.real, self.imag + other.imag, self.indeterminacy)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real - other, self.imag, self.indeterminacy)
        if isinstance(other, complex):
            return NeutrosophicComplexNumber(self.real - other.real, self.imag - other.imag, self.indeterminacy)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            # Indeterminacy: basitleştirilmiş model
            new_indeterminacy = (self.indeterminacy + other.indeterminacy +
                               self.magnitude_sq() * other.indeterminacy +
                               other.magnitude_sq() * self.indeterminacy)
            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        if isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real * other,
                self.imag * other,
                self.indeterminacy * other
            )
        return NotImplemented

    def __truediv__(self, divisor: Any) -> "NeutrosophicComplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicComplexNumber(
                self.real / divisor,
                self.imag / divisor,
                self.indeterminacy / divisor
            )
        return NotImplemented  # complex / NeutrosophicComplex desteklenmiyor

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                other - self.real,
                -self.imag,
                -self.indeterminacy
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def magnitude_sq(self) -> float:
        """Returns the squared magnitude of the complex part."""
        return self.real**2 + self.imag**2

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicComplexNumber):
            return (abs(self.real - other.real) < 1e-12 and
                    abs(self.imag - other.imag) < 1e-12 and
                    abs(self.indeterminacy - other.indeterminacy) < 1e-12)
        return False

@dataclass
class HyperrealNumber:
    """Represents a hyperreal number as a sequence of real numbers."""
    sequence: List[float]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.sequence = args[0]
        else:
            self.sequence = list(args)

    def __add__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            # Sequence'leri eşit uzunluğa getir
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a + b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] += other  # Sadece finite part'a ekle
            return HyperrealNumber(new_seq)
        return NotImplemented

    def __sub__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a - b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] -= other
            return HyperrealNumber(new_seq)
        return NotImplemented

    def __mul__(self, scalar: float) -> "HyperrealNumber":
        if isinstance(scalar, (int, float)):
            return HyperrealNumber([x * scalar for x in self.sequence])
        return NotImplemented

    def __rmul__(self, scalar: float) -> "HyperrealNumber":
        return self.__mul__(scalar)

    def __truediv__(self, divisor: float) -> "HyperrealNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Scalar division by zero.")
            return HyperrealNumber([x / divisor for x in self.sequence])
        raise TypeError("Only scalar division is supported.")

    def __mod__(self, divisor: float) -> "HyperrealNumber":
        if isinstance(divisor, (int, float)):
            return HyperrealNumber([x % divisor for x in self.sequence])
        raise TypeError("Modulo only supported with a scalar divisor.")

    def __str__(self) -> str:
        if len(self.sequence) <= 5:
            return f"Hyperreal{self.sequence}"
        return f"Hyperreal({self.sequence[:3]}...)" 

    @property
    def finite(self):
        """Returns the finite part (first component)"""
        return self.sequence[0] if self.sequence else 0.0

    @property
    def infinitesimal(self):
        """Returns the first infinitesimal part (second component)"""
        return self.sequence[1] if len(self.sequence) > 1 else 0.0

@dataclass
class BicomplexNumber:
    """Represents a bicomplex number with two complex components."""
    z1: complex  # First complex component
    z2: complex  # Second complex component

    def __add__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 + other, self.z2)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'BicomplexNumber' and '{type(other).__name__}'")

    def __sub__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 - other, self.z2)
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'BicomplexNumber' and '{type(other).__name__}'")

    def __mul__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(
                self.z1 * other.z1 - self.z2 * other.z2,
                self.z1 * other.z2 + self.z2 * other.z1
            )
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 * other, self.z2 * other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'BicomplexNumber' and '{type(other).__name__}'")

    def __truediv__(self, divisor: float) -> "BicomplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Division by zero")
            return BicomplexNumber(self.z1 / divisor, self.z2 / divisor)
        else:
            raise TypeError("Only scalar division is supported")

    def __str__(self) -> str:
        parts = []
        if self.z1 != 0j:
            parts.append(f"({self.z1.real}+{self.z1.imag}j)")
        if self.z2 != 0j:
            parts.append(f"({self.z2.real}+{self.z2.imag}j)e")
        return " + ".join(parts) if parts else "0"

def _parse_bicomplex(s: Any) -> BicomplexNumber:
    """
    Parses input into a BicomplexNumber.
    Accepts:
      - BicomplexNumber (returned as-is)
      - complex (used as z1, z2 = 0)
      - numeric (int/float) -> z1 = complex(num,0), z2 = 0
      - comma-separated 'a+bj,c+dj' or 'a,b,c,d' formats
      - explicit '(a+bj)+(c+dj)e' format
      - other strings like '1+2j'
    Returns fallback BicomplexNumber(0,0) on parse failures.
    """
    try:
        # If already correct type, return directly
        if isinstance(s, BicomplexNumber):
            return s

        # If numeric scalar (int/float), treat as real part of z1
        if isinstance(s, (int, float, np.floating, np.integer)):
            return BicomplexNumber(complex(float(s), 0.0), complex(0.0, 0.0))

        # If already a complex, use as z1
        if isinstance(s, complex):
            return BicomplexNumber(s, complex(0.0, 0.0))

        # If iterable (list/tuple) of numbers, try to map to components
        if hasattr(s, '__iter__') and not isinstance(s, str):
            parts = list(s)
            # If 4 numbers -> (r1,i1,r2,i2)
            if len(parts) == 4:
                return BicomplexNumber(complex(float(parts[0]), float(parts[1])),
                                       complex(float(parts[2]), float(parts[3])))
            # If 2 numbers -> (r1,i1)
            if len(parts) == 2:
                return BicomplexNumber(complex(float(parts[0]), float(parts[1])), complex(0.0, 0.0))
            # If 1 number -> scalar
            if len(parts) == 1:
                return BicomplexNumber(complex(float(parts[0]), 0.0), complex(0.0, 0.0))

        # Otherwise coerce to string and parse
        if not isinstance(s, str):
            s = str(s)

        s_clean = s.strip().replace(" ", "")

        # 1) Comma-separated numeric list: "a,b,c,d" or "a,b"
        if ',' in s_clean:
            parts = [p for p in s_clean.split(',') if p != '']
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 4:
                    return BicomplexNumber(complex(nums[0], nums[1]), complex(nums[2], nums[3]))
                elif len(nums) == 2:
                    return BicomplexNumber(complex(nums[0], nums[1]), complex(0.0, 0.0))
                elif len(nums) == 1:
                    return BicomplexNumber(complex(nums[0], 0.0), complex(0.0, 0.0))
            except ValueError:
                # not purely numeric comma format; fall through to other parsing
                pass

        # 2) Explicit "(a+bj)+(c+dj)e" form
        if 'e' in s_clean and '(' in s_clean:
            pattern = r'\(\s*([+-]?\d*\.?\d+)\s*([+-])\s*([\d\.]*)j\s*\)\s*(?:\+)\s*\(\s*([+-]?\d*\.?\d+)\s*([+-])\s*([\d\.]*)j\s*\)e'
            match = re.search(pattern, s_clean)
            if match:
                try:
                    z1_real = float(match.group(1))
                    z1_imag_sign = -1.0 if match.group(2) == '-' else 1.0
                    z1_imag_val = float(match.group(3)) if match.group(3) not in ['', None] else 1.0
                    z1_imag = z1_imag_sign * z1_imag_val

                    z2_real = float(match.group(4))
                    z2_imag_sign = -1.0 if match.group(5) == '-' else 1.0
                    z2_imag_val = float(match.group(6)) if match.group(6) not in ['', None] else 1.0
                    z2_imag = z2_imag_sign * z2_imag_val

                    return BicomplexNumber(complex(z1_real, z1_imag), complex(z2_real, z2_imag))
                except Exception:
                    pass

        # 3) Fallback: try Python complex() parsing for single complex string "a+bj"
        try:
            c = complex(s_clean)
            return BicomplexNumber(c, complex(0.0, 0.0))
        except Exception:
            # 4) As a last resort, try extracting first numeric token and use as real
            try:
                num_token = _extract_numeric_part(s_clean)
                return BicomplexNumber(complex(float(num_token), 0.0), complex(0.0, 0.0))
            except Exception:
                logger.warning("Bicomplex parsing failed for %r — returning zero bicomplex", s)
                return BicomplexNumber(complex(0.0, 0.0), complex(0.0, 0.0))

    except Exception as e:
        logger.exception("Unexpected error in _parse_bicomplex(%r): %s", s, e)
        return BicomplexNumber(complex(0.0, 0.0), complex(0.0, 0.0))

def _parse_universal(s: str, target_type: str) -> Any:
    """
    Universal parser - Çeşitli sayı türlerini string'den parse eder
    
    Args:
        s: Parse edilecek string
        target_type: Hedef tür ("real", "complex", "bicomplex")
    
    Returns:
        Parse edilmiş değer veya hata durumunda varsayılan değer
        
    Özellikler:
        - "real": Float'a çevirir, hata durumunda 0.0 döner
        - "complex": Karmaşık sayı formatlarını işler (1+2j, 3j, 4, vb.)
        - "bicomplex": _parse_bicomplex fonksiyonunu çağırır
    """
    if target_type == "real":
        try:
            return float(s.strip())
        except ValueError:
            return 0.0
    
    elif target_type == "complex":
        try:
            s_clean = s.strip().replace(" ", "")
            if 'j' not in s_clean:
                return complex(float(s_clean), 0.0)
            
            pattern = r'^([+-]?\d*\.?\d*)([+-]?\d*\.?\d*)j$'
            match = re.match(pattern, s_clean)
            if match:
                real_part = match.group(1)
                imag_part = match.group(2)
                
                if real_part in ['', '+', '-']:
                    real_part = real_part + '1' if real_part else '0'
                if imag_part in ['', '+', '-']:
                    imag_part = imag_part + '1' if imag_part else '0'
                
                return complex(float(real_part or 0), float(imag_part or 0))
            
            return complex(s_clean)
        except:
            return complex(0, 0)
    
    elif target_type == "bicomplex":
        # _parse_bicomplex'i çağır (tek parametreli)
        return _parse_bicomplex(s)
    
    return None

def kececi_bicomplex_algorithm(start: BicomplexNumber, add_val: BicomplexNumber, iterations: int, include_intermediate: bool = True) -> list:
    """
    Gerçek Keçeci algoritmasının bikompleks versiyonunu uygular
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Toplama işlemi
        current = current + add_val
        
        # 2. Keçeci algoritmasının özelliği: Mod alma ve asal sayı kontrolü
        # Gerçek algoritmada belirli bir modüle göre işlem yapılır
        mod_value = 100  # Bu değer algoritmaya göre ayarlanabilir
        
        # z1 ve z2 için mod alma (gerçek ve sanal kısımlar ayrı ayrı)
        current = BicomplexNumber(
            complex(current.z1.real % mod_value, current.z1.imag % mod_value),
            complex(current.z2.real % mod_value, current.z2.imag % mod_value)
        )
        
        # 3. Ara adımları ekle (Keçeci algoritmasının karakteristik özelliği)
        if include_intermediate:
            # Ara değerler için özel işlemler (örneğin: çarpma, bölme, vs.)
            intermediate = current * BicomplexNumber(complex(0.5, 0), complex(0, 0))
            sequence.append(intermediate)
        
        sequence.append(current)
        
        # 4. Asal sayı kontrolü (Keçeci algoritmasının önemli bir parçası)
        # Bu kısım algoritmanın detayına göre özelleştirilebilir
        magnitude = abs(current.z1) + abs(current.z2)
        if magnitude > 1 and all(magnitude % i != 0 for i in range(2, int(magnitude**0.5) + 1)):
            print(f"Keçeci Prime found at step {i}: {magnitude:.2f}")
        
        # 5. Özel durum: Belirli değerlere ulaşıldığında resetleme
        if abs(current.z1) < 1e-10 and abs(current.z2) < 1e-10:
            current = start  # Başa dön
    
    return sequence

# --- DAHA GERÇEKÇİ BİR VERSİYON ---
def kececi_bicomplex_advanced(start: BicomplexNumber, add_val: BicomplexNumber, 
                            iterations: int, include_intermediate: bool = True) -> list:
    """
    Gelişmiş Keçeci algoritması - daha karmaşık matematiksel işlemler içerir
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Temel toplama
        current = current + add_val
        
        # 2. Doğrusal olmayan dönüşümler (Keçeci algoritmasının özelliği)
        # Kök alma ve kuvvet alma işlemleri
        current = BicomplexNumber(
            complex(current.z1.real**0.5, current.z1.imag**0.5),
            complex(current.z2.real**0.5, current.z2.imag**0.5)
        )
        
        # 3. Modüler aritmetik
        mod_real = 50
        mod_imag = 50
        current = BicomplexNumber(
            complex(current.z1.real % mod_real, current.z1.imag % mod_imag),
            complex(current.z2.real % mod_real, current.z2.imag % mod_imag)
        )
        
        # 4. Ara adımlar
        if include_intermediate:
            # Çapraz çarpım ara değerleri
            cross_product = BicomplexNumber(
                complex(current.z1.real * current.z2.imag, 0),
                complex(0, current.z1.imag * current.z2.real)
            )
            sequence.append(cross_product)
        
        sequence.append(current)
        
        # 5. Dinamik sistem davranışı için feedback
        if i % 10 == 0 and i > 0:
            # Her 10 adımda bir küçük bir perturbasyon ekle
            perturbation = BicomplexNumber(
                complex(0.1 * np.sin(i), 0.1 * np.cos(i)),
                complex(0.05 * np.sin(i*0.5), 0.05 * np.cos(i*0.5))
            )
            current = current + perturbation
    
    return sequence

def _has_bicomplex_format(s: str) -> bool:
    """Checks if string has bicomplex format (comma-separated)."""
    return ',' in s and s.count(',') in [1, 3]  # 2 or 4 components

@dataclass
class NeutrosophicBicomplexNumber:
    def __init__(self, a, b, c, d, e, f, g, h):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)
        self.f = float(f)
        self.g = float(g)
        self.h = float(h)

    def __repr__(self):
        return f"NeutrosophicBicomplexNumber({self.a}, {self.b}, {self.c}, {self.d}, {self.e}, {self.f}, {self.g}, {self.h})"

    def __str__(self):
        return f"({self.a} + {self.b}i) + ({self.c} + {self.d}i)I + ({self.e} + {self.f}i)j + ({self.g} + {self.h}i)Ij"

    def __add__(self, other):
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(
                self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        return NotImplemented

    def __mul__(self, other):
        # Basitleştirilmiş çarpım (tam bicomplex kuralı karmaşık)
        if isinstance(other, (int, float)):
            return NeutrosophicBicomplexNumber(
                *(other * x for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h])
            )
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero")
            return NeutrosophicBicomplexNumber(
                self.a / scalar, self.b / scalar, self.c / scalar, self.d / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented

    def __eq__(self, other):
        """Equality with tolerance for float comparison."""
        if not isinstance(other, NeutrosophicBicomplexNumber):
            return False
        tol = 1e-12
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol 
                   for attr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    def __ne__(self, other):
        return not self.__eq__(other)

@dataclass
class SedenionNumber:
    def __init__(self, coeffs):
        if len(coeffs) != 16:
            raise ValueError("Sedenion must have 16 components")
        self.coeffs = list(map(float, coeffs))

    def __add__(self, other):
        if isinstance(other, SedenionNumber):
            return SedenionNumber([s + o for s, o in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return SedenionNumber([self.coeffs[0] + other] + self.coeffs[1:])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SedenionNumber):
            return SedenionNumber([s - o for s, o in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return SedenionNumber([self.coeffs[0] - other] + self.coeffs[1:])
        return NotImplemented

    def __mul__(self, other):
        # Sedenion çarpımı Cayley-Dickson yapısı ile tanımlanır ama oldukça karmaşıktır.
        # Basitleştirme: skaler çarpım veya element-wise çarpım DEĞİL.
        # Gerçek sedenion çarpımı 16x16 çarpım tablosu gerektirir.
        # Bu örnekte, yalnızca skaler çarpım ve Sedenion-Sedenion için çarpım tablosuz basitleştirilmiş hâli (örnek amaçlı) verilir.
        # Gerçek uygulama için sedenion multiplication table kullanılmalıdır.
        if isinstance(other, (int, float)):
            return SedenionNumber([c * other for c in self.coeffs])
        elif isinstance(other, SedenionNumber):
            # NOT: Gerçek sedenion çarpımı burada eksik (çok karmaşık). 
            # Element-wise çarpım matematiksel olarak doğru değildir ama örnek olsun diye konuldu.
            # Gerçek hâli için: https://en.wikipedia.org/wiki/Sedenion
            return NotImplemented  # Gerçek sedenion çarpımı oldukça karmaşıktır
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return SedenionNumber([c / other for c in self.coeffs])
        return NotImplemented

    def __str__(self):
        return "(" + ", ".join(f"{c:.2f}" for c in self.coeffs) + ")"

    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return list(self.coeffs)

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"

    def __repr__(self):
        return f"PathionNumber({', '.join(map(str, self.coeffs))})"
        #return f"PathionNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, PathionNumber):
            return PathionNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return PathionNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, PathionNumber):
            return PathionNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return PathionNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, PathionNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return PathionNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return PathionNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return PathionNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if not isinstance(other, PathionNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, PathionNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return PathionNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'PathionNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return PathionNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'PathionNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / PathionNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return PathionNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'PathionNumber'")

    # ------------------------------------------------------------------
    # Yeni eklenen yardımcı metodlar
    # ------------------------------------------------------------------
    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # compute and return the phase value
        return self._phase   # or whatever logic you need

@dataclass
class ChingonNumber:
    """64-bileşenli Chingon sayısı"""  # Açıklama düzeltildi
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 64:
            coeffs = list(coeffs) + [0.0] * (64 - len(coeffs))
            if len(coeffs) > 64:
                coeffs = coeffs[:64]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"ChingonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"ChingonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return ChingonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, ChingonNumber):
            return ChingonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return ChingonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, ChingonNumber):
            # Basitçe bileşen bazlı çarpma
            return ChingonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])  # ChingonNumber döndür
        else:
            # Skaler çarpma
            return ChingonNumber([c * float(other) for c in self.coeffs])  # ChingonNumber döndür
    
    def __mod__(self, divisor):
        return ChingonNumber([c % divisor for c in self.coeffs])  # ChingonNumber döndür
    
    def __eq__(self, other):
        if not isinstance(other, ChingonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, ChingonNumber):  # ChingonNumber ile karşılaştır
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return ChingonNumber([c / other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return ChingonNumber([c // other for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'ChingonNumber' and '{type(other).__name__}'")  # ChingonNumber
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / ChingonNumber"""
        if isinstance(other, (int, float)):
            return ChingonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])  # ChingonNumber döndür
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'ChingonNumber'")  # ChingonNumber

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # compute and return the phase value
        return self._phase   # or whatever logic you need

@dataclass
class RoutonNumber:
    """128-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 128:
            coeffs = list(coeffs) + [0.0] * (128 - len(coeffs))
            if len(coeffs) > 128:
                coeffs = coeffs[:128]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"RoutonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"RoutonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, RoutonNumber):
            return RoutonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return RoutonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, RoutonNumber):
            return RoutonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return RoutonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, RoutonNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return RoutonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return RoutonNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return RoutonNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if not isinstance(other, RoutonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, RoutonNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return RoutonNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'RoutonNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return RoutonNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'RoutonNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / RoutonNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return RoutonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'RoutonNumber'")

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # compute and return the phase value
        return self._phase   # or whatever logic you need

@dataclass
class VoudonNumber:
    """256-bileşenli Pathion sayısı"""
    
    def __init__(self, *coeffs):
        if len(coeffs) == 1 and hasattr(coeffs[0], '__iter__') and not isinstance(coeffs[0], str):
            coeffs = coeffs[0]
        
        if len(coeffs) != 256:
            coeffs = list(coeffs) + [0.0] * (256 - len(coeffs))
            if len(coeffs) > 256:
                coeffs = coeffs[:256]
        
        self.coeffs = [float(c) for c in coeffs]
    
    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]
    
    def __iter__(self):
        return iter(self.coeffs)
    
    def __getitem__(self, index):
        return self.coeffs[index]
    
    def __len__(self):
        return len(self.coeffs)
    
    def __str__(self):
        return f"VoudonNumber({', '.join(map(str, self.coeffs))})"
    
    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return f"VoudonNumber({self.coeffs})"
    
    def __add__(self, other):
        if isinstance(other, VoudonNumber):
            return VoudonNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler toplama
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += float(other)
            return VoudonNumber(new_coeffs)
    
    def __sub__(self, other):
        if isinstance(other, VoudonNumber):
            return VoudonNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= float(other)
            return VoudonNumber(new_coeffs)
    
    def __mul__(self, other):
        if isinstance(other, VoudonNumber):
            # Basitçe bileşen bazlı çarpma (gerçek Cayley-Dickson çarpımı yerine)
            return VoudonNumber([a * b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            # Skaler çarpma
            return VoudonNumber([c * float(other) for c in self.coeffs])
    
    def __mod__(self, divisor):
        return VoudonNumber([c % divisor for c in self.coeffs])
    
    def __eq__(self, other):
        if not isinstance(other, VoudonNumber):
            return NotImplemented
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)
        #if isinstance(other, VoudonNumber):
        #    return all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(self.coeffs, other.coeffs))
        #return False

    def __truediv__(self, other):
        """Bölme operatörü: / """
        if isinstance(other, (int, float)):
            # Skaler bölme
            return VoudonNumber([c / other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'VoudonNumber' and '{type(other).__name__}'")
    
    def __floordiv__(self, other):
        """Tam sayı bölme operatörü: // """
        if isinstance(other, (int, float)):
            # Skaler tam sayı bölme
            return VoudonNumber([c // other for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for //: 'VoudonNumber' and '{type(other).__name__}'")
    
    def __rtruediv__(self, other):
        """Sağdan bölme: other / VoudonNumber"""
        if isinstance(other, (int, float)):
            # Bu daha karmaşık olabilir, basitçe bileşen bazlı bölme
            return VoudonNumber([other / c if c != 0 else float('inf') for c in self.coeffs])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'VoudonNumber'")

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    def phase(self):
        # compute and return the phase value
        return self._phase   # or whatever logic you need

@dataclass
class OctonionNumber:
    """
    Represents an octonion number with 8 components.
    Implements octonion multiplication rules.
    """
    def __init__(self, *args):
        # Varsayılan
        self.w = self.x = self.y = self.z = self.e = self.f = self.g = self.h = 0.0
        
        if len(args) == 8:
            self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = args
        elif len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                if len(args[0]) == 8:
                    self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = args[0]
                else:
                    components = list(args[0]) + [0.0] * (8 - len(args[0]))
                    self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h = components
            elif isinstance(args[0], (int, float)):
                self.w = float(args[0])
        elif len(args) == 0:
            pass
        else:
            raise ValueError("Invalid arguments for OctonionNumber")

    @property
    def coeffs(self):
        """Octonion bileşenlerini liste olarak döner."""
        return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]

    def __add__(self, other):
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, OctonionNumber):
            return OctonionNumber(
                self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z,
                self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, OctonionNumber):
            # Mevcut octonion çarpımı (7-boyutlu çapraz çarpım kuralları)
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z \
                - self.e * other.e - self.f * other.f - self.g * other.g - self.h * other.h
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y \
                + self.e * other.f - self.f * other.e + self.g * other.h - self.h * other.g
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x \
                + self.e * other.g - self.g * other.e - self.f * other.h + self.h * other.f
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w \
                + self.e * other.h - self.h * other.e + self.f * other.g - self.g * other.f
            e = self.w * other.e - self.x * other.f - self.y * other.g - self.z * other.h \
                + self.e * other.w + self.f * other.x + self.g * other.y + self.h * other.z
            f = self.w * other.f + self.x * other.e - self.y * other.h + self.z * other.g \
                - self.e * other.x + self.f * other.w - self.g * other.z + self.h * other.y
            g = self.w * other.g + self.x * other.h + self.y * other.e - self.z * other.f \
                - self.e * other.y + self.f * other.z + self.g * other.w - self.h * other.x
            h = self.w * other.h - self.x * other.g + self.y * other.f + self.z * other.e \
                - self.e * other.z - self.f * other.y + self.g * other.x + self.h * other.w

            return OctonionNumber(w, x, y, z, e, f, g, h)

        elif isinstance(other, (int, float)):
            # Skaler çarpım: tüm bileşenler other ile çarpılır
            return OctonionNumber(
                self.w * other, self.x * other, self.y * other, self.z * other,
                self.e * other, self.f * other, self.g * other, self.h * other
            )

        return NotImplemented

    def __rmul__(self, other):
        # Skaler çarpımda değişme özelliği vardır: other * self == self * other
        if isinstance(other, (int, float)):
            return self.__mul__(other)  # self * other
        return NotImplemented


    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return OctonionNumber(
                self.w / scalar, self.x / scalar, self.y / scalar, self.z / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, OctonionNumber):
            return False
        tol = 1e-12
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol 
                   for attr in ['w', 'x', 'y', 'z', 'e', 'f', 'g', 'h'])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Octonion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.e:.3f}, {self.f:.3f}, {self.g:.3f}, {self.h:.3f})"

    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"
        #return str(self)

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]

    def phase(self):
        # compute and return the phase value
        return self._phase   # or whatever logic you need


@property
def coeffs(self):
    return [self.w, self.x, self.y, self.z, self.e, self.f, self.g, self.h]

class Constants:
    """Oktonyon sabitleri."""
    ZERO = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
    ONE = OctonionNumber(1, 0, 0, 0, 0, 0, 0, 0)
    I = OctonionNumber(0, 1, 0, 0, 0, 0, 0, 0)
    J = OctonionNumber(0, 0, 1, 0, 0, 0, 0, 0)
    K = OctonionNumber(0, 0, 0, 1, 0, 0, 0, 0)
    E = OctonionNumber(0, 0, 0, 0, 1, 0, 0, 0)
    F = OctonionNumber(0, 0, 0, 0, 0, 1, 0, 0)
    G = OctonionNumber(0, 0, 0, 0, 0, 0, 1, 0)
    H = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 1)


@dataclass
class NeutrosophicNumber:
    """Represents a neutrosophic number of the form t + iI + fF."""
    t: float  # truth
    i: float  # indeterminacy
    f: float  # falsity

    def __init__(self, t: float, i: float, f: float = 0.0):
        self.t = t
        self.i = i
        self.f = f

    def __add__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.t + other.t, self.i + other.i, self.f + other.f)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t + other, self.i, self.f)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(self.t - other.t, self.i - other.i, self.f - other.f)
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t - other, self.i, self.f)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicNumber":
        if isinstance(other, NeutrosophicNumber):
            return NeutrosophicNumber(
                self.t * other.t,
                self.t * other.i + self.i * other.t + self.i * other.i,
                self.t * other.f + self.f * other.t + self.f * other.f
            )
        if isinstance(other, (int, float)):
            return NeutrosophicNumber(self.t * other, self.i * other, self.f * other)
        return NotImplemented

    def __truediv__(self, divisor: float) -> "NeutrosophicNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicNumber(self.t / divisor, self.i / divisor, self.f / divisor)
        raise TypeError("Only scalar division is supported.")

    def __str__(self) -> str:
        parts = []
        if self.t != 0:
            parts.append(f"{self.t}")
        if self.i != 0:
            parts.append(f"{self.i}I")
        if self.f != 0:
            parts.append(f"{self.f}F")
        return " + ".join(parts) if parts else "0"

@dataclass
class NeutrosophicComplexNumber:
    """
    Represents a number with a complex part and an indeterminacy level.
    z = (a + bj) + cI, where I = indeterminacy.
    """

    def __init__(self, real: float = 0.0, imag: float = 0.0, indeterminacy: float = 0.0):
        self.real = float(real)
        self.imag = float(imag)
        self.indeterminacy = float(indeterminacy)

    def __repr__(self) -> str:
        return f"NeutrosophicComplexNumber(real={self.real}, imag={self.imag}, indeterminacy={self.indeterminacy})"

    def __str__(self) -> str:
        return f"({self.real}{self.imag:+}j) + {self.indeterminacy}I"

    def __add__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real + other.real,
                self.imag + other.imag,
                self.indeterminacy + other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real + other, self.imag, self.indeterminacy)
        if isinstance(other, complex):
            return NeutrosophicComplexNumber(self.real + other.real, self.imag + other.imag, self.indeterminacy)
        return NotImplemented

    def __sub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            return NeutrosophicComplexNumber(
                self.real - other.real,
                self.imag - other.imag,
                self.indeterminacy - other.indeterminacy
            )
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(self.real - other, self.imag, self.indeterminacy)
        if isinstance(other, complex):
            return NeutrosophicComplexNumber(self.real - other.real, self.imag - other.imag, self.indeterminacy)
        return NotImplemented

    def __mul__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, NeutrosophicComplexNumber):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            # Indeterminacy: basitleştirilmiş model
            new_indeterminacy = (self.indeterminacy + other.indeterminacy +
                               self.magnitude_sq() * other.indeterminacy +
                               other.magnitude_sq() * self.indeterminacy)
            return NeutrosophicComplexNumber(new_real, new_imag, new_indeterminacy)
        if isinstance(other, complex):
            new_real = self.real * other.real - self.imag * other.imag
            new_imag = self.real * other.imag + self.imag * other.real
            return NeutrosophicComplexNumber(new_real, new_imag, self.indeterminacy)
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                self.real * other,
                self.imag * other,
                self.indeterminacy * other
            )
        return NotImplemented

    def __truediv__(self, divisor: Any) -> "NeutrosophicComplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NeutrosophicComplexNumber(
                self.real / divisor,
                self.imag / divisor,
                self.indeterminacy / divisor
            )
        return NotImplemented  # complex / NeutrosophicComplex desteklenmiyor

    def __radd__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__add__(other)

    def __rsub__(self, other: Any) -> "NeutrosophicComplexNumber":
        if isinstance(other, (int, float)):
            return NeutrosophicComplexNumber(
                other - self.real,
                -self.imag,
                -self.indeterminacy
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "NeutrosophicComplexNumber":
        return self.__mul__(other)

    def magnitude_sq(self) -> float:
        """Returns the squared magnitude of the complex part."""
        return self.real**2 + self.imag**2

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NeutrosophicComplexNumber):
            return (abs(self.real - other.real) < 1e-12 and
                    abs(self.imag - other.imag) < 1e-12 and
                    abs(self.indeterminacy - other.indeterminacy) < 1e-12)
        return False

@dataclass
class HyperrealNumber:
    """Represents a hyperreal number as a sequence of real numbers."""
    sequence: List[float]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.sequence = args[0]
        else:
            self.sequence = list(args)

    def __add__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            # Sequence'leri eşit uzunluğa getir
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a + b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] += other  # Sadece finite part'a ekle
            return HyperrealNumber(new_seq)
        return NotImplemented

    def __sub__(self, other: Any) -> "HyperrealNumber":
        if isinstance(other, HyperrealNumber):
            max_len = max(len(self.sequence), len(other.sequence))
            seq1 = self.sequence + [0.0] * (max_len - len(self.sequence))
            seq2 = other.sequence + [0.0] * (max_len - len(other.sequence))
            return HyperrealNumber([a - b for a, b in zip(seq1, seq2)])
        elif isinstance(other, (int, float)):
            new_seq = self.sequence.copy()
            new_seq[0] -= other
            return HyperrealNumber(new_seq)
        return NotImplemented

    def __mul__(self, scalar: float) -> "HyperrealNumber":
        if isinstance(scalar, (int, float)):
            return HyperrealNumber([x * scalar for x in self.sequence])
        return NotImplemented

    def __rmul__(self, scalar: float) -> "HyperrealNumber":
        return self.__mul__(scalar)

    def __truediv__(self, divisor: float) -> "HyperrealNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Scalar division by zero.")
            return HyperrealNumber([x / divisor for x in self.sequence])
        raise TypeError("Only scalar division is supported.")

    def __mod__(self, divisor: float) -> "HyperrealNumber":
        if isinstance(divisor, (int, float)):
            return HyperrealNumber([x % divisor for x in self.sequence])
        raise TypeError("Modulo only supported with a scalar divisor.")

    def __str__(self) -> str:
        if len(self.sequence) <= 5:
            return f"Hyperreal{self.sequence}"
        return f"Hyperreal({self.sequence[:3]}...)" 

    @property
    def finite(self):
        """Returns the finite part (first component)"""
        return self.sequence[0] if self.sequence else 0.0

    @property
    def infinitesimal(self):
        """Returns the first infinitesimal part (second component)"""
        return self.sequence[1] if len(self.sequence) > 1 else 0.0

@dataclass
class BicomplexNumber:
    """Represents a bicomplex number with two complex components."""
    z1: complex  # First complex component
    z2: complex  # Second complex component

    def __add__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 + other.z1, self.z2 + other.z2)
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 + other, self.z2)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'BicomplexNumber' and '{type(other).__name__}'")

    def __sub__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(self.z1 - other.z1, self.z2 - other.z2)
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 - other, self.z2)
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'BicomplexNumber' and '{type(other).__name__}'")

    def __mul__(self, other: Any) -> "BicomplexNumber":
        if isinstance(other, BicomplexNumber):
            return BicomplexNumber(
                self.z1 * other.z1 - self.z2 * other.z2,
                self.z1 * other.z2 + self.z2 * other.z1
            )
        elif isinstance(other, (int, float, complex)):
            return BicomplexNumber(self.z1 * other, self.z2 * other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'BicomplexNumber' and '{type(other).__name__}'")

    def __truediv__(self, divisor: float) -> "BicomplexNumber":
        if isinstance(divisor, (int, float)):
            if divisor == 0:
                raise ZeroDivisionError("Division by zero")
            return BicomplexNumber(self.z1 / divisor, self.z2 / divisor)
        else:
            raise TypeError("Only scalar division is supported")

    def __str__(self) -> str:
        parts = []
        if self.z1 != 0j:
            parts.append(f"({self.z1.real}+{self.z1.imag}j)")
        if self.z2 != 0j:
            parts.append(f"({self.z2.real}+{self.z2.imag}j)e")
        return " + ".join(parts) if parts else "0"

def _parse_bicomplex(s: str) -> BicomplexNumber:
    """
    Kececi kütüphanesinin beklediği format: Tek parametre ile
    Sadece bicomplex parsing yapar
    """
    s_clean = s.strip().replace(" ", "")
    
    try:
        # Format 1: Comma-separated "z1_real,z1_imag,z2_real,z2_imag"
        if ',' in s_clean:
            parts = [float(p) for p in s_clean.split(',')]
            if len(parts) == 4:
                return BicomplexNumber(complex(parts[0], parts[1]), 
                                      complex(parts[2], parts[3]))
            elif len(parts) == 2:
                return BicomplexNumber(complex(parts[0], parts[1]), 
                                      complex(0, 0))
            elif len(parts) == 1:
                return BicomplexNumber(complex(parts[0], 0), 
                                      complex(0, 0))
        
        # Format 2: Explicit "(a+bj)+(c+dj)e"
        if 'e' in s_clean and '(' in s_clean:
            pattern = r'\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)\s*\+\s*\(([-\d.]+)\s*([+-]?)\s*([-\d.]*)j\)e'
            match = re.search(pattern, s_clean)
            
            if match:
                z1_real = float(match.group(1))
                z1_imag_sign = -1 if match.group(2) == '-' else 1
                z1_imag_val = float(match.group(3) or '1')
                z1_imag = z1_imag_sign * z1_imag_val
                
                z2_real = float(match.group(4))
                z2_imag_sign = -1 if match.group(5) == '-' else 1
                z2_imag_val = float(match.group(6) or '1')
                z2_imag = z2_imag_sign * z2_imag_val
                
                return BicomplexNumber(complex(z1_real, z1_imag), 
                                      complex(z2_real, z2_imag))
        
        # Format 3: Simple values
        try:
            if 'j' in s_clean:
                # Complex number parsing
                if 'j' not in s_clean:
                    return BicomplexNumber(complex(float(s_clean), 0), complex(0, 0))
                
                pattern = r'^([+-]?\d*\.?\d*)([+-]?\d*\.?\d*)j$'
                match = re.match(pattern, s_clean)
                if match:
                    real_part = match.group(1)
                    imag_part = match.group(2)
                    
                    if real_part in ['', '+', '-']:
                        real_part = real_part + '1' if real_part else '0'
                    if imag_part in ['', '+', '-']:
                        imag_part = imag_part + '1' if imag_part else '0'
                    
                    return BicomplexNumber(complex(float(real_part or 0), float(imag_part or 0)), 
                                          complex(0, 0))
            
            # Real number
            real_val = float(s_clean)
            return BicomplexNumber(complex(real_val, 0), complex(0, 0))
            
        except:
            pass
            
    except Exception as e:
        print(f"Bicomplex parsing error for '{s}': {e}")
    
    # Default fallback
    return BicomplexNumber(complex(0, 0), complex(0, 0))

def kececi_bicomplex_algorithm(start: BicomplexNumber, add_val: BicomplexNumber, iterations: int, include_intermediate: bool = True) -> list:
    """
    Gerçek Keçeci algoritmasının bikompleks versiyonunu uygular
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Toplama işlemi
        current = current + add_val
        
        # 2. Keçeci algoritmasının özelliği: Mod alma ve asal sayı kontrolü
        # Gerçek algoritmada belirli bir modüle göre işlem yapılır
        mod_value = 100  # Bu değer algoritmaya göre ayarlanabilir
        
        # z1 ve z2 için mod alma (gerçek ve sanal kısımlar ayrı ayrı)
        current = BicomplexNumber(
            complex(current.z1.real % mod_value, current.z1.imag % mod_value),
            complex(current.z2.real % mod_value, current.z2.imag % mod_value)
        )
        
        # 3. Ara adımları ekle (Keçeci algoritmasının karakteristik özelliği)
        if include_intermediate:
            # Ara değerler için özel işlemler (örneğin: çarpma, bölme, vs.)
            intermediate = current * BicomplexNumber(complex(0.5, 0), complex(0, 0))
            sequence.append(intermediate)
        
        sequence.append(current)
        
        # 4. Asal sayı kontrolü (Keçeci algoritmasının önemli bir parçası)
        # Bu kısım algoritmanın detayına göre özelleştirilebilir
        magnitude = abs(current.z1) + abs(current.z2)
        if magnitude > 1 and all(magnitude % i != 0 for i in range(2, int(magnitude**0.5) + 1)):
            print(f"Keçeci Prime found at step {i}: {magnitude:.2f}")
        
        # 5. Özel durum: Belirli değerlere ulaşıldığında resetleme
        if abs(current.z1) < 1e-10 and abs(current.z2) < 1e-10:
            current = start  # Başa dön
    
    return sequence

# --- DAHA GERÇEKÇİ BİR VERSİYON ---
def kececi_bicomplex_advanced(start: BicomplexNumber, add_val: BicomplexNumber, 
                            iterations: int, include_intermediate: bool = True) -> list:
    """
    Gelişmiş Keçeci algoritması - daha karmaşık matematiksel işlemler içerir
    """
    sequence = [start]
    current = start
    
    for i in range(iterations):
        # 1. Temel toplama
        current = current + add_val
        
        # 2. Doğrusal olmayan dönüşümler (Keçeci algoritmasının özelliği)
        # Kök alma ve kuvvet alma işlemleri
        current = BicomplexNumber(
            complex(current.z1.real**0.5, current.z1.imag**0.5),
            complex(current.z2.real**0.5, current.z2.imag**0.5)
        )
        
        # 3. Modüler aritmetik
        mod_real = 50
        mod_imag = 50
        current = BicomplexNumber(
            complex(current.z1.real % mod_real, current.z1.imag % mod_imag),
            complex(current.z2.real % mod_real, current.z2.imag % mod_imag)
        )
        
        # 4. Ara adımlar
        if include_intermediate:
            # Çapraz çarpım ara değerleri
            cross_product = BicomplexNumber(
                complex(current.z1.real * current.z2.imag, 0),
                complex(0, current.z1.imag * current.z2.real)
            )
            sequence.append(cross_product)
        
        sequence.append(current)
        
        # 5. Dinamik sistem davranışı için feedback
        if i % 10 == 0 and i > 0:
            # Her 10 adımda bir küçük bir perturbasyon ekle
            perturbation = BicomplexNumber(
                complex(0.1 * np.sin(i), 0.1 * np.cos(i)),
                complex(0.05 * np.sin(i*0.5), 0.05 * np.cos(i*0.5))
            )
            current = current + perturbation
    
    return sequence

def _has_bicomplex_format(s: str) -> bool:
    """Checks if string has bicomplex format (comma-separated)."""
    return ',' in s and s.count(',') in [1, 3]  # 2 or 4 components

@dataclass
class NeutrosophicBicomplexNumber:
    def __init__(self, a, b, c, d, e, f, g, h):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)
        self.f = float(f)
        self.g = float(g)
        self.h = float(h)

    def __repr__(self):
        return f"NeutrosophicBicomplexNumber({self.a}, {self.b}, {self.c}, {self.d}, {self.e}, {self.f}, {self.g}, {self.h})"

    def __str__(self):
        return f"({self.a} + {self.b}i) + ({self.c} + {self.d}i)I + ({self.e} + {self.f}i)j + ({self.g} + {self.h}i)Ij"

    def __add__(self, other):
        if isinstance(other, NeutrosophicBicomplexNumber):
            return NeutrosophicBicomplexNumber(
                self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h
            )
        return NotImplemented

    def __mul__(self, other):
        # Basitleştirilmiş çarpım (tam bicomplex kuralı karmaşık)
        if isinstance(other, (int, float)):
            return NeutrosophicBicomplexNumber(
                *(other * x for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h])
            )
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero")
            return NeutrosophicBicomplexNumber(
                self.a / scalar, self.b / scalar, self.c / scalar, self.d / scalar,
                self.e / scalar, self.f / scalar, self.g / scalar, self.h / scalar
            )
        return NotImplemented

    def __eq__(self, other):
        """Equality with tolerance for float comparison."""
        if not isinstance(other, NeutrosophicBicomplexNumber):
            return False
        tol = 1e-12
        return all(abs(getattr(self, attr) - getattr(other, attr)) < tol 
                   for attr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    def __ne__(self, other):
        return not self.__eq__(other)

@dataclass
class SedenionNumber:
    def __init__(self, coeffs):
        if len(coeffs) != 16:
            raise ValueError("Sedenion must have 16 components")
        self.coeffs = list(map(float, coeffs))

    def __add__(self, other):
        if isinstance(other, SedenionNumber):
            return SedenionNumber([s + o for s, o in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return SedenionNumber([self.coeffs[0] + other] + self.coeffs[1:])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SedenionNumber):
            return SedenionNumber([s - o for s, o in zip(self.coeffs, other.coeffs)])
        elif isinstance(other, (int, float)):
            return SedenionNumber([self.coeffs[0] - other] + self.coeffs[1:])
        return NotImplemented

    def __mul__(self, other):
        # Sedenion çarpımı Cayley-Dickson yapısı ile tanımlanır ama oldukça karmaşıktır.
        # Basitleştirme: skaler çarpım veya element-wise çarpım DEĞİL.
        # Gerçek sedenion çarpımı 16x16 çarpım tablosu gerektirir.
        # Bu örnekte, yalnızca skaler çarpım ve Sedenion-Sedenion için çarpım tablosuz basitleştirilmiş hâli (örnek amaçlı) verilir.
        # Gerçek uygulama için sedenion multiplication table kullanılmalıdır.
        if isinstance(other, (int, float)):
            return SedenionNumber([c * other for c in self.coeffs])
        elif isinstance(other, SedenionNumber):
            # NOT: Gerçek sedenion çarpımı burada eksik (çok karmaşık). 
            # Element-wise çarpım matematiksel olarak doğru değildir ama örnek olsun diye konuldu.
            # Gerçek hâli için: https://en.wikipedia.org/wiki/Sedenion
            return NotImplemented  # Gerçek sedenion çarpımı oldukça karmaşıktır
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return SedenionNumber([c / other for c in self.coeffs])
        return NotImplemented

    def __str__(self):
        return "(" + ", ".join(f"{c:.2f}" for c in self.coeffs) + ")"

    def __repr__(self):
        return f"({', '.join(map(str, self.coeffs))})"

    def components(self):
        """Bileşen listesini (Python list) döndürür."""
        return self.coeffs.tolist()

    def magnitude(self) -> float:
        """
        Euclidean norm = √( Σ_i coeff_i² )
        NumPy’nin `linalg.norm` fonksiyonu C‑hızında hesaplar.
        """
        return float(np.linalg.norm(self.coeffs))

    def __hash__(self):
        # NaN ve -0.0 gibi durumları göz önünde bulundurun
        return hash(tuple(np.round(self.coeffs, decimals=10)))

    @property
    def real(self) -> float:
        """İlk bileşen – “gerçek” kısım."""
        return float(self.coeffs[0])
    #def real(self):
    #    Gerçek kısım (ilk bileşen)
    #    return self.coeffs[0]

@dataclass
class CliffordNumber:
    def __init__(self, basis_dict: Dict[str, float]):
        """CliffordNumber constructor."""
        # Sadece sıfır olmayan değerleri sakla
        self.basis = {k: float(v) for k, v in basis_dict.items() if abs(float(v)) > 1e-10}
    
    @property
    def dimension(self) -> int:
        """Vector space dimension'ını otomatik hesaplar."""
        max_index = 0
        for key in self.basis.keys():
            if key:  # scalar değilse
                # '12', '123' gibi string'lerden maksimum rakamı bul
                if key.isdigit():
                    max_index = max(max_index, max(int(c) for c in key))
        return max_index

    def __add__(self, other):
        if isinstance(other, CliffordNumber):
            new_basis = self.basis.copy()
            for k, v in other.basis.items():
                new_basis[k] = new_basis.get(k, 0.0) + v
                # Sıfıra yakın değerleri temizle
                if abs(new_basis[k]) < 1e-10:
                    del new_basis[k]
            return CliffordNumber(new_basis)
        elif isinstance(other, (int, float)):
            new_basis = self.basis.copy()
            new_basis[''] = new_basis.get('', 0.0) + other
            return CliffordNumber(new_basis)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, CliffordNumber):
            new_basis = self.basis.copy()
            for k, v in other.basis.items():
                new_basis[k] = new_basis.get(k, 0.0) - v
                if abs(new_basis[k]) < 1e-10:
                    del new_basis[k]
            return CliffordNumber(new_basis)
        elif isinstance(other, (int, float)):
            new_basis = self.basis.copy()
            new_basis[''] = new_basis.get('', 0.0) - other
            return CliffordNumber(new_basis)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return CliffordNumber({k: v * other for k, v in self.basis.items()})
        elif isinstance(other, CliffordNumber):
            # Basit Clifford çarpımı (e_i^2 = +1 varsayımıyla)
            new_basis = {}
            
            for k1, v1 in self.basis.items():
                for k2, v2 in other.basis.items():
                    # Skaler çarpım
                    if k1 == '':
                        product_key = k2
                        sign = 1.0
                    elif k2 == '':
                        product_key = k1
                        sign = 1.0
                    else:
                        # Vektör çarpımı: e_i * e_j
                        combined = sorted(k1 + k2)
                        product_key = ''.join(combined)
                        
                        # Basitleştirilmiş: e_i^2 = +1, anti-commutative
                        sign = 1.0
                        # Burada gerçek Clifford cebir kuralları uygulanmalı
                    
                    new_basis[product_key] = new_basis.get(product_key, 0.0) + sign * v1 * v2
            
            return CliffordNumber(new_basis)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return CliffordNumber({k: v / other for k, v in self.basis.items()})
        return NotImplemented

    def __str__(self):
        parts = []
        if '' in self.basis and abs(self.basis['']) > 1e-10:
            parts.append(f"{self.basis['']:.2f}")
        
        sorted_keys = sorted([k for k in self.basis if k != ''], key=lambda x: (len(x), x))
        for k in sorted_keys:
            v = self.basis[k]
            if abs(v) > 1e-10:
                sign = '+' if v > 0 and parts else ''
                parts.append(f"{sign}{v:.2f}e{k}")
        
        result = "".join(parts).replace("+-", "-")
        return result if result else "0.0"

    @classmethod
    def parse(cls, s) -> 'CliffordNumber':
        """Class method olarak parse metodu"""
        return _parse_clifford(s)

    def __repr__(self):
        return self.__str__()


@dataclass
class DualNumber:

    real: float
    dual: float

    def __init__(self, real, dual):
        self.real = float(real)
        self.dual = float(dual)
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real + other, self.dual)
        raise TypeError
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real - other, self.dual)
        raise TypeError
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real * other, self.dual * other)
        raise TypeError
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            return DualNumber(self.real / other, self.dual / other)
        elif isinstance(other, DualNumber):
            if other.real == 0:
                raise ZeroDivisionError
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
        raise TypeError
    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            return DualNumber(self.real // other, self.dual // other)
        raise TypeError
    def __eq__(self, other):
        if isinstance(other, DualNumber):
            return self.real == other.real and self.dual == other.dual
        elif isinstance(other, (int, float)):
            return self.real == other and self.dual == 0
        return False
    def __str__(self):
        return f"{self.real} + {self.dual}ε"
    def __repr__(self):
        return self.__str__() # __repr__ eklenmiş
    def __int__(self):
        return int(self.real) # int() dönüşümü eklenmiş
    def __radd__(self, other):
       return self.__add__(other)  # commutative
    def __rsub__(self, other):
       if isinstance(other, (int, float)):
           return DualNumber(other - self.real, -self.dual)
       return NotImplemented

    def __neg__(self):
       return DualNumber(-self.real, -self.dual)

    def __hash__(self):
       return hash((self.real, self.dual))


@dataclass
class SplitcomplexNumber:
    def __init__(self, real, split):
        self.real = float(real)
        self.split = float(split)

    def __add__(self, other):
        if isinstance(other, SplitcomplexNumber):
            return SplitcomplexNumber(self.real + other.real, self.split + other.split)
        elif isinstance(other, (int, float)):
            return SplitcomplexNumber(self.real + other, self.split)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SplitcomplexNumber):
            return SplitcomplexNumber(self.real - other.real, self.split - other.split)
        elif isinstance(other, (int, float)):
            return SplitcomplexNumber(self.real - other, self.split)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, SplitcomplexNumber):
            # (a + bj) * (c + dj) = (ac + bd) + (ad + bc)j, çünkü j² = +1
            real = self.real * other.real + self.split * other.split
            split = self.real * other.split + self.split * other.real
            return SplitcomplexNumber(real, split)
        elif isinstance(other, (int, float)):
            return SplitcomplexNumber(self.real * other, self.split * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return SplitcomplexNumber(self.real / other, self.split / other)
        elif isinstance(other, SplitcomplexNumber):
            # (a + bj) / (c + dj) = ?
            # Payda: (c + dj)(c - dj) = c² - d² (çünkü j² = 1)
            # Yani bölme yalnızca c² ≠ d² ise tanımlıdır.
            a, b = self.real, self.split
            c, d = other.real, other.split
            norm = c * c - d * d
            if abs(norm) < 1e-10:
                raise ZeroDivisionError("Split-complex division by zero (null divisor)")
            real = (a * c - b * d) / norm
            split = (b * c - a * d) / norm
            return SplitcomplexNumber(real, split)
        return NotImplemented

    def __str__(self):
        return f"{self.real:.2f} + {self.split:.2f}j'"

    def __repr__(self):
        return f"({self.real}, {self.split}j')"


# Yardımcı fonksiyonlar
def _extract_numeric_part(s: Any) -> str:
    """
    Return the first numeric token found in s as string (supports scientific notation).
    Robust for None and non-string inputs.
    """
    if s is None:
        return "0"
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # match optional sign, digits, optional decimal, optional exponent
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return m.group(0) if m else "0"

def _parse_complex(s) -> complex:
    """Bir string'i veya sayıyı complex sayıya dönüştürür.
    "real,imag", "real+imag(i/j)", "real", "imag(i/j)" formatlarını destekler.
    Float ve int tiplerini de doğrudan kabul eder.
    """
    # Eğer zaten complex sayıysa doğrudan döndür
    if isinstance(s, complex):
        return s
    
    # Eğer float veya int ise doğrudan complex'e dönüştür
    if isinstance(s, (float, int)):
        return complex(s)
    
    # String işlemleri için önce string'e dönüştür
    if isinstance(s, str):
        s = s.strip().replace('J', 'j').replace('i', 'j') # Hem J hem i yerine j kullan
    else:
        s = str(s).strip().replace('J', 'j').replace('i', 'j')
    
    # 1. Eğer "real,imag" formatındaysa
    if ',' in s:
        parts = s.split(',')
        if len(parts) == 2:
            try:
                return complex(float(parts[0]), float(parts[1]))
            except ValueError:
                pass # Devam et

    # 2. Python'ın kendi complex() dönüştürücüsünü kullanmayı dene (örn: "1+2j", "3j", "-5")
    try:
        return complex(s)
    except ValueError:
        # 3. Sadece real kısmı varsa (örn: "5")
        try:
            return complex(float(s), 0)
        except ValueError:
            # 4. Sadece sanal kısmı varsa (örn: "2j", "j")
            if s.endswith('j'):
                try:
                    imag_val = float(s[:-1]) if s[:-1] else 1.0 # "j" -> 1.0j
                    return complex(0, imag_val)
                except ValueError:
                    pass
            
            raise ValueError(f"Geçersiz kompleks sayı formatı: '{s}'")


def convert_to_float(value: Any) -> float:
    """
    Convert various Keçeci number types to a float (best-effort).
    Raises TypeError if conversion is not possible.
    Rules:
      - int/float -> float
      - complex -> real part (float)
      - numpy-quaternion or objects with attribute 'w' -> float(w)
      - objects with 'real' attribute -> float(real)
      - objects with 'coeffs' iterable -> float(first coeff)
      - objects with 'sequence' iterable -> float(first element)
    """
    # Direct numeric types
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)

    if isinstance(value, complex):
        return float(value.real)

    # quaternion-like
    try:
        #if isinstance(value, quaternion):
        #    return float(value.w)
        if quaternion is not None and isinstance(value, quaternion):
            comps = [value.w, value.x, value.y, value.z]
            if not all(is_near_integer(c) for c in comps):
                return False
            return sympy.isprime(int(round(float(comps[0]))))
    except Exception:
        pass

    # Generic attributes
    if hasattr(value, 'real'):
        try:
            return float(getattr(value, 'real'))
        except Exception:
            pass

    if hasattr(value, 'w'):
        try:
            return float(getattr(value, 'w'))
        except Exception:
            pass

    if hasattr(value, 'coeffs'):
        try:
            coeffs = getattr(value, 'coeffs')
            if isinstance(coeffs, np.ndarray):
                if coeffs.size > 0:
                    return float(coeffs.flatten()[0])
            else:
                # list/iterable
                it = list(coeffs)
                if it:
                    return float(it[0])
        except Exception:
            pass

    if hasattr(value, 'sequence'):
        try:
            seq = getattr(value, 'sequence')
            if seq and len(seq) > 0:
                return float(seq[0])
        except Exception:
            pass

    # TernaryNumber: digits -> decimal
    if hasattr(value, 'digits'):
        try:
            digits = list(value.digits)
            dec = 0
            for i, d in enumerate(reversed(digits)):
                dec += int(d) * (3 ** i)
            return float(dec)
        except Exception:
            pass

    raise TypeError(f"Cannot convert {type(value).__name__} to float.")

def safe_add(added_value, ask_unit, direction):
    """
    Adds ±ask_unit to added_value using native algebraic operations.

    This function performs: `added_value + (ask_unit * direction)`
    It assumes that both operands support algebraic addition and scalar multiplication.

    Parameters
    ----------
    added_value : Any
        The base value (e.g., DualNumber, OctonionNumber, CliffordNumber).
    ask_unit : Same type as added_value
        The unit increment to add or subtract.
    direction : int
        Either +1 or -1, determining the sign of the increment.

    Returns
    -------
    Same type as added_value
        Result of `added_value + (ask_unit * direction)`.

    Raises
    ------
    TypeError
        If `ask_unit` does not support multiplication by an int,
        or if `added_value` does not support addition with `ask_unit`.
    """
    try:
        # Scale the unit: ask_unit * (+1 or -1)
        if not hasattr(ask_unit, '__mul__'):
            raise TypeError(f"Type '{type(ask_unit).__name__}' does not support scalar multiplication (missing __mul__).")
        scaled_unit = ask_unit * direction

        # Add to the current value
        if not hasattr(added_value, '__add__'):
            raise TypeError(f"Type '{type(added_value).__name__}' does not support addition (missing __add__).")
        result = added_value + scaled_unit

        return result

    except Exception as e:
        # Daha açıklayıcı hata mesajı
        msg = f"safe_add failed: Cannot compute {repr(added_value)} + ({direction} * {repr(ask_unit)})"
        raise TypeError(f"{msg} → {type(e).__name__}: {e}") from e


def _parse_neutrosophic(s) -> Tuple[float, float, float]:
    """Parses neutrosophic string into (t, i, f) tuple."""
    # Eğer zaten tuple ise doğrudan döndür
    if isinstance(s, (tuple, list)) and len(s) >= 3:
        return float(s[0]), float(s[1]), float(s[2])
    
    # Sayısal tipse sadece t değeri olarak işle
    if isinstance(s, (float, int, complex)):
        return float(s), 0.0, 0.0
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s_clean = s.strip().replace(" ", "").upper()
    
    # VİRGÜL formatı: t,i,f (3 parametre)
    if ',' in s_clean:
        parts = s_clean.split(',')
        try:
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]), float(parts[1]), 0.0
            elif len(parts) == 1:
                return float(parts[0]), 0.0, 0.0
        except ValueError:
            pass

    # Eski formatları destekle
    try:
        if 'I' in s_clean or 'F' in s_clean:
            # Basit parsing
            t_part = s_clean
            i_val, f_val = 0.0, 0.0
            
            if 'I' in s_clean:
                parts = s_clean.split('I')
                t_part = parts[0]
                i_val = float(parts[1]) if parts[1] and parts[1] not in ['', '+', '-'] else 1.0
            
            if 'F' in t_part:
                parts = t_part.split('F')
                t_val = float(parts[0]) if parts[0] and parts[0] not in ['', '+', '-'] else 0.0
                f_val = float(parts[1]) if len(parts) > 1 and parts[1] not in ['', '+', '-'] else 1.0
            else:
                t_val = float(t_part) if t_part not in ['', '+', '-'] else 0.0
            
            return t_val, i_val, f_val
        else:
            # Sadece sayısal değer
            return float(s_clean), 0.0, 0.0
    except ValueError:
        return 0.0, 0.0, 0.0  # Default

def _parse_hyperreal(s) -> Tuple[float, float]:
    """Parses hyperreal string into (finite, infinitesimal) tuple."""
    # Eğer zaten tuple ise doğrudan döndür
    if isinstance(s, (tuple, list)) and len(s) >= 2:
        return float(s[0]), float(s[1])
    
    # Sayısal tipse sadece finite değeri olarak işle
    if isinstance(s, (float, int, complex)):
        return float(s), 0.0
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s_clean = s.strip().replace(" ", "")
    
    # VİRGÜL formatı: finite,infinitesimal
    if ',' in s_clean:
        parts = s_clean.split(',')
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        elif len(parts) == 1:
            try:
                return float(parts[0]), 0.0
            except ValueError:
                pass

    # Eski 'a+be' formatını destekle
    if 'e' in s_clean:
        try:
            parts = s_clean.split('e')
            finite = float(parts[0]) if parts[0] not in ['', '+', '-'] else 0.0
            infinitesimal = float(parts[1]) if len(parts) > 1 and parts[1] not in ['', '+', '-'] else 1.0
            return finite, infinitesimal
        except ValueError:
            pass
    
    # Sadece sayısal değer
    try:
        return float(s_clean), 0.0
    except ValueError:
        return 0.0, 0.0  # Default

def _parse_quaternion_from_csv(s) -> quaternion:
    """Virgülle ayrılmış string'i veya sayıyı quaternion'a dönüştürür.
    
    Args:
        s: Dönüştürülecek değer. Şu formatları destekler:
            - quaternion nesnesi (doğrudan döndürülür)
            - float, int, complex sayılar (skaler quaternion)
            - String ("w,x,y,z" veya "scalar" formatında)
            - Diğer tipler (string'e dönüştürülerek işlenir)
    
    Returns:
        quaternion: Dönüştürülmüş kuaterniyon
    
    Raises:
        ValueError: Geçersiz format veya sayısal olmayan bileşenler durumunda
    """
    # Eğer zaten quaternion ise doğrudan döndür
    if isinstance(s, quaternion):
        return s
    
    # Sayısal tipse skaler quaternion olarak işle
    if isinstance(s, (float, int)):
        return quaternion(float(s), 0, 0, 0)
    
    # Complex sayı için özel işlem
    if isinstance(s, complex):
        # Complex sayının sadece gerçek kısmını al
        return quaternion(float(s.real), 0, 0, 0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    
    # Boş string kontrolü
    if not s:
        raise ValueError(f"Boş string quaternion'a dönüştürülemez")
    
    # String'i virgülle ayır
    parts_str = s.split(',')
    
    # Tüm parçaları float'a dönüştürmeyi dene
    parts_float = []
    for p in parts_str:
        p = p.strip()
        if not p:
            raise ValueError(f"Boş bileşen bulundu: '{s}'")
        
        try:
            # Önce normal float olarak dene
            parts_float.append(float(p))
        except ValueError:
            # Float olarak parse edilemezse complex olarak dene
            try:
                # 'i' karakterini 'j' ile değiştir (complex fonksiyonu 'j' bekler)
                complex_str = p.replace('i', 'j').replace('I', 'J')
                # Eğer 'j' yoksa ve sayı değilse hata ver
                if 'j' not in complex_str.lower():
                    raise ValueError(f"Geçersiz sayı formatı: '{p}'")
                
                c = complex(complex_str)
                parts_float.append(float(c.real))
            except ValueError:
                raise ValueError(f"quaternion bileşeni sayı olmalı: '{p}' (string: '{s}')")

    if len(parts_float) == 4:
        return quaternion(*parts_float)
    elif len(parts_float) == 1:  # Sadece skaler değer
        return quaternion(parts_float[0], 0, 0, 0)
    else:
        raise ValueError(f"Geçersiz quaternion formatı. 1 veya 4 bileşen bekleniyor, {len(parts_float)} alındı: '{s}'")

def _has_comma_format(s: Any) -> bool:
    """
    True if value is a string and contains a comma (CSV-like format).
    Guard against non-strings.
    """
    if s is None:
        return False
    if not isinstance(s, str):
        s = str(s)
    # Consider comma-format only when there's at least one digit and a comma
    return ',' in s and bool(re.search(r'\d', s))

def _is_complex_like(s: Any) -> bool:
    """
    Check if s looks like a complex literal (contains 'j'/'i' or +-/ with j).
    Accepts non-strings by attempting to str() them.
    """
    if s is None:
        return False
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # quick checks
    if 'j' in s or 'i' in s:
        return True
    # pattern like "a+bi" or "a-bi"
    if re.search(r'[+-]\d', s) and ('+' in s or '-' in s):
        # avoid classifying comma-separated lists as complex
        if ',' in s:
            return False
        return True
    return False

def _parse_neutrosophic_bicomplex(s) -> NeutrosophicBicomplexNumber:
    """
    Parses string or numbers into NeutrosophicBicomplexNumber.
    """
    # Eğer zaten NeutrosophicBicomplexNumber ise doğrudan döndür
    if isinstance(s, NeutrosophicBicomplexNumber):
        return s
    
    # Sayısal tipse tüm bileşenler 0, sadece ilk bileşen değerli
    if isinstance(s, (float, int, complex)):
        values = [float(s)] + [0.0] * 7
        return NeutrosophicBicomplexNumber(*values)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    try:
        parts = s.split(',')
        if len(parts) != 8:
            raise ValueError(f"Expected 8 components, got {len(parts)}")
        values = [float(part.strip()) for part in parts]
        return NeutrosophicBicomplexNumber(*values)
    except Exception as e:
        raise ValueError(f"Invalid NeutrosophicBicomplex format: '{s}' → {e}")


def _parse_octonion(s) -> OctonionNumber:
    """String'i veya sayıyı OctonionNumber'a dönüştürür.
    w,x,y,z,e,f,g,h:e0,e1,e2,e3,e4,e5,e6,e7
    """
    # Eğer zaten OctonionNumber ise doğrudan döndür
    if isinstance(s, OctonionNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) skaler olarak işle
    if isinstance(s, (float, int, complex)):
        scalar = float(s)
        return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s_clean = s.strip()
    
    # Eğer virgül içermiyorsa, skaler olarak kabul et
    if ',' not in s_clean:
        try:
            scalar = float(s_clean)
            return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
        except ValueError:
            raise ValueError(f"Invalid octonion format: '{s}'")
    
    # Virgülle ayrılmışsa
    try:
        parts = [float(p.strip()) for p in s_clean.split(',')]
        if len(parts) == 8:
            return OctonionNumber(*parts)  # 8 parametre olarak gönder
        else:
            # Eksik veya fazla bileşen için default
            scalar = parts[0] if parts else 0.0
            return OctonionNumber(scalar, 0, 0, 0, 0, 0, 0, 0)
    except ValueError as e:
        raise ValueError(f"Invalid octonion format: '{s}'") from e


def _parse_sedenion(s) -> SedenionNumber:
    """String'i veya sayıyı SedenionNumber'a dönüştürür."""
    # Eğer zaten SedenionNumber ise doğrudan döndür
    if isinstance(s, SedenionNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) skaler olarak işle
    if isinstance(s, (float, int, complex)):
        scalar_val = float(s)
        return SedenionNumber([scalar_val] + [0.0] * 15)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 16:
        try:
            return SedenionNumber(list(map(float, parts)))
        except ValueError as e:
            raise ValueError(f"Geçersiz sedenion bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1: # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return SedenionNumber([scalar_val] + [0.0] * 15)
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler sedenion değeri: '{s}' -> {e}") from e

    raise ValueError(f"Sedenion için 16 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_pathion(s) -> PathionNumber:
    """String'i veya sayıyı PathionNumber'a dönüştürür."""
    if isinstance(s, PathionNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return PathionNumber(float(s), *[0.0] * 31)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return PathionNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 32:  # Pathion 32 bileşenli olmalı
        try:
            return PathionNumber(*map(float, parts))  # 32 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz pathion bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return PathionNumber(scalar_val, *[0.0] * 31)  # 32 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler pathion değeri: '{s}' -> {e}") from e

    raise ValueError(f"Pathion için 32 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_chingon(s) -> ChingonNumber:
    """String'i veya sayıyı ChingonNumber'a dönüştürür."""
    if isinstance(s, ChingonNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return ChingonNumber(float(s), *[0.0] * 63)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return ChingonNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 64:  # Pathion 32 bileşenli olmalı
        try:
            return ChingonNumber(*map(float, parts))  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz chingon bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return ChingonNumber(scalar_val, *[0.0] * 63)  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler Chingon değeri: '{s}' -> {e}") from e

    raise ValueError(f"Chingon için 64 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_routon(s) -> RoutonNumber:
    """String'i veya sayıyı RoutonNumber'a dönüştürür."""
    if isinstance(s, RoutonNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return RoutonNumber(float(s), *[0.0] * 127)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return RoutonNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 128:  # Pathion 32 bileşenli olmalı
        try:
            return RoutonNumber(*map(float, parts))  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz routon bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return RoutonNumber(scalar_val, *[0.0] * 127)  # 128 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler routon değeri: '{s}' -> {e}") from e

    raise ValueError(f"Routon için 64 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")

def _parse_voudon(s) -> VoudonNumber:
    """String'i veya sayıyı VoudonNumber'a dönüştürür."""
    if isinstance(s, VoudonNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return VoudonNumber(float(s), *[0.0] * 255)
    
    if hasattr(s, '__iter__') and not isinstance(s, str):
        return VoudonNumber(s)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    # Köşeli parantezleri kaldır (eğer varsa)
    s = s.strip('[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 256:  # Pathion 32 bileşenli olmalı
        try:
            return VoudonNumber(*map(float, parts))  # 64 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz voudon bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:  # Sadece skaler değer girildiğinde
        try:
            scalar_val = float(parts[0])
            return VoudonNumber(scalar_val, *[0.0] * 255)  # 256 parametre
        except ValueError as e:
            raise ValueError(f"Geçersiz skaler voudon değeri: '{s}' -> {e}") from e

    raise ValueError(f"Voudon için 64 bileşen veya tek skaler bileşen gerekir. Verilen: '{s}' ({len(parts)} bileşen)")


def _parse_clifford(s) -> CliffordNumber:
    """Algebraik string'i CliffordNumber'a dönüştürür (ör: '1.0+2.0e1')."""
    if isinstance(s, CliffordNumber):
        return s
    
    if isinstance(s, (float, int, complex)):
        return CliffordNumber({'': float(s)})
    
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip().replace(' ', '').replace('^', '')  # ^ işaretini kaldır
    basis_dict = {}
    
    # Daha iyi regex pattern: +-1.23e12 formatını yakala
    pattern = r'([+-]?)(\d*\.?\d+)(?:e(\d+))?|([+-]?)(?:e(\d+))'
    matches = re.findall(pattern, s)
    
    for match in matches:
        sign_str, coeff_str, basis1, sign_str2, basis2 = match
        
        # Hangi grup match oldu?
        if coeff_str or basis1:
            sign = -1.0 if sign_str == '-' else 1.0
            coeff = float(coeff_str) if coeff_str else 1.0
            basis_key = basis1 if basis1 else ''
        else:
            sign = -1.0 if sign_str2 == '-' else 1.0
            coeff = 1.0
            basis_key = basis2
        
        value = sign * coeff
        basis_dict[basis_key] = basis_dict.get(basis_key, 0.0) + value
    
    # Ayrıca +e1, -e2 gibi ifadeleri yakala
    pattern2 = r'([+-])e(\d+)'
    matches2 = re.findall(pattern2, s)
    
    for sign_str, basis_key in matches2:
        sign = -1.0 if sign_str == '-' else 1.0
        basis_dict[basis_key] = basis_dict.get(basis_key, 0.0) + sign

    return CliffordNumber(basis_dict)


def _parse_dual(s) -> DualNumber:
    """String'i veya sayıyı DualNumber'a dönüştürür."""
    # Eğer zaten DualNumber ise doğrudan döndür
    if isinstance(s, DualNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) real kısım olarak işle
    if isinstance(s, (float, int, complex)):
        return DualNumber(float(s), 0.0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    parts = [p.strip() for p in s.split(',')]
    
    # Sadece ilk iki bileşeni al
    if len(parts) >= 2:
        try:
            return DualNumber(float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    elif len(parts) == 1: # Sadece real kısım verilmiş
        try:
            return DualNumber(float(parts[0]), 0.0)
        except ValueError:
            pass

    raise ValueError(f"Geçersiz Dual sayı formatı: '{s}' (Real, Dual veya sadece Real bekleniyor)")


def _parse_splitcomplex(s) -> SplitcomplexNumber:
    """String'i veya sayıyı SplitcomplexNumber'a dönüştürür."""
    # Eğer zaten SplitcomplexNumber ise doğrudan döndür
    if isinstance(s, SplitcomplexNumber):
        return s
    
    # Eğer sayısal tipse (float, int, complex) real kısım olarak işle
    if isinstance(s, (float, int, complex)):
        return SplitcomplexNumber(float(s), 0.0)
    
    # String işlemleri için önce string'e dönüştür
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 2:
        try:
            return SplitcomplexNumber(float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    elif len(parts) == 1: # Sadece real kısım verilmiş
        try:
            return SplitcomplexNumber(float(parts[0]), 0.0)
        except ValueError:
            pass

    raise ValueError(f"Geçersiz Split-Complex sayı formatı: '{s}' (Real, Split veya sadece Real bekleniyor)")


def generate_octonion(w, x, y, z, e, f, g, h):
    """8 bileşenden bir oktonyon oluşturur."""
    return OctonionNumber(w, x, y, z, e, f, g, h)

# Bazı önemli oktonyon sabitleri
ZERO = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 0)
ONE = OctonionNumber(1, 0, 0, 0, 0, 0, 0, 0)
I = OctonionNumber(0, 1, 0, 0, 0, 0, 0, 0)
J = OctonionNumber(0, 0, 1, 0, 0, 0, 0, 0)
K = OctonionNumber(0, 0, 0, 1, 0, 0, 0, 0)
E = OctonionNumber(0, 0, 0, 0, 1, 0, 0, 0)
F = OctonionNumber(0, 0, 0, 0, 0, 1, 0, 0)
G = OctonionNumber(0, 0, 0, 0, 0, 0, 1, 0)
H = OctonionNumber(0, 0, 0, 0, 0, 0, 0, 1)


def _parse_quaternion(s: str) -> quaternion:
    """Parses user string ('a+bi+cj+dk' or scalar) into a quaternion."""
    s_clean = s.replace(" ", "").lower()
    if not s_clean:
        raise ValueError("Input cannot be empty.")

    try:
        val = float(s_clean)
        return quaternion(val, val, val, val)
    except ValueError:
        pass
    
    s_temp = re.sub(r'([+-])([ijk])', r'\g<1>1\g<2>', s_clean)
    if s_temp.startswith(('i', 'j', 'k')):
        s_temp = '1' + s_temp
    
    pattern = re.compile(r'([+-]?\d*\.?\d*)([ijk])?')
    matches = pattern.findall(s_temp)
    
    parts = {'w': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    for value_str, component in matches:
        if not value_str:
            continue
        value = float(value_str)
        if component == 'i':
            parts['x'] += value
        elif component == 'j':
            parts['y'] += value
        elif component == 'k':
            parts['z'] += value
        else:
            parts['w'] += value
            
    return quaternion(parts['w'], parts['x'], parts['y'], parts['z'])

def _parse_superreal(s) -> SuperrealNumber:
    """String'i veya sayıyı SuperrealNumber'a dönüştürür."""
    if isinstance(s, SuperrealNumber):
        return s

    if isinstance(s, (float, int)):
        return SuperrealNumber(float(s), 0.0)

    if isinstance(s, complex):
        return SuperrealNumber(s.real, s.imag)

    if hasattr(s, '__iter__') and not isinstance(s, str):
        if len(s) == 2:
            return SuperrealNumber(float(s[0]), float(s[1]))
        else:
            raise ValueError("SuperrealNumber için 2 bileşen gereklidir.")

    # String işlemleri
    if not isinstance(s, str):
        s = str(s)

    s = s.strip().strip('()[]')
    parts = [p.strip() for p in s.split(',')]

    if len(parts) == 2:
        try:
            real = float(parts[0])
            split = float(parts[1])
            return SuperrealNumber(real, split)
        except ValueError as e:
            raise ValueError(f"Geçersiz SuperrealNumber bileşen değeri: '{s}' -> {e}") from e
    elif len(parts) == 1:
        try:
            real = float(parts[0])
            return SuperrealNumber(real, 0.0)
        except ValueError as e:
            raise ValueError(f"Geçersiz SuperrealNumber skaler değeri: '{s}' -> {e}") from e
    else:
        raise ValueError("SuperrealNumber için 1 veya 2 bileşen gereklidir.")

def _parse_ternary(s) -> TernaryNumber:
    """String'i veya sayıyı TernaryNumber'a dönüştürür."""
    if isinstance(s, TernaryNumber):
        return s

    if isinstance(s, (float, int)):
        # Sayıyı üçlü sayı sistemine dönüştür (örneğin, 11 -> "102")
        # Burada basitçe skaler bir değer olarak işleniyor
        return TernaryNumber.from_decimal(int(s))

    if hasattr(s, '__iter__') and not isinstance(s, str):
        return TernaryNumber(list(s))

    # String işlemleri
    if not isinstance(s, str):
        s = str(s)

    s = s.strip().strip('()[]')
    # Üçlü sayı sistemindeki geçersiz karakterleri kontrol et
    if not all(c in '012' for c in s):
        raise ValueError(f"Geçersiz üçlü sayı formatı: '{s}'")

    return TernaryNumber.from_ternary_string(s)

def get_random_type(num_iterations: int, fixed_start_raw: str = "0", fixed_add_base_scalar: float = 9.0) -> List[Any]:
    """Generates Keçeci Numbers for a randomly selected type."""
    random_type_choice = random.randint(1, 22)
    type_names_list = [
        "Positive Real", "Negative Real", "Complex", "Float", "Rational", 
        "Quaternion", "Neutrosophic", "Neutrosophic Complex", "Hyperreal", 
        "Bicomplex", "Neutrosophic Bicomplex", "Octonion", "Sedenion", "Clifford", "Dual", "Split-Complex",
        "Pathion", "Chingon", "Routon", "Voudon", "Super Real", "Ternary",
    ]
    logger.info("Randomly selected Keçeci Number Type: %d (%s)", random_type_choice, type_names_list[random_type_choice-1])
    
    return get_with_params(
        kececi_type_choice=random_type_choice, 
        iterations=num_iterations,
        start_value_raw=fixed_start_raw,
        add_value_raw=fixed_add_base_scalar
    )

def find_kececi_prime_number(kececi_numbers_list: List[Any]) -> Optional[int]:
    """Finds the Keçeci Prime Number from a generated sequence."""
    if not kececi_numbers_list:
        return None

    integer_prime_reps = [
        rep for num in kececi_numbers_list
        if is_prime(num) and (rep := _get_integer_representation(num)) is not None
    ]

    if not integer_prime_reps:
        return None

    counts = collections.Counter(integer_prime_reps)
    repeating_primes = [(freq, prime) for prime, freq in counts.items() if freq > 1]
    if not repeating_primes:
        return None
    
    _, best_prime = max(repeating_primes)
    return best_prime

def print_detailed_report(sequence: List[Any], params: Dict[str, Any], show_all: bool = False) -> None:
    """Generates and logs a detailed report of the sequence results.

    Args:
        sequence: generated sequence (list)
        params: dict of parameters used to generate the sequence
        show_all: if True, include full sequence in the log; otherwise only preview
    """
    if not sequence:
        logger.info("--- REPORT ---\nSequence could not be generated.")
        return

    logger.info("\n" + "="*50)
    logger.info("--- DETAILED SEQUENCE REPORT ---")
    logger.info("="*50)

    logger.info("[Parameters Used]")
    logger.info("  - Keçeci Type:   %s (%s)", params.get('type_name', 'N/A'), params.get('type_choice'))
    logger.info("  - Start Value:   %r", params.get('start_val'))
    logger.info("  - Increment:     %r", params.get('add_val'))
    logger.info("  - Keçeci Steps:  %s", params.get('steps'))

    logger.info("[Sequence Summary]")
    logger.info("  - Total Numbers Generated: %d", len(sequence))

    kpn = find_kececi_prime_number(sequence)
    logger.info("  - Keçeci Prime Number (KPN): %s", kpn if kpn is not None else "Not found")

    preview_count = min(len(sequence), 40)
    logger.info("  --- First %d Numbers ---", preview_count)
    for i in range(preview_count):
        logger.info("    %d: %s", i, sequence[i])

    if len(sequence) > preview_count:
        logger.info("  --- Last %d Numbers ---", preview_count)
        for i in range(len(sequence) - preview_count, len(sequence)):
            logger.info("    %d: %s", i, sequence[i])

    if show_all:
        logger.info("--- FULL SEQUENCE ---")
        for i, num in enumerate(sequence):
            logger.info("%d: %s", i, num)
        logger.info("="*50)

def _is_divisible(value: Any, divisor: int, kececi_type: int) -> bool:
    """
    Robust divisibility check across Keçeci types.

    Returns True if value is divisible by divisor according to the semantics
    of the given kececi_type, otherwise False.
    """
    TOLERANCE = 1e-12

    if divisor == 0:
        return False

    def _float_mod_zero(x: Any, divisor: int, tol: float = 1e-12) -> bool:
        """
        Module-level helper: returns True if float(x) % divisor ≈ 0 within tol.
        Safe against exceptions (returns False on error).
        """
        try:
            return math.isclose(float(x) % divisor, 0.0, abs_tol=tol)
        except Exception:
            return False

    def _int_mod_zero(x: int) -> bool:
        try:
            return int(round(x)) % divisor == 0
        except Exception:
            return False

    def _fraction_mod_zero(fr: Fraction) -> bool:
        # fr = n/d  -> divisible by divisor iff n % (d*divisor) == 0
        try:
            return fr.numerator % (fr.denominator * divisor) == 0
        except Exception:
            return False

    def _complex_mod_zero(c: complex) -> bool:
        return _float_mod_zero(c.real) and _float_mod_zero(c.imag)

    def _iterable_mod_zero(iterable) -> bool:
        try:
            for c in iterable:
                if isinstance(c, Fraction):
                    if not _fraction_mod_zero(c):
                        return False
                elif isinstance(c, complex):
                    if not _complex_mod_zero(c):
                        return False
                elif isinstance(c, (int, np.integer)):
                    if not _int_mod_zero(c):
                        return False
                else:
                    if not _float_mod_zero(c):
                        return False
            return True
        except Exception:
            return False

    try:
        # DualNumber: check .real
        if kececi_type == TYPE_DUAL:
            if hasattr(value, 'real'):
                return _float_mod_zero(getattr(value, 'real'))
            # fallback: element-wise if iterable
            if hasattr(value, 'coeffs') or hasattr(value, '__iter__'):
                return _iterable_mod_zero(getattr(value, 'coeffs', value))
            return False

        # Positive/Negative Real: only accept near-integers
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            if isinstance(value, (int, np.integer)):
                return _int_mod_zero(value)
            if isinstance(value, Fraction):
                return _fraction_mod_zero(value)
            try:
                val = float(value)
                if is_near_integer(val):
                    return _int_mod_zero(val)
            except Exception:
                pass
            return False

        # Float
        if kececi_type == TYPE_FLOAT:
            try:
                return _float_mod_zero(float(value))
            except Exception:
                return False

        # Rational (Fraction)
        if kececi_type == TYPE_RATIONAL:
            if isinstance(value, Fraction):
                return _fraction_mod_zero(value)
            # try to coerce
            try:
                fr = Fraction(value)
                return _fraction_mod_zero(fr)
            except Exception:
                return False

        # Complex
        if kececi_type == TYPE_COMPLEX:
            try:
                c = value if isinstance(value, complex) else _parse_complex(value)
                return _complex_mod_zero(c)
            except Exception:
                return False

        # quaternion-like (numpy quaternion or object with w,x,y,z)
        # quaternion branch within _is_divisible:
        if kececi_type == TYPE_QUATERNION:
            try:
                if quaternion is not None and isinstance(value, quaternion):
                    comps = [value.w, value.x, value.y, value.z]
                    return all(_float_mod_zero(c, divisor) for c in comps)
                if hasattr(value, 'w') and hasattr(value, 'x'):
                    components = [getattr(value, a) for a in ('w', 'x', 'y', 'z')]
                    return all(_float_mod_zero(c, divisor) for c in components)
                # fallback: iterable
                if hasattr(value, 'coeffs') or hasattr(value, '__iter__'):
                    return _iterable_mod_zero(getattr(value, 'coeffs', value))
            except Exception:
                return False
            return False

        # Neutrosophic (t,i,f) or objects with a,b attributes
        if kececi_type == TYPE_NEUTROSOPHIC:
            try:
                if hasattr(value, 't') and hasattr(value, 'i'):
                    return _float_mod_zero(value.t) and _float_mod_zero(value.i)
                if hasattr(value, 'a') and hasattr(value, 'b'):
                    return _float_mod_zero(value.a) and _float_mod_zero(value.b)
            except Exception:
                return False
            return False

        # Neutrosophic Complex
        if kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            try:
                comps = []
                if hasattr(value, 'real') and hasattr(value, 'imag'):
                    comps.append(value.real)
                    comps.append(value.imag)
                if hasattr(value, 'indeterminacy'):
                    comps.append(value.indeterminacy)
                return all(_float_mod_zero(c, divisor) for c in comps) if comps else False
            except Exception:
                return False

        # Hyperreal: check all sequence components
        if kececi_type == TYPE_HYPERREAL:
            if hasattr(value, 'sequence') and isinstance(value.sequence, (list, tuple)):
                return all(_float_mod_zero(x) for x in value.sequence)
            return False

        # Bicomplex
        if kececi_type == TYPE_BICOMPLEX:
            try:
                if hasattr(value, 'z1') and hasattr(value, 'z2'):
                    return _complex_mod_zero(value.z1) and _complex_mod_zero(value.z2)
            except Exception:
                return False
            return False

        # Neutrosophic Bicomplex
        if kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            try:
                comps = [getattr(value, attr) for attr in ['a','b','c','d','e','f','g','h'] if hasattr(value, attr)]
                return all(_float_mod_zero(c, divisor) for c in comps) if comps else False
            except Exception:
                return False

        # Octonion / Sedenion / Pathion / Chingon / Routon / Voudon / generic hypercomplex with coeffs
        if kececi_type in [TYPE_OCTONION, TYPE_SEDENION, TYPE_PATHION, TYPE_CHINGON, TYPE_ROUTON, TYPE_VOUDON]:
            try:
                if hasattr(value, 'coeffs'):
                    coeffs = getattr(value, 'coeffs')
                    return _iterable_mod_zero(coeffs)
                # fallback: attributes like w,x,y,z,... or iterable
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    return _iterable_mod_zero(value)
                # single scalar fallback
                return _float_mod_zero(value)
            except Exception:
                return False

        # Clifford
        if kececi_type == TYPE_CLIFFORD:
            try:
                if hasattr(value, 'basis') and isinstance(value.basis, dict):
                    return all(_float_mod_zero(v) for v in value.basis.values())
                return False
            except Exception:
                return False

        # Split-complex
        if kececi_type == TYPE_SPLIT_COMPLEX:
            try:
                if hasattr(value, 'real') and hasattr(value, 'split'):
                    return _float_mod_zero(value.real) and _float_mod_zero(value.split)
            except Exception:
                return False
            return False

        # Superreal
        if kececi_type == TYPE_SUPERREAL:
            try:
                if hasattr(value, 'real') and hasattr(value, 'split'):
                    return _float_mod_zero(value.real) and _float_mod_zero(value.split)
            except Exception:
                return False
            return False

        # Ternary
        if kececi_type == TYPE_TERNARY:
            try:
                if hasattr(value, 'to_decimal'):
                    dec = value.to_decimal()
                    return _int_mod_zero(dec)
                if hasattr(value, 'digits'):
                    dec = 0
                    for i, d in enumerate(reversed(list(value.digits))):
                        dec += int(d) * (3 ** i)
                    return _int_mod_zero(dec)
                # if iterable digits
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    parts = list(value)
                    dec = 0
                    for i, d in enumerate(reversed(parts)):
                        dec += int(d) * (3 ** i)
                    return _int_mod_zero(dec)
            except Exception:
                return False
            return False

        # Fallback: try coeffs -> iterable -> numeric coercion
        if hasattr(value, 'coeffs'):
            return _iterable_mod_zero(getattr(value, 'coeffs'))
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            return _iterable_mod_zero(value)

        # Last resort: numeric conversion
        try:
            return _float_mod_zero(float(value))
        except Exception:
            return False

    except (TypeError, AttributeError, ValueError, ZeroDivisionError):
        return False

def _get_integer_representation(n_input: Any) -> Optional[int]:
    """
    Extracts the primary integer component from supported Keçeci number types.

    Returns:
        absolute integer value (int) when a meaningful integer representation exists,
        otherwise None.
    """
    try:
        # None early exit
        if n_input is None:
            return None

        # Direct ints (including numpy ints)
        if isinstance(n_input, (int, np.integer)):
            return abs(int(n_input))

        # Fractions: only return if it's an integer fraction (denominator == 1)
        if isinstance(n_input, Fraction):
            if n_input.denominator == 1:
                return abs(int(n_input.numerator))
            return None

        # Floats: accept only if near integer
        if isinstance(n_input, (float, np.floating)):
            if is_near_integer(n_input):
                return abs(int(round(float(n_input))))
            return None

        # Complex: require imag ≈ 0 and real near-integer
        if isinstance(n_input, complex):
            if abs(n_input.imag) < 1e-12 and is_near_integer(n_input.real):
                return abs(int(round(n_input.real)))
            return None

        # numpy-quaternion or other quaternion types where 'w' is scalar part
        try:
            # `quaternion` type from numpy-quaternion has attribute 'w'
            #if isinstance(n_input, quaternion):
            #    w = getattr(n_input, 'w', None)
            #    if w is not None and is_near_integer(w):
            #        return abs(int(round(float(w)))
            if quaternion is not None and isinstance(n_input, quaternion):
                if is_near_integer(n_input.w):
                    return abs(int(round(n_input.w)))
                return None
        except Exception:
            # If quaternion type is not available or isinstance check fails, continue
            pass

        # If object exposes 'coeffs' (list/np.array), use first component
        if hasattr(n_input, 'coeffs'):
            coeffs = getattr(n_input, 'coeffs')
            # numpy array
            if isinstance(coeffs, np.ndarray):
                if coeffs.size > 0 and is_near_integer(coeffs.flatten()[0]):
                    return abs(int(round(float(coeffs.flatten()[0]))))
                return None
            # list/tuple-like
            try:
                # convert to list (works for many iterables)
                c0 = list(coeffs)[0]
                if is_near_integer(c0):
                    return abs(int(round(float(c0))))
                return None
            except Exception:
                # can't iterate coeffs reliably
                pass

        # Some classes expose 'coefficients' name instead
        if hasattr(n_input, 'coefficients'):
            try:
                c0 = list(getattr(n_input, 'coefficients'))[0]
                if is_near_integer(c0):
                    return abs(int(round(float(c0))))
            except Exception:
                pass

        # Try common scalar attributes in order of likelihood
        for attr in ('w', 'real', 't', 'a', 'value'):
            if hasattr(n_input, attr):
                val = getattr(n_input, attr)
                # If this is complex-like, use real part
                if isinstance(val, complex):
                    if abs(val.imag) < 1e-12 and is_near_integer(val.real):
                        return abs(int(round(val.real)))
                else:
                    try:
                        if is_near_integer(val):
                            return abs(int(round(float(val))))
                    except Exception:
                        pass

        # CliffordNumber: check basis dict scalar part ''
        if hasattr(n_input, 'basis') and isinstance(getattr(n_input, 'basis'), dict):
            scalar = n_input.basis.get('', 0)
            try:
                if is_near_integer(scalar):
                    return abs(int(round(float(scalar))))
            except Exception:
                pass

        # DualNumber / Superreal / others: if they expose .real attribute (and it's numeric)
        if hasattr(n_input, 'real') and not isinstance(n_input, (complex, float, int, np.floating, np.integer)):
            try:
                real_val = getattr(n_input, 'real')
                if is_near_integer(real_val):
                    return abs(int(round(float(real_val))))
            except Exception:
                pass

        # TernaryNumber: convert digits to decimal
        if hasattr(n_input, 'digits'):
            try:
                digits = list(n_input.digits)
                decimal_value = 0
                for i, d in enumerate(reversed(digits)):
                    decimal_value += int(d) * (3 ** i)
                return abs(int(decimal_value))
            except Exception:
                pass

        # HyperrealNumber: use finite part (sequence[0]) if present
        if hasattr(n_input, 'sequence') and isinstance(getattr(n_input, 'sequence'), (list, tuple)):
            seq = getattr(n_input, 'sequence')
            if seq:
                try:
                    if is_near_integer(seq[0]):
                        return abs(int(round(float(seq[0]))))
                except Exception:
                    pass

        # Fallback: try numeric coercion + is_near_integer
        try:
            if is_near_integer(n_input):
                return abs(int(round(float(n_input))))
        except Exception:
            pass

        # If nothing matched, return None
        return None

    except Exception:
        # On any unexpected failure, return None rather than raising
        return None


def is_prime(n_input: Any) -> bool:
    """
    Checks if a given number (or its principal component) is prime
    using the robust sympy.isprime function.
    """
    # Adım 1: Karmaşık sayı türünden tamsayıyı çıkarma (Bu kısım aynı kalıyor)
    value_to_check = _get_integer_representation(n_input)

    # Adım 2: Tamsayı geçerli değilse False döndür
    if value_to_check is None:
        return False
    
    # Adım 3: Asallık testini sympy'ye bırak
    # sympy.isprime, 2'den küçük sayılar (1, 0, negatifler) için zaten False döndürür.
    return sympy.isprime(value_to_check)


def is_near_integer(x, tol=1e-12):
    """
    Checks if a number (or its real part) is close to an integer.
    Useful for float-based primality and divisibility checks.
    """
    try:
        if isinstance(x, complex):
            # Sadece gerçek kısım önemli, imajiner sıfıra yakın olmalı
            if abs(x.imag) > tol:
                return False
            x = x.real
        elif isinstance(x, (list, tuple)):
            return False  # Desteklenmeyen tip

        # Genel durum: float veya int
        x = float(x)
        return abs(x - round(x)) < tol
    except:
        return False

def is_prime_like(value: Any, kececi_type: int) -> bool:
    """
    Heuristic to check whether `value` should be treated as a "prime candidate"
    under Keçeci logic for the given `kececi_type`.

    Strategy:
    - Prefer using _get_integer_representation when possible.
    - For quaternion/hypercomplex types require that component(s) are near-integers.
    - For ternary convert to decimal first.
    """
    try:
        # First, try a general integer extraction
        n = _get_integer_representation(value)
        if n is not None:
            return sympy.isprime(int(n))

        # Handle quaternion specifically: require all components near-integer then test skalar
        if kececi_type == TYPE_QUATERNION:
            try:
                #if isinstance(value, quaternion):
                    #comps = [value.w, value.x, value.y, value.z]
                if quaternion is not None and isinstance(value, quaternion):
                    comps = [value.w, value.x, value.y, value.z]
                    return all(_float_mod_zero(c, divisor) for c in comps)
                elif hasattr(value, 'w') and hasattr(value, 'x'):
                    comps = [getattr(value, a) for a in ('w','x','y','z')]
                elif hasattr(value, 'coeffs'):
                    comps = list(getattr(value, 'coeffs'))
                else:
                    return False
                if not all(is_near_integer(c) for c in comps):
                    return False
                return sympy.isprime(int(round(float(comps[0]))))
            except Exception:
                return False

        # Hypercomplex families: check coeffs exist and are integer-like, test first (scalar) component
        if kececi_type in [TYPE_OCTONION, TYPE_SEDENION, TYPE_PATHION, TYPE_CHINGON, TYPE_ROUTON, TYPE_VOUDON]:
            try:
                if hasattr(value, 'coeffs'):
                    coeffs = list(getattr(value, 'coeffs'))
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    coeffs = list(value)
                else:
                    return False
                if not coeffs:
                    return False
                if not all(is_near_integer(c) for c in coeffs):
                    return False
                return sympy.isprime(int(round(float(coeffs[0]))))
            except Exception:
                return False

        # Ternary
        if kececi_type == TYPE_TERNARY:
            try:
                if hasattr(value, 'to_decimal'):
                    dec = value.to_decimal()
                elif hasattr(value, 'digits'):
                    dec = 0
                    for i, d in enumerate(reversed(list(value.digits))):
                        dec += int(d) * (3 ** i)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    parts = list(value)
                    dec = 0
                    for i, d in enumerate(reversed(parts)):
                        dec += int(d) * (3 ** i)
                else:
                    return False
                return sympy.isprime(int(dec))
            except Exception:
                return False

        # Clifford: scalar basis part
        if kececi_type == TYPE_CLIFFORD:
            try:
                if hasattr(value, 'basis') and isinstance(value.basis, dict):
                    scalar = value.basis.get('', 0)
                    if is_near_integer(scalar):
                        return sympy.isprime(int(round(float(scalar))))
                return False
            except Exception:
                return False

        # Superreal: require integer-like real (and optional split)
        if kececi_type == TYPE_SUPERREAL:
            try:
                if hasattr(value, 'real'):
                    real = getattr(value, 'real')
                    if is_near_integer(real):
                        return sympy.isprime(int(round(float(real))))
                return False
            except Exception:
                return False

        # Fallback conservative behavior: not prime-like
        return False

    except Exception:
        return False

def generate_kececi_vectorial(q0_str, c_str, u_str, iterations):
    """
    Keçeci Haritası'nı tam vektörel toplama ile üreten geliştirilmiş fonksiyon.
    Bu, kütüphanenin ana üretim fonksiyonu olabilir.
    Tüm girdileri metin (string) olarak alarak esneklik sağlar.
    """
    try:
        # Girdi metinlerini kuaterniyon nesnelerine dönüştür
        w, x, y, z = map(float, q0_str.split(','))
        q0 = quaternion(w, x, y, z)
        
        cw, cx, cy, cz = map(float, c_str.split(','))
        c = quaternion(cw, cx, cy, cz)

        uw, ux, uy, uz = map(float, u_str.split(','))
        u = quaternion(uw, ux, uy, uz)

    except (ValueError, IndexError):
        raise ValueError("Girdi metinleri 'w,x,y,z' formatında olmalıdır.")

    trajectory = [q0]
    prime_events = []
    current_q = q0

    for i in range(iterations):
        y = current_q + c
        processing_val = y

        while True:
            scalar_int = int(processing_val.w)

            if scalar_int % 2 == 0:
                next_q = processing_val / 2.0
                break
            elif scalar_int % 3 == 0:
                next_q = processing_val / 3.0
                break
            elif is_prime(scalar_int):
                if processing_val == y:
                    prime_events.append((i, scalar_int))
                processing_val += u
                continue
            else:
                next_q = processing_val
                break
        
        trajectory.append(next_q)
        current_q = next_q
        
    return trajectory, prime_events

def analyze_all_types(iterations=120, additional_params=None):
    """
    Performs automated analysis on all Keçeci number types.
    - Uses module-level helpers (_find_kececi_zeta_zeros, _compute_gue_similarity, get_with_params, _plot_comparison).
    - Avoids heavy imports at module import time by importing lazily where needed.
    - Iterates over 1..TYPE_TERNARY (inclusive).
    Returns:
        (sorted_by_zeta, sorted_by_gue)
    """
    print("Automated Analysis for Keçeci Types")
    print("=" * 80)

    include_intermediate = True
    results = []

    # Default parameter sets (keçeçi testleri için örnekler)
    param_sets = [
        ('2.0', '3.0'),
        ('1+1j', '0.5+0.5j'),
        ('1.0,0.0,0.0,0.0', '0.1,0.0,0.0,0.0'),
        ('0.8,0.1,0.1', '0.0,0.05,0.0'),
        ('1.0', '0.1'),
        ('102', '1'),
    ]

    if additional_params:
        param_sets.extend(additional_params)

    type_names = {
        1: "Positive Real", 2: "Negative Real", 3: "Complex", 4: "Float", 5: "Rational",
        6: "Quaternion", 7: "Neutrosophic", 8: "Neutro-Complex", 9: "Hyperreal", 10: "Bicomplex",
        11: "Neutro-Bicomplex", 12: "Octonion", 13: "Sedenion", 14: "Clifford", 15: "Dual",
        16: "Split-Complex", 17: "Pathion", 18: "Chingon", 19: "Routon", 20: "Voudon",
        21: "Super Real", 22: "Ternary",
    }

    # Iterate all defined types (inclusive)
    for kececi_type in range(TYPE_POSITIVE_REAL, TYPE_TERNARY + 1):
        name = type_names.get(kececi_type, f"Type {kececi_type}")
        best_zeta_score = 0.0
        best_gue_score = 0.0
        best_params = None

        print(f"\nAnalyzing type {kececi_type} ({name})...")

        for start, add in param_sets:
            try:
                # generate sequence (get_with_params is defined in this module)
                sequence = get_with_params(
                    kececi_type_choice=kececi_type,
                    iterations=iterations,
                    start_value_raw=start,
                    add_value_raw=add,
                    include_intermediate_steps=include_intermediate
                )

                if not sequence or len(sequence) < 20:
                    # Skip too-short sequences
                    # (analysis routines expect some minimal data)
                    print(f"  Skipped (insufficient length): params {start}, {add}")
                    continue

                # Lazy import heavy helper functions (they exist in-module)
                try:
                    zzeros, zeta_score = _find_kececi_zeta_zeros(sequence, tolerance=0.5)
                except Exception as zz_err:
                    zzeros, zeta_score = [], 0.0
                    print(f"  Warning: _find_kececi_zeta_zeros failed for {name} with params {start},{add}: {zz_err}")

                try:
                    gue_score, gue_p = _compute_gue_similarity(sequence)
                except Exception as gue_err:
                    gue_score, gue_p = 0.0, 0.0
                    print(f"  Warning: _compute_gue_similarity failed for {name} with params {start},{add}: {gue_err}")

                if zeta_score > best_zeta_score or (zeta_score == best_zeta_score and gue_score > best_gue_score):
                    best_zeta_score = zeta_score
                    best_gue_score = gue_score
                    best_params = (start, add)

            except Exception as e:
                print(f"  Error analyzing params ({start}, {add}) for type {kececi_type}: {e}")
                continue

        if best_params:
            results.append({
                'type': kececi_type,
                'name': name,
                'start': best_params[0],
                'add': best_params[1],
                'zeta_score': best_zeta_score,
                'gue_score': best_gue_score
            })
        else:
            print(f"  No successful parameter set found for {name}.")

    # Sort and display results
    sorted_by_zeta = sorted(results, key=lambda x: x['zeta_score'], reverse=True)
    sorted_by_gue = sorted(results, key=lambda x: x['gue_score'], reverse=True)

    # Plot comparison if there are results (lazy-plot)
    try:
        if sorted_by_zeta or sorted_by_gue:
            _plot_comparison(sorted_by_zeta, sorted_by_gue)
    except Exception as plot_err:
        print(f"Plotting failed: {plot_err}")

    return sorted_by_zeta, sorted_by_gue

def _load_zeta_zeros(filename="zeta.txt"):
    """
    Loads Riemann zeta zeros from a text file.
    Each line should contain one floating-point number representing the imaginary part of a zeta zero.
    Lines that are empty or start with '#' are ignored.
    Returns: numpy.ndarray of zeros, or empty array if file not found / invalid.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        zeta_zeros = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                zeta_zeros.append(float(line))
            except ValueError:
                logger.warning("Invalid line skipped in %s: %r", filename, line)
        logger.info("%d zeta zeros loaded from %s.", len(zeta_zeros), filename)
        return np.array(zeta_zeros)
    except FileNotFoundError:
        logger.warning("Zeta zeros file '%s' not found.", filename)
        return np.array([])
    except Exception as e:
        logger.exception("Error while loading zeta zeros from %s: %s", filename, e)
        return np.array([])


def _compute_gue_similarity(sequence, tolerance=0.5):
    """
    Measures how closely the frequency spectrum of a Keçeci sequence matches the GUE (Gaussian Unitary Ensemble) statistics.
    Uses Kolmogorov-Smirnov test against Wigner-Dyson distribution.
    Args:
        sequence (list): The Keçeci number sequence.
        tolerance (float): Not used here; kept for interface consistency.
    Returns:
        tuple: (similarity_score, p_value)
    """
    from . import _get_integer_representation

    values = [val for z in sequence if (val := _get_integer_representation(z)) is not None]
    if len(values) < 10:
        return 0.0, 0.0

    values = np.array(values) - np.mean(values)
    N = len(values)
    powers = np.abs(fft(values))**2
    freqs = fftfreq(N)

    mask = (freqs > 0)
    freqs_pos = freqs[mask]
    powers_pos = powers[mask]

    if len(powers_pos) == 0:
        return 0.0, 0.0

    peaks, _ = find_peaks(powers_pos, height=np.max(powers_pos)*1e-7)
    strong_freqs = freqs_pos[peaks]

    if len(strong_freqs) < 2:
        return 0.0, 0.0

    # Scale so the strongest peak aligns with the first Riemann zeta zero
    peak_freq = strong_freqs[np.argmax(powers_pos[peaks])]
    scale_factor = 14.134725 / peak_freq
    scaled_freqs = np.sort(strong_freqs * scale_factor)

    # Compute level spacings
    if len(scaled_freqs) < 2:
        return 0.0, 0.0
    diffs = np.diff(scaled_freqs)
    if np.mean(diffs) == 0:
        return 0.0, 0.0
    diffs_norm = diffs / np.mean(diffs)

    # Generate GUE sample using Wigner-Dyson distribution
    def wigner_dyson(s):
        return (32 / np.pi) * s**2 * np.exp(-4 * s**2 / np.pi)

    s_gue = np.linspace(0.01, 3.0, 1000)
    p_gue = wigner_dyson(s_gue)
    p_gue = p_gue / np.sum(p_gue)
    sample_gue = np.random.choice(s_gue, size=1000, p=p_gue)

    # Perform KS test
    ks_stat, ks_p = ks_2samp(diffs_norm, sample_gue)
    similarity_score = 1.0 - ks_stat

    return similarity_score, ks_p

def _plot_comparison(zeta_results, gue_results):
    """
    Generates bar charts comparing the performance of Keçeci types in matching Riemann zeta zeros and GUE statistics.
    Args:
        zeta_results (list): Results sorted by zeta matching score.
        gue_results (list): Results sorted by GUE similarity score.
    """
    # Riemann Zeta Matching Plot
    plt.figure(figsize=(14, 7))
    types = [r['name'] for r in zeta_results]
    scores = [r['zeta_score'] for r in zeta_results]
    colors = ['skyblue'] * len(scores)
    if scores:
        colors[0] = 'red'
    bars = plt.bar(types, scores, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Riemann Zeta Matching Score")
    plt.title("Keçeci Types vs Riemann Zeta Zeros")
    plt.grid(True, alpha=0.3)
    if bars:
        bars[0].set_edgecolor('darkred')
        bars[0].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()

    # GUE Similarity Plot
    plt.figure(figsize=(14, 7))
    types = [r['name'] for r in gue_results]
    scores = [r['gue_score'] for r in gue_results]
    colors = ['skyblue'] * len(scores)
    if scores:
        colors[0] = 'red'
    bars = plt.bar(types, scores, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("GUE Similarity Score")
    plt.title("Keçeci Types vs GUE Statistics")
    plt.grid(True, alpha=0.3)
    if bars:
        bars[0].set_edgecolor('darkred')
        bars[0].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()


def _find_kececi_zeta_zeros(sequence, tolerance=0.5):
    """
    Estimates the zeros of the Keçeci Zeta Function from the spectral peaks of the sequence.
    Compares them to known Riemann zeta zeros.
    Args:
        sequence (list): The Keçeci number sequence.
        tolerance (float): Maximum distance for a match between Keçeci and Riemann zeros.
    Returns:
        tuple: (list of Keçeci zeta zeros, matching score)
    """
    from . import _get_integer_representation

    values = [val for z in sequence if (val := _get_integer_representation(z)) is not None]
    if len(values) < 10:
        return [], 0.0

    values = np.array(values) - np.mean(values)
    N = len(values)
    powers = np.abs(fft(values))**2
    freqs = fftfreq(N)

    mask = (freqs > 0)
    freqs_pos = freqs[mask]
    powers_pos = powers[mask]

    if len(powers_pos) == 0:
        return [], 0.0

    peaks, _ = find_peaks(powers_pos, height=np.max(powers_pos)*1e-7)
    strong_freqs = freqs_pos[peaks]

    if len(strong_freqs) < 2:
        return [], 0.0

    # Scale so the strongest peak aligns with the first Riemann zeta zero
    peak_freq = strong_freqs[np.argmax(powers_pos[peaks])]
    scale_factor = 14.134725 / peak_freq
    scaled_freqs = np.sort(strong_freqs * scale_factor)

    # Find candidate zeros by analyzing the Keçeci Zeta Function
    t_vals = np.linspace(0, 650, 10000)
    zeta_vals = np.array([sum((scaled_freqs + 1e-10)**(- (0.5 + 1j * t))) for t in t_vals])
    minima, _ = find_peaks(-np.abs(zeta_vals), height=-0.5*np.max(np.abs(zeta_vals)), distance=5)
    kececi_zeta_zeros = t_vals[minima]

    # Load Riemann zeta zeros for comparison
    zeta_zeros_imag = _load_zeta_zeros("zeta.txt")
    if len(zeta_zeros_imag) == 0:
        return kececi_zeta_zeros, 0.0

    # Calculate matching score
    close_matches = [kz for kz in kececi_zeta_zeros if min(abs(kz - zeta_zeros_imag)) < tolerance]
    score = len(close_matches) / len(kececi_zeta_zeros) if kececi_zeta_zeros.size > 0 else 0.0

    return kececi_zeta_zeros, score


def _pair_correlation(ordered_zeros, max_gap=3.0, bin_size=0.1):
    """
    Computes the pair correlation of a list of ordered zeros.
    This function calculates the normalized spacings between all pairs of zeros
    and returns a histogram of their distribution.
    Args:
        ordered_zeros (numpy.ndarray): Sorted array of zero locations (e.g., Keçeci or Riemann zeta zeros).
        max_gap (float): Maximum normalized gap to consider.
        bin_size (float): Size of bins for the histogram.
    Returns:
        tuple: (bin_centers, histogram) - The centers of the bins and the normalized histogram values.
    """
    n = len(ordered_zeros)
    if n < 2:
        return np.array([]), np.array([])

    # Compute average spacing for normalization
    avg_spacing = np.mean(np.diff(ordered_zeros))
    normalized_zeros = ordered_zeros / avg_spacing

    # Compute all pairwise gaps within max_gap
    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(normalized_zeros[j] - normalized_zeros[i])
            if gap <= max_gap:
                gaps.append(gap)

    # Generate histogram
    bins = np.arange(0, max_gap + bin_size, bin_size)
    hist, _ = np.histogram(gaps, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, hist


def _gue_pair_correlation(s):
    """
    Theoretical pair correlation function for the Gaussian Unitary Ensemble (GUE).
    This function is used as a reference for comparing the statistical distribution
    of eigenvalues (or zeta zeros) in quantum chaotic systems.
    Args:
        s (numpy.ndarray or float): Normalized spacing(s).
    Returns:
        numpy.ndarray or float: The GUE pair correlation value(s) at s.
    """
    return 1 - np.sinc(s)**2


def analyze_pair_correlation(sequence, title="Pair Correlation of Keçeci Zeta Zeros"):
    """
    Analyzes and plots the pair correlation of Keçeci Zeta zeros derived from a Keçeci sequence.
    Compares the empirical pair correlation to the theoretical GUE prediction.
    Performs a Kolmogorov-Smirnov test to quantify the similarity.
    Args:
        sequence (list): A Keçeci number sequence.
        title (str): Title for the resulting plot.
    """
    from . import _get_integer_representation

    # Extract integer representations and remove DC component
    values = [val for z in sequence if (val := _get_integer_representation(z)) is not None]
    if len(values) < 10:
        print("Insufficient data.")
        return

    values = np.array(values) - np.mean(values)
    N = len(values)
    powers = np.abs(fft(values))**2
    freqs = fftfreq(N)

    # Filter positive frequencies
    mask = (freqs > 0)
    freqs_pos = freqs[mask]
    powers_pos = powers[mask]

    if len(powers_pos) == 0:
        print("No positive frequencies found.")
        return

    # Find spectral peaks
    peaks, _ = find_peaks(powers_pos, height=np.max(powers_pos)*1e-7)
    strong_freqs = freqs_pos[peaks]

    if len(strong_freqs) < 2:
        print("Insufficient frequency peaks.")
        return

    # Scale frequencies so the strongest peak aligns with the first Riemann zeta zero
    peak_freq = strong_freqs[np.argmax(powers_pos[peaks])]
    scale_factor = 14.134725 / peak_freq
    scaled_freqs = np.sort(strong_freqs * scale_factor)

    # Estimate Keçeci Zeta zeros by finding minima of |ζ_Kececi(0.5 + it)|
    t_vals = np.linspace(0, 650, 10000)
    zeta_vals = np.array([sum((scaled_freqs + 1e-10)**(- (0.5 + 1j * t))) for t in t_vals])
    minima, _ = find_peaks(-np.abs(zeta_vals), height=-0.5*np.max(np.abs(zeta_vals)), distance=5)
    kececi_zeta_zeros = t_vals[minima]

    if len(kececi_zeta_zeros) < 2:
        print("Insufficient Keçeci zeta zeros found.")
        return

    # Compute pair correlation
    bin_centers, hist = _pair_correlation(kececi_zeta_zeros, max_gap=3.0, bin_size=0.1)
    gue_corr = _gue_pair_correlation(bin_centers)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, hist, 'o-', label="Keçeci Zeta Zeros", linewidth=2)
    plt.plot(bin_centers, gue_corr, 'r-', label="GUE (Theoretical)", linewidth=2)
    plt.title(title)
    plt.xlabel("Normalized Spacing (s)")
    plt.ylabel("Pair Correlation Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_p = ks_2samp(hist, gue_corr)
    print(f"Pair Correlation KS Test: Statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

# ==============================================================================
# --- CORE GENERATOR ---
# ==============================================================================
def unified_generator(kececi_type: int, start_input_raw: str, add_input_raw: str, iterations: int, include_intermediate_steps: bool = False) -> List[Any]:
    """
    Unified generator with input validation and robust division/ask handling.
    """
    if not (TYPE_POSITIVE_REAL <= kececi_type <= TYPE_TERNARY):
        raise ValueError(f"Invalid Keçeci Number Type: {kececi_type}")

    # Sanitize raw inputs
    start_input_raw = "0" if start_input_raw is None or str(start_input_raw).strip() == "" else start_input_raw
    add_input_raw = "1" if add_input_raw is None or str(add_input_raw).strip() == "" else add_input_raw

    use_integer_division = False
    current_value = None
    add_value_typed = None
    ask_unit = None

    # Helper for safe division choosing appropriate operator
    def _safe_divide(val, divisor, integer_mode):
        try:
            if integer_mode:
                # prefer __floordiv__ if available, fallback to integer division of float
                if hasattr(val, '__floordiv__'):
                    return val // divisor
                else:
                    return type(val)(val / divisor) if not isinstance(val, (int, float)) else val // divisor
            else:
                if hasattr(val, '__truediv__'):
                    return val / divisor
                else:
                    # fallback numeric
                    return type(val)(float(val) / float(divisor))
        except Exception as e:
            logger.debug("Division failed for %r by %s: %s", val, divisor, e)
            raise

    # --- Type-specific parsing with defensive error reporting ---
    try:
        if kececi_type in [TYPE_POSITIVE_REAL, TYPE_NEGATIVE_REAL]:
            try:
                current_value = int(float(start_input_raw))
                add_value_typed = int(float(add_input_raw))
            except Exception as e:
                logger.error("Failed to parse integer inputs: start=%r add=%r -> %s", start_input_raw, add_input_raw, e)
                raise
            ask_unit = 1
            use_integer_division = True

        elif kececi_type == TYPE_FLOAT:
            current_value = float(start_input_raw)
            add_value_typed = float(add_input_raw)
            ask_unit = 1.0

        elif kececi_type == TYPE_RATIONAL:
            current_value = Fraction(start_input_raw)
            add_value_typed = Fraction(add_input_raw)
            ask_unit = Fraction(1)

        elif kececi_type == TYPE_COMPLEX:
            current_value = _parse_complex(start_input_raw)
            add_value_typed = _parse_complex(add_input_raw)
            ask_unit = 1 + 1j

        elif kececi_type == TYPE_QUATERNION:
            current_value = _parse_quaternion_from_csv(start_input_raw)
            add_value_typed = _parse_quaternion_from_csv(add_input_raw)
            ask_unit = quaternion(1, 1, 1, 1)

        elif kececi_type == TYPE_NEUTROSOPHIC:
            t, i, f = _parse_neutrosophic(start_input_raw)
            current_value = NeutrosophicNumber(t, i, f)
            t_inc, i_inc, f_inc = _parse_neutrosophic(add_input_raw)
            add_value_typed = NeutrosophicNumber(t_inc, i_inc, f_inc)
            ask_unit = NeutrosophicNumber(1, 1, 1)

        elif kececi_type == TYPE_NEUTROSOPHIC_COMPLEX:
            s_complex = _parse_complex(start_input_raw)
            current_value = NeutrosophicComplexNumber(s_complex.real, s_complex.imag, 0.0)
            a_complex = _parse_complex(add_input_raw)
            add_value_typed = NeutrosophicComplexNumber(a_complex.real, a_complex.imag, 0.0)
            ask_unit = NeutrosophicComplexNumber(1, 1, 1)

        elif kececi_type == TYPE_HYPERREAL:
            finite, infinitesimal = _parse_hyperreal(start_input_raw)
            current_value = HyperrealNumber([finite, infinitesimal])
            finite_inc, infinitesimal_inc = _parse_hyperreal(add_input_raw)
            add_value_typed = HyperrealNumber([finite_inc, infinitesimal_inc])
            ask_unit = HyperrealNumber([1.0, 1.0])

        elif kececi_type == TYPE_BICOMPLEX:
            current_value = _parse_bicomplex(start_input_raw)
            add_value_typed = _parse_bicomplex(add_input_raw)
            ask_unit = BicomplexNumber(complex(1, 1), complex(1, 1))

        elif kececi_type == TYPE_NEUTROSOPHIC_BICOMPLEX:
            current_value = _parse_neutrosophic_bicomplex(start_input_raw)
            add_value_typed = _parse_neutrosophic_bicomplex(add_input_raw)
            ask_unit = NeutrosophicBicomplexNumber(1, 1, 1, 1, 1, 1, 1, 1)

        elif kececi_type == TYPE_OCTONION:
            current_value = _parse_octonion(start_input_raw)
            add_value_typed = _parse_octonion(add_input_raw)
            ask_unit = OctonionNumber(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        elif kececi_type == TYPE_SEDENION:
            current_value = _parse_sedenion(start_input_raw)
            add_value_typed = _parse_sedenion(add_input_raw)
            ask_unit = SedenionNumber([1.0] + [0.0] * 15)

        elif kececi_type == TYPE_CLIFFORD:
            current_value = _parse_clifford(start_input_raw)
            add_value_typed = _parse_clifford(add_input_raw)
            ask_unit = CliffordNumber({'': 1.0})

        elif kececi_type == TYPE_DUAL:
            current_value = _parse_dual(start_input_raw)
            add_value_typed = _parse_dual(add_input_raw)
            ask_unit = DualNumber(1.0, 1.0)

        elif kececi_type == TYPE_SPLIT_COMPLEX:
            current_value = _parse_splitcomplex(start_input_raw)
            add_value_typed = _parse_splitcomplex(add_input_raw)
            ask_unit = SplitcomplexNumber(1.0, 1.0)

        elif kececi_type == TYPE_PATHION:
            current_value = _parse_pathion(start_input_raw)
            add_value_typed = _parse_pathion(add_input_raw)
            ask_unit = PathionNumber([1.0] + [0.0] * 31)

        elif kececi_type == TYPE_CHINGON:
            current_value = _parse_chingon(start_input_raw)
            add_value_typed = _parse_chingon(add_input_raw)
            ask_unit = ChingonNumber([1.0] + [0.0] * 63)

        elif kececi_type == TYPE_ROUTON:
            current_value = _parse_routon(start_input_raw)
            add_value_typed = _parse_routon(add_input_raw)
            ask_unit = RoutonNumber([1.0] + [0.0] * 127)

        elif kececi_type == TYPE_VOUDON:
            current_value = _parse_voudon(start_input_raw)
            add_value_typed = _parse_voudon(add_input_raw)
            ask_unit = VoudonNumber([1.0] + [0.0] * 255)

        elif kececi_type == TYPE_SUPERREAL:
            current_value = _parse_superreal(start_input_raw)
            add_value_typed = _parse_superreal(add_input_raw)
            ask_unit = SuperrealNumber(1.0, 0.0)

        elif kececi_type == TYPE_TERNARY:
            current_value = _parse_ternary(start_input_raw)
            add_value_typed = _parse_ternary(add_input_raw)
            ask_unit = TernaryNumber([1])

        else:
            raise ValueError(f"Unsupported Keçeci type: {kececi_type}")

    except (ValueError, TypeError) as e:
        logger.exception("Failed to initialize type %s with start=%r add=%r: %s", kececi_type, start_input_raw, add_input_raw, e)
        return []

    # Generation loop
    clean_trajectory = [current_value]
    # full_log = [current_value]
    # full_log sadece include_intermediate_steps True ise tutulacak (bellek kullanımını azaltmak için)
    full_log = [current_value] if include_intermediate_steps else None
    last_divisor_used = None
    ask_counter = 0  # 0: +ask_unit, 1: -ask_unit

    for step in range(iterations):
        # 1. Add
        try:
            added_value = current_value + add_value_typed
        except Exception as e:
            logger.exception("Addition failed at step %s: %s", step, e)
            # cannot continue if addition is invalid for type
            break

        next_q = added_value
        divided_successfully = False
        modified_value = None

        # Choose which divisor to try first
        primary_divisor = 3 if last_divisor_used == 2 or last_divisor_used is None else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3

        # 2. Try divisions defensively
        for divisor in (primary_divisor, alternative_divisor):
            try:
                if _is_divisible(added_value, divisor, kececi_type):
                    try:
                        next_q = _safe_divide(added_value, divisor, use_integer_division)
                        last_divisor_used = divisor
                        divided_successfully = True
                    except Exception:
                        # if safe divide failed, try simple numeric fallback
                        try:
                            next_q = added_value / divisor
                            last_divisor_used = divisor
                            divided_successfully = True
                        except Exception:
                            logger.debug("Fallback division failed for %r by %s", added_value, divisor)
                    break
            except Exception as e:
                logger.debug("Divisibility check failed for %r by %s: %s", added_value, divisor, e)
                continue

        # 3. If division not successful and value looks prime-like, try ask_unit modification
        if not divided_successfully and is_prime_like(added_value, kececi_type):
            direction = 1 if ask_counter == 0 else -1
            try:
                modified_value = safe_add(added_value, ask_unit, direction)
            except Exception as e:
                logger.debug("safe_add failed for %r with ask_unit=%r direction=%s: %s", added_value, ask_unit, direction, e)
                # try native operation if possible
                try:
                    modified_value = added_value + (ask_unit * direction)
                except Exception as e2:
                    logger.debug("native ask alteration also failed: %s", e2)
                    modified_value = None

            # Toggle ask counter only if modification occurred
            if modified_value is not None:
                ask_counter = 1 - ask_counter
                # try divisions again with modified value
                for divisor in (primary_divisor, alternative_divisor):
                    try:
                        if _is_divisible(modified_value, divisor, kececi_type):
                            try:
                                next_q = _safe_divide(modified_value, divisor, use_integer_division)
                                last_divisor_used = divisor
                                divided_successfully = True
                            except Exception:
                                try:
                                    next_q = modified_value / divisor
                                    last_divisor_used = divisor
                                    divided_successfully = True
                                except Exception:
                                    logger.debug("Division on modified_value failed for %r by %s", modified_value, divisor)
                            break
                    except Exception as e:
                        logger.debug("Divisibility check on modified_value failed: %s", e)
                        continue

        # 4. Logging and book-keeping
        #full_log.append(added_value)
        if include_intermediate_steps:
            full_log.append(added_value)
        #if modified_value is not None:
        if modified_value is not None and include_intermediate_steps:
            full_log.append(modified_value)
        #if not full_log or next_q != full_log[-1]:
        # Ve next_q eklenmeden önce veya sonra:
        if include_intermediate_steps:
            # next_q muhtemelen full_log son elemanı değilse yine ekle
            if not full_log or next_q != full_log[-1]:
                full_log.append(next_q)

        clean_trajectory.append(next_q)
        current_value = next_q

    # Sonuç formatlarken:
    #formatted_sequence = [format_fraction(x) if isinstance(x, Fraction) else x for x in (full_log if include_intermediate_steps else clean_trajectory)]
    formatted_sequence = [format_fraction(x) if isinstance(x, Fraction) else x
                          for x in (full_log if include_intermediate_steps else clean_trajectory)]
    return formatted_sequence

def get_with_params(
    kececi_type_choice: int,
    iterations: int,
    start_value_raw: str,
    add_value_raw: str,
    include_intermediate_steps: bool = False
) -> List[Any]:
    """
    Common entry point: validates inputs early, logs info instead of printing.
    """
    logger.info("Generating Keçeci Sequence: Type %s, Steps %s", kececi_type_choice, iterations)
    logger.debug("Start: %r, Addition: %r, Include intermediate: %s", start_value_raw, add_value_raw, include_intermediate_steps)

    # Basic input sanitation
    if start_value_raw is None:
        start_value_raw = "0"
    if add_value_raw is None:
        # choose a conservative default for increment
        add_value_raw = "1"

    try:
        generated_sequence = unified_generator(
            kececi_type=kececi_type_choice,
            start_input_raw=start_value_raw,
            add_input_raw=add_value_raw,
            iterations=iterations,
            include_intermediate_steps=include_intermediate_steps
        )

        if not generated_sequence:
            logger.warning("Sequence generation failed or returned empty for type %s with start=%r add=%r", kececi_type_choice, start_value_raw, add_value_raw)
            return []

        logger.info("Generated %d numbers for type %s", len(generated_sequence), kececi_type_choice)
        # preview
        preview_start = [str(x) for x in generated_sequence[:5]]
        preview_end = [str(x) for x in generated_sequence[-5:]] if len(generated_sequence) > 5 else []

        logger.debug("First 5: %s", preview_start)
        if preview_end:
            logger.debug("Last 5: %s", preview_end)

        # Keçeci Prime Number check
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            logger.info("Keçeci Prime Number (KPN) found: %s", kpn)
        else:
            logger.info("No Keçeci Prime Number found in the sequence.")

        return generated_sequence

    except Exception as e:
        logger.exception("ERROR during sequence generation: %s", e)
        return []

def get_interactive(auto_values: Optional[Dict[str, str]] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Interactively (or programmatically via auto_values) gets parameters to generate a Keçeci sequence.

    If auto_values is provided, keys can include:
        'type_choice' (int or str), 'start_val' (str), 'add_val' (str),
        'steps' (int or str), 'show_details' ('y'/'n').

    If auto_values is None, function behaves interactively and prints a type menu.
    """
    # Local prompt function: use auto_values if present otherwise input()
    def _ask(key: str, prompt: str, default: str) -> str:
        if auto_values and key in auto_values:
            return str(auto_values[key])
        try:
            return input(prompt).strip() or default
        except Exception:
            # In non-interactive contexts where input is not available, use default
            logger.debug("input() failed for prompt %r — using default %r", prompt, default)
            return default

    interactive_mode = (auto_values is None)
    logger.info("Keçeci Numbers Interactive Generator (interactive=%s)", interactive_mode)

    # If interactive, present the full menu of type options so users see 1-22 choices
    if interactive_mode:
        menu_lines = [
            "  1: Positive Real    2: Negative Real      3: Complex",
            "  4: Float            5: Rational           6: Quaternion",
            "  7: Neutrosophic     8: Neutro-Complex     9: Hyperreal",
            " 10: Bicomplex       11: Neutro-Bicomplex  12: Octonion",
            " 13: Sedenion        14: Clifford          15: Dual",
            " 16: Split-Complex   17: Pathion           18: Chingon",
            " 19: Routon          20: Voudon            21: SuperReal",
            " 22: Ternary"
        ]
        logger.info("Available Keçeci Number Types:")
        for line in menu_lines:
            logger.info(line)

    # Defaults
    DEFAULT_TYPE = 3
    DEFAULT_STEPS = 40
    DEFAULT_SHOW_DETAILS = "no"

    default_start_values = {
        1: "2.5", 2: "-5.0", 3: "1+1j", 4: "3.14", 5: "3.5",
        6: "1.0,0.0,0.0,0.0", 7: "0.6,0.2,0.1", 8: "1+1j", 9: "0.0,0.001",
        10: "1.0,0.5,0.3,0.2", 11: "1.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0",
        12: "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0", 13: "1.0" + ",0.0"*15,
        14: "1.0+2.0e1+3.0e12", 15: "1.0,0.1", 16: "1.0,0.5",
        17: "1.0" + ",0.0"*31, 18: "1.0" + ",0.0"*63, 19: "1.0" + ",0.0"*127,
        20: "1.0" + ",0.0"*255, 21: "3.0,0.5", 22: "12",
    }

    default_add_values = {
        1: "0.5", 2: "-0.5", 3: "0.1+0.1j", 4: "0.1", 5: "0.1",
        6: "0.1,0.0,0.0,0.0", 7: "0.1,0.0,0.0", 8: "0.1+0.1j", 9: "0.0,0.001",
        10: "0.1,0.0,0.0,0.0", 11: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        12: "0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0", 13: "0.1" + ",0.0"*15,
        14: "0.1+0.2e1", 15: "0.1,0.0", 16: "0.1,0.0",
        17: "1.0" + ",0.0"*31, 18: "1.0" + ",0.0"*63, 19: "1.0" + ",0.0"*127,
        20: "1.0" + ",0.0"*255, 21: "1.5,2.7", 22: "1",
    }

    # Ask for inputs (uses _ask which respects auto_values when provided)
    type_input_raw = _ask('type_choice', f"Select Keçeci Number Type (1-22) [default: {DEFAULT_TYPE}]: ", str(DEFAULT_TYPE))
    try:
        type_choice = int(type_input_raw)
        if not (1 <= type_choice <= 22):
            logger.warning("Invalid type_choice %r, using default %s", type_choice, DEFAULT_TYPE)
            type_choice = DEFAULT_TYPE
    except Exception:
        logger.warning("Could not parse type_choice %r, using default %s", type_input_raw, DEFAULT_TYPE)
        type_choice = DEFAULT_TYPE

    start_prompt = _ask('start_val', f"Enter start value [default: {default_start_values[type_choice]}]: ", default_start_values[type_choice])
    add_prompt = _ask('add_val', f"Enter increment value [default: {default_add_values[type_choice]}]: ", default_add_values[type_choice])
    steps_raw = _ask('steps', f"Enter number of Keçeci steps [default: {DEFAULT_STEPS}]: ", str(DEFAULT_STEPS))
    try:
        num_kececi_steps = int(steps_raw)
        if num_kececi_steps <= 0:
            logger.warning("Non-positive steps %r, using default %d", num_kececi_steps, DEFAULT_STEPS)
            num_kececi_steps = DEFAULT_STEPS
    except Exception:
        logger.warning("Could not parse steps %r, using default %d", steps_raw, DEFAULT_STEPS)
        num_kececi_steps = DEFAULT_STEPS

    show_detail_raw = _ask('show_details', f"Include intermediate steps? (y/n) [default: {DEFAULT_SHOW_DETAILS}]: ", DEFAULT_SHOW_DETAILS)
    show_details = str(show_detail_raw).strip().lower() in ['y', 'yes']

    sequence = get_with_params(
        kececi_type_choice=type_choice,
        iterations=num_kececi_steps,
        start_value_raw=start_prompt,
        add_value_raw=add_prompt,
        include_intermediate_steps=show_details
    )

    params = {
        "type_choice": type_choice,
        "start_val": start_prompt,
        "add_val": add_prompt,
        "steps": num_kececi_steps,
        "detailed_view": show_details
    }
    logger.info("Using parameters: Type=%s, Start=%r, Add=%r, Steps=%s, Details=%s", type_choice, start_prompt, add_prompt, num_kececi_steps, show_details)
    return sequence, params

# ==============================================================================
# --- ANALYSIS AND PLOTTING ---
# ==============================================================================

def find_period(sequence: List[Any], min_repeats: int = 3) -> Optional[List[Any]]:
    """
    Checks if the end of a sequence has a repeating cycle (period).

    Args:
        sequence: The list of numbers to check.
        min_repeats: How many times the cycle must repeat to be considered stable.

    Returns:
        The repeating cycle as a list if found, otherwise None.
    """
    if len(sequence) < 4:  # Çok kısa dizilerde periyot aramak anlamsız
        return None

    # Olası periyot uzunluklarını dizinin yarısına kadar kontrol et
    for p_len in range(1, len(sequence) // min_repeats):
        # Dizinin sonundan potansiyel döngüyü al
        candidate_cycle = sequence[-p_len:]
        
        # Döngünün en az `min_repeats` defa tekrar edip etmediğini kontrol et
        is_periodic = True
        for i in range(1, min_repeats):
            start_index = -(i + 1) * p_len
            end_index = -i * p_len
            
            # Dizinin o bölümünü al
            previous_block = sequence[start_index:end_index]

            # Eğer bloklar uyuşmuyorsa, bu periyot değildir
            if candidate_cycle != previous_block:
                is_periodic = False
                break
        
        # Eğer döngü tüm kontrollerden geçtiyse, periyodu bulduk demektir
        if is_periodic:
            return candidate_cycle

    # Hiçbir periyot bulunamadı
    return None

def is_quaternion_like(obj):
    if isinstance(obj, quaternion):
        return True
    if hasattr(obj, 'components'):
        comp = np.array(obj.components)
        return comp.size == 4
    if all(hasattr(obj, attr) for attr in ['w', 'x', 'y', 'z']):
        return True
    if hasattr(obj, 'scalar') and hasattr(obj, 'vector') and isinstance(obj.vector, (list, np.ndarray)) and len(obj.vector) == 3:
        return True
    return False

def is_neutrosophic_like(obj):
    """NeutrosophicNumber gibi görünen objeleri tanır (t,i,f veya a,b vs.)"""
    return (hasattr(obj, 't') and hasattr(obj, 'i') and hasattr(obj, 'f')) or \
           (hasattr(obj, 'a') and hasattr(obj, 'b')) or \
           (hasattr(obj, 'value') and hasattr(obj, 'indeterminacy')) or \
           (hasattr(obj, 'determinate') and hasattr(obj, 'indeterminate'))

def _pca_var_sum(pca) -> float:
    """
    Safely return sum of PCA explained variance ratio.
    Returns 0.0 if pca has no explained_variance_ratio_ or values are NaN/invalid.
    """
    try:
        arr = getattr(pca, "explained_variance_ratio_", None)
        if arr is None:
            return 0.0
        s = float(np.nansum(arr))
        if not np.isfinite(s):
            return 0.0
        return s
    except Exception:
        return 0.0

# Yardımcı fonksiyon: Bileşen dağılımı grafiği
def _plot_component_distribution(ax, elem, all_keys, seq_length=1):
    """Bileşen dağılımını gösterir"""
    if seq_length == 1:
        # Tek veri noktası için bileşen değerleri
        components = []
        values = []
        
        for key in all_keys:
            if key == '':
                components.append('Scalar')
            else:
                components.append(f'e{key}')
            values.append(elem.basis.get(key, 0.0))
        
        bars = ax.bar(components, values, alpha=0.7, color='tab:blue')
        ax.set_title("Component Values")
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
    else:
        # Çoklu veri ama PCA yapılamıyor
        ax.text(0.5, 0.5, f"Need ≥2 data points and ≥2 features\n(Current: {seq_length} points, {len(all_keys)} features)", 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title("Insufficient for PCA")

def plot_octonion_3d(octonion_sequence, title="3D Octonion Trajectory"):
    """
    Plots the trajectory of octonion numbers in 3D space using the first three imaginary components (x, y, z).
    Args:
        octonion_sequence (list): List of OctonionNumber objects.
        title (str): Title of the plot.
    """
    if not octonion_sequence:
        print("Empty sequence. Nothing to plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Octonion bileşenlerini ayıkla (w: gerçek, x/y/z: ilk üç sanal bileşen)
    x = [o.x for o in octonion_sequence]
    y = [o.y for o in octonion_sequence]
    z = [o.z for o in octonion_sequence]

    # 3D uzayda çiz
    ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(x[0], y[0], z[0], c='g', s=100, label='Start', depthshade=True)
    ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='End', depthshade=True)

    # Eksen etiketleri ve başlık
    ax.set_xlabel('X (i)')
    ax.set_ylabel('Y (j)')
    ax.set_zlabel('Z (k)')
    ax.set_title(title)

    # Legend ve grid
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Otomatik Periyot Tespiti ve Keçeci Asal Analizi
def analyze_kececi_sequence(sequence, kececi_type):
    """
    Analyzes a Keçeci sequence for periodicity and Keçeci Prime Numbers (KPN).
    Args:
        sequence (list): List of Keçeci numbers.
        kececi_type (int): Type of Keçeci number (e.g., TYPE_OCTONION).
    Returns:
        dict: Analysis results including periodicity and KPNs.
    """
    results = {
        "periodicity": None,
        "kececi_primes": [],
        "prime_indices": []
    }

    # Periyot tespiti
    for window in range(2, len(sequence) // 2):
        is_periodic = True
        for i in range(len(sequence) - window):
            if sequence[i] != sequence[i + window]:
                is_periodic = False
                break
        if is_periodic:
            results["periodicity"] = window
            break

    # Keçeci Asal sayıları tespit et
    for idx, num in enumerate(sequence):
        if is_prime_like(num, kececi_type):
            integer_rep = _get_integer_representation(num)
            if integer_rep is not None and sympy.isprime(integer_rep):
                results["kececi_primes"].append(integer_rep)
                results["prime_indices"].append(idx)

    return results

# Makine Öğrenimi Entegrasyonu: PCA ve Kümelenme Analizi
def apply_pca_clustering(sequence, n_components=2):
    """
    Applies PCA and clustering to a Keçeci sequence for dimensionality reduction and pattern discovery.
    Args:
        sequence (list): List of Keçeci numbers.
        n_components (int): Number of PCA components.
    Returns:
        tuple: (pca_result, clusters) - PCA-transformed data and cluster labels.
    """
    # Sayıları sayısal vektörlere dönüştür
    vectors = []
    for num in sequence:
        if isinstance(num, OctonionNumber):
            vectors.append(num.coeffs)
        elif isinstance(num, Fraction):
            vectors.append([float(num)])
        else:
            vectors.append([float(num)])

    # PCA uygula
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(vectors)

    # Kümelenme (K-Means)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    return pca_result, clusters

# Etkileşimli Görselleştirme (Plotly DASH)
def generate_interactive_plot(sequence, kececi_type):
    """
    Generates an interactive 3D plot using Plotly for Keçeci sequences.
    Args:
        sequence (list): List of Keçeci numbers.
        kececi_type (int): Type of Keçeci number.
    """
    import plotly.graph_objects as go

    if kececi_type == TYPE_OCTONION:
        x = [num.x for num in sequence]
        y = [num.y for num in sequence]
        z = [num.z for num in sequence]
    elif kececi_type == TYPE_COMPLEX:
        x = [num.real for num in sequence]
        y = [num.imag for num in sequence]
        z = [0] * len(sequence)
    else:
        x = range(len(sequence))
        y = [float(num) for num in sequence]
        z = [0] * len(sequence)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=5, color=z, colorscale='Viridis'),
        line=dict(width=2)
    )])

    fig.update_layout(
        title=f"Interactive 3D Plot: Keçeci Type {kececi_type}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()

# Keçeci Varsayımı Test Aracı
def test_kececi_conjecture(sequence: List[Any], add_value: Any, kececi_type: Optional[int] = None, max_steps: int = 1000) -> bool:
    """
    Tests the Keçeci Conjecture for a given starting `sequence`.
    - sequence: initial list-like of Keçeci numbers (will be copied).
    - add_value: typed increment (must be of compatible type with elements).
    - kececi_type: optional type constant (used by is_prime_like); if None, fallback to is_prime.
    - max_steps: maximum additional steps to try.
    Returns True if a Keçeci-prime is reached within max_steps, otherwise False.
    """
    traj = list(sequence)
    if not traj:
        raise ValueError("sequence must contain at least one element")

    for step in range(max_steps):
        last = traj[-1]
        # Check prime-like condition
        try:
            if kececi_type is not None:
                if is_prime_like(last, kececi_type):
                    return True
            else:
                # fallback: try is_prime on integer rep
                if is_prime(last):
                    return True
        except Exception:
            # If prime test fails, continue attempts
            pass

        # Compute next element: prefer safe_add, else try native addition
        next_val = None
        try:
            next_val = safe_add(last, add_value, +1)
        except Exception:
            try:
                next_val = last + add_value
            except Exception:
                # cannot add -> abort
                return False

        traj.append(next_val)

    return False

def format_fraction(value):
    """Fraction nesnelerini güvenli bir şekilde formatlar."""
    if isinstance(value, Fraction):
        return float(value)  # veya str(value)
    return value

def plot_numbers(sequence: List[Any], title: str = "Keçeci Number Sequence Analysis"):
    """
    Tüm 22 Keçeci Sayı türü için detaylı görselleştirme sağlar.
    """

    if not sequence:
        print("Sequence is empty. Nothing to plot.")
        return

    # Ensure numpy is available for plotting functions
    try:
        import numpy as np
    except ImportError:
        print("Numpy not installed. Cannot plot effectively.")
        return

    try:
        from sklearn.decomposition import PCA
        use_pca = True
    except ImportError:
        use_pca = False
        print("scikit-learn kurulu değil. PCA olmadan çizim yapılıyor...")


    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # `sequence` is the iterable you want to visualise
    first_elem = sequence[0]

    # --- 1. Fraction (Rational)
    if isinstance(first_elem, Fraction):
        # Tüm elemanları `float` olarak dönüştür
        float_vals = [float(x) for x in sequence]
        # float_vals = [float(x) if isinstance(x, (int, float, Fraction)) else float(x.value) for x in sequence]
        # Pay ve paydaları ayrı ayrı al
        numerators = [x.numerator for x in sequence]
        denominators = [x.denominator for x in sequence]

        # GridSpec ile 4 alt grafik oluştur
        gs = GridSpec(2, 2, figure=fig)

        # 1. Grafik: Float değerleri
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(float_vals, 'o-', color='tab:blue')
        ax1.set_title("Fraction as Float")
        ax1.set_ylabel("Value")

        # 2. Grafik: Pay ve payda değerleri
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(numerators, 's-', label='Numerator', color='tab:orange')
        ax2.plot(denominators, '^-', label='Denominator', color='tab:green')
        ax2.set_title("Numerator & Denominator")
        ax2.legend()

        # 3. Grafik: Pay/Payda oranı
        ax3 = fig.add_subplot(gs[1, 0])
        ratios = [n / d for n, d in zip(numerators, denominators)]
        ax3.plot(ratios, 'o-', color='tab:purple')
        ax3.set_title("Numerator/Denominator Ratio")
        ax3.set_ylabel("n/d")

        # 4. Grafik: Pay vs Payda dağılımı
        ax4 = fig.add_subplot(gs[1, 1])
        sc = ax4.scatter(numerators, denominators, c=range(len(sequence)), cmap='plasma', s=30)
        ax4.set_title("Numerator vs Denominator Trajectory")
        ax4.set_xlabel("Numerator")
        ax4.set_ylabel("Denominator")
        plt.colorbar(sc, ax=ax4, label="Step")

    # --- 2. int, float (Positive/Negative Real, Float)
    elif isinstance(first_elem, (int, float)):
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([float(x) for x in sequence], 'o-', color='tab:blue', markersize=5)
        ax.set_title("Real Number Sequence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    # --- 3. Complex
    elif isinstance(first_elem, complex):
        real_parts = [z.real for z in sequence]
        imag_parts = [z.imag for z in sequence]
        magnitudes = [abs(z) for z in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_parts, 'o-', color='tab:blue')
        ax1.set_title("Real Part")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(imag_parts, 'o-', color='tab:red')
        ax2.set_title("Imaginary Part")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:purple')
        ax3.set_title("Magnitude |z|")

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(real_parts, imag_parts, '.-', alpha=0.7)
        ax4.scatter(real_parts[0], imag_parts[0], c='g', s=100, label='Start')
        ax4.scatter(real_parts[-1], imag_parts[-1], c='r', s=100, label='End')
        ax4.set_title("Complex Plane")
        ax4.set_xlabel("Re(z)")
        ax4.set_ylabel("Im(z)")
        ax4.legend()
        ax4.axis('equal')
        ax4.grid(True, alpha=0.3)

    # --- 4. quaternion
    # Check for numpy-quaternion's quaternion type, or a custom one with 'components' or 'w,x,y,z': çıkarıldı:  and len(getattr(first_elem, 'components', [])) == 4) or \
    elif isinstance(first_elem, quaternion) or (hasattr(first_elem, 'components') == 4) or \
         (hasattr(first_elem, 'w') and hasattr(first_elem, 'x') and hasattr(first_elem, 'y') and hasattr(first_elem, 'z')):

        try:
            comp = np.array([
                (q.w, q.x, q.y, q.z) if hasattr(q, 'w') else q.components
                for q in sequence
            ])

            w, x, y, z = comp.T
            magnitudes = np.linalg.norm(comp, axis=1)
            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(2, 2, figure=fig)

            # Component time‑series
            ax1 = fig.add_subplot(gs[0, 0])
            labels = ['w', 'x', 'y', 'z']
            for i, label in enumerate(labels):
                ax1.plot(comp[:, i], label=label, alpha=0.8)
            ax1.set_title("Quaternion Components")
            ax1.legend()

            # Magnitude plot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(magnitudes, 'o-', color='tab:purple')
            ax2.set_title("Magnitude |q|")

            # 3‑D trajectory of the vector part (x, y, z)
            ax3 = fig.add_subplot(gs[1, :], projection='3d')
            ax3.plot(x, y, z, alpha=0.7)
            ax3.scatter(x[0], y[0], z[0], c='g', s=100, label='Start')
            ax3.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='End')
            ax3.set_title("3D Trajectory (x,y,z)")
            ax3.set_xlabel("x");
            ax3.set_ylabel("y");
            ax3.set_zlabel("z")
            ax3.legend()

        except Exception as e:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='red')


    # --- 5. OctonionNumber
    elif isinstance(first_elem, OctonionNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(4):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.7)
        ax1.set_title("e0-e3 Components")
        ax1.legend(ncol=2)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(4, 8):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.7)
        ax2.set_title("e4-e7 Components")
        ax2.legend(ncol=2)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:purple')
        ax3.set_title("Magnitude |o|")

        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        ax4.plot(coeffs[:, 1], coeffs[:, 2], coeffs[:, 3], alpha=0.7)
        ax4.set_title("3D (e1,e2,e3)")
        ax4.set_xlabel("e1");
        ax4.set_ylabel("e2");
        ax4.set_zlabel("e3")

    # --- 6. SedenionNumber
    elif isinstance(first_elem, SedenionNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(8):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax1.set_title("Sedenion e0-e7")
        ax1.legend(ncol=2, fontsize=6)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(8, 16):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax2.set_title("e8-e15")
        ax2.legend(ncol=2, fontsize=6)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:purple')
        ax3.set_title("Magnitude |s|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 7. CliffordNumber
    elif isinstance(first_elem, CliffordNumber):
        all_keys = sorted(first_elem.basis.keys(), key=lambda x: (len(x), x))
        values = {k: [elem.basis.get(k, 0.0) for elem in sequence] for k in all_keys}
        scalar = values.get('', [0]*len(sequence))
        vector_keys = [k for k in all_keys if len(k) == 1]

        # GERÇEK özellik sayısını hesapla (sıfır olmayan bileşenler)
        non_zero_features = 0
        for key in all_keys:
            if any(abs(elem.basis.get(key, 0.0)) > 1e-10 for elem in sequence):
                non_zero_features += 1

        # Her zaman 2x2 grid kullan
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        # 1. Grafik: Skaler ve Vektör Bileşenleri
        ax1.plot(scalar, 'o-', label='Scalar', color='black', linewidth=2)

        # Sadece sıfır olmayan vektör bileşenlerini göster
        visible_vectors = 0
        for k in vector_keys:
            if any(abs(v) > 1e-10 for v in values[k]):
                ax1.plot(values[k], 'o-', label=f'Vec {k}', alpha=0.7, linewidth=1.5)
                visible_vectors += 1
            if visible_vectors >= 3:
                break

        ax1.set_title("Scalar & Vector Components Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Grafik: Bivector Magnitude
        bivector_mags = [sum(v**2 for k, v in elem.basis.items() if len(k) == 2)**0.5 for elem in sequence]
        ax2.plot(bivector_mags, 'o-', color='tab:green', linewidth=2, label='Bivector Magnitude')
        ax2.set_title("Bivector Magnitude Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Grafik: PCA - ARTIK PCA GÖSTERİYORUZ
        if use_pca and len(sequence) >= 2 and non_zero_features >= 2:
            try:
                # Tüm bileşenleri içeren matris oluştur
                matrix_data = []
                for elem in sequence:
                    row = []
                    for key in all_keys:
                        row.append(elem.basis.get(key, 0.0))
                    matrix_data.append(row)

                matrix = np.array(matrix_data)

                # PCA uygula
                pca = PCA(n_components=min(2, matrix.shape[1]))
                proj = pca.fit_transform(matrix)

                sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)),
                               cmap='plasma', s=50, alpha=0.8)
                ax3.set_title(f"PCA Projection ({non_zero_features} features)\nVariance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

                cbar = plt.colorbar(sc, ax=ax3)
                cbar.set_label("Time Step")

                ax3.plot(proj[:, 0], proj[:, 1], 'gray', linestyle='--', alpha=0.5)
                ax3.grid(True, alpha=0.3)

            except Exception as e:
                ax3.text(0.5, 0.5, f"PCA Error: {str(e)[:30]}",
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            # PCA yapılamazsa bilgi göster
            ax3.text(0.5, 0.5, f"Need ≥2 data points and ≥2 features\n(Current: {len(sequence)} points, {non_zero_features} features)",
                   ha='center', va='center', transform=ax3.transAxes)
            if not use_pca:
                ax3.text(0.5, 0.65, "Install sklearn for PCA",
                        ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Insufficient for PCA")


    # --- 8. DualNumber
    elif isinstance(first_elem, DualNumber):
        real_vals = [x.real for x in sequence]
        dual_vals = [x.dual for x in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_vals, 'o-', color='tab:blue')
        ax1.set_title("Real Part")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(dual_vals, 'o-', color='tab:orange')
        ax2.set_title("Dual Part (ε)")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(real_vals, dual_vals, '.-')
        ax3.set_title("Real vs Dual")
        ax3.set_xlabel("Real")
        ax3.set_ylabel("Dual")

        ax4 = fig.add_subplot(gs[1, 1])
        ratios = [d/r if r != 0 else 0 for r, d in zip(real_vals, dual_vals)]
        ax4.plot(ratios, 'o-', color='tab:purple')
        ax4.set_title("Dual/Real Ratio")

    # --- 9. SplitcomplexNumber
    elif isinstance(first_elem, SplitcomplexNumber):
        real_vals = [x.real for x in sequence]
        split_vals = [x.split for x in sequence]
        u_vals = [r + s for r, s in zip(real_vals, split_vals)]
        v_vals = [r - s for r, s in zip(real_vals, split_vals)]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_vals, 'o-', color='tab:green')
        ax1.set_title("Real Part")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(split_vals, 'o-', color='tab:brown')
        ax2.set_title("Split Part (j)")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(real_vals, split_vals, '.-')
        ax3.set_title("Trajectory (Real vs Split)")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(u_vals, label='u = r+j')
        ax4.plot(v_vals, label='v = r-j')
        ax4.set_title("Light-Cone Coordinates")
        ax4.legend()

    # --- 10. NeutrosophicNumber
    elif isinstance(first_elem, NeutrosophicNumber):
        # NeutrosophicNumber sınıfının arayüzünü biliyoruz, hasattr gerekmez
        # Sınıfın public attribute'larına doğrudan erişim
        try:
            t_vals = [x.t for x in sequence]
            i_vals = [x.i for x in sequence]
            f_vals = [x.f for x in sequence]
        except AttributeError:
            # Eğer attribute yoksa, alternatif arayüzleri deneyebiliriz
            # Veya hata fırlatabiliriz
            try:
                t_vals = [x.a for x in sequence]
                i_vals = [x.b for x in sequence]
                f_vals = [0] * len(sequence)  # f yoksa sıfır
            except AttributeError:
                try:
                    t_vals = [x.value for x in sequence]
                    i_vals = [x.indeterminacy for x in sequence]
                    f_vals = [0] * len(sequence)
                except AttributeError:
                    # Hiçbiri yoksa boş liste
                    t_vals = i_vals = f_vals = []

        gs = GridSpec(2, 2, figure=fig)

        # 1. t, i, f zaman içinde
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_vals, 'o-', label='Truth (t)', color='tab:blue')
        ax1.plot(i_vals, 's-', label='Indeterminacy (i)', color='tab:orange')
        ax1.plot(f_vals, '^-', label='Falsity (f)', color='tab:red')
        ax1.set_title("Neutrosophic Components")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. t vs i
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(t_vals, i_vals, c=range(len(t_vals)), cmap='viridis', s=30)
        ax2.set_title("t vs i Trajectory")
        ax2.set_xlabel("Truth (t)")
        ax2.set_ylabel("Indeterminacy (i)")
        plt.colorbar(ax2.collections[0], ax=ax2, label="Step")

        # 3. t vs f
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(t_vals, f_vals, c=range(len(t_vals)), cmap='plasma', s=30)
        ax3.set_title("t vs f Trajectory")
        ax3.set_xlabel("Truth (t)")
        ax3.set_ylabel("Falsity (f)")
        plt.colorbar(ax3.collections[0], ax=ax3, label="Step")

        # 4. Magnitude (t² + i² + f²)
        magnitudes = [np.sqrt(t**2 + i**2 + f**2) for t, i, f in zip(t_vals, i_vals, f_vals)]
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(magnitudes, 'o-', color='tab:purple')
        ax4.set_title("Magnitude √(t²+i²+f²)")
        ax4.set_ylabel("|n|")

    # --- 11. NeutrosophicComplexNumber
    elif isinstance(first_elem, NeutrosophicComplexNumber):
        # Sınıfın arayüzünü biliyoruz
        real_parts = [x.real for x in sequence]
        imag_parts = [x.imag for x in sequence]
        indeter_parts = [x.indeterminacy for x in sequence]
        magnitudes_z = [abs(complex(x.real, x.imag)) for x in sequence]

        gs = GridSpec(2, 2, figure=fig)

        # 1. Complex Plane
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(real_parts, imag_parts, '.-', alpha=0.7)
        ax1.scatter(real_parts[0], imag_parts[0], c='g', s=100, label='Start')
        ax1.scatter(real_parts[-1], imag_parts[-1], c='r', s=100, label='End')
        ax1.set_title("Complex Plane")
        ax1.set_xlabel("Re(z)")
        ax1.set_ylabel("Im(z)")
        ax1.legend()
        ax1.axis('equal')

        # 2. Indeterminacy over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(indeter_parts, 'o-', color='tab:purple')
        ax2.set_title("Indeterminacy Level")
        ax2.set_ylabel("I")

        # 3. |z| vs Indeterminacy
        ax3 = fig.add_subplot(gs[1, 0])
        sc = ax3.scatter(magnitudes_z, indeter_parts, c=range(len(magnitudes_z)), cmap='viridis', s=30)
        ax3.set_title("Magnitude vs Indeterminacy")
        ax3.set_xlabel("|z|")
        ax3.set_ylabel("I")
        plt.colorbar(sc, ax=ax3, label="Step")

        # 4. Real vs Imag colored by I
        ax4 = fig.add_subplot(gs[1, 1])
        sc2 = ax4.scatter(real_parts, imag_parts, c=indeter_parts, cmap='plasma', s=40)
        ax4.set_title("Real vs Imag (colored by I)")
        ax4.set_xlabel("Re(z)")
        ax4.set_ylabel("Im(z)")
        plt.colorbar(sc2, ax=ax4, label="Indeterminacy")

    # --- 12. HyperrealNumber
    elif isinstance(first_elem, HyperrealNumber):
        # Sınıfın arayüzünü biliyoruz
        seq_len = min(len(first_elem.sequence), 5)  # İlk 5 bileşen
        data = np.array([x.sequence[:seq_len] for x in sequence])
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(seq_len):
            ax1.plot(data[:, i], label=f'ε^{i}', alpha=0.8)
        ax1.set_title("Hyperreal Components")
        ax1.legend(ncol=2)

        ax2 = fig.add_subplot(gs[0, 1])
        magnitudes = np.linalg.norm(data, axis=1)
        ax2.plot(magnitudes, 'o-', color='tab:purple')
        ax2.set_title("Magnitude")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(data[:, 0], 'o-', label='Standard Part')
        ax3.set_title("Standard Part (ε⁰)")
        ax3.legend()

        ax4 = fig.add_subplot(gs[1, 1])
        sc = ax4.scatter(data[:, 0], data[:, 1], c=range(len(data)), cmap='viridis')
        ax4.set_title("Standard vs Infinitesimal")
        ax4.set_xlabel("Standard")
        ax4.set_ylabel("ε¹")
        plt.colorbar(sc, ax=ax4, label="Step")

    # --- 13. BicomplexNumber
    elif isinstance(first_elem, BicomplexNumber):
        # Sınıfın arayüzünü biliyoruz
        z1_real = [x.z1.real for x in sequence]
        z1_imag = [x.z1.imag for x in sequence]
        z2_real = [x.z2.real for x in sequence]
        z2_imag = [x.z2.imag for x in sequence]
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(z1_real, label='Re(z1)')
        ax1.plot(z1_imag, label='Im(z1)')
        ax1.set_title("Bicomplex z1")
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(z2_real, label='Re(z2)')
        ax2.plot(z2_imag, label='Im(z2)')
        ax2.set_title("Bicomplex z2")
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(z1_real, z1_imag, '.-')
        ax3.set_title("z1 Trajectory")
        ax3.set_xlabel("Re(z1)")
        ax3.set_ylabel("Im(z1)")

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(z2_real, z2_imag, '.-')
        ax4.set_title("z2 Trajectory")
        ax4.set_xlabel("Re(z2)")
        ax4.set_ylabel("Im(z2)")

    # --- 14. NeutrosophicBicomplexNumber ---
    elif isinstance(first_elem, NeutrosophicBicomplexNumber):
        # Sınıfın - a, b, c, d, e, f, g, h attribute'ları var
        try:
            # Doğru attribute isimlerini kullanıyoruz
            comps = np.array([
                [float(getattr(x, attr))
                 for attr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
                for x in sequence
            ])
            magnitudes = np.linalg.norm(comps, axis=1)
            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            for i, label in enumerate(['a', 'b', 'c', 'd']):
                ax1.plot(comps[:, i], label=label, alpha=0.7)
            ax1.set_title("First 4 Components")
            ax1.legend()

            ax2 = fig.add_subplot(gs[0, 1])
            for i, label in enumerate(['e', 'f', 'g', 'h']):
                ax2.plot(comps[:, i + 4], label=label, alpha=0.7)
            ax2.set_title("Last 4 Components")
            ax2.legend()

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(magnitudes, 'o-', color='tab:purple')
            ax3.set_title("Magnitude")

            ax4 = fig.add_subplot(gs[1, 1])
            sc = ax4.scatter(comps[:, 0], comps[:, 1], c=range(len(comps)), cmap='plasma')
            ax4.set_title("a vs b Trajectory")
            ax4.set_xlabel("a")
            ax4.set_ylabel("b")
            plt.colorbar(sc, ax=ax4, label="Step")

        except Exception as e:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Plot error: {e}", ha='center', va='center', color='red')
            ax.set_xticks([])
            ax.set_yticks([])

    # --- 15. Pathion
    elif isinstance(first_elem, PathionNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(8):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax1.set_title("PathionNumber e0-e7")
        ax1.legend(ncol=2, fontsize=6)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(8, 16):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.8)
        ax2.set_title("e8-e15")
        ax2.legend(ncol=2, fontsize=6)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:red')
        ax3.set_title("Magnitude |p|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 16. Chingon
    elif isinstance(first_elem, ChingonNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(16):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.5)
        ax1.set_title("ChingonNumber e0-e15")
        ax1.legend(ncol=4, fontsize=4)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(16, 32):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.5)
        ax2.set_title("e16-e31")
        ax2.legend(ncol=4, fontsize=4)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:green')
        ax3.set_title("Magnitude |c|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


    # --- 17. Routon
    elif isinstance(first_elem, RoutonNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(32):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.3)
        ax1.set_title("RoutonNumber e0-e31")
        ax1.legend(ncol=4, fontsize=3)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(32, 64):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.3)
        ax2.set_title("e32-e63")
        ax2.legend(ncol=4, fontsize=3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:blue')
        ax3.set_title("Magnitude |r|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 18. Voudon
    elif isinstance(first_elem, VoudonNumber):
        coeffs = np.array([x.coeffs for x in sequence])
        magnitudes = np.linalg.norm(coeffs, axis=1)
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(64):
            ax1.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.2)
        ax1.set_title("VoudonNumber e0-e63")
        ax1.legend(ncol=4, fontsize=2)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(64, 128):
            ax2.plot(coeffs[:, i], label=f'e{i}', alpha=0.6, linewidth=0.2)
        ax2.set_title("e64-e127")
        ax2.legend(ncol=4, fontsize=2)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(magnitudes, 'o-', color='tab:orange')
        ax3.set_title("Magnitude |v|")

        if use_pca:
            try:
                pca = PCA(n_components=2)
                if len(sequence) > 2:
                    proj = pca.fit_transform(coeffs)
                    ax4 = fig.add_subplot(gs[1, 1])
                    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                    var_sum = _pca_var_sum(pca)
                    ax4.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                    plt.colorbar(sc, ax=ax4, label="Iteration")
            except Exception as e:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


    # --- 21. Super Real
    elif isinstance(first_elem, SuperrealNumber):
        # real ve split bileşenlerini ayır
        reals = np.array([x.real for x in sequence])
        splits = np.array([x.split for x in sequence])

        gs = GridSpec(2, 2, figure=fig)  # 2 satır, 2 sütun
        #gs = GridSpec(2, 1, figure=fig)

        # Real bileşenini çizdir
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(reals, 'o-', color='tab:blue', label='Real')
        ax1.set_title("Real Component")
        ax1.legend()

        # Split bileşenini çizdir
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(splits, 'o-', color='tab:red', label='Split')
        ax2.set_title("Split Component")
        ax2.legend()

        if use_pca and len(sequence) > 2:
            try:
                # PCA için veriyi hazırla
                data = np.column_stack((reals, splits))
                # Minimum gereksinimler: en az 3 örnek ve en az 2 değişken (burada 2 var)
                if data.shape[0] >= 3:
                    # finite ve non-NaN satırları seç
                    mask = np.all(np.isfinite(data), axis=1)
                    data_clean = data[mask]
                    if data_clean.shape[0] >= 3:
                        try:
                            pca = PCA(n_components=2)
                            proj = pca.fit_transform(data_clean)

                            # Güvenli varyans toplamı helper'ı kullanın
                            var_sum = _pca_var_sum(pca)

                            # 2D çizim (projisyon iki bileşenli olduğu için daha uygun)
                            ax3 = fig.add_subplot(gs[:, 1])
                            sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                            ax3.set_title(f"PCA Projection (Var: {var_sum:.3f})")
                            ax3.set_xlabel("PC1")
                            ax3.set_ylabel("PC2")
                            plt.colorbar(sc, ax=ax3, label="Iteration")

                            # Eğer 3D görmek isterseniz, alternatif:
                            # ax3 = fig.add_subplot(gs[:, 1], projection='3d')
                            # ax3.scatter(proj[:,0], proj[:,1], np.zeros_like(proj[:,0]), c=range(len(proj)), cmap='viridis', s=25)
                            # ax3.set_title(f"PCA Projection (Var: {var_sum:.3f})")

                        except Exception as e:
                            logger.exception("PCA failed for Superreal data: %s", e)
                            ax3 = fig.add_subplot(gs[:, 1])
                            ax3.text(0.5, 0.5, f"PCA Error: {str(e)[:80]}", ha='center', va='center', fontsize=10)
                            ax3.set_title("PCA Projection (Error)")
                    else:
                        ax3 = fig.add_subplot(gs[:, 1])
                        ax3.text(0.5, 0.5, "Insufficient finite data for PCA", ha='center', va='center')
                        ax3.set_title("PCA Projection (Insufficient data)")
                else:
                    ax3 = fig.add_subplot(gs[:, 1])
                    ax3.text(0.5, 0.5, "Need ≥3 samples for PCA", ha='center', va='center')
                    ax3.set_title("PCA Projection (Not enough samples)")
            except Exception as e:
                ax3 = fig.add_subplot(gs[:, 1])
                ax3.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax3 = fig.add_subplot(gs[:, 1])
            ax3.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)

    # --- 22. Ternary
    elif isinstance(first_elem, TernaryNumber):
        # Tüm TernaryNumber nesnelerinin digits uzunluğunu belirle
        max_length = max(len(x.digits) for x in sequence)

        # Her bir TernaryNumber nesnesinin digits listesini max_length uzunluğuna tamamla
        padded_digits = []
        for x in sequence:
            padded_digit = x.digits + [0] * (max_length - len(x.digits))
            padded_digits.append(padded_digit)

        # NumPy dizisine dönüştür
        digits = np.array(padded_digits)

        gs = GridSpec(2, 2, figure=fig)  # 2 satır, 2 sütun

        # Her bir rakamın dağılımını çizdir
        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(digits.shape[1]):
            ax1.plot(digits[:, i], 'o-', alpha=0.6, label=f'digit {i}')
        ax1.set_title("Ternary Digits")
        ax1.legend(ncol=4, fontsize=6)

        # Üçlü sayı sistemindeki değerleri ondalık sisteme çevirip çizdir
        decimal_values = np.array([x.to_decimal() for x in sequence])
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(decimal_values, 'o-', color='tab:green')
        ax2.set_title("Decimal Values")

        if use_pca and len(sequence) > 2:
            try:
                # PCA için veriyi hazırla
                pca = PCA(n_components=2)
                proj = pca.fit_transform(digits)

                # PCA projeksiyonunu çizdir
                ax3 = fig.add_subplot(gs[1, :])  # 2. satırın tamamını kullan
                sc = ax3.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=25)
                ax3.set_title(f"PCA Projection (Var: {sum(pca.explained_variance_ratio_):.3f})")
                plt.colorbar(sc, ax=ax3, label="Iteration")
            except Exception as e:
                ax3 = fig.add_subplot(gs[1, :])
                ax3.text(0.5, 0.5, f"PCA Error: {e}", ha='center', va='center', fontsize=10)
        else:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.text(0.5, 0.5, "Install sklearn\nfor PCA", ha='center', va='center', fontsize=10)


    # --- 19. Bilinmeyen tip
    else:
        ax = fig.add_subplot(1, 1, 1)
        type_name = type(first_elem).__name__
        ax.text(0.5, 0.5, f"Plotting not implemented\nfor '{type_name}'",
                ha='center', va='center', fontsize=14, fontweight='bold', color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    # If user runs module directly, configure basic logging to console for demonstration.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Keçeci Numbers Module - Demonstration")
    logger.info("This script demonstrates the generation of various Keçeci Number types.")

    STEPS = 40
    START_VAL = "2.5"
    ADD_VAL = 3.0

    all_types = {
        "Positive Real": TYPE_POSITIVE_REAL, "Negative Real": TYPE_NEGATIVE_REAL,
        "Complex": TYPE_COMPLEX, "Float": TYPE_FLOAT, "Rational": TYPE_RATIONAL,
        "Quaternion": TYPE_QUATERNION, "Neutrosophic": TYPE_NEUTROSOPHIC,
        "Neutrosophic Complex": TYPE_NEUTROSOPHIC_COMPLEX, "Hyperreal": TYPE_HYPERREAL,
        "Bicomplex": TYPE_BICOMPLEX, "Neutrosophic Bicomplex": TYPE_NEUTROSOPHIC_BICOMPLEX,
        "Octonion": TYPE_OCTONION, "Sedenion": TYPE_SEDENION, "Clifford": TYPE_CLIFFORD, 
        "Dual": TYPE_DUAL, "Splitcomplex": TYPE_SPLIT_COMPLEX, "Pathion": TYPE_PATHION,
        "Chingon": TYPE_CHINGON, "Routon": TYPE_ROUTON, "Voudon": TYPE_VOUDON,
        "Super Real": TYPE_SUPERREAL, "Ternary": TYPE_TERNARY,
    }

    for name, type_id in all_types.items():
        start = "-5" if type_id == TYPE_NEGATIVE_REAL else "2+3j" if type_id in [TYPE_COMPLEX, TYPE_BICOMPLEX] else START_VAL
        try:
            seq = get_with_params(type_id, STEPS, start, ADD_VAL)
            if seq:
                logger.info("Generated sequence for %s (len=%d).", name, len(seq))
                # Optional: plot for a few selected types to avoid overloading user's environment
        except Exception as e:
            logger.exception("Demo generation failed for type %s: %s", name, e)

    logger.info("Demonstration finished.")
