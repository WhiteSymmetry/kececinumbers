# -*- coding: utf-8 -*-
# test_sample.py
"""
Comprehensive unit tests for the kececinumbers module.
Tests core functionality, number type generation, and mathematical properties.
"""

import unittest
import numpy as np
import quaternion
from kececinumbers import (
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

    # Core generator
    unified_generator,

    # Helper functions
    is_prime,
    _get_integer_representation,
    find_kececi_prime_number,

    # Custom classes
    NeutrosophicNumber,
    NeutrosophicComplexNumber,
    HyperrealNumber,
    BicomplexNumber,
    NeutrosophicBicomplexNumber,
)


class TestKececiNumbers(unittest.TestCase):

    def setUp(self):
        """Run before each test."""
        self.iterations = 10

    def test_positive_real_generation(self):
        """Test sequence generation for positive real numbers."""
        seq = unified_generator(
            kececi_type=TYPE_POSITIVE_REAL,
            start_input_raw="5",
            add_input_raw=3.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], int)
        self.assertGreaterEqual(seq[0], 0)

    def test_negative_real_generation(self):
        """Test sequence generation for negative real numbers."""
        seq = unified_generator(
            kececi_type=TYPE_NEGATIVE_REAL,
            start_input_raw="-7",
            add_input_raw=2.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], int)
        self.assertLessEqual(seq[0], 0)

    def test_float_generation(self):
        """Test sequence generation for float numbers."""
        seq = unified_generator(
            kececi_type=TYPE_FLOAT,
            start_input_raw="3.14",
            add_input_raw=1.5,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], float)

    def test_complex_generation(self):
        """Test sequence generation for complex numbers."""
        seq = unified_generator(
            kececi_type=TYPE_COMPLEX,
            start_input_raw="2+3j",
            add_input_raw=1.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], complex)
        self.assertEqual(seq[0].real, 2.0)
        self.assertEqual(seq[0].imag, 3.0)

    def test_rational_generation(self):
        """Test sequence generation for rational numbers."""
        seq = unified_generator(
            kececi_type=TYPE_RATIONAL,
            start_input_raw="7/3",
            add_input_raw=2.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        from fractions import Fraction
        self.assertIsInstance(seq[0], Fraction)
        self.assertEqual(str(seq[0]), "7/3")

    def test_quaternion_generation(self):
        """Test sequence generation for quaternions."""
        seq = unified_generator(
            kececi_type=TYPE_QUATERNION,
            start_input_raw="1.0,2.0,3.0,4.0",
            add_input_raw="1.0,0,0,0",
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], np.quaternion)
        # Check if parsing worked
        q = seq[0]
        self.assertAlmostEqual(q.w, 1.0)
        self.assertAlmostEqual(q.x, 2.0)
        self.assertAlmostEqual(q.y, 3.0)
        self.assertAlmostEqual(q.z, 4.0)

    def test_neutrosophic_generation(self):
        """Test sequence generation for neutrosophic numbers."""
        seq = unified_generator(
            kececi_type=TYPE_NEUTROSOPHIC,
            start_input_raw="5+2I",
            add_input_raw=1.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], NeutrosophicNumber)
        self.assertEqual(seq[0].a, 5.0)
        self.assertEqual(seq[0].b, 2.0)

    def test_neutrosophic_complex_generation(self):
        """Test sequence generation for neutrosophic complex numbers."""
        seq = unified_generator(
            kececi_type=TYPE_NEUTROSOPHIC_COMPLEX,
            start_input_raw="1-2j",
            add_input_raw=1.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], NeutrosophicComplexNumber)
        self.assertAlmostEqual(seq[0].real, 1.0)
        self.assertAlmostEqual(seq[0].imag, -2.0)
        self.assertAlmostEqual(seq[0].indeterminacy, 0.0)

    def test_hyperreal_generation(self):
        """Test sequence generation for hyperreal numbers."""
        seq = unified_generator(
            kececi_type=TYPE_HYPERREAL,
            start_input_raw="5+3e",
            add_input_raw=1.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], HyperrealNumber)
        self.assertGreater(len(seq[0].sequence), 0)

    def test_bicomplex_generation(self):
        """Test sequence generation for bicomplex numbers."""
        seq = unified_generator(
            kececi_type=TYPE_BICOMPLEX,
            start_input_raw="2+1j",
            add_input_raw=1.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], BicomplexNumber)
        z1, z2 = seq[0].z1, seq[0].z2
        self.assertIsInstance(z1, complex)
        self.assertIsInstance(z2, complex)

    def test_neutrosophic_bicomplex_generation(self):
        """Test sequence generation for neutrosophic bicomplex numbers."""
        seq = unified_generator(
            kececi_type=TYPE_NEUTROSOPHIC_BICOMPLEX,
            start_input_raw="1+2j",
            add_input_raw=1.0,
            iterations=self.iterations
        )
        self.assertTrue(len(seq) > 0)
        self.assertIsInstance(seq[0], NeutrosophicBicomplexNumber)

    def test_get_integer_representation(self):
        """Test _get_integer_representation for various types."""
        self.assertEqual(_get_integer_representation(42), 42)
        self.assertEqual(_get_integer_representation(-15), 15)
        self.assertEqual(_get_integer_representation(3.14), 3)
        self.assertEqual(_get_integer_representation(complex(5, 3)), 5)
        self.assertEqual(_get_integer_representation(np.quaternion(7, 1, 2, 3)), 7)
        self.assertEqual(_get_integer_representation(NeutrosophicNumber(9.5, 2.1)), 9)
        self.assertEqual(_get_integer_representation(HyperrealNumber([10.1, 11.2])), 10)
        self.assertEqual(_get_integer_representation(None), None)

    def test_is_prime_with_supported_types(self):
        """Test is_prime function with various number types."""
        # Prime numbers
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(7))

        # Non-prime
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))

        # Complex: uses real part
        self.assertTrue(is_prime(complex(3, 4)))  # 3 is prime
        self.assertFalse(is_prime(complex(4, 5)))  # 4 is not prime

        # Quaternion: uses scalar part
        q = np.quaternion(5, 1, 2, 3)
        self.assertTrue(is_prime(q))  # 5 is prime

        # Neutrosophic: uses 'a' component
        n = NeutrosophicNumber(7.0, 1.0)
        self.assertTrue(is_prime(n))  # 7 is prime

        # Invalid input
        self.assertFalse(is_prime("invalid"))
        self.assertFalse(is_prime(None))

    def test_find_kececi_prime_number(self):
        """Test if a repeating prime is correctly identified."""
        # Simulate a sequence where 3 appears multiple times
        mock_sequence = [
            2, 3, 4, 5, 3, 6, 3  # 3 repeats
        ]
        kpn = find_kececi_prime_number(mock_sequence)
        self.assertEqual(kpn, 3)

        # No prime repeats
        mock_sequence2 = [2, 3, 5, 7]  # all prime but no repeat
        kpn2 = find_kececi_prime_number(mock_sequence2)
        self.assertIsNone(kpn2)

        # Empty sequence
        self.assertIsNone(find_kececi_prime_number([]))

    def test_divisibility_and_logic_flow(self):
        """Basic check that generator does not crash and produces expected types."""
        seq = unified_generator(
            kececi_type=TYPE_POSITIVE_REAL,
            start_input_raw="10",
            add_input_raw=2.0,
            iterations=5
        )
        self.assertGreater(len(seq), 5)
        # Should include integers, possibly divided by 2 or 3
        for item in seq:
            self.assertIsInstance(item, int)

    def test_edge_case_empty_start(self):
        """Test handling of invalid or empty input."""
        seq = unified_generator(
            kececi_type=TYPE_FLOAT,
            start_input_raw="",  # empty
            add_input_raw=1.0,
            iterations=1
        )
        self.assertEqual(len(seq), 0)  # should fail gracefully

    def test_unsupported_type(self):
        """Test that unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            unified_generator(
                kececi_type=99,  # invalid
                start_input_raw="0",
                add_input_raw=1.0,
                iterations=1
            )


if __name__ == '__main__':
    unittest.main()
