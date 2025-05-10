# kececinumbers.py

import matplotlib.pyplot as plt
import random
import numpy as np
import math
from fractions import Fraction
import quaternion # pip install numpy numpy-quaternion


# --- Helper Functions ---
def is_prime(n_input):
    """
    Checks if a given number (or its relevant part for complex types)
    is prime.
    For Rational numbers, its integer part is used.
    For Complex/Quaternion numbers, its real/scalar part is used.
    """
    value_to_check = 0
    if isinstance(n_input, (int, float)):
        value_to_check = abs(int(n_input))
    elif isinstance(n_input, Fraction):
        value_to_check = abs(int(n_input)) # Integer part of the Fraction
    elif isinstance(n_input, complex):
        value_to_check = abs(int(n_input.real))
    elif isinstance(n_input, np.quaternion): # numpy-quaternion type check
        value_to_check = abs(int(n_input.w)) # Scalar (real) part
    else: # Default for other cases
        try:
            value_to_check = abs(int(n_input))
        except (ValueError, TypeError):
            return False # Consider not prime

    if value_to_check < 2:
        return False
    for i in range(2, int(value_to_check**0.5) + 1):
        if value_to_check % i == 0:
            return False
    return True

# --- Main Keçeci Number Generator ---
def unified_generator(kececi_type, start_input_raw, add_input_base_scalar, iterations):
    """
    Calculates Unified Keçeci Numbers.
    kececi_type: 1 (Positive Real), 2 (Negative Real), 3 (Complex), 4 (Floating-Point),
                 5 (Rational), 6 (Quaternion)
    start_input_raw: Starting value (as string or number, appropriate for the type)
    add_input_base_scalar: Base scalar value for increment (interpreted based on type)
    iterations: Number of iterations
    """
    sequence = []
    current_value = None
    _add_value_typed = None # Type-specific add value
    ask_unit = None
    use_integer_division = False # Generally False, True only for int-based types

    # Set initial value, add value, and ASK unit based on type
    if kececi_type == 1: # Positive Real (treated as integer)
        current_value = int(start_input_raw)
        _add_value_typed = int(add_input_base_scalar)
        ask_unit = 1
        use_integer_division = True
    elif kececi_type == 2: # Negative Real (treated as integer)
        current_value = int(start_input_raw)
        _add_value_typed = int(add_input_base_scalar) # Usually negative
        ask_unit = 1
        use_integer_division = True
    elif kececi_type == 3: # Complex Numbers
        # Adım 1: start_input_raw'ı bir complex sayıya dönüştür
        if isinstance(start_input_raw, complex):
            # Zaten bir complex sayı ise, doğrudan kullan
            start_complex_val = start_input_raw
        elif isinstance(start_input_raw, (int, float)):
            # Eğer int veya float ise, s+sj yap (kullanıcının beklentisi buysa)
            s_scalar = float(start_input_raw)
            start_complex_val = complex(s_scalar, s_scalar)
        else: # String ise (örn: "1+2j", "3", "3.0")
            try:
                # Doğrudan complex sayı string'i mi diye dene (örn: "1+2j")
                start_complex_val = complex(str(start_input_raw))
            except ValueError:
                # Eğer complex() başarısız olduysa, skaler bir sayı string'i olabilir (örn: "3")
                # Bu durumda s+sj yapalım
                try:
                    s_scalar_from_string = float(str(start_input_raw))
                    start_complex_val = complex(s_scalar_from_string, s_scalar_from_string)
                except ValueError:
                    raise ValueError(f"Cannot convert start_input_raw '{start_input_raw}' to a complex number.")

        current_value = start_complex_val

        # Adım 2: add_input_base_scalar'dan _add_value_typed oluştur
        a_scalar_for_add = float(add_input_base_scalar)
        _add_value_typed = complex(a_scalar_for_add, a_scalar_for_add) # a+aj

        ask_unit = 1 + 1j
        # use_integer_division = False (Zaten varsayılan)
    elif kececi_type == 4: # Floating-Point Numbers
        current_value = float(start_input_raw)
        _add_value_typed = float(add_input_base_scalar)
        ask_unit = 1.0
    elif kececi_type == 5: # Rational Numbers
        if isinstance(start_input_raw, Fraction):
            current_value = start_input_raw
        else: # String ("3/2", "5") or number (5 -> 5/1)
            current_value = Fraction(str(start_input_raw))
        _add_value_typed = Fraction(str(add_input_base_scalar)) # Convert scalar base to Fraction
        ask_unit = Fraction(1, 1)
    elif kececi_type == 6: # Quaternions
        s_val_q_raw = float(start_input_raw) if not isinstance(start_input_raw, np.quaternion) else start_input_raw
        a_val_q_scalar = float(add_input_base_scalar)
        if isinstance(s_val_q_raw, np.quaternion):
            current_value = s_val_q_raw
        else: # From scalar, make q(s,s,s,s)
            current_value = np.quaternion(s_val_q_raw, s_val_q_raw, s_val_q_raw, s_val_q_raw)
        _add_value_typed = np.quaternion(a_val_q_scalar, a_val_q_scalar, a_val_q_scalar, a_val_q_scalar)
        ask_unit = np.quaternion(1, 1, 1, 1) # Like (1+1i+1j+1k)
    else:
        raise ValueError("Invalid Keçeci Number Type")

    sequence.append(current_value)
    last_divisor_used = None # Last divisor used (2 or 3)
    ask_counter = 0 # 0: first time +unit, 1: second time -unit

    for _ in range(iterations):
        added_value = current_value + _add_value_typed
        sequence.append(added_value)

        value_for_primality_check = added_value
        
        primary_divisor = 3 if last_divisor_used is None or last_divisor_used == 2 else 2
        alternative_divisor = 2 if primary_divisor == 3 else 3
        
        divided_successfully = False
        result_value = None 

        for divisor_candidate in [primary_divisor, alternative_divisor]:
            can_divide = False
            if kececi_type in [1, 2]: 
                can_divide = (added_value % divisor_candidate == 0)
            elif kececi_type == 3: 
                can_divide = math.isclose(added_value.real % divisor_candidate, 0)
            elif kececi_type == 4: 
                can_divide = math.isclose(added_value % divisor_candidate, 0) or \
                             math.isclose(added_value % divisor_candidate, divisor_candidate)
            elif kececi_type == 5: 
                if divisor_candidate != 0:
                    quotient_rational = added_value / divisor_candidate
                    can_divide = (quotient_rational.denominator == 1)
            elif kececi_type == 6: 
                can_divide = math.isclose(added_value.w % divisor_candidate, 0)

            if can_divide:
                if use_integer_division:
                    result_value = added_value // divisor_candidate
                else:
                    result_value = added_value / divisor_candidate
                last_divisor_used = divisor_candidate
                divided_successfully = True
                break 

        if not divided_successfully:
            if is_prime(value_for_primality_check):
                modified_value = None
                if ask_counter == 0:
                    modified_value = added_value + ask_unit
                    ask_counter = 1
                else: 
                    modified_value = added_value - ask_unit
                    ask_counter = 0
                sequence.append(modified_value)
                
                current_target_for_division_mod = modified_value
                divided_after_modification = False
                for divisor_candidate_mod in [primary_divisor, alternative_divisor]:
                    can_divide_mod = False
                    if kececi_type in [1, 2]:
                        can_divide_mod = (current_target_for_division_mod % divisor_candidate_mod == 0)
                    elif kececi_type == 3:
                        can_divide_mod = math.isclose(current_target_for_division_mod.real % divisor_candidate_mod, 0)
                    elif kececi_type == 4:
                        can_divide_mod = math.isclose(current_target_for_division_mod % divisor_candidate_mod, 0) or \
                                         math.isclose(current_target_for_division_mod % divisor_candidate_mod, divisor_candidate_mod)
                    elif kececi_type == 5:
                        if divisor_candidate_mod != 0:
                            quotient_rational_mod = current_target_for_division_mod / divisor_candidate_mod
                            can_divide_mod = (quotient_rational_mod.denominator == 1)
                    elif kececi_type == 6:
                        can_divide_mod = math.isclose(current_target_for_division_mod.w % divisor_candidate_mod, 0)

                    if can_divide_mod:
                        if use_integer_division:
                            result_value = current_target_for_division_mod // divisor_candidate_mod
                        else:
                            result_value = current_target_for_division_mod / divisor_candidate_mod
                        last_divisor_used = divisor_candidate_mod
                        divided_after_modification = True
                        break
                if not divided_after_modification:
                    result_value = modified_value 
            else: 
                result_value = added_value
        
        sequence.append(result_value)
        current_value = result_value
        
    return sequence

# --- Control Mechanisms (Exportable Functions) ---
def get_interactive():
    """
    Interactively gets parameters from the user and generates Keçeci Numbers.
    """
    print("Keçeci Number Types:")
    print("1: Positive Real Numbers (Integer: e.g., 1)")
    print("2: Negative Real Numbers (Integer: e.g., -3)")
    print("3: Complex Numbers (e.g., 3+4j)")
    print("4: Floating-Point Numbers (e.g., 2.5)")
    print("5: Rational Numbers (e.g., 3/2, 5)")
    print("6: Quaternions (scalar start input becomes q(s,s,s,s): e.g.,  1 or 2.5)")
    
    while True:
        try:
            type_choice = int(input("Please select Keçeci Number Type (1-6): "))
            if 1 <= type_choice <= 6: break
            else: print("Invalid type.")
        except ValueError: print("Please enter a numeric value.")

    start_prompt = "Enter the starting number (e.g., 0 or 2.5, complex:3+4j, rational: 3/4, quaternions: 1)  : "
    if type_choice == 3: start_prompt += "(e.g., 3+4j or just 3): "
    elif type_choice == 5: start_prompt += "(e.g., 7/2 or 5): "

    start_input_val_raw = input(start_prompt) 

    while True:
        try:
            add_base_scalar_val = float(input("Enter the base scalar value for increment (e.g., 9): "))
            break
        except ValueError: print("Please enter a numeric value.")
            
    while True:
        try:
            num_iterations = int(input("Enter the number of iterations (positive integer: e.g., 30): "))
            if num_iterations > 0: break
            else: print("Number of iterations must be positive.")
        except ValueError: print("Please enter an integer value.")
            
    return unified_generator(type_choice, start_input_val_raw, add_base_scalar_val, num_iterations)

def get_with_params(kececi_type_choice, iterations, start_value_raw="0", add_value_base_scalar=9.0, fixed_params=True, random_range_factor=10):
    """
    Generates Keçeci Numbers with specified or randomized parameters.
    If fixed_params is False, start_value_raw and add_value_base_scalar are used as bases for randomization.
    random_range_factor influences the range of random values.
    """
    actual_start_raw = start_value_raw
    actual_add_base = add_value_base_scalar

    if not fixed_params:
        # Basic randomization logic, can be expanded
        if kececi_type_choice == 1: # Positive
            actual_start_raw = str(random.randint(0, int(random_range_factor)))
            actual_add_base = float(random.randint(1, int(random_range_factor*1.5)))
        elif kececi_type_choice == 2: # Negative
            actual_start_raw = str(random.randint(-int(random_range_factor), 0))
            actual_add_base = float(random.randint(-int(random_range_factor*1.5), -1))
        elif kececi_type_choice == 3: # Complex
            actual_start_raw = str(random.uniform(-random_range_factor/2, random_range_factor/2))
            actual_add_base = random.uniform(1, random_range_factor/2)
        elif kececi_type_choice == 4: # Float
            actual_start_raw = str(random.uniform(-random_range_factor, random_range_factor))
            actual_add_base = random.uniform(0.1, random_range_factor/2)
        elif kececi_type_choice == 5: # Rational
            num = random.randint(-random_range_factor, random_range_factor)
            den = random.randint(1, int(random_range_factor/2) if random_range_factor/2 >=1 else 1)
            actual_start_raw = f"{num}/{den}"
            actual_add_base = float(random.randint(1,random_range_factor))
        elif kececi_type_choice == 6: # Quaternion
            actual_start_raw = str(random.uniform(-random_range_factor/2, random_range_factor/2))
            actual_add_base = random.uniform(1, random_range_factor/2)
    else: # Fixed parameters, but adjust for negative type if default add is positive
        if kececi_type_choice == 2 and actual_add_base > 0:
            actual_add_base = -actual_add_base


    return unified_generator(kececi_type_choice, actual_start_raw, actual_add_base, iterations)


def get_random_type(num_iterations, use_fixed_params_for_selected_type=True,
                    fixed_start_raw="0", fixed_add_base_scalar=9.0, random_factor=10):
    """
    Generates Keçeci Numbers for a randomly selected type.
    """
    random_type_choice = random.randint(1, 6)
    type_names_list = ["Positive Integer", "Negative Integer", "Complex", "Float", "Rational", "Quaternion"]
    print(f"Randomly selected Keçeci Number Type: {random_type_choice} ({type_names_list[random_type_choice-1]})")
    
    return get_with_params(random_type_choice, num_iterations,
                           start_value_raw=fixed_start_raw,
                           add_value_base_scalar=fixed_add_base_scalar,
                           fixed_params=use_fixed_params_for_selected_type,
                           random_range_factor=random_factor)

# --- Plotting Function (can be called from the notebook) ---
def plot_numbers(sequence, title="Keçeci Numbers"):
    """
    Plots the generated Keçeci Number sequence.
    """
    plt.figure(figsize=(14, 8))
    
    if not sequence:
        print("Sequence is empty, nothing to plot.")
        return

    first_elem = sequence[0]

    if isinstance(first_elem, np.quaternion):
        w_parts = [q.w for q in sequence]
        # x_parts = [q.x for q in sequence] # Optional
        # y_parts = [q.y for q in sequence] # Optional
        # z_parts = [q.z for q in sequence] # Optional
        vector_norms = [np.sqrt(q.x**2 + q.y**2 + q.z**2) for q in sequence]

        plt.subplot(2, 1, 1)
        plt.plot(w_parts, marker='o', linestyle='-', label='w (Scalar Part)')
        plt.title(title + " - Quaternion Scalar Part (w)")
        plt.xlabel("Index"); plt.ylabel("Value"); plt.grid(True); plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(vector_norms, marker='x', linestyle='--', color='purple', label='Vector Part Norm (|xi+yj+zk|)')
        plt.title(title + " - Quaternion Vector Part Norm")
        plt.xlabel("Index"); plt.ylabel("Value"); plt.grid(True); plt.legend()

    elif isinstance(first_elem, complex):
        real_parts = [n.real for n in sequence]
        imag_parts = [n.imag for n in sequence]
        
        plt.subplot(2, 1, 1)
        plt.plot(real_parts, marker='o', linestyle='-', label='Real Part')
        plt.title(title + " - Complex Real Part"); plt.xlabel("Index")
        plt.ylabel("Value"); plt.grid(True); plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(imag_parts, marker='x', linestyle='--', color='red', label='Imaginary Part')
        plt.title(title + " - Complex Imaginary Part"); plt.xlabel("Index")
        plt.ylabel("Value"); plt.grid(True); plt.legend()
        
        plt.figure(figsize=(8,8)) 
        plt.plot(real_parts, imag_parts, marker='.', linestyle='-')
        if real_parts:
            plt.plot(real_parts[0], imag_parts[0], 'go', markersize=8, label='Start')
            plt.plot(real_parts[-1], imag_parts[-1], 'ro', markersize=8, label='End')
        plt.title(title + " - Trajectory in Complex Plane"); plt.xlabel("Real Axis")
        plt.ylabel("Imaginary Axis"); plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5); plt.grid(True); plt.legend(); plt.axis('equal')

    elif isinstance(first_elem, Fraction):
        float_sequence = [float(f) for f in sequence] 
        plt.plot(float_sequence, marker='o', linestyle='-')
        plt.title(title + " (Rational Numbers - plotted as float)")
        plt.xlabel("Index"); plt.ylabel("Value (float)"); plt.grid(True)
    
    else: 
        try:
            numeric_sequence = np.array(sequence, dtype=float) 
            plt.plot(numeric_sequence, marker='o', linestyle='-')
        except ValueError:
            print("Sequence contains non-plottable values. Plotting only numeric ones.")
            plt.plot([x for x in sequence if isinstance(x, (int, float))], marker='o', linestyle='-')
        plt.title(title); plt.xlabel("Index")
        plt.ylabel("Value"); plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# --- DEFINITIONS (as a multiline string, can be accessed as kececinumbers.DEFINITIONS) ---
DEFINITIONS = """
Keçeci NumberS UNIFIED DEFINITION

A Keçeci Number sequence is derived from a `start_input_raw` and an `add_input_base_scalar`.
These inputs are interpreted according to the selected Keçeci Number Type to become the `current_value` and a type-specific `_add_value_typed`.

In each step:
1.  `added_value = current_value + _add_value_typed`.
2.  `added_value` is recorded in the sequence.
3.  A `result_value` is obtained by applying the Division Rule and, if necessary, the ASK Rule to `added_value`.
4.  `result_value` is recorded in the sequence.
5.  `current_value = result_value`.
This process is repeated for the specified `number_of_iterations`.

Division Rule:
*   A `last_divisor_used` (2 or 3) is tracked.
*   `primary_divisor`: If `last_divisor_used` is nonexistent (first step) or 2, it's 3; if it's 3, it's 2.
*   `alternative_divisor`: The other of `primary_divisor` (2 or 3).
*   `value_to_check_division`: This is `added_value` (or `modified_value` after ASK).
    *   The part of this value used for divisibility depends on the number type (e.g., real/scalar part for complex/quaternion, the fraction itself for rational).
*   First, division by `primary_divisor` is attempted:
    *   If `value_to_check_division` (or its relevant part) is "perfectly divisible" by `primary_divisor` (in a type-specific sense, e.g., for rationals, the result is an integer),
        then `result_value = value_to_check_division / primary_divisor`. `last_divisor_used = primary_divisor`.
*   If unsuccessful, division by `alternative_divisor` is attempted:
    *   If `value_to_check_division` (or its relevant part) is "perfectly divisible" by `alternative_divisor`,
        then `result_value = value_to_check_division / alternative_divisor`. `last_divisor_used = alternative_divisor`.

Primality and ASK (Augment/Shrink then Check) Rule (if Division Fails):
*   `value_for_primality_check`: The part/representation of `added_value` used for primality testing (e.g., integer part, real part).
*   If `is_prime(value_for_primality_check)` is true:
    *   An `ask_counter` (0 or 1) is used.
    *   `ask_unit`: A type-specific unit value (e.g., 1 for real, 1+1j for complex, Fraction(1,1) for rational, quaternion(1,1,1,1) for quaternion).
    *   If `ask_counter` is 0: `modified_value = added_value + ask_unit`, `ask_counter = 1`.
    *   If `ask_counter` is 1: `modified_value = added_value - ask_unit`, `ask_counter = 0`.
    *   `modified_value` is added to the sequence.
    *   The Division Rule above is re-attempted on this `modified_value`.
        *   If division is successful, `result_value` is the quotient.
        *   If unsuccessful, `result_value = modified_value`.
*   If `is_prime(value_for_primality_check)` is false (and it wasn't divisible):
    *   `result_value = added_value`.

Number Types and Specifics:
1.  Positive Real Numbers (Treated as Integer):
    *   Start/Increment: Positive integers.
    *   Division: Integer division (`//`). Perfect divisibility: `% == 0`.
    *   ASK unit: `1`. Primality: `abs(int(number))`.
2.  Negative Real Numbers (Treated as Integer):
    *   Start/Increment: Generally negative integers.
    *   Division: Integer division (`//`). Perfect divisibility: `% == 0`.
    *   ASK unit: `1`. Primality: `abs(int(number))`.
3.  Complex Numbers (`complex`):
    *   Start/Increment: Complex numbers. Scalar input `s` is interpreted as `s+sj` for start, and scalar `a` as `a+aj` for increment.
    *   Division: Complex division (`/`). Perfect divisibility: Real part is `math.isclose(real_part % integer_divisor, 0)`.
    *   ASK unit: `1+1j`. Primality: `abs(int(number.real))`.
4.  Floating-Point Numbers (Treated as Float):
    *   Start/Increment: Decimal numbers.
    *   Division: Float division (`/`). Perfect divisibility: `math.isclose(number % integer_divisor, 0)`.
    *   ASK unit: `1.0`. Primality: `abs(int(number))`.
5.  Rational Numbers (`fractions.Fraction`):
    *   Start/Increment: Fractions. Scalar input `s` is interpreted as `Fraction(s,1)`.
    *   Division: Fraction division (`/`). Perfect divisibility: `(fraction / integer_divisor).denominator == 1`.
    *   ASK unit: `Fraction(1,1)`. Primality: `abs(int(fraction))`.
6.  Quaternions (`numpy.quaternion`):
    *   Start/Increment: Quaternions. Scalar input `s` is interpreted as `q(s,s,s,s)` for start, and scalar `a` as `q(a,a,a,a)` for increment.
    *   Division: Quaternion division (`/`). Perfect divisibility: Scalar (w) part is `math.isclose(w_part % integer_divisor, 0)`.
    *   ASK unit: `quaternion(1,1,1,1)`. Primality: `abs(int(number.w))`.
"""

# Constants for Kececi Types (makes it easier to use from outside)
TYPE_POSITIVE_REAL = 1
TYPE_NEGATIVE_REAL = 2
TYPE_COMPLEX = 3
TYPE_FLOAT = 4
TYPE_RATIONAL = 5
TYPE_QUATERNION = 6

# You can remove the __main__ block if this is purely a module
# or keep it for testing the module directly.
if __name__ == "__main__":
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

    # Basic test
    print("\nRunning a quick test for Complex numbers (fixed params):")
    test_seq = get_with_params(TYPE_COMPLEX, 10, start_value_raw="1", add_value_base_scalar=2.0, fixed_params=True)
    if test_seq:
        for i, val in enumerate(test_seq[:5]):
             if isinstance(val, complex): print(f" {i}: {val.real:.1f}{val.imag:+.1f}j")
             else: print(f" {i}: {val}")
        # plot_numbers(test_seq, "Module Test - Complex") # Uncomment to plot if running directly
