# kececinumbers.py

import matplotlib.pyplot as plt
import random
import numpy as np
import math
from fractions import Fraction
import quaternion # pip install numpy numpy-quaternion
import collections


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
    # Optimize for 2 and even numbers
    if value_to_check == 2:
        return True
    if value_to_check % 2 == 0:
        return False
    # Check only odd divisors up to sqrt(n)
    for i in range(3, int(value_to_check**0.5) + 1, 2):
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
        if isinstance(start_input_raw, complex):
            start_complex_val = start_input_raw
        elif isinstance(start_input_raw, (int, float)):
            s_scalar = float(start_input_raw)
            start_complex_val = complex(s_scalar, s_scalar)
        else: 
            try:
                start_complex_val = complex(str(start_input_raw))
            except ValueError:
                try:
                    s_scalar_from_string = float(str(start_input_raw))
                    start_complex_val = complex(s_scalar_from_string, s_scalar_from_string)
                except ValueError:
                    raise ValueError(f"Cannot convert start_input_raw '{start_input_raw}' to a complex number.")
        current_value = start_complex_val
        a_scalar_for_add = float(add_input_base_scalar)
        _add_value_typed = complex(a_scalar_for_add, a_scalar_for_add) # a+aj
        ask_unit = 1 + 1j
    elif kececi_type == 4: # Floating-Point Numbers
        current_value = float(start_input_raw)
        _add_value_typed = float(add_input_base_scalar)
        ask_unit = 1.0
    elif kececi_type == 5: # Rational Numbers
        if isinstance(start_input_raw, Fraction):
            current_value = start_input_raw
        else: 
            current_value = Fraction(str(start_input_raw))
        _add_value_typed = Fraction(str(add_input_base_scalar)) 
        ask_unit = Fraction(1, 1)
    elif kececi_type == 6: # Quaternions
        s_val_q_raw = float(start_input_raw) if not isinstance(start_input_raw, np.quaternion) else start_input_raw
        a_val_q_scalar = float(add_input_base_scalar)
        if isinstance(s_val_q_raw, np.quaternion):
            current_value = s_val_q_raw
        else: 
            current_value = np.quaternion(s_val_q_raw, s_val_q_raw, s_val_q_raw, s_val_q_raw)
        _add_value_typed = np.quaternion(a_val_q_scalar, a_val_q_scalar, a_val_q_scalar, a_val_q_scalar)
        ask_unit = np.quaternion(1, 1, 1, 1) 
    else:
        raise ValueError("Invalid Keçeci Number Type")

    sequence.append(current_value)
    last_divisor_used = None 
    ask_counter = 0 

    actual_iterations_done = 0
    while actual_iterations_done < iterations:
        added_value = current_value + _add_value_typed
        sequence.append(added_value)
        actual_iterations_done += 1
        if actual_iterations_done >= iterations: break


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
                can_divide = math.isclose(added_value.real % divisor_candidate, 0) and math.isclose(added_value.imag % divisor_candidate, 0)
            elif kececi_type == 4: 
                can_divide = math.isclose(added_value % divisor_candidate, 0) or \
                             math.isclose(added_value % divisor_candidate, divisor_candidate) # handles floating point precision
            elif kececi_type == 5: 
                if divisor_candidate != 0:
                    quotient_rational = added_value / divisor_candidate
                    can_divide = (quotient_rational.denominator == 1)
            elif kececi_type == 6: 
                # For quaternions, typically division is by scalar. Check if all components are divisible.
                # Or, as per your current `is_prime`, just check the scalar part.
                # Let's stick to the scalar part for consistency with `is_prime` for now.
                can_divide = math.isclose(added_value.w % divisor_candidate, 0) and \
                             math.isclose(added_value.x % divisor_candidate, 0) and \
                             math.isclose(added_value.y % divisor_candidate, 0) and \
                             math.isclose(added_value.z % divisor_candidate, 0)


            if can_divide:
                if use_integer_division:
                    result_value = added_value // divisor_candidate
                else:
                    result_value = added_value / divisor_candidate # For complex, float, rational, quaternion
                last_divisor_used = divisor_candidate
                divided_successfully = True
                break 

        if not divided_successfully:
            if is_prime(value_for_primality_check): # is_prime checks the relevant part (e.g., real part)
                modified_value = None
                if ask_counter == 0:
                    modified_value = added_value + ask_unit
                    ask_counter = 1
                else: 
                    modified_value = added_value - ask_unit
                    ask_counter = 0
                sequence.append(modified_value)
                actual_iterations_done += 1
                if actual_iterations_done >= iterations: 
                    result_value = modified_value # End with modified value if it's the last step
                    break


                current_target_for_division_mod = modified_value
                divided_after_modification = False
                for divisor_candidate_mod in [primary_divisor, alternative_divisor]: # Re-use primary/alternative logic
                    can_divide_mod = False
                    if kececi_type in [1, 2]:
                        can_divide_mod = (current_target_for_division_mod % divisor_candidate_mod == 0)
                    elif kececi_type == 3:
                        can_divide_mod = math.isclose(current_target_for_division_mod.real % divisor_candidate_mod, 0) and \
                                         math.isclose(current_target_for_division_mod.imag % divisor_candidate_mod, 0)
                    elif kececi_type == 4:
                        can_divide_mod = math.isclose(current_target_for_division_mod % divisor_candidate_mod, 0) or \
                                         math.isclose(current_target_for_division_mod % divisor_candidate_mod, divisor_candidate_mod)
                    elif kececi_type == 5:
                        if divisor_candidate_mod != 0:
                            quotient_rational_mod = current_target_for_division_mod / divisor_candidate_mod
                            can_divide_mod = (quotient_rational_mod.denominator == 1)
                    elif kececi_type == 6:
                        can_divide_mod = math.isclose(current_target_for_division_mod.w % divisor_candidate_mod, 0) and \
                                         math.isclose(current_target_for_division_mod.x % divisor_candidate_mod, 0) and \
                                         math.isclose(current_target_for_division_mod.y % divisor_candidate_mod, 0) and \
                                         math.isclose(current_target_for_division_mod.z % divisor_candidate_mod, 0)


                    if can_divide_mod:
                        if use_integer_division:
                            result_value = current_target_for_division_mod // divisor_candidate_mod
                        else:
                            result_value = current_target_for_division_mod / divisor_candidate_mod
                        last_divisor_used = divisor_candidate_mod # Update last_divisor_used
                        divided_after_modification = True
                        break
                if not divided_after_modification:
                    result_value = modified_value 
            else: # Not prime and not divisible
                result_value = added_value
        
        sequence.append(result_value)
        actual_iterations_done += 1
        if actual_iterations_done >= iterations: break
        current_value = result_value
        
    return sequence[:iterations+1] # Ensure correct length, as we add start + iterations*2 steps


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
    if type_choice == 3: start_prompt = "Enter starting complex number (e.g., 3+4j or just 3 for 3+3j): "
    elif type_choice == 5: start_prompt = "Enter starting rational (e.g., 7/2 or 5 for 5/1): "

    start_input_val_raw = input(start_prompt) 

    while True:
        try:
            add_base_scalar_val = float(input("Enter the base scalar value for increment (e.g., 9): "))
            break
        except ValueError: print("Please enter a numeric value.")
            
    while True:
        try:
            # Iterations'ı Keçeci adımları olarak düşünelim (her adımda 2 sayı eklenir: added_value, result_value)
            num_kececi_steps = int(input("Enter the number of Keçeci steps (positive integer: e.g., 15, generates ~30 numbers): "))
            if num_kececi_steps > 0: break
            else: print("Number of Keçeci steps must be positive.")
        except ValueError: print("Please enter an integer value.")
            
    generated_sequence = unified_generator(type_choice, start_input_val_raw, add_base_scalar_val, num_kececi_steps)
    
    # *** YENİ EKLENEN KISIM BAŞLANGICI ***
    if generated_sequence:
        print(f"\nGenerated Keçeci Sequence (first 20 of {len(generated_sequence)}): {generated_sequence[:20]}...")
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            print(f"Keçeci Prime Number for this sequence: {kpn}")
        else:
            print("No Keçeci Prime Number found for this sequence.")
    else:
        print("No sequence generated.")
    # *** YENİ EKLENEN KISIM SONU ***
            
    return generated_sequence # Fonksiyon yine de diziyi döndürmeli

def get_with_params(kececi_type_choice, iterations, start_value_raw="0", add_value_base_scalar=9.0, fixed_params=True, random_range_factor=10):
    """
    Generates Keçeci Numbers with specified or randomized parameters.
    If fixed_params is False, start_value_raw and add_value_base_scalar are used as bases for randomization.
    random_range_factor influences the range of random values.
    """
    actual_start_raw = start_value_raw
    actual_add_base = add_value_base_scalar

    if not fixed_params:
        if kececi_type_choice == 1: 
            actual_start_raw = str(random.randint(0, int(random_range_factor)))
            actual_add_base = float(random.randint(1, int(random_range_factor*1.5)))
        elif kececi_type_choice == 2: 
            actual_start_raw = str(random.randint(-int(random_range_factor), 0))
            actual_add_base = float(random.randint(-int(random_range_factor*1.5), -1))
        elif kececi_type_choice == 3: 
            re_start = random.uniform(-random_range_factor/2, random_range_factor/2)
            im_start = random.uniform(-random_range_factor/2, random_range_factor/2)
            actual_start_raw = f"{re_start}{im_start:+}j" # String formatında complex
            actual_add_base = random.uniform(1, random_range_factor/2)
        elif kececi_type_choice == 4: 
            actual_start_raw = str(random.uniform(-random_range_factor, random_range_factor))
            actual_add_base = random.uniform(0.1, random_range_factor/2)
        elif kececi_type_choice == 5: 
            num = random.randint(-random_range_factor, random_range_factor)
            den = random.randint(1, int(random_range_factor/2) if random_range_factor/2 >=1 else 1)
            actual_start_raw = f"{num}/{den}"
            actual_add_base = float(random.randint(1,random_range_factor))
        elif kececi_type_choice == 6: 
            # Quaternion için başlangıç ve ekleme skaler olsun, unified_generator q(s,s,s,s) yapsın
            actual_start_raw = str(random.uniform(-random_range_factor/2, random_range_factor/2))
            actual_add_base = random.uniform(1, random_range_factor/2)
    else: 
        if kececi_type_choice == 2 and float(actual_add_base) > 0: # add_value_base_scalar float olabilir
            actual_add_base = -abs(float(actual_add_base))


    generated_sequence = unified_generator(kececi_type_choice, actual_start_raw, actual_add_base, iterations)
    
    # *** YENİ EKLENEN KISIM BAŞLANGICI ***
    if generated_sequence:
        print(f"\nGenerated Keçeci Sequence (using get_with_params, first 20 of {len(generated_sequence)}): {generated_sequence[:20]}...")
        kpn = find_kececi_prime_number(generated_sequence)
        if kpn is not None:
            print(f"Keçeci Prime Number for this sequence: {kpn}")
        else:
            print("No Keçeci Prime Number found for this sequence.")
    else:
        print("No sequence generated by get_with_params.")
    # *** YENİ EKLENEN KISIM SONU ***

    return generated_sequence


def get_random_type(num_iterations, use_fixed_params_for_selected_type=True,
                    fixed_start_raw="0", fixed_add_base_scalar=9.0, random_factor=10):
    """
    Generates Keçeci Numbers for a randomly selected type.
    """
    random_type_choice = random.randint(1, 6)
    type_names_list = ["Positive Integer", "Negative Integer", "Complex", "Float", "Rational", "Quaternion"]
    print(f"\nRandomly selected Keçeci Number Type: {random_type_choice} ({type_names_list[random_type_choice-1]})")
    
    # get_with_params fonksiyonu zaten KPN'yi yazdıracak, bu yüzden burada tekrar yazdırmaya gerek yok.
    # Sadece get_with_params'ı çağırıyoruz.
    generated_sequence = get_with_params(random_type_choice, num_iterations,
                                         start_value_raw=fixed_start_raw,
                                         add_value_base_scalar=fixed_add_base_scalar,
                                         fixed_params=use_fixed_params_for_selected_type,
                                         random_range_factor=random_factor)
    return generated_sequence # get_with_params zaten KPN yazdırıyor.

# --- Plotting Function (can be called from the notebook) ---
def plot_numbers(sequence, title="Keçeci Numbers"):
    """
    Plots the generated Keçeci Number sequence.
    """
    plt.figure(figsize=(14, 8))
    
    if not sequence:
        print("Sequence is empty, nothing to plot.")
        plt.title(title + " (Empty Sequence)")
        plt.text(0.5, 0.5, "Empty Sequence", ha='center', va='center', fontsize=16)
        plt.show()
        return

    first_elem = sequence[0]

    if isinstance(first_elem, np.quaternion):
        # Filter out non-quaternion if any mixed types (should not happen with unified_generator)
        q_sequence = [q for q in sequence if isinstance(q, np.quaternion)]
        if not q_sequence:
            print("No quaternion data to plot.")
            plt.title(title + " (No Quaternion Data)")
            plt.text(0.5, 0.5, "No Quaternion Data", ha='center', va='center', fontsize=16)
            plt.show()
            return

        w_parts = [q.w for q in q_sequence]
        vector_norms = [np.sqrt(q.x**2 + q.y**2 + q.z**2) for q in q_sequence]

        plt.subplot(2, 1, 1)
        plt.plot(w_parts, marker='o', linestyle='-', label='w (Scalar Part)')
        plt.title(title + " - Quaternion Scalar Part (w)")
        plt.xlabel("Index"); plt.ylabel("Value"); plt.grid(True); plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(vector_norms, marker='x', linestyle='--', color='purple', label='Vector Part Norm (|xi+yj+zk|)')
        plt.title(title + " - Quaternion Vector Part Norm")
        plt.xlabel("Index"); plt.ylabel("Value"); plt.grid(True); plt.legend()

    elif isinstance(first_elem, complex):
        c_sequence = [c for c in sequence if isinstance(c, complex)]
        if not c_sequence:
            print("No complex data to plot.")
            plt.title(title + " (No Complex Data)")
            plt.text(0.5, 0.5, "No Complex Data", ha='center', va='center', fontsize=16)
            plt.show()
            return

        real_parts = [n.real for n in c_sequence]
        imag_parts = [n.imag for n in c_sequence]
        
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
        if real_parts: # Check if list is not empty
            plt.plot(real_parts[0], imag_parts[0], 'go', markersize=10, label='Start')
            plt.plot(real_parts[-1], imag_parts[-1], 'ro', markersize=10, label='End')
        plt.title(title + " - Trajectory in Complex Plane"); plt.xlabel("Real Axis")
        plt.ylabel("Imaginary Axis"); plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5); plt.grid(True); plt.legend(); plt.axis('equal')

    elif isinstance(first_elem, Fraction):
        f_sequence = [f for f in sequence if isinstance(f, Fraction)]
        if not f_sequence:
            print("No rational data to plot.")
            plt.title(title + " (No Rational Data)")
            plt.text(0.5, 0.5, "No Rational Data", ha='center', va='center', fontsize=16)
            plt.show()
            return
        float_sequence = [float(f) for f in f_sequence] 
        plt.plot(float_sequence, marker='o', linestyle='-')
        plt.title(title + " (Rational Numbers - plotted as float)")
        plt.xlabel("Index"); plt.ylabel("Value (float)"); plt.grid(True)
    
    else: # Integers, Floats, or mixed (if any error in generation)
        try:
            # Attempt to convert all to float for plotting
            numeric_sequence = [float(x) for x in sequence if isinstance(x, (int, float, np.number))]
            if not numeric_sequence: # If all were non-numeric after filtering
                 raise ValueError("No numeric data to plot after filtering.")
            plt.plot(numeric_sequence, marker='o', linestyle='-')
        except (ValueError, TypeError):
            print(f"Warning: Sequence for '{title}' contains non-standard numeric or mixed types. Attempting basic plot.")
            # Fallback for truly mixed or unplottable types: plot what you can as numbers
            plottable_part = []
            for x in sequence:
                try: plottable_part.append(float(x))
                except: pass # Ignore non-convertible
            if plottable_part:
                 plt.plot(plottable_part, marker='o', linestyle='-')
            else:
                 print("Could not plot any part of the sequence.")
                 plt.title(title + " (Non-Numeric or Unplottable Data)")
                 plt.text(0.5, 0.5, "Non-Numeric or Unplottable Data", ha='center', va='center', fontsize=16)

        plt.title(title); plt.xlabel("Index")
        plt.ylabel("Value"); plt.grid(True)
    
    plt.tight_layout()
    # plt.show() # Genellikle notebook'ta %matplotlib inline ile otomatik gösterilir.
                 # .py script'te bu satırın yorumunu kaldırmak gerekebilir.

def find_kececi_prime_number(kececi_numbers_list):
    """
    Verilen Keçeci sayıları listesinden Keçeci Asal Sayısını bulur.
    Keçeci Asal Sayısı, listede en sık tekrarlayan (veya reel/skaler kısmı asal olan) sayının
    asal tamsayı temsilcisinin en sık tekrarlayanıdır.
    Eğer frekanslar eşitse, daha büyük olan asal tamsayı temsilcisi tercih edilir.
    """
    if not kececi_numbers_list:
        # Modül içinde çağrıldığı için print yerine None dönmesi daha sessiz olur.
        return None

    integer_prime_representations = []
    for num_original in kececi_numbers_list:
        if is_prime(num_original): 
            value_checked = 0
            if isinstance(num_original, (int, float)):
                value_checked = abs(int(num_original))
            elif isinstance(num_original, Fraction):
                value_checked = abs(int(num_original))
            elif isinstance(num_original, complex):
                value_checked = abs(int(num_original.real))
            elif isinstance(num_original, np.quaternion):
                value_checked = abs(int(num_original.w))
            else:
                try:
                    value_checked = abs(int(num_original))
                except (ValueError, TypeError):
                    continue 
            integer_prime_representations.append(value_checked)

    if not integer_prime_representations:
        return None

    counts = collections.Counter(integer_prime_representations)
    repeating_primes_info = []
    for prime_int_val, freq in counts.items():
        if freq > 1: # Sadece tekrarlayan asallar
            repeating_primes_info.append((freq, prime_int_val))

    if not repeating_primes_info:
        return None
    
    # En yüksek frekansa sahip olanı bul. Frekanslar eşitse,
    # max() ikinci elemanı (prime_int_val) kullanarak büyük olanı seçer.
    # (frekans, sayı) tuple'ları karşılaştırılır.
    try:
        best_freq, kececi_prime_integer = max(repeating_primes_info)
    except ValueError: # repeating_primes_info boşsa (yukarıdaki if ile yakalanmalı ama ekstra güvence)
        return None
    
    return kececi_prime_integer

# Constants for Keçeci Types (makes it easier to use from outside)
TYPE_POSITIVE_REAL = 1
TYPE_NEGATIVE_REAL = 2
TYPE_COMPLEX = 3
TYPE_FLOAT = 4
TYPE_RATIONAL = 5
TYPE_QUATERNION = 6

# --- DEFINITIONS (as a multiline string, can be accessed as kececinumbers.DEFINITIONS) ---
DEFINITIONS = """
Keçeci NumberS UNIFIED DEFINITION

A Keçeci Number sequence is derived from a `start_input_raw` and an `add_input_base_scalar`.
These inputs are interpreted according to the selected Keçeci Number Type to become the `current_value` and a type-specific `_add_value_typed`.

In each "Keçeci step" (which typically adds 2 or 3 numbers to the sequence):
1.  `added_value = current_value + _add_value_typed`.
2.  `added_value` is recorded in the sequence.
3.  A `result_value` is obtained by applying the Division Rule and, if necessary, the ASK Rule to `added_value`.
4.  `result_value` is recorded in the sequence.
5.  `current_value = result_value`.
This process is repeated for the specified `number_of_iterations` (Keçeci steps).

Division Rule:
*   A `last_divisor_used` (2 or 3) is tracked.
*   `primary_divisor`: If `last_divisor_used` is nonexistent (first step) or 2, it's 3; if it's 3, it's 2.
*   `alternative_divisor`: The other of `primary_divisor` (2 or 3).
*   `value_to_check_division`: This is `added_value` (or `modified_value` after ASK).
    *   The part of this value used for divisibility depends on the number type (e.g., real/scalar part for complex/quaternion, the fraction itself for rational).
    *   For Complex/Quaternion, all components should be divisible by the integer divisor for perfect division.
*   First, division by `primary_divisor` is attempted:
    *   If `value_to_check_division` (or its relevant part) is "perfectly divisible" by `primary_divisor` (in a type-specific sense, e.g., for rationals, the result is an integer),
        then `result_value = value_to_check_division / primary_divisor`. `last_divisor_used = primary_divisor`.
*   If unsuccessful, division by `alternative_divisor` is attempted:
    *   If `value_to_check_division` (or its relevant part) is "perfectly divisible" by `alternative_divisor`,
        then `result_value = value_to_check_division / alternative_divisor`. `last_divisor_used = alternative_divisor`.

Primality and ASK (Augment/Shrink then Check) Rule (if Division Fails):
*   `value_for_primality_check`: The part/representation of `added_value` used for primality testing (e.g., integer part of real, real part of complex).
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
    *   Division: Complex division (`/`). Perfect divisibility: Both real and imaginary parts are `math.isclose(part % integer_divisor, 0)`.
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
    *   Division: Quaternion division (`/`). Perfect divisibility: All w,x,y,z parts are `math.isclose(part % integer_divisor, 0)`.
    *   ASK unit: `quaternion(1,1,1,1)`. Primality: `abs(int(number.w))`.
"""


if __name__ == "__main__":
    print("Keçeci Numbers Module Loaded.")
    print("This module provides functions to generate and plot Keçeci Numbers.")
    print("Example: Use 'import kececinumbers as kn' in your script/notebook.")
    print("\nAvailable functions:")
    print("- kn.get_interactive()")
    print("- kn.get_with_params(kececi_type, iterations, ...)")
    print("- kn.get_random_type(iterations, ...)")
    print("- kn.plot_numbers(sequence, title)")
    print("- kn.find_kececi_prime_number(sequence)") # Explicitly list it
    print("- kn.unified_generator(...) (low-level)")
    print("\nAccess definitions by printing kn.DEFINITIONS") # Changed to instruction
    print("\nAccess type constants like: kn.TYPE_COMPLEX")

    print("\nRunning a quick test for Complex numbers (fixed params, 5 Keçeci steps):")
    # `iterations` in get_with_params is number of Keçeci steps
    test_seq = get_with_params(TYPE_COMPLEX, 5, start_value_raw="1+1j", add_value_base_scalar=2.0, fixed_params=True)
    # KPN will be printed by get_with_params

    # Example of calling find_kececi_prime_number directly if needed
    # if test_seq:
    #     kpn_direct = find_kececi_prime_number(test_seq)
    #     if kpn_direct is not None:
    #         print(f"Direct call to find_kececi_prime_number result: {kpn_direct}")
    #     else:
    #         print("Direct call: No Keçeci Prime Number found.")

    # print("\nRunning a quick test for Negative Integers (10 Keçeci steps):")
    # test_seq_neg = get_random_type(num_iterations=10) # KPN will be printed
