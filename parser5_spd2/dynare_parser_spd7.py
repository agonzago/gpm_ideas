import re
import sympy
import numpy as np
from collections import OrderedDict
import copy
import os
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

# --- IRF Function (Unchanged) ---
def irf(P, Q, shock_index, horizon=40):
    """
    Compute impulse responses for y_t = P y_{t-1} + Q e_t,
    for a specific shock index.

    Parameters
    ----------
    P : ndarray (n x n)
        Transition matrix.
    Q : ndarray (n x n_shock)
        Shock impact matrix.
    shock_index : int
        Index of the shock to simulate (0, 1, or 2).
    horizon : int, optional
        Number of periods for the IRF. The default is 40.

    Returns
    -------
    ndarray
        Array of shape (horizon, n) with responses over time.
    """
    n = P.shape[0]
    n_shock = Q.shape[1]
    if shock_index < 0 or shock_index >= n_shock:
        raise ValueError(f"shock_index must be between 0 and {n_shock-1}")

    y_resp = np.zeros((horizon, n))
    # Initial impulse: only one shock is non-zero at t=0
    e0 = np.zeros((n_shock, 1))
    e0[shock_index] = 1.0

    # y_0 = P * y_{-1} + Q * e_0. Assume y_{-1} = 0.
    y_current = Q @ e0

    y_resp[0, :] = y_current.flatten()

    # Subsequent periods: e_t = 0 for t > 0
    et = np.zeros((n_shock, 1))
    for t in range(1, horizon):
        y_current = P @ y_current # + Q @ et (which is zero)
        y_resp[t, :] = y_current.flatten()

    y_resp[np.abs(y_resp)< 1e-15]=0 # Set small values to zero1
    return y_resp

def solve_quadratic_matrix_equation(A, B, C, initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    # ... (initialization as before) ...
    n = A.shape[0]
    if initial_guess is None: initial_guess = np.zeros_like(A)

    E = C.copy(); F_init = A.copy(); Bbar = B.copy() # Rename initial F
    Bbar += A @ initial_guess
    try: lu_Bbar = lu_factor(Bbar)
    except (ValueError, np.linalg.LinAlgError): return None, 0, 1.0 # Factorization failed

    E = lu_solve(lu_Bbar, E)    # E = Bbar^-1 * C  (Note: This is -E0 from paper if Bbar=B)
    F = lu_solve(lu_Bbar, F_init) # F = Bbar^-1 * A  (Note: This is -Y0 and -F0 from paper if Bbar=B)

    # Initialize X, Y, Ek, Fk based on Algorithm 1
    # Note: Paper uses E0=-BinvC, F0=-BinvA, X0=-BinvC, Y0=-BinvA
    # Code calculates E = BinvC, F = BinvA
    # So, initialize matching Algorithm 1:
    Xk = -E - initial_guess  # Matches X0 if guess=0
    Yk = -F                  # Matches Y0 if guess=0
    Ek = -E                  # Matches E0 if guess=0
    Fk = -F                  # Matches F0 if guess=0


    X_new = np.zeros_like(Xk); Y_new = np.zeros_like(Yk)
    E_new = np.zeros_like(Ek); F_new = np.zeros_like(Fk)
    I = np.eye(n, dtype=A.dtype); solved = False; iter_count = 0; relative_diff = np.inf

    for i in range(1, max_iter + 1):
        iter_count = i
        # --- Use Xk, Yk, Ek, Fk representing current iterates ---
        try: lu_EI = lu_factor(I - Yk @ Xk)
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Factor EI iter {i}"); return Xk + initial_guess, i, np.inf
        try: lu_FI = lu_factor(I - Xk @ Yk)
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Factor FI iter {i}"); return Xk + initial_guess, i, np.inf

        # E_new = Ek @ (I - Yk @ Xk)^-1 @ Ek
        try: temp_E = lu_solve(lu_EI, Ek); E_new = Ek @ temp_E
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc E_new iter {i}"); return Xk + initial_guess, i, np.inf

        # F_new = Fk @ (I - Xk @ Yk)^-1 @ Fk
        try: temp_F = lu_solve(lu_FI, Fk); F_new = Fk @ temp_F
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc F_new iter {i}"); return Xk + initial_guess, i, np.inf

        # X_new = Xk + Fk @ (I - Xk @ Yk)^-1 @ Xk @ Ek
        try: temp_X = Xk @ Ek; temp_X = lu_solve(lu_FI, temp_X); X_new = Xk + Fk @ temp_X
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc X_new iter {i}"); return Xk + initial_guess, i, np.inf

        # --- CORRECTED Y_new calculation ---
        # Y_new = Yk + Ek @ (I - Yk @ Xk)^-1 @ Yk @ Fk
        try:
            temp_Y = Yk @ Fk # Use current Yk and Fk
            temp_Y = lu_solve(lu_EI, temp_Y)
            Y_new = Yk + Ek @ temp_Y
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc Y_new iter {i}"); return Xk + initial_guess, i, np.inf
        # --- End Correction ---

        # Convergence check (use relative change in X)
        X_diff_norm = norm(X_new - Xk, ord='fro')
        X_norm = norm(X_new, ord='fro')
        relative_diff = X_diff_norm / (X_norm + 1e-12)
        if verbose: print(f"Iter {i}: Rel Change X = {relative_diff:e}")
        if relative_diff < tol: solved = True; break

        # Update iterates for next loop
        Xk[:], Yk[:], Ek[:], Fk[:] = X_new, Y_new, E_new, F_new

    # Final solution incorporates initial guess
    X_sol = X_new + initial_guess if solved else Xk + initial_guess

    # Final residual check (optional but good)
    AX2 = A @ (X_sol @ X_sol); AX2_norm = norm(AX2, 'fro')
    residual = AX2 + B @ X_sol + C
    residual_ratio = norm(residual, 'fro') / (AX2_norm + 1e-15)
    if not solved: print(f"SDA Warning: No converge. Rel diff:{relative_diff:.2e}, Res ratio:{residual_ratio:.2e}")

    return X_sol, iter_count, residual_ratio

# --- Q Computation Function (Unchanged) ---
def compute_Q(A, B, D, P):
    """
    Once P satisfies A P^2 + B P + C=0, we can solve for Q in

    (A P + B)*Q + D = 0   =>   (A P + B)*Q = -D   =>   Q = -(A P + B)^{-1} D.

    This Q is such that  y_t = P y_{t-1} + Q e_t .
    For dimension n=2, D is typically 2x1 if there's 1 shock.
    """
    APB = A @ P + B
    try:
        # Use pseudo-inverse for potential robustness, or solve directly
        # invAPB = np.linalg.inv(APB)
        # Q = - invAPB @ D
        Q = np.linalg.solve(APB, -D) # More stable than inv()
    except np.linalg.LinAlgError:
        print("Cannot solve (A P + B)Q = -D. Matrix (A P + B) might be singular.")
        print(f"Condition number of (A P + B): {np.linalg.cond(APB)}")
        # Optionally return None or raise error
        return None
    return Q

# --- Symbol Creation (Unchanged) ---
def create_timed_symbol(base_name, time_shift):
    """Creates a sympy symbol with a time suffix."""
    if time_shift == -1:
        return sympy.symbols(f"{base_name}_m1")
    elif time_shift == 1:
        return sympy.symbols(f"{base_name}_p1")
    elif time_shift == 0:
        return sympy.symbols(base_name)
    else:
        # Handle multi-step leads/lags if needed directly by symbol name
        # This function might not be strictly needed if aux vars handle naming
        return sympy.symbols(f"{base_name}_t{time_shift:+}")

# --- FIXED Declaration Extractor ---
def extract_declarations(model_string):
    """
    Extracts variables, shocks, and parameters using regex, ensuring only
    declarations *before* the 'model;' block are considered.
    Handles comma/space separation and terminating semicolons robustly.
    Correctly extracts only parameter names, ignoring assignments.
    """
    declarations = {
        'var': [],
        'varexo': [],
        'parameters': []
    }

    # --- Pre-processing: Remove Comments ---
    processed_content = re.sub(
        r'/\*.*?\*/',
        '',
        model_string,
        flags=re.DOTALL
    )
    lines = processed_content.split('\n')
    cleaned_lines = [
        re.sub(r'(//|%).*$', '', line).strip()
        for line in lines
    ]
    processed_content = " \n ".join(cleaned_lines)

    # --- Find the 'model;' marker ---
    model_marker = re.search(r'\bmodel\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    if not model_marker:
        # If you require the model block, raise an error
        # raise ValueError("Could not find 'model;' marker in the definition.")
        # If declarations might exist without a model block (unlikely for Dynare):
        print("Warning: 'model;' marker not found. Processing all declarations found.")
        content_to_search = processed_content
    else:
        # Only search in the content *before* the model marker
        content_to_search = processed_content[:model_marker.start()]

    # --- Regex Extraction within the restricted content ---
    # Look for declaration keywords followed by content, until the *next* declaration keyword or end of the search section
    block_matches = re.finditer(
        r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters)\b|$)',
        content_to_search,
        re.DOTALL | re.IGNORECASE
    )

    def process_block_content(content_str, block_type):
        """Helper to clean and split names from block content string."""
        if not content_str:
            return []

        content = content_str.strip()
        # Find the first semicolon, process only content before it
        first_semicolon_match = re.search(r';', content)
        if first_semicolon_match:
            content = content[:first_semicolon_match.start()].strip()
        else:
            # If no semicolon found (e.g., last block), use the whole cleaned content
            pass # Content is already stripped

        # Replace internal newlines with spaces for splitting/regex
        content = content.replace('\n', ' ')

        names = []
        if block_type == 'parameters':
            # Capture identifiers not part of assignments (handles declarations like `param1, param2;`
            # and ignores later `param1 = value;`)
            # This regex finds identifiers potentially followed by '=', but only captures the identifier itself.
            # It relies on the block content ending before the assignments section.
            names = re.findall(r'\b([a-zA-Z_]\w*)\b', content)
             # Basic filter for common keywords that might slip through if syntax is very loose
            keywords = {'var', 'varexo', 'parameters', 'model', 'end'}
            names = [n for n in names if n not in keywords]

        else: # var, varexo
            # Split by comma or whitespace
            raw_names = re.split(r'[,\s]+', content)
            cleaned_names = [
                name.strip()
                for name in raw_names
                if name.strip() # Ensure not empty string
                   and re.fullmatch(r'[a-zA-Z_]\w*', name) # Check if it's a valid identifier
            ]
            names = cleaned_names

        # Remove duplicates, preserve order
        return list(dict.fromkeys(names).keys())

    for match in block_matches:
        block_keyword = match.group(1).lower()
        block_content_raw = match.group(2)
        extracted_names = process_block_content(block_content_raw, block_keyword)
        # Use extend to handle multiple blocks of the same type (though unusual)
        declarations[block_keyword].extend(extracted_names)

    # Final deduplication across potentially multiple blocks
    final_declarations = {}
    for key, lst in declarations.items():
        final_declarations[key] = list(dict.fromkeys(lst).keys())


    # --- Extract parameter assignments separately ---
    # Find parameters block again, including assignments section until 'model;'
    param_assignment_content = ""
    param_block_match = re.search(
        r'(?i)\bparameters\b(.*?)(?=\bmodel\b\s*;)',
        processed_content, # Search in the full preprocessed content
        re.DOTALL | re.IGNORECASE
    )
    if param_block_match:
        param_assignment_content = param_block_match.group(1)
        # Clean comments from this specific section again if needed, though outer cleaning might suffice
        # param_assignment_content = re.sub(r'(//|%).*$', '', param_assignment_content, flags=re.MULTILINE)


    # Extract assignments like name = value;
    assignments = {}
    assignment_matches = re.finditer(
        r'\b([a-zA-Z_]\w*)\b\s*=\s*([^;]+);',
        param_assignment_content
    )
    parameter_names_declared = final_declarations.get('parameters', [])
    for match in assignment_matches:
        name = match.group(1)
        value_str = match.group(2).strip()
        if name in parameter_names_declared: # Only store assignments for declared parameters
            try:
                # Attempt to convert to float, handle potential errors
                assignments[name] = float(value_str)
            except ValueError:
                print(f"Warning: Could not parse value '{value_str}' for parameter '{name}'. Skipping assignment.")
        else:
             print(f"Warning: Assignment found for '{name}', but it was not in the initial parameter declaration list. Ignoring.")


    return (
        final_declarations.get('var', []),
        final_declarations.get('varexo', []),
        parameter_names_declared, # Return the list of declared names
        assignments # Return the dictionary of assigned values
    )

# --- FIXED Model Equation Extractor ---
def extract_model_equations(model_string):
    """
    Extracts equations from the 'model; ... end;' block using a stricter regex.
    Handles the semicolon immediately following 'model'.
    """
    # Pre-processing: Remove comments first
    processed_content = re.sub(
        r'/\*.*?\*/',
        '',
        model_string,
        flags=re.DOTALL
    )
    lines = processed_content.split('\n')
    cleaned_lines = [
        re.sub(r'(//|%).*$', '', line).strip()
        for line in lines
    ]
    processed_content = " ".join(cleaned_lines) # Join with spaces to handle line breaks within equations

    # Stricter Regex: Looks for 'model;', captures non-greedily until 'end;'
    model_match = re.search(
        r'(?i)\bmodel\b\s*;(.*?)\bend\b\s*;', # Requires 'end;'
        processed_content,
        re.DOTALL | re.IGNORECASE # DOTALL for newline matching, IGNORECASE
    )
    if not model_match:
        raise ValueError(
            "Could not find 'model; ... end;' block."
        )

    model_content = model_match.group(1)
    # Split equations by semicolon, filter out empty strings
    equations_raw = [
        eq.strip()
        for eq in model_content.split(';')
        if eq.strip() # Ensure equation is not just whitespace
    ]

    # Process equations to be in 'LHS - RHS = 0' form
    processed_equations = []
    for line in equations_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts
                # Ensure correct formatting for sympy parsing: enclose in parentheses
                processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else:
                print(f"Warning: Skipping malformed equation line: '{line}'")
        else:
            # If no '=', assume it's already in 'expr = 0' form.
            print(f"Warning: Equation '{line}' has no '='. Assuming it's already in 'expr = 0' form.")
            processed_equations.append(line) # Append as is

    return processed_equations


# --- Main Parser Function (Integrates fixes) ---
def parse_lambdify_and_order_model(model_string):
    """
    Parses a model string with declarations, generates symbolic matrices,
    orders them, and returns lambdified functions. Incorporates fixes
    for declaration and model block extraction.
    """
    print("--- Parsing Model Declarations ---")
    # Now returns assignments as well
    declared_vars, shock_names, param_names, param_assignments = extract_declarations(model_string)

    if not declared_vars:
        raise ValueError("No variables declared in 'var' block.")
    if not shock_names:
        raise ValueError("No shocks declared in 'varexo' block.")
    if not param_names:
        raise ValueError("No parameters declared in 'parameters' block.")


    print(f"Declared Variables: {declared_vars}")
    print(f"Declared Shocks: {shock_names}")
    print(f"Declared Parameters: {param_names}")
    print(f"Parsed Parameter Assignments: {param_assignments}")


    print("\n--- Parsing Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    print(f"Found {len(raw_equations)} equations in model block.")
    # for i, eq in enumerate(raw_equations):
    #     print(f"  Eq {i}: {eq}")

    # --- Handling Leads/Lags & Auxiliaries ---
    print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars) # Start with declared vars
    aux_variables = OrderedDict() # Store definition string for each aux var
    processed_equations = list(raw_equations) # Copy to modify
    # Regex to find var(time_shift)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]
        eq_idx += 1 # Increment here, so adding to list doesn't break index
        modified_eq = eq
        matches = list(var_time_regex.finditer(eq))

        # Process matches in reverse to avoid index issues during replacement
        for match in reversed(matches):
            base_name = match.group(1)
            time_shift = int(match.group(2))

            # Check if the base name is a known endogenous variable (original or already auxiliary)
            # Or if it's a parameter or shock (which shouldn't have time shifts applied typically)
            if base_name not in endogenous_vars and base_name not in aux_variables:
                if base_name in param_names or base_name in shock_names:
                    continue # Skip parameters/shocks
                # If it's an unknown symbol, we might warn or ignore
                # print(f"Warning: Found time shift for undeclared symbol '{base_name}' in eq: {eq}. Ignoring.")
                continue

            # --- Handle Leads > 1 ---
            if time_shift > 1:
                aux_needed_defs = [] # Track definitions needed for this specific lead
                target_aux_for_replacement = base_name # Start with base name for +1 lead
                if target_aux_for_replacement not in endogenous_vars:
                    target_aux_for_replacement = f"aux_{base_name}_lead_p{time_shift-1}"

                # Create auxiliary variables from +1 up to time_shift-1
                for k in range(1, time_shift): # Need aux vars up to k = time_shift - 1
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        # Define aux_k in terms of aux_{k-1} or base_name
                        if k == 1:
                            prev_var_for_def = base_name
                        else:
                            prev_var_for_def = f"aux_{base_name}_lead_p{k-1}"

                        # Definition: aux_k(t) = prev_var(t+1)  =>  aux_k - prev_var(+1) = 0
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(+1)"
                        aux_variables[aux_name] = def_eq_str # Store definition
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name) # Add aux var to the list

                # The variable replacing the original var(time_shift) expression
                # is the aux var for lead k=time_shift-1, shifted forward by one period (+1)
                # Example: y(+3) -> needs aux_y_p1, aux_y_p2. Replace y(+3) with aux_y_p2(+1)
                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                # print(f"  Replaced {match.group(0)} with {replacement}")

                # Add the necessary definition equations to the processed list if not already there
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        print(f"  Adding aux def: {def_eq} = 0")
                        processed_equations.append(def_eq)

            # --- Handle Lags < -1 ---
            elif time_shift < -1:
                aux_needed_defs = []
                # Create auxiliary variables from -1 down to time_shift+1
                for k in range(1, abs(time_shift)): # Need aux vars up to k = abs(time_shift) - 1
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        # Define aux_k in terms of aux_{k-1} or base_name
                        if k == 1:
                            prev_var_for_def = base_name
                        else:
                            prev_var_for_def = f"aux_{base_name}_lag_m{k-1}"

                        # Definition: aux_k(t) = prev_var(t-1)  =>  aux_k - prev_var(-1) = 0
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(-1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                # The variable replacing the original var(time_shift) expression
                # is the aux var for lag k = abs(time_shift) - 1, shifted backward by one period (-1)
                # Example: y(-3) -> needs aux_y_m1, aux_y_m2. Replace y(-3) with aux_y_m2(-1)
                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                # print(f"  Replaced {match.group(0)} with {replacement}")


                # Add the necessary definition equations
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        print(f"  Adding aux def: {def_eq} = 0")
                        processed_equations.append(def_eq)

        # Update the equation in the list if it was modified
        if modified_eq != eq:
            processed_equations[eq_idx - 1] = modified_eq
            # print(f"  Updated Eq {eq_idx-1}: {modified_eq}")


    # --- Final Variable/Equation Counts ---
    initial_vars_ordered = list(endogenous_vars) # Final list including auxiliaries
    num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations)
    num_shocks = len(shock_names)

    print(f"\nTotal variables after processing leads/lags ({num_vars}): {initial_vars_ordered}")
    print(f"Total equations after processing leads/lags ({num_eq}):")
    # for i, eq in enumerate(processed_equations):
    #      print(f"  Eq {i}: {eq}")

    if num_vars != num_eq:
        # Provide more context in the error message
        print("\nError Details:")
        print(f"  Original Declared Vars: {declared_vars}")
        print(f"  Auxiliary Vars Added: {list(aux_variables.keys())}")
        print(f"  Final Variable List ({num_vars}): {initial_vars_ordered}")
        print(f"\n  Original Equations: {raw_equations}")
        print(f"  Auxiliary Equations Added: {list(aux_variables.values())}")
        print(f"  Final Equation List ({num_eq}): {processed_equations}")
        raise ValueError(
            f"Model not square after processing leads/lags: {num_vars} vars vs {num_eq} eqs."
        )

    print("\nModel is square.")

    # --- Symbolic Representation ---
    print("\n--- Creating Symbolic Representation ---")
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names} # Shocks are time t
    var_syms = {} # Stores {'var': {'m1': sym, 't': sym, 'p1': sym}}

    # Create symbols for y(t-1), y(t), y(t+1) for all endogenous variables
    all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in initial_vars_ordered:
        sym_m1 = create_timed_symbol(var, -1)
        sym_t  = create_timed_symbol(var, 0) # Use base name for time t
        sym_p1 = create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
        all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])

    # Create a local dictionary for sympy's parse_expr
    local_dict = {str(s): s for s in all_syms_for_parsing}
    # Add common math functions if they might appear (like log, exp)
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})


    # Use sympy's recommended parser setup
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, rationalize)
    transformations = (standard_transformations +
                       (implicit_multiplication_application, rationalize))

    sym_equations = []
    print("Parsing equations into symbolic form...")
    for i, eq_str in enumerate(processed_equations):
        # Substitute var(time) notation with corresponding _m1, _t, _p1 symbols
        eq_str_sym = eq_str

        def replace_var_time(match):
            """Callback function for re.sub"""
            base_name = match.group(1)
            time_shift = int(match.group(2))

            if base_name in shock_names:
                if time_shift == 0:
                    return str(shock_syms[base_name]) # Shocks are e_t
                else:
                     raise ValueError(f"Shocks like {base_name} should only appear at time t (0), not ({time_shift}) in equation: {eq_str}")
            elif base_name in var_syms:
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0:  return str(var_syms[base_name]['t'])
                if time_shift == 1:  return str(var_syms[base_name]['p1'])
                # If shifts > 1 or < -1 remain, it means aux var logic failed
                raise ValueError(f"Unexpected time shift {time_shift} for {base_name} remaining after auxiliary var processing in eq: {eq_str}")
            elif base_name in param_syms:
                # Parameters shouldn't have time shifts
                 raise ValueError(f"Parameter {base_name} cannot have time shift ({time_shift}) in equation: {eq_str}")
            elif base_name in local_dict: # e.g. log, exp
                 return match.group(0) # Keep function call syntax like log(x)
            else:
                # This case should ideally not be reached if all vars/params are declared
                print(f"Warning: Symbol '{base_name}' with time shift ({time_shift}) in eq {i} ('{eq_str}') is not a declared var, shock, or parameter. Treating as symbolic.")
                # Ensure the symbol exists in the local_dict if encountered first time here
                if base_name not in local_dict:
                     local_dict[base_name] = sympy.symbols(base_name)
                # Attempt to create timed symbol, though it might not be used correctly later
                timed_sym_str = str(create_timed_symbol(base_name, time_shift))
                if timed_sym_str not in local_dict:
                     local_dict[timed_sym_str] = sympy.symbols(timed_sym_str)
                return timed_sym_str


        # Apply the replacement for var(time) patterns
        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)

        # Replace remaining base variable names (implicitly time t) with their 't' symbol string
        # Sort by length descending to replace longer names first (e.g., RES_X before X)
        all_known_base_names = sorted(list(var_syms.keys()) + param_names + shock_names, key=len, reverse=True)
        for name in all_known_base_names:
             # Use word boundaries to avoid partial replacements (e.g., replacing 'R' in 'RR_GAP')
             pattern = r'\b' + re.escape(name) + r'\b'
             if name in var_syms:
                 replacement = str(var_syms[name]['t']) # Base name implies time t
             elif name in param_syms:
                 replacement = str(param_syms[name])
             elif name in shock_syms:
                 replacement = str(shock_syms[name]) # Shock implies time t
             else:
                 continue # Should not happen if all_known_base_names is correct

             eq_str_sym = re.sub(pattern, replacement, eq_str_sym)

        # print(f"  Processed Eq {i}: {eq_str_sym}") # Debug output

        try:
            # Check for any remaining undeclared symbols before parsing
            current_symbols = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq_str_sym))
            unknown_symbols = current_symbols - set(local_dict.keys()) - {'log', 'exp', 'sqrt', 'abs'} # Exclude known functions
             # Filter out numbers
            unknown_symbols = {s for s in unknown_symbols if not re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)}


            if unknown_symbols:
                print(f"Warning: Potential undeclared symbols found in eq {i} ('{eq_str_sym}'): {unknown_symbols}. Adding them to local_dict.")
                for sym_str in unknown_symbols:
                    if sym_str not in local_dict:
                         local_dict[sym_str] = sympy.symbols(sym_str)

            # Parse the fully substituted string
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            sym_equations.append(sym_eq)
            # print(f"    Sympy Eq {i}: {sym_eq}") # Debug output
        except Exception as e:
            print(f"\nError parsing equation {i}:")
            print(f"  Original: '{eq_str}'")
            print(f"  Substituted: '{eq_str_sym}'")
            print(f"  Local dict keys: {sorted(local_dict.keys())}")
            print(f"  Sympy error: {e}")
            raise # Re-raise the exception to stop execution

    print("Symbolic parsing completed.")

    # --- Generate Initial Symbolic Matrices ---
    print("\n--- Generating Initial Symbolic Matrices (A, B, C, D) ---")
    # Dimensions: num_eq x num_vars for A, B, C; num_eq x num_shocks for D
    sympy_A = sympy.zeros(num_eq, num_vars) # Coefficients of y(t-1)
    sympy_B = sympy.zeros(num_eq, num_vars) # Coefficients of y(t)
    sympy_C = sympy.zeros(num_eq, num_vars) # Coefficients of y(t+1)
    sympy_D = sympy.zeros(num_eq, num_shocks) # Coefficients of e(t)

    for i, eq in enumerate(sym_equations):
        # Use .expand() to ensure terms are separated, e.g., (a+b)*x -> a*x + b*x
        # Use .expand().coeff() which is generally safer than .diff() for linear coefficients
        eq_expanded = sympy.expand(eq)

        # Get coefficients for y(t-1) terms -> Matrix A
        for j, var in enumerate(initial_vars_ordered):
            sympy_A[i, j] = eq_expanded.coeff(var_syms[var]['m1'])

        # Get coefficients for y(t) terms -> Matrix B
        for j, var in enumerate(initial_vars_ordered):
            sympy_B[i, j] = eq_expanded.coeff(var_syms[var]['t'])

        # Get coefficients for y(t+1) terms -> Matrix C
        for j, var in enumerate(initial_vars_ordered):
            sympy_C[i, j] = eq_expanded.coeff(var_syms[var]['p1'])

        # Get coefficients for e(t) shock terms -> Matrix D
        for k, shk in enumerate(shock_names):
            sympy_D[i, k] = eq_expanded.coeff(shock_syms[shk])

        # Optional: Check for remaining constant terms (should be zero if model is linearized correctly)
        # constant_term = eq_expanded
        # for sym in all_syms_for_parsing:
        #     constant_term = constant_term.subs(sym, 0)
        # if not constant_term.is_zero:
        #     print(f"Warning: Non-zero constant term '{constant_term}' found in equation {i} after extracting coefficients.")


    initial_info = {
        'A': copy.deepcopy(sympy_A), 'B': copy.deepcopy(sympy_B),
        'C': copy.deepcopy(sympy_C), 'D': copy.deepcopy(sympy_D),
        'vars': list(initial_vars_ordered), 'eqs': list(processed_equations)
    }
    print("Symbolic matrices A, B, C, D generated.")
    # print("Initial Symbolic A:\n", sympy_A) # Optional: Print for debugging


    # --- Classify Variables (Simpler Classification for Ordering) ---
    print("\n--- Classifying Variables for Ordering ---")
    # Based on Dynare's typical ordering: backward, both, forward, static
    # This simplified version: (Exo + Aux Lags), (Forward/Backward Endo), (Static Endo)

    # Identify exogenous processes (like AR processes defined explicitly)
    # Heuristic: Look for equations defining RES_ variables
    exo_process_vars = [] # Vars defined by AR-like processes
    exo_defining_eq_indices = {} # Map var -> eq_index

    # Identify potential candidates based on name prefix
    potential_exo_vars = [v for v in initial_vars_ordered if v.startswith("RES_")]
    potential_aux_lag_vars = [v for v in initial_vars_ordered if v.startswith("aux_") and "_lag_" in v]

    # Simple check for defining equations (e.g., var(t) = rho*var(t-1) + shock)
    # More robust check would analyze matrix structure
    for var in potential_exo_vars:
         j = initial_vars_ordered.index(var)
         for i in range(num_eq):
             # Check if eq i looks like: coeff*var(t) + coeff*var(t-1) + coeff*shock = 0
             # Simplified check: Does B[i,j] seem dominant and non-zero?
             # Is C[i,:] zero for this row?
             # Is A[i,j] non-zero?
             # Are other B[i,k] zero?
             # This is complex, relying on simple naming convention for now.
             # A better way: Check if the *original* equations define them simply.
             original_eq_index = -1
             for idx, eq_str in enumerate(processed_equations):
                 # Check if equation is the defining equation for 'var' or its aux lags
                 # e.g., "RES_X - rho*RES_X(-1) - SHK_X" or "aux_RESX_m1 - RES_X(-1)"
                 if var in eq_str and f"{var}(-1)" in eq_str or f"{var} =" in eq_str:
                     original_eq_index = idx
                     break
             if original_eq_index != -1:
                  is_defining = True # Assume for now
                  # Basic check: is var(t+1) absent from this equation?
                  if not sympy_C.row(i).is_zero_matrix:
                       is_defining = False
                  # Basic check: is var(t) present?
                  if sympy_B[i, j].is_zero:
                       is_defining = False

                  if is_defining:
                      if var not in exo_process_vars: exo_process_vars.append(var)
                      if var not in exo_defining_eq_indices: exo_defining_eq_indices[var] = i # Store first match
                      # Let's also add associated aux lag vars
                      for aux_var in potential_aux_lag_vars:
                          if base_name_from_aux(aux_var) == var:
                              if aux_var not in exo_process_vars: exo_process_vars.append(aux_var)
                              # Try to find the aux var's defining equation too
                              aux_j = initial_vars_ordered.index(aux_var)
                              for aux_i in range(num_eq):
                                   # aux_var(t) = base_var(t-1) or prev_aux(t-1)
                                   if not sympy_C.row(aux_i).is_zero_matrix: continue
                                   if sympy_B[aux_i, aux_j].is_zero: continue # Needs aux(t) coeff
                                   # Check if A has the expected lagged term coeff
                                   # This gets complex quickly. Rely on simple grouping for now.
                                   if aux_var not in exo_defining_eq_indices:
                                       # Simple association: map aux var to its base var's eq for sorting? No, find its *own* def eq.
                                       for eq_k, eq_k_str in enumerate(processed_equations):
                                           if eq_k_str.startswith(f"{aux_var} -"):
                                                exo_defining_eq_indices[aux_var] = eq_k
                                                break


    # All variables involved in exogenous AR processes or their lags
    backward_exo_vars = sorted([v for v in initial_vars_ordered if v in exo_process_vars or (v.startswith("aux_") and "_lag_" in v and base_name_from_aux(v) in exo_process_vars)], key=initial_vars_ordered.index)

    # Remaining variables classify based on presence in A and C columns
    forward_backward_endo_vars = []
    static_endo_vars = []

    remaining_vars = [v for v in initial_vars_ordered if v not in backward_exo_vars]

    for var in remaining_vars:
        j = initial_vars_ordered.index(var)
        has_lag = not sympy_A.col(j).is_zero_matrix
        has_lead = not sympy_C.col(j).is_zero_matrix

        if has_lag or has_lead:
            forward_backward_endo_vars.append(var)
        else:
            static_endo_vars.append(var) # Only appears at time t

    print("\nCategorized Variables:")
    print(f"  Backward/Exo Group: {backward_exo_vars}")
    print(f"  Forward/Backward Endo: {forward_backward_endo_vars}")
    print(f"  Static Endo: {static_endo_vars}")

    # --- Determine New Variable Order ---
    # Order: Backward/Exo -> Forward/Backward -> Static
    ordered_vars = backward_exo_vars + forward_backward_endo_vars + static_endo_vars

    if len(ordered_vars) != len(initial_vars_ordered):
        raise ValueError(f"Variable reordering failed. Length mismatch: {len(ordered_vars)} vs {len(initial_vars_ordered)}")
    if set(ordered_vars) != set(initial_vars_ordered):
         raise ValueError(f"Variable reordering failed. Content mismatch: {set(initial_vars_ordered) - set(ordered_vars)} missing, {set(ordered_vars) - set(initial_vars_ordered)} extra.")

    # Create permutation index map: old_index -> new_index
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]
    print(f"\nNew Variable Order ({len(ordered_vars)}): {ordered_vars}")

    # --- Determine New Equation Order ---
    # Try to match equations to variables based on category and index
    # Order: Equations for Backward/Exo -> Equations for Fwd/Bwd -> Equations for Static
    eq_perm_indices = []
    used_eq_indices = set()

    # Helper to find an unused equation index
    def find_unused_eq(used_set, total_eqs):
        for i in range(total_eqs):
            if i not in used_set:
                return i
        return -1 # Should not happen if counts match

    # 1. Assign equations for backward/exo variables (prioritize defining equations)
    for var in backward_exo_vars:
        assigned_eq = -1
        if var in exo_defining_eq_indices and exo_defining_eq_indices[var] not in used_eq_indices:
            assigned_eq = exo_defining_eq_indices[var]
        else:
            # Fallback: find first unused equation (less ideal)
            fallback_eq = find_unused_eq(used_eq_indices, num_eq)
            if fallback_eq != -1:
                assigned_eq = fallback_eq
                print(f"Warning: Assigning fallback eq {fallback_eq} to backward/exo var '{var}'")
            else:
                 raise RuntimeError(f"Cannot find unused equation for backward/exo var '{var}'")

        if assigned_eq != -1:
            eq_perm_indices.append(assigned_eq)
            used_eq_indices.add(assigned_eq)

    # 2. Assign equations for forward/backward endogenous variables
    # Heuristic: Try to find equations where these variables appear prominently
    # Simple approach: Assign remaining unused equations in order
    num_bw_eq = len(eq_perm_indices)
    for i in range(len(forward_backward_endo_vars)):
        assigned_eq = find_unused_eq(used_eq_indices, num_eq)
        if assigned_eq != -1:
            eq_perm_indices.append(assigned_eq)
            used_eq_indices.add(assigned_eq)
        else:
            raise RuntimeError(f"Cannot find unused equation for fwd/bwd var index {i}")


    # 3. Assign equations for static endogenous variables
    for i in range(len(static_endo_vars)):
         assigned_eq = find_unused_eq(used_eq_indices, num_eq)
         if assigned_eq != -1:
             eq_perm_indices.append(assigned_eq)
             used_eq_indices.add(assigned_eq)
         else:
             raise RuntimeError(f"Cannot find unused equation for static var index {i}")


    if len(eq_perm_indices) != num_eq:
        raise ValueError(f"Equation permutation construction failed. Length mismatch: {len(eq_perm_indices)} vs {num_eq}")
    if len(used_eq_indices) != num_eq:
         raise ValueError(f"Equation permutation construction failed. Not all equations used/assigned uniquely.")

    print(f"\nEquation permutation indices (new row i <- old row eq_perm_indices[i]): {eq_perm_indices}")

    # --- Reorder Symbolic Matrices ---
    print("\n--- Reordering Symbolic Matrices ---")
    # Use extract to reorder rows (equations) and columns (variables)
    sympy_A_ord = sympy_A.extract(eq_perm_indices, var_perm_indices)
    sympy_B_ord = sympy_B.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C.extract(eq_perm_indices, var_perm_indices)
    # D only needs rows reordered (shocks order is fixed)
    sympy_D_ord = sympy_D.extract(eq_perm_indices, list(range(num_shocks)))

    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}
    print("Symbolic reordering complete.")
    # print("Ordered Symbolic B:\n", sympy_B_ord) # Optional debug print

    # --- Lambdify ---
    print("\n--- Lambdifying Ordered Matrices ---")
    # Parameters are the input arguments for the functions
    param_sym_list = [param_syms[p] for p in param_names]
    try:
        # Use 'numpy' module for numerical evaluation
        func_A = sympy.lambdify(param_sym_list, sympy_A_ord, modules='numpy')
        func_B = sympy.lambdify(param_sym_list, sympy_B_ord, modules='numpy')
        func_C = sympy.lambdify(param_sym_list, sympy_C_ord, modules='numpy')
        func_D = sympy.lambdify(param_sym_list, sympy_D_ord, modules='numpy')
        print("Lambdification successful.")
    except Exception as e:
        print(f"Error during lambdification: {e}")
        # Print details about the matrix that failed if possible
        if 'func_A' not in locals(): print("Failed on matrix A")
        elif 'func_B' not in locals(): print("Failed on matrix B")
        elif 'func_C' not in locals(): print("Failed on matrix C")
        elif 'func_D' not in locals(): print("Failed on matrix D")
        raise

    return (func_A, func_B, func_C, func_D,
            ordered_vars, shock_names, param_names, param_assignments,
            symbolic_matrices_ordered, initial_info)


def base_name_from_aux(aux_name):
    """Helper to extract base variable name from auxiliary variable name"""
    match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p\d+", aux_name)
    if match_lead:
        return match_lead.group(1)
    match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", aux_name)
    if match_lag:
        return match_lag.group(1)
    return None # Or return original name if not matching aux pattern


# --- Example usage ---
if __name__ == "__main__":
    try:
        # Assume the script is in the same directory as the .dyn file
        # or provide the full path.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_path = os.path.join(script_dir, "qpm_simpl1.dyn")

        # Change CWD (optional, might be needed if the .dyn file uses relative paths internally)
        # os.chdir(script_dir)
        # print(f"Current Working Directory set to: {os.getcwd()}")


        if not os.path.exists(mod_file_path):
             raise FileNotFoundError(f"Model file not found at: {mod_file_path}")

        print(f"Reading model definition from: {mod_file_path}")
        with open(mod_file_path, 'r') as f:
            model_def = f.read()

        # --- PARSE ---
        (func_A, func_B, func_C, func_D,
         ordered_vars, shocks, param_names, param_assignments,
         sym_matrices_ord, initial_info) = parse_lambdify_and_order_model(model_def)

        print("\n\n--- Parser Results ---")
        print("Parameter Names (for function arguments):", param_names)
        print("Parameter Assignments Found:", param_assignments)
        print("Shock Names:", shocks)
        print("Final Ordered Variables:", ordered_vars)

        # --- EVALUATE ---
        # Use parameter values found in the file, otherwise provide defaults
        test_param_values = param_assignments.copy() # Start with parsed values
        # Add defaults for any parameters declared but not assigned in the file
        default_test_values = {
            'b1': 0.7, 
            'b4': 0.7, 
            'a1': 0.5, 
            'a2': 0.1, 
            'g1': 0.7, 
            'g2': 0.3,
            'g3': 0.25, 
            'rho_L_GDP_GAP': 0.75, 
            'rho_DLA_CPI': 0.75,
            'rho_rs': 0.75, 
            'rho_rs2': 0.1 # Using 0.1 based on original example
        }
        missing_params = []
        test_args = []
        print("\n--- Evaluating Matrices with Parameters ---")
        for p in param_names:
            if p in test_param_values:
                test_args.append(test_param_values[p])
                # print(f"  Using value for {p}: {test_param_values[p]}")
            elif p in default_test_values:
                test_args.append(default_test_values[p])
                print(f"  Warning: Param '{p}' not assigned in file. Using default value: {default_test_values[p]}.")
            else:
                missing_params.append(p)
                print(f"  ERROR: Param '{p}' declared but no value assigned in file or defaults. Using 0.0.")
                test_args.append(0.0) # Fallback, likely problematic

        if not missing_params:
            print("\nEvaluating numerical matrices A, B, C, D...")
            A_num = func_A(*test_args)
            B_num = func_B(*test_args)
            C_num = func_C(*test_args)
            D_num = func_D(*test_args)

            # Optional: Print numerical matrices
            with np.printoptions(precision=3, suppress=True, linewidth=120):
                 # print("\nNumerical A (ordered):\n", A_num)
                 print("\nNumerical B (ordered):\n", B_num) # B is often informative
                 # print("\nNumerical C (ordered):\n", C_num)
                 # print("\nNumerical D (ordered):\n", D_num)

            # --- SOLVE ---
            print("\n--- Solving Quadratic Matrix Equation: A P^2 + B P + C = 0 ---")
            P_sol, iter_count, residual_ratio = solve_quadratic_matrix_equation(A_num, B_num, C_num, tol=1e-12, verbose=False)

            if P_sol is None:
                print("\nERROR: Quadratic solver failed to compute P.")
            else:
                print(f"Solver iterations: {iter_count}, final residual ratio: {residual_ratio:.2e}")
                if residual_ratio > 1e-6:
                    print("Warning: Solver residual ratio is high, solution P might be inaccurate.")

                # Ensure P is stable (eigenvalues < 1 in magnitude)
                try:
                    eigenvalues = np.linalg.eigvals(P_sol)
                    max_eig = np.max(np.abs(eigenvalues))
                    print(f"Maximum eigenvalue magnitude of P: {max_eig:.6f}")
                    if max_eig >= 1.0 - 1e-9: # Allow for slight numerical inaccuracy
                        print("Warning: Solution P might be unstable or borderline stable (max eigenvalue >= 1).")
                    else:
                        print("Solution P appears stable.")
                except np.linalg.LinAlgError:
                    print("Warning: Could not compute eigenvalues of P.")


                # --- Compute Q ---
                print("\n--- Computing Q matrix ---")
                Q_sol = compute_Q(A_num, B_num, D_num, P_sol)

                if Q_sol is None:
                    print("ERROR: Failed to compute Q matrix. (A P + B) may be singular.")
                else:
                    print("Q matrix computed successfully.")
                    # print("Numerical Q (ordered):\n", Q_sol) # Optional print

                    # --- COMPUTE IRFs ---
                    print("\n--- Computing Impulse Response Functions ---")
                    # Choose shock index (0=GDP_GAP, 1=DLA_CPI, 2=RS)
                    shock_index_to_plot = 2 # Example: Monetary policy shock (SHK_RS)
                    horizon = 40
                    print(f"Computing IRFs for shock: {shocks[shock_index_to_plot]} (index {shock_index_to_plot})")

                    irf_vals = irf(P_sol, Q_sol, shock_index=shock_index_to_plot, horizon=horizon)

                    # --- PLOT IRFs ---
                    print(f"\n--- Plotting IRFs for selected variables ---")
                    # Select subset of variables to plot
                    # Use variables from the *ordered* list
                    vars_to_plot_names = ["L_GDP_GAP", "DLA_CPI", "RS", "RR_GAP", "RES_RS", "RES_L_GDP_GAP", "RES_DLA_CPI"]
                    # Ensure the variables exist in the final ordered list
                    vars_to_plot_indices = []
                    valid_vars_to_plot = []
                    for v_name in vars_to_plot_names:
                        if v_name in ordered_vars:
                            vars_to_plot_indices.append(ordered_vars.index(v_name))
                            valid_vars_to_plot.append(v_name)
                        else:
                            print(f"Warning: Variable '{v_name}' not found in ordered list, cannot plot.")


                    if valid_vars_to_plot:
                        plt.figure(figsize=(12, 8))
                        plt.suptitle(f"Impulse Responses to a Unit {shocks[shock_index_to_plot]} Shock", fontsize=14)

                        num_plots = len(valid_vars_to_plot)
                        # Adjust layout dynamically
                        cols = 3 if num_plots > 4 else 2
                        rows = (num_plots + cols - 1) // cols

                        for i, var_name in enumerate(valid_vars_to_plot):
                            idx = vars_to_plot_indices[i]
                            plt.subplot(rows, cols, i + 1)
                            plt.plot(range(horizon), irf_vals[:, idx], label=f'{var_name}')
                            plt.axhline(0, color='black', linewidth=0.7, linestyle=':')
                            plt.title(f"{var_name}")
                            # plt.xlabel("Quarters") # Add only to bottom plots if desired
                            # plt.ylabel("Response")
                            plt.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

                        # Add common labels if needed
                        # fig = plt.gcf()
                        # fig.text(0.5, 0.01, 'Quarters', ha='center', va='center')
                        # fig.text(0.02, 0.5, 'Response', ha='center', va='center', rotation='vertical')

                        plt.show()
                    else:
                        print("No valid variables selected for plotting.")

        else:
            print("\nSkipping numerical evaluation, solving, and plotting due to missing parameter values.")
            print(f"Missing parameters: {missing_params}")

    except FileNotFoundError as e:
         print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nAn error occurred during parsing or validation: {e}")
        import traceback
        traceback.print_exc()
    except np.linalg.LinAlgError as e:
         print(f"\nA linear algebra error occurred (possibly singular matrix): {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()