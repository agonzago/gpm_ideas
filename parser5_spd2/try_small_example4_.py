# --- (Keep previous functions unchanged: irf, solvers, Q, symbol creation, extraction) ---
    """
    The function `parse_lambdify_and_order_model` parses a model definition, orders variables and
    equations, creates symbolic representations, and lambdifies matrices for numerical evaluation, with
    an example usage demonstrating solving, computing Q, IRFs, and plotting.
    
    :param P: The parameter matrix P is a solution to a quadratic matrix equation. It is computed based
    on the matrices A, B, and C using the `solve_quadratic_matrix_equation` function. The matrix P
    represents the solution to the quadratic equation and is used in subsequent calculations such as
    computing the Q matrix
    :param Q: Q is a matrix that represents the impact of structural shocks on the endogenous variables
    in a model. It is computed using the `compute_Q` function, which takes as input the matrices A, B,
    D, and P. The matrix Q is calculated as the solution to the equation AP + B
    :param shock_index: The `shock_index` parameter is the index of the shock for which you want to
    compute Impulse Response Functions (IRFs). It corresponds to the position of the shock in the list
    of shock names declared in the model. The shock index is used to identify which shock's IRFs to
    compute
    :param horizon: The `horizon` parameter specifies the number of periods into the future for which
    you want to compute impulse response functions (IRFs) after a shock. It determines the length of the
    IRF time series, defaults to 40 (optional)
    :return: The code is returning a tuple containing the following elements:
    1. `func_A`: A lambda function representing the numerical evaluation of the symbolic matrix A.
    2. `func_B`: A lambda function representing the numerical evaluation of the symbolic matrix B.
    3. `func_C`: A lambda function representing the numerical evaluation of the symbolic matrix C.
    4. `func_D`: A lambda function representing the numerical
    """
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
    n = P.shape[0]; n_shock = Q.shape[1]
    if not 0 <= shock_index < n_shock: raise ValueError(f"shock_index {shock_index} out of bounds for {n_shock} shocks")
    y_resp = np.zeros((horizon, n)); e0 = np.zeros((n_shock, 1)); e0[shock_index] = 1.0
    y_current = Q @ e0; y_resp[0, :] = y_current.flatten()
    for t in range(1, horizon): y_current = P @ y_current; y_resp[t, :] = y_current.flatten()
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
    APB = A @ P + B
    try: Q = np.linalg.solve(APB, -D)
    except np.linalg.LinAlgError: print(f"Cannot solve Q. Cond(APB):{np.linalg.cond(APB)}"); return None
    return Q

# --- Symbol Creation (Unchanged) ---
def create_timed_symbol(base_name, time_shift):
    if time_shift == -1: return sympy.symbols(f"{base_name}_m1")
    if time_shift == 1: return sympy.symbols(f"{base_name}_p1")
    if time_shift == 0: return sympy.symbols(base_name)
    return sympy.symbols(f"{base_name}_t{time_shift:+}")

# --- Declaration Extractor (Unchanged) ---
def extract_declarations(model_string):
    declarations = {'var': [], 'varexo': [], 'parameters': []}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    model_marker = re.search(r'\bmodel\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    content_to_search = processed_content[:model_marker.start()] if model_marker else processed_content
    if not model_marker: print("Warning: 'model;' marker not found.")

    block_matches = re.finditer(
        r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters)\b|$)',
        content_to_search, re.DOTALL | re.IGNORECASE)

    def process_block_content(content_str, block_type):
        content = content_str.strip()
        first_semicolon_match = re.search(r';', content)
        if first_semicolon_match: content = content[:first_semicolon_match.start()].strip()
        content = content.replace('\n', ' ')
        names = []
        if block_type == 'parameters':
            names = re.findall(r'\b([a-zA-Z_]\w*)\b', content)
            keywords = {'var', 'varexo', 'parameters', 'model', 'end'}
            names = [n for n in names if n not in keywords]
        else:
            raw_names = re.split(r'[,\s]+', content)
            names = [name.strip() for name in raw_names if name.strip() and re.fullmatch(r'[a-zA-Z_]\w*', name)]
        return list(dict.fromkeys(names).keys())

    for match in block_matches:
        block_keyword = match.group(1).lower()
        extracted_names = process_block_content(match.group(2), block_keyword)
        declarations[block_keyword].extend(extracted_names)

    final_declarations = {key: list(dict.fromkeys(lst).keys()) for key, lst in declarations.items()}

    param_assignment_content = ""
    param_block_match = re.search(
        r'(?i)\bparameters\b(.*?)(?=\bmodel\b\s*;)', processed_content, re.DOTALL | re.IGNORECASE)
    if param_block_match: param_assignment_content = param_block_match.group(1)

    assignments = {}
    assignment_matches = re.finditer(r'\b([a-zA-Z_]\w*)\b\s*=\s*([^;]+);', param_assignment_content)
    parameter_names_declared = final_declarations.get('parameters', [])
    for match in assignment_matches:
        name = match.group(1); value_str = match.group(2).strip()
        if name in parameter_names_declared:
            try: assignments[name] = float(value_str)
            except ValueError: print(f"Warning: Could not parse value '{value_str}' for param '{name}'.")
    return (final_declarations.get('var', []), final_declarations.get('varexo', []),
            parameter_names_declared, assignments)

# --- Model Equation Extractor (Unchanged) ---
def extract_model_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    model_match = re.search(r'(?i)\bmodel\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not model_match: raise ValueError("Could not find 'model; ... end;' block.")
    model_content = model_match.group(1)
    equations_raw = [eq.strip() for eq in model_content.split(';') if eq.strip()]
    processed_equations = []
    for line in equations_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2: processed_equations.append(f"({parts[0].strip()}) - ({parts[1].strip()})")
            else: print(f"Warning: Skipping malformed equation: '{line}'")
        else: processed_equations.append(line)
    return processed_equations


# --- Helper function ---
def base_name_from_aux(aux_name, lag_level=False):
    """Helper to extract base variable name from auxiliary variable name"""
    match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p(\d+)", aux_name)
    if match_lead:
        base = match_lead.group(1); level = int(match_lead.group(2))
        return base if not lag_level else (f"aux_{base}_lead_p{level-1}" if level > 1 else base)
    match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m(\d+)", aux_name)
    if match_lag:
         base = match_lag.group(1); level = int(match_lag.group(2))
         return base if not lag_level else (f"aux_{base}_lag_m{level-1}" if level > 1 else base)
    return None


# --- Main Parser Function (Corrected Debugging Code) ---
def parse_lambdify_and_order_model(model_string):
    print("--- Parsing Model Declarations ---")
    declared_vars, shock_names, param_names, param_assignments = extract_declarations(model_string) # Correct variable name is shock_names
    if not declared_vars: raise ValueError("No variables declared.")
    if not shock_names: raise ValueError("No shocks declared.")
    if not param_names: raise ValueError("No parameters declared.")
    print(f"Declared Variables ({len(declared_vars)}): {declared_vars}")
    print(f"Declared Shocks ({len(shock_names)}): {shock_names}") # Use shock_names
    print(f"Declared Parameters ({len(param_names)}): {param_names}")
    print(f"Parsed Parameter Assignments: {param_assignments}")

    print("\n--- Parsing Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    print(f"Found {len(raw_equations)} equations in model block.")

    print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars); aux_variables = OrderedDict(); processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]; eq_idx += 1; modified_eq = eq
        matches = list(var_time_regex.finditer(eq))
        for match in reversed(matches): # Process backwards
            base_name = match.group(1); time_shift = int(match.group(2))
            if base_name not in endogenous_vars and base_name not in aux_variables:
                if base_name in param_names or base_name in shock_names: continue # Use shock_names
                continue
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        def_eq_str = f"{aux_name} - {prev_var}(+1)"
                        aux_variables[aux_name] = def_eq_str; aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)"; start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        print(f"  Adding aux lead def: {def_eq} = 0"); processed_equations.append(def_eq)
            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        def_eq_str = f"{aux_name} - {prev_var}(-1)"
                        aux_variables[aux_name] = def_eq_str; aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)"; start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        print(f"  Adding aux lag def: {def_eq} = 0"); processed_equations.append(def_eq)
        if modified_eq != eq: processed_equations[eq_idx - 1] = modified_eq

    initial_vars_ordered = list(endogenous_vars); num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations); num_shocks = len(shock_names) # Use shock_names
    print(f"\nTotal variables after processing ({num_vars}): {initial_vars_ordered}")
    print(f"Total equations after processing ({num_eq})")
    if num_vars != num_eq: raise ValueError(f"Model not square: {num_vars} vars vs {num_eq} eqs.")
    print("Model is square.")

    print("\n--- Creating Symbolic Representation ---")
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names} # Use shock_names
    var_syms = {}; all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in initial_vars_ordered:
        sym_m1, sym_t, sym_p1 = (create_timed_symbol(var, s) for s in [-1, 0, 1])
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
        all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])
    local_dict = {str(s): s for s in all_syms_for_parsing}
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
           implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    sym_equations = []
    print("Parsing equations into symbolic form...")
    for i, eq_str in enumerate(processed_equations):
        eq_str_sym = eq_str
        def replace_var_time(match):
            base_name = match.group(1); time_shift = int(match.group(2))
            if base_name in shock_names: # Use shock_names
                if time_shift == 0: return str(shock_syms[base_name])
                else: raise ValueError(f"Shock {base_name}({time_shift}) invalid: {eq_str}")
            elif base_name in var_syms:
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0:  return str(var_syms[base_name]['t'])
                if time_shift == 1:  return str(var_syms[base_name]['p1'])
                raise ValueError(f"Unexpected shift {time_shift} for {base_name}: {eq_str}")
            elif base_name in param_syms: raise ValueError(f"Param {base_name}({time_shift}) invalid: {eq_str}")
            elif base_name in local_dict: return match.group(0)
            else: print(f"Warning: Untimed sym '{base_name}' eq {i}: '{eq_str}'"); return base_name
        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)
        all_known_base_names = sorted(list(var_syms.keys()) + param_names + shock_names, key=len, reverse=True) # Use shock_names
        for name in all_known_base_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name in var_syms: replacement = str(var_syms[name]['t'])
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in shock_names: replacement = str(shock_syms[name]) # Use shock_names
            else: continue
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        try:
            current_symbols = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq_str_sym))
            unknown = current_symbols - set(local_dict.keys()) - {'log','exp','sqrt','abs'}
            unknown = {s for s in unknown if not re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)}
            if unknown: print(f"Warning: Undeclared syms eq {i}: {unknown}"); local_dict.update({s:sympy.symbols(s) for s in unknown if s not in local_dict})
            sym_equations.append(parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations))
        except Exception as e: print(f"\nError parsing eq {i}: '{eq_str}' -> '{eq_str_sym}'\nError: {e}"); raise
    print("Symbolic parsing completed.")

    print("\n--- Generating Initial Symbolic Matrices (A, B, C, D) ---")
    sympy_A, sympy_B, sympy_C = (sympy.zeros(num_eq, num_vars) for _ in range(3))
    sympy_D = sympy.zeros(num_eq, num_shocks) # Use num_shocks
    print("Populating symbolic matrices...")
    for i, eq in enumerate(sym_equations):
        eq_expanded = sympy.expand(eq)
        for j, var in enumerate(initial_vars_ordered):
            sympy_A[i, j] = eq_expanded.coeff(var_syms[var]['p1'])
            sympy_B[i, j] = eq_expanded.coeff(var_syms[var]['t'])
            sympy_C[i, j] = eq_expanded.coeff(var_syms[var]['m1'])
        for k, shk in enumerate(shock_names): # Use shock_names
            sympy_D[i, k] = eq_expanded.coeff(shock_syms[shk])
    initial_info = {'A': sympy_A.copy(), 'B': sympy_B.copy(), 'C': sympy_C.copy(), 'D': sympy_D.copy(),
                    'vars': list(initial_vars_ordered), 'eqs': list(processed_equations)}
    print("Initial symbolic matrices generated.")

    print("\n--- Classifying Variables for Ordering ---")
    exo_process_vars = []; exo_defining_eq_indices = {}
    potential_exo_vars = [v for v in initial_vars_ordered if v.startswith("RES_")]
    potential_aux_lag_vars = [v for v in initial_vars_ordered if v.startswith("aux_") and "_lag_" in v]
    for eq_idx, eq_str in enumerate(processed_equations):
        match = re.match(r"\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*-\s*\((.*)\)", eq_str) or \
                re.match(r"\s*([a-zA-Z_]\w*)\s*-\s*(.*)", eq_str)
        if match:
            lhs_var = match.group(1).strip()
            if lhs_var in potential_exo_vars and lhs_var not in exo_defining_eq_indices:
                if f"{lhs_var}(-1)" in match.group(2):
                     # print(f"  Assoc var '{lhs_var}' with eq {eq_idx}") # Verbose
                     exo_defining_eq_indices[lhs_var] = eq_idx
                     if lhs_var not in exo_process_vars: exo_process_vars.append(lhs_var)
            if lhs_var in potential_aux_lag_vars and lhs_var not in exo_defining_eq_indices:
                 base = base_name_from_aux(lhs_var); prev = f"{base}(-1)" if not "_m" in lhs_var else f"{base_name_from_aux(lhs_var, lag_level=True)}(-1)"
                 if prev in match.group(2):
                     # print(f"  Assoc aux '{lhs_var}' with eq {eq_idx}") # Verbose
                     exo_defining_eq_indices[lhs_var] = eq_idx
                     if lhs_var not in exo_process_vars: exo_process_vars.append(lhs_var)

    backward_exo_vars = sorted([v for v in initial_vars_ordered if v in exo_process_vars or (v.startswith("aux_") and "_lag_" in v and base_name_from_aux(v) in exo_process_vars)], key=initial_vars_ordered.index)
    forward_backward_endo_vars = []; static_endo_vars = []
    remaining_vars = [v for v in initial_vars_ordered if v not in backward_exo_vars]
    for var in remaining_vars:
        j = initial_vars_ordered.index(var)
        has_lag = not sympy_A.col(j).is_zero_matrix; has_lead = not sympy_C.col(j).is_zero_matrix
        if has_lag or has_lead: forward_backward_endo_vars.append(var)
        else: static_endo_vars.append(var)
    print("\nCategorized Variables:")
    print(f"  Backward/Exo Group ({len(backward_exo_vars)}): {backward_exo_vars}")
    print(f"  Forward/Backward Endo ({len(forward_backward_endo_vars)}): {forward_backward_endo_vars}")
    print(f"  Static Endo ({len(static_endo_vars)}): {static_endo_vars}")

    ordered_vars = backward_exo_vars + forward_backward_endo_vars + static_endo_vars
    if len(ordered_vars) != num_vars: raise ValueError("Var reordering len mismatch")
    if set(ordered_vars) != set(initial_vars_ordered): raise ValueError("Var reordering content mismatch")
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]
    print(f"\nNew Variable Order ({len(ordered_vars)}): {ordered_vars}")

    eq_perm_indices = []; used_eq_indices = set()
    def find_unused_eq(used_set, total_eqs): return next((i for i in range(total_eqs) if i not in used_set), -1)
    for var in backward_exo_vars:
        assigned_eq = -1
        if var in exo_defining_eq_indices and exo_defining_eq_indices[var] not in used_eq_indices: assigned_eq = exo_defining_eq_indices[var]
        else: fallback_eq = find_unused_eq(used_eq_indices, num_eq); assigned_eq = fallback_eq if fallback_eq != -1 else assigned_eq # Removed warning print
        if assigned_eq != -1: eq_perm_indices.append(assigned_eq); used_eq_indices.add(assigned_eq)
        else: raise RuntimeError(f"Cannot find eq for backward var '{var}'")
    for _ in forward_backward_endo_vars: # Use _ as var name not needed
        assigned_eq = find_unused_eq(used_eq_indices, num_eq)
        if assigned_eq != -1: eq_perm_indices.append(assigned_eq); used_eq_indices.add(assigned_eq)
        else: raise RuntimeError(f"Cannot find eq for fwd/bwd var")
    for _ in static_endo_vars: # Use _ as var name not needed
        assigned_eq = find_unused_eq(used_eq_indices, num_eq)
        if assigned_eq != -1: eq_perm_indices.append(assigned_eq); used_eq_indices.add(assigned_eq)
        else: raise RuntimeError(f"Cannot find eq for static var")
    if len(eq_perm_indices) != num_eq: raise ValueError("Eq permutation length mismatch")
    print(f"\nEquation permutation indices ({len(eq_perm_indices)}): {eq_perm_indices}")

    print("\n--- Reordering Symbolic Matrices ---")
    sympy_A_ord = sympy_A.extract(eq_perm_indices, var_perm_indices)
    sympy_B_ord = sympy_B.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C.extract(eq_perm_indices, var_perm_indices)
    sympy_D_ord = sympy_D.extract(eq_perm_indices, list(range(num_shocks))) # Use num_shocks
    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}
    print("Symbolic reordering complete.")

    # --- !!! DEBUGGING POINT 1: Inspect Symbolic Matrices !!! ---
    print("\n--- DEBUG: Inspecting Ordered Symbolic Matrices (Entries related to RES_ vars) ---")
    res_vars_indices_ordered = {v: ordered_vars.index(v) for v in ordered_vars if v.startswith("RES_")}
    aux_lag_vars_indices_ordered = {v: ordered_vars.index(v) for v in ordered_vars if v.startswith("aux_") and "_lag_" in v}
    res_eq_indices_ordered = {}
    for var_name, initial_eq_idx in exo_defining_eq_indices.items():
         if initial_eq_idx in eq_perm_indices: res_eq_indices_ordered[var_name] = eq_perm_indices.index(initial_eq_idx)
         else: print(f"Warning: Def eq {initial_eq_idx} for {var_name} not in permutation.")
    print(f"  Ordered RES_ variable indices: {res_vars_indices_ordered}")
    print(f"  Ordered Aux Lag variable indices: {aux_lag_vars_indices_ordered}")
    print(f"  Ordered RES_/Aux defining equation indices: {res_eq_indices_ordered}")

    target_var = "RES_L_GDP_GAP"; target_shock = "SHK_L_GDP_GAP"; rho_param_name = 'rho_L_GDP_GAP'
    rho_param_sym = param_syms.get(rho_param_name)
    if target_var in res_vars_indices_ordered and target_var in res_eq_indices_ordered and rho_param_sym:
        row_idx = res_eq_indices_ordered[target_var]; col_idx = res_vars_indices_ordered[target_var]
        # --- FIX HERE: Use shock_names ---
        shock_col_idx = shock_names.index(target_shock) if target_shock in shock_names else -1
        print(f"\n  Checking Eq for {target_var} (Row {row_idx}, Var Col {col_idx}, Shock Col {shock_col_idx}):")
        sym_a_entry = sympy_A_ord[row_idx, col_idx]
        print(f"    Symbolic A[{row_idx}, {col_idx}] (coeff {target_var}_m1): {sym_a_entry}")
        if not (sym_a_entry == -rho_param_sym or sym_a_entry == sympy.Mul(-1, rho_param_sym)): print(f"      WARNING: Expected {-rho_param_sym} differs.")
        sym_b_entry = sympy_B_ord[row_idx, col_idx]
        print(f"    Symbolic B[{row_idx}, {col_idx}] (coeff {target_var}): {sym_b_entry}")
        if sym_b_entry != 1: print(f"      WARNING: Expected B coeff 1, got {sym_b_entry}")
        if shock_col_idx != -1:
            sym_d_entry = sympy_D_ord[row_idx, shock_col_idx]
            print(f"    Symbolic D[{row_idx}, {shock_col_idx}] (coeff {target_shock}): {sym_d_entry}")
            if sym_d_entry != -1: print(f"      WARNING: Expected D coeff -1, got {sym_d_entry}")
    else: print(f"  Could not perform detailed symbolic check for {target_var}.")

    target_var_rs = "RES_RS"; target_shock_rs = "SHK_RS"; rho1_param_name = "rho_rs"; rho2_param_name = "rho_rs2"; aux_lag_var = "aux_RES_RS_lag_m1"
    if (target_var_rs in res_vars_indices_ordered and target_var_rs in res_eq_indices_ordered and
        aux_lag_var in aux_lag_vars_indices_ordered and rho1_param_name in param_syms and rho2_param_name in param_syms):
        row_idx_rs = res_eq_indices_ordered[target_var_rs]; col_idx_rs = res_vars_indices_ordered[target_var_rs]
        col_idx_aux = aux_lag_vars_indices_ordered[aux_lag_var]
        # --- FIX HERE: Use shock_names ---
        shock_col_idx_rs = shock_names.index(target_shock_rs) if target_shock_rs in shock_names else -1
        rho1_sym = param_syms[rho1_param_name]; rho2_sym = param_syms[rho2_param_name]
        print(f"\n  Checking Eq for {target_var_rs} (Row {row_idx_rs}, Var Col {col_idx_rs}, Aux Col {col_idx_aux}, Shock Col {shock_col_idx_rs}):")
        sym_a1_entry = sympy_A_ord[row_idx_rs, col_idx_rs] # Coeff of RES_RS_m1
        sym_a2_entry = sympy_A_ord[row_idx_rs, col_idx_aux] # Coeff of aux_RES_RS_lag_m1_m1
        print(f"    Symbolic A[{row_idx_rs}, {col_idx_rs}] (coeff {target_var_rs}_m1): {sym_a1_entry}")
        if not (sym_a1_entry == -rho1_sym or sym_a1_entry == sympy.Mul(-1, rho1_sym)): print(f"      WARNING: Expected {-rho1_sym}, got {sym_a1_entry}")
        print(f"    Symbolic A[{row_idx_rs}, {col_idx_aux}] (coeff {aux_lag_var}_m1): {sym_a2_entry}")
        if not (sym_a2_entry == -rho2_sym or sym_a2_entry == sympy.Mul(-1, rho2_sym)): print(f"      WARNING: Expected {-rho2_sym}, got {sym_a2_entry}")
        sym_b_entry_rs = sympy_B_ord[row_idx_rs, col_idx_rs] # Coeff of RES_RS(t)
        print(f"    Symbolic B[{row_idx_rs}, {col_idx_rs}] (coeff {target_var_rs}): {sym_b_entry_rs}")
        if sym_b_entry_rs != 1: print(f"      WARNING: Expected B coeff 1, got {sym_b_entry_rs}")
        if shock_col_idx_rs != -1:
            sym_d_entry_rs = sympy_D_ord[row_idx_rs, shock_col_idx_rs]
            print(f"    Symbolic D[{row_idx_rs}, {shock_col_idx_rs}] (coeff {target_shock_rs}): {sym_d_entry_rs}")
            if sym_d_entry_rs != -1: print(f"      WARNING: Expected D coeff -1, got {sym_d_entry_rs}")
    else: print(f"  Could not perform detailed symbolic check for {target_var_rs}.")

    print("\n--- Lambdifying Ordered Matrices ---")
    param_sym_list = [param_syms[p] for p in param_names]
    print("\n--- DEBUG: Checking Lambdify Inputs ---")
    print(f"  Parameter names order ({len(param_names)}): {param_names}")
    print(f"  Parameter symbols order for lambdify ({len(param_sym_list)}): {param_sym_list}")

    try:
        func_A = sympy.lambdify(param_sym_list, sympy_A_ord, modules='numpy')
        func_B = sympy.lambdify(param_sym_list, sympy_B_ord, modules='numpy')
        func_C = sympy.lambdify(param_sym_list, sympy_C_ord, modules='numpy')
        func_D = sympy.lambdify(param_sym_list, sympy_D_ord, modules='numpy')
        print("Lambdification successful.")
    except Exception as e: print(f"Lambdify Error: {e}"); raise

    # Return shock_names as well for use in main block
    return (func_A, func_B, func_C, func_D, ordered_vars, shock_names, param_names, param_assignments,
            symbolic_matrices_ordered, initial_info) # Use shock_names


# --- Example usage (Corrected Debugging Code) ---
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_name = "qpm_simpl1.dyn"
        mod_file_path = os.path.join(script_dir, mod_file_name)
        if not os.path.exists(mod_file_path): raise FileNotFoundError(f"File '{mod_file_name}' not found in {script_dir}")
        print(f"Reading model definition from: {mod_file_path}")
        with open(mod_file_path, 'r') as f: model_def = f.read()

        # --- PARSE --- Get shock_names back
        (func_A, func_B, func_C, func_D, ordered_vars, shock_names_ret, param_names, param_assignments,
         sym_matrices_ord, initial_info) = parse_lambdify_and_order_model(model_def) # Use shock_names_ret

        print("\n\n--- Parser Results ---")
        print("Parameter Names:", param_names)
        print("Param Assignments:", param_assignments)
        print("Shock Names:", shock_names_ret) # Use shock_names_ret
        print("Final Ordered Variables:", ordered_vars)

        # --- EVALUATE ---
        test_param_values = param_assignments.copy()
        default_test_values = { 'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
                                'rho_L_GDP_GAP': 0.75, 'rho_DLA_CPI': 0.75, 'rho_rs': 0.0, 'rho_rs2': 0.00 }
        missing_params = []; test_args = []
        print("\n--- Evaluating Matrices with Parameters ---")
        print("Building test_args list based on param_names order:")
        for i, p_name in enumerate(param_names):
            value, source = None, "Not Found"
            if p_name in test_param_values: value, source = test_param_values[p_name], "Assignment"
            elif p_name in default_test_values: value, source = default_test_values[p_name], "Default"
            else: missing_params.append(p_name); value, source = 0.0, "MISSING (0.0)"
            test_args.append(value); print(f"  Arg {i}: name='{p_name}', value={value} ({source})")

        if not missing_params:
            print("\nEvaluating numerical matrices A, B, C, D...")
            A_num = func_A(*test_args); B_num = func_B(*test_args)
            C_num = func_C(*test_args); D_num = func_D(*test_args)
            print("Numerical matrices evaluated.")

            # --- !!! DEBUGGING POINT 4: Inspect Numerical Matrices !!! ---
            print("\n--- DEBUG: Inspecting Numerical Matrices (Entries related to RES_ vars) ---")
            res_vars_idx_num = {v: ordered_vars.index(v) for v in ordered_vars if v.startswith("RES_")}
            aux_lag_idx_num = {v: ordered_vars.index(v) for v in ordered_vars if v.startswith("aux_") and "_lag_" in v}
            # Need the equation permutation map again or re-infer it
            # Re-infer row logic (less robust than passing the map)
            if "RES_L_GDP_GAP" in res_vars_idx_num:
                target_var = "RES_L_GDP_GAP"; target_shock = "SHK_L_GDP_GAP"; rho_param_name = 'rho_L_GDP_GAP'
                col_idx_res = res_vars_idx_num[target_var]; likely_row_idx = -1
                for r in range(A_num.shape[0]):
                    if np.isclose(B_num[r, col_idx_res], 1.0) and not np.isclose(A_num[r, col_idx_res], 0.0): likely_row_idx = r; break
                if likely_row_idx != -1:
                    print(f"\n  Checking Eq likely for {target_var} (Row {likely_row_idx}, Var Col {col_idx_res}):")
                    col_idx = res_vars_idx_num[target_var]
                    # --- FIX HERE: Use shock_names_ret ---
                    shock_col_idx = shock_names_ret.index(target_shock) if target_shock in shock_names_ret else -1
                    num_a = A_num[likely_row_idx, col_idx]; exp_a = -test_args[param_names.index(rho_param_name)] if rho_param_name in param_names else 'N/A'
                    print(f"    Num A[{likely_row_idx},{col_idx}]: {num_a:.4f} (Exp: {exp_a})")
                    num_b = B_num[likely_row_idx, col_idx]
                    print(f"    Num B[{likely_row_idx},{col_idx}]: {num_b:.4f} (Exp: 1.0)")
                    if shock_col_idx != -1:
                        num_d = D_num[likely_row_idx, shock_col_idx]
                        print(f"    Num D[{likely_row_idx},{shock_col_idx}]: {num_d:.4f} (Exp: -1.0)")
                else: print(f"  Could not identify likely row for {target_var} num check.")

            if "RES_RS" in res_vars_idx_num:
                target_var_rs = "RES_RS"; target_shock_rs = "SHK_RS"; rho1, rho2 = "rho_rs", "rho_rs2"; aux = "aux_RES_RS_lag_m1"
                col_rs = res_vars_idx_num[target_var_rs]; col_aux = aux_lag_idx_num.get(aux, -1)
                # --- FIX HERE: Use shock_names_ret ---
                shock_col_rs = shock_names_ret.index(target_shock_rs) if target_shock_rs in shock_names_ret else -1
                likely_row_rs = -1
                for r in range(A_num.shape[0]):
                    if np.isclose(B_num[r, col_rs], 1.0) and not np.isclose(A_num[r, col_rs], 0.0) and \
                       (col_aux == -1 or not np.isclose(A_num[r, col_aux], 0.0)): likely_row_rs = r; break
                if likely_row_rs != -1:
                    print(f"\n  Checking Eq likely for {target_var_rs} (Row {likely_row_rs}, Var Col {col_rs}, Aux Col {col_aux}):")
                    num_a1 = A_num[likely_row_rs, col_rs]; exp_a1 = -test_args[param_names.index(rho1)] if rho1 in param_names else 'N/A'
                    print(f"    Num A[{likely_row_rs},{col_rs}]: {num_a1:.4f} (Exp: {exp_a1})")
                    if col_aux != -1:
                        num_a2 = A_num[likely_row_rs, col_aux]; exp_a2 = -test_args[param_names.index(rho2)] if rho2 in param_names else 'N/A'
                        print(f"    Num A[{likely_row_rs},{col_aux}]: {num_a2:.4f} (Exp: {exp_a2})")
                    num_b = B_num[likely_row_rs, col_rs]
                    print(f"    Num B[{likely_row_rs},{col_rs}]: {num_b:.4f} (Exp: 1.0)")
                    if shock_col_rs != -1:
                        num_d = D_num[likely_row_rs, shock_col_rs]
                        print(f"    Num D[{likely_row_rs},{shock_col_rs}]: {num_d:.4f} (Exp: -1.0)")
                else: print(f"  Could not identify likely row for {target_var_rs} num check.")


            # --- SOLVE, Q, IRF, PLOT ---
            print("\n--- Solving, Computing Q, IRFs, and Plotting ---")
            P_sol, iter_count, residual_ratio = solve_quadratic_matrix_equation(A_num, B_num, C_num, tol=1e-12, verbose=False)
            if P_sol is None: print("\nERROR: Solver failed.")
            else:
                print(f"Solver done: iters={iter_count}, res={residual_ratio:.2e}")
                if residual_ratio > 1e-6: print("Warning: High solver residual.")
                try: max_eig = np.max(np.abs(np.linalg.eigvals(P_sol))); print(f"Max eig P: {max_eig:.6f} {'(Stable)' if max_eig < 1.0 else '(UNSTABLE)'}")
                except np.linalg.LinAlgError: print("Warning: eig(P) failed.")
                Q_sol = compute_Q(A_num, B_num, D_num, P_sol)
                if Q_sol is None: print("ERROR: Failed compute Q.")
                else:
                    print("Q matrix computed.")
                    shock_idx = 2; horizon = 40 # Example: SHK_RS
                    print(f"Computing IRFs for shock {shock_names_ret[shock_idx]}...") # Use shock_names_ret
                    irf_vals = irf(P_sol, Q_sol, shock_index=shock_idx, horizon=horizon)
                    print(f"Plotting IRFs...")
                    vars_to_plot = ["L_GDP_GAP", "DLA_CPI", "RS", "RR_GAP", "RES_RS", "RES_L_GDP_GAP", "RES_DLA_CPI"]
                    valid_vars = []; indices = []
                    for v in vars_to_plot:
                        if v in ordered_vars: valid_vars.append(v); indices.append(ordered_vars.index(v))
                        else: print(f"Warn: Var '{v}' not in ordered list for plot.")
                    if valid_vars:
                        plt.figure(figsize=(12, 8)); n_plots = len(valid_vars); cols = 3 if n_plots > 4 else 2; rows = (n_plots + cols - 1)//cols
                        plt.suptitle(f"IRFs to Unit {shock_names_ret[shock_idx]} Shock", fontsize=14) # Use shock_names_ret
                        for i, name in enumerate(valid_vars):
                            plt.subplot(rows, cols, i + 1); plt.plot(range(horizon), irf_vals[:, indices[i]])
                            plt.axhline(0, c='k', lw=0.7, ls=':'); plt.title(name); plt.grid(True, ls='--', alpha=0.6)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
                    else: print("No valid vars to plot.")
        else: print(f"\nSkipping num steps due to missing params: {missing_params}")
    except FileNotFoundError as e: print(f"\nError: {e}")
    except ValueError as e: print(f"\nValueError: {e}"); import traceback; traceback.print_exc()
    except np.linalg.LinAlgError as e: print(f"\nLinAlgError: {e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nUnexpected Error: {e}"); import traceback; traceback.print_exc()