# -*- coding: utf-8 -*-
"""
Enhanced Dynare Parser and State-Space Solver

This script parses a Dynare-like model file, including trend and
measurement equation components, solves the stationary part of the model
using a quadratic matrix equation solver (SDA), builds the augmented
state-space representation including trends, and computes impulse responses
for both state variables and observable variables.
"""

import re
import sympy
import numpy as np
from collections import OrderedDict
import copy
import os
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve, block_diag
import matplotlib.pyplot as plt

# --- Helper Functions ---

def plot_irfs(irf_values, var_names, horizon, title="Impulse Responses"):
    """ Simple IRF plotting function """
    num_vars = irf_values.shape[1]
    if num_vars == 0:
        print(f"No variables to plot for: {title}")
        return

    # Dynamically determine grid layout
    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols

    plt.figure(figsize=(min(5*cols, 18), 3*rows)) # Adjust figsize
    plt.suptitle(title, fontsize=14)
    time = range(horizon)

    for i, var_name in enumerate(var_names):
        plt.subplot(rows, cols, i + 1)
        plt.plot(time, irf_values[:, i], label=var_name)
        plt.axhline(0, color='black', linewidth=0.7, linestyle=':') # Zero line
        plt.title(var_name)
        plt.grid(True, linestyle='--', alpha=0.6)
        # Add x-label only to bottom row plots for clarity
        if i >= num_vars - cols:
             plt.xlabel("Horizon")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    
def create_timed_symbol(base_name, time_shift):
    """Creates a sympy symbol with a time suffix (_m1, _t, _p1)."""
    if time_shift == -1:
        return sympy.symbols(f"{base_name}_m1")
    elif time_shift == 1:
        return sympy.symbols(f"{base_name}_p1")
    elif time_shift == 0:
        # Use base name for time t for cleaner symbolic expressions
        return sympy.symbols(base_name)
    else:
        # Should have been handled by auxiliary variables
        raise ValueError(f"Unexpected time shift {time_shift} for {base_name}")

def base_name_from_aux(aux_name):
    """Helper to extract base variable name from auxiliary variable name"""
    match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p\d+", aux_name)
    if match_lead:
        return match_lead.group(1)
    match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", aux_name)
    if match_lag:
        return match_lag.group(1)
    # Return original name if it doesn't match aux patterns
    return aux_name

def symbolic_jacobian(equations, variables):
    """Computes the symbolic Jacobian matrix."""
    num_eq = len(equations)
    num_var = len(variables)
    jacobian = sympy.zeros(num_eq, num_var)
    for i, eq in enumerate(equations):
        for j, var in enumerate(variables):
            jacobian[i, j] = sympy.diff(eq, var)
    return jacobian

def robust_lambdify(args, expr, modules='numpy'):
    """
    Attempts to lambdify, providing more context on error.
    Handles cases where expr is already numerical (e.g., zero matrix).
    """
    if isinstance(expr, (int, float)) or \
       (isinstance(expr, np.ndarray) and np.issubdtype(expr.dtype, np.number)):
        # If it's already numerical, return a function that ignores args
        # and returns the constant value.
        return lambda *a: expr
    if isinstance(expr, (sympy.Matrix, sympy.ImmutableMatrix)):
        # Check if the matrix contains any symbols
        if not expr.free_symbols:
            # If no free symbols, convert to numpy array and return lambda
            try:
                numerical_matrix = np.array(expr.tolist(), dtype=float)
                return lambda *a: numerical_matrix
            except (TypeError, ValueError) as e:
                 print(f"Warning: Could not convert symbol-free matrix {expr} to NumPy array: {e}")
                 # Fallback to standard lambdify, which might handle it or fail
                 pass # Proceed to standard lambdify below

    try:
        # Standard lambdify for symbolic expressions
        return sympy.lambdify(args, expr, modules=modules)
    except Exception as e:
        print(f"Error during lambdify. Arguments: {args}")
        print(f"Expression causing error:\n{expr}")
        raise e

# --- Core Model Parsing and Solving Functions (Adapted from original) ---

def extract_declarations(model_string):
    """
    Extracts variables, shocks, and parameters using regex, ensuring only
    declarations *before* the 'model;' block are considered.
    Handles comma/space separation and terminating semicolons robustly.
    Correctly extracts only parameter names, ignoring assignments.
    Also extracts parameter assignments separately.
    """
    declarations = {
        'var': [],
        'varexo': [],
        'parameters': []
    }
    param_assignments = {}

    # --- Pre-processing: Remove Comments ---
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL) # Block comments
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines] # Line comments
    processed_content = " \n ".join(cleaned_lines)

    # --- Find the 'model;' marker ---
    model_marker = re.search(r'\bmodel\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    if not model_marker:
        print("Warning: 'model;' marker not found. Processing all declarations found.")
        content_to_search = processed_content
        param_assignment_content_search_area = processed_content # Search everywhere for assignments
    else:
        content_to_search = processed_content[:model_marker.start()]
        # Search for assignments only up to the model block
        param_assignment_content_search_area = content_to_search

    # --- Regex Extraction for Declarations ---
    block_matches = re.finditer(
        r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters|model)\b|$)',
        content_to_search,
        re.DOTALL | re.IGNORECASE
    )

    def process_block_content(content_str, block_type):
        """Helper to clean and split names from block content string."""
        if not content_str: return []
        content = content_str.strip()
        first_semicolon_match = re.search(r';', content)
        if first_semicolon_match:
            content = content[:first_semicolon_match.start()].strip()
        content = content.replace('\n', ' ')
        names = []
        # Split by comma or whitespace, filter empty strings and ensure valid identifiers
        raw_names = re.split(r'[,\s]+', content)
        cleaned_names = [name for name in raw_names if name and re.fullmatch(r'[a-zA-Z_]\w*', name)]
        # Basic keyword filter
        keywords = {'var', 'varexo', 'parameters', 'model', 'end'}
        names = [n for n in cleaned_names if n not in keywords]
        return list(dict.fromkeys(names).keys()) # Remove duplicates, preserve order

    for match in block_matches:
        block_keyword = match.group(1).lower()
        block_content_raw = match.group(2)
        extracted_names = process_block_content(block_content_raw, block_keyword)
        declarations[block_keyword].extend(extracted_names)

    final_declarations = {key: list(dict.fromkeys(lst).keys()) for key, lst in declarations.items()}

    # --- Extract parameter assignments separately ---
    # Search within the designated area (before model block if found)
    assignment_matches = re.finditer(
        r'\b([a-zA-Z_]\w*)\b\s*=\s*([^;]+);',
        param_assignment_content_search_area
    )
    parameter_names_declared = final_declarations.get('parameters', [])
    for match in assignment_matches:
        name = match.group(1)
        value_str = match.group(2).strip()
        if name in parameter_names_declared:
            try:
                param_assignments[name] = float(value_str)
            except ValueError:
                print(f"Warning: Could not parse value '{value_str}' for parameter '{name}'. Skipping assignment.")
        # else: # Optional warning for assignments to non-declared params
            # print(f"Warning: Assignment found for '{name}', but it was not in the parameter declaration. Ignoring.")

    return (
        final_declarations.get('var', []),
        final_declarations.get('varexo', []),
        parameter_names_declared,
        param_assignments
    )

def extract_model_equations(model_string):
    """
    Extracts equations from the 'model; ... end;' block.
    """
    # Pre-processing: Remove comments first
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)

    model_match = re.search(
        r'(?i)\bmodel\b\s*;(.*?)\bend\b\s*;',
        processed_content,
        re.DOTALL | re.IGNORECASE
    )
    if not model_match:
        raise ValueError("Could not find 'model; ... end;' block.")

    model_content = model_match.group(1)
    equations_raw = [eq.strip() for eq in model_content.split(';') if eq.strip()]

    processed_equations = []
    for line in equations_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts
                processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else:
                print(f"Warning: Skipping malformed equation line: '{line}'")
        else:
            print(f"Warning: Equation '{line}' has no '='. Assuming it's 'expr = 0'.")
            processed_equations.append(line)

    return processed_equations

def parse_lambdify_and_order_model(model_string):
    """
    Parses the stationary part of the model, handles leads/lags,
    orders variables/equations, and returns lambdified matrices.
    """
    print("--- Parsing Stationary Model Declarations ---")
    declared_vars, shock_names, param_names, param_assignments = extract_declarations(model_string)

    if not declared_vars: raise ValueError("No variables declared in 'var' block.")
    if not shock_names: raise ValueError("No shocks declared in 'varexo' block.")
    if not param_names: raise ValueError("No parameters declared in 'parameters' block.")

    print(f"Declared Variables: {declared_vars}")
    print(f"Declared Shocks: {shock_names}")
    print(f"Declared Parameters: {param_names}")
    print(f"Parsed Parameter Assignments: {param_assignments}")

    print("\n--- Parsing Stationary Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    print(f"Found {len(raw_equations)} equations in model block.")

    # --- Handling Leads/Lags & Auxiliaries ---
    print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars)
    aux_variables = OrderedDict() # Stores definition string for each aux var
    processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]
        eq_idx += 1
        modified_eq = eq
        matches = list(var_time_regex.finditer(eq))

        # Process matches in reverse order to avoid index issues
        for match in reversed(matches):
            base_name = match.group(1)
            time_shift = int(match.group(2))

            # Skip if not an endogenous variable or already processed aux
            if base_name not in endogenous_vars and base_name not in aux_variables:
                continue

            # --- Handle Leads > 1 ---
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(+1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]

                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        # print(f"  Adding aux lead def: {def_eq} = 0")
                        processed_equations.append(def_eq)

            # --- Handle Lags < -1 ---
            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(-1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]

                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        # print(f"  Adding aux lag def: {def_eq} = 0")
                        processed_equations.append(def_eq)

        if modified_eq != eq:
            processed_equations[eq_idx - 1] = modified_eq
            # print(f"  Updated Eq {eq_idx-1}: {modified_eq}")

    initial_vars_ordered = list(endogenous_vars)
    num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations)
    num_shocks = len(shock_names)

    print(f"Total variables after processing leads/lags ({num_vars}): {initial_vars_ordered}")
    # print(f"Total equations after processing leads/lags ({num_eq}):")
    # for i, eq in enumerate(processed_equations): print(f"  Eq {i}: {eq}")

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
            f"Stationary model not square after processing leads/lags: {num_vars} vars vs {num_eq} eqs."
        )
    print("Stationary model is square.")

    # --- Symbolic Representation ---
    print("\n--- Creating Symbolic Representation (Stationary Model) ---")
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names}
    var_syms = {}
    all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in initial_vars_ordered:
        sym_m1 = create_timed_symbol(var, -1)
        sym_t  = create_timed_symbol(var, 0)
        sym_p1 = create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
        all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])

    local_dict = {str(s): s for s in all_syms_for_parsing}
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                          implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))

    sym_equations = []
    print("Parsing stationary equations into symbolic form...")
    for i, eq_str in enumerate(processed_equations):
        eq_str_sym = eq_str
        def replace_var_time(match):
            base_name, time_shift_str = match.groups()
            time_shift = int(time_shift_str)
            if base_name in shock_names:
                if time_shift == 0: return str(shock_syms[base_name])
                else: raise ValueError(f"Shock {base_name}({time_shift}) invalid.")
            elif base_name in var_syms:
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0:  return str(var_syms[base_name]['t'])
                if time_shift == 1:  return str(var_syms[base_name]['p1'])
                raise ValueError(f"Unexpected time shift {time_shift} for {base_name} after aux processing.")
            elif base_name in param_syms:
                raise ValueError(f"Parameter {base_name}({time_shift}) invalid.")
            elif base_name in local_dict: # e.g. log, exp
                return match.group(0)
            else: # Unknown symbol - add dynamically if needed, but warn
                # print(f"Warning: Symbol '{base_name}' with time shift {time_shift} in eq {i} ('{eq_str}') is undeclared. Treating symbolically.")
                if base_name not in local_dict: local_dict[base_name] = sympy.symbols(base_name)
                timed_sym_str = str(create_timed_symbol(base_name, time_shift))
                if timed_sym_str not in local_dict: local_dict[timed_sym_str] = sympy.symbols(timed_sym_str)
                return timed_sym_str

        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)

        # Replace remaining base names (implicitly time t)
        all_known_base_names = sorted(list(var_syms.keys()) + param_names + shock_names, key=len, reverse=True)
        for name in all_known_base_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name in var_syms: replacement = str(var_syms[name]['t'])
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in shock_names: replacement = str(shock_syms[name])
            else: continue
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)

        try:
            # Check for remaining undeclared symbols before parsing
            current_symbols = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq_str_sym))
            known_keys = set(local_dict.keys()) | {'log', 'exp', 'sqrt', 'abs'}
            unknown_symbols = {s for s in current_symbols if s not in known_keys and not re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)}

            if unknown_symbols:
                print(f"Warning: Potential undeclared symbols found in eq {i} ('{eq_str_sym}'): {unknown_symbols}. Adding to local_dict.")
                for sym_str in unknown_symbols:
                    if sym_str not in local_dict: local_dict[sym_str] = sympy.symbols(sym_str)

            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            sym_equations.append(sym_eq)
        except Exception as e:
            print(f"\nError parsing stationary equation {i}: '{eq_str}' -> '{eq_str_sym}'")
            print(f"Local dict keys: {sorted(local_dict.keys())}")
            print(f"Sympy error: {e}")
            raise

    print("Symbolic parsing completed.")

    # --- Generate Initial Symbolic Matrices A P^2 + B P + C = 0, D for shocks ---
    print("\n--- Generating Initial Symbolic Matrices (A, B, C, D) ---")
    # Here A coeffs y(t+1), B coeffs y(t), C coeffs y(t-1)
    # This matches the convention needed for the `solve_quadratic_matrix_equation`
    # Note the definition change: A = dF/dy_{t+1}, B = dF/dy_t, C = dF/dy_{t-1}
    sympy_A_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t+1)
    sympy_B_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t)
    sympy_C_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t-1)
    sympy_D_quad = sympy.zeros(num_eq, num_shocks) # Coeffs of e(t) (multiplied by -1 for Q calc)

    var_p1_syms = [var_syms[v]['p1'] for v in initial_vars_ordered]
    var_t_syms  = [var_syms[v]['t']  for v in initial_vars_ordered]
    var_m1_syms = [var_syms[v]['m1'] for v in initial_vars_ordered]
    shock_t_syms = [shock_syms[s] for s in shock_names]

    for i, eq in enumerate(sym_equations):
        # Use Jacobian calculation for robustness with non-linear terms (though model is linear)
        for j, var_p1 in enumerate(var_p1_syms): sympy_A_quad[i, j] = sympy.diff(eq, var_p1)
        for j, var_t  in enumerate(var_t_syms):  sympy_B_quad[i, j] = sympy.diff(eq, var_t)
        for j, var_m1 in enumerate(var_m1_syms): sympy_C_quad[i, j] = sympy.diff(eq, var_m1)
        for k, shk_t in enumerate(shock_t_syms): sympy_D_quad[i, k] = -sympy.diff(eq, shk_t) # Note the minus sign

    initial_info = {
        'A': copy.deepcopy(sympy_C_quad), # Store based on y_t = P y_{t-1} + Q e_t form
        'B': copy.deepcopy(sympy_B_quad),
        'C': copy.deepcopy(sympy_A_quad),
        'D': copy.deepcopy(sympy_D_quad),
        'vars': list(initial_vars_ordered),
        'eqs': list(processed_equations)
    }
    print("Symbolic matrices A, B, C, D generated (for quadratic solver).")

    # --- Classify Variables (Simplified for Ordering) ---
    print("\n--- Classifying Variables for Ordering (Stationary Model) ---")
    # Heuristic: RES_ and aux_lag are backward, others are forward/both
    backward_exo_vars = []
    forward_backward_endo_vars = []
    static_endo_vars = [] # Variables appearing only at time t

    potential_backward = [v for v in initial_vars_ordered if v.startswith("RES_") or (v.startswith("aux_") and "_lag_" in v)]
    remaining_vars = [v for v in initial_vars_ordered if v not in potential_backward]

    # Check matrix columns for actual dependencies
    for var in potential_backward:
        j = initial_vars_ordered.index(var)
        # Check if it has a lead dependency (appears in A_quad)
        has_lead = not sympy_A_quad.col(j).is_zero_matrix
        if has_lead:
            # If an RES_ or aux_lag var has a lead, it's not purely backward
            # This might indicate a model specification issue or complex aux var interaction
            print(f"Warning: Potential backward var '{var}' has lead dependency. Classifying as forward/backward.")
            forward_backward_endo_vars.append(var)
        else:
            backward_exo_vars.append(var)

    for var in remaining_vars:
         j = initial_vars_ordered.index(var)
         has_lag = not sympy_C_quad.col(j).is_zero_matrix # Appears with t-1?
         has_lead = not sympy_A_quad.col(j).is_zero_matrix # Appears with t+1?
         if has_lag or has_lead:
             forward_backward_endo_vars.append(var)
         else:
             # Only appears at time t (in B_quad)
             static_endo_vars.append(var)

    print("\nCategorized Variables:")
    print(f"  Backward/Exo Group: {backward_exo_vars}")
    print(f"  Forward/Backward Endo: {forward_backward_endo_vars}")
    print(f"  Static Endo: {static_endo_vars}")

    # --- Determine New Variable Order ---
    ordered_vars = backward_exo_vars + forward_backward_endo_vars + static_endo_vars
    if len(ordered_vars) != len(initial_vars_ordered) or set(ordered_vars) != set(initial_vars_ordered):
        raise ValueError("Variable reordering failed.")
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]
    print(f"\nNew Variable Order ({len(ordered_vars)}): {ordered_vars}")

    # --- Determine New Equation Order (Simple heuristic: match blocks) ---
    # Find defining equations for backward vars first
    eq_perm_indices = []
    used_eq_indices = set()
    # Find defining eq for each backward var (heuristic: eq where B[i,j] != 0 and C[i,j] == 0, maybe A[i,j] != 0)
    # Simpler: Use aux definitions and RES definitions
    aux_def_patterns = {name: re.compile(fr"^\s*{name}\s*-\s*{base_name_from_aux(name)}\s*\(\s*-1\s*\)\s*$") for name in aux_variables if "_lag_" in name}
    res_def_patterns = {name: re.compile(fr"^\s*{name}\s*-\s*.*{name}\s*\(\s*-1\s*\).*") for name in initial_vars_ordered if name.startswith("RES_")}

    assigned_eq_for_var = {}

    # Assign defining equations for aux lags first
    for aux_var in [v for v in backward_exo_vars if v.startswith("aux_")]:
        if aux_var in assigned_eq_for_var: continue
        pattern = aux_def_patterns.get(aux_var)
        if pattern:
            found = False
            for i, eq_str in enumerate(processed_equations):
                if i not in used_eq_indices and pattern.match(eq_str.replace(" ","")):
                    eq_perm_indices.append(i)
                    used_eq_indices.add(i)
                    assigned_eq_for_var[aux_var] = i
                    found = True
                    break
            # if not found: print(f"Warning: Could not find unique defining eq for aux lag '{aux_var}'")

    # Assign defining equations for RES vars
    for res_var in [v for v in backward_exo_vars if v.startswith("RES_")]:
         if res_var in assigned_eq_for_var: continue
         pattern = res_def_patterns.get(res_var)
         if pattern:
            found = False
            potential_matches = []
            for i, eq_str in enumerate(processed_equations):
                 if i not in used_eq_indices and pattern.match(eq_str):
                     potential_matches.append(i)
            if len(potential_matches) == 1:
                 i = potential_matches[0]
                 eq_perm_indices.append(i)
                 used_eq_indices.add(i)
                 assigned_eq_for_var[res_var] = i
                 found = True
            # elif len(potential_matches) > 1:
                 # print(f"Warning: Found multiple potential defining eqs for RES var '{res_var}': {potential_matches}")
            # else:
                 # print(f"Warning: Could not find defining eq for RES var '{res_var}'")

    # Assign remaining equations heuristically or just sequentially
    remaining_eq_indices = [i for i in range(num_eq) if i not in used_eq_indices]
    eq_perm_indices.extend(remaining_eq_indices)

    if len(eq_perm_indices) != num_eq:
        raise ValueError(f"Equation permutation construction failed. Length mismatch: {len(eq_perm_indices)} vs {num_eq}")
    if len(set(eq_perm_indices)) != num_eq:
         raise ValueError("Equation permutation construction failed. Indices not unique.")

    print(f"\nEquation permutation indices (new row i <- old row eq_perm_indices[i]): {eq_perm_indices}")

    # --- Reorder Symbolic Matrices ---
    print("\n--- Reordering Symbolic Matrices (Stationary Model) ---")
    # Use extract based on permutations derived
    sympy_A_ord = sympy_A_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_B_ord = sympy_B_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_D_ord = sympy_D_quad.extract(eq_perm_indices, list(range(num_shocks))) # Rows reordered

    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}
    print("Symbolic reordering complete.")

    # --- Lambdify ---
    print("\n--- Lambdifying Ordered Matrices (Stationary Model) ---")
    param_sym_list = [param_syms[p] for p in param_names] # Ensure consistent order

    func_A = robust_lambdify(param_sym_list, sympy_A_ord)
    func_B = robust_lambdify(param_sym_list, sympy_B_ord)
    func_C = robust_lambdify(param_sym_list, sympy_C_ord)
    func_D = robust_lambdify(param_sym_list, sympy_D_ord)
    print("Lambdification successful.")

    return (func_A, func_B, func_C, func_D,
            ordered_vars, shock_names, param_names, param_assignments,
            symbolic_matrices_ordered, initial_info)

# --- SDA Solver and Q Computation ---

def solve_quadratic_matrix_equation(A, B, C, initial_guess=None, tol=1e-12, max_iter=100, verbose=False):
    """Solves A X^2 + B X + C = 0 for X using the SDA algorithm."""
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n) or C.shape != (n, n):
        raise ValueError("Input matrices A, B, C must be square and conformable.")
    if initial_guess is None:
        initial_guess = np.zeros_like(A)
    elif initial_guess.shape != (n, n):
        raise ValueError("Initial guess must match matrix dimensions.")

    # Initialization based on Binder & Pesaran (1995) / SGU representation alignment
    # We want to solve A P^2 + B P + C = 0
    # Algorithm 1 solves Gamma_1 X^2 + Gamma_0 X + Gamma_{-1} = 0
    # Match: Gamma_1 = A, Gamma_0 = B, Gamma_{-1} = C

    # Initial setup (matches Algorithm 1 notation with our matrices)
    E = C.copy() # E = Gamma_{-1}
    F_init = A.copy() # F_init = Gamma_1
    Bbar = B.copy() # Bbar = Gamma_0
    Bbar += A @ initial_guess # Adjust B based on initial guess: Bbar = B + A * X_guess

    try:
        lu_Bbar, piv_Bbar = lu_factor(Bbar)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"SDA Error: Factorization of (B + A*X_guess) failed. Cond: {np.linalg.cond(Bbar):.2e}. Error: {e}")
        return None, 0, np.inf

    # Calculate E0 = -(B + A*Xg)^-1 * C
    # Calculate F0 = -(B + A*Xg)^-1 * A
    try:
        E0 = -lu_solve((lu_Bbar, piv_Bbar), E)
        F0 = -lu_solve((lu_Bbar, piv_Bbar), F_init)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"SDA Error: Initial solve for E0/F0 failed. Error: {e}")
        return None, 0, np.inf

    # Initialize iterates (matching Algorithm 1 notation)
    Xk = E0  # X_0 = E_0
    Yk = F0  # Y_0 = F_0
    Ek = E0  # E_0 = E_0
    Fk = F0  # F_0 = F_0

    X_new = np.zeros_like(Xk); Y_new = np.zeros_like(Yk)
    E_new = np.zeros_like(Ek); F_new = np.zeros_like(Fk)
    I = np.eye(n, dtype=A.dtype); solved = False; iter_count = 0; relative_diff = np.inf

    for i in range(1, max_iter + 1):
        iter_count = i
        # Precompute factors (I - Yk @ Xk) and (I - Xk @ Yk)
        try: M1 = I - Yk @ Xk; lu_M1, piv_M1 = lu_factor(M1)
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Factor (I-YX) iter {i}. Cond:{np.linalg.cond(M1):.2e}"); return Xk + initial_guess, i, np.inf
        try: M2 = I - Xk @ Yk; lu_M2, piv_M2 = lu_factor(M2)
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Factor (I-XY) iter {i}. Cond:{np.linalg.cond(M2):.2e}"); return Xk + initial_guess, i, np.inf

        # Update E, F, X, Y using SDA updates (Algorithm 1)
        try: temp_E = lu_solve((lu_M1, piv_M1), Ek); E_new = Ek @ temp_E
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc E_new iter {i}"); return Xk + initial_guess, i, np.inf
        try: temp_F = lu_solve((lu_M2, piv_M2), Fk); F_new = Fk @ temp_F
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc F_new iter {i}"); return Xk + initial_guess, i, np.inf
        try: temp_X = Xk @ Ek; temp_X = lu_solve((lu_M2, piv_M2), temp_X); X_new = Xk + Fk @ temp_X
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc X_new iter {i}"); return Xk + initial_guess, i, np.inf
        try: temp_Y = Yk @ Fk; temp_Y = lu_solve((lu_M1, piv_M1), temp_Y); Y_new = Yk + Ek @ temp_Y
        except (ValueError, np.linalg.LinAlgError): print(f"SDA Error: Calc Y_new iter {i}"); return Xk + initial_guess, i, np.inf

        # Convergence check (relative change in X)
        X_diff_norm = norm(X_new - Xk, ord='fro')
        X_norm = norm(X_new, ord='fro')
        relative_diff = X_diff_norm / (X_norm + 1e-12)
        if verbose: print(f"Iter {i}: Rel Change X = {relative_diff:e}")
        if relative_diff < tol: solved = True; break

        # Update iterates for next loop
        Xk, Yk, Ek, Fk = X_new, Y_new, E_new, F_new

    # Final solution P = X_final + X_guess
    X_sol = X_new + initial_guess if solved else Xk + initial_guess

    # Final residual check
    residual = A @ (X_sol @ X_sol) + B @ X_sol + C
    residual_norm = norm(residual, 'fro')
    term_norms = norm(A @ X_sol @ X_sol, 'fro') + norm(B @ X_sol, 'fro') + norm(C, 'fro')
    residual_ratio = residual_norm / (term_norms + 1e-15) # Relative to sum of term norms

    if not solved:
        print(f"SDA Warning: Did not converge within {max_iter} iterations.")
        print(f" Final relative change in X: {relative_diff:.2e}, Residual ratio: {residual_ratio:.2e}")
    elif verbose:
         print(f"SDA Converged in {iter_count} iterations. Final residual ratio: {residual_ratio:.2e}")

    return X_sol, iter_count, residual_ratio

def compute_Q(A, B, D, P):
    """
    Computes Q for y_t = P y_{t-1} + Q e_t, given A P^2 + B P + C = 0.
    Requires solving (A P + B) Q = D. (Note: D here has opposite sign
    from the derivation dF/de, hence no minus sign needed below).
    A, B, P are (n x n), D is (n x n_shock). Q will be (n x n_shock).
    """
    n = A.shape[0]
    n_shock = D.shape[1]
    if A.shape != (n, n) or B.shape != (n, n) or P.shape != (n, n) or D.shape[0] != n:
        raise ValueError("Dimension mismatch in compute_Q inputs.")

    APB = A @ P + B
    try:
        Q = np.linalg.solve(APB, D)
    except np.linalg.LinAlgError:
        cond_num = np.linalg.cond(APB)
        print(f"Cannot compute Q: Matrix (A P + B) might be singular. Condition number: {cond_num:.2e}")
        # Try pseudo-inverse as a fallback? Might indicate deeper issues.
        # pinv_APB = np.linalg.pinv(APB)
        # Q = pinv_APB @ D
        # print("Attempted computation with pseudo-inverse.")
        return None # Return None to indicate failure

    if Q.shape != (n, n_shock):
         raise RuntimeError(f"Computed Q has unexpected shape {Q.shape}. Expected ({n}, {n_shock}).")

    return Q

# --- Trend and Measurement Equation Parsing Functions ---

def extract_trend_declarations(model_string):
    """Extracts trend variables and trend shocks."""
    trend_vars = []
    trend_shocks = []
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)

    # Find trend_model block to limit search scope
    trend_model_marker = re.search(r'\btrend_model\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    search_area = processed_content
    if trend_model_marker:
        search_area = processed_content[:trend_model_marker.start()]

    # Extract trends_vars
    match_tv = re.search(r'(?i)\btrends_vars\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match_tv:
        content = match_tv.group(1).replace('\n', ' ').strip()
        trend_vars = [v for v in re.split(r'[,\s]+', content) if v and re.fullmatch(r'[a-zA-Z_]\w*', v)]
        trend_vars = list(dict.fromkeys(trend_vars).keys()) # Unique, ordered

    # Extract varexo_trends
    match_vt = re.search(r'(?i)\bvarexo_trends\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match_vt:
        content = match_vt.group(1).replace('\n', ' ').strip()
        trend_shocks = [s for s in re.split(r'[,\s]+', content) if s and re.fullmatch(r'[a-zA-Z_]\w*', s)]
        trend_shocks = list(dict.fromkeys(trend_shocks).keys()) # Unique, ordered

    return trend_vars, trend_shocks

def extract_trend_equations(model_string):
    """Extracts equations from the 'trend_model; ... end;' block."""
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)

    match = re.search(r'(?i)\btrend_model\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match:
        print("Warning: 'trend_model; ... end;' block not found.")
        return []

    content = match.group(1)
    eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]

    processed_equations = []
    for line in eq_raw:
        # Handle potentially non-standard syntax like G_TREND(-1) on LHS
        # Standardize to VAR(t) = expression_involving(t-1)_and_shocks(t)
        if '=' in line:
            parts = line.split('=', 1)
            lhs, rhs = parts[0].strip(), parts[1].strip()
            # Check if LHS has a time shift, if so, assume it means VAR(t)
            lhs_match = re.match(r'([a-zA-Z_]\w*)\s*(\(\s*([+-]?\d+)\s*\))?', lhs)
            if lhs_match:
                base_lhs = lhs_match.group(1)
                # If LHS is like G_TREND(-1), rewrite eq as G_TREND = RHS (with adjustments)
                # This requires careful parsing of RHS too. Simpler: assume standard syntax.
                if lhs_match.group(2) and lhs_match.group(3) != '0':
                    print(f"Warning: Trend equation '{line}' has non-standard LHS '{lhs}'. Assuming it defines '{base_lhs}(t)'. Check model definition.")
                    processed_equations.append(f"({base_lhs}) - ({rhs})")
                else: # Standard LHS (just variable name, implies time t)
                    processed_equations.append(f"({base_lhs}) - ({rhs})")
            else:
                 print(f"Warning: Skipping malformed trend equation line: '{line}' - Cannot parse LHS")
        else:
            print(f"Warning: Trend equation '{line}' has no '='. Assuming 'expr = 0'.")
            processed_equations.append(line)

    return processed_equations

def extract_observation_declarations(model_string):
    """Extracts observable variable names."""
    obs_vars = []
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)

    # Find measurement_equations block to limit search scope
    meas_eq_marker = re.search(r'\bmeas(?:urement)?_equations\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    search_area = processed_content
    if meas_eq_marker:
        search_area = processed_content[:meas_eq_marker.start()]

    match = re.search(r'(?i)\bvarobs\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).replace('\n', ' ').strip()
        obs_vars = [v for v in re.split(r'[,\s]+', content) if v and re.fullmatch(r'[a-zA-Z_]\w*', v)]
        obs_vars = list(dict.fromkeys(obs_vars).keys()) # Unique, ordered

    return obs_vars

def extract_measurement_equations(model_string):
    """Extracts equations from the 'measument_equations; ... end;' block."""
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)

    match = re.search(r'(?i)\bmeas(?:urement)?_equations\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match:
        print("Warning: 'measument_equations; ... end;' block not found.")
        return []

    content = match.group(1)
    eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]

    processed_equations = []
    for line in eq_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts
                # Form: ObsVar - (expression involving states/trends) = 0
                processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else:
                 print(f"Warning: Skipping malformed measurement equation line: '{line}'")
        else:
             print(f"Warning: Measurement equation '{line}' has no '='. Assuming 'expr = 0'.")
             processed_equations.append(line)

    return processed_equations

def extract_trend_shock_stderrs(model_string):
    """Extracts standard errors from the 'trend_shocks; ... end;' block."""
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)

    match = re.search(r'(?i)\btrend_shocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match:
        # print("Info: 'trend_shocks; ... end;' block not found.") # Optional info message
        return {} # Return empty dict if block is missing

    content = match.group(1)
    # Example line: var SHK_X; stderr 0.1;
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)

    for m in stderr_matches:
        shock_name = m.group(1)
        stderr_val_str = m.group(2).strip()
        try:
            stderr_val = float(stderr_val_str)
            # Define parameter name convention, e.g., sigma_SHK_X
            param_name = f"sigma_{shock_name}"
            stderrs[param_name] = stderr_val
        except ValueError:
            print(f"Warning: Could not parse stderr value '{stderr_val_str}' for trend shock '{shock_name}'. Skipping.")

    return stderrs

# --- State-Space Building Functions ---

def build_trend_matrices(trend_equations, trend_vars, trend_shocks, param_names, param_assignments):
    """
    Builds symbolic P_trends and Q_trends matrices.
    Assumes trend model is VAR(1)-like: T(t) = P_trends * T(t-1) + Q_trends * shocks(t).
    Handles contemporaneous definitions by substitution or exclusion from state.
    """
    print("\n--- Building Trend State-Space Matrices (P_trends, Q_trends) ---")

    # Identify trend variables defined contemporaneously (RHS has time t vars)
    contemporaneous_defs = {}
    state_trend_vars = [] # Trends that evolve based on t-1 state
    defining_equations_for_state_trends = []

    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    base_var_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')

    print("Analyzing trend equations for state definition...")
    for eq_str in trend_equations:
        eq_parts = eq_str.split('-', 1) # Assumes "LHS - (RHS) = 0" form
        if len(eq_parts) != 2: continue
        lhs_str = eq_parts[0].strip().strip('()')
        rhs_str = eq_parts[1].strip().strip('()')

        if lhs_str not in trend_vars:
            print(f"Warning: LHS '{lhs_str}' in trend eq '{eq_str}' is not a declared trend variable. Skipping.")
            continue

        # Check RHS for time (t) dependencies on *other* trend vars
        rhs_symbols = set(base_var_regex.findall(rhs_str))
        has_contemporaneous_rhs = False
        for sym in rhs_symbols:
            if sym in trend_vars and sym != lhs_str:
                # Check if this sym appears without (-1)
                explicit_lag = f"{sym}(-1)"
                if sym in rhs_str and explicit_lag not in rhs_str:
                     has_contemporaneous_rhs = True
                     break
        # Check for time shifts other than -1 on RHS
        for match in var_time_regex.finditer(rhs_str):
            base, shift = match.group(1), int(match.group(2))
            if base in trend_vars and shift != -1:
                has_contemporaneous_rhs = True
                break

        if has_contemporaneous_rhs:
            print(f"  Trend '{lhs_str}' defined contemporaneously: {eq_str}. Will handle in observation matrix.")
            contemporaneous_defs[lhs_str] = rhs_str # Store RHS expression
        else:
            # This variable is part of the backward-looking state
            if lhs_str not in state_trend_vars:
                 state_trend_vars.append(lhs_str)
            defining_equations_for_state_trends.append(eq_str)

    if not state_trend_vars:
        print("Warning: No state trend variables identified (all might be contemporaneous?). Returning empty matrices.")
        # Return empty lambdified functions that return empty arrays
        return lambda *a: np.empty((0,0)), lambda *a: np.empty((0,0)), [], {}

    print(f"Identified state trend variables ({len(state_trend_vars)}): {state_trend_vars}")
    num_state_trends = len(state_trend_vars)
    num_trend_shocks = len(trend_shocks)

    # Create symbols for state trends (t, t-1), shocks (t), parameters
    param_syms = {p: sympy.symbols(p) for p in param_names}
    trend_shock_syms = {s: sympy.symbols(s) for s in trend_shocks}
    trend_var_syms = {} # {'var': {'m1': sym, 't': sym}}
    all_syms = set(param_syms.values()) | set(trend_shock_syms.values())
    for var in state_trend_vars:
        sym_m1 = create_timed_symbol(var, -1)
        sym_t  = create_timed_symbol(var, 0)
        trend_var_syms[var] = {'m1': sym_m1, 't': sym_t}
        all_syms.update([sym_m1, sym_t])
    # Add symbols for any trends appearing on RHS of state trend equations
    # (even if defined contemporaneously themselves, their t-1 value might be needed)
    for var in trend_vars:
         if var not in trend_var_syms:
             sym_m1 = create_timed_symbol(var, -1)
             sym_t = create_timed_symbol(var, 0) # May not be used directly
             trend_var_syms[var] = {'m1': sym_m1, 't': sym_t}
             all_syms.add(sym_m1)
             all_syms.add(sym_t)


    local_dict = {str(s): s for s in all_syms}
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                          implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))

    # Parse only the defining equations for the state trends
    sym_trend_equations = []
    print("Parsing state trend equations into symbolic form...")
    map_state_trend_to_eq_idx = {} # Track which eq defines which state var
    eq_idx_counter = 0

    # Ensure equations list matches state vars list order/presence
    parsed_eq_for_state_var = {var: None for var in state_trend_vars}
    final_sym_equations_ordered = []


    for i, eq_str in enumerate(defining_equations_for_state_trends):
        eq_parts = eq_str.split('-', 1)
        lhs_var = eq_parts[0].strip().strip('()')

        if lhs_var not in state_trend_vars: continue # Should not happen based on filtering

        eq_str_sym = eq_str
        # Substitute var(time) -> var_m1, var_t etc.
        def replace_trend_time(match):
            base, shift_str = match.groups()
            shift = int(shift_str)
            if base in trend_shock_syms:
                if shift == 0: return str(trend_shock_syms[base])
                else: raise ValueError(f"Trend shock {base}({shift}) invalid.")
            elif base in trend_var_syms:
                 # For state trends, only t-1 allowed on RHS; t on LHS.
                 # For other trends, t-1 allowed on RHS.
                 if shift == -1: return str(trend_var_syms[base]['m1'])
                 if shift == 0 and base == lhs_var: return str(trend_var_syms[base]['t']) # LHS var at time t
                 # Any other shift or time t var on RHS is an error here (should be contemporaneous)
                 raise ValueError(f"Unexpected term {base}({shift}) in state trend eq for {lhs_var}.")
            elif base in param_syms:
                raise ValueError(f"Parameter {base}({shift}) invalid.")
            else: # Unknown symbol - treat symbolically but warn
                # print(f"Warning: Undeclared symbol {base}({shift}) in trend eq '{eq_str}'.")
                sym = sympy.symbols(f"{base}_t{shift:+}")
                if str(sym) not in local_dict: local_dict[str(sym)] = sym
                return str(sym)

        eq_str_sym = var_time_regex.sub(replace_trend_time, eq_str_sym)

        # Replace remaining base names (implicitly t for LHS, error for RHS)
        all_base_names = sorted(list(trend_var_syms.keys()) + param_names + list(trend_shock_syms.keys()), key=len, reverse=True)
        for name in all_base_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name == lhs_var: # LHS base name is time t
                 replacement = str(trend_var_syms[name]['t'])
            elif name in trend_var_syms: # Any other base name on RHS is error (must be t-1)
                 # Need a check to ensure it's not implicitly time t on RHS
                 # This requires careful regex, maybe skip this replace and rely on timed version
                 continue # Safer to rely on explicit (-1) notation parsing
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in trend_shock_syms: replacement = str(trend_shock_syms[name]) # time t
            else: continue
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)

        try:
            # Parse the fully substituted string: LHS(t) - RHS(t-1, shocks_t) = 0
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            # Store based on the state variable it defines
            if parsed_eq_for_state_var[lhs_var] is not None:
                 print(f"Warning: Multiple defining equations found for state trend var '{lhs_var}'. Using the last one.")
            parsed_eq_for_state_var[lhs_var] = sym_eq

        except Exception as e:
            print(f"\nError parsing state trend equation: '{eq_str}' -> '{eq_str_sym}'")
            print(f"Local dict keys: {sorted(local_dict.keys())}")
            print(f"Sympy error: {e}")
            raise

    # Assemble equations in the order of state_trend_vars
    for var in state_trend_vars:
         eq = parsed_eq_for_state_var[var]
         if eq is None:
             raise RuntimeError(f"Could not find or parse defining equation for state trend variable '{var}'.")
         final_sym_equations_ordered.append(eq)

    print("Symbolic parsing of state trend equations completed.")

    # Build symbolic matrices P_trends, Q_trends
    # T(t) = P_trends * T(t-1) + Q_trends * shocks_t
    # Our equations are: Var(t) - RHS(t-1, shocks_t) = 0
    # => Var(t) = RHS(...)
    # P_trends[i,j] = d(RHS_i) / d(Var_j(t-1)) = - d(Eq_i) / d(Var_j(t-1))  (since Eq = Var(t) - RHS)
    # Q_trends[i,k] = d(RHS_i) / d(Shock_k(t)) = - d(Eq_i) / d(Shock_k(t))

    sympy_P_trends = sympy.zeros(num_state_trends, num_state_trends)
    sympy_Q_trends = sympy.zeros(num_state_trends, num_trend_shocks)

    state_trend_m1_syms = [trend_var_syms[v]['m1'] for v in state_trend_vars]
    trend_shock_t_syms = [trend_shock_syms[s] for s in trend_shocks]

    for i, eq in enumerate(final_sym_equations_ordered):
        for j, var_m1 in enumerate(state_trend_m1_syms):
            sympy_P_trends[i, j] = -sympy.diff(eq, var_m1)
        for k, shock_t in enumerate(trend_shock_t_syms):
            sympy_Q_trends[i, k] = -sympy.diff(eq, shock_t)

    print("Symbolic P_trends and Q_trends matrices generated.")
    # print("Symbolic P_trends:\n", sympy_P_trends)
    # print("Symbolic Q_trends:\n", sympy_Q_trends)


    # Lambdify
    print("Lambdifying trend matrices...")
    param_sym_list = [param_syms[p] for p in param_names] # Use consistent order
    func_P_trends = robust_lambdify(param_sym_list, sympy_P_trends)
    func_Q_trends = robust_lambdify(param_sym_list, sympy_Q_trends)
    print("Lambdification successful.")

    return func_P_trends, func_Q_trends, state_trend_vars, contemporaneous_defs

def build_observation_matrix(measurement_equations, obs_vars, stationary_vars,
                             trend_state_vars, contemporaneous_trend_defs,
                             param_names, param_assignments):
    """
    Builds the symbolic observation matrix Omega.
    Omega maps the augmented state [stationary_vars(t), trend_state_vars(t)]
    to observable variables obs_vars(t).
    Handles substitution of contemporaneous trend definitions.
    obs(t) = Omega * [stat_vars(t), trend_state_vars(t)] + H * xi(t) (Assume H=0)
    Equations are parsed as: obs_var(t) - RHS(...) = 0
    Omega[i, j] = d(RHS_i) / d(state_j(t)) = - d(Eq_i) / d(state_j(t))
    """
    print("\n--- Building Observation Matrix (Omega) ---")
    num_obs = len(obs_vars)
    num_stationary = len(stationary_vars)
    num_trend_state = len(trend_state_vars)
    num_augmented_state = num_stationary + num_trend_state

    if num_obs == 0:
        print("No observable variables declared. Returning empty Omega.")
        return lambda *a: np.empty((0, num_augmented_state)), []

    if len(measurement_equations) != num_obs:
         raise ValueError(f"Number of measurement equations ({len(measurement_equations)}) does not match number of varobs ({num_obs}).")

    # Create symbols for states (t), observables (t), parameters
    param_syms = {p: sympy.symbols(p) for p in param_names}
    obs_syms = {v: sympy.symbols(v) for v in obs_vars}
    # Need symbols for *all* variables that might appear on RHS (stationary and all trends)
    all_rhs_vars = stationary_vars + trend_state_vars + list(contemporaneous_trend_defs.keys())
    rhs_var_syms = {v: sympy.symbols(v) for v in all_rhs_vars} # Time t symbols

    all_syms = set(param_syms.values()) | set(obs_syms.values()) | set(rhs_var_syms.values())
    local_dict = {str(s): s for s in all_syms}
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                          implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))

    sym_measurement_equations = []
    print("Parsing measurement equations into symbolic form...")
    parsed_eq_for_obs_var = {var: None for var in obs_vars}

    for eq_str in measurement_equations:
        eq_parts = eq_str.split('-', 1)
        if len(eq_parts) != 2: continue
        lhs_var = eq_parts[0].strip().strip('()')

        if lhs_var not in obs_vars:
            print(f"Warning: LHS '{lhs_var}' in meas eq '{eq_str}' is not a declared varobs. Skipping.")
            continue

        # Substitute contemporaneous trend definitions into RHS *before* parsing fully
        rhs_str = eq_parts[1].strip().strip('()')
        rhs_processed = rhs_str
        # Iteratively substitute until no more substitutions are possible
        for _ in range(len(contemporaneous_trend_defs) + 1): # Limit iterations
            made_substitution = False
            for contemp_var, contemp_expr_str in contemporaneous_trend_defs.items():
                pattern = r'\b' + re.escape(contemp_var) + r'\b'
                if re.search(pattern, rhs_processed):
                     rhs_processed = re.sub(pattern, f"({contemp_expr_str})", rhs_processed)
                     made_substitution = True
            if not made_substitution: break # Exit loop if no changes in a pass
        # Reconstruct the equation string for parsing
        eq_str_subbed = f"{lhs_var} - ({rhs_processed})"

        # Now parse the substituted equation string
        # Replace base names with time t symbols
        eq_str_sym = eq_str_subbed
        all_base_names_obs = sorted(list(obs_syms.keys()) + list(rhs_var_syms.keys()) + param_names, key=len, reverse=True)
        for name in all_base_names_obs:
             pattern = r'\b' + re.escape(name) + r'\b'
             if name in obs_syms: replacement = str(obs_syms[name])
             elif name in rhs_var_syms: replacement = str(rhs_var_syms[name])
             elif name in param_syms: replacement = str(param_syms[name])
             else: continue
             eq_str_sym = re.sub(pattern, replacement, eq_str_sym)

        try:
            # Parse: ObsVar(t) - RHS(stat_vars(t), trend_state_vars(t), params) = 0
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            if parsed_eq_for_obs_var[lhs_var] is not None:
                 print(f"Warning: Multiple measurement equations found for obs var '{lhs_var}'. Using the last one.")
            parsed_eq_for_obs_var[lhs_var] = sym_eq
        except Exception as e:
            print(f"\nError parsing measurement equation: '{eq_str}' -> subbed '{eq_str_subbed}' -> sym '{eq_str_sym}'")
            print(f"Local dict keys: {sorted(local_dict.keys())}")
            print(f"Sympy error: {e}")
            raise

    # Assemble equations in the order of obs_vars
    final_sym_equations_ordered = []
    for var in obs_vars:
         eq = parsed_eq_for_obs_var[var]
         if eq is None:
             raise RuntimeError(f"Could not find or parse measurement equation for observable variable '{var}'.")
         final_sym_equations_ordered.append(eq)
    print("Symbolic parsing of measurement equations completed.")


    # Build symbolic Omega matrix
    sympy_Omega = sympy.zeros(num_obs, num_augmented_state)
    # Augmented state vector order: [stationary_vars(t), trend_state_vars(t)]
    augmented_state_syms = [rhs_var_syms[v] for v in stationary_vars] + \
                           [rhs_var_syms[v] for v in trend_state_vars]

    for i, eq in enumerate(final_sym_equations_ordered):
        for j, state_sym in enumerate(augmented_state_syms):
            sympy_Omega[i, j] = -sympy.diff(eq, state_sym)

    print("Symbolic Omega matrix generated.")
    # print("Symbolic Omega:\n", sympy_Omega)

    # Lambdify
    print("Lambdifying observation matrix...")
    param_sym_list = [param_syms[p] for p in param_names] # Consistent order
    func_Omega = robust_lambdify(param_sym_list, sympy_Omega)
    print("Lambdification successful.")

    return func_Omega, obs_vars

def build_augmented_state_space(P_stat, Q_stat, P_trend, Q_trend, Omega_func,
                                stat_vars, trend_state_vars, obs_vars,
                                stat_shocks, trend_shocks, param_args):
    """
    Combines stationary and trend dynamics into an augmented state-space.

    Returns numerical matrices P_aug, Q_aug, Omega.
    P_stat, Q_stat, P_trend, Q_trend are numerical matrices.
    Omega_func is a lambda function.
    param_args is the list of parameter values.
    """
    print("\n--- Building Augmented State-Space System ---")

    n_stat = P_stat.shape[0]
    n_trend = P_trend.shape[0]
    n_aug = n_stat + n_trend
    n_stat_shock = Q_stat.shape[1]
    n_trend_shock = Q_trend.shape[1]
    n_aug_shock = n_stat_shock + n_trend_shock
    n_obs = len(obs_vars)

    # Build P_aug: block diagonal [P_stat, P_trend]
    P_aug = block_diag(P_stat, P_trend)

    # Build Q_aug: block diagonal [Q_stat, Q_trend]
    # Need to handle cases where one block is empty (e.g., no trends)
    if n_stat > 0 and n_trend > 0:
        # SciPy's block_diag handles this directly if inputs are 2D
        Q_aug = block_diag(Q_stat, Q_trend)
    elif n_stat > 0: # Only stationary part
        Q_aug = Q_stat
    elif n_trend > 0: # Only trend part
        Q_aug = Q_trend
    else: # Empty system
        Q_aug = np.empty((0, 0))

    # Ensure Q_aug dimensions are correct if blocks were combined
    if Q_aug.shape != (n_aug, n_aug_shock):
        # This might happen if one set of shocks is empty but the other is not
        # We need to manually construct Q_aug with zero blocks
        Q_aug = np.zeros((n_aug, n_aug_shock), dtype=P_stat.dtype)
        if n_stat > 0 and n_stat_shock > 0:
            Q_aug[:n_stat, :n_stat_shock] = Q_stat
        if n_trend > 0 and n_trend_shock > 0:
            Q_aug[n_stat:, n_stat_shock:] = Q_trend

    # Evaluate Omega
    Omega = Omega_func(*param_args)
    if Omega.shape != (n_obs, n_aug):
        raise ValueError(f"Evaluated Omega matrix has wrong shape {Omega.shape}. Expected ({n_obs}, {n_aug}).")

    # Define orderings
    augmented_state_vars = stat_vars + trend_state_vars
    augmented_shocks = stat_shocks + trend_shocks

    print(f"Augmented state dimension: {n_aug}")
    print(f"Augmented shock dimension: {n_aug_shock}")
    print(f"Observable dimension: {n_obs}")

    return (P_aug, Q_aug, Omega,
            augmented_state_vars, augmented_shocks, obs_vars)

def plot_irfs(irf_values, var_names, horizon, title="Impulse Responses"):
    """ Simple IRF plotting function """
    num_vars = irf_values.shape[1]
    if num_vars == 0:
        print(f"No variables to plot for: {title}")
        return

    # Dynamically determine grid layout
    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols

    plt.figure(figsize=(min(5*cols, 18), 3*rows)) # Adjust figsize
    plt.suptitle(title, fontsize=14)
    time = range(horizon)

    for i, var_name in enumerate(var_names):
        plt.subplot(rows, cols, i + 1)
        plt.plot(time, irf_values[:, i], label=var_name)
        plt.axhline(0, color='black', linewidth=0.7, linestyle=':') # Zero line
        plt.title(var_name)
        plt.grid(True, linestyle='--', alpha=0.6)
        # Add x-label only to bottom row plots for clarity
        if i >= num_vars - cols:
             plt.xlabel("Horizon")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

# --- IRF Calculation Functions ---

def irf(P, Q, shock_index, horizon=40):
    """
    Compute impulse responses for y_t = P y_{t-1} + Q e_t,
    for a specific shock index (unit shock).
    P is (n x n), Q is (n x n_shock).
    Returns (horizon x n) array of state responses.
    """
    n = P.shape[0]
    n_shock = Q.shape[1]
    if shock_index < 0 or shock_index >= n_shock:
        raise ValueError(f"shock_index must be between 0 and {n_shock-1}")

    y_resp = np.zeros((horizon, n))
    e0 = np.zeros((n_shock, 1))
    e0[shock_index] = 1.0 # Unit shock

    # y_0 = P*y_{-1} + Q*e_0. Assume y_{-1} = 0.
    y_current = Q @ e0
    y_resp[0, :] = y_current.flatten()

    # Subsequent periods: e_t = 0 for t > 0
    for t in range(1, horizon):
        y_current = P @ y_current
        y_resp[t, :] = y_current.flatten()

    # Set very small values to zero for cleaner output
    y_resp[np.abs(y_resp) < 1e-14] = 0.0
    return y_resp

def irf_observables(P_aug, Q_aug, Omega, shock_index, horizon=40):
    """
    Compute impulse responses for observable variables.
    obs(t) = Omega * state_aug(t)
    shock_index refers to the index in the augmented shock vector.
    Returns (horizon x n_obs) array of observable responses.
    """
    n_aug = P_aug.shape[0]
    n_aug_shock = Q_aug.shape[1]
    n_obs = Omega.shape[0]

    if shock_index < 0 or shock_index >= n_aug_shock:
         raise ValueError(f"Augmented shock_index must be between 0 and {n_aug_shock-1}")
    if Omega.shape[1] != n_aug:
         raise ValueError(f"Omega columns ({Omega.shape[1]}) must match P_aug dimension ({n_aug}).")

    # 1. Compute IRF for the augmented state vector
    state_irf = irf(P_aug, Q_aug, shock_index, horizon) # shape (horizon, n_aug)

    # 2. Map state responses to observable responses: obs_irf = state_irf @ Omega^T
    # Need to transpose Omega because state_irf has time on rows, state on columns
    obs_irf = state_irf @ Omega.T # shape (horizon, n_obs)

     # Set very small values to zero
    obs_irf[np.abs(obs_irf) < 1e-14] = 0.0
    return obs_irf


# --- Main Execution ---
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # --- Use the full model file provided by user ---
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn") # Make sure this file exists

        if not os.path.exists(mod_file_path):
             raise FileNotFoundError(f"Model file not found at: {mod_file_path}")

        print(f"Reading full model definition from: {mod_file_path}")
        with open(mod_file_path, 'r') as f:
            model_def = f.read()

        # --- STEP 1: Parse Stationary Model ---
        (func_A, func_B, func_C, func_D,
         ordered_stat_vars, stat_shocks, param_names_stat, param_assignments_stat,
         sym_matrices_ord_stat, _) = parse_lambdify_and_order_model(model_def)

        # --- STEP 2: Parse Trend Components ---
        print("\n--- Parsing Trend and Observation Components ---")
        trend_vars, trend_shocks = extract_trend_declarations(model_def)
        trend_equations = extract_trend_equations(model_def)
        obs_vars = extract_observation_declarations(model_def)
        measurement_equations = extract_measurement_equations(model_def)
        trend_stderr_params = extract_trend_shock_stderrs(model_def) # dict: {sigma_SHK_X: val}

        print(f"Declared Trend Variables: {trend_vars}")
        print(f"Declared Trend Shocks: {trend_shocks}")
        print(f"Parsed Trend Equations ({len(trend_equations)}):")
        # for eq in trend_equations: print(f"  {eq}")
        print(f"Declared Observation Variables: {obs_vars}")
        print(f"Parsed Measurement Equations ({len(measurement_equations)}):")
        # for eq in measurement_equations: print(f"  {eq}")
        print(f"Parsed Trend Shock Stderr Parameters: {trend_stderr_params}")

        # --- STEP 3: Combine Parameters ---
        # Combine parameters declared in 'parameters' block and those from 'trend_shocks' stderr
        all_param_names = list(dict.fromkeys(param_names_stat + list(trend_stderr_params.keys())).keys())
        all_param_assignments = param_assignments_stat.copy()
        all_param_assignments.update(trend_stderr_params) # Add/overwrite with stderr values

        print(f"\nCombined Parameter List ({len(all_param_names)}): {all_param_names}")
        print(f"Combined Parameter Assignments: {all_param_assignments}")

        # --- STEP 4: Build Trend and Observation Matrices ---
        # Note: Need to pass combined param list/values to builders
        (func_P_trends, func_Q_trends,
        ordered_trend_state_vars, contemp_trend_defs) = build_trend_matrices(
            trend_equations, trend_vars, trend_shocks, all_param_names, all_param_assignments
        )

        (func_Omega, ordered_obs_vars) = build_observation_matrix(
            measurement_equations, obs_vars, ordered_stat_vars,
            ordered_trend_state_vars, contemp_trend_defs,
            all_param_names, all_param_assignments
        )

        # --- STEP 5: Evaluate Numerical Matrices ---
        test_param_values = all_param_assignments.copy()
        # Add defaults if needed (using example defaults for stationary part)
        default_test_values = {
            'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1,
            'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
            'rho_L_GDP_GAP': 0.75, 'rho_DLA_CPI': 0.75,
            'rho_rs': 0.75, 'rho_rs2': 0.01 # From example, note this was 0.1 in python code before
        }
        # Add defaults for sigma parameters if not parsed (e.g., set to 1 or 0)
        for shk in trend_shocks:
            pname = f"sigma_{shk}"
            if pname not in test_param_values:
                default_val = 1.0 if "SHK" in shk else 0.0 # Heuristic: assume measurement shocks 0? Use 1 for now.
                default_test_values[pname] = default_val
                print(f"  Note: Trend shock std err param '{pname}' not found in trend_shocks block. Using default: {default_val}")

        missing_params = []
        test_args = []
        print("\n--- Evaluating ALL Matrices with Combined Parameters ---")
        for p in all_param_names:
            if p in test_param_values:
                test_args.append(test_param_values[p])
                # print(f"  Using value for {p}: {test_param_values[p]}")
            elif p in default_test_values:
                test_args.append(default_test_values[p])
                print(f"  Warning: Param '{p}' not assigned. Using default value: {default_test_values[p]}.")
            else:
                missing_params.append(p)
                print(f"  ERROR: Param '{p}' declared but no value assigned or default. Using 0.0.")
                test_args.append(0.0) # Fallback

        if missing_params:
            print(f"\nERROR: Missing parameter values: {missing_params}. Cannot proceed.")
        else:
            # --- Create argument list specifically for stationary functions ---
            # Use the order from param_names_stat which was used for lambdification
            stat_test_args = []
            print("\nExtracting arguments for stationary model functions...")
            for p_stat in param_names_stat:
                if p_stat in test_param_values:
                    stat_test_args.append(test_param_values[p_stat])
                    # print(f"  Adding {p_stat} = {test_param_values[p_stat]}")
                else:
                    # This case should ideally not happen if defaults worked correctly
                    raise ValueError(f"Internal Error: Parameter '{p_stat}' needed for stationary model not found in evaluated values.")
            print(f"Extracted {len(stat_test_args)} arguments for stationary functions.")

            print("\nEvaluating numerical matrices A, B, C, D (Stationary)...")
            # --- Call stationary functions with the correct number of arguments ---
            A_num_stat = func_A(*stat_test_args)
            B_num_stat = func_B(*stat_test_args)
            C_num_stat = func_C(*stat_test_args)
            D_num_stat = func_D(*stat_test_args)

            # --- STEP 6: Solve Stationary Model ---
            print("\n--- Solving Stationary Quadratic Matrix Equation: A P^2 + B P + C = 0 ---")
            P_sol_stat, iter_count, residual_ratio = solve_quadratic_matrix_equation(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, verbose=False
            )

            if P_sol_stat is None:
                print("\nERROR: Stationary quadratic solver failed to compute P.")
            else:
                print(f"Solver iterations: {iter_count}, final residual ratio: {residual_ratio:.2e}")
                if residual_ratio > 1e-6: print("Warning: Solver residual ratio is high.")

                try:
                    eigenvalues = np.linalg.eigvals(P_sol_stat)
                    max_eig = np.max(np.abs(eigenvalues))
                    print(f"Maximum eigenvalue magnitude of P_stationary: {max_eig:.6f}")
                    if max_eig >= 1.0 - 1e-9: print("Warning: Stationary solution P might be unstable.")
                    else: print("Stationary solution P appears stable.")
                except np.linalg.LinAlgError: print("Warning: Could not compute eigenvalues of P_stationary.")

                print("\n--- Computing Q matrix (Stationary) ---")
                Q_sol_stat = compute_Q(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)

                if Q_sol_stat is None:
                    print("ERROR: Failed to compute Q_stationary.")
                else:
                    print("Q_stationary matrix computed successfully.")

                    # --- STEP 7: Evaluate Trend Matrices ---
                    print("\nEvaluating numerical matrices P_trends, Q_trends...")
                    P_num_trend = func_P_trends(*test_args)
                    Q_num_trend = func_Q_trends(*test_args)

                    # Scale Q_trends by standard deviations if desired
                    # Q_trends = Q_trends @ diag(sigmas_trend)
                    trend_sigmas = []
                    for shk in trend_shocks:
                        pname = f"sigma_{shk}"
                        if pname in all_param_assignments:
                            trend_sigmas.append(all_param_assignments[pname])
                        else:
                            trend_sigmas.append(1.0) # Default if not specified
                            print(f"Warning: Using std dev=1 for trend shock {shk}")

                    if Q_num_trend.size > 0 and len(trend_sigmas) == Q_num_trend.shape[1]:
                        print("Scaling Q_trends by standard deviations.")
                        Q_num_trend_scaled = Q_num_trend @ np.diag(trend_sigmas)
                    else:
                        Q_num_trend_scaled = Q_num_trend # Use unscaled if shapes mismatch or empty

                    # Optional: Scale Q_stat by std dev params if they exist (usually not explicit)
                    # For now, assume unit shocks for stationary part or incorporate into param file if needed.
                    Q_sol_stat_scaled = Q_sol_stat # Assuming unit shocks for stationary model

                    # --- STEP 8: Build Augmented System ---
                    (P_aug, Q_aug, Omega_num,
                    aug_state_vars, aug_shocks, obs_vars_ordered) = build_augmented_state_space(
                        P_sol_stat, Q_sol_stat_scaled, # Use scaled Q_stat if available
                        P_num_trend, Q_num_trend_scaled, # Use scaled Q_trend
                        func_Omega,
                        ordered_stat_vars, ordered_trend_state_vars, obs_vars,
                        stat_shocks, trend_shocks, test_args
                    )

                    print("\n--- Augmented System Matrices (Numerical) ---")
                    with np.printoptions(precision=3, suppress=True, linewidth=120):
                        print("P_augmented:\n", P_aug)
                        print("\nQ_augmented (scaled by std devs):\n", Q_aug)
                        print("\nOmega:\n", Omega_num)
                        print("\nAugmented State Variables:", aug_state_vars)
                        print("Augmented Shocks:", aug_shocks)
                        print("Observable Variables:", obs_vars_ordered)


                    # --- STEP 9: Compute and Plot IRFs ---
                    print("\n--- Computing Impulse Response Functions (Augmented System) ---")
                    horizon = 40

                    # Example: Choose shock - find index in the *augmented* shock list
                    shock_name_to_plot = "SHK_RS" # Example: Stationary shock
                    # shock_name_to_plot = "SHK_L_GDP_TREND" # Example: Trend shock

                    if shock_name_to_plot not in aug_shocks:
                        print(f"ERROR: Selected shock '{shock_name_to_plot}' not found in augmented shock list: {aug_shocks}")
                    else:
                        shock_index_aug = aug_shocks.index(shock_name_to_plot)
                        print(f"Computing IRFs for shock: {shock_name_to_plot} (index {shock_index_aug} in augmented list)")

                        # --- IRFs for Augmented State Variables ---
                        irf_states_aug = irf(P_aug, Q_aug, shock_index=shock_index_aug, horizon=horizon)

                        print(f"\n--- Plotting IRFs for Augmented State Variables ---")
                        # Select subset of augmented state variables to plot
                        state_vars_to_plot = list(dict.fromkeys(ordered_stat_vars + ordered_trend_state_vars)) # All unique states
                        # Or select a subset:
                        # state_vars_to_plot = ["L_GDP_GAP", "DLA_CPI", "RS", "L_GDP_TREND", "PI_TREND"]

                        valid_state_indices = [aug_state_vars.index(v) for v in state_vars_to_plot if v in aug_state_vars]
                        valid_state_names = [v for v in state_vars_to_plot if v in aug_state_vars]

                        if valid_state_names:
                            plt.figure(figsize=(15, 10))
                            plt.suptitle(f"Impulse Responses (Augmented State) to a {shock_name_to_plot} Shock (Std Dev Scaled)", fontsize=14)
                            num_plots = len(valid_state_names)
                            cols = 4 if num_plots > 9 else 3
                            rows = (num_plots + cols - 1) // cols
                            for i, var_name in enumerate(valid_state_names):
                                idx = valid_state_indices[i]
                                plt.subplot(rows, cols, i + 1)
                                plt.plot(range(horizon), irf_states_aug[:, idx], label=f'{var_name}')
                                plt.axhline(0, color='black', linewidth=0.7, linestyle=':')
                                plt.title(f"{var_name}")
                                plt.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            plt.show()
                        else:
                            print("No valid state variables selected for plotting.")


                        # --- IRFs for Observable Variables ---
                        if obs_vars_ordered:
                            irf_observables_vals = irf_observables(P_aug, Q_aug, Omega_num, shock_index=shock_index_aug, horizon=horizon)

                            print(f"\n--- Plotting IRFs for Observable Variables ---")
                            plt.figure(figsize=(12, 6))
                            plt.suptitle(f"Impulse Responses (Observables) to a {shock_name_to_plot} Shock (Std Dev Scaled)", fontsize=14)
                            num_plots = len(obs_vars_ordered)
                            cols = 3 if num_plots > 4 else 2
                            rows = (num_plots + cols - 1) // cols
                            for i, var_name in enumerate(obs_vars_ordered):
                                plt.subplot(rows, cols, i + 1)
                                plt.plot(range(horizon), irf_observables_vals[:, i], label=f'{var_name}')
                                plt.axhline(0, color='black', linewidth=0.7, linestyle=':')
                                plt.title(f"{var_name}")
                                plt.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            plt.show()
                        else:
                            print("\nNo observable variables to plot.")

    # --- Error Handling ---
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