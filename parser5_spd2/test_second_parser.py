import re
import sympy
import numpy as np
from collections import OrderedDict
import copy

def create_timed_symbol(base_name, time_shift):
    """Creates a sympy symbol with a time suffix."""
    if time_shift == -1:
        return sympy.symbols(f"{base_name}_m1")
    elif time_shift == 1:
        return sympy.symbols(f"{base_name}_p1")
    elif time_shift == 0:
        return sympy.symbols(base_name)
    else:
        return sympy.symbols(f"{base_name}_t{time_shift:+}")


def extract_declarations(model_string):
    """
    Extracts variables, shocks, and parameters using regex to find blocks.
    Handles comma/space separation and terminating semicolons robustly.
    Correctly extracts only parameter names, ignoring assignments.
    """
    declarations = {'var': [], 'varexo': [], 'parameters': []}

    # --- Pre-processing ---
    # 1. Remove block comments /* ... */
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    # 2. Remove line comments // ... and % ...
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)

    # --- Regex Extraction ---
    block_matches = re.finditer(r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters|model)\b|$)', processed_content, re.DOTALL)

    def process_block_content(content_str, block_type):
        """Helper to clean and split names from block content string."""
        if not content_str: return []
        # Remove any terminating ';' and surrounding whitespace
        content = content_str.strip()
        if content.endswith(';'):
             content = content[:-1].strip()

        # Replace newlines possibly captured within content with spaces
        content = content.replace('\n', ' ')

        names = []
        if block_type == 'parameters':
            # Find valid identifiers that are likely parameter names
            # Regex: Find words starting with a letter or underscore,
            # followed by word characters, that are NOT immediately
            # preceded by an '=' sign (to avoid capturing values).
            # We look for names followed by ',', ';', '=', or whitespace/end.
            # This specifically targets the names themselves.
            potential_names = re.findall(r'\b([a-zA-Z_]\w*)\b', content)
            # Filter out potential keywords or numbers if necessary, although
            # the \b check helps. We primarily want to discard the values
            # from assignments. Let's refine the findall:
            names = re.findall(r'\b([a-zA-Z_]\w*)\b\s*(?:=.*?)?(?=[,;\s]|$)', content)
            # Explanation of refined regex:
            # \b([a-zA-Z_]\w*)\b : Capture a valid identifier (Group 1)
            # \s*               : Match optional whitespace
            # (?:=.*?)?         : Optionally match (non-capturingly) an equals sign
            #                     followed by anything non-greedily (the assignment value)
            # (?=[,;\s]|$)      : Positive lookahead: ensure the match is followed by
            #                     a comma, semicolon, whitespace, or end of string.
            #                     This helps delimit the parameter declaration/assignment.

            # Example: "b1 = 0.7; b2, b3;"
            # Should find 'b1', 'b2', 'b3'

        else: # For 'var' and 'varexo'
            # Split by comma or space, filter empty strings
            raw_names = re.split(r'[,\s]+', content)
            cleaned_names = [name.strip() for name in raw_names if name.strip()]
            names = cleaned_names

        # Remove duplicates while preserving order
        return list(dict.fromkeys(names).keys())


    for match in block_matches:
        block_keyword = match.group(1).lower()
        block_content_raw = match.group(2)
        names = process_block_content(block_content_raw, block_keyword)
        declarations[block_keyword].extend(names)

    # Final cleanup for duplicates across potential multiple declarations (unlikely but safe)
    final_declarations = {}
    for key in declarations:
        final_declarations[key] = list(dict.fromkeys(declarations[key]).keys())

    return final_declarations['var'], final_declarations['varexo'], final_declarations['parameters']


# --- Rest of the script remains the same ---

def extract_model_equations(model_string):
    """
    Extracts equations from the 'model; ... end;' block.
    Handles the semicolon immediately following 'model'.
    """
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n')
    cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)

    model_match = re.search(r'(?i)\bmodel\b\s*;(.*?)(\bend\b\s*;|\Z)', processed_content, re.DOTALL)
    if not model_match:
        raise ValueError("Could not find 'model; ... end;' block or 'model;' statement.")
    model_content = model_match.group(1)

    equations_raw = [eq.strip() for eq in model_content.split(';') if eq.strip()]
    processed_equations = []
    for line in equations_raw:
        if '=' in line:
            lhs, rhs = line.split('=', 1)
            processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
        else:
            print(f"Warning: Equation '{line}' has no '='. Assuming 'expr = 0'.")
            processed_equations.append(line)
    return processed_equations


def parse_lambdify_and_order_model(model_string):
    """
    Parses a model string with declarations, generates symbolic matrices,
    orders them, and returns lambdified functions.
    (Uses updated declaration/model extraction)
    """
    print("--- Parsing Model Declarations ---")
    declared_vars, shock_names, param_names = extract_declarations(model_string)
    if not declared_vars: raise ValueError("No variables declared in 'var' block.")
    if not shock_names: raise ValueError("No shocks declared in 'varexo' block.")

    print(f"Declared Variables: {declared_vars}")
    print(f"Declared Shocks: {shock_names}")
    print(f"Declared Parameters: {param_names}") # Should be correct now

    print("\n--- Parsing Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    print(f"Found {len(raw_equations)} equations in model block.")

    # --- Handling Leads/Lags & Auxiliaries ---
    print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars)
    aux_variables = OrderedDict()
    processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]; eq_idx += 1
        modified_eq = eq; matches = list(var_time_regex.finditer(eq))
        for match in reversed(matches):
            base_name = match.group(1); time_shift = int(match.group(2))
            if base_name not in endogenous_vars and base_name not in aux_variables:
                 if base_name in param_names or base_name in shock_names: continue
                 continue
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev = f"aux_{base_name}_lead_p{k-1}" if k > 1 else base_name
                        def_eq_str = f"{aux_name} - {prev}(+1)"; aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"; replacement = f"{target_aux}(+1)"
                start, end = match.span(); modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                for def_eq in aux_needed_defs:
                     if def_eq not in processed_equations: print(f"  Adding aux def: {def_eq} = 0"); processed_equations.append(def_eq)
            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                     aux_name = f"aux_{base_name}_lag_m{k}"
                     if aux_name not in aux_variables:
                          prev = f"aux_{base_name}_lag_m{k-1}" if k > 1 else base_name
                          def_eq_str = f"{aux_name} - {prev}(-1)"; aux_variables[aux_name] = def_eq_str
                          aux_needed_defs.append(def_eq_str)
                          if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"; replacement = f"{target_aux}(-1)"
                start, end = match.span(); modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                for def_eq in aux_needed_defs:
                     if def_eq not in processed_equations: print(f"  Adding aux def: {def_eq} = 0"); processed_equations.append(def_eq)
        if modified_eq != eq: processed_equations[eq_idx-1] = modified_eq
    initial_vars_ordered = list(endogenous_vars); num_vars = len(initial_vars_ordered); num_eq = len(processed_equations); num_shocks = len(shock_names)
    if num_vars != num_eq: raise ValueError(f"Model not square: {num_vars} vars vs {num_eq} eqs. Vars: {initial_vars_ordered}")
    print(f"\nFinal variable list ({num_vars}): {initial_vars_ordered}"); print(f"Final equation list ({num_eq})")

    # --- Symbolic Representation ---
    print("\n--- Creating Symbolic Representation ---")
    param_syms = {p: sympy.symbols(p) for p in param_names}; shock_syms = {s: sympy.symbols(s) for s in shock_names}
    var_syms = {}; all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in initial_vars_ordered:
        sym_m1 = create_timed_symbol(var, -1); sym_t = create_timed_symbol(var, 0); sym_p1 = create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}; all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])
    sym_equations = []; local_dict = {str(s): s for s in all_syms_for_parsing}
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, rationalize
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    for i, eq_str in enumerate(processed_equations):
        eq_str_sym = eq_str
        def replace_var_time(match):
            base_name = match.group(1); time_shift = int(match.group(2))
            if base_name in shock_names: return base_name
            elif base_name in var_syms:
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0: return str(var_syms[base_name]['t'])
                if time_shift == 1: return str(var_syms[base_name]['p1'])
                raise ValueError(f"Unhandled time shift {time_shift} for {base_name} in eq '{eq_str}'.")
            elif base_name in param_syms: return base_name
            else:
                 if base_name not in ['log', 'exp', 'sqrt', 'abs']:
                      print(f"Warning: Symbol '{base_name}' in eq {i} not declared. Treating as symbolic.");
                      if base_name not in local_dict: local_dict[base_name] = sympy.symbols(base_name)
                 return match.group(0)
        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)
        all_known_names = list(var_syms.keys()) + param_names + shock_names; sorted_names = sorted(all_known_names, key=len, reverse=True)
        for name in sorted_names:
             pattern = r'\b' + re.escape(name) + r'\b'; replacement = ''
             if name in var_syms: replacement = str(var_syms[name]['t'])
             elif name in param_syms: replacement = str(param_syms[name])
             elif name in shock_names: replacement = str(shock_syms[name])
             if replacement:
                  if name in eq_str_sym: eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        try:
            current_symbols_in_expr = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq_str_sym))
            for sym_str in current_symbols_in_expr:
                if sym_str not in local_dict and sym_str not in ['log', 'exp', 'sqrt', 'abs']:
                    try: float(sym_str); continue
                    except ValueError: pass
                    if '_m1' in sym_str or '_p1' in sym_str or '_t' in sym_str: continue
                    if sym_str not in param_names and sym_str not in shock_names and sym_str not in initial_vars_ordered:
                         print(f"Info: Adding symbol '{sym_str}' from eq {i} to local_dict."); local_dict[sym_str] = sympy.symbols(sym_str)
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations); sym_equations.append(sym_eq)
        except Exception as e: print(f"Error parsing eq {i}: '{eq_str}' -> '{eq_str_sym}'\nLocal dict keys: {sorted(list(local_dict.keys()))}\nSympy error: {e}"); raise

    # --- Generate Initial Symbolic Matrices ---
    print("\n--- Generating Initial Symbolic Matrices (A, B, C, D) ---")
    sympy_A = sympy.zeros(num_eq, num_vars); sympy_B = sympy.zeros(num_eq, num_vars); sympy_C = sympy.zeros(num_eq, num_vars); sympy_D = sympy.zeros(num_eq, num_shocks)
    for i, eq in enumerate(sym_equations):
        eq_expanded = sympy.expand(eq)
        for j, var in enumerate(initial_vars_ordered):
            sympy_A[i, j] = eq_expanded.coeff(var_syms[var]['m1']); sympy_B[i, j] = eq_expanded.coeff(var_syms[var]['t']); sympy_C[i, j] = eq_expanded.coeff(var_syms[var]['p1'])
        for k, shk in enumerate(shock_names): sympy_D[i, k] = eq_expanded.coeff(shock_syms[shk])
    initial_info = {'A': copy.deepcopy(sympy_A), 'B': copy.deepcopy(sympy_B),'C': copy.deepcopy(sympy_C), 'D': copy.deepcopy(sympy_D),'vars': list(initial_vars_ordered), 'eqs': list(processed_equations)}

    # --- Classify Variables ---
    print("\n--- Classifying Variables ---")
    exo_process_vars = [v for v in declared_vars if v.startswith("RES_")] # Base check on declared
    exo_defining_eq_indices = {}
    for var in exo_process_vars:
         if var not in initial_vars_ordered: continue
         j = initial_vars_ordered.index(var); found_eq = False
         for i in range(num_eq):
              coeff_t = sympy_B[i, j]; is_defining_coeff = False
              if isinstance(coeff_t, sympy.Number):
                   if coeff_t == 1 or coeff_t == -1: is_defining_coeff = True
              elif coeff_t is sympy.S.One or coeff_t is sympy.S.NegativeOne: is_defining_coeff = True
              if is_defining_coeff:
                   is_defining_row_b = all(sympy_B[i, col].is_zero for col in range(num_vars) if col != j)
                   if is_defining_row_b:
                        is_defining_row_c = sympy_C.row(i).is_zero_matrix
                        if is_defining_row_c:
                             only_self_lags = all(sympy_A[i, col].is_zero for col in range(num_vars) if col != j)
                             if only_self_lags: exo_defining_eq_indices[var] = i; found_eq = True; break
         if not found_eq: print(f"Warning: No defining eq found for exo var '{var}'.")
    backward_exo_vars = []; forward_backward_endo_vars = []; static_vars = []
    for j, var in enumerate(initial_vars_ordered):
        is_exo_related = False
        if var in exo_process_vars: is_exo_related = True
        elif var.startswith("aux_") and "_lag_" in var:
             match = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", var)
             if match:
                  parent_var = match.group(1);
                  if parent_var in exo_process_vars: is_exo_related = True
        if is_exo_related: backward_exo_vars.append(var); continue
        is_lagged_self = not sympy_A.col(j).is_zero_matrix; is_leaded_self = not sympy_C.col(j).is_zero_matrix
        if not is_lagged_self and not is_leaded_self: static_vars.append(var)
        else: forward_backward_endo_vars.append(var)
    print(f"\nCategorized Variables:\n  Backward Exogenous: {backward_exo_vars}\n  Forward/Backward Endogenous: {forward_backward_endo_vars}\n  Static: {static_vars}")

    # --- Determine New Variable Order ---
    ordered_vars = backward_exo_vars + forward_backward_endo_vars + static_vars
    if len(ordered_vars) != len(initial_vars_ordered): raise ValueError(f"Var reordering failed. Missing: {set(initial_vars_ordered)-set(ordered_vars)}, Extra: {set(ordered_vars)-set(initial_vars_ordered)}")
    print(f"\nNew Variable Order ({len(ordered_vars)}): {ordered_vars}"); var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]

    # --- Determine New Equation Order ---
    eq_perm_indices = []; used_eq_indices = set()
    # Function to find alternative unused equation
    def find_alt_eq(used_set, num_equations):
        for i in range(num_equations):
            if i not in used_set: return i
        return -1 # No unused equation found

    for var_type_list, var_type_name in [(backward_exo_vars, "backward_exo"), (forward_backward_endo_vars, "forward/backward"), (static_vars, "static")]:
        for var in var_type_list:
            assigned_eq = -1
            old_var_idx = initial_vars_ordered.index(var)
            # Priority 1: Identified defining equation (for exo)
            if var in exo_defining_eq_indices and exo_defining_eq_indices[var] not in used_eq_indices:
                assigned_eq = exo_defining_eq_indices[var]
            # Priority 2: Equation matching original variable index
            elif old_var_idx < num_eq and old_var_idx not in used_eq_indices:
                 assigned_eq = old_var_idx
            # Priority 3: Any other unused equation
            else:
                 alt_eq = find_alt_eq(used_eq_indices, num_eq)
                 if alt_eq != -1:
                      assigned_eq = alt_eq
                      print(f"Warning: Assigning fallback unused equation {assigned_eq} to {var_type_name} var {var}")
                 else: # Should not happen if square
                      print(f"Error: No unused equation available for {var_type_name} var {var} when trying fallback.")

            if assigned_eq != -1 and assigned_eq not in used_eq_indices:
                eq_perm_indices.append(assigned_eq); used_eq_indices.add(assigned_eq)
            elif assigned_eq != -1: # Equation was already used (e.g., identified exo eq index == another var's original index)
                 print(f"Warning: Equation {assigned_eq} intended for {var} was already used. Finding alternative.")
                 alt_eq = find_alt_eq(used_eq_indices, num_eq)
                 if alt_eq != -1:
                      eq_perm_indices.append(alt_eq); used_eq_indices.add(alt_eq)
                      print(f"Warning: Using alternative unused equation {alt_eq} for {var_type_name} var {var}")
                 else:
                      print(f"Error: Could not find alternative unused equation for {var_type_name} var {var}")
            else: # Should not happen if assigned_eq remained -1
                 print(f"Error: Failed to assign an equation for {var_type_name} var {var}")

    if len(eq_perm_indices) != num_eq or len(used_eq_indices) != num_eq: raise ValueError(f"Eq permutation invalid. Length {len(eq_perm_indices)}, Used {len(used_eq_indices)}, NumEq {num_eq}")
    print(f"\nEquation permutation: {eq_perm_indices}")

    # --- Reorder Symbolic Matrices ---
    print("\n--- Reordering Symbolic Matrices ---")
    sympy_A_ord = sympy_A.extract(eq_perm_indices, var_perm_indices); sympy_B_ord = sympy_B.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C.extract(eq_perm_indices, var_perm_indices); sympy_D_ord = sympy_D.extract(eq_perm_indices, list(range(num_shocks)))
    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}; print("Symbolic reordering complete.")

    # --- Lambdify ---
    print("\n--- Lambdifying Ordered Matrices ---")
    param_sym_list = [param_syms[p] for p in param_names]
    try:
        func_A = sympy.lambdify(param_sym_list, sympy_A_ord, modules='numpy'); func_B = sympy.lambdify(param_sym_list, sympy_B_ord, modules='numpy')
        func_C = sympy.lambdify(param_sym_list, sympy_C_ord, modules='numpy'); func_D = sympy.lambdify(param_sym_list, sympy_D_ord, modules='numpy')
        print("Lambdification successful.")
    except Exception as e: print(f"Error during lambdification: {e}"); raise

    return (func_A, func_B, func_C, func_D, ordered_vars, shock_names, param_names, symbolic_matrices_ordered, initial_info)


# # --- Example Usage ---
# model_def = """
# var
#     // Main variables
#     L_GDP_GAP, DLA_CPI RS RR_GAP // Mixed comma/space separation
#     // Exogenous states declared as variables
#     RES_L_GDP_GAP
#     RES_DLA_CPI
#     RES_RS  
#     ; // Terminating semicolon for the block

# varexo SHK_L_GDP_GAP SHK_DLA_CPI SHK_RS; // Space separated, terminating semicolon

# parameters b1, b4, a1, a2, g1, g2, g3, rho_DLA_CPI, rho_L_GDP_GAP, rho_rs, rho_rs2; // Comma separated, terminating semicolon

#     // Parameter values defined later - parser should only get names
#     b1 = 0.7;          // Output persistence
#     b4 = 0.7;          // MCI weight
#     a1 = 0.5;          // Inflation persistence
#     a2 = 0.1;          // RMC passthrough
#     g1 = 0.7;         // Interest rate smoothing
#     g2 = 0.3;          // Inflation response
#     g3 = 0.25;         // Output gap response
#     rho_L_GDP_GAP  =0.75;
#     rho_DLA_CPI   =0.75;
#     rho_rs      =0.7;
#     rho_rs2      =0.1;

# model; // Semicolon immediately after model
#     // Aggregate demand: L_GDP_GAP - ( (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*RR_GAP(+1) + RES_L_GDP_GAP ) = 0
#     L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + temp_lag- b4*RR_GAP(+1) + RES_L_GDP_GAP;


#     // Core Inflation: DLA_CPI - ( a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI ) = 0
#     DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;


#     // Monetary policy reaction function: RS - ( g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS ) = 0
#     RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS;


#     // Definition: RR_GAP - ( RS - DLA_CPI(+1) ) = 0
#     RR_GAP = RS - DLA_CPI(+1);

#     // Exogenous process: RES_L_GDP_GAP - ( rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP ) = 0
#     RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;

#     // Exogenous process: RES_DLA_CPI - ( rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI ) = 0
#     RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI;

#     // Exogenous process: RES_RS - ( rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) + SHK_RS ) = 0
#     RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) + SHK_RS;


# end; // Semicolon after end
# """
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.getcwd()
mod_file_path = os.path.join(script_dir, 'qpm_model.dyn')
if not os.path.isfile(mod_file_path):
    raise FileNotFoundError(f"Mod file not found: {mod_file_path}")
with open(mod_file_path, 'r') as f:
    model_def = f.read()


try:
    (func_A, func_B, func_C, func_D,
     ordered_vars, shocks, param_names,
     sym_matrices_ord, initial_info) = parse_lambdify_and_order_model(model_def)

    print("\n\n--- Results ---")
    print("\nParameter Names (for function arguments):", param_names)
    print("\nShock Names:", shocks)
    print("\nFinal Ordered Variables:", ordered_vars)

    # --- Test the lambdified functions ---
    test_param_values = {
        'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
        'rho_L_GDP_GAP': 0.75, 'rho_DLA_CPI': 0.75, 'rho_rs': 0.75, 'rho_rs2': 0.1
    }
    test_args = []
    missing_params = []
    for p in param_names:
         if p in test_param_values: test_args.append(test_param_values[p])
         else:
              missing_params.append(p)
              print(f"Warning: Param '{p}' not in test values. Using 0.0.")
              test_args.append(0.0)

    if not missing_params:
        print("\n--- Testing Lambdified Functions with Example Parameters ---")
        A_num = func_A(*test_args)
        B_num = func_B(*test_args)
        C_num = func_C(*test_args)
        D_num = func_D(*test_args)
        with np.printoptions(precision=3, suppress=True, linewidth=120):
            print("\nNumerical A (ordered):\n", A_num)
            print("\nNumerical B (ordered):\n", B_num)
            print("\nNumerical C (ordered):\n", C_num)
            print("\nNumerical D (ordered):\n", D_num)
    else:
        print("\nSkipping numerical evaluation due to missing parameter values.")

except Exception as e:
     print(f"\nAn error occurred: {e}")
     import traceback
     traceback.print_exc()