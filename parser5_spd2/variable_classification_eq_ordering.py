import sympy
import re
import collections

# --- Inputs (Copied from your example) ---

# Target order for variables/equations
state_vars_ordered_names = [
    'RES_DLA_CPI', 'RES_L_GDP_GAP', 'RES_RS', 'aux_RES_RS_lag1',
    'DLA_CPI', 'L_GDP_GAP', 'RS', 'RR_GAP',
    'aux_DLA_CPI_lead1', 'aux_DLA_CPI_lead2'
]

# Unordered equations (as strings for easier parsing in standalone)
# Note: In your class, you'll have these as Sympy objects already
equations_str = [
    "L_GDP_GAP - L_GDP_GAP_m1*b1 + L_GDP_GAP_p1*(b1 - 1) - RES_L_GDP_GAP + RR_GAP_p1*b4", # Eq 0 -> L_GDP_GAP
    "DLA_CPI - DLA_CPI_m1*a1 + DLA_CPI_p1*(a1 - 1) - L_GDP_GAP*a2 - RES_DLA_CPI",       # Eq 1 -> DLA_CPI
    "-RES_RS + RS - RS_m1*g1 + (g1 - 1)*(DLA_CPI_p1 + L_GDP_GAP*g3 + aux_DLA_CPI_lead2_p1*g2)", # Eq 2 -> RS
    "DLA_CPI_p1 + RR_GAP - RS",                                                              # Eq 3 -> RR_GAP
    "RES_L_GDP_GAP - RES_L_GDP_GAP_m1*rho_L_GDP_GAP - SHK_L_GDP_GAP",                      # Eq 4 -> RES_L_GDP_GAP
    "RES_DLA_CPI - RES_DLA_CPI_m1*rho_DLA_CPI - SHK_DLA_CPI",                              # Eq 5 -> RES_DLA_CPI
    "RES_RS - RES_RS_m1*rho_rs - SHK_RS - aux_RES_RS_lag1_m1*rho_rs2",                     # Eq 6 -> RES_RS
    "-DLA_CPI_p1 + aux_DLA_CPI_lead1",                                                     # Eq 7 -> aux_DLA_CPI_lead1
    "-aux_DLA_CPI_lead1_p1 + aux_DLA_CPI_lead2",                                           # Eq 8 -> aux_DLA_CPI_lead2
    "-RES_RS_m1 + aux_RES_RS_lag1"                                                         # Eq 9 -> aux_RES_RS_lag1
]

# --- Symbol Generation (Necessary for parsing) ---
# Extract all potential base names and create symbols
all_symbols = set()
base_vars = set(['L_GDP_GAP', 'DLA_CPI', 'RS', 'RR_GAP', 'RES_L_GDP_GAP', 'RES_DLA_CPI', 'RES_RS'])
aux_vars = set(['aux_DLA_CPI_lead1', 'aux_DLA_CPI_lead2', 'aux_RES_RS_lag1'])
params = set(['b1', 'b4', 'a1', 'a2', 'g1', 'g2', 'g3', 'rho_DLA_CPI', 'rho_L_GDP_GAP', 'rho_rs', 'rho_rs2'])
shocks = set(['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS'])

# Add base names
all_symbols.update(base_vars)
all_symbols.update(aux_vars)
all_symbols.update(params)
all_symbols.update(shocks)

# Add potential leads/lags (up to 2 for this example)
for var in base_vars.union(aux_vars):
    for k in range(1, 3):
        all_symbols.add(f"{var}_p{k}")
        all_symbols.add(f"{var}_m{k}")

# Create sympy symbols
symbols_dict = {name: sympy.Symbol(name) for name in all_symbols}

# --- Parse Equations ---
unordered_equations_sympy = []
print("Parsing equations...")
for i, eq_str in enumerate(equations_str):
    try:
        # Assume equations are already in LHS = 0 form implicitly
        lhs_expr = sympy.parse_expr(eq_str, local_dict=symbols_dict, evaluate=False)
        unordered_equations_sympy.append(sympy.Eq(lhs_expr, 0))
        print(f"  Parsed Eq {i}: {sympy.sstr(lhs_expr, full_prec=False)} = 0")
    except Exception as e:
        print(f"Error parsing equation {i} ('{eq_str}'): {e}")
        # Handle error appropriately, maybe exit or skip
        exit()
print("-" * 20)

# --- Reordering Logic ---

# Heuristic function to identify the main variable (at time t) in an equation
# This is specific to the structure observed in the example
def get_main_variable_from_eq(equation, all_target_vars_set):
    """Finds the variable symbol appearing without _p or _m suffix."""
    contemporaneous_vars = []
    for sym in equation.lhs.free_symbols:
        # Check if it's one of our target variables and appears without suffix
        if sym.name in all_target_vars_set and not re.search(r'_[pm]\d+$', sym.name):
             # Special check for aux vars - they are the main var if they appear
             if sym.name.startswith("aux_"):
                 return sym.name # Aux vars define their own equations
             contemporaneous_vars.append(sym.name)

    # For non-aux equations, we need a clearer rule based on the example:
    # - If only one contemporaneous var, that's it.
    # - If multiple (like RS and RES_RS in Eq 2), need a tie-breaker.
    #   In Eq 2, RS seems primary. In Eq 3, RR_GAP seems primary.
    #   Let's refine based on known structure for this example:
    eq_str_repr = str(equation.lhs) # Rough check
    if 'L_GDP_GAP - L_GDP_GAP_m1' in eq_str_repr: return 'L_GDP_GAP'
    if 'DLA_CPI - DLA_CPI_m1' in eq_str_repr: return 'DLA_CPI'
    if 'RS - RS_m1' in eq_str_repr: return 'RS' # Matches Eq 2
    if 'RR_GAP - RS' in eq_str_repr: return 'RR_GAP' # Matches Eq 3
    if 'RES_L_GDP_GAP - RES_L_GDP_GAP_m1' in eq_str_repr: return 'RES_L_GDP_GAP'
    if 'RES_DLA_CPI - RES_DLA_CPI_m1' in eq_str_repr: return 'RES_DLA_CPI'
    if 'RES_RS - RES_RS_m1' in eq_str_repr and 'aux_RES_RS_lag1' in eq_str_repr: return 'RES_RS'

    # Fallback: if exactly one contemporaneous var found among targets
    if len(contemporaneous_vars) == 1:
         return contemporaneous_vars[0]

    # If logic fails for an equation
    # print(f"Warning: Could not determine main variable for eq: {equation}")
    return None


# Build the mapping from variable name to equation
equation_map = {}
used_indices = set()
all_targets_set = set(state_vars_ordered_names)

print("Building variable-to-equation map...")
for i, eq in enumerate(unordered_equations_sympy):
    main_var = get_main_variable_from_eq(eq, all_targets_set)
    if main_var:
        if main_var in equation_map:
            print(f"  Warning: Variable '{main_var}' seems to be defined by multiple equations (Eq {equation_map[main_var][0]} and Eq {i}). Using first match.")
        else:
            print(f"  Mapping: Equation {i} -> Variable '{main_var}'")
            equation_map[main_var] = (i, eq) # Store original index and equation
            used_indices.add(i)
    else:
         print(f"  Warning: Could not identify main variable for Equation {i}: {sympy.sstr(eq.lhs, full_prec=False)}")

# Check if all equations were mapped
if len(used_indices) != len(unordered_equations_sympy):
    print("Error: Not all original equations were mapped to a variable!")
    unmapped_indices = set(range(len(unordered_equations_sympy))) - used_indices
    print(f"Unmapped equation indices: {unmapped_indices}")
print("-" * 20)

# Create the ordered list
ordered_equations_sympy = []
print("Reordering equations based on state variable order...")
for var_name in state_vars_ordered_names:
    if var_name in equation_map:
        original_index, eq = equation_map[var_name]
        ordered_equations_sympy.append(eq)
        print(f"  Adding Eq {original_index} (for '{var_name}') to ordered list.")
    else:
        print(f"Error: No equation found in map for variable '{var_name}'!")
        # Handle error: maybe append a placeholder or raise Exception
        ordered_equations_sympy.append(None) # Placeholder for missing
print("-" * 20)

# --- Output ---
print("\n--- Original Unordered Equations ---")
for i, eq in enumerate(unordered_equations_sympy):
    print(f"Eq {i}: {sympy.sstr(eq.lhs, full_prec=False)} = 0")

print("\n--- Target State Variable Order ---")
print(state_vars_ordered_names)

print("\n--- Reordered Equations (Should match target order) ---")
if len(ordered_equations_sympy) == len(state_vars_ordered_names):
    for i, eq in enumerate(ordered_equations_sympy):
        # --- THIS IS THE CORRECTED LINE ---
        if eq is not None:
        # --- END CORRECTION ---
             main_var_check = get_main_variable_from_eq(eq, all_targets_set) # Re-check mapping
             print(f"Ordered Eq {i} (Target: '{state_vars_ordered_names[i]}', Found Main Var: '{main_var_check}'): {sympy.sstr(eq.lhs, full_prec=False)} = 0")
        else:
             print(f"Ordered Eq {i} (Target: '{state_vars_ordered_names[i]}'): MISSING EQUATION")
else:
    print(f"Error: Length mismatch! Ordered equations: {len(ordered_equations_sympy)}, Target variables: {len(state_vars_ordered_names)}")