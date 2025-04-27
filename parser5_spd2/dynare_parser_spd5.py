
import sympy
import re
import numpy as np
import os
import pickle
import collections
import sys
import importlib.util
import datetime

# Keep the time_shift_expression function as it is in your original code
def time_shift_expression(expr, shift, parser_symbols, var_names_set):
    # ... (Keep the original implementation from dynare_parser_spd5.py)
    if shift == 0:
        return expr
    subs_dict = {}
    atoms = expr.free_symbols
    for atom in atoms:
        atom_name = atom.name
        base_name = None
        current_k = 0
        is_var_type = False
        match_aux_lead = re.match(r"(aux_\w+_lead)(\d+)$", atom_name, re.IGNORECASE)
        match_aux_lag  = re.match(r"(aux_\w+_lag)(\d+)$", atom_name, re.IGNORECASE)
        match_lead = re.match(r"(\w+)_p(\d+)$", atom_name)
        match_lag  = re.match(r"(\w+)_m(\d+)$", atom_name)
        if match_aux_lead:
            base_name = match_aux_lead.group(1)
            current_k = int(match_aux_lead.group(2))
            is_var_type = True
        elif match_aux_lag:
            base_name = match_aux_lag.group(1)
            current_k = -int(match_aux_lag.group(2))
            is_var_type = True
        elif match_lead:
            base_name_cand = match_lead.group(1)
            # Check if it's a real variable lead, not aux_something_p1
            if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                base_name = base_name_cand
                current_k = int(match_lead.group(2))
                is_var_type = True
            # Handle aux leads like aux_VAR_lead1_p1 -> aux_VAR_lead2
            elif base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                 base_name = base_name_cand # Keep the aux name
                 current_k = int(match_lead.group(2))
                 is_var_type = True
            else:
                base_name = None
        elif match_lag:
            base_name_cand = match_lag.group(1)
            # Check if it's a real variable lag, not aux_something_m1
            if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                base_name = base_name_cand
                current_k = -int(match_lag.group(2))
                is_var_type = True
            # Handle aux lags like aux_VAR_lag1_m1 -> aux_VAR_lag2
            elif base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                 base_name = base_name_cand # Keep the aux name
                 current_k = -int(match_lag.group(2))
                 is_var_type = True
            else:
                base_name = None
        elif atom_name in var_names_set:
            base_name = atom_name
            current_k = 0
            is_var_type = True

        if is_var_type:
            new_k = current_k + shift
            # Determine the correct base name (strip aux_ prefix if needed for target name)
            clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
            if clean_base_match:
                clean_base = clean_base_match.group(1)
            else:
                clean_base = base_name # Could be original var or an aux var itself

            # Construct the new symbol name based on the target time index new_k
            if new_k == 0:
                 # If the original was an aux var like aux_VAR_lead1 and shift is -1, target is VAR
                if base_name.startswith("aux_") and clean_base in var_names_set:
                     new_sym_name = clean_base
                # If the original was a base var like VAR, target is VAR
                elif base_name in var_names_set and not base_name.startswith("aux_"):
                     new_sym_name = base_name
                # If original was aux_VAR_lag1 and shift is +1, target is VAR
                elif base_name.startswith("aux_") and clean_base in var_names_set:
                     new_sym_name = clean_base
                # If original was already an aux_ variable name itself (not _p/_m)
                elif base_name in parser_symbols and base_name.startswith("aux_"):
                     new_sym_name = base_name # No shift if target is t=0 for an aux var itself
                else:
                     # Default or error case? Maybe keep the base_name if it's an aux var name
                     # print(f"Warning: Ambiguous shift to t=0 for {atom_name} with shift {shift}. Resulting base: {base_name}")
                     new_sym_name = base_name # Fallback, might be incorrect if symbol doesn't exist
            elif new_k > 0:
                # If the base is an aux name like 'aux_VAR_lead1', shifting +1 makes 'aux_VAR_lead1_p1'
                if base_name.startswith("aux_"):
                    new_sym_name = f"{base_name}_p{new_k}"
                # If the base is a regular variable 'VAR', shifting +1 makes 'VAR_p1'
                else:
                    new_sym_name = f"{clean_base}_p{new_k}"
            else: # new_k < 0
                # If the base is an aux name like 'aux_VAR_lag1', shifting -1 makes 'aux_VAR_lag1_m1'
                if base_name.startswith("aux_"):
                     new_sym_name = f"{base_name}_m{abs(new_k)}"
                # If the base is a regular variable 'VAR', shifting -1 makes 'VAR_m1'
                else:
                     new_sym_name = f"{clean_base}_m{abs(new_k)}"

            # Ensure the target symbol exists or create it
            if new_sym_name not in parser_symbols:
                # print(f"Debug: Creating shifted symbol '{new_sym_name}' from '{atom_name}' (base='{base_name}', clean='{clean_base}') with shift {shift}, k={current_k}->{new_k}")
                parser_symbols[new_sym_name] = sympy.Symbol(new_sym_name)

            # Add to substitution dictionary
            subs_dict[atom] = parser_symbols[new_sym_name]

    # Apply substitutions
    try:
        # Using xreplace is generally safer for structural replacement
        shifted_expr = expr.xreplace(subs_dict)
    except Exception:
        # Fallback to subs if xreplace fails (e.g., with non-Sympy objects)
        try:
            shifted_expr = expr.subs(subs_dict)
        except Exception as e2:
            print(f"Error: Both xreplace and subs failed in time_shift_expression for expression: {expr} with subs {subs_dict}. Error: {e2}")
            # Return the original expression if substitution fails completely
            shifted_expr = expr

    return shifted_expr


class DynareParser:
    """
    Parses a Dynare-style .mod file, performs model reduction for UQME form,
    and optionally generates a Python function for numerical Jacobians.
    Allows specifying the state vector order.
    Adheres to standard Python style (no semicolons for line splitting).
    """
    def __init__(self, mod_file_path):
        """Initializes the parser by reading and parsing the .mod file."""
        self.mod_file_path = mod_file_path
        self.param_names = []
        self.var_names = []
        self.var_names_set = set()
        self.shock_names = []
        self.equations_str = []
        self.symbols = {}
        self.sympy_equations_original = []
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        self.static_subs = {}
        self.equations_after_static_elim = []
        self.equations_after_static_sub = []
        self.aux_lead_vars = {} # Stores aux_name -> symbol mapping
        self.aux_lag_vars = {}  # Stores aux_name -> symbol mapping
        self.aux_var_definitions = []
        self.equations_after_aux_sub = []
        self.final_dynamic_var_names = [] # List of names (strings)
        self.state_vars_ordered = []      # List of symbols in final order
        self.state_var_map = {}           # Map symbol -> index
        self.final_equations_for_jacobian = []
        self.last_param_values = {}
        self.forced_state_order = None    # Will store user-provided order if given

        print(f"--- Parsing Mod File: {self.mod_file_path} ---")
        self._parse_mod_file()
        self.var_names_set = set(self.var_names) # Create set after parsing
        print("\n--- Creating Initial Symbols ---")
        self._create_initial_sympy_symbols()
        print("\n--- Parsing Equations to Sympy ---")
        self._parse_equations_to_sympy()

        # Add aux var names to var_names_set *after* initial parsing if needed?
        # No, aux vars are created later. var_names_set should reflect only 'var' declared ones.

        print(f"Parser initialized. Vars:{len(self.var_names)}, Params:{len(self.param_names)}, Shocks:{len(self.shock_names)}")
        print(f"Original {len(self.sympy_equations_original)} equations parsed.")

    # Keep _parse_mod_file, _create_initial_sympy_symbols, _replace_dynare_timing,
    # _parse_equations_to_sympy, _analyze_variable_timing,
    # _identify_and_eliminate_static_vars, _substitute_static_vars,
    # _handle_aux_vars as they are in dynare_parser_spd5.py

    # ==================================================
    #  Previous methods (_parse_mod_file etc.) go here
    # ==================================================
    def _parse_mod_file(self):
        """Reads .mod file content and extracts declarations."""
        if not os.path.isfile(self.mod_file_path):
            raise FileNotFoundError(f"Mod file not found: {self.mod_file_path}")
        try:
            with open(self.mod_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise IOError(f"Error reading mod file {self.mod_file_path}: {e}") from e
        # Remove comments
        content = re.sub(r"//.*", "", content) # Single line comments
        content = re.sub(r"%.*", "", content) # Dynare % comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL) # Multi-line comments

        # Extract declarations using regex (case-insensitive, handling various spacing)
        var_pattern = re.compile(r"var\s+(.*?);", re.IGNORECASE | re.DOTALL)
        varexo_pattern = re.compile(r"varexo\s+(.*?);", re.IGNORECASE | re.DOTALL)
        parameters_pattern = re.compile(r"parameters\s+(.*?);", re.IGNORECASE | re.DOTALL)
        model_pattern = re.compile(r"model\s*(?:\(linear\))?\s*;\s*(.*?)\s*end\s*;", re.IGNORECASE | re.DOTALL) # Added optional (linear)

        # Find variables
        var_match = var_pattern.search(content)
        if var_match:
            # Split by space or comma, strip whitespace/commas from each item
            self.var_names = [v.strip().rstrip(',') for v in re.split(r'[,\s]+', var_match.group(1)) if v.strip()]
        else:
            print("Warning: 'var' block not found.")

        # Find exogenous variables (shocks)
        varexo_match = varexo_pattern.search(content)
        if varexo_match:
            self.shock_names = [s.strip().rstrip(',') for s in re.split(r'[,\s]+', varexo_match.group(1)) if s.strip()]
        else:
            print("Warning: 'varexo' block not found.")

        # Find parameters
        parameters_match = parameters_pattern.search(content)
        if parameters_match:
            self.param_names = [p.strip().rstrip(',') for p in re.split(r'[,\s]+', parameters_match.group(1)) if p.strip()]
        else:
            print("Warning: 'parameters' block not found.")

        # Find model block and extract equations
        model_match = model_pattern.search(content)
        if model_match:
            eq_block = model_match.group(1)
            # Split by semicolon, remove empty strings, strip whitespace
            raw_equations = [eq.strip() for eq in eq_block.split(';') if eq.strip()]
            # Further clean up potential tags like [name='...']
            self.equations_str = [re.sub(r"\[.*?\]", "", eq).strip() for eq in raw_equations if eq.strip()]
        else:
            raise ValueError("Model block ('model;...end;') not found or parsed correctly.")

        # --- Logging ---
        print(f"Found Variables ({len(self.var_names)}): {self.var_names}")
        print(f"Found Parameters ({len(self.param_names)}): {self.param_names}")
        print(f"Found Shocks ({len(self.shock_names)}): {self.shock_names}")
        print(f"Found {len(self.equations_str)} Equations (raw).")
        # Basic validation
        if not self.var_names: print("Warning: No variables declared.")
        if not self.equations_str: print("Warning: No equations found in model block.")


    def _create_initial_sympy_symbols(self):
        """Creates symbols for variables, parameters, and shocks."""
        all_names = self.var_names + self.param_names + self.shock_names
        for name in all_names:
            if name and name.isidentifier() and name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
            elif name in self.symbols:
                 # print(f"Debug: Symbol '{name}' already exists.")
                 pass # Symbol already exists, possibly from timing replacement
            else:
                # Provide more context for the warning
                source_list = "[unknown]"
                if name in self.var_names: source_list = "var"
                elif name in self.param_names: source_list = "parameters"
                elif name in self.shock_names: source_list = "varexo"
                print(f"Warning: Skipping invalid or empty name '{name}' found in '{source_list}' declaration.")

        # Add common constants if they aren't parameters
        common_consts = {'pi': sympy.pi}
        for const_name, const_sym in common_consts.items():
             if const_name not in self.symbols:
                  self.symbols[const_name] = const_sym

        print(f"Created {len(self.symbols)} initial symbols (incl. potentially pi).")


    def _replace_dynare_timing(self, eqstr):
        """
        Replaces VAR(+k) with VAR_pk and VAR(-k) with VAR_mk symbols
        in a single equation string. Also handles VAR, VAR(+0), VAR(-0).
        Ensures created symbols are added to self.symbols.
        """
        # Pattern for VAR(k), VAR(+k), VAR(-k) where k is an integer
        # Allows optional sign for k=0, e.g. Y(0), Y(+0), Y(-0) all become Y
        pat = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+\-]?)(\d+)\s*\)')

        output_str = eqstr
        replacements = [] # Store (start_index, end_index, new_symbol_name)
        needed_symbols = set() # Track new symbols needed

        for match in pat.finditer(eqstr):
            start_index, end_index = match.span()
            var_name, sign, num_str = match.groups()

            # Only replace if the base name is a declared variable
            if var_name in self.var_names_set:
                num = int(num_str)
                replacement_name = None

                if num == 0:
                    replacement_name = var_name # Y(0), Y(+0), Y(-0) -> Y
                elif sign == '+':
                    replacement_name = f"{var_name}_p{num}"
                    needed_symbols.add(replacement_name)
                elif sign == '-':
                    replacement_name = f"{var_name}_m{num}"
                    needed_symbols.add(replacement_name)
                else: # Sign is empty, implies current time if num > 0? Dynare requires sign for non-zero.
                      # Let's assume Dynare syntax is strict: VAR(k) means VAR_pk if k>0, VAR_mk if k<0
                      # However, the pattern requires sign +/- for non-zero. This case shouldn't be hit often.
                      # If it *does* mean current time, treat as num=0.
                      # If it's an error in the mod file like Y(1), treat as Y_p1? Let's assume the latter.
                    if num > 0 :
                         print(f"Warning: Interpreting '{match.group(0)}' as '{var_name}_p{num}' due to missing sign (strict Dynare usually requires '+').")
                         replacement_name = f"{var_name}_p{num}"
                         needed_symbols.add(replacement_name)
                    else: # Should not happen if num != 0 and sign is empty
                         replacement_name = var_name # Fallback

                if replacement_name:
                    replacements.append((start_index, end_index, replacement_name))

        # Perform replacements from right to left to avoid index issues
        for start, end, replacement_name in sorted(replacements, key=lambda x: x[0], reverse=True):
            output_str = output_str[:start] + replacement_name + output_str[end:]

        # Add any newly needed symbols (VAR_pk, VAR_mk) to the main symbol dictionary
        for sym_name in needed_symbols:
            if sym_name not in self.symbols:
                self.symbols[sym_name] = sympy.Symbol(sym_name)
                # print(f"Debug: Added timing symbol '{sym_name}'")


        # Handle standalone variables (without parentheses) - ensure they are in symbols
        # This is mostly covered by _create_initial_sympy_symbols, but good check.
        # We rely on sympy.parse_expr later to find symbols.

        return output_str


    def _parse_equations_to_sympy(self):
        """Parses cleaned equation strings into Sympy Eq objects (LHS-RHS=0)."""
        self.sympy_equations_original = []
        for i, eq_str in enumerate(self.equations_str):
            if not eq_str: # Skip empty lines potentially left after cleaning
                continue

            try:
                # Replace Dynare timing like Y(+1) with Y_p1 etc.
                processed_eq_str = self._replace_dynare_timing(eq_str)

                # Check for '=' to split LHS and RHS
                if '=' in processed_eq_str:
                    lhs_str, rhs_str = processed_eq_str.split('=', 1)
                    # Parse LHS and RHS separately
                    # evaluate=False prevents immediate evaluation like 1+1 -> 2
                    lhs_expr = sympy.parse_expr(lhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    rhs_expr = sympy.parse_expr(rhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    # Create equation LHS - RHS = 0
                    # Use sympy.Eq for representation, though we mostly use the .lhs part later
                    self.sympy_equations_original.append(sympy.Eq(lhs_expr - rhs_expr, 0))
                else:
                    # If no '=', assume the whole string is the expression that should equal zero
                    expr = sympy.parse_expr(processed_eq_str, local_dict=self.symbols, evaluate=False)
                    self.sympy_equations_original.append(sympy.Eq(expr, 0))

            except (SyntaxError, TypeError, NameError) as e:
                # Catch common parsing errors
                print(f"ERROR parsing equation {i+1}: '{eq_str}'")
                print(f"  Processed string attempt: '{processed_eq_str}'")
                # Identify potentially undefined symbols
                try:
                    # Attempt parsing again to find undefined names
                     sympy.parse_expr(processed_eq_str, local_dict=self.symbols, evaluate=False)
                except NameError as ne:
                     print(f"  Potential undefined name(s): {ne}")
                except Exception as pe:
                     print(f"  Underlying Sympy error: {type(pe).__name__}: {pe}")

                print(f"  Original error: {type(e).__name__}: {e}")
                # Provide context: which symbols were available?
                # print(f"  Available symbols: {list(self.symbols.keys())}")
                raise ValueError(f"Failed to parse equation {i+1}. Check syntax and ensure all variables/parameters are declared.") from e
            except Exception as e:
                 # Catch other unexpected errors
                 print(f"UNEXPECTED ERROR parsing equation {i+1}: '{eq_str}'")
                 print(f"  Processed string attempt: '{processed_eq_str}'")
                 print(f"  Error: {type(e).__name__}: {e}")
                 raise ValueError(f"Unexpected error parsing equation {i+1}.") from e


    def _analyze_variable_timing(self):
        """Analyzes lead/lag structure for each variable across all equations."""
        print("\n--- Stage 1: Analyzing Variable Timing ---")
        # Use the set of base variable names
        variable_base_names = self.var_names_set
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})

        max_k_check = 15 # Maximum lead/lag to check (arbitrary but usually sufficient)

        for eq in self.sympy_equations_original:
            try:
                # Get all free symbols in the equation's LHS (assuming RHS is 0)
                free_symbols_in_eq = eq.lhs.free_symbols
            except AttributeError:
                print(f"Warning: Skipping timing analysis for non-equation object: {eq}")
                continue

            # Check each symbol against known patterns for variables
            for sym in free_symbols_in_eq:
                sym_name = sym.name
                base_name = None
                lag = 0
                lead = 0

                # Check for lead pattern: VAR_p<k>
                lead_match = re.match(r"(\w+)_p(\d+)", sym_name)
                if lead_match:
                    base_name = lead_match.group(1)
                    lead = int(lead_match.group(2))
                else:
                    # Check for lag pattern: VAR_m<k>
                    lag_match = re.match(r"(\w+)_m(\d+)", sym_name)
                    if lag_match:
                        base_name = lag_match.group(1)
                        lag = int(lag_match.group(2)) # Positive number for lag amount
                    else:
                        # If no lead/lag pattern, check if it's a base variable name
                        if sym_name in variable_base_names:
                            base_name = sym_name
                            # Appears contemporaneously

                # If the symbol corresponds to a declared variable (base or timed)
                if base_name and base_name in variable_base_names:
                    if lead > 0:
                        self.var_timing_info[base_name]['max_lead'] = max(self.var_timing_info[base_name]['max_lead'], lead)
                    elif lag > 0:
                        # Store lag as negative value internally
                        self.var_timing_info[base_name]['min_lag'] = min(self.var_timing_info[base_name]['min_lag'], -lag)
                    else: # num == 0 implies current time
                        self.var_timing_info[base_name]['appears_current'] = True

        # --- Logging ---
        print("Variable Timing Analysis Results:")
        found_any_timing = False
        for var_name in sorted(self.var_names): # Iterate through original declared vars
            if var_name in self.var_timing_info:
                info = self.var_timing_info[var_name]
                print(f"- {var_name}: Current={info['appears_current']}, MaxLead={info['max_lead']}, MinLag={info['min_lag']}")
                found_any_timing = True
            else:
                # Check if the variable just wasn't found in *any* equation
                 symbol_exists = var_name in self.symbols
                 if symbol_exists:
                      print(f"- {var_name}: Not found in any equations.")
                 else:
                      print(f"- {var_name}: Declared but symbol not created (unexpected).")

        if not found_any_timing and self.var_names:
             print("Warning: No timing information gathered. Were variables used in equations?")


    def _identify_and_eliminate_static_vars(self):
        """
        Identifies static variables (appearing only contemporaneously across *all* their occurrences)
        and attempts to solve for them. Removes the defining equation.
        Uses standard Python formatting.
        """
        print("\n--- Stage 2: Identifying and Eliminating Static Variables ---")
        self.static_subs = {} # Stores {static_sym: solution_expr}
        self.equations_after_static_elim = [] # Equations remaining after removal
        output_lines = ["Static Variable Identification and Elimination Log:"]

        # Get symbols for all declared variables
        var_syms = {self.symbols[v] for v in self.var_names if v in self.symbols}
        if not var_syms:
            print("No variable symbols found. Skipping static analysis.")
            self.equations_after_static_elim = list(self.sympy_equations_original)
            return

        # Identify variables that appear *only* contemporaneously
        candidate_static_syms = set()
        non_static_syms = set() # Variables known to have leads/lags

        for var_name in self.var_names:
             info = self.var_timing_info.get(var_name)
             if info:
                  # If it has leads or lags, it's definitely not static
                  if info['max_lead'] > 0 or info['min_lag'] < 0:
                       if var_name in self.symbols:
                            non_static_syms.add(self.symbols[var_name])
                  # If it appears only currently (and has no leads/lags), it's a candidate
                  elif info['appears_current'] and info['max_lead'] == 0 and info['min_lag'] == 0:
                       if var_name in self.symbols:
                            candidate_static_syms.add(self.symbols[var_name])
                  # Else: Variable declared but doesn't appear at all? Ignore.
             # else: Variable declared but not found in equations. Ignore.


        line = f"Potential static variables (appear only at t=0): {[v.name for v in sorted(candidate_static_syms, key=lambda s:s.name)]}"
        print(line); output_lines.append(line)
        line = f"Non-static variables (have leads/lags): {[v.name for v in sorted(non_static_syms, key=lambda s:s.name)]}"
        print(line); output_lines.append(line)

        if not candidate_static_syms:
            print("No candidate static variables found based on timing analysis.")
            output_lines.append("No candidate static variables found.")
            self.equations_after_static_elim = list(self.sympy_equations_original)
            return

        # Iteratively solve for static candidates
        remaining_equations = list(self.sympy_equations_original)
        solved_statics_syms = set()
        made_change_in_iteration = True
        iteration = 0
        max_iterations = len(candidate_static_syms) + 2 # Safety break

        while made_change_in_iteration and iteration < max_iterations and candidate_static_syms:
            iteration += 1
            made_change_in_iteration = False
            next_remaining_equations = []
            solved_this_round = set()
            equations_used_this_round = set() # Track equations used for solving

            output_lines.append(f"\n--- Solving Iteration {iteration} ---")
            print(f"--- Static Solving Iteration {iteration} ---")

            # Find equations that might define *one* unsolved static variable
            for eq_idx, eq in enumerate(remaining_equations):
                if eq in equations_used_this_round: # Skip if already used
                    continue
                try:
                    eq_lhs = eq.lhs
                    eq_atoms = eq_lhs.free_symbols
                except AttributeError:
                    # Keep non-equation objects if any slipped through (shouldn't happen)
                    next_remaining_equations.append(eq)
                    continue

                # What static candidates are in this equation?
                statics_in_eq = eq_atoms.intersection(candidate_static_syms)
                # What *already solved* statics are in this equation?
                known_solved_in_eq = eq_atoms.intersection(solved_statics_syms)
                # What *unsolved* static candidates are in this equation?
                unsolved_statics_in_eq = statics_in_eq - solved_statics_syms

                # Try to solve if exactly one *unsolved* static candidate appears
                if len(unsolved_statics_in_eq) == 1:
                    static_cand_sym = list(unsolved_statics_in_eq)[0]
                    # Substitute already known static solutions into this equation
                    current_subs = {s: expr for s, expr in self.static_subs.items()}
                    eq_lhs_subbed = eq_lhs.subs(current_subs)

                    # Check if the candidate *still* appears after substitution
                    if static_cand_sym not in eq_lhs_subbed.free_symbols:
                        # This can happen if substitutions eliminated the variable.
                        # The equation might become 0=0 (redundant) or just not define the variable anymore.
                        # We should keep this equation for now unless it simplifies to 0=0.
                        if sympy.simplify(eq_lhs_subbed) == 0:
                            line = f"- Equation {eq_idx+1} ({sympy.sstr(eq_lhs)}=0) became 0=0 after subs; removing." 
                            print(line)
                            output_lines.append(line)
                            equations_used_this_round.add(eq) # Mark as 'used' (removed)
                        else:
                            # Equation didn't define the variable after all, keep it.
                            if eq not in equations_used_this_round:
                                next_remaining_equations.append(eq)
                        continue # Move to the next equation

                    # Attempt to solve for the candidate
                    try:
                        # Use sympy.solve
                        solution_list = sympy.solve(eq_lhs_subbed, static_cand_sym)

                        if isinstance(solution_list, list) and len(solution_list) == 1:
                            solution = solution_list[0]

                            # Check if the solution depends on the variable itself (solve failed implicitly)
                            if static_cand_sym in solution.free_symbols:
                                line = f"- Warn: Solution for {static_cand_sym.name} from eq {eq_idx+1} still contains the variable. Cannot use."
                                print(line); output_lines.append(line)
                                if eq not in equations_used_this_round: # Keep the equation if solve failed
                                     next_remaining_equations.append(eq)
                            else:
                                # Successful solution!
                                line = f"- Solved: {static_cand_sym.name} = {solution} (from eq {eq_idx+1}: {sympy.sstr(eq_lhs_subbed)} = 0)"
                                print(line); output_lines.append(line)
                                self.static_subs[static_cand_sym] = solution
                                solved_this_round.add(static_cand_sym)
                                made_change_in_iteration = True
                                equations_used_this_round.add(eq) # Mark equation as used

                        elif isinstance(solution_list, dict) and static_cand_sym in solution_list :
                              # Handle solve returning a dictionary
                              solution = solution_list[static_cand_sym]
                              if static_cand_sym in solution.free_symbols:
                                  line = f"- Warn: Solution for {static_cand_sym.name} from eq {eq_idx+1} still contains the variable. Cannot use."
                                  print(line); output_lines.append(line)
                                  if eq not in equations_used_this_round:
                                       next_remaining_equations.append(eq)
                              else:
                                  line = f"- Solved: {static_cand_sym.name} = {solution} (from eq {eq_idx+1}: {sympy.sstr(eq_lhs_subbed)} = 0)"
                                  print(line); output_lines.append(line)
                                  self.static_subs[static_cand_sym] = solution
                                  solved_this_round.add(static_cand_sym)
                                  made_change_in_iteration = True
                                  equations_used_this_round.add(eq)
                        else:
                            # Solve returned empty list, multiple solutions, or unexpected format
                            line = f"- Warn: sympy.solve for {static_cand_sym.name} from eq {eq_idx+1} yielded no unique solution ({solution_list})."
                            print(line); output_lines.append(line)
                            if eq not in equations_used_this_round: # Keep the equation
                                next_remaining_equations.append(eq)

                    except NotImplementedError:
                        line = f"- Warn: sympy.solve cannot solve equation {eq_idx+1} for {static_cand_sym.name} (NotImplementedError)."
                        print(line); output_lines.append(line)
                        if eq not in equations_used_this_round: # Keep the equation
                             next_remaining_equations.append(eq)
                    except Exception as e:
                        line = f"- Warn: Error solving eq {eq_idx+1} for {static_cand_sym.name}. Error: {e}"
                        print(line); output_lines.append(line)
                        if eq not in equations_used_this_round: # Keep the equation
                             next_remaining_equations.append(eq)

                else:
                    # Equation has 0 or >1 unsolved static candidates, keep it for next round or substitution
                     if eq not in equations_used_this_round:
                          next_remaining_equations.append(eq)

            # Update solved set and remaining candidates
            solved_statics_syms.update(solved_this_round)
            candidate_static_syms -= solved_this_round # Remove solved ones from candidates
            remaining_equations = next_remaining_equations # Update list for next iteration

            if made_change_in_iteration:
                output_lines.append(f"Solved in iteration {iteration}: {[s.name for s in solved_this_round]}")
                output_lines.append(f"Remaining static candidates: {[s.name for s in candidate_static_syms]}")
            elif iteration == 1 and not solved_statics_syms: # No progress on first try
                output_lines.append("\nNo static vars solved in iteration 1.")
                break # Exit loop if no progress initially
            elif not made_change_in_iteration:
                 output_lines.append(f"\nNo further static variables solved in iteration {iteration}.")
                 break # Exit if no progress in a later iteration


        # Final list of equations is those remaining after the solving loop
        self.equations_after_static_elim = remaining_equations

        # --- Logging ---
        line = f"\nStatic analysis complete. Solved {len(solved_statics_syms)} static variables: {[v.name for v in sorted(solved_statics_syms, key=lambda s:s.name)]}"
        print(line); output_lines.append(line)
        if candidate_static_syms:
            line = f"Unsolved static candidates remaining: {[v.name for v in sorted(candidate_static_syms, key=lambda s:s.name)]}"
            print(line); output_lines.append(line)
        line = f"Number of equations remaining after static elimination: {len(self.equations_after_static_elim)}"
        print(line); output_lines.append(line)

        # Save the detailed log if needed (optional)
        # self._save_intermediate_file("static_elim_log.txt", output_lines)


    def _substitute_static_vars(self):
        """
        Substitutes solved static variables into the remaining equations.
        Handles necessary time shifts for static variable solutions.
        Uses standard Python formatting.
        """
        print("\n--- Stage 3: Substituting Static Variables ---")
        output_lines = ["Substituting static variables log:"]

        if not self.static_subs:
            print("No static substitutions to perform.")
            output_lines.append("No static substitutions found.")
            # Ensure the list is copied
            self.equations_after_static_sub = list(self.equations_after_static_elim)
            return

        # Identify the maximum lead/lag needed for substitutions
        max_lead_needed = 0
        min_lag_needed = 0
        static_syms_to_sub = set(self.static_subs.keys())

        for eq in self.equations_after_static_elim:
            try:
                eq_atoms = eq.lhs.free_symbols
            except AttributeError:
                continue # Skip non-equations

            for atom in eq_atoms:
                atom_name = atom.name
                base_name = None
                k = 0
                is_static_related = False

                # Check lead format: VAR_pk or aux_VAR_leadk_pk etc.
                match_lead = re.match(r"(.+)_p(\d+)$", atom_name)
                if match_lead:
                    base_name_candidate = match_lead.group(1)
                    # The base could be a static var name OR an aux name derived from one
                    # We need to check if the ultimate original variable was static.
                    # This is tricky. Let's check if the direct base symbol is static.
                    potential_base_sym = self.symbols.get(base_name_candidate)
                    if potential_base_sym in static_syms_to_sub:
                         base_name = base_name_candidate
                         k = int(match_lead.group(2))
                         is_static_related = True

                if not is_static_related:
                    # Check lag format: VAR_mk or aux_VAR_lagk_mk etc.
                    match_lag = re.match(r"(.+)_m(\d+)$", atom_name)
                    if match_lag:
                        base_name_candidate = match_lag.group(1)
                        potential_base_sym = self.symbols.get(base_name_candidate)
                        if potential_base_sym in static_syms_to_sub:
                            base_name = base_name_candidate
                            k = int(match_lag.group(2)) # Positive lag number
                            is_static_related = True

                if is_static_related:
                    if k > 0 and match_lead: # Lead
                        max_lead_needed = max(max_lead_needed, k)
                    elif k > 0 and match_lag: # Lag
                        min_lag_needed = min(min_lag_needed, -k) # Store as negative

        output_lines.append(f"Max lead needed for static subs: {max_lead_needed}")
        output_lines.append(f"Min lag needed for static subs: {min_lag_needed}")

        # Create the full substitution dictionary including time shifts
        full_subs_dict = {}
        print(f"Creating substitution rules for {len(self.static_subs)} static variables...")
        for static_var_sym, solution_expr in self.static_subs.items():
            static_var_name = static_var_sym.name
            # Add the base case (t=0)
            full_subs_dict[static_var_sym] = solution_expr
            output_lines.append(f"  Rule: {static_var_name} -> {solution_expr}")

            # Add leads (t+k)
            for k in range(1, max_lead_needed + 1):
                lead_key_name = f"{static_var_name}_p{k}"
                if lead_key_name in self.symbols:
                    lead_key_sym = self.symbols[lead_key_name]
                    # Shift the *solution expression* forward by k periods
                    # Need access to all symbols and the base variable set for shifting
                    try:
                        lead_solution = time_shift_expression(solution_expr, k, self.symbols, self.var_names_set)
                        full_subs_dict[lead_key_sym] = lead_solution
                        output_lines.append(f"  Rule: {lead_key_name} -> {lead_solution}")
                    except Exception as e:
                         print(f"ERROR creating lead rule for {lead_key_name}: {e}")
                         output_lines.append(f"  ERROR Rule: {lead_key_name} -> [Shift Failed: {e}]")


            # Add lags (t-k)
            for k in range(1, abs(min_lag_needed) + 1):
                lag_key_name = f"{static_var_name}_m{k}"
                if lag_key_name in self.symbols:
                    lag_key_sym = self.symbols[lag_key_name]
                     # Shift the *solution expression* backward by k periods
                    try:
                        lag_solution = time_shift_expression(solution_expr, -k, self.symbols, self.var_names_set)
                        full_subs_dict[lag_key_sym] = lag_solution
                        output_lines.append(f"  Rule: {lag_key_name} -> {lag_solution}")
                    except Exception as e:
                         print(f"ERROR creating lag rule for {lag_key_name}: {e}")
                         output_lines.append(f"  ERROR Rule: {lag_key_name} -> [Shift Failed: {e}]")

        # Apply substitutions to the remaining equations
        self.equations_after_static_sub = []
        num_equations_before = len(self.equations_after_static_elim)
        print(f"Applying {len(full_subs_dict)} substitution rules to {num_equations_before} equations...")

        for i, eq in enumerate(self.equations_after_static_elim):
            try:
                original_lhs = eq.lhs
                # Use xreplace for potentially more robust substitution
                substituted_lhs = original_lhs.xreplace(full_subs_dict)

                # Try simplifying the result
                try:
                    # Use simplify, but consider expand().simplify() or factor() if needed
                    simplified_lhs = sympy.simplify(substituted_lhs)
                except Exception as e:
                    print(f"Warning: Simplification failed for substituted equation {i+1}. Error: {e}. Using unsimplified result.")
                    simplified_lhs = substituted_lhs # Keep the unsimplified version

                # Create the new equation object
                subbed_eq = sympy.Eq(simplified_lhs, 0)
                self.equations_after_static_sub.append(subbed_eq)

                # Optional: Log the change
                # if simplified_lhs != original_lhs:
                #     output_lines.append(f"Eq {i+1} changed: {sympy.sstr(original_lhs)} -> {sympy.sstr(simplified_lhs)}")
                # else:
                #     output_lines.append(f"Eq {i+1} unchanged: {sympy.sstr(original_lhs)}")

            except AttributeError:
                # Keep non-equation objects if they exist
                self.equations_after_static_sub.append(eq)
                output_lines.append(f"Eq {i+1} was not an equation object, kept as is.")
            except Exception as e:
                print(f"ERROR substituting static vars in equation {i+1}: ({sympy.sstr(eq.lhs)}=0). Error: {e}")
                # Decide whether to keep the original or skip it
                self.equations_after_static_sub.append(eq) # Keep original on error
                output_lines.append(f"ERROR substituting Eq {i+1}. Kept original. Error: {e}")


        line = f"Substitution complete. {len(self.equations_after_static_sub)} equations remain."
        print(line); output_lines.append(line)

        # Optional: Save detailed log
        # self._save_intermediate_file("static_sub_log.txt", output_lines)


    def _handle_aux_vars(self):
        """
        Identifies leads > +1 and lags < -1 in dynamic variables.
        Creates auxiliary variables and corresponding definition equations.
        Substitutes the original long leads/lags with aux variable timings.
        Uses standard Python formatting.
        """
        print("\n--- Stage 4: Handling Auxiliary Variables for Long Leads/Lags ---")
        self.aux_lead_vars = {} # Map: aux_lead_name -> symbol
        self.aux_lag_vars = {}  # Map: aux_lag_name -> symbol
        self.aux_var_definitions = [] # List of Sympy Eq for aux definitions
        current_equations = list(self.equations_after_static_sub) # Start from post-static-sub equations
        subs_long_leads_lags = {} # Map: original_long_lead/lag_sym -> new_aux_sym

        # Identify dynamic variables (non-static ones)
        static_var_names = {s.name for s in self.static_subs.keys()}
        dynamic_base_var_names = [
            v for v in self.var_names if v not in static_var_names
        ]
        dynamic_base_var_syms = {self.symbols[v] for v in dynamic_base_var_names if v in self.symbols}

        # --- Handle LEADS > +1 ---
        leads_to_replace = collections.defaultdict(int) # Map: base_var_name -> max_lead_k > 1

        # Find maximum lead k > 1 for each dynamic variable
        for eq in current_equations:
            try:
                for atom in eq.lhs.free_symbols:
                    match = re.match(r"(\w+)_p(\d+)", atom.name)
                    if match:
                        base, k_str = match.groups()
                        k = int(k_str)
                        # Check if it's a dynamic variable and lead is > 1
                        if k > 1 and base in dynamic_base_var_names:
                            leads_to_replace[base] = max(leads_to_replace[base], k)
            except AttributeError:
                continue # Skip non-equations

        if leads_to_replace:
            print("Creating auxiliary LEAD variables...")
            output_lines_aux = ["Auxiliary Lead Variable Creation:"]
            for var_name in sorted(leads_to_replace.keys()):
                max_lead_k = leads_to_replace[var_name]
                output_lines_aux.append(f"- Variable '{var_name}' needs aux leads up to k={max_lead_k-1} (for original lead {max_lead_k})")

                # Create chain of aux vars: aux_VAR_lead1, aux_VAR_lead2, ... aux_VAR_lead{k-1}
                for k in range(1, max_lead_k): # Need aux vars up to k-1 level
                    aux_name = f"aux_{var_name}_lead{k}"
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                        output_lines_aux.append(f"  - Created symbol: {aux_name}")
                    else:
                         output_lines_aux.append(f"  - Symbol exists: {aux_name}")

                    aux_sym = self.symbols[aux_name]
                    self.aux_lead_vars[aux_name] = aux_sym # Store name -> symbol mapping

                    # Define the auxiliary variable: aux_VAR_lead{k} = (previous term)(+1)
                    if k == 1:
                        # Definition: aux_VAR_lead1 = VAR(+1)
                        var_p1_name = f"{var_name}_p1"
                        if var_p1_name not in self.symbols:
                            self.symbols[var_p1_name] = sympy.Symbol(var_p1_name)
                        rhs_sym = self.symbols[var_p1_name]
                        output_lines_aux.append(f"  - Definition: {aux_name} = {var_p1_name}")
                    else:
                        # Definition: aux_VAR_lead{k} = aux_VAR_lead{k-1}(+1)
                        prev_aux_name = f"aux_{var_name}_lead{k-1}"
                        prev_aux_p1_name = f"{prev_aux_name}_p1" # Name of the lead of the previous aux var
                        if prev_aux_p1_name not in self.symbols:
                            self.symbols[prev_aux_p1_name] = sympy.Symbol(prev_aux_p1_name)
                        rhs_sym = self.symbols[prev_aux_p1_name]
                        output_lines_aux.append(f"  - Definition: {aux_name} = {prev_aux_p1_name}")

                    # Create the equation: aux_sym - rhs_sym = 0
                    new_eq = sympy.Eq(aux_sym - rhs_sym, 0)
                    # Avoid adding duplicate definitions if structure overlaps somehow
                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)

                    # Define substitution rule for the *original* long lead
                    # Original VAR(+(k+1)) should be replaced by aux_VAR_lead{k}(+1)
                    original_long_lead_name = f"{var_name}_p{k+1}"
                    if original_long_lead_name in self.symbols:
                         original_long_lead_sym = self.symbols[original_long_lead_name]
                         # The replacement is the lead of the current aux var
                         aux_lead_k_p1_name = f"{aux_name}_p1"
                         if aux_lead_k_p1_name not in self.symbols:
                                self.symbols[aux_lead_k_p1_name] = sympy.Symbol(aux_lead_k_p1_name)
                         replacement_sym = self.symbols[aux_lead_k_p1_name]
                         subs_long_leads_lags[original_long_lead_sym] = replacement_sym
                         output_lines_aux.append(f"  - Substitution Rule: {original_long_lead_name} -> {aux_lead_k_p1_name}")

            # Optional: Save aux lead creation log
            # self._save_intermediate_file("aux_lead_creation_log.txt", output_lines_aux)


        # --- Handle LAGS < -1 ---
        lags_to_replace = collections.defaultdict(int) # Map: base_var_name -> max_lag_k > 1

        # Find maximum lag k > 1 for each dynamic variable
        for eq in current_equations:
             try:
                for atom in eq.lhs.free_symbols:
                    match = re.match(r"(\w+)_m(\d+)", atom.name)
                    if match:
                        base, k_str = match.groups()
                        k = int(k_str)
                        # Check if it's a dynamic variable and lag is > 1
                        if k > 1 and base in dynamic_base_var_names:
                            lags_to_replace[base] = max(lags_to_replace[base], k)
             except AttributeError:
                 continue

        if lags_to_replace:
            print("Creating auxiliary LAG variables...")
            output_lines_aux = ["Auxiliary Lag Variable Creation:"] # Reset log for lags
            for var_name in sorted(lags_to_replace.keys()):
                max_lag_k = lags_to_replace[var_name]
                output_lines_aux.append(f"- Variable '{var_name}' needs aux lags up to k={max_lag_k-1} (for original lag -{max_lag_k})")

                # Create chain: aux_VAR_lag1, ..., aux_VAR_lag{k-1}
                for k in range(1, max_lag_k):
                    aux_name = f"aux_{var_name}_lag{k}"
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                        output_lines_aux.append(f"  - Created symbol: {aux_name}")
                    else:
                         output_lines_aux.append(f"  - Symbol exists: {aux_name}")

                    aux_sym = self.symbols[aux_name]
                    self.aux_lag_vars[aux_name] = aux_sym

                    # Define the auxiliary variable: aux_VAR_lag{k} = (previous term)(-1)
                    if k == 1:
                        # Definition: aux_VAR_lag1 = VAR(-1)
                        var_m1_name = f"{var_name}_m1"
                        if var_m1_name not in self.symbols:
                            self.symbols[var_m1_name] = sympy.Symbol(var_m1_name)
                        rhs_sym = self.symbols[var_m1_name]
                        output_lines_aux.append(f"  - Definition: {aux_name} = {var_m1_name}")
                    else:
                        # Definition: aux_VAR_lag{k} = aux_VAR_lag{k-1}(-1)
                        prev_aux_name = f"aux_{var_name}_lag{k-1}"
                        prev_aux_m1_name = f"{prev_aux_name}_m1"
                        if prev_aux_m1_name not in self.symbols:
                            self.symbols[prev_aux_m1_name] = sympy.Symbol(prev_aux_m1_name)
                        rhs_sym = self.symbols[prev_aux_m1_name]
                        output_lines_aux.append(f"  - Definition: {aux_name} = {prev_aux_m1_name}")

                    # Create equation: aux_sym - rhs_sym = 0
                    new_eq = sympy.Eq(aux_sym - rhs_sym, 0)
                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)

                    # Define substitution rule for the *original* long lag
                    # Original VAR(-(k+1)) should be replaced by aux_VAR_lag{k}(-1)
                    original_long_lag_name = f"{var_name}_m{k+1}"
                    if original_long_lag_name in self.symbols:
                        original_long_lag_sym = self.symbols[original_long_lag_name]
                        # Replacement is the lag of the current aux var
                        aux_lag_k_m1_name = f"{aux_name}_m1"
                        if aux_lag_k_m1_name not in self.symbols:
                            self.symbols[aux_lag_k_m1_name] = sympy.Symbol(aux_lag_k_m1_name)
                        replacement_sym = self.symbols[aux_lag_k_m1_name]
                        subs_long_leads_lags[original_long_lag_sym] = replacement_sym
                        output_lines_aux.append(f"  - Substitution Rule: {original_long_lag_name} -> {aux_lag_k_m1_name}")

            # Optional: Save aux lag creation log
            # self._save_intermediate_file("aux_lag_creation_log.txt", output_lines_aux)


        # --- Apply Substitutions for Long Leads/Lags ---
        self.equations_after_aux_sub = []
        if not subs_long_leads_lags:
            print("No long leads (>1) or lags (<-1) found requiring auxiliary variables.")
            self.equations_after_aux_sub = list(current_equations) # No changes needed
        else:
            print(f"\nApplying {len(subs_long_leads_lags)} long lead/lag substitutions to {len(current_equations)} equations...")
            output_lines_sub = ["Auxiliary Variable Substitution Log:"]
            for i, eq in enumerate(current_equations):
                try:
                    original_lhs = eq.lhs
                    # Apply the substitutions collected earlier
                    subbed_lhs = original_lhs.xreplace(subs_long_leads_lags)

                    # Try simplifying (optional, but good practice)
                    try:
                        simplified_lhs = sympy.simplify(subbed_lhs)
                    except Exception as e:
                        # print(f"Warning: Simplification failed for aux-substituted equation {i+1}. Error: {e}")
                        simplified_lhs = subbed_lhs # Use unsimplified

                    subbed_eq = sympy.Eq(simplified_lhs, 0)
                    self.equations_after_aux_sub.append(subbed_eq)

                    # Log if changed
                    # if simplified_lhs != original_lhs:
                    #      output_lines_sub.append(f"Eq {i+1} changed: {sympy.sstr(original_lhs)} -> {sympy.sstr(simplified_lhs)}")

                except AttributeError:
                    self.equations_after_aux_sub.append(eq) # Keep non-equations
                except Exception as e:
                    print(f"ERROR substituting aux vars in equation {i+1}: {sympy.sstr(eq.lhs)} = 0. Error: {e}")
                    self.equations_after_aux_sub.append(eq) # Keep original on error
                    output_lines_sub.append(f"ERROR substituting Eq {i+1}. Kept original. Error: {e}")

            # Optional: Save log
            # self._save_intermediate_file("aux_sub_log.txt", output_lines_sub)

        # Define the final set of dynamic variable *names* (including aux)
        self.final_dynamic_var_names = (dynamic_base_var_names
                                        + list(self.aux_lead_vars.keys())
                                        + list(self.aux_lag_vars.keys()))

        print(f"Auxiliary variable handling complete.")
        print(f"Total auxiliary lead variables created: {len(self.aux_lead_vars)}")
        print(f"Total auxiliary lag variables created: {len(self.aux_lag_vars)}")
        print(f"Total auxiliary variable definitions added: {len(self.aux_var_definitions)}")
        print(f"Number of equations after aux substitution: {len(self.equations_after_aux_sub)}")
        print(f"Final dynamic variable names (incl. aux): {len(self.final_dynamic_var_names)}")
        # print(f"  Names: {sorted(self.final_dynamic_var_names)}") # Optional: list them


    # ===========================================
    # Stage 5: Define State Vector (MODIFIED)
    # ===========================================
    def _define_state_vector(self):
        """
        Defines the ordered state vector y_t. Uses a user-provided order
        (self.forced_state_order) if available and valid, otherwise defaults
        to alphabetical sorting within categories (base, aux_lag, aux_lead).
        Updates self.state_vars_ordered (list of symbols) and self.state_var_map.
        """
        print("\n--- Stage 5: Defining State Vector ---")
        final_dynamic_names_set = set(self.final_dynamic_var_names)
        n_dynamic_vars = len(final_dynamic_names_set)

        if n_dynamic_vars == 0:
            raise ValueError("Cannot define state vector: No final dynamic variables identified (incl. aux).")

        ordered_names = [] # This will hold the final ordered list of names

        # --- Check for User-Provided Order ---
        if self.forced_state_order is not None:
            print(f"Attempting to use user-provided state vector order ({len(self.forced_state_order)} variables).")
            provided_order = self.forced_state_order
            provided_order_set = set(provided_order)

            # --- Validation ---
            valid = True
            error_msg = ""
            if not isinstance(provided_order, (list, tuple)):
                error_msg = "Forced state order must be a list or tuple."
                valid = False
            elif len(provided_order) != n_dynamic_vars:
                error_msg = (f"Forced state order length ({len(provided_order)}) "
                             f"does not match the number of dynamic variables ({n_dynamic_vars}).")
                valid = False
            elif len(provided_order_set) != len(provided_order):
                 error_msg = "Forced state order contains duplicate variable names."
                 valid = False
            elif provided_order_set != final_dynamic_names_set:
                missing_in_provided = final_dynamic_names_set - provided_order_set
                extra_in_provided = provided_order_set - final_dynamic_names_set
                error_msg = "Forced state order does not match the set of dynamic variables."
                if missing_in_provided: error_msg += f" Missing: {sorted(list(missing_in_provided))}"
                if extra_in_provided: error_msg += f" Extra: {sorted(list(extra_in_provided))}"
                valid = False

            if valid:
                print("User-provided order is valid. Using it.")
                ordered_names = list(provided_order) # Use the user's list
            else:
                print(f"ERROR: Invalid user-provided state order: {error_msg}")
                # Fallback to default ordering or raise error? Let's raise an error.
                raise ValueError(f"Invalid forced_state_order: {error_msg}")

        # --- Default Ordering Logic ---
        else:
            print("No user-provided order found. Using default ordering:")
            print("  (Base Vars (alpha) + Aux Lag Vars (alpha) + Aux Lead Vars (alpha))")
            # Sort alphabetically within categories
            base_dynamic_vars = sorted([s for s in self.final_dynamic_var_names if not s.startswith("aux_")])
            aux_lag_vars_list = sorted([s for s in self.final_dynamic_var_names if s.startswith("aux_") and "_lag" in s])
            aux_lead_vars_list = sorted([s for s in self.final_dynamic_var_names if s.startswith("aux_") and "_lead" in s])
            ordered_names = base_dynamic_vars + aux_lag_vars_list + aux_lead_vars_list

        # --- Convert names to symbols and store ---
        self.state_vars_ordered = [] # Ensure it's reset
        missing_symbols = []
        for name in ordered_names:
            if name in self.symbols:
                self.state_vars_ordered.append(self.symbols[name])
            else:
                # This indicates an internal inconsistency if it happens
                missing_symbols.append(name)

        if missing_symbols:
            print(f"FATAL ERROR: The following state variable names do not have corresponding symbols:")
            print(f"  {missing_symbols}")
            print(f"  Available symbols: {list(self.symbols.keys())}")
            raise KeyError(f"Symbols missing for state variables: {missing_symbols}. This indicates a parser bug.")

        # Create the symbol-to-index map based on the final order
        self.state_var_map = {sym: i for i, sym in enumerate(self.state_vars_ordered)}

        # --- Logging ---
        final_order_names = [s.name for s in self.state_vars_ordered]
        line = f"Final State Vector Defined (size {len(self.state_vars_ordered)}):"
        print(line)
        # Print first few and last few if very long
        max_print = 20
        if len(final_order_names) <= max_print:
             print(f"  Order: {final_order_names}")
        else:
             print(f"  Order: {final_order_names[:max_print//2]} ... {final_order_names[-max_print//2:]}")


    # ==================================================
    # Stages 6, 7 and helpers (_build_final_equations,
    # get_numerical_ABCD, _generate_matrix_assignments_code_helper,
    # generate_matrix_function_file) go here
    # ==================================================
    def _build_final_equations(self):
        """Combines equations after aux substitution and aux definitions. Checks count."""
        print("\n--- Stage 6: Building Final Equation System ---")

        # The final system includes:
        # 1. Original equations after static and aux substitutions
        # 2. Definitions of the auxiliary variables
        self.final_equations_for_jacobian = list(self.equations_after_aux_sub) + list(self.aux_var_definitions)

        # --- Sanity Check: Count vs State Variables ---
        n_state = len(self.state_vars_ordered)
        n_eqs = len(self.final_equations_for_jacobian)

        line1 = f"\nChecking counts: N_States = {n_state}, N_Equations = {n_eqs}"
        print(line1)

        if n_state == 0 and n_eqs == 0 :
             print("Warning: Model resulted in zero state variables and zero equations.")
             # This might be valid for a purely static model, though unusual for this parser.
             return # Allow processing to continue if possible? Or raise error? Let's allow.

        if n_state == 0 and n_eqs > 0:
            # Should not happen if logic is correct
            raise ValueError("State vector has zero size, but final equations exist.")
        if n_eqs == 0 and n_state > 0:
             # Should not happen if logic is correct
             raise ValueError("Final equation list is empty, but state variables exist.")

        if n_state != n_eqs:
            line2 = f"FATAL ERROR: State count ({n_state}) does not match final equation count ({n_eqs})!"
            print("!"*60 + "\n" + line2)
            print("\nState Variables:")
            state_names_list = [s.name for s in self.state_vars_ordered]
            print(f"  ({len(state_names_list)}): {state_names_list}")

            print("\nFinal Equations:")
            max_eq_print = 25 # Print more for debugging mismatches
            for i, eq in enumerate(self.final_equations_for_jacobian):
                 if i >= max_eq_print:
                      print(f"  ... ({n_eqs - max_eq_print} more)")
                      break
                 try:
                      print(f"  {i+1}: {sympy.sstr(eq.lhs)} = 0")
                 except:
                      print(f"  {i+1}: [Error displaying equation object: {type(eq)}]")

            print("\nPotential Causes:")
            print("- Errors during static variable elimination/substitution.")
            print("- Errors during auxiliary variable creation/substitution.")
            print("- Incorrect counting of final dynamic variables or equations.")
            print("\n" + "!"*60)
            # Save the state variables and equations for debugging
            self.save_final_equations_to_txt("ERROR_final_equations.txt")
            with open("ERROR_state_vars.txt", "w") as f:
                 f.write(f"State Vars ({n_state}):\n")
                 f.write("\n".join(state_names_list))

            raise ValueError(f"Equation count mismatch ({n_eqs}) vs state count ({n_state}) after final assembly. Check intermediate files and ERROR files.")
        else:
            line2 = "State and equation counts match. Proceeding to Jacobian calculation."
            print(line2)


    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        """
        Calculates numerical A, B, C, D matrices from the final equations
        and the ordered state vector (self.state_vars_ordered).

        Args:
            param_dict_values (dict): Dictionary mapping parameter names to values.
            file_path (str, optional): Path to save the resulting matrices (.pkl). Defaults to None.

        Returns:
            tuple: (A_num, B_num, C_num, D_num, state_names, shock_names)
                   where A, B, C, D are numpy arrays, and names are lists of strings.
        """
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")

        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
             # Handle the case where the model might be empty (e.g., static only)
             if not self.state_vars_ordered and not self.final_equations_for_jacobian:
                  print("Model appears empty (no dynamic states/equations). Returning empty matrices.")
                  n_state = 0
                  n_shocks = len(self.shock_names)
                  state_names = []
                  shock_names = self.shock_names
                  A_num = np.zeros((0, 0), dtype=float)
                  B_num = np.zeros((0, 0), dtype=float)
                  C_num = np.zeros((0, 0), dtype=float)
                  D_num = np.zeros((0, n_shocks), dtype=float)
                  # Save if requested
                  if file_path:
                        self._save_final_matrices(file_path, A_num, B_num, C_num, D_num,
                                                state_names, shock_names, self.param_names, param_dict_values)
                  return A_num, B_num, C_num, D_num, state_names, shock_names
             else:
                  # If one is populated but not the other, it's an error from previous stages
                  raise ValueError("Preprocessing incomplete: Final equations or ordered state vector not available (or inconsistent). Cannot calculate Jacobians.")


        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)
        state_names = [s.name for s in self.state_vars_ordered] # Use the final ordered names
        shock_names = self.shock_names

        # --- Prepare Parameter Substitutions ---
        self.last_param_values = param_dict_values # Store for potential saving/debugging
        param_subs = {}
        missing_params = []
        for p_name in self.param_names: # Iterate through declared parameters
            if p_name in param_dict_values:
                 if p_name in self.symbols:
                      param_subs[self.symbols[p_name]] = param_dict_values[p_name]
                 else:
                      # Parameter declared but symbol not created? Should not happen.
                      print(f"Warning: Declared parameter '{p_name}' has no symbol.")
            else:
                 missing_params.append(p_name)

        if missing_params:
             raise ValueError(f"Missing parameter values for: {missing_params}")
        if not param_subs and self.param_names:
             print("Warning: No parameter values were substituted (dictionary might be empty or names mismatch).")


        # --- Define Symbolic Vectors based on the ORDERED state vector ---
        # state_vec_t is directly from self.state_vars_ordered
        state_vec_t = sympy.Matrix(self.state_vars_ordered)

        # Create y_{t+1} vector (VAR_p1, AUX_p1, etc.)
        state_vec_tp1_list = []
        for state_sym in self.state_vars_ordered:
            lead_name = f"{state_sym.name}_p1"
            if lead_name not in self.symbols:
                self.symbols[lead_name] = sympy.Symbol(lead_name) # Create if missing
            state_vec_tp1_list.append(self.symbols[lead_name])
        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)

        # Create y_{t-1} vector (VAR_m1, AUX_m1, etc.)
        state_vec_tm1_list = []
        for state_sym in self.state_vars_ordered:
            lag_name = f"{state_sym.name}_m1"
            if lag_name not in self.symbols:
                self.symbols[lag_name] = sympy.Symbol(lag_name) # Create if missing
            state_vec_tm1_list.append(self.symbols[lag_name])
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)

        # Shock vector e_t
        shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
        if len(shock_syms_list) != n_shocks:
             print(f"Warning: Number of shock symbols ({len(shock_syms_list)}) does not match declared varexo count ({n_shocks}).")
             # Find missing shock symbols
             missing_shock_syms = [s for s in self.shock_names if s not in self.symbols]
             if missing_shock_syms: print(f"  Missing shock symbols: {missing_shock_syms}")

        shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None # Handle case of zero shocks

        # Equation vector F(y_{t+1}, y_t, y_{t-1}, e_t) = 0
        # Ensure all elements are expressions (using .lhs)
        eq_expressions = []
        for i, eq in enumerate(self.final_equations_for_jacobian):
            if hasattr(eq, 'lhs'):
                 eq_expressions.append(eq.lhs)
            else:
                 raise TypeError(f"Item {i} in final_equations_for_jacobian is not a Sympy Eq object: {eq}")
        eq_vec = sympy.Matrix(eq_expressions)


        # --- Calculate Symbolic Jacobians ---
        print(f"Calculating symbolic Jacobians (State dim: {n_state}, Shock dim: {n_shocks})...")
        try:
            # Jacobian of F wrt y_{t+1} -> A
            A_sym = eq_vec.jacobian(state_vec_tp1)
            # Jacobian of F wrt y_t -> B
            B_sym = eq_vec.jacobian(state_vec_t)
            # Jacobian of F wrt y_{t-1} -> C
            C_sym = eq_vec.jacobian(state_vec_tm1)

            # Jacobian of F wrt e_t -> -D (note the negative sign)
            if shock_vec is not None and n_shocks > 0:
                D_sym = -eq_vec.jacobian(shock_vec)
            else:
                # Create zero matrix if no shocks
                D_sym = sympy.zeros(n_state, n_shocks)

        except Exception as e:
            print(f"FATAL ERROR during symbolic Jacobian calculation: {e}")
            # Optionally: try to identify the problematic derivative
            # ... (more sophisticated error checking could go here) ...
            raise RuntimeError("Symbolic Jacobian calculation failed.") from e

        # --- Substitute Numerical Parameter Values ---
        print("Substituting parameter values into symbolic matrices...")
        try:
            # Use evalf(subs=...) for numerical evaluation
            #tolist().astype(float) converts Sympy matrix to numpy array
            A_num = np.array(A_sym.evalf(subs=param_subs).tolist()).astype(float)
            B_num = np.array(B_sym.evalf(subs=param_subs).tolist()).astype(float)
            C_num = np.array(C_sym.evalf(subs=param_subs).tolist()).astype(float)

            # Handle D matrix substitution
            if n_shocks > 0:
                D_num = np.array(D_sym.evalf(subs=param_subs).tolist()).astype(float)
            else:
                # Ensure D_num is correctly shaped even with 0 shocks
                D_num = np.zeros((n_state, 0), dtype=float)

        except Exception as e:
            print(f"FATAL ERROR during numerical substitution (evalf): {e}")
            # Try to find which matrix failed?
            # ...
            raise RuntimeError("Numerical substitution of parameters failed.") from e


        # --- Validate Matrix Dimensions ---
        expected_A_shape = (n_state, n_state)
        expected_B_shape = (n_state, n_state)
        expected_C_shape = (n_state, n_state)
        expected_D_shape = (n_state, n_shocks)

        if A_num.shape != expected_A_shape: raise ValueError(f"Matrix A shape error: Expected {expected_A_shape}, Got {A_num.shape}")
        if B_num.shape != expected_B_shape: raise ValueError(f"Matrix B shape error: Expected {expected_B_shape}, Got {B_num.shape}")
        if C_num.shape != expected_C_shape: raise ValueError(f"Matrix C shape error: Expected {expected_C_shape}, Got {C_num.shape}")
        if D_num.shape != expected_D_shape: raise ValueError(f"Matrix D shape error: Expected {expected_D_shape}, Got {D_num.shape}")

        print("Numerical matrices A, B, C, D calculated successfully.")

        # --- Save Results (Optional) ---
        if file_path:
             # Pass the actual parameter values used for saving metadata
            self._save_final_matrices(file_path, A_num, B_num, C_num, D_num,
                                      state_names, shock_names, self.param_names, param_dict_values)

        return A_num, B_num, C_num, D_num, state_names, shock_names


    def _generate_matrix_assignments_code_helper(self, matrix_sym, matrix_name):
        """
        Generates Python code lines for initializing and assigning non-zero
        elements of a symbolic matrix (A, B, C, or D) to a numpy array.

        Args:
            matrix_sym (sympy.Matrix): The symbolic matrix.
            matrix_name (str): The name for the numpy array ('A', 'B', 'C', 'D').

        Returns:
            str: A block of Python code as a single string.
        """
        try:
            rows, cols = matrix_sym.shape
        except Exception as e:
            # Handle cases where matrix might be None or not have shape
            print(f"Warning: Could not get shape for symbolic matrix '{matrix_name}'. Error: {e}")
            # Decide how to handle: return empty init or raise error?
            # Let's return an empty init for robustness, assuming cols=0 if rows is known?
            # This needs careful handling based on where it's called.
            # For now, assume it has a shape if called.
            raise ValueError(f"Cannot generate code for matrix '{matrix_name}' without shape.") from e

        indent = "    " # 4 spaces for indentation inside the function
        code_lines = []

        # Initialize the numpy array with zeros
        # Handle the case of 0 columns correctly (e.g., for D with no shocks)
        code_lines.append(f"{indent}{matrix_name} = np.zeros(({rows}, {cols}), dtype=float)")

        # Find and assign non-zero elements
        assignments = []
        for r in range(rows):
            for c in range(cols):
                try:
                    element = matrix_sym[r, c]
                except IndexError:
                    # Should not happen if looping within shape bounds
                    print(f"Warning: IndexError accessing {matrix_name}[{r},{c}]")
                    continue

                # Check if the element is explicitly non-zero
                # Use sympy.S.Zero for reliable comparison with symbolic zero
                if element != 0 and element is not sympy.S.Zero:
                    try:
                        # Convert the symbolic element to a Python string
                        # full_prec=False might truncate long floats, consider True if needed
                        # Use sympy.pycode for potentially better Python code generation
                        # expr_str = sympy.pycode(element)
                        # Or stick to sstr for more direct representation
                        expr_str = sympy.sstr(element, full_prec=True) # Use full precision

                        # Replace intermediate power notation if needed (e.g., Pow(x, 2) -> x**2)
                        # This is often handled by sstr/pycode, but manual cleanup might be req'd
                        # expr_str = expr_str.replace("Pow(", "(").replace(", ", "**") # Basic replacement

                        assignments.append(f"{indent}{matrix_name}[{r}, {c}] = {expr_str}")
                    except Exception as str_e:
                        print(f"Warning: Could not convert element {matrix_name}[{r},{c}] to string. Element: {element}. Error: {str_e}")
                        # Add a comment indicating the failure
                        assignments.append(f"{indent}# Error assigning {matrix_name}[{r},{c}]. Symbolic value: {element}")


        if assignments:
            code_lines.append(f"{indent}# Fill {matrix_name} non-zero elements")
            code_lines.extend(assignments)
        # else: No non-zero elements found, just the zeros init line remains

        return "\n".join(code_lines)


    def generate_matrix_function_file(self, filename="jacobian_matrices.py"):
        """
        Generates a Python file containing a function `jacobian_matrices(theta)`
        that takes a list/array of parameter values `theta` (in the order defined
        in the .mod file's `parameters` block) and returns the numerical A, B, C, D
        matrices, plus state and shock names.

        Args:
            filename (str): The name of the Python file to generate.
        """
        function_name = "jacobian_matrices" # Standardized name
        print(f"\n--- Generating Python Function File: {filename} ---")

        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
             if not self.state_vars_ordered and not self.final_equations_for_jacobian:
                   # Handle empty model case: generate a function returning empty matrices
                   print("Warning: Model is empty. Generating function returning empty matrices.")
                   n_state = 0
                   n_shocks = len(self.shock_names)
                   state_names = []
                   shock_names = self.shock_names
                   # Generate minimal code for this case
                   file_lines = [
                       f"# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'",
                       f"# Generated: {datetime.datetime.now().isoformat()}",
                       "# Model appears empty (no dynamic states/equations).",
                       "import numpy as np", "",
                       f"def {function_name}(theta):",
                       f"    # Model is empty. Expected {len(self.param_names)} parameters.",
                       f"    if len(theta) != {len(self.param_names)}: raise ValueError('Incorrect parameter count')",
                       f"    A = np.zeros((0, 0), dtype=float)",
                       f"    B = np.zeros((0, 0), dtype=float)",
                       f"    C = np.zeros((0, 0), dtype=float)",
                       f"    D = np.zeros((0, {n_shocks}), dtype=float)",
                       f"    state_names = []",
                       f"    shock_names = {repr(shock_names)}",
                       f"    return A, B, C, D, state_names, shock_names", ""
                   ]
                   final_file_content = "\n".join(file_lines)
                   # Write the file
                   try:
                       dir_name = os.path.dirname(filename)
                       if dir_name: os.makedirs(dir_name, exist_ok=True)
                       with open(filename, "w", encoding='utf-8') as f:
                           f.write(final_file_content)
                       print(f"Successfully generated empty function file: {filename}")
                   except Exception as e:
                       print(f"Error writing empty function file {filename}: {e}")
                   return # Stop here for empty model
             else:
                raise ValueError("Preprocessing must be run successfully first (final equations/states missing or inconsistent).")


        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)

        # --- Re-calculate Symbolic Jacobians (ensure they are fresh) ---
        # This repeats the logic from get_numerical_ABCD, but without substitution
        print("Recalculating symbolic Jacobians for code generation...")
        try:
            state_vec_t = sympy.Matrix(self.state_vars_ordered)
            state_vec_tp1_list = [self.symbols.get(f"{s.name}_p1", sympy.Symbol(f"{s.name}_p1")) for s in self.state_vars_ordered]
            state_vec_tm1_list = [self.symbols.get(f"{s.name}_m1", sympy.Symbol(f"{s.name}_m1")) for s in self.state_vars_ordered]
            state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
            state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)

            shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
            shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None

            eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])

            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            D_sym = -eq_vec.jacobian(shock_vec) if shock_vec is not None and n_shocks > 0 else sympy.zeros(n_state, n_shocks)
            print("Symbolic Jacobians recalculated.")

        except Exception as e:
            print(f"FATAL ERROR: Could not recalculate symbolic Jacobians for code generation: {e}")
            raise RuntimeError("Jacobian calculation failed during code generation.") from e


        # --- Identify Parameters Used in Matrices ---
        # Collect all free symbols from all symbolic matrices
        all_matrix_symbols = set().union(
            *(mat.free_symbols for mat in [A_sym, B_sym, C_sym, D_sym] if mat is not None)
        )
        # Filter these down to only those that are declared parameters
        parameter_symbols_in_matrices = {
            sym for sym in all_matrix_symbols if sym.name in self.param_names
        }
        used_params_ordered_by_declaration = [
            p_name for p_name in self.param_names if self.symbols.get(p_name) in parameter_symbols_in_matrices
        ]

        # Check if any symbols in matrices are *not* params, states, or known symbols (potential issue)
        # state_related_symbols = set(state_vec_t) | set(state_vec_tp1) | set(state_vec_tm1) | set(shock_vec if shock_vec else [])
        # unknown_symbols = all_matrix_symbols - parameter_symbols_in_matrices - state_related_symbols - set(self.symbols.values())
        # if unknown_symbols:
        #      print(f"Warning: Unknown symbols found in symbolic matrices: {unknown_symbols}")


        # --- Generate Python Code Strings for Each Matrix ---
        print("Generating Python code strings for matrix assignments...")
        try:
            code_A = self._generate_matrix_assignments_code_helper(A_sym, 'A')
            code_B = self._generate_matrix_assignments_code_helper(B_sym, 'B')
            code_C = self._generate_matrix_assignments_code_helper(C_sym, 'C')
            code_D = self._generate_matrix_assignments_code_helper(D_sym, 'D')
            print("Code string generation complete.")
        except Exception as e:
             print(f"FATAL ERROR: Failed to generate code strings for matrices: {e}")
             raise RuntimeError("Code string generation failed.") from e


        # --- Assemble the Full Python File Content ---
        file_lines = []
        file_lines.append(f"# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'")
        file_lines.append(f"# Generated: {datetime.datetime.now().isoformat()}")
        file_lines.append("# This function computes the A, B, C, D matrices for the system:")
        file_lines.append("# 0 = A E[y_{t+1}] + B y_t + C y_{t-1} + D e_t")
        file_lines.append("")
        file_lines.append("import numpy as np")
        # Import math functions often used in Dynare models if needed
        # Add more imports if specific functions (exp, log, sqrt, etc.) are common
        file_lines.append("from math import *") # Use carefully, consider explicit imports
        file_lines.append("")
        file_lines.append(f"def {function_name}(theta):")
        file_lines.append('    """')
        file_lines.append("    Computes the numerical A, B, C, D matrices for the solved model.")
        file_lines.append("")
        file_lines.append("    Parameters:")
        file_lines.append("    -----------")
        file_lines.append("    theta : list or numpy array")
        file_lines.append(f"        Vector of parameter values in the order defined in the .mod file:")
        file_lines.append(f"        {self.param_names}")
        file_lines.append("")
        file_lines.append("    Returns:")
        file_lines.append("    --------")
        file_lines.append("    A, B, C : numpy.ndarray")
        file_lines.append("        State space matrices (n_states x n_states).")
        file_lines.append("    D : numpy.ndarray")
        file_lines.append("        Shock impact matrix (n_states x n_shocks).")
        file_lines.append("    state_names : list[str]")
        file_lines.append("        Ordered list of state variable names.")
        file_lines.append("    shock_names : list[str]")
        file_lines.append("        Ordered list of shock names.")
        file_lines.append('    """')
        file_lines.append("")

        # Unpack parameters from theta using the original declared order
        file_lines.append("    # --- Unpack Parameters ---")
        n_params_expected = len(self.param_names)
        file_lines.append(f"    expected_param_count = {n_params_expected}")
        file_lines.append("    if len(theta) != expected_param_count:")
        file_lines.append("        raise ValueError(f'Expected {expected_param_count} parameters in theta, but got {len(theta)}.')")
        file_lines.append("")
        file_lines.append("    try:")
        # Only unpack parameters actually used in the matrices
        if not used_params_ordered_by_declaration and self.param_names:
             file_lines.append(f"{indent}# No declared parameters seem to be used in the matrices.")
        elif not self.param_names:
             file_lines.append(f"{indent}# Model has no declared parameters.")
        else:
             for i, p_name in enumerate(self.param_names):
                  # Check if this parameter was actually needed for A,B,C,D
                  if p_name in used_params_ordered_by_declaration:
                       file_lines.append(f"        {p_name} = theta[{i}]")
                  else:
                       # Optionally add commented-out line for unused params
                       file_lines.append(f"        # {p_name} = theta[{i}] # (Unused in final matrices)")

        file_lines.append("    except IndexError:")
        file_lines.append("        # This should be caught by the length check above, but added for safety")
        file_lines.append("        raise IndexError('Failed to unpack parameters from theta vector.')")
        file_lines.append("")

        # Add matrix initialization and assignments
        file_lines.append("    # --- Initialize and Fill Matrices ---")
        file_lines.append(code_A)
        file_lines.append("")
        file_lines.append(code_B)
        file_lines.append("")
        file_lines.append(code_C)
        file_lines.append("")
        file_lines.append(code_D)
        file_lines.append("")

        # Prepare return values
        file_lines.append("    # --- Return Results ---")
        # Embed the state and shock names directly into the function
        # Use repr() for safe string representation of the lists
        final_state_names = [s.name for s in self.state_vars_ordered]
        file_lines.append(f"    state_names = {repr(final_state_names)}")
        file_lines.append(f"    shock_names = {repr(self.shock_names)}")
        file_lines.append("")
        file_lines.append("    return A, B, C, D, state_names, shock_names")
        file_lines.append("") # Final newline

        final_file_content = "\n".join(file_lines)

        # --- Write the generated code to file ---
        try:
            dir_name = os.path.dirname(filename)
            if dir_name: # Ensure directory exists if filename includes path
                os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(final_file_content)
            print(f"Successfully generated function file: {filename}")
        except IOError as e:
            print(f"ERROR: Could not write function file '{filename}'. Check permissions. Error: {e}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while writing function file '{filename}': {e}")
            raise


    # ==================================================
    # process_model (MODIFIED to accept forced_state_order)
    # and helpers (_save_intermediate_file, _save_final_matrices) go here
    # ==================================================
    def process_model(self, param_dict_values_or_list,
                      output_dir_intermediate=None,
                      output_dir_final=None,
                      generate_function=True,
                      forced_state_order=None): # <-- New optional argument
        """
        Runs the full parsing and matrix generation pipeline.

        Args:
            param_dict_values_or_list: Dict or ordered list/array of parameter values.
                                       If list/array, order must match `self.param_names`.
            output_dir_intermediate: Directory to save intermediate text files.
            output_dir_final: Directory to save final .pkl and .py files.
            generate_function: Boolean flag to generate the python function file.
            forced_state_order (list[str], optional):
                A list of variable names specifying the desired order for the state vector y_t.
                If provided, it must contain exactly the same variables as the parser
                identifies as final dynamic variables (including aux). Defaults to None,
                in which case the parser uses its default ordering.

        Returns:
            tuple: (A, B, C, D, state_names, shock_names) on success, None on failure.
                   Matrices are numpy arrays, names are lists of strings.
        """
        print("\n--- Starting Model Processing Pipeline ---")
        base_name = os.path.splitext(os.path.basename(self.mod_file_path))[0]

        # --- Parameter Handling ---
        param_dict_values = {}
        if isinstance(param_dict_values_or_list, (list, tuple, np.ndarray)):
            if len(param_dict_values_or_list) != len(self.param_names):
                raise ValueError(
                    f"Input parameter list/array length ({len(param_dict_values_or_list)}) "
                    f"does not match declared parameters ({len(self.param_names)}): {self.param_names}"
                )
            # Create dict from list using declared parameter order
            param_dict_values = {
                name: val for name, val in zip(self.param_names, param_dict_values_or_list)
            }
            print(f"Received parameter values as list/array, mapped to: {list(param_dict_values.keys())}")
        elif isinstance(param_dict_values_or_list, dict):
            param_dict_values = param_dict_values_or_list
            # Check for missing/extra keys compared to declared parameters
            declared_set = set(self.param_names)
            provided_set = set(param_dict_values.keys())
            missing_keys = declared_set - provided_set
            extra_keys = provided_set - declared_set
            if missing_keys:
                # This is now an error, values must be provided for all declared params
                raise ValueError(f"Input parameter dict is missing values for declared parameters: {sorted(list(missing_keys))}")
            if extra_keys:
                print(f"Warning: Input parameter dict contains keys not in declared parameters: {sorted(list(extra_keys))}")
            print(f"Received parameter values as dictionary.")
        else:
            raise TypeError("Input 'param_dict_values_or_list' must be a dict, list, tuple, or numpy array.")

        # --- Store Forced State Order ---
        # Store the user's choice in the instance variable for _define_state_vector to access
        self.forced_state_order = forced_state_order
        if self.forced_state_order:
             print(f"User has requested a specific state vector order.")

        # --- Prepare Output Paths ---
        fpaths_inter = {i: None for i in range(7)} # Intermediate files
        final_matrices_pkl = None
        function_py = None

        if output_dir_intermediate:
            os.makedirs(output_dir_intermediate, exist_ok=True)
            inter_names = [
                "0_original_eqs", "1_timing", "2_static_elim", "3_static_sub",
                "4_aux_handling", "5_state_def", "6_final_eqs"
            ]
            fpaths_inter = {
                i: os.path.join(output_dir_intermediate, f"{i}_{base_name}_{name}.txt")
                for i, name in enumerate(inter_names)
            }

        if output_dir_final:
            os.makedirs(output_dir_final, exist_ok=True)
            final_matrices_pkl = os.path.join(output_dir_final, f"{base_name}_matrices.pkl")
            if generate_function:
                function_py = os.path.join(output_dir_final, f"{base_name}_jacobian_matrices.py")


        # --- Execute Processing Stages ---
        try:
            # Stage 0 (already done in init)
            self._save_intermediate_file(fpaths_inter[0], ["Stage 0 Parsed Equations"],
                                         self.sympy_equations_original, "Original Sympy Equations")
            # Stage 1
            self._analyze_variable_timing()
            timing_lines = ["Stage 1 Timing Analysis Complete"]
            # Create summary for timing file
            for var_name in sorted(self.var_names):
                info = self.var_timing_info.get(var_name, None)
                if info:
                    timing_lines.append(f"- {var_name}: Curr={info['appears_current']}, Lead={info['max_lead']}, Lag={info['min_lag']}")
                else:
                    timing_lines.append(f"- {var_name}: Not found in equations.")
            self._save_intermediate_file(fpaths_inter[1], timing_lines)

            # Stage 2
            self._identify_and_eliminate_static_vars()
            static_lines = ["Stage 2 Static Elimination Complete",
                            f"Solved Static Vars: {[s.name for s in self.static_subs.keys()]}"]
            self._save_intermediate_file(fpaths_inter[2], static_lines,
                                         self.equations_after_static_elim, "Equations After Static Elimination")

            # Stage 3
            self._substitute_static_vars()
            self._save_intermediate_file(fpaths_inter[3], ["Stage 3 Static Substitution Complete"],
                                         self.equations_after_static_sub, "Equations After Static Substitution")

            # Stage 4
            self._handle_aux_vars()
            aux_lines = ["Stage 4 Aux Handling Complete",
                         f"Final dynamic vars (names): {self.final_dynamic_var_names}",
                         f"Aux Lead Vars: {list(self.aux_lead_vars.keys())}",
                         f"Aux Lag Vars: {list(self.aux_lag_vars.keys())}"]
            # Combine equations after sub + definitions for saving
            eqs_after_aux_handling = self.equations_after_aux_sub + self.aux_var_definitions
            self._save_intermediate_file(fpaths_inter[4], aux_lines,
                                         eqs_after_aux_handling, "Equations After Aux Handling (Substituted + Definitions)")

            # Stage 5 (Uses self.forced_state_order if set)
            self._define_state_vector()
            state_lines = ["Stage 5 State Definition Complete"]
            state_order_names = [s.name for s in self.state_vars_ordered]
            state_lines.append(f"State Vector ({len(state_order_names)}): {state_order_names}")
            self._save_intermediate_file(fpaths_inter[5], state_lines)

            # Stage 6
            self._build_final_equations()
            build_lines = ["Stage 6 Final Equation Build Complete",
                           f"N_States={len(self.state_vars_ordered)}, N_Eqs={len(self.final_equations_for_jacobian)}"]
            self._save_intermediate_file(fpaths_inter[6], build_lines,
                                         self.final_equations_for_jacobian, "Final Equation System for Jacobian")

            # Stage 7
            # Pass the validated dictionary of parameter values
            A, B, C, D, state_names, shock_names = self.get_numerical_ABCD(
                param_dict_values,
                file_path=final_matrices_pkl # Pass path for optional saving
            )

            # --- Generate Python Function (Optional) ---
            if generate_function and function_py:
                self.generate_matrix_function_file(filename=function_py)

            print("\n--- Model Processing Successful ---")
            return A, B, C, D, state_names, shock_names

        except Exception as e:
            print(f"\n--- FATAL ERROR during model processing: {type(e).__name__}: {e} ---")
            import traceback
            traceback.print_exc()
            # Optionally save final equations even on error if they exist
            if hasattr(self, 'final_equations_for_jacobian') and self.final_equations_for_jacobian:
                 self.save_final_equations_to_txt(filename="ERROR_final_equations_on_fail.txt")
            return None # Indicate failure


    def _save_intermediate_file(self, file_path, lines, equations=None, equations_title="Equations"):
        """Saves intermediate step details to a text file for debugging."""
        if not file_path:
            # print("Debug: No file path provided for intermediate file, skipping save.")
            return # Skip if no path is given

        try:
            # Ensure directory exists
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(file_path, "w", encoding='utf-8') as f:
                f.write(f"--- {os.path.basename(file_path)} ---\n")
                f.write(f"--- Model: {self.mod_file_path} ---\n")
                f.write(f"--- Generated: {datetime.datetime.now().isoformat()} ---\n\n")
                # Write header lines
                f.write("\n".join(lines))
                f.write("\n") # Add a newline

                # Write equations if provided
                if equations is not None:
                    f.write(f"\n--- {equations_title} ({len(equations)}) ---\n")
                    if equations:
                        for i, eq in enumerate(equations):
                            if eq is not None and hasattr(eq, 'lhs'):
                                # Use sstr for readable output, full_prec=False for brevity
                                try:
                                     f.write(f"  Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                                except Exception as e:
                                     f.write(f"  Eq {i+1}: [Error converting LHS to string: {e}]\n")
                            elif eq is None:
                                f.write(f"  Eq {i+1}: [None Equation Placeholder]\n")
                            else:
                                # Handle unexpected items in the list
                                f.write(f"  Eq {i+1}: [Non-Equation Object: {type(eq)}] {str(eq)}\n")
                    else:
                        f.write("  [No equations in this list]\n")

            # print(f"Intermediate file saved: {file_path}") # Optional: log success

        except Exception as e:
            print(f"Warning: Could not save intermediate file '{file_path}'. Error: {type(e).__name__} - {e}")


    def _save_final_matrices(self, file_path, A, B, C, D, state_names, shock_names, param_names, param_values_used):
        """
        Saves the final numerical matrices (A, B, C, D) and associated metadata
        to a pickle file (.pkl) and a human-readable text file (.txt).

        Args:
            file_path (str): Base path for saving (e.g., 'output/model_matrices.pkl').
                             The .txt file will have the same base name.
            A, B, C, D (numpy.ndarray): The numerical matrices.
            state_names (list[str]): Ordered list of state variable names.
            shock_names (list[str]): Ordered list of shock names.
            param_names (list[str]): Ordered list of ALL declared parameter names.
            param_values_used (dict): Dictionary of parameter name -> value used for this calculation.
        """
        if not file_path:
            print("Warning: No file path provided for saving final matrices.")
            return

        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # --- Prepare Data for Pickle ---
        matrix_data = {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'state_names': state_names,
            'shock_names': shock_names,
            'param_names_ordered': param_names, # All declared params in order
            'param_values_used': param_values_used, # Values used for this run
            'model_file': self.mod_file_path,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # --- Save Pickle File ---
        try:
            with open(file_path, "wb") as f:
                pickle.dump(matrix_data, f)
            print(f"Matrices and metadata saved to pickle file: {file_path}")
        except Exception as e:
            print(f"ERROR: Could not save matrices to pickle file '{file_path}'. Error: {e}")
            # Decide whether to continue to save TXT file or stop
            # return # Option: stop if pickle fails

        # --- Save Human-Readable Text File ---
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        try:
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write(f"# Numerical Matrices and Metadata\n")
                f.write(f"# Model File: {self.mod_file_path}\n")
                f.write(f"# Generated: {matrix_data['timestamp']}\n\n")

                f.write(f"State Names (Order: {len(state_names)}):\n{state_names}\n\n")
                f.write(f"Shock Names ({len(shock_names)}):\n{shock_names}\n\n")
                f.write(f"Declared Parameter Names (Order: {len(param_names)}):\n{param_names}\n\n")
                f.write(f"Parameter Values Used ({len(param_values_used)}):\n")
                # Pretty print the dictionary
                for pname, pval in param_values_used.items():
                     f.write(f"  {pname}: {pval}\n")
                f.write("\n")


                # Configure numpy printing options for readability
                np.set_printoptions(linewidth=200, precision=6, suppress=True, threshold=np.inf)

                f.write(f"A Matrix ({A.shape}):\n{np.array2string(A, separator=', ')}\n\n")
                f.write(f"B Matrix ({B.shape}):\n{np.array2string(B, separator=', ')}\n\n")
                f.write(f"C Matrix ({C.shape}):\n{np.array2string(C, separator=', ')}\n\n")

                f.write(f"D Matrix ({D.shape}):\n")
                if D.size > 0: # Check if D is not empty
                    f.write(f"{np.array2string(D, separator=', ')}\n\n")
                elif D.shape[1] == 0: # Handle n_states x 0 case
                    f.write("[No shocks defined/used]\n\n")
                else: # Handle 0 x n_shocks case (empty model)
                     f.write("[Model is empty or has no states]\n\n")

            print(f"Human-readable matrices saved to text file: {txt_path}")
        except Exception as e:
            print(f"Warning: Could not save matrices to text file '{txt_path}'. Error: {e}")


    def save_final_equations_to_txt(self, filename="final_equations.txt"):
        """Saves the final system of equations used for Jacobian calculation to a text file."""
        print(f"\n--- Saving Final Equations to: {filename} ---")

        if not hasattr(self, 'final_equations_for_jacobian') or not self.final_equations_for_jacobian:
             # Check if the model was empty from the start
             if not hasattr(self, 'state_vars_ordered') or not self.state_vars_ordered:
                  message = "Model processing resulted in no final equations (likely an empty or static model)."
                  print(f"Warning: {message} File not saved.")
             else:
                  message = "Final equations list not generated or is empty. Cannot save."
                  print(f"Warning: {message} File not saved.")
             # Write a file indicating this? Or just skip? Skipping seems reasonable.
             return

        try:
            # Ensure directory exists
            dir_name = os.path.dirname(filename)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"# Final System Equations Used for Jacobian Calculation\n")
                f.write(f"# Model: {os.path.basename(self.mod_file_path)}\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                n_eqs = len(self.final_equations_for_jacobian)
                f.write(f"# Number of Equations: {n_eqs}\n")

                # Add state variable order if available
                if hasattr(self, 'state_vars_ordered') and self.state_vars_ordered:
                    state_order_names = [s.name for s in self.state_vars_ordered]
                    f.write(f"# State Variables Order ({len(state_order_names)}): {state_order_names}\n\n")
                else:
                    f.write("# State variable order not determined at time of saving.\n\n")

                # Write the equations
                for i, eq in enumerate(self.final_equations_for_jacobian):
                    if eq is not None and hasattr(eq, 'lhs'):
                        try:
                            # Use sstr for more readable output
                            f.write(f"Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                        except Exception as e:
                             f.write(f"Eq {i+1}: [Error converting equation LHS to string: {e}]\n")
                    elif eq is None:
                        f.write(f"Eq {i+1}: [None Equation Placeholder]\n")
                    else:
                        # Log unexpected object type
                        f.write(f"Eq {i+1}: [Invalid Equation Object Type: {type(eq)}] {str(eq)}\n")

            print(f"Successfully saved final equations ({n_eqs}) to {filename}")

        except Exception as e:
            print(f"ERROR: Could not write final equations file '{filename}'. Error: {type(e).__name__} - {e}")


# ==================================================
# Main Execution Block (Example Usage - MODIFIED)
# ==================================================
if __name__ == "__main__":
    # Determine script directory dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    print(f"Script directory: {script_dir}")
    os.chdir(script_dir) # Change working directory to script dir for relative paths

    mod_file = "qpm_model.dyn" # Assumed to be in the same directory
    output_dir_inter = os.path.join(script_dir, "model_files_intermediate_final") # Subfolder
    output_dir_final = os.path.join(script_dir, "model_files_numerical_final")   # Subfolder

    # Ensure output directories exist
    os.makedirs(output_dir_inter, exist_ok=True)
    os.makedirs(output_dir_final, exist_ok=True)

    # --- Parameters ---
    # Use the same values as the manual example (AFTER fixing rho_rs)
    parameter_values_dict = {
        'b1': 0.7,  
        'b4': 0.7,
        'a1': 0.5,   
        'a2': 0.1,
        'g1': 0.7, 
        'g2': 0.3, 
        'g3': 0.25,
        'rho_L_GDP_GAP': 0.75,
        'rho_DLA_CPI':   0.75,
        'rho_rs':        0.75, 
        'rho_rs2':       0.01
    }
    print(f"\nParameter values used: {parameter_values_dict}")

    # --- Initialize Parser ---
    try:
        parser = DynareParser(mod_file)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Model file '{mod_file}' not found in directory '{script_dir}'.")
        print("Please ensure the .dyn file exists in the same directory as the script.")
        sys.exit(1)
    except ValueError as ve: # Catch model block parsing errors etc.
         print(f"\nFATAL ERROR initializing parser: {ve}")
         sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR during parser initialization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Prepare Parameter Vector (theta) ---
    # Create theta based on the order the parser found in the .mod file
    try:
        parameter_theta = [parameter_values_dict[pname] for pname in parser.param_names]
        print(f"\nParameter vector 'theta' created (order: {parser.param_names})")
    except KeyError as e:
        print(f"\nFATAL ERROR: Parameter '{e}' declared in .mod file but missing from provided dictionary.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR creating parameter vector 'theta': {e}")
        sys.exit(1)

    # --- Define the Desired State Order (from manual example) ---
    # This order MUST exactly match the variables the parser identifies
    # as dynamic after static/aux handling. Run once without forced order
    # to see the parser's final_dynamic_var_names if unsure.
    manual_state_order = [
        'RES_DLA_CPI', 'RES_L_GDP_GAP', 'RES_RS', 'aux_RES_RS_lag1',
        'DLA_CPI', 'L_GDP_GAP', 'RS',
        'aux_DLA_CPI_lead1', 'aux_DLA_CPI_lead2'
    ]
    print(f"\nRequesting specific state order: {manual_state_order}")


    # --- Process Model with Forced Order ---
    # Pass the parameter vector 'theta' and the desired state order
    result = parser.process_model(
        parameter_theta, # Pass the list/array
        output_dir_intermediate=output_dir_inter,
        output_dir_final=output_dir_final,
        generate_function=True,
        forced_state_order=manual_state_order # Pass the desired order
    )

    # --- Check and Use Results ---
    if result:
        A_direct, B_direct, C_direct, D_direct, state_names_direct, shock_names_direct = result
        print("\n--- Results from Parser (using forced order) ---")
        print("State Order:", state_names_direct) # Should match manual_state_order
        print(f"Shock Order: {shock_names_direct}")
        print(f"Matrix Shapes: A:{A_direct.shape}, B:{B_direct.shape}, C:{C_direct.shape}, D:{D_direct.shape}")

        # Optional: Print small matrices
        if A_direct.size <= 100:
             np.set_printoptions(linewidth=150, precision=4, suppress=True)
             print("A Matrix:\n", A_direct)
             print("B Matrix:\n", B_direct)
             print("C Matrix:\n", C_direct)
             print("D Matrix:\n", D_direct)

        # Save final equations for reference
        parser.save_final_equations_to_txt(os.path.join(output_dir_final, f"{base_name}_final_equations_forced_order.txt"))

        # --- Test Generated Function (if created) ---
        function_file = os.path.join(output_dir_final, f"{base_name}_jacobian_matrices.py")
        if os.path.exists(function_file):
            print(f"\n--- Testing Generated Function ({function_file}) ---")
            abs_function_file = os.path.abspath(function_file)
            module_name = os.path.splitext(os.path.basename(function_file))[0]

            try:
                # Import the generated module
                spec = importlib.util.spec_from_file_location(module_name, abs_function_file)
                if spec is None or spec.loader is None:
                     print(f"Error: Could not create module spec from {abs_function_file}")
                else:
                    mod_matrices = importlib.util.module_from_spec(spec)
                    # Add to sys.modules BEFORE execution to handle potential relative imports within the generated file (if any)
                    sys.modules[module_name] = mod_matrices
                    spec.loader.exec_module(mod_matrices)

                    # Call the function
                    print(f"Calling generated function '{module_name}.jacobian_matrices'...")
                    A_f, B_f, C_f, D_f, states_f, shocks_f = mod_matrices.jacobian_matrices(parameter_theta)
                    print("Function call successful.")

                    # Compare results
                    print("Comparing matrices from direct calculation vs generated function...")
                    all_match = True
                    try:
                        np.testing.assert_allclose(A_direct, A_f, atol=1e-8, rtol=1e-8, err_msg="A matrix mismatch")
                        print("  - A matrices match.")
                    except AssertionError as ae:
                        print(f"  - !!! {ae} !!!")
                        all_match = False

                    try:
                        np.testing.assert_allclose(B_direct, B_f, atol=1e-8, rtol=1e-8, err_msg="B matrix mismatch")
                        print("  - B matrices match.")
                    except AssertionError as ae:
                        print(f"  - !!! {ae} !!!")
                        all_match = False

                    try:
                        np.testing.assert_allclose(C_direct, C_f, atol=1e-8, rtol=1e-8, err_msg="C matrix mismatch")
                        print("  - C matrices match.")
                    except AssertionError as ae:
                        print(f"  - !!! {ae} !!!")
                        all_match = False

                    try:
                        np.testing.assert_allclose(D_direct, D_f, atol=1e-8, rtol=1e-8, err_msg="D matrix mismatch")
                        print("  - D matrices match.")
                    except AssertionError as ae:
                        print(f"  - !!! {ae} !!!")
                        all_match = False

                    # Compare names
                    if state_names_direct == states_f:
                        print("  - State names match.")
                    else:
                        print(f"  - !!! State names mismatch !!!")
                        print(f"    Direct: {state_names_direct}")
                        print(f"    Func:   {states_f}")
                        all_match = False

                    if shock_names_direct == shocks_f:
                        print("  - Shock names match.")
                    else:
                         print(f"  - !!! Shock names mismatch !!!")
                         print(f"    Direct: {shock_names_direct}")
                         print(f"    Func:   {shocks_f}")
                         all_match = False


                    if all_match:
                        print(">>> Generated function test PASSED. <<<")
                    else:
                        print(">>> Generated function test FAILED. <<<")


            except ImportError as ie:
                 print(f"Error importing generated module: {ie}")
                 print("Check if the generated file has syntax errors or missing dependencies.")
            except AttributeError as ae:
                 print(f"Error accessing function in generated module: {ae}")
                 print(f"Does '{function_file}' contain the function 'jacobian_matrices'?")
            except Exception as test_e:
                print(f"ERROR testing generated function: {type(test_e).__name__}: {test_e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nGenerated function file not found: {function_file}. Cannot test.")
    else:
        print("\n--- Model processing FAILED. ---")
        # Error messages should have been printed during the process_model call
        sys.exit(1) # Exit with error code if processing failed