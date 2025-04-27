import sympy
import re
import numpy as np
import os
import pickle
import collections # For defaultdict
import sys
import importlib.util # For testing generated function
import datetime # For timestamp

# --- Helper Function for Time Shifting Expressions ---
# (Using the robust version from previous answers - adheres to standard Python)
def time_shift_expression(expr, shift, parser_symbols, var_names_set):
    """
    Shifts the time index of variables within a Sympy expression. Handles base,
    lead (_p), lag (_m), aux_lead, and aux_lag vars based on parser_symbols.
    Uses standard Python formatting.
    """
    if shift == 0:
        return expr

    subs_dict = {}
    atoms = expr.free_symbols

    for atom in atoms:
        atom_name = atom.name
        base_name = None
        current_k = 0
        is_var_type = False # Includes base vars, aux_lead, aux_lag

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
            if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                 base_name = base_name_cand
                 current_k = int(match_lead.group(2))
                 is_var_type = True
            else:
                 base_name = None
        elif match_lag:
             base_name_cand = match_lag.group(1)
             if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                 base_name = base_name_cand
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
            if new_k == 0:
                clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
                if clean_base_match:
                    clean_base = clean_base_match.group(1)
                else:
                    clean_base = base_name
                if clean_base in var_names_set:
                    new_sym_name = clean_base
                else:
                     if base_name in parser_symbols:
                         new_sym_name = base_name
                     else:
                         continue
            elif new_k > 0:
                clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
                if clean_base_match: clean_base = clean_base_match.group(1)
                else: clean_base = base_name
                prefix = "aux_" if base_name.lower().startswith("aux_") else ""
                new_sym_name = f"{prefix}{clean_base}_p{new_k}"
            else: # new_k < 0
                clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
                if clean_base_match: clean_base = clean_base_match.group(1)
                else: clean_base = base_name
                prefix = "aux_" if base_name.lower().startswith("aux_") else ""
                new_sym_name = f"{prefix}{clean_base}_m{abs(new_k)}"

            if new_sym_name not in parser_symbols:
                parser_symbols[new_sym_name] = sympy.Symbol(new_sym_name)
            subs_dict[atom] = parser_symbols[new_sym_name]

    try:
        shifted_expr = expr.xreplace(subs_dict)
    except Exception:
        try:
            shifted_expr = expr.subs(subs_dict)
        except Exception as e2:
            print(f"Error: Both xreplace and subs failed in time_shift_expression: {e2}")
            shifted_expr = expr
    return shifted_expr


class DynareParser:
    """
    Parses a Dynare-style .mod file, performs model reduction for UQME form,
    and optionally generates a Python function for numerical Jacobians.
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
        # Intermediate stages' results
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        self.static_subs = {}
        self.equations_after_static_elim = []
        self.equations_after_static_sub = []
        self.aux_lead_vars = {}
        self.aux_lag_vars = {}
        self.aux_var_definitions = []
        self.equations_after_aux_sub = []
        self.final_dynamic_var_names = []
        self.state_vars_ordered = []
        self.state_var_map = {}
        self.final_equations_for_jacobian = []
        self.last_param_values = {} # Store last used params

        print(f"--- Parsing Mod File: {self.mod_file_path} ---")
        self._parse_mod_file()
        self.var_names_set = set(self.var_names)
        print("\n--- Creating Initial Symbols ---")
        self._create_initial_sympy_symbols()
        print("\n--- Parsing Equations to Sympy ---")
        self._parse_equations_to_sympy()
        print(f"Parser initialized. Vars:{len(self.var_names)}, Params:{len(self.param_names)}, Shocks:{len(self.shock_names)}")
        print(f"Original {len(self.sympy_equations_original)} equations parsed.")

    # --- Stage 0 Methods ---
    def _parse_mod_file(self):
        """Reads .mod file content and extracts declarations."""
        if not os.path.isfile(self.mod_file_path):
            raise FileNotFoundError(f"Mod file not found: {self.mod_file_path}")
        try:
            with open(self.mod_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise IOError(f"Error reading mod file {self.mod_file_path}: {e}") from e

        # Remove comments first
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        # Define patterns (case-insensitive, non-greedy)
        var_pattern = re.compile(r"var\s+(.*?);", re.IGNORECASE | re.DOTALL)
        varexo_pattern = re.compile(r"varexo\s+(.*?);", re.IGNORECASE | re.DOTALL)
        parameters_pattern = re.compile(r"parameters\s+(.*?);", re.IGNORECASE | re.DOTALL)
        model_pattern = re.compile(r"model\s*;\s*(.*?)\s*end\s*;", re.IGNORECASE | re.DOTALL)

        # Extract and clean content
        var_match = var_pattern.search(content)
        if var_match:
            # Remove trailing commas just in case
            self.var_names = [v.strip().rstrip(',') for v in var_match.group(1).split() if v.strip()]
        else:
            print("Warning: 'var' block not found.")

        varexo_match = varexo_pattern.search(content)
        if varexo_match:
            self.shock_names = [s.strip().rstrip(',') for s in varexo_match.group(1).split() if s.strip()]
        else:
            print("Warning: 'varexo' block not found.")

        parameters_match = parameters_pattern.search(content)
        if parameters_match:
            self.param_names = [p.strip().rstrip(',') for p in parameters_match.group(1).split() if p.strip()]
        else:
            print("Warning: 'parameters' block not found.")

        model_match = model_pattern.search(content)
        if model_match:
            eq_block = model_match.group(1)
            # Split equations by semicolon, clean whitespace
            self.equations_str = [eq.strip() for eq in eq_block.split(';') if eq.strip()]
        else:
            raise ValueError("Model block ('model;...end;') not found or parsed correctly.")

        print(f"Found Variables: {self.var_names}")
        print(f"Found Parameters: {self.param_names}")
        print(f"Found Shocks: {self.shock_names}")
        print(f"Found {len(self.equations_str)} Equations (raw).")

    def _create_initial_sympy_symbols(self):
        """Creates symbols for variables, parameters, and shocks."""
        all_names = self.var_names + self.param_names + self.shock_names
        for name in all_names:
            # Check if name is valid Python identifier (optional but good practice)
            if name and name.isidentifier() and name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
            elif name in self.symbols:
                 pass # Symbol already exists
            else:
                 print(f"Warning: Skipping invalid or empty name found in declarations: '{name}'")
        print(f"Created {len(self.symbols)} initial symbols.")

    def _replace_dynare_timing(self, eqstr):
        """Replaces VAR(+k) with VAR_pk and VAR(-k) with VAR_mk symbols."""
        # Pattern to find VAR(+-k)
        # Ensures VAR starts with letter/underscore, followed by alphanumeric/_
        # Allows optional spaces around operator and number
        pat = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+\-])\s*(\d+)\s*\)')
        output_str = eqstr
        replacements = [] # Store replacements as (start_index, end_index, new_string)
        needed_symbols = set() # Track symbols needed for leads/lags

        for match in pat.finditer(eqstr):
            start_index, end_index = match.span()
            var_name, sign, num_str = match.groups()

            # Only replace if var_name is a declared variable
            if var_name in self.var_names_set:
                num = int(num_str)
                if num == 0: # Ignore VAR(0)
                    continue
                # Construct new name: VAR_pk or VAR_mk
                new_name = f"{var_name}_{'p' if sign == '+' else 'm'}{num}"
                replacements.append((start_index, end_index, new_name))
                needed_symbols.add(new_name)

        # Perform replacements from right-to-left to avoid index shifts
        for start, end, replacement_name in sorted(replacements, key=lambda x: x[0], reverse=True):
            output_str = output_str[:start] + replacement_name + output_str[end:]

        # Ensure all needed lead/lag symbols exist in the central dictionary
        for sym_name in needed_symbols:
            if sym_name not in self.symbols:
                self.symbols[sym_name] = sympy.Symbol(sym_name)

        return output_str

    def _parse_equations_to_sympy(self):
        """Parses cleaned equation strings into Sympy Eq objects (LHS-RHS=0)."""
        self.sympy_equations_original = []
        for i, eq_str in enumerate(self.equations_str):
            if not eq_str:
                continue # Skip empty lines
            try:
                # Replace VAR(+k) etc. with VAR_pk etc.
                processed_eq_str = self._replace_dynare_timing(eq_str)

                # Split equation by '=' if present
                if '=' in processed_eq_str:
                    lhs_str, rhs_str = processed_eq_str.split('=', 1)
                    # Parse LHS and RHS into SymPy expressions without evaluating
                    lhs_expr = sympy.parse_expr(lhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    rhs_expr = sympy.parse_expr(rhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    # Create equation LHS - RHS = 0
                    self.sympy_equations_original.append(sympy.Eq(lhs_expr - rhs_expr, 0))
                else:
                    # If no '=', assume the expression should equal zero
                    expr = sympy.parse_expr(processed_eq_str, local_dict=self.symbols, evaluate=False)
                    self.sympy_equations_original.append(sympy.Eq(expr, 0))

            except Exception as e:
                print(f"ERROR parsing equation {i+1}: '{eq_str}'")
                print(f"  Processed string: '{processed_eq_str}'")
                print(f"  Sympy error: {type(e).__name__}: {e}")
                # Optionally raise the error to stop processing
                raise ValueError(f"Failed to parse equation {i+1}") from e

    # ===========================================
    # Stage 1: Variable Timing Analysis (Readable)
    # ===========================================
    def _analyze_variable_timing(self): # Removed file_path argument
        """Analyzes lead/lag structure for each variable across all equations."""
        print("\n--- Stage 1: Analyzing Variable Timing ---")
        # Use existing symbols dictionary
        variable_symbols = {self.symbols[v] for v in self.var_names if v in self.symbols}
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})

        # Max lead/lag to check (can be adjusted if models have longer horizons)
        max_k_check = 15

        for eq in self.sympy_equations_original:
            # Get all free symbols appearing in the equation's LHS (since eq is LHS-RHS=0)
            try:
                free_symbols_in_eq = eq.lhs.free_symbols
            except AttributeError:
                print(f"Warning: Skipping equation analysis for non-expression object: {eq}")
                continue # Skip if equation object is invalid

            for var_sym in variable_symbols:
                var_name = var_sym.name
                # Check if the base variable (time t) appears
                if var_sym in free_symbols_in_eq:
                    self.var_timing_info[var_name]['appears_current'] = True

                # Check for leads (_p) and lags (_m)
                for k in range(1, max_k_check + 1):
                    lead_k_name = f"{var_name}_p{k}"
                    lag_k_name = f"{var_name}_m{k}"

                    # Check lead
                    if lead_k_name in self.symbols and self.symbols[lead_k_name] in free_symbols_in_eq:
                        self.var_timing_info[var_name]['max_lead'] = max(self.var_timing_info[var_name]['max_lead'], k)

                    # Check lag
                    if lag_k_name in self.symbols and self.symbols[lag_k_name] in free_symbols_in_eq:
                        self.var_timing_info[var_name]['min_lag'] = min(self.var_timing_info[var_name]['min_lag'], -k)

        # Print summary after analyzing all equations
        print("Variable Timing Analysis Results:")
        for var_name in sorted(self.var_names): # Print in declared order
             if var_name in self.var_timing_info:
                 info = self.var_timing_info[var_name]
                 print(f"- {var_name}: Current={info['appears_current']}, MaxLead={info['max_lead']}, MinLag={info['min_lag']}")
             else:
                 print(f"- {var_name}: Not found in equations.") # Should not happen if declared

    # ===========================================
    # Stage 2: Identify and Eliminate Static Vars (Readable - User Criteria)
    # ===========================================
    def _identify_and_eliminate_static_vars(self): # Removed file_path argument
        """
        Identifies static vars using refined user criteria and eliminates defining equations.
        Uses standard Python formatting.
        """
        print("\n--- Stage 2: Identifying and Eliminating Static Variables ---")
        self.static_subs = {} # Reset results
        self.equations_after_static_elim = []
        output_lines = ["Static Variable Identification and Elimination:"] # For potential logging

        var_syms = {self.symbols[v] for v in self.var_names if v in self.symbols}
        dynamic_vars_syms = set()

        # 1. Equation-by-Equation Dynamic Check
        output_lines.append("\nChecking for dynamic variables (contemporaneous + lead/lag in same eq):")
        for i, eq in enumerate(self.sympy_equations_original):
            try: eq_lhs = eq.lhs; eq_atoms = eq_lhs.free_symbols
            except AttributeError: continue # Skip invalid equations
            current_vars_in_eq = eq_atoms.intersection(var_syms)

            for var_sym in current_vars_in_eq:
                if var_sym in dynamic_vars_syms: continue
                appears_contemporaneous = var_sym in eq_atoms
                appears_with_lead_lag = False
                for k in range(1, 15):
                    pk_sym = self.symbols.get(f"{var_sym.name}_p{k}")
                    mk_sym = self.symbols.get(f"{var_sym.name}_m{k}")
                    if (pk_sym and pk_sym in eq_atoms) or (mk_sym and mk_sym in eq_atoms):
                        appears_with_lead_lag = True; break
                if appears_contemporaneous and appears_with_lead_lag:
                    if var_sym not in dynamic_vars_syms:
                         line = f"- Variable '{var_sym.name}' identified as dynamic from equation {i+1}."
                         print(line); output_lines.append(line); dynamic_vars_syms.add(var_sym)

        line = f"\nTotal dynamic variables identified: {len(dynamic_vars_syms)} {[v.name for v in sorted(dynamic_vars_syms, key=lambda s:s.name)]}"
        print(line); output_lines.append(line)

        # 2. Identify Candidate Static Variables
        candidate_static_vars = var_syms - dynamic_vars_syms
        line = f"Candidate static variables: {[v.name for v in sorted(candidate_static_vars, key=lambda s:s.name)]}"
        print(line); output_lines.append(line)

        if not candidate_static_vars:
             print("No candidate static variables found."); output_lines.append("No candidate static variables found.")
             self.equations_after_static_elim = list(self.sympy_equations_original)
             # No need to save intermediate file here, handled by process_model
             return

        # 3. Iteratively Find Defining Equations and Solve
        remaining_equations = list(self.sympy_equations_original)
        solved_statics_syms = set(); made_change = True; iteration = 0; max_iterations = len(candidate_static_vars) + 2
        while made_change and iteration < max_iterations:
            iteration += 1; made_change = False; next_remaining_equations = []
            solved_this_round = set(); potential_defs_this_round = collections.defaultdict(list)
            output_lines.append(f"\n--- Solving Iteration {iteration} ---")

            # First pass: Identify potential definitions
            for eq in remaining_equations:
                try: eq_lhs = eq.lhs; eq_atoms = eq_lhs.free_symbols
                except AttributeError: next_remaining_equations.append(eq); continue # Keep invalid eq? Or raise? Keep for now.
                found_potential_def_for_unsolved = False; candidate_statics_in_eq = eq_atoms.intersection(candidate_static_vars)
                for static_cand in candidate_statics_in_eq:
                    appears_only_contemporaneously = True
                    if static_cand not in eq_atoms: appears_only_contemporaneously = False
                    else:
                        for k in range(1, 15):
                            pk_sym = self.symbols.get(f"{static_cand.name}_p{k}"); mk_sym = self.symbols.get(f"{static_cand.name}_m{k}")
                            if (pk_sym and pk_sym in eq_atoms) or (mk_sym and mk_sym in eq_atoms): appears_only_contemporaneously = False; break
                    if appears_only_contemporaneously:
                        potential_defs_this_round[static_cand].append(eq)
                        if static_cand not in solved_statics_syms: found_potential_def_for_unsolved = True
                if not found_potential_def_for_unsolved: next_remaining_equations.append(eq)

            # Second pass: Attempt to solve
            # output_lines.append("Attempting to solve potential definitions:") # Optional verbose log
            eqs_used_this_round = set()
            for static_cand, eq_list in potential_defs_this_round.items():
                if static_cand in solved_statics_syms: continue
                if len(eq_list) == 1:
                    defining_eq = eq_list[0]; eq_lhs = defining_eq.lhs
                    if defining_eq in eqs_used_this_round:
                         if defining_eq not in next_remaining_equations: next_remaining_equations.append(defining_eq)
                         continue
                    try:
                        solution_list = sympy.solve(eq_lhs, static_cand)
                        if isinstance(solution_list, (list, tuple)) and len(solution_list) == 1:
                            current_subs = {s: expr for s, expr in self.static_subs.items()}; solution = solution_list[0].subs(current_subs)
                            if static_cand in solution.free_symbols:
                                line = f"- Warn: Solved {static_cand.name} depends on self! Eq: {sympy.sstr(eq_lhs)}=0"; print(line); output_lines.append(line)
                                if defining_eq not in next_remaining_equations: next_remaining_equations.append(defining_eq)
                            else:
                                line = f"- Solved: {static_cand.name} = {solution} (from eq: {sympy.sstr(eq_lhs)} = 0)"; print(line); output_lines.append(line)
                                self.static_subs[static_cand] = solution; solved_this_round.add(static_cand); made_change = True; eqs_used_this_round.add(defining_eq)
                        else:
                            # line = f"- Info: No unique solve {static_cand.name} from {sympy.sstr(eq_lhs)}=0"; output_lines.append(line); # Optional verbose
                            if defining_eq not in next_remaining_equations: next_remaining_equations.append(defining_eq)
                    except Exception as e:
                        line = f"- Warn: Solve fail {static_cand.name}, Eq: {sympy.sstr(eq_lhs)}=0. Err: {e}"; print(line); output_lines.append(line);
                        if defining_eq not in next_remaining_equations: next_remaining_equations.append(defining_eq)
                elif len(eq_list) > 1:
                    # line = f"- Info: Multiple potential defs for {static_cand.name}. Keeping eqs."; output_lines.append(line); # Optional verbose
                    for eq_pot in eq_list:
                        if eq_pot not in next_remaining_equations: next_remaining_equations.append(eq_pot)

            solved_statics_syms.update(solved_this_round)
            remaining_equations = next_remaining_equations
            if made_change: output_lines.append(f"Solved in iteration {iteration}: {[s.name for s in solved_this_round]}")
            elif iteration == 1 and not solved_statics_syms: output_lines.append("\nNo static vars solved in iteration 1.")

        self.equations_after_static_elim = remaining_equations
        line = f"\nStatic vars solved: {[v.name for v in sorted(solved_statics_syms, key=lambda s:s.name)]}"; print(line); output_lines.append(line)
        line = f"Remaining equations after static elim: {len(self.equations_after_static_elim)}"; print(line); output_lines.append(line)
        # Saving handled by process_model

    # ===========================================
    # Stage 3: Substitute Static Variables (Readable)
    # ===========================================
    def _substitute_static_vars(self): # Removed file_path argument
        """
        Substitutes solved static vars into remaining equations. Uses standard Python.
        """
        print("\n--- Stage 3: Substituting Static Variables ---")
        output_lines = ["Substituting static variables:"] # For potential logging
        if not self.static_subs:
            print("No static substitutions to perform.")
            self.equations_after_static_sub = list(self.equations_after_static_elim)
            # Saving handled by process_model
            return

        # Build the full substitution dictionary
        full_subs_dict = {}
        max_lead_needed = 0; min_lag_needed = 0
        static_syms_to_sub = set(self.static_subs.keys())
        for eq in self.equations_after_static_elim:
             for atom in eq.lhs.free_symbols:
                 base_name = None; k = 0; is_static_related = False
                 match_lead = re.match(r"(\w+)_p(\d+)", atom.name); match_lag = re.match(r"(\w+)_m(\d+)", atom.name)
                 if match_lead: base_name=match_lead.group(1); k=int(match_lead.group(2)); is_static = self.symbols.get(base_name) in static_syms_to_sub; if is_static: max_lead_needed = max(max_lead_needed, k)
                 elif match_lag: base_name=match_lag.group(1); k=int(match_lag.group(2)); is_static = self.symbols.get(base_name) in static_syms_to_sub; if is_static: min_lag_needed = min(min_lag_needed, -k)
        # output_lines.append(f"Max lead needed for static subs: {max_lead_needed}") # Optional log
        # output_lines.append(f"Min lag needed for static subs: {min_lag_needed}") # Optional log

        for static_var, solution in self.static_subs.items():
            full_subs_dict[static_var] = solution
            for k in range(1, max_lead_needed + 1):
                lead_key = f"{static_var.name}_p{k}"
                if lead_key in self.symbols:
                    lead_solution = time_shift_expression(solution, k, self.symbols, self.var_names_set)
                    full_subs_dict[self.symbols[lead_key]] = lead_solution
                    # output_lines.append(f"- Sub rule: {lead_key} -> {sympy.sstr(lead_solution)}") # Optional log
            for k in range(1, abs(min_lag_needed) + 1):
                lag_key = f"{static_var.name}_m{k}"
                if lag_key in self.symbols:
                    lag_solution = time_shift_expression(solution, -k, self.symbols, self.var_names_set)
                    full_subs_dict[self.symbols[lag_key]] = lag_solution
                    # output_lines.append(f"- Sub rule: {lag_key} -> {sympy.sstr(lag_solution)}") # Optional log

        # Apply substitutions
        self.equations_after_static_sub = []
        print(f"Applying {len(full_subs_dict)} substitution rules to {len(self.equations_after_static_elim)} equations...")
        # output_lines.append("\nApplying substitutions:") # Optional log
        for i, eq in enumerate(self.equations_after_static_elim):
            substituted_lhs = eq.lhs.xreplace(full_subs_dict)
            try:
                simplified_lhs = sympy.simplify(substituted_lhs)
            except Exception as e:
                print(f"Simplify failed eq{i+1}: {e}")
                simplified_lhs = substituted_lhs
            subbed_eq = sympy.Eq(simplified_lhs, 0)
            self.equations_after_static_sub.append(subbed_eq)
            # line2 = f"  Eq {i+1} substituted: {sympy.sstr(subbed_eq.lhs, full_prec=False)} = 0"; output_lines.append(line2) # Optional log
        line = f"Substitution complete. {len(self.equations_after_static_sub)} equations remain."
        print(line)
        # output_lines.append(line) # Optional log
        # Saving handled by process_model

    # ===========================================
    # Stage 4: Handle Long Leads/Lags (Readable & Corrected)
    # ===========================================
    def _handle_aux_vars(self): # Removed file_path argument
        """
        Handles leads > +1 and lags < -1. Uses standard Python formatting.
        """
        print("\n--- Stage 4: Handling Aux Vars (Readable & Corrected Sub Target) ---")
        self.aux_lead_vars = {}; self.aux_lag_vars = {}; self.aux_var_definitions = []
        current_equations = list(self.equations_after_static_sub)
        subs_long_leads_lags = {};
        dynamic_base_var_names = [v for v in self.var_names if v not in [s.name for s in self.static_subs.keys()]]
        # output_lines = ["Handling long leads/lags using 'aux_VAR_leadK'/'aux_VAR_lagK':"] # Optional log

        # --- Identify and Define Aux Lead Vars ---
        leads_to_replace = collections.defaultdict(int)
        for eq in current_equations:
             for atom in eq.lhs.free_symbols:
                 match = re.match(r"(\w+)_p(\d+)", atom.name)
                 if match: base, k = match.groups(); k=int(k); if k > 1 and base in dynamic_base_var_names: leads_to_replace[base] = max(leads_to_replace[base], k)
        if leads_to_replace:
             # output_lines.append("\nCreating auxiliary LEAD variables:") # Optional log
             print("Creating auxiliary LEAD variables...")
             for var_name in sorted(leads_to_replace.keys()):
                 max_lead = leads_to_replace[var_name]
                 for k in range(1, max_lead):
                     aux_name = f"aux_{var_name}_lead{k}"
                     if aux_name not in self.symbols: self.symbols[aux_name] = sympy.Symbol(aux_name)
                     self.aux_lead_vars[aux_name] = self.symbols[aux_name]
                     if k == 1: var_p1_name = f"{var_name}_p1"; if var_p1_name not in self.symbols: self.symbols[var_p1_name] = sympy.Symbol(var_p1_name); rhs_sym = self.symbols[var_p1_name]
                     else: prev_aux_name = f"aux_{var_name}_lead{k-1}"; prev_aux_p1_name = f"{prev_aux_name}_p1"; if prev_aux_p1_name not in self.symbols: self.symbols[prev_aux_p1_name] = sympy.Symbol(prev_aux_p1_name); rhs_sym = self.symbols[prev_aux_p1_name]
                     new_eq = sympy.Eq(self.symbols[aux_name] - rhs_sym, 0)
                     if new_eq not in self.aux_var_definitions: self.aux_var_definitions.append(new_eq); # def_line = f"- Added aux var '{aux_name}'. Def: {new_eq.lhs}=0"; print(def_line); # Optional log
                     orig_lead_key = f"{var_name}_p{k+1}"; aux_lead_k_p1_name = f"{aux_name}_p1";
                     if orig_lead_key in self.symbols:
                         if aux_lead_k_p1_name not in self.symbols: self.symbols[aux_lead_k_p1_name] = sympy.Symbol(aux_lead_k_p1_name)
                         subs_long_leads_lags[self.symbols[orig_lead_key]] = self.symbols[aux_lead_k_p1_name]; # sub_line = f"  - Sub rule created: {orig_lead_key} -> {aux_lead_k_p1_name}"; print(sub_line) # Optional log

        # --- Identify and Define Aux Lag Vars ---
        lags_to_replace = collections.defaultdict(int)
        for eq in current_equations:
             for atom in eq.lhs.free_symbols:
                 match = re.match(r"(\w+)_m(\d+)", atom.name);
                 if match: base, k = match.groups(); k=int(k); if k > 1 and base in dynamic_base_var_names: lags_to_replace[base] = max(lags_to_replace[base], k)
        if lags_to_replace:
             # output_lines.append("\nCreating auxiliary LAG variables:") # Optional log
             print("Creating auxiliary LAG variables...")
             for var_name in sorted(lags_to_replace.keys()):
                  max_lag = lags_to_replace[var_name]
                  for k in range(1, max_lag):
                      aux_name = f"aux_{var_name}_lag{k}"
                      if aux_name not in self.symbols: self.symbols[aux_name] = sympy.Symbol(aux_name)
                      self.aux_lag_vars[aux_name] = self.symbols[aux_name]
                      if k == 1: var_m1_name = f"{var_name}_m1"; if var_m1_name not in self.symbols: self.symbols[var_m1_name] = sympy.Symbol(var_m1_name); rhs_sym = self.symbols[var_m1_name]
                      else: prev_aux_name = f"aux_{var_name}_lag{k-1}"; prev_aux_m1_name = f"{prev_aux_name}_m1"; if prev_aux_m1_name not in self.symbols: self.symbols[prev_aux_m1_name] = sympy.Symbol(prev_aux_m1_name); rhs_sym = self.symbols[prev_aux_m1_name]
                      new_eq = sympy.Eq(self.symbols[aux_name] - rhs_sym, 0)
                      if new_eq not in self.aux_var_definitions: self.aux_var_definitions.append(new_eq); # def_line = f"- Added aux var '{aux_name}'. Def: {new_eq.lhs}=0"; print(def_line); # Optional log
                      orig_lag_key = f"{var_name}_m{k+1}"; aux_lag_m1_key = f"{aux_name}_m1";
                      if orig_lag_key in self.symbols:
                          if aux_lag_m1_key not in self.symbols: self.symbols[aux_lag_m1_key] = sympy.Symbol(aux_lag_m1_key)
                          subs_long_leads_lags[self.symbols[orig_lag_key]] = self.symbols[aux_lag_m1_key]; # sub_line = f"  - Sub rule created: {orig_lag_key} -> {aux_lag_m1_key}"; print(sub_line) # Optional log

        # --- Apply Substitutions ---
        self.equations_after_aux_sub = []
        print(f"\nApplying {len(subs_long_leads_lags)} long lead/lag substitutions...")
        for i, eq in enumerate(current_equations):
             subbed_lhs = eq.lhs.xreplace(subs_long_leads_lags)
             try: simplified_lhs = sympy.simplify(subbed_lhs)
             except Exception: simplified_lhs = subbed_lhs
             subbed_eq = sympy.Eq(simplified_lhs, 0); self.equations_after_aux_sub.append(subbed_eq)
             # line2 = f"  Eq {i+1} substituted: {sympy.sstr(subbed_eq.lhs, full_prec=False)} = 0"; output_lines.append(line2) # Optional log

        # Finalize dynamic var list
        self.final_dynamic_var_names = dynamic_base_var_names + list(self.aux_lead_vars.keys()) + list(self.aux_lag_vars.keys())
        print(f"Aux handling complete. Final dynamic vars: {len(self.final_dynamic_var_names)}")
        # Saving handled by process_model

    # ===========================================
    # Stage 5: Define State Vector (Readable)
    # ===========================================
    def _define_state_vector(self): # Removed file_path argument
        """Defines the ordered state vector. Uses standard Python."""
        print("\n--- Stage 5: Defining State Vector ---")
        state_candidate_names = list(self.final_dynamic_var_names)
        if not state_candidate_names: raise ValueError("Cannot define state: final_dynamic_var_names empty.")

        # Simple ordering: Base Vars (alpha), Aux Lag (alpha), Aux Lead (alpha)
        base_dynamic_vars = sorted([s for s in state_candidate_names if not s.startswith("aux_")])
        aux_lag_vars_list = sorted([s for s in state_candidate_names if s.startswith("aux_") and "_lag" in s])
        aux_lead_vars_list = sorted([s for s in state_candidate_names if s.startswith("aux_") and "_lead" in s])

        # Ensure all names map to existing symbols before creating list
        ordered_names = base_dynamic_vars + aux_lag_vars_list + aux_lead_vars_list
        self.state_vars_ordered = []
        for name in ordered_names:
             if name in self.symbols:
                  self.state_vars_ordered.append(self.symbols[name])
             else:
                  # This indicates an internal inconsistency
                  print(f"Error: State variable name '{name}' not found in symbols dictionary during state vector creation.")
                  raise KeyError(f"Symbol for state variable '{name}' missing.")

        self.state_var_map = {sym: i for i, sym in enumerate(self.state_vars_ordered)}
        line = f"Final State Vector Defined (size {len(self.state_vars_ordered)}):"
        print(line); state_names_list = [s.name for s in self.state_vars_ordered]; print(f"  Order: {state_names_list}")
        # Saving handled by process_model

    # ===========================================
    # Stage 6: Build Final Equations & Check (Readable)
    # ===========================================
    def _build_final_equations(self): # Removed file_path argument
        """Combines equations and checks count. Uses standard Python."""
        print("\n--- Stage 6: Building Final Equation System ---")
        # Combine substituted dynamic equations + all aux definitions
        self.final_equations_for_jacobian = list(self.equations_after_aux_sub) + list(self.aux_var_definitions)

        # Check count matches state vector size
        n_state = len(self.state_vars_ordered)
        n_eqs = len(self.final_equations_for_jacobian)
        line1 = f"\nChecking counts: N_States = {n_state}, N_Equations = {n_eqs}"
        print(line1)

        if n_state == 0: raise ValueError("State vector has zero size.")
        if n_eqs == 0: raise ValueError("Final equation list is empty.")
        if n_state != n_eqs:
            line2 = f"ERROR: State count ({n_state}) != equation count ({n_eqs})!"
            print("!"*60 + "\n" + line2); print("State Vars:", [s.name for s in self.state_vars_ordered]); print("Equations:")
            max_eq_print = 20
            for i, eq in enumerate(self.final_equations_for_jacobian[:max_eq_print]): print(f"  {i+1}: {sympy.sstr(eq.lhs)} = 0")
            if n_eqs > max_eq_print: print(f"  ... ({n_eqs - max_eq_print} more)")
            print("Preprocessing failed." + "\n" + "!"*60)
            raise ValueError("Equation count mismatch after final assembly.")
        else:
             line2 = "State and equation counts match."
             print(line2)
        # Saving handled by process_model

    # ===========================================
    # get_numerical_ABCD (Readable & Correct)
    # ===========================================
    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        # (Keep the last corrected, readable version of this method)
        # ... (Readable implementation from Response #16) ...
        """ Calculates numerical A, B, C, D from final equations and states."""
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing incomplete: Final equations or ordered state vector not available.")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)
        self.last_param_values = param_dict_values
        param_subs = {self.symbols[p]: v for p, v in param_dict_values.items() if p in self.symbols}

        # --- Create symbolic vectors (Readable Version) ---
        state_vec_t = sympy.Matrix(self.state_vars_ordered)
        state_vec_tp1_list = []
        state_vec_tm1_list = []

        for state_sym in self.state_vars_ordered:
            lead_name = f"{state_sym.name}_p1"
            lag_name = f"{state_sym.name}_m1"
            if lead_name not in self.symbols: self.symbols[lead_name] = sympy.Symbol(lead_name)
            state_vec_tp1_list.append(self.symbols[lead_name])
            if lag_name not in self.symbols: self.symbols[lag_name] = sympy.Symbol(lag_name)
            state_vec_tm1_list.append(self.symbols[lag_name])

        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)
        shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
        shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None
        eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])
        # --- End Create symbolic vectors ---

        print("Calculating Jacobians...")
        try:
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            if shock_vec and n_shocks > 0 : D_sym = -eq_vec.jacobian(shock_vec)
            else: D_sym = sympy.zeros(n_state, n_shocks)
        except Exception as e: print(f"Error during symbolic Jacobian calculation: {e}"); raise

        print("Substituting parameter values...")
        try:
            A_num = np.array(A_sym.evalf(subs=param_subs).tolist(), dtype=float)
            B_num = np.array(B_sym.evalf(subs=param_subs).tolist(), dtype=float)
            C_num = np.array(C_sym.evalf(subs=param_subs).tolist(), dtype=float)
            if n_shocks > 0: D_num = np.array(D_sym.evalf(subs=param_subs).tolist(), dtype=float)
            else: D_num = np.zeros((n_state, 0), dtype=float)
        except Exception as e: print(f"Error during numerical substitution (evalf): {e}"); raise

        expected_d_shape = (n_state, n_shocks)
        if A_num.shape!=(n_state,n_state) or B_num.shape!=(n_state,n_state) or C_num.shape!=(n_state,n_state) or D_num.shape!=expected_d_shape:
            print(f"ERROR: Final matrix dimension mismatch!")
            raise ValueError("Matrix dimension mismatch.")
        print("Numerical matrices A, B, C, D calculated.")

        if file_path: self._save_final_matrices(file_path, A_num, B_num, C_num, D_num, [s.name for s in self.state_vars_ordered], self.shock_names, self.param_names)
        return A_num, B_num, C_num, D_num, [s.name for s in self.state_vars_ordered], self.shock_names


    # --- Helper Function (Internal Use Only) ---
    def _generate_matrix_assignments_code_helper(self, matrix_sym, matrix_name):
        # (Keep the latest correct & readable version of this helper)
        # ... (Implementation from Response #16) ...
        """Generates element assignment lines, getting dims from matrix_sym."""
        try: rows, cols = matrix_sym.shape
        except Exception as e: raise ValueError(f"Shape error for {matrix_name}") from e
        indent = "    "; code_lines = [f"{indent}{matrix_name} = np.zeros(({rows}, {cols}), dtype=float)"]; assignments = []
        for r in range(rows):
            for c in range(cols):
                try: element = matrix_sym[r, c]
                except IndexError: continue
                if element != 0 and element is not sympy.S.Zero:
                    try: expr_str = sympy.sstr(element, full_prec=False); assignments.append(f"{indent}{matrix_name}[{r}, {c}] = {expr_str}")
                    except Exception as str_e: print(f"Warn: Str convert fail {matrix_name}[{r},{c}]: {str_e}"); assignments.append(f"{indent}# Warn: Fail {matrix_name}[{r},{c}] = {element}")
        if assignments: code_lines.append(f"{indent}# Fill {matrix_name} non-zero elements"); code_lines.extend(assignments)
        return "\n".join(code_lines)

    # ===========================================
    # generate_matrix_function_file (Readable & No Internal Docstring/Example)
    # ===========================================
    def generate_matrix_function_file(self, filename="jacobian_matrices.py"):
        # (Keep the latest correct & readable version of this method)
        # ... (Implementation from Response #16, calling internal helper) ...
        """ Generates jacobian_matrices(theta) -> A, B, C, D with clean code. """
        function_name = "jacobian_matrices"; print(f"\n--- Generating Python Function File (Clean): {filename} ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered: raise ValueError("Preprocessing must be run successfully first.")
        n_state = len(self.state_vars_ordered); n_shocks = len(self.shock_names)
        try: # Recalculate Jacobians
            state_vec_t = sympy.Matrix(self.state_vars_ordered); state_vec_tp1_list = []; state_vec_tm1_list = []
            for s in self.state_vars_ordered: ln=f"{s.name}_p1"; mn=f"{s.name}_m1";
            if ln not in self.symbols: self.symbols[ln]=sympy.Symbol(ln); state_vec_tp1_list.append(self.symbols[ln])
            else: state_vec_tp1_list.append(self.symbols[ln])
            if mn not in self.symbols: self.symbols[mn]=sympy.Symbol(mn); state_vec_tm1_list.append(self.symbols[mn])
            else: state_vec_tm1_list.append(self.symbols[mn])
            state_vec_tp1=sympy.Matrix(state_vec_tp1_list); state_vec_tm1=sympy.Matrix(state_vec_tm1_list)
            shock_syms_list=[self.symbols[s] for s in self.shock_names if s in self.symbols]; shock_vec=sympy.Matrix(shock_syms_list) if shock_syms_list else None
            eq_vec=sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])
            print("Calculating symbolic Jacobians..."); A_sym=eq_vec.jacobian(state_vec_tp1); B_sym=eq_vec.jacobian(state_vec_t); C_sym=eq_vec.jacobian(state_vec_tm1)
            if shock_vec and n_shocks > 0 : D_sym = -eq_vec.jacobian(shock_vec)
            else: D_sym = sympy.zeros(n_state, n_shocks)
            print("Jacobians calculated.")
        except Exception as e: print(f"Error calculating Jacobians: {e}"); raise
        ordered_params_from_mod = self.param_names; param_symbols_in_matrices = set().union(*(mat.free_symbols for mat in [A_sym, B_sym, C_sym, D_sym]))
        used_params_ordered = [p for p in ordered_params_from_mod if p in [s.name for s in param_symbols_in_matrices]]; param_indices = {p: i for i, p in enumerate(ordered_params_from_mod)}
        print("Generating Python code strings for matrices...");
        code_A = self._generate_matrix_assignments_code_helper(A_sym, 'A'); code_B = self._generate_matrix_assignments_code_helper(B_sym, 'B');
        code_C = self._generate_matrix_assignments_code_helper(C_sym, 'C'); code_D = self._generate_matrix_assignments_code_helper(D_sym, 'D')
        print("Code generation complete.")
        file_lines = []; file_lines.append(f"# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'"); file_lines.append(f"# Generated: {datetime.datetime.now().isoformat()}"); file_lines.append("import numpy as np"); file_lines.append("from math import *"); file_lines.append(""); file_lines.append(f"def {function_name}(theta):")
        file_lines.append("    # Unpack parameters (using original order)"); file_lines.append(f"    expected_len = {len(ordered_params_from_mod)}")
        file_lines.append(f"    if len(theta) != expected_len: raise ValueError(f'Expected {{expected_len}} parameters, got {{len(theta)}}')")
        file_lines.append("    try:")
        for p_name in used_params_ordered: idx = param_indices[p_name]; file_lines.append(f"        {p_name} = theta[{idx}]")
        file_lines.append("    except IndexError: raise IndexError(f'Parameter vector theta has incorrect length.')")
        file_lines.append(""); file_lines.append("    # Initialize and fill matrices")
        file_lines.append(code_A); file_lines.append(""); file_lines.append(code_B); file_lines.append("")
        file_lines.append(code_C); file_lines.append(""); file_lines.append(code_D); file_lines.append("")
        file_lines.append("    # --- Return results ---"); file_lines.append(f"    state_names = {repr([s.name for s in self.state_vars_ordered])}"); file_lines.append(f"    shock_names = {repr(self.shock_names)}"); file_lines.append(""); file_lines.append("    return A, B, C, D, state_names, shock_names")
        final_file_content = "\n".join(file_lines)
        try:
            dir_name = os.path.dirname(filename);
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f: f.write(final_file_content)
            print(f"Successfully generated function file: {filename}")
        except Exception as e: print(f"Error writing function file {filename}: {e}")


    # ===========================================
    # process_model (Corrected Internal Calls and Return Value)
    # ===========================================
    def process_model(self, param_dict_values_or_list, output_dir_intermediate=None,
                      output_dir_final=None, generate_function=True):
        """
        Runs the full parsing and matrix generation pipeline. Correctly returns results.
        Uses standard Python formatting. Internal calls don't handle file paths directly.

        Args:
            param_dict_values_or_list: Dict or ordered list/array of parameter values.
            output_dir_intermediate: Directory to save intermediate text files.
            output_dir_final: Directory to save final .pkl and .py files.
            generate_function: Boolean flag to generate the python function file.

        Returns:
            tuple: (A, B, C, D, state_names, shock_names) on success, None on failure.
        """
        # --- Parameter Input Handling (Readable) ---
        param_dict_values = {}
        if isinstance(param_dict_values_or_list, (list, tuple, np.ndarray)):
             if len(param_dict_values_or_list) != len(self.param_names):
                 raise ValueError(f"Input parameter list/array length ({len(param_dict_values_or_list)}) != declared parameters ({len(self.param_names)})")
             param_dict_values = {name: val for name, val in zip(self.param_names, param_dict_values_or_list)}
             # print("Received parameter list/array, converted to dict.")
        elif isinstance(param_dict_values_or_list, dict):
            param_dict_values = param_dict_values_or_list
            missing_keys = set(self.param_names) - set(param_dict_values.keys())
            if missing_keys: print(f"Warning: Input dict missing keys: {missing_keys}")
        else: raise TypeError("Input must be a dict, list, tuple, or numpy array.")
        # --- End Parameter Input Handling ---

        # --- Define file paths (Readable) ---
        base_name = os.path.splitext(os.path.basename(self.mod_file_path))[0]
        fpaths_inter = {i: None for i in range(7)} # Increased size for extra save
        final_matrices_pkl = None; function_py = None
        if output_dir_intermediate:
            os.makedirs(output_dir_intermediate, exist_ok=True)
            inter_names = ["0_original_eqs", "1_timing", "2_static_elim", "3_static_sub", "4_aux_handling", "5_state_def", "6_final_eqs"]
            fpaths_inter = {i: os.path.join(output_dir_intermediate, f"{i}_{base_name}_{name}.txt") for i, name in enumerate(inter_names)}
        if output_dir_final:
             os.makedirs(output_dir_final, exist_ok=True)
             final_matrices_pkl = os.path.join(output_dir_final, f"{base_name}_matrices.pkl")
             if generate_function: function_py = os.path.join(output_dir_final, f"{base_name}_jacobian_matrices.py")
        # --- End Define file paths ---

        # --- Run pipeline steps ---
        try:
            print("\n--- Starting Model Processing Pipeline ---")
            # Save original parsed equations
            self._save_intermediate_file(fpaths_inter[0], ["Stage 0 Parsed Equations"], self.sympy_equations_original, "Original Sympy Equations")

            # Call internal methods WITHOUT file_path argument
            self._analyze_variable_timing()
            timing_lines = ["Stage 1 Timing Analysis Complete"]
            for var_name in sorted(self.var_names):
                if var_name in self.var_timing_info: info = self.var_timing_info[var_name]; timing_lines.append(f"- {var_name}: Curr={info['appears_current']}, Lead={info['max_lead']}, Lag={info['min_lag']}")
            self._save_intermediate_file(fpaths_inter[1], timing_lines)

            self._identify_and_eliminate_static_vars()
            static_lines = ["Stage 2 Static Elimination Complete", f"Solved: {[s.name for s in self.static_subs.keys()]}"]
            self._save_intermediate_file(fpaths_inter[2], static_lines, self.equations_after_static_elim, "Equations After Static Elimination")

            self._substitute_static_vars()
            self._save_intermediate_file(fpaths_inter[3], ["Stage 3 Static Substitution Complete"], self.equations_after_static_sub, "Equations After Static Substitution")

            self._handle_aux_vars()
            aux_lines = ["Stage 4 Aux Handling Complete", f"Final dynamic vars (incl aux): {self.final_dynamic_var_names}"]
            self._save_intermediate_file(fpaths_inter[4], aux_lines, self.equations_after_aux_sub + self.aux_var_definitions, "Equations After Aux Handling (Substituted + Definitions)")

            self._define_state_vector()
            state_lines = ["Stage 5 State Definition Complete", f"State Vector ({len(self.state_vars_ordered)}): {[s.name for s in self.state_vars_ordered]}"]
            self._save_intermediate_file(fpaths_inter[5], state_lines)

            self._build_final_equations()
            build_lines = ["Stage 6 Final Equation Build Complete", f"N_States={len(self.state_vars_ordered)}, N_Eqs={len(self.final_equations_for_jacobian)}"]
            self._save_intermediate_file(fpaths_inter[6], build_lines, self.final_equations_for_jacobian, "Final Equation System for Jacobian")

            # Get numerical matrices using the processed data
            A, B, C, D, state_names, shock_names = self.get_numerical_ABCD(
                param_dict_values, # Pass the DICT here
                file_path=final_matrices_pkl # Save numerical matrices
            )

            # Generate the function file if requested
            if generate_function and function_py:
                self.generate_matrix_function_file(filename=function_py)

            print("\n--- Model Processing Successful ---")
            # --- CORRECT RETURN VALUE ---
            return A, B, C, D, state_names, shock_names
            # --- END CORRECTION ---

        except Exception as e:
            print(f"\n--- ERROR during model processing: {type(e).__name__}: {e} ---")
            import traceback
            traceback.print_exc()
            return None # Indicate failure
        # --- End Run pipeline steps ---

    # --- Helper methods ---
    # Add _save_intermediate_file, _save_final_matrices, save_final_equations_to_txt here
    # Ensure they are defined within the class or imported correctly
    # (Using implementations from previous answers)
    def _save_intermediate_file(self, file_path, lines, equations=None, equations_title="Equations"):
        # (Implementation from previous answer)
        if not file_path: return
        try:
            dir_name = os.path.dirname(file_path);
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(f"--- {os.path.basename(file_path)} ---\n"); f.write(f"--- Generated: {datetime.datetime.now().isoformat()} ---\n\n"); f.write("\n".join(lines))
                if equations is not None:
                     f.write(f"\n\n{equations_title} ({len(equations)}):\n")
                     if equations:
                          for i, eq in enumerate(equations):
                              if eq is not None and hasattr(eq, 'lhs'): f.write(f"  {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                              elif eq is None: f.write(f"  {i+1}: [None Equation Placeholder]\n")
                              else: f.write(f"  {i+1}: [Non-Equation Object: {type(eq)}] {str(eq)}\n")
                     else: f.write("  [No equations in this list]\n")
            # print(f"Intermediate results saved to {file_path}") # Optional verbose
        except Exception as e: print(f"Warning: Could not save intermediate file {file_path}. Error: {e}")

    def _save_final_matrices(self, file_path, A, B, C, D, state_names, shock_names, param_names):
        # (Implementation from previous answer)
        if not file_path: return
        matrix_data = {'A': A, 'B': B, 'C': C, 'D': D,'state_names': state_names,'shock_names': shock_names,'param_names': param_names,'timestamp': datetime.datetime.now().isoformat()}
        dir_name = os.path.dirname(file_path);
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        try:
            with open(file_path, "wb") as f: pickle.dump(matrix_data, f); print(f"Matrices saved to {file_path}")
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w", encoding='utf-8') as f:
                 f.write(f"# Numerical Matrices for {os.path.basename(self.mod_file_path)}\n"); f.write(f"# Generated: {matrix_data['timestamp']}\n\n")
                 f.write(f"State Names (Order: {len(state_names)}):\n" + str(state_names) + "\n\n"); f.write(f"Shock Names ({len(shock_names)}):\n" + str(shock_names) + "\n\n"); f.write(f"Parameter Names ({len(param_names)}):\n" + str(param_names) + "\n\n")
                 np.set_printoptions(linewidth=200, precision=6, suppress=True, threshold=np.inf)
                 f.write(f"A Matrix ({A.shape}):\n" + np.array2string(A, separator=', ') + "\n\n"); f.write(f"B Matrix ({B.shape}):\n" + np.array2string(B, separator=', ') + "\n\n")
                 f.write(f"C Matrix ({C.shape}):\n" + np.array2string(C, separator=', ') + "\n\n"); f.write(f"D Matrix ({D.shape}):\n")
                 if D.size > 0: f.write(np.array2string(D, separator=', ') + "\n\n")
                 else: f.write("[No shocks]\n\n")
            print(f"Human-readable matrices saved to {txt_path}")
        except Exception as e: print(f"Warning: Could not save matrices file {file_path} or {txt_path}. Error: {e}")

    def save_final_equations_to_txt(self, filename="final_equations.txt"):
        # (Implementation from previous answer)
        print(f"\n--- Saving Final Equations to: {filename} ---")
        if not hasattr(self, 'final_equations_for_jacobian') or not self.final_equations_for_jacobian: print("Warning: Final equations not generated yet. File not saved."); return
        try:
            dir_name = os.path.dirname(filename);
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"# Final System Equations Used for Jacobian ({len(self.final_equations_for_jacobian)} equations)\n"); f.write(f"# Model: {os.path.basename(self.mod_file_path)}\n"); f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                if hasattr(self, 'state_vars_ordered') and self.state_vars_ordered: f.write(f"# State Variables Order ({len(self.state_vars_ordered)}): {[s.name for s in self.state_vars_ordered]}\n\n")
                else: f.write("# State variable order not determined yet.\n\n")
                for i, eq in enumerate(self.final_equations_for_jacobian):
                    if eq is not None and hasattr(eq, 'lhs'): f.write(f"Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                    elif eq is None: f.write(f"Eq {i+1}: [None Equation Placeholder]\n")
                    else: f.write(f"Eq {i+1}: [Invalid Equation Object] {str(eq)}\n")
            print(f"Successfully saved final equations to {filename}")
        except Exception as e: print(f"Error writing final equations file {filename}: {e}")

# ===========================================
# End of DynareParser Class
# ===========================================

# ===========================================
# Example Usage Script (at the end of the file)
# ===========================================
if __name__ == "__main__":
    # --- Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    os.chdir(script_dir)
    mod_file = "qpm_model.dyn" # Make sure this points to your model file
    output_dir_inter = "model_files_intermediate_final"
    output_dir_final = "model_files_numerical_final"
    os.makedirs(output_dir_inter, exist_ok=True)
    os.makedirs(output_dir_final, exist_ok=True)

    # --- Define parameters DICT ---
    parameter_values_dict = { 'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25, 'rho_DLA_CPI': 0.75, 'rho_L_GDP_GAP': 0.75, 'rho_rs': 0.75, 'rho_rs2': 0.01 }

    # --- Instantiate parser ---
    try: parser = DynareParser(mod_file)
    except FileNotFoundError: print(f"ERROR: Model file '{mod_file}' not found in {script_dir}. Please create it."); sys.exit(1)
    except Exception as e: print(f"Error initializing parser: {e}"); sys.exit(1)

    # --- Create theta list IN ORDER ---
    try: parameter_theta = [parameter_values_dict[pname] for pname in parser.param_names]; print(f"\nTheta created (order: {parser.param_names})")
    except KeyError as e: print(f"\nERROR: Param '{e}' missing from dict."); sys.exit(1)
    except Exception as e: print(f"\nERROR creating theta list: {e}"); sys.exit(1)

    # --- Process the model ---
    result = parser.process_model(parameter_theta, output_dir_intermediate=output_dir_inter, output_dir_final=output_dir_final, generate_function=True)

    # --- Check Results & Test ---
    if result:
        A_direct, B_direct, C_direct, D_direct, state_names_direct, shock_names_direct = result
        print("\n--- Results from Direct Calculation ---"); print("States:", state_names_direct); print(f"A:{A_direct.shape} B:{B_direct.shape} C:{C_direct.shape} D:{D_direct.shape}")
        function_file = os.path.join(output_dir_final, f"{os.path.splitext(mod_file)[0]}_jacobian_matrices.py")
        if os.path.exists(function_file):
            print("\n--- Testing Generated Function ---"); abs_function_file = os.path.abspath(function_file); module_name = os.path.splitext(os.path.basename(function_file))[0]
            try:
                spec = importlib.util.spec_from_file_location(module_name, abs_function_file);
                if spec is None: print(f"Error: Could not load spec for {module_name}")
                else:
                    mod_matrices = importlib.util.module_from_spec(spec); sys.modules[module_name] = mod_matrices; spec.loader.exec_module(mod_matrices)
                    A_f, B_f, C_f, D_f, states_f, shocks_f = mod_matrices.jacobian_matrices(parameter_theta) # Call correct function name
                    print("Function call successful. Comparing matrices...")
                    try: # Assertions
                        assert np.allclose(A_direct, A_f, atol=1e-8, equal_nan=True), "A mismatch"; assert np.allclose(B_direct, B_f, atol=1e-8, equal_nan=True), "B mismatch"
                        assert np.allclose(C_direct, C_f, atol=1e-8, equal_nan=True), "C mismatch"; assert np.allclose(D_direct, D_f, atol=1e-8, equal_nan=True), "D mismatch"
                        assert state_names_direct == states_f, "State mismatch"; assert shock_names_direct == shocks_f, "Shock mismatch"
                        print("Generated function tested successfully.")
                    except AssertionError as ae: print(f"!!! Assertion Error: {ae} !!!")
            except Exception as test_e: print(f"Error testing generated func: {test_e}"); import traceback; traceback.print_exc()
        else: print(f"\nGenerated file not found: {function_file}. Cannot test.")
    else: print("\nModel processing failed.")