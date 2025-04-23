import sympy
import re
import numpy as np
import os
import pickle
import collections # For defaultdict

class DynareParser:
    def __init__(self, mod_file_path):
        self.mod_file_path = mod_file_path
        self.param_names = []
        self.var_names = []
        self.shock_names = []
        self.equations_str = [] # Raw strings from model block

        self.symbols = {} # All symbols (vars, params, shocks, aux)
        self.sympy_equations = [] # Original equations as Sympy Eq objects

        # --- Intermediate Preprocessing Attributes ---
        # Stage 1: Timing Analysis
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        # Stage 2: Static Handling
        self.static_candidates = set()
        self.static_definitions = {} # Map static var symbol -> solving equation
        self.static_subs = {} # Map static var symbol -> solution expression
        self.equations_after_static = [] # Sympy Eq objects
        # Stage 3: Aux Variables
        self.aux_vars = {} # Map aux_name -> symbol
        self.aux_var_definitions = [] # Equations defining aux vars
        self.equations_after_aux = [] # Equations after substituting long leads/lags
        # Stage 4: State Definition
        self.state_vars_ordered = [] # Final list of state var symbols
        self.state_var_map = {} # Map symbol -> index
        self.final_equations_for_jacobian = [] # List of Sympy Eq for Jacobian

        # --- Run Parsing and Initial Setup ---
        print("--- Parsing Mod File ---")
        self._parse_mod_file()
        print("\n--- Creating Initial Symbols ---")
        self._create_initial_sympy_symbols()
        print("\n--- Parsing Equations to Sympy ---")
        self._parse_equations_to_sympy()

    # ===========================================
    # Stage 0: Parsing and Initial Setup
    # ===========================================
    def _parse_mod_file(self):
        """Robustly parses the .mod file sections."""
        with open(self.mod_file_path, 'r') as f:
            content = f.read()

        # 1. Remove comments first
        content = re.sub(r"//.*", "", content) # Line comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL) # Block comments

        # 2. Define patterns for blocks (case-insensitive, non-greedy)
        var_pattern = re.compile(r"var\s+(.*?);", re.IGNORECASE | re.DOTALL)
        varexo_pattern = re.compile(r"varexo\s+(.*?);", re.IGNORECASE | re.DOTALL)
        parameters_pattern = re.compile(r"parameters\s+(.*?);", re.IGNORECASE | re.DOTALL)
        model_pattern = re.compile(r"model\s*;\s*(.*?)\s*end\s*;", re.IGNORECASE | re.DOTALL)

        # 3. Extract and clean content
        var_match = var_pattern.search(content)
        if var_match:
            self.var_names = [v.strip() for v in var_match.group(1).split() if v.strip()]
        else:
            print("Warning: 'var' block not found.")

        varexo_match = varexo_pattern.search(content)
        if varexo_match:
            self.shock_names = [s.strip() for s in varexo_match.group(1).split() if s.strip()]
        else:
            print("Warning: 'varexo' block not found.")

        parameters_match = parameters_pattern.search(content)
        if parameters_match:
            # Remove trailing commas from parameters
            self.param_names = [p.strip().rstrip(',') for p in parameters_match.group(1).split() if p.strip()]
        else:
            print("Warning: 'parameters' block not found.")

        model_match = model_pattern.search(content)
        if model_match:
            eq_block = model_match.group(1)
            # Split equations by semicolon, clean whitespace
            raw_equations = [eq.strip() for eq in eq_block.split(';') if eq.strip()]
            self.equations_str = raw_equations
        else:
            print("Error: 'model; ... end;' block not found or parsed correctly.")

        print(f"Found Variables: {self.var_names}")
        print(f"Found Parameters: {self.param_names}")
        print(f"Found Shocks: {self.shock_names}")
        print(f"Found {len(self.equations_str)} Equations (raw):")
        # for i, eq in enumerate(self.equations_str):
        #     print(f"  {i+1}: {eq}")

    def _create_initial_sympy_symbols(self):
        """Creates symbols for variables, parameters, and shocks."""
        for name in self.var_names:
            if name and name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
        for name in self.param_names:
            if name and name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
        for name in self.shock_names:
             if name and name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
        print(f"Created {len(self.symbols)} initial symbols.")

    def _replace_lead_lag(self, expr_str):
        """Replaces var(+k) with var_pk and var(-k) with var_mk symbols."""
        # Find max lead/lag needed in this specific string to create symbols on the fly
        max_lead_local = 0
        max_lag_local = 0
        lead_lag_pattern_find = re.compile(r"\b([a-zA-Z_]\w*)\s*\(([+-])(\d+)\)")
        for match in lead_lag_pattern_find.finditer(expr_str):
             var, sign, num_str = match.groups()
             num = int(num_str)
             if sign == '+':
                 max_lead_local = max(max_lead_local, num)
             else:
                 max_lag_local = max(max_lag_local, num)

        # Ensure necessary lead/lag symbols exist in self.symbols
        for var_name in self.var_names:
             for k in range(1, max_lead_local + 1):
                 key = f"{var_name}_p{k}"
                 if key not in self.symbols:
                     self.symbols[key] = sympy.Symbol(key)
             for k in range(1, max_lag_local + 1):
                 key = f"{var_name}_m{k}"
                 if key not in self.symbols:
                     self.symbols[key] = sympy.Symbol(key)

        # Perform the replacement
        def repl(match):
            var, sign, num_str = match.groups()
            num = int(num_str)
            if var in self.var_names: # Only replace for known variables
                 key = f"{var}_{'p' if sign == '+' else 'm'}{num}"
                 if key in self.symbols:
                    return self.symbols[key].name
                 else:
                    # Should not happen if symbols created above
                    raise ValueError(f"Symbol '{key}' was not created for '{expr_str}'")
            else:
                # Not a variable, maybe a function call - leave it unchanged
                return match.group(0)

        # Regex to avoid replacing within names, requires careful lookarounds or tokenization
        # Simpler approach: Replace based on known variable list
        processed_str = expr_str
        lead_lag_pattern_replace = re.compile(r"\b([a-zA-Z_]\w*)\s*\(([+-])(\d+)\)")
        processed_str = lead_lag_pattern_replace.sub(repl, processed_str)
        return processed_str

    def _parse_equations_to_sympy(self):
        """Parses cleaned equation strings into Sympy Eq objects."""
        self.sympy_equations = []
        for i, eq_str in enumerate(self.equations_str):
            if not eq_str: continue
            try:
                processed_eq_str = self._replace_lead_lag(eq_str)
                if '=' in processed_eq_str:
                    lhs_str, rhs_str = processed_eq_str.split('=', 1)
                    # Add parentheses around RHS for safety before subtracting
                    lhs = sympy.parse_expr(lhs_str.strip(), local_dict=self.symbols)
                    rhs = sympy.parse_expr(rhs_str.strip(), local_dict=self.symbols)
                    self.sympy_equations.append(sympy.Eq(lhs - rhs, 0))
                else:
                    # Assume expression should be zero
                    expr = sympy.parse_expr(processed_eq_str, local_dict=self.symbols)
                    self.sympy_equations.append(sympy.Eq(expr, 0))
                # print(f"  Parsed Eq {i+1}: {self.sympy_equations[-1]}")
            except Exception as e:
                print(f"Error parsing equation: {eq_str}")
                print(f"Processed string: {processed_eq_str}")
                print(f"Sympy error: {e}")
                # Decide whether to raise error or just skip the equation
                # raise # Or continue

        print(f"Successfully parsed {len(self.sympy_equations)} equations into Sympy.")


    # ===========================================
    # Stage 1: Variable Timing Analysis
    # ===========================================
    def _analyze_variable_timing(self, file_path=None):
        """Analyzes lead/lag structure for each variable across all equations."""
        print("\n--- Stage 1: Analyzing Variable Timing ---")
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})

        var_syms = {sympy.Symbol(v) for v in self.var_names}

        for eq in self.sympy_equations:
            atoms = eq.lhs.free_symbols # LHS since eq is LHS - RHS = 0

            for var_sym in var_syms:
                base_name = var_sym.name
                if var_sym in atoms:
                    self.var_timing_info[var_sym]['appears_current'] = True

                # Check leads
                for k in range(1, 10): # Check reasonable max lead
                    lead_key = f"{base_name}_p{k}"
                    if lead_key in self.symbols and self.symbols[lead_key] in atoms:
                        self.var_timing_info[var_sym]['max_lead'] = max(self.var_timing_info[var_sym]['max_lead'], k)

                # Check lags
                for k in range(1, 10): # Check reasonable max lag
                    lag_key = f"{base_name}_m{k}"
                    if lag_key in self.symbols and self.symbols[lag_key] in atoms:
                         # min_lag stores negative value, so use -k
                        self.var_timing_info[var_sym]['min_lag'] = min(self.var_timing_info[var_sym]['min_lag'], -k)

        print("Variable Timing Analysis Results:")
        output_lines = []
        for var_sym in var_syms:
            info = self.var_timing_info[var_sym]
            line = f"- {var_sym.name}: Current={info['appears_current']}, MaxLead={info['max_lead']}, MinLag={info['min_lag']}"
            print(line)
            output_lines.append(line)

        if file_path:
            with open(file_path, "w") as f:
                f.write("Variable Timing Analysis Results:\n")
                f.write("\n".join(output_lines))
            print(f"Timing info saved to {file_path}")

        return self.var_timing_info

    # ===========================================
    # Stage 2: Identify and Solve Static Vars
    # ===========================================
    def _identify_and_solve_static(self, file_path=None):
        """Identifies and attempts to solve for static variables."""
        print("\n--- Stage 2: Identifying and Solving Static Variables ---")
        self.static_candidates = set()
        self.static_definitions = {}
        self.static_subs = {}
        dynamic_eqs_candidates = list(self.sympy_equations) # Start with all

        # Identify potential static variables (those *never* appearing with lead/lag)
        var_syms = {sympy.Symbol(v) for v in self.var_names}
        for var_sym in var_syms:
             info = self.var_timing_info[var_sym]
             if info['max_lead'] == 0 and info['min_lag'] == 0:
                 self.static_candidates.add(var_sym)
        print(f"Potential static candidates: {[v.name for v in self.static_candidates]}")

        solved_statics = set()
        made_change = True
        remaining_equations = list(self.sympy_equations)

        output_lines = [f"Potential static candidates: {[v.name for v in self.static_candidates]}"]

        while made_change:
            made_change = False
            next_remaining_equations = []
            solved_this_round = set()

            for eq in remaining_equations:
                eq_atoms = eq.lhs.free_symbols
                current_vars_in_eq = eq_atoms.intersection(var_syms) # Only non-lagged/lead symbols
                static_candidates_in_eq = current_vars_in_eq.intersection(self.static_candidates)
                unsolved_statics_in_eq = static_candidates_in_eq - solved_statics

                # Check if equation contains any leads or lags AT ALL
                has_leads_lags = any( (s.name.endswith(tuple(f"_p{i}" for i in range(1,10))) or \
                                      s.name.endswith(tuple(f"_m{i}" for i in range(1,10)))) \
                                     for s in eq_atoms)

                if not has_leads_lags and len(unsolved_statics_in_eq) == 1:
                    # This eq potentially defines the single remaining static var
                    static_var_to_solve = list(unsolved_statics_in_eq)[0]

                    # Check if it depends only on known parameters, shocks, or *already solved* statics
                    depends_only_on_known = True
                    for atom in eq_atoms:
                        if atom == static_var_to_solve: continue
                        is_param = atom in {sympy.Symbol(p) for p in self.param_names}
                        is_shock = atom in {sympy.Symbol(s) for s in self.shock_names}
                        is_solved_static = atom in solved_statics
                        if not (is_param or is_shock or is_solved_static):
                            depends_only_on_known = False
                            break

                    if depends_only_on_known:
                        try:
                            solution = sympy.solve(eq.lhs, static_var_to_solve)
                            if isinstance(solution, list) and len(solution) == 1:
                                solution = solution[0]
                                # Substitute already known statics into the solution
                                solution = solution.subs(self.static_subs)

                                line = f"- Solved for {static_var_to_solve.name}: {solution}"
                                print(line)
                                output_lines.append(line)

                                self.static_definitions[static_var_to_solve] = eq
                                self.static_subs[static_var_to_solve] = solution
                                solved_this_round.add(static_var_to_solve)
                                made_change = True
                            else:
                                line = f"- Warning: Could not uniquely solve for {static_var_to_solve.name} from {eq}. Solution: {solution}"
                                print(line)
                                output_lines.append(line)
                                next_remaining_equations.append(eq) # Keep eq
                        except Exception as e:
                            line = f"- Warning: sympy.solve failed for {static_var_to_solve.name} from {eq}. Error: {e}"
                            print(line)
                            output_lines.append(line)
                            next_remaining_equations.append(eq) # Keep eq
                    else:
                         next_remaining_equations.append(eq) # Depends on unsolved var
                else:
                     next_remaining_equations.append(eq) # Contains leads/lags or multiple unsolved statics

            solved_statics.update(solved_this_round)
            remaining_equations = next_remaining_equations

        self.equations_after_static = remaining_equations
        print(f"\nStatic variables solved: {[v.name for v in solved_statics]}")
        print(f"Remaining equations after static identification: {len(self.equations_after_static)}")

        if file_path:
            with open(file_path, "w") as f:
                 f.write("Static Variable Identification and Solution:\n")
                 f.write("\n".join(output_lines))
                 f.write(f"\nStatic variables solved: {[v.name for v in solved_statics]}\n")
                 f.write(f"Remaining equations: {len(self.equations_after_static)}\n")
                 # for eq in self.equations_after_static: f.write(f"  {eq}\n")
            print(f"Static analysis info saved to {file_path}")

        return self.static_subs, self.equations_after_static


    # ===========================================
    # Stage 3: Substitute Static Variables
    # ===========================================
    def _substitute_static_vars(self, file_path=None):
        """Substitutes solved static variables into the remaining dynamic equations."""
        print("\n--- Stage 3: Substituting Static Variables ---")
        if not self.static_subs:
            print("No static variables to substitute.")
            self.equations_after_static_sub = list(self.equations_after_static) # Use previous result
            return self.equations_after_static_sub

        subs_dict_with_leads_lags = {}
        for static_var, solution in self.static_subs.items():
            base_name = static_var.name
            subs_dict_with_leads_lags[static_var] = solution # Substitute current time
            # Substitute leads
            for k in range(1, 10):
                lead_key = f"{base_name}_p{k}"
                if lead_key in self.symbols:
                    lead_sym = self.symbols[lead_key]
                    # Apply lead to the solution expression
                    lead_solution = solution.copy()
                    subs_lead = {}
                    for atom in solution.free_symbols:
                        # Shift time for all variables, params/shocks unchanged
                        atom_match_p = re.match(r"(\w+)_p(\d+)", atom.name)
                        atom_match_m = re.match(r"(\w+)_m(\d+)", atom.name)
                        atom_match_base = atom.name in self.var_names # Base variable name

                        if atom_match_p:
                             an, ak = atom_match_p.groups()
                             new_sym_name = f"{an}_p{int(ak)+k}"
                        elif atom_match_m:
                             an, ak = atom_match_m.groups()
                             new_lag = int(ak) - k
                             if new_lag > 0: new_sym_name = f"{an}_m{new_lag}"
                             elif new_lag == 0: new_sym_name = an
                             else: new_sym_name = f"{an}_p{abs(new_lag)}"
                        elif atom_match_base:
                             new_sym_name = f"{atom.name}_p{k}"
                        else: # Parameter or shock
                             continue # No change

                        if new_sym_name not in self.symbols: # Ensure symbol exists
                            self.symbols[new_sym_name] = sympy.Symbol(new_sym_name)
                        subs_lead[atom] = self.symbols[new_sym_name]

                    subs_dict_with_leads_lags[lead_sym] = solution.subs(subs_lead)

            # Substitute lags (similar logic, shift time back)
            for k in range(1, 10):
                lag_key = f"{base_name}_m{k}"
                if lag_key in self.symbols:
                    lag_sym = self.symbols[lag_key]
                    lag_solution = solution.copy()
                    subs_lag = {}
                    for atom in solution.free_symbols:
                        atom_match_p = re.match(r"(\w+)_p(\d+)", atom.name)
                        atom_match_m = re.match(r"(\w+)_m(\d+)", atom.name)
                        atom_match_base = atom.name in self.var_names

                        if atom_match_p:
                             an, ak = atom_match_p.groups()
                             new_lead = int(ak) - k
                             if new_lead > 0: new_sym_name = f"{an}_p{new_lead}"
                             elif new_lead == 0: new_sym_name = an
                             else: new_sym_name = f"{an}_m{abs(new_lead)}"
                        elif atom_match_m:
                             an, ak = atom_match_m.groups()
                             new_sym_name = f"{an}_m{int(ak)+k}"
                        elif atom_match_base:
                             new_sym_name = f"{atom.name}_m{k}"
                        else: # Parameter or shock
                            continue

                        if new_sym_name not in self.symbols:
                           self.symbols[new_sym_name] = sympy.Symbol(new_sym_name)
                        subs_lag[atom] = self.symbols[new_sym_name]

                    subs_dict_with_leads_lags[lag_sym] = solution.subs(subs_lag)


        self.equations_after_static_sub = []
        output_lines = ["Substituting static variables:"]
        for i, eq in enumerate(self.equations_after_static):
            subbed_eq = sympy.Eq(sympy.simplify(eq.lhs.subs(subs_dict_with_leads_lags)), 0)
            self.equations_after_static_sub.append(subbed_eq)
            line = f"  Eq {i+1} original : {eq}"
            line2 = f"  Eq {i+1} substituted: {subbed_eq}"
            # print(line)
            # print(line2)
            output_lines.append(line)
            output_lines.append(line2)

        print(f"Substitution complete. {len(self.equations_after_static_sub)} equations remain.")

        if file_path:
            with open(file_path, "w") as f:
                f.write("Static Variable Substitution Results:\n")
                f.write("\n".join(output_lines))
            print(f"Static substitution info saved to {file_path}")

        return self.equations_after_static_sub

    # ===========================================
    # Stage 4: Handle Long Leads/Lags (Aux Vars)
    # ===========================================
    def _handle_aux_vars_revised(self, file_path=None):
        """Handles leads > +1 and lags < -1 by creating auxiliary variables."""
        print("\n--- Stage 4: Handling Auxiliary Vars (Long Leads/Lags) ---")
        self.aux_vars = {}
        self.aux_var_definitions = []
        current_equations = list(self.equations_after_static_sub)
        subs_long_leads = {}
        final_dynamic_vars = list(self.var_names) # Start with original
        final_dynamic_vars = [v for v in final_dynamic_vars if v not in [s.name for s in self.static_subs.keys()]]

        output_lines = ["Handling long leads/lags:"]

        # --- Handle Leads > +1 ---
        leads_to_replace = collections.defaultdict(int) # var_name -> max_lead_k
        for eq in current_equations:
             for atom in eq.lhs.free_symbols:
                 match_lead = re.match(r"(\w+)_p(\d+)", atom.name)
                 if match_lead:
                     base_name, lead_k = match_lead.groups()
                     lead_k = int(lead_k)
                     if lead_k > 1:
                         leads_to_replace[base_name] = max(leads_to_replace[base_name], lead_k)

        if leads_to_replace:
             output_lines.append("\nCreating auxiliary LEAD variables:")
             for var_name, max_lead_needed in leads_to_replace.items():
                 for k in range(1, max_lead_needed): # Aux for +2 up to +max_lead
                     aux_name = f"AUX_{var_name}_p{k}"
                     if aux_name not in self.symbols:
                         aux_sym = sympy.Symbol(aux_name)
                         self.symbols[aux_name] = aux_sym
                         self.aux_vars[aux_name] = aux_sym
                         final_dynamic_vars.append(aux_name) # Add aux var to list
                         # Define the aux equation: AUX_var_pk = Prev_AUX(+1) or Var(+1)
                         if k == 1: # AUX_p1 defines var(+2)
                             # Eq: AUX_var_p1 = var(+1) => AUX_var_p1 - var_p1 = 0
                             var_p1_sym = self.symbols[f"{var_name}_p1"]
                             new_eq = sympy.Eq(aux_sym - var_p1_sym, 0)
                         else: # AUX_pk defines var(+(k+1))
                             # Eq: AUX_var_pk = AUX_var_p(k-1)(+1) => AUX_var_pk - AUX_var_p(k-1)_p1 = 0
                             prev_aux_p1_name = f"AUX_{var_name}_p{k-1}_p1"
                             if prev_aux_p1_name not in self.symbols:
                                 self.symbols[prev_aux_p1_name] = sympy.Symbol(prev_aux_p1_name)
                             prev_aux_p1_sym = self.symbols[prev_aux_p1_name]
                             new_eq = sympy.Eq(aux_sym - prev_aux_p1_sym, 0)

                         self.aux_var_definitions.append(new_eq)
                         line = f"- Added aux var '{aux_name}' with definition: {new_eq}"
                         print(line)
                         output_lines.append(line)

                     # Add substitution rule for the original equations
                     # Replace var(+(k+1)) symbol with AUX_var_pk symbol
                     orig_lead_pKp1_key = f"{var_name}_p{k+1}"
                     if orig_lead_pKp1_key in self.symbols:
                         subs_long_leads[self.symbols[orig_lead_pKp1_key]] = self.symbols[aux_name]

        # --- Handle Lags < -1 (Create LAG state variables) ---
        # This is now handled in state definition, but track need here
        lags_to_handle = collections.defaultdict(int) # var_name -> max_lag_k
        for eq in current_equations:
             for atom in eq.lhs.free_symbols:
                 match_lag = re.match(r"(\w+)_m(\d+)", atom.name)
                 if match_lag:
                     base_name, lag_k = match_lag.groups()
                     lag_k = int(lag_k)
                     if lag_k > 1:
                         lags_to_handle[base_name] = max(lags_to_handle[base_name], lag_k)

        if lags_to_handle:
            output_lines.append("\nTracking required LAG variables (created in state definition):")
            for var_name, max_lag_k in lags_to_handle.items():
                 for k in range(1, max_lag_k): # Need LAG_m1 up to LAG_m(k-1)
                      lag_state_name = f"LAG_{var_name}_m{k}"
                      line = f"- Need state for {var_name}(-{k+1}) represented by '{lag_state_name}'"
                      print(line)
                      output_lines.append(line)
                      if lag_state_name not in self.symbols:
                          self.symbols[lag_state_name] = sympy.Symbol(lag_state_name)
                      # We don't add these to final_dynamic_vars here, state def does.

        # Apply long lead substitutions and add aux definitions
        self.equations_after_aux = [eq.subs(subs_long_leads) for eq in current_equations]
        self.equations_after_aux.extend(self.aux_var_definitions)

        # Store the final list of dynamic var names (including aux leads, excluding statics)
        self.final_dynamic_var_names = final_dynamic_vars

        print(f"\nTotal dynamic variables (incl. aux leads): {len(self.final_dynamic_var_names)}")
        print(f"Total equations after aux handling: {len(self.equations_after_aux)}")

        if file_path:
             with open(file_path, "w") as f:
                 f.write("Auxiliary Variable Handling Results:\n")
                 f.write("\n".join(output_lines))
                 f.write(f"\nFinal dynamic variable names: {self.final_dynamic_var_names}\n")
                 f.write(f"Final equations: {len(self.equations_after_aux)}\n")
                 # for eq in self.equations_after_aux: f.write(f"  {eq}\n")
             print(f"Auxiliary var info saved to {file_path}")

        return self.equations_after_aux

    # ===========================================
    # Stage 5: Define State Vector
    # ===========================================
    def _define_state_vector_revised(self, file_path=None):
        """Defines the ordered state vector including lagged states."""
        print("\n--- Stage 5: Defining State Vector ---")
        self.state_vars_ordered = []
        pred_part = []
        mixed_part = [] # Includes aux leads

        # Re-analyze timing based on *final* variables and *final* equations
        temp_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'is_pred': False})
        final_var_syms = {sympy.Symbol(v) for v in self.final_dynamic_var_names}

        for eq in self.equations_after_aux:
            atoms = eq.lhs.free_symbols
            for var_sym in final_var_syms:
                base_name = var_sym.name
                has_lead = False
                min_lag = 0
                for k in range(1, 10): # Leads
                    if f"{base_name}_p{k}" in self.symbols and self.symbols[f"{base_name}_p{k}"] in atoms:
                        has_lead = True; break
                for k in range(1, 10): # Lags
                    if f"{base_name}_m{k}" in self.symbols and self.symbols[f"{base_name}_m{k}"] in atoms:
                         min_lag = min(min_lag, -k)
                # Update overall timing
                if has_lead: temp_timing_info[var_sym]['max_lead'] = 1 # Just track if *any* lead
                temp_timing_info[var_sym]['min_lag'] = min(temp_timing_info[var_sym]['min_lag'], min_lag)

        # Classify based on final timing
        current_pred_vars = []
        current_mixed_vars = [] # Includes aux_leads
        lag_requirements = collections.defaultdict(int) # base_name -> max_lag_k

        for var_sym in final_var_syms:
            info = temp_timing_info[var_sym]
            max_lag_needed = abs(info['min_lag'])
            if max_lag_needed > 0:
                lag_requirements[var_sym.name] = max(lag_requirements[var_sym.name], max_lag_needed)

            if info['max_lead'] > 0 or var_sym.name.startswith("AUX_"): # Has leads or is aux lead var
                 current_mixed_vars.append(var_sym)
            else: # No leads detected in final equations
                 current_pred_vars.append(var_sym)

        # Build state vector: Predetermined + Lags, then Mixed + Lags
        state_set = set() # To avoid duplicates

        # Predetermined part
        for var_sym in sorted(current_pred_vars, key=lambda s: s.name): # Sort for consistency
             if var_sym not in state_set:
                 pred_part.append(var_sym)
                 state_set.add(var_sym)
             # Add required lagged states
             max_lag = lag_requirements.get(var_sym.name, 0)
             for k in range(1, max_lag): # If max_lag=2, need LAG_m1
                 lag_state_name = f"LAG_{var_sym.name}_m{k}"
                 if lag_state_name in self.symbols:
                     lag_sym = self.symbols[lag_state_name]
                     if lag_sym not in state_set:
                         pred_part.append(lag_sym)
                         state_set.add(lag_sym)
                 else: # Should have been created in aux handling if needed
                      print(f"Warning: Symbol for required lag state '{lag_state_name}' not found.")

        # Mixed part
        for var_sym in sorted(current_mixed_vars, key=lambda s: s.name):
             if var_sym not in state_set:
                 mixed_part.append(var_sym)
                 state_set.add(var_sym)
             # Add required lagged states (unlikely for aux vars, but possible for orig mixed)
             if not var_sym.name.startswith("AUX_"):
                  max_lag = lag_requirements.get(var_sym.name, 0)
                  for k in range(1, max_lag):
                      lag_state_name = f"LAG_{var_sym.name}_m{k}"
                      if lag_state_name in self.symbols:
                          lag_sym = self.symbols[lag_state_name]
                          if lag_sym not in state_set:
                              mixed_part.append(lag_sym)
                              state_set.add(lag_sym)
                      else:
                           print(f"Warning: Symbol for required lag state '{lag_state_name}' not found.")


        self.state_vars_ordered = pred_part + mixed_part
        self.state_var_map = {sym: i for i, sym in enumerate(self.state_vars_ordered)}

        print(f"Final State Vector (size {len(self.state_vars_ordered)}):")
        output_lines = [f"Final State Vector (size {len(self.state_vars_ordered)}):"]
        state_names_list = [s.name for s in self.state_vars_ordered]
        print(f"  {state_names_list}")
        output_lines.append(f"  {state_names_list}")

        if file_path:
            with open(file_path, "w") as f:
                f.write("State Vector Definition:\n")
                f.write("\n".join(output_lines))
            print(f"State vector info saved to {file_path}")

        return self.state_vars_ordered

    # ===========================================
    # Stage 6: Build Final Equations & Check
    # ===========================================
    def _build_final_equations(self, file_path=None):
        """Adds LAG identities and ensures equation count matches state count."""
        print("\n--- Stage 6: Building Final Equation System ---")
        self.final_equations_for_jacobian = []
        lag_identities = []
        processed_lag_states = set()

        output_lines = ["Building final equations:"]

        # Create identity equations for LAG states
        for state_sym in self.state_vars_ordered:
            match_lag_state = re.match(r"LAG_(\w+)_m(\d+)", state_sym.name)
            if match_lag_state and state_sym not in processed_lag_states:
                base_name, lag_k = match_lag_state.groups()
                lag_k = int(lag_k)

                # Identity: state(t) = previous_state(t-1)
                # LAG_X_mk(t) = LAG_X_m(k-1)(t-1) or LAG_X_mk(t) = X(t-1) if k=1
                state_now = state_sym # LAG_X_mk
                if lag_k == 1:
                    # Need symbol for X(t-1), which is base_name + "_m1"
                    prev_state_lagged_name = f"{base_name}_m1"
                else:
                    # Need symbol for LAG_X_m(k-1)(t-1)
                    prev_state_lagged_name = f"LAG_{base_name}_m{k-1}_m1"

                # Ensure the symbol for the RHS (lagged previous state) exists
                if prev_state_lagged_name not in self.symbols:
                    self.symbols[prev_state_lagged_name] = sympy.Symbol(prev_state_lagged_name)
                prev_state_lagged_sym = self.symbols[prev_state_lagged_name]

                identity_eq = sympy.Eq(state_now - prev_state_lagged_sym, 0)
                lag_identities.append(identity_eq)
                processed_lag_states.add(state_sym)
                line = f"- Added LAG identity: {identity_eq}"
                print(line)
                output_lines.append(line)


        # Combine original dynamic equations (after aux) + LAG identities + AUX definitions
        # Note: equations_after_aux already includes aux_var_definitions
        combined_eqs = list(self.equations_after_aux) + lag_identities

        # Ensure the count matches state vector size
        n_state = len(self.state_vars_ordered)
        n_eqs = len(combined_eqs)
        print(f"\nChecking counts: N_States = {n_state}, N_Equations = {n_eqs}")
        output_lines.append(f"\nChecking counts: N_States = {n_state}, N_Equations = {n_eqs}")


        if n_state != n_eqs:
            line = f"ERROR: State count ({n_state}) does not match equation count ({n_eqs})!"
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(line)
            print("Preprocessing failed. Check model structure, static/aux handling.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            output_lines.append(line)
            output_lines.append("Preprocessing failed.")
            # Decide whether to raise error or return empty
            raise ValueError("Equation count mismatch after preprocessing.")
            # self.final_equations_for_jacobian = [] # Indicate failure
        else:
             self.final_equations_for_jacobian = combined_eqs
             print("State and equation counts match.")
             output_lines.append("State and equation counts match.")


        if file_path:
            with open(file_path, "w") as f:
                 f.write("Final Equation System Build:\n")
                 f.write("\n".join(output_lines))
                 f.write(f"\nFinal equations ({len(self.final_equations_for_jacobian)}):\n")
                 # for eq in self.final_equations_for_jacobian: f.write(f"  {eq}\n")
            print(f"Final equation info saved to {file_path}")

        return self.final_equations_for_jacobian


    # ===========================================
    # Stage 7: Get Numerical Matrices
    # ===========================================
    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        """ Calculates numerical A, B, C, D from final equations and states."""
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing Stages 1-6 must be run successfully first.")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)

        # Prepare substitution dict for parameters
        param_subs = {self.symbols[p]: v for p, v in param_dict_values.items() if p in self.symbols}

        # Create symbolic vectors for state@t+1, t, t-1 based on ordered state list
        state_vec_t = sympy.Matrix(self.state_vars_ordered)

        # Need to map state(t+1) and state(t-1) carefully
        state_vec_tp1_list = []
        state_vec_tm1_list = []
        for state_sym in self.state_vars_ordered:
             base_name = state_sym.name.replace("LAG_","").split("_m")[0] # Get original name
             is_lag_state = state_sym.name.startswith("LAG_")

             # Lead: X(t+1) becomes X_p1, LAG_X_m1(t+1) becomes LAG_X_m1_p1
             lead_name = f"{state_sym.name}_p1"
             if lead_name not in self.symbols: self.symbols[lead_name] = sympy.Symbol(lead_name)
             state_vec_tp1_list.append(self.symbols[lead_name])

             # Lag: X(t-1) becomes X_m1, LAG_X_m1(t-1) becomes LAG_X_m1_m1
             lag_name = f"{state_sym.name}_m1"
             if lag_name not in self.symbols: self.symbols[lag_name] = sympy.Symbol(lag_name)
             state_vec_tm1_list.append(self.symbols[lag_name])

        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)
        shock_vec = sympy.Matrix([self.symbols[s] for s in self.shock_names])
        eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian]) # LHS of Eq(LHS-RHS, 0)

        # Calculate Jacobians
        print("Calculating Jacobians...")
        try:
            # Note sign conventions: A E[x(t+1)] + B x(t) + C x(t-1) + D eps(t) = 0
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            D_sym = eq_vec.jacobian(shock_vec)
        except Exception as e:
             print(f"Error during symbolic Jacobian calculation: {e}")
             raise

        # Substitute parameter values
        print("Substituting parameter values...")
        try:
            A_num = np.array(A_sym.subs(param_subs).tolist(), dtype=float)
            B_num = np.array(B_sym.subs(param_subs).tolist(), dtype=float)
            C_num = np.array(C_sym.subs(param_subs).tolist(), dtype=float)
            D_num = np.array(D_sym.subs(param_subs).tolist(), dtype=float)
        except Exception as e:
             print(f"Error during numerical substitution: {e}")
             print("Check parameter dictionary.")
             raise

        # Final checks
        if A_num.shape != (n_state, n_state) or \
           B_num.shape != (n_state, n_state) or \
           C_num.shape != (n_state, n_state) or \
           D_num.shape != (n_state, n_shocks):
            print(f"ERROR: Final matrix dimension mismatch!")
            print(f"Expected A,B,C: ({n_state},{n_state}), Got A:{A_num.shape}, B:{B_num.shape}, C:{C_num.shape}")
            print(f"Expected D: ({n_state},{n_shocks}), Got D:{D_num.shape}")
            raise ValueError("Matrix dimension mismatch.")

        print("Numerical matrices A, B, C, D successfully calculated.")

        # Save numerical matrices using pickle
        if file_path:
            matrix_data = {
                'A': A_num, 'B': B_num, 'C': C_num, 'D': D_num,
                'state_names': [s.name for s in self.state_vars_ordered],
                'shock_names': self.shock_names
            }
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(matrix_data, f)
            print(f"Numerical matrices saved to {file_path}")

        return A_num, B_num, C_num, D_num, [s.name for s in self.state_vars_ordered], self.shock_names

    # --- Helper to run all steps ---
    def run_preprocessing(self, output_dir="model_files_intermediate"):
        """Runs all preprocessing stages and saves intermediate results."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timing_file = os.path.join(output_dir, "1_timing_analysis.txt")
            static_file = os.path.join(output_dir, "2_static_analysis.txt")
            static_sub_file = os.path.join(output_dir, "3_static_substitution.txt")
            aux_file = os.path.join(output_dir, "4_aux_handling.txt")
            state_file = os.path.join(output_dir, "5_state_definition.txt")
            final_eq_file = os.path.join(output_dir, "6_final_equations.txt")
        else:
             timing_file=static_file=static_sub_file=aux_file=state_file=final_eq_file=None

        self._analyze_variable_timing(file_path=timing_file)
        self._identify_and_solve_static(file_path=static_file)
        self._substitute_static_vars(file_path=static_sub_file)
        self._handle_aux_vars_revised(file_path=aux_file)
        self._define_state_vector_revised(file_path=state_file)
        self._build_final_equations(file_path=final_eq_file)
        print("\n--- Preprocessing Complete ---")


# Example Usage
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # 1) Parse and generate all core & trend code
    mod_file = "qpm_simpl1_with_trends.dyn" # Replace with your actual file path
    output_dir_inter = "model_files_intermediate"
    output_dir_final = "model_files_numerical"
    final_matrices_file = os.path.join(output_dir_final, "reduced_matrices.pkl")


    parser = DynareParser(mod_file)
    parser.run_preprocessing(output_dir=output_dir_inter)

    # Define parameter values (replace with actual values for your model)
    parameter_values = {
        'b1': 0.5, 'b4': 0.1,
        'a1': 0.5, 'a2': 0.1,
        'g1': 0.8, 'g2': 0.0, 'g3': 0.2,
        'rho_DLA_CPI': 0.5, 'rho_L_GDP_GAP': 0.8, 'rho_rs': 0.7, 'rho_rs2': 0.15
    }

    try:
        # Calculate and save final numerical matrices
        A, B, C, D, state_names, shock_names = parser.get_numerical_ABCD(
            parameter_values,
            file_path=final_matrices_file
        )

        print("\n--- Final Numerical Matrices ---")
        print("State Names:", state_names)
        print("Shock Names:", shock_names)
        print("A shape:", A.shape)
        print("B shape:", B.shape)
        print("C shape:", C.shape)
        print("D shape:", D.shape)

        # Now you can load 'reduced_matrices.pkl' in your solver script

    except Exception as e:
        print(f"\n--- Error during final matrix calculation: ---")
        print(e)
        import traceback
        traceback.print_exc()