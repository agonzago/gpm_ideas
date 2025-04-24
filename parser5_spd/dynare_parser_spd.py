import sympy
import re
import numpy as np
import os
import pickle
import collections # For defaultdict
import sys
import importlib.util # For testing generated function

# --- Helper Function for Time Shifting Expressions ---
def time_shift_expression(expr, shift, parser_symbols, var_names_set):
    """
    Shifts the time index of variables within a Sympy expression. Handles base,
    lead (_p), lag (_m), aux_lead, and aux_lag vars based on parser_symbols.
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

        # Prioritize matching aux first, then base lead/lag, then base
        match_aux_lead = re.match(r"(aux_\w+_lead)(\d+)$", atom_name, re.IGNORECASE)
        match_aux_lag  = re.match(r"(aux_\w+_lag)(\d+)$", atom_name, re.IGNORECASE)
        match_lead = re.match(r"(\w+)_p(\d+)$", atom_name)
        match_lag  = re.match(r"(\w+)_m(\d+)$", atom_name)

        if match_aux_lead:
            base_name = match_aux_lead.group(1) # e.g., "aux_DLA_CPI_lead"
            current_k = int(match_aux_lead.group(2))
            is_var_type = True
        elif match_aux_lag:
            base_name = match_aux_lag.group(1) # e.g., "aux_RES_RS_lag"
            current_k = -int(match_aux_lag.group(2))
            is_var_type = True
        elif match_lead:
            base_name_cand = match_lead.group(1)
            # Avoid interpreting aux_..._p as base_p
            if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                 base_name = base_name_cand
                 current_k = int(match_lead.group(2))
                 is_var_type = True
            else:
                 base_name = None # Not a variable lead/lag
        elif match_lag:
             base_name_cand = match_lag.group(1)
             if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                 base_name = base_name_cand
                 current_k = -int(match_lag.group(2))
                 is_var_type = True
             else:
                 base_name = None
        elif atom_name in var_names_set: # Base variable name
            base_name = atom_name
            current_k = 0
            is_var_type = True

        if is_var_type:
            new_k = current_k + shift
            if new_k == 0:
                # Need the actual base name (strip aux_, _lead, _lag)
                clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
                if clean_base_match:
                    clean_base = clean_base_match.group(1)
                else:
                    clean_base = base_name # Assumes base_name was correct if not aux

                if clean_base in var_names_set:
                    new_sym_name = clean_base
                else:
                     # Fallback if base name was something like 'aux_VAR_lead' (no number)
                     # This shouldn't happen if base_name derived correctly from numbered aux vars
                     print(f"Warning: Could not find base var for '{atom_name}' during shift to t=0. Using original base '{base_name}'.")
                     if base_name in parser_symbols:
                         new_sym_name = base_name
                     else:
                         continue # Cannot determine target symbol

            elif new_k > 0:
                # Construct lead name using _p suffix
                clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
                if clean_base_match: clean_base = clean_base_match.group(1)
                else: clean_base = base_name
                prefix = "aux_" if base_name.lower().startswith("aux_") else ""
                new_sym_name = f"{prefix}{clean_base}_p{new_k}" # Standard _p suffix
            else: # new_k < 0
                # Construct lag name using _m suffix
                clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)", base_name, re.IGNORECASE)
                if clean_base_match: clean_base = clean_base_match.group(1)
                else: clean_base = base_name
                prefix = "aux_" if base_name.lower().startswith("aux_") else ""
                new_sym_name = f"{prefix}{clean_base}_m{abs(new_k)}" # Standard _m suffix

            # Ensure the new symbol exists
            if new_sym_name not in parser_symbols:
                parser_symbols[new_sym_name] = sympy.Symbol(new_sym_name)
            subs_dict[atom] = parser_symbols[new_sym_name]
        # else: atom is a parameter or shock, leave unchanged

    try:
        # Use xreplace for potentially more robust substitution of symbols
        shifted_expr = expr.xreplace(subs_dict)
    except Exception as e:
        print(f"Warning: xreplace failed during time_shift_expression for expr: {expr}. Trying subs. Error: {e}")
        try:
            shifted_expr = expr.subs(subs_dict) # Fallback to subs
        except Exception as e2:
            print(f"Error: Fallback subs also failed in time_shift_expression. Error: {e2}")
            shifted_expr = expr # Return original on error
    return shifted_expr


class DynareParser:
    def __init__(self, mod_file_path):
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
        self.aux_lead_vars = {}
        self.aux_lag_vars = {}
        self.aux_var_definitions = []
        self.equations_after_aux_sub = []
        self.final_dynamic_var_names = []
        self.state_vars_ordered = []
        self.state_var_map = {}
        self.final_equations_for_jacobian = []
        self.last_param_values = {}

        print(f"--- Parsing Mod File: {self.mod_file_path} ---")
        self._parse_mod_file()
        self.var_names_set = set(self.var_names)
        print("\n--- Creating Initial Symbols ---")
        self._create_initial_sympy_symbols()
        print("\n--- Parsing Equations to Sympy ---")
        self._parse_equations_to_sympy()
        print(f"Original {len(self.sympy_equations_original)} equations parsed.")

    def _parse_mod_file(self):
        try:
            with open(self.mod_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"ERROR: Mod file not found at {self.mod_file_path}")
            raise
        # Remove comments
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        # Extract blocks
        var_pattern = re.compile(r"var\s+(.*?);", re.IGNORECASE | re.DOTALL)
        varexo_pattern = re.compile(r"varexo\s+(.*?);", re.IGNORECASE | re.DOTALL)
        parameters_pattern = re.compile(r"parameters\s+(.*?);", re.IGNORECASE | re.DOTALL)
        model_pattern = re.compile(r"model\s*;\s*(.*?)\s*end\s*;", re.IGNORECASE | re.DOTALL)
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
            self.param_names = [p.strip().rstrip(',') for p in parameters_match.group(1).split() if p.strip()]
        else:
            print("Warning: 'parameters' block not found.")
        model_match = model_pattern.search(content)
        if model_match:
            self.equations_str = [eq.strip() for eq in model_match.group(1).split(';') if eq.strip()]
        else:
            raise ValueError("Model block not found or parsed correctly.")
        print(f"Found Variables: {self.var_names}")
        print(f"Found Parameters: {self.param_names}")
        print(f"Found Shocks: {self.shock_names}")
        print(f"Found {len(self.equations_str)} Equations (raw).")

    def _create_initial_sympy_symbols(self):
        all_names = self.var_names + self.param_names + self.shock_names
        for name in all_names:
            if name and name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
        print(f"Created {len(self.symbols)} initial symbols.")

    def _replace_dynare_timing_with_sympy_names(self, expr_str):
        sympy_names_needed = set()
        processed_str = expr_str
        replacements = []
        lead_lag_pattern = re.compile(r"\b([a-zA-Z_]\w*)\s*\(([+-]?)(\d+)\)")
        for match in lead_lag_pattern.finditer(processed_str):
            start, end = match.span()
            var, sign, num_str = match.groups()
            num = int(num_str)
            if var in self.var_names_set:
                if num != 0:
                    sym_name = f"{var}_{'p' if sign == '+' else 'm'}{num}"
                    replacements.append({'start': start, 'end': end, 'sym_name': sym_name})
                    sympy_names_needed.add(sym_name)
        new_processed_str = processed_str
        for rep in sorted(replacements, key=lambda x: x['start'], reverse=True):
            new_processed_str = new_processed_str[:rep['start']] + rep['sym_name'] + new_processed_str[rep['end']:]
        for name in sympy_names_needed:
            if name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
        return new_processed_str

    def _parse_equations_to_sympy(self):
        self.sympy_equations_original = []
        for i, eq_str in enumerate(self.equations_str):
            if not eq_str:
                continue
            try:
                processed_eq_str = self._replace_dynare_timing_with_sympy_names(eq_str)
                if '=' in processed_eq_str:
                    lhs_str, rhs_str = processed_eq_str.split('=', 1)
                    # Use evaluate=False to prevent immediate evaluation by sympy
                    lhs = sympy.parse_expr(lhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    rhs = sympy.parse_expr(rhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    self.sympy_equations_original.append(sympy.Eq(lhs - rhs, 0))
                else:
                    expr = sympy.parse_expr(processed_eq_str, local_dict=self.symbols, evaluate=False)
                    self.sympy_equations_original.append(sympy.Eq(expr, 0))
            except Exception as e:
                print(f"Error parsing equation: {eq_str}")
                print(f"Processed: {processed_eq_str}")
                print(f"Error: {e}")
                raise

    def _analyze_variable_timing(self, file_path=None):
        print("\n--- Stage 1: Analyzing Variable Timing ---")
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        var_syms = {self.symbols[v] for v in self.var_names}
        output_lines = ["Variable Timing Analysis Results:"]
        for eq in self.sympy_equations_original:
            atoms = eq.lhs.free_symbols
            for var_sym in var_syms:
                base_name = var_sym.name
                if var_sym in atoms:
                    self.var_timing_info[var_sym]['appears_current'] = True
                for k in range(1, 10):
                    pk = f"{base_name}_p{k}"; mk = f"{base_name}_m{k}"
                    if pk in self.symbols and self.symbols[pk] in atoms:
                        self.var_timing_info[var_sym]['max_lead'] = max(self.var_timing_info[var_sym]['max_lead'], k)
                    if mk in self.symbols and self.symbols[mk] in atoms:
                        self.var_timing_info[var_sym]['min_lag'] = min(self.var_timing_info[var_sym]['min_lag'], -k)
        for var_name in self.var_names:
             var_sym = self.symbols[var_name]
             info = self.var_timing_info[var_sym]
             line = f"- {var_sym.name}: Current={info['appears_current']}, MaxLead={info['max_lead']}, MinLag={info['min_lag']}"
             print(line)
             output_lines.append(line)
        if file_path:
            self._save_intermediate_file(file_path, output_lines, self.sympy_equations_original, "Original Equations")
        return self.var_timing_info

    # ===========================================
    # Stage 2: Identify and Eliminate Static Vars (ACCURATE DYNAMIC CHECK)
    # ===========================================
    def _identify_and_eliminate_static(self, file_path=None):
        """
        Identifies static variables based on the correct definition:
        1. Dynamic Vars Check: Iterate through each equation. If a variable
           appears contemporaneously AND with a lead/lag *in that same equation*,
           it's marked as dynamic.
        2. Candidate Static: Variables not marked as dynamic.
        3. Defining Eq & Solve: Iterate through equations to find ones that
           uniquely define an unsolved candidate static variable (where the
           candidate appears ONLY contemporaneously in that eq).
        Eliminates the defining equation after solving.
        """
        print("\n--- Stage 2: Identifying and Eliminating Static Variables ---")
        self.static_subs = {} # Reset results
        self.equations_after_static_elim = []
        output_lines = ["Static Variable Identification and Elimination:"]

        var_syms = {self.symbols[v] for v in self.var_names}
        dynamic_vars_syms = set() # Store confirmed dynamic variables

        # 1. Equation-by-Equation Dynamic Check
        output_lines.append("\nChecking for dynamic variables (contemporaneous + lead/lag in same eq):")
        for i, eq in enumerate(self.sympy_equations_original):
            eq_lhs = eq.lhs
            eq_atoms = eq_lhs.free_symbols
            current_vars_in_eq = eq_atoms.intersection(var_syms)

            for var_sym in current_vars_in_eq:
                if var_sym in dynamic_vars_syms: # Already confirmed dynamic
                    continue

                appears_contemporaneous = var_sym in eq_atoms
                appears_with_lead_lag = False
                for k in range(1, 10): # Check leads/lags for this var *in this eq*
                    pk_sym = self.symbols.get(f"{var_sym.name}_p{k}")
                    mk_sym = self.symbols.get(f"{var_sym.name}_m{k}")
                    if (pk_sym and pk_sym in eq_atoms) or \
                       (mk_sym and mk_sym in eq_atoms):
                        appears_with_lead_lag = True
                        break

                if appears_contemporaneous and appears_with_lead_lag:
                    if var_sym not in dynamic_vars_syms:
                         line = f"- Variable '{var_sym.name}' identified as dynamic from equation {i+1}."
                         print(line)
                         output_lines.append(line)
                         dynamic_vars_syms.add(var_sym)

        line = f"\nTotal dynamic variables identified: {len(dynamic_vars_syms)} {[v.name for v in dynamic_vars_syms]}"
        print(line)
        output_lines.append(line)

        # 2. Identify Candidate Static Variables
        candidate_static_vars = var_syms - dynamic_vars_syms
        line = f"Candidate static variables (not identified as dynamic): {[v.name for v in candidate_static_vars]}"
        print(line)
        output_lines.append(line)

        if not candidate_static_vars:
             print("No candidate static variables found.")
             output_lines.append("No candidate static variables found.")
             # All original equations are kept if no statics exist
             self.equations_after_static_elim = list(self.sympy_equations_original)
             if file_path:
                 self._save_intermediate_file(file_path, output_lines, self.equations_after_static_elim, "Equations After Static Elimination")
             return self.static_subs, self.equations_after_static_elim

        # 3. Iteratively Find Defining Equations and Solve for Candidate Statics
        remaining_equations = list(self.sympy_equations_original)
        solved_statics_syms = set()
        made_change = True
        iteration = 0

        while made_change:
            iteration += 1
            made_change = False
            next_remaining_equations = []
            solved_this_round = set()
            potential_defs_this_round = collections.defaultdict(list)

            output_lines.append(f"\n--- Solving Iteration {iteration} ---")

            # First pass: Identify potential definitions among remaining equations
            for eq in remaining_equations:
                eq_lhs = eq.lhs
                eq_atoms = eq_lhs.free_symbols
                found_potential_def_for_unsolved = False

                # Find candidate statics present in this equation
                candidate_statics_in_eq = eq_atoms.intersection(candidate_static_vars)

                for static_cand in candidate_statics_in_eq:
                    # Check if this candidate appears ONLY contemporaneously in *this* eq
                    appears_only_contemporaneously = True
                    if static_cand not in eq_atoms: # Check if symbol actually present
                        appears_only_contemporaneously = False
                    else:
                        for k in range(1, 10):
                            pk_sym = self.symbols.get(f"{static_cand.name}_p{k}")
                            mk_sym = self.symbols.get(f"{static_cand.name}_m{k}")
                            if (pk_sym and pk_sym in eq_atoms) or \
                               (mk_sym and mk_sym in eq_atoms):
                                appears_only_contemporaneously = False
                                break

                    if appears_only_contemporaneously:
                        # This equation is a potential definition for this static_cand
                        potential_defs_this_round[static_cand].append(eq)
                        if static_cand not in solved_statics_syms:
                            found_potential_def_for_unsolved = True

                # If eq doesn't potentially define any *currently unsolved* static var, keep it.
                # Otherwise, hold it until solving pass.
                if not found_potential_def_for_unsolved:
                    next_remaining_equations.append(eq)


            # Second pass: Attempt to solve based on unique potential definitions
            output_lines.append("Attempting to solve potential definitions:")
            eqs_used_this_round = set()
            for static_cand, eq_list in potential_defs_this_round.items():
                if static_cand in solved_statics_syms:
                    continue # Already solved

                # Check if only ONE potential defining equation was found for this *unsolved* static
                if len(eq_list) == 1:
                    defining_eq = eq_list[0]
                    eq_lhs = defining_eq.lhs

                    # Check if this equation has already been used to solve another static var this round
                    if defining_eq in eqs_used_this_round:
                        # Add eq back if it wasn't already kept and wasn't used
                        if defining_eq not in next_remaining_equations:
                             next_remaining_equations.append(defining_eq)
                        continue

                    # Try to solve
                    try:
                        solution_list = sympy.solve(eq_lhs, static_cand)
                        if isinstance(solution_list, (list, tuple)) and len(solution_list) == 1:
                            # Substitute already solved statics *into the solution itself*
                            current_subs = {s: expr for s, expr in self.static_subs.items()}
                            solution = solution_list[0].subs(current_subs)

                            # Final check: solution shouldn't contain the variable itself
                            if static_cand in solution.free_symbols:
                                line = f"- Warn: Solution for {static_cand.name} depends on itself! Eq: {sympy.sstr(eq_lhs)}=0"
                                print(line); output_lines.append(line)
                                # Keep the problematic equation
                                if defining_eq not in next_remaining_equations:
                                     next_remaining_equations.append(defining_eq)
                            else:
                                line = f"- Solved: {static_cand.name} = {solution} (from eq: {sympy.sstr(eq_lhs)} = 0)"
                                print(line); output_lines.append(line)
                                self.static_subs[static_cand] = solution
                                solved_this_round.add(static_cand)
                                made_change = True
                                eqs_used_this_round.add(defining_eq) # Mark eq as used
                                # Defining equation successfully used, DO NOT keep it

                        else: # Solve failed or non-unique
                            line = f"- Info: No unique solve for {static_cand.name} from potential def {sympy.sstr(eq_lhs)}=0. Sol: {solution_list}"
                            # print(line) # Verbose
                            output_lines.append(line)
                            # Keep the equation
                            if defining_eq not in next_remaining_equations:
                                 next_remaining_equations.append(defining_eq)

                    except NotImplementedError:
                         line = f"- Warn: sympy.solve NotImplementedError for {static_cand.name}. Eq: {sympy.sstr(eq_lhs)}=0"
                         print(line); output_lines.append(line)
                         if defining_eq not in next_remaining_equations: next_remaining_equations.append(defining_eq)
                    except Exception as e: # Catch other solve errors
                        line = f"- Warn: sympy.solve failed for {static_cand.name} from {sympy.sstr(eq_lhs)}=0. Error: {e}"
                        print(line); output_lines.append(line)
                        if defining_eq not in next_remaining_equations: next_remaining_equations.append(defining_eq)

                elif len(eq_list) > 1: # Multiple potential defining equations
                    line = f"- Info: Multiple potential defining equations found for {static_cand.name}. Keeping equations."
                    # print(line) # Verbose
                    output_lines.append(line)
                    # Add all potential equations back if they weren't already kept
                    for eq_pot in eq_list:
                        if eq_pot not in next_remaining_equations:
                             next_remaining_equations.append(eq_pot)
                # else: len(eq_list) == 0, means this static_cand didn't have a potential definition this round


            solved_statics_syms.update(solved_this_round)
            remaining_equations = next_remaining_equations
            if made_change:
                 output_lines.append(f"Solved in iteration {iteration}: {[s.name for s in solved_this_round]}")
            elif iteration == 1 and not solved_statics_syms:
                 output_lines.append("\nNo static variables could be solved in the first iteration.")

        self.equations_after_static_elim = remaining_equations
        line = f"\nStatic vars solved: {[v.name for v in solved_statics_syms]}"
        print(line); output_lines.append(line)
        line = f"Remaining equations after static elim: {len(self.equations_after_static_elim)}"
        print(line); output_lines.append(line)

        if file_path:
            self._save_intermediate_file(file_path, output_lines, self.equations_after_static_elim, "Equations After Static Elimination")
        return self.static_subs, self.equations_after_static_elim
    
    def _substitute_static_vars(self, file_path=None):
        print("\n--- Stage 3: Substituting Static Variables ---")
        output_lines = ["Substituting static variables:"]
        if not self.static_subs:
            print("No static vars to substitute.")
            output_lines.append("No static vars to substitute.")
            self.equations_after_static_sub = list(self.equations_after_static_elim)
            if file_path:
                self._save_intermediate_file(file_path, output_lines, self.equations_after_static_sub, "Equations After Static Substitution")
            return self.equations_after_static_sub

        subs_dict_full = {}
        max_lead_needed = 0
        min_lag_needed = 0
        static_syms_to_sub = set(self.static_subs.keys())

        # Determine max lead/lag of STATIC variables needed in remaining equations
        for eq in self.equations_after_static_elim:
            for atom in eq.lhs.free_symbols:
                base_name = None
                k = 0
                is_static_related = False
                match_lead = re.match(r"(\w+)_p(\d+)", atom.name)
                match_lag = re.match(r"(\w+)_m(\d+)", atom.name)
                if match_lead:
                    base_name = match_lead.group(1)
                    k = int(match_lead.group(2))
                    if self.symbols.get(base_name) in static_syms_to_sub:
                        max_lead_needed = max(max_lead_needed, k)
                elif match_lag:
                    base_name = match_lag.group(1)
                    k = int(match_lag.group(2))
                    if self.symbols.get(base_name) in static_syms_to_sub:
                         min_lag_needed = min(min_lag_needed, -k)

        output_lines.append(f"Max lead needed for static subs: {max_lead_needed}")
        output_lines.append(f"Min lag needed for static subs: {min_lag_needed}")

        # Create the full substitution dictionary
        for static_var, solution in self.static_subs.items():
            subs_dict_full[static_var] = solution # Current time
            for k in range(1, max_lead_needed + 1):
                lead_key = f"{static_var.name}_p{k}"
                if lead_key in self.symbols:
                    lead_solution = time_shift_expression(solution, k, self.symbols, self.var_names_set)
                    subs_dict_full[self.symbols[lead_key]] = lead_solution
                    output_lines.append(f"- Sub rule: {lead_key} -> {sympy.sstr(lead_solution)}")
            for k in range(1, abs(min_lag_needed) + 1):
                lag_key = f"{static_var.name}_m{k}"
                if lag_key in self.symbols:
                    lag_solution = time_shift_expression(solution, -k, self.symbols, self.var_names_set)
                    subs_dict_full[self.symbols[lag_key]] = lag_solution
                    output_lines.append(f"- Sub rule: {lag_key} -> {sympy.sstr(lag_solution)}")

        # Apply substitutions
        self.equations_after_static_sub = []
        output_lines.append("\nApplying substitutions:")
        for i, eq in enumerate(self.equations_after_static_elim):
            substituted_lhs = eq.lhs.xreplace(subs_dict_full)
            try:
                # Use evaluate=False during simplify if needed
                simplified_lhs = sympy.simplify(substituted_lhs)
            except Exception as e:
                print(f"Simplify failed eq{i+1}: {e}")
                simplified_lhs = substituted_lhs
            subbed_eq = sympy.Eq(simplified_lhs, 0)
            self.equations_after_static_sub.append(subbed_eq)
            line2 = f"  Eq {i+1} substituted: {sympy.sstr(subbed_eq.lhs, full_prec=False)} = 0"
            output_lines.append(line2)
        line = f"Substitution complete. {len(self.equations_after_static_sub)} equations remain."
        print(line)
        output_lines.append(line)
        if file_path:
            self._save_intermediate_file(file_path, output_lines, self.equations_after_static_sub, "Equations After Static Substitution")
        return self.equations_after_static_sub

    def _handle_aux_vars(self, file_path=None):
        print("\n--- Stage 4: Handling Aux Vars (User Naming) ---")
        self.aux_lead_vars = {}
        self.aux_lag_vars = {}
        self.aux_var_definitions = []
        current_equations = list(self.equations_after_static_sub)
        subs_long_leads_lags = {}
        dynamic_base_var_names = [v for v in self.var_names if v not in [s.name for s in self.static_subs.keys()]]
        output_lines = ["Handling long leads/lags using 'aux_VAR_leadK'/'aux_VAR_lagK':"]

        # --- Leads > +1 ---
        leads_to_replace = collections.defaultdict(int)
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                match = re.match(r"(\w+)_p(\d+)", atom.name)
                if match:
                    base, k = match.groups()
                    k = int(k)
                    if k > 1 and base in dynamic_base_var_names:
                        leads_to_replace[base] = max(leads_to_replace[base], k)

        if leads_to_replace:
            output_lines.append("\nCreating aux LEAD variables:")
            for var_name in sorted(leads_to_replace.keys()):
                max_lead = leads_to_replace[var_name]
                for k in range(1, max_lead):
                    aux_name = f"aux_{var_name}_lead{k}"
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    # Always add/update in aux_lead_vars map
                    self.aux_lead_vars[aux_name] = self.symbols[aux_name]

                    if k == 1:
                        var_p1_name = f"{var_name}_p1"
                        if var_p1_name not in self.symbols: self.symbols[var_p1_name] = sympy.Symbol(var_p1_name)
                        new_eq = sympy.Eq(self.symbols[aux_name] - self.symbols[var_p1_name], 0)
                    else:
                        prev_aux_name = f"aux_{var_name}_lead{k-1}"
                        prev_aux_p1_name = f"{prev_aux_name}_p1"
                        if prev_aux_p1_name not in self.symbols: self.symbols[prev_aux_p1_name] = sympy.Symbol(prev_aux_p1_name)
                        new_eq = sympy.Eq(self.symbols[aux_name] - self.symbols[prev_aux_p1_name], 0)

                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)
                        line = f"- Added aux var '{aux_name}'. Def: {new_eq.lhs}=0"
                        print(line)
                        output_lines.append(line)

                    orig_lead_key = f"{var_name}_p{k+1}"
                    if orig_lead_key in self.symbols:
                        subs_long_leads_lags[self.symbols[orig_lead_key]] = self.symbols[aux_name]

        # --- Lags < -1 ---
        lags_to_replace = collections.defaultdict(int)
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                match = re.match(r"(\w+)_m(\d+)", atom.name)
                if match:
                    base, k = match.groups()
                    k = int(k)
                    if k > 1 and base in dynamic_base_var_names:
                        lags_to_replace[base] = max(lags_to_replace[base], k)

        if lags_to_replace:
            output_lines.append("\nCreating aux LAG variables:")
            for var_name in sorted(lags_to_replace.keys()):
                 max_lag = lags_to_replace[var_name]
                 for k in range(1, max_lag):
                     aux_name = f"aux_{var_name}_lag{k}"
                     if aux_name not in self.symbols:
                         self.symbols[aux_name] = sympy.Symbol(aux_name)
                     # Always add/update in aux_lag_vars map
                     self.aux_lag_vars[aux_name] = self.symbols[aux_name]

                     if k == 1:
                         var_m1_name = f"{var_name}_m1"
                         if var_m1_name not in self.symbols: self.symbols[var_m1_name] = sympy.Symbol(var_m1_name)
                         new_eq = sympy.Eq(self.symbols[aux_name] - self.symbols[var_m1_name], 0)
                     else:
                         prev_aux_name = f"aux_{var_name}_lag{k-1}"
                         prev_aux_m1_name = f"{prev_aux_name}_m1"
                         if prev_aux_m1_name not in self.symbols: self.symbols[prev_aux_m1_name] = sympy.Symbol(prev_aux_m1_name)
                         new_eq = sympy.Eq(self.symbols[aux_name] - self.symbols[prev_aux_m1_name], 0)

                     if new_eq not in self.aux_var_definitions:
                         self.aux_var_definitions.append(new_eq)
                         line = f"- Added aux var '{aux_name}'. Def: {new_eq.lhs}=0"
                         print(line)
                         output_lines.append(line)

                     orig_lag_key = f"{var_name}_m{k+1}"
                     aux_lag_m1_key = f"{aux_name}_m1"
                     if orig_lag_key in self.symbols:
                         if aux_lag_m1_key not in self.symbols:
                             self.symbols[aux_lag_m1_key] = sympy.Symbol(aux_lag_m1_key)
                         subs_long_leads_lags[self.symbols[orig_lag_key]] = self.symbols[aux_lag_m1_key]

        # Apply substitutions
        self.equations_after_aux_sub = []
        output_lines.append("\nApplying long lead/lag substitutions:")
        for i, eq in enumerate(current_equations):
             subbed_lhs = eq.lhs.xreplace(subs_long_leads_lags)
             try:
                 simplified_lhs = sympy.simplify(subbed_lhs)
             except Exception:
                 simplified_lhs = subbed_lhs
             subbed_eq = sympy.Eq(simplified_lhs, 0)
             self.equations_after_aux_sub.append(subbed_eq)
             line2 = f"  Eq {i+1} substituted: {sympy.sstr(subbed_eq.lhs, full_prec=False)} = 0"
             output_lines.append(line2)

        # Combine base names + all aux var names for the next stage
        self.final_dynamic_var_names = dynamic_base_var_names + list(self.aux_lead_vars.keys()) + list(self.aux_lag_vars.keys())

        line = f"\nSubst complete. State vars (base+aux_lead+aux_lag): {len(self.final_dynamic_var_names)}"
        print(line)
        output_lines.append(line)
        print(f"  {self.final_dynamic_var_names}")
        line = f"Eqs after aux sub (excl defs): {len(self.equations_after_aux_sub)}"
        print(line)
        output_lines.append(line)
        line = f"Aux var defs (lead+lag): {len(self.aux_var_definitions)}"
        print(line)
        output_lines.append(line)

        if file_path:
            self._save_intermediate_file(file_path, output_lines, self.equations_after_aux_sub + self.aux_var_definitions, "Equations After Aux Handling")
        return self.equations_after_aux_sub, self.aux_var_definitions

    def _define_state_vector(self, file_path=None):
        print("\n--- Stage 5: Defining State Vector ---")
        state_candidate_names = list(self.final_dynamic_var_names)
        state_candidate_syms = {self.symbols[name] for name in state_candidate_names if name in self.symbols}
        pred_vars = []
        mixed_vars = []
        final_system_equations = self.equations_after_aux_sub + self.aux_var_definitions

        for state_sym in state_candidate_syms:
            has_lead_p1 = False
            lead_p1_sym_name = f"{state_sym.name}_p1"
            if lead_p1_sym_name in self.symbols:
                 lead_p1_sym = self.symbols[lead_p1_sym_name]
                 for eq in final_system_equations:
                      # Check if the lead term appears in the equation's free symbols
                      if lead_p1_sym in eq.lhs.free_symbols:
                          has_lead_p1 = True
                          break

            # Classify: AUX_lead are mixed, others classified by presence of _p1
            if state_sym.name.startswith("aux_") and "_lead" in state_sym.name:
                mixed_vars.append(state_sym)
            elif has_lead_p1:
                 mixed_vars.append(state_sym)
            else: # Base predetermined and AUX_lag states
                pred_vars.append(state_sym)

        pred_orig = sorted([s for s in pred_vars if not s.name.startswith("aux_")], key=lambda x: x.name)
        pred_lags = sorted([s for s in pred_vars if s.name.startswith("aux_") and "_lag" in s.name], key=lambda x: x.name)
        mixed_orig = sorted([s for s in mixed_vars if not s.name.startswith("aux_")], key=lambda x: x.name)
        mixed_leads = sorted([s for s in mixed_vars if s.name.startswith("aux_") and "_lead" in s.name], key=lambda x: x.name)

        self.state_vars_ordered = pred_orig + pred_lags + mixed_orig + mixed_leads
        self.state_var_map = {sym: i for i, sym in enumerate(self.state_vars_ordered)}

        line = f"Final State Vector (size {len(self.state_vars_ordered)}):"
        print(line)
        state_names_list = [s.name for s in self.state_vars_ordered]
        print(f"  Predetermined ({len(pred_orig + pred_lags)}): {[s.name for s in pred_orig + pred_lags]}")
        print(f"  Mixed ({len(mixed_orig + mixed_leads)}): {[s.name for s in mixed_orig + mixed_leads]}")
        output_lines = [line, f"  State Vector: {state_names_list}"]
        if file_path:
            self._save_intermediate_file(file_path, output_lines)
        return self.state_vars_ordered

    def _build_final_equations(self, file_path=None):
        print("\n--- Stage 6: Building Final Equation System ---")
        # Combined equations are substituted dynamic equations + all aux definitions
        self.final_equations_for_jacobian = list(self.equations_after_aux_sub) + list(self.aux_var_definitions)
        output_lines = ["Final equations = (Substituted dynamic equations + All aux definitions):"]
        n_state = len(self.state_vars_ordered)
        n_eqs = len(self.final_equations_for_jacobian)
        line1 = f"\nChecking counts: N_States = {n_state}, N_Equations = {n_eqs}"
        print(line1)
        output_lines.append(line1)
        if n_state != n_eqs:
            line2 = f"ERROR: State count ({n_state}) != equation count ({n_eqs})!"
            print("!"*30 + "\n" + line2)
            print("State Vars:", [s.name for s in self.state_vars_ordered])
            print("Equations considered:")
            for i, eq in enumerate(self.final_equations_for_jacobian):
                print(f"  {i+1}: {sympy.sstr(eq.lhs)} = 0")
            print("Preprocessing failed." + "\n" + "!"*30)
            output_lines.append(line2)
            output_lines.append("Preprocessing failed.")
            raise ValueError("Equation count mismatch after final assembly.")
        else:
             line2 = "State and equation counts match."
             print(line2)
             output_lines.append(line2)
        if file_path:
            self._save_intermediate_file(file_path, output_lines, self.final_equations_for_jacobian, "Final Equation System")
        return self.final_equations_for_jacobian

    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing failed.")
        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)
        self.last_param_values = param_dict_values
        param_subs = {self.symbols[p]: v for p, v in param_dict_values.items() if p in self.symbols}

        # Create symbolic vectors ensuring all needed symbols exist
        state_vec_t = sympy.Matrix(self.state_vars_ordered)
        state_vec_tp1_list = []
        state_vec_tm1_list = []
        needed_jacobian_symbols = set(state_vec_t) # Start with t=0 symbols
        for state_sym in self.state_vars_ordered:
             lead_name = f"{state_sym.name}_p1"; lag_name = f"{state_sym.name}_m1"
             if lead_name not in self.symbols: self.symbols[lead_name] = sympy.Symbol(lead_name)
             state_vec_tp1_list.append(self.symbols[lead_name])
             needed_jacobian_symbols.add(self.symbols[lead_name])
             if lag_name not in self.symbols: self.symbols[lag_name] = sympy.Symbol(lag_name)
             state_vec_tm1_list.append(self.symbols[lag_name])
             needed_jacobian_symbols.add(self.symbols[lag_name])
        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)
        shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
        shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None
        eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])
        needed_jacobian_symbols.update(eq_vec.free_symbols) # Add symbols from equations

        print("Calculating Jacobians...")
        try:
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            if shock_vec and n_shocks > 0 :
                D_sym = -eq_vec.jacobian(shock_vec) # Sign change for standard form
            else:
                D_sym = sympy.zeros(n_state, n_shocks)
        except Exception as e:
            print(f"Error during symbolic Jacobian calculation: {e}")
            # Debug: Check which symbols might be missing or causing issues
            print("Symbols needed for Jacobians:", needed_jacobian_symbols)
            print("Symbols available:", self.symbols.keys())
            raise

        print("Substituting parameter values...")
        try:
            # N evaluates the symbolic matrix with substitutions
            A_num = np.array(A_sym.evalf(subs=param_subs).tolist(), dtype=float)
            B_num = np.array(B_sym.evalf(subs=param_subs).tolist(), dtype=float)
            C_num = np.array(C_sym.evalf(subs=param_subs).tolist(), dtype=float)
            if n_shocks > 0:
                D_num = np.array(D_sym.evalf(subs=param_subs).tolist(), dtype=float)
            else:
                D_num = np.zeros((n_state, 0), dtype=float)
        except Exception as e:
            print(f"Error during numerical substitution (evalf): {e}")
            raise

        expected_d_shape = (n_state, n_shocks)
        if A_num.shape!=(n_state,n_state) or B_num.shape!=(n_state,n_state) or C_num.shape!=(n_state,n_state) or D_num.shape!=expected_d_shape:
            print(f"ERROR: Matrix dimension mismatch!")
            print(f" A:{A_num.shape}, B:{B_num.shape}, C:{C_num.shape}, D:{D_num.shape}")
            print(f" Expected A,B,C:({n_state},{n_state}), Expected D:{expected_d_shape}")
            raise ValueError("Matrix dimension mismatch.")
        print("Numerical matrices A, B, C, D calculated.")
        if file_path:
            self._save_final_matrices(file_path, A_num, B_num, C_num, D_num)
        return A_num, B_num, C_num, D_num, [s.name for s in self.state_vars_ordered], self.shock_names


    # ===========================================
    # generate_matrix_function_file (FINAL - ROBUST CODE GENERATION FOR JAX/NUMPYRO)
    # ===========================================
    def generate_matrix_function_file(self, filename="jacobian_matrices.py"):
        """
        Generates a Python file defining jacobian_matrices(theta) -> A, B, C, D.
        - Accepts parameters as an ordered list/array `theta`.
        - Uses element-by-element assignment for matrix construction.
        - Replaces standard math functions with numpy equivalents (np.*)
          for better JAX/NumPyro compatibility.
        """
        function_name = "jacobian_matrices"
        print(f"\n--- Generating Python Function File: {filename} ({function_name}) ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing must be run successfully first (requires final equations and state order).")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)

        # 1. Recalculate Symbolic Jacobians (Ensure they are available)
        try:
            state_vec_t=sympy.Matrix(self.state_vars_ordered)
            state_vec_tp1_list=[]; state_vec_tm1_list=[]
            for s in self.state_vars_ordered: # Ensure p1/m1 symbols exist
                ln=f"{s.name}_p1"; mn=f"{s.name}_m1"
                if ln not in self.symbols: self.symbols[ln]=sympy.Symbol(ln)
                state_vec_tp1_list.append(self.symbols[ln])
                if mn not in self.symbols: self.symbols[mn]=sympy.Symbol(mn)
                state_vec_tm1_list.append(self.symbols[mn])
            state_vec_tp1=sympy.Matrix(state_vec_tp1_list); state_vec_tm1=sympy.Matrix(state_vec_tm1_list)
            shock_syms_list=[self.symbols[s] for s in self.shock_names if s in self.symbols]; shock_vec=sympy.Matrix(shock_syms_list) if shock_syms_list else None
            eq_vec=sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])

            print("Calculating symbolic Jacobians...")
            A_sym=eq_vec.jacobian(state_vec_tp1)
            B_sym=eq_vec.jacobian(state_vec_t)
            C_sym=eq_vec.jacobian(state_vec_tm1)
            if shock_vec and n_shocks > 0 : D_sym = -eq_vec.jacobian(shock_vec) # Standard form requires -D
            else: D_sym = sympy.zeros(n_state, n_shocks)
            print("Jacobians calculated.")
        except Exception as e:
            print(f"Error calculating Jacobians for function generation: {e}"); raise

        # 2. Prepare Parameter Info
        ordered_params_from_mod = self.param_names
        param_symbols_in_matrices = set().union(*(mat.free_symbols for mat in [A_sym, B_sym, C_sym, D_sym]))
        used_params_ordered = [p for p in ordered_params_from_mod if p in [s.name for s in param_symbols_in_matrices]]
        param_indices = {p: i for i, p in enumerate(ordered_params_from_mod)}

        # 3. Generate Python Code for Matrix Assignments
        def generate_matrix_assignments_code(matrix_sym, matrix_name, rows, cols):
            """Generates the element assignment lines for one matrix, using np.* funcs."""
            indent = "    " # 4 spaces
            code_lines = [f"{indent}{matrix_name} = np.zeros(({rows}, {cols}), dtype=float)"]
            # Mappings from sympy function names/operators to numpy equivalents
            sympy_to_numpy_map = {
                'exp': 'np.exp', 'log': 'np.log', 'sqrt': 'np.sqrt',
                'Abs': 'np.abs', 'sign': 'np.sign',
                'sin': 'np.sin', 'cos': 'np.cos', 'tan': 'np.tan',
                'asin': 'np.arcsin', 'acos': 'np.arccos', 'atan': 'np.arctan',
                'sinh': 'np.sinh', 'cosh': 'np.cosh', 'tanh': 'np.tanh',
                # Add more as needed based on functions used in models
            }
            # Regex for function names (ensuring word boundary)
            func_regex = re.compile(r"\b(" + "|".join(sympy_to_numpy_map.keys()) + r")(?=\()")

            # Regex for power operator (more robust)
            # Matches: base**exponent, base**(exponent), (base)**exponent, (base)**(exponent)
            # Where base can be a variable name or a parenthesized expression
            # And exponent is a number (int or float)
            pow_regex = re.compile(r"(\b[a-zA-Z_]\w*\b|\([^)]+\))\s*\*\*\s*(\d+\.?\d*|\([\d\s.+\-*/]+\))")


            for r in range(rows):
                for c in range(cols):
                    element = matrix_sym[r, c]
                    if element != 0 and element is not sympy.S.Zero : # Ensure it's actually non-zero
                        try:
                            # Get standard string representation
                            expr_str = sympy.sstr(element, full_prec=False)

                            # Replace function names using regex
                            expr_str = func_regex.sub(lambda m: sympy_to_numpy_map[m.group(1)], expr_str)

                            # Replace power operator: x**y -> np.power(x, y)
                            # Need to handle potential nested powers carefully if they occur
                            # Iteratively replace until no more ** are found
                            while "**" in expr_str:
                                original_str = expr_str # Store original for comparison
                                # Apply power regex - this needs refinement for nested/complex cases
                                # A simpler approach might be sufficient for typical models
                                expr_str = pow_regex.sub(r'np.power(\1, \2)', expr_str)
                                if expr_str == original_str: # Avoid infinite loop if regex fails
                                     print(f"Warning: Power replacement stopped for {matrix_name}[{r},{c}]. String: {expr_str}")
                                     break

                            # Append the assignment line
                            code_lines.append(f"{indent}{matrix_name}[{r}, {c}] = {expr_str}")
                        except Exception as str_e:
                             print(f"Warning: Could not convert element ({r},{c}) of {matrix_name} to code. Element: {element}. Error: {str_e}")
                             code_lines.append(f"{indent}# Error generating code for {matrix_name}[{r}, {c}] = {element}")

            # Return joined code lines, ensuring last line has newline if code exists
            return "\n".join(code_lines) if code_lines[1:] else code_lines[0]

        print("Generating Python code strings for matrices using np.* functions...")
        code_A = generate_matrix_assignments_code(A_sym, 'A', n_state, n_state)
        code_B = generate_matrix_assignments_code(B_sym, 'B', n_state, n_state)
        code_C = generate_matrix_assignments_code(C_sym, 'C', n_state, n_state)
        code_D = generate_matrix_assignments_code(D_sym, 'D', n_state, n_shocks)
        print("Code generation complete.")

        # 4. Assemble the Python File Content
        param_list_str_for_doc = str(ordered_params_from_mod)
        # Add line breaks for readability if param list is long
        if len(param_list_str_for_doc) > 70:
            import textwrap
            param_list_str_for_doc = textwrap.fill(param_list_str_for_doc, width=70,
                                                  initial_indent=' ' * 36,
                                                  subsequent_indent=' ' * 36).strip()


        file_content = f"""\
# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'
# Timestamp: {datetime.datetime.now().isoformat()}
import numpy as np
# Note: No 'from math import *' needed as we aim for np.* functions

def {function_name}(theta):
    \"\"\"
    Calculates the A, B, C, D matrices for the reduced state-space model
    in the form: A E[z(t+1)] + B z(t) + C z(t-1) + D eps(t) = 0

    Args:
        theta (list or np.ndarray): Array/list of parameter values in the
                                    order defined in the .mod file:
                                    {param_list_str_for_doc}

    Returns:
        tuple: (A, B, C, D, state_names, shock_names)
    \"\"\"
    # Unpack parameters using the order from the .mod file
    expected_len = {len(ordered_params_from_mod)}
    if len(theta) != expected_len:
        raise ValueError(f"Expected {{expected_len}} parameters, but got {{len(theta)}}")
    try:
"""
        # Parameter unpacking code
        unpacking_code = []
        for p_name in used_params_ordered:
            idx = param_indices[p_name]
            unpacking_code.append(f"        {p_name} = theta[{idx}]")
        file_content += "\n".join(unpacking_code) + "\n"

        file_content += f"""\
    except IndexError:
        raise IndexError(f"Parameter vector 'theta' has incorrect length {{len(theta)}}, expected {{expected_len}}.")

    # Initialize and fill matrices using element-by-element assignments
{code_A}

{code_B}

{code_C}

{code_D}

    state_names = {repr([s.name for s in self.state_vars_ordered])}
    shock_names = {repr(self.shock_names)}

    # Dimension checks (optional but good)
    n_state = len(state_names)
    n_shocks = len(shock_names)
    expected_d_shape = (n_state, n_shocks)
    if A.shape != (n_state, n_state) or B.shape != (n_state, n_state) or \\
       C.shape != (n_state, n_state):
        raise AssertionError("Internal error: A, B, or C shape mismatch.")
    if not (n_shocks == 0 and D.size == 0) and D.shape != expected_d_shape:
        raise AssertionError(f"D shape mismatch: {{D.shape}} vs {{expected_d_shape}}")

    return A, B, C, D, state_names, shock_names

# Example Usage Block within the generated file:
if __name__ == '__main__':
    print(f"Example Usage for {function_name}:")
    # Parameter order expected matches the .mod file 'parameters' block
    parameter_order_comment = '# Order: ' + ', '.join(ordered_params_from_mod)
    print(parameter_order_comment)

    example_theta_values = [
"""
        # Example theta values code
        example_theta_code = []
        test_params_dict = self.last_param_values if hasattr(self, 'last_param_values') and self.last_param_values else {}
        if not test_params_dict: print(f"Warn: Using defaults (0.5) for {function_name} example.")
        for i, p_name in enumerate(ordered_params_from_mod):
            val = test_params_dict.get(p_name, 0.5)
            example_theta_code.append(f"        {val}, # {i}: {p_name}")
        file_content += "\n".join(example_theta_code) + "\n"

        file_content += f"""\
    ]
    example_theta = np.array(example_theta_values)

    if len(example_theta) == {len(ordered_params_from_mod)}:
        try:
            A, B, C, D, states, shocks = {function_name}(example_theta)
            print("\\n--- Matrices Calculated by Example Usage ---")
            print(f"Model: '{os.path.basename(self.mod_file_path)}'")
            print("State Names:", states)
            print(f"A shape: {{A.shape}}, B shape: {{B.shape}}")
            print(f"C shape: {{C.shape}}, D shape: {{D.shape}}")
        except ValueError as e: print(f"Error calling generated function: {{e}}")
        except Exception as e: print(f"An error occurred: {{e}}"); import traceback; traceback.print_exc()
    else: print(f"Error: Example theta length mismatch.")

"""
        # 5. Write the Python File
        try:
            # Add timestamp for tracking generation time
            import datetime
            file_content = f"# Generated: {datetime.datetime.now().isoformat()}\n" + file_content
            dir_name = os.path.dirname(filename)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f: f.write(file_content)
            print(f"Successfully generated function file: {filename}")
        except Exception as e: print(f"Error writing function file {filename}: {e}")

    def _save_intermediate_file(self, file_path, lines, equations=None, equations_title="Equations"):
        if not file_path: return
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(f"--- {os.path.basename(file_path)} ---\n\n"); f.write("\n".join(lines))
                if equations:
                     f.write(f"\n\n{equations_title} ({len(equations)}):\n")
                     for i, eq in enumerate(equations):
                         f.write(f"  {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
            print(f"Intermediate results saved to {file_path}")
        except Exception as e: print(f"Warning: Could not save intermediate file {file_path}. Error: {e}")

    def _save_final_matrices(self, file_path, A, B, C, D):
        matrix_data = {'A': A, 'B': B, 'C': C, 'D': D,'state_names': [s.name for s in self.state_vars_ordered],'shock_names': self.shock_names,'param_names': self.param_names}
        dir_name = os.path.dirname(file_path);
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        try:
            with open(file_path, "wb") as f: pickle.dump(matrix_data, f); print(f"Matrices saved to {file_path}")
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w", encoding='utf-8') as f:
                 f.write("State Names:\n" + str(matrix_data['state_names']) + "\n\n"); f.write("Shock Names:\n" + str(matrix_data['shock_names']) + "\n\n")
                 np.set_printoptions(linewidth=200, precision=4, suppress=True)
                 f.write("A Matrix:\n" + np.array2string(A) + "\n\n"); f.write("B Matrix:\n" + np.array2string(B) + "\n\n")
                 f.write("C Matrix:\n" + np.array2string(C) + "\n\n"); f.write("D Matrix:\n" + np.array2string(D) + "\n\n")
            print(f"Human-readable matrices saved to {txt_path}")
        except Exception as e: print(f"Warning: Could not save matrices file {file_path}. Error: {e}")

    def process_model(self, param_dict_values, output_dir_intermediate=None, output_dir_final=None, generate_function=True):
        # Define file paths
        base_name = os.path.splitext(os.path.basename(self.mod_file_path))[0]
        fpaths_inter = {i: None for i in range(6)}; final_matrices_pkl = None; function_py = None
        if output_dir_intermediate:
            os.makedirs(output_dir_intermediate, exist_ok=True)
            inter_names = ["timing", "static_elim", "static_sub", "aux_handling", "state_def", "final_eqs"]
            fpaths_inter = {i: os.path.join(output_dir_intermediate, f"{i+1}_{base_name}_{name}.txt") for i, name in enumerate(inter_names)}
        if output_dir_final:
             os.makedirs(output_dir_final, exist_ok=True)
             final_matrices_pkl = os.path.join(output_dir_final, f"{base_name}_matrices.pkl")
             if generate_function: function_py = os.path.join(output_dir_final, f"{base_name}_matrices.py")
        # Run pipeline
        try:
            self._analyze_variable_timing(file_path=fpaths_inter[0])
            self._identify_and_eliminate_static(file_path=fpaths_inter[1])
            self._substitute_static_vars(file_path=fpaths_inter[2])
            self._handle_aux_vars(file_path=fpaths_inter[3]) # Creates aux vars and definitions
            self._define_state_vector(file_path=fpaths_inter[4]) # Defines order including aux vars
            self._build_final_equations(file_path=fpaths_inter[5]) # Combines substituted + definitions
            A, B, C, D, state_names, shock_names = self.get_numerical_ABCD(param_dict_values, file_path=final_matrices_pkl)
            if generate_function and function_py: self.generate_matrix_function_file(filename=function_py)
            print("\n--- Model Processing Successful ---")
            return A, B, C, D, state_names, shock_names
        except Exception as e: print("\n--- ERROR during model processing ---"); import traceback; traceback.print_exc(); return None

# Example Usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    os.chdir(script_dir)
    mod_file = "qpm_model.dyn"
    output_dir_inter = "model_files_intermediate_final"
    output_dir_final = "model_files_numerical_final"
    if not os.path.exists(mod_file):
        print(f"Creating '{mod_file}' for testing.")
        with open(mod_file, "w") as f: f.write("""
var L_GDP_GAP DLA_CPI RS RR_GAP RES_L_GDP_GAP RES_DLA_CPI RES_RS;
varexo SHK_L_GDP_GAP SHK_DLA_CPI SHK_RS;
parameters b1 b4 a1 a2 g1 g2 g3 rho_DLA_CPI rho_L_GDP_GAP rho_rs rho_rs2;
model;
L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*RR_GAP(+1) + RES_L_GDP_GAP;
DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS;
RR_GAP = RS - DLA_CPI(+1);
RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;
RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI;
RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) + SHK_RS;
end;""")
    parameter_values = { 'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25, 'rho_DLA_CPI': 0.75, 'rho_L_GDP_GAP': 0.75, 'rho_rs': 0.75, 'rho_rs2': 0.01 }
    parser = DynareParser(mod_file)
    result = parser.process_model(parameter_values, output_dir_intermediate=output_dir_inter, output_dir_final=output_dir_final, generate_function=True)
    if result:
        A, B, C, D, state_names, shock_names = result
        print("\n--- Results from process_model ---"); print("States:", state_names); print(f"A:{A.shape} B:{B.shape} C:{C.shape} D:{D.shape}")
        function_file = os.path.join(output_dir_final, f"{os.path.splitext(mod_file)[0]}_matrices.py")
        if os.path.exists(function_file):
            print("\n--- Testing Generated Function ---"); abs_function_file = os.path.abspath(function_file); module_name = os.path.splitext(os.path.basename(function_file))[0]
            spec = importlib.util.spec_from_file_location(module_name, abs_function_file);
            if spec is None: print(f"Error: Could not load spec for {module_name}")
            else:
                mod_matrices = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod_matrices)
                try:
                    A_f, B_f, C_f, D_f, states_f, shocks_f = mod_matrices.get_abcd_matrices(parameter_values)
                    assert np.allclose(A, A_f, atol=1e-8), "A mismatch"; assert np.allclose(B, B_f, atol=1e-8), "B mismatch"
                    assert np.allclose(C, C_f, atol=1e-8), "C mismatch"; assert np.allclose(D, D_f, atol=1e-8), "D mismatch"
                    assert state_names == states_f, "State mismatch"; assert shock_names == shocks_f, "Shock mismatch"
                    print("Generated function tested successfully.")
                except Exception as test_e: print(f"Error testing generated function: {test_e}")
    else: print("\nModel processing failed.")