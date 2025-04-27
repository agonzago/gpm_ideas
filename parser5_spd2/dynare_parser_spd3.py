# -*- coding: utf-8 -*-
"""
dynare_parser_spd4.py # Renamed for clarity
A simplified Dynare-style parser implementing standard static reduction
and final classification/ordering for block-structured solving.
Corrected dependency detection logic.
No semicolons are used to terminate lines.
"""

import os
import re
import sympy
import numpy as np
import pickle
import collections
import importlib.util
import datetime
import sys # Added for sys.exit

# Add imports at the top of the file if not already present
import numpy as np
import scipy.linalg # For lu_factor, lu_solve

# --- time_shift_expression function (Keep as before) ---
def time_shift_expression(expr, shift, parser_symbols, var_names_set):
    """
    Shift the time index of variables (x_p1, x_m2, etc.) in 'expr' by 'shift' steps.
    Does not use semicolons.
    """
    if shift == 0:
        return expr

    subs_dict = {}
    # Ensure parser_symbols is accessible or passed correctly if needed globally
    # In the class context, self.symbols is used

    for atom in expr.free_symbols:
        nm = atom.name
        base = None
        k = 0
        is_var = False

        # Regex matching remains the same
        m_aux_lead = re.match(r'(aux_\w+_lead)(\d+)$', nm)
        m_aux_lag  = re.match(r'(aux_\w+_lag)(\d+)$', nm)
        m_lead     = re.match(r'(\w+)_p(\d+)$', nm)
        m_lag      = re.match(r'(\w+)_m(\d+)$', nm)

        if m_aux_lead:
            base = m_aux_lead.group(1)
            k = int(m_aux_lead.group(2))
            is_var = True
        elif m_aux_lag:
            base = m_aux_lag.group(1)
            k = -int(m_aux_lag.group(2))
            is_var = True
        elif m_lead:
            vb = m_lead.group(1)
            # Use the provided var_names_set
            if vb in var_names_set and not vb.lower().startswith('aux_'):
                base = vb
                k = int(m_lead.group(2))
                is_var = True
        elif m_lag:
            vb = m_lag.group(1)
            # Use the provided var_names_set
            if vb in var_names_set and not vb.lower().startswith('aux_'):
                base = vb
                k = -int(m_lag.group(2))
                is_var = True
        # Check base variable case *after* specific lead/lag/aux checks
        elif nm in var_names_set: # Check if the atom itself is a base variable
             # Avoid misinterpreting aux base names if they coincide with var names
             is_aux_base = nm.startswith('aux_') and ('_lag' in nm or '_lead' in nm)
             if not is_aux_base:
                 base = nm
                 k = 0
                 is_var = True


        if is_var:
            new_k = k + shift
            new_name = "" # Initialize new_name

            # Determine base name correctly (handle aux prefix)
            mo = re.match(r'aux_(\w+)_(?:lead|lag)', base)
            current_base_name = mo.group(1) if mo else base

            # Determine prefix for new name based on original base type
            prf = 'aux_' if base.startswith('aux_') else ''

            # Construct new name based on new time index
            if new_k == 0:
                # If shifted to time t, use the core variable name
                new_name = current_base_name
            elif new_k > 0:
                # Positive shift -> lead term
                new_name = f'{prf}{current_base_name}_p{new_k}'
            else: # new_k < 0
                # Negative shift -> lag term
                new_name = f'{prf}{current_base_name}_m{abs(new_k)}'

            # Ensure the new symbol exists in the central dictionary
            if new_name not in parser_symbols:
                # print(f"DEBUG: Creating shifted symbol: {new_name}") # Debug print
                parser_symbols[new_name] = sympy.Symbol(new_name)
            # Add substitution rule to the dictionary
            subs_dict[atom] = parser_symbols[new_name]

    # Apply all substitutions at once
    return expr.xreplace(subs_dict)


class DynareParser:
    """
    Parses a Dynare-style .mod file, performs model reduction,
    classifies final dynamic variables [Backward, Mixed, Forward], orders state vector,
    and prepares for block-structured solving using SDA. V4 with refined logic.
    """

    # --- __init__ and basic parsing methods (keep from previous) ---
    def __init__(self, mod_file_path):
        self.mod_file_path = mod_file_path; self.param_names = []; self.var_names = []
        self.var_names_set = set(); self.shock_names = []; self.equations_str = []
        self.symbols = {}; self.sympy_equations_original = []
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        self.static_subs = {}; self.equations_after_static_elim = []
        self.equations_after_static_sub = []; self.aux_lead_vars = {}
        self.aux_lag_vars = {}; self.aux_var_definitions = []
        self.equations_after_aux_sub = []; self.final_dynamic_var_names = []
        self.state_vars_ordered = []; self.state_var_map = {}
        self.final_equations_for_jacobian = []; self.last_param_values = {}
        self.var_dependencies = {} # NEW: Store dependency info here
        self._parse_mod_file(); self.var_names_set = set(self.var_names)
        self._create_initial_sympy_symbols(); self._parse_equations_to_sympy()
        print(f"Parser initialized. Vars:{len(self.var_names)}, Params:{len(self.param_names)}, Shocks:{len(self.shock_names)}")

    def _parse_mod_file(self):
        if not os.path.isfile(self.mod_file_path): raise FileNotFoundError(f"{self.mod_file_path}")
        try:
            with open(self.mod_file_path, 'r', encoding='utf-8') as f: content = f.read()
        except Exception as e: raise IOError(f"Reading {self.mod_file_path}: {e}")
        content = re.sub(r'//.*', '', content); content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        vpat = re.compile(r'var\s+(.*?);', re.I | re.DOTALL); vxpat = re.compile(r'varexo\s+(.*?);', re.I | re.DOTALL)
        ppat = re.compile(r'parameters\s+(.*?);', re.I | re.DOTALL); mpat = re.compile(r'model\s*;\s*(.*?)\s*end\s*;', re.I | re.DOTALL)
        vm = vpat.search(content); vxm = vxpat.search(content); pm = ppat.search(content); mm = mpat.search(content)
        if vm: self.var_names = [x.strip().rstrip(',') for x in vm.group(1).split() if x.strip()]
        if vxm: self.shock_names = [x.strip().rstrip(',') for x in vxm.group(1).split() if x.strip()]
        if pm: self.param_names = [x.strip().rstrip(',') for x in pm.group(1).split() if x.strip()]
        if mm: self.equations_str = [e.strip() for e in mm.group(1).split(';') if e.strip()]
        else: raise ValueError("Model block not found.")

    def _create_initial_sympy_symbols(self):
        for name in self.var_names + self.param_names + self.shock_names:
            if name not in self.symbols: self.symbols[name] = sympy.Symbol(name)

    def _replace_dynare_timing(self, eqstr):
        pat = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([+\-])\s*(\d+)\s*\)')
        output_str = eqstr; replacements = []; needed_symbols = set()
        for match in pat.finditer(eqstr):
            s, e = match.span(); vn, sign, num = match.groups()
            if vn in self.var_names_set:
                k = int(num); new_name = f"{vn}_p{k}" if sign == '+' else f"{vn}_m{k}"
                replacements.append((s, e, new_name)); needed_symbols.add(new_name)
        for s, e, rp in sorted(replacements, key=lambda x: x[0], reverse=True): output_str = output_str[:s] + rp + output_str[e:]
        for sn in needed_symbols:
            if sn not in self.symbols: self.symbols[sn] = sympy.Symbol(sn)
        return output_str

    def _parse_equations_to_sympy(self):
        self.sympy_equations_original = []
        for eq_str in self.equations_str:
            if not eq_str: continue
            processed_eq_str = self._replace_dynare_timing(eq_str)
            try:
                if '=' in processed_eq_str:
                    lhs_s, rhs_s = processed_eq_str.split('=', 1)
                    lhs = sympy.parse_expr(lhs_s.strip(), local_dict=self.symbols, evaluate=False)
                    rhs = sympy.parse_expr(rhs_s.strip(), local_dict=self.symbols, evaluate=False)
                    self.sympy_equations_original.append(sympy.Eq(lhs - rhs, 0))
                else:
                    expr = sympy.parse_expr(processed_eq_str, local_dict=self.symbols, evaluate=False)
                    self.sympy_equations_original.append(sympy.Eq(expr, 0))
            except Exception as e: raise ValueError(f"Parsing equation '{eq_str}': {e}") from e

    def _analyze_variable_timing(self, max_k=10): # Keep as is for initial info
        variable_symbols = {self.symbols[v] for v in self.var_names if v in self.symbols}
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        for eq in self.sympy_equations_original:
            free_symbols_in_eq = eq.lhs.free_symbols
            for var_sym in variable_symbols:
                var_name = var_sym.name
                if var_sym in free_symbols_in_eq: self.var_timing_info[var_name]['appears_current'] = True
                for k in range(1, max_k + 1):
                    lk, mk = f"{var_name}_p{k}", f"{var_name}_m{k}"
                    if lk in self.symbols and self.symbols[lk] in free_symbols_in_eq: self.var_timing_info[var_name]['max_lead'] = max(self.var_timing_info[var_name]['max_lead'], k)
                    if mk in self.symbols and self.symbols[mk] in free_symbols_in_eq: self.var_timing_info[var_name]['min_lag'] = min(self.var_timing_info[var_name]['min_lag'], -k)

    # --- Static Elimination, Substitution, Aux Vars (Keep corrected V3/V4) ---
    # Use the _identify_and_eliminate_static_vars from response #6
    def _identify_and_eliminate_static_vars(self):
        """
        Identifies variables that are candidates for static elimination (never appear
        with leads/lags globally) and attempts to solve for them iteratively using
        equations where they appear only contemporaneously. Populates self.static_subs
        and self.equations_after_static_elim. Implements Dynare-like reduction logic.
        """
        print("\n--- Stage 2: Identifying and Eliminating Static Variables ---")
        if not self.sympy_equations_original:
            print("Warning: No original equations to process.")
            self.equations_after_static_elim = []
            return

        all_var_symbols = {self.symbols[v] for v in self.var_names if v in self.symbols}
        if not all_var_symbols:
            print("Warning: No variable symbols found.")
            self.equations_after_static_elim = list(self.sympy_equations_original)
            return

        # --- Step 1: Global Dynamic Check ---
        print("Performing global check for dynamic variables...")
        dynamic_var_symbols = set()
        max_lead_lag_check = 15 # Arbitrary depth, adjust if needed

        for eq in self.sympy_equations_original:
            atoms_in_eq = eq.lhs.free_symbols
            for var_sym in all_var_symbols:
                if var_sym in dynamic_var_symbols: continue # Already classified

                var_name = var_sym.name
                # Check for leads/lags of this variable in this equation
                for k in range(1, max_lead_lag_check + 1):
                    lead_k_name = f"{var_name}_p{k}"
                    lag_k_name = f"{var_name}_m{k}"
                    if (lead_k_name in self.symbols and self.symbols[lead_k_name] in atoms_in_eq) or \
                       (lag_k_name in self.symbols and self.symbols[lag_k_name] in atoms_in_eq):
                        dynamic_var_symbols.add(var_sym)
                        # print(f"  '{var_name}' marked dynamic due to '{lead_k_name or lag_k_name}' in Eq.") # Debug
                        break # Stop checking k for this var in this eq

        # --- Step 2: Identify Candidate Statics ---
        candidate_static_symbols = all_var_symbols - dynamic_var_symbols
        if not candidate_static_symbols:
            print("No candidate static variables found (all variables appear dynamic somewhere).")
            self.equations_after_static_elim = list(self.sympy_equations_original)
            self.static_subs = {}
            return
        print(f"Candidate static variables: {[s.name for s in candidate_static_symbols]}")

        # --- Step 3: Iterative Solving ---
        equations_pool = list(enumerate(self.sympy_equations_original)) # Keep original index
        self.static_subs = {} # Stores {static_symbol: solved_expression}
        solved_static_symbols = set()
        used_equations_indices = set() # Track indices of equations used for solving

        iteration = 0
        max_iterations = len(candidate_static_symbols) + 2 # Safety break slightly larger than needed
        changed_overall = True # Track if any progress is made across all passes in an iteration

        print("Attempting iterative elimination of candidate static variables...")
        while changed_overall and iteration < max_iterations:
            iteration += 1
            changed_overall = False
            if verbose_debug := False: print(f"  Starting Iteration {iteration}...") # Changed to False for cleaner output

            made_progress_this_pass = True
            while made_progress_this_pass: # Inner loop for multiple passes per iteration
                made_progress_this_pass = False
                
                # Track indices solved in *this specific pass* to avoid immediate reuse
                solved_indices_this_pass = set()

                for i, eq in equations_pool:
                    if i in used_equations_indices or i in solved_indices_this_pass:
                        continue # Skip already used equations

                    # Substitute known static variables found in previous iterations/passes
                    try:
                        current_eq_lhs = eq.lhs.xreplace(self.static_subs)
                        current_eq_free_symbols = current_eq_lhs.free_symbols
                    except Exception as e:
                        print(f"    Warning: Substitution failed for Eq index {i}. Skipping. Error: {e}")
                        continue # Skip this equation if substitution fails

                    # Identify candidate static vars present ONLY AT TIME T in THIS equation
                    potential_static_here_t_only = set()
                    for cand_sym in candidate_static_symbols:
                        if cand_sym in current_eq_free_symbols:
                            # Check if ANY lead/lag of this candidate exists in THIS substituted eq
                            cand_name = cand_sym.name
                            has_lead_lag_in_eq = False
                            for k in range(1, max_lead_lag_check + 1):
                                lead_k_name = f"{cand_name}_p{k}"
                                lag_k_name = f"{cand_name}_m{k}"
                                if (lead_k_name in self.symbols and self.symbols[lead_k_name] in current_eq_free_symbols) or \
                                   (lag_k_name in self.symbols and self.symbols[lag_k_name] in current_eq_free_symbols):
                                    has_lead_lag_in_eq = True
                                    break
                            if not has_lead_lag_in_eq:
                                potential_static_here_t_only.add(cand_sym)

                    # Find which of these are not yet solved
                    unsolved_target_candidates = potential_static_here_t_only - solved_static_symbols

                    # If exactly one unsolved candidate static variable remains (at time t only)
                    if len(unsolved_target_candidates) == 1:
                        target_static_var = list(unsolved_target_candidates)[0]
                        if verbose_debug: print(f"    Attempting to solve Eq {i+1} for candidate '{target_static_var.name}'")

                        try:
                            # Solve the substituted equation LHS = 0 for the target
                            solutions = sympy.solve(current_eq_lhs, target_static_var)

                            if len(solutions) == 1:
                                solved_expr = solutions[0]
                                # Final check: ensure solution is not self-referential
                                if target_static_var not in solved_expr.free_symbols:
                                    print(f"      Solved: {target_static_var.name} = {sympy.sstr(solved_expr, full_prec=False)}")
                                    self.static_subs[target_static_var] = solved_expr
                                    solved_static_symbols.add(target_static_var)
                                    used_equations_indices.add(i) # Mark globally used
                                    solved_indices_this_pass.add(i) # Mark used in this pass
                                    changed_overall = True      # Progress in outer loop
                                    made_progress_this_pass = True # Progress in inner loop
                                    # Don't process this eq again this iteration
                                else:
                                     if verbose_debug: print(f"      Solve failed (self-ref): '{target_static_var.name}' from Eq {i+1}")
                            else:
                                 if verbose_debug: print(f"      Solve failed ({len(solutions)} sols): '{target_static_var.name}' from Eq {i+1}")
                        except NotImplementedError:
                             if verbose_debug: print(f"      Solve failed (NotImplemented): '{target_static_var.name}' from Eq {i+1}")
                        except Exception as e:
                             print(f"      Solve failed (Error: {e}): '{target_static_var.name}' from Eq {i+1}")
                # End of loop through equations for this pass
            # End of inner while loop (passes)

            if not changed_overall and iteration > 0: # Check outer loop progress
                 # print(f"  No static variables solved in Iteration {iteration}.") # Reduce verbosity
                 break # Exit outer loop if no progress in a full iteration

        # --- Step 4: Final Equation List ---
        # Filter the *original* list based on indices NOT used for solving
        self.equations_after_static_elim = [
            eq for i, eq in enumerate(self.sympy_equations_original) if i not in used_equations_indices
        ]

        if iteration >= max_iterations:
             print(f"Warning: Static elimination stopped after {max_iterations} iterations.")

        print(f"Static elimination finished. Solved for {len(self.static_subs)} static variables:")
        for var, expr in self.static_subs.items():
            print(f"  {var.name} = {sympy.sstr(expr, full_prec=False)}")
        print(f"{len(self.equations_after_static_elim)} equations remain after elimination.")

    # Use _substitute_static_vars from response #6
    def _substitute_static_vars(self):
        """
        Substitutes the solved static variables (and their leads/lags)
        into the remaining equations. Populates `self.equations_after_static_sub`.
        Internal method.
        """
        print("\n--- Stage 3: Substituting Static Variables ---")
        if not self.static_subs:
            print("No static substitutions to perform.")
            # If no static vars were eliminated, the remaining equations are the input
            self.equations_after_static_sub = list(self.equations_after_static_elim)
            return

        # Build the full substitution dictionary including leads and lags
        full_subs_dict = {}
        max_lead_lag_shift = 10 # How far to create shifted substitutions
        print("Building substitution dictionary for static variables and their leads/lags...")

        for static_var_sym, solved_expr in self.static_subs.items():
            static_var_name = static_var_sym.name
            # Add substitution for the current time variable
            full_subs_dict[static_var_sym] = solved_expr
            # print(f"  Sub rule (t=0): {static_var_name} -> {solved_expr}") # Debug

            # Create substitutions for leads (+k)
            for k in range(1, max_lead_lag_shift + 1):
                lead_k_name = f"{static_var_name}_p{k}"
                # Check if this lead symbol actually exists (was used in original model)
                if lead_k_name in self.symbols:
                    lead_sym = self.symbols[lead_k_name]
                    # Time-shift the solved expression by +k
                    shifted_expr = time_shift_expression(solved_expr, k, self.symbols, self.var_names_set)
                    full_subs_dict[lead_sym] = shifted_expr
                    # print(f"  Sub rule (t=+{k}): {lead_k_name} -> {shifted_expr}") # Debug

            # Create substitutions for lags (-k)
            for k in range(1, max_lead_lag_shift + 1):
                lag_k_name = f"{static_var_name}_m{k}"
                # Check if this lag symbol actually exists
                if lag_k_name in self.symbols:
                    lag_sym = self.symbols[lag_k_name]
                    # Time-shift the solved expression by -k
                    shifted_expr = time_shift_expression(solved_expr, -k, self.symbols, self.var_names_set)
                    full_subs_dict[lag_sym] = shifted_expr
                    # print(f"  Sub rule (t=-{k}): {lag_k_name} -> {shifted_expr}") # Debug

        # Apply substitutions to the remaining equations
        self.equations_after_static_sub = []
        print(f"Applying {len(full_subs_dict)} substitution rules to {len(self.equations_after_static_elim)} remaining equations...")
        for i, eq in enumerate(self.equations_after_static_elim):
            try:
                # Substitute using the full dictionary
                substituted_lhs = eq.lhs.xreplace(full_subs_dict)
                # Attempt to simplify the resulting expression
                # Use try-except as simplify can sometimes fail or be very slow
                try:
                    simplified_lhs = sympy.simplify(substituted_lhs)
                except Exception as simplify_error:
                    print(f"  Warning: Simplification failed for eq {i+1}. Using unsimplified. Error: {simplify_error}")
                    simplified_lhs = substituted_lhs

                new_eq = sympy.Eq(simplified_lhs, 0)
                self.equations_after_static_sub.append(new_eq)
                # print(f"  Eq {i+1} substituted: {sympy.sstr(new_eq.lhs, full_prec=False)} = 0") # Verbose
            except Exception as sub_error:
                print(f"Error substituting in equation {i+1}: {eq}")
                print(f"Substitution error: {sub_error}")
                # Decide whether to raise error or skip the equation
                raise ValueError(f"Failed to substitute static vars in eq {i+1}") from sub_error

        print(f"Static substitution complete. {len(self.equations_after_static_sub)} equations remain.")

    # Use _handle_aux_vars from response #6
    def _handle_aux_vars(self, file_path=None):
        """
        Identifies leads > +1 and lags < -1 in the post-static-substitution
        equations. Creates auxiliary variables and equations to reduce the system
        to first order (t-1, t, t+1). Populates `self.aux_lead_vars`,
        `self.aux_lag_vars`, `self.aux_var_definitions`, and
        `self.equations_after_aux_sub`. Internal method.
        Optionally saves details to file_path.
        """
        print("\n--- Stage 4: Handling Long Leads (>+1) and Lags (<-1) ---")
        self.aux_lead_vars = {} # name -> symbol
        self.aux_lag_vars = {}  # name -> symbol
        self.aux_var_definitions = [] # List of sympy.Eq defining aux vars
        # Start with equations after static substitution
        current_equations = list(self.equations_after_static_sub)
        subs_long_leads_lags = {} # Combined substitutions for long leads/lags
        output_lines = ["--- Stage 4 Log ---"] # For optional file saving

        # Identify base dynamic variable names (excluding static ones already removed)
        static_var_names = set(s.name for s in self.static_subs.keys())
        # Use self.var_names (original list) minus static ones
        dynamic_base_var_names = [v for v in self.var_names if v not in static_var_names]
        output_lines.append(f"Base dynamic variables considered: {dynamic_base_var_names}")

        max_lead_lag_check = 15 # Max lead/lag to check for

        # --- 1. Identify and Define Auxiliary Lead Variables (for leads > +1) ---
        leads_to_replace = collections.defaultdict(int) # base_var_name -> max_lead_k
        print("Scanning for leads > +1...")
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                match = re.match(r"(\w+)_p(\d+)", atom.name)
                if match:
                    base, k_str = match.groups()
                    k = int(k_str)
                    # Check if it's a dynamic variable and lead is > 1
                    if k > 1 and base in dynamic_base_var_names:
                        leads_to_replace[base] = max(leads_to_replace[base], k)

        if leads_to_replace:
            output_lines.append("\nCreating auxiliary LEAD variables:")
            print("Creating auxiliary LEAD variables...")
            # Sort for deterministic order
            for var_name in sorted(leads_to_replace.keys()):
                max_lead = leads_to_replace[var_name]
                output_lines.append(f"  Variable '{var_name}' needs aux leads up to k={max_lead-1} (to replace leads up to +{max_lead})")
                # Create aux vars aux_lead1, aux_lead2, ..., aux_lead(max_lead - 1)
                for k in range(1, max_lead): # Need k up to max_lead-1
                    aux_name = f"aux_{var_name}_lead{k}"
                    output_lines.append(f"    Processing aux lead: {aux_name}")

                    # Create symbol for the aux variable itself if it doesn't exist
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    self.aux_lead_vars[aux_name] = self.symbols[aux_name]

                    # --- Define the auxiliary equation ---
                    # aux_lead_k(t) = E[aux_lead_{k-1}(t+1)] or E[var(t+1)] for k=1
                    if k == 1:
                        # Definition: aux_VAR_lead1 = VAR_p1
                        rhs_base_name = f"{var_name}_p1"
                    else:
                        # Definition: aux_VAR_leadK = aux_VAR_lead(K-1)_p1
                        rhs_base_name = f"aux_{var_name}_lead{k-1}_p1"

                    # Ensure the RHS symbol exists
                    if rhs_base_name not in self.symbols:
                         self.symbols[rhs_base_name] = sympy.Symbol(rhs_base_name)
                    rhs_sym = self.symbols[rhs_base_name]

                    # Create the equation: aux_name - rhs_sym = 0
                    new_eq = sympy.Eq(self.symbols[aux_name] - rhs_sym, 0)

                    # Add definition only if it's unique
                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)
                        def_line = f"      Definition: {self.symbols[aux_name]} = {rhs_sym}"
                        print(def_line)
                        output_lines.append(def_line)

                    # --- Create the substitution rule ---
                    # Rule: Replace original_lead Var_p(k+1) with aux_lead_k_p1
                    original_lead_name = f"{var_name}_p{k+1}" # Lead we want to replace
                    replacement_lead_name = f"{aux_name}_p1" # Lead of the k-th aux var

                    # Only add rule if the original lead actually exists/was used
                    if original_lead_name in self.symbols:
                        # Ensure the replacement symbol (lead of aux var) exists
                        if replacement_lead_name not in self.symbols:
                            self.symbols[replacement_lead_name] = sympy.Symbol(replacement_lead_name)
                        # Store the substitution rule: original_symbol -> replacement_symbol
                        subs_long_leads_lags[self.symbols[original_lead_name]] = self.symbols[replacement_lead_name]
                        sub_line = f"      Substitution Rule: {original_lead_name} -> {replacement_lead_name}"
                        print(sub_line)
                        output_lines.append(sub_line)
        else:
             print("No leads > +1 found.")
             output_lines.append("No leads > +1 found.")


        # --- 2. Identify and Define Auxiliary Lag Variables (for lags < -1) ---
        lags_to_replace = collections.defaultdict(int) # base_var_name -> max_lag_k (|k|)
        print("\nScanning for lags < -1...")
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                match = re.match(r"(\w+)_m(\d+)", atom.name)
                if match:
                    base, k_str = match.groups()
                    k = int(k_str)
                    # Check if it's a dynamic variable and lag is > 1 (magnitude)
                    if k > 1 and base in dynamic_base_var_names:
                        lags_to_replace[base] = max(lags_to_replace[base], k)

        if lags_to_replace:
            output_lines.append("\nCreating auxiliary LAG variables:")
            print("Creating auxiliary LAG variables...")
            for var_name in sorted(lags_to_replace.keys()):
                max_lag = lags_to_replace[var_name]
                output_lines.append(f"  Variable '{var_name}' needs aux lags up to k={max_lag-1} (to replace lags up to -{max_lag})")
                # Create aux vars aux_lag1, ..., aux_lag(max_lag - 1)
                for k in range(1, max_lag):
                    aux_name = f"aux_{var_name}_lag{k}"
                    output_lines.append(f"    Processing aux lag: {aux_name}")

                    # Create symbol for the aux variable itself
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    self.aux_lag_vars[aux_name] = self.symbols[aux_name]

                    # --- Define the auxiliary equation ---
                    # aux_lag_k(t) = aux_lag_{k-1}(t-1) or var(t-1) for k=1
                    if k == 1:
                        # Definition: aux_VAR_lag1 = VAR_m1
                        rhs_base_name = f"{var_name}_m1"
                    else:
                        # Definition: aux_VAR_lagK = aux_VAR_lag(K-1)_m1
                        rhs_base_name = f"aux_{var_name}_lag{k-1}_m1"

                    # Ensure RHS symbol exists
                    if rhs_base_name not in self.symbols:
                        self.symbols[rhs_base_name] = sympy.Symbol(rhs_base_name)
                    rhs_sym = self.symbols[rhs_base_name]

                    # Create the equation: aux_name - rhs_sym = 0
                    new_eq = sympy.Eq(self.symbols[aux_name] - rhs_sym, 0)

                    # Add definition if unique
                    if new_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(new_eq)
                        def_line = f"      Definition: {self.symbols[aux_name]} = {rhs_sym}"
                        print(def_line)
                        output_lines.append(def_line)

                    # --- Create the substitution rule ---
                    # Rule: Replace original_lag Var_m(k+1) with aux_lag_k_m1
                    original_lag_name = f"{var_name}_m{k+1}" # Lag we want to replace
                    replacement_lag_name = f"{aux_name}_m1" # Lag of the k-th aux var

                    # Only add rule if the original lag exists
                    if original_lag_name in self.symbols:
                        # Ensure replacement symbol exists
                        if replacement_lag_name not in self.symbols:
                            self.symbols[replacement_lag_name] = sympy.Symbol(replacement_lag_name)
                        # Store substitution rule
                        subs_long_leads_lags[self.symbols[original_lag_name]] = self.symbols[replacement_lag_name]
                        sub_line = f"      Substitution Rule: {original_lag_name} -> {replacement_lag_name}"
                        print(sub_line)
                        output_lines.append(sub_line)
        else:
             print("No lags < -1 found.")
             output_lines.append("No lags < -1 found.")


        # --- 3. Apply Substitutions ---
        self.equations_after_aux_sub = [] # Reset the list
        output_lines.append(f"\nApplying {len(subs_long_leads_lags)} long lead/lag substitutions to {len(current_equations)} equations...")
        print(f"\nApplying {len(subs_long_leads_lags)} long lead/lag substitutions...")
        for i, eq in enumerate(current_equations):
            try:
                subbed_lhs = eq.lhs.xreplace(subs_long_leads_lags)
                # Optional simplification (can be slow or fail)
                try:
                    simplified_lhs = sympy.simplify(subbed_lhs)
                except Exception:
                    simplified_lhs = subbed_lhs # Fallback

                subbed_eq = sympy.Eq(simplified_lhs, 0)
                self.equations_after_aux_sub.append(subbed_eq)
                # eq_line = f"  Eq {i+1} substituted: {sympy.sstr(subbed_eq.lhs, full_prec=False)} = 0"
                # output_lines.append(eq_line) # Keep log concise for console
            except Exception as sub_error:
                print(f"Error substituting long leads/lags in equation {i+1}: {eq}")
                print(f"Substitution error: {sub_error}")
                raise ValueError(f"Failed to substitute aux vars in eq {i+1}") from sub_error

        # --- 4. Finalize and Log ---
        # Update the list of *all* dynamic variables (original non-static + all aux)
        # Use the keys from the aux dictionaries
        self.final_dynamic_var_names = dynamic_base_var_names \
                                     + list(self.aux_lead_vars.keys()) \
                                     + list(self.aux_lag_vars.keys())

        final_line1 = f"Aux handling complete. System reduced to first order."
        final_line2 = f"  Total final dynamic variables (incl. aux): {len(self.final_dynamic_var_names)}"
        final_line3 = f"  Original equations after substitution: {len(self.equations_after_aux_sub)}"
        final_line4 = f"  Auxiliary variable definition equations: {len(self.aux_var_definitions)}"
        print(final_line1); print(final_line2); print(final_line3); print(final_line4)
        output_lines.extend(["", final_line1, final_line2, final_line3, final_line4])
        # print(f"  Final dynamic variable names: {self.final_dynamic_var_names}") # Optional verbose print

        if file_path:
            # Use the internal helper to save the log
            self._save_intermediate_file(file_path, output_lines,
                                        self.equations_after_aux_sub + self.aux_var_definitions,
                                        "Equations After Aux Handling (Substituted + Definitions)")

    # --- Refined Dependency Builder ---
    def _build_dependency_info(self):
        """
        Analyzes final equations to build dependency information for each variable.
        Stores results in self.var_dependencies. V6 Refined.
        """
        print("Building dependency information (V6 Refined)...")
        # Reset dependencies
        self.var_dependencies = {
            var_name: {'has_lag_dep': False, 'has_lead_dep': False}
            for var_name in self.final_dynamic_var_names
        }

        if not self.final_dynamic_var_names or not self.final_equations_for_jacobian:
            print("Error: Cannot build dependencies - final variables/equations missing.")
            return # Keep dependencies empty

        # --- Iterate through each target variable ---
        for target_var_name in self.final_dynamic_var_names:
            target_var_sym = self.symbols.get(target_var_name)
            if not target_var_sym: continue

            # Scan ALL equations where the target variable appears (at time t)
            for eq in self.final_equations_for_jacobian:
                try:
                    free_symbols_in_eq = eq.lhs.free_symbols
                except AttributeError: continue

                if target_var_sym in free_symbols_in_eq:
                    # Check other symbols in THIS equation for leads/lags
                    for sym in free_symbols_in_eq:
                        if sym == target_var_sym: continue # Skip self

                        sym_name = sym.name
                        # Use regex to check suffix _m<digits> or _p<digits>
                        if re.search(r'_m\d+$', sym_name):
                            self.var_dependencies[target_var_name]['has_lag_dep'] = True
                        elif re.search(r'_p\d+$', sym_name):
                            self.var_dependencies[target_var_name]['has_lead_dep'] = True

                    # Optimization: Stop checking equations for this var if both found
                    if self.var_dependencies[target_var_name]['has_lag_dep'] and \
                       self.var_dependencies[target_var_name]['has_lead_dep']:
                        break

        print("Dependency information built.")

    
    def _build_final_equations(self):
        """
        Combines substituted original equations and aux definitions.
        Sets self.final_equations_for_jacobian.
        """
        print("\n--- Stage 6: Building Final Equation System for Jacobians ---")
        self.final_equations_for_jacobian = list(self.equations_after_aux_sub) + list(self.aux_var_definitions)
        n_eqs = len(self.final_equations_for_jacobian)
        n_vars = len(self.final_dynamic_var_names) # Check against total vars defined
        print(f"Final system built with {n_eqs} equations and {n_vars} dynamic variables.")
        # Note: Will check n_eqs == n_states later after state ordering is confirmed


    def _define_state_vector(self, file_path=None):
        """
        Classifies dynamic variables based on the time indices of symbols
        within their corresponding final equation, following the user's logic.
        Orders variables as [Backward, Mixed, Forward].
        Assumes a one-to-one mapping between final_dynamic_var_names and
        final_equations_for_jacobian for non-auxiliary variables.
        """
        print("\n--- Stage 5: Classifying and Ordering State Variables (User Logic) ---")
        output_lines = ["Classifying and Ordering State Variables (User Logic):"]

        # --- Input Checks ---
        if not hasattr(self, 'final_dynamic_var_names') or not self.final_dynamic_var_names:
            raise ValueError("Stage 4 must run first (final_dynamic_var_names missing).")
        if not hasattr(self, 'final_equations_for_jacobian') or not self.final_equations_for_jacobian:
            raise ValueError("Stage 6 must run first (final_equations_for_jacobian missing).")
        if len(self.final_dynamic_var_names) != len(self.final_equations_for_jacobian):
            raise ValueError(f"Fatal Error: Mismatch variables ({len(self.final_dynamic_var_names)}) "
                             f"vs equations ({len(self.final_equations_for_jacobian)}).")

        backward_vars, mixed_vars, forward_vars = [], [], []
        classified_vars_names = set()

        # --- Step 1: Pre-classify Aux Vars ---
        print("Pre-classifying Aux variables by name...")
        output_lines.append("\nPre-classifying Aux variables by name:")
        final_var_symbols_map = {name: self.symbols[name] for name in self.final_dynamic_var_names if name in self.symbols}

        for var_name, var_sym in final_var_symbols_map.items():
            # Use the name convention directly
            if var_name.startswith("aux_") and "_lag" in var_name:
                 line = f"  {var_name} -> BACKWARD (Aux Lag)"
                 print(line); output_lines.append(line)
                 backward_vars.append(var_sym)
                 classified_vars_names.add(var_name)
            elif var_name.startswith("aux_") and "_lead" in var_name:
                 line = f"  {var_name} -> FORWARD (Aux Lead)"
                 print(line); output_lines.append(line)
                 forward_vars.append(var_sym)
                 classified_vars_names.add(var_name)

        # --- Step 2: Classify remaining variables based on associated equation ---
        print("Classifying remaining (non-aux) variables...")
        output_lines.append("\nClassifying remaining (non-aux) variables:")

        # Assume k-th equation corresponds to k-th variable *in the final list*
        # This requires careful construction in _build_final_equations if order matters.
        # A safer mapping might be needed for robustness in complex cases.
        for i, var_name in enumerate(self.final_dynamic_var_names):
            if var_name in classified_vars_names:
                continue # Already classified as aux

            if var_name not in self.symbols:
                 print(f"Warning: Variable '{var_name}' not found in symbols dictionary. Skipping.")
                 continue

            var_sym = self.symbols[var_name]
            # Assume the i-th equation corresponds to the i-th variable name
            # This relies on the order preserved/constructed in _build_final_equations
            if i >= len(self.final_equations_for_jacobian):
                 print(f"Error: No equation found for variable index {i} ('{var_name}').")
                 # Defaulting to backward might hide errors. Consider raising an error.
                 backward_vars.append(var_sym) # Default fallback
                 continue

            equation = self.final_equations_for_jacobian[i]
            equation_symbols = equation.lhs.free_symbols # Assuming LHS=0 form

            has_lag = False
            has_lead = False

            # Check for *any* lead or lag symbol within this equation
            for sym in equation_symbols:
                if not isinstance(sym, sympy.Symbol): continue
                # Check name for lag/lead indicators (_m<k> or _p<k>)
                if sym.name.endswith(tuple(f"_m{k}" for k in range(1, 15))):
                    has_lag = True
                elif sym.name.endswith(tuple(f"_p{k}" for k in range(1, 15))):
                    has_lead = True
                if has_lag and has_lead: # Optimization
                    break

            # --- Determine classification based on presence of *any* lead/lag ---
            classification = "Unknown"
            if has_lag and has_lead:
                classification = "MIXED"
                mixed_vars.append(var_sym)
            elif has_lag: # Only lag (or current) terms found
                classification = "BACKWARD"
                backward_vars.append(var_sym)
            elif has_lead: # Only lead (or current) terms found
                classification = "FORWARD"
                forward_vars.append(var_sym)
            else: # Neither lag nor lead found (only current terms)
                classification = "BACKWARD (Static-like)"
                backward_vars.append(var_sym)

            line = f"  {var_name} (Eq {i+1}) -> {classification} (Eq has any lag: {has_lag}, Eq has any lead: {has_lead})"
            print(line)
            output_lines.append(line)
            classified_vars_names.add(var_name) # Mark as classified

        # --- Step 3 & 4: Ordering, Checks, Map ---
        # Sort symbols within each category alphabetically by name
        backward_vars.sort(key=lambda s: s.name)
        mixed_vars.sort(key=lambda s: s.name)
        forward_vars.sort(key=lambda s: s.name)

        # Combine ordered lists: Backward + Mixed + Forward (adjust order if needed)
        self.state_vars_ordered = backward_vars + mixed_vars + forward_vars
        self.state_var_map = {s: i for i, s in enumerate(self.state_vars_ordered)}

        # --- Final Checks ---
        expected_vars_set = set(self.final_dynamic_var_names)
        final_classified_names = {s.name for s in self.state_vars_ordered}

        if len(self.state_vars_ordered) != len(expected_vars_set):
             print("\nError: Mismatch between expected final variables and classified state variables!")
             # ... (Error reporting as before) ...
             raise RuntimeError("State variable classification size mismatch.")
        elif final_classified_names != expected_vars_set:
             print("\nWarning: Classified state variables differ from expected final dynamic variables!")
             # ... (Warning reporting as before) ...

        print("\nFinal variable classification and ordering complete.")
        line = f"  Order: Backward ({len(backward_vars)}), Mixed ({len(mixed_vars)}), Forward ({len(forward_vars)})"
        print(line); output_lines.append(line)
        line = f"  Total State Variables: {len(self.state_vars_ordered)}"
        print(line); output_lines.append(line)
        line = f"  Final Ordered State Vector Names: {[s.name for s in self.state_vars_ordered]}"
        print(line); output_lines.append(line)

        if file_path:
            self._save_intermediate_file(file_path, output_lines)

        return self.state_vars_ordered
    # --- get_numerical_ABCD (Keep as before) ---
    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        # ... (Keep implementation from response #6, it uses self.state_vars_ordered) ...
        """
        Calculates numerical A, B, C, D matrices from the final, ordered equations
        and state vector, substituting parameter values.
        D = -d(eq)/d(shock). Respects the order in `self.state_vars_ordered`.
        Optionally saves matrices to a file.

        Args:
            param_dict_values (dict): Dictionary mapping parameter names (str) to values (float).
            file_path (str, optional): Path to save numerical matrices (.pkl + .txt).

        Returns:
            tuple: (A_num, B_num, C_num, D_num, state_names, shock_names)
                   Matrices are numpy arrays. state_names is an ordered list.
        """
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing incomplete: Final equations or ordered state vector not available.")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)
        n_eqs = len(self.final_equations_for_jacobian)

        if n_eqs != n_state:
             print(f"Warning: Non-square system detected ({n_eqs} equations, {n_state} states). Jacobians might not be square.")

        # Store params for potential later use (e.g., function generation)
        self.last_param_values = param_dict_values

        # Create substitution dictionary for parameters
        param_subs = {}
        for p_name, p_value in param_dict_values.items():
            if p_name in self.symbols:
                param_subs[self.symbols[p_name]] = p_value
            else:
                # This shouldn't happen if parameters are parsed correctly
                print(f"Warning: Parameter '{p_name}' from input dict not found in model symbols.")

        # --- Create symbolic vectors based on ORDERED state variables ---
        print("Building symbolic vectors (t+1, t, t-1) using ordered states...")
        state_vec_t = sympy.Matrix(self.state_vars_ordered) # ORDERED vector at time t

        state_vec_tp1_list = []
        state_vec_tm1_list = []
        # Create lead/lag versions corresponding to the ordered state vector
        for state_sym in self.state_vars_ordered:
            lead_name = f"{state_sym.name}_p1"
            lag_name = f"{state_sym.name}_m1"

            # Ensure lead symbol exists (should have been created if needed)
            if lead_name not in self.symbols:
                self.symbols[lead_name] = sympy.Symbol(lead_name)
            state_vec_tp1_list.append(self.symbols[lead_name])

            # Ensure lag symbol exists
            if lag_name not in self.symbols:
                self.symbols[lag_name] = sympy.Symbol(lag_name)
            state_vec_tm1_list.append(self.symbols[lag_name])

        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list) # ORDERED vector at t+1
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list) # ORDERED vector at t-1

        # Create shock vector
        shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
        shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None

        # Create equation vector (LHS of final equations)
        # The order of equations here determines the row order of the Jacobians.
        # As noted before, we haven't explicitly reordered equations yet.
        eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])
        # --- End Create symbolic vectors ---

        print("Calculating symbolic Jacobians...")
        try:
            # Calculate Jacobians w.r.t the ORDERED state vectors
            A_sym = eq_vec.jacobian(state_vec_tp1) # d(Eq)/d(State_{t+1})
            B_sym = eq_vec.jacobian(state_vec_t)   # d(Eq)/d(State_t)
            C_sym = eq_vec.jacobian(state_vec_tm1) # d(Eq)/d(State_{t-1})

            # Calculate D matrix Jacobian: D = -d(eq)/d(shock)
            if shock_vec and n_shocks > 0:
                D_sym = -eq_vec.jacobian(shock_vec)
            else:
                # Create a zero matrix of correct shape if no shocks
                D_sym = sympy.zeros(n_eqs, n_shocks)
        except Exception as e:
            print(f"Error during symbolic Jacobian calculation: {e}")
            # Add more debug info if needed, e.g., print shapes
            print(f"  Equation vector shape: {eq_vec.shape}")
            print(f"  State(t+1) vector shape: {state_vec_tp1.shape}")
            print(f"  State(t)   vector shape: {state_vec_t.shape}")
            print(f"  State(t-1) vector shape: {state_vec_tm1.shape}")
            print(f"  Shock vector shape: {shock_vec.shape if shock_vec else 'None'}")
            raise

        print("Substituting parameter values into Jacobians...")
        try:
            # --- Numerical Substitution using evalf for potential speedup ---
            A_num = np.array(A_sym.evalf(subs=param_subs), dtype=float)
            B_num = np.array(B_sym.evalf(subs=param_subs), dtype=float)
            C_num = np.array(C_sym.evalf(subs=param_subs), dtype=float)
            # D matrix substitution
            if n_shocks > 0:
                D_num = np.array(D_sym.evalf(subs=param_subs), dtype=float)
            else:
                D_num = np.zeros((n_eqs, 0), dtype=float) # Correct shape for zero shocks

        except Exception as e:
            print(f"Error during numerical substitution (evalf): {e}")
            # Provide context for debugging
            print("Attempting substitution on symbolic matrices:")
            # print("A_sym:", A_sym) # Can be very large
            # print("B_sym:", B_sym)
            # print("C_sym:", C_sym)
            # print("D_sym:", D_sym)
            raise

        # --- Final checks and Output ---
        # Expected shape based on number of states (vars) and shocks
        expected_state_shape = (n_eqs, n_state)
        expected_shock_shape = (n_eqs, n_shocks)

        # Verify shapes after substitution
        if A_num.shape != expected_state_shape or \
           B_num.shape != expected_state_shape or \
           C_num.shape != expected_state_shape or \
           D_num.shape != expected_shock_shape:
            print("ERROR: Final numerical matrix dimension mismatch!")
            print(f" A shape: {A_num.shape}, Expected: {expected_state_shape}")
            print(f" B shape: {B_num.shape}, Expected: {expected_state_shape}")
            print(f" C shape: {C_num.shape}, Expected: {expected_state_shape}")
            print(f" D shape: {D_num.shape}, Expected: {expected_shock_shape}")
            raise ValueError("Numerical matrix dimension mismatch after substitution.")

        print("Numerical matrices A, B, C, D calculated successfully.")
        state_names_ordered = [s.name for s in self.state_vars_ordered]

        # --- Save (Optional) ---
        if file_path:
            # Use the internal helper method to save
            # Make sure _save_final_matrices signature matches
            self._save_final_matrices(file_path, A_num, B_num, C_num, D_num, state_names_ordered, self.shock_names, self.param_names)


        return A_num, B_num, C_num, D_num, state_names_ordered, self.shock_names

    # --- _generate_matrix_assignments_code_helper (Keep as before) ---
    def _generate_matrix_assignments_code_helper(self, matrix_sym, matrix_name):
        # ... (Keep implementation from response #6) ...
        """
        Generates Python code lines for element-wise matrix assignments.
        Internal helper function. Uses standard Python formatting.
        """
        try:
            rows, cols = matrix_sym.shape
        except Exception as e:
            print(f"Error getting shape for matrix {matrix_name}. Matrix: {matrix_sym}")
            raise ValueError(f"Could not get shape for symbolic matrix {matrix_name}") from e

        indent = "    " # Standard 4 spaces
        # Initialize the matrix creation line correctly indented
        code_lines = [f"{indent}{matrix_name} = np.zeros(({rows}, {cols}), dtype=float)"]
        assignments = []

        # Iterate using actual dimensions from the symbolic matrix
        for r in range(rows):
            for c in range(cols):
                try:
                    element = matrix_sym[r, c]
                except IndexError:
                    # Defensive check
                    print(f"Internal Error: Index ({r},{c}) out of bounds for {matrix_name} shape {matrix_sym.shape}")
                    continue

                # Check for structural zero before converting to string
                if element != 0 and element is not sympy.S.Zero:
                    try:
                        # Get standard Python string representation
                        expr_str = sympy.sstr(element, full_prec=False)
                        # Add assignment line with standard indentation
                        assignments.append(f"{indent}{matrix_name}[{r}, {c}] = {expr_str}")
                    except Exception as str_e:
                        print(f"Warning: String conversion failed for {matrix_name}[{r},{c}]: {str_e}")
                        assignments.append(f"{indent}# Error generating code for {matrix_name}[{r},{c}] = {element}")

        # Add the assignment block only if there are non-zero elements found
        if assignments:
            code_lines.append(f"{indent}# Fill {matrix_name} non-zero elements")
            code_lines.extend(assignments)

        # Return the complete code block as a single string
        return "\n".join(code_lines)
        
    # --- generate_matrix_function_file (Keep as before, with typo fix) ---
    def generate_matrix_function_file(self, filename="jacobian_matrices.py"):
        # ... (Keep implementation from response #6, including IndexError fix) ...
        """
        Generates a Python file containing a function `jacobian_matrices(theta)`
        that computes the numerical A, B, C, D matrices for given parameters `theta`.
        The generated function uses the ordered state vector.
        No semicolons are used. Includes corrected IndexError handling.

        Args:
            filename (str): The name of the Python file to generate.
        """
        function_name = "jacobian_matrices" # Hardcoded function name
        print(f"\n--- Generating Python Function File: {filename} ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing must complete (final equations, ordered states) before generating function.")

        n_state = len(self.state_vars_ordered)
        n_shocks = len(self.shock_names)
        n_eqs = len(self.final_equations_for_jacobian) # Number of rows

        # --- 1. Recalculate Symbolic Jacobians (using ordered states) ---
        try:
            state_vec_t = sympy.Matrix(self.state_vars_ordered)
            state_vec_tp1_list = [self.symbols.get(f"{s.name}_p1", sympy.Symbol(f"{s.name}_p1")) for s in state_vec_t]
            state_vec_tm1_list = [self.symbols.get(f"{s.name}_m1", sympy.Symbol(f"{s.name}_m1")) for s in state_vec_t]
            state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
            state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)
            shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
            shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None
            eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])

            print("Calculating symbolic Jacobians for function generation...")
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            D_sym = -eq_vec.jacobian(shock_vec) if shock_vec and n_shocks > 0 else sympy.zeros(n_eqs, n_shocks)
            print("Symbolic Jacobians calculated.")
        except Exception as e:
            print(f"Error calculating symbolic Jacobians for function generation: {e}")
            raise

        # --- 2. Prepare Parameter Info ---
        ordered_params_from_mod = self.param_names
        param_symbols_in_matrices = set().union(*(mat.free_symbols for mat in [A_sym, B_sym, C_sym, D_sym] if mat is not None))
        used_param_symbols = {s for s in param_symbols_in_matrices if s.name in self.param_names}
        used_params_ordered = [p for p in ordered_params_from_mod if p in {s.name for s in used_param_symbols}]
        param_indices = {p_name: i for i, p_name in enumerate(ordered_params_from_mod)}

        # --- 3. Generate Python code strings for matrix assignments ---
        print("Generating Python code strings for matrices...")
        try:
            code_A = self._generate_matrix_assignments_code_helper(A_sym, 'A')
            code_B = self._generate_matrix_assignments_code_helper(B_sym, 'B')
            code_C = self._generate_matrix_assignments_code_helper(C_sym, 'C')
            code_D = self._generate_matrix_assignments_code_helper(D_sym, 'D')
            print("Python code strings generated.")
        except Exception as e:
            print(f"Error generating matrix assignment code: {e}")
            raise

        # --- 4. Assemble File Content ---
        file_lines = []
        file_lines.append(f"# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'")
        file_lines.append(f"# Generated: {datetime.datetime.now().isoformat()}")
        file_lines.append("# DO NOT EDIT MANUALLY - Re-run the parser instead")
        file_lines.append("")
        file_lines.append("import numpy as np")
        file_lines.append("from math import exp, log, log10, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, erf, sqrt, pi # Common functions")
        file_lines.append("")
        file_lines.append(f"def {function_name}(theta):")
        indent = "    "
        file_lines.append(f"{indent}# theta: A list or numpy array of parameter values in SPECIFIC order.")
        file_lines.append(f"{indent}# Expected order: {ordered_params_from_mod}")
        file_lines.append("")
        file_lines.append(f"{indent}# --- Parameter Unpacking ---")
        file_lines.append(f"{indent}expected_len = {len(ordered_params_from_mod)}")
        file_lines.append(f"{indent}if len(theta) != expected_len:")
        # Corrected f-string for ValueError
        file_lines.append(f"{indent}    raise ValueError(f'Expected {{expected_len}} parameters, but got {{len(theta)}} parameters.')")
        file_lines.append("")
        file_lines.append(f"{indent}try:")
        for p_name in used_params_ordered:
            idx = param_indices[p_name]
            file_lines.append(f"{indent}    {p_name} = theta[{idx}] # Index {idx}")
        if not used_params_ordered:
            file_lines.append(f"{indent}    pass")
        file_lines.append(f"{indent}except IndexError:") # Corrected block
        file_lines.append(f"{indent}    # Error message includes expected length") # Corrected block
        file_lines.append(f"{indent}    raise IndexError(f'theta has incorrect length. Expected {{expected_len}}.')") # Corrected block
        file_lines.append("")
        file_lines.append(f"{indent}# --- Matrix Calculations ---")
        file_lines.append(code_A)
        file_lines.append("")
        file_lines.append(code_B)
        file_lines.append("")
        file_lines.append(code_C)
        file_lines.append("")
        file_lines.append(code_D)
        file_lines.append("")
        file_lines.append(f"{indent}# --- Return results ---")
        state_names_repr = repr([s.name for s in self.state_vars_ordered])
        shock_names_repr = repr(self.shock_names)
        file_lines.append(f"{indent}state_names = {state_names_repr}")
        file_lines.append(f"{indent}shock_names = {shock_names_repr}")
        file_lines.append("")
        file_lines.append(f"{indent}return A, B, C, D, state_names, shock_names")

        final_file_content = "\n".join(file_lines)
        # --- End Assemble File Content ---

        # --- 5. Write File ---
        try:
            dir_name = os.path.dirname(filename)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(final_file_content)
            print(f"Successfully generated function file: {filename}")
        except Exception as e:
            print(f"Error writing function file {filename}: {e}")

    # --- process_model (Corrected Order) ---
    def process_model(self, param_dict_values_or_list, output_dir_intermediate=None,
                    output_dir_final=None, generate_function=True):
        """
        Runs the full parsing and matrix generation pipeline, including corrected
        static reduction and final variable classification/ordering. Returns matrices.
        Uses standard Python formatting. Corrected pipeline order.
        """
        # Parameter handling (keep as before)
        param_dict_values = {}
        if isinstance(param_dict_values_or_list, (list, tuple, np.ndarray)): # ... (rest of param handling)
             if len(param_dict_values_or_list) != len(self.param_names): raise ValueError(f"Input param length mismatch")
             param_dict_values = {name: val for name, val in zip(self.param_names, param_dict_values_or_list)}
        elif isinstance(param_dict_values_or_list, dict): # ... (rest of param handling)
            param_dict_values = param_dict_values_or_list; # ... (checks) ...
        else: raise TypeError("param_dict_values_or_list must be dict, list, tuple, or numpy array.")

        # Define file paths (keep as before)
        base_name = os.path.splitext(os.path.basename(self.mod_file_path))[0]; fpaths_inter = {}; final_matrices_pkl = None; function_py = None
        if output_dir_intermediate: # ... (define intermediate paths) ...
             os.makedirs(output_dir_intermediate, exist_ok=True); inter_names = ["timing", "static_elim", "static_sub", "aux_handling", "final_eqs", "state_def_ordered"]
             fpaths_inter = {name: os.path.join(output_dir_intermediate, f"{i+1}_{base_name}_{name}.txt") for i, name in enumerate(inter_names)}
        if output_dir_final: # ... (define final paths) ...
             os.makedirs(output_dir_final, exist_ok=True); final_matrices_pkl = os.path.join(output_dir_final, f"{base_name}_matrices.pkl")
             if generate_function: function_py = os.path.join(output_dir_final, f"{base_name}_jacobian_matrices.py")


        # --- Run pipeline steps with CORRECTED ORDER ---
        try:
            print("\n--- Starting Model Processing Pipeline (Corrected Order) ---")
            # Stage 1: Timing Analysis
            self._analyze_variable_timing()
            self._save_intermediate_file(fpaths_inter.get("timing"), ["Stage 1: Timing Analysis Results", str(self.var_timing_info)], self.sympy_equations_original, "Original Equations")

            # Stage 2: Static Elimination
            self._identify_and_eliminate_static_vars()
            static_lines = [f"  {s.name} = {sympy.sstr(e, full_prec=False)}" for s, e in self.static_subs.items()]
            self._save_intermediate_file(fpaths_inter.get("static_elim"), ["Stage 2: Static Elimination", f"Solved {len(self.static_subs)} static vars:"] + static_lines, self.equations_after_static_elim, "Eqs After Elimination")

            # Stage 3: Static Substitution
            self._substitute_static_vars()
            self._save_intermediate_file(fpaths_inter.get("static_sub"), ["Stage 3: Static Substitution"], self.equations_after_static_sub, "Eqs After Substitution")

            # Stage 4: Auxiliary Variable Handling
            self._handle_aux_vars(file_path=fpaths_inter.get("aux_handling"))

            # --- ** CORRECTED ORDER ** ---
            # Stage 6 (was 5): Build Final Equations FIRST
            self._build_final_equations()
            self._save_intermediate_file(fpaths_inter.get("final_eqs"), ["Stage 6: Final Equation System"], self.final_equations_for_jacobian, "Final Equations for Jacobian")


            
            # Stage 5 (was 6): Define State Vector (Classification & Ordering) NEXT
            # This now uses the correctly populated self.final_equations_for_jacobian via _build_dependency_info
            self._define_state_vector()
            print(f"Final dynamic variables: {self.final_dynamic_var_names}")
            print(self.final_dynamic_var_names)
            print(f"state_var_orderd: {self.state_vars_ordered}")
            print(f"final_equations_for_jacobian: {self.final_equations_for_jacobian}")
            


            state_lines = ["Stage 5: State Vector Definition (Ordered)", f"Ordered State Vector ({len(self.state_vars_ordered)}): {[s.name for s in self.state_vars_ordered]}"]
            self._save_intermediate_file(fpaths_inter.get("state_def_ordered"), state_lines)


            # Stage 7: Calculate Numerical Matrices
            A, B, C, D, state_names, shock_names = self.get_numerical_ABCD(
                param_dict_values,
                file_path=final_matrices_pkl
            )

            # Stage 8: Generate Function File (Optional)
            if generate_function and function_py:
                self.generate_matrix_function_file(filename=function_py)

            print("\n--- Model Processing Successful ---")
            return A, B, C, D, state_names, shock_names

        except Exception as e:
            print(f"\n--- ERROR during model processing pipeline: {type(e).__name__}: {e} ---")
            import traceback
            traceback.print_exc()
            return None

    # --- Helper methods (_save_...) and Solvers (solve_spd_sf1, calculate_Q, compute_irf) ---
    # ... (Keep implementations from response #7) ...
    def _save_intermediate_file(self, file_path, lines, equations=None, equations_title="Equations"):
        """Saves intermediate processing steps to a text file."""
        if not file_path: return # Don't save if no path provided
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True) # Ensure dir exists

            with open(file_path, "w", encoding='utf-8') as f:
                f.write(f"--- {os.path.basename(file_path)} ---\n")
                f.write(f"--- Generated: {datetime.datetime.now().isoformat()} ---\n\n")
                f.write("\n".join(lines))
                if equations is not None: # Check if equations were provided
                    f.write(f"\n\n{equations_title} ({len(equations)}):\n")
                    if equations: # Check if the list is not empty
                         for i, eq in enumerate(equations):
                            # Handle potential non-Eq objects gracefully
                            if hasattr(eq, 'lhs'):
                                f.write(f"  {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                            else:
                                f.write(f"  {i+1}: [Non-Equation Object: {type(eq)}] {str(eq)}\n")
                    else:
                         f.write("  [No equations in this list]\n")
            # print(f"Intermediate results saved to {file_path}") # Keep console less verbose
        except Exception as e:
            print(f"Warning: Could not save intermediate file {file_path}. Error: {e}")

    # Corrected signature to match how it's called in get_numerical_ABCD
    def _save_final_matrices(self, file_path, A, B, C, D, state_names, shock_names, param_names):
        """Saves final numerical matrices and names to .pkl and .txt files."""
        if not file_path: return # Don't save if no path provided

        matrix_data = {
            'A': A, 'B': B, 'C': C, 'D': D,
            'state_names': state_names, # Assumed to be ordered
            'shock_names': shock_names,
            'param_names': param_names, # Original order for reference
            'timestamp': datetime.datetime.now().isoformat()
         }
        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name: os.makedirs(dir_name, exist_ok=True)

        try:
            # Save as pickle
            with open(file_path, "wb") as f:
                pickle.dump(matrix_data, f)
            print(f"Matrices saved to {file_path}")

            # Save as human-readable text
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write(f"# Numerical Matrices for {os.path.basename(self.mod_file_path)}\n")
                f.write(f"# Generated: {matrix_data['timestamp']}\n\n")
                f.write(f"State Names (Order: {len(state_names)}):\n" + str(state_names) + "\n\n")
                f.write(f"Shock Names ({len(shock_names)}):\n" + str(shock_names) + "\n\n")
                f.write(f"Parameter Names ({len(param_names)}):\n" + str(param_names) + "\n\n")

                # Configure numpy printing
                np.set_printoptions(linewidth=200, precision=6, suppress=True, threshold=np.inf)
                f.write(f"A Matrix ({A.shape}):\n" + np.array2string(A, separator=', ') + "\n\n")
                f.write(f"B Matrix ({B.shape}):\n" + np.array2string(B, separator=', ') + "\n\n")
                f.write(f"C Matrix ({C.shape}):\n" + np.array2string(C, separator=', ') + "\n\n")
                f.write(f"D Matrix ({D.shape}):\n")
                if D.size > 0: # Only print if not empty
                     f.write(np.array2string(D, separator=', ') + "\n\n")
                else:
                     f.write("[No shocks]\n\n")
            print(f"Human-readable matrices saved to {txt_path}")
        except Exception as e:
            print(f"Warning: Could not save matrices file {file_path} or {txt_path}. Error: {e}")

    def save_final_equations_to_txt(self, filename="final_equations.txt"):
        """Saves the final list of equations used for Jacobians to a text file."""
        print(f"\n--- Saving Final Equations for Jacobian to: {filename} ---")
        if not hasattr(self, 'final_equations_for_jacobian') or not self.final_equations_for_jacobian:
            print("Warning: Final equations list is empty. File not saved.")
            return
        try:
            # Ensure directory exists
            dir_name = os.path.dirname(filename)
            if dir_name: os.makedirs(dir_name, exist_ok=True)

            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"# Final System Equations Used for Jacobian Calculation ({len(self.final_equations_for_jacobian)} equations)\n")
                f.write(f"# Model: {os.path.basename(self.mod_file_path)}\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")

                if hasattr(self, 'state_vars_ordered') and self.state_vars_ordered:
                    f.write(f"# State Variables Order ({len(self.state_vars_ordered)}): {[s.name for s in self.state_vars_ordered]}\n\n")
                else:
                    f.write("# State variable order not determined at time of saving.\n\n")

                for i, eq in enumerate(self.final_equations_for_jacobian):
                    # Ensure it's a sympy equation and format nicely
                    if hasattr(eq, 'lhs'):
                         f.write(f"Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                    else:
                         f.write(f"Eq {i+1}: [Invalid Equation Object] {str(eq)}\n")

            print(f"Successfully saved final equations to {filename}")
        except Exception as e:
            print(f"Error writing final equations file {filename}: {e}")

    def solve_spd_sf1(self, A, B, C, initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
        """
        Solves A*P^2 + B*P + C = 0 using the Structure-Preserving Doubling
        Algorithm (SDA) based on the First Standard Form (SF1), corresponding
        to Algorithm 1/5 in Huber, Meyer-Gohde, Saecker (2024).

        Args:
            A (np.ndarray): Coefficient matrix for P^2.
            B (np.ndarray): Coefficient matrix for P.
            C (np.ndarray): Constant coefficient matrix.
            initial_guess (np.ndarray, optional): Initial guess P0 for P.
                                                  Defaults to zero matrix if None.
            tol (float, optional): Convergence tolerance for the norm of the update to X.
                                   Defaults to 1e-14.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            verbose (bool, optional): If True, prints iteration progress. Defaults to False.

        Returns:
            tuple: (P, converged) where:
                   P (np.ndarray): The computed solution matrix.
                   converged (bool): True if the algorithm converged within max_iter, False otherwise.
        """
        n = A.shape[0] # Assuming square matrices
        if A.shape != (n, n) or B.shape != (n, n) or C.shape != (n, n):
            raise ValueError("Input matrices A, B, C must be square and have the same dimension.")

        # --- Initialization (Algorithm 5) ---
        if initial_guess is None:
            P0 = np.zeros_like(A)
            guess_provided = False
        else:
            if initial_guess.shape != (n, n):
                 raise ValueError(f"Initial guess shape {initial_guess.shape} must match matrix dimension ({n},{n}).")
            P0 = initial_guess.copy()
            guess_provided = True

        if verbose:
            print(f"--- Starting SDA SF1 (Max Iter: {max_iter}, Tol: {tol}) ---")
            if guess_provided:
                print("  Using provided initial guess P0.")
            else:
                print("  Using zero matrix as initial guess P0.")

        # Form B_bar = B + A @ P0
        B_bar = B + A @ P0

        # Factorize B_bar: lu_piv contains LU factors and pivot info
        try:
            lu_piv_B_bar = scipy.linalg.lu_factor(B_bar, check_finite=False)
            # Simple singularity check (more robust needed for edge cases)
            if np.any(np.abs(np.diag(lu_piv_B_bar[0])) < np.finfo(float).eps * n):
                 print("Warning: Matrix B_bar = B + A@P0 appears singular or ill-conditioned during LU factorization.")
                 # Returning P0 and False as per Julia code's apparent behavior on LU failure
                 return P0, False
        except np.linalg.LinAlgError:
            print("Error: LU factorization failed for B_bar = B + A@P0 (likely singular).")
            return P0, False # Indicate failure

        # Compute initial E0, F0, X0, Y0
        try:
            E0 = -scipy.linalg.lu_solve(lu_piv_B_bar, C, check_finite=False)
            F0 = -scipy.linalg.lu_solve(lu_piv_B_bar, A, check_finite=False)
        except np.linalg.LinAlgError:
             print("Error: Solving initial E0/F0 failed after LU factorization (should not happen if LU succeeded).")
             return P0, False

        X0 = -E0 - P0
        Y0 = -F0

        # Initialize iteration variables
        X = X0.copy()
        Y = Y0.copy()
        E = E0.copy()
        F = F0.copy()

        # Preallocate temps (optional, minor speedup for large n)
        # temp1 = np.empty_like(X)
        # temp2 = np.empty_like(X)
        # temp3 = np.empty_like(X)
        II = np.identity(n)

        converged = False
        final_iter = max_iter

        # --- Iteration Loop ---
        if verbose:
            print("  Iter | Norm(X_update)")
            print("--------|--------------")

        for i in range(1, max_iter + 1):
            # --- Calculate necessary inverses implicitly ---
            # temp1 = I - Y @ X
            # temp2 = I - X @ Y
            try:
                mat_I_YX = II - Y @ X
                lu_piv_I_YX = scipy.linalg.lu_factor(mat_I_YX, check_finite=False)
                if np.any(np.abs(np.diag(lu_piv_I_YX[0])) < np.finfo(float).eps * n):
                     print(f"Warning: Matrix (I - Yk*Xk) appears singular at iteration {i}.")
                     final_iter = i; converged = False; break # Exit loop

                mat_I_XY = II - X @ Y
                lu_piv_I_XY = scipy.linalg.lu_factor(mat_I_XY, check_finite=False)
                if np.any(np.abs(np.diag(lu_piv_I_XY[0])) < np.finfo(float).eps * n):
                     print(f"Warning: Matrix (I - Xk*Yk) appears singular at iteration {i}.")
                     final_iter = i; converged = False; break # Exit loop

            except np.linalg.LinAlgError as e:
                print(f"Error: LU factorization failed during iteration {i}: {e}")
                final_iter = i; converged = False; break # Exit loop

            # --- Calculate Updates ---
            try:
                # E_update_term = solve(I - YX, E) -> temp3
                temp3 = scipy.linalg.lu_solve(lu_piv_I_YX, E, check_finite=False)
                E_new = E @ temp3

                # F_update_term = solve(I - XY, F) -> temp3
                temp3 = scipy.linalg.lu_solve(lu_piv_I_XY, F, check_finite=False)
                F_new = F @ temp3

                # X_update = F @ solve(I - XY, X @ E)
                temp_XE = X @ E
                temp_invFI_XE = scipy.linalg.lu_solve(lu_piv_I_XY, temp_XE, check_finite=False)
                X_update = F @ temp_invFI_XE

                # Y_update = E @ solve(I - YX, Y @ F)
                temp_YF = Y @ F
                temp_invEI_YF = scipy.linalg.lu_solve(lu_piv_I_YX, temp_YF, check_finite=False)
                Y_update = E @ temp_invEI_YF

            except np.linalg.LinAlgError as e:
                 print(f"Error: Solving system failed during update calculation at iteration {i}: {e}")
                 final_iter = i; converged = False; break

            # --- Check Convergence ---
            # Use norm of the update applied to X
            norm_X_update = np.linalg.norm(X_update)
            if verbose:
                 print(f"  {i:4d} | {norm_X_update:12.4e}")

            if norm_X_update < tol:
                converged = True
                final_iter = i
                break # Exit loop

            # --- Update for next iteration ---
            X += X_update
            Y += Y_update
            E = E_new # No need to copy if we assign directly
            F = F_new

        # --- Post-Loop ---
        if not converged and verbose:
            print(f"SDA SF1 did not converge within {max_iter} iterations. Final Norm(X_update): {norm_X_update:.4e}")
        elif converged and verbose:
            print(f"SDA SF1 converged in {final_iter} iterations.")

        # Final solution P = X_k + P0
        P = X + P0

        # Optional: Calculate final residual (for diagnostics)
        if verbose or not converged:
            try:
                Residual = A @ P @ P + B @ P + C
                residual_norm = np.linalg.norm(Residual)
                # Normalize relative to norm(C) or norm(B@P) if C is zero?
                # Simple norm for now.
                print(f"  Final Residual Norm ||A*P^2 + B*P + C|| = {residual_norm:.4e}")
            except Exception as e:
                print(f"  Could not compute final residual norm: {e}")


        return P, converged

    def calculate_Q(self, A, B, C, D, P):
        """
        Calculates the shock impact matrix Q = -(A*P + B)^(-1) * D.

        Requires the solution matrix P from solving the quadratic equation.

        Args:
            A (np.ndarray): Coefficient matrix for P^2 (or E_t[x_{t+1}]).
            B (np.ndarray): Coefficient matrix for P (or x_t).
            C (np.ndarray): Constant coefficient matrix (or related to x_{t-1}).
            D (np.ndarray): Coefficient matrix for shocks epsilon_t.
            P (np.ndarray): Solution matrix from AP^2 + BP + C = 0.

        Returns:
            np.ndarray or None: The calculated Q matrix, or None if calculation fails.
        """
        if P is None:
            print("Error: Cannot calculate Q without a valid solution matrix P.")
            return None
        if D is None or D.shape[1] == 0:
             print("No shocks defined (D matrix is empty/None). Q matrix is zero.")
             n_state = A.shape[0]
             n_shocks = 0
             return np.zeros((n_state, n_shocks)) # Return correctly shaped zero matrix

        n_state, n_shocks = D.shape
        if A.shape != (n_state, n_state) or B.shape != (n_state, n_state) or P.shape != (n_state, n_state):
             raise ValueError("Matrices A, B, P must be square and conformable with D.")

        # Calculate -(A*P + B)
        M = -(A @ P + B)

        # Solve M * Q = D for Q using LU factorization for stability
        try:
            lu_piv_M = scipy.linalg.lu_factor(M, check_finite=False)
            # Simple singularity check
            if np.any(np.abs(np.diag(lu_piv_M[0])) < np.finfo(float).eps * n_state):
                 print("Warning: Matrix -(A*P + B) appears singular during LU factorization for Q calculation.")
                 return None
            Q = scipy.linalg.lu_solve(lu_piv_M, D, check_finite=False)
            print("Matrix Q calculated successfully.")
            return Q
        except np.linalg.LinAlgError as e:
            print(f"Error: Solving for Q failed (matrix -(A*P + B) likely singular): {e}")
            return None
        except Exception as e:
             print(f"An unexpected error occurred during Q calculation: {e}")
             return None


    def compute_irf(self, P, Q, n_periods, shock_index=None, shock_vector=None, shock_size=1.0):
        """
        Computes Impulse Response Functions (IRFs) for the linear system
        x_t = P * x_{t-1} + Q * epsilon_t.

        Requires the state transition matrix P and the shock impact matrix Q.

        Args:
            P (np.ndarray): State transition matrix (n_state x n_state).
            Q (np.ndarray): Shock impact matrix (n_state x n_shocks).
            n_periods (int): Number of periods to simulate the IRF.
            shock_index (int, optional): Index of the single shock to simulate (0-based).
                                         If provided, shock_vector is ignored.
            shock_vector (np.ndarray, optional): A specific shock vector (n_shocks,)
                                                  for the initial impulse. Used if shock_index is None.
            shock_size (float, optional): Size of the initial shock (if using shock_index).
                                          Defaults to 1.0.

        Returns:
            tuple: (irf_array, state_names) where:
                   irf_array (np.ndarray): Array of IRFs (n_state x n_periods).
                   state_names (list): Ordered list of state variable names.
                   Returns (None, None) if inputs are invalid.
        """
        if P is None or Q is None:
            print("Error: Cannot compute IRFs without valid P and Q matrices.")
            return None, None
        if not self.state_vars_ordered:
             print("Error: State vector ordering not defined. Run preprocessing first.")
             return None, None

        n_state = P.shape[0]
        if Q.shape[0] != n_state:
             raise ValueError(f"Q matrix rows ({Q.shape[0]}) must match P matrix dimension ({n_state}).")
        n_shocks = Q.shape[1]

        # --- Determine initial shock epsilon_0 ---
        epsilon_0 = np.zeros(n_shocks)
        if shock_index is not None:
            if not 0 <= shock_index < n_shocks:
                raise ValueError(f"shock_index ({shock_index}) out of bounds for {n_shocks} shocks.")
            epsilon_0[shock_index] = shock_size
            print(f"Simulating IRF for shock index {shock_index} (size {shock_size}).")
        elif shock_vector is not None:
            if shock_vector.shape != (n_shocks,):
                 raise ValueError(f"shock_vector shape ({shock_vector.shape}) must be ({n_shocks},).")
            epsilon_0 = shock_vector.copy()
            print(f"Simulating IRF for provided shock vector.")
        elif n_shocks > 0:
             # Default to first shock if none specified and shocks exist
             print(f"Warning: No shock specified, defaulting to first shock (index 0, size {shock_size}).")
             epsilon_0[0] = shock_size
             shock_index = 0
        else:
             print("Warning: No shocks in the model (n_shocks=0). IRF will be all zeros.")
             # Return zeros immediately if no shocks
             irf_array = np.zeros((n_state, n_periods))
             state_names = [s.name for s in self.state_vars_ordered]
             return irf_array, state_names

        # --- Simulate IRF ---
        irf_array = np.zeros((n_state, n_periods))
        # Initial impact: x_0 = Q * epsilon_0
        x_current = Q @ epsilon_0
        if n_periods > 0:
             irf_array[:, 0] = x_current

        # Subsequent periods: x_t = P * x_{t-1}
        for t in range(1, n_periods):
            x_current = P @ x_current
            irf_array[:, t] = x_current

        state_names = [s.name for s in self.state_vars_ordered]
        print(f"IRF calculated for {n_periods} periods.")
        return irf_array, state_names

# ===========================================
# Example Usage Script (Updated)
# ===========================================
if __name__ == "__main__":

    # --- Setup (same as before) ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    os.chdir(script_dir)
    mod_file = "qpm_model.dyn"
    output_dir_inter = "model_files_intermediate_final"
    output_dir_final = "model_files_numerical_final"
    os.makedirs(output_dir_inter, exist_ok=True)
    os.makedirs(output_dir_final, exist_ok=True)
    print(f"Script directory: {script_dir}")
    print(f"Using mod file: {os.path.abspath(mod_file)}")
    print(f"Intermediate output: {os.path.abspath(output_dir_inter)}")
    print(f"Final output: {os.path.abspath(output_dir_final)}")

    # --- Define parameters DICT (same as before) ---
    parameter_values_dict = {
        'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7,
        'g2': 0.3, 'g3': 0.25, 'rho_DLA_CPI': 0.75,
        'rho_L_GDP_GAP': 0.75, 'rho_rs': 0.75, 'rho_rs2': 0.01
    }
    print(f"\nParameter values provided: {parameter_values_dict}")

    # --- Instantiate parser (same as before) ---
    try:
        parser = DynareParser(mod_file)
    except Exception as e:
        print(f"Error initializing parser: {type(e).__name__}: {e}")
        sys.exit(1)

    # --- Create theta list IN DECLARED ORDER (same as before) ---
    try:
        missing_params = [pname for pname in parser.param_names if pname not in parameter_values_dict]
        if missing_params:
            print(f"\nERROR: Input dict missing params: {missing_params}")
            sys.exit(1)
        parameter_theta = [parameter_values_dict[pname] for pname in parser.param_names]
        print(f"\nTheta vector created ({len(parameter_theta)} values)")
    except Exception as e:
        print(f"\nERROR creating theta list: {e}")
        sys.exit(1)

    # --- Process the model (same as before) ---
    result_matrices = parser.process_model(
        parameter_theta,
        output_dir_intermediate=output_dir_inter,
        output_dir_final=output_dir_final,
        generate_function=True
    )

    if not result_matrices:
        print("\nModel processing failed. Exiting.")
        sys.exit(1)

    A_num, B_num, C_num, D_num, state_names, shock_names = result_matrices
    print("\n--- Model Processed Successfully ---")
    print(f"State Names (Ordered): {state_names}")
    print(f"Shock Names: {shock_names}")
    print(f"Matrix Shapes: A:{A_num.shape}, B:{B_num.shape}, C:{C_num.shape}, D:{D_num.shape}")

    # --- Solve using SDA SF1 ---
    print("\n--- Solving using SDA SF1 ---")
    # Can provide an initial guess, e.g., from a previous solve or zero
    # initial_p_guess = np.zeros_like(A_num)
    initial_p_guess = None # Use zero default

    P_sol, converged = parser.solve_spd_sf1(A_num, B_num, C_num,
                                           initial_guess=initial_p_guess,
                                           tol=1e-12, # Tighter tolerance maybe
                                           max_iter=150,
                                           verbose=True)

    if converged:
        print("SDA SF1 converged to solution P.")
        # print("P matrix (first 5x5):\n", P_sol[:5, :5]) # Optional print

        # --- Calculate Q matrix ---
        print("\n--- Calculating Q matrix ---")
        Q_sol = parser.calculate_Q(A_num, B_num, C_num, D_num, P_sol)

        if Q_sol is not None:
            print(f"Q matrix calculated ({Q_sol.shape}).")
            # print("Q matrix (first 5 rows):\n", Q_sol[:5, :]) # Optional print

            # --- Compute IRFs ---
            print("\n--- Computing Impulse Responses ---")
            n_periods_irf = 40
            if shock_names:
                 shock_to_plot_index = 0 # Index of the shock in shock_names
                 shock_to_plot_name = shock_names[shock_to_plot_index]
                 print(f"Calculating IRF for shock: '{shock_to_plot_name}' (index {shock_to_plot_index})")

                 irf_array, irf_state_names = parser.compute_irf(P_sol, Q_sol, n_periods_irf,
                                                                shock_index=shock_to_plot_index,
                                                                shock_size=0.01) # Example shock size

                 if irf_array is not None:
                     print("IRF calculation successful.")
                     # --- Plotting (Optional) ---
                     try:
                         import matplotlib.pyplot as plt
                         print("Plotting IRFs for selected variables...")
                         vars_to_plot = ['L_GDP_GAP', 'DLA_CPI', 'RS'] # Example variables
                         plt.figure(figsize=(12, 8))
                         for var_name in vars_to_plot:
                              if var_name in irf_state_names:
                                   var_index = irf_state_names.index(var_name)
                                   plt.plot(range(n_periods_irf), irf_array[var_index, :], label=var_name, marker='o', markersize=3)
                              else:
                                   print(f"  Variable '{var_name}' not found in ordered state names for plotting.")

                         plt.title(f'Impulse Responses to a 1% {shock_to_plot_name} Shock')
                         plt.xlabel('Periods')
                         plt.ylabel('Deviation from Steady State')
                         plt.grid(True, linestyle='--', alpha=0.6)
                         plt.legend()
                         plt.axhline(0, color='black', linewidth=0.5)
                         plot_filename = os.path.join(output_dir_final, f"{os.path.splitext(mod_file)[0]}_irf_{shock_to_plot_name}.png")
                         plt.savefig(plot_filename)
                         print(f"IRF plot saved to {plot_filename}")
                         # plt.show() # Uncomment to display plot interactively
                         plt.close() # Close the plot window

                     except ImportError:
                          print("\nMatplotlib not found. Skipping IRF plotting.")
                     except Exception as plot_err:
                          print(f"An error occurred during plotting: {plot_err}")

            else:
                 print("No shocks defined in the model. Cannot compute IRFs.")

        else:
            print("Q matrix calculation failed. Cannot compute IRFs.")
    else:
        print("SDA SF1 did not converge. Cannot compute Q or IRFs.")

    # --- Test Generated Function (Optional, same as before) ---
    # ... (include the test code from previous response if desired) ...

    print("\n--- Parser Script Finished ---")