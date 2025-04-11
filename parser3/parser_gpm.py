# parser_gpm.py
# Complete implementation based on structured workflow V4
# Aiming for correct state classification and substitutions.

import re
import json
import os
import numpy as np
import pandas as pd
import sympy as sy
import sys

# Helper JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.generic,)): return obj.item()
        return json.JSONEncoder.default(self, obj)

class DynareParser:
    """
    Parses a Dynare .mod file, transforms it for Klein solution format
    (t+1 notation, states as lags), and generates output files.
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path; self.parameters = pd.Series(dtype=float)
        self.var_list = []; self.varexo_list = []; self.original_equations = []
        self.content = ""; self.param_names_declared = []
        # Attributes populated by parse()
        self.exo_process_info = {}; self.state_variables = []; self.control_variables = []
        self.all_variables = []; self.future_variables = []; self.auxiliary_variables = []
        self.auxiliary_equations = []; self.transformed_equations = []
        self.final_shock_to_process_var_map = {}; self.state_to_shock_map = {}
        self.shock_to_state_map = {}; self.zero_persistence_processes = []
        self.endogenous_states = set(); self.exo_with_shocks = set(); self.exo_without_shocks = set()

    # --- Stage 1: Basic Parsing ---
    def _read_and_preprocess(self):
        print("--- Reading & Preprocessing ---")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f: self.content = f.read()
            self.content = re.sub(r'//.*|%.*', '', self.content)
            self.content = re.sub(r'/\*.*?\*/', '', self.content, flags=re.DOTALL)
            self.content = re.sub(r'\s+', ' ', self.content).strip()
        except Exception as e: raise IOError(f"Error reading/preprocessing {self.file_path}: {e}")

    def _parse_declarations(self):
        print("--- Parsing Declarations ---")
        # Variables
        m_var = re.search(r'var\s+(.*?);', self.content, re.I | re.DOTALL)
        if m_var: self.var_list = [v for v in re.findall(r'\b([a-zA-Z_]\w*)\b', m_var.group(1)) if v]
        else: print("Warning: 'var' declaration not found.")
        # Shocks
        m_exo = re.search(r'varexo\s+(.*?);', self.content, re.I | re.DOTALL)
        if m_exo: self.varexo_list = [v for v in re.findall(r'\b([a-zA-Z_]\w*)\b', m_exo.group(1)) if v]
        else: print("Warning: 'varexo' declaration not found.")
        self.shocks = list(self.varexo_list)
        # Parameters
        m_param_decl = re.search(r'parameters\s+(.*?);', self.content, re.I | re.DOTALL)
        if m_param_decl: self.param_names_declared = [p for p in re.findall(r'\b([a-zA-Z_]\w*)\b', m_param_decl.group(1)) if p]
        else: print("Warning: 'parameters' declaration not found.")
        param_val_pattern = re.compile(r'\b([a-zA-Z_]\w*)\b\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*;')
        param_dict = {}
        for name, val_str in param_val_pattern.findall(self.content):
             if not self.param_names_declared or name in self.param_names_declared:
                 try: param_dict[name] = float(val_str)
                 except ValueError: print(f"Warning: Cannot parse value for param '{name}'.")
        for p in self.param_names_declared:
            if p not in param_dict: param_dict[p] = np.nan; print(f"Warning: Param '{p}' unassigned.")
        ordered_keys = self.param_names_declared if self.param_names_declared else sorted(param_dict.keys())
        self.parameters = pd.Series(param_dict).reindex(ordered_keys)
        print(f"  Found {len(self.var_list)} vars, {len(self.varexo_list)} shocks, {len(self.parameters)} params.")

    def _parse_model_block(self):
        print("--- Parsing Model Block ---")
        m_model = re.search(r'model\s*;(.*?)\s*end\s*;', self.content, re.I | re.DOTALL)
        if m_model: self.original_equations = [eq.strip() for eq in m_model.group(1).strip().split(';') if eq.strip()]
        else: raise ValueError("'model;'...'end;' block not found or empty.")
        print(f"  Found {len(self.original_equations)} original equations.")

    # --- Stage 2: Identify Exogenous Processes ---
    def _identify_exogenous_processes(self):
        """
        Identifies exogenous process variables via driving shocks. V4.
        Sets self.exo_process_info. Raises ValueError on multi-shock errors.
        Ensures pv is assigned before use.
        """
        print("\n--- Step 2: Identifying Exogenous Processes ---")
        # --- Input Checks ---
        if not hasattr(self, 'original_equations') or not self.original_equations:
             print("  Error: Original equations not available (self.original_equations).")
             self.exo_process_info = {}
             return
        if not hasattr(self, 'varexo_list'): # Shocks list
             print("  Warning: No exogenous shocks (varexo) declared.")
             self.varexo_list = [] # Ensure it exists as an empty list
        if not hasattr(self, 'parameters') or not isinstance(self.parameters, pd.Series):
             print("  Warning: Parameters not parsed correctly (should be Series).")
             if not hasattr(self, 'parameters'): self.parameters = pd.Series(dtype=float)

        self.exo_process_info = {} # Reset the dictionary for this run

        # --- Iterate through original equations ---
        for eq in self.original_equations: # Use the original list
            eq_clean = eq.strip()
            if not eq_clean or "=" not in eq_clean:
                continue # Skip empty or non-assignment lines

            # --- Safely Split Equation ---
            try:
                left, right = [s.strip() for s in eq_clean.split("=", 1)]
                # --- CRITICAL ASSIGNMENT of Process Variable ---
                process_variable = left # Assign Potential LHS variable (pv)
                # -------------------------------------------
            except ValueError:
                 print(f"  Warning: Skipping malformed equation line: {eq_clean}")
                 continue # Skip this equation entirely

            # --- Validate LHS ---
            if not process_variable:
                 print(f"  Warning: Skipping equation with empty LHS: {eq_clean}")
                 continue # Skip this equation

            # --- Find Shocks on RHS ---
            # Use word boundaries (\b) to ensure full match
            found_shocks = [
                s for s in self.varexo_list
                if re.search(r'\b' + re.escape(s) + r'\b', right)
            ]

            # --- Process based on Shocks Found ---
            if len(found_shocks) == 1:
                # === Exactly One Shock: Process This Equation ===
                driving_shock = found_shocks[0]

                # --- Check for Redefinition using the ASSIGNED process_variable ---
                # 'process_variable' is guaranteed to be defined here if this block is reached
                if process_variable in self.exo_process_info:
                    print(f"  Warning: Redefinition of exogenous process variable '{process_variable}'. Overwriting with equation: {eq_clean}")
                # -----------------------------------------------------------------

                # --- Analyze AR Structure & Persistence ---
                ar_params = {}
                is_zero_persistence = True
                max_ar_lag = 5

                for k in range(1, max_ar_lag + 1):
                    lag_pattern_re = rf'\b{re.escape(process_variable)}\(\s*-{k}\s*\)\b'
                    # Look for 'param * lag_term'
                    mult_match = re.search(rf'(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*\*\s*{lag_pattern_re}', right)
                    if mult_match:
                         param_name = mult_match.group(1)
                         if param_name in self.parameters.index:
                              ar_params[k] = param_name
                              if abs(self.parameters.get(param_name, 0.0)) > 1e-9:
                                   is_zero_persistence = False
                         # else: print(f"Warning: Potential AR param '{param_name}' not found.")

                if ar_params and is_zero_persistence: # Recheck if all found params are zero
                     all_params_near_zero = all(abs(self.parameters.get(p, 0.0)) <= 1e-9 for p in ar_params.values())
                     is_zero_persistence = all_params_near_zero

                # Store results
                self.exo_process_info[process_variable] = {
                    'driving_shock': driving_shock, 'equation': eq,
                    'right_side': right, 'left_side': left,
                    'ar_params': ar_params, 'zero_persistence': is_zero_persistence
                }
                print(f"  Identified: Var='{process_variable}', Shock='{driving_shock}', ZP={is_zero_persistence}")

            elif len(found_shocks) > 1:
                # === Multiple Shocks: Raise Error ===
                # 'process_variable' is also guaranteed to be defined here
                error_msg = (f"Equation defining '{process_variable}' contains multiple shocks "
                             f"({', '.join(found_shocks)}) on the RHS. This parser requires "
                             f"each exogenous process variable to be driven by a single shock.\n"
                             f"Equation: {eq_clean}")
                raise ValueError(error_msg)

            # If len(found_shocks) == 0, it's treated as an endogenous equation.

        print(f"--- Finished identification. Found {len(self.exo_process_info)} shock-driven processes. ---")
    
    # --- Stage 3: Rewrite Exogenous Equations (In Memory) ---
    def _rewrite_exogenous_equations(self):
        print("\n--- Step 3: Rewriting Exogenous Equations ---")
        rewritten_exo_eqs = {}; max_lag = 10
        if not self.exo_process_info: print("  No processes to rewrite."); return {}
        for pv, info in self.exo_process_info.items():
            rhs=info['right_side']; shock=info['driving_shock']; orig_eq=info['equation']
            if not all([rhs, shock, orig_eq]): continue
            new_l = f"{pv}_p"; current_rhs = rhs
            for k in range(max_lag, 0, -1): # Apply lag substitution
                pat = rf'\b{re.escape(pv)}\(\s*-{k}\s*\)\b'; repl = pv if k == 1 else f"{pv}_lag{k-1}"
                current_rhs = re.sub(pat, repl, current_rhs)
            if not re.search(r'\b'+re.escape(shock)+r'\b', current_rhs): # Ensure shock present
                if info['zero_persistence'] and len(current_rhs.replace(shock,'').split())==0: current_rhs=shock
                elif shock not in current_rhs: current_rhs += f" + {shock}"
            rewritten_eq = f"{new_l} = {current_rhs}"
            rewritten_exo_eqs[pv] = rewritten_eq
            print(f"  Rewriting '{orig_eq}' -> '{rewritten_eq}'")
        print(f"--- Finished rewriting {len(rewritten_exo_eqs)} exo eqs. ---")
        return rewritten_exo_eqs

    # --- Stage 4: Analyze ALL Potential Equations ---
    def _analyze_all_equations(self, equations_to_analyze):
        print(f"\n--- Step 4: Analyzing Variables in {len(equations_to_analyze)} Combined Equations ---")
        variable_shifts = {}; all_base_variables = set()
        var_pat = re.compile(r'\b([a-zA-Z_]\w*)\b')
        shift_pat = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]\d+)\s*\)')
        params = set(self.parameters.index); shocks = set(self.varexo_list)
        for eq in equations_to_analyze:
            pot_names = set(var_pat.findall(eq)); curr_vars = pot_names - params - shocks
            all_base_variables.update(v for v in curr_vars if v)
            shifts_found = set()
            for var, shift_str in shift_pat.findall(eq):
                if var in curr_vars: shifts_found.add(var); variable_shifts.setdefault(var,set()).add(int(shift_str))
            for var in curr_vars:
                if var not in shifts_found and re.search(rf'\b{re.escape(var)}\b(?!\s*\()', eq): variable_shifts.setdefault(var,set()).add(0)
        all_base_variables = {v for v in all_base_variables if v}
        variable_shifts = {k: v for k, v in variable_shifts.items() if k and v}
        print(f"  Analysis found {len(all_base_variables)} unique potential vars.")
        return variable_shifts, all_base_variables

    # --- Stage 5: Generate Auxiliary Definitions ---
    def _generate_auxiliaries(self, variable_shifts, all_base_variables):
        """ Creates aux equations and classifies vars. State = _lag rule. """
        print("\n--- Step 5: Generating Auxiliaries & Classifying Vars (State=Lag Rule) ---")
        transformation_map = {}  # Maps X(-k) -> X_lagk
        aux_equations = []
        processed_aux_eqs = set()
        model_vars = {'state_variables': set(), 'control_variables': set(), 'aux_variables': set(),
                      'all_variables': set(), 'future_variables': set()}
        parameter_names = set(self.parameters.index); shock_names = set(self.varexo_list)

        # Pass 1: Initial classification based on ALL detected base names
        print("  Pass 1: Initial Classification...")
        for var_name in all_base_variables:
            if var_name in shock_names or var_name.endswith("_p") or var_name in parameter_names: continue
            model_vars['all_variables'].add(var_name)
            if re.search(r'_lag\d*$', var_name): model_vars['state_variables'].add(var_name)
            else: model_vars['control_variables'].add(var_name)

        # Pass 2: Generate auxiliaries based on SHIFTS
        print("  Pass 2: Generating Aux Equations based on shifts...")
        unique_vars_with_shifts = set(variable_shifts.keys()) | \
                                  {re.sub(r'_lag\d*$', '', v) for v in all_base_variables if '_lag' in v} | \
                                  {re.sub(r'_lead\d*$', '', v) for v in all_base_variables if '_lead' in v}
        unique_vars_with_shifts -= (shock_names | parameter_names)

        max_lag_needed = {}; max_lead_needed = {}
        for var in unique_vars_with_shifts:
            shifts = variable_shifts.get(var, set())
            max_lag_needed[var] = abs(min((s for s in shifts if s <= 0), default=0))
            max_lead_needed[var] = max((s for s in shifts if s >= 0), default=0)

        for base_name in unique_vars_with_shifts:
            # Ensure base var is known & classified
            if base_name not in model_vars['all_variables']: model_vars['all_variables'].add(base_name)
            if base_name not in model_vars['state_variables']: model_vars['control_variables'].add(base_name)

            # Generate Lag Aux
            if max_lag_needed[base_name] > 0:
                prev = base_name
                for i in range(1, max_lag_needed[base_name] + 1):
                    curr = f"{base_name}_lag" if i == 1 else f"{base_name}_lag{i}"
                    model_vars['all_variables'].add(curr); model_vars['state_variables'].add(curr)
                    model_vars['aux_variables'].add(curr); model_vars['control_variables'].discard(curr)
                    transformation_map[f"{base_name}(-{i})"] = curr
                    eq = f"{curr}_p = {prev}"; cp = f"{curr}_p"
                    model_vars['all_variables'].add(cp); model_vars['future_variables'].add(cp)
                    if eq not in processed_aux_eqs: aux_equations.append(eq); processed_aux_eqs.add(eq)
                    if prev not in model_vars['all_variables']: model_vars['all_variables'].add(prev) # Ensure RHS exists
                    prev = curr
            # Generate Lead Aux
            if max_lead_needed[base_name] > 0:
                fp = f"{base_name}_p"; model_vars['all_variables'].add(fp); model_vars['future_variables'].add(fp)
                if max_lead_needed[base_name] > 1:
                    prev_p = fp
                    for i in range(1, max_lead_needed[base_name]): # Need lead up to N-1 if N exists
                         curr = f"{base_name}_lead{i}"
                         model_vars['all_variables'].add(curr); model_vars['control_variables'].add(curr)
                         model_vars['aux_variables'].add(curr)
                         eq = f"{curr} = {prev_p}"
                         if eq not in processed_aux_eqs: aux_equations.append(eq); processed_aux_eqs.add(eq)
                         prev_p = f"{curr}_p"
                         model_vars['all_variables'].add(prev_p); model_vars['future_variables'].add(prev_p)

        # Final Consolidation
        model_vars['control_variables'] -= model_vars['state_variables']
        model_vars['all_variables'] = {v for v in model_vars['all_variables'] if not v.endswith('_p')}
        model_vars['state_variables'] = {v for v in model_vars['state_variables'] if not v.endswith('_p')}
        model_vars['control_variables'] = {v for v in model_vars['control_variables'] if not v.endswith('_p')}
        model_vars['future_variables'] = {v for v in model_vars['future_variables']}

        print(f"  Generated {len(aux_equations)} aux equations.")
        print(f"  Final Plan States ({len(model_vars['state_variables'])}): {sorted(list(model_vars['state_variables']))}")
        print(f"  Final Plan Controls ({len(model_vars['control_variables'])}): {sorted(list(model_vars['control_variables']))}")

        return transformation_map, aux_equations, model_vars

    # --- Stage 6: Apply Final Substitutions ---
    def _apply_substitutions(self, equations_to_transform, transformation_map, model_vars_dict):
        """ Applies final lag/lead substitutions """
        print(f"\n--- Step 6: Applying Final Substitutions to {len(equations_to_transform)} Equations ---")
        fully_transformed_eqs = []
        all_final_vars = model_vars_dict.get('state_variables', set()) | model_vars_dict.get('control_variables', set())
        stems = sorted(list(all_final_vars | \
                      {re.sub(r'_lag\d*$', '', v) for v in all_final_vars if '_lag' in v} | \
                      {re.sub(r'_lead\d*$', '', v) for v in all_final_vars if '_lead' in v}),
                   key=len, reverse=True)
        print(f"  Substituting based on {len(stems)} stems...")
        for eq_in in equations_to_transform:
            eq_out = eq_in
            for stem in stems:
                if not stem: continue
                # Lags: VAR(-k) -> VAR_lagk
                for k in range(10, 0, -1):
                    mk=f"{stem}(-{k})"; pat=rf'\b{re.escape(stem)}\(\s*-{k}\s*\)\b'
                    if mk in transformation_map: eq_out=re.sub(pat, transformation_map[mk], eq_out)
                # Leads: VAR(+1)->VAR_p, VAR(+k)->VAR_lead{k-1}_p
                pat_p1=rf'\b{re.escape(stem)}\(\s*\+1\s*\)\b'; eq_out=re.sub(pat_p1, f'{stem}_p', eq_out)
                for k in range(2, 10):
                    pat_pk=rf'\b{re.escape(stem)}\(\s*\+{k}\s*\)\b'; repl=f'{stem}_lead{k-1}_p'
                    eq_out=re.sub(pat_pk, repl, eq_out)
            if eq_out.strip(): fully_transformed_eqs.append(eq_out)
        print(f"  Finished substitutions. Resulting model equations: {len(fully_transformed_eqs)}")
        return fully_transformed_eqs

    # --- Main Orchestration Method ---
    def parse(self):
        """ Main parsing function - Simplified & Direct V4 """
        print("\n--- Starting Dynare File Parsing (Simplified V4) ---")
        # Step 1: Initial Parse
        self._read_and_preprocess(); self._parse_declarations(); self._parse_model_block()
        if not self.original_equations: raise ValueError("No model equations found.")

        # Step 2: Identify Exogenous Processes
        self._identify_exogenous_processes() # -> self.exo_process_info

        # Step 3: Rewrite ONLY Exogenous Equations (In Memory)
        rewritten_exo_eqs_dict = self._rewrite_exogenous_equations()

        # Step 4a: Compile Base Equations for Analysis/Substitution
        base_equations = []
        processed_exo_vars = set()
        exo_pvars = set(self.exo_process_info.keys())
        for eq in self.original_equations:
            lvar = eq.split("=", 1)[0].strip() if "=" in eq else None
            if lvar and lvar in exo_pvars:
                if lvar in rewritten_exo_eqs_dict: base_equations.append(rewritten_exo_eqs_dict[lvar]); processed_exo_vars.add(lvar)
                else: base_equations.append(eq) # Fallback if rewrite failed
            else: base_equations.append(eq) # Original endogenous
        for pvar, req in rewritten_exo_eqs_dict.items(): # Add potentially missed
            if pvar not in processed_exo_vars: base_equations.append(req)

        # Step 4b: Analyze these equations
        variable_shifts, all_analyzed_base_vars = self._analyze_all_equations(base_equations)

        # Step 5: Generate Auxiliaries & Classify Vars
        transformation_map, aux_equations, model_vars_dict = \
            self._generate_auxiliaries(variable_shifts, all_analyzed_base_vars)
        self.auxiliary_equations = list(aux_equations)

        # --- Step 6: Final Substitution Pass ---
        # Apply substitutions to the combined list 'base_equations'
        self.transformed_equations = self._apply_substitutions(
            base_equations, transformation_map, model_vars_dict
        )

        # --- Step 7: Consolidate & Finalize Instance Attributes ---
        print("\n--- Step 7: Consolidating Results ---")
        self.state_variables = sorted(list(model_vars_dict.get('state_variables', set())))
        self.control_variables = sorted(list(model_vars_dict.get('control_variables', set())))
        self.auxiliary_variables = sorted(list(model_vars_dict.get('aux_variables', set())))
        self.future_variables = sorted(list(model_vars_dict.get('future_variables', set())))
        self.all_variables = self.state_variables + self.control_variables
        self.final_shock_to_process_var_map = {info['driving_shock']: pvar for pvar, info in self.exo_process_info.items() if 'driving_shock' in info }
        self.endogenous_states.clear(); self.exo_with_shocks.clear(); self.exo_without_shocks.clear()
        for sv in self.state_variables: # Classify final states
            bn=re.sub(r'_lag\d*$','',sv)
            if bn in self.exo_process_info: self.exo_with_shocks.add(sv)
            else: self.endogenous_states.add(sv)
        self.state_variables = sorted(list(self.endogenous_states)) + sorted(list(self.exo_with_shocks)) + sorted(list(self.exo_without_shocks))
        self.all_variables = self.state_variables + self.control_variables # Ensure final order

        self.state_to_shock_map = {s: info['driving_shock'] for s in self.state_variables for bn in [re.sub(r'_lag\d*$', '', s)] if bn in self.exo_process_info for info in [self.exo_process_info[bn]] if 'driving_shock' in info }
        self.shock_to_state_map = {}; [self.shock_to_state_map.setdefault(sh, []).append(st) for st, sh in self.state_to_shock_map.items()]
        self.zero_persistence_processes = [ pvar for pvar, info in self.exo_process_info.items() if info.get('zero_persistence', False) ]
        print(f"    Final States ({len(self.state_variables)}): {self.state_variables}")
        print(f"    Final Controls ({len(self.control_variables)}): {self.control_variables}")

        # Format equations list for output (transformed model + aux)
        self.equations = self.format_transformed_equations(self.transformed_equations, self.auxiliary_equations)
        print("--- Dynare File Parsing Finished ---")
        # Return final dictionary matching target structure
        output_dict = {
             'parameters': list(self.parameters.keys()), 'param_values': self.parameters.to_dict(),
             'states': self.state_variables, 'controls': self.control_variables,
             'all_variables': self.all_variables, 'shocks': self.shocks,
             'equations': self.equations,
             'final_shock_to_process_var_map': self.final_shock_to_process_var_map,
        }
        return output_dict

    # --- Helper to format equations ---
    def format_transformed_equations(self, main_equations, aux_equations):
        formatted = []
        for i, eq in enumerate(main_equations + aux_equations):
            eq_s=eq.strip(); idx=i+1; fmt_eq = eq_s # Default if no '='
            if not eq_s: continue
            if "=" in eq_s: l,r=eq_s.split("=",1); fmt_eq=f"{r.strip()} - ({l.strip()})"
            formatted.append({f"eq{idx}": fmt_eq})
        return formatted

    # --- File Generation Method ---
    def parse_and_generate_files(self, output_dir):
        print(f"\n--- Parsing and Generating Files for: {self.file_path} ---")
        print(f"--- Output Directory: {output_dir} ---")
        try: parse_results_dict = self.parse() # Runs the full workflow
        except Exception as e: print(f"ERROR: Parsing failed: {e}"); import traceback; traceback.print_exc(); raise
        os.makedirs(output_dir, exist_ok=True)
        # Save JSON
        json_path = os.path.join(output_dir, "model.json")
        print(f"  Saving model definition to: {json_path}")
        try:
            keys_to_keep = {'parameters', 'param_values', 'states', 'controls', 'all_variables', 'shocks', 'equations', 'final_shock_to_process_var_map'}
            filtered_dict = {k: v for k, v in parse_results_dict.items() if k in keys_to_keep}
            with open(json_path, 'w') as f: json.dump(filtered_dict, f, indent=2, cls=NumpyEncoder)
            print(f"  Model JSON saved successfully.")
        except Exception as e: print(f"ERROR: Failed to save model.json: {e}")
        # Generate Jacobian
        jacobian_path = os.path.join(output_dir, "jacobian_evaluator.py")
        print(f"  Generating Jacobian evaluator: {jacobian_path}")
        try:
            if not self.transformed_equations and not self.auxiliary_equations: raise RuntimeError("No equations for Jacobian.")
            self.generate_jacobian_evaluator(output_file=jacobian_path)
            print(f"  Jacobian evaluator generated successfully.")
        except Exception as e: print(f"ERROR: Failed to generate jacobian_evaluator.py: {e}"); import traceback; traceback.print_exc()
        # Generate Structure
        structure_path = os.path.join(output_dir, "model_structure.py")
        print(f"  Generating model structure: {structure_path}")
        try:
            structure_dict = self.generate_model_structure()
            with open(structure_path, 'w') as f:
                f.write("import numpy as np\n\n"); f.write(f"indices = {repr(structure_dict.get('indices', {}))}\n\n")
                r = structure_dict.get('R_struct'); c = structure_dict.get('C_selection'); d = structure_dict.get('D_struct')
                f.write(f"R_struct = np.array({repr(r.tolist() if r is not None else [])})\n\n")
                f.write(f"C_selection = np.array({repr(c.tolist() if c is not None else [])})\n\n")
                f.write(f"D_struct = np.array({repr(d.tolist() if d is not None else [])})\n\n")
                f.write("# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls\n\n")
                f.write(f"labels = {repr(structure_dict.get('labels', {}))}\n")
            print(f"  Model structure saved successfully.")
        except Exception as e: print(f"ERROR: Failed to generate model_structure.py: {e}")
        print(f"\n--- File Generation Process Finished for: {self.file_path} ---")
        return os.path.abspath(output_dir)

    # --- Include generate_jacobian_evaluator and generate_model_structure ---
    def generate_jacobian_evaluator(self, output_file=None):
        print("  Generating Jacobian evaluator function...")
        if not self.transformed_equations and not self.auxiliary_equations: raise ValueError("Eqs missing for Jacobian.")
        formatted_eqs_list = self.format_transformed_equations(self.transformed_equations, self.auxiliary_equations)
        system_equations_str = [list(eq_dict.values())[0] for eq_dict in formatted_eqs_list]
        current_vars = getattr(self, 'all_variables', []); future_vars = [v+"_p" for v in current_vars]
        shock_vars = getattr(self, 'varexo_list', []); param_names = list(self.parameters.index)
        if not current_vars: print("Warning: No variables for Jacobian."); return "def evaluate_jacobians(theta): return np.array([]), np.array([]), np.array([])"
        print(f"    Jacobian w.r.t. {len(current_vars)} vars, {len(shock_vars)} shocks.")
        current_syms_map={v: sy.symbols(v) for v in current_vars}; future_syms_map={v: sy.symbols(v) for v in future_vars}
        shock_syms_map={s: sy.symbols(s) for s in shock_vars}; param_syms_map={p: sy.symbols(p) for p in param_names}
        local_sym_map = {**current_syms_map, **future_syms_map, **shock_syms_map, **param_syms_map}
        current_syms_list=[current_syms_map[v] for v in current_vars]; future_syms_list=[future_syms_map.get(v, None) for v in future_vars if future_syms_map.get(v, None) is not None] # Filter None
        # Ensure future_syms_list has the same length as current_vars by adding placeholders if needed, though ideally all _p vars exist
        if len(future_syms_list) != len(current_vars):
            print(f"Warning: Mismatch between future symbols ({len(future_syms_list)}) and current vars ({len(current_vars)}). Jacobian A might be wrong size.")
            # Pad or adjust future_syms_list based on `future_vars`? Risky. Best if all _p exist.
            future_syms_list = [future_syms_map.get(f"{v}_p", sy.symbols(f"{v}_p_missing")) for v in current_vars] # Create placeholders

        shock_syms_list=[shock_syms_map[s] for s in shock_vars]; param_syms_list=[param_syms_map[p] for p in param_names]
        F_vector = []
        print(f"    Parsing {len(system_equations_str)} equations with Sympy...")
        for i, eq_str in enumerate(system_equations_str):
            try: F_vector.append(sy.sympify(eq_str, locals=local_sym_map))
            except (sy.SympifyError, SyntaxError, TypeError, AttributeError) as e: raise ValueError(f"Sympy parsing failed eq {i+1}: {eq_str}\nError: {e}")
        F_matrix = sy.Matrix(F_vector); print(f"    System Matrix F shape: {F_matrix.shape}")
        print(f"    Computing Jacobians..."); A_sym = F_matrix.jacobian(future_syms_list); B_sym = F_matrix.jacobian(current_syms_list)
        C_sym = F_matrix.jacobian(shock_syms_list) if shock_vars else sy.zeros(F_matrix.rows, 0)
        print(f"    Symbolic Jacobians shapes: A={A_sym.shape}, B={B_sym.shape}, C={C_sym.shape}")
        n_eqs, n_vars = B_sym.shape; n_shocks = C_sym.shape[1]
        # Check dimensions
        if n_vars != len(current_vars): print(f"Warning: Jacobian B columns ({n_vars}) != variables ({len(current_vars)})")
        if A_sym.shape[1] != len(current_vars): print(f"Warning: Jacobian A columns ({A_sym.shape[1]}) != variables ({len(current_vars)})")
        if n_shocks != len(shock_vars): print(f"Warning: Jacobian C columns ({n_shocks}) != shocks ({len(shock_vars)})")

        header = ["import numpy as np", "# Generated by DynareParser","", "def evaluate_jacobians(theta):", f"    \"\"\"Evaluates A=dF/dxp, B=dF/dx, C=dF/de.\n    Params: {param_names}\n    Vars: {current_vars}\n    Shocks: {shock_vars}\"\"\"","    # Unpack params"]
        param_unpack = [f"    {p} = theta[{i}]" for i, p in enumerate(param_names)]
        # Use dimensions determined from Jacobian calculation
        init_matrices = ["",f"    A=np.zeros(({n_eqs},{n_vars}))",f"    B=np.zeros(({n_eqs},{n_vars}))",f"    C=np.zeros(({n_eqs},{n_shocks}))",""]
        body_A = ["    # Jacobian A = dF/dxp"]; body_B = ["    # Jacobian B = dF/dx"]; body_C = ["    # Jacobian C = dF/de"]
        sympy_to_numpy = {'exp': 'np.exp', 'log': 'np.log', 'sqrt': 'np.sqrt'}
        print("    Generating code strings for Jacobian elements...");
        for r in range(n_eqs):
            for c in range(n_vars):
                if A_sym[r, c]!=0: body_A.append(f"    A[{r},{c}] = {sy.pycode(A_sym[r,c],user_functions=sympy_to_numpy)}")
                if B_sym[r, c]!=0: body_B.append(f"    B[{r},{c}] = {sy.pycode(B_sym[r,c],user_functions=sympy_to_numpy)}")
            for c in range(n_shocks):
                if C_sym[r, c]!=0: body_C.append(f"    C[{r},{c}] = {sy.pycode(C_sym[r,c],user_functions=sympy_to_numpy)}")
        footer = ["", "    return A, B, C"]; full_code = "\n".join(header+param_unpack+init_matrices+body_A+[""]+body_B+[""]+body_C+footer)
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f: f.write(full_code)
                print(f"    Jacobian code written to {output_file}")
            except Exception as e: print(f"    ERROR writing Jacobian file: {e}")
        else: return full_code
        return None

    def generate_model_structure(self):
        print("\n--- Generating Model Structure Components (State=Lag Rule) ---")
        state_vars=getattr(self,'state_variables',[]); control_vars=getattr(self,'control_variables',[])
        all_vars=getattr(self,'all_variables',[]); shock_vars=getattr(self,'varexo_list',[])
        n_states=len(state_vars); n_controls=len(control_vars); n_vars=len(all_vars); n_shocks=len(shock_vars)
        if n_vars==0: return {'indices': {}, 'R_struct': np.array([]), 'C_selection': np.array([]), 'D_struct': np.array([]), 'labels': {}}
        print(f"  Dims: St={n_states}, Co={n_controls}, Tot={n_vars}, Sh={n_shocks}")
        R_struct=np.zeros((n_states,n_shocks)); C_selection=np.zeros((n_vars,n_states))
        state_indices=[all_vars.index(v) for v in state_vars if v in all_vars]
        if len(state_indices)==n_states: [C_selection.__setitem__((state_idx,i),1.0) for i,state_idx in enumerate(state_indices)]
        else: print(f"Warn: State idx mismatch C_sel ({len(state_indices)} vs {n_states})")
        D_struct=np.zeros((n_vars,n_shocks))
        shock_map=getattr(self, 'final_shock_to_process_var_map', {})
        if shock_map:
            for shock,pvar in shock_map.items():
                try: j=shock_vars.index(shock); i=all_vars.index(pvar); D_struct[i,j]=1.0
                except (ValueError,AttributeError,IndexError): pass
        labels={'state_labels':state_vars, 'control_labels':control_vars, 'variable_labels':all_vars, 'shock_labels':shock_vars}
        indices={'n_states':n_states, 'n_controls':n_controls, 'n_vars':n_vars, 'n_shocks':n_shocks,
                 'n_endogenous_states':len(getattr(self,'endogenous_states',set())),'n_exo_states_ws':len(getattr(self,'exo_with_shocks',set())),
                 'n_exo_states_wos':len(getattr(self,'exo_without_shocks',set())),'zero_persistence_processes':getattr(self,'zero_persistence_processes',[])}
        print(f"  Structure components generated.")
        return {'indices': indices, 'R_struct': R_struct, 'C_selection': C_selection, 'D_struct': D_struct, 'labels': labels}
