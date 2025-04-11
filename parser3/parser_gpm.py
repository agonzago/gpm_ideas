import re
import json
import os
import numpy as np
import scipy.linalg as la
import sys
import sympy as sy
import matplotlib.pyplot as plt

class DynareParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.var_list = []
        self.varexo_list = []
        self.parameters = {}
        self.equations = []
        self.model_text = ""
        self.state_variables = []
        self.control_variables = []
        self.all_variables = []
        self.auxiliary_variables = []

    def read_dynare_file(self):
        """Read and preprocess the Dynare .mod file content"""
        with open(self.file_path, 'r') as file:
            self.content = file.read()
        self.preprocess_content()  # Clean content immediately after reading

    def preprocess_content(self):
        """Remove comments and clean up content before parsing"""
        # Remove single-line comments
        self.content = re.sub(r'//.*', '', self.content)
        # Remove extra whitespace
        self.content = re.sub(r'\s+', ' ', self.content)

    def parse_variables(self):
        """Extract variable declarations from the Dynare file"""
        var_section = re.search(r'var\s+(.*?);', self.content, re.DOTALL)
        if var_section:
            var_text = var_section.group(1)
            # No need for comment removal here since we preprocessed
            var_list = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', var_text)
            self.var_list = [v for v in var_list if v]
        
    def parse_exogenous(self):
        """Extract exogenous variable declarations from the Dynare file"""
        varexo_section = re.search(r'varexo\s+(.*?);', self.content, re.DOTALL)
        if varexo_section:
            varexo_text = varexo_section.group(1)
            # Remove comments and split by whitespace
            varexo_text = re.sub(r'//.*', '', varexo_text)
            varexo_list = [v.strip() for v in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', varexo_text)]
            self.varexo_list = [v for v in varexo_list if v]  # Remove empty strings
            
    def parse_parameters(self):
        """Extract parameter declarations and values from the Dynare file"""
        # Get parameter names
        params_section = re.search(r'parameters\s+(.*?);', self.content, re.DOTALL)
        if params_section:
            params_text = params_section.group(1)
            params_text = re.sub(r'//.*', '', params_text)  # Remove comments
            param_list = [p.strip() for p in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', params_text)]
            param_list = [p for p in param_list if p]  # Remove empty strings
            
            # Initialize parameters dictionary
            for param in param_list:
                self.parameters[param] = None
                
            # Get parameter values
            for param in param_list:
                param_value = re.search(rf'{param}\s*=\s*([0-9.-]+)', self.content)
                if param_value:
                    self.parameters[param] = float(param_value.group(1))
    
    def parse_model(self):
        """Extract the model equations from the Dynare file"""
        model_section = re.search(r'model;(.*?)end;', self.content, re.DOTALL)
        if model_section:
            self.model_text = model_section.group(1).strip()
            
            # Split by semicolons to get individual equations
            equations = re.split(r';', self.model_text)
            equations = [eq.strip() for eq in equations if eq.strip()]
            
            self.equations = equations

    def identify_exogenous_processes(self):
        """
        Identify exogenous process variables by finding the equations
        where the declared exogenous shocks appear. Flags error if multiple
        shocks are found in the RHS of a single equation defining a variable.

        Returns:
            Dictionary mapping the *process variable name* to details about
            the exogenous process (shock, equation, etc.).
            e.g., {'RES_RS': {'driving_shock': 'SHK_RS', 'equation': '...', ...}}
        """
        exo_processes = {} # Key: process variable name

        for eq in self.equations:
            eq_clean = eq.strip()
            if "=" not in eq_clean:
                continue

            left, right = [s.strip() for s in eq_clean.split("=", 1)]
            process_variable = left.strip() # Variable being defined

            # Find ALL shocks on the RHS
            found_shocks = []
            for shock in self.varexo_list:
                if re.search(r'\b' + re.escape(shock) + r'\b', right):
                    found_shocks.append(shock)

            # --- Process based on shocks found ---
            if len(found_shocks) == 1:
                # Exactly one shock found - this defines an exogenous process
                driving_shock = found_shocks[0]

                # --- Check for Re-definition ---
                if process_variable in exo_processes:
                    print(f"Warning: Exogenous process variable '{process_variable}' seems to be defined "
                        f"by multiple equations/shocks ('{exo_processes[process_variable]['driving_shock']}' and '{driving_shock}'). "
                        f"Using the definition from equation: {eq}")
                    # Decide how to handle - overwrite or error? Overwriting for now.

                # Store information keyed by the PROCESS variable name
                exo_processes[process_variable] = {
                    'driving_shock': driving_shock,
                    'equation': eq,
                    'right_side': right,
                    'left_side': left
                }

                # Analyze AR structure (same as before)
                ar_params = {}
                # ... (AR parameter detection logic remains the same) ...
                for k in range(1, 5):
                    lag_pattern = rf'\b{re.escape(process_variable)}\(\s*-{k}\s*\)\b'
                    match = re.search(rf'(\b[a-zA-Z_]\w*\b)\s*\*\s*{lag_pattern}', right)
                    if match:
                        param_name = match.group(1)
                        if param_name in self.parameters:
                            ar_params[k] = param_name
                        # else: print warning if needed

                exo_processes[process_variable]['ar_params'] = ar_params

                # Check for zero persistence (same as before)
                is_zero_persistence = True
                # ... (zero persistence check logic remains the same) ...
                for lag, param in ar_params.items():
                    if param in self.parameters and abs(self.parameters[param]) > 1e-10:
                        is_zero_persistence = False
                        break
                exo_processes[process_variable]['zero_persistence'] = is_zero_persistence

                print(f"Identified exogenous process: Var='{process_variable}', driving_shock='{driving_shock}', "
                    f"AR_params={list(ar_params.values())}, zero_persist={is_zero_persistence}")


            elif len(found_shocks) > 1:
                # --- ERROR: Multiple shocks found ---
                raise ValueError(f"Equation defining '{process_variable}' contains multiple shocks "
                                f"({', '.join(found_shocks)}) on the RHS. This is not allowed.\n"
                                f"Equation: {eq}")

            # If len(found_shocks) == 0, this equation doesn't define a primary exogenous process.

        # Store this map keyed by process variable name
        self.exo_process_info = exo_processes # Renamed for clarity
        return exo_processes

    def rewrite_exogenous_equations_with_correct_timing(self):
        """
        Rewrites equations defining exogenous processes (identified via shocks).
        Uses self.exo_process_info. Modifies self.equations in place.
        """
        print("\n--- Rewriting Exogenous Process Equations (Shock-Driven ID, Process Key) ---")

        if not hasattr(self, 'exo_process_info'):
            self.identify_exogenous_processes() # Ensure it's run

        if not self.exo_process_info:
            print("No exogenous processes identified to rewrite.")
            return {}

        eq_rewrites = {}

        # Iterate through {process_variable: info_dict}
        for process_variable, process_info in self.exo_process_info.items():
            original_eq = process_info['equation']
            right_side = process_info['right_side']
            driving_shock = process_info['driving_shock'] # Shock associated with this process

            # Apply correct timing: Z_t = f(Z_{t-k}) + SHK_t ==> Z_{t+1} = f(Z_{t+1-k}) + SHK_t
            # Notation:             Z_p = f(Z, Z_lag, ...) + SHK

            new_left = f"{process_variable}_p"
            new_right = right_side

            # Rewrite lags Z(-k) -> Z_lag{k-1}
            for lag in range(10, 0, -1):
                old_pattern = rf'\b{re.escape(process_variable)}\(\s*-{lag}\s*\)\b'
                new_pattern = process_variable if lag == 1 else f"{process_variable}_lag{lag-1}"
                new_right = re.sub(old_pattern, new_pattern, new_right)

            # Keep original shock name on RHS (SHK)
            if not re.search(r'\b' + re.escape(driving_shock) + r'\b', new_right):
                # Handle if shock disappeared (e.g., zero persistence Z = SHK)
                if process_info['zero_persistence'] and new_right.strip() == '':
                    new_right = driving_shock # If eq was just Z = SHK, rewrite Z_p = SHK
                elif driving_shock not in new_right:
                    print(f"Warning: Shock '{driving_shock}' missing from rewritten RHS for '{process_variable}'. Re-adding.")
                    new_right += f" + {driving_shock}"

            new_eq = f"{new_left} = {new_right}"
            eq_rewrites[original_eq] = new_eq
            print(f"Rewriting '{original_eq}' -> '{new_eq}'")

            # Update map with rewritten info
            self.exo_process_info[process_variable]['rewritten_equation'] = new_eq
            # ... (add rewritten_left, rewritten_right if needed elsewhere) ...

        # Apply rewrites (same as before)
        modified_equations = [eq_rewrites.get(eq, eq) for eq in self.equations]
        self.equations = modified_equations
        print("Exogenous equation rewrite complete.")
        return eq_rewrites

    def analyze_model_variables(self):
        """
        First pass: Analyze all variables and their time shifts across the entire model.
        
        Returns:
            variable_shifts: Dictionary mapping each variable to its set of time shifts
            all_variables: Set of all base variable names found in the model
        """
        variable_shifts = {}  # Maps variables to their time shifts
        all_variables = set()  # All base variable names
        
        # Process each equation to find variables and their time shifts
        for equation in self.equations:
            # Remove comments and clean up
            equation = re.sub(r'//.*', '', equation).strip()
            
            # Find all base variables (excluding parameters)
            base_vars = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', equation))
            base_vars = base_vars - set(self.parameters.keys())
            
            # Add to all_variables set
            all_variables.update(base_vars)
            
            # For each variable, find all its lead/lag patterns
            for var_name in base_vars:
                # Initialize if not already in dictionary
                if var_name not in variable_shifts:
                    variable_shifts[var_name] = set()
                    variable_shifts[var_name].add(0)  # Always include current period
                
                # Find all lead/lag patterns for this variable
                lead_lag_pattern = rf'{var_name}\(\s*([+-]?\d+)\s*\)'
                lead_lag_matches = re.findall(lead_lag_pattern, equation)
                
                # Add all time shifts found for this variable
                for time_shift_str in lead_lag_matches:
                    variable_shifts[var_name].add(int(time_shift_str))
        
        return variable_shifts, all_variables

    def create_transformation_plan(self, variable_shifts):
        """
        Create transformation plan.
        State Definition: Variable is a state IFF its name ends with _lag or _lagN.
        Contemporaneous RES_* variables are controls.
        """
        # ... (initial setup: transformation_map, aux_equations, etc.) ...

        model_variables = {
            'state_variables': set(),
            'control_variables': set(),
            'aux_variables': set(),
            'all_variables': set(),
            'future_variables': set()
        }

        all_detected_var_names = set(variable_shifts.keys())

        # --- Pass 1: Register variables and apply STRICT state classification ---
        for var_name in all_detected_var_names:
            if var_name in self.varexo_list or var_name.endswith("_p"):
                continue
            model_variables['all_variables'].add(var_name)

            # --- STRICT Rule: State ONLY if it ends with _lag ---
            if re.search(r'_lag\d*$', var_name):
                model_variables['state_variables'].add(var_name)
            elif var_name not in self.varexo_list: # Otherwise, it's a control
                model_variables['control_variables'].add(var_name)

        # --- Pass 2: Process shifts and generate auxiliary structures ---
        # The logic here should remain largely the same as the previous version,
        # ensuring that *if* a lag (like X_lag, X_lag2, RES_RS_lag) is needed
        # (either by finding X(-k) or detecting X_lagk directly),
        # it is added to the correct sets (all_variables, state_variables, aux_variables)
        # and its defining equation (X_lag_p = X or X_lag2_p = X_lag) is generated.
        # The classification done in Pass 1 is definitive for state/control status.

        for var_name, shifts in variable_shifts.items():
            # ... (skip shocks) ...

            base_name = re.sub(r'_lag\d*$', '', var_name)
            is_already_lag = re.search(r'_lag\d*$', var_name) is not None

            # --- Handle Lags ---
            min_lag_shift = min((s for s in shifts if s <= 0), default=1)
            current_lag_level = 0
            if is_already_lag:
                lag_match = re.search(r'_lag(\d*)$', var_name)
                current_lag_level = 1
                if lag_match and lag_match.group(1):
                    current_lag_level = int(lag_match.group(1))
            effective_min_lag = min(min_lag_shift, -current_lag_level if current_lag_level > 0 else 0)

            if effective_min_lag < 0:
                # Ensure base variable exists and is classified (likely as control)
                if base_name not in model_variables['all_variables']:
                    model_variables['all_variables'].add(base_name)
                    if base_name not in model_variables['state_variables'] and \
                        base_name not in self.varexo_list:
                        model_variables['control_variables'].add(base_name)


                prev_lag_var = base_name
                for i in range(1, abs(effective_min_lag) + 1):
                    curr_lag_var = f"{base_name}_lag" if i == 1 else f"{base_name}_lag{i}"

                    # Add lag variable (it MUST be a state by the rule)
                    model_variables['all_variables'].add(curr_lag_var)
                    model_variables['state_variables'].add(curr_lag_var) # Confirms state status
                    model_variables['aux_variables'].add(curr_lag_var)
                    if curr_lag_var in model_variables['control_variables']:
                        model_variables['control_variables'].remove(curr_lag_var) # Remove if wrongly added

                    # Define mapping if needed
                    if -i in shifts:
                        transformation_map[f"{base_name}(-{i})"] = curr_lag_var

                    # Create auxiliary equation
                    aux_eq = f"{curr_lag_var}_p = {prev_lag_var}"
                    if aux_eq not in processed_aux_eqs:
                        # ... (add aux_eq, register _p vars, etc. - same as before) ...
                        aux_equations.append(aux_eq)
                        processed_aux_eqs.add(aux_eq)
                        model_variables['all_variables'].add(f"{curr_lag_var}_p")
                        model_variables['future_variables'].add(f"{curr_lag_var}_p")
                        model_variables['all_variables'].add(prev_lag_var)


                    prev_lag_var = curr_lag_var

            # --- Handle Leads ---
            # Logic remains the same as before (leads -> controls, aux eqs up to n-1)
            # Ensure base variable is classified correctly (likely control)
            # ... (lead handling code from previous correct version) ...
            max_lead_shift = max((s for s in shifts if s >= 0), default=-1)
            if max_lead_shift >= 0: # Current or future shifts exist
                if base_name not in model_variables['all_variables']:
                    model_variables['all_variables'].add(base_name)
                    if base_name not in model_variables['state_variables'] and \
                        base_name not in self.varexo_list:
                        model_variables['control_variables'].add(base_name)
            # ... (rest of lead code: creating lead vars, aux eqs etc.)

        # --- Final Consolidation ---
        # Ensure sets are disjoint based on the strict _lag rule
        model_variables['control_variables'] = model_variables['control_variables'] - model_variables['state_variables']
        # Remove _p vars
        model_variables['state_variables'] = {v for v in model_variables['state_variables'] if not v.endswith('_p')}
        model_variables['control_variables'] = {v for v in model_variables['control_variables'] if not v.endswith('_p')}
        model_variables['all_variables'] = {v for v in model_variables['all_variables'] if not v.endswith('_p')}

        # ... (Debug prints) ...

        return transformation_map, aux_equations, model_variables

    def apply_transformation(self):
        """
        Complete model transformation.
        Final Classification Rule: State iff ends with _lag.
        """
        print("\n=== Applying Model Transformation (State=Lag Rule) ===")

        # Step 1: Rewrite exogenous equations (as before)
        self.rewrite_exogenous_equations_with_correct_timing()

        # Step 2: Analyze variables and shifts *after* rewriting
        variable_shifts, all_base_variables = self.analyze_model_variables()

        # Step 3: Create transformation plan (uses the modified logic)
        transformation_map, aux_equations, model_variables_dict = self.create_transformation_plan(variable_shifts)

        # Step 4: Apply transformations to all equations (as before)
        transformed_equations = []
        # ... (logic for substituting lags/leads remains the same) ...
        for i, equation in enumerate(self.equations):
            clean_eq = re.sub(r'//.*', '', equation).strip()
            transformed_eq = clean_eq
            eq_vars = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', clean_eq))
            eq_vars = eq_vars & all_base_variables
            for var_name in sorted(list(eq_vars), key=len, reverse=True):
                for k in range(10, 0, -1):
                    orig_expr_pattern = rf'{re.escape(var_name)}\(\s*-{k}\s*\)'
                    map_key = f"{var_name}(-{k})"
                    if map_key in transformation_map:
                        transformed_eq = re.sub(orig_expr_pattern, transformation_map[map_key], transformed_eq)
                orig_expr_p1 = rf'{re.escape(var_name)}\(\s*\+1\s*\)'
                transformed_eq = re.sub(orig_expr_p1, f'{var_name}_p', transformed_eq)
                for j in range(2, 10):
                    orig_expr_pn = rf'{re.escape(var_name)}\(\s*\+{j}\s*\)'
                    replacement = f'{var_name}_lead{j-1}_p'
                    transformed_eq = re.sub(orig_expr_pn, replacement, transformed_eq)
            transformed_equations.append(transformed_eq)

        # Step 5: Update intermediate properties
        self.transformed_equations = transformed_equations
        self.auxiliary_equations = aux_equations

        # --- Step 6: Final Variable Classification & Ordering ---
        # Get the sets identified by create_transformation_plan
        identified_states = model_variables_dict['state_variables'] # Should ONLY contain _lag vars now
        identified_controls = model_variables_dict['control_variables'] # Should include RES_* vars
        identified_all_vars = model_variables_dict['all_variables'] # All non-_p vars
        identified_aux_vars = model_variables_dict['aux_variables']
        identified_future_vars = model_variables_dict['future_variables']

        # Analyze exogenous processes based on transformed model to map shocks
        exo_info = self.analyze_exogenous_processes_with_correct_timing()
        self.shock_to_state_map = exo_info.get('shock_to_state_map', {})
        self.state_to_shock_map = exo_info.get('state_to_shock_map', {})
        self.zero_persistence_processes = exo_info.get('zero_persistence_processes', [])

        # Classify the identified states (_lag vars) into endogenous/exogenous
        self.endogenous_states = set()
        self.exo_with_shocks = set()
        self.exo_without_shocks = set()

        shock_driven_exo_bases = set()
        for shock, state_target in self.shock_to_state_map.items():
            # Map points to the base RES name (e.g., RES_X from RES_X or RES_X_lag)
            base_name = re.sub(r'_lag\d*$', '', state_target)
            if base_name.startswith("RES_"):
                shock_driven_exo_bases.add(base_name)

        for state_var in identified_states: # Iterate only over vars ending in _lag
            if state_var.startswith("RES_"):
                base_name = re.sub(r'_lag\d*$', '', state_var)
                if base_name in shock_driven_exo_bases:
                    self.exo_with_shocks.add(state_var)
                else:
                    self.exo_without_shocks.add(state_var)
            else:
                self.endogenous_states.add(state_var)

        # Assign final sorted lists
        self.state_variables = sorted(list(self.endogenous_states)) + \
                            sorted(list(self.exo_with_shocks)) + \
                            sorted(list(self.exo_without_shocks))

        # Controls are everything in identified_controls
        self.control_variables = sorted(list(identified_controls))

        # Ensure consistency: All vars = states + controls
        self.all_variables = self.state_variables + self.control_variables
        # Double check identified_all_vars matches - potential debug step
        if set(self.all_variables) != identified_all_vars:
            print("Warning: Final all_variables mismatch with identified_all_vars!")
            print(f"  Final: {set(self.all_variables)}")
            print(f"  Identified: {identified_all_vars}")
            # Fallback or recalculate? Let's use the constructed list for now.

        self.auxiliary_variables = sorted(list(identified_aux_vars))
        self.future_variables = sorted(list(identified_future_vars))


        # Step 7: Format equations (as before)
        formatted_equations = self.format_transformed_equations(transformed_equations, aux_equations)

        print(f"\nModel transformation complete (State=Lag Rule):")
        print(f"  States ({len(self.state_variables)}): {self.state_variables}")
        print(f"  Controls ({len(self.control_variables)}): {self.control_variables}")
        print(f"  All Variables ({len(self.all_variables)}): {self.all_variables}")
        print(f"  Future Vars ({len(self.future_variables)}): {self.future_variables}")
        print(f"  Aux Equations: {len(self.auxiliary_equations)}")

        # Prepare the final output dictionary (as before)
        output = {
            'equations': formatted_equations,
            'state_variables': self.state_variables,
            'control_variables': self.control_variables,
            'auxiliary_variables': self.auxiliary_variables,
            'all_variables': self.all_variables,
            'future_variables': self.future_variables,
            'endogenous_states': sorted(list(self.endogenous_states)),
            'exo_with_shocks': sorted(list(self.exo_with_shocks)),
            'exo_without_shocks': sorted(list(self.exo_without_shocks)),
            'shock_to_state_map': self.shock_to_state_map,
            'state_to_shock_map': self.state_to_shock_map,
            'zero_persistence_processes': self.zero_persistence_processes,
            'parameters': list(self.parameters.keys()),
            'param_values': self.parameters,
            'shocks': self.varexo_list,
        }
        # output['output_text'] = self.generate_output_text(formatted_equations)

        return output

    def analyze_exogenous_processes_with_correct_timing(self):
        """
        Analyze exogenous processes in the model with the correct timing convention.
        
        Returns:
            Dictionary with exogenous process analysis results
        """
        print("\n--- Analyzing Exogenous Processes with Correct Timing ---")
        
        # Initialize mappings
        shock_to_state_map = {}
        state_to_shock_map = {}
        zero_persistence_processes = []
        
        # Identify all exogenous processes (RES_* variables)
        exo_processes = {}
        for var in self.all_variables:
            if var.startswith("RES_"):
                exo_processes[var] = {
                    'lags': [v for v in self.all_variables if v.startswith(f"{var}_lag")]
                }
        
        # Analyze exogenous process equations
        for eq in self.transformed_equations:
            if "=" not in eq:
                continue
                
            left, right = [s.strip() for s in eq.split("=", 1)]
            
            # Check for exogenous process equations (now with _p on the left side)
            for process_name in exo_processes.keys():
                if left == f"{process_name}_p":
                    # This is the equation for this exogenous process
                    exo_processes[process_name]['equation'] = eq
                    
                    # Check for AR parameters
                    ar_params = re.findall(r'(rho_\w+)\s*\*', right)
                    exo_processes[process_name]['ar_params'] = ar_params
                    
                    # Check for zero persistence
                    is_zero_persistence = True
                    for param in ar_params:
                        if param in self.parameters and abs(self.parameters[param]) > 1e-10:
                            is_zero_persistence = False
                            break
                    
                    if is_zero_persistence:
                        zero_persistence_processes.append(process_name)
                    
                    # Check which shock drives this process
                    for shock in self.varexo_list:
                        if shock in right:
                            exo_processes[process_name]['shock'] = shock
                            
                            # Important: Map the shock to both the process
                            # and its first lag variable if it exists
                            shock_to_state_map[shock] = process_name
                            state_to_shock_map[process_name] = shock
                            
                            # Also map to lag variable
                            lag_var = f"{process_name}_lag"
                            if lag_var in self.all_variables:
                                shock_to_state_map[shock] = lag_var
                                state_to_shock_map[lag_var] = shock
                            
                            print(f"  Process {process_name} is driven by shock {shock}")
                            if is_zero_persistence:
                                print(f"  Process {process_name} has ZERO PERSISTENCE")
        
        return {
            'shock_to_state_map': shock_to_state_map,
            'state_to_shock_map': state_to_shock_map,
            'zero_persistence_processes': zero_persistence_processes,
            'exo_processes': exo_processes
        }

    def format_transformed_equations(self, main_equations, aux_equations):
        """Format transformed equations for output"""
        formatted_equations = []
        
        # Process main equations
        for i, equation in enumerate(main_equations):
            # Convert equation to standard form (right side - left side = 0)
            if "=" in equation:
                left_side, right_side = equation.split("=", 1)
                formatted_eq = f"{right_side.strip()} - ({left_side.strip()})"
            else:
                formatted_eq = equation
            
            eq_dict = {f"eq{i+1}": formatted_eq}
            formatted_equations.append(eq_dict)
        
        # Process auxiliary equations
        for i, aux_equation in enumerate(aux_equations):
            left_side, right_side = aux_equation.split("=", 1)
            formatted_eq = f"{right_side.strip()} - ({left_side.strip()})"
            
            eq_dict = {f"eq{len(main_equations) + i + 1}": formatted_eq}
            formatted_equations.append(eq_dict)
        
        return formatted_equations
    
    def generate_output_text(self, formatted_equations):
        """Generate output text in the required format"""
        output_text = "equations = {\n"
        
        for i, eq_dict in enumerate(formatted_equations):
            for eq_name, eq_value in eq_dict.items():
                output_text += f'\t{{"{eq_name}": "{eq_value}"}}'
                if i < len(formatted_equations) - 1:
                    output_text += ",\n"
                else:
                    output_text += "\n"
        
        output_text += "};\n\n"
        
        output_text += "variables = ["
        output_text += ", ".join([f'"{var}"' for var in self.all_variables])
        output_text += "];\n\n"
        
        output_text += "parameters = ["
        output_text += ", ".join([f'"{param}"' for param in self.parameters.keys()])
        output_text += "];\n\n"
        
        for param, value in self.parameters.items():
            output_text += f"{param} = {value};\n"
        
        output_text += "\n"
        
        output_text += "shocks = ["
        output_text += ", ".join([f'"{shock}"' for shock in self.varexo_list])
        output_text += "];\n"
        
        return output_text
    
    def parse(self):
        """Main parsing function with correct timing implementation"""
        self.read_dynare_file()
        self.parse_variables()
        self.parse_exogenous()
        self.parse_parameters()
        self.parse_model()
        
        # Apply the model transformation with correct timing
        output = self.apply_transformation()
        
        # Add other necessary output fields
        output['parameters'] = list(self.parameters.keys())
        output['param_values'] = self.parameters
        output['shocks'] = self.varexo_list
        output['output_text'] = self.generate_output_text(output['equations'])
        
        return output
    
    def save_json(self, output_file):
        """Save the parsed model to a JSON file"""
        output = self.parse()
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Model parsed and saved to {output_file}")
        return output

    def generate_jacobian_evaluator(self, output_file=None):
        """
        Generate a Python function that evaluates the Jacobian matrices for the model.
        
        Args:
            output_file (str, optional): Path to save the generated Python code
                
        Returns:
            str: The generated Python code for the Jacobian evaluator
        """
        print("Generating Jacobian evaluator...")
        
        # First, apply the model transformation if it hasn't been done yet
        if not hasattr(self, 'transformed_equations') or not self.transformed_equations:
            print("Applying model transformation first...")
            self.apply_transformation()

        # Get the relevant model components after transformation
        variables = self.state_variables + self.control_variables
        exogenous = self.varexo_list
        parameters = list(self.parameters.keys())
        
        print("State Variables Order:", self.state_variables)
        print("Control Variables Order:", self.control_variables)
        
        # Create variables with "_p" suffix for t+1 variables
        variables_p = [var + "_p" for var in variables]
        
        # Create symbolic variables for all model components
        var_symbols = {var: sy.symbols(var) for var in variables}
        var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
        exo_symbols = {exo: sy.symbols(exo) for exo in exogenous}
        param_symbols = {param: sy.symbols(param) for param in parameters}
        
        # Combine all symbols
        all_symbols = {**var_symbols, **var_p_symbols, **exo_symbols, **param_symbols}
        
        # Get endogenous equations from the formatted equations
        formatted_equations = self.format_transformed_equations(self.transformed_equations, self.auxiliary_equations)
        endogenous_eqs = {}
        for eq_dict in formatted_equations:
            endogenous_eqs.update(eq_dict)
        
        # Parse endogenous equations into sympy expressions
        equations = []
        success_count = 0
        error_count = 0
        
        for eq_name, eq_str in endogenous_eqs.items():
            # Convert string to sympy expression
            eq_expr = eq_str
            for name, symbol in all_symbols.items():
                # Use regex to match whole words only
                pattern = r'\b' + re.escape(name) + r'\b'
                eq_expr = re.sub(pattern, str(symbol), eq_expr)
            
            # Try to parse the expression
            try:
                expr = sy.sympify(eq_expr)
                equations.append(expr)
                success_count += 1
            except Exception as e:
                print(f"Failed to parse equation {eq_name}: {eq_str}")
                print(f"Error: {str(e)}")
                # Try to recover by using a placeholder
                equations.append(sy.sympify("0"))
                error_count += 1
        
        print(f"Parsed {success_count} equations successfully, {error_count} with errors")
        
        # Create system as sympy Matrix
        F = sy.Matrix(equations)
        
        # Compute Jacobians for endogenous system
        X_symbols = [var_symbols[var] for var in variables]
        X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
        Z_symbols = [exo_symbols[exo] for exo in exogenous]  
        
        # A = ∂F/∂X_p (Jacobian with respect to future variables)
        print("Computing A matrix...")
        A_symbolic = -F.jacobian(X_p_symbols)
        
        # B = -∂F/∂X (negative Jacobian with respect to current variables)
        print("Computing B matrix...")
        B_symbolic = F.jacobian(X_symbols)
        
        # C = -∂F/∂Z (negative Jacobian with respect to exogenous processes)
        print("Computing C matrix...")
        C_symbolic = F.jacobian(Z_symbols)
        
        print("Generating output code...")
        
        # Generate code for the Jacobian evaluation function
        function_code = [
            "import numpy as np",
            "",
            "def evaluate_jacobians(theta):",
            "    \"\"\"",
            "    Evaluates Jacobian matrices for the Klein method and VAR representation",
            "    ",
            "    Args:",
            "        theta: List or array of parameter values in the order of:",
            f"            {parameters}",
            "        ",
            "    Returns:",
            "        a: Matrix ∂F/∂X_p (Jacobian with respect to future variables)",
            "        b: Matrix -∂F/∂X (negative Jacobian with respect to current variables)",
            "        c: Matrix -∂F/∂Z (negative Jacobian with respect to exogenous processes)",
            "    \"\"\"",
            "    # Unpack parameters from theta"
        ]
        
        # Add parameter unpacking
        for i, param in enumerate(parameters):
            function_code.append(f"    {param} = theta[{i}]")
        
        # Initialize matrices
        function_code.extend([
            "",
            f"    a = np.zeros(({len(equations)}, {len(variables)}))",
            f"    b = np.zeros(({len(equations)}, {len(variables)}))",
            f"    c = np.zeros(({len(equations)}, {len(exogenous)}))"   
        ])
        
        # Add A matrix elements
        function_code.append("")
        function_code.append("    # A matrix elements")
        for i in range(A_symbolic.rows):
            for j in range(A_symbolic.cols):
                if A_symbolic[i, j] != 0:
                    expr = str(A_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        # Replace symbol with parameter name
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    a[{i}, {j}] = {expr}")
        
        # Add B matrix elements
        function_code.append("")
        function_code.append("    # B matrix elements")
        for i in range(B_symbolic.rows):
            for j in range(B_symbolic.cols):
                if B_symbolic[i, j] != 0:
                    expr = str(B_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    b[{i}, {j}] = {expr}")
        
        # Add C matrix elements
        function_code.append("")
        function_code.append("    # C matrix elements")
        for i in range(C_symbolic.rows):
            for j in range(C_symbolic.cols):
                if C_symbolic[i, j] != 0:
                    expr = str(C_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    c[{i}, {j}] = {expr}")
        
        # Return all matrices
        function_code.append("")
        function_code.append("    return a, b, c")
        
        # Join all lines to form the complete function code
        complete_code = "\n".join(function_code)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(complete_code)
            print(f"Jacobian evaluator saved to {output_file}")
        
        return complete_code

    def generate_model_structure(self):
        """
        Generates structural components potentially useful for forming a
        state-space representation:
            s_t = P @ s_{t-1} + Q @ e_t
            x_t = F @ s_t + G @ e_t
        where s_t are states and x_t = [states; controls] are all variables.

        Returns:
            Dictionary with structural components for state space setup.
        """
        print("\n--- Generating Model Structure Components ---")

        # Use the final classified variables after apply_transformation
        state_vars = self.state_variables       # Only _lag variables
        control_vars = self.control_variables   # Includes RES_*
        all_vars = self.all_variables           # states + controls in order
        shock_vars = self.varexo_list

        n_states = len(state_vars)
        n_controls = len(control_vars)
        n_vars = len(all_vars) # n_states + n_controls
        n_shocks = len(shock_vars)

        # Find indices corresponding to state/control groups
        state_indices = [all_vars.index(v) for v in state_vars]
        control_indices = [all_vars.index(v) for v in control_vars]

        print(f"Dimensions: States={n_states}, Controls={n_controls}, Total Vars={n_vars}, Shocks={n_shocks}")
        if n_vars != n_states + n_controls:
            print(f"Warning: Mismatch in variable counts: {n_vars} != {n_states} + {n_controls}")

        # --- R: Shock-to-State Mapping ---
        # Matrix mapping shocks e_t to the state variables s_t they *directly* affect
        # This primarily comes from the definition of the exogenous AR processes
        # R has shape (n_states, n_shocks)
        R_struct = np.zeros((n_states, n_shocks))
        print("\nConstructing R_struct matrix (shock -> state direct impact):")
        if hasattr(self, 'state_to_shock_map'):
            for i, state_name in enumerate(state_vars):
                # Check if this state variable has a shock mapped to it
                if state_name in self.state_to_shock_map:
                    shock_name = self.state_to_shock_map[state_name]
                    try:
                        j = shock_vars.index(shock_name)
                        R_struct[i, j] = 1.0 # Shock j directly drives state i
                        print(f"  {shock_name} -> {state_name} (R_struct[{i}, {j}] = 1.0)")
                    except ValueError:
                        print(f"  Warning: Shock {shock_name} from state_to_shock_map not found in shock list.")
                # Handle cases where the base RES process maps to the shock
                # e.g., shock_map might have SHK_X -> RES_X, but RES_X_lag is the state
                elif state_name.endswith("_lag"):
                    base_name = re.sub(r'_lag\d*$', '', state_name)
                    if base_name in self.state_to_shock_map:
                        shock_name = self.state_to_shock_map[base_name]
                        try:
                            j = shock_vars.index(shock_name)
                            # Even if shock hits RES_X, the impact on RES_X_lag state
                            # is typically through the transition, not direct impact matrix R_struct.
                            # R_struct represents the Q matrix structure in s_t = P*s_{t-1} + Q*e_t
                            # Q maps e_t to s_t. If SHK_X -> RES_X and RES_X_lag_p = RES_X,
                            # then SHK_X doesn't directly cause RES_X_lag, it causes RES_X which becomes RES_X_lag next period.
                            # However, for zero-persistence shocks where RES_X_p = SHK_X,
                            # and if RES_X were treated as a state, R would have an entry.
                            # Since only _lag are states, R_struct likely remains zero here unless
                            # the shock *directly* appears in the RES_X_lag_p equation (unusual).
                            # Let's assume R_struct is only for shocks hitting the *state* variable itself directly.
                            # We might need a separate matrix later for the full system dynamics.
                            # For now, let's stick to direct shock->state mappings.
                            # If SHK_X is mapped to RES_RS_lag in state_to_shock_map, this works.
                            # If it's mapped to RES_RS, it doesn't go in R_struct.
                            pass # Keep R_struct[i,j] = 0 unless shock maps *directly* to the _lag variable
                        except ValueError:
                            print(f"  Warning: Shock {shock_name} from state_to_shock_map (base) not found.")

        else:
            print("  Warning: state_to_shock_map not found in parser results.")


        # --- Selection Matrix for Observables ---
        # We define "observables" x_t as the combined vector [states; controls]
        # C_selection maps the state vector s_t to the state portion of x_t
        # C_selection has shape (n_vars, n_states)
        C_selection = np.zeros((n_vars, n_states))
        C_selection[state_indices, :] = np.eye(n_states) # Selects states
        print("\nConstructing C_selection matrix (selects states from state vector):")
        print(f"  Shape: {C_selection.shape}")


        # --- Direct Shock Impact on Observables ---
        # D_struct maps shocks e_t to the combined observable vector x_t = [states; controls]
        # This is important for zero-persistence shocks that affect controls contemporaneously
        # or shocks directly hitting states (captured via R_struct in the state part).
        # D_struct has shape (n_vars, n_shocks)
        D_struct = np.zeros((n_vars, n_shocks))
        # Map shocks hitting states directly (from R_struct) into the state portion of D_struct
        D_struct[state_indices, :] = R_struct
        print("\nConstructing D_struct matrix (direct shock impact on states/controls):")
        # Check for shocks directly impacting *control* variables (like RES_X if treated as control)
        # A shock SHK_X affects RES_X contemporaneously if RES_X_p = rho*RES_X + SHK_X
        # Since RES_X is a control, we need an entry in D_struct mapping SHK_X to RES_X position.
        if hasattr(self, 'state_to_shock_map'): # Re-using state_to_shock_map, assuming it maps SHK_X -> RES_X
            for control_name in control_vars:
                if control_name in self.state_to_shock_map: # If a control (like RES_X) is mapped from a shock
                    shock_name = self.state_to_shock_map[control_name]
                    try:
                        j = shock_vars.index(shock_name)       # Shock index
                        i = all_vars.index(control_name) # Control variable index in all_vars
                        D_struct[i, j] = 1.0
                        print(f"  {shock_name} -> {control_name} (Control) (D_struct[{i}, {j}] = 1.0)")
                    except ValueError:
                        print(f"  Warning: Shock {shock_name} or Control {control_name} mapping issue.")
        print(f"  Shape: {D_struct.shape}")

        # --- Store labels ---
        # Ensure labels only contain the necessary lists based on final classification
        labels = {
            'state_labels': state_vars,
            'control_labels': control_vars,
            'variable_labels': all_vars, # Combined list [states; controls]
            'shock_labels': shock_vars,
            # Include mappings if they exist
            'shock_to_state_map': getattr(self, 'shock_to_state_map', {}),
            'state_to_shock_map': getattr(self, 'state_to_shock_map', {})
        }

        # --- Store indices ---
        indices = {
            'n_states': n_states,
            'n_controls': n_controls,
            'n_vars': n_vars, # Total variables (states + controls)
            'n_shocks': n_shocks,
            # Keep old keys if needed elsewhere, but clarify meaning
            'n_endogenous': len(getattr(self, 'endogenous_states', [])),
            'n_exo_states': len(getattr(self, 'exo_with_shocks', [])) + len(getattr(self, 'exo_without_shocks', [])),
            'zero_persistence_processes': getattr(self, 'zero_persistence_processes', [])
        }


        # Note: B_structure and C_structure from the prompt seemed specific to a
        # particular state-space setup. The R_struct, C_selection, D_struct generated
        # here are more fundamental building blocks. The final state-space matrices
        # P, Q, F, G will depend on the solution matrices (self.p, self.f) and these structures.

        return {
            'indices': indices,
            'R_struct': R_struct,         # Maps shocks to states they directly drive (e.g., for Q matrix)
            'C_selection': C_selection,   # Selects states from state vector (e.g., for F matrix)
            'D_struct': D_struct,         # Maps shocks to *all* variables they directly drive (e.g., for G matrix)
            'labels': labels
        }

    @staticmethod
    def parse_and_generate_files(dynare_file, output_dir):
        """Run the parser and generate all required files for later use"""
        parser = DynareParser(dynare_file)
        # Parse model (implicitly calls apply_transformation)
        model_json_data = parser.parse() # Use the dictionary returned by parse

        # Save JSON (already done by parse if you modify it, or save here)
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "model.json")
        with open(json_path, 'w') as f:
            # Save the relevant parts from model_json_data or parser attributes
            json.dump({
                'parameters': parser.parameters.keys().tolist(),
                'param_values': parser.parameters.to_dict(),
                'state_variables': parser.state_variables,
                'control_variables': parser.control_variables,
                'all_variables': parser.all_variables,
                'shocks': parser.varexo_list,
                'equations': model_json_data.get('equations', []) # Get equations if parse returns them
                # Add other relevant info if needed
            }, f, indent=2)
        print(f"Model JSON saved to {json_path}")


        # Generate Jacobian file
        parser.generate_jacobian_evaluator(os.path.join(output_dir, "jacobian_evaluator.py"))

        # --- Generate structure file with correct timing ---
        structure = parser.generate_model_structure() # Call the updated method
        structure_path = os.path.join(output_dir, "model_structure.py")
        with open(structure_path, 'w') as f:
            f.write("import numpy as np\n\n")
            # Use repr for cleaner output of lists/dicts/arrays
            f.write(f"indices = {repr(structure['indices'])}\n\n")
            # Use np.array representation for arrays
            f.write(f"R_struct = np.array({repr(structure['R_struct'].tolist())})\n\n")
            f.write(f"C_selection = np.array({repr(structure['C_selection'].tolist())})\n\n")
            f.write(f"D_struct = np.array({repr(structure['D_struct'].tolist())})\n\n")
            f.write(f"# Note: R_struct maps shocks to states directly hit.\n")
            f.write(f"# C_selection selects states from the state vector.\n")
            f.write(f"# D_struct maps shocks to the combined variable vector [states; controls] they directly hit.\n\n")
            f.write(f"labels = {repr(structure['labels'])}\n")
        print(f"Model structure saved to {structure_path}")

        print(f"All model files generated in {output_dir}")
        # Return the path or data if needed
        return output_dir
    
