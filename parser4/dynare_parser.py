#!/usr/bin/env python3
# dynare_parser.py
# Parser for Dynare model files, transforming them for use with Klein's solution method

import re
import os
import json
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


class DynareParser:
    """
    Parser for Dynare model files, converting them to a format suitable 
    for Klein's solution method with proper handling of leads and lags.
    """
    
    def __init__(self, input_file):
        """Initialize the parser with the input Dynare file path."""
        self.input_file = input_file
        self.content = ""
        self.clean_content = ""
        self.var_list = []
        self.varexo_list = []
        self.parameters = {}
        self.param_values = {}
        self.original_equations = []
        self.equations_with_timing = []
        self.auxiliary_vars = []
        self.auxiliary_eqs = []
        self.final_equations = []
        self.state_variables = []
        self.control_variables = []
        self.shock_to_process_var_map = {}
        self.var_lead_lag_info = {}
        
    def read_file(self):
        """Read the Dynare file and store its content."""
        try:
            with open(self.input_file, 'r') as f:
                self.content = f.read()
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    def clean_file(self, out_folder):
        """Clean the file content from comments and empty lines."""
        # Remove block comments
        clean_content = re.sub(r'/\*.*?\*/', '', self.content, flags=re.DOTALL)
        # Remove line comments
        clean_content = re.sub(r'//.*|%.*', '', clean_content)
        # Normalize whitespace
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        self.clean_content = clean_content
        
        # Save clean content to file
        full_path = os.path.join(out_folder, "clean_file.txt")
        with open(full_path, 'w') as f:
            f.write(self.clean_content)
        
        return True
    
    def extract_declarations(self):
        """Extract variable, shock, and parameter declarations from the file."""
        # Extract var declarations
        var_match = re.search(r'var\s+(.*?);', self.clean_content, re.DOTALL)
        if var_match:
            var_block = var_match.group(1)
            # Remove comments within the var block if any remain
            var_block = re.sub(r'//.*|%.*', '', var_block)
            # Extract variable names
            self.var_list = [v.strip() for v in re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', var_block)]
            
        # Extract varexo declarations
        varexo_match = re.search(r'varexo\s+(.*?);', self.clean_content, re.DOTALL)
        if varexo_match:
            varexo_block = varexo_match.group(1)
            varexo_block = re.sub(r'//.*|%.*', '', varexo_block)
            self.varexo_list = [v.strip() for v in re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', varexo_block)]
            
        # Extract parameters
        param_match = re.search(r'parameters\s+(.*?);', self.clean_content, re.DOTALL)
        if param_match:
            param_block = param_match.group(1)
            param_block = re.sub(r'//.*|%.*', '', param_block)
            self.parameters = [p.strip() for p in re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', param_block)]
        
        # Extract parameter values
        for param in self.parameters:
            param_value_match = re.search(rf'{param}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?);', self.clean_content)
            if param_value_match:
                self.param_values[param] = float(param_value_match.group(1))
            else:
                # If no value found, set to NaN
                self.param_values[param] = float('nan')
                
        return True
    
    def extract_model(self, out_folder):
        """Extract model equations from the Dynare file."""
        model_match = re.search(r'model\s*;(.*?)end\s*;', self.clean_content, re.DOTALL)
        if model_match:
            model_block = model_match.group(1).strip()
            # Split into equations
            equations = [eq.strip() for eq in model_block.split(';') if eq.strip()]
            self.original_equations = equations
            
            # Save original equations to file
            full_path = os.path.join(out_folder, "clean_file.txt")
            with open(full_path, 'w') as f:
                f.write("Parameters:\n")
                for param, value in self.param_values.items():
                    f.write(f"{param} = {value};\n")
                f.write("\nVariables:\n")
                f.write(", ".join(self.var_list) + "\n")
                f.write("\nShocks:\n")
                f.write(", ".join(self.varexo_list) + "\n")
                f.write("\nModel Equations:\n")
                for eq in self.original_equations:
                    f.write(eq + ";\n")
                    
        return True
    
    def identify_shock_equations(self, out_folder):
        """Identify which equations contain shocks and their associated variables."""
        for eq in self.original_equations:
            # Skip equations without =
            if "=" not in eq:
                continue
                
            left, right = [s.strip() for s in eq.split("=", 1)]
            
            # Find all shocks present in this equation
            present_shocks = [shock for shock in self.varexo_list if re.search(rf'\b{re.escape(shock)}\b', right)]
            
            if len(present_shocks) == 1:
                # This is an exogenous process equation
                shock = present_shocks[0]
                variable = left.split("(")[0].strip() if "(" in left else left
                self.shock_to_process_var_map[shock] = variable
                
                # Check if this is the exogenous process equation and lead it forward
                # Use a better regex pattern to find lag terms of the form variable(-n)
                lag_pattern = rf'{re.escape(variable)}\s*\(\s*-\d+\s*\)'
                if variable in self.var_list and re.search(lag_pattern, right):
                    # This is an exogenous process, lead it forward
                    print(f"Leading forward exogenous process equation: {eq}")
                    new_eq = self._lead_forward_equation(eq)
                    self.equations_with_timing.append(new_eq)
                else:
                    self.equations_with_timing.append(eq)
            else:
                # Regular equation, keep as is
                self.equations_with_timing.append(eq)
                
        # Save equations with correct timing
        full_path = os.path.join(out_folder, "clean_file_with_correct_timing.txt")
        with open(full_path, 'w') as f:
            for eq in self.equations_with_timing:
                f.write(eq + ";\n")
                
        return True
    
    def _lead_forward_equation(self, equation):
        """Lead forward an exogenous process equation by one period."""
        if "=" not in equation:
            return equation
            
        left, right = [s.strip() for s in equation.split("=", 1)]
        variable = left.split("(")[0].strip() if "(" in left else left
        
        # Modify left side to add (+1)
        new_left = f"{variable}(+1)"
        
        # Modify right side: replace var(-n) with var(-(n-1))
        new_right = right
        
        # Find all instances of var(-n) and replace them
        # More robust pattern that ensures we only match the specific variable name
        lag_pattern = re.compile(rf'\b({re.escape(variable)})\(\s*-(\d+)\s*\)')
        
        matches = list(lag_pattern.finditer(right))
        # Process in reverse order to avoid changing positions as we replace
        for match in reversed(matches):
            full_match = match.group(0)
            var_name = match.group(1)
            lag = int(match.group(2))
            
            if lag == 1:
                # var(-1) -> var
                replacement = var_name
            else:
                # var(-n) -> var(-(n-1))
                replacement = f"{var_name}(-{lag-1})"
                
            # Replace just this instance at the specific position
            start, end = match.span()
            new_right = new_right[:start] + replacement + new_right[end:]
            
        # Debug output
        print(f"  Original: {left} = {right}")
        print(f"  Led forward: {new_left} = {new_right}")
            
        return f"{new_left} = {new_right}"
    
    def analyze_variable_leads_lags(self):
        """Analyze all equations to identify the maximum lead and lag for each variable."""
        for var in self.var_list:
            self.var_lead_lag_info[var] = {'max_lead': 0, 'max_lag': 0}
            
        for eq in self.equations_with_timing:
            for var in self.var_list:
                # Use word boundary \b to ensure we match complete variable names
                # This prevents partial matches of variable names that are substrings of others
                lead_pattern = re.compile(rf'\b{re.escape(var)}\(\s*\+(\d+)\s*\)')
                lag_pattern = re.compile(rf'\b{re.escape(var)}\(\s*-(\d+)\s*\)')
                
                # Check for leads
                for match in lead_pattern.finditer(eq):
                    lead = int(match.group(1))
                    if lead > self.var_lead_lag_info[var]['max_lead']:
                        self.var_lead_lag_info[var]['max_lead'] = lead
                        print(f"Found lead {lead} for variable {var} in equation: {eq}")
                        
                # Check for lags
                for match in lag_pattern.finditer(eq):
                    lag = int(match.group(1))
                    if lag > self.var_lead_lag_info[var]['max_lag']:
                        self.var_lead_lag_info[var]['max_lag'] = lag
                        print(f"Found lag {lag} for variable {var} in equation: {eq}")
        
        # Print summary of leads and lags           
        print("\nVariable lead/lag summary:")
        for var, info in self.var_lead_lag_info.items():
            if info['max_lead'] > 0 or info['max_lag'] > 0:
                print(f"  {var}: max_lead={info['max_lead']}, max_lag={info['max_lag']}")
                        
        return True
    
    def generate_auxiliary_variables(self, out_folder):
        """Generate auxiliary variables and equations for leads and lags."""
        for var, info in self.var_lead_lag_info.items():
            # Generate auxiliary variables for lags
            if info['max_lag'] > 0:
                # Create state variable for the first lag
                lag_var = f"{var}_lag"
                if lag_var not in self.auxiliary_vars:
                    self.auxiliary_vars.append(lag_var)
                    self.state_variables.append(lag_var)
                    # Add auxiliary equation: lag_var_p = var
                    self.auxiliary_eqs.append(f"{lag_var}_p = {var}")
                
                # Create additional lag variables if needed
                prev_lag_var = lag_var
                for i in range(2, info['max_lag'] + 1):
                    curr_lag_var = f"{var}_lag{i}"
                    if curr_lag_var not in self.auxiliary_vars:
                        self.auxiliary_vars.append(curr_lag_var)
                        self.state_variables.append(curr_lag_var)
                        # Add auxiliary equation: lag_var{i}_p = lag_var{i-1}
                        self.auxiliary_eqs.append(f"{curr_lag_var}_p = {prev_lag_var}")
                    prev_lag_var = curr_lag_var
            
            # Generate auxiliary variables for leads
            if info['max_lead'] > 1:  # Only need auxiliary for lead > 1
                for i in range(1, info['max_lead']):
                    lead_var = f"{var}_lead{i}"
                    if lead_var not in self.auxiliary_vars:
                        self.auxiliary_vars.append(lead_var)
                        self.control_variables.append(lead_var)
                        
                        if i == 1:
                            # Add auxiliary equation: lead_var = var_p
                            self.auxiliary_eqs.append(f"{lead_var} = {var}_p")
                        else:
                            # Add auxiliary equation: lead_var = lead_var{i-1}_p
                            self.auxiliary_eqs.append(f"{lead_var} = {var}_lead{i-1}_p")
                            
        # Save file with auxiliary variables and equations
        full_path = os.path.join(out_folder, "clean_file_with_correct_timing_and_auxiliary_variables.txt")
        with open(full_path, 'w') as f:
            f.write("Original Equations:\n")
            for eq in self.equations_with_timing:
                f.write(eq + ";\n")
            f.write("\nAuxiliary Variables:\n")
            f.write(", ".join(self.auxiliary_vars) + "\n")
            f.write("\nAuxiliary Equations:\n")
            for eq in self.auxiliary_eqs:
                f.write(eq + ";\n")
                
        return True
    
    def substitute_leads_lags(self, out_folder):
        """Replace all lead and lag notations with the appropriate auxiliary variables."""
        # Process original equations
        for eq in self.equations_with_timing:
            new_eq = eq
            
            # Process variables in order of length (longest first) to avoid substring issues
            sorted_vars = sorted(self.var_list, key=len, reverse=True)
            
            for var in sorted_vars:
                # We'll use a two-stage approach:
                # 1. First identify all matches with their positions
                # 2. Then replace them in reverse order to avoid shifting positions
                
                # --- Replace leads ---
                # First replace var(+1) with var_p
                lead1_pattern = re.compile(rf'\b{re.escape(var)}\(\s*\+1\s*\)')
                matches = list(lead1_pattern.finditer(new_eq))
                for match in reversed(matches):
                    start, end = match.span()
                    new_eq = new_eq[:start] + f"{var}_p" + new_eq[end:]
                
                # Then replace var(+n) with var_lead{n-1}_p for n > 1
                lead_pattern = re.compile(rf'\b{re.escape(var)}\(\s*\+(\d+)\s*\)')
                matches = list(lead_pattern.finditer(new_eq))
                for match in reversed(matches):
                    lead = int(match.group(1))
                    if lead > 1:
                        start, end = match.span()
                        replacement = f"{var}_lead{lead-1}_p"
                        new_eq = new_eq[:start] + replacement + new_eq[end:]
                
                # --- Replace lags ---
                # First replace var(-1) with var_lag
                lag1_pattern = re.compile(rf'\b{re.escape(var)}\(\s*-1\s*\)')
                matches = list(lag1_pattern.finditer(new_eq))
                for match in reversed(matches):
                    start, end = match.span()
                    new_eq = new_eq[:start] + f"{var}_lag" + new_eq[end:]
                
                # Then replace var(-n) with var_lag{n} for n > 1
                lag_pattern = re.compile(rf'\b{re.escape(var)}\(\s*-(\d+)\s*\)')
                matches = list(lag_pattern.finditer(new_eq))
                for match in reversed(matches):
                    lag = int(match.group(1))
                    if lag > 1:
                        start, end = match.span()
                        replacement = f"{var}_lag{lag}"
                        new_eq = new_eq[:start] + replacement + new_eq[end:]
            
            print(f"Original: {eq}")
            print(f"Substituted: {new_eq}")
            self.final_equations.append(new_eq)
            
        # Add auxiliary equations
        self.final_equations.extend(self.auxiliary_eqs)

        # Add all auxiliary variables
        updated_var_names = self.var_list
        for aux_var in self.auxiliary_vars:
            if aux_var not in updated_var_names:
                updated_var_names.append(aux_var)
        
        #Add the updated variable names to the list
        self.var_names = updated_var_names

        # Save the file with substitutions
        full_path = os.path.join(out_folder, "clean_file_with_auxiliary_variables_substituted.txt")
        with open(full_path, 'w') as f:
            for eq in self.final_equations:
                f.write(eq + ";\n")
                
        return True
    
    def classify_variables(self):
        """
        Simplified classification of variables:
        - State variables: variables with "_lag" in their name or that appear with lags in final equations
        - Controls: all other variables
        """
        # Initialize lists
        state_variables = []
        control_variables = []
        
        # Ensure we're working with the complete list of variables including auxiliaries
        # Identify variables that are states (either have "_lag" suffix or appear with lags)
        for var in self.var_names:
            # Check if this is a lag variable (by naming convention)
            if "_lag" in var:
                state_variables.append(var)
                continue
                
            # # Check if this variable appears with a lag in any final equation
            # is_state = False
            # for eq in self.final_equations:
            #     # Look for var(-1) pattern
            #     if f"{var}(-1)" in eq or f"{var}[-1]" in eq:
            #         is_state = True
            #         break                    
            # if is_state:
            #     state_variables.append(var)
            else:
                control_variables.append(var)
        
        # Set the classified variables
        self.state_variables = state_variables
        self.control_variables = control_variables
        
        # Print classification results
        print("Variable Classification:")
        print(f"  States: {len(self.state_variables)}")
        print(f"  Controls: {len(self.control_variables)}")
        
        return True
        
    def format_equations_for_json(self):
        """Format equations for JSON output in the required form."""
        formatted_equations = []
        
        for i, eq in enumerate(self.final_equations, 1):
            if "=" in eq:
                left, right = [s.strip() for s in eq.split("=", 1)]
                # Format: right - (left)
                formatted_eq = f"{right} - ({left})"
                formatted_equations.append({f"eq{i}": formatted_eq})
            else:
                # If there's no =, just use the equation as is
                formatted_equations.append({f"eq{i}": eq.strip()})
                
        return formatted_equations
    
    def generate_json_output(self, out_folder):
        """Generate JSON output with the model information."""
        # Format equations
        formatted_equations = self.format_equations_for_json()
        
        # Create the JSON structure
        model_json = {
            "parameters": self.parameters,
            "param_values": self.param_values,
            "states": self.state_variables,
            "controls": self.control_variables,
            "all_variables": self.state_variables + self.control_variables,
            "shocks": self.varexo_list,
            "equations": formatted_equations,
            "shock_to_process_var_map": self.shock_to_process_var_map
        }
        
        # Write to file
        full_path = os.path.join(out_folder, "model_json.json")
        with open(full_path, 'w') as f:
            json.dump(model_json, f, indent=2)
            
        return model_json
    
    def generate_jacobian_matrices(self, out_folder):
        """Generate Jacobian matrices A, B, C using symbolic differentiation."""
        # Create symbolic variables for parameters
        param_symbols = {param: sp.Symbol(param) for param in self.parameters}
        
        # Create symbolic variables for all model variables (at t and t+1)
        all_vars = self.state_variables + self.control_variables
        var_symbols = {var: sp.Symbol(var) for var in all_vars}
        future_var_symbols = {f"{var}_p": sp.Symbol(f"{var}_p") for var in all_vars}
        
        # Create symbolic variables for shocks
        shock_symbols = {shock: sp.Symbol(shock) for shock in self.varexo_list}
        
        # Combine all symbols
        all_symbols = {**param_symbols, **var_symbols, **future_var_symbols, **shock_symbols}
        
        # Parse equations into symbolic expressions
        symbolic_eqs = []
        
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        for eq_dict in self.format_equations_for_json():
            for eq_str in eq_dict.values():
                try:
                    # Parse the equation string into a symbolic expression
                    sym_eq = parse_expr(eq_str, local_dict=all_symbols, transformations=transformations)
                    symbolic_eqs.append(sym_eq)
                except Exception as e:
                    print(f"Error parsing equation '{eq_str}': {e}")
                    symbolic_eqs.append(sp.sympify(0))  # Use a dummy equation if parsing fails
        
        # Create list of variables in the correct order
        x_vars = [var_symbols[var] for var in all_vars]
        x_p_vars = [future_var_symbols[f"{var}_p"] for var in all_vars]
        shock_vars = [shock_symbols[shock] for shock in self.varexo_list]
        
        # Compute Jacobians
        F = sp.Matrix(symbolic_eqs)
        A = F.jacobian(x_p_vars)  # ∂equations/∂x_p
        B = -F.jacobian(x_vars)    # -∂equations/∂x
        C = -F.jacobian(shock_vars) if shock_vars else sp.zeros(len(symbolic_eqs), 0)  # -∂equations/∂shocks
        
        # Generate Python code for Jacobian evaluation
        jacobian_code = [
            "import numpy as np",
            "",
            "def evaluate_jacobians(theta):",
            "    \"\"\"",
            "    Computes the Jacobian matrices A, B, C for the model.",
            "    ",
            "    Args:",
            "        theta: Array of parameter values in the order specified by the model.",
            "    ",
            "    Returns:",
            "        A, B, C: Jacobian matrices",
            "    \"\"\"",
            "    # Unpack parameters"
        ]
        
        # Add parameter assignments
        for i, param in enumerate(self.parameters):
            jacobian_code.append(f"    {param} = theta[{i}]")
        
        jacobian_code.extend([
            "",
            f"    # Initialize matrices",
            f"    A = np.zeros(({len(symbolic_eqs)}, {len(all_vars)}))",
            f"    B = np.zeros(({len(symbolic_eqs)}, {len(all_vars)}))",
            f"    C = np.zeros(({len(symbolic_eqs)}, {len(self.varexo_list)}))",
            ""
        ])
        
        # Add code for A matrix
        jacobian_code.append("    # A matrix elements (∂equations/∂x_p)")
        for i in range(A.rows):
            for j in range(A.cols):
                if A[i, j] != 0:
                    code_str = str(A[i, j]).replace("exp", "np.exp").replace("log", "np.log").replace("sqrt", "np.sqrt")
                    jacobian_code.append(f"    A[{i}, {j}] = {code_str}")
        
        # Add code for B matrix
        jacobian_code.append("")
        jacobian_code.append("    # B matrix elements (-∂equations/∂x)")
        for i in range(B.rows):
            for j in range(B.cols):
                if B[i, j] != 0:
                    code_str = str(B[i, j]).replace("exp", "np.exp").replace("log", "np.log").replace("sqrt", "np.sqrt")
                    jacobian_code.append(f"    B[{i}, {j}] = {code_str}")
        
        # Add code for C matrix
        if C.cols > 0:
            jacobian_code.append("")
            jacobian_code.append("    # C matrix elements (-∂equations/∂shocks)")
            for i in range(C.rows):
                for j in range(C.cols):
                    if C[i, j] != 0:
                        code_str = str(C[i, j]).replace("exp", "np.exp").replace("log", "np.log").replace("sqrt", "np.sqrt")
                        jacobian_code.append(f"    C[{i}, {j}] = {code_str}")
        
        # Return statement
        jacobian_code.extend([
            "",
            "    return A, B, C"
        ])
        
        # Write to file
        full_path = os.path.join(out_folder, "jacobian_matrices.py")
        with open(full_path, 'w') as f:
            f.write('\n'.join(jacobian_code))
            
        return True
    
    def generate_model_structure(self, out_folder):
        """Generate model structure with indices and selection matrices."""
        all_vars = self.state_variables + self.control_variables
        
        # Calculate indices
        n_states = len(self.state_variables)
        n_controls = len(self.control_variables)
        n_vars = len(all_vars)
        n_shocks = len(self.varexo_list)
        
        # Create R structure (shock->state direct impact)
        R_struct = np.zeros((n_states, n_shocks))
        
        # Create C selection matrix (selects states)
        C_selection = np.zeros((n_vars, n_states))
        for i, state in enumerate(self.state_variables):
            state_idx = all_vars.index(state)
            C_selection[state_idx, i] = 1.0
            
        # Create D structure (shock->var direct impact)
        D_struct = np.zeros((n_vars, n_shocks))
        for shock, var in self.shock_to_process_var_map.items():
            if shock in self.varexo_list and var in all_vars:
                shock_idx = self.varexo_list.index(shock)
                var_idx = all_vars.index(var)
                D_struct[var_idx, shock_idx] = 1.0
                
        # Create indices dictionary
        indices = {
            'n_states': n_states,
            'n_controls': n_controls,
            'n_vars': n_vars,
            'n_shocks': n_shocks,
            'n_endogenous_states': 0,  # This would need to be calculated based on further analysis
            'n_exo_states_ws': 0,      # Same here
            'n_exo_states_wos': 0,     # Same here
            'zero_persistence_processes': []  # This would need to be calculated
        }
        
        # Create labels dictionary
        labels = {
            'state_labels': self.state_variables,
            'control_labels': self.control_variables,
            'variable_labels': all_vars,
            'shock_labels': self.varexo_list
        }
        
        # Create model structure file
        structure_code = [
            "import numpy as np",
            "",
            f"indices = {indices}",
            "",
            f"R_struct = np.array({R_struct.tolist()})",
            "",
            f"C_selection = np.array({C_selection.tolist()})",
            "",
            f"D_struct = np.array({D_struct.tolist()})",
            "",
            "# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls",
            "",
            f"labels = {labels}"
        ]
        
        # Write to file
        full_path = os.path.join(out_folder, "model_structure.py")
        with open(full_path, 'w') as f:
            f.write('\n'.join(structure_code))
            
        return True
    
    def clean_log_files(self, folder_path):
        """
        Checks if the specified folder exists. If it does, deletes the entire folder
        and all its contents recursively.

        Args:
            folder_path (str): The path to the directory to be cleaned (deleted).
        """
        import shutil
        print(f"--- Running clean_log_files for: '{folder_path}' ---")

        # 1. Check if the path exists and is a directory
        if os.path.isdir(folder_path):
            print(f"Directory '{folder_path}' exists. Proceeding with deletion...")
            try:
                # 2. Delete the directory tree
                shutil.rmtree(folder_path)
                print(f"Successfully deleted directory: '{folder_path}'")
            except OSError as e:
                # Handle potential errors during deletion (e.g., permissions, file in use)
                print(f"Error deleting directory '{folder_path}': {e}")
            except Exception as e:
                # Catch any other unexpected errors
                print(f"An unexpected error occurred while deleting '{folder_path}': {e}")
        # Optional: Check if the path exists but is not a directory
        elif os.path.exists(folder_path):
            print(f"Path '{folder_path}' exists but is NOT a directory. No action taken.")
        else:
            # 3. If the directory doesn't exist, just report it
            print(f"Directory '{folder_path}' does not exist. No action needed.")

        print(f"--- Finished clean_log_files for: '{folder_path}' ---")

    def parse(self, out_folder="model_files"):
        """Execute the full parsing process."""
        print("Starting Dynare model parsing...")
        #Step 0: Delete the output folder if it exists
        self.clean_log_files(out_folder)

        # Create output folder if it doesn't exist
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(f"Created output folder: {out_folder}")

        # Step 1: Read and clean the file
        if not self.read_file():
            return False
        if not self.clean_file(out_folder):
            return False
            
        # Step 2: Extract declarations
        if not self.extract_declarations():
            return False
            
        # Step 3: Extract model equations
        if not self.extract_model(out_folder):
            return False
            
        # Step 4: Identify and lead forward exogenous equations
        if not self.identify_shock_equations(out_folder):
            return False
            
        # Step 5: Analyze variables for leads and lags
        if not self.analyze_variable_leads_lags():
            return False
            
        # Step 6: Generate auxiliary variables and equations
        if not self.generate_auxiliary_variables(out_folder):
            return False
            
        # Step 7: Substitute leads and lags with auxiliary variables
        if not self.substitute_leads_lags(out_folder):
            return False
            
        # Step 8: Classify variables as state or control
        if not self.classify_variables():
            return False
            
        # Step 9: Generate JSON output
        model_json = self.generate_json_output(out_folder)
            
        # Step 10: Generate Jacobian matrices
        if not self.generate_jacobian_matrices(out_folder):
            return False
            
        # Step 11: Generate model structure
        if not self.generate_model_structure(out_folder):
            return False
            
        print("Dynare model parsing completed successfully!")
        return True

# def main():
#     """Main function to run the parser on a file."""
#     # import argparse
    
#     # parser = argparse.ArgumentParser(description='Parse a Dynare model file for use with Klein solution method.')
#     # parser.add_argument('file', help='The Dynare model file to parse')
#     # args = parser.parse_args()
#     import os
#     script_dir = os.path.dirname(__file__)
#     os.chdir(script_dir)
#     dynare_file = "qpm_simpl1.dyn"
#     dynare_parser = DynareParser(dynare_file)
#     success = dynare_parser.parse()
    
#     if success:
#         print("Files generated:")
#         print("  - clean_file.txt")
#         print("  - clean_file_with_correct_timing.txt")
#         print("  - clean_file_with_correct_timing_and_auxiliary_variables.txt")
#         print("  - clean_file_with_auxiliary_variables_substituted.txt")
#         print("  - model_json.json")
#         print("  - jacobian_matrices.py")
#         print("  - model_structure.py")
#     else:
#         print("Parsing failed.")
        
# if __name__ == "__main__":
#     main()