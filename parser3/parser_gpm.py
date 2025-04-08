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
        Create a comprehensive transformation plan for all variables in the model.
        
        Args:
            variable_shifts: Dictionary mapping each variable to its set of time shifts
        
        Returns:
            transformation_map: Dictionary mapping original variable expressions to transformed names
            aux_equations: List of auxiliary equations needed
            model_variables: Dictionary of all variable types in the transformed model
        """
        transformation_map = {}  # Maps original expressions to transformed variable names
        aux_equations = []       # List of auxiliary equations
        processed_aux_eqs = set() # Track which auxiliary equations have been added
        
        # Track different types of variables in the transformed model
        model_variables = {
            'state_variables': set(),     # Variables with _lag suffix
            'control_variables': set(),   # Non-state, non-exogenous, non-future variables
            'aux_variables': set(),       # Auxiliary variables
            'all_variables': set(),       # All variables in transformed model
            'lead_variables': set()       # Variables with _lead suffix
        }
        
        # For each variable, create transformation rules and auxiliary equations
        for var_name, shifts in variable_shifts.items():
            # Skip exogenous variables (shocks) in the main variable lists
            if var_name in self.varexo_list:
                continue
                
            # Process all time shifts for this variable
            for shift in sorted(shifts):  # Sort shifts to ensure consistent processing
                # Current period (shift = 0)
                if shift == 0:
                    # No transformation needed
                    transformation_map[var_name] = var_name
                    model_variables['all_variables'].add(var_name)
                    
                    # Add to control variables (unless it's an exogenous shock)
                    if var_name not in self.varexo_list:
                        model_variables['control_variables'].add(var_name)
                
                # One-period lead (shift = 1)
                elif shift == 1:
                    orig_expr = f"{var_name}(+1)"
                    transformed_var = f"{var_name}_p"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    # Note: _p variables are not considered control variables
                
                # One-period lag (shift = -1)
                elif shift == -1:
                    orig_expr = f"{var_name}(-1)"
                    transformed_var = f"{var_name}_lag"  # Standard naming convention for first lag
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    model_variables['state_variables'].add(transformed_var)
                    
                    # Add auxiliary equation for the lag if not already added
                    aux_eq = f"{transformed_var}_p = {var_name}"
                    if aux_eq not in processed_aux_eqs:
                        aux_equations.append(aux_eq)
                        processed_aux_eqs.add(aux_eq)
                    model_variables['aux_variables'].add(transformed_var)
                
                # Multi-period leads (shift > 1)
                elif shift > 1:
                    # Define direct transformation for the equation
                    orig_expr = f"{var_name}(+{shift})"
                    transformed_var = f"{var_name}_lead{shift}"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    model_variables['lead_variables'].add(transformed_var)
                    model_variables['control_variables'].add(transformed_var)
                    
                    # Generate auxiliary variables and equations for all leads
                    # For lead variables, we need: var_lead = var_p, var_lead2 = var_lead_p, etc.
                    for i in range(1, shift + 1):
                        # Add to all variables
                        lead_var = f"{var_name}_lead{i}"
                        model_variables['all_variables'].add(lead_var)
                        model_variables['lead_variables'].add(lead_var)
                        model_variables['control_variables'].add(lead_var)
                        
                        # Create auxiliary equations according to the pattern:
                        # var_lead = var_p
                        # var_lead2 = var_lead_p
                        # var_lead3 = var_lead2_p
                        if i == 1:
                            # First lead relates to the base variable
                            aux_eq = f"{lead_var} = {var_name}_p"
                        else:
                            # Higher leads relate to the previous lead
                            prev_lead = f"{var_name}_lead{i-1}"
                            aux_eq = f"{lead_var} = {prev_lead}_p"
                        
                        # Only add if not already processed
                        if aux_eq not in processed_aux_eqs:
                            aux_equations.append(aux_eq)
                            processed_aux_eqs.add(aux_eq)
                            model_variables['aux_variables'].add(lead_var)
                
                # Multi-period lags (shift < -1)
                elif shift < -1:
                    abs_shift = abs(shift)
                    orig_expr = f"{var_name}({shift})"
                    transformed_var = f"{var_name}_lag{abs_shift}"
                    transformation_map[orig_expr] = transformed_var
                    model_variables['all_variables'].add(transformed_var)
                    model_variables['state_variables'].add(transformed_var)
                    
                    # Generate auxiliary variables and equations for all lags
                    # We need to ensure proper chain of auxiliary equations:
                    # var_lag_p = var
                    # var_lag2_p = var_lag
                    # var_lag3_p = var_lag2
                    
                    # First make sure we have the first lag defined
                    first_lag_var = f"{var_name}_lag"  # Standard name for first lag
                    model_variables['all_variables'].add(first_lag_var)
                    model_variables['state_variables'].add(first_lag_var)
                    
                    first_lag_eq = f"{first_lag_var}_p = {var_name}"
                    if first_lag_eq not in processed_aux_eqs:
                        aux_equations.append(first_lag_eq)
                        processed_aux_eqs.add(first_lag_eq)
                        model_variables['aux_variables'].add(first_lag_var)
                    
                    # Now create higher-order lags recursively
                    prev_lag_var = first_lag_var  # Start with the first lag
                    for i in range(2, abs_shift + 1):
                        curr_lag_var = f"{var_name}_lag{i}"
                        model_variables['all_variables'].add(curr_lag_var)
                        model_variables['state_variables'].add(curr_lag_var)
                        
                        # This is the key correction: curr_lag_p = prev_lag
                        aux_eq = f"{curr_lag_var}_p = {prev_lag_var}"
                        
                        if aux_eq not in processed_aux_eqs:
                            aux_equations.append(aux_eq)
                            processed_aux_eqs.add(aux_eq)
                            model_variables['aux_variables'].add(curr_lag_var)
                        
                        # Update prev_lag for next iteration
                        prev_lag_var = curr_lag_var
        
        return transformation_map, aux_equations, model_variables

    def apply_transformation(self):
        """
        Two-pass transformation of the model with proper handling of exogenous processes:
        1. Analyze all variables and their time shifts
        2. Create a comprehensive transformation plan
        3. Apply transformations consistently across all equations
        4. Update model variables and add auxiliary equations
        5. Correctly analyze exogenous processes with proper timing
        
        Returns:
            Dictionary with transformed model information
        """
        # First pass: Analyze variables and their shifts
        variable_shifts, all_variables = self.analyze_model_variables()
        
        # Create transformation plan based on the analysis
        transformation_map, aux_equations, model_variables = self.create_transformation_plan(variable_shifts)
        
        # Apply transformations to all equations
        transformed_equations = []
        for i, equation in enumerate(self.equations):
            # Remove comments and clean up
            clean_eq = re.sub(r'//.*', '', equation).strip()
            transformed_eq = clean_eq
            
            # Process each variable in the equation
            for var_name in all_variables:
                # Replace var(+1) with var_p
                transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*\+1\s*\)', f'{var_name}_p', transformed_eq)
                
                # Replace var(-1) with var_lag
                transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*-1\s*\)', f'{var_name}_lag', transformed_eq)
                
                # Replace var(+n) with var_leadn for n > 1
                for j in range(2, 10):  # Assume no leads greater than +9
                    transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*\+{j}\s*\)', f'{var_name}_lead{j}', transformed_eq)
                
                # Replace var(-n) with var_lagn for n > 1
                for j in range(2, 10):  # Assume no lags greater than -9
                    transformed_eq = re.sub(rf'{re.escape(var_name)}\(\s*-{j}\s*\)', f'{var_name}_lag{j}', transformed_eq)
            
            transformed_equations.append(transformed_eq)
        
        # Update class properties
        self.transformed_equations = transformed_equations
        self.auxiliary_equations = aux_equations
        
        # Remove _p variables and shock variables from main model variables
        endogenous_vars = set([v for v in model_variables['all_variables'] 
                            if not v.endswith('_p') and v not in self.varexo_list])
        
        # Identify all state variables
        all_state_variables = list(set(model_variables['state_variables']))

        # Analyze exogenous processes with correct timing convention
        exogenous_process_info = self.analyze_exogenous_processes_with_correct_timing()
        
        # Update state variables with exogenous process classification
        self.endogenous_states = exogenous_process_info['endogenous_states']
        self.exo_with_shocks = exogenous_process_info['exo_with_shocks']
        self.exo_without_shocks = exogenous_process_info['exo_without_shocks']
        self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
        self.shock_to_state_map = exogenous_process_info['shock_to_state_map']
        self.state_to_shock_map = exogenous_process_info['state_to_shock_map']
        self.zero_persistence_processes = exogenous_process_info['zero_persistence_processes']
        
        # Control variables are endogenous variables that are not state variables
        self.control_variables = list(endogenous_vars - set(self.state_variables))
        
        # Remove shock variables from all_variables
        self.all_variables = list(endogenous_vars)
        
        # Keep auxiliary variables as defined
        self.auxiliary_variables = list(set(model_variables['aux_variables']))
        
        # Format equations for output
        formatted_equations = self.format_transformed_equations(transformed_equations, aux_equations)
        
        return {
            'equations': formatted_equations,
            'state_variables': self.state_variables,
            'control_variables': self.control_variables,
            'auxiliary_variables': self.auxiliary_variables,
            'all_variables': self.all_variables,
            'endogenous_states': self.endogenous_states,
            'exo_with_shocks': self.exo_with_shocks,
            'exo_without_shocks': self.exo_without_shocks,
            'shock_to_state_map': self.shock_to_state_map,
            'state_to_shock_map': self.state_to_shock_map,
            'zero_persistence_processes': self.zero_persistence_processes
        }

    def analyze_exogenous_processes_with_correct_timing(self):
        """
        Analyze exogenous processes in the model with the correct timing convention.
        This implementation follows the state space specification document, where:
        
        Even if exogenous processes are written in Dynare as:
            z_t = ρ * z_t-1 + ε_t
        They should be interpreted in the state space framework as:
            z_t+1 = ρ * z_t + ε_t+1
            
        This timing convention ensures that shocks at time t move z_t and 
        all other variables in the system contemporaneously.
        
        Returns:
            Dictionary with exogenous process analysis results
        """
        print("\n--- Analyzing Exogenous Processes with Correct Timing ---")
        
        # Initialize mapping dictionaries
        shock_to_state_map = {}
        state_to_shock_map = {}
        zero_persistence_processes = []
        
        # Identify all exogenous processes (variables starting with "RES_")
        exo_processes = {}  # Maps base names to lists of their lag variables
        for var in self.state_variables:
            if var.startswith("RES_") and "_lag" in var:
                # Extract base name (e.g., "RES_RS" from "RES_RS_lag")
                base_name = var.split("_lag")[0]
                
                # Extract lag number
                if var.endswith("_lag"):
                    lag = 1
                else:
                    lag_suffix = var.split("_lag")[1]
                    lag = int(lag_suffix) if lag_suffix.isdigit() else 1
                
                # Add to processes dictionary
                if base_name not in exo_processes:
                    exo_processes[base_name] = []
                exo_processes[base_name].append((lag, var))
        
        # Sort each process's lags
        for process in exo_processes.values():
            process.sort()
        
        print(f"Found {len(exo_processes)} exogenous processes:")
        for base, lags in exo_processes.items():
            lag_vars = [var for _, var in lags]
            print(f"  {base}: {lag_vars}")
        
        # Find exogenous process definitions in equations
        exo_process_equations = {}
        for eq_idx, equation in enumerate(self.equations):
            clean_eq = re.sub(r'//.*', '', equation).strip()
            if "=" not in clean_eq:
                continue
                
            left, right = [s.strip() for s in clean_eq.split("=", 1)]
            
            # Check if this equation defines an exogenous process
            for base_name in exo_processes.keys():
                if base_name == left:
                    exo_process_equations[base_name] = {
                        'equation': clean_eq,
                        'right_side': right,
                        'index': eq_idx
                    }
                    print(f"Found exogenous process equation: {clean_eq}")
        
        # Analyze each exogenous process equation
        for base_name, eq_info in exo_process_equations.items():
            right_side = eq_info['right_side']
            
            # Check for AR parameters and their values
            ar_params = re.findall(r'(rho_\w+)\s*\*', right_side)
            print(f"  Process {base_name} AR parameters: {ar_params}")
            
            # Check if this is a zero-persistence process
            is_zero_persistence = True
            for param in ar_params:
                if param in self.parameters and abs(self.parameters[param]) > 1e-10:
                    is_zero_persistence = False
                    break
            
            if is_zero_persistence:
                print(f"  Process {base_name} has ZERO PERSISTENCE")
                zero_persistence_processes.append(base_name)
            
            # Find shocks that appear in this equation
            for shock in self.varexo_list:
                if shock in right_side:
                    print(f"  Process {base_name} is driven by shock {shock}")
                    
                    # FIXED TIMING: Map shock to the CURRENT variable, not its lag
                    # This follows the specification document's convention where:
                    # z_t+1 = ρ * z_t + ε_t+1
                    # The shock at time t affects z_t directly, not z_t+1 
                    if base_name in exo_processes and exo_processes[base_name]:
                        # Get the base name itself, which will be a state variable
                        base_var = base_name
                        
                        # For the lag state mapping, use the first lag
                        first_lag = exo_processes[base_name][0][1]
                        
                        # Map shock to current exogenous variable
                        shock_to_state_map[shock] = first_lag
                        state_to_shock_map[first_lag] = shock
                        
                        print(f"  Mapped shock {shock} to state {first_lag} (correct timing)")
        
        # Categorize state variables
        endogenous_states = []
        exo_with_shocks = []
        exo_without_shocks = []
        
        for var in self.state_variables:
            if var.startswith("RES_") and "_lag" in var:
                # Exogenous state
                if var in state_to_shock_map:
                    exo_with_shocks.append(var)
                else:
                    exo_without_shocks.append(var)
            else:
                # Endogenous state
                endogenous_states.append(var)
        
        # Return categorized results
        return {
            'endogenous_states': endogenous_states,
            'exo_with_shocks': exo_with_shocks,
            'exo_without_shocks': exo_without_shocks,
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
        Generate the structural components of the state space representation
        that correctly implements the timing convention from the specification document.
        
        Returns:
            Dictionary with structural components for state space
        """
        print("\n--- Generating Model Structure with Correct Timing ---")
        
        # Calculate dimensions
        n_endogenous = len(self.endogenous_states)
        n_exo_with_shocks = len(self.exo_with_shocks)
        n_exo_without_shocks = len(self.exo_without_shocks)
        n_controls = len(self.control_variables)
        n_shocks = len(self.varexo_list)
        n_exo_states = n_exo_with_shocks + n_exo_without_shocks
        n_states = n_endogenous + n_exo_states
        
        print(f"Dimensions: {n_endogenous} endogenous + {n_exo_states} exogenous = {n_states} total states")
        print(f"Controls: {n_controls}, Shocks: {n_shocks}")
        
        # Create shock selection matrix R that maps shocks to exogenous states
        # KEY FIX: This implements the correct timing convention where shocks at t
        # affect exogenous processes at t, not t+1
        R = np.zeros((n_exo_states, n_shocks))
        
        # Fill R matrix using shock-to-state mapping
        print("\nConstructing R matrix (shock -> exogenous state):")
        for shock_idx, shock_name in enumerate(self.varexo_list):
            if shock_name in self.shock_to_state_map:
                state_var = self.shock_to_state_map[shock_name]
                
                try:
                    # Find position in state vector
                    state_idx = self.state_variables.index(state_var)
                    # Calculate position relative to exogenous state section
                    exo_idx = state_idx - n_endogenous
                    
                    if 0 <= exo_idx < n_exo_states:
                        R[exo_idx, shock_idx] = 1.0
                        print(f"  {shock_name} -> {state_var} (R[{exo_idx}, {shock_idx}] = 1.0)")
                    else:
                        print(f"  Error: Invalid exo_idx {exo_idx} for {state_var}")
                except ValueError:
                    print(f"  Error: State {state_var} not found in state_variables")
        
        # Structures defining state-space matrices with correct timing
        # B_structure maps shocks to state transitions but with CORRECT TIMING
        B_structure = np.zeros((n_states, n_shocks))
        
        # Fill in the exogenous part of B - maps shocks to exogenous states
        B_structure[n_endogenous:, :] = R
        
        # C_structure maps states to observables
        C_structure = np.zeros((n_controls + n_states, n_states))
        
        # States appear directly in observables
        C_structure[n_controls:, :] = np.eye(n_states)
        
        # D_structure maps shocks directly to observables
        # KEY FIX: With the correct timing convention, shocks at t directly affect observables at t
        D_structure = np.zeros((n_controls + n_states, n_shocks))
        
        # Direct shock to exogenous state mapping
        # This is crucial for zero-persistence processes
        D_structure[n_controls + n_endogenous:, :] = R
        
        # Handle zero-persistence processes specially
        # For zero-persistence processes, shocks directly affect controls
        if hasattr(self, 'zero_persistence_processes') and self.zero_persistence_processes:
            print("\nHandling zero-persistence processes:")
            
            # For each zero-persistence process
            for process_name in self.zero_persistence_processes:
                print(f"  Processing zero-persistence process: {process_name}")
                
                # Find which shock drives this process
                for shock_name, state_var in self.shock_to_state_map.items():
                    if state_var.startswith(f"{process_name}_lag"):
                        print(f"    Process {process_name} is driven by shock {shock_name}")
                        
                        # Find which control variables depend on this process
                        for i, control_var in enumerate(self.control_variables):
                            # Check each equation to see if this control depends on the process
                            for equation in self.equations:
                                clean_eq = re.sub(r'//.*', '', equation).strip()
                                
                                # If equation defines this control and contains the process
                                if "=" in clean_eq and control_var in clean_eq.split("=")[0].strip() and process_name in clean_eq:
                                    shock_idx = self.varexo_list.index(shock_name)
                                    print(f"    Control {control_var} depends on {process_name} - adding direct shock effect")
                                    
                                    # Add direct shock effect to control variable
                                    # This is the key fix for zero-persistence processes
                                    D_structure[i, shock_idx] = 1.0
        
        # Store indices for later use
        indices = {
            'n_endogenous': n_endogenous,
            'n_exo_states': n_exo_states,
            'n_controls': n_controls,
            'n_shocks': n_shocks,
            'n_states': n_states,
            'n_observables': n_controls + n_states,
            'zero_persistence_processes': self.zero_persistence_processes if hasattr(self, 'zero_persistence_processes') else []
        }
        
        # Store labels and mappings
        labels = {
            'state_labels': self.state_variables,
            'observable_labels': self.control_variables + self.state_variables,
            'shock_labels': self.varexo_list,
            'shock_to_state_map': self.shock_to_state_map,
            'state_to_shock_map': self.state_to_shock_map,
            'zero_persistence_processes': self.zero_persistence_processes if hasattr(self, 'zero_persistence_processes') else []
        }
        
        return {
            'indices': indices,
            'R': R,
            'B_structure': B_structure,
            'C_structure': C_structure,
            'D_structure': D_structure,
            'labels': labels
        }

    @staticmethod
    def parse_and_generate_files(dynare_file, output_dir):
        """Run the parser and generate all required files for later use"""
        # 1. Parse model and save JSON
        parser = DynareParser(dynare_file)
        model_json = parser.save_json(os.path.join(output_dir, "model.json"))

        # 2. Generate Jacobian file
        parser.generate_jacobian_evaluator(os.path.join(output_dir, "jacobian_evaluator.py"))

        # 3. Generate structure file with correct timing
        structure = parser.generate_model_structure()
        with open(os.path.join(output_dir, "model_structure.py"), 'w') as f:
            f.write("import numpy as np\n\n")
            f.write(f"indices = {repr(structure['indices'])}\n\n")
            f.write(f"R = np.array({repr(structure['R'].tolist())})\n\n")
            f.write(f"B_structure = np.array({repr(structure['B_structure'].tolist())})\n\n")
            f.write(f"C_structure = np.array({repr(structure['C_structure'].tolist())})\n\n")
            f.write(f"D = np.array({repr(structure['D_structure'].tolist())})\n\n")
            f.write(f"labels = {repr(structure['labels'])}\n")

        print(f"All model files generated in {output_dir} with correct timing convention")



#!/usr/bin/env python3

"""
Test script for the fixed DynareParser implementation.
This script tests the parser with a focus on correct timing convention for exogenous variables.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("\n===== Testing Fixed DynareParser Implementation =====\n")
    
    # Define input and output paths
    dynare_file = "qpm_simpl1.dyn"  # Input Dynare file
    output_dir = "model_files_test"  # Directory for generated files
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import the fixed parser
        # Adjust this import based on where you've saved the fixed parser
       # from fixed_parser import DynareParser
        
        # Step 1: Parse the Dynare file
        print("\n=== Step 1: Parsing Dynare File ===")
        parser = DynareParser(dynare_file)
        parser.read_dynare_file()
        parser.parse_variables()
        parser.parse_exogenous()
        parser.parse_parameters()
        parser.parse_model()
        
        print(f"Successfully parsed {dynare_file}")
        print(f"Found {len(parser.var_list)} variables, {len(parser.varexo_list)} exogenous variables, and {len(parser.parameters)} parameters")
        
        # Step 2: Display model equations
        print("\n=== Step 2: Original Model Equations ===")
        for i, eq in enumerate(parser.equations):
            print(f"Equation {i+1}: {eq}")
        
        # Step 3: Apply transformation with correct timing
        print("\n=== Step 3: Applying Transformation with Correct Timing ===")
        transformed_model = parser.apply_transformation()
        
        # Step 4: Display state and control variables
        print("\n=== Step 4: Variable Classification ===")
        print("\nState Variables:")
        print("  Endogenous states:", transformed_model['endogenous_states'])
        print("  Exogenous states with shocks:", transformed_model['exo_with_shocks'])
        print("  Exogenous states without shocks:", transformed_model['exo_without_shocks'])
        print("\nControl Variables:")
        print("  ", transformed_model['control_variables'])
        print("\nAuxiliary Variables:")
        print("  ", transformed_model['auxiliary_variables'])
        
        # Step 5: Check timing convention
        print("\n=== Step 5: Verifying Exogenous Process Timing ===")
        print("Shock to State Mapping (should map shocks to exogenous states at time t):")
        for shock, state in transformed_model['shock_to_state_map'].items():
            print(f"  {shock} -> {state}")
        
        # Check for zero-persistence processes
        if 'zero_persistence_processes' in transformed_model:
            print("\nZero-Persistence Processes:")
            print("  ", transformed_model['zero_persistence_processes'])
        
        # Step 6: Generate model structure with correct timing
        print("\n=== Step 6: Generating Model Structure ===")
        structure = parser.generate_model_structure()
        
        # Display R matrix (shock to exogenous state mapping)
        print("\nR Matrix (shock to exogenous state mapping):")
        print(structure['R'])
        
        # Display D matrix (direct shock effects)
        print("\nD Matrix (direct shock effects):")
        print(structure['D_structure'])
        
        # Step 7: Generate and save all required files
        print("\n=== Step 7: Generating Output Files ===")
        model_json = parser.save_json(os.path.join(output_dir, "model.json"))
        
        # Generate Jacobian evaluator
        jacobian_code = parser.generate_jacobian_evaluator(os.path.join(output_dir, "jacobian_evaluator.py"))
        print(f"Generated Jacobian evaluator at {os.path.join(output_dir, 'jacobian_evaluator.py')}")
        
        # Generate model structure
        structure = parser.generate_model_structure()
        with open(os.path.join(output_dir, "model_structure.py"), 'w') as f:
            f.write("import numpy as np\n\n")
            f.write(f"indices = {repr(structure['indices'])}\n\n")
            f.write(f"R = np.array({repr(structure['R'].tolist())})\n\n")
            f.write(f"B_structure = np.array({repr(structure['B_structure'].tolist())})\n\n")
            f.write(f"C_structure = np.array({repr(structure['C_structure'].tolist())})\n\n")
            f.write(f"D = np.array({repr(structure['D_structure'].tolist())})\n\n")
            f.write(f"labels = {repr(structure['labels'])}\n")
        print(f"Generated model structure at {os.path.join(output_dir, 'model_structure.py')}")
        
        print("\n=== Test Summary ===")
        print(f"Successfully parsed {dynare_file} with correct timing convention")
        print(f"Generated output files in {output_dir}")
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())