#%%
import re
import json
import os
import numpy as np
import scipy.linalg as la
import sys
import sympy as sy
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm

import numpy as np
import pandas as pd

# Load Jacobian evaluator function
import importlib

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
                    
                    # CORRECTED: Generate auxiliary variables and equations for all lags
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
        Two-pass transformation of the model with improved handling of exogenous processes:
        1. Analyze all variables and their time shifts
        2. Create a comprehensive transformation plan
        3. Apply transformations consistently across all equations
        4. Update model variables and add auxiliary equations
        5. Analyze exogenous processes and create shock-to-state mappings
        
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

        # Initialize mapping dictionaries for shock-to-state relationships
        self.shock_to_state_map = {}  # Maps shock names to state variables
        self.state_to_shock_map = {}  # Maps state variables to shock names
        
        # ---- ENHANCED CODE: Better categorization of exogenous process states ----
        # Group state variables by exogenous processes and their lags
        exo_processes = {}  # Dict mapping base names to lists of (lag, var_name)
        endogenous_states = []
        
        # First pass: Categorize state variables
        for var in all_state_variables:
            if var.startswith("RES_") and "_lag" in var:
                # This is an exogenous process state
                base_name = var.split("_lag")[0]  # Get base name before _lag
                
                # Extract lag number
                if var.endswith("_lag"):
                    lag = 1  # First lag
                else:
                    # Try to extract lag number after _lag
                    lag_suffix = var.split("_lag")[1]
                    if lag_suffix and lag_suffix.isdigit():
                        lag = int(lag_suffix)
                    else:
                        lag = 1  # Default to first lag if not specified
                
                # Add to the exogenous processes dictionary
                if base_name not in exo_processes:
                    exo_processes[base_name] = []
                exo_processes[base_name].append((lag, var))
            else:
                # This is an endogenous state
                endogenous_states.append(var)
        
        # Sort each exogenous process by lag
        for process in exo_processes.values():
            process.sort()  # Sort by lag (but preserve process ordering)
        
        # Second pass: Examine equations to find shock relationships
        for equation in self.equations:
            # Clean equation for analysis
            clean_eq = re.sub(r'//.*', '', equation).strip()
            
            # Look for exogenous processes that appear in this equation
            for base_name, process_lags in exo_processes.items():
                if base_name in clean_eq:
                    # This equation contains an exogenous process
                    # Look for shocks that appear in the same equation
                    for shock in self.varexo_list:
                        if shock in clean_eq:
                            # Found a shock that appears in the same equation as the process
                            if process_lags:  # If there are any lags for this process
                                state_var = process_lags[0][1]  # First lag gets direct shock
                                self.shock_to_state_map[shock] = state_var
                                self.state_to_shock_map[state_var] = shock
                                break
        
        # Extract variables that receive direct shocks (first lag of each process)
        exo_with_shocks = []
        exo_without_shocks = []
        
        for process_name, process_lags in exo_processes.items():
            if process_lags:  # If there are any lags for this process
                state_var = process_lags[0][1]  # First lag gets direct shock
                
                # Check if we found a shock for this state variable
                if state_var in self.state_to_shock_map:
                    exo_with_shocks.append(state_var)
                else:
                    # No shock found, but we still need to track it
                    exo_with_shocks.append(state_var)
                    print(f"Warning: No shock found for exogenous process state {state_var}")
                    
                # Higher lags don't get direct shocks
                for _, var in process_lags[1:]:
                    exo_without_shocks.append(var)
        
        # Store these categorizations for later use - NO SORTING
        self.endogenous_states = endogenous_states  
        self.exo_with_shocks = exo_with_shocks      
        self.exo_without_shocks = exo_without_shocks
        
        # Update the state_variables list with the correct ordering for state space
        self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
        # ---- END ENHANCED CODE ----
        
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
            'state_to_shock_map': self.state_to_shock_map
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
    
    def prepare_json_output(self):
        """Prepare the final JSON output"""
        # Combine all variables
        self.all_variables = self.state_variables + self.control_variables
        
        # Format equations
        formatted_equations = self.format_transformed_equations()
        
        # Create the JSON structure
        output = {
            "equations": formatted_equations,
            "state_variables": self.state_variables,
            "control_variables": self.control_variables,
            "parameters": list(self.parameters.keys()),
            "param_values": self.parameters,
            "shocks": self.varexo_list,
            "output_text": self.generate_output_text(formatted_equations)
        }
        
        return output
    
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
        """Main parsing function, updated with two-pass transformation"""
        self.read_dynare_file()
        self.parse_variables()
        self.parse_exogenous()
        self.parse_parameters()
        self.parse_model()
        
        # Use the two-pass transformation approach
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

    def analyze_exogenous_processes(self):
        """
        Analyze exogenous processes in the model and map them to their corresponding shocks
        by examining the equations themselves.
        """
        # Initialize mapping dictionaries
        self.shock_to_state_map = {}  # Maps shock names to state variables
        self.state_to_shock_map = {}  # Maps state variables to shock names
        
        # Identify exogenous process states and categorize them
        exo_processes = {}  # Dict mapping base names to lists of (lag, var_name)
        endogenous_states = []
        
        # First pass: Identify all exogenous process states
        for var in self.state_variables:
            if var.startswith("RES_") and "_lag" in var:
                # This is an exogenous process state
                base_name = var.split("_lag")[0]  # Get base name before _lag
                
                # Extract lag number
                if var.endswith("_lag"):
                    lag = 1  # First lag
                else:
                    # Try to extract lag number after _lag
                    lag_suffix = var.split("_lag")[1]
                    if lag_suffix and lag_suffix.isdigit():
                        lag = int(lag_suffix)
                    else:
                        lag = 1  # Default to first lag if not specified
                
                # Add to the exogenous processes dictionary
                if base_name not in exo_processes:
                    exo_processes[base_name] = []
                exo_processes[base_name].append((lag, var))
            else:
                # This is an endogenous state
                endogenous_states.append(var)
        
        # Sort each exogenous process by lag
        for process in exo_processes.values():
            process.sort()  # Sort by lag
        
        # Second pass: Examine equations to find connections between RES_ variables and shocks
        for equation in self.equations:
            # Clean equation for analysis
            clean_eq = re.sub(r'//.*', '', equation).strip()
            
            # Look for exogenous processes that appear in this equation
            for base_name, process_lags in exo_processes.items():
                if base_name in clean_eq:
                    # This equation contains an exogenous process
                    # Now look for shocks that appear in the same equation
                    for shock in self.varexo_list:
                        if shock in clean_eq:
                            # Found a shock that appears in the same equation as the process
                            # This likely means the shock drives this process
                            if process_lags:  # If there are any lags for this process
                                state_var = process_lags[0][1]  # First lag gets direct shock
                                self.shock_to_state_map[shock] = state_var
                                self.state_to_shock_map[state_var] = shock
                                break
        
        # Extract variables that receive direct shocks (first lag of each process)
        exo_with_shocks = []
        exo_without_shocks = []
        
        for process_name, process_lags in exo_processes.items():
            if process_lags:  # If there are any lags for this process
                state_var = process_lags[0][1]  # First lag gets direct shock
                
                # Check if we found a shock for this state variable
                if state_var in self.state_to_shock_map:
                    exo_with_shocks.append(state_var)
                else:
                    # No shock found, but we still need to track it
                    exo_with_shocks.append(state_var)
                    print(f"Warning: No shock found for exogenous process state {state_var}")
                    
                # Higher lags don't get direct shocks
                for _, var in process_lags[1:]:
                    exo_without_shocks.append(var)
        
        # Store these categorizations for later use
        self.endogenous_states = endogenous_states  # No need to sort
        self.exo_with_shocks = exo_with_shocks      # No need to sort 
        self.exo_without_shocks = exo_without_shocks  # No need to sort
        
        # Update the state_variables list with the correct ordering for state space
        self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
        
        return self.shock_to_state_map

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

#        endogenous_states, exo_with_shocks, exo_without_shocks = self.sort_state_variables()

        # Get the relevant model components after transformation
        variables = self.state_variables + self.control_variables
        exogenous = self.varexo_list
        parameters = list(self.parameters.keys())
        
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

    def generate_doubling_jacobian_evaluator(self, output_file=None):
        """
        Generate Python code for the Jacobian matrices needed for the doubling algorithm.
        
        The model follows the structure:
        0 = A E_t[y_{t+1}] + B y_t + C y_{t-1} + D ε_t
        
        Args:
            self: A DynareParser instance with parsed model
            output_file (str, optional): Path to save the generated Python code
                
        Returns:
            str: The generated Python code for the Jacobian evaluator
        """
        print("Generating Jacobian evaluator for doubling algorithm...")
        
        # Get model components from parser
        variables = self.state_variables + self.control_variables
        exogenous = self.varexo_list
        parameters = list(self.parameters.keys())
        
        # Variables with "_p" suffix for t+1 variables
        variables_p = [var + "_p" for var in variables]
        
        # Create symbolic variables for all model components
        var_symbols = {var: sy.symbols(var) for var in variables}
        var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
        exo_symbols = {exo: sy.symbols(exo) for exo in exogenous}
        param_symbols = {param: sy.symbols(param) for param in parameters}
        
        # Combine all symbols for equation parsing
        all_symbols = {**var_symbols, **var_p_symbols, **exo_symbols, **param_symbols}
        
        # Get equations from the transformed model
        formatted_equations = self.format_transformed_equations(
            self.transformed_equations, 
            self.auxiliary_equations
        )
        
        # Parse endogenous equations into sympy expressions
        equations = []
        success_count = 0
        error_count = 0
        
        for eq_dict in formatted_equations:
            for eq_name, eq_str in eq_dict.items():
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
        
        # Get variable counts for matrix dimensions
        n_states = len(self.state_variables)
        n_vars = len(variables)
        n_shocks = len(exogenous)
        
        # Extract symbols for different variable types
        future_symbols = [var_p_symbols[var_p] for var_p in variables_p]
        current_symbols = [var_symbols[var] for var in variables]
        # State variables at t-1 (for past variables)
        past_symbols = [var_symbols[var] for var in self.state_variables]
        shock_symbols = [exo_symbols[exo] for exo in exogenous]
        
        # Compute Jacobians for the model structure:
        # 0 = A E_t[y_{t+1}] + B y_t + C y_{t-1} + D ε_t
        print("Computing A matrix (coefficient on future variables)...")
        A_symbolic = F.jacobian(future_symbols)
        
        print("Computing B matrix (coefficient on current variables)...")
        B_symbolic = F.jacobian(current_symbols)
        
        print("Computing C matrix (coefficient on past state variables)...")
        # Only state variables have t-1 values
        C_symbolic = sy.zeros(n_vars, n_states)
        for i in range(len(equations)):
            for j, state_var in enumerate(self.state_variables):
                lag_var = state_var + "_lag"
                if lag_var in all_symbols:
                    C_symbolic[i, j] = sy.diff(equations[i], all_symbols[lag_var])
        
        print("Computing D matrix (coefficient on shock variables)...")
        D_symbolic = F.jacobian(shock_symbols)
        
        print("Generating output code...")
        
        # Generate code for the doubling algorithm Jacobian evaluation function
        function_code = [
            "import numpy as np",
            "",
            "def evaluate_doubling_jacobians(theta):",
            "    \"\"\"",
            "    Evaluates Jacobian matrices for the doubling algorithm",
            "    ",
            "    For the model structure: 0 = A E_t[y_{t+1}] + B y_t + C y_{t-1} + D ε_t",
            "    ",
            "    Args:",
            "        theta: List or array of parameter values in the order of:",
            f"            {parameters}",
            "        ",
            "    Returns:",
            "        A_plus: Matrix for future variables (coefficient on t+1 variables)",
            "        A_zero: Matrix for current variables (coefficient on t variables)",
            "        A_minus: Matrix for past variables (coefficient on t-1 variables)",
            "        shock_impact: Matrix for shock impacts (n_vars x n_shocks)",
            "        state_indices: Indices of state variables",
            "        control_indices: Indices of control variables",
            "    \"\"\"",
            "    # Unpack parameters from theta"
        ]
        
        # Add parameter unpacking
        for i, param in enumerate(parameters):
            function_code.append(f"    {param} = theta[{i}]")
        
        # Initialize matrices
        function_code.extend([
            "",
            f"    n_vars = {n_vars}",
            f"    n_states = {n_states}",
            f"    n_shocks = {n_shocks}",
            f"    A_plus = np.zeros((n_vars, n_vars))",
            f"    A_zero = np.zeros((n_vars, n_vars))",
            f"    A_minus = np.zeros((n_vars, n_states))",
            f"    shock_impact = np.zeros((n_vars, n_shocks))"   
        ])
        
        # Add A_plus matrix elements (future variables)
        function_code.append("")
        function_code.append("    # A_plus matrix elements (future variables)")
        for i in range(A_symbolic.rows):
            for j in range(A_symbolic.cols):
                if A_symbolic[i, j] != 0:
                    expr = str(A_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        # Replace symbol with parameter name
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    A_plus[{i}, {j}] = {expr}")
        
        # Add A_zero matrix elements (current variables)
        function_code.append("")
        function_code.append("    # A_zero matrix elements (current variables)")
        for i in range(B_symbolic.rows):
            for j in range(B_symbolic.cols):
                if B_symbolic[i, j] != 0:
                    expr = str(B_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    A_zero[{i}, {j}] = {expr}")
        
        # Add A_minus matrix elements (past variables)
        function_code.append("")
        function_code.append("    # A_minus matrix elements (past variables)")
        for i in range(C_symbolic.rows):
            for j in range(C_symbolic.cols):
                if C_symbolic[i, j] != 0:
                    expr = str(C_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    A_minus[{i}, {j}] = {expr}")
        
        # Add shock_impact matrix elements (shock terms)
        function_code.append("")
        function_code.append("    # shock_impact matrix elements (shock impacts)")
        for i in range(D_symbolic.rows):
            for j in range(D_symbolic.cols):
                if D_symbolic[i, j] != 0:
                    expr = str(D_symbolic[i, j])
                    # Clean up the expression
                    for param in parameters:
                        pattern = r'\b' + re.escape(str(param_symbols[param])) + r'\b'
                        expr = re.sub(pattern, param, expr)
                    function_code.append(f"    shock_impact[{i}, {j}] = {expr}")
        
        # Add state and control indices
        function_code.extend([
            "",
            "    # Indices of state and control variables",
            f"    state_indices = {list(range(n_states))}",
            f"    control_indices = {list(range(n_states, n_vars))}"
        ])
        
        # Return matrices
        function_code.append("")
        function_code.append("    return A_plus, A_zero, A_minus, shock_impact, state_indices, control_indices")
        
        # Join all lines to form the complete function code
        complete_code = "\n".join(function_code)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(complete_code)
            print(f"Doubling algorithm Jacobian evaluator saved to {output_file}")
        
        return complete_code

    def generate_model_structure(self):
        """
        Generate the structural components of the state space representation
        that don't depend on parameter values.

        Args:
            self: DynareParser instance with model information

        Returns:
            structure: Dictionary with structural components for state space
        """
        # Get variable counts and indices
        n_endogenous = len(self.endogenous_states)
        n_exo_with_shocks = len(self.exo_with_shocks)
        n_exo_without_shocks = len(self.exo_without_shocks)
        n_controls = len(self.control_variables)
        n_shocks = len(self.varexo_list)
        n_exo_states = n_exo_with_shocks + n_exo_without_shocks
        n_states = n_endogenous + n_exo_states

        # Create shock selection matrix R
        R = np.zeros((n_exo_states, n_shocks))

        # Fill R matrix using shock_to_state_map
        for shock_idx, shock_name in enumerate(self.varexo_list):
            if shock_name in self.shock_to_state_map:
                state_var = self.shock_to_state_map[shock_name]
                try:
                    # Find position in full state list
                    state_full_idx = self.state_variables.index(state_var)
                    # Calculate position in exogenous state vector
                    exo_state_idx = state_full_idx - n_endogenous
                    if 0 <= exo_state_idx < n_exo_states:
                        R[exo_state_idx, shock_idx] = 1.0
                except ValueError:
                    print(f"Warning: State variable {state_var} not found in state_variables")

        # Create B matrix structure (shock impacts on states)
        B_structure = np.zeros((n_states, n_shocks))
        B_structure[n_endogenous:, :] = R

        # Create C matrix structure (maps states to observables)
        n_observables = n_controls + n_states
        C_structure = np.zeros((n_observables, n_states))

        # States mapped one-to-one (identity matrix part)
        C_structure[n_controls:, :] = np.eye(n_states)

        # D matrix (direct shock impact on observables)
        D = np.zeros((n_observables, n_shocks))

        # Store indices for fast matrix creation later
        indices = {
            'n_endogenous': n_endogenous,
            'n_exo_states': n_exo_states,
            'n_controls': n_controls,
            'n_shocks': n_shocks,
            'n_states': n_states,
            'n_observables': n_observables
        }

        # Create labels for the state space
        labels = {
            'state_labels': self.state_variables,
            'observable_labels': self.control_variables + self.state_variables,
            'shock_labels': self.varexo_list
        }

        return {
            'indices': indices,
            'R': R,
            'B_structure': B_structure,
            'C_structure': C_structure,
            'D': D,
            'labels': labels
        }

    def parse_and_generate_files(dynare_file, output_dir):
        """Run the parser and generate all required files for later use"""
        # 1. Parse model and save JSON
        parser = DynareParser(dynare_file)
        model_json = parser.save_json(os.path.join(output_dir, "model.json"))

        # 2. Generate Jacobian file
        parser.generate_jacobian_evaluator(os.path.join(output_dir, "jacobian_evaluator.py"))

        # 3. Generate structure file
        structure = parser.generate_model_structure()
        with open(os.path.join(output_dir, "model_structure.py"), 'w') as f:
            f.write("import numpy as np\n\n")
            f.write(f"indices = {repr(structure['indices'])}\n\n")
            f.write(f"R = np.array({repr(structure['R'].tolist())})\n\n")
            f.write(f"B_structure = np.array({repr(structure['B_structure'].tolist())})\n\n")
            f.write(f"C_structure = np.array({repr(structure['C_structure'].tolist())})\n\n")
            f.write(f"D = np.array({repr(structure['D'].tolist())})\n\n")
            f.write(f"labels = {repr(structure['labels'])}\n")

        print(f"All model files generated in {output_dir}")


# # end of class parser 

# import json
# import importlib.util
# import sys
# import numpy as np
# import os
# from scipy.linalg import lu_factor, lu_solve
# import pandas as pd
# import matplotlib.pyplot as plt

# # Klein's method (copied from your parse_claude.py, but now in this module)
# def klein(a=None,b=None,n_states=None,eigenvalue_warnings=True):
#     # ... (rest of the klein function) ...
#     s,t,alpha,beta,q,z = la.ordqz(A=a,B=b,sort='ouc',output='complex')

#     # Components of the z matrix
#     z11 = z[0:n_states,0:n_states]
    
#     z21 = z[n_states:,0:n_states]
    
#     # number of nonpredetermined variables
#     n_costates = np.shape(a)[0] - n_states
    
#     if n_states>0:
#         if np.linalg.matrix_rank(z11)<n_states:
#             sys.exit("Invertibility condition violated. Check model equations or parameter values.")

#     s11 = s[0:n_states,0:n_states];
#     if n_states>0:
#         z11i = la.inv(z11)

#     else:
#         z11i = z11


#     # Components of the s,t,and q matrices   
#     t11 = t[0:n_states,0:n_states]
#     # Verify that there are exactly n_states stable (inside the unit circle) eigenvalues:
#     stab = 0

#     # if n_states>0:
#     #     if np.abs(t[n_states-1,n_states-1])>np.abs(s[n_states-1,n_states-1]):
#     #         if eigenvalue_warnings:
#     #             print('Warning: Too few stable eigenvalues. Check model equations or parameter values.')
#     #         stab = -1

#     # if n_states<n_states+n_costates:
#     #     if np.abs(t[n_states,n_states])<np.abs(s[n_states,n_states]):
#     #         if eigenvalue_warnings:
#     #             print('Warning: Too many stable eigenvalues. Check model equations or parameter values.')
#     #         stab = 1

#     # Compute the generalized eigenvalues
#     tii = np.diag(t)
#     sii = np.diag(s)
#     eig = np.zeros(np.shape(tii),dtype=np.complex128)
#     # eig = np.zeros(np.shape(tii))

#     for k in range(len(tii)):
#         if np.abs(sii[k])>0:
#             # eig[k] = np.abs(tii[k])/np.abs(sii[k])
#             eig[k] = tii[k]/sii[k]    
#         else:
#             eig[k] = np.inf



#     # Solution matrix coefficients on the endogenous state
#     if n_states>0:
#             dyn = np.linalg.solve(s11,t11)
#     else:
#         dyn = np.array([])


#     f = z21.dot(z11i)
#     p = z11.dot(dyn).dot(z11i)

#     f = np.real(f)
#     p = np.real(p)

#     return f, p,stab,eig


class ModelSolver:
    def __init__(self, output_dir):
        """
        Initializes the ModelSolver by loading pre-computed model components.

        Args:
            output_dir (str): Path to the directory containing the model files.
        """
        self.output_dir = output_dir
        self.load_model()
        self.load_jacobian_evaluator()
        self.load_model_structure()

    def load_model(self):
        """Loads the model from the JSON file."""
        model_path = os.path.join(self.output_dir, "model.json")
        with open(model_path, 'r') as f:
            self.model = json.load(f)

    def load_jacobian_evaluator(self):
        """Loads the Jacobian evaluator function from the Python file."""
        jac_path = os.path.join(self.output_dir, "jacobian_evaluator.py")
        spec = importlib.util.spec_from_file_location("jacobian_evaluator", jac_path)
        self.jacobian_module = importlib.util.module_from_spec(spec)
        sys.modules["jacobian_evaluator"] = self.jacobian_module
        spec.loader.exec_module(self.jacobian_module)
        self.evaluate_jacobians = self.jacobian_module.evaluate_jacobians

    def load_model_structure(self):
        """Loads the pre-computed model structure from the Python file."""
        struct_path = os.path.join(self.output_dir, "model_structure.py")
        spec = importlib.util.spec_from_file_location("model_structure", struct_path)
        struct_module = importlib.util.module_from_spec(spec)
        sys.modules["model_structure"] = struct_module
        spec.loader.exec_module(struct_module)

        self.indices = struct_module.indices
        self.R = struct_module.R
        self.B_structure = struct_module.B_structure
        self.C_structure = struct_module.C_structure
        self.D = struct_module.D
        self.labels = struct_module.labels

    def solve_and_create_state_space(self, params):
        """
        Solves the model and creates the state-space representation.

        Args:
            params (list): List of parameter values.

        Returns:
            dict: Dictionary containing the state-space matrices (A, B, C, D) and metadata.
        """
        # Evaluate Jacobians
        a, b, c = self.evaluate_jacobians(params)

        # Solve model with Klein's method
        f, p, stab, eig = klein(a, b, self.indices['n_states'])

        # Create state space
        ss = {
            'A': p.copy(),
            'B': self.B_structure.copy(),
            'C': self.C_structure.copy(),
            'D': self.D.copy(),
        }

        # Fill in policy function part of C matrix
        ss['C'][:self.indices['n_controls'], :] = f

        # Fill in B matrix (shock impacts on states)
        ss['B'][self.indices['n_endogenous']:, :] = self.R  # Ensure R is still used

        # Add metadata
        ss.update({
            'state_labels': self.labels['state_labels'],
            'observable_labels': self.labels['observable_labels'],
            'shock_labels': self.labels['shock_labels'],
            'n_states': self.indices['n_states'],
            'n_shocks': self.indices['n_shocks'],
            'n_observables': self.indices['n_observables'],
        })

        self.f = f  # Store f and p for IRF generation
        self.p = p
        self.stab = stab
        self.eig = eig

        # Store state space representation
        

        return ss

    def impulse_response(self, state_space, shock_idx=0, shock_size=1.0, periods=40):
        """
        Computes impulse response function for a given shock using state space representation.

        Args:
            state_space (dict): Dictionary containing the state-space matrices.
            shock_idx (int): Index of the shock.
            shock_size (float): Size of the shock.
            periods (int): Number of periods to simulate.

        Returns:
            DataFrame: DataFrame with IRF results.
        """
        import pandas as pd

        # Get state space matrices
        A = state_space['A']
        B = state_space['B']
        C = state_space['C']

        # Initialize state vector
        x = np.zeros(state_space['n_states'])

        # Initialize arrays for IRFs
        states_irf = np.zeros((periods, state_space['n_states']))
        obs_irf = np.zeros((periods, state_space['n_observables']))

        # Apply shock at t=0
        shock = np.zeros(state_space['n_shocks'])
        shock[shock_idx] = shock_size
        x = B @ shock

        # Store period 0 results
        states_irf[0, :] = x
        obs_irf[0, :] = C @ x

        # Simulate forward
        for t in range(1, periods):
            x = A @ states_irf[t - 1, :]
            states_irf[t, :] = x
            obs_irf[t, :] = C @ x

        # Create DataFrame with results
        irf_data = {}

        # Add observable variables
        for i, var in enumerate(state_space['observable_labels']):
            irf_data[var] = obs_irf[:, i]

        # Create DataFrame
        irf_df = pd.DataFrame(irf_data)

        # Add shock name for reference
        shock_name = state_space['shock_labels'][shock_idx] if shock_idx < len(
            state_space['shock_labels']) else f"shock_{shock_idx}"
        irf_df.attrs['shock_name'] = shock_name

        # ... after calculating f, p ...
        print("--- Original Script ---")
        print("B matrix (first 5x5 from Original logic):\n", B[:5, :5])
        print("Norm of B_orig:", np.linalg.norm(B))


        # # Find index for SHK_RS
        # try:
        #     shock_name_to_compare = 'SHK_RS'
        #     shock_idx_orig = parser.varexo_list.index(shock_name_to_compare)
        #     x0_orig_calc = B_orig[:, shock_idx_orig] * 1.0 # Assuming shock size 1.0
        #     print(f"x0 for {shock_name_to_compare} (Original logic):\n", x0_orig_calc)
        #     print(f"Index of non-zero element in x0_orig_calc: {np.argmax(np.abs(x0_orig_calc))}")
        # except ValueError:
        #     print(f"Shock {shock_name_to_compare} not found in original varexo_list")


        return irf_df

    def plot_irf(self, irf_df, variables_to_plot, shock_name, figsize=(12, 8)):
        """
        Plots selected variables from IRF results.

        Args:
            irf_df (DataFrame): DataFrame with IRF results.
            variables_to_plot (list): List of variables to plot.
            shock_name (str): Name of the shock (for title).
            figsize (tuple): Figure size.
        """
        plt.figure(figsize=figsize)

        for var in variables_to_plot:
            if var in irf_df.columns:
                plt.plot(irf_df.index, irf_df[var], label=var)
            else:
                print(f"Warning: Variable {var} not found in IRF results")

        plt.xlabel('Periods')
        plt.ylabel('Deviation from Steady State')
        plt.title(f'Impulse Responses to {shock_name} Shock')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        plt.tight_layout()
        plt.show()

    def print_model_details(self):
        """Print detailed model information from parser components we've seen in the code context"""
        # Header
        print("="*60)
        print("MODEL DETAILS".center(60))
        print("="*60)
        
        # Variables (from Context 3)
        print("\nVARIABLES:")
        print(f"State variables ({len(self.labels['state_labels'])}): {', '.join(self.labels['state_labels'])}")
        print(f"Control variables ({len(self.labels['observable_labels']) - len(self.labels['state_labels'])}): {', '.join([v for v in self.labels['observable_labels'] if v not in self.labels['state_labels']])}")
        print(f"Exogenous shocks ({len(self.model['shocks'])}): {', '.join(self.model['shocks'])}")  # From Context 2
        
        # Equations (from Context 2)
        print(f"\nEQUATIONS: {len(self.model['equations'])} total")  # From equations list in Context 2
        
        # Solution matrices (from Context 3)
        print("\nKLEIN SOLUTION MATRICES:")
        print(f"f matrix ({self.f.shape}): Policy functions mapping states to controls")
        print("First 5 rows/columns:")
        print(self.f[:5, :5])  # Show top-left corner
        
        print(f"\np matrix ({self.p.shape}): State transition matrix")
        print("First 5 rows/columns:")
        print(self.p[:5, :5])
        
        # Matrix-var relationships (from Context 3)
        print("\nSTRUCTURE:")
        print(f"Total equations: {len(self.model['equations'])}")
        print(f"State variables: {len(self.labels['state_labels'])}")
        print(f"Control variables: {len( [v for v in self.labels['observable_labels'] if v not in self.labels['state_labels']])}")
        print(f"Exogenous shocks: {len(self.model['shocks'])}")

        # Eigenvalues (from Context 3)
        print('Eigenvalues:')
        for i, val in enumerate(np.abs(self.eig)):
            print(f"  λ_{i+1}: {val:.6e}")



        
        print("="*60)

def solve_quadratic_matrix_equation_doubling(A, B, C, initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    """
    Solve the quadratic matrix equation A*X^2 + B*X + C = 0 using the structure-preserving doubling algorithm.
    
    This is a Python translation of the Julia function from quadratic_matrix_equation.jl.
    
    Args:
        A: Matrix for future variables (coefficient on t+1 variables)
        B: Matrix for current variables (coefficient on t variables)
        C: Matrix for past variables (coefficient on t-1 variables)
        initial_guess: Initial guess for the solution (optional)
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        verbose: Whether to print progress
        
    Returns:
        X: Solution matrix
        iter_count: Number of iterations performed
        reached_tol: Final residual
    """
    # Check if initial guess is provided
    guess_provided = True
    if initial_guess is None or initial_guess.size == 0:
        guess_provided = False
        initial_guess = np.zeros_like(A)
    
    # Initialize matrices
    E = C.copy()
    F = A.copy()
    
    # Compute B̄ = B + A*initial_guess
    B_bar = B.copy()
    B_bar += A @ initial_guess
    
    # LU factorization of B̄
    try:
        B_lu, B_piv = lu_factor(B_bar)
    except:
        if verbose:
            print("LU factorization of B_bar failed")
        return A, 0, 1.0
    
    # Compute initial values E and F
    E = lu_solve((B_lu, B_piv), C)
    F = lu_solve((B_lu, B_piv), A)
    
    # Initial X and Y
    X = -E - initial_guess
    Y = -F
    
    # Preallocate temporary matrices
    X_new = np.zeros_like(X)
    Y_new = np.zeros_like(Y)
    E_new = np.zeros_like(E)
    F_new = np.zeros_like(F)
    
    temp1 = np.zeros_like(Y)
    temp2 = np.zeros_like(Y)
    temp3 = np.zeros_like(Y)
    
    n = X.shape[0]
    II = np.eye(n)
    
    Xtol = 1.0
    Ytol = 1.0
    
    solved = False
    iter_count = max_iter
    
    # Main iteration loop
    for i in range(1, max_iter + 1):
        # Compute EI = I - Y*X
        np.matmul(Y, X, out=temp1)
        temp1 = II - temp1
        
        # Factorize EI
        try:
            EI_lu, EI_piv = lu_factor(temp1)
        except:
            if verbose:
                print(f"LU factorization of EI failed at iteration {i}")
            return A, i, 1.0
        
        # Compute E = E * EI^(-1) * E
        temp3 = lu_solve((EI_lu, EI_piv), E)
        np.matmul(E, temp3, out=E_new)
        
        # Compute FI = I - X*Y
        np.matmul(X, Y, out=temp2)
        temp2 = II - temp2
        
        # Factorize FI
        try:
            FI_lu, FI_piv = lu_factor(temp2)
        except:
            if verbose:
                print(f"LU factorization of FI failed at iteration {i}")
            return A, i, 1.0
        
        # Compute F = F * FI^(-1) * F
        temp3 = lu_solve((FI_lu, FI_piv), F)
        np.matmul(F, temp3, out=F_new)
        
        # Compute X_new = X + F * FI^(-1) * X * E
        np.matmul(X, E, out=temp3)
        temp3 = lu_solve((FI_lu, FI_piv), temp3)
        np.matmul(F, temp3, out=X_new)
        
        if i > 5 or guess_provided:
            Xtol = norm(X_new)
        
        X_new += X
        
        # Compute Y_new = Y + E * EI^(-1) * Y * F
        np.matmul(Y, F, out=temp3)
        temp3 = lu_solve((EI_lu, EI_piv), temp3)
        np.matmul(E, temp3, out=Y_new)
        
        if i > 5 or guess_provided:
            Ytol = norm(Y_new)
        
        Y_new += Y
        
        # Check for convergence
        if Xtol < tol:
            solved = True
            iter_count = i
            break
        
        # Update values for next iteration
        X[:] = X_new
        Y[:] = Y_new
        E[:] = E_new
        F[:] = F_new
    
    # Compute the final X
    X_new += initial_guess
    X = X_new
    
    # Compute the residual
    AXX = A @ X @ X
    AXXnorm = norm(AXX)
    AXX += B @ X
    AXX += C
    
    reached_tol = norm(AXX) / (AXXnorm + 1e-20)
    
    if verbose:
        print(f"Doubling algorithm finished in {iter_count} iterations with tolerance {reached_tol}")
    
    return X, iter_count, reached_tol

def solve_quadratic_matrix_equation(A, B, C, T, initial_guess=None, 
                                quadratic_matrix_equation_algorithm="doubling",
                                tol=1e-14, acceptance_tol=1e-8, verbose=False):
    """
    Wrapper function for solving the quadratic matrix equation, similar to the Julia implementation.
    
    Args:
        A: Matrix for future variables 
        B: Matrix for current variables
        C: Matrix for past variables
        T: Timing structure (from parser)
        initial_guess: Initial guess for the solution (optional)
        quadratic_matrix_equation_algorithm: Algorithm to use (only 'doubling' is fully implemented)
        tol: Tolerance for convergence
        acceptance_tol: Acceptance tolerance for convergence
        verbose: Whether to print progress
        
    Returns:
        sol: Solution matrix
        converged: Whether the solution converged
    """
    # Check if initial guess is valid
    if initial_guess is not None and initial_guess.size > 0:
        X = initial_guess
        
        # Check if initial guess is already a solution
        AXX = A @ X @ X
        AXXnorm = norm(AXX)
        AXX += B @ X
        AXX += C
        
        reached_tol = norm(AXX) / (AXXnorm + 1e-20)
        
        if reached_tol < (acceptance_tol * len(initial_guess) / 1e6):
            if verbose:
                print(f"Quadratic matrix equation solver: previous solution has tolerance {reached_tol}")
            return initial_guess, True
    
    # Solve using doubling algorithm
    sol, iterations, reached_tol = solve_quadratic_matrix_equation_doubling(
        A, B, C, 
        initial_guess=initial_guess, 
        tol=tol, 
        max_iter=100, 
        verbose=verbose
    )
    
    if verbose:
        print(f"Quadratic matrix equation solver: doubling - converged: {reached_tol < acceptance_tol} in {iterations} iterations to tolerance: {reached_tol}")
    
    return sol, reached_tol < acceptance_tol

def solve_quadratic_matrix_equation_doubling(A, B, C, initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    """
    Solve the quadratic matrix equation A*X^2 + B*X + C = 0 using the structure-preserving doubling algorithm.
    
    Parameters:
    -----------
    A : ndarray
        Coefficient matrix for X^2 term
    B : ndarray
        Coefficient matrix for X term
    C : ndarray
        Constant term
    initial_guess : ndarray, optional
        Initial guess for X
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Whether to print progress
    
    Returns:
    --------
    X : ndarray
        Solution to the quadratic matrix equation
    iter_count : int
        Number of iterations performed
    reached_tol : float
        Final error tolerance reached
    """
    guess_provided = True
    
    if initial_guess is None or initial_guess.size == 0:
        guess_provided = False
        initial_guess = np.zeros_like(A)
    
    # Initialize matrices
    E = C.copy()
    F = A.copy()
    
    # Compute B̄ = B + A*initial_guess
    B_bar = B.copy()
    B_bar += A @ initial_guess
    
    # LU factorization of B̄
    try:
        B_lu, B_piv = lu_factor(B_bar)
    except:
        return A, 0, 1.0
    
    # Compute initial values
    E = lu_solve((B_lu, B_piv), C)
    F = lu_solve((B_lu, B_piv), A)
    
    X = -E - initial_guess
    Y = -F
    
    # Preallocate temporary matrices
    X_new = np.zeros_like(X)
    Y_new = np.zeros_like(Y)
    E_new = np.zeros_like(E)
    F_new = np.zeros_like(F)
    
    temp1 = np.zeros_like(Y)
    temp2 = np.zeros_like(Y)
    temp3 = np.zeros_like(Y)
    
    n = X.shape[0]
    II = np.eye(n)
    
    Xtol = 1.0
    Ytol = 1.0
    
    solved = False
    iter_count = max_iter
    
    # Main iteration loop
    for i in range(1, max_iter + 1):
        # Compute EI = I - Y*X
        np.matmul(Y, X, out=temp1)
        temp1 = II - temp1
        
        # Factorize EI
        try:
            EI_lu, EI_piv = lu_factor(temp1)
        except:
            return A, i, 1.0
        
        # Compute E = E * EI^(-1) * E
        temp3 = lu_solve((EI_lu, EI_piv), E)
        np.matmul(E, temp3, out=E_new)
        
        # Compute FI = I - X*Y
        np.matmul(X, Y, out=temp2)
        temp2 = II - temp2
        
        # Factorize FI
        try:
            FI_lu, FI_piv = lu_factor(temp2)
        except:
            return A, i, 1.0
        
        # Compute F = F * FI^(-1) * F
        temp3 = lu_solve((FI_lu, FI_piv), F)
        np.matmul(F, temp3, out=F_new)
        
        # Compute X_new = X + F * FI^(-1) * X * E
        np.matmul(X, E, out=temp3)
        temp3 = lu_solve((FI_lu, FI_piv), temp3)
        np.matmul(F, temp3, out=X_new)
        
        if i > 5 or guess_provided:
            Xtol = norm(X_new)
        
        X_new += X
        
        # Compute Y_new = Y + E * EI^(-1) * Y * F
        np.matmul(Y, F, out=temp3)
        temp3 = lu_solve((EI_lu, EI_piv), temp3)
        np.matmul(E, temp3, out=Y_new)
        
        if i > 5 or guess_provided:
            Ytol = norm(Y_new)
        
        Y_new += Y
        
        # Check for convergence
        if Xtol < tol:
            solved = True
            iter_count = i
            break
        
        # Update values for next iteration
        X[:] = X_new
        Y[:] = Y_new
        E[:] = E_new
        F[:] = F_new
    
    # Compute the final X
    X_new += initial_guess
    X = X_new
    
    # Compute the residual
    AXX = A @ X @ X
    AXXnorm = norm(AXX)
    AXX += B @ X
    AXX += C
    
    reached_tol = norm(AXX) / AXXnorm
    
    return X, iter_count, reached_tol

def build_state_space_from_solution_doubling(F, P, shock_impact, state_indices, control_indices):
    """
    Convert the solution matrices F and P to state space form.
    
    Args:
        F: Control-to-state mapping from solution
        P: State transition matrix from solution
        shock_impact: Shock impact matrix
        state_indices: Indices of state variables
        control_indices: Indices of control variables
        
    Returns:
        A: State transition matrix
        B: Shock impact matrix
    """
    n_states = len(state_indices)
    n_controls = len(control_indices)
    n_shocks = shock_impact.shape[1]
    n_variables = n_states + n_controls
    
    # Create state space matrices
    A = np.zeros((n_variables, n_variables))
    B = np.zeros((n_variables, n_shocks))
    
    # Fill state transition matrix
    # Controls depend on states
    A[:n_controls, n_controls:] = F
    
    # States transition according to P
    A[n_controls:, n_controls:] = P
    
    # Shock impact matrix
    # Controls directly affected by shocks
    B[:n_controls, :] = shock_impact[:n_controls, :]
    
    # States affected by shocks
    B[n_controls:, :] = shock_impact[n_controls:, :]
    
    return A, B

def impulse_response_doubling(A, B, periods=40, shock_idx=0, shock_size=1.0):
    """
    Compute impulse response function for a given shock.
    
    Args:
        A: State transition matrix
        B: Shock impact matrix
        periods: Number of periods to simulate
        shock_idx: Index of the shock
        shock_size: Size of the shock
        
    Returns:
        irf: Impulse response matrix (periods x variables)
    """
    n_variables = A.shape[0]
    
    # Initialize impulse response matrix
    irf = np.zeros((periods, n_variables))
    
    # Initial shock
    irf[0, :] = B[:, shock_idx] * shock_size
    
    # Propagate through the state space
    for t in range(1, periods):
        irf[t, :] = A @ irf[t-1, :]
    
    return irf

def solve_dsge_klein_representation(A_plus, A_zero, A_minus, 
                                    state_indices, control_indices,
                                    initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    """
    Solves a DSGE model using the doubling algorithm and returns the Klein representation matrices.
    
    Parameters:
    -----------
    A_plus : ndarray
        Jacobian with respect to future variables (t+1)
    A_zero : ndarray
        Jacobian with respect to current variables (t)
    A_minus : ndarray
        Jacobian with respect to past variables (t-1)
    state_indices : list
        Indices of predetermined state variables in the model
    control_indices : list
        Indices of control variables in the model
    initial_guess : ndarray, optional
        Initial guess for X
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Whether to print progress
    
    Returns:
    --------
    F : ndarray
        Policy function matrix (control variables as function of state variables)
    P : ndarray
        State transition matrix (evolution of state variables)
    converged : bool
        Whether the solution converged
    X : ndarray
        The original solution to the quadratic matrix equation
    """
    # First solve the quadratic matrix equation using the doubling algorithm
    X, iterations, residual = solve_quadratic_matrix_equation_doubling(
        A_plus, A_zero, A_minus, initial_guess, tol, max_iter, verbose
    )
    
    if residual > tol:
        if verbose:
            print(f"Solution did not converge. Residual: {residual}")
        return None, None, False, X
    
    # Construct the full transition matrix
    n_vars = A_zero.shape[0]
    full_transition = np.zeros((n_vars, n_vars))
    
    # The structure depends on how the model is arranged, but generally:
    # 1. For predetermined variables (states), we use the law of motion from the model
    # 2. For control variables, we use the policy function X
    
    # Extract state-to-state transition (P matrix)
    n_states = len(state_indices)
    P = np.zeros((n_states, n_states))
    
    # Assuming X gives the full set of relationships
    # We need to extract the parts relevant to state transitions
    for i, row_idx in enumerate(state_indices):
        for j, col_idx in enumerate(state_indices):
            # This assumes X contains the full transition structure
            # The exact indexing depends on the structure of X
            P[i, j] = X[row_idx, col_idx]
    
    # Extract control-to-state relationships (F matrix)
    n_controls = len(control_indices)
    F = np.zeros((n_controls, n_states))
    
    for i, row_idx in enumerate(control_indices):
        for j, col_idx in enumerate(state_indices):
            F[i, j] = X[row_idx, col_idx]
    
    return F, P, True, X

def klein(a=None,b=None,n_states=None,eigenvalue_warnings=True):

    '''Solves linear dynamic models with the form of:
    
                a*Et[x(t+1)] = b*x(t)       
                
        [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
        a vector of nonpredetermined costate variables. z(t) is a vector of exogenous forcing variables with 
        autocorrelation matrix phi. The solution to the model is a set of matrices f, n, p, l such that:

                u(t)   = f*s(t)
                s(t+1) = p*s(t).

        The solution algorithm is based on Klein (2000) and his solab.m Matlab program.

    Args:
        a:                      (Numpy ndarray) Coefficient matrix on future-dated variables
        b:                      (Numpy ndarray) Coefficient matrix on current-dated variables
        c:                      (Numpy ndarray) Coefficient matrix on exogenous forcing variables
        n_states:               (int) Number of state variables
        eigenvalue_warnings:    (bool) Whether to print warnings that there are too many or few eigenvalues. Default: True

    Returns:
        f:          (Numpy ndarray) Solution matrix coeffients on s(t) for u(t)
        p:          (Numpy ndarray) Solution matrix coeffients on s(t) for s(t+1)
        stab:       (int) Indicates solution stability and uniqueness
                        stab == 1: too many stable eigenvalues
                        stab == -1: too few stable eigenvalues
                        stab == 0: just enoughstable eigenvalues
        eig:        The generalized eigenvalues from the Schur decomposition

    '''

    s,t,alpha,beta,q,z = la.ordqz(A=a,B=b,sort='ouc',output='complex')

    # Components of the z matrix
    z11 = z[0:n_states,0:n_states]
    
    z21 = z[n_states:,0:n_states]
    
    # number of nonpredetermined variables
    n_costates = np.shape(a)[0] - n_states
    
    if n_states>0:
        if np.linalg.matrix_rank(z11)<n_states:
            sys.exit("Invertibility condition violated. Check model equations or parameter values.")

    s11 = s[0:n_states,0:n_states];
    if n_states>0:
        z11i = la.inv(z11)

    else:
        z11i = z11


    # Components of the s,t,and q matrices   
    t11 = t[0:n_states,0:n_states]
    # Verify that there are exactly n_states stable (inside the unit circle) eigenvalues:
    stab = 0

    # if n_states>0:
    #     if np.abs(t[n_states-1,n_states-1])>np.abs(s[n_states-1,n_states-1]):
    #         if eigenvalue_warnings:
    #             print('Warning: Too few stable eigenvalues. Check model equations or parameter values.')
    #         stab = -1

    # if n_states<n_states+n_costates:
    #     if np.abs(t[n_states,n_states])<np.abs(s[n_states,n_states]):
    #         if eigenvalue_warnings:
    #             print('Warning: Too many stable eigenvalues. Check model equations or parameter values.')
    #         stab = 1

    # Compute the generalized eigenvalues
    tii = np.diag(t)
    sii = np.diag(s)
    eig = np.zeros(np.shape(tii),dtype=np.complex128)
    # eig = np.zeros(np.shape(tii))

    for k in range(len(tii)):
        if np.abs(sii[k])>0:
            # eig[k] = np.abs(tii[k])/np.abs(sii[k])
            eig[k] = tii[k]/sii[k]    
        else:
            eig[k] = np.inf



    # Solution matrix coefficients on the endogenous state
    if n_states>0:
            dyn = np.linalg.solve(s11,t11)
    else:
        dyn = np.array([])


    f = z21.dot(z11i)
    p = z11.dot(dyn).dot(z11i)

    f = np.real(f)
    p = np.real(p)

    return f, p,stab,eig

def create_state_space_representation(parser, f, p):
    """
    Create state space representation from Klein solution matrices
    
    Args:
        parser: DynareParser instance with model information
        f: Control policy function matrix from Klein solution
        p: State transition matrix from Klein solution
        
    Returns:
        A: State transition matrix for state space representation
        B: Shock impact matrix for state space representation
        C: Observation matrix
        D: Direct shock impact on observables (often zero)
    """
    # Get variable counts
    n_endogenous = len(parser.endogenous_states)
    n_exo_with_shocks = len(parser.exo_with_shocks)
    n_exo_without_shocks = len(parser.exo_without_shocks)
    n_controls = len(parser.control_variables)
    n_shocks = len(parser.varexo_list)
    
    # Total exogenous states
    n_exo_states = n_exo_with_shocks + n_exo_without_shocks
    n_states = n_endogenous + n_exo_states
    
    # Extract submatrices from F (control policy functions)
    # F maps states to controls: c_t = F s_t
    Fcx = f[:, :n_endogenous]  # Control responses to endogenous states
    Fcz = f[:, n_endogenous:]  # Control responses to exogenous states
    
    # Extract submatrices from P (state transitions)
    # P maps current states to future states: s_{t+1} = P s_t
    Pxx = p[:n_endogenous, :n_endogenous]  # Endogenous state transitions
    Pxz = p[:n_endogenous, n_endogenous:]  # Impact of exogenous on endogenous
    Pzz = p[n_endogenous:, n_endogenous:]  # Exogenous AR processes
    
    # Create shock selection matrix R
    # This maps structural shocks to exogenous states: z_t = Pzz z_{t-1} + R ε_t
    R = np.zeros((n_exo_states, n_shocks))
    
    # Fill R matrix using shock_to_state_map
    for shock_idx, shock_name in enumerate(parser.varexo_list):
        if shock_name in parser.shock_to_state_map:
            state_var = parser.shock_to_state_map[shock_name]
            # Find position of this state in the exogenous states
            try:
                # First find position in full state list
                state_full_idx = parser.state_variables.index(state_var)
                # Calculate position in exogenous state vector
                exo_state_idx = state_full_idx - n_endogenous
                if 0 <= exo_state_idx < n_exo_states:
                    R[exo_state_idx, shock_idx] = 1.0
            except ValueError:
                print(f"Warning: State variable {state_var} not found in state_variables")
    
    # ---- STATE SPACE MATRICES ----
    
    # A matrix: state transition
    A = p.copy()  # The P matrix already captures the state transition
    
    # B matrix: shock impacts on states
    # Only exogenous states are directly affected by shocks
    B = np.zeros((n_states, n_shocks))
    B[n_endogenous:, :] = R
    
    # C matrix: maps states to observables
    # By default, we'll use the controls and states as observables
    n_observables = n_controls + n_states
    C = np.zeros((n_observables, n_states))
    
    # First rows: controls as functions of states (via policy function F)
    C[:n_controls, :] = f
    print(f"f matrix:")
    print(f)
    # Next rows: states mapped one-to-one (identity matrix)
    C[n_controls:, :] = np.eye(n_states)
    
    # D matrix: direct shock impact on observables
    # Usually zero in DSGE models, but controls might be directly affected
    D = np.zeros((n_observables, n_shocks))
    
    # Create labels for the state space
    state_labels = parser.state_variables
    observable_labels = parser.control_variables + parser.state_variables
    shock_labels = parser.varexo_list
    
    return {
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'state_labels': state_labels,
        'observable_labels': observable_labels,
        'shock_labels': shock_labels,
        'n_states': n_states,
        'n_shocks': n_shocks,
        'n_observables': n_observables
    }

def simulate_irf_state_space(ss, shock_idx=0, shock_size=1.0, periods=40):
    """
    Simulate impulse responses using state space representation
    
    Args:
        ss: State space representation (output from create_state_space_representation)
        shock_idx: Index of shock to simulate
        shock_size: Size of the shock
        periods: Number of periods for IRF
        
    Returns:
        DataFrame with IRF results
    """
    import pandas as pd
    
    # Get state space matrices
    A = ss['A']
    B = ss['B']
    C = ss['C']
    
    # Initialize state vector
    x = np.zeros(ss['n_states'])
    
    # Initialize arrays for IRFs
    states_irf = np.zeros((periods, ss['n_states']))
    obs_irf = np.zeros((periods, ss['n_observables']))
    
    # Apply shock at t=0
    # The shock affects the state via B matrix
    shock = np.zeros(ss['n_shocks'])
    shock[shock_idx] = shock_size
    x = B @ shock
    
    # Store period 0 results
    states_irf[0, :] = x
    obs_irf[0, :] = C @ x
    
    # Simulate forward
    for t in range(1, periods):
        x = A @ states_irf[t-1, :]
        states_irf[t, :] = x
        obs_irf[t, :] = C @ x
    
    # Create DataFrame with results
    irf_data = {}
    
    # Add observable variables
    for i, var in enumerate(ss['observable_labels']):
        irf_data[var] = obs_irf[:, i]
    
    # Create DataFrame
    irf_df = pd.DataFrame(irf_data)
    
    # Add shock name for reference
    shock_name = ss['shock_labels'][shock_idx] if shock_idx < len(ss['shock_labels']) else f"shock_{shock_idx}"
    irf_df.attrs['shock_name'] = shock_name
    
    return irf_df

def compare_irf_methods(parser, f, p):
    """
    Compare IRFs from different methods
    
    Args:
        parser: DynareParser instance
        f: Control policy matrix
        p: State transition matrix
    """
    # Original IRF method
    irf_original = generate_irfs(parser, f, p)
    
    # State space IRF method
    ss = create_state_space_representation(parser, f, p)
    
    # Compare for each shock
    for shock_idx, shock_name in enumerate(parser.varexo_list):
        if shock_name in irf_original:
            print(f"Comparing IRFs for shock: {shock_name}")
            
            # Generate IRF using state space
            irf_ss = simulate_irf_state_space(ss, shock_idx=shock_idx)
            
            # Select some key variables to compare
            vars_to_compare = ['RS', 'DLA_CPI', 'L_GDP_GAP', 'RR_GAP']
            vars_to_compare = [v for v in vars_to_compare if v in irf_original[shock_name].columns]
            
            # Compare values
            for var in vars_to_compare:
                original = irf_original[shock_name][var].values[:10]
                ss_method = irf_ss[var].values[:10]
                
                # Calculate difference
                max_diff = np.max(np.abs(original - ss_method))
                print(f"  {var}: Max difference = {max_diff:.8f}")
                
                if max_diff > 1e-6:
                    print(f"    Warning: Differences detected for {var}")
                    print(f"    Original: {original[:5]}")
                    print(f"    SS method: {ss_method[:5]}")

def generate_irfs(parser, F, P, shock_size=1.0, T=40):
    """
    Generate IRFs using Klein's state-space representation with proper AR(1) dynamics.
    
    Args:
        parser: DynareParser instance with model information
        F: Control policy function matrix from Klein solution
        P: State transition matrix from Klein solution
        shock_size: Size of the shock impulse
        T: Number of periods for IRF
        
    Returns:
        irf_results: Dictionary mapping shock names to IRF DataFrames
    """
    irf_results = {}
    n_states = len(parser.state_variables)
    
    # For each shock
    for shock_name in parser.varexo_list:
        # Use shock-to-state mappings to find which state variable receives this shock
        if shock_name in parser.shock_to_state_map:
            # Get state variable that receives this shock
            state_var = parser.shock_to_state_map[shock_name]
            
            # Find state variable index in the state_variables list
            state_idx = parser.state_variables.index(state_var)
            
            # Create initial state vector
            x0 = np.zeros(n_states)
            x0[state_idx] = shock_size  # Apply shock of given size
            
            # Compute IRF
            irf_df = ir(F, P, x0, T=T, parser=parser)
            
            # Store results
            irf_results[shock_name] = irf_df
        else:
            print(f"Warning: No state variable mapping found for shock {shock_name}")
    
    return irf_results

def ir(F, P, x0, T=40, parser=None):
    """
    Compute impulse responses and return as pandas DataFrame
    
    Args:
        F: Control policy function matrix
        P: State transition matrix
        x0: Initial state vector
        T: Number of periods
        parser: DynareParser instance with variable names
        
    Returns:
        DataFrame with IRF results
    """
    import pandas as pd
    
    nx = P.shape[0]  # Number of state variables
    ny = F.shape[0]  # Number of control variables
    
    # Create augmented observation matrix that maps states to all variables
    MX = np.vstack([F, np.eye(nx)])
    
    # Initialize responses
    IR = np.zeros((T, ny + nx))
    
    # Set initial state
    x = x0.copy()
    
    # Compute responses
    for t in range(T):
        IR[t, :] = MX @ x  # Map states to all variables
        x = P @ x          # Transition states to next period
    
    # Get variable names from parser
    variable_names = parser.control_variables + parser.state_variables
    
    # Create DataFrame
    df = pd.DataFrame(IR, columns=variable_names)
    
    return df

def plot_irf(irf_df, variables_to_plot, shock_name, figsize=(12, 8)):
    """
    Plot selected variables from IRF results
    
    Args:
        irf_df: DataFrame with IRF results
        variables_to_plot: List of variables to plot
        shock_name: Name of the shock (for title)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for var in variables_to_plot:
        if var in irf_df.columns:
            plt.plot(irf_df.index, irf_df[var], label=var)
        else:
            print(f"Warning: Variable {var} not found in IRF results")
    
    plt.xlabel('Periods')
    plt.ylabel('Deviation from Steady State')
    plt.title(f'Impulse Responses to {shock_name} Shock')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()

def print_model_details(parser):
    """Print detailed model information from parser components we've seen in the code context"""
    # Header
    print("="*60)
    print("MODEL DETAILS".center(60))
    print("="*60)
    
    # Variables (from Context 3)
    print("\nVARIABLES:")
    print(f"State variables ({len(parser.state_variables)}): {', '.join(parser.state_variables)}")
    print(f"Control variables ({len(parser.control_variables)}): {', '.join(parser.control_variables)}")
    print(f"Exogenous shocks ({len(parser.varexo_list)}): {', '.join(parser.varexo_list)}")  # From Context 2
    
    # Equations (from Context 2)
    print(f"\nEQUATIONS: {len(parser.equations)} total")  # From equations list in Context 2
    
    # Solution matrices (from Context 3)
    print("\nKLEIN SOLUTION MATRICES:")
    print(f"f matrix ({parser.f.shape}): Policy functions mapping states to controls")
    print("First 3 rows/columns:")
    print(parser.f[:3, :3])  # Show top-left corner
    
    print(f"\np matrix ({parser.p.shape}): State transition matrix")
    print("First 3 rows/columns:")
    print(parser.p[:3, :3])
    
    # Matrix-var relationships (from Context 3)
    print("\nSTRUCTURE:")
    print(f"Total equations: {len(parser.equations)}")
    print(f"State variables: {len(parser.state_variables)}")
    print(f"Control variables: {len(parser.control_variables)}")
    print(f"Exogenous shocks: {len(parser.varexo_list)}")

    # Eigenvalues (from Context 3)
    print('Eigenvalues:')
    for i, val in enumerate(np.abs(parser.eig)):
        print(f"  λ_{i+1}: {val:.6e}")

    print("="*60)



# Example usage (driver script)
if __name__ == "__main__":
    import os
    import numpy as np
    #from parse_claude import DynareParser  # Import your parser
    #from model_solver import ModelSolver  # Import the new ModelSolver

    # Define the output directory for storing the generated files
    output_dir = "model_files"  # Or any directory you prefer
    os.makedirs(output_dir, exist_ok=True)

    # Define the path to your Dynare model file
    dynare_file = "qpm_simpl1.dyn" #os.path.join(os.path.dirname(os.path.abspath(__file__)), "qpm_simpl1.dyn")  # Replace with your Dynare file

    # 1. Generate the necessary files (JSON model, Jacobian, structure)
    DynareParser.parse_and_generate_files(dynare_file, output_dir)

    # 2. Create an instance of the ModelSolver
    solver = ModelSolver(output_dir)

    # 3. Define parameter values
    # Example: Use the parameter values from the JSON file
    with open(os.path.join(output_dir, "model.json"), 'r') as f:
        model_data = json.load(f)
    initial_params = [model_data['param_values'][p] for p in model_data['parameters']]

    # 4. Solve the model and get the state-space representation
    state_space = solver.solve_and_create_state_space(initial_params)

    # Print some model details from structure
    solver.print_model_details()

    # 5. Generate impulse responses
    irf_df = solver.impulse_response(state_space, shock_idx=0)  # Shock index 0
    print("\nImpulse Response to shock 0:\n", irf_df.head())

    # 6. Plot impulse responses for a specific shock
    variables_to_plot = [
        "RR_GAP",
        "RS",
        "DLA_CPI",
        "L_GDP_GAP",
        "RES_RS_lag",
        "RES_L_GDP_GAP_lag",
        "RES_DLA_CPI_lag"
    ]
    solver.plot_irf(irf_df, variables_to_plot, "Interest Rate")




# # Example usage
# if __name__ == "__main__":
#     import os 
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     dynare_file = os.path.join(script_dir, "qpm_simpl1.dyn")
#     parser = DynareParser(dynare_file)
#     parser.save_json(os.path.join(script_dir,"transformed_model_claude.json"))
#     parser.generate_jacobian_evaluator("_jacobian_evaluator.py")

#     #Evaluate Jacobians
#     # Load Jacobian evaluator function
#     import importlib.util
#     spec = importlib.util.spec_from_file_location("jacobian_evaluator", "_jacobian_evaluator.py")
#     jacobian_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(jacobian_module)
#     evaluate_jacobians = jacobian_module.evaluate_jacobians
    
#     # Get parameter values
#     param_values = list(parser.parameters.values())
    
#     #Number of state variables#print_model_output(parser)

    
#     n_states = len(parser.state_variables)
#     n_exogenous = len(parser.varexo_list)
#     n_equations = len(parser.all_variables)
#     n_controls = n_equations - n_states
    
#     # Evaluate Jacobians
#     a, b, c = evaluate_jacobians(param_values)

#     f, p, stab, eig = klein(a=a, b=b, n_states=n_states, eigenvalue_warnings=True)
#     parser.f = f
#     parser.p = p
#     parser.stab = stab
#     parser.eig= eig

#     print_model_details(parser)
#     print("Done!")

#     # Generate IRFs
#     irf_results = generate_irfs(parser, f, p, shock_size=1.0, T=40)

#     # Plot IRFs for a specific shock
#     variables_to_plot = [
#         "RR_GAP",
#         "RS",    
#         "DLA_CPI",
#         "L_GDP_GAP",   
#         "RES_RS_lag",
#         "RES_L_GDP_GAP_lag",
#         "RES_DLA_CPI_lag"
#     ]

#     # I still need lag z_t in the state space form to get the responses alignig with dynare. Say in my notation a shock to RES is really a shock to RES_LAG (as state) and RES_ is the value at t+1 of RES.  
#     # This commes from the klein's solution that has Z(t+1) and K(t+1). The IRFs look OK. I guess. 
#     plot_irf(irf_results['SHK_RS'], variables_to_plot, 'Interest Rate', figsize=(12, 8))

#     compare_irf_methods(parser, f, p)

    ##parser.generate_doubling_jacobian_evaluator("doubling_jacobian_evaluator.py")

    # Get Jacobians
#A_plus, A_zero, A_minus, shock_impact, state_indices, control_indices = evaluate_doubling_jacobians(parameters)

# # Solve the quadratic matrix equation
# solution_matrix, converged = solve_quadratic_matrix_equation(
#     A_plus, A_zero, A_minus, 
#     T=timings_object,  # You'll need to pass your timing structure
#     initial_guess=None,
#     verbose=True
# )

# # If needed, build state space form
# A, B = build_state_space_from_solution(
#     F=solution_matrix[:n_controls, :n_states],
#     P=solution_matrix[n_controls:, :n_states],
#     shock_impact=shock_impact,
#     state_indices=state_indices,
#     control_indices=control_indices
# )

# # Compute impulse responses
# irf = impulse_response(A, B, periods=40, shock_idx=0)