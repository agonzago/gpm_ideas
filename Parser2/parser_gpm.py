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

    # def analyze_exogenous_processes(self):
    #     """
    #     Analyze exogenous processes in the model and map them to their corresponding shocks
    #     by examining the equations themselves.
    #     """
    #     # Initialize mapping dictionaries
    #     self.shock_to_state_map = {}  # Maps shock names to state variables
    #     self.state_to_shock_map = {}  # Maps state variables to shock names
        
    #     # Identify exogenous process states and categorize them
    #     exo_processes = {}  # Dict mapping base names to lists of (lag, var_name)
    #     endogenous_states = []
        
    #     # First pass: Identify all exogenous process states
    #     for var in self.state_variables:
    #         if var.startswith("RES_") and "_lag" in var:
    #             # This is an exogenous process state
    #             base_name = var.split("_lag")[0]  # Get base name before _lag
                
    #             # Extract lag number
    #             if var.endswith("_lag"):
    #                 lag = 1  # First lag
    #             else:
    #                 # Try to extract lag number after _lag
    #                 lag_suffix = var.split("_lag")[1]
    #                 if lag_suffix and lag_suffix.isdigit():
    #                     lag = int(lag_suffix)
    #                 else:
    #                     lag = 1  # Default to first lag if not specified
                
    #             # Add to the exogenous processes dictionary
    #             if base_name not in exo_processes:
    #                 exo_processes[base_name] = []
    #             exo_processes[base_name].append((lag, var))
    #         else:
    #             # This is an endogenous state
    #             endogenous_states.append(var)
        
    #     # Sort each exogenous process by lag
    #     for process in exo_processes.values():
    #         process.sort()  # Sort by lag
        
    #     # Second pass: Examine equations to find connections between RES_ variables and shocks
    #     for equation in self.equations:
    #         # Clean equation for analysis
    #         clean_eq = re.sub(r'//.*', '', equation).strip()
            
    #         # Look for exogenous processes that appear in this equation
    #         for base_name, process_lags in exo_processes.items():
    #             if base_name in clean_eq:
    #                 # This equation contains an exogenous process
    #                 # Now look for shocks that appear in the same equation
    #                 for shock in self.varexo_list:
    #                     if shock in clean_eq:
    #                         # Found a shock that appears in the same equation as the process
    #                         # This likely means the shock drives this process
    #                         if process_lags:  # If there are any lags for this process
    #                             state_var = process_lags[0][1]  # First lag gets direct shock
    #                             self.shock_to_state_map[shock] = state_var
    #                             self.state_to_shock_map[state_var] = shock
    #                             break
        
    #     # Extract variables that receive direct shocks (first lag of each process)
    #     exo_with_shocks = []
    #     exo_without_shocks = []
        
    #     for process_name, process_lags in exo_processes.items():
    #         if process_lags:  # If there are any lags for this process
    #             state_var = process_lags[0][1]  # First lag gets direct shock
                
    #             # Check if we found a shock for this state variable
    #             if state_var in self.state_to_shock_map:
    #                 exo_with_shocks.append(state_var)
    #             else:
    #                 # No shock found, but we still need to track it
    #                 exo_with_shocks.append(state_var)
    #                 print(f"Warning: No shock found for exogenous process state {state_var}")
                    
    #             # Higher lags don't get direct shocks
    #             for _, var in process_lags[1:]:
    #                 exo_without_shocks.append(var)
        
    #     # Store these categorizations for later use
    #     self.endogenous_states = endogenous_states  # No need to sort
    #     self.exo_with_shocks = exo_with_shocks      # No need to sort 
    #     self.exo_without_shocks = exo_without_shocks  # No need to sort
        
    #     # Update the state_variables list with the correct ordering for state space
    #     self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
        
    #     return self.shock_to_state_map

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
        

        # In your ORIGINAL script, find where variables are finalized
        # Let's assume you have lists like 'original_state_vars' and 'original_control_vars'
        print("Original State Variables Order:", self.state_variables)
        print("Original Control Variables Order:", self.control_variables)
        
        # The combined list used for Jacobians would be original_state_vars + original_control_vars
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

    # def generate_model_structure(self):
    #     """
    #     Generate the structural components of the state space representation
    #     that don't depend on parameter values.

    #     Args:
    #         self: DynareParser instance with model information

    #     Returns:
    #         structure: Dictionary with structural components for state space
    #     """
    #     # Get variable counts and indices
    #     n_endogenous = len(self.endogenous_states)
    #     n_exo_with_shocks = len(self.exo_with_shocks)
    #     n_exo_without_shocks = len(self.exo_without_shocks)
    #     n_controls = len(self.control_variables)
    #     n_shocks = len(self.varexo_list)
    #     n_exo_states = n_exo_with_shocks + n_exo_without_shocks
    #     n_states = n_endogenous + n_exo_states

    #     # Create shock selection matrix R
    #     R = np.zeros((n_exo_states, n_shocks))

    #     # Fill R matrix using shock_to_state_map
    #     for shock_idx, shock_name in enumerate(self.varexo_list):
    #         if shock_name in self.shock_to_state_map:
    #             state_var = self.shock_to_state_map[shock_name]
    #             try:
    #                 # Find position in full state list
    #                 state_full_idx = self.state_variables.index(state_var)
    #                 # Calculate position in exogenous state vector
    #                 exo_state_idx = state_full_idx - n_endogenous
    #                 if 0 <= exo_state_idx < n_exo_states:
    #                     R[exo_state_idx, shock_idx] = 1.0
    #             except ValueError:
    #                 print(f"Warning: State variable {state_var} not found in state_variables")

    #     # Create B matrix structure (shock impacts on states)
    #     B_structure = np.zeros((n_states, n_shocks))
    #     B_structure[n_endogenous:, :] = R

    #     # Create C matrix structure (maps states to observables)
    #     n_observables = n_controls + n_states
    #     C_structure = np.zeros((n_observables, n_states))

    #     # States mapped one-to-one (identity matrix part)
    #     C_structure[n_controls:, :] = np.eye(n_states)

    #     # D matrix (direct shock impact on observables)
    #     D = np.zeros((n_observables, n_shocks))

    #     # Store indices for fast matrix creation later
    #     indices = {
    #         'n_endogenous': n_endogenous,
    #         'n_exo_states': n_exo_states,
    #         'n_controls': n_controls,
    #         'n_shocks': n_shocks,
    #         'n_states': n_states,
    #         'n_observables': n_observables
    #     }

    #     # Create labels for the state space
    #     labels = {
    #         'state_labels': self.state_variables,
    #         'observable_labels': self.control_variables + self.state_variables,
    #         'shock_labels': self.varexo_list
    #     }

    #     return {
    #         'indices': indices,
    #         'R': R,
    #         'B_structure': B_structure,
    #         'C_structure': C_structure,
    #         'D': D,
    #         'labels': labels
    #     }

    # def analyze_exogenous_processes(self):
    #     """
    #     Analyze exogenous processes in the model and map them to their corresponding shocks
    #     by examining the equations themselves.
        
    #     Returns:
    #         Dictionary mapping shock names to corresponding state variables
    #     """
    #     # Initialize mapping dictionaries
    #     self.shock_to_state_map = {}  # Maps shock names to state variables
    #     self.state_to_shock_map = {}  # Maps state variables to shock names
        
    #     # Identify all RES_ variables (exogenous processes)
    #     exo_processes = {}  # Map base names to lists of (lag, var_name)
    #     for var in self.state_variables:
    #         if var.startswith("RES_") and "_lag" in var:
    #             # This is an exogenous process state
    #             base_name = var.split("_lag")[0]  # Get base name before _lag
                
    #             # Extract lag number
    #             if var.endswith("_lag"):
    #                 lag = 1  # First lag
    #             else:
    #                 # Try to extract lag number after _lag
    #                 lag_suffix = var.split("_lag")[1]
    #                 if lag_suffix and lag_suffix.isdigit():
    #                     lag = int(lag_suffix)
    #                 else:
    #                     lag = 1  # Default to first lag
                
    #             # Add to the exogenous processes dictionary
    #             if base_name not in exo_processes:
    #                 exo_processes[base_name] = []
    #             exo_processes[base_name].append((lag, var))
        
    #     # Sort processes by lag
    #     for process in exo_processes.values():
    #         process.sort()  # Sort by lag
        
    #     # Find exogenous process equations
    #     exo_eq_map = {}  # Maps base names to their equations
    #     for eq_idx, equation in enumerate(self.equations):
    #         # Clean equation for analysis
    #         clean_eq = re.sub(r'//.*', '', equation).strip()
            
    #         # Check each exogenous process
    #         for base_name in exo_processes.keys():
    #             # Current period base name (without _lag)
    #             if base_name in clean_eq and "=" in clean_eq:
    #                 # This equation defines the exogenous process
    #                 left, right = clean_eq.split("=", 1)
    #                 if base_name in left.strip():
    #                     exo_eq_map[base_name] = {
    #                         'equation': clean_eq,
    #                         'right_side': right.strip(),
    #                         'index': eq_idx
    #                     }
        
    #     # Extract shock relationships from the exogenous process equations
    #     print("Analyzing exogenous process equations:")
    #     for base_name, eq_info in exo_eq_map.items():
    #         print(f"  {base_name}: {eq_info['equation']}")
            
    #         # Check which shocks appear in this equation
    #         for shock in self.varexo_list:
    #             if shock in eq_info['right_side']:
    #                 # Get the first lag state variable for this process
    #                 if base_name in exo_processes and exo_processes[base_name]:
    #                     first_lag_var = exo_processes[base_name][0][1]
                        
    #                     # Map shock to state and vice versa
    #                     self.shock_to_state_map[shock] = first_lag_var
    #                     self.state_to_shock_map[first_lag_var] = shock
                        
    #                     print(f"    Found shock mapping: {shock} → {first_lag_var}")
    #                 else:
    #                     print(f"    Warning: No lag variables found for {base_name}")
            
    #         # If no shocks found but there should be one
    #         if base_name in exo_processes and exo_processes[base_name] and base_name not in [self.state_to_shock_map.get(v[1]) for v in exo_processes[base_name]]:
    #             print(f"    Warning: No shock found for exogenous process {base_name}")
        
    #     # Identify and categorize state variables
    #     endogenous_states = []
    #     exo_with_shocks = []
    #     exo_without_shocks = []
        
    #     for var in self.state_variables:
    #         if var.startswith("RES_") and "_lag" in var:
    #             # Check if this state is directly affected by a shock
    #             if var in self.state_to_shock_map:
    #                 exo_with_shocks.append(var)
    #             else:
    #                 # Check if this is a higher lag of a shocked variable
    #                 base_name = var.split("_lag")[0]
    #                 has_shocked_lag = False
    #                 if base_name in exo_processes:
    #                     for _, lag_var in exo_processes[base_name]:
    #                         if lag_var in self.state_to_shock_map:
    #                             has_shocked_lag = True
    #                             break
                    
    #                 # If it's a higher lag of a shocked variable, add to without_shocks
    #                 # Otherwise, treat as an exogenous state (no shocks found at all)
    #                 if has_shocked_lag:
    #                     exo_without_shocks.append(var)
    #                 else:
    #                     # Last resort: check all equations
    #                     has_shock = False
    #                     for equation in self.equations:
    #                         if base_name in equation:
    #                             for shock in self.varexo_list:
    #                                 if shock in equation:
    #                                     has_shock = True
    #                                     break
                        
    #                     if has_shock:
    #                         # There is some shock, but we couldn't find the direct mapping
    #                         # Still treat it as potentially shocked
    #                         exo_with_shocks.append(var)
    #                     else:
    #                         exo_without_shocks.append(var)
    #         else:
    #             # This is an endogenous state
    #             endogenous_states.append(var)
        
    #     # Store these categorizations for later use - ORDER MATTERS
    #     self.endogenous_states = endogenous_states
    #     self.exo_with_shocks = exo_with_shocks
    #     self.exo_without_shocks = exo_without_shocks
        
    #     # Update the state_variables list with the correct ordering for state space
    #     self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
        
    #     # Print summary
    #     print("\nExogenous Process Analysis Summary:")
    #     print(f"  Endogenous states: {len(endogenous_states)}")
    #     print(f"  Exogenous states with direct shocks: {len(exo_with_shocks)}")
    #     print(f"  Exogenous states without direct shocks: {len(exo_without_shocks)}")
    #     print(f"  Shock-to-state mappings: {len(self.shock_to_state_map)}")
        
    #     # Add extra diagnostics for AR parameters
    #     print("\nChecking for potential zero-persistence cases:")
    #     for base_name, eq_info in exo_eq_map.items():
    #         eq = eq_info['equation']
    #         # Look for AR coefficients like "rho_*"
    #         rho_params = re.findall(r'(rho_\w+)', eq)
    #         if rho_params:
    #             print(f"  {base_name}: AR parameters {rho_params}")
                
    #             # Check if these parameters could be zero
    #             for param in rho_params:
    #                 if param in self.parameters and self.parameters[param] == 0:
    #                     print(f"    WARNING: {param} = 0 (zero persistence in {base_name})")
    #         else:
    #             print(f"  {base_name}: No AR parameters found")
        
    #     return self.shock_to_state_map

    # def generate_model_structure(self):
    #     """
    #     Generate the structural components of the state space representation
    #     that don't depend on parameter values.

    #     Returns:
    #         structure: Dictionary with structural components for state space
    #     """
    #     # Get variable counts and indices
    #     n_endogenous = len(self.endogenous_states)
    #     n_exo_with_shocks = len(self.exo_with_shocks)
    #     n_exo_without_shocks = len(self.exo_without_shocks)
    #     n_controls = len(self.control_variables)
    #     n_shocks = len(self.varexo_list)
    #     n_exo_states = n_exo_with_shocks + n_exo_without_shocks
    #     n_states = n_endogenous + n_exo_states

    #     # Create shock selection matrix R
    #     R = np.zeros((n_exo_states, n_shocks))

    #     # Fill R matrix using shock_to_state_map - CRITICAL FOR ZERO PERSISTENCE CASES
    #     print("\nBuilding R matrix (shock → exogenous state mapping):")
        
    #     # First, handle direct mappings from shock_to_state_map
    #     for shock_idx, shock_name in enumerate(self.varexo_list):
    #         if shock_name in self.shock_to_state_map:
    #             state_var = self.shock_to_state_map[shock_name]
    #             try:
    #                 # Find position in full state list
    #                 state_full_idx = self.state_variables.index(state_var)
    #                 # Calculate position in exogenous state vector
    #                 exo_state_idx = state_full_idx - n_endogenous
    #                 if 0 <= exo_state_idx < n_exo_states:
    #                     R[exo_state_idx, shock_idx] = 1.0
    #                     print(f"  {shock_name} → {state_var} (R[{exo_state_idx}, {shock_idx}] = 1.0)")
    #                 else:
    #                     print(f"  Warning: Invalid exo_state_idx {exo_state_idx} for {state_var}")
    #             except ValueError:
    #                 print(f"  Warning: State variable {state_var} not found in state_variables")
        
    #     # Additional check for zero-persistence exogenous processes
    #     # If an AR process has zero persistence, ensure shock impacts are captured
    #     for shock_idx, shock_name in enumerate(self.varexo_list):
    #         for equation in self.equations:
    #             clean_eq = re.sub(r'//.*', '', equation).strip()
                
    #             # Look for equations containing both the shock and some RES_ variable
    #             if shock_name in clean_eq and "=" in clean_eq:
    #                 left, right = clean_eq.split("=", 1)
    #                 left = left.strip()
    #                 right = right.strip()
                    
    #                 # If a RES_ variable is on the left side and the shock appears on the right
    #                 res_vars = [v for v in re.findall(r'\b(RES_[A-Za-z0-9_]+)\b', left) if v in self.all_variables]
                    
    #                 for res_var in res_vars:
    #                     # See if this res_var has a lag in state_variables
    #                     res_lag_vars = [v for v in self.state_variables if v.startswith(f"{res_var}_lag")]
                        
    #                     if res_lag_vars and shock_name in right:
    #                         # Found a potential direct path - check for zero persistence 
    #                         # by looking for rho_ parameters
    #                         rho_params = re.findall(r'(rho_\w+)', clean_eq)
    #                         zero_persistence = True
                            
    #                         # Check if all rho parameters are zero or missing
    #                         for param in rho_params:
    #                             if param in self.parameters and self.parameters[param] != 0:
    #                                 zero_persistence = False
    #                                 break
                            
    #                         if zero_persistence:
    #                             # This process has zero persistence - ensure shock is mapped
    #                             for lag_var in res_lag_vars:
    #                                 try:
    #                                     state_full_idx = self.state_variables.index(lag_var)
    #                                     exo_state_idx = state_full_idx - n_endogenous
                                        
    #                                     if 0 <= exo_state_idx < n_exo_states and R[exo_state_idx, shock_idx] == 0:
    #                                         # Only set if not already set
    #                                         R[exo_state_idx, shock_idx] = 1.0
    #                                         print(f"  ZERO PERSISTENCE: Added {shock_name} → {lag_var} (R[{exo_state_idx}, {shock_idx}] = 1.0)")
    #                                 except ValueError:
    #                                     pass

    #     # Verify R matrix is properly configured
    #     if not np.any(R):
    #         print("  WARNING: R matrix has all zeros - shock transmission won't work!")
    #     else:
    #         print(f"  R matrix shape: {R.shape}")
    #         print(f"  Non-zero elements: {np.count_nonzero(R)}")

    #     # Create structure matrices for state space
    #     B_structure = np.zeros((n_states, n_shocks))
    #     B_structure[n_endogenous:, :] = R

    #     # Create C matrix structure (maps states to observables)
    #     n_observables = n_controls + n_states
    #     C_structure = np.zeros((n_observables, n_states))

    #     # States mapped one-to-one (identity matrix part)
    #     C_structure[n_controls:, :] = np.eye(n_states)

    #     # Create D matrix for direct shock impact (important for zero persistence)
    #     D_structure = np.zeros((n_observables, n_shocks))
        
    #     # Fill in D for direct shock effects on exogenous states
    #     D_structure[n_controls+n_endogenous:n_controls+n_endogenous+n_exo_states, :] = R
        
    #     # Also examine direct shock effects on controls
    #     # For each control variable, check if it depends on an exogenous process
    #     for i, control_var in enumerate(self.control_variables):
    #         for j, shock_name in enumerate(self.varexo_list):
    #             # Check all equations to see if this control depends on an exogenous process
    #             # that is directly affected by this shock
    #             for equation in self.equations:
    #                 clean_eq = re.sub(r'//.*', '', equation).strip()
                    
    #                 # If the equation defines this control variable
    #                 if "=" in clean_eq and control_var in clean_eq.split("=")[0].strip():
    #                     # Check if the right side contains any RES_ variable and the shock
    #                     right_side = clean_eq.split("=")[1].strip()
                        
    #                     # Find all RES_ variables on the right side
    #                     res_vars = re.findall(r'\b(RES_[A-Za-z0-9_]+)\b', right_side)
                        
    #                     for res_var in res_vars:
    #                         # Check if this RES variable is related to the shock
    #                         for state_var, shock in self.state_to_shock_map.items():
    #                             if shock == shock_name and state_var.startswith(f"{res_var}_lag"):
    #                                 # Found a path from shock to control through a RES variable
    #                                 # Now check for zero persistence
    #                                 rho_params = re.findall(r'(rho_\w+)', equation)
    #                                 zero_persistence = True
                                    
    #                                 for param in rho_params:
    #                                     if param in self.parameters and self.parameters[param] != 0:
    #                                         zero_persistence = False
    #                                         break
                                    
    #                                 # With zero persistence, shock directly affects control
    #                                 if zero_persistence:
    #                                     D_structure[i, j] = 1.0
    #                                     print(f"  Direct effect: {shock_name} → {control_var} (D[{i}, {j}] = 1.0)")

    #     # Store indices for fast matrix creation later
    #     indices = {
    #         'n_endogenous': n_endogenous,
    #         'n_exo_states': n_exo_states,
    #         'n_controls': n_controls,
    #         'n_shocks': n_shocks,
    #         'n_states': n_states,
    #         'n_observables': n_observables
    #     }

    #     # Create labels for the state space
    #     labels = {
    #         'state_labels': self.state_variables,
    #         'observable_labels': self.control_variables + self.state_variables,
    #         'shock_labels': self.varexo_list,
    #         'shock_to_state_map': self.shock_to_state_map,
    #         'state_to_shock_map': self.state_to_shock_map
    #     }

    #     return {
    #         'indices': indices,
    #         'R': R,
    #         'B_structure': B_structure,
    #         'C_structure': C_structure,
    #         'D_structure': D_structure,  # Added D_structure for direct effects
    #         'labels': labels
    #     }

    def analyze_exogenous_processes(self):
        """
        Analyze exogenous processes in the model and map them to their corresponding shocks.
        
        This function:
        1. Identifies all exogenous processes (variables with prefix RES_)
        2. Maps these processes to their corresponding shocks
        3. Identifies processes with zero persistence
        
        Returns:
            Dictionary mapping shock names to corresponding state variables
        """
        print("\n--- Analyzing Exogenous Processes ---")
        
        # Initialize mapping dictionaries
        self.shock_to_state_map = {}  # Maps shock names to state variables
        self.state_to_shock_map = {}  # Maps state variables to shock names
        
        # First, identify all exogenous processes in the model
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
        
        # Next, find the equation for each exogenous process
        for eq_idx, equation in enumerate(self.equations):
            clean_eq = re.sub(r'//.*', '', equation).strip()
            if "=" not in clean_eq:
                continue
                
            left, right = [s.strip() for s in clean_eq.split("=", 1)]
            
            # Check if this equation defines an exogenous process
            for base_name in exo_processes.keys():
                if base_name == left:
                    print(f"\nFound equation for {base_name}: {clean_eq}")
                    
                    # Look for shocks in the right-hand side
                    for shock in self.varexo_list:
                        if shock in right:
                            print(f"  Process {base_name} is driven by shock {shock}")
                            
                            # Map this shock to the first lag of the process
                            if base_name in exo_processes and exo_processes[base_name]:
                                first_lag = exo_processes[base_name][0][1]
                                self.shock_to_state_map[shock] = first_lag
                                self.state_to_shock_map[first_lag] = shock
                            
                    # Check for zero persistence by looking for AR parameters
                    ar_params = re.findall(r'(rho_\w+)\s*\*', clean_eq)
                    print(f"  AR parameters: {ar_params}")
                    
                    zero_persistence = True
                    for param in ar_params:
                        if param in self.parameters:
                            value = self.parameters[param]
                            print(f"    {param} = {value}")
                            if abs(value) > 1e-10:  # Non-zero persistence
                                zero_persistence = False
                    
                    if zero_persistence:
                        print(f"  WARNING: {base_name} has ZERO PERSISTENCE")
        
        # Categorize state variables
        endogenous_states = []
        exo_with_shocks = []
        exo_without_shocks = []
        
        for var in self.state_variables:
            if var.startswith("RES_") and "_lag" in var:
                # Exogenous state
                if var in self.state_to_shock_map:
                    exo_with_shocks.append(var)
                else:
                    exo_without_shocks.append(var)
            else:
                # Endogenous state
                endogenous_states.append(var)
        
        # Store categorized states
        self.endogenous_states = endogenous_states
        self.exo_with_shocks = exo_with_shocks
        self.exo_without_shocks = exo_without_shocks
        
        # Update state_variables with correct ordering
        self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
        
        # Print summary
        print("\nState Variable Categorization:")
        print(f"  Endogenous states: {len(endogenous_states)}")
        print(f"  Exogenous states with shocks: {len(exo_with_shocks)}")
        print(f"  Exogenous states without shocks: {len(exo_without_shocks)}")
        
        return self.shock_to_state_map


    def generate_model_structure(self):
        """
        Generate the structural components of the state space representation
        that don't depend on parameter values, following the correct mathematical formulation.

        Returns:
            Dictionary with structural components for state space
        """
        print("\n--- Generating Model Structure ---")
        
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
        
        # Define direct shock impacts
        # First, check which exogenous processes have zero persistence
        zero_persistence_processes = set()
        
        for equation in self.equations:
            clean_eq = re.sub(r'//.*', '', equation).strip()
            if "=" not in clean_eq:
                continue
                
            left, right = [s.strip() for s in clean_eq.split("=", 1)]
            
            # Check if this is an exogenous process equation
            if left.startswith("RES_"):
                # Check if any AR parameters are in the equation
                ar_params = re.findall(r'(rho_\w+)\s*\*', clean_eq)
                
                all_zero = True
                for param in ar_params:
                    if param in self.parameters and abs(self.parameters[param]) > 1e-10:
                        all_zero = False
                        break
                
                if all_zero:
                    zero_persistence_processes.add(left)
                    print(f"Zero persistence detected for {left}")
        
        # Structures defining state-space matrices
        # B_structure maps shocks to state transitions
        B_structure = np.zeros((n_states, n_shocks))
        
        # Fill in the exogenous part of B - maps shocks to exogenous states
        B_structure[n_endogenous:, :] = R
        
        # C_structure maps states to observables
        C_structure = np.zeros((n_controls + n_states, n_states))
        
        # States appear directly in observables
        C_structure[n_controls:, :] = np.eye(n_states)
        
        # D_structure maps shocks directly to observables
        D_structure = np.zeros((n_controls + n_states, n_shocks))
        
        # Direct shock to exogenous state mapping
        D_structure[n_controls + n_endogenous:, :] = R
        
        # Store indices for later use
        indices = {
            'n_endogenous': n_endogenous,
            'n_exo_states': n_exo_states,
            'n_controls': n_controls,
            'n_shocks': n_shocks,
            'n_states': n_states,
            'n_observables': n_controls + n_states,
            'zero_persistence_processes': list(zero_persistence_processes)
        }
        
        # Store labels and mappings
        labels = {
            'state_labels': self.state_variables,
            'observable_labels': self.control_variables + self.state_variables,
            'shock_labels': self.varexo_list,
            'shock_to_state_map': self.shock_to_state_map,
            'state_to_shock_map': self.state_to_shock_map,
            'zero_persistence_processes': list(zero_persistence_processes)
        }
        
        return {
            'indices': indices,
            'R': R,
            'B_structure': B_structure,
            'C_structure': C_structure,
            'D_structure': D_structure,
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

