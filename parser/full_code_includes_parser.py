# dsge_trend_filter.py

import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
import importlib.util
import simdkalman
import re
import scipy.linalg as la
from scipy.linalg import lu_factor, lu_solve, norm
from typing import Dict, List, Optional, Tuple, Any
import sympy as sy  # Import sympy
# Utility functions for trend components
# def calculate_trend_positions(model_specs, obs_vars, n_states, n_shocks):
#     """
#     Calculate positions and parameter names for trend components.

#     Returns:
#         dict: Information about trend component positions and parameters
#     """
#     trend_info = {
#         'state_labels': [],
#         'components': {},
#         'total_states': 0,
#         'param_positions': {},
#         'shock_std_param_names': [],
#         'total_trend_shocks': 0  # New: Total number of trend shocks
#     }

#     pos = n_states
#     shock_pos = n_shocks
#     total_trend_shocks = 0 # Initialize counter

#     for obs_var in obs_vars:
#         trend_type = model_specs[obs_var]['trend']
#         var_info = {'type': trend_type, 'components': {}}

#         if trend_type == 'random_walk':
#             var_info['components']['level'] = {
#                 'state_pos': pos,
#                 'shock_pos': shock_pos
#             }
#             param_name = f"{obs_var}_level_shock_std"  # Add _std suffix
#             trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
#             trend_info['shock_std_param_names'].append(param_name)
#             total_trend_shocks += 1  # Increment counter

#             trend_info['state_labels'].append(f"{obs_var}_level")
#             pos += 1
#             shock_pos += 1

#         elif trend_type == 'second_difference':
#             # ... (Similar modifications for other trend types)
#             var_info['components']['level'] = {
#                 'state_pos': pos,
#                 'shock_pos': shock_pos
#             }
#             param_name = f"{obs_var}_level_shock_std"  # Add _std suffix
#             trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
#             trend_info['shock_std_param_names'].append(param_name)
#             total_trend_shocks += 1 # Increment counter

#             trend_info['state_labels'].append(f"{obs_var}_level")
#             pos += 1
#             shock_pos += 1

#             var_info['components']['slope'] = {
#                 'state_pos': pos,
#                 'shock_pos': shock_pos
#             }
#             param_name = f"{obs_var}_slope_shock_std"  # Add _std suffix
#             trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
#             trend_info['shock_std_param_names'].append(param_name)
#             total_trend_shocks += 1 # Increment counter
#             trend_info['state_labels'].append(f"{obs_var}_slope")
#             pos += 1
#             shock_pos += 1

#             var_info['components']['curvature'] = {
#                 'state_pos': pos,
#                 'shock_pos': shock_pos
#             }
#             param_name = f"{obs_var}_curvature_shock_std"  # Add _std suffix
#             trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
#             trend_info['shock_std_param_names'].append(param_name)
#             total_trend_shocks += 1 # Increment counter
#             trend_info['state_labels'].append(f"{obs_var}_curvature")
#             pos += 1
#             shock_pos += 1

#         elif trend_type == 'constant_mean':
#             # ... (Similar modifications for other trend types)
#             var_info['components']['level'] = {
#                 'state_pos': pos,
#                 'shock_pos': shock_pos
#             }
#             param_name = f"{obs_var}_mean_shock_std"  # Add _std suffix
#             trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
#             trend_info['shock_std_param_names'].append(param_name)
#             total_trend_shocks += 1  # Increment counter

#             trend_info['state_labels'].append(f"{obs_var}_mean")
#             pos += 1
#             shock_pos += 1

#         # Store information for this variable
#         trend_info['components'][obs_var] = var_info

#     trend_info['total_states'] = pos - n_states
#     trend_info['total_trend_shocks'] = total_trend_shocks # Store the total count

#     return trend_info

# def build_trend_transition(T_matrix, trend_info, n_states):
#     """
#     Build the transition matrix blocks for trend components.
    
#     Args:
#         T_matrix: Transition matrix to fill
#         trend_info: Trend component position information
#         n_states: Number of original state variables
#     """
#     # Process each observed variable
#     for obs_var, var_info in trend_info['components'].items():
#         trend_type = var_info['type']
        
#         # Set transition dynamics based on trend type
#         if trend_type == 'random_walk':
#             # Level follows random walk: level_t = level_{t-1} + shock
#             level_pos = var_info['components']['level']['state_pos']
#             T_matrix[level_pos, level_pos] = 1.0
            
#         elif trend_type == 'second_difference':
#             # Level follows: level_t = level_{t-1} + slope_{t-1} + shock
#             # Slope follows: slope_t = slope_{t-1} + curvature_{t-1} + shock
#             # Curvature follows: curvature_t = curvature_{t-1} + shock
#             level_pos = var_info['components']['level']['state_pos']
#             slope_pos = var_info['components']['slope']['state_pos']
#             curv_pos = var_info['components']['curvature']['state_pos']
            
#             # Level depends on itself and slope
#             T_matrix[level_pos, level_pos] = 1.0
#             T_matrix[level_pos, slope_pos] = 1.0
            
#             # Slope depends on itself and curvature
#             T_matrix[slope_pos, slope_pos] = 1.0
#             T_matrix[slope_pos, curv_pos] = 1.0
            
#             # Curvature depends only on itself
#             T_matrix[curv_pos, curv_pos] = 1.0
            
#         elif trend_type == 'constant_mean':
#             # Constant mean doesn't change: mean_t = mean_{t-1} + small_shock
#             mean_pos = var_info['components']['level']['state_pos']
#             T_matrix[mean_pos, mean_pos] = 1.0

# def build_trend_selection(R_matrix, trend_info, n_states, n_shocks):
#     """
#     Build the selection matrix blocks for trend components.
    
#     Args:
#         R_matrix: Selection matrix to fill
#         trend_info: Trend component position information
#         n_states: Number of original state variables
#         n_shocks: Number of original shock variables
#     """
#     # For each variable and its trend components
#     for obs_var, var_info in trend_info['components'].items():
#         for comp_name, comp_info in var_info['components'].items():
#             state_pos = comp_info['state_pos']
#             shock_pos = comp_info['shock_pos']
            
#             # Direct mapping from shock to state
#             R_matrix[state_pos, shock_pos] = 1.0

# def build_trend_observation(Z_matrix, trend_info, obs_vars, observable_labels):
#     """
#     Build the observation matrix blocks for trend components.
    
#     Args:
#         Z_matrix: Observation matrix to fill
#         trend_info: Trend component position information
#         obs_vars: List of observed variable names
#         observable_labels: List of all observable labels
#     """
#     # For each observed variable
#     for obs_var in obs_vars:
#         # Find base variable (without _obs)
#         base_var = obs_var.replace("_obs", "")
        
#         # Find the index in the observable list
#         try:
#             var_idx = observable_labels.index(base_var)
#         except ValueError:
#             continue  # Skip if not found
        
#         # Get trend components for this variable
#         var_info = trend_info['components'].get(obs_var, {})
#         if not var_info:
#             continue
            
#         # Map the appropriate trend component to the observation
#         if 'level' in var_info['components']:
#             level_pos = var_info['components']['level']['state_pos']
#             Z_matrix[var_idx, level_pos] = 1.0





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
        
        # Identify direct shock states
        direct_shock_states = [process_lags[0][1] for process_name, process_lags in exo_processes.items() if process_lags]
        self.direct_shock_states = direct_shock_states

        # Format equations for output
        formatted_equations = self.format_transformed_equations(transformed_equations, aux_equations)

        # Store these categorizations for later use - NO SORTING
        self.endogenous_states = endogenous_states  
        self.exo_with_shocks = exo_with_shocks      
        self.exo_without_shocks = exo_without_shocks
        
        # Update the state_variables list with the correct ordering for state space
        self.state_variables = self.endogenous_states + self.exo_with_shocks + self.exo_without_shocks
    
        
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
            'direct_shock_states': self.direct_shock_states #   Keep track of states with direct shocks
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


    def generate_model_structure(self):
        """
        Generate the structural components of the state space representation
        with consistent variable ordering.
        """
        # Get variable counts
        n_endogenous = len(self.endogenous_states)
        n_exo_with_shocks = len(self.exo_with_shocks)
        n_exo_without_shocks = len(self.exo_without_shocks)
        n_controls = len(self.control_variables)
        n_shocks = len(self.varexo_list)
        n_exo_states = n_exo_with_shocks + n_exo_without_shocks
        n_states = n_endogenous + n_exo_states
        
        # Create indices dictionary
        indices = {
            'n_endogenous': n_endogenous,
            'n_exo_states': n_exo_states,
            'n_controls': n_controls,
            'n_shocks': n_shocks,
            'n_states': n_states,
            'n_observables': n_controls + n_states
        }
        
        # Create shock selection matrix R
        R = np.zeros((n_exo_states, n_shocks))
        
        # Map shocks to exogenous states using the state_to_shock_map
        # First create inverse mapping from shock to exo state position
        shock_to_exo_pos = {}
        for exo_pos, state_var in enumerate(self.exo_with_shocks):
            if state_var in self.state_to_shock_map:
                shock_name = self.state_to_shock_map[state_var]
                shock_to_exo_pos[shock_name] = exo_pos
        
        # Fill R matrix using the mapping
        for shock_idx, shock_name in enumerate(self.varexo_list):
            if shock_name in shock_to_exo_pos:
                exo_state_idx = shock_to_exo_pos[shock_name]
                R[exo_state_idx, shock_idx] = 1.0
        
        # Create B matrix structure
        B_structure = np.zeros((n_states, n_shocks))
        B_structure[n_endogenous:, :] = R
        
        # Create C matrix structure
        C_structure = np.zeros((indices['n_observables'], n_states))
        
        # States mapped one-to-one in C matrix
        C_structure[n_controls:, :] = np.eye(n_states)
        
        # D matrix
        D = np.zeros((indices['n_observables'], n_shocks))
        
        # Create labels with consistent ordering
        labels = {
            'state_labels': self.state_variables,
            'observable_labels': self.control_variables + self.state_variables,
            'shock_labels': self.varexo_list
        }
        
        structure = {
            'indices': indices,
            'R': R,
            'B_structure': B_structure,
            'C_structure': C_structure,
            'D': D,
            'labels': labels
        }
        
        # Add trend structures if needed
        if hasattr(self, 'obs_vars') and self.obs_vars and hasattr(self, 'model_specs'):
            # Build trend structures (implementation from previous function)
            structure = self._add_trend_structures(structure)
        
        return structure

    def _add_trend_structures(self, structure):
        """Helper method to build trend structures"""
        n_observables = len(self.obs_vars)
        
        # Count trend states
        n_trend_states = 0
        for obs_var in self.obs_vars:
            trend_type = self.model_specs[obs_var]['trend']
            if trend_type == 'random_walk':
                n_trend_states += 1
            elif trend_type == 'second_difference':
                n_trend_states += 3
            elif trend_type == 'constant_mean':
                n_trend_states += 1
        
        # Create trend matrices
        if n_trend_states > 0:
            T_trend = np.zeros((n_trend_states, n_trend_states))
            R_trend = np.eye(n_trend_states)
            C_trend = np.zeros((n_observables, n_trend_states))
            
            trend_pos = 0
            for i, obs_var in enumerate(self.obs_vars):
                trend_type = self.model_specs[obs_var]['trend']
                
                if trend_type == 'random_walk':
                    C_trend[i, trend_pos] = 1.0
                    T_trend[trend_pos, trend_pos] = 1.0
                    trend_pos += 1
                elif trend_type == 'second_difference':
                    C_trend[i, trend_pos] = 1.0
                    T_trend[trend_pos, trend_pos] = 1.0
                    T_trend[trend_pos, trend_pos+1] = 1.0
                    T_trend[trend_pos+1, trend_pos+1] = 1.0
                    T_trend[trend_pos+1, trend_pos+2] = 1.0
                    T_trend[trend_pos+2, trend_pos+2] = 1.0
                    trend_pos += 3
                elif trend_type == 'constant_mean':
                    C_trend[i, trend_pos] = 1.0
                    T_trend[trend_pos, trend_pos] = 1.0
                    trend_pos += 1
            
            structure['T_trend_structure'] = T_trend.tolist()
            structure['R_trend_structure'] = R_trend.tolist()
            structure['C_trend_structure'] = C_trend.tolist()
            structure['n_trend_states'] = n_trend_states
            
            # Add observation mapping
            structure['obs_mapping'] = self._create_obs_mapping()
        
        return structure

    def _create_obs_mapping(self):
        """Helper method to create observation mapping"""
        obs_mapping = {}
        
        for i, obs_var in enumerate(self.obs_vars):
            cycle_var = self.model_specs[obs_var]['cycle']
            
            if cycle_var in self.control_variables:
                obs_mapping[obs_var] = {
                    'type': 'control',
                    'index': self.control_variables.index(cycle_var),
                    'obs_index': i
                }
            elif cycle_var in self.state_variables:
                obs_mapping[obs_var] = {
                    'type': 'state',
                    'index': self.state_variables.index(cycle_var),
                    'obs_index': i
                }
        
        return obs_mapping
    

import numpy as np
import os
import sys
import json
import importlib.util
from typing import Dict, List, Optional, Tuple, Any

class ModelSolver:
    """
    Solves the DSGE model and creates the core state-space representation 
    based on parameter values. Designed to work with AugmentedStateSpace class.
    """
    def __init__(self, output_dir, model_specs, obs_vars):
        """
        Initialize the ModelSolver.

        Args:
            output_dir: Directory containing model files ('model.json',
                        'jacobian_evaluator.py', 'model_structure.py').
            model_specs: Dictionary with specifications for observed variables.
            obs_vars: List of observed variable names (must match data columns).
        """
        self.output_dir = output_dir
        self.model_specs = model_specs
        self.obs_vars = obs_vars

        # Load model components
        self.load_model()               # Loads self.model (from model.json)
        self.load_jacobian_evaluator()  # Loads self.evaluate_jacobians
        self.load_model_structure()     # Loads self.indices, self.labels, self.R
        
        # Define parameters ordering
        self._define_theta_order()

        # Validate model specifications against loaded model/labels
        self.check_spec_vars()

        # Initialize matrices that will be set when solve_model is called
        self.f = None  # Policy function for controls: c_t = f*s_t
        self.p = None  # State transition matrix: s_{t+1} = p*s_t
        
        # Derived matrices (set after solve_model is called)
        self.state_transition = None  # Same as p, for compatibility
        self.impulse_matrix = None  # B matrix for shock impact
        
        # Additional properties for compatibility with AugmentedStateSpace
        self.var_names = self.model.get('all_variables', [])
        self.param_names = self.model.get('parameters', [])
        self.shock_names = self.model.get('shocks', [])

    def check_spec_vars(self):
        """Check that model specifications match variables in the model."""
        if not hasattr(self, 'model') or not hasattr(self, 'labels'):
            print("Warning: Cannot check spec vars, model or labels not loaded.")
            return

        list_model_names = self.model.get('all_variables', [])
        if not list_model_names:
            print("Warning: 'all_variables' empty in model.json, cannot check cycle vars.")
            return

        all_cycle_vars_found = True
        for obs_var, spec in self.model_specs.items():
            # Check cycle variable existence
            cycle_var = spec.get('cycle')
            if cycle_var is None:
                print(f"Warning: 'cycle' key missing in model_specs for {obs_var}.")
                continue  # Skip if spec incomplete
            if cycle_var not in list_model_names:
                print(f"Error: Cycle variable '{cycle_var}' for '{obs_var}' not found in model variables.")
                all_cycle_vars_found = False

            # Check trend specification existence and type
            if 'trend' not in spec:
                print(f"Error: Trend specification missing for {obs_var}")
                all_cycle_vars_found = False  # Treat as error
                continue
            trend_type = spec['trend']
            valid_trends = ['random_walk', 'second_difference', 'constant_mean']
            if trend_type not in valid_trends:
                print(f"Error: Invalid trend type '{trend_type}' for {obs_var}. Must be one of {valid_trends}")
                all_cycle_vars_found = False

        if not all_cycle_vars_found:
            raise ValueError("Model specification check failed. See errors above.")
        print("Model specification check passed.")
        
    def _define_theta_order(self):
        """
        Defines the strict order of parameters expected in the 'theta' vector.
        Stores the order in self.theta_param_names.
        """
        # 1. DSGE core parameters (from param_labels)
        dsge_core_params = list(self.labels['param_labels'])

        # 2. DSGE shock standard deviations
        dsge_shock_std_params = [f"{state}_std" for state in self.model.get('direct_shock_states', [])]

        # 3. Trend shock standard deviations - using observable variables
        trend_shock_std_params = [f"{obs_var}_level_shock_std" for obs_var in self.obs_vars]
        
        # Add slope shock stds for second_difference trends
        for obs_var, spec in self.model_specs.items():
            if spec.get('trend') == 'second_difference':
                trend_shock_std_params.append(f"{obs_var}_slope_shock_std")

        # Combine all parameter names IN ORDER
        self.theta_param_names = dsge_core_params + dsge_shock_std_params + trend_shock_std_params

        # Store counts for easy splitting later
        self.n_dsge_core_params = len(dsge_core_params)
        self.n_dsge_shock_std_params = len(dsge_shock_std_params)
        self.n_trend_shock_std_params = len(trend_shock_std_params)

        # Total expected length of theta
        self.expected_theta_length = self.n_dsge_core_params + self.n_dsge_shock_std_params + self.n_trend_shock_std_params 

        print(f"Theta vector order defined. Expected length: {self.expected_theta_length}")

    def load_model(self):
        """Load the model from the JSON file."""
        model_path = os.path.join(self.output_dir, "model.json")
        try:
            with open(model_path, 'r') as f:
                self.model = json.load(f)
            # Ensure essential keys are present
            if 'parameters' not in self.model:
                raise KeyError("'parameters' key missing in model.json")
            if 'all_variables' not in self.model:
                raise KeyError("'all_variables' key missing in model.json")

        except FileNotFoundError:
            print(f"Error: model.json not found at {model_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {model_path}")
            raise
        except KeyError as e:
            print(f"Error: Missing essential key in model.json: {e}")
            raise

    def load_jacobian_evaluator(self):
        """Load the Jacobian evaluator function."""
        jac_path = os.path.join(self.output_dir, "jacobian_evaluator.py")
        try:
            spec = importlib.util.spec_from_file_location("jacobian_evaluator", jac_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {jac_path}")
            self.jacobian_module = importlib.util.module_from_spec(spec)
            sys.modules["jacobian_evaluator"] = self.jacobian_module  # Add to sys.modules
            spec.loader.exec_module(self.jacobian_module)
            self.evaluate_jacobians = getattr(self.jacobian_module, 'evaluate_jacobians')
        except FileNotFoundError:
            print(f"Error: jacobian_evaluator.py not found at {jac_path}")
            raise
        except AttributeError:
            print(f"Error: 'evaluate_jacobians' function not found in {jac_path}")
            raise
        except Exception as e:
            print(f"Error loading Jacobian evaluator from {jac_path}: {e}")
            raise

    def load_model_structure(self):
        """Load the pre-computed model structure."""
        struct_path = os.path.join(self.output_dir, "model_structure.py")
        try:
            spec = importlib.util.spec_from_file_location("model_structure", struct_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {struct_path}")

            struct_module = importlib.util.module_from_spec(spec)
            sys.modules["model_structure"] = struct_module  # Add to sys.modules
            spec.loader.exec_module(struct_module)

            # Load required attributes
            self.indices = getattr(struct_module, 'indices')
            self.labels = getattr(struct_module, 'labels')
            self.R = getattr(struct_module, 'R')  # Shock-to-state mapping
            self.B_structure = getattr(struct_module, 'B_structure')
            self.C_structure = getattr(struct_module, 'C_structure')
            self.D = getattr(struct_module, 'D')

            # Load observation mapping if available
            if hasattr(struct_module, 'obs_mapping'):
                self.obs_mapping = getattr(struct_module, 'obs_mapping')

            # Validate loaded structure
            required_indices = ['n_states', 'n_shocks', 'n_controls', 'n_observables', 'n_endogenous']
            if not all(k in self.indices for k in required_indices):
                raise AttributeError(f"Missing required keys in 'indices' from {struct_path}")
            
            required_labels = ['state_labels', 'observable_labels', 'shock_labels', 'param_labels']
            if not all(k in self.labels for k in required_labels):
                raise AttributeError(f"Missing required keys in 'labels' from {struct_path}")

            # Check dimensions
            n_exo_states = self.indices['n_states'] - self.indices['n_endogenous']
            if self.R.shape != (n_exo_states, self.indices['n_shocks']):
                print(f"Warning: Shape of loaded R {self.R.shape} does not match expected ({n_exo_states}, {self.indices['n_shocks']})")

        except FileNotFoundError:
            print(f"Error: model_structure.py not found at {struct_path}")
            raise
        except AttributeError as e:
            print(f"Error: Missing expected attribute in {struct_path}: {e}")
            raise
        except Exception as e:
            print(f"Error loading model structure from {struct_path}: {e}")
            raise

    def solve_model(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the DSGE model using Klein's method.
        
        Args:
            theta: NumPy array containing parameter values in the predefined order
                  (only the core DSGE parameters are used here)
        
        Returns:
            f: Policy function matrix for controls
            p: State transition matrix
        """
        # Input validation
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)  # Ensure theta is a numpy array
        if theta.ndim != 1:
            raise ValueError(f"Input 'theta' must be a 1D array, but got shape {theta.shape}")
        
        # Extract DSGE core parameters (only these affect Klein solution)
        dsge_core_params = theta[:self.n_dsge_core_params]
        
        # Get Jacobians from evaluator
        try:
            a, b, c_jac = self.evaluate_jacobians(dsge_core_params)
        except Exception as e:
            print(f"Error during Jacobian evaluation: {e}")
            raise
        
        # Solve using Klein method
        try:
            f, p, stab, eig = klein(a, b, self.indices['n_states'])
            
            # Store results for later use
            self.f = f  # Policy function for controls
            self.p = p  # State transition matrix
            self.state_transition = p  # Alias for compatibility
            
            # Create impulse matrix (B) from structure
            n_states = self.indices['n_states']
            n_shocks = self.indices['n_shocks']
            
            # B matrix needs to be constructed based on shock std devs
            B = np.zeros((n_states, n_shocks))
            n_endogenous = self.indices['n_endogenous']
            
            # Only exogenous states receive shocks directly
            # The R matrix defines which shock affects which state
            B[n_endogenous:, :] = self.R
            
            self.impulse_matrix = B
            
            if stab != 0:
                print("Warning: Klein solver indicates potential instability.")
                
            return f, p
            
        except Exception as e:
            print(f"Error during Klein solution: {e}")
            raise

    def create_shock_covariance_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Creates the shock covariance matrix for DSGE shocks.
        
        Args:
            theta: Parameter vector containing shock standard deviations
        
        Returns:
            QQ: Shock covariance matrix (n_shocks × n_shocks)
        """
        n_shocks = self.indices['n_shocks']
        
        # Extract shock standard deviations from theta
        shock_std_start = self.n_dsge_core_params
        shock_std_end = shock_std_start + self.n_dsge_shock_std_params
        
        shock_stds = theta[shock_std_start:shock_std_end]
        
        # Create diagonal covariance matrix using variances (std^2)
        QQ = np.diag(shock_stds**2)
        
        return QQ
    
    def create_theta_vector(self, param_dict: Dict[str, float]) -> np.ndarray:
        """
        Creates parameter vector from dictionary in the correct order.
        
        Args:
            param_dict: Dictionary of parameter names and values
            
        Returns:
            theta: Parameter vector in the correct order
        """
        # Check for missing parameters
        missing_params = set(self.theta_param_names) - set(param_dict.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Create ordered theta array
        theta = np.array([param_dict[param] for param in self.theta_param_names])
        
        return theta
    
    def get_model_components(self, theta: np.ndarray = None) -> Dict[str, Any]:
        """
        Returns core model components needed by AugmentedStateSpace with consistent ordering.
        """
        # Solve model if theta provided and model not already solved
        if theta is not None:
            self.solve_model(theta)
            self.QQ = self.create_shock_covariance_matrix(theta)
        
        # Check if model has been solved
        if self.f is None or self.p is None:
            raise ValueError("Model has not been solved. Call solve_model first.")
        
        # Ensure var_names follows the same order used in Jacobian computation
        # The order should be: state_variables followed by control_variables
        var_names = self.model['state_variables'] + self.model['control_variables']
        
        # Ensure shock_names follows the order from the parser
        shock_names = self.model['shocks']
        
        # Return components with consistent ordering
        return {
            "f": self.f,
            "p": self.p,
            "state_transition": self.state_transition,
            "impulse_matrix": self.impulse_matrix,
            "QQ": getattr(self, 'QQ', np.eye(self.indices['n_shocks'])),
            "var_names": var_names,
            "shock_names": shock_names,
            "param_names": self.model['parameters'],
            "n_states": self.indices['n_states'],
            "n_controls": self.indices['n_controls'],
            "n_shocks": self.indices['n_shocks'],
            "n_endogenous": self.indices['n_endogenous']
        }



class DataProcessor:
    """
    Handles data preparation and reshaping for the simdkalman library.
    Optimizes by doing the heavy transformation work once and caching results.
    """
    
    def __init__(self, obs_vars: List[str]):
        """
        Initialize the data processor.
        
        Args:
            obs_vars: List of observed variable names in the data
        """
        self.obs_vars = obs_vars
        self.n_obs_vars = len(obs_vars)
        self._cached_data = None
        self._cached_dates = None
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[Any]]:
        """
        Prepare DataFrame data for use with the simdkalman library.
        Caches results to avoid repeated transformation.
        
        Args:
            data: Pandas DataFrame with observed data
                Must have the observed variables as columns
                
        Returns:
            reshaped_data: numpy array with shape (1, n_timesteps, n_obs_vars)
            dates: list of datetime indices from the original DataFrame
        """
        # Return cached data if available and data hasn't changed
        if self._cached_data is not None and self._is_same_data(data):
            return self._cached_data, self._cached_dates
        
        # Ensure all required columns are present
        for var in self.obs_vars:
            if var not in data.columns:
                raise ValueError(f"Required observed variable {var} not found in data")
        
        # Extract observed data in the correct order
        obs_data = data[self.obs_vars].values
        
        # Store dates for reference
        dates = data.index.tolist()
        
        # Reshape to (1, n_timesteps, n_observed_vars) for simdkalman
        n_timesteps = obs_data.shape[0]
        reshaped_data = obs_data.reshape(1, n_timesteps, self.n_obs_vars)
        
        # Handle missing values (NaN) if any
        # simdkalman can handle NaNs, but we should ensure all values are float
        reshaped_data = reshaped_data.astype(float)
        
        # Cache the results
        self._cached_data = reshaped_data
        self._cached_dates = dates
        
        return reshaped_data, dates
    
    def _is_same_data(self, data: pd.DataFrame) -> bool:
        """
        Check if the provided data matches the cached data.
        
        Args:
            data: DataFrame to check against cache
            
        Returns:
            True if data matches cached data, False otherwise
        """
        if self._cached_data is None or self._cached_dates is None:
            return False
        
        if len(data) != len(self._cached_dates):
            return False
        
        if not data.index.equals(pd.DatetimeIndex(self._cached_dates)):
            return False
        
        return True
    
    def process_results(self, simdkalman_results, state_labels: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Process the results from simdkalman into pandas DataFrames with proper indexing.
        
        Args:
            simdkalman_results: Results from simdkalman.KalmanFilter.smooth()
            state_labels: List of state variable names
            
        Returns:
            processed_results: Dictionary with DataFrames for filtered states, smoothed states, etc.
        """
        # Extract dates from cache
        if self._cached_dates is None:
            raise ValueError("No cached dates available. Run prepare_data first.")
        
        dates = self._cached_dates
        
        # Get smoothed state means (n_batches, n_timesteps, n_states)
        smoothed_means = simdkalman_results.smoothed_means[0]  # Take first batch
        
        # Create DataFrame for smoothed states
        smoothed_states_df = pd.DataFrame(
            smoothed_means, 
            index=dates,
            columns=state_labels
        )
        
        # Get filtered state means
        filtered_means = simdkalman_results.filtered_means[0]  # Take first batch
        
        # Create DataFrame for filtered states
        filtered_states_df = pd.DataFrame(
            filtered_means, 
            index=dates,
            columns=state_labels
        )
        
        # Get state covariances (diagonal elements for state variances)
        n_timesteps = smoothed_means.shape[0]
        n_states = smoothed_means.shape[1]
        
        smoothed_vars = np.zeros((n_timesteps, n_states))
        for t in range(n_timesteps):
            # Extract diagonal elements (variances)
            smoothed_vars[t, :] = np.diag(simdkalman_results.smoothed_covariances[0, t])
        
        # Create DataFrame for smoothed state variances
        smoothed_vars_df = pd.DataFrame(
            smoothed_vars,
            index=dates,
            columns=[f"{state}_var" for state in state_labels]
        )
        
        # Package results
        processed_results = {
            'smoothed_states': smoothed_states_df,
            'filtered_states': filtered_states_df,
            'smoothed_variances': smoothed_vars_df,
            # Original simdkalman results for advanced analysis
            'raw_results': simdkalman_results
        }
        
        return processed_results


class AugmentedStateSpace:
    def __init__(self, model_solver, model_specs, param_dict):
        """
        Initializes the AugmentedStateSpace class with the DSGE model solution and 
        creates an augmented state space representation with trend components.

        Args:
            model_solver: The ModelSolver instance with solved DSGE model
            model_specs: Dictionary with specifications for observed variables
            param_dict: Dictionary of parameter names and values
        """
        self.model_solver = model_solver
        self.model_specs = model_specs
        self.param_dict = param_dict
        
        # Get model dimensions
        self.n_states = model_solver.indices['n_states']
        self.n_shocks = model_solver.indices['n_shocks']
        self.n_observables = len(model_specs)
        
        # Get parameter order from model_solver
        self.theta_param_names = model_solver.theta_param_names
        
        # Create theta vector in correct order
        self.theta = self._create_theta_vector()
        
        # Initialize trend-related attributes
        self.trend_variables = list(model_specs.keys())
        self.trend_types = {}
        self.n_trend_states = 0
        self.n_trend_shocks = 0
        self.trend_shock_names = {}

        # Constants for trend types
        self.TREND_NONE = 0
        self.TREND_RW = 1
        self.TREND_SECOND_DIFF = 2
        self.TREND_CONSTANT = 3

        # Process trend specifications and create constant matrices
        self._process_trend_specifications()
        self._create_constant_trend_matrices()
        
        # Solve core DSGE model
        model_solver.solve_model(self.theta)
        
        # Get core model components
        self.model_components = model_solver.get_model_components()
        
        # Build the augmented state space
        self._build_augmented_state_space()

    def _create_theta_vector(self):
        """
        Creates the theta vector in the order defined by model_solver.
        """
        theta = np.zeros(len(self.theta_param_names))
        for i, param_name in enumerate(self.theta_param_names):
            if param_name in self.param_dict:
                theta[i] = self.param_dict[param_name]
            else:
                raise ValueError(f"Parameter '{param_name}' not found in param_dict.")
        return theta

    def _process_trend_specifications(self):
        """
        Processes the trend specifications from the model_specs dictionary and
        counts the number of trend states needed.
        """
        for obs_var, spec in self.model_specs.items():
            trend_type_str = spec.get('trend', 'none')
            cycle_variable = spec.get('cycle')

            if cycle_variable is None:
                raise ValueError(f"Cycle variable not defined for observable {obs_var}")

            if trend_type_str == "random_walk":
                trend_type = self.TREND_RW
                self.n_trend_states += 1
                self.n_trend_shocks += 1
            elif trend_type_str == "second_difference":
                trend_type = self.TREND_SECOND_DIFF
                self.n_trend_states += 2  # level and growth states
                self.n_trend_shocks += 2  # level and growth shocks
            elif trend_type_str == "constant_mean":
                trend_type = self.TREND_CONSTANT
                self.n_trend_states += 1
                self.n_trend_shocks += 1
            elif trend_type_str == "none":
                trend_type = self.TREND_NONE
            else:
                raise ValueError(f"Invalid trend type: {trend_type_str} for variable {obs_var}")

            self.trend_types[obs_var] = trend_type
            # The shock name is the observable name + level_shock
            self.trend_shock_names[obs_var] = f"{obs_var}_level_shock" 

    def _create_constant_trend_matrices(self):
        """
        Creates constant parts of A_trend and B_trend matrices based on trend specifications.
        """
        n_trend_states = self.n_trend_states
        n_trend_shocks = self.n_trend_shocks
        
        # Create matrices with correct dimensions
        self.constant_A_trend = np.zeros((n_trend_states, n_trend_states))
        self.constant_B_trend = np.zeros((n_trend_states, n_trend_shocks))

        trend_state_index = 0
        trend_shock_index = 0

        for obs_var in self.trend_variables:
            trend_type = self.trend_types[obs_var]

            if trend_type == self.TREND_RW:
                # Random walk: level_t = level_{t-1} + shock
                self.constant_A_trend[trend_state_index, trend_state_index] = 1.0
                self.constant_B_trend[trend_state_index, trend_shock_index] = 1.0
                trend_state_index += 1
                trend_shock_index += 1
                
            elif trend_type == self.TREND_SECOND_DIFF:
                # Second difference: 
                # level_t = level_{t-1} + growth_{t-1} + shock_level
                # growth_t = growth_{t-1} + shock_growth
                
                # Level transition
                self.constant_A_trend[trend_state_index, trend_state_index] = 1.0  # level to level
                self.constant_A_trend[trend_state_index, trend_state_index + 1] = 1.0  # growth to level
                self.constant_B_trend[trend_state_index, trend_shock_index] = 1.0  # shock to level
                
                # Growth transition
                self.constant_A_trend[trend_state_index + 1, trend_state_index + 1] = 1.0  # growth to growth
                self.constant_B_trend[trend_state_index + 1, trend_shock_index + 1] = 1.0  # shock to growth
                
                trend_state_index += 2
                trend_shock_index += 2
                
            elif trend_type == self.TREND_CONSTANT:
                # Constant mean: mean_t = mean_{t-1} + small_shock
                self.constant_A_trend[trend_state_index, trend_state_index] = 1.0
                self.constant_B_trend[trend_state_index, trend_shock_index] = 0.01  # Very small effect
                trend_state_index += 1
                trend_shock_index += 1

    def _build_augmented_state_space(self):
        """
        Builds the augmented state-space matrices (A_aug, B_aug, C_aug, Q_aug).
        """
        # 1. Get A and B matrices from the DSGE solution
        self.A = self.model_components['state_transition']  # P matrix from Klein
        self.B = self.model_components['impulse_matrix']    # DSGE impulse matrix
        
        # Print diagnostics for debugging
        print(f"DSGE A matrix shape: {self.A.shape}")
        print(f"DSGE B matrix shape: {self.B.shape}")
        print(f"Trend A matrix shape: {self.constant_A_trend.shape}")
        print(f"Trend B matrix shape: {self.constant_B_trend.shape}")

        # 2. Create augmented A, B, C matrices
        self.A_aug = self._create_A_aug()
        self.B_aug = self._create_B_aug()
        self.C_aug = self._create_C_aug()

        # 3. Create the selection matrix H (identity if all variables observed)
        self.H = np.eye(self.n_observables)  # Simple case: all specified variables observed

        # 4. Create the augmented Q matrix (shock covariances)
        self.Q_aug = self._create_augmented_Q()

    def _create_A_aug(self):
        """
        Creates the augmented state transition matrix A_aug.
        
        Following the specification:
        A_aug = [A  0]
                [0  A_trend]
                
        Returns:
            A_aug: Augmented state transition matrix
        """
        n_states = self.n_states
        n_trend_states = self.n_trend_states
        
        # Create the block diagonal matrix
        A_aug = np.zeros((n_states + n_trend_states, n_states + n_trend_states))
        
        # Add DSGE state transition to upper-left block
        A_aug[:n_states, :n_states] = self.A
        
        # Add trend transition to lower-right block
        A_aug[n_states:, n_states:] = self.constant_A_trend
        
        return A_aug

    def _create_B_aug(self):
        """
        Creates the augmented shock selection matrix B_aug.
        """
        n_states = self.n_states
        n_trend_states = self.n_trend_states
        n_shocks = self.n_shocks
        n_trend_shocks = self.n_trend_shocks
        
        # Create the block diagonal matrix
        B_aug = np.zeros((n_states + n_trend_states, n_shocks + n_trend_shocks))
        
        # Add DSGE shock selection to upper-left block
        # The model_solver should have already set up the correct shock ordering
        B_aug[:n_states, :n_shocks] = self.B
        
        # Add trend shock selection to lower-right block
        B_aug[n_states:, n_shocks:] = self.constant_B_trend
        
        return B_aug
        
    def _create_C_aug(self):
        """
        Creates the augmented observation matrix C_aug using correct variable ordering.
        """
        n_observable = self.n_observables
        n_states = self.n_states
        n_trend_states = self.n_trend_states
        
        # Get the policy function matrix F from Klein solution
        F = self.model_components['f']
        
        # Get actual variable ordering from model JSON
        state_vars = self.model_solver.model.get('state_variables', [])
        control_vars = self.model_solver.model.get('control_variables', [])
        
        # Create augmented observation matrix
        C_aug = np.zeros((n_observable, n_states + n_trend_states))
        
        # Track current trend state index
        trend_state_index = n_states
        
        # Process each observable variable
        for i, obs_var in enumerate(self.trend_variables):
            cycle_variable = self.model_specs[obs_var]['cycle']
            
            if cycle_variable in state_vars:
                state_idx = state_vars.index(cycle_variable)
                C_aug[i, state_idx] = 1.0
            elif cycle_variable in control_vars:
                control_idx = control_vars.index(cycle_variable)
                C_aug[i, :n_states] = F[control_idx, :]
            else:
                raise ValueError(f"Cycle variable {cycle_variable} not found in model variables")
            
            # Add trend component
            trend_type = self.trend_types[obs_var]
            if trend_type == self.TREND_RW or trend_type == self.TREND_CONSTANT:
                C_aug[i, trend_state_index] = 1.0
                trend_state_index += 1
            elif trend_type == self.TREND_SECOND_DIFF:
                C_aug[i, trend_state_index] = 1.0
                trend_state_index += 2
        
        return C_aug

    def _create_augmented_Q(self):
        """
        Creates the augmented process noise covariance matrix Q_aug.
        
        Returns:
            Q_aug: Augmented shock covariance matrix
        """
        n_shocks = self.n_shocks
        n_trend_shocks = self.n_trend_shocks
        n_aug_shocks = n_shocks + n_trend_shocks

        Q_aug = np.zeros((n_aug_shocks, n_aug_shocks))

        # DSGE shock covariances
        if 'QQ' in self.model_components:
            Q_aug[:n_shocks, :n_shocks] = self.model_components['QQ']
        else:
            # Extract shock variances from theta
            dsge_shock_stds_start = self.model_solver.n_dsge_core_params
            dsge_shock_stds_end = dsge_shock_stds_start + self.model_solver.n_dsge_shock_std_params
            dsge_shock_stds = self.theta[dsge_shock_stds_start:dsge_shock_stds_end]
            Q_aug[:n_shocks, :n_shocks] = np.diag(dsge_shock_stds**2)

        # Trend shock variances - match them in order with the B_trend matrix
        trend_shock_index = 0
        for obs_var in self.trend_variables:
            trend_type = self.trend_types[obs_var]
            
            # Level shock
            level_param_name = f"{obs_var}_level_shock_std"
            if level_param_name in self.theta_param_names:
                param_index = self.theta_param_names.index(level_param_name)
                Q_aug[n_shocks + trend_shock_index, n_shocks + trend_shock_index] = self.theta[param_index]**2
            else:
                #print(f"Warning: Trend shock parameter '{level_param_name}' not found. Using default.")
                sys.stderr.write(f"Warning: Trend shock parameter '{level_param_name}' not found.\n")
                sys.exit(1)

            trend_shock_index += 1
            
            # Growth shock for second difference trends
            if trend_type == self.TREND_SECOND_DIFF:
                growth_param_name = f"{obs_var}_slope_shock_std"
                if growth_param_name in self.theta_param_names:
                    param_index = self.theta_param_names.index(growth_param_name)
                    Q_aug[n_shocks + trend_shock_index, n_shocks + trend_shock_index] = self.theta[param_index]**2
                else:
                    Q_aug[n_shocks + trend_shock_index, n_shocks + trend_shock_index] = 0.01
                    print(f"Warning: Trend shock parameter '{growth_param_name}' not found. Using default.")
                
                trend_shock_index += 1

        return Q_aug

    def get_state_space(self):
        """
        Returns the complete state-space representation.
        
        Returns:
            dict: State-space matrices and information
        """
        return {
            'A': self.A_aug,           # Augmented state transition
            'B': self.B_aug,           # Augmented shock selection
            'C': self.C_aug,           # Augmented observation matrix
            'H': self.H,               # Selection matrix for observables
            'Q': self.Q_aug,           # Augmented shock covariance
            'n_states': self.n_states,
            'n_trend_states': self.n_trend_states,
            'n_total_states': self.n_states + self.n_trend_states,
            'n_shocks': self.n_shocks,
            'n_trend_shocks': self.n_trend_shocks,
            'n_observables': self.n_observables
        }

    def compute_irfs(self, shock_name, periods=40, shock_size=1.0):
        """
        Computes the impulse response functions (IRFs) for a given shock.

        Args:
            shock_name: The name of the shock to compute the IRF for
            periods: The number of periods for the IRF
            shock_size: Size of the shock in standard deviations

        Returns:
            tuple: (irfs, state_irfs) with impulse responses for observables and states
        """
        # Get shock index - FIXED THIS PART FOR MORE ACCURATE IRFs
        shock_index = None
        
        # Check if it's a DSGE core shock
        if shock_name in self.model_components['shock_names']:
            shock_index = self.model_components['shock_names'].index(shock_name)
            shock_is_dsge = True
            print(f"Found DSGE shock '{shock_name}' at index {shock_index}")
        else:
            # Check if it's a trend shock
            for i, obs_var in enumerate(self.trend_variables):
                trend_shock = f"{obs_var}_level_shock"
                if trend_shock == shock_name:
                    shock_index = self.n_shocks + i
                    shock_is_dsge = False
                    print(f"Found trend shock '{shock_name}' at index {shock_index}")
                    break
                
                # Check for growth shock in second difference trends
                if self.trend_types[obs_var] == self.TREND_SECOND_DIFF:
                    growth_shock = f"{obs_var}_growth_shock"
                    if growth_shock == shock_name:
                        # Need to calculate correct index based on previous shocks
                        shock_is_dsge = False
                        trend_shock_counter = 0
                        for j, other_var in enumerate(self.trend_variables):
                            if j < i:
                                trend_shock_counter += 1
                                if self.trend_types[other_var] == self.TREND_SECOND_DIFF:
                                    trend_shock_counter += 1
                            else:
                                break
                        shock_index = self.n_shocks + trend_shock_counter  # Skip level shock
                        print(f"Found trend growth shock '{shock_name}' at index {shock_index}")
                        break
        
        if shock_index is None:
            raise ValueError(f"Shock '{shock_name}' not found in model")

        # Initialize the state vector
        x = np.zeros(self.A_aug.shape[0])

        # Apply the shock - correct way with shock_size
        epsilon = np.zeros(self.B_aug.shape[1])
        epsilon[shock_index] = shock_size
        
        # First state is B*ε
        x = self.B_aug @ epsilon
        
        # Store all observables and states
        n_total_states = self.n_states + self.n_trend_states
        obs_irfs = np.zeros((periods, self.n_observables))
        state_irfs = np.zeros((periods, n_total_states))
        
        # Record the initial response
        y = self.C_aug @ x
        obs_irfs[0, :] = y
        state_irfs[0, :] = x
        
        # Simulate the model forward
        for t in range(1, periods):
            # Update the state vector without additional shocks
            x = self.A_aug @ x
            
            # Store states
            state_irfs[t, :] = x
            
            # Compute observed variables
            y = self.C_aug @ x
            obs_irfs[t, :] = y

        # Convert to dictionary for observables
        irfs = {}
        for i, var in enumerate(self.trend_variables):
            irfs[var] = obs_irfs[:, i]

        return irfs, state_irfs

    def kalman_filter(self, data):
        """
        Applies Kalman filter/smoother to the data using the augmented state space.
        
        Args:
            data: Numpy array with shape (n_timesteps, n_observables)
                or (1, n_timesteps, n_observables)
                
        Returns:
            dict: Filtered/smoothed states and related statistics
        """
        import simdkalman
        
        # Reshape data for simdkalman if needed
        if data.ndim == 2:
            # Add batch dimension (simdkalman expects [batch, time, obs])
            data = data.reshape(1, data.shape[0], data.shape[1])
        
        # Process noise covariance
        Q_state = self.B_aug @ self.Q_aug @ self.B_aug.T
        
        # Create Kalman filter with our state space
        kf = simdkalman.KalmanFilter(
            state_transition=self.A_aug,
            process_noise=Q_state,
            observation_model=self.C_aug,
            observation_noise=np.zeros((self.n_observables, self.n_observables))  # Assuming no measurement error
        )
        
        # Run smoother
        results = kf.smooth(data)
        
        # Process results
        smoothed_states = results.smoothed_means[0]  # Remove batch dimension
        smoothed_covs = results.smoothed_covariances[0]
        
        # Separate DSGE and trend components
        dsge_states = smoothed_states[:, :self.n_states]
        trend_states = smoothed_states[:, self.n_states:]
        
        return {
            'smoothed_states': smoothed_states,
            'dsge_states': dsge_states,
            'trend_states': trend_states,
            'smoothed_covs': smoothed_covs,
            'raw_results': results
        }

def klein(a=None, b=None, n_states=None, eigenvalue_warnings=True):
    """
    Solves linear dynamic models using Klein's method.
    
    The model has the form:
        a*E_t[x(t+1)] = b*x(t)
    
    where x(t) = [s(t); u(t)] combines predetermined states s(t)
    and non-predetermined controls u(t).
    
    The solution takes the form:
        u(t)   = f*s(t)       (policy function)
        s(t+1) = p*s(t)       (state transition)
    
    Args:
        a: Coefficient matrix on future-dated variables
        b: Coefficient matrix on current-dated variables
        n_states: Number of predetermined state variables
        eigenvalue_warnings: Whether to print warnings about eigenvalues
        
    Returns:
        f: Policy function matrix
        p: State transition matrix
        stab: Stability indicator
        eig: Generalized eigenvalues
    """
    # Use scipy's linalg for QZ decomposition
    from scipy import linalg
    
    s, t, alpha, beta, q, z = linalg.ordqz(A=a, B=b, sort='ouc', output='complex')

    # Components of the z matrix
    z11 = z[0:n_states, 0:n_states]
    z21 = z[n_states:, 0:n_states]
    
    # number of nonpredetermined variables
    n_costates = np.shape(a)[0] - n_states
    
    if n_states > 0:
        if np.linalg.matrix_rank(z11) < n_states:
            sys.exit("Invertibility condition violated. Check model equations or parameter values.")

    s11 = s[0:n_states, 0:n_states]
    if n_states > 0:
        z11i = linalg.inv(z11)
    else:
        z11i = z11

    # Components of the s, t, and q matrices   
    t11 = t[0:n_states, 0:n_states]
    
    # Verify that there are exactly n_states stable eigenvalues:
    stab = 0

    # Compute the generalized eigenvalues
    tii = np.diag(t)
    sii = np.diag(s)
    eig = np.zeros(np.shape(tii), dtype=np.complex128)

    for k in range(len(tii)):
        if np.abs(sii[k]) > 0:
            eig[k] = tii[k]/sii[k]    
        else:
            eig[k] = np.inf

    # Solution matrix coefficients on the endogenous state
    if n_states > 0:
        dyn = np.linalg.solve(s11, t11)
    else:
        dyn = np.array([])

    f = z21.dot(z11i)
    p = z11.dot(dyn).dot(z11i)

    f = np.real(f)
    p = np.real(p)

    return f, p, stab, eig

def parse_and_generate_files(dynare_file, output_dir, obs_vars=None, model_specs=None):
    """Run the parser and generate all required files, including trend
    structures.

    Args:
        dynare_file (str): Path to the Dynare model file.
        output_dir (str): Path to the output directory.
        obs_vars (list, optional): List of observed variables. Defaults to None.
        model_specs (dict, optional): Model specifications dictionary. Defaults to None.
    """
    # Ensure output directory exists
    if not os.path.isfile(dynare_file):
        raise FileNotFoundError(f"The Dynare file '{dynare_file}' does not exist.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse the Dynare file
    print(f"Parsing Dynare file: {dynare_file}")            
    parser = DynareParser(dynare_file)
    
    # Set observed variables and model specifications if provided
    if obs_vars is not None:
        parser.obs_vars = obs_vars
    if model_specs is not None:
        parser.model_specs = model_specs
    
    # Parse and save the model JSON
    model_json = parser.save_json(os.path.join(output_dir, "model.json"))
    
    # Generate the Jacobian evaluator
    parser.generate_jacobian_evaluator(os.path.join(output_dir, "jacobian_evaluator.py"))
    
    # Generate model structure with trend components
    structure = parser.generate_model_structure()
    
    # Write model_structure.py with all components
    with open(os.path.join(output_dir, "model_structure.py"), 'w') as f:
        f.write("import numpy as np\n\n")
        f.write(f"indices = {repr(structure['indices'])}\n\n")
        f.write(f"R = np.array({repr(structure['R'].tolist())})\n\n")
        f.write(f"B_structure = np.array({repr(structure['B_structure'].tolist())})\n\n")
        f.write(f"C_structure = np.array({repr(structure['C_structure'].tolist())})\n\n")
        f.write(f"D = np.array({repr(structure['D'].tolist())})\n\n")
        f.write(f"labels = {repr(structure['labels'])}\n\n")
        
        # Write trend structures if they exist
        if 'T_trend_structure' in structure:
            f.write(f"T_trend_structure = np.array({repr(structure['T_trend_structure'])})\n\n")
        if 'R_trend_structure' in structure:
            f.write(f"R_trend_structure = np.array({repr(structure['R_trend_structure'])})\n\n")
        if 'C_trend_structure' in structure:
            f.write(f"C_trend_structure = np.array({repr(structure['C_trend_structure'])})\n\n")
        if 'n_trend_states' in structure:
            f.write(f"n_trend_states = {structure['n_trend_states']}\n\n")
        if 'obs_mapping' in structure:
            f.write(f"obs_mapping = {repr(structure['obs_mapping'])}\n\n")
    
    print(f"All model files generated in {output_dir}")

# import os
# import sys
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from scipy import linalg

# Import our classes
# from model_solver import ModelSolver
# from augmented_state_space import AugmentedStateSpace
# from dynare_parser import DynareParser, parse_and_generate_files

# Example usage - Main script
if __name__ == "__main__":
    # --- Assume these steps were run previously ---
    # 1. DynareParser generated model.json, jacobian_evaluator.py, model_structure.py
    # ----------------------------------------------
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    output_dir = "model_files"  # Directory containing generated files
    dynare_file = "qpm_simpl1.dyn"  # Just needed for reference if re-parsing

    # Define model specifications (as used by parser and solver)
    model_specs = {
        "rs_obs": {"trend": "random_walk", "cycle": "RS"},
        "dla_cpi_obs": {"trend": "random_walk", "cycle": "DLA_CPI"},
        "l_gdp_obs": {"trend": "random_walk", "cycle": "L_GDP_GAP"}
    }
    observed_variables = list(model_specs.keys())

    # Optionally regenerate the necessary files (JSON model, Jacobian, structure)
    # Comment this out if not needed
    parse_and_generate_files(dynare_file, output_dir, 
                            obs_vars=observed_variables, 
                            model_specs=model_specs)

    # Ensure output directory exists (it should if parser ran)
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found. Ensure DynareParser ran successfully.")
        sys.exit(1)

    # --- Main workflow ---
    # 1. Create ModelSolver instance (loads structure, defines theta order)
    try:
        solver = ModelSolver(output_dir, model_specs, observed_variables)
    except Exception as e:
        print(f"Failed to initialize ModelSolver: {e}")
        sys.exit(1)

    # 2. Load and prepare data (using DataProcessor or similar)
    try:
        us_data = pd.read_csv('transformed_data_us.csv', index_col='Date', parse_dates=True)
        # Ensure data has the 'observed_variables' columns
        if not all(v in us_data.columns for v in observed_variables):
            missing_vars = [v for v in observed_variables if v not in us_data.columns]
            raise ValueError(f"Data file missing required columns: {missing_vars}")

        # Use only the necessary columns in the correct order
        data_for_filter = us_data[observed_variables].values
        # Keep dates for results processing
        dates = us_data.index 
    except FileNotFoundError:
        print("Error: Data file 'transformed_data_us.csv' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

    # 3. Define parameter values in dictionary form
    initial_param_dict = {
        # DSGE Core Params
        'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
        'rho_DLA_CPI': 0.75, 'rho_L_GDP_GAP': 0.75, 'rho_rs': 0.8, 'rho_rs2': 0.1,
        # DSGE Shock STDs 
        'RES_L_GDP_GAP_lag_std': 1.0,
        'RES_RS_lag_std': 1.0,
        'RES_DLA_CPI_lag_std': 1.0,
        # Trend Shock STDs
        'rs_obs_level_shock_std': 0.5,
        'dla_cpi_obs_level_shock_std': 0.5,
        'l_gdp_obs_level_shock_std': 0.1
    }

    # 4. Create augmented state space
    augmented_model = AugmentedStateSpace(solver, model_specs, initial_param_dict)
    
    # # 5. Compute IRFs
    # shock_name = 'SHK_RS'  # This should match one in solver.shock_names
    # irf_results, state_irfs = augmented_model.compute_irfs(
    #     shock_name=shock_name,
    #     periods=40
    # )
    
    # # Plot IRFs
    # plt.figure(figsize=(15, 10))
    # for i, (var, values) in enumerate(irf_results.items()):
    #     plt.subplot(2, 2, i+1)
    #     plt.plot(values)
    #     plt.title(f"Response of {var} to {shock_name} shock")
    #     plt.grid(True)
    #     plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"IRF_{shock_name}.png")
    # plt.close()
    # plt.show()
    # # 6. Run Kalman filter on the data
#     filter_results = augmented_model.kalman_filter(data_for_filter)
    
#     # Create a DataFrame with filtered/smoothed states
#     smoothed_states_df = pd.DataFrame(
#         filter_results['smoothed_states'], 
#         index=dates,
#         columns=[f"State_{i}" for i in range(filter_results['smoothed_states'].shape[1])]
#     )
    
#     # Extract trend states for reporting
#     trend_states_df = pd.DataFrame(
#         filter_results['trend_states'],
#         index=dates,
#         columns=[f"{var}_trend" for var in observed_variables]
#     )
    
#     # Plot trend components
#     plt.figure(figsize=(15, 10))
#     for i, var in enumerate(observed_variables):
#         plt.subplot(len(observed_variables), 1, i+1)
        
#         # Plot observed data
#         plt.plot(dates, us_data[var], 'b-', label=f"Observed {var}")
        
#         # Plot trend component
#         plt.plot(dates, trend_states_df[f"{var}_trend"], 'r-', label=f"Trend {var}")
        
#         plt.title(f"Decomposition of {var}")
#         plt.legend()
#         plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig("trend_decomposition.png")
#     plt.close()
    
#     print("Analysis complete. Check the output plots.")




# # def klein(a=None, b=None, n_states=None, eigenvalue_warnings=True):
# #     """
# #     Solves linear dynamic models with the form of:
    
# #     a*Et[x(t+1)] = b*x(t)       
            
#     [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
#     a vector of nonpredetermined costate variables.
    
#     The solution to the model is a set of matrices f, p such that:
    
#     u(t)   = f*s(t)
#     s(t+1) = p*s(t)
    
#     The solution algorithm is based on Klein (2000) and his solab.m Matlab program.
    
#     Args:
#         a: Coefficient matrix on future-dated variables
#         b: Coefficient matrix on current-dated variables
#         n_states: Number of state variables
#         eigenvalue_warnings: Whether to print warnings about eigenvalues
        
#     Returns:
#         f: Solution matrix coefficients on s(t) for u(t)
#         p: Solution matrix coefficients on s(t) for s(t+1)
#         stab: Stability indicator
#         eig: Generalized eigenvalues
#     """
#     s, t, alpha, beta, q, z = la.ordqz(A=a, B=b, sort='ouc', output='complex')

#     # Components of the z matrix
#     z11 = z[0:n_states, 0:n_states]
#     z21 = z[n_states:, 0:n_states]
    
#     # number of nonpredetermined variables
#     n_costates = np.shape(a)[0] - n_states
    
#     if n_states > 0:
#         if np.linalg.matrix_rank(z11) < n_states:
#             sys.exit("Invertibility condition violated. Check model equations or parameter values.")

#     s11 = s[0:n_states, 0:n_states]
#     if n_states > 0:
#         z11i = la.inv(z11)
#     else:
#         z11i = z11

#     # Components of the s, t, and q matrices   
#     t11 = t[0:n_states, 0:n_states]
    
#     # Verify that there are exactly n_states stable eigenvalues:
#     stab = 0

#     # Compute the generalized eigenvalues
#     tii = np.diag(t)
#     sii = np.diag(s)
#     eig = np.zeros(np.shape(tii), dtype=np.complex128)

#     for k in range(len(tii)):
#         if np.abs(sii[k]) > 0:
#             eig[k] = tii[k]/sii[k]    
#         else:
#             eig[k] = np.inf

#     # Solution matrix coefficients on the endogenous state
#     if n_states > 0:
#         dyn = np.linalg.solve(s11, t11)
#     else:
#         dyn = np.array([])

#     f = z21.dot(z11i)
#     p = z11.dot(dyn).dot(z11i)

#     f = np.real(f)
#     p = np.real(p)

#     return f, p, stab, eig



# # Example usage - Main script
# if __name__ == "__main__":

#     # --- Assume these steps were run previously ---
#     # 1. DynareParser generated model.json, jacobian_evaluator.py, model_structure.py
#     # ----------------------------------------------
#     script_path = os.path.abspath(__file__)
#     script_dir = os.path.dirname(script_path)
#     os.chdir(script_dir)
#     print(f"Current working directory: {os.getcwd()}")

#     output_dir = "model_files" # Directory containing generated files
#     dynare_file = "qpm_simpl1.dyn" # Just needed for reference if re-parsing
   

#     # Define model specifications (as used by parser and solver)
#     model_specs = {
#         "rs_obs": {"trend": "random_walk", "cycle": "RS"},
#         "dla_cpi_obs": {"trend": "random_walk", "cycle": "DLA_CPI"},
#         "l_gdp_obs": {"trend": "random_walk", "cycle": "L_GDP_GAP"}
#     }
#     observed_variables = list(model_specs.keys())


#     # 1. Generate the necessary files (JSON model, Jacobian, structure)
#     parse_and_generate_files(dynare_file, output_dir, 
#                             obs_vars=observed_variables, 
#                             model_specs=model_specs)


#     # Ensure output directory exists (it should if parser ran)
#     if not os.path.exists(output_dir):
#         print(f"Error: Output directory '{output_dir}' not found. Ensure DynareParser ran successfully.")
#         sys.exit(1)

#     # --- Main workflow ---
#     # 1. Create ModelSolver instance (loads structure, defines theta order)
#     #try:
#     solver = ModelSolver(output_dir, model_specs, observed_variables)
#     # except Exception as e:
#     #     print(f"Failed to initialize ModelSolver: {e}")
#     #     sys.exit(1)

#     # 2. Load and prepare data (using DataProcessor or similar)
#     try:
#         us_data = pd.read_csv('transformed_data_us.csv', index_col='Date', parse_dates=True)
#         # Ensure data has the 'observed_variables' columns
#         if not all(v in us_data.columns for v in observed_variables):
#             missing_vars = [v for v in observed_variables if v not in us_data.columns]
#             raise ValueError(f"Data file missing required columns: {missing_vars}")

#         # Use only the necessary columns in the correct order
#         data_for_filter = us_data[observed_variables].values
#         # Reshape for simdkalman: (1, n_timesteps, n_observables)
#         data_array = data_for_filter.reshape(1, data_for_filter.shape[0], data_for_filter.shape[1])
#         data_array = data_array.astype(float) # Ensure float type
#         dates = us_data.index # Keep dates for results processing

#     except FileNotFoundError:
#         print("Error: Data file 'transformed_data_us.csv' not found.")
#         sys.exit(1)
#     except ValueError as e:
#         print(f"Error preparing data: {e}")
#         sys.exit(1)


#     # 3. Define parameter values `theta` IN THE CORRECT ORDER
#     #    Get the order from solver.theta_param_names
#     #    Example values (replace with your actual draw/initial values)
#     initial_param_dict = {
#         # DSGE Core Params (Order from labels['param_labels'])
#         'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
#         'rho_DLA_CPI': 0.75, 'rho_L_GDP_GAP': 0.75, 'rho_rs': 0.8, 'rho_rs2': 0.1,
#         # DSGE Shock STDs (Order from labels['shock_labels']) - Assuming shocks are RES_...
#         'RES_L_GDP_GAP_lag_std': 1.0, # Parameter name should end in _std
#         'RES_RS_lag_std': 1.0,
#         'RES_DLA_CPI_lag_std': 1.0,
#         # Trend Shock STDs (Order from trend_info calculation)
#         'rs_obs_level_shock_std': 0.5,
#         'dla_cpi_obs_level_shock_std': 0.5,
#         'l_gdp_obs_level_shock_std': 0.1 # Adjust name based on trend_info output
#         # Measurement Error STDs (Order from obs_vars) - OMITTED FOR NOW
#         # 'rs_obs_meas_error_std': 0.1,
#         # 'dla_cpi_obs_meas_error_std': 0.1,
#         # 'l_gdp_obs_meas_error_std': 0.1
#     }
#     smoothed_results = solver.solve_and_filter_calibrated_model(initial_param_dict, data_array)
    
    
#     fig, irf_data = solver.compute_irf(
#                     param_dict=initial_param_dict,
#                     shock_name='SHK_RS',  # Name must match one in labels['shock_labels']
#                     shock_size=1.0,       # Size in standard deviations
#                     periods=40,           # Number of periods to simulate
#                     vars_to_plot=['RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP']  # Variables to plot
#     ) 
#     # # Construct theta array using the defined order
#     # try:
#     #     theta_values = [initial_param_dict[pname] for pname in solver.theta_param_names]
#     #     theta = np.array(theta_values)
#     #     print(f"Constructed theta vector with {len(theta)} values.")
#     # except KeyError as e:
#     #     print(f"Error: Parameter '{e}' not found in initial_param_dict. Check parameter names.")
#     #     # Find missing/mismatched names:
#     #     provided_keys = set(initial_param_dict.keys())
#     #     expected_keys = set(solver.theta_param_names)
#     #     print(f"Missing from dict: {expected_keys - provided_keys}")
#     #     print(f"Extra in dict: {provided_keys - expected_keys}")
#     #     sys.exit(1)


#     # # 4. Update state space and run smoother
#     # try:
#     #     # Option 1: Call update_state_space then create filter manually
#     #     # T, Q_state, Z, H = solver.update_state_space(theta)
#     #     # kf = simdkalman.KalmanFilter(state_transition=T, process_noise=Q_state, observation_model=Z, observation_noise=H)
#     #     # simdkalman_results = kf.smooth(data_array)

#     #     # Option 2: Use helper method if defined
#     #     simdkalman_results = solver.run_filter_smoother(theta, data_array)

#     #     print("Smoother run successfully.")

#     # except Exception as e:
#     #     print(f"Error running smoother: {e}")
#     #     # Add more diagnostics here if needed
#     #     sys.exit(1)

#     # # 5. Process results (using DataProcessor or manually)
#     # # Get state labels (including augmented trend states)
#     # augmented_state_labels = solver.labels['state_labels'] + solver.trend_info['state_labels']

#     # # Extract smoothed means
#     # smoothed_means = simdkalman_results.smoothed_means[0] # Get first batch

#     # # Create DataFrame
#     # results_df = pd.DataFrame(smoothed_means, index=dates, columns=augmented_state_labels)

#     # print("\nSmoothed States (Head):")
#     # print(results_df.head())

#     # # Extract and print specific components as before
#     # print("\nSmoothed trend components:")
#     # trend_columns = [col for col in results_df.columns if any(x in col for x in ['level', 'slope', 'curvature', 'mean'])]
#     # print(results_df[trend_columns].head())

#     # print("\nSmoothed cyclical components (DSGE states):")
#     # # Assuming cycle vars match DSGE state names (adjust if needed)
#     # cycle_columns = [spec['cycle'] for spec in model_specs.values() if spec['cycle'] in results_df.columns]
#     # print(results_df[cycle_columns].head())
