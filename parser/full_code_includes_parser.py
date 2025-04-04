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
def calculate_trend_positions(model_specs, obs_vars, n_states, n_shocks):
    """
    Calculate positions and parameter names for trend components.

    Returns:
        dict: Information about trend component positions and parameters
    """
    trend_info = {
        'state_labels': [],
        'components': {},
        'total_states': 0,
        'param_positions': {},
        'shock_std_param_names': [],
        'total_trend_shocks': 0  # New: Total number of trend shocks
    }

    pos = n_states
    shock_pos = n_shocks
    total_trend_shocks = 0 # Initialize counter

    for obs_var in obs_vars:
        trend_type = model_specs[obs_var]['trend']
        var_info = {'type': trend_type, 'components': {}}

        if trend_type == 'random_walk':
            var_info['components']['level'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            param_name = f"{obs_var}_level_shock_std"  # Add _std suffix
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name)
            total_trend_shocks += 1  # Increment counter

            trend_info['state_labels'].append(f"{obs_var}_level")
            pos += 1
            shock_pos += 1

        elif trend_type == 'second_difference':
            # ... (Similar modifications for other trend types)
            var_info['components']['level'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            param_name = f"{obs_var}_level_shock_std"  # Add _std suffix
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name)
            total_trend_shocks += 1 # Increment counter

            trend_info['state_labels'].append(f"{obs_var}_level")
            pos += 1
            shock_pos += 1

            var_info['components']['slope'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            param_name = f"{obs_var}_slope_shock_std"  # Add _std suffix
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name)
            total_trend_shocks += 1 # Increment counter
            trend_info['state_labels'].append(f"{obs_var}_slope")
            pos += 1
            shock_pos += 1

            var_info['components']['curvature'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            param_name = f"{obs_var}_curvature_shock_std"  # Add _std suffix
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name)
            total_trend_shocks += 1 # Increment counter
            trend_info['state_labels'].append(f"{obs_var}_curvature")
            pos += 1
            shock_pos += 1

        elif trend_type == 'constant_mean':
            # ... (Similar modifications for other trend types)
            var_info['components']['level'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            param_name = f"{obs_var}_mean_shock_std"  # Add _std suffix
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name)
            total_trend_shocks += 1  # Increment counter

            trend_info['state_labels'].append(f"{obs_var}_mean")
            pos += 1
            shock_pos += 1

        # Store information for this variable
        trend_info['components'][obs_var] = var_info

    trend_info['total_states'] = pos - n_states
    trend_info['total_trend_shocks'] = total_trend_shocks # Store the total count

    return trend_info

def build_trend_transition(T_matrix, trend_info, n_states):
    """
    Build the transition matrix blocks for trend components.
    
    Args:
        T_matrix: Transition matrix to fill
        trend_info: Trend component position information
        n_states: Number of original state variables
    """
    # Process each observed variable
    for obs_var, var_info in trend_info['components'].items():
        trend_type = var_info['type']
        
        # Set transition dynamics based on trend type
        if trend_type == 'random_walk':
            # Level follows random walk: level_t = level_{t-1} + shock
            level_pos = var_info['components']['level']['state_pos']
            T_matrix[level_pos, level_pos] = 1.0
            
        elif trend_type == 'second_difference':
            # Level follows: level_t = level_{t-1} + slope_{t-1} + shock
            # Slope follows: slope_t = slope_{t-1} + curvature_{t-1} + shock
            # Curvature follows: curvature_t = curvature_{t-1} + shock
            level_pos = var_info['components']['level']['state_pos']
            slope_pos = var_info['components']['slope']['state_pos']
            curv_pos = var_info['components']['curvature']['state_pos']
            
            # Level depends on itself and slope
            T_matrix[level_pos, level_pos] = 1.0
            T_matrix[level_pos, slope_pos] = 1.0
            
            # Slope depends on itself and curvature
            T_matrix[slope_pos, slope_pos] = 1.0
            T_matrix[slope_pos, curv_pos] = 1.0
            
            # Curvature depends only on itself
            T_matrix[curv_pos, curv_pos] = 1.0
            
        elif trend_type == 'constant_mean':
            # Constant mean doesn't change: mean_t = mean_{t-1} + small_shock
            mean_pos = var_info['components']['level']['state_pos']
            T_matrix[mean_pos, mean_pos] = 1.0

def build_trend_selection(R_matrix, trend_info, n_states, n_shocks):
    """
    Build the selection matrix blocks for trend components.
    
    Args:
        R_matrix: Selection matrix to fill
        trend_info: Trend component position information
        n_states: Number of original state variables
        n_shocks: Number of original shock variables
    """
    # For each variable and its trend components
    for obs_var, var_info in trend_info['components'].items():
        for comp_name, comp_info in var_info['components'].items():
            state_pos = comp_info['state_pos']
            shock_pos = comp_info['shock_pos']
            
            # Direct mapping from shock to state
            R_matrix[state_pos, shock_pos] = 1.0

def build_trend_observation(Z_matrix, trend_info, obs_vars, observable_labels):
    """
    Build the observation matrix blocks for trend components.
    
    Args:
        Z_matrix: Observation matrix to fill
        trend_info: Trend component position information
        obs_vars: List of observed variable names
        observable_labels: List of all observable labels
    """
    # For each observed variable
    for obs_var in obs_vars:
        # Find base variable (without _obs)
        base_var = obs_var.replace("_obs", "")
        
        # Find the index in the observable list
        try:
            var_idx = observable_labels.index(base_var)
        except ValueError:
            continue  # Skip if not found
        
        # Get trend components for this variable
        var_info = trend_info['components'].get(obs_var, {})
        if not var_info:
            continue
            
        # Map the appropriate trend component to the observation
        if 'level' in var_info['components']:
            level_pos = var_info['components']['level']['state_pos']
            Z_matrix[var_idx, level_pos] = 1.0





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
        that don't depend on parameter values, including parts for
        stochastic trends.

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

        # Correctly define n_observables based on observed variables
        n_observables = len(self.obs_vars) if hasattr(self, 'obs_vars') and self.obs_vars else 0

        # Create indices dictionary with corrected dimensions
        indices = {
            'n_endogenous': n_endogenous,
            'n_exo_states': n_exo_states,
            'n_controls': n_controls,
            'n_shocks': n_shocks,
            'n_states': n_states,
            'n_observables': n_observables
        }
        
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

        # Create C matrix structure with correct dimensions
        C_structure = np.zeros((n_observables, n_states))

        # D matrix (direct shock impact on observables)
        D = np.zeros((n_observables, n_shocks))

        # Create labels for the state space
        labels = {
            'state_labels': self.state_variables,
            'observable_labels': self.control_variables + self.state_variables,
            'shock_labels': self.varexo_list,
            'param_labels': list(self.parameters.keys())
        }

        # Initialize structure dictionary
        structure = {
            'indices': indices,
            'R': R,
            'B_structure': B_structure,
            'C_structure': C_structure,
            'D': D,
            'labels': labels
        }

        # Only process trend and observation mapping if we have observed variables and model specs
        if hasattr(self, 'obs_vars') and self.obs_vars and hasattr(self, 'model_specs'):
            # First pass: Count total trend states
            n_trend_states = 0
            for obs_var in self.obs_vars:
                trend_type = self.model_specs[obs_var]['trend']
                if trend_type == 'random_walk':
                    n_trend_states += 1
                elif trend_type == 'second_difference':
                    n_trend_states += 3
                elif trend_type == 'constant_mean':
                    n_trend_states += 1
            
            # If we have trend states, create the trend matrices
            if n_trend_states > 0:
                # Create matrices with correct dimensions
                T_trend_structure = np.zeros((n_trend_states, n_trend_states))
                R_trend_structure = np.eye(n_trend_states)  # Identity for direct shock mapping
                C_trend_structure = np.zeros((n_observables, n_trend_states))
                
                # Second pass: Fill in structures
                trend_pos = 0
                for i, obs_var in enumerate(self.obs_vars):
                    trend_type = self.model_specs[obs_var]['trend']
                    
                    if trend_type == 'random_walk':
                        # Level maps directly to observation
                        C_trend_structure[i, trend_pos] = 1.0
                        # Level follows random walk
                        T_trend_structure[trend_pos, trend_pos] = 1.0
                        trend_pos += 1
                        
                    elif trend_type == 'second_difference':
                        # Level maps directly to observation
                        C_trend_structure[i, trend_pos] = 1.0
                        # Set transition dynamics
                        T_trend_structure[trend_pos, trend_pos] = 1.0  # Level on itself
                        T_trend_structure[trend_pos, trend_pos+1] = 1.0  # Level on slope
                        T_trend_structure[trend_pos+1, trend_pos+1] = 1.0  # Slope on itself
                        T_trend_structure[trend_pos+1, trend_pos+2] = 1.0  # Slope on curvature
                        T_trend_structure[trend_pos+2, trend_pos+2] = 1.0  # Curvature on itself
                        trend_pos += 3
                        
                    elif trend_type == 'constant_mean':
                        # Mean maps directly to observation
                        C_trend_structure[i, trend_pos] = 1.0
                        # Mean follows AR(1) with coefficient near 1
                        T_trend_structure[trend_pos, trend_pos] = 1.0
                        trend_pos += 1
                
                # Add trend structures to output
                structure['T_trend_structure'] = T_trend_structure.tolist()
                structure['R_trend_structure'] = R_trend_structure.tolist()
                structure['C_trend_structure'] = C_trend_structure.tolist()
                structure['n_trend_states'] = n_trend_states
            
            # Create observation mapping
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
                    # Update C_structure for direct state observations
                    C_structure[i, self.state_variables.index(cycle_var)] = 1.0
                else:
                    print(f"Warning: Cycle variable {cycle_var} for {obs_var} not found in model variables")
            
            # Add observation mapping to structure
            structure['obs_mapping'] = obs_mapping

        return structure
    
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

class ModelSolver:
    """
    Solves the DSGE model and updates the state-space representation based
    on parameter values using a dedicated update function.
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

        # --- Pre-calculate trend structure ---
        # Needs n_states and n_shocks from the core DSGE model structure
        self.trend_info = calculate_trend_positions(
            self.model_specs,
            self.obs_vars,
            self.indices['n_states'],
            self.indices['n_shocks']
        )
        # --- Store the order of parameters expected by theta ---
        self._define_theta_order()

        # Validate model specifications against loaded model/labels
        self.check_spec_vars()

        # Initialize solved state (optional)
        self.f = None # Policy function
        self.p = None # DSGE state transition

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
                continue # Skip if spec incomplete
            if cycle_var not in list_model_names:
                print(f"Error: Cycle variable '{cycle_var}' for '{obs_var}' not found in model variables.")
                all_cycle_vars_found = False

            # Check trend specification existence and type
            if 'trend' not in spec:
                print(f"Error: Trend specification missing for {obs_var}")
                all_cycle_vars_found = False # Treat as error
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
        # 1. DSGE core parameters (must match jacobian_evaluator expectation)
        #    Load this order from self.labels['param_labels'] which came from parser
        dsge_core_params = list(self.labels['param_labels'])

        # 2. DSGE shock standard deviations (shock in exo_variables with shocks)
        dsge_shock_std_params = [f"{state}_std" for state in self.model.get('direct_shock_states', [])]

        # 3. Trend shock standard deviations (use order from trend_info calculation)
        trend_shock_std_params = list(self.trend_info['shock_std_param_names'])

        # 4. Measurement error standard deviations (use order of self.obs_vars)
        #    We assume zero for now, but define the names for future use
        meas_error_std_params = [f"{obs_var}_meas_error_std" for obs_var in self.obs_vars]

        # Combine all parameter names IN ORDER
        self.theta_param_names = dsge_core_params + dsge_shock_std_params + trend_shock_std_params # + meas_error_std_params (Add when needed)

        # Store counts for easy splitting later
        self.n_dsge_core_params = len(dsge_core_params)
        self.n_dsge_shock_std_params = len(dsge_shock_std_params)
        self.n_trend_shock_std_params = len(trend_shock_std_params)
        # self.n_meas_error_std_params = len(meas_error_std_params) # Add when needed

        # Total expected length of theta
        self.expected_theta_length = self.n_dsge_core_params + self.n_dsge_shock_std_params + self.n_trend_shock_std_params # + self.n_meas_error_std_params

        print(f"Theta vector order defined. Expected length: {self.expected_theta_length}")
        # print(f"Theta order: {self.theta_param_names}") # Uncomment for debugging

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
            sys.modules["jacobian_evaluator"] = self.jacobian_module # Add to sys.modules
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
            sys.modules["model_structure"] = struct_module # Add to sys.modules
            spec.loader.exec_module(struct_module)

            # Load required attributes
            self.indices = getattr(struct_module, 'indices')
            self.labels = getattr(struct_module, 'labels')
            self.R = getattr(struct_module, 'R') # Shock-to-state mapping
            self.B_structure = getattr(struct_module, 'B_structure')
            self.C_structure = getattr(struct_module, 'C_structure')
            self.D = getattr(struct_module, 'D')

            # Load observation mapping if available
            if hasattr(struct_module, 'obs_mapping'):
                self.obs_mapping = getattr(struct_module, 'obs_mapping')
            
            # Load trend structures if available
            if hasattr(struct_module, 'T_trend_structure'):
                self.T_trend_structure = getattr(struct_module, 'T_trend_structure')
            if hasattr(struct_module, 'R_trend_structure'):
                self.R_trend_structure = getattr(struct_module, 'R_trend_structure')
            if hasattr(struct_module, 'C_trend_structure'):
                self.C_trend_structure = getattr(struct_module, 'C_trend_structure')
            if hasattr(struct_module, 'n_trend_states'):
                self.n_trend_states = getattr(struct_module, 'n_trend_states')

            # Validate loaded structure
            required_indices = ['n_states', 'n_shocks', 'n_controls', 'n_observables', 'n_endogenous']
            if not all(k in self.indices for k in required_indices):
                raise AttributeError(f"Missing required keys in 'indices' from {struct_path}")
            
            required_labels = ['state_labels', 'observable_labels', 'shock_labels', 'param_labels']
            if not all(k in self.labels for k in required_labels):
                raise AttributeError(f"Missing required keys in 'labels' from {struct_path}")
            
            # Basic check on R dimensions
            n_exo_states = self.indices['n_states'] - self.indices['n_endogenous']
            if self.R.shape != (n_exo_states, self.indices['n_shocks']):
                print(f"Warning: Shape of loaded R {self.R.shape} does not match expected ({n_exo_states}, {self.indices['n_shocks']})")

            # Check C_structure dimensions
            n_observables = self.indices['n_observables']
            n_states = self.indices['n_states']
            if self.C_structure.shape != (n_observables, n_states):
                print(f"Warning: Shape of loaded C_structure {self.C_structure.shape} does not match expected ({n_observables}, {n_states})")

        except FileNotFoundError:
            print(f"Error: model_structure.py not found at {struct_path}")
            raise
        except AttributeError as e:
            print(f"Error: Missing expected attribute in {struct_path}: {e}")
            raise
        except Exception as e:
            print(f"Error loading model structure from {struct_path}: {e}")
            raise

    def update_state_space(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates and returns the numerical state-space matrices (T, Q_state, Z, H)
        based on the provided parameter vector 'theta'.

        Assumes zero measurement error for now (H=0).

        Args:
            theta: NumPy array containing parameter values in the predefined order:
                1. DSGE core parameters
                2. DSGE shock standard deviations
                3. Trend shock standard deviations
                (Measurement error std devs would be next if used)

        Returns:
            T: Augmented state transition matrix (n_aug_states x n_aug_states)
            Q_state: Augmented process noise covariance matrix (n_aug_states x n_aug_states)
            Z: Augmented observation matrix (n_observables x n_aug_states)
            H: Measurement noise covariance matrix (n_observables x n_observables)
        """
        # --- 1. Input Validation and Parameter Splitting ---
        if not isinstance(theta, np.ndarray):
            theta = np.array(theta) # Ensure theta is a numpy array
        if theta.ndim != 1:
            raise ValueError(f"Input 'theta' must be a 1D array, but got shape {theta.shape}")
        if len(theta) != self.expected_theta_length:
            raise ValueError(f"Incorrect length for 'theta'. Expected {self.expected_theta_length}, got {len(theta)}.")

        # Split theta based on pre-calculated counts
        split1 = self.n_dsge_core_params
        split2 = split1 + self.n_dsge_shock_std_params
        split3 = split2 + self.n_trend_shock_std_params

        dsge_core_params = theta[:split1]
        dsge_shock_stds = theta[split1:split2]
        trend_shock_stds = theta[split2:split3]
        # meas_error_stds = theta[split3:] # Uncomment when using measurement errors

        # --- 2. Get Dimensions from loaded model structure ---
        n_states = self.indices['n_states']
        n_shocks = self.indices['n_shocks']
        n_controls = self.indices['n_controls']
        n_observables = self.indices['n_observables']
        n_endogenous = self.indices['n_endogenous']
        n_exo_states = n_states - n_endogenous

        n_trend_states = self.trend_info['total_states']
        n_trend_shocks = self.trend_info['total_trend_shocks']
        n_aug_states = n_states + n_trend_states
        n_aug_shocks = n_shocks + n_trend_shocks

        # --- 3. Solve DSGE Part ---
        try:
            a, b, c_jac = self.evaluate_jacobians(dsge_core_params)
        except Exception as e:
            print(f"Error during Jacobian evaluation: {e}")
            raise

        try:
            # Pass only n_states (predetermined vars) to Klein
            f, p, stab, eig = klein(a, b, n_states)
            self.f = f # Store policy function
            self.p = p # Store DSGE state transition
            # if stab != 0:
            #     print("Warning: Klein solver indicates potential instability.")
        except Exception as e:
            print(f"Error during Klein solution: {e}")
            raise

        # --- 4. Build Augmented State Transition Matrix T ---
        T = np.zeros((n_aug_states, n_aug_states))
        # DSGE state transition
        T[:n_states, :n_states] = self.p
        # Trend component transition
        T[n_states:, n_states:] = np.array(self.T_trend_structure)

        # --- 5. Build Augmented Shock Selection Matrix `selection` ---
        selection = np.zeros((n_aug_states, n_aug_shocks))
        # DSGE shock selection
        selection[n_endogenous:n_states, :n_shocks] = self.R
        # Trend shock selection
        selection[n_states:, n_shocks:] = np.array(self.R_trend_structure)

        # --- 6. Build Process Noise Covariance Q_state ---
        # Combine all shock std devs
        all_shock_stds = np.concatenate((dsge_shock_stds, trend_shock_stds))
        # Square them to get variances
        all_shock_vars = all_shock_stds ** 2
        # Check for negative variances
        # if np.any(all_shock_vars < 0):
        #     print("Warning: Negative variances detected from shock std devs in theta. Taking absolute value.")
        #     all_shock_vars = np.abs(all_shock_vars)
        # Create diagonal matrix of shock variances
        Q_param = np.diag(all_shock_vars)
        # Calculate the state process noise covariance
        Q_state = selection @ Q_param @ selection.T

        # --- 7. Build Augmented Observation Matrix Z ---
        Z = np.zeros((n_observables, n_aug_states))
        
        # Use observation mapping from structure
        for obs_var, mapping in self.obs_mapping.items():
            i = mapping['obs_index']
            if mapping['type'] == 'control':
                # Apply policy function for control variables
                control_idx = mapping['index']
                Z[i, :n_states] = self.f[control_idx, :]
            elif mapping['type'] == 'state':
                # Direct mapping for state variables
                state_idx = mapping['index']
                Z[i, state_idx] = 1.0
        
        # Add trend contributions
        Z[:, n_states:] = np.array(self.C_trend_structure)

        # --- 8. Build Measurement Noise Covariance H ---
        # Assuming zero measurement error for now
        H = np.zeros((n_observables, n_observables))

        # --- 9. Return the constructed matrices ---
        return T, Q_state, Z, H

    def solve_and_filter_calibrated_model(self, param_dict, data_array):
        """
        Solves the model using calibrated parameters and runs the Kalman filter.
        
        Args:
            param_dict: Dictionary of parameter names and values
            data_array: Observed data (shape: 1, n_timesteps, n_observables)
            
        Returns:
            filter_results: Results from Kalman filter/smoother
            model_matrices: Dictionary with T, Q_state, Z, H matrices
            
        Raises:
            ValueError: If parameters are missing or unexpected
        """
        # Validate parameters
        missing_params = set(self.theta_param_names) - set(param_dict.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Check for unexpected parameters
        unexpected_params = set(param_dict.keys()) - set(self.theta_param_names)
        if unexpected_params:
            raise ValueError(f"Unexpected parameters not in model: {', '.join(unexpected_params)}")
        
        # Create ordered theta array
        theta = np.array([param_dict[param] for param in self.theta_param_names])
        
        # Solve the model once
        T, Q_state, Z, H = self.update_state_space(theta)
        
        # Create Kalman filter with solved model
        kf = simdkalman.KalmanFilter(
            state_transition=T, 
            process_noise=Q_state,
            observation_model=Z,
            observation_noise=H
        )
        
        # Run smoother
        smoothed_results = kf.smooth(data_array)
        
        return {
            'filter_results': smoothed_results,
            'model_matrices': {
                'T': T,
                'Q_state': Q_state,
                'Z': Z,
                'H': H
            }
        }

    def compute_irf(self, param_dict=None, theta=None, shock_name=None, shock_size=1.0, 
                    periods=40, vars_to_plot=None, figsize=(15, 10)):
        """
        Computes and plots impulse response functions for the model.
        
        Args:
            param_dict: Dictionary of parameter names and values (optional)
            theta: Parameter vector in correct order (optional)
            shock_name: Name of the shock to simulate (must be in self.labels['shock_labels'])
            shock_size: Size of the shock in standard deviations
            periods: Number of periods to simulate
            vars_to_plot: List of variables to plot (if None, plots all observed variables and selected states)
            figsize: Figure size for the plot
            
        Returns:
            fig: The matplotlib figure containing the IRF plots
            irf_data: DataFrame containing all IRF responses
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Ensure we have parameters
        if theta is None and param_dict is not None:
            # Create ordered theta array
            theta = np.array([param_dict[param] for param in self.theta_param_names])
        elif theta is None and param_dict is None:
            raise ValueError("Either param_dict or theta must be provided")
        
        # Validate shock name
        if shock_name is None:
            raise ValueError("shock_name must be provided")
        
        if shock_name not in self.labels['shock_labels']:
            raise ValueError(f"Shock '{shock_name}' not found. Available shocks: {self.labels['shock_labels']}")
        
        # Get shock index
        shock_idx = self.labels['shock_labels'].index(shock_name)
        
        # Get DSGE core parameters
        core_params = theta[:self.n_dsge_core_params]
        
        # Get Jacobians
        a, b, c = self.evaluate_jacobians(core_params)
        
        # Solve model with Klein's method
        f, p, stab, eig = klein(a, b, self.indices['n_states'])
        
        # Create simple IRF calculation method
        def compute_simple_irf(F, P, x0, periods):
            # Number of state variables
            n_states = P.shape[0]
            # Number of control variables
            n_controls = F.shape[0]
            
            # Create IRF storage
            irf = np.zeros((periods, n_controls + n_states))
            
            # Set initial state
            x = x0.copy()
            
            # Compute IRF
            for t in range(periods):
                # Controls = F * states
                controls = F @ x
                # Store results for this period
                irf[t, :n_controls] = controls
                irf[t, n_controls:] = x
                # Transition to next period
                x = P @ x
            
            return irf
        
        # Now we need to figure out which state receives the shock
        # For now, let's directly use the shock index for mapping
        n_states = self.indices['n_states']
        n_endogenous = self.indices['n_endogenous']
        n_shocks = self.indices['n_shocks']
        
        # Create initial state vector
        x0 = np.zeros(n_states)
        
        # For shock in the DSGE part
        if shock_idx < n_shocks:
            # Check if we're dealing with an exogenous shock
            if shock_name.startswith("SHK_"):
                # Find corresponding state variable (usually starts with RES_)
                # This is the most critical part - correctly mapping shocks to states
                state_name = "RES_" + shock_name[4:] + "_lag"
                
                # Find position in state variables
                if state_name in self.labels['state_labels']:
                    state_idx = self.labels['state_labels'].index(state_name)
                    
                    # Get shock standard deviation
                    shock_std_idx = self.n_dsge_core_params + shock_idx
                    if shock_std_idx < len(theta):
                        shock_std = theta[shock_std_idx]
                        # Apply shock to the state
                        x0[state_idx] = shock_size * shock_std
                    else:
                        x0[state_idx] = shock_size
                else:
                    # Try direct mapping based on R matrix
                    if shock_idx < n_shocks:
                        # Look at the R matrix to find which state receives this shock
                        for i in range(self.R.shape[0]):
                            if self.R[i, shock_idx] != 0:
                                # Found the state that receives this shock
                                state_idx = n_endogenous + i
                                if state_idx < n_states:
                                    # Get shock standard deviation
                                    shock_std_idx = self.n_dsge_core_params + shock_idx
                                    if shock_std_idx < len(theta):
                                        shock_std = theta[shock_std_idx]
                                        # Apply shock to the state
                                        x0[state_idx] = shock_size * shock_std
                                    else:
                                        x0[state_idx] = shock_size
                                    break
            
            # Compute DSGE IRF
            dsge_irf = compute_simple_irf(f, p, x0, periods)
            
            # Get variable names
            control_names = self.labels.get('control_labels', self.model.get('control_variables', []))
            state_names = self.labels['state_labels']
            
            # If control_names is not available, derive it from all_variables and state_variables
            if not control_names:
                all_vars = self.model.get('all_variables', [])
                control_names = [v for v in all_vars if v not in state_names]
            
            # Create DSGE IRF DataFrame
            var_names = control_names + state_names
            dsge_df = pd.DataFrame(dsge_irf, columns=var_names)
            
            # Now handle trend IRF if needed
            if hasattr(self, 'trend_info') and self.trend_info['total_states'] > 0:
                # Get trend state space matrices
                n_trend = self.trend_info['total_states']
                
                # Create trend transition matrix if available
                if hasattr(self, 'T_trend_structure'):
                    T_trend = np.array(self.T_trend_structure)
                    
                    # Initialize trend state
                    trend_state = np.zeros(n_trend)
                    
                    # Create trend IRF storage
                    trend_irf = np.zeros((periods, n_trend))
                    
                    # Compute trend IRF
                    for t in range(periods):
                        trend_irf[t, :] = trend_state
                        trend_state = T_trend @ trend_state
                    
                    # Create trend DataFrame
                    trend_df = pd.DataFrame(trend_irf, columns=self.trend_info['state_labels'])
                    
                    # Combine DSGE and trend
                    combined_df = pd.concat([dsge_df, trend_df], axis=1)
                else:
                    combined_df = dsge_df
            else:
                combined_df = dsge_df
            
            # Create base variable names mapping (without _lag suffix)
            base_vars = {}
            for var in var_names:
                if var.endswith('_lag'):
                    base_name = var[:-4]
                    base_vars[base_name] = var
                else:
                    base_vars[var] = var
            
            # Determine variables to plot
            if vars_to_plot is None:
                # Choose sensible defaults
                vars_to_plot = []
                
                # Add main model variables (excluding RES_ variables)
                for var in base_vars.keys():
                    if not var.startswith('RES_'):
                        vars_to_plot.append(var)
                
                # Limit to at most 12 variables
                vars_to_plot = vars_to_plot[:12]
            
            # Create figure for plotting
            n_vars = len(vars_to_plot)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Plot each variable
            for i, var in enumerate(vars_to_plot):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Check if variable exists directly in results
                    if var in combined_df.columns:
                        column = var
                    elif var in base_vars:
                        # Use the base variable mapping
                        column = base_vars[var]
                        if column not in combined_df.columns:
                            ax.text(0.5, 0.5, f"Variable '{var}' mapped to '{column}' not found", 
                                ha='center', va='center')
                            continue
                    else:
                        # Try to find a fuzzy match
                        matches = [col for col in combined_df.columns 
                                if col == var or col.startswith(var + '_')]
                        
                        if matches:
                            column = matches[0]
                        else:
                            ax.text(0.5, 0.5, f"Variable '{var}' not found", 
                                ha='center', va='center')
                            continue
                    
                    # Plot the variable
                    combined_df[column].plot(ax=ax, color='blue', linewidth=2)
                    
                    # Display original name for clarity
                    display_name = var if var == column else f"{var} (as {column})"
                    ax.set_title(display_name)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Periods')
                    ax.set_ylabel('Deviation')
            
            # Hide unused subplots
            for i in range(n_vars, len(axes)):
                axes[i].set_visible(False)
            
            # Add overall title
            fig.suptitle(f"Impulse Response Functions for {shock_size}σ {shock_name} Shock", 
                        fontsize=16, y=1.02)
            fig.tight_layout()
            
            # Print available variables for debugging
            print("Available variables in IRF results:", sorted(combined_df.columns.tolist()))
            
            return fig, combined_df
        else:
            raise ValueError(f"Unable to properly map shock {shock_name} to a state variable")


    def run_filter_smoother(self, theta: np.ndarray, data_array: np.ndarray):
        """
        Updates state space, creates Kalman filter, and runs smoother.

        Args:
            theta: Parameter vector.
            data_array: Observed data, shape (1, n_timesteps, n_observables).

        Returns:
            simdkalman results object.
        """
        # 1. Get updated state-space matrices
        T, Q_state, Z, H = self.update_state_space(theta)

        # Re-create filter if matrices were modified
        kf = simdkalman.KalmanFilter(state_transition=T, 
                                    process_noise=Q_state, 
                                    observation_model=Z, 
                                    observation_noise=H)


        try:
            smoothed_results = kf.smooth(data_array)
            return smoothed_results
        except Exception as e:
            print(f"Error during Kalman smoothing: {e}")
            # You might want to inspect the matrices T, Q_state, Z, H here
            # print("T:\n", T)
            # print("Q_state:\n", Q_state)
            # print("Z:\n", Z)
            # print("H:\n", H)
            raise # Re-raise the error after printing info




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


def klein(a=None, b=None, n_states=None, eigenvalue_warnings=True):
    """
    Solves linear dynamic models with the form of:
    
    a*Et[x(t+1)] = b*x(t)       
            
    [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
    a vector of nonpredetermined costate variables.
    
    The solution to the model is a set of matrices f, p such that:
    
    u(t)   = f*s(t)
    s(t+1) = p*s(t)
    
    The solution algorithm is based on Klein (2000) and his solab.m Matlab program.
    
    Args:
        a: Coefficient matrix on future-dated variables
        b: Coefficient matrix on current-dated variables
        n_states: Number of state variables
        eigenvalue_warnings: Whether to print warnings about eigenvalues
        
    Returns:
        f: Solution matrix coefficients on s(t) for u(t)
        p: Solution matrix coefficients on s(t) for s(t+1)
        stab: Stability indicator
        eig: Generalized eigenvalues
    """
    s, t, alpha, beta, q, z = la.ordqz(A=a, B=b, sort='ouc', output='complex')

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
        z11i = la.inv(z11)
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



# Example usage - Main script
if __name__ == "__main__":

    # --- Assume these steps were run previously ---
    # 1. DynareParser generated model.json, jacobian_evaluator.py, model_structure.py
    # ----------------------------------------------
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    output_dir = "model_files" # Directory containing generated files
    dynare_file = "qpm_simpl1.dyn" # Just needed for reference if re-parsing
   

    # Define model specifications (as used by parser and solver)
    model_specs = {
        "rs_obs": {"trend": "random_walk", "cycle": "RS"},
        "dla_cpi_obs": {"trend": "random_walk", "cycle": "DLA_CPI"},
        "l_gdp_obs": {"trend": "random_walk", "cycle": "L_GDP_GAP"}
    }
    observed_variables = list(model_specs.keys())


    # 1. Generate the necessary files (JSON model, Jacobian, structure)
    parse_and_generate_files(dynare_file, output_dir, 
                            obs_vars=observed_variables, 
                            model_specs=model_specs)


    # Ensure output directory exists (it should if parser ran)
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found. Ensure DynareParser ran successfully.")
        sys.exit(1)

    # --- Main workflow ---
    # 1. Create ModelSolver instance (loads structure, defines theta order)
    #try:
    solver = ModelSolver(output_dir, model_specs, observed_variables)
    # except Exception as e:
    #     print(f"Failed to initialize ModelSolver: {e}")
    #     sys.exit(1)

    # 2. Load and prepare data (using DataProcessor or similar)
    try:
        us_data = pd.read_csv('transformed_data_us.csv', index_col='Date', parse_dates=True)
        # Ensure data has the 'observed_variables' columns
        if not all(v in us_data.columns for v in observed_variables):
            missing_vars = [v for v in observed_variables if v not in us_data.columns]
            raise ValueError(f"Data file missing required columns: {missing_vars}")

        # Use only the necessary columns in the correct order
        data_for_filter = us_data[observed_variables].values
        # Reshape for simdkalman: (1, n_timesteps, n_observables)
        data_array = data_for_filter.reshape(1, data_for_filter.shape[0], data_for_filter.shape[1])
        data_array = data_array.astype(float) # Ensure float type
        dates = us_data.index # Keep dates for results processing

    except FileNotFoundError:
        print("Error: Data file 'transformed_data_us.csv' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)


    # 3. Define parameter values `theta` IN THE CORRECT ORDER
    #    Get the order from solver.theta_param_names
    #    Example values (replace with your actual draw/initial values)
    initial_param_dict = {
        # DSGE Core Params (Order from labels['param_labels'])
        'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
        'rho_DLA_CPI': 0.75, 'rho_L_GDP_GAP': 0.75, 'rho_rs': 0.8, 'rho_rs2': 0.1,
        # DSGE Shock STDs (Order from labels['shock_labels']) - Assuming shocks are RES_...
        'RES_L_GDP_GAP_lag_std': 1.0, # Parameter name should end in _std
        'RES_RS_lag_std': 1.0,
        'RES_DLA_CPI_lag_std': 1.0,
        # Trend Shock STDs (Order from trend_info calculation)
        'rs_obs_level_shock_std': 0.5,
        'dla_cpi_obs_level_shock_std': 0.5,
        'l_gdp_obs_level_shock_std': 0.1 # Adjust name based on trend_info output
        # Measurement Error STDs (Order from obs_vars) - OMITTED FOR NOW
        # 'rs_obs_meas_error_std': 0.1,
        # 'dla_cpi_obs_meas_error_std': 0.1,
        # 'l_gdp_obs_meas_error_std': 0.1
    }
    smoothed_results = solver.solve_and_filter_calibrated_model(initial_param_dict, data_array)
    
    
    fig, irf_data = solver.compute_irf(
                    param_dict=initial_param_dict,
                    shock_name='SHK_RS',  # Name must match one in labels['shock_labels']
                    shock_size=1.0,       # Size in standard deviations
                    periods=40,           # Number of periods to simulate
                    vars_to_plot=['RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP']  # Variables to plot
    ) 
    # # Construct theta array using the defined order
    # try:
    #     theta_values = [initial_param_dict[pname] for pname in solver.theta_param_names]
    #     theta = np.array(theta_values)
    #     print(f"Constructed theta vector with {len(theta)} values.")
    # except KeyError as e:
    #     print(f"Error: Parameter '{e}' not found in initial_param_dict. Check parameter names.")
    #     # Find missing/mismatched names:
    #     provided_keys = set(initial_param_dict.keys())
    #     expected_keys = set(solver.theta_param_names)
    #     print(f"Missing from dict: {expected_keys - provided_keys}")
    #     print(f"Extra in dict: {provided_keys - expected_keys}")
    #     sys.exit(1)


    # # 4. Update state space and run smoother
    # try:
    #     # Option 1: Call update_state_space then create filter manually
    #     # T, Q_state, Z, H = solver.update_state_space(theta)
    #     # kf = simdkalman.KalmanFilter(state_transition=T, process_noise=Q_state, observation_model=Z, observation_noise=H)
    #     # simdkalman_results = kf.smooth(data_array)

    #     # Option 2: Use helper method if defined
    #     simdkalman_results = solver.run_filter_smoother(theta, data_array)

    #     print("Smoother run successfully.")

    # except Exception as e:
    #     print(f"Error running smoother: {e}")
    #     # Add more diagnostics here if needed
    #     sys.exit(1)

    # # 5. Process results (using DataProcessor or manually)
    # # Get state labels (including augmented trend states)
    # augmented_state_labels = solver.labels['state_labels'] + solver.trend_info['state_labels']

    # # Extract smoothed means
    # smoothed_means = simdkalman_results.smoothed_means[0] # Get first batch

    # # Create DataFrame
    # results_df = pd.DataFrame(smoothed_means, index=dates, columns=augmented_state_labels)

    # print("\nSmoothed States (Head):")
    # print(results_df.head())

    # # Extract and print specific components as before
    # print("\nSmoothed trend components:")
    # trend_columns = [col for col in results_df.columns if any(x in col for x in ['level', 'slope', 'curvature', 'mean'])]
    # print(results_df[trend_columns].head())

    # print("\nSmoothed cyclical components (DSGE states):")
    # # Assuming cycle vars match DSGE state names (adjust if needed)
    # cycle_columns = [spec['cycle'] for spec in model_specs.values() if spec['cycle'] in results_df.columns]
    # print(results_df[cycle_columns].head())
