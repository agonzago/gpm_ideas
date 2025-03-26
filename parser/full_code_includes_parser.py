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
    Calculate positions for each trend component in the augmented state vector.
    
    Args:
        model_specs: Dictionary with specifications for observed variables
        obs_vars: List of observed variable names
        n_states: Number of original state variables
        n_shocks: Number of original shock variables
        
    Returns:
        dict: Information about trend component positions
    """
    trend_info = {
        'state_labels': [],       # Labels for trend state components
        'components': {},         # Component info for each observed variable
        'total_states': 0,        # Total number of trend states
        'param_positions': {}     # Maps parameter names to matrix positions
    }
    
    # Start position after original states
    pos = n_states
    shock_pos = n_shocks
    
    # Process each observed variable
    for obs_var in obs_vars:
        trend_type = model_specs[obs_var]['trend']
        var_info = {'type': trend_type, 'components': {}}
        
        # Add appropriate state components based on trend type
        if trend_type == 'random_walk':
            # Only level component
            var_info['components']['level'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            
            # Add shock parameter mapping
            param_name = f"{obs_var}_level_shock"
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            
            trend_info['state_labels'].append(f"{obs_var}_level")
            pos += 1
            shock_pos += 1
            
        elif trend_type == 'second_difference':
            # Level, slope, and curvature components
            var_info['components']['level'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            
            # Add shock parameter mapping
            param_name = f"{obs_var}_level_shock"
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            
            trend_info['state_labels'].append(f"{obs_var}_level")
            pos += 1
            shock_pos += 1
            
            var_info['components']['slope'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            
            # Add shock parameter mapping
            param_name = f"{obs_var}_slope_shock"
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            
            trend_info['state_labels'].append(f"{obs_var}_slope")
            pos += 1
            shock_pos += 1
            
            var_info['components']['curvature'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            
            # Add shock parameter mapping
            param_name = f"{obs_var}_curvature_shock"
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            
            trend_info['state_labels'].append(f"{obs_var}_curvature")
            pos += 1
            shock_pos += 1
            
        elif trend_type == 'constant_mean':
            # Only constant level (no dynamics)
            var_info['components']['level'] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            
            # Add shock parameter mapping
            param_name = f"{obs_var}_mean_shock"
            trend_info['param_positions'][param_name] = ('Q', shock_pos, shock_pos)
            
            trend_info['state_labels'].append(f"{obs_var}_mean")
            pos += 1
            shock_pos += 1
        
        # Store information for this variable
        trend_info['components'][obs_var] = var_info
    
    # Update total trend states
    trend_info['total_states'] = pos - n_states
    
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

        # Create indices dictionary (ALWAYS defined)
        indices = {
            'n_endogenous': n_endogenous,
            'n_exo_states': n_exo_states,
            'n_controls': n_controls,
            'n_shocks': n_shocks,
            'n_states': n_states,
            'n_observables': n_controls + n_states  # Corrected: Use n_controls + n_states
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

        # Create C matrix structure (maps states to observables)
        n_observables =  n_controls + n_states
        C_structure = np.zeros((n_observables, n_states))

        # States mapped one-to-one (identity matrix part)
        C_structure[n_controls:, :] = np.eye(n_states)

        # D matrix (direct shock impact on observables)
        D = np.zeros((n_observables, n_shocks))

        # Create labels for the state space
        labels = {
            'state_labels': self.state_variables,
            'observable_labels': self.control_variables + self.state_variables,
            'shock_labels': self.varexo_list
        }

        # 2. Add constant components for trend augmentation
        n_obs_vars = len(self.obs_vars) if hasattr(self, 'obs_vars') and self.obs_vars else 0 # Safe check if obs_vars
        n_trend_states = 3 * n_obs_vars

        # Create constant structure for augmented transition matrix
        T_trend_structure = np.zeros((n_trend_states, n_trend_states))
        for i in range(0, n_trend_states, 3):
            T_trend_structure[i, i] = 1.0  # Level on itself
            T_trend_structure[i, i + 1] = 1.0  # Level on slope
            T_trend_structure[i + 1, i + 1] = 1.0  # Slope on itself
            T_trend_structure[i + 1, i + 2] = 1.0  # Slope on curvature
            T_trend_structure[i + 2, i + 2] = 1.0  # Curvature on itself

        # Create constant structure for augmented selection matrix
        R_trend_structure = np.eye(n_trend_states)

        # Create constant structure for augmented observation matrix (mapping trends)
        C_trend_structure = np.zeros((len(self.all_variables), n_trend_states))

        structure = {
            'indices': indices,
            'R': R,
            'B_structure': B_structure,
            'C_structure': C_structure,
            'D': D,
            'labels': labels,
        }

        if hasattr(self, 'obs_vars') and self.obs_vars:  # Only add if obs_vars exists

            structure['T_trend_structure'] = T_trend_structure.tolist()  # Convert to lists for JSON
            structure['R_trend_structure'] = R_trend_structure.tolist()
            structure['C_trend_structure'] = C_trend_structure.tolist()

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

        parser = DynareParser(dynare_file) # create a parser without obs var
        if obs_vars is not None:
            parser.obs_vars = obs_vars # Add observed vars if available

        model_json = parser.save_json(os.path.join(output_dir, "model.json"))

        parser.generate_jacobian_evaluator(os.path.join(output_dir, "jacobian_evaluator.py"))

        structure = parser.generate_model_structure()
        with open(os.path.join(output_dir, "model_structure.py"), 'w') as f:
            f.write("import numpy as np\n\n")
            f.write(f"indices = {repr(structure['indices'])}\n\n")
            f.write(f"R = np.array({repr(structure['R'].tolist())})\n\n")
            f.write(f"B_structure = np.array({repr(structure['B_structure'].tolist())})\n\n")
            f.write(f"C_structure = np.array({repr(structure['C_structure'].tolist())})\n\n")
            f.write(f"D = np.array({repr(structure['D'].tolist())})\n\n")
            f.write(f"labels = {repr(structure['labels'])}\n")
            try:
                f.write(f"T_trend_structure = np.array({repr(structure['T_trend_structure'])})\n\n")
                f.write(f"R_trend_structure = np.array({repr(structure['R_trend_structure'])})\n\n")
                f.write(f"C_trend_structure = np.array({repr(structure['C_trend_structure'])})\n\n")
            except KeyError:
                print("Trend structure not found. If you are running a simple_model or a simple case you can ignore this.  ")

        print(f"All model files generated in {output_dir}")





class ParameterMapper:
    """
    Maps model parameters to their respective locations in state space matrices.
    This allows efficient updates of state space matrices without repeatedly
    looping through parameter names.
    """
    
    def __init__(self, indices, labels, model_specs, obs_vars, output_dir=None):
        """
        Initialize the parameter mapper with information about the model structure.
        
        Args:
            indices: Dictionary with model dimension information
            labels: Dictionary with variable labels
            model_specs: Dictionary with specifications for observed variables
            obs_vars: List of observed variable names
            output_dir: Directory to save/load mapping information (optional)
        """
        self.indices = indices
        self.labels = labels
        self.model_specs = model_specs
        self.obs_vars = obs_vars
        self.output_dir = output_dir
        
        # Extract dimensions
        self.n_endogenous = indices['n_endogenous']
        self.n_exo_states = indices['n_exo_states']
        self.n_states = indices['n_states']
        self.n_shocks = indices['n_shocks']
        self.n_controls = indices['n_controls']
        
        # Try to load mappings from file first
        if output_dir and self._load_mappings():
            print("Parameter mappings loaded from file.")
        else:
            # Build the mappings from scratch
            self._build_mappings()
            # Save mappings if output directory is provided
            if output_dir:
                self._save_mappings()
    
    def _build_mappings(self):
        """
        Build the mappings between parameters and matrix locations.
        """
        # Calculate trend information
        self.trend_info = calculate_trend_positions(
            self.model_specs, 
            self.obs_vars, 
            self.n_states, 
            self.n_shocks
        )
        
        # Dictionary to map parameters to their locations in matrices
        self.param_map = {}
        
        # 1. Map DSGE model parameters to the Jacobian parameter vector
        # These should come from the model.json file
        model_params = self.labels.get('param_labels', [])
        for i, param in enumerate(model_params):
            self.param_map[param] = ('DSGE', i)  # (matrix_type, position)
        
        # 2. Map shock standard deviations to Q matrix (process noise covariance)
        # a. DSGE shocks (map to original shock positions)
        shock_labels = self.labels['shock_labels']
        for i, shock in enumerate(shock_labels):
            param_name = f"{shock}_shock"
            # Map to position in the process noise covariance matrix (diagonal elements)
            self.param_map[param_name] = ('Q', i, i)
        
        # b. Add trend shock parameter mappings from trend_info
        self.param_map.update(self.trend_info['param_positions'])
        
        # 3. Map measurement error standard deviations to R matrix
        for i, obs_var in enumerate(self.obs_vars):
            # Find base variable (without _obs)
            base_var = obs_var.replace("_obs", "")
            # Find position in observation vector
            try:
                var_idx = self.labels['observable_labels'].index(base_var)
                # Map to position in the measurement noise covariance matrix
                self.param_map[f"{obs_var}_meas_error"] = ('R', var_idx, var_idx)
            except ValueError:
                # Skip if variable not found in observable_labels
                pass
    
    def _save_mappings(self):
        """
        Save parameter mappings to a file for efficient reuse.
        """
        if not self.output_dir:
            return
        
        # Create output path
        mapping_file = os.path.join(self.output_dir, "parameter_mappings.pkl")
        
        # Data to save
        data = {
            'param_map': self.param_map,
            'trend_info': self.trend_info
        }
        
        # Save to file
        with open(mapping_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save a JSON version for human readability
        json_file = os.path.join(self.output_dir, "parameter_mappings.json")
        
        # Convert to JSON-serializable format
        json_data = {
            'param_map': {k: list(v) for k, v in self.param_map.items()},
            'trend_info': self.trend_info
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Parameter mappings saved to {mapping_file} and {json_file}")
    
    def _load_mappings(self):
        """
        Load parameter mappings from file.
        
        Returns:
            bool: True if mappings were successfully loaded, False otherwise
        """
        if not self.output_dir:
            return False
        
        mapping_file = os.path.join(self.output_dir, "parameter_mappings.pkl")
        
        if not os.path.exists(mapping_file):
            return False
        
        try:
            with open(mapping_file, 'rb') as f:
                data = pickle.load(f)
            
            self.param_map = data['param_map']
            self.trend_info = data['trend_info']
            return True
        except Exception as e:
            print(f"Error loading parameter mappings: {e}")
            return False
    
    def build_process_noise_covariance(self, parameters):
        """
        Build the process noise covariance matrix (Q) using parameter values.
        
        Args:
            parameters: Dictionary mapping parameter names to values
            
        Returns:
            Q: Process noise covariance matrix
        """
        # Total number of shocks
        n_total_shocks = self.n_shocks + self.trend_info['total_states']
        
        # Initialize process noise covariance matrix
        Q = np.zeros((n_total_shocks, n_total_shocks))
        
        # Fill diagonal elements based on parameter mapping
        for param_name, param_value in parameters.items():
            if param_name in self.param_map:
                mapping = self.param_map[param_name]
                if mapping[0] == 'Q':
                    # Square the parameter value to get variance
                    Q[mapping[1], mapping[2]] = param_value**2
        
        return Q
    
    def build_measurement_noise_covariance(self, parameters):
        """
        Build the measurement noise covariance matrix (R) using parameter values.
        
        Args:
            parameters: Dictionary mapping parameter names to values
            
        Returns:
            R: Measurement noise covariance matrix
        """
        # Number of observables
        n_observables = self.n_controls + self.n_states
        
        # Initialize measurement noise covariance matrix (diagonal)
        R = np.zeros((n_observables, n_observables))
        
        # Fill diagonal elements based on parameter mapping
        for param_name, param_value in parameters.items():
            if param_name in self.param_map:
                mapping = self.param_map[param_name]
                if mapping[0] == 'R':
                    # Square the parameter value to get variance
                    R[mapping[1], mapping[2]] = param_value**2
        
        # Set a small default value for stability if no parameters mapped
        if np.all(R == 0):
            R = np.eye(n_observables) * 1e-5
        
        return R
    
    def get_dsge_parameters(self, parameters):
        """
        Extract DSGE model parameters in the correct order for the Jacobian evaluator.
        
        Args:
            parameters: Dictionary mapping parameter names to values
            
        Returns:
            dsge_params: List of parameter values in the correct order for evaluate_jacobians
        """
        # Find maximum index for DSGE parameters
        max_idx = -1
        for param_name, mapping in self.param_map.items():
            if mapping[0] == 'DSGE':
                max_idx = max(max_idx, mapping[1])
        
        # Initialize parameter vector with zeros
        dsge_params = np.zeros(max_idx + 1)
        
        # Fill parameter vector based on mapping
        for param_name, param_value in parameters.items():
            if param_name in self.param_map:
                mapping = self.param_map[param_name]
                if mapping[0] == 'DSGE':
                    dsge_params[mapping[1]] = param_value
        
        return dsge_params


class ModelSolver:
    """
    Enhanced ModelSolver class with improved state space construction
    and efficient parameter handling.
    """
    
    def __init__(self, output_dir, model_specs, obs_vars):
        """
        Initialize the ModelSolver.
        
        Args:
            output_dir: Directory containing model files
            model_specs: Dictionary with specifications for observed variables
            obs_vars: List of observed variable names
        """
        self.output_dir = output_dir
        self.model_specs = model_specs
        self.obs_vars = obs_vars
        
        # Load model components
        self.load_model()
        self.load_jacobian_evaluator()
        self.load_model_structure()
        
        # Initialize matrices for state space
        self.ss = None  # Will hold state space representation
        self.kf = None  # Will hold Kalman filter object
        
        # Create the parameter mapper
        self._create_parameter_mapper()
        
        # Validate model specifications
        self.check_spec_vars()
    
    def load_model(self):
        """Load the model from the JSON file."""
        model_path = os.path.join(self.output_dir, "model.json")
        with open(model_path, 'r') as f:
            self.model = json.load(f)
    
    def load_jacobian_evaluator(self):
        """Load the Jacobian evaluator function from the Python file."""
        jac_path = os.path.join(self.output_dir, "jacobian_evaluator.py")
        spec = importlib.util.spec_from_file_location("jacobian_evaluator", jac_path)
        self.jacobian_module = importlib.util.module_from_spec(spec)
        sys.modules["jacobian_evaluator"] = self.jacobian_module
        spec.loader.exec_module(self.jacobian_module)
        self.evaluate_jacobians = self.jacobian_module.evaluate_jacobians
    
    def load_model_structure(self):
        """Load the pre-computed model structure from the Python file."""
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
    
    def _create_parameter_mapper(self):
        """Create the parameter mapper for efficient parameter handling."""
        self.param_mapper = ParameterMapper(
            indices=self.indices,
            labels=self.labels,
            model_specs=self.model_specs,
            obs_vars=self.obs_vars,
            output_dir=self.output_dir
        )
        
        # Store trend info for easy access
        self.trend_info = self.param_mapper.trend_info
    
    def check_spec_vars(self):
        """Check that model specifications match variables in the model."""
        # Check observation variable specifications
        for obs_var, spec in self.model_specs.items():
            # Use the cycle variable directly for matching
            cycle_var = spec['cycle']
            if cycle_var not in self.model['all_variables']:
                raise ValueError(f"Cycle variable {cycle_var} for {obs_var} not found.")
            
            # Check trend specification
            if 'trend' not in spec:
                raise ValueError(f"Trend specification missing for {obs_var}")
                
            trend_type = spec['trend']
            valid_trends = ['random_walk', 'second_difference', 'constant_mean']
            if trend_type not in valid_trends:
                raise ValueError(f"Invalid trend type '{trend_type}' for {obs_var}. "
                               f"Must be one of {valid_trends}")
            
            # Check that names match observed variables
        # Check that all cycle variables (not base names) are in the model
        list_cycle_names = [spec['cycle'] for spec in self.model_specs.values()]
        list_model_names = self.model['all_variables']
        if not all(item in list_model_names for item in list_cycle_names):
            raise ValueError('Not all cycle variable names match those from the .mod file')
    
    def solve_and_create_state_space(self, params):
        """
        Solve the model and create the state-space representation.
        
        Args:
            params: List or array of parameter values.
            
        Returns:
            dict: State-space representation (A, B, C, D matrices and metadata).
        """
        # Evaluate Jacobians
        a, b, c = self.evaluate_jacobians(params)
        
        # Import klein function from appropriate place
        # If you have the function in this file, you can call it directly
        f, p, stab, eig = klein(a, b, self.indices['n_states'])
        
        # Store solution matrices
        self.f = f
        self.p = p
        self.stab = stab
        self.eig = eig
        
        # Create state space representation
        ss = {
            'A': p.copy(),
            'B': self.B_structure.copy(),
            'C': self.C_structure.copy(),
            'D': self.D.copy(),
        }
        
        # Fill in policy function part of C matrix
        ss['C'][:self.indices['n_controls'], :] = f
        
        # Fill in B matrix (shock impacts on states)
        ss['B'][self.indices['n_endogenous']:, :] = self.R
        
        # Add metadata
        ss.update({
            'state_labels': self.labels['state_labels'],
            'observable_labels': self.labels['observable_labels'],
            'shock_labels': self.labels['shock_labels'],
            'n_states': self.indices['n_states'],
            'n_shocks': self.indices['n_shocks'],
            'n_observables': self.indices['n_observables'],
        })
        
        return ss
    
    def augment_state_space(self):
        """
        Augment the state space model with stochastic trends.
        
        Returns:
            dict: Augmented state space representation.
        """
        # Get dimensions
        n_states = self.indices['n_states']
        n_shocks = self.indices['n_shocks']
        n_controls = self.indices['n_controls']
        n_observables = n_controls + n_states
        
        # Total trend states from trend_info
        n_trend_states = self.trend_info['total_states']
        
        # Total dimensions in augmented system
        n_aug_states = n_states + n_trend_states
        n_aug_shocks = n_shocks + n_trend_states
        
        # Create augmented state labels
        augmented_states = self.labels['state_labels'] + self.trend_info['state_labels']
        
        # Create augmented transition matrix (A matrix)
        T_augmented = np.zeros((n_aug_states, n_aug_states))
        
        # Fill in DSGE state transition part (upper left block)
        T_augmented[:n_states, :n_states] = self.p
        
        # Fill in trend transition blocks
        build_trend_transition(T_augmented, self.trend_info, n_states)
        
        # Create augmented selection matrix (B matrix)
        R_augmented = np.zeros((n_aug_states, n_aug_shocks))
        
        # Fill in DSGE selection part (upper left block)
        R_augmented[:n_states, :n_shocks] = self.B_structure
        
        # Fill in trend selection blocks
        build_trend_selection(R_augmented, self.trend_info, n_states, n_shocks)
        
        # Create augmented observation matrix (C matrix)
        Z_augmented = np.zeros((n_observables, n_aug_states))
        
        # Fill in DSGE observation part (left columns)
        Z_augmented[:, :n_states] = self.C_structure
        
        # Fill in trend observation blocks
        build_trend_observation(Z_augmented, self.trend_info, self.obs_vars, self.labels['observable_labels'])
        
        # Store augmented state space
        self.ss = {
            "augmented_states": augmented_states,
            "transition": T_augmented,
            "selection": R_augmented,
            "design": Z_augmented,
            "trend_info": self.trend_info  # Store for reference
        }
        
        return self.ss
    
    def create_simdkalman_filter(self, parameters):
        """
        Create simdkalman.KalmanFilter object with the augmented state space.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            simdkalman.KalmanFilter: Configured Kalman filter
        """
        if self.ss is None:
            raise ValueError("State space not created. Call augment_state_space first.")
        
        # Build process noise covariance matrix using parameter mapper
        Q = self.param_mapper.build_process_noise_covariance(parameters)
        
        # Build measurement noise covariance matrix
        R = self.param_mapper.build_measurement_noise_covariance(parameters)
        
        # Create actual process noise in state space
        Q_state = self.ss["selection"] @ Q @ self.ss["selection"].T
        
        # Create the filter
        self.kf = simdkalman.KalmanFilter(
            state_transition=self.ss["transition"],
            process_noise=Q_state,
            observation_model=self.ss["design"],
            observation_noise=R
        )
        
        return self.kf
    
    def simulate_states_with_trends(self, data_array):
        """
        Simulate states with stochastic trends using simdkalman.
        
        Args:
            data_array: Data array with shape (1, n_timesteps, n_observed_vars)
            
        Returns:
            Results from simdkalman.KalmanFilter.smooth()
        """
        if self.kf is None:
            raise ValueError("simdkalman filter not initialized. Call create_simdkalman_filter first.")
        
        # Run simulation smoother
        results = self.kf.smooth(data_array)
        
        return results


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


def run_smoother(data, parameters, model_specs, observed_variables, output_dir):
    """
    Run the simulation smoother on the augmented state space model.
    All parameter mappings and matrix positions are cached for efficiency.
    
    Args:
        data: Pandas DataFrame with observed data
        parameters: Dictionary mapping parameter names to values
        model_specs: Dictionary with specifications for observed variables
        observed_variables: List of observed variable names
        output_dir: Directory containing model files
        
    Returns:
        solver: ModelSolver instance
        results: Results from the simulation smoother
    """
    # 1. Create data processor
    data_processor = DataProcessor(observed_variables)
    
    # 2. Prepare data - only done once
    reshaped_data, dates = data_processor.prepare_data(data)
    
    # 3. Create ModelSolver instance
    solver = ModelSolver(output_dir, model_specs, observed_variables)
    
    # 4. Get DSGE parameters from mapper
    dsge_params = solver.param_mapper.get_dsge_parameters(parameters)
    
    # 5. Solve the DSGE model
    solver.solve_and_create_state_space(dsge_params)
    
    # 6. Augment the state space with stochastic trends
    solver.augment_state_space()
    
    # 7. Create simdkalman filter with parameters
    # This already uses the parameter mapper to efficiently build noise matrices
    solver.create_simdkalman_filter(parameters)
    
    # 8. Run simulation smoother
    simdkalman_results = solver.simulate_states_with_trends(reshaped_data)
    
    # 9. Process results
    processed_results = data_processor.process_results(
        simdkalman_results, 
        solver.ss["augmented_states"]
    )
    
    return solver, processed_results



# Example usage - Main script
if __name__ == "__main__":
    import pandas as pd
    
    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # Extract the directory from the script path
    script_dir = os.path.dirname(script_path)

    # Change the current working directory
    os.chdir(script_dir)

    # Verify the change
    print(f"Current working directory: {os.getcwd()}")

    # 1. Parse the Dynare model and generate model files
    dynare_file = "qpm_simpl1.dyn"  # Replace with your Dynare model file
    output_dir = "model_files"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Observed variables and their specifications. 
    # This dictionary provides important structural information for setting up the model.
    model_specs = {
        "rs_obs": {"trend": "random_walk", "cycle": "RS"},  # rs_obs has a random walk trend
        "dla_cpi_obs": {"trend": "random_walk", "cycle": "DLA_CPI"},  # dla_cpi_obs has a random-walk trend
        "l_gdp_obs": {"trend": "random_walk", "cycle": "L_GDP_GAP"}  # l_gdp_obs has a random walk trend
    }
    observed_variables = list(model_specs.keys())  # The observed variables used in the estimation

    DynareParser.parse_and_generate_files(
        dynare_file, output_dir, obs_vars=observed_variables, model_specs=model_specs
    )
    
    
    # 2. Load and prepare the data
    us_data = pd.read_csv('transformed_data_us.csv', index_col='Date', parse_dates=True)
    
    # 3. Set initial parameter values. Make sure these match your Dynare model parameters.
   # Define filtration parameters
    initial_param_values = {
        'b1': 0.7,
        'b4': 0.7,
        'a1': 0.5,
        'a2': 0.1,
        'g1': 0.7,
        'g2': 0.3,
        'g3': 0.25,
        'rho_DLA_CPI': 0.75,
        'rho_L_GDP_GAP': 0.75,
        'rho_rs': 0.8,
        'rho_rs2': 0.1,
        "RES_L_GDP_GAP_lag_shock": 1,
        "RES_RS_lag_shock": 1,
        "RES_DLA_CPI_lag_shock": 1,
        "rs_obs_level_shock": 0.5,
        "dla_cpi_obs_level_shock": 0.5,        
        "l_gdp_gap_obs_mean_shock": 0.1,
        "rs_obs_meas_error": 0.1,
        "dla_cpi_obs_meas_error": 0.1,
        "l_gdp_gap_obs_meas_error": 0.1
    }
    
    
    # 4. Run the smoother. This uses the parsed model and data.
    solver, results = run_smoother(
        us_data, initial_param_values, model_specs, observed_variables, output_dir
    )
    
    # 5. Process and analyze the results
    print("Smoothed trend components:")
    trend_columns = [col for col in results['smoothed_states'].columns if any(x in col for x in ['level', 'slope', 'curvature', 'mean'])]
    print(results['smoothed_states'][trend_columns].head())
    
    print("\nSmoothed cyclical components:")
    cycle_columns = [col for col in results['smoothed_states'].columns if col in ['RS', 'DLA_CPI', 'L_GDP_GAP']]  # Select cyclical component columns
    print(results['smoothed_states'][cycle_columns].head())

