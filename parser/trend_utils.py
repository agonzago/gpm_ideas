 trend_utils.py

import numpy as np
import json
import os

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
