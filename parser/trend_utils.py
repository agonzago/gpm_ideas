# --- START OF FILE trend_utils.py ---

# trend_utils.py

import numpy as np
# Removed json and os imports as they are not used here

def calculate_trend_positions(model_specs, obs_vars, n_states, n_shocks):
    """
    Calculate positions for each trend component in the augmented state vector
    and define the names for trend shock standard deviation parameters.

    Args:
        model_specs: Dictionary with specifications for observed variables
        obs_vars: List of observed variable names
        n_states: Number of original state variables
        n_shocks: Number of original shock variables

    Returns:
        dict: Information about trend component positions, including
              'shock_std_param_names' for use in theta.
    """
    trend_info = {
        'state_labels': [],       # Labels for trend state components
        'components': {},         # Component info for each observed variable
        'total_states': 0,        # Total number of trend states added
        'total_shocks': 0,        # Total number of trend shocks added
        'param_positions': {},    # Maps parameter names (_std) to Q matrix pos
        'shock_std_param_names': [] # Names of shock std params for theta order
    }

    # Start position after original states
    pos = n_states
    shock_pos = n_shocks # Start index for new shocks

    # Process each observed variable
    for obs_var in obs_vars:
        if obs_var not in model_specs:
             print(f"Warning (calculate_trend_positions): Spec missing for {obs_var}, skipping.")
             continue # Skip if spec missing

        trend_type = model_specs[obs_var]['trend']
        var_info = {'type': trend_type, 'components': {}}

        # Add appropriate state components based on trend type
        if trend_type == 'random_walk':
            # Only level component
            comp_name = 'level'
            state_label = f"{obs_var}_{comp_name}"
            param_name_std = f"{state_label}_shock_std" # Use _std convention

            var_info['components'][comp_name] = {
                'state_pos': pos,
                'shock_pos': shock_pos
            }
            # Map the STANDARD DEVIATION parameter name to the Q matrix position
            trend_info['param_positions'][param_name_std] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name_std)
            trend_info['state_labels'].append(state_label)
            pos += 1
            shock_pos += 1

        elif trend_type == 'second_difference':
            # Level component
            comp_name_level = 'level'
            state_label_level = f"{obs_var}_{comp_name_level}"
            param_name_level_std = f"{state_label_level}_shock_std"
            var_info['components'][comp_name_level] = {'state_pos': pos, 'shock_pos': shock_pos}
            trend_info['param_positions'][param_name_level_std] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name_level_std)
            trend_info['state_labels'].append(state_label_level)
            pos += 1; shock_pos += 1

            # Slope component
            comp_name_slope = 'slope'
            state_label_slope = f"{obs_var}_{comp_name_slope}"
            param_name_slope_std = f"{state_label_slope}_shock_std"
            var_info['components'][comp_name_slope] = {'state_pos': pos, 'shock_pos': shock_pos}
            trend_info['param_positions'][param_name_slope_std] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name_slope_std)
            trend_info['state_labels'].append(state_label_slope)
            pos += 1; shock_pos += 1

            # Curvature component
            comp_name_curv = 'curvature'
            state_label_curv = f"{obs_var}_{comp_name_curv}"
            param_name_curv_std = f"{state_label_curv}_shock_std"
            var_info['components'][comp_name_curv] = {'state_pos': pos, 'shock_pos': shock_pos}
            trend_info['param_positions'][param_name_curv_std] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name_curv_std)
            trend_info['state_labels'].append(state_label_curv)
            pos += 1; shock_pos += 1

        elif trend_type == 'constant_mean':
            # Only constant level (no dynamics) - using 'mean' name
            comp_name = 'mean'
            state_label = f"{obs_var}_{comp_name}"
            # Note: A constant mean typically doesn't have its own shock in this setup,
            # or if it does, it's often fixed small. Adjust naming if needed.
            param_name_std = f"{state_label}_shock_std"
            var_info['components']['level'] = {'state_pos': pos, 'shock_pos': shock_pos} # Still uses 'level' internally
            trend_info['param_positions'][param_name_std] = ('Q', shock_pos, shock_pos)
            trend_info['shock_std_param_names'].append(param_name_std)
            trend_info['state_labels'].append(state_label)
            pos += 1
            shock_pos += 1

        # Store information for this variable
        trend_info['components'][obs_var] = var_info

    # Update total trend states and shocks ADDED
    trend_info['total_states'] = pos - n_states
    trend_info['total_shocks'] = shock_pos - n_shocks # Number of trend shocks added

    # Add total augmented dimensions for convenience
    trend_info['n_aug_states'] = pos
    trend_info['n_aug_shocks'] = shock_pos

    return trend_info

# --- build_trend_transition --- (No changes needed from your version)
def build_trend_transition(T_matrix, trend_info, n_states):
    """
    Build the transition matrix blocks for trend components.

    Args:
        T_matrix: Transition matrix to fill (n_aug_states x n_aug_states)
        trend_info: Trend component position information
        n_states: Number of original state variables
    """
    for obs_var, var_info in trend_info['components'].items():
        trend_type = var_info['type']
        if trend_type == 'random_walk':
            level_pos = var_info['components']['level']['state_pos']
            if 0 <= level_pos < T_matrix.shape[0]: # Bounds check
                 T_matrix[level_pos, level_pos] = 1.0
        elif trend_type == 'second_difference':
             level_pos = var_info['components']['level']['state_pos']
             slope_pos = var_info['components']['slope']['state_pos']
             curv_pos = var_info['components']['curvature']['state_pos']
             if all(0 <= p < T_matrix.shape[0] for p in [level_pos, slope_pos, curv_pos]):
                  T_matrix[level_pos, level_pos] = 1.0
                  T_matrix[level_pos, slope_pos] = 1.0
                  T_matrix[slope_pos, slope_pos] = 1.0
                  T_matrix[slope_pos, curv_pos] = 1.0
                  T_matrix[curv_pos, curv_pos] = 1.0
        elif trend_type == 'constant_mean':
             mean_pos = var_info['components']['level']['state_pos']
             if 0 <= mean_pos < T_matrix.shape[0]:
                 T_matrix[mean_pos, mean_pos] = 1.0
    # Ensure off-diagonal blocks (DSGE states affecting trends, trends affecting DSGE) are zero


# --- build_trend_selection --- (No changes needed from your version)
def build_trend_selection(selection_matrix, trend_info, n_states, n_shocks):
    """
    Build the selection matrix blocks for trend components.

    Args:
        selection_matrix: Selection matrix to fill (n_aug_states x n_aug_shocks)
        trend_info: Trend component position information
        n_states: Number of original state variables
        n_shocks: Number of original shock variables
    """
    for obs_var, var_info in trend_info['components'].items():
        for comp_name, comp_info in var_info['components'].items():
            state_pos = comp_info['state_pos'] # Row index in selection_matrix
            shock_pos = comp_info['shock_pos'] # Column index in selection_matrix
            # Bounds checks
            if 0 <= state_pos < selection_matrix.shape[0] and 0 <= shock_pos < selection_matrix.shape[1]:
                # This maps the specific trend shock (at shock_pos) to the trend state (at state_pos)
                selection_matrix[state_pos, shock_pos] = 1.0
            else:
                 print(f"Warning (build_trend_selection): Invalid indices state={state_pos}, shock={shock_pos}")

# --- build_trend_observation --- (Needs model_specs passed in for cycle var mapping)
def build_trend_observation(Z_matrix, trend_info, model_specs, obs_vars, observable_labels):
    """
    Build the observation matrix blocks for trend components.
    Assumes trend components additively affect the corresponding cycle variable observation.

    Args:
        Z_matrix: Observation matrix to fill (n_observables x n_aug_states)
        trend_info: Trend component position information
        model_specs: Dictionary with specifications (needed for cycle var mapping)
        obs_vars: List of observed variable names (e.g., 'l_gdp_obs')
        observable_labels: List of all observable labels (DSGE cycle vars + DSGE states)
    """
    # Map from observable_labels name to its row index in Z
    obs_label_to_index = {label: i for i, label in enumerate(observable_labels)}

    for obs_var in obs_vars: # e.g., "l_gdp_obs"
        var_info = trend_info['components'].get(obs_var)
        if not var_info: continue # Skip if no trend info for this obs_var

        # --- Find the row in Z corresponding to the observation ---
        # The trend component affects the observation equation for the *cycle* variable
        cycle_var_name = model_specs.get(obs_var, {}).get('cycle')
        if cycle_var_name is None:
             print(f"Warning (build_trend_observation): Cycle variable name not found in model_specs for {obs_var}")
             continue

        try:
            # Find the row index in Z matrix corresponding to this cycle variable
            obs_row_idx = obs_label_to_index[cycle_var_name]
        except KeyError:
            print(f"Warning (build_trend_observation): Cycle variable '{cycle_var_name}' for {obs_var} not found in observable_labels.")
            continue
        # --- End Find Row ---

        # Map the appropriate trend state component(s) to this observation row
        # Assumes only the 'level' or 'mean' component adds to the observation
        trend_comp_to_map = None
        if 'level' in var_info['components']:
            trend_comp_to_map = 'level'
        elif 'mean' in var_info['components']: # Check if 'mean' is used
             # This depends on calculate_trend_positions using 'level' key even for 'constant_mean'
             if 'level' in var_info['components']: trend_comp_to_map = 'level'


        if trend_comp_to_map:
            trend_state_pos = var_info['components'][trend_comp_to_map]['state_pos'] # Column index in Z
            # Bounds check before assignment
            if 0 <= obs_row_idx < Z_matrix.shape[0] and 0 <= trend_state_pos < Z_matrix.shape[1]:
                 # Z[observation_row, state_column] = 1.0
                 # This means: obs_cycle_var = dsge_cycle_var + 1.0 * trend_level_state + ...
                 Z_matrix[obs_row_idx, trend_state_pos] = 1.0
            else:
                 print(f"Warning (build_trend_observation): Invalid indices row={obs_row_idx}, col={trend_state_pos}")

# --- END OF FILE trend_utils.py ---