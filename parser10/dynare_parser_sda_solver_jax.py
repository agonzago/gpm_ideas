# -*- coding: utf-8 -*-
"""
Enhanced Dynare Parser and State-Space Solver (JAX Compatible)

Parses Dynare-like models, solves using JAX-based SDA, builds augmented
state-space, computes IRFs. Includes symbolic pre-computation of matrix
elements and their gradients for use with JAX custom VJPs.
"""

import re
import sympy
import numpy as np
from collections import OrderedDict, namedtuple
import copy
import os
# from numpy.linalg import norm # No longer needed here
# from scipy.linalg import lu_factor, lu_solve, block_diag # No longer needed here
import matplotlib.pyplot as plt
import warnings

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.typing import ArrayLike
from typing import Tuple, Optional, List, Dict, Any, Callable
from jax import lax

# JAX machine epsilon
_JAX_EPS = jnp.finfo(jnp.float64).eps if jax.config.jax_enable_x64 else jnp.finfo(jnp.float32).eps

# --- Force CPU Execution (Optional) ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update("jax_platforms", "cpu")
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

# Ensure JAX is configured for float64 if enabled
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
# print(f"Using JAX with dtype: {_DEFAULT_DTYPE}")


# --- Helper Functions (Plotting, Simulation - Mostly unchanged) ---
def plot_irfs(irf_values, var_names, horizon, title="Impulse Responses"):
    """ Simple IRF plotting function """
    irf_values_np = np.asarray(irf_values)
    var_names_list = list(var_names)
    num_vars = irf_values_np.shape[1]
    if num_vars == 0:
        # print(f"No variables to plot for: {title}")
        return
    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(min(5*cols, 18), 3*rows), sharex=True, squeeze=False)
    axes = axes.flatten()
    plt.suptitle(title, fontsize=14)
    time = np.arange(horizon)
    for i, var_name in enumerate(var_names_list):
        if i < len(axes):
            ax = axes[i]
            ax.plot(time, irf_values_np[:, i], label=var_name)
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
            ax.set_title(var_name)
            ax.grid(True, linestyle='--', alpha=0.6)
            if (i // cols) == (rows - 1):
                ax.set_xlabel("Horizon")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def simulate_ssm_data(
    P: ArrayLike, R: ArrayLike, Omega: ArrayLike, T: int, key: jax.random.PRNGKey,
    state_init: Optional[ArrayLike] = None, measurement_noise_std: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    
    P_jax, R_jax, Omega_jax = jnp.asarray(P, dtype=_DEFAULT_DTYPE), jnp.asarray(R, dtype=_DEFAULT_DTYPE), jnp.asarray(Omega, dtype=_DEFAULT_DTYPE)
    n_state, n_shock, n_obs = P_jax.shape[0], (R_jax.shape[1] if R_jax.ndim == 2 and R_jax.shape[1] > 0 else 0), (Omega_jax.shape[0] if Omega_jax.ndim == 2 and Omega_jax.shape[0] > 0 else 0)

    if n_shock == 0: 
        R_jax = jnp.zeros((n_state, 0), dtype=_DEFAULT_DTYPE)
    if n_obs == 0: 
        Omega_jax = jnp.zeros((0, n_state), dtype=_DEFAULT_DTYPE)
    
    key_state, key_measure = jax.random.split(key)
    
    s_previous = jnp.asarray(state_init, dtype=_DEFAULT_DTYPE) if state_init is not None else jnp.zeros(n_state, dtype=_DEFAULT_DTYPE)
    state_shocks = jax.random.normal(key_state, shape=(T, n_shock), dtype=_DEFAULT_DTYPE) if n_shock > 0 else jnp.zeros((T, 0), dtype=_DEFAULT_DTYPE)
    measurement_noise = (jax.random.normal(key_measure, shape=(T, n_obs), dtype=_DEFAULT_DTYPE) * measurement_noise_std) if measurement_noise_std > 0.0 and n_obs > 0 else jnp.zeros((T, n_obs), dtype=_DEFAULT_DTYPE)
    
    def step(s_prev, inputs):
        e_t, eta_t = inputs
        s_t = P_jax @ s_prev + (R_jax @ e_t if n_shock > 0 else 0)
        y_t = (Omega_jax @ s_t + eta_t) if n_obs > 0 else jnp.zeros(0, dtype=_DEFAULT_DTYPE)
        return s_t, (s_t, y_t)
    _, (states_T, ys_T) = lax.scan(step, s_previous, (state_shocks, measurement_noise))

    return states_T, ys_T

def plot_simulation(sim_data, var_names, title="Simulated Data"):
    sim_data_np, var_names_list = np.asarray(sim_data), list(var_names)
    num_vars, T_sim = sim_data_np.shape[1], sim_data_np.shape[0]
    if num_vars == 0: 
        return
    
    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(min(5*cols,18), 3*rows), sharex=True, squeeze=False)
    axes = axes.flatten()
    
    plt.suptitle(title, fontsize=14)
    time_axis = np.arange(T_sim)
    for i, name in enumerate(var_names_list):
        if i < len(axes):
            ax = axes[i]; ax.plot(time_axis, sim_data_np[:, i], label=name); ax.set_title(name)
            ax.grid(True, linestyle='--', alpha=0.6)
            if (i // cols) == (rows - 1): ax.set_xlabel("Time")
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_simulation_with_trends_matched(
    sim_observables: np.ndarray, obs_var_names: list, sim_states: np.ndarray, state_var_names: list,
    trend_state_names: list, contemporaneous_trend_defs: Dict[str, str],
    mapping: Optional[Dict[str, str]] = None, title="Simulated Observables and Trends"
):
    sim_obs_np, sim_states_np = np.asarray(sim_observables), np.asarray(sim_states)
    time_axis = np.arange(sim_obs_np.shape[0])
    if mapping is None:
        mapping = {}
        for obs in obs_var_names:
            obs_base = obs.split('_OBS')[0]
            for trend in trend_state_names + list(contemporaneous_trend_defs.keys()):
                if obs_base == trend.split('_TREND')[0]: mapping[obs] = trend; break
    cols = 2; rows = (len(obs_var_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(min(6*cols,18), 4*rows), squeeze=False); axes=axes.flatten()
    plt.suptitle(title, fontsize=14)
    plotted_count = 0
    for i, obs_name in enumerate(obs_var_names):
        if i >= len(axes): break
        ax = axes[i]; plotted_count += 1
        try: ax.plot(time_axis, sim_obs_np[:, obs_var_names.index(obs_name)], label=f"{obs_name}", linewidth=2)
        except (ValueError, IndexError): continue
        trend_name, plotted_trend = mapping.get(obs_name), False
        if trend_name:
            if trend_name in state_var_names:
                try: ax.plot(time_axis, sim_states_np[:, state_var_names.index(trend_name)], label=f"{trend_name} (State)", linestyle='--', alpha=0.85); plotted_trend = True
                except (ValueError, IndexError): pass # print(f"Warning: State trend '{trend_name}' mapped but not found.")
            elif trend_name in contemporaneous_trend_defs:
                def_str = contemporaneous_trend_defs[trend_name]; parts = [p.strip() for p in def_str.split('+')]
                if len(parts) == 2:
                    try:
                        idx1, idx2 = state_var_names.index(parts[0]), state_var_names.index(parts[1])
                        derived_trend = sim_states_np[:, idx1] + sim_states_np[:, idx2]
                        ax.plot(time_axis, derived_trend, label=f"{trend_name} (Derived)", linestyle='--', alpha=0.85); plotted_trend = True
                    except ValueError: pass # print(f"Warning: Could not find states to derive '{trend_name}'.")
        if plotted_trend: ax.legend()
        ax.grid(True); ax.set_xlabel("Time")
    for j in range(plotted_count, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# --- Symbolic Processing and Lambdification ---

# Add these functions to dynare_parser_sda_solver_jax.py

def parse_and_order_stationary_model_symbolic(model_string, verbose=True):
    """
    Parses the stationary model, handles leads/lags,
    orders variables/equations based on Dynare conventions.
    
    Args:
        model_string (str): The full content of the Dynare file.
        verbose (bool): If True, prints progress information.
        
    Returns:
        dict: Dictionary containing parsed and ordered model components.
    """
    if verbose:  print("--- Parsing Stationary Model Declarations ---")
    
    # Extract variables, shocks, and parameters
    declared_vars, shock_names, param_names_declared, param_assignments_initial = extract_declarations(model_string)
    
    # Process parameter definitions
    inferred_sigma_params = [f"sigma_{shk}" for shk in shock_names]
    stat_stderr_values = extract_stationary_shock_stderrs(model_string)
    
    # Combine parameter information
    combined_param_names = list(dict.fromkeys(param_names_declared).keys())
    for p_sigma in inferred_sigma_params:
        if p_sigma not in combined_param_names:
            combined_param_names.append(p_sigma)
    
    combined_param_assignments = stat_stderr_values.copy()
    combined_param_assignments.update(param_assignments_initial)
    
    # Set defaults for sigma parameters if not found
    for p_sigma in inferred_sigma_params:
        if p_sigma not in combined_param_assignments:
            combined_param_assignments[p_sigma] = 1.0
    
    param_names = combined_param_names
    param_assignments = combined_param_assignments
    
    if verbose:
        print(f"Declared Variables: {len(declared_vars)}, Shocks: {len(shock_names)}, Parameters: {len(param_names)}")
    
    # Extract and process equations
    raw_equations = extract_model_equations(model_string)
    if verbose: print(f"Found {len(raw_equations)} equations in model block.")
    
    # Handle leads/lags and auxiliary variables
    endogenous_vars = list(declared_vars)
    aux_variables = OrderedDict()
    processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    
    # Process each equation
    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]
        eq_idx += 1
        modified_eq = eq
        matches = list(var_time_regex.finditer(eq))
        
        for match in reversed(matches):
            base_name = match.group(1)
            time_shift = int(match.group(2))
            
            # Skip if not an endogenous variable or already processed aux
            if base_name not in endogenous_vars and base_name not in aux_variables:
                continue
            
            # Handle leads > 1
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        def_eq_str = f"({aux_name}) - ({prev_var_for_def}(+1))"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)
                
                # Replace with appropriate auxiliary variable
                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                
                # Add auxiliary definitions
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        processed_equations.append(def_eq)
            
            # Handle lags < -1
            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        def_eq_str = f"({aux_name}) - ({prev_var_for_def}(-1))"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)
                
                # Replace with appropriate auxiliary variable
                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                
                # Add auxiliary definitions
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        processed_equations.append(def_eq)
        
        # Update equation if modified
        if modified_eq != eq:
            processed_equations[eq_idx - 1] = modified_eq
    
    # Finalize variable list
    initial_vars_ordered = list(endogenous_vars)
    num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations)
    
    # Ensure model is square
    if num_vars != num_eq:
        raise ValueError(f"Stationary model not square after processing leads/lags: {num_vars} vars vs {num_eq} eqs.")
    
    # Create symbolic representation
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names}
    var_syms = {}
    
    for var in initial_vars_ordered:
        sym_m1 = create_timed_symbol(var, -1)
        sym_t = create_timed_symbol(var, 0)
        sym_p1 = create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
    
    # Parse equations into symbolic form
    local_dict = {
        str(s): s for s in list(param_syms.values()) + list(shock_syms.values()) + 
        [v for var_dict in var_syms.values() for v in var_dict.values()]
    }
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                         implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    
    # Parse and convert equations
    sym_equations = []
    for i, eq_str in enumerate(processed_equations):
        eq_str_sym = eq_str
        # Replace timed variables
        def replace_var_time(match):
            base_name, time_shift_str = match.groups()
            time_shift = int(time_shift_str)
            
            if base_name in shock_names:
                if time_shift == 0: return str(shock_syms[base_name])
                else: raise ValueError(f"Shock {base_name}({time_shift}) invalid in eq {i}: '{eq_str}'.")
            elif base_name in var_syms:
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0: return str(var_syms[base_name]['t'])
                if time_shift == 1: return str(var_syms[base_name]['p1'])
                raise ValueError(f"Unexpected time shift {time_shift} for variable {base_name} in eq {i}.")
            elif base_name in param_syms:
                raise ValueError(f"Parameter {base_name}({time_shift}) is invalid in eq {i}.")
            
            # Unknown symbol - likely an error
            return match.group(0)
            
        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)
        
        # Replace base names - sort by length to avoid partial matches
        all_base_names = sorted(list(var_syms.keys()) + list(param_syms.keys()) + list(shock_syms.keys()), key=len, reverse=True)
        for name in all_base_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name in var_syms:
                replacement = str(var_syms[name]['t'])  # Current time
            elif name in param_syms:
                replacement = str(param_syms[name])
            elif name in shock_syms:
                replacement = str(shock_syms[name])
            else:
                continue
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        
        # Parse symbolic expression
        try:
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            sym_equations.append(sym_eq)
        except Exception as e:
            print(f"\nError parsing equation {i}: '{eq_str}'")
            print(f"Processed form: '{eq_str_sym}'")
            raise
    
    # Generate symbolic matrices for A*P^2 + B*P + C = 0
    # A: coefficients of y(t+1), B: coefficients of y(t), C: coefficients of y(t-1)
    sympy_A_quad = sympy.zeros(num_eq, num_vars)
    sympy_B_quad = sympy.zeros(num_eq, num_vars)
    sympy_C_quad = sympy.zeros(num_eq, num_vars)
    sympy_D_quad = sympy.zeros(num_eq, len(shock_names))
    
    # Extract variable symbols for Jacobian
    var_p1_syms = [var_syms[v]['p1'] for v in initial_vars_ordered]
    var_t_syms = [var_syms[v]['t'] for v in initial_vars_ordered]
    var_m1_syms = [var_syms[v]['m1'] for v in initial_vars_ordered]
    shock_t_syms = [shock_syms[s] for s in shock_names]
    
    # Compute Jacobians
    for i, eq in enumerate(sym_equations):
        for j, var_p1 in enumerate(var_p1_syms):
            sympy_A_quad[i, j] = sympy.diff(eq, var_p1)
        for j, var_t in enumerate(var_t_syms):
            sympy_B_quad[i, j] = sympy.diff(eq, var_t)
        for j, var_m1 in enumerate(var_m1_syms):
            sympy_C_quad[i, j] = sympy.diff(eq, var_m1)
        for k, shk_t in enumerate(shock_t_syms):
            sympy_D_quad[i, k] = -sympy.diff(eq, shk_t)  # Note minus sign
    
    # Store initial matrices
    initial_info = {
        'A': sympy_A_quad.copy(),
        'B': sympy_B_quad.copy(),
        'C': sympy_C_quad.copy(),
        'D': sympy_D_quad.copy(),
        'vars': list(initial_vars_ordered),
        'eqs': list(processed_equations)
    }
    
    # --- Classify Variables ---
    # Key part based on Model_reduction.pdf guidance
    if verbose: print("\n--- Classifying Variables for Ordering (Following Dynare Convention) ---")
    
    # 1. First identify potential backward looking variables
    potential_backward = [v for v in initial_vars_ordered if v.startswith("RES_") or 
                         (v.startswith("aux_") and "_lag_" in v)]
    
    backward_exo_vars = []
    forward_backward_endo_vars = []
    static_endo_vars = []
    
    # 2. Check if these potential backward variables are truly backward
    for var in potential_backward:
        j = initial_vars_ordered.index(var)
        # Check if it has lead dependency (appears in A = dF/dy(t+1))
        has_lead = not sympy_A_quad.col(j).is_zero_matrix
        if has_lead:
            forward_backward_endo_vars.append(var)
        else:
            backward_exo_vars.append(var)
    
    # 3. For remaining variables, check lead/lag dependencies
    remaining_vars = [v for v in initial_vars_ordered if v not in backward_exo_vars + forward_backward_endo_vars]
    for var in remaining_vars:
        j = initial_vars_ordered.index(var)
        has_lag = not sympy_C_quad.col(j).is_zero_matrix
        has_lead = not sympy_A_quad.col(j).is_zero_matrix
        
        if has_lead or has_lag:
            forward_backward_endo_vars.append(var)
        else:
            static_endo_vars.append(var)
    
    # 4. Final ordering: backward exo, then forward/backward, then static
    ordered_vars = backward_exo_vars + forward_backward_endo_vars + static_endo_vars
    
    # 5. Create permutation indices
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]
    
    # --- Determine Equation Order ---
    eq_perm_indices = []
    used_eq_indices = set()
    
    # Compile regex patterns for auxiliary/RES definitions
    aux_def_patterns = {
        name: re.compile(fr"^\s*\({name}\)\s*-\s*\({base_name_from_aux(name)}\s*\(\s*-1\s*\)\)$", re.IGNORECASE)
        for name in aux_variables if "_lag_" in name
    }
    
    res_def_patterns = {
        name: re.compile(fr"^\s*\({name}\)\s*-\s*.*\({name}\s*\(\s*-1\s*\)\).*", re.IGNORECASE)
        for name in initial_vars_ordered if name.startswith("RES_")
    }
    
    # Assign defining equations for backward variables first
    for var in backward_exo_vars:
        pattern = None
        if var.startswith("aux_") and "_lag_" in var:
            pattern = aux_def_patterns.get(var)
        elif var.startswith("RES_"):
            pattern = res_def_patterns.get(var)
            
        if pattern:
            for i, eq_str in enumerate(processed_equations):
                if i in used_eq_indices:
                    continue
                    
                if pattern.match(eq_str):
                    eq_perm_indices.append(i)
                    used_eq_indices.add(i)
                    break
    
    # Assign equations for forward/backward and static variables
    for var in forward_backward_endo_vars + static_endo_vars:
        for i, eq_str in enumerate(processed_equations):
            if i in used_eq_indices:
                continue
                
            if eq_str.strip().startswith(f"({var})"):
                eq_perm_indices.append(i)
                used_eq_indices.add(i)
                break
    
    # Add any remaining unassigned equations
    remaining_eq_indices = [i for i in range(num_eq) if i not in used_eq_indices]
    eq_perm_indices.extend(remaining_eq_indices)
    
    # Validate permutation indices
    if len(eq_perm_indices) != num_eq:
        raise ValueError(f"Equation permutation construction failed. Expected {num_eq} indices, got {len(eq_perm_indices)}.")
    
    if len(set(eq_perm_indices)) != num_eq:
        raise ValueError("Equation permutation construction failed. Indices not unique.")
    
    # --- Reorder Symbolic Matrices ---
    sympy_A_ord = sympy_A_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_B_ord = sympy_B_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_D_ord = sympy_D_quad.extract(eq_perm_indices, list(range(len(shock_names))))
    
    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}
    
    return {
        "equations_processed": processed_equations,
        "var_names_initial_order": initial_vars_ordered,
        "ordered_vars_final": ordered_vars,
        "shock_names": shock_names,
        "param_names_all": param_names,
        "param_assignments_default": param_assignments,
        "var_permutation_indices": var_perm_indices,
        "eq_permutation_indices": eq_perm_indices,
    }

def create_timed_symbol(base_name, time_shift):
    """Creates a SymPy symbol with appropriate time suffix."""
    suffix_map = {-1: "_m1", 0: "", 1: "_p1"}
    suffix = suffix_map.get(time_shift, f"_t{time_shift:+}")
    return sympy.symbols(f"{base_name}{suffix}")

def base_name_from_aux(aux_name):
    """Extracts base variable name from auxiliary variable name."""
    match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p\d+", aux_name)
    if match_lead:
        return match_lead.group(1)
    match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", aux_name)
    if match_lag:
        return match_lag.group(1)
    return aux_name

def _create_timed_symbol(base_name: str, time_shift: int, suffix_map: Dict[int, str] = {-1: "_m1", 0: "", 1: "_p1"}):
    """Creates a SymPy symbol with a time suffix."""
    return sympy.symbols(f"{base_name}{suffix_map.get(time_shift, f'_t{time_shift:+}')}")

def _base_name_from_aux(aux_name: str) -> str:
    """Extracts base variable name from an auxiliary variable name."""
    match = re.match(r"aux_([a-zA-Z_]\w*)_(?:lead_p|lag_m)\d+", aux_name)
    return match.group(1) if match else aux_name

def _robust_lambdify(args: List[sympy.Symbol], expr: Any, modules = None) -> Callable:
    """Lambdifies a SymPy expression, handling constants and ensuring JAX compatibility."""
    if modules is None:
        # Pass the actual jnp module object
        modules_list = [
            {'ImmutableDenseMatrix': jnp.array, 'MutableDenseMatrix': jnp.array}, # For SymPy Matrix -> jnp.array
            jnp,  # Pass the jax.numpy module object directly
            "math" # Can also include standard math for functions like exp, log if not covered by jnp
        ]
        # If you need functions directly from 'jax' itself (less common for numpy operations)
        # you could add 'jax' as a string: modules_list.append('jax')
    else:
        # If modules are explicitly passed, ensure they are in the correct format
        if isinstance(modules, str):
            modules_list = [modules]
        elif isinstance(modules, list) or isinstance(modules, tuple):
            modules_list = list(modules)
        else:
            raise TypeError(f"Unsupported type for modules: {type(modules)}")


    if not isinstance(expr, sympy.Expr) and not isinstance(expr, (sympy.MatrixBase)):
        def const_func(*params_tuple):
            return jnp.asarray(expr, dtype=_DEFAULT_DTYPE)
        return const_func

    if isinstance(expr, sympy.Expr) and not expr.free_symbols:
        try:
            const_val = expr.evalf()
            def const_expr_func(*params_tuple):
                return jnp.asarray(const_val, dtype=_DEFAULT_DTYPE)
            return const_expr_func
        except Exception:
            pass

    # Standard lambdify
    try:
        # Use the prepared modules_list
        return sympy.lambdify(args, expr, modules=modules_list)
    except Exception as e:
        # print(f"Error during robust_lambdify for expr: {expr} with args: {args}")
        # print(f"Using modules: {modules_list}")
        raise e

def generate_matrix_lambda_functions(
    equations_str: List[str],
    var_names_ordered: List[str],      # Base names of variables for this block
    shock_names_ordered: List[str],    # Base names of shocks for this block
    all_param_names_ordered: List[str],# Ordered list of ALL model parameter names
    model_type: str,                   # "stationary", "trend", or "observation"
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Generates lambdified JAX functions for matrix elements and their gradients
    with respect to parameters.

    Args:
        equations_str: List of equation strings ("LHS - RHS = 0").
        var_names_ordered: Base names of variables (e.g., Y, C for y_t, y_m1, y_p1).
                           For "observation", these are the obs_vars.
        shock_names_ordered: Base names of shocks. For "observation", these are
                             the augmented state variables.
        all_param_names_ordered: Global list of parameter names.
        model_type: Type of model equations.
        verbose: Print progress.

    Returns:
        Dictionary containing:
            - 'lambda_matrices': {
                'A': {'elements': [[fn_A00(params), ...]], 'grads': [[[fn_dA00dpk(params), ...]]]}, ...
              }
            - 'param_symbols': List of SymPy parameter symbols used (matches all_param_names_ordered).
            - 'var_symbols_timed': Dict mapping var_name to {'t': sym, 'm1': sym, 'p1': sym} (for stationary)
                                   or similar for other model types.
            - 'shock_symbols_timed': Dict mapping shock_name to {'t': sym}
    """
    if verbose: print(f"--- SymPy: Generating Lambdas for {model_type.upper()} ---")

    # 1. Create SymPy symbols
    param_symbols = [sympy.symbols(p) for p in all_param_names_ordered]
    param_map = {name: sym for name, sym in zip(all_param_names_ordered, param_symbols)}

    # Symbols for variables and shocks with time context
    var_symbols_timed = OrderedDict() # E.g. var_symbols_timed['Y']['t'] = Y_symbol
    shock_symbols_timed = OrderedDict() # E.g. shock_symbols_timed['SHK_X']['t'] = SHK_X_symbol

    # Define which time shifts are relevant for each model_type
    # This defines what d(resid)/d(var_at_time_X) we will compute
    # y_t: current value of var_names_ordered
    # y_m1: lagged value of var_names_ordered
    # y_p1: lead value of var_names_ordered
    # shocks_t: current value of shock_names_ordered (or state_t for obs model)

    # For parsing equations, create a flat mapping of "NAME(TIMESHIFT)" -> symbol_string
    parsing_symbol_map = {p: str(sym) for p, sym in param_map.items()} # Parameters are time-invariant

    if model_type == "stationary":
        for v_name in var_names_ordered:
            var_symbols_timed[v_name] = {
                'm1': _create_timed_symbol(v_name, -1),
                't': _create_timed_symbol(v_name, 0),
                'p1': _create_timed_symbol(v_name, 1)
            }
            parsing_symbol_map[f"{v_name}(-1)"] = str(var_symbols_timed[v_name]['m1'])
            parsing_symbol_map[f"{v_name}(0)"] = str(var_symbols_timed[v_name]['t'])
            parsing_symbol_map[f"{v_name}"] = str(var_symbols_timed[v_name]['t']) # Implicit current time
            parsing_symbol_map[f"{v_name}(+1)"] = str(var_symbols_timed[v_name]['p1'])
        for s_name in shock_names_ordered:
            shock_symbols_timed[s_name] = {'t': _create_timed_symbol(s_name, 0, suffix_map={0: "_shk"})} # Avoid clash
            parsing_symbol_map[f"{s_name}(0)"] = str(shock_symbols_timed[s_name]['t'])
            parsing_symbol_map[f"{s_name}"] = str(shock_symbols_timed[s_name]['t'])
    elif model_type == "trend":
        # Trend: y_t = P * y_{t-1} + Q * shocks_t
        for v_name in var_names_ordered: # These are the state_trend_vars
            var_symbols_timed[v_name] = {
                'm1': _create_timed_symbol(v_name, -1),
                't': _create_timed_symbol(v_name, 0) # LHS of definition
            }
            parsing_symbol_map[f"{v_name}(-1)"] = str(var_symbols_timed[v_name]['m1'])
            parsing_symbol_map[f"{v_name}"] = str(var_symbols_timed[v_name]['t'])
        for s_name in shock_names_ordered: # These are trend_shocks
            shock_symbols_timed[s_name] = {'t': _create_timed_symbol(s_name, 0, suffix_map={0: "_shk_tr"})}
            parsing_symbol_map[f"{s_name}"] = str(shock_symbols_timed[s_name]['t'])
    elif model_type == "observation":
        # Obs: obs_t = Omega * state_t
        for v_name in var_names_ordered: # These are obs_vars
            var_symbols_timed[v_name] = {'t': _create_timed_symbol(v_name, 0, suffix_map={0:"_obs"})} # LHS
            parsing_symbol_map[f"{v_name}"] = str(var_symbols_timed[v_name]['t'])
        for s_name in shock_names_ordered: # These are augmented_state_vars
            shock_symbols_timed[s_name] = {'t': _create_timed_symbol(s_name, 0, suffix_map={0:"_augst"})} # RHS
            parsing_symbol_map[f"{s_name}"] = str(shock_symbols_timed[s_name]['t']) # State vars appear at current time
    else:
        raise ValueError(f"Unknown model_type for symbol generation: {model_type}")

    # Create the local_dict for sympy.parse_expr
    local_parsing_dict = {str(sym): sym for sym_list_dict in var_symbols_timed.values() for sym in sym_list_dict.values()}
    local_parsing_dict.update({str(sym): sym for sym_list_dict in shock_symbols_timed.values() for sym in sym_list_dict.values()})
    local_parsing_dict.update(param_map)
    local_parsing_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})


    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                          implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    var_time_regex_parser = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)') # For VAR(k)
    base_name_regex_parser = re.compile(r'\b([a-zA-Z_]\w*)\b') # For VAR or PARAM


    sym_residuals = []
    if verbose: print("  Parsing equations to SymPy expressions...")
    for i, eq_str_orig in enumerate(equations_str):
        eq_str_mod = eq_str_orig
        # Replace VAR(k) with their symbol strings from parsing_symbol_map
        def replace_timed(match):
            base, shift = match.group(1), int(match.group(2))
            key = f"{base}({shift:+})" if shift !=0 else base # construct VAR(+1) or VAR
            if key in parsing_symbol_map: return parsing_symbol_map[key]
            if base in parsing_symbol_map: return parsing_symbol_map[base] # for implicit VAR
            # print(f"Warning (eq {i}): Timed var '{match.group(0)}' not in parsing_symbol_map for {model_type}. Keys: {list(parsing_symbol_map.keys())[:10]}")
            return match.group(0) # Fallback
        eq_str_mod = var_time_regex_parser.sub(replace_timed, eq_str_mod)

        # Replace remaining base names (implicit current time vars, params)
        # Sort by length to avoid partial replacement (e.g. "C" before "CPI")
        sorted_simple_names = sorted([k for k in parsing_symbol_map.keys() if '(' not in k], key=len, reverse=True)
        for name in sorted_simple_names:
            eq_str_mod = re.sub(r'\b' + re.escape(name) + r'\b', parsing_symbol_map[name], eq_str_mod)

        try:
            # Check for lingering unparsed symbols before sympify
            # remaining_names = set(re.findall(r'\b([a-zA-Z_]\w*(?!\s*\())((?<!\()\w*)\b', eq_str_mod)) # find VAR not VAR(
            remaining_names = set(re.findall(r'[a-zA-Z_]\w*(?!\s*\()', eq_str_mod))

            parsed_sym_names = set(local_parsing_dict.keys()) | {'log', 'exp', 'sqrt', 'abs'}
            problematic_symbols = []
            for rem_name_match in remaining_names:
                rem_name = rem_name_match # if isinstance(rem_name_match, tuple) else rem_name_match
                is_number = False
                try:
                    float(rem_name); is_number = True
                except ValueError: pass
                if not is_number and rem_name not in parsed_sym_names and not rem_name.endswith(("_m1", "_p1", "_shk", "_shk_tr", "_obs", "_augst")):
                    problematic_symbols.append(rem_name)
            if problematic_symbols and verbose:
                 pass
                 # print(f"    Notice (eq {i}): Potentially unparsed symbols in '{eq_str_mod}': {problematic_symbols}. Will try sympify.")

            sym_eq = parse_expr(eq_str_mod, local_dict=local_parsing_dict, transformations=transformations)
            sym_residuals.append(sym_eq)
        except Exception as e:
            # print(f"Error parsing equation {i} for {model_type}: '{eq_str_orig}'")
            # print(f"  Modified string for SymPy: '{eq_str_mod}'")
            # print(f"  Local parsing dict keys: {list(local_parsing_dict.keys())[:20]}...") # Print some keys
            raise ValueError(f"SymPy parsing failed for eq {i} ({model_type}): {e}")

    # 3. Compute Symbolic Jacobians (defining matrix elements)
    # These are symbolic expressions in terms of param_symbols
    # Example for Stationary: A_quad_ij = d(resid_i)/d(y_j(+1)), D_quad_ij = -d(resid_i)/d(shocks_j(t))
    symbolic_matrices = {} # Keyed by matrix name (e.g., 'A', 'P_trends')

    if model_type == "stationary":
        # jac_A_quad = d(resid) / d(y_p1)
        # jac_B_quad = d(resid) / d(y_t)
        # jac_C_quad = d(resid) / d(y_m1)
        # jac_D_quad_neg = d(resid) / d(shocks_t) -> D = -jac_D_quad_neg
        vars_p1_syms = [var_symbols_timed[v]['p1'] for v in var_names_ordered]
        vars_t_syms  = [var_symbols_timed[v]['t']  for v in var_names_ordered]
        vars_m1_syms = [var_symbols_timed[v]['m1'] for v in var_names_ordered]
        shocks_t_syms= [shock_symbols_timed[s]['t'] for s in shock_names_ordered]

        symbolic_matrices['A'] = sympy.Matrix(sym_residuals).jacobian(vars_p1_syms)
        symbolic_matrices['B'] = sympy.Matrix(sym_residuals).jacobian(vars_t_syms)
        symbolic_matrices['C'] = sympy.Matrix(sym_residuals).jacobian(vars_m1_syms)
        symbolic_matrices['D'] = -sympy.Matrix(sym_residuals).jacobian(shocks_t_syms) # Note negation
    elif model_type == "trend":
        # P_trends_ij = -d(resid_i)/d(y_j(-1)) (where resid = y_t - P*y_m1 - Q*shocks)
        # Q_trends_ij = -d(resid_i)/d(shocks_j(t))
        vars_m1_syms = [var_symbols_timed[v]['m1'] for v in var_names_ordered] # state_trend_vars(-1)
        shocks_t_syms= [shock_symbols_timed[s]['t'] for s in shock_names_ordered] # trend_shocks(t)
        symbolic_matrices['P_trends'] = -sympy.Matrix(sym_residuals).jacobian(vars_m1_syms)
        symbolic_matrices['Q_trends'] = -sympy.Matrix(sym_residuals).jacobian(shocks_t_syms)
    elif model_type == "observation":
        # Omega_ij = -d(resid_i)/d(state_j(t)) (where resid = obs_t - Omega*state_t)
        state_t_syms = [shock_symbols_timed[s]['t'] for s in shock_names_ordered] # augmented_state_vars(t)
        symbolic_matrices['Omega'] = -sympy.Matrix(sym_residuals).jacobian(state_t_syms)
    else:
        raise ValueError(f"Unknown model_type for symbolic Jacobians: {model_type}")

    # 4. Lambdify: Convert symbolic matrix elements and their gradients to JAX functions
    lambdas_dict_for_model = {} # E.g., {'A': {'elements': [[fn,...]], 'grads': [[[fn_dpk,...]]]}, ...}
    if verbose: print(f"  Lambdifying matrix elements and their gradients for {model_type}...")

    for mat_name, sym_matrix_expr in symbolic_matrices.items():
        num_rows, num_cols = sym_matrix_expr.shape
        element_fns_mat = [[None]*num_cols for _ in range(num_rows)]
        grad_fns_mat = [[None]*num_cols for _ in range(num_rows)]

        for r in range(num_rows):
            for c in range(num_cols):
                element_expr = sym_matrix_expr[r, c]
                element_fns_mat[r][c] = _robust_lambdify(param_symbols, element_expr)

                grads_for_element = [] # List of d(M_rc)/d(param_k) functions
                for p_sym in param_symbols:
                    grad_expr = sympy.diff(element_expr, p_sym)
                    grads_for_element.append(_robust_lambdify(param_symbols, grad_expr))
                grad_fns_mat[r][c] = grads_for_element
        lambdas_dict_for_model[mat_name] = {'elements': element_fns_mat, 'grads': grad_fns_mat}
        if verbose: print(f"    Processed matrix {mat_name} ({num_rows}x{num_cols})")

    if verbose: print(f"--- SymPy: Lambda generation for {model_type.upper()} COMPLETE ---")
    return {
        'lambda_matrices': lambdas_dict_for_model,
        'param_symbols': param_symbols, # For consistent ordering
        'var_symbols_timed': var_symbols_timed, # For inspection/debugging
        'shock_symbols_timed': shock_symbols_timed # For inspection/debugging
    }

def extract_stationary_shock_stderrs(model_string):
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\bshocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: return {}
    content = match.group(1)
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)
    parsed_count = 0
    for m in stderr_matches:
        shock_name = m.group(1); stderr_val_str = m.group(2).strip()
        try: stderr_val = float(stderr_val_str); param_name = f"sigma_{shock_name}"; stderrs[param_name] = stderr_val; parsed_count += 1
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for stat shock '{shock_name}'.")
    # if parsed_count > 0: print(f"   Parsed {parsed_count} stderr definitions from 'shocks;' block.") # Less verbose
    return stderrs

def extract_declarations(model_string):
    declarations = {'var': [], 'varexo': [], 'parameters': [] }
    param_assignments = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    model_marker = re.search(r'\bmodel\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    content_to_search = processed_content[:model_marker.start()] if model_marker else processed_content
    param_assignment_content_search_area = content_to_search
    block_matches = re.finditer(r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters|model)\b|$)', content_to_search, re.DOTALL | re.IGNORECASE)
    def process_block_content(content_str, block_type):
        if not content_str: return []
        content = content_str.strip(); first_semicolon_match = re.search(r';', content)
        if first_semicolon_match: content = content[:first_semicolon_match.start()].strip()
        content = content.replace('\n', ' '); names = []
        raw_names = re.split(r'[,\s]+', content)
        cleaned_names = [name for name in raw_names if name and re.fullmatch(r'[a-zA-Z_]\w*', name)]
        keywords = {'var', 'varexo', 'parameters', 'model', 'end'}; names = [n for n in cleaned_names if n not in keywords]
        return list(dict.fromkeys(names).keys())
    for match in block_matches:
        block_keyword = match.group(1).lower(); block_content_raw = match.group(2)
        extracted_names = process_block_content(block_content_raw, block_keyword)
        declarations[block_keyword].extend(extracted_names)
    final_declarations = {key: list(dict.fromkeys(lst).keys()) for key, lst in declarations.items()}
    assignment_matches = re.finditer(r'\b([a-zA-Z_]\w*)\b\s*=\s*([^;]+);', param_assignment_content_search_area)
    parameter_names_declared = final_declarations.get('parameters', [])
    for match in assignment_matches:
        name = match.group(1); value_str = match.group(2).strip()
        if name in parameter_names_declared:
            try: param_assignments[name] = float(value_str)
            except ValueError: print(f"Warning: Could not parse value '{value_str}' for param '{name}'.")
    return (final_declarations.get('var', []), final_declarations.get('varexo', []),
            parameter_names_declared, param_assignments)

def extract_model_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    model_match = re.search(r'(?i)\bmodel\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not model_match: raise ValueError("Could not find 'model; ... end;' block.")
    model_content = model_match.group(1)
    equations_raw = [eq.strip() for eq in model_content.split(';') if eq.strip()]
    processed_equations = []
    for line in equations_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2: lhs, rhs = parts; processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else: print(f"Warning: Skipping malformed equation line: '{line}'")
        else: print(f"Warning: Equation '{line}' has no '='. Assuming 'expr = 0'."); processed_equations.append(line)
    return processed_equations

# --- Trend/Observation Parsing Functions (Mostly unchanged) ---

def extract_trend_declarations(model_string):
    trend_vars = []; trend_shocks = []
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    trend_model_marker = re.search(r'\btrend_model\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    search_area = processed_content[:trend_model_marker.start()] if trend_model_marker else processed_content
    match_tv = re.search(r'(?i)\btrends_vars\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match_tv: content = match_tv.group(1).replace('\n', ' ').strip(); trend_vars = [v for v in re.split(r'[,\s]+', content) if v and re.fullmatch(r'[a-zA-Z_]\w*', v)]; trend_vars = list(dict.fromkeys(trend_vars).keys())
    match_vt = re.search(r'(?i)\bvarexo_trends\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match_vt: content = match_vt.group(1).replace('\n', ' ').strip(); trend_shocks = [s for s in re.split(r'[,\s]+', content) if s and re.fullmatch(r'[a-zA-Z_]\w*', s)]; trend_shocks = list(dict.fromkeys(trend_shocks).keys())
    return trend_vars, trend_shocks

def extract_trend_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\btrend_model\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: print("Warning: 'trend_model; ... end;' block not found."); return []
    content = match.group(1); eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]
    processed_equations = []
    for line in eq_raw:
        if '=' in line:
            parts = line.split('=', 1); lhs, rhs = parts[0].strip(), parts[1].strip()
            lhs_match = re.match(r'([a-zA-Z_]\w*)\s*(\(\s*([+-]?\d+)\s*\))?', lhs)
            if lhs_match:
                base_lhs = lhs_match.group(1)
                if lhs_match.group(2) and lhs_match.group(3) != '0': print(f"Warning: Trend eq '{line}' has non-standard LHS. Assuming it defines '{base_lhs}(t)'."); processed_equations.append(f"({base_lhs}) - ({rhs})")
                else: processed_equations.append(f"({base_lhs}) - ({rhs})")
            else: print(f"Warning: Skipping malformed trend eq line: '{line}' - Cannot parse LHS")
        else: print(f"Warning: Trend eq '{line}' has no '='. Assuming 'expr = 0'."); processed_equations.append(line)
    return processed_equations

def extract_observation_declarations(model_string):
    obs_vars = []
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    meas_eq_marker = re.search(r'\bmeas(?:urement)?_equations\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    search_area = processed_content[:meas_eq_marker.start()] if meas_eq_marker else processed_content
    match = re.search(r'(?i)\bvarobs\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match: content = match.group(1).replace('\n', ' ').strip(); obs_vars = [v for v in re.split(r'[,\s]+', content) if v and re.fullmatch(r'[a-zA-Z_]\w*', v)]; obs_vars = list(dict.fromkeys(obs_vars).keys())
    return obs_vars

def extract_measurement_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\bmeas(?:urement)?_equations\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: print("Warning: 'measurement_equations; ... end;' block not found."); return []
    content = match.group(1); eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]
    processed_equations = []
    for line in eq_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2: lhs, rhs = parts; processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else: print(f"Warning: Skipping malformed measurement eq line: '{line}'")
        else: print(f"Warning: Measurement eq '{line}' has no '='. Assuming 'expr = 0'."); processed_equations.append(line)
    return processed_equations

def extract_trend_shock_stderrs(model_string):
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\btrend_shocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: return {}
    content = match.group(1)
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)
    for m in stderr_matches:
        shock_name = m.group(1); stderr_val_str = m.group(2).strip()
        try: stderr_val = float(stderr_val_str); param_name = f"sigma_{shock_name}"; stderrs[param_name] = stderr_val
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for trend shock '{shock_name}'.")
    return stderrs

def extract_model_shock_stderrs(model_string):
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\bshocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: return {}
    content = match.group(1)
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)
    for m in stderr_matches:
        shock_name = m.group(1); stderr_val_str = m.group(2).strip()
        try: stderr_val = float(stderr_val_str); param_name = f"sigma_{shock_name}"; stderrs[param_name] = stderr_val
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for trend shock '{shock_name}'.")
    return stderrs

# --- Stationary Model Parsing, Ordering (using SymPy for classification now) ---
def parse_and_order_stationary_model_symbolic(model_string: str, verbose: bool = False):
    """
    Parses the stationary model, handles leads/lags, orders variables/equations
    symbolically, and prepares info for lambda generation.
    """
    if verbose: print("--- Parsing & Ordering Stationary Model (Symbolic) ---")
    # 1. Basic Declarations
    declared_vars, shock_names, param_names_declared, param_assignments_initial = extract_declarations(model_string)
    inferred_sigma_params = [f"sigma_{shk}" for shk in shock_names]
    stat_stderr_values = extract_stationary_shock_stderrs(model_string)
    param_names = list(dict.fromkeys(param_names_declared + inferred_sigma_params))
    param_assignments = stat_stderr_values.copy()
    param_assignments.update(param_assignments_initial)
    for p_sigma in inferred_sigma_params:
        if p_sigma not in param_assignments: param_assignments[p_sigma] = 1.0

    # 2. Equations and Auxiliaries
    raw_equations = extract_model_equations(model_string)
    endogenous_vars = list(declared_vars); aux_variables = OrderedDict(); processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    eq_idx = 0
    while eq_idx < len(processed_equations): # Aux var handling loop (simplified for brevity)
        eq = processed_equations[eq_idx]; eq_idx += 1; modified_eq = eq
        matches = list(var_time_regex.finditer(eq))
        for match in reversed(matches):
            base_name, time_shift = match.group(1), int(match.group(2))
            if base_name not in endogenous_vars and base_name not in aux_variables: continue
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        def_eq = f"({aux_name}) - ({prev_var}(+1))"; aux_variables[aux_name] = def_eq; aux_needed_defs.append(def_eq)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"; replacement = f"{target_aux}(+1)"
                modified_eq = modified_eq[:match.start()] + replacement + modified_eq[match.end():]
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations and f"({def_eq.split('-')[0].strip('()')}) - ({def_eq.split('-')[1].strip('()')})" not in processed_equations:
                        processed_equations.append(def_eq)
            elif time_shift < -1: # Similar for lags
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        def_eq = f"({aux_name}) - ({prev_var}(-1))"; aux_variables[aux_name] = def_eq; aux_needed_defs.append(def_eq)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"; replacement = f"{target_aux}(-1)"
                modified_eq = modified_eq[:match.start()] + replacement + modified_eq[match.end():]
                for def_eq in aux_needed_defs:
                     if def_eq not in processed_equations and f"({def_eq.split('-')[0].strip('()')}) - ({def_eq.split('-')[1].strip('()')})" not in processed_equations:
                        processed_equations.append(def_eq)
        if modified_eq != eq: processed_equations[eq_idx - 1] = modified_eq
    
    initial_vars_ordered = list(dict.fromkeys(endogenous_vars)) # Unique vars before Dynare-specific ordering
    num_vars, num_eq = len(initial_vars_ordered), len(processed_equations)
    if num_vars != num_eq: raise ValueError(f"Model not square after aux: {num_vars} vars vs {num_eq} eqs.")

    # 3. Symbolic Representation for Classification (similar to generate_matrix_lambda_functions)
    # This part creates temporary SymPy expressions of the UNORDERED system
    # just to classify variables and determine permutation order.
    # The actual lambdas for VJP will be generated later on the potentially reordered system
    # or using the permutation indices.

    temp_param_syms = {p: sympy.symbols(p) for p in param_names}
    temp_shock_syms = {s: sympy.symbols(s) for s in shock_names}
    temp_var_syms = {}
    temp_local_dict = {str(s): s for s in list(temp_param_syms.values()) + list(temp_shock_syms.values())}
    temp_local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})

    temp_parsing_symbol_map = {p: str(sym) for p, sym in temp_param_syms.items()}
    for v_name in initial_vars_ordered:
        m1_s, t_s, p1_s = _create_timed_symbol(v_name, -1), _create_timed_symbol(v_name, 0), _create_timed_symbol(v_name, 1)
        temp_var_syms[v_name] = {'m1': m1_s, 't': t_s, 'p1': p1_s}
        temp_local_dict.update({str(m1_s):m1_s, str(t_s):t_s, str(p1_s):p1_s})
        temp_parsing_symbol_map.update({f"{v_name}(-1)":str(m1_s), f"{v_name}(0)":str(t_s), f"{v_name}":str(t_s), f"{v_name}(+1)":str(p1_s)})
    for s_name in shock_names:
        shk_s = _create_timed_symbol(s_name, 0, suffix_map={0:"_shk"})
        temp_var_syms[s_name] = {'t': shk_s} # Store shock symbols here too for jacobian
        temp_local_dict.update({str(shk_s):shk_s})
        temp_parsing_symbol_map.update({f"{s_name}(0)":str(shk_s), f"{s_name}":str(shk_s)})


    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application, rationalize)
    transformations_sym = (standard_transformations + (implicit_multiplication_application, rationalize))
    
    temp_sym_residuals = []
    for i, eq_str_orig in enumerate(processed_equations):
        eq_str_mod = eq_str_orig
        def temp_replace_timed(match):
            base, shift = match.group(1), int(match.group(2))
            key = f"{base}({shift:+})" if shift !=0 else base
            return temp_parsing_symbol_map.get(key, temp_parsing_symbol_map.get(base, match.group(0)))
        eq_str_mod = var_time_regex.sub(temp_replace_timed, eq_str_mod)
        sorted_simple_names_temp = sorted([k for k in temp_parsing_symbol_map.keys() if '(' not in k], key=len, reverse=True)
        for name_s in sorted_simple_names_temp:
            eq_str_mod = re.sub(r'\b' + re.escape(name_s) + r'\b', temp_parsing_symbol_map[name_s], eq_str_mod)
        try:
            temp_sym_residuals.append(parse_expr(eq_str_mod, local_dict=temp_local_dict, transformations=transformations_sym))
        except Exception as e_sym_parse:
            raise ValueError(f"SymPy parsing for classification failed (eq {i}): {e_sym_parse} on '{eq_str_mod}'")

    # Symbolic Jacobians for classification (UNORDERED)
    # A_quad_unord_ij = d(resid_i)/d(y_j(+1)), C_quad_unord_ij = d(resid_i)/d(y_j(-1))
    vars_p1_syms_temp = [temp_var_syms[v]['p1'] for v in initial_vars_ordered]
    vars_m1_syms_temp = [temp_var_syms[v]['m1'] for v in initial_vars_ordered]
    
    A_quad_unord_sym = sympy.Matrix(temp_sym_residuals).jacobian(vars_p1_syms_temp)
    C_quad_unord_sym = sympy.Matrix(temp_sym_residuals).jacobian(vars_m1_syms_temp)

    # 4. Classify Variables (Dynare-like ordering based on symbolic structure)
    # Similar to parser10's parse_and_compute_matrices_jax_ad classification
    pb_vars, mf_vars, af_vars, s_vars = [], [], [], []
    aux_lead_vars = [v for v in initial_vars_ordered if v.startswith("aux_") and "_lead_" in v]
    aux_lag_vars = [v for v in initial_vars_ordered if v.startswith("aux_") and "_lag_" in v]

    for idx, var_name in enumerate(initial_vars_ordered):
        has_lag_dep = not C_quad_unord_sym.col(idx).is_zero_matrix
        has_lead_dep = not A_quad_unord_sym.col(idx).is_zero_matrix

        if var_name in aux_lead_vars: af_vars.append(var_name)
        elif var_name in aux_lag_vars: pb_vars.append(var_name)
        elif var_name.startswith("RES_"): pb_vars.append(var_name) # Typically RES_ are backward
        else: # Original declared vars
            if has_lead_dep and not has_lag_dep: mf_vars.append(var_name) # Purely forward (treat as mixed)
            elif not has_lead_dep and has_lag_dep: pb_vars.append(var_name) # Purely backward (original)
            elif not has_lead_dep and not has_lag_dep: s_vars.append(var_name) # Static
            else: mf_vars.append(var_name) # Mixed

    ordered_vars = sorted(pb_vars) + sorted(mf_vars) + sorted(af_vars) + sorted(s_vars)
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars] # new_idx -> old_idx

    # 5. Equation Ordering (heuristic based on variable type)
    eq_perm_indices = list(range(num_eq)) # Placeholder - for now, assume equations align with initial_vars_ordered
                                       # Or adapt the more complex eq ordering from parser10 if needed
    # More robust eq ordering (simplified from parser10's logic)
    temp_used_eq_indices = set()
    temp_eq_perm_indices = []
    # Equations defining purely backward and aux_lag vars first
    for var_k in sorted(pb_vars): # Process PURELY backward first
        found_eq = False
        for i, eq_str in enumerate(processed_equations):
            if i in temp_used_eq_indices: continue
            # Check if eq_str starts with "(var_k) -"
            if re.match(fr"^\s*\(\s*{re.escape(var_k)}\s*\)\s*-", eq_str):
                temp_eq_perm_indices.append(i); temp_used_eq_indices.add(i); found_eq = True; break
        # if not found_eq and verbose: print(f"  Warning: No clear defining eq for PB var '{var_k}'")
    # Equations defining aux_fwd vars next
    for var_k in sorted(af_vars):
        found_eq = False
        for i, eq_str in enumerate(processed_equations):
            if i in temp_used_eq_indices: continue
            if re.match(fr"^\s*\(\s*{re.escape(var_k)}\s*\)\s*-", eq_str):
                temp_eq_perm_indices.append(i); temp_used_eq_indices.add(i); found_eq = True; break
        # if not found_eq and verbose: print(f"  Warning: No clear defining eq for AF var '{var_k}'")
    # Remaining equations
    for i in range(num_eq):
        if i not in temp_used_eq_indices: temp_eq_perm_indices.append(i)
    
    if len(temp_eq_perm_indices) == num_eq and len(set(temp_eq_perm_indices)) == num_eq:
        eq_perm_indices = temp_eq_perm_indices
    # else:
        # if verbose: print("  Warning: Simplified eq ordering failed, using initial order.")


    if verbose:
        print(f"  Ordered Vars ({len(ordered_vars)}): {ordered_vars if len(ordered_vars) < 10 else str(ordered_vars[:5])+'...'} ")
        print(f"  Var Permutation (new_idx->old_idx): {var_perm_indices if len(var_perm_indices) < 10 else str(var_perm_indices[:5])+'...'}")
        print(f"  Eq Permutation (new_row->old_row): {eq_perm_indices if len(eq_perm_indices) < 10 else str(eq_perm_indices[:5])+'...'}")

    return {
        "equations_processed": processed_equations,       # For lambda generation
        "var_names_initial_order": initial_vars_ordered, # For lambda generation (defines y_t, y_m1, y_p1 bases)
        "ordered_vars_final": ordered_vars,              # Final Dynare-ordered variable list
        "shock_names": shock_names,
        "param_names_all": param_names,
        "param_assignments_default": param_assignments,
        "var_permutation_indices": var_perm_indices,     # new_idx -> old_idx for var columns
        "eq_permutation_indices": eq_perm_indices,       # new_idx -> old_idx for eq rows
    }


# --- SDA Solver and Q Computation (JAX Versions - Unchanged from parser10) ---
SDAState = namedtuple("SDAState", ["Xk", "Yk", "Ek", "Fk", "k", "converged", "rel_diff", "is_valid"])
_SDA_JITTER = 1e-14

def solve_quadratic_matrix_equation_jax(A, B, C, initial_guess=None, tol=1e-12, max_iter=500, verbose=False):
    dtype, n = A.dtype, A.shape[0]
    A_jax, B_jax, C_jax = jnp.asarray(A, dtype=dtype), jnp.asarray(B, dtype=dtype), jnp.asarray(C, dtype=dtype)
    X_guess = jnp.asarray(initial_guess, dtype=dtype) if initial_guess is not None else jnp.zeros_like(A_jax)
    E_init, F_init = C_jax, A_jax
    Bbar = B_jax + A_jax @ X_guess; I = jnp.eye(n, dtype=dtype)
    Bbar_reg = Bbar + _SDA_JITTER * I
    E0, F0 = -jax.scipy.linalg.solve(Bbar_reg, E_init, assume_a='gen'), -jax.scipy.linalg.solve(Bbar_reg, F_init, assume_a='gen')
    initial_solve_valid = jnp.all(jnp.isfinite(E0)) & jnp.all(jnp.isfinite(F0))
    def sda_scan_body(state, _):
        Xk, Yk, Ek, Fk, k, prev_converged, prev_rel_diff, prev_is_valid = state
        M1, M2 = I - Yk @ Xk + _SDA_JITTER * I, I - Xk @ Yk + _SDA_JITTER * I
        E_new, F_new = Ek @ jax.scipy.linalg.solve(M1, Ek, assume_a='gen'), Fk @ jax.scipy.linalg.solve(M2, Fk, assume_a='gen')
        X_new, Y_new = Xk + Fk @ jax.scipy.linalg.solve(M2, Xk @ Ek, assume_a='gen'), Yk + Ek @ jax.scipy.linalg.solve(M1, Yk @ Fk, assume_a='gen')
        current_rel_diff = jnp.linalg.norm(X_new - Xk, ord='fro') / jnp.maximum(jnp.linalg.norm(X_new, ord='fro'), 1e-15)
        current_step_valid = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(Y_new)) & jnp.all(jnp.isfinite(E_new)) & jnp.all(jnp.isfinite(F_new)) & jnp.isfinite(current_rel_diff)
        converged_this_step, current_is_valid = current_step_valid & (current_rel_diff < tol), prev_is_valid & current_step_valid
        current_converged = prev_converged | converged_this_step
        keep = prev_is_valid & current_step_valid & (~prev_converged)
        X_next, Y_next, E_next, F_next = (jnp.where(keep, n, o) for n,o in [(X_new,Xk), (Y_new,Yk), (E_new,Ek), (F_new,Fk)])
        next_rel_diff, next_converged = jnp.where(keep, current_rel_diff, prev_rel_diff), jnp.where(keep, current_converged, prev_converged)
        return SDAState(X_next, Y_next, E_next, F_next, k + 1, next_converged, next_rel_diff, current_is_valid), None
    final_state, _ = lax.scan(sda_scan_body, SDAState(E0,F0,E0,F0,0,False,jnp.inf,initial_solve_valid), None, max_iter)
    X_sol_scan, converged_flag = final_state.Xk + X_guess, final_state.converged & final_state.is_valid
    res = A_jax @ (X_sol_scan@X_sol_scan) + B_jax@X_sol_scan + C_jax; res_norm = jnp.linalg.norm(res,'fro')
    terms_norm = jnp.linalg.norm(A_jax@X_sol_scan@X_sol_scan,'fro') + jnp.linalg.norm(B_jax@X_sol_scan,'fro') + jnp.linalg.norm(C_jax,'fro')
    res_ratio = res_norm / jnp.maximum(terms_norm, 1e-15)
    return jnp.where(converged_flag, X_sol_scan, jnp.full_like(X_sol_scan, jnp.nan)), final_state.k, res_ratio, converged_flag

def compute_Q_jax(A, B, D, P, dtype=None):
    eff_dtype = dtype or getattr(A, 'dtype', _DEFAULT_DTYPE)
    A_jax,B_jax,D_jax,P_jax = (jnp.asarray(x, dtype=eff_dtype) for x in [A,B,D,P])
    n, n_shock = A_jax.shape[0], (D_jax.shape[1] if D_jax.ndim == 2 and D_jax.shape[1] > 0 else 0)
    if n_shock == 0: return jnp.zeros((n,0), dtype=eff_dtype)
    APB_reg = A_jax@P_jax + B_jax + _SDA_JITTER*jnp.eye(n, dtype=eff_dtype)
    return jax.scipy.linalg.solve(APB_reg, D_jax.reshape(n, n_shock) if D_jax.ndim==1 and n_shock==1 else D_jax, assume_a='gen')


# --- Trend/Observation Parsing & Lambda Generation Wrappers ---
# These will call the core generate_matrix_lambda_functions
def generate_trend_lambda_functions(
    model_string: str, all_param_names_ordered: List[str], verbose: bool = False
):
    if verbose: print("--- SymPy: Generating Lambdas for Trend Model ---")
    trend_vars_decl, trend_shocks_decl = extract_trend_declarations(model_string)
    trend_equations_str = extract_trend_equations(model_string)
    if not trend_equations_str:
        if verbose: print("  No trend equations found. Skipping lambda generation for trends.")
        return {'lambda_matrices': {}, 'param_symbols': [sympy.symbols(p) for p in all_param_names_ordered],
                'state_trend_vars': [], 'contemporaneous_trend_defs': {}}

    # Identify state trends vs contemporaneous definitions (simplified)
    state_trend_vars, contemp_defs, defining_eqs_state_trends = [], {}, []
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    base_var_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')
    for eq_str in trend_equations_str:
        match_lhs = re.match(r"\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*-\s*\((.*)\)\s*", eq_str)
        if not match_lhs: continue
        lhs_var, rhs_str = match_lhs.groups()
        if lhs_var not in trend_vars_decl: continue
        has_contemp_rhs = False
        for rmatch in var_time_regex.finditer(rhs_str):
            base, shift = rmatch.group(1), int(rmatch.group(2))
            if base in trend_vars_decl and base != lhs_var and shift >= 0: has_contemp_rhs = True; break
        if not has_contemp_rhs:
            for sym in set(base_var_regex.findall(rhs_str)):
                 if sym in trend_vars_decl and sym != lhs_var and f"{sym}(-1)" not in rhs_str: has_contemp_rhs = True; break
        if has_contemp_rhs: contemp_defs[lhs_var] = rhs_str
        else:
            if lhs_var not in state_trend_vars: state_trend_vars.append(lhs_var)
            defining_eqs_state_trends.append(eq_str)
    if not state_trend_vars:
         if verbose: print("  No state trend variables (only contemporaneous). Skipping specific lambda generation.")
         return {'lambda_matrices': {}, 'param_symbols': [sympy.symbols(p) for p in all_param_names_ordered],
                 'state_trend_vars': [], 'contemporaneous_trend_defs': contemp_defs}

    trend_lambdas = generate_matrix_lambda_functions(
        defining_eqs_state_trends, state_trend_vars, trend_shocks_decl,
        all_param_names_ordered, "trend", verbose
    )
    trend_lambdas['state_trend_vars'] = state_trend_vars
    trend_lambdas['contemporaneous_trend_defs'] = contemp_defs
    return trend_lambdas

def generate_observation_lambda_functions(
    model_string: str, all_param_names_ordered: List[str],
    ordered_stationary_vars: List[str], # From stationary model part
    ordered_state_trend_vars: List[str],# From trend model part (state trends only)
    contemporaneous_trend_defs: Dict[str, str], # From trend model part
    verbose: bool = False
):
    if verbose: print("--- SymPy: Generating Lambdas for Observation Model ---")
    obs_vars_decl = extract_observation_declarations(model_string)
    measurement_equations_str = extract_measurement_equations(model_string)
    if not measurement_equations_str or not obs_vars_decl:
        if verbose: print("  No measurement equations or varobs found. Skipping lambda generation for observations.")
        return {'lambda_matrices': {}, 'param_symbols': [sympy.symbols(p) for p in all_param_names_ordered], 'ordered_obs_vars': []}

    augmented_state_vars = ordered_stationary_vars + ordered_state_trend_vars
    
    # Substitute contemporaneous trend definitions into measurement equations
    processed_meas_eqs = []
    for eq_str in measurement_equations_str:
        match_lhs = re.match(r"\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*-\s*\((.*)\)\s*", eq_str)
        if not match_lhs: continue
        lhs_obs, rhs_str_orig = match_lhs.groups(); rhs_processed = rhs_str_orig
        for _iter in range(len(contemporaneous_trend_defs) + 1): # Max iterations for substitution
            made_sub = False
            for contemp_var, contemp_expr in contemporaneous_trend_defs.items():
                pattern = r'\b' + re.escape(contemp_var) + r'\b'
                if re.search(pattern, rhs_processed):
                    rhs_processed = re.sub(pattern, f"({contemp_expr})", rhs_processed); made_sub = True
            if not made_sub: break
        processed_meas_eqs.append(f"({lhs_obs}) - ({rhs_processed})")

    obs_lambdas = generate_matrix_lambda_functions(
        processed_meas_eqs, obs_vars_decl, augmented_state_vars,
        all_param_names_ordered, "observation", verbose
    )
    obs_lambdas['ordered_obs_vars'] = obs_vars_decl # Assuming parser returns them in order
    return obs_lambdas


# --- IRF Calculation Functions (JAX - Unchanged from parser10) ---
def irf(P, R, shock_index, horizon=40):
    P_jax, R_jax = jnp.asarray(P), jnp.asarray(R)
    n, n_shock = P_jax.shape[0], (R_jax.shape[1] if R_jax.ndim==2 and R_jax.size>0 else 0)
    if n_shock == 0: return jnp.zeros((horizon, n), dtype=P_jax.dtype)
    if not (0 <= shock_index < n_shock): raise ValueError(f"Shock index {shock_index} out of range for {n_shock} shocks.")
    y_resp, y_current = jnp.zeros((horizon,n),dtype=P_jax.dtype), R_jax[:,shock_index]
    y_resp = y_resp.at[0,:].set(y_current)
    if horizon > 1:
        _, y_scan = lax.scan(lambda y_prev, _: (P_jax@y_prev, P_jax@y_prev), y_current, None, horizon-1)
        y_resp = y_resp.at[1:,:].set(y_scan)
    return jnp.where(jnp.abs(y_resp) < 1e-14, 0.0, y_resp)

def irf_observables(P_aug, R_aug, Omega, shock_index, horizon=40):
    P_aug_jax, R_aug_jax, Omega_jax = jnp.asarray(P_aug), jnp.asarray(R_aug), jnp.asarray(Omega)
    n_aug, n_obs = P_aug_jax.shape[0], Omega_jax.shape[0]
    n_aug_shock = R_aug_jax.shape[1] if R_aug_jax.ndim==2 and R_aug_jax.size>0 else 0
    if n_aug_shock == 0: return jnp.zeros((horizon,n_obs),dtype=P_aug_jax.dtype)
    if not (0 <= shock_index < n_aug_shock): raise ValueError(f"Aug shock index {shock_index} out of range for {n_aug_shock} shocks.")
    if Omega_jax.shape[1]!=n_aug: raise ValueError(f"Omega cols ({Omega_jax.shape[1]})!=P_aug dim ({n_aug}).")
    if n_obs==0: return jnp.zeros((horizon,0),dtype=P_aug_jax.dtype)
    state_irf_vals = irf(P_aug_jax, R_aug_jax, shock_index, horizon)
    obs_irf_vals = state_irf_vals @ (Omega_jax.T if Omega_jax.ndim > 1 else Omega_jax.reshape(-1,1))
    return jnp.where(jnp.abs(obs_irf_vals) < 1e-14, 0.0, obs_irf_vals)

def construct_initial_state(
    n_aug: int, n_stat: int, aug_state_vars: List[str], key_init: jax.random.PRNGKey,
    initial_state_config: Optional[Dict[str, Dict[str, float]]] = None,
    default_trend_std: float = 0.01, dtype: jnp.dtype = _DEFAULT_DTYPE
) -> jax.Array:
    s0 = jnp.zeros(n_aug, dtype=dtype)
    n_trend = n_aug - n_stat
    if n_trend > 0:
        trend_means = jnp.zeros(n_trend, dtype=dtype)
        trend_stds = jnp.full(n_trend, default_trend_std, dtype=dtype)
        if initial_state_config:
            trend_state_names = aug_state_vars[n_stat:]
            for var_name, config in initial_state_config.items():
                if var_name in trend_state_names:
                    try:
                        idx = trend_state_names.index(var_name)
                        if "mean" in config: trend_means = trend_means.at[idx].set(float(config["mean"]))
                        if "std" in config: trend_stds = trend_stds.at[idx].set(jnp.maximum(0.0, float(config["std"])))
                    except (ValueError, TypeError, KeyError): pass # print(f"Warning processing init_state_config for {var_name}")
        s0 = s0.at[n_stat:].set(trend_means + trend_stds * jax.random.normal(key_init, (n_trend,), dtype=dtype))
    return s0

# --- Main Execution Guard (Example for testing this file) ---
if __name__ == "__main__":
    print("--- Dynare Parser (Symbolic Lambdas) & Solver Script ---")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn") # Ensure this example file exists
        if not os.path.exists(mod_file_path):
            raise FileNotFoundError(f"Example model file not found: {mod_file_path}")
        with open(mod_file_path, 'r') as f: model_def_content = f.read()

        # 1. Parse and Order Stationary Model Structure
        stationary_structure = parse_and_order_stationary_model_symbolic(model_def_content, verbose=True)
        print(f"\nStationary Model Structure Processed.")
        print(f"  Ordered Stationary Vars: {stationary_structure['ordered_vars_final'][:5]}...")
        print(f"  All Params: {stationary_structure['param_names_all'][:5]}...")

        # 2. Generate Lambdas for Stationary Part
        stationary_lambdas_info = generate_matrix_lambda_functions(
            equations_str=stationary_structure['equations_processed'],
            var_names_ordered=stationary_structure['var_names_initial_order'], # Use initial order for y_t, y_m1, y_p1 base
            shock_names_ordered=stationary_structure['shock_names'],
            all_param_names_ordered=stationary_structure['param_names_all'],
            model_type="stationary",
            verbose=True
        )
        print(f"Stationary Lambdas Generated. Example: A[0,0] is a function.")
        # Test a lambda function call
        test_param_vals_ordered = [stationary_structure['param_assignments_default'].get(p, 1.0)
                                   for p in stationary_structure['param_names_all']]
        if stationary_lambdas_info['lambda_matrices'].get('A'):
            A00_val = stationary_lambdas_info['lambda_matrices']['A']['elements'][0][0](*test_param_vals_ordered)
            # print(f"  Test: A[0,0] evaluated at default params: {A00_val}")
            dA00dp0_val = stationary_lambdas_info['lambda_matrices']['A']['grads'][0][0][0](*test_param_vals_ordered)
            # print(f"  Test: dA[0,0]/dparam0 evaluated: {dA00dp0_val}")


        # 3. Generate Lambdas for Trend Part
        trend_lambdas_info = generate_trend_lambda_functions(
            model_def_content, stationary_structure['param_names_all'], verbose=True
        )
        print(f"Trend Lambdas Generated. State trends: {trend_lambdas_info.get('state_trend_vars')}")

        # 4. Generate Lambdas for Observation Part
        obs_lambdas_info = generate_observation_lambda_functions(
            model_def_content, stationary_structure['param_names_all'],
            stationary_structure['ordered_vars_final'], # Use final ordered stat vars for Omega
            trend_lambdas_info.get('state_trend_vars', []),
            trend_lambdas_info.get('contemporaneous_trend_defs', {}),
            verbose=True
        )
        print(f"Observation Lambdas Generated. Ordered obs vars: {obs_lambdas_info.get('ordered_obs_vars')}")

        print("\n--- Symbolic parsing and lambda generation test complete ---")

        # Further steps (numerical evaluation, solving, IRFs) would be similar to the
        # original __main__ block but using these lambdas within a DynareModel class
        # that implements custom VJPs.

    except Exception as e:
        print(f"\n--- ERROR in __main__ ---")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# --- END OF MODIFIED FILE dynare_parser_sda_solver_jax.py ---