
# --- START OF FILE Dynare_parser_sda_solver.py ---

# -*- coding: utf-8 -*-
"""
Enhanced Dynare Parser and State-Space Solver (JAX Compatible)

Parses Dynare-like models, solves using JAX-based SDA, builds augmented
state-space, computes IRFs. Includes JAX AD matrix computation.
"""

import re
import sympy # Keep sympy ONLY for parsing expressions if needed, but try to avoid
import numpy as np
from collections import OrderedDict, namedtuple
import copy
import os
from numpy.linalg import norm # Still use numpy norm for final check in non-JAX solver if needed
from scipy.linalg import lu_factor, lu_solve, block_diag # block_diag used in non-JAX build_augmented
import matplotlib.pyplot as plt
import warnings # To manage warnings from exec

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.typing import ArrayLike
from typing import Tuple, Optional, List, Dict, Any, Callable
from jax import lax
import inspect  #this is for debugging
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
print(f"Using JAX with dtype: {_DEFAULT_DTYPE}")


# --- Helper Functions (plot_irfs, simulate_ssm_data, etc. - UNCHANGED) ---
# [Your existing helper functions: plot_irfs, simulate_ssm_data, plot_simulation, plot_simulation_with_trends_matched]
def plot_irfs(irf_values, var_names, horizon, title="Impulse Responses"):
    """ Simple IRF plotting function """
    # Convert JAX array to NumPy for plotting
    irf_values_np = np.asarray(irf_values)
    var_names_list = list(var_names) # Ensure it's a list

    num_vars = irf_values_np.shape[1]
    if num_vars == 0:
        print(f"No variables to plot for: {title}")
        return

    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(min(5*cols, 18), 3*rows), sharex=True, squeeze=False)
    axes = axes.flatten() # Flatten the axes array for easy iteration
    plt.suptitle(title, fontsize=14)
    time = np.arange(horizon)

    for i, var_name in enumerate(var_names_list):
        if i < len(axes): # Check if subplot exists
            ax = axes[i]
            ax.plot(time, irf_values_np[:, i], label=var_name)
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':') # Zero line
            ax.set_title(var_name)
            ax.grid(True, linestyle='--', alpha=0.6)
            # Add x-label only to bottom row plots for clarity
            if (i // cols) == (rows - 1): # Check if it's in the last row
                ax.set_xlabel("Horizon")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    # plt.show() # Remove automatic showing, let the main script decide


def simulate_ssm_data(
    P: ArrayLike,
    R: ArrayLike,
    Omega: ArrayLike,
    T: int,
    key: jax.random.PRNGKey,
    state_init: Optional[ArrayLike] = None,
    measurement_noise_std: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    """
    Simulate data from a linear state-space model using JAX:
      s_t = P @ s_{t-1} + R @ e_t
      y_t = Omega @ s_t + eta_t

    [Docstring omitted for brevity; see your original.]

    Returns:
        Tuple[jax.Array, jax.Array]:
          - states: Simulated states (s_1, ..., s_T), shape (T, n_state).
          - observables: Simulated observables (y_1, ..., y_T), shape (T, n_obs).
    """
    P_jax = jnp.asarray(P, dtype=_DEFAULT_DTYPE)
    R_jax = jnp.asarray(R, dtype=_DEFAULT_DTYPE)
    Omega_jax = jnp.asarray(Omega, dtype=_DEFAULT_DTYPE)

    n_state = P_jax.shape[0]
    n_shock = R_jax.shape[1] if R_jax.ndim == 2 and R_jax.shape[1] > 0 else 0
    n_obs = Omega_jax.shape[0] if Omega_jax.ndim == 2 and Omega_jax.shape[0] > 0 else 0

    if P_jax.shape != (n_state, n_state):
        raise ValueError(f"P matrix shape mismatch: expected ({n_state}, {n_state}), got {P_jax.shape}")
    if R_jax.shape != (n_state, n_shock):
        # Allow R to be empty if no shocks
        if not (n_shock == 0 and R_jax.shape in [(n_state,), (n_state, 0)]):
            raise ValueError(f"R matrix shape mismatch: expected ({n_state}, {n_shock}), got {R_jax.shape}")
        elif n_shock == 0: # Ensure R has correct shape even if empty
            R_jax = jnp.zeros((n_state, 0), dtype=_DEFAULT_DTYPE)

    if Omega_jax.shape != (n_obs, n_state):
        # Allow Omega to be empty if no obs
        if not (n_obs == 0 and Omega_jax.shape in [(0, n_state), (0,)]):
            raise ValueError(f"Omega matrix shape mismatch: expected ({n_obs}, {n_state}), got {Omega_jax.shape}")
        elif n_obs == 0: # Ensure Omega has correct shape even if empty
            Omega_jax = jnp.zeros((0, n_state), dtype=_DEFAULT_DTYPE)

    key_state, key_measure = jax.random.split(key)

    # Handle initial state
    if state_init is None:
        # print(
        #     "[simulate_ssm_data] Warning: No state_init provided! Initial state s_0 defaults to zeros. "
        #     "This may be unsuitable for models with trends or non-zero steady states. "
        #     "Provide a specific state_init for accurate simulation."
        # )
        s_previous = jnp.zeros(n_state, dtype=_DEFAULT_DTYPE)
    else:
        s_previous = jnp.asarray(state_init, dtype=_DEFAULT_DTYPE)
        if s_previous.shape != (n_state,):
            raise ValueError(f"state_init shape mismatch: expected ({n_state},), got {s_previous.shape}")

    # Draw state and measurement noises for all periods T
    state_shocks = jnp.zeros((T, n_shock), dtype=_DEFAULT_DTYPE)
    if n_shock > 0:
        state_shocks = jax.random.normal(key_state, shape=(T, n_shock), dtype=_DEFAULT_DTYPE)

    measurement_noise = jnp.zeros((T, n_obs), dtype=_DEFAULT_DTYPE)
    if measurement_noise_std > 0.0 and n_obs > 0:
        measurement_noise = jax.random.normal(key_measure, shape=(T, n_obs), dtype=_DEFAULT_DTYPE) * measurement_noise_std

    def step(s_prev, inputs):
        """
        Args:
            s_prev: State from the previous period (s_{t-1}).
            inputs: Tuple containing shocks for the current period (e_t, eta_t).
        Returns:
            Tuple: (s_t, (s_t, y_t))
        """
        e_t, eta_t = inputs

        # State evolution: s_t = P @ s_{t-1} + R @ e_t
        s_t = P_jax @ s_prev
        if n_shock > 0:
            s_t += R_jax @ e_t # Add shock impact

        # Observation: y_t = Omega @ s_t + eta_t
        y_t = jnp.zeros(n_obs, dtype=_DEFAULT_DTYPE) # Default if no obs
        if n_obs > 0:
            y_t = Omega_jax @ s_t
            if measurement_noise_std > 0.0:
                y_t += eta_t # Add measurement noise

        return s_t, (s_t, y_t)

    # Run the simulation using lax.scan
    _, (states_T, ys_T) = lax.scan(
        step, s_previous, (state_shocks, measurement_noise)
    )

    # states_T will have shape (T, n_state); ys_T will have shape (T, n_obs)
    return states_T, ys_T


def plot_simulation(sim_data, var_names, title="Simulated Data"):
    """Simple simulation plotting function"""
    sim_data_np = np.asarray(sim_data)  # Convert JAX array to NumPy for plotting
    var_names_list = list(var_names)
    num_vars = sim_data_np.shape[1]
    T = sim_data_np.shape[0]

    if num_vars == 0:
        print(f"No variables to plot for: {title}")
        return

    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(min(5 * cols, 18), 3 * rows), sharex=True, squeeze=False)
    axes = axes.flatten()
    plt.suptitle(title, fontsize=14)
    time = np.arange(T)

    for i, var_name in enumerate(var_names_list):
        if i < len(axes):
            ax = axes[i]
            ax.plot(time, sim_data_np[:, i], label=var_name)
            ax.set_title(var_name)
            ax.grid(True, linestyle='--', alpha=0.6)
            if (i // cols) == (rows - 1):
                ax.set_xlabel("Time")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_simulation_with_trends_matched(
    sim_observables: np.ndarray,
    obs_var_names: list,                   # ["L_GDP_OBS", ...]
    sim_states: np.ndarray,
    state_var_names: list,                 # e.g. ["L_GDP_GAP", ..., "L_GDP_TREND", ...]
    trend_state_names: list,               # e.g. ["L_GDP_TREND", "PI_TREND", ...]
    contemporaneous_trend_defs: Dict[str, str], # Add this argument! e.g. {"RS_TREND": "RR_TREND + PI_TREND"}
    mapping: Optional[Dict[str, str]] = None, # Optional: {"L_GDP_OBS": "L_GDP_TREND", ...}
    title="Simulated Observables and Trends"
):
    """
    Plots observables and their 'main' trend side-by-side.
    If the trend is defined contemporaneously (e.g., A = B + C),
    it computes and plots the derived trend using the simulated states B and C.
    """
    sim_obs_np = np.asarray(sim_observables)
    sim_states_np = np.asarray(sim_states)
    time = np.arange(sim_obs_np.shape[0])
    num_states = sim_states_np.shape[1]

    # --- Auto-generate mapping if None (unchanged) ---
    if mapping is None:
        mapping = {}
        for obs in obs_var_names:
            # Simple match based on the part before the first underscore or full name
            obs_base = obs.split('_OBS')[0] # Try removing _OBS suffix
            found_match = False
            for trend in trend_state_names: # Check actual state trends first
                trend_base = trend.split('_TREND')[0]
                if obs_base == trend_base:
                    mapping[obs] = trend
                    found_match = True
                    break
            if not found_match: # Check contemporaneous defs
                 for contemp_trend in contemporaneous_trend_defs.keys():
                     contemp_base = contemp_trend.split('_TREND')[0]
                     if obs_base == contemp_base:
                         mapping[obs] = contemp_trend
                         break
    # --- End Auto-mapping ---

    cols = 2
    rows = (len(obs_var_names) + cols - 1) // cols # Ensure enough rows
    fig, axes = plt.subplots(rows, cols, figsize=(min(6 * cols, 18), 4 * rows), squeeze=False)
    axes = axes.flatten()
    plt.suptitle(title, fontsize=14)

    plotted_count = 0
    for i, obs_name in enumerate(obs_var_names):
        if i >= len(axes): break # Avoid index error if more obs than subplots

        ax = axes[i]
        plotted_count += 1
        try:
            obs_idx = obs_var_names.index(obs_name)
            ax.plot(time, sim_obs_np[:, obs_idx], label=f"{obs_name}", linewidth=2)
        except (ValueError, IndexError):
             print(f"Warning: Could not find or plot observable '{obs_name}'.")
             continue


        # --- Find and Plot Matching Trend ---
        trend_name = mapping.get(obs_name)
        plotted_trend = False
        if trend_name:
            # 1. Check if trend is directly in the state vector
            if trend_name in state_var_names:
                try:
                    trend_idx = state_var_names.index(trend_name)
                    ax.plot(time, sim_states_np[:, trend_idx], label=f"{trend_name} (State)", linestyle='--', alpha=0.85)
                    plotted_trend = True
                    # print(f"  Plotted state trend '{trend_name}' for '{obs_name}'") # Debug
                except (ValueError, IndexError):
                     print(f"Warning: State trend '{trend_name}' mapped to '{obs_name}' but not found in state vector.")

            # 2. If not in state, check if defined contemporaneously
            elif trend_name in contemporaneous_trend_defs:
                definition_str = contemporaneous_trend_defs[trend_name]
                print(f"  Info: Trend '{trend_name}' for '{obs_name}' is defined as: {definition_str}. Attempting calculation...")
                # --- Simple Parser for 'VAR1 + VAR2' ---
                parts = [p.strip() for p in definition_str.split('+')]
                if len(parts) == 2:
                    var1_name, var2_name = parts[0], parts[1]
                    try:
                        idx1 = state_var_names.index(var1_name)
                        idx2 = state_var_names.index(var2_name)
                        # Calculate the derived trend
                        derived_trend = sim_states_np[:, idx1] + sim_states_np[:, idx2]
                        ax.plot(time, derived_trend, label=f"{trend_name} (Derived)", linestyle='--', alpha=0.85)
                        plotted_trend = True
                        print(f"    Successfully calculated and plotted derived trend '{trend_name}' = '{var1_name}' + '{var2_name}'")
                    except ValueError:
                        print(f"    Warning: Could not find state variables '{var1_name}' or '{var2_name}' needed to derive '{trend_name}'.")
                    except Exception as e_calc:
                         print(f"   Warning: Error calculating derived trend '{trend_name}': {e_calc}")
                else:
                     # Add more parsing here if needed (e.g., subtraction, parameters)
                     print(f"    Warning: Cannot plot derived trend '{trend_name}'. Plotting function only handles simple addition (e.g., 'VAR1 + VAR2') for now. Definition was: '{definition_str}'")
                # --- End Simple Parser ---

            else:
                 print(f"  Warning: Trend variable '{trend_name}' for observable '{obs_name}' not found in states or contemporaneous definitions.")

        if plotted_trend:
             ax.legend()
        ax.grid(True)
        ax.set_xlabel("Time")
        # --- End Trend Plotting ---

    # Hide any empty subplots
    for j in range(plotted_count, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])




# --- Dynare File Parsing Functions (Minimal changes needed) ---
# Keep extract_declarations, extract_model_equations, etc. as they are needed
# to get the raw strings and names before JAX AD step.
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

# --- Trend/Observation Parsing Functions (Unchanged) ---
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
                # Assume trend equations define current value based on past/shocks
                processed_equations.append(f"({base_lhs}) - ({rhs})")
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
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for model shock '{shock_name}'.")
    return stderrs

# --- JAX AD BASED MATRIX COMPUTATION ---

def _create_jax_resid_function(
    equations: List[str],
    var_names: List[str],
    shock_names: List[str],
    param_names: List[str],
    model_type: str = "stationary" # "stationary", "trend", "observation"
) -> Tuple[Callable, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Dynamically creates a JAX-compatible function evaluating equation residuals.

    Args:
        equations: List of equation strings (LHS - RHS).
        var_names: Ordered list of variable names for this model part.
        shock_names: Ordered list of shock names for this model part.
        param_names: Ordered list of *all* parameter names.
        model_type: Type of model equations being processed.

    Returns:
        tuple: (callable_function, var_map, shock_map, param_map)
            - callable_function: JAX function, signature depends on model_type.
            - var_map: Dictionary mapping variable names to indices.
            - shock_map: Dictionary mapping shock names to indices.
            - param_map: Dictionary mapping parameter names to indices.
    """
    var_map = {name: i for i, name in enumerate(var_names)}
    shock_map = {name: i for i, name in enumerate(shock_names)}
    param_map = {name: i for i, name in enumerate(param_names)}
    num_eq = len(equations)

    # Define function signature based on model type
    if model_type == "stationary":
        func_signature = "def _resid_func(y_t, y_m1, y_p1, shocks_t, params):"
        input_args = ["y_t", "y_m1", "y_p1", "shocks_t", "params"]
    elif model_type == "trend":
        # Trend model: y_t depends on y_{t-1} and shocks_t
        func_signature = "def _resid_func(y_t, y_m1, shocks_t, params):"
        input_args = ["y_t", "y_m1", "shocks_t", "params"]
    elif model_type == "observation":
        # Measurement eq: obs_t depends on state_t (stationary + trend)
        # 'var_names' here are the observable names
        # 'shock_names' here are the augmented state names (stat + trend)
        func_signature = "def _resid_func(obs_t, state_t, params):"
        input_args = ["obs_t", "state_t", "params"]
        # Rename maps for clarity in this context
        obs_map = var_map
        state_map = shock_map # Reuse shock_map logic for states
        var_map = obs_map
        shock_map = state_map
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    func_body_lines = [f"    # Dynamically generated JAX residual function for {model_type} model"]
    func_body_lines.append(f"    residuals = jnp.zeros({num_eq}, dtype=params.dtype)")

    # Known functions to map
    known_funcs = {'exp': 'jnp.exp', 'log': 'jnp.log', 'sqrt': 'jnp.sqrt', 'abs': 'jnp.abs'}
    # Regex for variables with explicit time shifts like VAR(+1), VAR(-1)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    # Regex for standalone variable/parameter/shock names
    name_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')

    for i, eq_str in enumerate(equations):
        processed_eq = eq_str # Start with original residual string "LHS - RHS"

        # 1. Replace known math functions
        for py_name, jnp_name in known_funcs.items():
            processed_eq = re.sub(r'\b' + py_name + r'\b', jnp_name, processed_eq)

        # 2. Replace timed variables (e.g., X(+1))
        def replace_timed_var(match):
            base_name, time_shift_str = match.groups()
            time_shift = int(time_shift_str)

            if base_name in var_map:
                var_idx = var_map[base_name]
                if model_type == "stationary":
                    if time_shift == 1: return f"y_p1[{var_idx}]"
                    if time_shift == 0: return f"y_t[{var_idx}]"
                    if time_shift == -1: return f"y_m1[{var_idx}]"
                elif model_type == "trend":
                    # Trends typically depend on t-1
                    if time_shift == 0: return f"y_t[{var_idx}]" # The LHS variable
                    if time_shift == -1: return f"y_m1[{var_idx}]"
                elif model_type == "observation":
                    # Observables depend on current state, vars are observables
                    if time_shift == 0: return f"obs_t[{var_idx}]" # The LHS observable
                    # Note: State variables on RHS won't have time shifts in typical definitions
            # Shocks are typically contemporaneous (time_shift == 0)
            elif base_name in shock_map and time_shift == 0:
                 shock_idx = shock_map[base_name]
                 if model_type == "stationary": return f"shocks_t[{shock_idx}]"
                 if model_type == "trend": return f"shocks_t[{shock_idx}]"
                 if model_type == "observation": # 'shocks' are states here
                     return f"state_t[{shock_idx}]" # Reference state_t

            # If it's none of the above with a time shift, it might be an error
            # or an auxiliary variable that should have been handled earlier
            print(f"Warning: Unhandled timed symbol '{match.group(0)}' in eq {i} ({model_type}). Treating as is.")
            return match.group(0) # Return original if not replaceable

        processed_eq = var_time_regex.sub(replace_timed_var, processed_eq)

        # 3. Replace remaining standalone names (vars, params, shocks at time t=0 implicitly)
        # Sort by length descending to handle overlapping names correctly (e.g., C before C_A)
        all_names_sorted = sorted(list(var_map.keys()) + list(shock_map.keys()) + list(param_map.keys()), key=len, reverse=True)

        for name in all_names_sorted:
            # Use word boundaries to avoid partial matches inside other words
            pattern = r'\b' + re.escape(name) + r'\b'
            replacement = None
            if name in param_map:
                replacement = f"params[{param_map[name]}]"
            elif name in var_map: # Variable at time t=0 (or obs_t, or trend_t)
                 if model_type == "stationary": replacement = f"y_t[{var_map[name]}]"
                 elif model_type == "trend": replacement = f"y_t[{var_map[name]}]" # LHS trend var
                 elif model_type == "observation": replacement = f"obs_t[{var_map[name]}]" # LHS observable
            elif name in shock_map: # Shock at t=0 (or state_t)
                 if model_type == "stationary": replacement = f"shocks_t[{shock_map[name]}]"
                 elif model_type == "trend": replacement = f"shocks_t[{shock_map[name]}]"
                 elif model_type == "observation": replacement = f"state_t[{shock_map[name]}]" # RHS state var

            if replacement:
                processed_eq = re.sub(pattern, replacement, processed_eq)

        # Add the processed equation line to the function body
        func_body_lines.append(f"    residuals = residuals.at[{i}].set({processed_eq})")

    func_body_lines.append("    return residuals")
    full_func_code = "\n".join([func_signature] + func_body_lines)

    # Define the function in a temporary scope
    temp_scope = {'jnp': jnp}
    try:
        # Suppress warnings about overwriting lambda/variable names if reused
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            exec(full_func_code, temp_scope)
    except Exception as e:
        print("--- ERROR during dynamic function compilation ---")
        print(f"Model Type: {model_type}")
        print("Generated Code:\n", full_func_code)
        print("Error:", e)
        raise RuntimeError(f"Failed to compile dynamic JAX function for {model_type} model.") from e

    # Extract the compiled function
    resid_func = temp_scope.get('_resid_func')
    if resid_func is None:
         raise RuntimeError(f"Could not find compiled function '_resid_func' in exec scope for {model_type}.")

    
    return resid_func, var_map, shock_map, param_map, full_func_code


def compute_matrices_jax_ad(
    equations: List[str],
    var_names: List[str],
    shock_names: List[str],
    param_names: List[str],
    param_values: Dict[str, float],
    model_type: str = "stationary",
    dtype: jnp.dtype = _DEFAULT_DTYPE
) -> Dict[str, jax.Array]:
    """
    Computes model matrices (A, B, C, D or P_trends, Q_trends or Omega) using JAX AD.

    Args:
        equations: List of equation strings (LHS - RHS = 0).
        var_names: Ordered list of variable names for this model part.
        shock_names: Ordered list of shock/state names relevant to the RHS.
        param_names: Ordered list of *all* parameter names in the model.
        param_values: Dictionary mapping parameter names to their numerical values.
        model_type: "stationary", "trend", or "observation".
        dtype: JAX dtype to use.

    Returns:
        Dictionary containing the computed matrices as JAX arrays.
          - For "stationary": {'A', 'B', 'C', 'D'}
          - For "trend": {'P_trends', 'Q_trends'}
          - For "observation": {'Omega'}
    """
    print(f"\n--- Computing Matrices via JAX AD for {model_type.upper()} Model ---")

    num_vars = len(var_names)
    num_shocks = len(shock_names)
    num_params = len(param_names)
    num_eq = len(equations)

    # Check consistency
    if model_type in ["stationary", "trend"] and num_eq != num_vars:
         print(f"Warning ({model_type}): Number of equations ({num_eq}) != number of variables ({num_vars}).")
    elif model_type == "observation" and num_eq != num_vars: # obs vars
         print(f"Warning ({model_type}): Number of equations ({num_eq}) != number of observable variables ({num_vars}).")

    # Generate the JAX residual function dynamically
    try:
        resid_func, var_map, shock_map, param_map,resid_func_code  = _create_jax_resid_function(
            equations, var_names, shock_names, param_names, model_type
        )
        print("DEBUG")
        print(f"\n--- Dynamically Generated Code for {model_type.upper()} resid_func ---")
        print(resid_func_code)
        print("--- End Generated Code ---")  
    except Exception as e:
        print(f"Error creating JAX residual function for {model_type}: {e}")
        raise

     
    # Prepare input arguments for jacobian evaluation
    # Evaluate at steady state (zeros for stationary gaps, shocks)
    # Parameters are evaluated at the provided values
    params_vec = jnp.array([param_values[p] for p in param_names], dtype=dtype)


    # --- Compute Jacobians based on model type ---
    matrices = {}

    if model_type == "stationary":
        # Inputs: y_t, y_m1, y_p1, shocks_t, params
        y_steady = jnp.zeros(num_vars, dtype=dtype)
        shocks_steady = jnp.zeros(num_shocks, dtype=dtype)

        # Compute Jacobians w.r.t. inputs
        # jac_A = d(resid) / d(y_p1) --> This IS the 'A' matrix (coeff of y_p1)
        jac_A = jax.jacobian(resid_func, argnums=2)(y_steady, y_steady, y_steady, shocks_steady, params_vec)
        # jac_B = d(resid) / d(y_t) --> This IS the 'B' matrix (coeff of y_t)
        jac_B = jax.jacobian(resid_func, argnums=0)(y_steady, y_steady, y_steady, shocks_steady, params_vec)
        # jac_C = d(resid) / d(y_m1) --> This IS the 'C' matrix (coeff of y_m1)
        jac_C = jax.jacobian(resid_func, argnums=1)(y_steady, y_steady, y_steady, shocks_steady, params_vec)
        # jac_D_neg = d(resid) / d(shocks_t) --> Need to negate for solver's D convention
        jac_D_neg = jax.jacobian(resid_func, argnums=3)(y_steady, y_steady, y_steady, shocks_steady, params_vec)

        matrices['A'] = jac_A
        matrices['B'] = jac_B
        matrices['C'] = jac_C
        matrices['D'] = -jac_D_neg # D = - d(resid) / d(shocks)

    elif model_type == "trend":
        # Inputs: y_t, y_m1, shocks_t, params
        # Assume linearization around some point, Jacobians constant for linear trends
        y_steady = jnp.zeros(num_vars, dtype=dtype) # Represents trend levels/changes
        shocks_steady = jnp.zeros(num_shocks, dtype=dtype)

        # jac_P_neg = d(resid) / d(y_m1) where resid = y_t - (P*y_m1 + Q*shocks + ...)
        # So, jac_P_neg = -P. We want P.
        jac_P_neg = jax.jacobian(resid_func, argnums=1)(y_steady, y_steady, shocks_steady, params_vec)
        # jac_Q_neg = d(resid) / d(shocks_t) = -Q. We want Q.
        jac_Q_neg = jax.jacobian(resid_func, argnums=2)(y_steady, y_steady, shocks_steady, params_vec)

        matrices['P_trends'] = -jac_P_neg
        matrices['Q_trends'] = -jac_Q_neg

    elif model_type == "observation":
        # Inputs: obs_t, state_t, params
        # Here, var_names = obs_vars, shock_names = augmented_state_vars
        num_obs = num_vars
        num_state = num_shocks # state_t corresponds to shocks_t input arg

        obs_steady = jnp.zeros(num_obs, dtype=dtype)
        state_steady = jnp.zeros(num_state, dtype=dtype)

        # jac_Omega_neg = d(resid) / d(state_t) where resid = obs_t - (Omega*state_t + ...)
        # So, jac_Omega_neg = -Omega. We want Omega.
        jac_Omega_neg = jax.jacobian(resid_func, argnums=1)(obs_steady, state_steady, params_vec)

        matrices['Omega'] = -jac_Omega_neg

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # # --- Validation ---
    print(f"\nValidating computed matrices for {model_type.upper()}...")
    for name, mat in matrices.items(): # Iterate through computed matrices ONLY
        if not jnp.all(jnp.isfinite(mat)):
            print(f"  WARNING ({model_type}): Matrix '{name}' contains non-finite values (NaN/Inf).")
            # print(mat) # Optionally print the matrix

        # Define expected shape based on the matrix NAME and context
        expected_shape = None
        if name in ['A', 'B', 'C']: # Stationary dynamics matrices
            expected_shape = (num_eq, num_vars)
        elif name == 'D': # Stationary shock matrix
            expected_shape = (num_eq, num_shocks)
        elif name == 'P_trends': # Trend dynamics matrix
             # Note: num_eq should equal num_vars for state trends
            expected_shape = (num_eq, num_vars)
        elif name == 'Q_trends': # Trend shock matrix
            expected_shape = (num_eq, num_shocks)
        elif name == 'Omega': # Observation matrix
             # Context: num_eq=num_obs, num_shocks=num_states for observation model
             expected_shape = (num_eq, num_shocks)

        # Check the shape if an expected shape was determined
        if expected_shape is not None:
            if mat.shape != expected_shape:
                 print(f"  WARNING ({model_type}): Matrix '{name}' shape mismatch. Got {mat.shape}, expected {expected_shape}")
            else:
                 print(f"  Matrix '{name}' shape OK: {mat.shape}")
        else:
             # This case should ideally not be reached if names are correct
             print(f"  Internal Warning: Could not determine expected shape for matrix '{name}' ({model_type}). Shape is {mat.shape}.")
    print(f"--- Validation for {model_type.upper()} complete ---")
    # --- End of Updated Validation ---


    print(f"--- JAX AD Matrix Computation for {model_type.upper()} COMPLETE ---")
    return matrices, resid_func_code


# --- SDA Solver and Q Computation (JAX Versions - UNCHANGED) ---
# [Your existing solve_quadratic_matrix_equation_jax and compute_Q_jax]
SDAState = namedtuple("SDAState", ["Xk", "Yk", "Ek", "Fk", "k", "converged", "rel_diff", "is_valid"])
_SDA_JITTER = 1e-14 # Small regularization factor

def solve_quadratic_matrix_equation_jax(A, B, C, initial_guess=None,
                                        tol=1e-12, max_iter=500,
                                        verbose=False): # verbose is ineffective in JIT
    """
    Solves A X^2 + B X + C = 0 for X using the SDA algorithm implemented with
    jax.lax.scan. Uses jnp.where for conditional updates after update step.
    Returns NaN solution if convergence fails or numerical issues occur.

    Args:
        A, B, C: JAX arrays for the quadratic matrix equation coefficients.
        initial_guess: Optional initial guess for the solution (JAX array).
        tol: Convergence tolerance for the relative change in X.
        max_iter: Fixed maximum number of iterations for the scan loop.
        verbose: (Not effective inside JIT/grad)

    Returns:
        Tuple: (X_sol, iter_count, residual_ratio, converged_flag)
               - X_sol: The computed solution (JAX array). NaN if solve failed.
               - iter_count: Iteration count (always max_iter in this version).
               - residual_ratio: Relative residual norm of the final state after scan.
               - converged_flag: Boolean JAX array indicating success.
    """
    dtype = A.dtype
    n = A.shape[0]
    A_jax = jnp.asarray(A, dtype=dtype)
    B_jax = jnp.asarray(B, dtype=dtype)
    C_jax = jnp.asarray(C, dtype=dtype)
    if initial_guess is None:
        X_guess = jnp.zeros_like(A_jax)
    else:
        X_guess = jnp.asarray(initial_guess, dtype=dtype)

    # Initial Setup
    E_init = C_jax
    F_init = A_jax
    Bbar = B_jax + A_jax @ X_guess
    I = jnp.eye(n, dtype=dtype)
    Bbar_reg = Bbar + _SDA_JITTER * I
    # Use jax.scipy.linalg.solve which handles potential singularity better (returns NaN/inf)
    E0 = jax.scipy.linalg.solve(Bbar_reg, -E_init, assume_a='gen') # Solve Bbar * E0 = -E_init
    F0 = jax.scipy.linalg.solve(Bbar_reg, -F_init, assume_a='gen') # Solve Bbar * F0 = -F_init
    initial_solve_valid = jnp.all(jnp.isfinite(E0)) & jnp.all(jnp.isfinite(F0))

    # Scan Loop Definition
    def sda_scan_body(state, _):
        Xk, Yk, Ek, Fk, k, prev_converged, prev_rel_diff, prev_is_valid = state
        M1 = I - Yk @ Xk + _SDA_JITTER * I; M2 = I - Xk @ Yk + _SDA_JITTER * I
        # Perform solves directly. Let NaN/inf propagate on failure.
        # temp_E = E @ M1^-1 @ Ek ; E_new = temp_E
        temp_E = jax.scipy.linalg.solve(M1, Ek, assume_a='gen'); E_new = Ek @ temp_E
        # temp_F = F @ M2^-1 @ Fk ; F_new = temp_F
        temp_F = jax.scipy.linalg.solve(M2, Fk, assume_a='gen'); F_new = Fk @ temp_F
        # temp_X = M2^-1 @ Xk @ Ek; X_new = Xk + Fk @ temp_X
        temp_X = Xk @ Ek; temp_X = jax.scipy.linalg.solve(M2, temp_X, assume_a='gen'); X_new = Xk + Fk @ temp_X
        # temp_Y = M1^-1 @ Yk @ Fk; Y_new = Yk + Ek @ temp_Y
        temp_Y = Yk @ Fk; temp_Y = jax.scipy.linalg.solve(M1, temp_Y, assume_a='gen'); Y_new = Yk + Ek @ temp_Y

        X_diff_norm = jnp.linalg.norm(X_new - Xk, ord='fro'); X_norm = jnp.linalg.norm(X_new, ord='fro')
        # Use safe division, default to inf if X_norm is too small
        current_rel_diff = jnp.where(X_norm > 1e-15, X_diff_norm / X_norm, jnp.inf)

        current_step_valid = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(Y_new)) & \
                             jnp.all(jnp.isfinite(E_new)) & jnp.all(jnp.isfinite(F_new)) & \
                             jnp.isfinite(current_rel_diff)
        converged_this_step = current_step_valid & (current_rel_diff < tol)
        current_is_valid = prev_is_valid & current_step_valid
        current_converged = prev_converged | converged_this_step

        # Only update state if the *previous* state was valid, the current step is valid,
        # AND convergence hadn't already occurred.
        keep_new_state_cond = prev_is_valid & current_step_valid & (~prev_converged)

        # Conditionally update state using jnp.where
        X_next = jnp.where(keep_new_state_cond, X_new, Xk); Y_next = jnp.where(keep_new_state_cond, Y_new, Yk)
        E_next = jnp.where(keep_new_state_cond, E_new, Ek); F_next = jnp.where(keep_new_state_cond, F_new, Fk)
        next_rel_diff = jnp.where(keep_new_state_cond, current_rel_diff, prev_rel_diff)
        # Ensure converged status persists if achieved
        # Use jnp.where: if we keep the new state, use its converged status, otherwise keep the old converged status
        next_converged = jnp.where(keep_new_state_cond, current_converged, prev_converged)
        next_is_valid = current_is_valid # is_valid tracks if *any* step became invalid
        next_state = SDAState(X_next, Y_next, E_next, F_next, k + 1, next_converged, next_rel_diff, next_is_valid)

        return next_state, None # Pass state to next iteration, no per-iteration output needed

    init_state = SDAState(Xk=E0, Yk=F0, Ek=E0, Fk=F0, k=0, converged=jnp.array(False), rel_diff=jnp.inf, is_valid=initial_solve_valid)
    final_state, _ = lax.scan(sda_scan_body, init_state, xs=None, length=max_iter)

    # Post-Scan Processing
    X_sol_scan = final_state.Xk + X_guess
    # Final convergence requires the loop to have finished AND the state to be valid AND converged flag set
    converged_flag = final_state.converged & final_state.is_valid
    iter_final = final_state.k # Will be max_iter due to lax.scan fixed length

    # Calculate final residual ratio (for info) using the computed solution
    residual = A_jax @ (X_sol_scan @ X_sol_scan) + B_jax @ X_sol_scan + C_jax
    residual_norm = jnp.linalg.norm(residual, 'fro')
    term_norms = (jnp.linalg.norm(A_jax @ X_sol_scan @ X_sol_scan, 'fro') +
                  jnp.linalg.norm(B_jax @ X_sol_scan, 'fro') +
                  jnp.linalg.norm(C_jax, 'fro'))
    # Safe division for residual ratio
    residual_ratio = jnp.where(term_norms > 1e-15, residual_norm / term_norms, jnp.inf)

    # Return NaN if convergence failed or state is invalid
    X_sol_final = jnp.where(converged_flag, X_sol_scan, jnp.full_like(X_sol_scan, jnp.nan))

    return X_sol_final, iter_final, residual_ratio, converged_flag

def compute_Q_jax(A: ArrayLike, B: ArrayLike, D: ArrayLike, P: ArrayLike,
                  dtype: Optional[jnp.dtype] = None) -> jax.Array:
    """
    Computes Q for y_t = P y_{t-1} + Q e_t using JAX. Solves (A P + B) Q = D.
    Allows NaNs to propagate if the solve fails.
    """
    effective_dtype = dtype if dtype is not None else (A.dtype if hasattr(A, 'dtype') else _DEFAULT_DTYPE)
    A_jax = jnp.asarray(A, dtype=effective_dtype); B_jax = jnp.asarray(B, dtype=effective_dtype)
    D_jax = jnp.asarray(D, dtype=effective_dtype); P_jax = jnp.asarray(P, dtype=effective_dtype)
    n = A_jax.shape[0]
    n_shock = D_jax.shape[1] if D_jax.ndim == 2 else (0 if D_jax.size == 0 else 1) # Handle empty/scalar D
    if D_jax.size == 0: return jnp.zeros((n, 0), dtype=effective_dtype) # Handle empty D explicitly

    APB = A_jax @ P_jax + B_jax
    # Add jitter for potentially ill-conditioned APB matrix
    APB_reg = APB + _SDA_JITTER * jnp.eye(n, dtype=effective_dtype)
    # Attempt solve, let NaNs propagate if it fails
    # Ensure D has correct shape if it's 1D
    if D_jax.ndim == 1: D_jax = D_jax.reshape(-1, 1)
    Q = jax.scipy.linalg.solve(APB_reg, D_jax, assume_a='gen')
    return Q


# --- Main Parsing and Ordering Logic (Modified to use JAX AD) ---
def parse_and_compute_matrices_jax_ad(model_string, verbose=True):
    """
    Parses the model, handles leads/lags, orders variables/equations according
    to Dynare convention (Backward -> Mixed -> Forward Aux -> Static),
    and computes the numerical matrices A, B, C, D using JAX AD.

    Args:
        model_string (str): The full content of the Dynare file.
        verbose (bool): If True, prints progress information.

    Returns:
        tuple: Contains:
            - A_jax, B_jax, C_jax, D_jax: Computed JAX arrays for *ordered* matrices.
            - ordered_vars (List[str]): Ordered list of stationary state variables.
            - shock_names (List[str]): List of stationary shock names.
            - param_names (List[str]): List of combined stationary parameter names.
            - param_assignments (Dict[str, float]): Dictionary of default parameter values.
            - var_perm_indices (List[int]): Indices for variable permutation (new -> old).
            - eq_perm_indices (List[int]): Indices for equation permutation (new -> old).
            - initial_vars (List[str]): Variable list before ordering.
            - stationary_resid_func_code (str): Code string of the generated residual function.
    """
    if verbose: print("--- Parsing Declarations ---")
    declared_vars, shock_names, param_names_declared, param_assignments_initial = extract_declarations(model_string)
    # Combine parameters (including inferred sigmas)
    inferred_sigma_params = [f"sigma_{shk}" for shk in shock_names]
    stat_stderr_values = extract_stationary_shock_stderrs(model_string)
    combined_param_names = list(dict.fromkeys(param_names_declared + inferred_sigma_params).keys())
    combined_param_assignments = stat_stderr_values.copy()
    combined_param_assignments.update(param_assignments_initial)
    # Ensure defaults for sigmas
    for p_sigma in inferred_sigma_params:
        if p_sigma not in combined_param_assignments:
            if verbose: print(f"Warning: Inferred sigma '{p_sigma}' using default 1.0.")
            combined_param_assignments[p_sigma] = 1.0
    param_names = combined_param_names
    param_assignments = combined_param_assignments

    if verbose: print(f"Variables: {len(declared_vars)}, Shocks: {len(shock_names)}, Params: {len(param_names)}")

    if verbose: print("\n--- Parsing Stationary Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    if verbose: print(f"Found {len(raw_equations)} raw equations.")

    # --- Handling Leads/Lags & Auxiliaries ---
    if verbose: print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars)
    aux_variables = OrderedDict() # Stores definition string for each aux var
    processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    # [Auxiliary variable handling logic - unchanged]
    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]
        eq_idx += 1
        modified_eq = eq
        matches = list(var_time_regex.finditer(eq))
        for match in reversed(matches): # Process matches in reverse order
            base_name = match.group(1); time_shift = int(match.group(2))
            if base_name not in endogenous_vars and base_name not in aux_variables: continue

            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        def_eq_str = f"({aux_name}) - ({prev_var_for_def}(+1))" # Enclose in ()
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                for def_eq in aux_needed_defs:
                    # Ensure aux equations are also added in the processed format
                    if def_eq not in processed_equations and f"({def_eq.split('-')[0].strip().strip('()')}) - ({def_eq.split('-')[1].strip().strip('()')})" not in processed_equations:
                        processed_equations.append(def_eq)

            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        def_eq_str = f"({aux_name}) - ({prev_var_for_def}(-1))" # Enclose in ()
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars: endogenous_vars.append(aux_name)
                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)"
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]
                for def_eq in aux_needed_defs:
                     # Ensure aux equations are also added in the processed format
                    if def_eq not in processed_equations and f"({def_eq.split('-')[0].strip().strip('()')}) - ({def_eq.split('-')[1].strip().strip('()')})" not in processed_equations:
                        processed_equations.append(def_eq)
        if modified_eq != eq:
            processed_equations[eq_idx - 1] = modified_eq
            # Re-evaluate equations count if changed
            num_eq = len(processed_equations)
    # --- End Auxiliary Handling ---


    initial_vars_ordered = list(dict.fromkeys(endogenous_vars).keys()) # Ensure unique vars
    num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations) # Update count after potential additions
    num_shocks = len(shock_names)

    if verbose: print(f"Total variables after processing leads/lags ({num_vars}): {initial_vars_ordered}")
    if verbose: print(f"Total equations after processing leads/lags ({num_eq}):\n" + "\n".join([f"  Eq {i}: {eq}" for i, eq in enumerate(processed_equations)]))

    if num_vars != num_eq:
        raise ValueError(f"Model not square after aux processing: {num_vars} vars vs {num_eq} eqs.")

    # --- Compute UNORDERED Matrices using JAX AD ---
    # We compute based on the `initial_vars_ordered` first, then reorder later
    if verbose: print("\n--- Computing Unordered A, B, C, D via JAX AD ---")
    matrices_unordered, stationary_resid_func_code = compute_matrices_jax_ad(
        equations=processed_equations,
        var_names=initial_vars_ordered,
        shock_names=shock_names,
        param_names=param_names,
        param_values=param_assignments,
        model_type="stationary",
        dtype=_DEFAULT_DTYPE
    )
    # The generated code is printed inside compute_matrices_jax_ad

    A_unord = matrices_unordered['A']
    B_unord = matrices_unordered['B']
    C_unord = matrices_unordered['C']
    D_unord = matrices_unordered['D']

    # --- Classify Variables & Determine Order (Based on UNORDERED Jacobians & Structure) ---
    if verbose: print("\n--- Classifying Variables for Ordering (Improved Logic) ---")
    # Use the computed numerical Jacobians (A_unord, C_unord) for classification
    pb_vars = [] # Purely backward (state vars defined by lags/shocks)
    mf_vars = [] # Mixed Fwd/Bwd Endogenous (original declared vars unless purely fwd/bwd)
    af_vars = [] # Auxiliary Forward (defined by leads)
    s_vars = []  # Static (should be empty or handled by substitution)

    # Tolerance for checking zero columns/norms
    zero_tol = 1e-9

    # Identify auxiliary lead/lag variables first
    aux_lead_vars = [v for v in initial_vars_ordered if v.startswith("aux_") and "_lead_" in v]
    aux_lag_vars = [v for v in initial_vars_ordered if v.startswith("aux_") and "_lag_" in v]
    original_declared_vars = [v for v in initial_vars_ordered if not v.startswith("aux_")]

    # Classify based on dependencies shown in Jacobians
    for var in initial_vars_ordered:
        j = initial_vars_ordered.index(var)
        # Check lag dependency (col j in C_unord is non-zero)
        has_lag = jnp.linalg.norm(C_unord[:, j]) > zero_tol
        # Check lead dependency (col j in A_unord is non-zero)
        has_lead = jnp.linalg.norm(A_unord[:, j]) > zero_tol

        is_aux_lead = var in aux_lead_vars
        is_aux_lag = var in aux_lag_vars
        is_res_var = var.startswith("RES_") # Treat RES_ vars as predetermined states

        if is_aux_lead:
            af_vars.append(var) # Auxiliary leads are forward-looking helpers
        elif is_aux_lag:
            # Aux lags are backward unless they somehow appear with a lead (unlikely)
            if has_lead:
                 if verbose: print(f"  Warning: Aux lag var '{var}' unexpectedly has lead dependency. Treating as Mixed.")
                 mf_vars.append(var) # Should not happen with typical aux defs
            else:
                 pb_vars.append(var)
        elif is_res_var:
             # RES_ vars are typically defined by lags/shocks -> purely backward
             if has_lead:
                 if verbose: print(f"  Warning: RES_ var '{var}' unexpectedly has lead dependency. Treating as Mixed.")
                 mf_vars.append(var) # Should not happen with typical AR(1) defs
             else:
                 pb_vars.append(var)
        else: # Original declared endogenous vars (not RES_)
             if has_lead and not has_lag:
                 # Purely forward endogenous (uncommon, e.g., jump variables not dependent on own lag)
                 # Group with mixed for now unless a clear distinction is needed later
                 mf_vars.append(var)
             elif not has_lead and has_lag:
                 # Purely backward endogenous (depends only on lags)
                 # Group with mixed for now.
                 mf_vars.append(var)
             elif not has_lead and not has_lag:
                 # Static variable
                 s_vars.append(var)
                 if verbose: print(f"  Warning: Identified static variable '{var}'. Static variables should ideally be substituted out for stability.")
             else: # has_lead and has_lag
                 # Mixed forward-backward variable
                 mf_vars.append(var)

    # Construct the final ordered list based on the desired Dynare-like structure:
    # Order: Purely Backward (RES_, aux_lag) -> Mixed Endogenous -> Auxiliary Forward -> Static
    # Ensure RES_ variables come first among purely backward
    pb_res_vars = sorted([v for v in pb_vars if v.startswith("RES_")])
    pb_aux_lag_vars = sorted([v for v in pb_vars if v.startswith("aux_")])
    ordered_pb_vars = pb_res_vars + pb_aux_lag_vars

    # Sort other categories alphabetically for consistency
    ordered_mf_vars = sorted(mf_vars)
    ordered_af_vars = sorted(af_vars)
    ordered_s_vars = sorted(s_vars)

    ordered_vars = ordered_pb_vars + ordered_mf_vars + ordered_af_vars + ordered_s_vars

    # Verification
    if len(ordered_vars) != len(initial_vars_ordered):
        raise ValueError(f"Variable reordering failed (length mismatch). Got {len(ordered_vars)}, expected {len(initial_vars_ordered)}")
    if set(ordered_vars) != set(initial_vars_ordered):
        lost = set(initial_vars_ordered) - set(ordered_vars)
        gained = set(ordered_vars) - set(initial_vars_ordered)
        raise ValueError(f"Variable reordering failed (set mismatch). Lost: {lost}, Gained: {gained}")

    # Permutation indices: mapping from new index -> old index
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]

    if verbose:
        print("\nVariable Classification Results:")
        print(f"  Purely Backward (Predetermined States; {len(ordered_pb_vars)}): {ordered_pb_vars}")
        print(f"  Mixed Fwd/Bwd Endogenous ({len(ordered_mf_vars)}): {ordered_mf_vars}")
        print(f"  Auxiliary Forward ({len(ordered_af_vars)}): {ordered_af_vars}")
        print(f"  Static ({len(ordered_s_vars)}): {ordered_s_vars}")
        print(f"\n==> Final Variable Order ({len(ordered_vars)}): {ordered_vars}")


    # --- Determine New Equation Order (Matching Variables) ---
    if verbose: print("\n--- Determining Equation Order (Matching Backward/Auxiliary Vars) ---")
    eq_perm_indices = []
    used_eq_indices = set()
    assigned_eq_for_var = {} # Track which equation defines which variable

    # Function to find the equation defining a specific variable (more robustly)
    def find_defining_equation(var_to_define, equations_list, current_used_indices):
        # Simple check: look for equation like "(VAR) - (RHS)"
        # Allow for potential spaces around variable name and parentheses
        pattern_lhs = re.compile(fr"^\s*\(\s*{re.escape(var_to_define)}\s*\)\s*-")
        potential_matches = []
        for i, eq_str in enumerate(equations_list):
            if i not in current_used_indices and pattern_lhs.search(eq_str):
                 potential_matches.append(i)

        if len(potential_matches) == 1:
            return potential_matches[0]
        elif len(potential_matches) > 1:
            # If multiple LHS matches, try to find the simplest definition
            # e.g., for aux_VAR_lag = VAR(-1), the equation is just "(aux_VAR_lag) - (VAR(-1))"
            if var_to_define.startswith("aux_"):
                 base_name = base_name_from_aux(var_to_define) # Need helper defined earlier
                 if "_lag_" in var_to_define:
                     target_rhs = f"{base_name}(-1)"
                 elif "_lead_" in var_to_define:
                     # Find the previous aux or base var
                     match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p(\d+)", var_to_define)
                     k = int(match_lead.group(2))
                     prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                     target_rhs = f"{prev_var_for_def}(+1)"
                 else: target_rhs=None

                 if target_rhs:
                      for i in potential_matches:
                           # Check if RHS matches the expected simple definition
                           eq_str = equations_list[i]
                           rhs_part = eq_str.split('-', 1)[1].strip().strip('()')
                           # Simple string comparison for aux definitions
                           if rhs_part == target_rhs:
                               if verbose: print(f"    Refined match for aux '{var_to_define}' to Eq {i} based on RHS.")
                               return i
            # If still ambiguous
            if verbose: print(f"  Warning: Found multiple potential defining eqs for '{var_to_define}': {potential_matches}. Cannot uniquely assign.")
            return None
        else:
            # if verbose: print(f"  Info: Could not find unique LHS defining eq for '{var_to_define}'.")
            return None

    # Helper function base_name_from_aux needed by find_defining_equation
    def base_name_from_aux(aux_name):
        match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p\d+", aux_name)
        if match_lead: return match_lead.group(1)
        match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", aux_name)
        if match_lag: return match_lag.group(1)
        return aux_name # Should not happen if called on aux vars

    # 1. Order equations for Purely Backward variables first (RES_ then aux_lag)
    if verbose: print("  Assigning equations for Purely Backward variables...")
    for var_name in ordered_pb_vars: # Use the ordered list
         eq_idx = find_defining_equation(var_name, processed_equations, used_eq_indices)
         if eq_idx is not None:
             if eq_idx not in used_eq_indices:
                 eq_perm_indices.append(eq_idx)
                 used_eq_indices.add(eq_idx)
                 assigned_eq_for_var[var_name] = eq_idx
                 if verbose: print(f"    Assigned Eq {eq_idx} to var '{var_name}'")
             else:
                  if verbose: print(f"    Skipping already used Eq {eq_idx} for var '{var_name}'")
         else:
             if verbose: print(f"    Could not assign unique equation for '{var_name}'")

    # 2. Order equations for Auxiliary Forward variables next
    if verbose: print("  Assigning equations for Auxiliary Forward variables...")
    for var_name in ordered_af_vars: # Use the ordered list
         eq_idx = find_defining_equation(var_name, processed_equations, used_eq_indices)
         if eq_idx is not None:
            if eq_idx not in used_eq_indices:
                 eq_perm_indices.append(eq_idx)
                 used_eq_indices.add(eq_idx)
                 assigned_eq_for_var[var_name] = eq_idx
                 if verbose: print(f"    Assigned Eq {eq_idx} to var '{var_name}'")
            else:
                  if verbose: print(f"    Skipping already used Eq {eq_idx} for var '{var_name}'")
         else:
             if verbose: print(f"    Could not assign unique equation for '{var_name}'")

    # 3. Add remaining equations (should correspond to Mixed and Static vars)
    if verbose: print("  Assigning remaining equations...")
    initial_indices_set = set(range(num_eq))
    remaining_eq_indices = sorted(list(initial_indices_set - used_eq_indices)) # Sort for consistency
    eq_perm_indices.extend(remaining_eq_indices)
    if verbose: print(f"    Added remaining Eq indices: {remaining_eq_indices}")

    # Validate permutation
    if len(eq_perm_indices) != num_eq:
        raise ValueError(f"Equation permutation construction failed. Length mismatch: {len(eq_perm_indices)} vs {num_eq}. Used: {used_eq_indices}")
    if len(set(eq_perm_indices)) != num_eq:
         duplicates = {x for x in eq_perm_indices if eq_perm_indices.count(x) > 1}
         raise ValueError(f"Equation permutation construction failed. Indices not unique. Duplicates: {duplicates}. Final list: {eq_perm_indices}")

    if verbose: print(f"\n==> Final Equation Order Indices (new row i <- old row eq_perm_indices[i]): {eq_perm_indices}")


    # --- Reorder Numerical Matrices ---
    if verbose: print("\n--- Reordering Numerical Matrices (A, B, C, D) ---")
    # Convert lists to JAX arrays for ix_
    eq_perm_indices_jax = jnp.array(eq_perm_indices)
    var_perm_indices_jax = jnp.array(var_perm_indices)

    # Use JAX arrays in jnp.ix_
    # Permutation: NewMatrix[i,j] = OldMatrix[row_map[i], col_map[j]]
    A_jax = A_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
    B_jax = B_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
    C_jax = C_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
    # D columns correspond to shocks (not variables), so only permute rows
    D_jax = D_unord[jnp.ix_(eq_perm_indices_jax, jnp.arange(num_shocks))]

    if verbose: print("Numerical reordering complete.")

    # Return the ORDERED numerical matrices and associated info
    return (A_jax, B_jax, C_jax, D_jax,
            ordered_vars, shock_names, param_names, param_assignments,
            var_perm_indices, eq_perm_indices, initial_vars_ordered,
            stationary_resid_func_code) # Pass the code string back

# --- Trend/Observation Matrix Builders (Modified to use JAX AD) ---

def build_trend_matrices_jax_ad(trend_equations, trend_vars, trend_shocks, param_names, param_assignments, verbose=True, dtype=_DEFAULT_DTYPE):
    if verbose: print("\n--- Building Trend Matrices (P_trends, Q_trends) via JAX AD ---")
    # Identify state trends (not defined contemporaneously)
    state_trend_vars = []
    defining_equations_for_state_trends = []
    contemporaneous_defs = {} # Store contemporaneous definitions for observation matrix later
    base_var_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')
    var_time_regex_trend = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    for eq_str in trend_equations:
        # Assume format " (LHS) - (RHS) "
        match = re.match(r"\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*-\s*\((.*)\)\s*", eq_str)
        if not match: continue
        lhs_var, rhs_str = match.groups()
        if lhs_var not in trend_vars: continue

        # Check if RHS depends on *other* trend vars at time t or leads
        has_contemporaneous_rhs = False
        # Check explicit time shifts first
        for rmatch in var_time_regex_trend.finditer(rhs_str):
            base, shift = rmatch.group(1), int(rmatch.group(2))
            if base in trend_vars and base != lhs_var and shift >= 0:
                has_contemporaneous_rhs = True; break
        # Check implicit time t vars (excluding LHS var itself)
        if not has_contemporaneous_rhs:
            rhs_symbols = set(base_var_regex.findall(rhs_str))
            for sym in rhs_symbols:
                 # Is it another trend variable appearing without (-1)?
                 if sym in trend_vars and sym != lhs_var and f"{sym}(-1)" not in rhs_str:
                     has_contemporaneous_rhs = True; break

        if has_contemporaneous_rhs:
            contemporaneous_defs[lhs_var] = rhs_str # Store the definition string
            if verbose: print(f"  Identified contemporaneous trend def: {lhs_var} = {rhs_str}")
        else:
            # This defines a state trend variable
            if lhs_var not in state_trend_vars:
                state_trend_vars.append(lhs_var)
                defining_equations_for_state_trends.append(eq_str)

    if not state_trend_vars:
        if verbose: print("Warning: No state trend variables identified. Returning empty matrices.")
        return jnp.empty((0,0), dtype=dtype), jnp.empty((0,0), dtype=dtype), [], {}

    if verbose: print(f"Identified state trend variables ({len(state_trend_vars)}): {state_trend_vars}")
    if verbose: print(f"Using defining equations: {defining_equations_for_state_trends}")

    # Compute matrices for the state trend variables
    trend_matrices, trend_resid_func_code = compute_matrices_jax_ad(
        equations=defining_equations_for_state_trends,
        var_names=state_trend_vars,   # y_t and y_m1 refer to these state trends
        shock_names=trend_shocks,     # shocks_t refer to these trend shocks
        param_names=param_names,
        param_values=param_assignments,
        model_type="trend",
        dtype=dtype
    )

    print("DEBUG")
    print("\n--- Trend Model Residual Function Code ---")
    print(trend_resid_func_code)

    P_trends = trend_matrices['P_trends']
    Q_trends = trend_matrices['Q_trends']

    return P_trends, Q_trends, state_trend_vars, contemporaneous_defs


def build_observation_matrix_jax_ad(measurement_equations, obs_vars, stationary_vars,
                                   trend_state_vars, contemporaneous_trend_defs,
                                   param_names, param_assignments, verbose=True, dtype=_DEFAULT_DTYPE):
    if verbose: print("\n--- Building Observation Matrix (Omega) via JAX AD ---")
    num_obs = len(obs_vars)
    num_stationary = len(stationary_vars)
    num_trend_state = len(trend_state_vars)
    augmented_state_vars = stationary_vars + trend_state_vars # Order matters!
    num_augmented_state = len(augmented_state_vars)

    if num_obs == 0:
        if verbose: print("No observable variables declared. Returning empty Omega.")
        return jnp.empty((0, num_augmented_state), dtype=dtype), []
    if len(measurement_equations) != num_obs:
        raise ValueError(f"Num measurement eqs ({len(measurement_equations)}) != num varobs ({num_obs}).")

    # Substitute contemporaneous trend definitions into measurement equations
    processed_meas_eqs = []
    if verbose: print("Substituting contemporaneous trend definitions into measurement equations...")
    for eq_str in measurement_equations:
        # Assume format " (LHS_obs) - (RHS) "
        match = re.match(r"\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*-\s*\((.*)\)\s*", eq_str)
        if not match: continue
        lhs_obs, rhs_str = match.groups()
        rhs_processed = rhs_str

        # Iteratively substitute definitions (simple, might not handle complex nesting)
        made_substitution = True
        iter_count = 0
        max_iter = 10 # Safety break
        while made_substitution and iter_count < max_iter:
            made_substitution = False
            iter_count += 1
            for contemp_var, contemp_expr_str in contemporaneous_trend_defs.items():
                pattern = r'\b' + re.escape(contemp_var) + r'\b'
                if re.search(pattern, rhs_processed):
                    # Substitute with parentheses for safety
                    rhs_processed = re.sub(pattern, f"({contemp_expr_str})", rhs_processed)
                    made_substitution = True
            if not made_substitution: break
        if iter_count == max_iter: print(f"Warning: Max substitution iterations reached for eq: {eq_str}")

        processed_eq_str = f"({lhs_obs}) - ({rhs_processed})"
        processed_meas_eqs.append(processed_eq_str)
        # if verbose and rhs_processed != rhs_str:
        #      print(f"  Substituted Eq: {processed_eq_str}")

    if verbose: print(f"Final measurement equations for JAX AD: {processed_meas_eqs}")

    # Compute Omega matrix using JAX AD
    obs_matrix_dict, obs_resid_func_code = compute_matrices_jax_ad(
        equations=processed_meas_eqs,
        var_names=obs_vars,              # 'obs_t' in resid func corresponds to these
        shock_names=augmented_state_vars,# 'state_t' in resid func corresponds to these
        param_names=param_names,
        param_values=param_assignments,
        model_type="observation",
        dtype=dtype
    )

    print("DEBUG")
    print("\n--- Obs Model Residual Function Code ---")
    print(obs_resid_func_code)
    Omega = obs_matrix_dict['Omega']

    return Omega, obs_vars # Return computed Omega and the ordered obs vars

# --- IRF Calculation Functions (JAX - UNCHANGED) ---
# [Your existing irf and irf_observables functions]
def irf(P: ArrayLike, R: ArrayLike, shock_index: int, horizon: int = 40) -> jax.Array:
    """
    Compute impulse responses for y_t = P y_{t-1} + R eps_t (eps_t ~ N(0,I)).
    Uses JAX arrays and lax.scan.

    Args:
        P: State transition matrix (n x n).
        R: Shock impact matrix (n x n_shock).
        shock_index: Index of the shock to perturb (0-based).
        horizon: Number of periods for the IRF.

    Returns:
        jax.Array: IRF values (horizon x n).
    """
    P_jax = jnp.asarray(P); R_jax = jnp.asarray(R)
    n = P_jax.shape[0];
    # Handle case where R might be empty or 1D
    if R_jax.ndim == 0 or R_jax.size == 0: n_shock = 0
    elif R_jax.ndim == 1: n_shock = 1
    else: n_shock = R_jax.shape[1]

    dtype = P_jax.dtype
    if n_shock == 0: return jnp.zeros((horizon, n), dtype=dtype) # No response if no shocks
    if not (0 <= shock_index < n_shock): raise ValueError(f"shock_index ({shock_index}) out of range [0, {n_shock})")

    y_resp = jnp.zeros((horizon, n), dtype=dtype)
    # Unit shock impulse at time 0
    if R_jax.ndim == 1:
        impulse = R_jax if shock_index == 0 else jnp.zeros_like(R_jax)
    else:
        impulse = R_jax[:, shock_index] # Select column corresponding to shock

    # y_0 = P*y_{-1} + impulse. Assume y_{-1} = 0.
    y_current = impulse
    y_resp = y_resp.at[0, :].set(y_current)

    def step(y_prev, _):
        y_next = P_jax @ y_prev
        return y_next, y_next

    # Scan for remaining steps
    # Handle horizon=1 case
    if horizon > 1:
        _, y_resp_scan = jax.lax.scan(step, y_current, xs=None, length=horizon - 1)
        y_resp = y_resp.at[1:, :].set(y_resp_scan)

    # Clean very small values
    y_resp_clean = jnp.where(jnp.abs(y_resp) < 1e-14, 0.0, y_resp)
    return y_resp_clean

def irf_observables(P_aug: ArrayLike, R_aug: ArrayLike, Omega: ArrayLike,
                     shock_index: int, horizon: int = 40) -> jax.Array:
    """
    Compute impulse responses for observable variables.
    obs(t) = Omega * state_aug(t)

    Args:
        P_aug: Augmented state transition matrix (n_aug x n_aug).
        R_aug: Augmented shock impact matrix (n_aug x n_aug_shock).
        Omega: Observation matrix (n_obs x n_aug).
        shock_index: Index of the shock in the *augmented* shock vector.
        horizon: Number of periods.

    Returns:
        jax.Array: Observable IRF values (horizon x n_obs).
    """
    P_aug_jax = jnp.asarray(P_aug); R_aug_jax = jnp.asarray(R_aug); Omega_jax = jnp.asarray(Omega)
    n_aug = P_aug_jax.shape[0];
    n_obs = Omega_jax.shape[0]

    # Handle empty/1D R_aug
    if R_aug_jax.ndim == 0 or R_aug_jax.size == 0: n_aug_shock = 0
    elif R_aug_jax.ndim == 1: n_aug_shock = 1
    else: n_aug_shock = R_aug_jax.shape[1]

    if n_aug_shock == 0: return jnp.zeros((horizon, n_obs), dtype=P_aug_jax.dtype)
    if not (0 <= shock_index < n_aug_shock): raise ValueError(f"Aug shock_index ({shock_index}) out of range [0, {n_aug_shock})")
    if Omega_jax.shape[1] != n_aug: raise ValueError(f"Omega columns ({Omega_jax.shape[1]}) != P_aug dim ({n_aug}).")
    if n_obs == 0: return jnp.zeros((horizon, 0), dtype=P_aug_jax.dtype) # No obs IRFs if no obs

    # Compute state IRF using the JAX function
    state_irf = irf(P_aug_jax, R_aug_jax, shock_index, horizon) # (horizon, n_aug)

    # Map state responses to observables: obs_irf = state_irf @ Omega.T
    # Ensure Omega is 2D even if n_obs=1
    if Omega_jax.ndim == 1: Omega_jax = Omega_jax.reshape(1, -1)
    obs_irf = state_irf @ Omega_jax.T # (horizon, n_obs)

    # Clean very small values
    obs_irf_clean = jnp.where(jnp.abs(obs_irf) < 1e-14, 0.0, obs_irf)
    return obs_irf_clean

# This function only is use to setup the initial conditions of the simulation
def construct_initial_state(
    n_aug: int,
    n_stat: int,
    aug_state_vars: List[str],
    key_init: jax.random.PRNGKey,
    initial_state_config: Optional[Dict[str, Dict[str, float]]] = None,
    default_trend_std: float = 0.01, # Default std for unspecified trends
    dtype: jnp.dtype = _DEFAULT_DTYPE
) -> jax.Array:
    """
    Constructs the initial state vector s0 for simulation or filtering.

    Allows setting specific means and standard deviations for trend variables
    via the initial_state_config dictionary. Stationary variables default to zero mean.

    Args:
        n_aug: Total number of augmented state variables.
        n_stat: Number of stationary variables.
        aug_state_vars: Ordered list of all augmented state variable names.
        key_init: JAX PRNG key for drawing initial random components.
        initial_state_config: Dictionary specifying initial states, e.g.,
            {
                "VAR_NAME_1": {"mean": value1, "std": std_dev1},
                "VAR_NAME_2": {"mean": value2}, # std defaults to default_trend_std
                "VAR_NAME_3": {"std": std_dev3}, # mean defaults to 0.0
                ...
            }. Applies primarily to trend variables.
        default_trend_std: Default standard deviation for trend variables not
                           specified in the config.
        dtype: JAX dtype for the state vector.

    Returns:
        jax.Array: The constructed initial state vector s0 (size n_aug).
    """
    n_trend = n_aug - n_stat
    s0 = jnp.zeros(n_aug, dtype=dtype) # Start with zeros for all states

    if n_trend > 0:
        # Set defaults for ALL trend variables first
        trend_means = jnp.zeros(n_trend, dtype=dtype)
        trend_stds = jnp.full(n_trend, default_trend_std, dtype=dtype)

        if initial_state_config:
            print("\n--- Applying Initial State Configuration for Trends ---")
            trend_state_vars = aug_state_vars[n_stat:] # Names of trend vars only

            for var_name, config in initial_state_config.items():
                if not isinstance(config, dict):
                    print(f"Warning: Invalid config format for '{var_name}' (expected dict). Skipping.")
                    continue

                if var_name in trend_state_vars:
                    try:
                        # Find index *within the trend block*
                        trend_idx = trend_state_vars.index(var_name)

                        # Get mean and std, falling back to defaults if not provided
                        mean_val = config.get("mean")
                        std_val = config.get("std")

                        if mean_val is not None:
                            val = float(mean_val)
                            trend_means = trend_means.at[trend_idx].set(val)
                            print(f"  Set initial mean for '{var_name}' (trend idx {trend_idx}) to: {val:.4f}")
                        # else: Mean remains default 0.0

                        if std_val is not None:
                            val = jnp.maximum(0.0, float(std_val)) # Ensure non-negative
                            trend_stds = trend_stds.at[trend_idx].set(val)
                            print(f"  Set initial std dev for '{var_name}' (trend idx {trend_idx}) to: {val:.4f}")
                        # else: Std dev remains default_trend_std

                    except ValueError: # Should not happen if var_name in trend_state_vars
                        print(f"Warning: Internal error finding index for trend var '{var_name}'.")
                    except (TypeError, KeyError) as e:
                        print(f"Warning: Invalid format/value in initial state config for '{var_name}': {config}. Error: {e}. Skipping.")
                else:
                    # Optionally warn if config given for a stationary var or unknown var
                    if var_name in aug_state_vars[:n_stat]:
                        print(f"Info: Initial state config provided for stationary var '{var_name}'. Ignoring (mean defaults to 0).")
                    else:
                        print(f"Warning: Variable '{var_name}' in initial_state_config not found in trend state list. Skipping.")


        # Draw initial trends from N(mean, std^2) using the final mean/std arrays
        print("\nDrawing initial trend states from N(mean, std^2)...")
        initial_trends = trend_means + trend_stds * jax.random.normal(key_init, (n_trend,), dtype=dtype)

        # Place the drawn values into the s0 vector
        s0 = s0.at[n_stat:].set(initial_trends)
        print(f"Initial state vector s0 constructed (shape {s0.shape}).")

    else:
        print("No trend variables, initial state s0 defaults to zeros.")


    return s0


# --- Main Execution Guard (Example Usage - Modified to use JAX AD path) ---
if __name__ == "__main__":
    print("--- Dynare Parser, JAX AD Solver, and Simulator Script ---")
    print("Running example parse/solve/IRF/simulate using JAX AD for matrices.")
    try:
        # --- Setup: Find model file ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the provided example file name
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        if not os.path.exists(mod_file_path):
             raise FileNotFoundError(f"Example model file not found: {mod_file_path}")
        with open(mod_file_path, 'r') as f:
            model_def = f.read()
        print(f"Loaded model from: {mod_file_path}")

        # --- STEP 1: Parse Stationary Model & Compute Matrices via JAX AD ---
        print("\n--- [1] Parsing Stationary Model & Computing Matrices (JAX AD) ---")
        (A_num_stat, B_num_stat, C_num_stat, D_num_stat, ordered_stat_vars, # New
        stat_shocks, param_names_all, param_assignments_all,
        var_perm_indices, eq_perm_indices, initial_stat_vars,
        stat_resid_code # Capture the code string if needed later
        ) = parse_and_compute_matrices_jax_ad(model_def, verbose=True)

        n_stat = len(ordered_stat_vars)
        n_s_shock = len(stat_shocks)

        # print(f"Found {n_stat} ordered stationary vars, {n_s_shock} stationary shocks.")
        # print(f"Computed A({A_num_stat.shape}), B({B_num_stat.shape}), C({C_num_stat.shape}), D({D_num_stat.shape}) via JAX AD.")

        # Add print statement for A, B, C, D
        print("\n--- Computed Stationary Matrices (Ordered, JAX AD) ---")
        print(f"A_num_stat (shape {A_num_stat.shape}):\n{A_num_stat}")
        print(f"B_num_stat (shape {B_num_stat.shape}):\n{B_num_stat}")
        print(f"C_num_stat (shape {C_num_stat.shape}):\n{C_num_stat}")
        print(f"D_num_stat (shape {D_num_stat.shape}):\n{D_num_stat}")

        # # Add print statements in STEP 4 after computing trend/obs matrices


        # --- STEP 2: Parse Trends/Observations Declarations & Equations ---
        print("\n--- [2] Parsing Trend/Observation Declarations & Equations ---")
        trend_vars, trend_shocks = extract_trend_declarations(model_def)
        trend_equations = extract_trend_equations(model_def)
        obs_vars = extract_observation_declarations(model_def)
        measurement_equations = extract_measurement_equations(model_def)
        trend_stderr_params = extract_trend_shock_stderrs(model_def)
        n_t_shock = len(trend_shocks)
        n_obs = len(obs_vars)
        print(f"Found {len(trend_vars)} trend vars, {n_t_shock} trend shocks, {n_obs} observable vars.")

        # --- STEP 3: Combine Parameters (Ensure consistency) ---
        # param_names_all and param_assignments_all already contain combined params
        # from the stationary parse step. We just need trend sigmas.
        param_assignments_all.update(trend_stderr_params)
        # Add trend sigmas to the name list if they weren't declared elsewhere
        for p_sigma in trend_stderr_params.keys():
            if p_sigma not in param_names_all:
                param_names_all.append(p_sigma)
        # Ensure defaults for trend sigmas if not set in trend_shocks block
        inferred_trend_sigmas = [f"sigma_{shk}" for shk in trend_shocks]
        for p_sigma in inferred_trend_sigmas:
            if p_sigma not in param_assignments_all:
                 print(f"Warning: Inferred trend sigma '{p_sigma}' using default 1.0.")
                 param_assignments_all[p_sigma] = 1.0
                 if p_sigma not in param_names_all: param_names_all.append(p_sigma)

        print(f"Total combined parameters: {len(param_names_all)}")

        # Apply example parameter overrides (as in original script)
        param_overrides = {
            'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12, 'g1': 0.7,
            'g2': 0.3, 'g3': 0.25, 'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
            'rho_rs': 0.75, 'rho_rs2': 0.01
        }
        param_assignments_all.update(param_overrides)
        print(f"Applied parameter overrides.")

        # --- STEP 4: Build Trend/Observation Matrices via JAX AD ---
        print("\n--- [4] Building Trend/Observation Matrices (JAX AD) ---")
        P_num_trend, Q_num_trend, ordered_trend_state_vars, contemp_trend_defs = build_trend_matrices_jax_ad(
            trend_equations, trend_vars, trend_shocks, param_names_all, param_assignments_all, verbose=True)
        n_trend = len(ordered_trend_state_vars)

        Omega_num, ordered_obs_vars = build_observation_matrix_jax_ad(
            measurement_equations, obs_vars, ordered_stat_vars, ordered_trend_state_vars,
            contemp_trend_defs, param_names_all, param_assignments_all, verbose=True)

        print(f"Identified {n_trend} trend state vars.")
        print(f"Computed P_trend({P_num_trend.shape}), Q_trend({Q_num_trend.shape}), Omega({Omega_num.shape}) via JAX AD.")

        print("\n--- Computed Trend/Observation Matrices (JAX AD) ---")
        print(f"P_num_trend (shape {P_num_trend.shape}):\n{P_num_trend}")
        print(f"Q_num_trend (shape {Q_num_trend.shape}):\n{Q_num_trend}")
        print(f"Omega_num (shape {Omega_num.shape}):\n{Omega_num}")

        # --- STEP 5: Solve Stationary Model (JAX SDA - using matrices from Step 1) ---
        print("\n--- [5] Solving Stationary Model (JAX SDA) ---")
        P_sol_stat, iter_count, residual_ratio, converged = solve_quadratic_matrix_equation_jax(
            A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500)

        if not converged or not jnp.all(jnp.isfinite(P_sol_stat)):
            raise RuntimeError(f"JAX SDA solver failed! Converged: {converged}, Residual Ratio: {residual_ratio:.2e}")
        else:
            print(f"JAX SDA converged. Residual ratio: {residual_ratio:.2e}")
            Q_sol_stat = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
            print("Q_stationary computed.")

        # # Add print statements in STEP 5 after solving for P_sol_stat and Q_sol_stat
        print("\n--- Solution Matrices for Stationary Model ---")
        print(f"P_sol_stat (shape {P_sol_stat.shape}):\n{P_sol_stat}")
        print(f"Q_sol_stat (shape {Q_sol_stat.shape}):\n{Q_sol_stat}")

        # --- STEP 6: Build R Matrices (Apply Std Devs) & Augmented System ---
        print("\n--- [6] Building R Matrices and Augmented System ---")
        shock_std_devs = {}
        aug_shocks = stat_shocks + trend_shocks
        for shock_name in aug_shocks:
            sigma_param_name = f"sigma_{shock_name}"
            std_dev = param_assignments_all.get(sigma_param_name, 1.0)
            if sigma_param_name not in param_assignments_all: # Should have defaults now
                print(f"Warning: Sigma param '{sigma_param_name}' missing, using 1.0 for '{shock_name}'.")
            shock_std_devs[shock_name] = std_dev

        stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in stat_shocks], dtype=P_sol_stat.dtype)
        trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in trend_shocks], dtype=P_sol_stat.dtype)

        # Check dimensions before matrix multiplication
        if Q_sol_stat.shape[1] != len(stat_std_devs_arr):
             print(f"Warning: Q_stat shape {Q_sol_stat.shape} mismatch with stat shocks {len(stat_std_devs_arr)}")
             # Handle empty case
             R_sol_stat = jnp.zeros((n_stat, 0), dtype=P_sol_stat.dtype) if n_s_shock == 0 else Q_sol_stat @ jnp.diag(stat_std_devs_arr)
        else:
             R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if n_s_shock > 0 else jnp.zeros((n_stat, 0), dtype=P_sol_stat.dtype)

        if Q_num_trend.shape[1] != len(trend_std_devs_arr):
             print(f"Warning: Q_trend shape {Q_num_trend.shape} mismatch with trend shocks {len(trend_std_devs_arr)}")
             R_num_trend = jnp.zeros((n_trend, 0), dtype=P_sol_stat.dtype) if n_t_shock == 0 else Q_num_trend @ jnp.diag(trend_std_devs_arr)
        else:
             R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if n_t_shock > 0 else jnp.zeros((n_trend, 0), dtype=P_sol_stat.dtype)

        # Build Augmented P_aug, R_aug
        n_aug = n_stat + n_trend
        n_aug_shock = n_s_shock + n_t_shock
        aug_state_vars = ordered_stat_vars + ordered_trend_state_vars # Combined state vector order

        P_aug = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
        R_aug = jnp.zeros((n_aug, n_aug_shock), dtype=P_aug.dtype)
        if n_stat > 0 and n_s_shock > 0 and R_sol_stat.shape == (n_stat, n_s_shock): 
            R_aug = R_aug.at[:n_stat, :n_s_shock].set(R_sol_stat)
        if n_trend > 0 and n_t_shock > 0 and R_num_trend.shape == (n_trend, n_t_shock): 
            R_aug = R_aug.at[n_stat:, n_s_shock:].set(R_num_trend)
        
        print(f"Augmented system built: n_aug={n_aug}, n_aug_shock={n_aug_shock}")
        print(f"Augmented state order: {aug_state_vars}")
        print(f"Augmented shock order: {aug_shocks}")


        print("\n--- Augmented System Matrices ---")
        print(f"P_aug (shape {P_aug.shape}):\n{P_aug}")
        print(f"R_aug (shape {R_aug.shape}):\n{R_aug}")
             
        # --- STEP 7: Compute IRFs (Example) ---
        print("\n--- [7] Computing Example IRFs (JAX) ---")
        shock_name_to_plot = "SHK_RS" # Example shock
        if shock_name_to_plot in aug_shocks:
            shock_index_aug = aug_shocks.index(shock_name_to_plot)
            horizon = 40

            irf_states_aug = irf(P_aug, R_aug, shock_index=shock_index_aug, horizon=horizon)
            irf_observables_vals = irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index_aug, horizon=horizon)
            print(f"IRFs computed for shock '{shock_name_to_plot}'.")

            # Plotting IRFs (Optional)
            print("\nDisplaying IRF plots...")
            plot_irfs(irf_observables_vals, ordered_obs_vars, horizon, title=f"Observable IRFs to {shock_name_to_plot} (JAX AD)")
            plot_irfs(irf_states_aug, aug_state_vars, horizon, title=f"Augmented State IRFs to {shock_name_to_plot} (JAX AD)")
            plt.show(block=False) # Non-blocking for subsequent plots
        else:
            print(f"Warning: Shock '{shock_name_to_plot}' not found in augmented shocks: {aug_shocks}. Skipping IRF example.")


        # --- STEP 8: Simulate Data ---
        print("\n--- [8] Simulating Data ---")
        T_sim = 200 # Simulation length
        key_master = jax.random.PRNGKey(42)
        key_init, key_sim = jax.random.split(key_master)

        # Construct initial state (Example: trends start near zero)

        T_sim = 200 # Simulation length
        key_master = jax.random.PRNGKey(42)
        key_init, key_sim = jax.random.split(key_master)

        # --- Define Initial State Configuration ---
        # Example: Set specific means/stds for some trend variables
        initial_state_configuration = {
            "L_GDP_TREND": {"mean": 10000.0, "std": 0.01},
            "G_TREND":     {"mean": 2.0/4.0,  "std": 0.01}, # Start growth slightly positive
            "PI_TREND":    {"mean": 2.0, "std": 0.00},              # Std defaults to default_trend_std (0.01)
            "RR_TREND":    {"mean": 1.0, "std": 0.01}               # Mean defaults to 0.0            
        }

        # --- Construct Initial State using the helper function ---
        s0 = construct_initial_state(
            n_aug=n_aug,
            n_stat=n_stat,
            aug_state_vars=aug_state_vars,
            key_init=key_init, # Use the specific key for initialization draws
            initial_state_config=initial_state_configuration,
            default_trend_std=0.01, # Default std for trends if not specified above
            dtype=_DEFAULT_DTYPE
        )

        print(f"Initial state s0 constructed.")
        print(f"Simulating {T_sim} periods...")

        measurement_noise_level = 0.0 # No measurement noise in this example run

        sim_states, sim_observables = simulate_ssm_data(
            P=P_aug, R=R_aug, Omega=Omega_num, T=T_sim, key=key_sim,
            state_init=s0, measurement_noise_std=measurement_noise_level
        )
        print("Simulation complete.")
        print(f"Simulated states shape: {sim_states.shape}")
        print(f"Simulated observables shape: {sim_observables.shape}")

        # --- STEP 9: Plot Simulation Results ---
        print("\n--- [9] Plotting Simulation Results ---")
        plot_simulation(sim_observables, ordered_obs_vars, title=f"Simulated Observables (JAX AD, T={T_sim})")

        # Plot observables with matched trends
        # Correctly identify trend state names from the ordered list        
        trend_state_names_plot = ordered_trend_state_vars # Actual state trends
        # This mapping can now include contemporaneous trends like RS_TREND
        mapping_sim = {
            "L_GDP_OBS": "L_GDP_TREND",
            "DLA_CPI_OBS": "PI_TREND",
            "PI_TREND_OBS": "PI_TREND", # PI_TREND is a state var
            "RS_OBS": "RS_TREND",      # RS_TREND is defined contemporaneously
        }

        # Pass the contemporaneous definitions dictionary
        plot_simulation_with_trends_matched(
            sim_observables=sim_observables,
            obs_var_names=ordered_obs_vars,
            sim_states=sim_states,
            state_var_names=aug_state_vars, # Full augmented state list
            trend_state_names=trend_state_names_plot, # Just the names of trends in the state
            contemporaneous_trend_defs=contemp_trend_defs, # Pass the definitions!
            mapping=mapping_sim,
            title="Simulated Observables and Trends (JAX AD)"
        )

        # Show all plots together
        print("\nDisplaying all plots...")
        plt.show() # Blocking call for the final plots

        

    # --- Error Handling ---
    except FileNotFoundError as e: print(f"\nError: {e}")
    except ValueError as e: print(f"\nValueError: {e}"); import traceback; traceback.print_exc()
    except RuntimeError as e: print(f"\nRuntimeError: {e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nUnexpected Error: {e}"); import traceback; traceback.print_exc()


# --- END OF FILE ---

