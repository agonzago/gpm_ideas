# --- START OF FILE Dynare_parser_sda_solver.py ---

# -*- coding: utf-8 -*-
"""
Enhanced Dynare Parser and State-Space Solver (JAX Compatible)

Parses Dynare-like models, solves using JAX-based SDA, builds augmented
state-space, computes IRFs.
"""

import re
import sympy
import numpy as np
from collections import OrderedDict, namedtuple
import copy
import os
from numpy.linalg import norm # Still use numpy norm for final check in non-JAX solver if needed
from scipy.linalg import lu_factor, lu_solve, block_diag # block_diag used in non-JAX build_augmented
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.typing import ArrayLike
from typing import Tuple, Optional, List, Dict, Any
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
print(f"Using JAX with dtype: {_DEFAULT_DTYPE}")

# --- Helper Functions (Mostly unchanged) ---

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
        print(
            "[simulate_ssm_data] Warning: No state_init provided! Initial state s_0 defaults to zeros. "
            "This may be unsuitable for models with trends or non-zero steady states. "
            "Provide a specific state_init for accurate simulation."
        )
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
    mapping: dict = None,                  # {"L_GDP_OBS": "L_GDP_TREND", ...}
    title="Simulated Observables and Trends"
):
    """
    Plots observables and their "main" trend side-by-side. If `mapping` is None, guesses by substring match.
    """
    sim_obs_np = np.asarray(sim_observables)
    sim_states_np = np.asarray(sim_states)
    time = np.arange(sim_obs_np.shape[0])

    if mapping is None:
        # Guess a mapping by substring; e.g., L_GDP_OBS -> L_GDP_TREND
        mapping = {}
        for obs in obs_var_names:
            for trend in trend_state_names:
                if obs.split('_')[0] in trend:
                    mapping[obs] = trend
                    break

    cols = 2
    rows = (len(obs_var_names) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(min(6 * cols, 18), 4 * rows), squeeze=False)
    axes = axes.flatten()
    plt.suptitle(title, fontsize=14)

    for i, obs_name in enumerate(obs_var_names):
        ax = axes[i]
        obs_idx = obs_var_names.index(obs_name)
        ax.plot(time, sim_obs_np[:, obs_idx], label=f"{obs_name}", linewidth=2)
        # Plot matching trend if found
        trend_name = mapping.get(obs_name)
        if trend_name and trend_name in state_var_names:
            trend_idx = state_var_names.index(trend_name)
            ax.plot(time, sim_states_np[:, trend_idx], label=f"{trend_name} (Trend)", linestyle='--', alpha=0.85)
        ax.legend()
        ax.grid(True)
    # Hide any empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Main functions of the parser 
def create_timed_symbol(base_name, time_shift):
    if time_shift == -1:
        return sympy.symbols(f"{base_name}_m1")
    elif time_shift == 1:
        return sympy.symbols(f"{base_name}_p1")
    elif time_shift == 0: 
        return sympy.symbols(base_name)
    else: 
        raise ValueError(f"Unexpected time shift {time_shift} for {base_name}")

def base_name_from_aux(aux_name):
    match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p\d+", aux_name)
    if match_lead: 
        return match_lead.group(1)
    match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", aux_name)
    if match_lag: 
        return match_lag.group(1)
    return aux_name

def symbolic_jacobian(equations, variables):
    num_eq = len(equations); num_var = len(variables)
    jacobian = sympy.zeros(num_eq, num_var)
    for i, eq in enumerate(equations):
        for j, var in enumerate(variables):
            jacobian[i, j] = sympy.diff(eq, var)
    return jacobian

def robust_lambdify(args, expr, modules='jax'):
    # Ensure expr is Sympy Matrix for consistency checks
    if not isinstance(expr, (sympy.Matrix, sympy.ImmutableMatrix)):
         if isinstance(expr, (int, float, np.number, jnp.number)):
              # If it's a scalar number, wrap it in a lambda
              return lambda *a: jnp.array(expr)
         elif isinstance(expr, (np.ndarray, jnp.ndarray)):
              # If it's already an array, wrap it
              return lambda *a: jnp.asarray(expr)
         else:
             # Try converting other types to Sympy Matrix if possible
             try: expr = sympy.Matrix(expr)
             except Exception: pass # Fall through if conversion fails


    if isinstance(expr, (sympy.Matrix, sympy.ImmutableMatrix)):
        if not expr.free_symbols:
            try:
                numerical_matrix = jnp.array(expr.tolist(), dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
                return lambda *a: numerical_matrix
            except (TypeError, ValueError) as e:
                 print(f"Warning: Could not convert symbol-free matrix {expr} to JAX array: {e}")

    try:
        # Use JAX backend
        # Need to map Sympy Matrix to jnp.array explicitly for some cases
        jax_modules = [{'ImmutableDenseMatrix': jnp.array, 'MutableDenseMatrix': jnp.array}, 'jax']
        return sympy.lambdify(args, expr, modules=jax_modules)
    except Exception as e:
        print(f"Error during lambdify (using JAX backend). Arguments: {args}")
        print(f"Expression causing error:\n{sympy.pretty(expr)}") # Pretty print expression
        raise e

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



def parse_lambdify_and_order_model(model_string, verbose=True):
    """
    Parses the stationary part of the model, handles leads/lags,
    orders variables/equations, generates symbolic matrices for the
    quadratic form A P^2 + B P + C = 0, and returns lambdified JAX functions.

    Args:
        model_string (str): The full content of the Dynare file.
        verbose (bool): If True, prints progress information.

    Returns:
        tuple: Contains:
            - func_A, func_B, func_C, func_D: Lambdified JAX functions for ordered matrices.
            - ordered_vars (List[str]): Ordered list of stationary state variables.
            - shock_names (List[str]): List of stationary shock names.
            - param_names (List[str]): List of combined stationary parameter names.
            - param_assignments (Dict[str, float]): Dictionary of default parameter values.
            - symbolic_matrices_ordered (Dict[str, sympy.Matrix]): Ordered symbolic matrices.
            - initial_info (Dict): Information about the initial (unordered) symbolic setup.
    """
    if verbose:  print("--- Parsing Stationary Model Declarations ---")
    declared_vars, shock_names, param_names_declared, param_assignments_initial = extract_declarations(model_string)
    inferred_sigma_params = [f"sigma_{shk}" for shk in shock_names]
    stat_stderr_values = extract_stationary_shock_stderrs(model_string)
    #sigma_stderr_assignments = {f"sigma_{shk}": val for shk, val in stat_stderr_values.items()}
    sigma_stderr_assignments = stat_stderr_values.copy()

    # --- Step 1d: Combine parameter NAMES ---
    combined_param_names = list(dict.fromkeys(param_names_declared).keys())
    added_sigmas = []
    for p_sigma in inferred_sigma_params:
        if p_sigma not in combined_param_names:
            combined_param_names.append(p_sigma)
            added_sigmas.append(p_sigma)
    # Optional: Print added sigmas only if verbose and list is not empty
    # if verbose and added_sigmas: print(f"   Added inferred stationary sigma parameters: {added_sigmas}")

    # --- Step 1e: Combine parameter ASSIGNMENTS ---
    combined_param_assignments = sigma_stderr_assignments.copy()
    combined_param_assignments.update(param_assignments_initial)
    # Ensure defaults exist for added sigma params if not set
    for p_sigma in added_sigmas:
        if p_sigma not in combined_param_assignments:
             if verbose: print(f"Warning: Inferred sigma parameter '{p_sigma}' has no default value. Setting default to 1.0.")
             combined_param_assignments[p_sigma] = 1.0

    # --- Use the combined lists from now on ---
    param_names = combined_param_names
    param_assignments = combined_param_assignments

    if verbose:
        # Simplified verbose output
        print(f"Declared Variables ({len(declared_vars)}), Shocks ({len(shock_names)}), Declared Params ({len(param_names_declared)})")
        print(f"==> Final Parameter List for Stationary Model ({len(param_names)}): {param_names}")
        print(f"==> Combined Initial Parameter Assignments: {param_assignments}")

    if verbose: print("\n--- Parsing Stationary Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    if verbose: print(f"Found {len(raw_equations)} equations in model block.")

    # --- Handling Leads/Lags & Auxiliaries ---
    if verbose:  print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars)
    aux_variables = OrderedDict() # Stores definition string for each aux var
    processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]
        eq_idx += 1
        modified_eq = eq
        matches = list(var_time_regex.finditer(eq))

        # Process matches in reverse order to avoid index issues
        for match in reversed(matches):
            base_name = match.group(1)
            time_shift = int(match.group(2))

            # Skip if not an endogenous variable or already processed aux
            if base_name not in endogenous_vars and base_name not in aux_variables:
                continue

            # --- Handle Leads > 1 ---
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        # Definition: aux_k = prev_var(+1) => aux_k - prev_var(+1) = 0
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(+1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)" # Replace original var(t+s) with aux(t+1)
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]

                # Add the definition equations if they aren't already there
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        processed_equations.append(def_eq)

            # --- Handle Lags < -1 ---
            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        # Definition: aux_k = prev_var(-1) => aux_k - prev_var(-1) = 0
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(-1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)" # Replace original var(t-s) with aux(t-1)
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]

                # Add the definition equations if they aren't already there
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        processed_equations.append(def_eq)

        if modified_eq != eq:
            processed_equations[eq_idx - 1] = modified_eq
            # if verbose: print(f"  Updated Eq {eq_idx-1}: {modified_eq}") # Can be very verbose

    initial_vars_ordered = list(endogenous_vars) # All vars including auxiliaries
    num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations)
    num_shocks = len(shock_names)

    if verbose: print(f"Total variables after processing leads/lags ({num_vars}): {initial_vars_ordered}")
    # if verbose: print(f"Total equations after processing leads/lags ({num_eq}):") # Avoid printing all eqs
    # for i, eq in enumerate(processed_equations): print(f"  Eq {i}: {eq}")

    if num_vars != num_eq:
        # Provide more context in the error message
        print("\nError Details:")
        print(f"  Original Declared Vars: {declared_vars}")
        print(f"  Auxiliary Vars Added: {list(aux_variables.keys())}")
        print(f"  Final Variable List ({num_vars}): {initial_vars_ordered}")
        print(f"\n  Original Equations: {raw_equations}")
        print(f"  Auxiliary Equations Added: {list(aux_variables.values())}")
        print(f"  Final Equation List ({num_eq}): {processed_equations}")
        raise ValueError(
            f"Stationary model not square after processing leads/lags: {num_vars} vars vs {num_eq} eqs."
        )
    if verbose: print("Stationary model is square.")

    # --- Symbolic Representation ---
    if verbose: print("\n--- Creating Symbolic Representation (Stationary Model) ---")
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names}
    var_syms = {} # Holds {'var': {'m1': sym, 't': sym, 'p1': sym}}
    all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in initial_vars_ordered:
        sym_m1 = create_timed_symbol(var, -1)
        sym_t  = create_timed_symbol(var, 0)
        sym_p1 = create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
        all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])

    local_dict = {str(s): s for s in all_syms_for_parsing}
    # Add common math functions known by sympy
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    # Add others if your models use them: 'ln', 'Abs', etc.

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                          implicit_multiplication_application, rationalize)
    # Define standard transformations for the parser
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))

    sym_equations = []
    if verbose: print("Parsing stationary equations into symbolic form...")
    for i, eq_str in enumerate(processed_equations):
        eq_str_sym = eq_str # Start with the string equation

        # Define a function to replace timed variables (var(time)) with their symbolic counterparts
        def replace_var_time(match):
            base_name, time_shift_str = match.groups()
            time_shift = int(time_shift_str)
            if base_name in shock_names:
                if time_shift == 0: return str(shock_syms[base_name])
                else: raise ValueError(f"Shock {base_name}({time_shift}) invalid in stationary model eq {i}: '{eq_str}'. Shocks are contemporaneous.")
            elif base_name in var_syms: # Endogenous or auxiliary variable
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0:  return str(var_syms[base_name]['t'])
                if time_shift == 1:  return str(var_syms[base_name]['p1'])
                # Should not happen after auxiliary variable processing
                raise ValueError(f"Unexpected time shift {time_shift} for variable {base_name} in eq {i}: '{eq_str}' after aux processing.")
            elif base_name in param_syms:
                raise ValueError(f"Parameter {base_name}({time_shift}) is invalid syntax in eq {i}: '{eq_str}'. Parameters are time-invariant.")
            elif base_name in local_dict: # e.g. log, exp - leave as is
                return match.group(0)
            else: # Unknown symbol with time shift - potential error or undeclared var
                # Create a symbolic representation but warn
                timed_sym_str = str(create_timed_symbol(base_name, time_shift))
                if timed_sym_str not in local_dict:
                    print(f"Warning: Symbol '{base_name}' with time shift {time_shift} in eq {i} ('{eq_str}') is undeclared. Treating symbolically as '{timed_sym_str}'.")
                    local_dict[timed_sym_str] = sympy.symbols(timed_sym_str)
                return timed_sym_str

        # Apply the replacement using regex substitution
        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)

        # Replace remaining base names (implicitly time t)
        # Sort by length descending to replace longer names first (e.g., ABC before AB)
        all_known_base_names = sorted(list(var_syms.keys()) + param_names + shock_names, key=len, reverse=True)
        for name in all_known_base_names:
            # Use word boundaries to avoid partial matches within other names
            pattern = r'\b' + re.escape(name) + r'\b'
            if name in var_syms: replacement = str(var_syms[name]['t'])
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in shock_names: replacement = str(shock_syms[name])
            else: continue # Should not happen if list is correct
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)

        try:
            # Check for remaining undeclared symbols (likely errors) before parsing
            current_symbols = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq_str_sym))
            known_keys = set(local_dict.keys()) | {'log', 'exp', 'sqrt', 'abs'} # Add known functions
            # Exclude numbers
            unknown_symbols = {s for s in current_symbols if s not in known_keys and not re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)}

            if unknown_symbols:
                # This is usually an error (typo, undeclared var/param)
                print(f"ERROR: Potential undeclared symbols found parsing eq {i} ('{eq_str_sym}'): {unknown_symbols}. Check model definition.")
                # Decide whether to raise error or try to continue by adding them
                # raise ValueError(f"Undeclared symbols found: {unknown_symbols}")
                print("Attempting to continue by adding symbols to local_dict.")
                for sym_str in unknown_symbols:
                    if sym_str not in local_dict: local_dict[sym_str] = sympy.symbols(sym_str)


            # Parse the string into a sympy expression
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            sym_equations.append(sym_eq)
        except Exception as e:
            print(f"\n--- Error parsing stationary equation {i} ---")
            print(f"Original Eq String : '{eq_str}'")
            print(f"Processed Sym String: '{eq_str_sym}'")
            print(f"Local dict keys    : {sorted(local_dict.keys())}")
            print(f"Sympy error        : {e}")
            raise

    if verbose: print("Symbolic parsing completed.")

    # --- Generate Initial Symbolic Matrices A P^2 + B P + C = 0, D for shocks ---
    if verbose: print("\n--- Generating Initial Symbolic Matrices (A, B, C, D) ---")
    # Convention: A coeffs y(t+1), B coeffs y(t), C coeffs y(t-1)
    # This matches the convention needed for the `solve_quadratic_matrix_equation`
    # Note the definition change: A_quad = dF/dy_{t+1}, B_quad = dF/dy_t, C_quad = dF/dy_{t-1}
    sympy_A_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t+1)
    sympy_B_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t)
    sympy_C_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t-1)
    sympy_D_quad = sympy.zeros(num_eq, num_shocks) # Coeffs of e(t) (multiplied by -1 for Q calc)

    # Get lists of symbolic variables for Jacobian calculation
    var_p1_syms = [var_syms[v]['p1'] for v in initial_vars_ordered]
    var_t_syms  = [var_syms[v]['t']  for v in initial_vars_ordered]
    var_m1_syms = [var_syms[v]['m1'] for v in initial_vars_ordered]
    shock_t_syms = [shock_syms[s] for s in shock_names]

    # Compute Jacobians
    for i, eq in enumerate(sym_equations):
        # Use Jacobian calculation for robustness with non-linear terms (though model is assumed linear)
        for j, var_p1 in enumerate(var_p1_syms): sympy_A_quad[i, j] = sympy.diff(eq, var_p1)
        for j, var_t  in enumerate(var_t_syms):  sympy_B_quad[i, j] = sympy.diff(eq, var_t)
        for j, var_m1 in enumerate(var_m1_syms): sympy_C_quad[i, j] = sympy.diff(eq, var_m1)
        # For D, we need coefficient of shock e(t). Derivative is dF/de(t). We store -dF/de(t).
        for k, shk_t in enumerate(shock_t_syms): sympy_D_quad[i, k] = -sympy.diff(eq, shk_t) # Note the minus sign

    # Store initial symbolic matrices (using solver convention A=coeff(p1), B=coeff(t), C=coeff(m1))
    # Keep a copy before reordering if needed
    initial_info = {
        'A': copy.deepcopy(sympy_A_quad), # Store based on quad eq: A P^2 + B P + C = 0 convention
        'B': copy.deepcopy(sympy_B_quad),
        'C': copy.deepcopy(sympy_C_quad),
        'D': copy.deepcopy(sympy_D_quad),
        'vars': list(initial_vars_ordered), # Keep original order info
        'eqs': list(processed_equations)   # Keep original order info
    }
    if verbose: print("Symbolic matrices A, B, C, D generated (for quadratic solver).")

    # --- Classify Variables (Simplified for Ordering) ---
    if verbose: print("\n--- Classifying Variables for Ordering (Stationary Model) ---")
    # Heuristic: RES_ and aux_lag are backward. Others depend on jacobians.
    backward_exo_vars = []
    forward_backward_endo_vars = []
    static_endo_vars = [] # Variables appearing only at time t

    potential_backward = [v for v in initial_vars_ordered if v.startswith("RES_") or (v.startswith("aux_") and "_lag_" in v)]
    remaining_vars = [v for v in initial_vars_ordered if v not in potential_backward]

    # Check matrix columns for actual dependencies
    for var in potential_backward:
        j = initial_vars_ordered.index(var)
        # Check if it has a lead dependency (appears in A_quad = dF/dy(t+1))
        has_lead = not sympy_A_quad.col(j).is_zero_matrix
        if has_lead:
            if verbose: print(f"Warning: Potential backward var '{var}' has lead dependency. Classifying as forward/backward.")
            forward_backward_endo_vars.append(var)
        else:
            backward_exo_vars.append(var)

    for var in remaining_vars:
         j = initial_vars_ordered.index(var)
         has_lag = not sympy_C_quad.col(j).is_zero_matrix # Appears with t-1? (dF/dy(t-1) != 0)
         has_lead = not sympy_A_quad.col(j).is_zero_matrix # Appears with t+1? (dF/dy(t+1) != 0)
         if has_lag or has_lead:
             forward_backward_endo_vars.append(var)
         else:
             # Only appears at time t (in B_quad = dF/dy(t))
             static_endo_vars.append(var)

    if verbose:
        print(f"  Backward/Exo Group ({len(backward_exo_vars)}): {backward_exo_vars}")
        print(f"  Forward/Backward Endo ({len(forward_backward_endo_vars)}): {forward_backward_endo_vars}")
        print(f"  Static Endo ({len(static_endo_vars)}): {static_endo_vars}")

    # --- Determine New Variable Order ---
    ordered_vars = backward_exo_vars + forward_backward_endo_vars + static_endo_vars
    if len(ordered_vars) != len(initial_vars_ordered) or set(ordered_vars) != set(initial_vars_ordered):
        raise ValueError("Variable reordering failed (loss/gain of variables).")
    # Permutation indices: mapping from new index -> old index
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]
    if verbose: print(f"\nNew Variable Order ({len(ordered_vars)}): {ordered_vars}")

    # --- Determine New Equation Order (Heuristic) ---
    eq_perm_indices = []
    used_eq_indices = set()
    # Compile regex patterns for aux/RES definitions
    # Pattern: aux_VAR_lag_mK - VAR(-1) = 0 (allow spaces, ignore case maybe?)
    aux_def_patterns = {
        name: re.compile(fr"^\s*{name}\s*-\s*{base_name_from_aux(name)}\s*\(\s*-1\s*\)\s*$", re.IGNORECASE)
        for name in aux_variables if "_lag_" in name
    }
    # Pattern: RES_VAR - ... RES_VAR(-1) ... = 0 (less precise)
    res_def_patterns = {
        name: re.compile(fr"^\s*{name}\s*-\s*.*{name}\s*\(\s*-1\s*\).*", re.IGNORECASE)
        for name in initial_vars_ordered if name.startswith("RES_")
    }

    assigned_eq_for_var = {} # Track which equation defines which variable

    # Assign defining equations for aux lags first
    for aux_var in [v for v in backward_exo_vars if v.startswith("aux_")]:
        if aux_var in assigned_eq_for_var: continue # Already assigned
        pattern = aux_def_patterns.get(aux_var)
        # --- Corrected Check ---
        if pattern: # Check if pattern exists
            found = False
            for i, eq_str in enumerate(processed_equations):
                # Clean equation string slightly for matching
                cleaned_eq = eq_str.replace(" ", "")
                if i not in used_eq_indices and pattern.match(cleaned_eq):
                    eq_perm_indices.append(i)
                    used_eq_indices.add(i)
                    assigned_eq_for_var[aux_var] = i
                    found = True
                    break
            # if not found and verbose: print(f"Warning: Could not find unique defining eq for aux lag '{aux_var}'")
        # --- End Corrected Check ---

    # Assign defining equations for RES vars
    for res_var in [v for v in backward_exo_vars if v.startswith("RES_")]:
         if res_var in assigned_eq_for_var: continue
         pattern = res_def_patterns.get(res_var)
         # --- Corrected Check ---
         if pattern: # Check if pattern exists
            found = False
            potential_matches = []
            for i, eq_str in enumerate(processed_equations):
                 if i not in used_eq_indices and pattern.match(eq_str):
                     potential_matches.append(i)

            if len(potential_matches) == 1:
                 i = potential_matches[0]
                 eq_perm_indices.append(i)
                 used_eq_indices.add(i)
                 assigned_eq_for_var[res_var] = i
                 found = True
            elif len(potential_matches) > 1 and verbose:
                 print(f"Warning: Found multiple potential defining eqs for RES var '{res_var}': {potential_matches}. Using none for ordering.")
            # elif not potential_matches and verbose:
            #      print(f"Warning: Could not find defining eq pattern match for RES var '{res_var}'")
         # --- End Corrected Check ---

    # Assign remaining equations (likely forward-looking or static) sequentially
    remaining_eq_indices = [i for i in range(num_eq) if i not in used_eq_indices]
    eq_perm_indices.extend(remaining_eq_indices)

    # Validate permutation
    if len(eq_perm_indices) != num_eq:
        raise ValueError(f"Equation permutation construction failed. Length mismatch: {len(eq_perm_indices)} vs {num_eq}")
    if len(set(eq_perm_indices)) != num_eq:
         # This indicates a logic error in assigning equations
         raise ValueError("Equation permutation construction failed. Indices not unique.")

    if verbose: print(f"\nEquation permutation indices (new row i <- old row eq_perm_indices[i]): {eq_perm_indices}")

    # --- Reorder Symbolic Matrices ---
    if verbose: print("\n--- Reordering Symbolic Matrices (Stationary Model) ---")
    # Use sympy .extract() method: matrix.extract(row_indices, col_indices)
    # Rows are permuted by eq_perm_indices, Cols by var_perm_indices
    sympy_A_ord = sympy_A_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_B_ord = sympy_B_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C_quad.extract(eq_perm_indices, var_perm_indices)
    # D matrix rows are permuted, columns (shocks) remain in original order
    sympy_D_ord = sympy_D_quad.extract(eq_perm_indices, list(range(num_shocks)))

    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}
    if verbose: print("Symbolic reordering complete.")

    # --- Lambdify ---
    if verbose: print("\n--- Lambdifying Ordered Matrices (Stationary Model) ---")
    # Ensure consistent parameter order for lambdification
    param_sym_list = [param_syms[p] for p in param_names]

    # Lambdify the *ordered* matrices
    func_A = robust_lambdify(param_sym_list, sympy_A_ord, modules='jax')
    func_B = robust_lambdify(param_sym_list, sympy_B_ord, modules='jax')
    func_C = robust_lambdify(param_sym_list, sympy_C_ord, modules='jax')
    func_D = robust_lambdify(param_sym_list, sympy_D_ord, modules='jax')
    if verbose: print("Lambdification successful.")

    # Return all necessary components
    return (func_A, func_B, func_C, func_D,
            ordered_vars, shock_names, param_names, param_assignments,
            symbolic_matrices_ordered, initial_info)

# --- END OF CORRECTED parse_lambdify_and_order_model FUNCTION ---

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


# --- SDA Solver and Q Computation (JAX Versions) ---

# # Define a state tuple for clarity in the scan
# # Added is_valid flag to track numerical stability during iteration
# SDAState = namedtuple("SDAState", ["Xk", "Yk", "Ek", "Fk", "k", "converged", "rel_diff", "is_valid"])
# _SDA_JITTER = 1e-14 # Small regularization factor

# def solve_quadratic_matrix_equation_jax(A, B, C, initial_guess=None,
#                                         tol=1e-12, max_iter=500,
#                                         verbose=False): # verbose is ineffective in JIT
#     """
#     Solves A X^2 + B X + C = 0 for X using the SDA algorithm implemented with
#     jax.lax.scan. Uses jnp.where for conditional updates after update step.
#     Returns NaN solution if convergence fails or numerical issues occur.

#     Args:
#         A, B, C: JAX arrays for the quadratic matrix equation coefficients.
#         initial_guess: Optional initial guess for the solution (JAX array).
#         tol: Convergence tolerance for the relative change in X.
#         max_iter: Fixed maximum number of iterations for the scan loop.
#         verbose: (Not effective inside JIT/grad)

#     Returns:
#         Tuple: (X_sol, iter_count, residual_ratio, converged_flag)
#                - X_sol: The computed solution (JAX array). NaN if solve failed.
#                - iter_count: Iteration count (always max_iter in this version).
#                - residual_ratio: Relative residual norm of the final state after scan.
#                - converged_flag: Boolean JAX array indicating success.
#     """
#     dtype = A.dtype
#     n = A.shape[0]
#     A_jax = jnp.asarray(A, dtype=dtype)
#     B_jax = jnp.asarray(B, dtype=dtype)
#     C_jax = jnp.asarray(C, dtype=dtype)
#     if initial_guess is None:
#         X_guess = jnp.zeros_like(A_jax)
#     else:
#         X_guess = jnp.asarray(initial_guess, dtype=dtype)

#     # Initial Setup
#     E_init = C_jax
#     F_init = A_jax
#     Bbar = B_jax + A_jax @ X_guess
#     I = jnp.eye(n, dtype=dtype)
#     Bbar_reg = Bbar + _SDA_JITTER * I
#     E0 = -jax.scipy.linalg.solve(Bbar_reg, E_init, assume_a='gen')
#     F0 = -jax.scipy.linalg.solve(Bbar_reg, F_init, assume_a='gen')
#     initial_solve_valid = jnp.all(jnp.isfinite(E0)) & jnp.all(jnp.isfinite(F0))

#     # Scan Loop Definition
#     def sda_scan_body(state, _):
#         Xk, Yk, Ek, Fk, k, prev_converged, prev_rel_diff, prev_is_valid = state
#         M1 = I - Yk @ Xk + _SDA_JITTER * I; M2 = I - Xk @ Yk + _SDA_JITTER * I
#         # Perform solves directly. Let NaN/inf propagate on failure.
#         temp_E = jax.scipy.linalg.solve(M1, Ek, assume_a='gen'); E_new = Ek @ temp_E
#         temp_F = jax.scipy.linalg.solve(M2, Fk, assume_a='gen'); F_new = Fk @ temp_F
#         temp_X = Xk @ Ek; temp_X = jax.scipy.linalg.solve(M2, temp_X, assume_a='gen'); X_new = Xk + Fk @ temp_X
#         temp_Y = Yk @ Fk; temp_Y = jax.scipy.linalg.solve(M1, temp_Y, assume_a='gen'); Y_new = Yk + Ek @ temp_Y

#         X_diff_norm = jnp.linalg.norm(X_new - Xk, ord='fro'); X_norm = jnp.linalg.norm(X_new, ord='fro')
#         current_rel_diff = X_diff_norm / jnp.maximum(X_norm, 1e-15)
#         current_step_valid = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(Y_new)) & \
#                              jnp.all(jnp.isfinite(E_new)) & jnp.all(jnp.isfinite(F_new)) & \
#                              jnp.isfinite(current_rel_diff)
#         converged_this_step = current_step_valid & (current_rel_diff < tol)
#         current_is_valid = prev_is_valid & current_step_valid
#         current_converged = prev_converged | converged_this_step
#         keep_new_state_cond = prev_is_valid & current_step_valid & (~prev_converged)

#         X_next = jnp.where(keep_new_state_cond, X_new, Xk); Y_next = jnp.where(keep_new_state_cond, Y_new, Yk)
#         E_next = jnp.where(keep_new_state_cond, E_new, Ek); F_next = jnp.where(keep_new_state_cond, F_new, Fk)
#         next_rel_diff = jnp.where(keep_new_state_cond, current_rel_diff, prev_rel_diff)
#         next_converged = jnp.where(keep_new_state_cond, current_converged, prev_converged)
#         next_is_valid = current_is_valid
#         next_state = SDAState(X_next, Y_next, E_next, F_next, k + 1, next_converged, next_rel_diff, next_is_valid)
#         return next_state, None

#     init_state = SDAState(Xk=E0, Yk=F0, Ek=E0, Fk=F0, k=0, converged=jnp.array(False), rel_diff=jnp.inf, is_valid=initial_solve_valid)
#     final_state, _ = lax.scan(sda_scan_body, init_state, xs=None, length=max_iter)

#     # Post-Scan Processing
#     X_sol_scan = final_state.Xk + X_guess
#     converged_flag = final_state.converged & final_state.is_valid # Final check
#     iter_final = final_state.k

#     # Calculate final residual ratio (for info)
#     residual = A_jax @ (X_sol_scan @ X_sol_scan) + B_jax @ X_sol_scan + C_jax
#     residual_norm = jnp.linalg.norm(residual, 'fro')
#     term_norms = (jnp.linalg.norm(A_jax @ X_sol_scan @ X_sol_scan, 'fro') +
#                   jnp.linalg.norm(B_jax @ X_sol_scan, 'fro') +
#                   jnp.linalg.norm(C_jax, 'fro'))
#     residual_ratio = residual_norm / jnp.maximum(term_norms, 1e-15)

#     # Return NaN if convergence failed or state is invalid
#     X_sol_final = jnp.where(converged_flag, X_sol_scan, jnp.full_like(X_sol_scan, jnp.nan))

#     return X_sol_final, iter_final, residual_ratio, converged_flag

# In dynare_parser_engine.py

# Add actual_converged_iter to SDAState
SDAState = namedtuple("SDAState", ["Xk", "Yk", "Ek", "Fk", "k", "converged", "rel_diff", "is_valid", "actual_converged_iter"])
_SDA_JITTER = 1e-14

def solve_quadratic_matrix_equation_jax(A, B, C, initial_guess=None,
                                        tol=1e-12, max_iter=500, # Keep max_iter as the loop length
                                        verbose=False):
    # ... (initial setup as before) ...
    dtype = A.dtype # Add this
    n = A.shape[0]  # Add this
    I = jnp.eye(n, dtype=dtype) # Add this
    E_init = C
    F_init = A
    Bbar = B + A @ (initial_guess if initial_guess is not None else jnp.zeros_like(A))
    Bbar_reg = Bbar + _SDA_JITTER * I
    E0 = jax.scipy.linalg.solve(Bbar_reg, -E_init, assume_a='gen')
    F0 = jax.scipy.linalg.solve(Bbar_reg, -F_init, assume_a='gen')
    initial_solve_valid = jnp.all(jnp.isfinite(E0)) & jnp.all(jnp.isfinite(F0))


    def sda_scan_body(state, _):
        Xk, Yk, Ek, Fk, k_iter, prev_converged, prev_rel_diff, prev_is_valid, prev_actual_conv_iter = state # Unpack new state
        
        M1 = I - Yk @ Xk + _SDA_JITTER * I; M2 = I - Xk @ Yk + _SDA_JITTER * I
        temp_E = jax.scipy.linalg.solve(M1, Ek, assume_a='gen'); E_new = Ek @ temp_E
        temp_F = jax.scipy.linalg.solve(M2, Fk, assume_a='gen'); F_new = Fk @ temp_F
        temp_X = Xk @ Ek; temp_X = jax.scipy.linalg.solve(M2, temp_X, assume_a='gen'); X_new = Xk + Fk @ temp_X
        temp_Y = Yk @ Fk; temp_Y = jax.scipy.linalg.solve(M1, temp_Y, assume_a='gen'); Y_new = Yk + Ek @ temp_Y

        X_diff_norm = jnp.linalg.norm(X_new - Xk, ord='fro'); X_norm = jnp.linalg.norm(X_new, ord='fro')
        current_rel_diff = jnp.where(X_norm > 1e-15, X_diff_norm / X_norm, jnp.inf)

        current_step_valid = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(Y_new)) & \
                             jnp.all(jnp.isfinite(E_new)) & jnp.all(jnp.isfinite(F_new)) & \
                             jnp.isfinite(current_rel_diff)
        converged_this_step = current_step_valid & (current_rel_diff < tol)
        
        current_is_valid = prev_is_valid & current_step_valid
        current_converged = prev_converged | converged_this_step # Overall convergence status

        # Update actual_converged_iter: if not previously converged but converged now, record k_iter + 1
        current_actual_conv_iter = jnp.where(
            (~prev_converged) & converged_this_step,
            k_iter + 1,
            prev_actual_conv_iter
        )

        keep_new_state_cond = prev_is_valid & current_step_valid & (~prev_converged)

        X_next = jnp.where(keep_new_state_cond, X_new, Xk); Y_next = jnp.where(keep_new_state_cond, Y_new, Yk)
        E_next = jnp.where(keep_new_state_cond, E_new, Ek); F_next = jnp.where(keep_new_state_cond, F_new, Fk)
        next_rel_diff = jnp.where(keep_new_state_cond, current_rel_diff, prev_rel_diff)
        next_converged = current_converged # Keep the updated overall convergence
        next_is_valid = current_is_valid
        
        next_state = SDAState(X_next, Y_next, E_next, F_next, k_iter + 1, next_converged, next_rel_diff, next_is_valid, current_actual_conv_iter)
        return next_state, None

    # Initialize with actual_converged_iter = max_iter (or a placeholder like -1 or inf)
    init_state = SDAState(Xk=E0, Yk=F0, Ek=E0, Fk=F0, k=0, converged=jnp.array(False), rel_diff=jnp.inf, is_valid=initial_solve_valid, actual_converged_iter=jnp.array(max_iter, dtype=jnp.int32))
    final_state, _ = lax.scan(sda_scan_body, init_state, xs=None, length=max_iter)

    X_sol_scan = final_state.Xk + (initial_guess if initial_guess is not None else jnp.zeros_like(A))
    converged_flag = final_state.converged & final_state.is_valid
    
    # loop_iterations_taken = final_state.k # This will always be max_iter
    actual_conv_iter_val = final_state.actual_converged_iter # This is what we want

    residual = A @ (X_sol_scan @ X_sol_scan) + B @ X_sol_scan + C
    residual_norm = jnp.linalg.norm(residual, 'fro')
    term_norms = (jnp.linalg.norm(A @ X_sol_scan @ X_sol_scan, 'fro') +
                  jnp.linalg.norm(B @ X_sol_scan, 'fro') +
                  jnp.linalg.norm(C, 'fro'))
    residual_ratio = jnp.where(term_norms > 1e-15, residual_norm / term_norms, jnp.inf)
    X_sol_final = jnp.where(converged_flag, X_sol_scan, jnp.full_like(X_sol_scan, jnp.nan))

    # Return the actual convergence iteration along with other results
    return X_sol_final, actual_conv_iter_val, residual_ratio, converged_flag

def compute_Q_jax(A: ArrayLike, B: ArrayLike, D: ArrayLike, P: ArrayLike,
                  dtype: Optional[jnp.dtype] = None) -> jax.Array:
    """
    Computes Q for y_t = P y_{t-1} + Q e_t using JAX. Solves (A P + B) Q = D.
    Allows NaNs to propagate if the solve fails.
    """
    effective_dtype = dtype if dtype is not None else (A.dtype if hasattr(A, 'dtype') else jnp.float64)
    A_jax = jnp.asarray(A, dtype=effective_dtype); B_jax = jnp.asarray(B, dtype=effective_dtype)
    D_jax = jnp.asarray(D, dtype=effective_dtype); P_jax = jnp.asarray(P, dtype=effective_dtype)
    n = A_jax.shape[0]
    n_shock = D_jax.shape[1] if D_jax.ndim == 2 else 0
    if n_shock == 0: return jnp.zeros((n, 0), dtype=effective_dtype)

    APB = A_jax @ P_jax + B_jax
    # Add jitter for potentially ill-conditioned APB matrix
    APB_reg = APB + _SDA_JITTER * jnp.eye(n, dtype=effective_dtype)
    # Attempt solve, let NaNs propagate if it fails
    Q = jax.scipy.linalg.solve(APB_reg, D_jax, assume_a='gen')
    return Q


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


# --- State-Space Building Functions (JAX Compatible) ---

def build_trend_matrices(trend_equations, trend_vars, trend_shocks, param_names, param_assignments, verbose=True):
    if verbose: print("\n--- Building Trend State-Space Matrices (P_trends, Q_trends) ---")
    contemporaneous_defs = {}; state_trend_vars = []; defining_equations_for_state_trends = []
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    base_var_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')
    # if verbose: print("Analyzing trend equations for state definition...")
    for eq_str in trend_equations:
        eq_parts = eq_str.split('-', 1);
        if len(eq_parts) != 2: continue
        lhs_str = eq_parts[0].strip().strip('()'); rhs_str = eq_parts[1].strip().strip('()')
        if lhs_str not in trend_vars: continue
        rhs_symbols = set(base_var_regex.findall(rhs_str)); has_contemporaneous_rhs = False
        for sym in rhs_symbols:
            if sym in trend_vars and sym != lhs_str:
                 explicit_lag = f"{sym}(-1)"
                 if sym in rhs_str and explicit_lag not in rhs_str: has_contemporaneous_rhs = True; break
        for match in var_time_regex.finditer(rhs_str):
            base, shift = match.group(1), int(match.group(2))
            if base in trend_vars and shift != -1: has_contemporaneous_rhs = True; break
        if has_contemporaneous_rhs:
            # if verbose: print(f"  Trend '{lhs_str}' defined contemporaneously: {eq_str}.")
            contemporaneous_defs[lhs_str] = rhs_str
        else:
            if lhs_str not in state_trend_vars: state_trend_vars.append(lhs_str)
            defining_equations_for_state_trends.append(eq_str)

    if not state_trend_vars:
        if verbose: print("Warning: No state trend variables identified. Returning empty matrices.")
        # Return lambdas returning empty JAX arrays
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        return lambda *a: jnp.empty((0,0), dtype=dtype), lambda *a: jnp.empty((0,0), dtype=dtype), [], {}

    # if verbose: print(f"Identified state trend variables ({len(state_trend_vars)}): {state_trend_vars}")
    num_state_trends = len(state_trend_vars); num_trend_shocks = len(trend_shocks)
    param_syms = {p: sympy.symbols(p) for p in param_names}; trend_shock_syms = {s: sympy.symbols(s) for s in trend_shocks}
    trend_var_syms = {}; all_syms = set(param_syms.values()) | set(trend_shock_syms.values())
    for var in state_trend_vars: sym_m1 = create_timed_symbol(var, -1); sym_t = create_timed_symbol(var, 0); trend_var_syms[var] = {'m1': sym_m1, 't': sym_t}; all_syms.update([sym_m1, sym_t])
    for var in trend_vars:
         if var not in trend_var_syms: sym_m1 = create_timed_symbol(var, -1); sym_t = create_timed_symbol(var, 0); trend_var_syms[var] = {'m1': sym_m1, 't': sym_t}; all_syms.add(sym_m1); all_syms.add(sym_t)
    local_dict = {str(s): s for s in all_syms}; local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    sym_trend_equations = []; # if verbose: print("Parsing state trend equations into symbolic form...")
    parsed_eq_for_state_var = {var: None for var in state_trend_vars}; final_sym_equations_ordered = []
    for i, eq_str in enumerate(defining_equations_for_state_trends):
        eq_parts = eq_str.split('-', 1); lhs_var = eq_parts[0].strip().strip('()')
        if lhs_var not in state_trend_vars: continue
        eq_str_sym = eq_str
        def replace_trend_time(match):
            base, shift_str = match.groups(); shift = int(shift_str)
            if base in trend_shock_syms:
                if shift == 0: return str(trend_shock_syms[base])
                else: raise ValueError(f"Trend shock {base}({shift}) invalid.")
            elif base in trend_var_syms:
                 if shift == -1: return str(trend_var_syms[base]['m1'])
                 if shift == 0 and base == lhs_var: return str(trend_var_syms[base]['t'])
                 raise ValueError(f"Unexpected term {base}({shift}) in state trend eq for {lhs_var}.")
            elif base in param_syms: raise ValueError(f"Parameter {base}({shift}) invalid.")
            else: sym = sympy.symbols(f"{base}_t{shift:+d}"); return str(sym) # Use +d for sign
        eq_str_sym = var_time_regex.sub(replace_trend_time, eq_str_sym)
        all_base_names = sorted(list(trend_var_syms.keys()) + param_names + list(trend_shock_syms.keys()), key=len, reverse=True)
        for name in all_base_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name == lhs_var: replacement = str(trend_var_syms[name]['t'])
            elif name in trend_var_syms: continue
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in trend_shock_syms: replacement = str(trend_shock_syms[name])
            else: continue
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        try: sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations); parsed_eq_for_state_var[lhs_var] = sym_eq
        except Exception as e: print(f"Error parsing trend eq: '{eq_str}' -> '{eq_str_sym}'\n{e}"); raise
    for var in state_trend_vars:
         eq = parsed_eq_for_state_var[var]
         if eq is None: raise RuntimeError(f"Missing defining eq for state trend var '{var}'.")
         final_sym_equations_ordered.append(eq)
    # if verbose: print("Symbolic parsing of state trend equations completed.")
    sympy_P_trends = sympy.zeros(num_state_trends, num_state_trends); sympy_Q_trends = sympy.zeros(num_state_trends, num_trend_shocks)
    state_trend_m1_syms = [trend_var_syms[v]['m1'] for v in state_trend_vars]; trend_shock_t_syms = [trend_shock_syms[s] for s in trend_shocks]
    for i, eq in enumerate(final_sym_equations_ordered):
        for j, var_m1 in enumerate(state_trend_m1_syms): sympy_P_trends[i, j] = -sympy.diff(eq, var_m1)
        for k, shock_t in enumerate(trend_shock_t_syms): sympy_Q_trends[i, k] = -sympy.diff(eq, shock_t)
    # if verbose: print("Symbolic P_trends and Q_trends matrices generated.")
    # if verbose: print("Lambdifying trend matrices...")
    param_sym_list = [param_syms[p] for p in param_names]
    func_P_trends = robust_lambdify(param_sym_list, sympy_P_trends)
    func_Q_trends = robust_lambdify(param_sym_list, sympy_Q_trends)
    # if verbose: print("Lambdification successful.")
    return func_P_trends, func_Q_trends, state_trend_vars, contemporaneous_defs

def build_observation_matrix(measurement_equations, obs_vars, stationary_vars,
                             trend_state_vars, contemporaneous_trend_defs,
                             param_names, param_assignments, verbose=True):
    if verbose: print("\n--- Building Observation Matrix (Omega) ---")
    num_obs = len(obs_vars); num_stationary = len(stationary_vars); num_trend_state = len(trend_state_vars)
    num_augmented_state = num_stationary + num_trend_state
    if num_obs == 0:
        if verbose: print("No observable variables declared. Returning empty Omega.")
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        return lambda *a: jnp.empty((0, num_augmented_state), dtype=dtype), []
    if len(measurement_equations) != num_obs: raise ValueError(f"Num measurement eqs ({len(measurement_equations)}) != num varobs ({num_obs}).")

    param_syms = {p: sympy.symbols(p) for p in param_names}; obs_syms = {v: sympy.symbols(v) for v in obs_vars}
    all_rhs_vars = stationary_vars + trend_state_vars + list(contemporaneous_trend_defs.keys())
    rhs_var_syms = {v: sympy.symbols(v) for v in all_rhs_vars}
    all_syms = set(param_syms.values()) | set(obs_syms.values()) | set(rhs_var_syms.values())
    local_dict = {str(s): s for s in all_syms}; local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    sym_measurement_equations = []; # if verbose: print("Parsing measurement equations into symbolic form...")
    parsed_eq_for_obs_var = {var: None for var in obs_vars}
    for eq_str in measurement_equations:
        eq_parts = eq_str.split('-', 1);
        if len(eq_parts) != 2: continue
        lhs_var = eq_parts[0].strip().strip('()')
        if lhs_var not in obs_vars: continue
        rhs_str = eq_parts[1].strip().strip('()'); rhs_processed = rhs_str
        for _ in range(len(contemporaneous_trend_defs) + 1):
            made_substitution = False
            for contemp_var, contemp_expr_str in contemporaneous_trend_defs.items():
                pattern = r'\b' + re.escape(contemp_var) + r'\b'
                if re.search(pattern, rhs_processed): rhs_processed = re.sub(pattern, f"({contemp_expr_str})", rhs_processed); made_substitution = True
            if not made_substitution: break
        eq_str_subbed = f"{lhs_var} - ({rhs_processed})"
        eq_str_sym = eq_str_subbed
        all_base_names_obs = sorted(list(obs_syms.keys()) + list(rhs_var_syms.keys()) + param_names, key=len, reverse=True)
        for name in all_base_names_obs:
             pattern = r'\b' + re.escape(name) + r'\b'
             if name in obs_syms: replacement = str(obs_syms[name])
             elif name in rhs_var_syms: replacement = str(rhs_var_syms[name])
             elif name in param_syms: replacement = str(param_syms[name])
             else: continue
             eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        try: sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations); parsed_eq_for_obs_var[lhs_var] = sym_eq
        except Exception as e: print(f"Error parsing measurement eq: '{eq_str}' -> '{eq_str_sym}'\n{e}"); raise
    final_sym_equations_ordered = []
    for var in obs_vars:
         eq = parsed_eq_for_obs_var[var]
         if eq is None: raise RuntimeError(f"Missing measurement eq for observable '{var}'.")
         final_sym_equations_ordered.append(eq)
    # if verbose: print("Symbolic parsing of measurement equations completed.")
    sympy_Omega = sympy.zeros(num_obs, num_augmented_state)
    augmented_state_syms = [rhs_var_syms[v] for v in stationary_vars] + [rhs_var_syms[v] for v in trend_state_vars]
    for i, eq in enumerate(final_sym_equations_ordered):
        for j, state_sym in enumerate(augmented_state_syms):
            sympy_Omega[i, j] = -sympy.diff(eq, state_sym)
    # if verbose: print("Symbolic Omega matrix generated.")
    # if verbose: print("Lambdifying observation matrix...")
    param_sym_list = [param_syms[p] for p in param_names]
    func_Omega = robust_lambdify(param_sym_list, sympy_Omega)
    # if verbose: print("Lambdification successful.")
    return func_Omega, obs_vars

# --- IRF Calculation Functions (JAX) ---

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
    n = P_jax.shape[0]; n_shock = R_jax.shape[1]
    dtype = P_jax.dtype
    if not (0 <= shock_index < n_shock): raise ValueError(f"shock_index must be 0 <= index < {n_shock}")

    y_resp = jnp.zeros((horizon, n), dtype=dtype)
    # Unit shock impulse at time 0
    impulse = R_jax[:, shock_index] # Select column corresponding to shock

    # y_0 = P*y_{-1} + impulse. Assume y_{-1} = 0.
    y_current = impulse
    y_resp = y_resp.at[0, :].set(y_current)

    def step(y_prev, _):
        y_next = P_jax @ y_prev
        return y_next, y_next

    # Scan for remaining steps
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
    n_aug = P_aug_jax.shape[0]; n_aug_shock = R_aug_jax.shape[1]; n_obs = Omega_jax.shape[0]
    if not (0 <= shock_index < n_aug_shock): raise ValueError(f"Aug shock_index must be 0 <= index < {n_aug_shock}")
    if Omega_jax.shape[1] != n_aug: raise ValueError(f"Omega columns ({Omega_jax.shape[1]}) != P_aug dim ({n_aug}).")

    # Compute state IRF using the JAX function
    state_irf = irf(P_aug_jax, R_aug_jax, shock_index, horizon) # (horizon, n_aug)

    # Map state responses to observables: obs_irf = state_irf @ Omega.T
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


# --- Add this import at the top with other imports ---
import jax.random

# --- [ PREVIOUS CODE: All functions including simulate_ssm_data, plot_simulation, etc. ] ---
# --- [ PREVIOUS CODE: parse_lambdify_and_order_model, build_trend_matrices, etc. ] ---

# --- Main Execution Guard (Example Usage - Usually called from wrapper) ---
if __name__ == "__main__":
    print("--- Dynare Parser, Solver, and Simulator Script ---")
    print("Running example parse/solve/IRF/simulate.")
    try:
        # --- Setup: Find model file ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        if not os.path.exists(mod_file_path):
             raise FileNotFoundError(f"Example model file not found: {mod_file_path}")
        with open(mod_file_path, 'r') as f:
            model_def = f.read()
        print(f"Loaded model from: {mod_file_path}")

        # --- STEP 1: Parse Stationary Model ---
        print("\n--- Parsing Stationary Model ---")
        (func_A, func_B, func_C, func_D, ordered_stat_vars, stat_shocks,
         param_names_stat, param_assignments_stat, _, _) = parse_lambdify_and_order_model(model_def, verbose=False)
        n_stat = len(ordered_stat_vars)
        n_s_shock = len(stat_shocks)
        print(f"Found {n_stat} stationary vars, {n_s_shock} stationary shocks.")

        # --- STEP 2: Parse Trends/Observations ---
        print("\n--- Parsing Trend/Observation Model ---")
        trend_vars, trend_shocks = extract_trend_declarations(model_def)
        trend_equations = extract_trend_equations(model_def)
        obs_vars = extract_observation_declarations(model_def)
        measurement_equations = extract_measurement_equations(model_def)
        trend_stderr_params = extract_trend_shock_stderrs(model_def)
        # Extract model shock std deviations using the new function
      #  model_stderr_params = extract_model_shock_stderrs(model_def)

        n_t_shock = len(trend_shocks)
      # n_t_model_shock = len(model_stderr_params)
        n_obs = len(obs_vars)
        print(f"Found {len(trend_vars)} trend vars, {n_t_shock} trend shocks, {n_obs} observable vars.")

        # --- STEP 3: Combine Parameters ---
        print("\n--- Combining Parameters ---")
        all_param_names = list( dict.fromkeys(param_names_stat +  list(trend_stderr_params.keys())).keys())
        all_param_assignments = param_assignments_stat.copy()
        all_param_assignments.update(trend_stderr_params)
        
        # Apply example parameter overrides (adjust as needed)
        param_overrides = {
            'b1': 0.75, 
            'b4': 0.75, 
            'a1': 0.5, 
            'a2': 0.1, 
            'g1': 0.7,
            'g2': 0.3, 
            'g3': 0.25, 
            'rho_L_GDP_GAP': 0.8, 
            'rho_DLA_CPI': 0.7,
            'rho_rs': 0.75, 
            'rho_rs2': 0.01
        }
        all_param_assignments.update(param_overrides)
        print(f"Total parameters: {len(all_param_names)}. Applied overrides.")

        # --- STEP 4: Build Trend/Observation Matrix Functions ---
        print("\n--- Building Trend/Observation Matrix Functions ---")
        (func_P_trends, func_Q_trends, ordered_trend_state_vars, contemp_trend_defs) = build_trend_matrices(
            trend_equations, trend_vars, trend_shocks, all_param_names, all_param_assignments, verbose=False)
        n_trend = len(ordered_trend_state_vars)
        (func_Omega, ordered_obs_vars) = build_observation_matrix(
            measurement_equations, obs_vars, ordered_stat_vars, ordered_trend_state_vars,
            contemp_trend_defs, all_param_names, all_param_assignments, verbose=False)
        print(f"Identified {n_trend} trend state vars. Built trend/obs function generators.")

        # --- STEP 5: Evaluate Numerical Stationary Matrices ---
        print("\n--- Evaluating Numerical Stationary Matrices ---")
        stat_param_values = [all_param_assignments.get(p, 1.0) for p in param_names_stat] # Use 1.0 default for safety
        A_num_stat = jnp.asarray(func_A(*stat_param_values))
        B_num_stat = jnp.asarray(func_B(*stat_param_values))
        C_num_stat = jnp.asarray(func_C(*stat_param_values))
        D_num_stat = jnp.asarray(func_D(*stat_param_values))

        # --- STEP 6: Solve Stationary Model (JAX SDA) ---
        print("\n--- Solving Stationary Model (JAX SDA) ---")
        P_sol_stat, iter_count, residual_ratio, converged = solve_quadratic_matrix_equation_jax(
            A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500)

        if not converged or not jnp.all(jnp.isfinite(P_sol_stat)):
            raise RuntimeError(f"JAX SDA solver failed to converge! Converged: {converged}, Residual Ratio: {residual_ratio:.2e}")
        else:
            print(f"JAX SDA converged. Residual ratio: {residual_ratio:.2e}")
            Q_sol_stat = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
            print("Q_stationary computed.")

        # --- STEP 7: Evaluate Numerical Trend/Observation Matrices ---
        print("\n--- Evaluating Numerical Trend/Observation Matrices ---")
        all_param_values = [all_param_assignments.get(p, 1.0) for p in all_param_names]
        P_num_trend = jnp.asarray(func_P_trends(*all_param_values))
        Q_num_trend = jnp.asarray(func_Q_trends(*all_param_values))
        Omega_num = jnp.asarray(func_Omega(*all_param_values))
        print("P_trend, Q_trend, Omega evaluated.")

        # --- STEP 8: Build R Matrices (Apply Std Devs) & Augmented System ---
        print("\n--- Building R Matrices and Augmented System ---")
        shock_std_devs = {}
        aug_shocks = stat_shocks + trend_shocks
        for shock_name in aug_shocks:
            sigma_param_name = f"sigma_{shock_name}"
            # Use assignment, default to 1.0 if sigma parameter missing from assignments
            std_dev = all_param_assignments.get(sigma_param_name, 1.0)
            if sigma_param_name not in all_param_assignments:
                print(f"Warning: Sigma parameter '{sigma_param_name}' not assigned, using default std dev 1.0 for shock '{shock_name}'.")
            shock_std_devs[shock_name] = std_dev

        stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in stat_shocks], dtype=P_sol_stat.dtype)
        trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in trend_shocks], dtype=P_sol_stat.dtype)

        R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if n_s_shock > 0 else jnp.zeros((n_stat, 0), dtype=P_sol_stat.dtype)
        R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if n_t_shock > 0 else jnp.zeros((n_trend, 0), dtype=P_sol_stat.dtype)

        # Build Augmented P_aug, R_aug
        n_aug = n_stat + n_trend
        n_aug_shock = n_s_shock + n_t_shock
        aug_state_vars = ordered_stat_vars + ordered_trend_state_vars # Combined state vector order

        P_aug = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
        R_aug = jnp.zeros((n_aug, n_aug_shock), dtype=P_aug.dtype)
        if n_stat > 0 and n_s_shock > 0: R_aug = R_aug.at[:n_stat, :n_s_shock].set(R_sol_stat)
        if n_trend > 0 and n_t_shock > 0: R_aug = R_aug.at[n_stat:, n_s_shock:].set(R_num_trend)
        print(f"Augmented system built: n_aug={n_aug}, n_aug_shock={n_aug_shock}")
        print(f"Augmented state order: {aug_state_vars}")

        # --- STEP 9: Compute IRFs (Example) ---
        print("\n--- Computing Example IRFs (JAX) ---")
        shock_name_to_plot = "SHK_RS" # Example shock
        if shock_name_to_plot in aug_shocks:
            shock_index_aug = aug_shocks.index(shock_name_to_plot)
            horizon = 40

            irf_states_aug = irf(P_aug, R_aug, shock_index=shock_index_aug, horizon=horizon)
            irf_observables_vals = irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index_aug, horizon=horizon)
            print(f"IRFs computed for shock '{shock_name_to_plot}'.")

            # Plotting IRFs (Optional)
            # plot_irfs(irf_observables_vals, ordered_obs_vars, horizon, title=f"Observable IRFs to {shock_name_to_plot}")
            # plot_irfs(irf_states_aug, aug_state_vars, horizon, title=f"Augmented State IRFs to {shock_name_to_plot}")
            # plt.show() # Call show later after simulation plots
        else:
            print(f"Warning: Shock '{shock_name_to_plot}' not found in augmented shocks: {aug_shocks}. Skipping IRF example.")


        # --- STEP 10: Simulate Data ---
        print("\n--- Simulating Data ---")
        T_sim = 200 # Simulation length
        key_master = jax.random.PRNGKey(42) # Master random seed
        key_init, key_sim = jax.random.split(key_master) # Split key for init state and simulation path

        # --- Specify Initial State Distribution for Trends ---
        # Define mean and std dev for the *initial* values (s_0) of the trend states
        # These should correspond to the 'ordered_trend_state_vars' list
        # Example: Assume all trends start near zero with some small variance
        trend_init_means = jnp.zeros(n_trend, dtype=_DEFAULT_DTYPE)
        trend_init_stds = jnp.ones(n_trend, dtype=_DEFAULT_DTYPE) * 0.1 # Small initial variance

        # Example: Customize specific trend initial means if needed
        if 'L_GDP_TREND' in ordered_trend_state_vars:
            idx = ordered_trend_state_vars.index('L_GDP_TREND')
            trend_init_means = trend_init_means.at[idx].set(10.0) # Start potential GDP at 1.0
            trend_init_stds = trend_init_stds.at[idx].set(0.01)

        if 'G_TREND' in ordered_trend_state_vars:
            idx = ordered_trend_state_vars.index('G_TREND')
            trend_init_means = trend_init_means.at[idx].set(2.0) # Start potential GDP at 1.0
            trend_init_stds = trend_init_stds.at[idx].set(0.002)

        if 'PI_TREND' in ordered_trend_state_vars:
            idx = ordered_trend_state_vars.index('PI_TREND')
            trend_init_means = trend_init_means.at[idx].set(2.0) # Start potential GDP at 1.0
            trend_init_stds = trend_init_stds.at[idx].set(0.01)

        if 'RR_TREND' in ordered_trend_state_vars:
            idx = ordered_trend_state_vars.index('RR_TREND')
            trend_init_means = trend_init_means.at[idx].set(1.0) # Start potential GDP at 1.0
            trend_init_stds = trend_init_stds.at[idx].set(0.1)


        # Construct the initial state s_0
        s0 = jnp.zeros(n_aug, dtype=_DEFAULT_DTYPE) # Initialize augmented state vector
        # Set stationary parts to zero (steady state) - already done by zeros init

        # Draw initial trend states from specified distribution N(mean, std^2)
        if n_trend > 0:
            initial_trends = trend_init_means + trend_init_stds * jax.random.normal(key_init, (n_trend,), dtype=_DEFAULT_DTYPE)
            s0 = s0.at[n_stat:].set(initial_trends) # Place draws into the trend part of s0

        print(f"Initial state s0 constructed. Stationary=0, Trends drawn from N(mean, std^2).")
        print(f"Simulating {T_sim} periods...")

        # Specify measurement noise level (if desired)
        measurement_noise_level = 0.0

        sim_states, sim_observables = simulate_ssm_data(
            P=P_aug,
            R=R_aug,
            Omega=Omega_num,
            T=T_sim,
            key=key_sim, # Use the second key for the simulation path
            state_init=s0,
            measurement_noise_std=measurement_noise_level
        )
        print("Simulation complete.")
        print(f"Simulated states shape: {sim_states.shape}")
        print(f"Simulated observables shape: {sim_observables.shape}")

        # --- STEP 11: Plot Simulation Results ---
        print("\n--- Plotting Simulation Results ---")

        # Plot observables only
        plot_simulation(sim_observables, ordered_obs_vars, title=f"Simulated Observables (T={T_sim}, Meas. Noise Std={measurement_noise_level})")

        # Plot observables together with underlying trends
        trend_indices_in_aug_state = list(range(n_stat, n_aug)) # Indices of trends in the augmented state
        trend_state_names = ["L_GDP_TREND", "G_TREND", "PI_TREND", "RS_TREND", "RR_TREND"]
        mapping = {
            "L_GDP_OBS": "L_GDP_TREND",
            "DLA_CPI_OBS": "PI_TREND",
            "PI_TREND_OBS": "PI_TREND",
            "RS_OBS": "RS_TREND",
            # etc, match as appropriate
        }
        plot_simulation_with_trends_matched(
            sim_observables, ordered_obs_vars,
            sim_states, aug_state_vars, trend_state_names, mapping=mapping
        )

        # Show all plots (IRFs from Step 9 and Simulation from Step 11)
        print("\nDisplaying plots...")
        plt.show()

    # --- Error Handling ---
    except FileNotFoundError as e: print(f"\nError: {e}")
    except ValueError as e: print(f"\nValueError: {e}"); import traceback; traceback.print_exc()
    except RuntimeError as e: print(f"\nRuntimeError: {e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nUnexpected Error: {e}"); import traceback; traceback.print_exc()

# --- END OF FILE ---

# # --- Main Execution Guard (Example Usage - Usually called from wrapper) ---
# if __name__ == "__main__":
#     print("--- Dynare Parser and Solver Script ---")
#     print("This script defines parsing, solving, and IRF functions.")
#     print("It is typically imported and used by a wrapper class (e.g., DynareModel).")
#     print("Running this script directly will perform an example parse/solve/IRF.")
#     try:
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
#         if not os.path.exists(mod_file_path): raise FileNotFoundError(f"Example model file not found: {mod_file_path}")
#         with open(mod_file_path, 'r') as f: model_def = f.read()

#         # STEP 1: Parse Stationary
#         (func_A, func_B, func_C, func_D, ordered_stat_vars, stat_shocks,
#          param_names_stat, param_assignments_stat, _, _) = parse_lambdify_and_order_model(model_def, verbose=False)

#         # STEP 2: Parse Trends/Obs
#         trend_vars, trend_shocks = extract_trend_declarations(model_def)
#         trend_equations = extract_trend_equations(model_def)
#         obs_vars = extract_observation_declarations(model_def)
#         measurement_equations = extract_measurement_equations(model_def)
#         trend_stderr_params = extract_trend_shock_stderrs(model_def)

#         # STEP 3: Combine Params
#         all_param_names = list(dict.fromkeys(param_names_stat + list(trend_stderr_params.keys())).keys())
#         all_param_assignments = param_assignments_stat.copy()
#         all_param_assignments.update(trend_stderr_params)
#         # Example values update (use defaults from main script if needed)
#         all_param_assignments.update({ 'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12, 'g1': 0.7, 'g2': 0.3, 'g3': 0.25, 'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7, 'rho_rs': 0.75, 'rho_rs2': 0.05 })


#         # STEP 4: Build Trend/Obs Matrices
#         (func_P_trends, func_Q_trends, ordered_trend_state_vars, contemp_trend_defs) = build_trend_matrices(
#             trend_equations, trend_vars, trend_shocks, all_param_names, all_param_assignments, verbose=False)
#         (func_Omega, ordered_obs_vars) = build_observation_matrix(
#             measurement_equations, obs_vars, ordered_stat_vars, ordered_trend_state_vars,
#             contemp_trend_defs, all_param_names, all_param_assignments, verbose=False)

#         # STEP 5: Evaluate Numerical Matrices
#         test_args = [all_param_assignments.get(p, 0.0) for p in all_param_names] # Use 0.0 default if missing (shouldn't happen)
#         stat_test_args = [all_param_assignments.get(p, 0.0) for p in param_names_stat]

#         A_num_stat = jnp.asarray(func_A(*stat_test_args))
#         B_num_stat = jnp.asarray(func_B(*stat_test_args))
#         C_num_stat = jnp.asarray(func_C(*stat_test_args))
#         D_num_stat = jnp.asarray(func_D(*stat_test_args))

#         # STEP 6: Solve Stationary (JAX)
#         print("\nSolving stationary model using JAX SDA...")
#         P_sol_stat, iter_count, residual_ratio, converged = solve_quadratic_matrix_equation_jax(
#             A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500)

#         if not converged or not jnp.all(jnp.isfinite(P_sol_stat)):
#             print(f"ERROR: JAX SDA solver failed. Converged: {converged}, Residual Ratio: {residual_ratio:.2e}")
#         else:
#             print(f"JAX SDA converged in {iter_count} iterations (fixed). Residual ratio: {residual_ratio:.2e}")
#             Q_sol_stat = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
#             print("Q_stationary computed.")

#             # STEP 7: Evaluate Trend/Obs (JAX)
#             P_num_trend = jnp.asarray(func_P_trends(*test_args))
#             Q_num_trend = jnp.asarray(func_Q_trends(*test_args))
#             Omega_num = jnp.asarray(func_Omega(*test_args))

#             # Scale Q by std devs to get R
#             shock_std_devs = {}
#             for shock_name in stat_shocks + trend_shocks:
#                  sigma_param_name = f"sigma_{shock_name}"
#                  shock_std_devs[shock_name] = all_param_assignments.get(sigma_param_name, 1.0) # Default 1.0 if sigma missing

#             stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in stat_shocks], dtype=P_sol_stat.dtype)
#             trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in trend_shocks], dtype=P_sol_stat.dtype)

#             R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if len(stat_shocks)>0 else jnp.zeros((P_sol_stat.shape[0], 0), dtype=P_sol_stat.dtype)
#             R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if len(trend_shocks)>0 else jnp.zeros((P_num_trend.shape[0], 0), dtype=P_sol_stat.dtype)

#             # STEP 8: Build Augmented System (JAX)
#             n_stat = P_sol_stat.shape[0]; n_trend = P_num_trend.shape[0]
#             n_s_shock = R_sol_stat.shape[1]; n_t_shock = R_num_trend.shape[1]
#             n_aug = n_stat + n_trend; n_aug_shock = n_s_shock + n_t_shock

#             P_aug = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
#             R_aug = jnp.zeros((n_aug, n_aug_shock), dtype=P_aug.dtype)
#             if n_stat > 0 and n_s_shock > 0: R_aug = R_aug.at[:n_stat, :n_s_shock].set(R_sol_stat)
#             if n_trend > 0 and n_t_shock > 0: R_aug = R_aug.at[n_stat:, n_s_shock:].set(R_num_trend)

#             print("\nAugmented Matrices (JAX):")
#             # print("P_aug:\n", P_aug); print("R_aug:\n", R_aug); print("Omega:\n", Omega_num)

#             # STEP 9: Compute IRFs (JAX)
#             print("\nComputing IRFs using JAX...")
#             shock_name_to_plot = "SHK_RS"
#             aug_shocks = stat_shocks + trend_shocks
#             shock_index_aug = aug_shocks.index(shock_name_to_plot)
#             horizon = 40

#             irf_states_aug = irf(P_aug, R_aug, shock_index=shock_index_aug, horizon=horizon)
#             irf_observables_vals = irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index_aug, horizon=horizon)
#             print(f"IRFs computed for shock '{shock_name_to_plot}'.")

#             # Plotting (Optional - uses matplotlib which works with JAX arrays via np.asarray)
#             aug_state_vars = ordered_stat_vars + ordered_trend_state_vars
#             print("\nPlotting IRFs...")
#             plot_irfs(irf_observables_vals, ordered_obs_vars, horizon, title=f"Observable IRFs to {shock_name_to_plot}")
#             plot_irfs(irf_states_aug, aug_state_vars, horizon, title=f"Augmented State IRFs to {shock_name_to_plot}")
#             plt.show() # Show plots if run directly

#     except FileNotFoundError as e: print(f"\nError: {e}")
#     except ValueError as e: print(f"\nValueError: {e}"); import traceback; traceback.print_exc()
#     except RuntimeError as e: print(f"\nRuntimeError: {e}"); import traceback; traceback.print_exc()
#     except Exception as e: print(f"\nUnexpected Error: {e}"); import traceback; traceback.print_exc()


# --- END OF FILE Dynare_parser_sda_solver.py ---
# ```