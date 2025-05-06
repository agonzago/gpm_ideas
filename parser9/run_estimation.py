# --- START OF FILE run_estimation.py ---
import os
import time
import jax
import jax.numpy as jnp
import numpy as onp # For plotting or data loading if needed
import matplotlib.pyplot as plt
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any
import re  

# --- JAX/Dynamax/Numpyro Setup ---
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

# --- Library Imports ---
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    NUMPYRO_AVAILABLE = True
    print("Numpyro imported successfully.")
    try:
        num_devices_to_use = jax.local_device_count()
        numpyro.set_host_device_count(num_devices_to_use)
        print(f"Numpyro configured to use {num_devices_to_use} host devices.")
    except Exception as e_np_config:
        print(f"Warning: Could not configure numpyro device count: {e_np_config}")
except ImportError:
    NUMPYRO_AVAILABLE = False
    print("Warning: numpyro not found. Estimation disabled.")

# try:
#     from dynamax.linear_gaussian_ssm import LinearGaussianSSM,  lgssm_filter
#     DYNAMAX_AVAILABLE = True
#     print("Dynamax imported successfully.")
# except ImportError:
#     DYNAMAX_AVAILABLE = False
#     print("Warning: dynamax not found. Likelihood calculation will fail.")

# --- Import your custom Kalman Filter ---
try:
    # Assume Kalman_filter_jax.py is in the same directory or path
    from Kalman_filter_jax import KalmanFilter
    KALMAN_FILTER_JAX_AVAILABLE = True
    print("Custom KalmanFilter imported successfully.")
except ImportError:
    KALMAN_FILTER_JAX_AVAILABLE = False
    print("Warning: Kalman_filter_jax.py not found. Likelihood calculation will fail.")
# --- End custom Kalman Filter import ---


# --- Import from Parser ---
# Assume Dynare_parser_sda_solver.py is in the same directory or PYTHONPATH
from dynare_parser_sda_solver_jax import (
    extract_declarations,
    extract_model_equations,
    extract_trend_declarations,
    extract_trend_equations,
    extract_observation_declarations,
    extract_measurement_equations,
    extract_stationary_shock_stderrs,
    extract_trend_shock_stderrs,
    parse_and_compute_matrices_jax_ad, # Used for initial parse/ordering
    compute_matrices_jax_ad,           # Used within solve method
    build_trend_matrices_jax_ad,       # Used within solve method
    build_observation_matrix_jax_ad,   # Used within solve method
    solve_quadratic_matrix_equation_jax,
    compute_Q_jax,
    construct_initial_state, # Helper for simulation/initialization
    simulate_ssm_data, # For generating test data
    plot_simulation_with_trends_matched # For plotting
    # Add other necessary imports like plot_irfs if needed
)


# --- Dynare Model Class using JAX AD ---

class DynareModelWithJAXAD:
    """
    Represents a Dynare model, computing matrices using JAX AD and
    providing methods for solving and likelihood calculation (via Dynamax).
    """
    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose
        self._parsed = False
        self._parse_model() # Parse structure on initialization

    def _parse_model(self):
        """Parses the model structure (equations, names) once."""
        if self._parsed:
            return

        if self.verbose: print("--- Parsing Model Structure ---")
        with open(self.mod_file_path, 'r') as f:
            model_def = f.read()

        # --- Stationary Part ---
        # Use parse_and_compute_matrices_jax_ad primarily for parsing and ordering info
        # We will recompute matrices with sampled params later
        try:
            (_, _, _, _,
             self.ordered_stat_vars, self.stat_shocks, self.param_names_stat,
             self.param_assignments_stat, self.var_perm_indices_stat,
             self.eq_perm_indices_stat, self.initial_stat_vars, _ # Ignore matrices/code
             ) = parse_and_compute_matrices_jax_ad(model_def, verbose=self.verbose)

            # Store processed equations (needed for re-computation)
            # Re-run relevant parts of parse_and_compute_matrices_jax_ad without AD
            self.declared_vars, _, _, _ = extract_declarations(model_def)
            raw_equations_stat = extract_model_equations(model_def)
            # Re-run aux variable handling to get processed equations
            _vars = list(self.declared_vars)
            _aux = {}
            self.processed_equations_stat = list(raw_equations_stat)
            _var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
            _eq_idx = 0
            while _eq_idx < len(self.processed_equations_stat):
                 _eq = self.processed_equations_stat[_eq_idx]; _eq_idx += 1; _mod_eq = _eq
                 _matches = list(_var_time_regex.finditer(_eq))
                 # [Copy the exact auxiliary variable handling loop from parser here]
                 for _match in reversed(_matches):
                     _base, _shift = _match.group(1), int(_match.group(2))
                     if _base not in _vars and _base not in _aux: continue
                     if _shift > 1:
                         _defs = []; _target = ""
                         for k in range(1, _shift):
                             _aux_n = f"aux_{_base}_lead_p{k}"
                             if _aux_n not in _aux:
                                 _prev = _base if k == 1 else f"aux_{_base}_lead_p{k-1}"
                                 _def_str = f"({_aux_n}) - ({_prev}(+1))"; _aux[_aux_n] = _def_str; _defs.append(_def_str)
                                 if _aux_n not in _vars: _vars.append(_aux_n)
                         _target = f"aux_{_base}_lead_p{_shift-1}"
                         _repl = f"{_target}(+1)"; _start, _end = _match.span(); _mod_eq = _mod_eq[:_start] + _repl + _mod_eq[_end:]
                         for _def in _defs:
                              if _def not in self.processed_equations_stat and f"({_def.split('-')[0].strip().strip('()')}) - ({_def.split('-')[1].strip().strip('()')})" not in self.processed_equations_stat: self.processed_equations_stat.append(_def)
                     elif _shift < -1:
                         _defs = []; _target = ""
                         for k in range(1, abs(_shift)):
                             _aux_n = f"aux_{_base}_lag_m{k}"
                             if _aux_n not in _aux:
                                 _prev = _base if k == 1 else f"aux_{_base}_lag_m{k-1}"
                                 _def_str = f"({_aux_n}) - ({_prev}(-1))"; _aux[_aux_n] = _def_str; _defs.append(_def_str)
                                 if _aux_n not in _vars: _vars.append(_aux_n)
                         _target = f"aux_{_base}_lag_m{abs(_shift)-1}"
                         _repl = f"{_target}(-1)"; _start, _end = _match.span(); _mod_eq = _mod_eq[:_start] + _repl + _mod_eq[_end:]
                         for _def in _defs:
                             if _def not in self.processed_equations_stat and f"({_def.split('-')[0].strip().strip('()')}) - ({_def.split('-')[1].strip().strip('()')})" not in self.processed_equations_stat: self.processed_equations_stat.append(_def)
                 if _mod_eq != _eq: self.processed_equations_stat[_eq_idx - 1] = _mod_eq


        except Exception as e:
            print(f"Error during initial stationary model parse: {e}")
            raise

        # --- Trend Part ---
        self.trend_vars, self.trend_shocks = extract_trend_declarations(model_def)
        self.trend_equations = extract_trend_equations(model_def)
        self.trend_stderr_params = extract_trend_shock_stderrs(model_def)

        # --- Observation Part ---
        self.obs_vars = extract_observation_declarations(model_def)
        self.measurement_equations = extract_measurement_equations(model_def)

        # --- Combine Parameters ---
        _stat_sigma_params = [f"sigma_{shk}" for shk in self.stat_shocks]
        _trend_sigma_params = [f"sigma_{shk}" for shk in self.trend_shocks]
        self.all_param_names = list(dict.fromkeys(
            self.param_names_stat + list(self.trend_stderr_params.keys()) + _stat_sigma_params + _trend_sigma_params
        ).keys())

        self.default_param_assignments = self.param_assignments_stat.copy()
        self.default_param_assignments.update(self.trend_stderr_params)
        # Ensure defaults for ALL sigmas (including stationary if not in shocks block)
        for p_sigma in _stat_sigma_params + _trend_sigma_params:
             if p_sigma not in self.default_param_assignments:
                 self.default_param_assignments[p_sigma] = 1.0 # Default std dev

        # --- Store dimensions ---
        self.n_stat = len(self.ordered_stat_vars)
        self.n_s_shock = len(self.stat_shocks)
        self.n_t_shock = len(self.trend_shocks)
        self.n_obs = len(self.obs_vars)
        # Trend state vars determined during solve
        self.n_trend = -1 # Will be set later
        self.n_aug = -1
        self.n_aug_shock = -1
        self.aug_state_vars = []
        self.aug_shocks = []
        self.ordered_trend_state_vars = []
        self.contemp_trend_defs = {}
        self.ordered_obs_vars = []

        self._parsed = True
        if self.verbose: print("--- Model Structure Parsing Complete ---")


    # def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
    #     """
    #     Solves the model for given parameters using JAX AD matrices.

    #     Args:
    #         param_dict: Dictionary mapping parameter names to values.

    #     Returns:
    #         Dictionary containing:
    #          - P_aug (jax.Array): Augmented transition matrix.
    #          - R_aug (jax.Array): Augmented shock impact matrix (scaled by std dev).
    #          - Omega (jax.Array): Observation matrix.
    #          - solution_valid (bool): True if solve successful, False otherwise.
    #          - ordered_trend_state_vars (List[str]): Names of trend state vars.
    #          - contemp_trend_defs (Dict): Contemporaneous trend definitions.
    #          - ordered_obs_vars (List[str]): Names of observable vars.
    #          - aug_state_vars (List[str]): Names of augmented state vars.
    #          - aug_shocks (List[str]): Names of augmented shocks.
    #          - n_aug, n_aug_shock, n_obs (int): Dimensions.
    #     """
    #     if not self._parsed: self._parse_model()

    #     results = {"solution_valid": False} # Default to invalid

    #     try:
    #         # --- STEP 1: Compute Stationary Matrices (Unordered) ---
    #         matrices_unordered, _ = compute_matrices_jax_ad(
    #             equations=self.processed_equations_stat,
    #             var_names=self.initial_stat_vars, # Use initial order
    #             shock_names=self.stat_shocks,
    #             param_names=self.all_param_names, # Use combined list
    #             param_values=param_dict,
    #             model_type="stationary",
    #             dtype=_DEFAULT_DTYPE
    #         )
    #         A_unord = matrices_unordered['A']
    #         B_unord = matrices_unordered['B']
    #         C_unord = matrices_unordered['C']
    #         D_unord = matrices_unordered['D']

    #          # --- STEP 1b: Reorder Stationary Matrices ---
    #         # Use pre-calculated permutation indices from __init__
    #         eq_perm_indices_jax = jnp.array(self.eq_perm_indices_stat)
    #         var_perm_indices_jax = jnp.array(self.var_perm_indices_stat)
    #         A_num_stat = A_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
    #         B_num_stat = B_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
    #         C_num_stat = C_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
    #         D_num_stat = D_unord[jnp.ix_(eq_perm_indices_jax, jnp.arange(self.n_s_shock))]


    #         # --- STEP 2: Solve Stationary Model ---
    #         P_sol_stat, _, _, converged = solve_quadratic_matrix_equation_jax(
    #             A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500
    #         )
    #         if not converged or not jnp.all(jnp.isfinite(P_sol_stat)):
    #             if self.verbose: print("Warning: Stationary SDA solver failed.")
    #             return results # Return invalid results
    #         Q_sol_stat = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
    #         if not jnp.all(jnp.isfinite(Q_sol_stat)):
    #              if self.verbose: print("Warning: Stationary Q computation failed (likely singular A*P+B).")
    #              return results


    #         # --- STEP 3: Build Trend Matrices ---
    #         P_num_trend, Q_num_trend, self.ordered_trend_state_vars, self.contemp_trend_defs = build_trend_matrices_jax_ad(
    #             self.trend_equations, self.trend_vars, self.trend_shocks,
    #             self.all_param_names, param_dict, verbose=self.verbose)
    #         self.n_trend = len(self.ordered_trend_state_vars)


    #         # --- STEP 4: Build Observation Matrix ---
    #         Omega_num, self.ordered_obs_vars = build_observation_matrix_jax_ad(
    #             self.measurement_equations, self.obs_vars, self.ordered_stat_vars,
    #             self.ordered_trend_state_vars, self.contemp_trend_defs,
    #             self.all_param_names, param_dict, verbose=self.verbose)


    #         # --- STEP 5: Build R Matrices & Augmented System ---
    #         shock_std_devs = {}
    #         self.aug_shocks = self.stat_shocks + self.trend_shocks
    #         for shock_name in self.aug_shocks:
    #             sigma_param_name = f"sigma_{shock_name}"
    #             std_dev = param_dict.get(sigma_param_name)
    #             if std_dev is None:
    #                 print(f"CRITICAL WARNING: Sigma parameter '{sigma_param_name}' missing in param_dict during solve!")
    #                 std_dev = 1.0 # Fallback, but indicates an issue
    #             shock_std_devs[shock_name] = jnp.maximum(std_dev, 1e-9) # Ensure positivity for diag

    #         stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE)
    #         trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE)

    #         R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if self.n_s_shock > 0 else jnp.zeros((self.n_stat, 0), dtype=_DEFAULT_DTYPE)
    #         R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if self.n_t_shock > 0 else jnp.zeros((self.n_trend, 0), dtype=_DEFAULT_DTYPE)

    #         # Build Augmented P_aug, R_aug
    #         self.n_aug = self.n_stat + self.n_trend
    #         self.n_aug_shock = self.n_s_shock + self.n_t_shock
    #         self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars

    #         P_aug = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
    #         R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=P_aug.dtype)
    #         if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock): R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
    #         if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock): R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)

    #         # --- Store results ---
    #         results["P_aug"] = P_aug
    #         results["R_aug"] = R_aug # Note: This is R = Q_impact @ diag(std)
    #         results["Omega"] = Omega_num
    #         results["solution_valid"] = True
    #         results["ordered_trend_state_vars"] = self.ordered_trend_state_vars
    #         results["contemp_trend_defs"] = self.contemp_trend_defs
    #         results["ordered_obs_vars"] = self.ordered_obs_vars
    #         results["aug_state_vars"] = self.aug_state_vars
    #         results["aug_shocks"] = self.aug_shocks
    #         results["n_aug"] = self.n_aug
    #         results["n_aug_shock"] = self.n_aug_shock
    #         results["n_obs"] = self.n_obs

    #         # Final validation
    #         if not jnp.all(jnp.isfinite(P_aug)) or not jnp.all(jnp.isfinite(R_aug)) or not jnp.all(jnp.isfinite(Omega_num)):
    #             if self.verbose: print("Warning: NaNs/Infs found in final augmented matrices.")
    #             results["solution_valid"] = False


    #     except Exception as e:
    #         if self.verbose: print(f"Error during model solve: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         results["solution_valid"] = False # Ensure invalid on error

    #     return results

# Inside DynareModelWithJAXAD class

    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Solves the model for given parameters using JAX AD matrices.
        Returns results dictionary including a JAX boolean 'solution_valid'.
        """
        if not self._parsed: self._parse_model()

        results = {"solution_valid": jnp.array(False)} # Default to invalid (JAX bool)
        P_aug, R_aug, Omega_num = None, None, None # Initialize

        try:
            # --- STEP 1: Compute and Reorder Stationary Matrices ---
            matrices_unordered, _ = compute_matrices_jax_ad(
                self.processed_equations_stat, self.initial_stat_vars, self.stat_shocks,
                self.all_param_names, param_dict, "stationary", _DEFAULT_DTYPE
            )
            A_unord, B_unord, C_unord, D_unord = (matrices_unordered['A'], matrices_unordered['B'],
                                                matrices_unordered['C'], matrices_unordered['D'])
            eq_perm_indices_jax = jnp.array(self.eq_perm_indices_stat)
            var_perm_indices_jax = jnp.array(self.var_perm_indices_stat)
            A_num_stat = A_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
            B_num_stat = B_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
            C_num_stat = C_unord[jnp.ix_(eq_perm_indices_jax, var_perm_indices_jax)]
            D_num_stat = D_unord[jnp.ix_(eq_perm_indices_jax, jnp.arange(self.n_s_shock))]

            # --- STEP 2: Solve Stationary Model ---
            P_sol_stat, _, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500
            )
            # Check convergence immediately (converged_stat is a JAX bool)
            # Also check if P_sol_stat contains NaN (implicitly checked by converged_stat)
            valid_stat_solve = converged_stat

            # Compute Q only if solve was valid
            Q_sol_stat = jnp.where(
                 valid_stat_solve,
                 compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat),
                 jnp.full_like(D_num_stat, jnp.nan) # Return NaN Q if P invalid
            )
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat)) # Check Q finiteness

            # --- STEP 3 & 4: Build Trend & Observation Matrices ---
            P_num_trend, Q_num_trend, self.ordered_trend_state_vars, self.contemp_trend_defs = build_trend_matrices_jax_ad(
                self.trend_equations, self.trend_vars, self.trend_shocks,
                self.all_param_names, param_dict, verbose=False) # Less verbose solve
            self.n_trend = len(self.ordered_trend_state_vars)

            Omega_num, self.ordered_obs_vars = build_observation_matrix_jax_ad(
                self.measurement_equations, self.obs_vars, self.ordered_stat_vars,
                self.ordered_trend_state_vars, self.contemp_trend_defs,
                self.all_param_names, param_dict, verbose=False)

            # --- STEP 5: Build R Matrices & Augmented System ---
            # ... (calculate R_sol_stat, R_num_trend using shock_std_devs) ...
            shock_std_devs = {}
            self.aug_shocks = self.stat_shocks + self.trend_shocks
            for shock_name in self.aug_shocks:
                sigma_param_name = f"sigma_{shock_name}"
                std_dev = param_dict.get(sigma_param_name, 1.0) # Default 1 if missing (shouldn't happen)
                # Ensure positivity and non-zero for diag
                shock_std_devs[shock_name] = jnp.maximum(jnp.abs(std_dev), 1e-9)

            stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE)

            R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if self.n_s_shock > 0 else jnp.zeros((self.n_stat, 0), dtype=_DEFAULT_DTYPE)
            R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if self.n_t_shock > 0 else jnp.zeros((self.n_trend, 0), dtype=_DEFAULT_DTYPE)

            self.n_aug = self.n_stat + self.n_trend
            self.n_aug_shock = self.n_s_shock + self.n_t_shock
            self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars

            P_aug = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=P_aug.dtype)
            # Use jnp.where to handle potentially NaN R_sol_stat
            if self.n_stat > 0 and self.n_s_shock > 0: R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
            if self.n_trend > 0 and self.n_t_shock > 0: R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)


            # --- Final Validity Check (JAX compatible) ---
            all_finite = (
                jnp.all(jnp.isfinite(P_aug)) &
                jnp.all(jnp.isfinite(R_aug)) &
                jnp.all(jnp.isfinite(Omega_num))
            )
            # Solution is valid only if stat solve converged AND Q computation worked AND final matrices finite
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite

            # Store results (matrices might contain NaNs if !solution_valid_final)
            results["P_aug"] = P_aug
            results["R_aug"] = R_aug
            results["Omega"] = Omega_num
            results["solution_valid"] = solution_valid_final # Store JAX bool
            results["ordered_trend_state_vars"] = self.ordered_trend_state_vars # Python list
            results["contemp_trend_defs"] = self.contemp_trend_defs # Python dict
            results["ordered_obs_vars"] = self.ordered_obs_vars # Python list
            results["aug_state_vars"] = self.aug_state_vars # Python list
            results["aug_shocks"] = self.aug_shocks # Python list
            results["n_aug"] = self.n_aug # Python int
            results["n_aug_shock"] = self.n_aug_shock # Python int
            results["n_obs"] = self.n_obs # Python int


        except Exception as e:
            if self.verbose: print(f"Exception during model solve: {e}")
            # Ensure results dict contains default valid=False and potentially None matrices
            results["solution_valid"] = jnp.array(False) # Ensure JAX bool False on error
            # Optionally clear potentially partially computed matrices
            results["P_aug"], results["R_aug"], results["Omega"] = None, None, None

        return results
    

    # def log_likelihood(self,
    #                 param_dict: Dict[str, float],
    #                 ys: jax.Array,
    #                 H_obs: jax.Array, # Observation noise COVARIANCE
    #                 init_x_mean: jax.Array,
    #                 init_P_cov: jax.Array) -> float:
    #     """
    #     Computes the log-likelihood using Dynamax filter, handling potential
    #     solver failures gracefully using jax.lax.cond based *only* on the validity flag.
    #     Relies on solve() returning valid=False or filter failing for bad params.
    #     """
    #     # Ensure prerequisites are met before entering JAX-traced logic
    #     if not DYNAMAX_AVAILABLE:
    #         # This error is better raised during setup than returning -inf repeatedly
    #         raise RuntimeError("Dynamax library is required for log_likelihood.")
    #     if ys is None or H_obs is None or init_x_mean is None or init_P_cov is None:
    #         raise ValueError("Missing required input (ys, H_obs, init_x_mean, or init_P_cov).")


    #     # --- Define functions for jax.lax.cond branches ---
    #     def _calculate_likelihood(pd): # Accepts the param_dict
    #         """Calculates likelihood assuming solve was successful."""
    #         # Re-solve inside the branch to ensure tracing works correctly with cond
    #         # This might seem redundant but helps with JAX control flow primitives
    #         solution = self.solve(pd)

    #         # Since this branch is only executed if the initial solve outside cond was valid,
    #         # we proceed directly, but wrap in try/except for runtime filter/numerical errors.
    #         try:
    #             # Extract results - assume they are valid based on the outer cond check
    #             P_aug = solution["P_aug"]
    #             R_aug = solution["R_aug"]
    #             Omega = solution["Omega"]
    #             n_aug = solution["n_aug"] # Get actual dimension used in solve
    #             n_obs = solution["n_obs"]

    #             # Calculate Q_dyn and add jitter
    #             # Use jnp.maximum to prevent issues if std dev param is exactly zero
    #             # (although priors/transforms should ideally prevent this)
    #             Q_dyn = R_aug @ R_aug.T
    #             q_jitter = 1e-8 # Slightly increased jitter for robustness
    #             Q_dyn = Q_dyn + q_jitter * jnp.eye(n_aug, dtype=Q_dyn.dtype)

    #             # Instantiate and Initialize Dynamax LGSSM
    #             lgssm = LinearGaussianSSM(n_aug, n_obs, input_dim=0)
    #             lgssm_params, _ = lgssm.initialize(
    #                 initial_mean=init_x_mean, initial_covariance=init_P_cov,
    #                 dynamics_weights=P_aug, dynamics_covariance=Q_dyn,
    #                 emission_weights=Omega, emission_covariance=H_obs
    #             )

    #             # Compute Log Likelihood via Filtering
    #             filtered_results = lgssm_filter(lgssm_params, emissions=ys)
    #             log_prob = filtered_results.marginal_loglik

    #             # Return likelihood or -inf if filter itself produced non-finite result
    #             # Ensure dtype matches the default float type for consistency
    #             return jnp.where(jnp.isfinite(log_prob), log_prob, jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE))

    #         except Exception as e:
    #              # Catch potential runtime errors *within* this valid branch
    #              # (e.g., LinAlgError from filter if Q_dyn still numerically bad)
    #              if self.verbose: print(f"[LogLik Debug] Exception in _calculate_likelihood branch: {type(e).__name__}: {e}")
    #              return jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)

    #     def _return_invalid_likelihood(pd): # Accepts param_dict (operand)
    #         """Returns -inf when solve failed."""
    #         # Match the expected return type (JAX scalar array)
    #         return jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)

    #     # --- Perform the solve ONCE to get the validity flag ---
    #     # Crucially, ensure solve() itself does not contain Python `if` on tracers
    #     try:
    #          solution_check = self.solve(param_dict)
    #          # Ensure is_valid is a JAX boolean scalar, default to False if key missing
    #          is_valid = jnp.asarray(solution_check.get("solution_valid", False))

    #     except Exception as e_solve_outer:
    #          # Catch error during the initial solve attempt
    #          if self.verbose: print(f"[LogLik Debug] Exception during initial solve: {type(e_solve_outer).__name__}: {e_solve_outer}")
    #          return jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)

# --- Inside DynareModelWithJAXAD class ---

    def log_likelihood(self,
                       param_dict: Dict[str, float],
                       ys: jax.Array,
                       H_obs: jax.Array, # Observation noise COVARIANCE
                       init_x_mean: jax.Array,
                       init_P_cov: jax.Array) -> float:
        """
        Computes the log-likelihood using the custom KalmanFilter class,
        handling potential solver failures gracefully using jax.lax.cond.
        """
        # Ensure prerequisites are met before entering JAX-traced logic
        if not KALMAN_FILTER_JAX_AVAILABLE:
            raise RuntimeError("Custom KalmanFilter class is required.")
        if ys is None or H_obs is None or init_x_mean is None or init_P_cov is None:
            raise ValueError("Missing required input (ys, H_obs, init_x_mean, or init_P_cov).")

        # --- Define functions for jax.lax.cond branches ---
        def _calculate_likelihood(pd): # Accepts the param_dict
            """Calculates likelihood assuming solve was successful."""
            # Re-solve inside the branch
            solution = self.solve(pd)

            # Assume validity based on outer cond, wrap calculation in try/except
            try:
                # Extract results
                P_aug = solution["P_aug"]
                R_aug = solution["R_aug"] # NOTE: Your KF expects R s.t. Q=R@R.T
                Omega = solution["Omega"]
                # Dimensions are needed for KF constructor if not passed explicitly
                n_aug = solution["n_aug"]
                n_obs = solution["n_obs"]
                n_shocks = solution["n_aug_shock"]

                # Add jitter directly to covariance matrices if KF needs it internally
                # Or ensure KF handles it. Let's assume KF adds jitter if needed.

                # --- Instantiate your KalmanFilter ---
                kf = KalmanFilter(
                    T = P_aug,
                    R = R_aug,     # Pass R directly
                    C = Omega,
                    H = H_obs,     # Observation noise COVARIANCE
                    init_x = init_x_mean,
                    init_P = init_P_cov
                )

                # --- Compute Log Likelihood using your filter's method ---
                log_prob = kf.log_likelihood(ys)

                # Return likelihood or -inf if filter itself produced non-finite result
                return jnp.where(jnp.isfinite(log_prob), log_prob, jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE))

            except Exception as e:
                 if self.verbose: print(f"[LogLik Debug] Exception in _calculate_likelihood branch (using custom KF): {type(e).__name__}: {e}")
                 # Optional: Print traceback for internal errors
                 # import traceback
                 # traceback.print_exc()
                 return jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)

        def _return_invalid_likelihood(pd): # Accepts param_dict (operand)
            """Returns -inf when solve failed."""
            return jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)

        # --- Perform the solve ONCE to get the validity flag ---
        try:
             solution_check = self.solve(param_dict)
             is_valid = jnp.asarray(solution_check.get("solution_valid", False))
        except Exception as e_solve_outer:
             if self.verbose: print(f"[LogLik Debug] Exception during initial validity solve: {type(e_solve_outer).__name__}: {e_solve_outer}")
             return jnp.array(-jnp.inf, dtype=_DEFAULT_DTYPE)

        # --- Use lax.cond ---
        log_prob = jax.lax.cond(
            pred=is_valid,
            true_fun=_calculate_likelihood,
            false_fun=_return_invalid_likelihood,
            operand=param_dict # Pass parameters needed by the branches
        )

        return log_prob
# --- End of Revised log_likelihood ---



def numpyro_model(
    model_instance: DynareModelWithJAXAD, # Pass the instantiated model
    user_priors: List[Dict[str, Any]], # List of prior specs for estimated params
    fixed_param_values: Dict[str, float], # Values for non-estimated params
    ys: Optional[jax.Array] = None, # Observations, None if prior predictive
    H_obs: Optional[jax.Array] = None, # Fixed observation noise covariance
    init_x_mean: Optional[jax.Array] = None, # Fixed initial state mean
    init_P_cov: Optional[jax.Array] = None # Fixed initial state covariance
):
    """
    Numpyro model function for estimating a subset of Dynare parameters.

    Args:
        model_instance: An instantiated DynareModelWithJAXAD object.
        user_priors: List defining priors for parameters TO BE ESTIMATED.
                     Example: [{"name": "p1", "prior": "Normal", "args": {"loc":0,"scale":1}}, ...]
        fixed_param_values: Dictionary containing name:value pairs for parameters
                            NOT estimated (fixed at these values).
        ys: Observed data array (num_timesteps, n_obs).
        H_obs: Observation noise covariance matrix.
        init_x_mean: Initial state mean vector.
        init_P_cov: Initial state covariance matrix.
    """
    if not NUMPYRO_AVAILABLE:
        raise RuntimeError("Numpyro is required for this model function.")

    # Dictionary to hold all parameter values (sampled and fixed) for this MCMC step
    params_for_likelihood = {}

    # Set of parameters that have priors specified by the user
    estimated_param_names = {p_spec["name"] for p_spec in user_priors}

    # --- Sample Parameters with Priors ---
    for prior_spec in user_priors:
        name = prior_spec["name"]
        dist_name = prior_spec.get("prior", "").lower()
        args = prior_spec.get("args", {})

        # Ensure args values are JAX arrays with correct dtype
        dist_args_processed = {k: jnp.asarray(v, dtype=_DEFAULT_DTYPE) for k, v in args.items()}

        sampled_value = None
        try:
            if dist_name == "normal":
                loc = dist_args_processed.get("loc", 0.0)
                scale = jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7) # Ensure positive scale
                sampled_value = numpyro.sample(name, dist.Normal(loc, scale))
            elif dist_name == "beta":
                c1 = jnp.maximum(dist_args_processed.get("concentration1", 1.0), 1e-7) # Ensure positive
                c2 = jnp.maximum(dist_args_processed.get("concentration2", 1.0), 1e-7) # Ensure positive
                sampled_value = numpyro.sample(name, dist.Beta(c1, c2))
            elif dist_name == "gamma":
                conc = jnp.maximum(dist_args_processed.get("concentration", 1.0), 1e-7) # Ensure positive
                rate = jnp.maximum(dist_args_processed.get("rate", 1.0), 1e-7) # Ensure positive
                sampled_value = numpyro.sample(name, dist.Gamma(conc, rate=rate))
            elif dist_name == "invgamma":
                conc = jnp.maximum(dist_args_processed.get("concentration", 1.0), 1e-7) # Ensure positive
                # Numpyro uses scale parameter (which is 1/rate)
                scale = jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7) # Ensure positive
                sampled_value = numpyro.sample(name, dist.InverseGamma(conc, scale))
            # Add other distributions as needed (e.g., Uniform)
            # elif dist_name == "uniform":
            #     low = dist_args_processed.get("low", 0.0)
            #     high = dist_args_processed.get("high", 1.0)
            #     sampled_value = numpyro.sample(name, dist.Uniform(low, high))
            else:
                raise NotImplementedError(f"Prior distribution '{dist_name}' not implemented or specified for '{name}'.")

            params_for_likelihood[name] = sampled_value

        except KeyError as e:
             raise ValueError(f"Missing required argument for prior '{dist_name}' for parameter '{name}': {e}")
        except Exception as e_dist:
             raise RuntimeError(f"Error sampling parameter '{name}' with prior '{dist_name}': {e_dist}")

    # --- Add Fixed Parameters ---
    for name, value in fixed_param_values.items():
        if name not in estimated_param_names:
            # Important: Convert fixed value to JAX array for type consistency
            params_for_likelihood[name] = jnp.asarray(value, dtype=_DEFAULT_DTYPE)

    # --- Verify all parameters are present ---
    missing_keys = set(model_instance.all_param_names) - set(params_for_likelihood.keys())
    if missing_keys:
        # Provide more info in the error
        raise RuntimeError(f"Internal Error: Parameters missing before likelihood calculation: {missing_keys}. Estimated: {estimated_param_names}, Fixed: {set(fixed_param_values.keys())}")
    extra_keys = set(params_for_likelihood.keys()) - set(model_instance.all_param_names)
    if extra_keys:
         raise RuntimeError(f"Internal Error: Extra parameters found before likelihood calculation: {extra_keys}")

    # # --- Calculate Log Likelihood ---
    # if ys is not None:
    #     if not DYNAMAX_AVAILABLE:
    #          raise RuntimeError("Dynamax needed for likelihood calculation.")
    #     if H_obs is None or init_x_mean is None or init_P_cov is None:
    #          raise ValueError("H_obs, init_x_mean, init_P_cov are required when ys is provided.")

        # Use the combined dictionary of sampled and fixed parameters
    log_prob = model_instance.log_likelihood(
        params_for_likelihood,
        ys,
        H_obs,
        init_x_mean,
        init_P_cov
    )
    # Register the log likelihood with Numpyro
    numpyro.factor("log_likelihood", log_prob)



# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Make sure this points to your actual model file
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file) # Allow override via env var

    # Simulation settings (for generating test data)
    num_sim_steps = 200
    sim_seed = 123
    sim_measurement_noise_std = 0.001 # Add some noise to make estimation non-trivial

    # Estimation settings
    run_estimation_flag = True # Set to False to only simulate/test solve
    mcmc_seed = 456
    mcmc_chains = jax.local_device_count() # Use available CPUs/devices
    mcmc_warmup = 500
    mcmc_samples = 1000
    mcmc_target_accept = 0.85

    # --- [1] Initialize the Model Wrapper ---
    print(f"\n--- [1] Initializing Dynare Model ({mod_file_path}) ---")
    init_start_time = time.time()
    try:
        model = DynareModelWithJAXAD(mod_file_path, verbose=False) # Less verbose init
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {mod_file_path}")
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize DynareModelWithJAXAD: {e}")
        import traceback
        traceback.print_exc()
        exit()
    init_end_time = time.time()
    print(f"Model initialized ({init_end_time - init_start_time:.2f} seconds).")
    print(f"  Found {len(model.all_param_names)} parameters total.")


    # --- [2] Define Parameter Set for Simulation ---
    # Use defaults and apply overrides for simulation
    sim_param_values = model.default_param_assignments.copy()
    # Apply overrides used in previous examples
    # sim_param_overrides = {
    #     'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12, 'g1': 0.7,
    #     'g2': 0.3, 'g3': 0.25, 'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
    #     'rho_rs': 0.75, 'rho_rs2': 0.05,
    #     # Ensure sigma overrides are included if different from defaults
    #     "sigma_SHK_L_GDP_GAP": 0.10,
    #     "sigma_SHK_DLA_CPI": 1.5, # Using dyn file value
    #     "sigma_SHK_RS": 0.5, # Using dyn file value
    #     "sigma_SHK_L_GDP_TREND": 0.001, # Using dyn file value
    #     "sigma_SHK_G_TREND": 0.0, # Make growth constant for clearer simulation
    #     "sigma_SHK_PI_TREND": 1e-10, # Using dyn file value
    #     "sigma_SHK_RR_TREND": 0.1, # Using dyn file value
    # }
    #sim_param_values.update(sim_param_overrides)

    # Verify all model parameters are present
    missing_sim_params = [p for p in model.all_param_names if p not in sim_param_values]
    if missing_sim_params:
        print(f"FATAL ERROR: Missing simulation parameter values: {missing_sim_params}")
        exit()
    print("\n--- [2] Simulation parameter set defined ---")

    # --- [3] Generate Simulated Data ---
    print("\n--- [3] Simulating Data ---")
    sim_key_master = random.PRNGKey(sim_seed)
    sim_key_init, sim_key_path = random.split(sim_key_master)

    # Solve once with sim params to get dimensions
    sim_solution = model.solve(sim_param_values)
    if not sim_solution["solution_valid"]:
         print("FATAL ERROR: Cannot solve model with simulation parameters.")
         exit()

    # Define initial state configuration for simulation
    sim_initial_state_config = {
        "L_GDP_TREND": {"mean": 10000.0, "std": 0.01},
        "G_TREND":     {"mean": 0.5, "std": 0.0}, # Exact start for constant growth
        "PI_TREND":    {"mean": 2.0, "std": 0.0}, # Exact start
        "RR_TREND":    {"mean": 1.0, "std": 0.01}
    }
    s0_sim = construct_initial_state(
        n_aug=sim_solution["n_aug"], n_stat=model.n_stat,
        aug_state_vars=sim_solution["aug_state_vars"], key_init=sim_key_init,
        initial_state_config=sim_initial_state_config, dtype=_DEFAULT_DTYPE
    )

    # Ensure observation noise matrix is correct size and positive definite
    n_obs_sim = sim_solution["n_obs"]
    H_obs_sim = jnp.eye(n_obs_sim, dtype=_DEFAULT_DTYPE) * (sim_measurement_noise_std**2)

    sim_start_time = time.time()
    try:
        sim_states, sim_observables = simulate_ssm_data(
            P=sim_solution["P_aug"],
            R=sim_solution["R_aug"],
            Omega=sim_solution["Omega"],
            T=num_sim_steps,
            key=sim_key_path,
            state_init=s0_sim,
            # Pass H_obs to simulate_ssm_data if it handles obs noise, otherwise add manually
            measurement_noise_std=sim_measurement_noise_std # Assuming simulate_ssm_data adds it based on std
        )
        sim_end_time = time.time()
        print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")

        # Plot simulation results
        plot_simulation_with_trends_matched(
            sim_observables, sim_solution["ordered_obs_vars"],
            sim_states, sim_solution["aug_state_vars"],
            sim_solution["ordered_trend_state_vars"], sim_solution["contemp_trend_defs"],
            title=f"Simulated Data (Meas Noise Std={sim_measurement_noise_std:.2e})"
        )
        plt.show(block=False)

    except Exception as e_sim:
        sim_end_time = time.time()
        print(f"FATAL ERROR during simulation: {e_sim}")
        import traceback
        traceback.print_exc()
        exit()


    print("\n--- [4] Defining Priors for Estimation ---")
    # ONLY list parameters you want to estimate here
    user_priors = [
        # Example: Estimate persistence and response coefficients
        # {"name": "b1", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}}, # Mean ~0.75
        # {"name": "a1", "prior": "beta", "args": {"concentration1": 20.0, "concentration2": 15.0}}, # Mean ~0.57
        # {"name": "g1", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}}, # Mean ~0.75
        # {"name": "g2", "prior": "normal", "args": {"loc": 0.3, "scale": 0.1}},
        # {"name": "g3", "prior": "normal", "args": {"loc": 0.25, "scale": 0.1}},
        # {"name": "rho_L_GDP_GAP", "prior": "Beta", "args": {"concentration1": 40.0, "concentration2": 10.0}}, # Mean ~0.8
        # # Example: Estimate key shock std devs
        # {"name": "sigma_SHK_L_GDP_GAP", "prior": "invgamma", "args": {"concentration": 4.0, "scale": 0.1}}, # Approx Mean 0.03
        # {"name": "sigma_SHK_DLA_CPI", "prior": "invgamma", "args": {"concentration": 3.0, "scale": 0.5}}, # Approx Mean 0.25 (adjust if needed)
         {"name": "sigma_SHK_RS", "prior": "invgamma", "args": {"concentration": 3.0, "scale": 0.2}}, # Approx Mean 0.1
    ]
    estimated_param_names_set = {p["name"] for p in user_priors}


    # --- Determine Fixed Parameters and their values ---
    fixed_params = {}
    print("  Parameters treated as FIXED (using .dyn file values):")
    for name in model.all_param_names:
        if name not in estimated_param_names_set:
            # Get value from parsed defaults (which include .dyn assignments)
            value = model.default_param_assignments.get(name)
            if value is None:
                # This should not happen if parser guarantees defaults
                raise ValueError(f"Critical: No default value found for parameter '{name}' intended to be fixed.")
            fixed_params[name] = value
            print(f"    - {name} = {value:.4f}")


           

    # --- [5] Run Estimation ---
    if run_estimation_flag and NUMPYRO_AVAILABLE: #and DYNAMAX_AVAILABLE:
        print(f"\n--- [5] Running Bayesian Estimation (Estimating {len(user_priors)} parameters) ---")
        mcmc_key = random.PRNGKey(mcmc_seed)

        # Define fixed inputs for the likelihood (unchanged)
        H_obs_est = H_obs_sim # Use same observation noise as simulation
        n_aug_est = sim_solution["n_aug"]
        init_x_mean_est = jnp.zeros(n_aug_est, dtype=_DEFAULT_DTYPE)
        init_x_mean_est = s0_sim
        init_P_cov_est = jnp.eye(n_aug_est, dtype=_DEFAULT_DTYPE) * 1.0




        # --- Initial values ONLY for ESTIMATED parameters ---
        init_values_mcmc = {}
        print("  Setting initial MCMC values from simulation parameters for:")
        for p_spec in user_priors:
             name = p_spec["name"]
             if name in sim_param_values:
                 init_values_mcmc[name] = sim_param_values[name]
                 print(f"    - {name} = {sim_param_values[name]:.4f}")
             else:
                 # Fallback if sim value not available (shouldn't happen here)
                 init_values_mcmc[name] = fixed_params.get(name, 0.0) # Or sample from prior mean
                 print(f"    - {name} (using fixed/default as fallback init)")

        init_strategy = init_to_value(values=init_values_mcmc) # Pass only estimated params



     # --- [DEBUG] Test Log Likelihood at Initial Parameters ---
        print("\n--- [DEBUG] Testing log_likelihood at initial parameters ---")
        # Combine initial estimated parameters with fixed parameters
        initial_params_full = fixed_params.copy()
        initial_params_full.update(init_values_mcmc)
    
        # Ensure all parameters are present (sanity check)
        if set(initial_params_full.keys()) != set(model.all_param_names):
            print("ERROR: Mismatch between initial_params_full and model.all_param_names!")
        else:
            # Temporarily set verbose=True in the model instance for detailed output
            model.verbose = True
            try:
                initial_log_lik = model.log_likelihood(
                    initial_params_full,
                    sim_observables,
                    H_obs_est,
                    init_x_mean_est,
                    init_P_cov_est
                )
                print(f"--- [DEBUG] Log Likelihood at initial params: {initial_log_lik} ---")
                if not jnp.isfinite(initial_log_lik):
                    print("--- [DEBUG] FAILURE: Initial log likelihood is non-finite! ---")
                    # Optionally exit here if you want to stop before MCMC
                    # exit()
                else:
                    print("--- [DEBUG] SUCCESS: Initial log likelihood is finite. ---")

            except Exception as e_debug:
                print(f"--- [DEBUG] FAILURE: Error during initial log_likelihood test: {e_debug} ---")
                import traceback
                traceback.print_exc()
                # Optionally exit
                # exit()
            finally:
                # Set verbose back to False if needed for MCMC run
                model.verbose = False # Set back for cleaner MCMC logs
        # --- [END DEBUG] ---

        # Instantiate NUTS kernel (unchanged)
        kernel = NUTS(numpyro_model, init_strategy=init_strategy, target_accept_prob=mcmc_target_accept)

        # Instantiate MCMC (unchanged)
        mcmc = MCMC(
            kernel, 
            num_warmup=mcmc_warmup, 
            num_samples=mcmc_samples,
            num_chains=mcmc_chains, 
            progress_bar=(mcmc_chains == 1),
            chain_method='parallel' if mcmc_chains > 1 else 'sequential',
            jit_model_args=True
        )

        # Run MCMC - Pass user_priors and fixed_params
        print(f"Starting MCMC...")
        est_start_time = time.time()
        try:
            mcmc.run(
                mcmc_key,
                model,               # Positional arg 1 -> model_instance
                user_priors,         # Positional arg 2 -> user_priors
                fixed_params,        # Positional arg 3 -> fixed_param_values
                sim_observables,     # Positional arg 4 -> ys
                H_obs_est,           # Positional arg 5 -> H_obs
                init_x_mean_est,     # Positional arg 6 -> init_x_mean
                init_P_cov_est       # Positional arg 7 -> init_P_cov
            )
            est_end_time = time.time()
            print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")

            # --- [6] Analyze Estimation Results ---
            print("\n--- [6] Estimation Summary (Estimated Parameters) ---")
            # Print summary only for estimated parameters
            
            mcmc.print_summary()
            posterior_samples = mcmc.get_samples()

            # Plotting trace plots (Optional)
            try:
                import arviz as az
                print(" Generating trace plots...")
                az_data = az.from_numpyro(mcmc) # Arviz automatically gets sampled vars
                az.plot_trace(az_data) # Plots all estimated parameters
                plt.suptitle("Trace Plots (Estimated Parameters)", y=1.02)
                plt.tight_layout()
                plt.show(block=False)
                print(" Trace plots generated (may be in background).")
            except ImportError:
                print(" Install arviz (`pip install arviz`) to see trace plots.")
            except Exception as e_trace:
                print(f" Could not generate trace plots: {e_trace}")

        except Exception as e_est:
            est_end_time = time.time()
            print(f"\n--- Estimation FAILED ({est_end_time - est_start_time:.2f} seconds) ---")
            print(f"An error occurred during estimation: {e_est}")
            import traceback
            traceback.print_exc()

    elif not run_estimation_flag:
        print("\n--- [5] Skipping Estimation (run_estimation_flag=False) ---")
    else:
        print("\n--- [5] Skipping Estimation (Numpyro or Dynamax not available) ---")


    print(f"\n--- Script finished ---")
    # Keep plots open if shown
    if run_estimation_flag and NUMPYRO_AVAILABLE: #and DYNAMAX_AVAILABLE:
       print("Close plot windows to exit.")
       plt.show()


# --- END OF FILE run_estimation.py ---