# --- START OF FILE run_estimation.py (Modified) ---
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

# --- Import your custom Kalman Filter ---
try:
    # Assume Kalman_filter_jax.py is in the same directory or path
    from Kalman_filter_jax import KalmanFilter # This remains the same
    KALMAN_FILTER_JAX_AVAILABLE = True
    print("Custom KalmanFilter imported successfully.")
except ImportError:
    KALMAN_FILTER_JAX_AVAILABLE = False
    print("Warning: Kalman_filter_jax.py not found. Likelihood calculation will fail.")
# --- End custom Kalman Filter import ---

# --- Import from Parser Engine (was Dynare_parser_sda_solver_correct_order.py) ---
from dynare_parser_engine import (
    extract_declarations,
    extract_model_equations,
    extract_trend_declarations,
    extract_trend_equations,
    extract_observation_declarations,
    extract_measurement_equations,
    extract_stationary_shock_stderrs,
    extract_trend_shock_stderrs,
    parse_lambdify_and_order_model,    # Key function for stationary part
    build_trend_matrices as build_trend_matrices_lambdified, # Will return lambdified funcs
    build_observation_matrix as build_observation_matrix_lambdified, # Will return lambdified funcs
    solve_quadratic_matrix_equation_jax, # SDA solver
    compute_Q_jax,                     # Q computation
    construct_initial_state,           # Helper for simulation/initialization
    simulate_ssm_data,                 # For generating test data
    plot_simulation_with_trends_matched,# For plotting
    plot_irfs                          # For plotting IRFs
)

# --- Dynare Model Class using Lambdified Functions ---

class DynareModelWithLambdified:
    """
    Represents a Dynare model, computing matrices using pre-lambdified functions
    derived from symbolic Jacobians, and providing methods for solving and
    likelihood calculation.
    """
    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose
        self._parsed = False

        # Lambdified functions will be stored here
        self.func_A_stat = None
        self.func_B_stat = None
        self.func_C_stat = None
        self.func_D_stat = None
        self.func_P_trends = None
        self.func_Q_trends = None
        self.func_Omega = None

        # Ordered parameter lists for each set of lambdified functions
        self.param_names_for_stat_funcs = []
        self.param_names_for_trend_funcs = [] # Should be all_param_names
        self.param_names_for_obs_funcs = []   # Should be all_param_names

        self._parse_model() # Parse structure and create lambdified functions on initialization

    def _parse_model(self):
        """Parses the model structure, generates symbolic matrices, and lambdifies them."""
        if self._parsed:
            return

        if self.verbose: print("--- Parsing Model Structure & Lambdifying Matrices ---")
        with open(self.mod_file_path, 'r') as f:
            model_def = f.read()

        # --- Stationary Part ---
        # `parse_lambdify_and_order_model` handles aux vars, ordering, and lambdification
        try:
            (self.func_A_stat, self.func_B_stat, self.func_C_stat, self.func_D_stat,
             self.ordered_stat_vars, self.stat_shocks, self.param_names_for_stat_funcs,
             self.param_assignments_stat, _, self.initial_info_stat # Store initial_info if needed
             ) = parse_lambdify_and_order_model(model_def, verbose=self.verbose)
            # self.param_names_for_stat_funcs is the ordered list of param names for A,B,C,D
        except Exception as e:
            print(f"Error during stationary model parsing and lambdification: {e}")
            import traceback; traceback.print_exc()
            raise

        # --- Trend Part Declarations ---
        self.trend_vars, self.trend_shocks = extract_trend_declarations(model_def)
        self.trend_equations = extract_trend_equations(model_def)
        self.trend_stderr_params = extract_trend_shock_stderrs(model_def) # e.g. {"sigma_SHK_TREND_X": val}

        # --- Observation Part Declarations ---
        self.obs_vars = extract_observation_declarations(model_def)
        self.measurement_equations = extract_measurement_equations(model_def)

        # --- Combine Parameters (Crucial for Trend and Observation lambdified functions) ---
        # Start with parameters known from the stationary model solution
        current_param_names = list(self.param_names_for_stat_funcs)
        current_param_assignments = self.param_assignments_stat.copy()

        # Add parameters from trend shock stderr definitions
        for p_name, p_val in self.trend_stderr_params.items():
            if p_name not in current_param_assignments:
                current_param_assignments[p_name] = p_val
            if p_name not in current_param_names:
                current_param_names.append(p_name) # Add if new

        # Add any inferred sigma parameters for trend shocks if not already present
        inferred_trend_sigmas = [f"sigma_{shk}" for shk in self.trend_shocks]
        for p_sigma_trend in inferred_trend_sigmas:
            if p_sigma_trend not in current_param_assignments:
                current_param_assignments[p_sigma_trend] = 1.0 # Default std dev
                if self.verbose: print(f"Defaulting inferred trend sigma '{p_sigma_trend}' to 1.0")
            if p_sigma_trend not in current_param_names:
                current_param_names.append(p_sigma_trend)

        # `self.all_param_names` will be the master list for trend/obs lambdification
        # The order matters for the lambdified functions.
        # We use the order derived from `param_names_for_stat_funcs` and append new ones.
        self.all_param_names = list(dict.fromkeys(current_param_names)) # Maintain order, ensure unique
        self.default_param_assignments = current_param_assignments

        # --- Lambdify Trend Matrices ---
        # `build_trend_matrices_lambdified` from dynare_parser_engine.py
        # It takes all_param_names for its lambdification process.
        try:
            (self.func_P_trends, self.func_Q_trends, self.ordered_trend_state_vars,
             self.contemp_trend_defs) = build_trend_matrices_lambdified(
                self.trend_equations, self.trend_vars, self.trend_shocks,
                self.all_param_names, # Use the combined, ordered list
                self.default_param_assignments,
                verbose=self.verbose
            )
            self.param_names_for_trend_funcs = list(self.all_param_names) # Store order
        except Exception as e:
            print(f"Error during trend model lambdification: {e}")
            import traceback; traceback.print_exc()
            raise

        # --- Lambdify Observation Matrix ---
        # `build_observation_matrix_lambdified` from dynare_parser_engine.py
        try:
            (self.func_Omega, self.ordered_obs_vars) = build_observation_matrix_lambdified(
                self.measurement_equations, self.obs_vars, self.ordered_stat_vars,
                self.ordered_trend_state_vars, self.contemp_trend_defs,
                self.all_param_names, # Use the combined, ordered list
                self.default_param_assignments,
                verbose=self.verbose
            )
            self.param_names_for_obs_funcs = list(self.all_param_names) # Store order
        except Exception as e:
            print(f"Error during observation model lambdification: {e}")
            import traceback; traceback.print_exc()
            raise

        # --- Store dimensions ---
        self.n_stat = len(self.ordered_stat_vars)
        self.n_s_shock = len(self.stat_shocks)
        self.n_t_shock = len(self.trend_shocks)
        self.n_obs = len(self.ordered_obs_vars)
        self.n_trend = len(self.ordered_trend_state_vars)
        self.n_aug = self.n_stat + self.n_trend
        self.n_aug_shock = self.n_s_shock + self.n_t_shock
        self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars
        self.aug_shocks = self.stat_shocks + self.trend_shocks

        self._parsed = True
        if self.verbose: print("--- Model Structure Parsing & Lambdification Complete ---")

# In run_estimation.py, DynareModelWithLambdified.solve()

    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        if not self._parsed:
            if self.verbose: print("[solve()] Warning: Model not parsed. Calling _parse_model() now.")
            self._parse_model()

        results = {"solution_valid": jnp.array(False, dtype=jnp.bool_)}
        P_aug, R_aug, Omega_num = None, None, None # Initialize to None
        
        # Conditional printing for verbosity, using jax.debug.print for traced values
        if self.verbose:
            # jax.debug.print("Solving with parameters (verbose mode): {params}", params=param_dict) # Prints entire dict, can be large
            # Selective printing:
            keys_to_print_solve_verbose = ['sigma_SHK_RS', 'rho_L_GDP_GAP', 'b1', 'a1']
            # Create a sub-dict for printing to avoid iterating over potentially traced dict in Python
            print_subset = {k: param_dict.get(k) for k in keys_to_print_solve_verbose if k in param_dict}
            if print_subset:
                 jax.debug.print("  [solve() verbose] Input param_dict (subset): {subset}", subset=print_subset)


        try:
            # --- STEP 1: Evaluate Stationary Matrices ---
            stat_param_values_ordered = [
                param_dict.get(p_name, self.default_param_assignments.get(p_name, 1.0))
                for p_name in self.param_names_for_stat_funcs
            ]
            A_num_stat = jnp.asarray(self.func_A_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            B_num_stat = jnp.asarray(self.func_B_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            C_num_stat = jnp.asarray(self.func_C_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            D_num_stat = jnp.asarray(self.func_D_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)

            # --- STEP 2: Solve Stationary Model ---
            # INCREASED max_iter FROM 30 to 200
            P_sol_stat, actual_iters, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=200 
            )
            
            if self.verbose:
                jax.debug.print("  [solve() debug] SDA actual_iters: {iters}", iters=actual_iters)

            valid_stat_solve = converged_stat & jnp.all(jnp.isfinite(P_sol_stat))
            
            Q_sol_stat_if_valid = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
            Q_sol_stat_if_invalid = jnp.full_like(D_num_stat, jnp.nan) # Ensure D_num_stat exists for shape
            
            Q_sol_stat = jnp.where(
                 valid_stat_solve,
                 Q_sol_stat_if_valid,
                 Q_sol_stat_if_invalid 
            )
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat))

            # --- STEP 3: Evaluate Trend & Observation Matrices ---
            all_param_values_ordered = [
                param_dict.get(p_name, self.default_param_assignments.get(p_name, 1.0))
                for p_name in self.all_param_names
            ]
            P_num_trend = jnp.asarray(self.func_P_trends(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            Q_num_trend = jnp.asarray(self.func_Q_trends(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            Omega_num = jnp.asarray(self.func_Omega(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)

            # --- STEP 4: Build R Matrices & Augmented System ---
            shock_std_devs = {}
            for shock_name in self.aug_shocks:
                sigma_param_name = f"sigma_{shock_name}"
                std_dev = param_dict.get(sigma_param_name, self.default_param_assignments.get(sigma_param_name, 1.0))
                shock_std_devs[shock_name] = jnp.maximum(jnp.abs(std_dev), 1e-9) # Ensure positive std dev
            
            stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE)
            
            # Handle cases where Q_sol_stat might be NaN due to invalid solve earlier
            R_sol_stat_if_q_valid = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if self.n_s_shock > 0 and Q_sol_stat.shape[1] == len(stat_std_devs_arr) else jnp.zeros((self.n_stat, 0), dtype=_DEFAULT_DTYPE)
            R_sol_stat = jnp.where(valid_q_compute, R_sol_stat_if_q_valid, jnp.full((self.n_stat, self.n_s_shock if self.n_s_shock > 0 else 0), jnp.nan, dtype=_DEFAULT_DTYPE))

            R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if self.n_t_shock > 0 and Q_num_trend.shape[1] == len(trend_std_devs_arr) else jnp.zeros((self.n_trend, 0), dtype=_DEFAULT_DTYPE)

            P_aug_if_valid = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
            P_aug = jnp.where(valid_stat_solve, P_aug_if_valid, jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE))

            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=P_aug.dtype) # Initialize with correct dtype
            if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock):
                 R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat) # R_sol_stat might be NaN here
            if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock):
                 R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)


            # --- Final Validity Check (all JAX operations) ---
            # R_aug can contain NaNs if R_sol_stat was NaN.
            all_finite_check = (
                jnp.all(jnp.isfinite(P_aug)) &
                jnp.all(jnp.isfinite(R_aug)) & # This will be false if R_sol_stat was NaN
                jnp.all(jnp.isfinite(Omega_num))
            )
            # solution_valid_final depends on valid_stat_solve, valid_q_compute, and then all_finite_check
            # If valid_stat_solve is False, P_sol_stat -> P_aug might be NaN.
            # If valid_q_compute is False, Q_sol_stat -> R_sol_stat -> R_aug might be NaN.
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite_check
            solution_valid_final_jax = jnp.asarray(solution_valid_final, dtype=jnp.bool_)

            if self.verbose:
                jax.debug.print("  [solve() validity details] valid_stat_solve: {x}", x=valid_stat_solve)
                jax.debug.print("  [solve() validity details] valid_q_compute: {x}", x=valid_q_compute)
                jax.debug.print("  [solve() validity details] all_finite (P_aug,R_aug,Omega): {x}", x=all_finite_check)
                jax.debug.print("  [solve() validity details] solution_valid_final_JAX_BOOL: {x}", x=solution_valid_final_jax)
            
            results["P_aug"] = P_aug
            results["R_aug"] = R_aug
            results["Omega"] = Omega_num
            results["solution_valid"] = solution_valid_final_jax
            results["ordered_trend_state_vars"] = self.ordered_trend_state_vars; results["contemp_trend_defs"] = self.contemp_trend_defs
            results["ordered_obs_vars"] = self.ordered_obs_vars; results["aug_state_vars"] = self.aug_state_vars
            results["aug_shocks"] = self.aug_shocks; results["n_aug"] = self.n_aug; results["n_aug_shock"] = self.n_aug_shock
            results["n_obs"] = self.n_obs


        except Exception as e:
            if self.verbose:
                # This print is in Python scope, so direct printing of e is fine.
                print(f"[solve()] Exception during model solve's try block: {type(e).__name__}: {e}")
            results["solution_valid"] = jnp.array(False, dtype=jnp.bool_) 
            results["P_aug"] = None 
            results["R_aug"] = None
            results["Omega"] = None
            results["ordered_trend_state_vars"] = self.ordered_trend_state_vars if hasattr(self, 'ordered_trend_state_vars') else []
            results["contemp_trend_defs"] = self.contemp_trend_defs if hasattr(self, 'contemp_trend_defs') else {}
            results["ordered_obs_vars"] = self.ordered_obs_vars if hasattr(self, 'ordered_obs_vars') else []
            results["aug_state_vars"] = self.aug_state_vars if hasattr(self, 'aug_state_vars') else []
            results["aug_shocks"] = self.aug_shocks if hasattr(self, 'aug_shocks') else []
            results["n_aug"] = self.n_aug if hasattr(self, 'n_aug') else -1 
            results["n_aug_shock"] = self.n_aug_shock if hasattr(self, 'n_aug_shock') else -1
            results["n_obs"] = self.n_obs 

        if self.verbose:
            jax.debug.print("  [solve() end] Returning solution_valid: {valid_flag}", valid_flag=results.get('solution_valid'))
        return results
    
    def log_likelihood(self,
                    param_dict: Dict[str, float], # Params from Numpyro
                    ys: jax.Array,             # Full ys data
                    H_obs: jax.Array,          # Full H matrix
                    init_x_mean: jax.Array,
                    init_P_cov: jax.Array,
                    static_valid_obs_idx: jax.Array, 
                    static_n_obs_actual: int         
                    ) -> float:
        if not KALMAN_FILTER_JAX_AVAILABLE:
            raise RuntimeError("Custom KalmanFilter class is required.")

        LARGE_NEG_VALUE = -1e10 
        desired_dtype = _DEFAULT_DTYPE

        if self.verbose:
            # Selective printing for log_likelihood
            keys_to_print_loglik_verbose = ['sigma_SHK_RS', 'rho_L_GDP_GAP']
            print_subset_loglik = {k: param_dict.get(k) for k in keys_to_print_loglik_verbose if k in param_dict}
            if print_subset_loglik:
                 jax.debug.print("\n[LogLik Top] Called. Estimating params (subset): {subset}", subset=print_subset_loglik)
            jax.debug.print("  LL static_n_obs_actual: {n_actual}", n_actual=static_n_obs_actual)


        def _calculate_likelihood_branch(pd_operand_branch): # Renamed operand
            # This is the "true" branch of the outer lax.cond.
            # self.solve is called again here.
            solution_inner = self.solve(pd_operand_branch) 
            solution_inner_valid_tracer = solution_inner.get("solution_valid", jnp.array(False, dtype=jnp.bool_))

            def ll_if_inner_solve_valid():
                P_aug_inner = solution_inner["P_aug"]
                R_aug_inner = solution_inner["R_aug"]
                Omega_sol_inner = solution_inner["Omega"]
                
                # This Python if P_aug_inner is None should ideally not be hit if solution_inner_valid_tracer is True
                # and P_aug_inner is correctly populated. However, JAX might trace it.
                # To be safe, ensure P_aug_inner, etc., are JAX arrays even if representing failure.
                # The current self.solve returns NaNs for P_aug if solution_valid is false.
                
                # This check might be problematic if P_aug_inner is a tracer that *could* be None
                # Python `if P_aug_inner is None` on a JAX tracer is an error.
                # However, self.solve ensures P_aug, R_aug, Omega are JAX arrays (possibly with NaNs) or None
                # only if an exception occurs *before* their assignment.
                # If solution_inner_valid_tracer is True, P_aug_inner should be a valid JAX array.
                # The problem arises if solution_inner_valid_tracer is False, P_aug_inner has NaNs,
                # and this branch is taken due to JAX tracing both sides of the *outer* cond.
                # This check for None is only problematic if P_aug_inner itself is a tracer.
                # If it's a concrete None (due to an early exception in `solve`), this branch of the *outer* cond
                # shouldn't be taken if the outer `is_valid` was also False.
                # Let's assume if this branch is taken, P_aug_inner is a JAX array (possibly with NaNs if inner_solve_valid is False)

                n_obs_full_model = self.n_obs
                # Construct C_obs_static_for_kf, H_obs_static_for_kf, I_obs_static_for_kf based on Omega_sol_inner
                # This part is fine as it uses JAX ops or static values.
                if static_n_obs_actual == n_obs_full_model:
                    C_obs_static_for_kf_val = Omega_sol_inner
                    H_obs_static_for_kf_val = H_obs
                    I_obs_static_for_kf_val = jnp.eye(n_obs_full_model, dtype=desired_dtype)
                elif static_n_obs_actual > 0:
                    C_obs_static_for_kf_val = Omega_sol_inner.take(static_valid_obs_idx, axis=0)
                    H_obs_temp = H_obs.take(static_valid_obs_idx, axis=0)
                    H_obs_static_for_kf_val = H_obs_temp.take(static_valid_obs_idx, axis=1)
                    I_obs_static_for_kf_val = jnp.eye(static_n_obs_actual, dtype=desired_dtype)
                else: # static_n_obs_actual == 0
                    C_obs_static_for_kf_val = jnp.empty((0, Omega_sol_inner.shape[1] if Omega_sol_inner is not None and Omega_sol_inner.ndim > 1 else self.n_aug ), dtype=desired_dtype)
                    H_obs_static_for_kf_val = jnp.empty((0, 0), dtype=desired_dtype)
                    I_obs_static_for_kf_val = jnp.empty((0, 0), dtype=desired_dtype)
                
                try:
                    # We need to ensure P_aug_inner, R_aug_inner, Omega_sol_inner are valid *before* KF instantiation
                    # This is what solution_inner_valid_tracer is for.
                    kf = KalmanFilter(T=P_aug_inner, R=R_aug_inner, C=Omega_sol_inner, H=H_obs, init_x=init_x_mean, init_P=init_P_cov)
                    raw_log_prob = kf.log_likelihood(ys, static_valid_obs_idx, static_n_obs_actual,
                                                     C_obs_static_for_kf_val, H_obs_static_for_kf_val, I_obs_static_for_kf_val)
                    raw_log_prob_scalar = jnp.asarray(raw_log_prob, dtype=desired_dtype).reshape(())
                    safe_ll = jnp.where(jnp.isfinite(raw_log_prob_scalar), raw_log_prob_scalar, jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype))
                    if self.verbose: jax.debug.print("  [LogLik _calc_ll -> ll_if_inner_solve_valid] KF raw: {x}, safe: {y}", x=raw_log_prob_scalar, y=safe_ll)
                    return safe_ll
                except Exception as e_kf_inner: # This Python try-except is problematic in JIT
                    if self.verbose: print(f"  [LogLik _calc_ll -> ll_if_inner_solve_valid] Exception in KF: {type(e_kf_inner).__name__}: {e_kf_inner}") # Won't run if JITted
                    return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)

            def ll_if_inner_solve_invalid():
                if self.verbose: jax.debug.print("  [LogLik _calc_ll -> ll_if_inner_solve_invalid] Inner solve was invalid.")
                return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)

            return jax.lax.cond(
                solution_inner_valid_tracer,
                ll_if_inner_solve_valid,
                ll_if_inner_solve_invalid
            )

        def _return_invalid_likelihood_branch(pd_operand_branch): # Renamed operand
            if self.verbose:
                keys_to_print_invalid_branch = ['sigma_SHK_RS', 'rho_L_GDP_GAP']
                print_subset_invalid = {k: pd_operand_branch.get(k) for k in keys_to_print_invalid_branch if k in pd_operand_branch}
                if print_subset_invalid:
                    jax.debug.print("  [LogLik _return_invalid] Outer solution was invalid. Params (subset): {subset}", subset=print_subset_invalid)
            return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)

        # Perform the initial solve for validity check
        # This call to self.solve is outside any lax.cond in this function itself.
        try:
            solution_outer_check = self.solve(param_dict) # param_dict can have JVP tracers here
            is_valid_outer = solution_outer_check.get("solution_valid", jnp.array(False, dtype=jnp.bool_))
            is_valid_outer = jnp.asarray(is_valid_outer, dtype=jnp.bool_) 

            if self.verbose:
                jax.debug.print("  [LogLik ValidityCheck] Outer solve for validity: is_valid_outer_raw={raw}, is_valid_for_cond={cond}",
                                raw=solution_outer_check.get('solution_valid'), cond=is_valid_outer)
        except Exception as e_solve_outer: # Python try-except
            if self.verbose: print(f"  [LogLik ValidityCheck] Exception during outer validity solve: {type(e_solve_outer).__name__}: {e_solve_outer}")
            is_valid_outer = jnp.array(False, dtype=jnp.bool_)

        # Main conditional execution for likelihood calculation
        log_prob_final_val = jax.lax.cond(
            pred=is_valid_outer, 
            true_fun=_calculate_likelihood_branch, # Calls solve again internally
            false_fun=_return_invalid_likelihood_branch,
            operand=param_dict 
        )
        
        final_log_prob_clean = jnp.where(jnp.isfinite(log_prob_final_val), log_prob_final_val, jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype))
        if self.verbose:
            jax.debug.print("  [LogLik End] Final log_prob returned: {lp}. (is_valid_outer for cond was: {iv})", lp=final_log_prob_clean, iv=is_valid_outer)
        return final_log_prob_clean

    
# --- Kalman Integration Test (unchanged from prompt) ---
def test_kalman_integration(model, param_values, obs_data, H_obs, init_x, init_P):
    """
    Simple diagnostic function to test the integration between the model and Kalman filter.
    Performs each step separately to identify potential issues.
    """
    print("\n=== RUNNING KALMAN INTEGRATION TEST ===")
    
    print("\nStep 1: Model Solution")
    try:
        solution = model.solve(param_values)
        # Use jnp.all() for JAX boolean scalar or array with one element
        if not jnp.all(solution["solution_valid"]):
            print("✗ Model solution is INVALID")
            print(f"  Debug - solution_valid flag from model.solve(): {solution['solution_valid']}")
            return False, {"error": "Invalid model solution"}
        
        P_aug = solution["P_aug"]
        R_aug = solution["R_aug"]
        Omega = solution["Omega"] # This is C_full for the KF
        n_aug = solution["n_aug"]
        n_obs = solution["n_obs"] # This is n_obs_full for the KF
        
        print(f"✓ Model solution valid")
        print(f"  - State dimension (n_aug): {n_aug}")
        print(f"  - Observation dimension (n_obs_full): {n_obs}") # Clarify this is full
        print(f"  - P_aug shape: {P_aug.shape}")
        print(f"  - R_aug shape: {R_aug.shape}")
        print(f"  - Omega (C_full) shape: {Omega.shape}")

        if P_aug is None or R_aug is None or Omega is None or \
           not jnp.all(jnp.isfinite(P_aug)) or \
           not jnp.all(jnp.isfinite(R_aug)) or \
           not jnp.all(jnp.isfinite(Omega)):
            print("✗ Model solution matrices are None or contain NaN/Inf values")
            return False, {"error": "None or NaN/Inf in model solution matrices"}

    except Exception as e:
        print(f"✗ Error solving model: {e}")
        import traceback; traceback.print_exc()
        return False, {"error": f"Model solution error: {str(e)}"}
    
    print("\nStep 2: Dimension Checks")
    try:
        expected_R_shape_cols = R_aug.shape[1] if R_aug.ndim == 2 and R_aug.size > 0 else 0
        
        expected_shapes = {
            "P_aug": (n_aug, n_aug), "R_aug": (n_aug, expected_R_shape_cols),
            "Omega": (n_obs, n_aug), "H_obs": (n_obs, n_obs), # H_obs is H_full
            "init_x": (n_aug,), "init_P": (n_aug, n_aug),
            "obs_data": (obs_data.shape[0], n_obs)
        }
        actual_shapes = {
            "P_aug": P_aug.shape, "R_aug": R_aug.shape, "Omega": Omega.shape,
            "H_obs": H_obs.shape, "init_x": init_x.shape, "init_P": init_P.shape,
            "obs_data": obs_data.shape
        }
        mismatches = [f"{name}: expected {expected}, got {actual_shapes[name]}" for name, expected in expected_shapes.items() if actual_shapes[name] != expected]
        if mismatches:
            print("✗ Dimension mismatches found:"); [print(f"  - {m}") for m in mismatches]
            return False, {"error": "Dimension mismatches", "details": mismatches}
        print("✓ All dimensions match expected values")
    except Exception as e:
        print(f"✗ Error checking dimensions: {e}"); return False, {"error": f"Dimension check error: {str(e)}"}
    
    print("\nStep 3: Check for NaN/Inf Values in KF Inputs (H_obs, init_x, init_P)")
    kf_direct_inputs = { "H_obs": H_obs, "init_x": init_x, "init_P": init_P }
    if any(not jnp.all(jnp.isfinite(matrix)) for matrix in kf_direct_inputs.values()):
        for name, matrix in kf_direct_inputs.items():
            if not jnp.all(jnp.isfinite(matrix)): print(f"✗ Matrix {name} for KF contains NaN/Inf")
        return False, {"error": "NaN/Inf values in direct KF input matrices"}
    print("✓ All direct KF input matrices contain finite values")
    
    print("\nStep 4: Direct Kalman Filter Test")
    log_lik_direct_kf = None
    try:
        # KF is instantiated with C_full (Omega from solution) and H_full (H_obs from input)
        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega, H=H_obs, init_x=init_x, init_P=init_P)
        print("✓ KalmanFilter instantiated successfully")

        # --- PREPARE STATIC NaN INFO for direct KF calls ---
        # obs_data is concrete here
        first_obs_slice_test = onp.asarray(obs_data[0])
        _valid_obs_idx_py_test = onp.where(~onp.isnan(first_obs_slice_test))[0]
        static_valid_obs_idx_test_jnp = jnp.array(_valid_obs_idx_py_test, dtype=jnp.int32)
        static_n_obs_actual_test_py = len(_valid_obs_idx_py_test) # Python int

        # Omega is C_full from solution, H_obs is H_full from input to test_kalman_integration
        # n_obs is the full number of observables (rows in Omega and H_obs)
        if static_n_obs_actual_test_py == n_obs: # No NaNs or all NaNs if n_obs_actual is 0
            C_obs_static_test = Omega
            H_obs_static_test = H_obs
            I_obs_static_test = jnp.eye(n_obs, dtype=_DEFAULT_DTYPE)
        elif static_n_obs_actual_test_py > 0: # Some NaNs
            C_obs_static_test = Omega[static_valid_obs_idx_test_jnp, :]
            H_obs_static_test = H_obs[jnp.ix_(static_valid_obs_idx_test_jnp, static_valid_obs_idx_test_jnp)]
            I_obs_static_test = jnp.eye(static_n_obs_actual_test_py, dtype=_DEFAULT_DTYPE)
        else: # All observations are NaN
            C_obs_static_test = jnp.empty((0, n_aug), dtype=Omega.dtype) # n_aug is kf.n_state
            H_obs_static_test = jnp.empty((0, 0), dtype=H_obs.dtype)
            I_obs_static_test = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
        # --- END PREPARE STATIC NaN INFO ---

        print("  - Testing filter operation...")
        # Call kf.filter with the new signature
        filter_results = kf.filter(obs_data,
                                   static_valid_obs_idx_test_jnp,
                                   static_n_obs_actual_test_py,
                                   C_obs_static_test,
                                   H_obs_static_test,
                                   I_obs_static_test)
        print("  ✓ filter() completed successfully")
            
        keys_to_check = ['x_pred', 'P_pred', 'x_filt', 'P_filt', 'log_likelihood_contributions']
        for key_to_check in keys_to_check:
            if key_to_check not in filter_results: print(f"  ✗ Missing key: {key_to_check}"); return False, {"error": f"Missing key {key_to_check}"}
            if not jnp.all(jnp.isfinite(filter_results[key_to_check])): print(f"  ✗ Non-finite in: {key_to_check}"); return False, {"error": f"Non-finite in {key_to_check}"}
        print("  ✓ Filter results look valid")
            
        print("  - Testing log_likelihood calculation (direct kf call)...")
        # Call kf.log_likelihood with the new signature
        log_lik_direct_kf = kf.log_likelihood(obs_data,
                                              static_valid_obs_idx_test_jnp,
                                              static_n_obs_actual_test_py,
                                              C_obs_static_test,
                                              H_obs_static_test,
                                              I_obs_static_test)
        print(f"  ✓ kf.log_likelihood() completed: {log_lik_direct_kf}")
        if not jnp.isfinite(log_lik_direct_kf):
            print("  ✗ Log-likelihood is not finite from direct kf.log_likelihood()"); return False, {"error": "Non-finite LL from kf", "value": float(log_lik_direct_kf)}
                
        print("  - Testing model.log_likelihood()...")
        # For model.log_likelihood, it internally computes C_obs_static etc.
        # We just need to pass it the basic static_valid_obs_idx and static_n_obs_actual
        # that were derived from obs_data for the kf direct call.
        model_ll = model.log_likelihood(param_values, 
                                        obs_data, 
                                        H_obs, 
                                        init_x, 
                                        init_P,
                                        static_valid_obs_idx_test_jnp,  # From earlier in test_kalman
                                        static_n_obs_actual_test_py) # From earlier in test_kalman
        print(f"  ✓ model.log_likelihood() completed: {model_ll}")
        
        if not jnp.isfinite(model_ll):
            print("  ✗ Log-likelihood is not finite from model.log_likelihood()"); return False, {"error": "Non-finite LL from model", "value": float(model_ll)}

        ll_diff = jnp.abs(log_lik_direct_kf - model_ll)
        if ll_diff > 1e-6: print(f"  ⚠ Warning: Difference between direct KF and model log-likelihood: {ll_diff}")
        else: print("  ✓ Direct KF and model log-likelihood match")
                    
    except Exception as e_kf_test: # Renamed to avoid conflict
        print(f"✗ Error during Kalman Filter test: {e_kf_test}")
        import traceback; traceback.print_exc()
        return False, {"error": f"KalmanFilter test error: {str(e_kf_test)}"}
    
    print("\n=== KALMAN INTEGRATION TEST PASSED ===")
    return True, {"log_likelihood": float(log_lik_direct_kf) if log_lik_direct_kf is not None else None}


# --- Numpyro Model Function (unchanged from prompt, uses model_instance.log_likelihood) ---
# In run_estimation.py

def numpyro_model_fixed(
    model_instance: DynareModelWithLambdified,
    user_priors: List[Dict[str, Any]],
    fixed_param_values: Dict[str, float],
    ys: Optional[jax.Array], # This will still be the full ys data
    H_obs: Optional[jax.Array],
    init_x_mean: Optional[jax.Array],
    init_P_cov: Optional[jax.Array],
    # NEW: Pass precomputed static NaN info
    static_valid_obs_idx_for_kf: jax.Array,
    static_n_obs_actual_for_kf: int
):
    if not NUMPYRO_AVAILABLE:
        raise RuntimeError("Numpyro is required for this model function.")

    # Static NaN info is NOW PASSED IN, no need to compute from ys here.

    params_for_likelihood = {}
    estimated_param_names = {p_spec["name"] for p_spec in user_priors}

    # ... (Sampling parameters as before - this part is fine) ...
    for prior_spec in user_priors:
        name = prior_spec["name"]; dist_name = prior_spec.get("prior", "").lower(); args = prior_spec.get("args", {})
        dist_args_processed = {k: jnp.asarray(v, dtype=_DEFAULT_DTYPE) for k, v in args.items()}
        sampled_value = None
        try:
            if dist_name == "normal": sampled_value = numpyro.sample(name, dist.Normal(dist_args_processed.get("loc", 0.0), jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)))
            elif dist_name == "beta": sampled_value = numpyro.sample(name, dist.Beta(jnp.maximum(dist_args_processed.get("concentration1", 1.0), 1e-7), jnp.maximum(dist_args_processed.get("concentration2", 1.0), 1e-7)))
            elif dist_name == "gamma": sampled_value = numpyro.sample(name, dist.Gamma(jnp.maximum(dist_args_processed.get("concentration", 1.0), 1e-7), rate=jnp.maximum(dist_args_processed.get("rate", 1.0), 1e-7)))
            elif dist_name == "invgamma":
                conc = jnp.maximum(dist_args_processed.get("concentration", 1.0), 1e-7)
                user_scale = jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)
                rate_param = 1.0 / user_scale # InverseGamma in Numpyro takes rate = 1/scale
                sampled_value = numpyro.sample(name, dist.InverseGamma(conc, rate=rate_param))
            elif dist_name == "uniform": sampled_value = numpyro.sample(name, dist.Uniform(dist_args_processed.get("low", 0.0), dist_args_processed.get("high", 1.0)))
            elif dist_name == "halfnormal": sampled_value = numpyro.sample(name, dist.HalfNormal(jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)))
            else: raise NotImplementedError(f"Prior distribution '{dist_name}' not implemented for '{name}'.")
            params_for_likelihood[name] = sampled_value
        except KeyError as e: raise ValueError(f"Missing arg for prior '{dist_name}' for '{name}': {e}")
        except Exception as e_dist: raise RuntimeError(f"Error sampling '{name}' with prior '{dist_name}': {e_dist}")

    for name, value in fixed_param_values.items():
        if name not in estimated_param_names:
            params_for_likelihood[name] = jnp.asarray(value, dtype=_DEFAULT_DTYPE)

    missing_keys = set(model_instance.all_param_names) - set(params_for_likelihood.keys())
    if missing_keys: raise RuntimeError(f"Internal Error: Parameters missing: {missing_keys}.")
    extra_keys = set(params_for_likelihood.keys()) - set(model_instance.all_param_names)
    if extra_keys: raise RuntimeError(f"Internal Error: Extra parameters found: {extra_keys}")

    if ys is not None: # Only proceed if actual data is provided
        if H_obs is None or init_x_mean is None or init_P_cov is None:
            raise ValueError("H_obs, init_x_mean, init_P_cov are required when ys is provided.")
        
        log_prob = model_instance.log_likelihood(
            params_for_likelihood,
            ys, H_obs, init_x_mean, init_P_cov,
            # Pass the precomputed static NaN info received by this function:
            static_valid_obs_idx_for_kf,
            static_n_obs_actual_for_kf
        )
        
        safe_log_prob = jax.lax.cond(jnp.isfinite(log_prob), lambda x: x, lambda _: jnp.array(-1e10, dtype=_DEFAULT_DTYPE), log_prob)
        numpyro.factor("log_likelihood", safe_log_prob)
    # If ys is None (e.g. prior predictive), log_likelihood is not computed,
    # and numpyro.factor is not called, which is correct.


# --- Main Execution Block (largely unchanged, but now uses DynareModelWithLambdified) ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn") # Ensure this file exists
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file)

    num_sim_steps = 200
    sim_seed = 123
    sim_measurement_noise_std = 0.01 # Slightly increased for better identifiability

    run_estimation_flag = True
    mcmc_seed = 456
    mcmc_chains = 1 # For faster debugging, jax.local_device_count() for parallel
    mcmc_warmup = 200 # Reduced for speed
    mcmc_samples = 300 # Reduced for speed
    mcmc_target_accept = 0.8

    print(f"\n--- [1] Initializing Dynare Model ({mod_file_path}) using Lambdified Engine ---")
    init_start_time = time.time()
    try:
        # USE THE NEW MODEL CLASS
        model = DynareModelWithLambdified(mod_file_path, verbose=True) # Set verbose=True for detailed parsing logs
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {mod_file_path}"); 
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize DynareModelWithLambdified: {e}")
        import traceback; 
        traceback.print_exc(); 
        exit()


    init_end_time = time.time()
    print(f"Model initialized ({init_end_time - init_start_time:.2f} seconds).")
    print(f"  Found {len(model.all_param_names)} parameters total: {model.all_param_names}")
    print(f"  Stat func params ({len(model.param_names_for_stat_funcs)}): {model.param_names_for_stat_funcs}")

    sim_param_values = model.default_param_assignments.copy()
    # Example overrides (ensure these are in all_param_names and defaults exist)
    sim_overrides = { 'b1': 0.7, 
                     'a1': 0.6, 
                     'g1':0.65, 
                     'rho_L_GDP_GAP': 0.85, 
                     "sigma_SHK_RS": 0.25}
    

    for k_override, v_override in sim_overrides.items():
        if k_override in sim_param_values:
            sim_param_values[k_override] = v_override
        else:
            print(f"Warning: Override parameter '{k_override}' not in model's default assignments. Ignoring override.")

    missing_sim_params = [p for p in model.all_param_names if p not in sim_param_values]
    if missing_sim_params:
        print(f"FATAL ERROR: Missing simulation parameter values for: {missing_sim_params}")
        # Attempt to fill with 1.0 as a last resort, but this indicates an issue
        for p_missing in missing_sim_params: 
            sim_param_values[p_missing] = 1.0

        print(f"Filled missing sim_params with 1.0: {missing_sim_params}")

        # exit() # Or allow to continue with warning
    print("\n--- [2] Simulation parameter set defined ---")

    print("\n--- [3] Simulating Data ---")
    sim_key_master = random.PRNGKey(sim_seed)
    sim_key_init, sim_key_path = random.split(sim_key_master)

    sim_solution = model.solve(sim_param_values)
    if not sim_solution["solution_valid"]:
         print("FATAL ERROR: Cannot solve model with simulation parameters.")
         print("Problematic parameters for solve:", sim_param_values)
         exit()
    
    sim_initial_state_config = { # From original main block
        "L_GDP_TREND": {"mean": 10.0, "std": 0.01}, 
        "G_TREND": {"mean": 2.0, "std": 0.002},
        "PI_TREND": {"mean": 2.0, "std": 0.01}, 
        "RR_TREND": {"mean": 1.0, "std": 0.1}
    }
    s0_sim = construct_initial_state(
        n_aug=sim_solution["n_aug"], 
        n_stat=model.n_stat,
        aug_state_vars=sim_solution["aug_state_vars"],
        key_init=sim_key_init,
        initial_state_config=sim_initial_state_config, 
        dtype=_DEFAULT_DTYPE
    )
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
            measurement_noise_std=sim_measurement_noise_std
        )
        sim_end_time = time.time()
        print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")
        plot_simulation_with_trends_matched(
            sim_observables, 
            sim_solution["ordered_obs_vars"],
            sim_states, 
            sim_solution["aug_state_vars"],
            sim_solution["ordered_trend_state_vars"], 
            sim_solution["contemp_trend_defs"],
            title=f"Simulated Data (Meas Noise Std={sim_measurement_noise_std:.2e})"
        )
        plt.show(block=False)
    except Exception as e_sim:
        print(f"FATAL ERROR during simulation: {e_sim}"); 
        import traceback; 
        traceback.print_exc(); 
        exit()

    print("\n--- [4] Defining Priors for Estimation ---")
    user_priors = [
        {"name": "sigma_SHK_RS", "prior": "invgamma", "args": {"concentration": 3.0, "scale": 0.2}}, # True was 0.25
        {"name": "rho_L_GDP_GAP", "prior": "beta", "args": {"concentration1": 40.0, "concentration2": 15.0}}, # True was 0.85
        # {"name": "b1", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}}, # True was 0.7
    ]
    estimated_param_names_set = {p["name"] for p in user_priors}
    fixed_params = {}
    print("  Parameters treated as FIXED (using .dyn file defaults or sim_param_values):")
    for name in model.all_param_names:
        if name not in estimated_param_names_set:
            value = sim_param_values.get(name, model.default_param_assignments.get(name)) # Prioritize sim_param_values for fixed
            if value is None: raise ValueError(f"Critical: No value for fixed parameter '{name}'.")
            fixed_params[name] = value
            # print(f"    - {name} = {value:.4f}")

    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE:
        print(f"\n--- [5] Running Bayesian Estimation (Estimating {len(user_priors)} parameters) ---")
        mcmc_key = random.PRNGKey(mcmc_seed)
        H_obs_est = H_obs_sim # Use same observation noise as simulation
        init_x_mean_est = s0_sim
        init_P_cov_est = jnp.eye(sim_solution["n_aug"], dtype=_DEFAULT_DTYPE) * 0.1

        # --- COMPUTE STATIC NaN INFO FOR MCMC ---
        # This uses the concrete `sim_observables` data
        if sim_observables is not None and sim_observables.shape[0] > 0:
            first_obs_slice_mcmc = onp.asarray(sim_observables[0])
            _valid_obs_idx_py_mcmc = onp.where(~onp.isnan(first_obs_slice_mcmc))[0]
            static_valid_obs_idx_mcmc_jnp = jnp.array(_valid_obs_idx_py_mcmc, dtype=jnp.int32)
            static_n_obs_actual_mcmc_py = len(_valid_obs_idx_py_mcmc)
        else: # Should not happen if simulation ran correctly
            static_valid_obs_idx_mcmc_jnp = jnp.array(onp.arange(model.n_obs), dtype=jnp.int32) # Fallback: assume all obs
            static_n_obs_actual_mcmc_py = model.n_obs
        # --- END COMPUTE STATIC NaN INFO FOR MCMC ---


        init_values_mcmc = {}
        print("  Setting initial MCMC values from simulation parameters for estimated vars:")
        for p_spec in user_priors:
             name = p_spec["name"]
             # Get true value from sim_param_values, fallback to fixed_params (which includes defaults), then 0.5
             true_value = sim_param_values.get(name)
             if true_value is None: true_value = fixed_params.get(name, 0.5) # Should not happen if logic above is correct

             init_values_mcmc[name] = true_value 
             print(f"    - {name} = {init_values_mcmc[name]:.4f} (True value for sim: {sim_param_values.get(name, 'N/A')})")
        init_strategy = init_to_value(values=init_values_mcmc)


        # --- Optional: Run Kalman Integration Test before MCMC ---
        # test_params_for_kf_integration = sim_param_values.copy() # Use the true sim values for this test
        # print("\n--- [Pre-MCMC] Running Kalman Integration Test with simulation parameters ---")
        # test_passed, test_info = test_kalman_integration(
        #     model, test_params_for_kf_integration, sim_observables, H_obs_est, init_x_mean_est, init_P_cov_est
        # )
        # if not test_passed:
        #     print(f"FATAL: Kalman integration test FAILED. Error: {test_info.get('error')}. Details: {test_info.get('details')}")
        #     # exit() # Exit if the test fails with true parameters, as MCMC will likely fail.
        # else:
        #     print(f"Kalman integration test PASSED with simulation parameters. LogLik: {test_info.get('log_likelihood')}")


        kernel = NUTS(numpyro_model_fixed, init_strategy=init_strategy, target_accept_prob=mcmc_target_accept)
        mcmc = MCMC(kernel, num_warmup=mcmc_warmup, num_samples=mcmc_samples, num_chains=mcmc_chains, progress_bar=True, chain_method='sequential' if mcmc_chains==1 else 'parallel')
        
        print(f"Starting MCMC ({mcmc_chains} chain(s), {mcmc_warmup} warmup, {mcmc_samples} samples)...")
        est_start_time = time.time()
        try:
            mcmc.run(
                mcmc_key,
                model,               # Arg 0 -> model_instance
                user_priors,         # Arg 1 -> user_priors
                fixed_params,        # Arg 2 -> fixed_param_values
                sim_observables,     # Arg 3 -> ys
                H_obs_est,           # Arg 4 -> H_obs
                init_x_mean_est,     # Arg 5 -> init_x_mean
                init_P_cov_est,      # Arg 6 -> init_P_cov
                # NEW ARGS for static NaN info:
                static_valid_obs_idx_mcmc_jnp, # Arg 7
                static_n_obs_actual_mcmc_py    # Arg 8
            )
            est_end_time = time.time()
            print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")
            print("\n--- [6] Estimation Summary (Estimated Parameters) ---")
            mcmc.print_summary()
            posterior_samples = mcmc.get_samples()
            try:
                import arviz as az
                print(" Generating trace plots...")
                az_data = az.from_numpyro(mcmc)
                az.plot_trace(az_data); plt.suptitle("Trace Plots (Estimated Parameters)", y=1.02); plt.tight_layout(); plt.show(block=False)
            except ImportError: print(" Install arviz (`pip install arviz`) to see trace plots.")
            except Exception as e_trace: print(f" Could not generate trace plots: {e_trace}")
        except Exception as e_est:
            print(f"\n--- Estimation FAILED ---"); print(f"Error: {e_est}"); import traceback; traceback.print_exc()
    else: print("\n--- [5] Skipping Estimation ---")

    print(f"\n--- Script finished ---")
    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE : 
       print("Close plot windows to exit."); plt.show()

# --- END OF FILE run_estimation.py (Modified) ---