# --- START OF FILE run_estimation.py (Cleaned) ---
import os
import time
import jax
import jax.numpy as jnp
import numpy as onp # For plotting or data loading if needed
import matplotlib.pyplot as plt
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import traceback # Keep for top-level error reporting

# --- JAX/Dynamax/Numpyro Setup ---
# Keep initial setup prints
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update("jax_platforms", "cpu")
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

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

try:
    from Kalman_filter_jax_prev import KalmanFilter
    KALMAN_FILTER_JAX_AVAILABLE = True
    print("Custom KalmanFilter imported successfully.")
except ImportError:
    KALMAN_FILTER_JAX_AVAILABLE = False
    print("Warning: Kalman_filter_jax.py not found. Likelihood calculation will fail.")

from dynare_parser_engine import (
    # Keep necessary imports
    parse_lambdify_and_order_model,
    build_trend_matrices as build_trend_matrices_lambdified,
    build_observation_matrix as build_observation_matrix_lambdified,
    solve_quadratic_matrix_equation_jax,
    compute_Q_jax,
    construct_initial_state,
    simulate_ssm_data,
    plot_simulation_with_trends_matched,
    plot_irfs
)

# --- Dynare Model Class (Cleaned solve/log_likelihood) ---

class DynareModelWithLambdified:
    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose # Verbosity now controlled solely by this flag
        self._parsed = False
        self.func_A_stat = None; self.func_B_stat = None; self.func_C_stat = None; self.func_D_stat = None
        self.func_P_trends = None; self.func_Q_trends = None; self.func_Omega = None
        self.param_names_for_stat_funcs = []; self.param_names_for_trend_funcs = []; self.param_names_for_obs_funcs = []
        self._parse_model()

    def _parse_model(self):
        if self._parsed: return
        if self.verbose: print("--- Parsing Model Structure & Lambdifying Matrices ---") # Keep if verbose
        # ... (rest of _parse_model remains the same, relying on the verbose flag) ...
        with open(self.mod_file_path, 'r') as f: model_def = f.read()
        from dynare_parser_engine import ( # Re-import locally if needed, or ensure they are accessible
             extract_declarations, extract_model_equations, extract_trend_declarations,
             extract_trend_equations, extract_observation_declarations, extract_measurement_equations,
             extract_stationary_shock_stderrs, extract_trend_shock_stderrs
        )
        try:
            (self.func_A_stat, self.func_B_stat, self.func_C_stat, self.func_D_stat,
             self.ordered_stat_vars, self.stat_shocks, self.param_names_for_stat_funcs,
             self.param_assignments_stat, _, self.initial_info_stat
             ) = parse_lambdify_and_order_model(model_def, verbose=self.verbose) # Pass verbosity down

            self.trend_vars, self.trend_shocks = extract_trend_declarations(model_def)
            self.trend_equations = extract_trend_equations(model_def)
            self.trend_stderr_params = extract_trend_shock_stderrs(model_def)
            self.obs_vars = extract_observation_declarations(model_def)
            self.measurement_equations = extract_measurement_equations(model_def)

            current_param_names = list(self.param_names_for_stat_funcs)
            current_param_assignments = self.param_assignments_stat.copy()
            for p_name, p_val in self.trend_stderr_params.items():
                if p_name not in current_param_assignments: current_param_assignments[p_name] = p_val
                if p_name not in current_param_names: current_param_names.append(p_name)
            inferred_trend_sigmas = [f"sigma_{shk}" for shk in self.trend_shocks]
            for p_sigma_trend in inferred_trend_sigmas:
                if p_sigma_trend not in current_param_assignments:
                    current_param_assignments[p_sigma_trend] = 1.0
                    if self.verbose: print(f"Defaulting inferred trend sigma '{p_sigma_trend}' to 1.0")
                if p_sigma_trend not in current_param_names: current_param_names.append(p_sigma_trend)

            self.all_param_names = list(dict.fromkeys(current_param_names))
            self.default_param_assignments = current_param_assignments

            (self.func_P_trends, self.func_Q_trends, self.ordered_trend_state_vars,
             self.contemp_trend_defs) = build_trend_matrices_lambdified(
                self.trend_equations, self.trend_vars, self.trend_shocks,
                self.all_param_names, self.default_param_assignments, verbose=self.verbose # Pass verbosity
            )
            self.param_names_for_trend_funcs = list(self.all_param_names)

            (self.func_Omega, self.ordered_obs_vars) = build_observation_matrix_lambdified(
                self.measurement_equations, self.obs_vars, self.ordered_stat_vars,
                self.ordered_trend_state_vars, self.contemp_trend_defs,
                self.all_param_names, self.default_param_assignments, verbose=self.verbose # Pass verbosity
            )
            self.param_names_for_obs_funcs = list(self.all_param_names)

            self.n_stat = len(self.ordered_stat_vars); self.n_s_shock = len(self.stat_shocks)
            self.n_t_shock = len(self.trend_shocks); self.n_obs = len(self.ordered_obs_vars)
            self.n_trend = len(self.ordered_trend_state_vars); self.n_aug = self.n_stat + self.n_trend
            self.n_aug_shock = self.n_s_shock + self.n_t_shock
            self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars
            self.aug_shocks = self.stat_shocks + self.trend_shocks

            self._parsed = True
            if self.verbose: print("--- Model Structure Parsing & Lambdification Complete ---")

        except Exception as e:
            print(f"Error during model parsing: {e}")
            traceback.print_exc()
            raise


    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        # REMOVED jax.debug.print statements from here
        if not self._parsed: self._parse_model()

        results = {"solution_valid": jnp.array(False, dtype=jnp.bool_)}
        try:
            stat_param_values_ordered = [param_dict.get(p, self.default_param_assignments.get(p, 1.0)) for p in self.param_names_for_stat_funcs]
            A_num_stat = jnp.asarray(self.func_A_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            B_num_stat = jnp.asarray(self.func_B_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            C_num_stat = jnp.asarray(self.func_C_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            D_num_stat = jnp.asarray(self.func_D_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)

            P_sol_stat, _, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=10 # Keep increased max_iter
            )
            valid_stat_solve = converged_stat & jnp.all(jnp.isfinite(P_sol_stat))
            
            Q_sol_stat_if_valid = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
            Q_sol_stat_if_invalid = jnp.full_like(D_num_stat, jnp.nan)
            Q_sol_stat = jnp.where(valid_stat_solve, Q_sol_stat_if_valid, Q_sol_stat_if_invalid)
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat))

            all_param_values_ordered = [param_dict.get(p, self.default_param_assignments.get(p, 1.0)) for p in self.all_param_names]
            P_num_trend = jnp.asarray(self.func_P_trends(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            Q_num_trend = jnp.asarray(self.func_Q_trends(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            Omega_num = jnp.asarray(self.func_Omega(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)

            shock_std_devs = { f"sigma_{shk}": jnp.maximum(jnp.abs(param_dict.get(f"sigma_{shk}", self.default_param_assignments.get(f"sigma_{shk}", 1.0))), 1e-9) for shk in self.aug_shocks }
            stat_std_devs_arr = jnp.array([shock_std_devs[f"sigma_{shk}"] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[f"sigma_{shk}"] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE)
            
            R_sol_stat_if_q_valid = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if self.n_s_shock > 0 and Q_sol_stat.shape[1] == len(stat_std_devs_arr) else jnp.zeros((self.n_stat, 0), dtype=_DEFAULT_DTYPE)
            R_sol_stat = jnp.where(valid_q_compute, R_sol_stat_if_q_valid, jnp.full((self.n_stat, self.n_s_shock if self.n_s_shock > 0 else 0), jnp.nan, dtype=_DEFAULT_DTYPE))
            R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if self.n_t_shock > 0 and Q_num_trend.shape[1] == len(trend_std_devs_arr) else jnp.zeros((self.n_trend, 0), dtype=_DEFAULT_DTYPE)

            P_aug_if_valid = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
            P_aug = jnp.where(valid_stat_solve, P_aug_if_valid, jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE))

            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=P_aug.dtype)
            if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock): R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
            if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock): R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)

            all_finite_check = jnp.all(jnp.isfinite(P_aug)) & jnp.all(jnp.isfinite(R_aug)) & jnp.all(jnp.isfinite(Omega_num))
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite_check
            
            results["P_aug"] = P_aug; results["R_aug"] = R_aug; results["Omega"] = Omega_num
            results["solution_valid"] = jnp.asarray(solution_valid_final, dtype=jnp.bool_)
            results["ordered_trend_state_vars"] = self.ordered_trend_state_vars; results["contemp_trend_defs"] = self.contemp_trend_defs
            results["ordered_obs_vars"] = self.ordered_obs_vars; results["aug_state_vars"] = self.aug_state_vars
            results["aug_shocks"] = self.aug_shocks; results["n_aug"] = self.n_aug; results["n_aug_shock"] = self.n_aug_shock
            results["n_obs"] = self.n_obs

        except Exception as e:
            # Keep this simple error reporting, but remove verbose flag check for it
            print(f"[solve()] Exception during model solve: {type(e).__name__}: {e}") 
            results["solution_valid"] = jnp.array(False, dtype=jnp.bool_) 
            results["P_aug"], results["R_aug"], results["Omega"] = None, None, None
            # Populate other results with safe defaults
            results["ordered_trend_state_vars"] = getattr(self, 'ordered_trend_state_vars', [])
            results["contemp_trend_defs"] = getattr(self, 'contemp_trend_defs', {})
            results["ordered_obs_vars"] = getattr(self, 'ordered_obs_vars', [])
            results["aug_state_vars"] = getattr(self, 'aug_state_vars', [])
            results["aug_shocks"] = getattr(self, 'aug_shocks', [])
            results["n_aug"] = getattr(self, 'n_aug', -1)
            results["n_aug_shock"] = getattr(self, 'n_aug_shock', -1)
            results["n_obs"] = getattr(self, 'n_obs', -1)

        return results


    def log_likelihood(self,
                    param_dict: Dict[str, float], 
                    ys: jax.Array, H_obs: jax.Array,
                    init_x_mean: jax.Array, init_P_cov: jax.Array,
                    static_valid_obs_idx: jax.Array, static_n_obs_actual: int
                    ) -> float:
        # REMOVED jax.debug.print statements from here
        if not KALMAN_FILTER_JAX_AVAILABLE: raise RuntimeError("Custom KalmanFilter class is required.")
        LARGE_NEG_VALUE = -1e10; desired_dtype = _DEFAULT_DTYPE

        def _calculate_likelihood_branch(pd_operand_branch):
            solution_inner = self.solve(pd_operand_branch) # Solve is now cleaner internally
            solution_inner_valid_tracer = solution_inner.get("solution_valid", jnp.array(False, dtype=jnp.bool_))

            def ll_if_inner_solve_valid():
                P_aug_inner = solution_inner["P_aug"]
                R_aug_inner = solution_inner["R_aug"]
                Omega_sol_inner = solution_inner["Omega"]
                n_obs_full_model = self.n_obs

                if static_n_obs_actual == n_obs_full_model:
                    C_obs_static_for_kf_val = Omega_sol_inner
                    H_obs_static_for_kf_val = H_obs
                    I_obs_static_for_kf_val = jnp.eye(n_obs_full_model, dtype=desired_dtype)
                elif static_n_obs_actual > 0:
                    C_obs_static_for_kf_val = Omega_sol_inner.take(static_valid_obs_idx, axis=0)
                    H_obs_temp = H_obs.take(static_valid_obs_idx, axis=0)
                    H_obs_static_for_kf_val = H_obs_temp.take(static_valid_obs_idx, axis=1)
                    I_obs_static_for_kf_val = jnp.eye(static_n_obs_actual, dtype=desired_dtype)
                else:
                    n_aug_shape = Omega_sol_inner.shape[1] if Omega_sol_inner is not None and Omega_sol_inner.ndim > 1 else self.n_aug
                    C_obs_static_for_kf_val = jnp.empty((0, n_aug_shape), dtype=desired_dtype)
                    H_obs_static_for_kf_val = jnp.empty((0, 0), dtype=desired_dtype)
                    I_obs_static_for_kf_val = jnp.empty((0, 0), dtype=desired_dtype)
                
                try: # Keep try-except around KF for safety, though it should be more robust now
                    kf = KalmanFilter(T=P_aug_inner, R=R_aug_inner, C=Omega_sol_inner, H=H_obs, init_x=init_x_mean, init_P=init_P_cov)
                    raw_log_prob = kf.log_likelihood(ys, static_valid_obs_idx, static_n_obs_actual,
                                                     C_obs_static_for_kf_val, H_obs_static_for_kf_val, I_obs_static_for_kf_val)
                    raw_log_prob_scalar = jnp.asarray(raw_log_prob, dtype=desired_dtype).reshape(())
                    safe_ll = jnp.where(jnp.isfinite(raw_log_prob_scalar), raw_log_prob_scalar, jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype))
                    return safe_ll
                except Exception: 
                    return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)

            def ll_if_inner_solve_invalid():
                return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)

            return jax.lax.cond(
                solution_inner_valid_tracer,
                ll_if_inner_solve_valid,
                ll_if_inner_solve_invalid
            )

        def _return_invalid_likelihood_branch(pd_operand_branch): 
            return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)

        try:
            solution_outer_check = self.solve(param_dict)
            is_valid_outer = solution_outer_check.get("solution_valid", jnp.array(False, dtype=jnp.bool_))
            is_valid_outer = jnp.asarray(is_valid_outer, dtype=jnp.bool_)
        except Exception: # Catch errors during the initial solve check
            is_valid_outer = jnp.array(False, dtype=jnp.bool_)

        log_prob_final_val = jax.lax.cond(
            pred=is_valid_outer, 
            true_fun=_calculate_likelihood_branch,
            false_fun=_return_invalid_likelihood_branch,
            operand=param_dict 
        )
        final_log_prob_clean = jnp.where(jnp.isfinite(log_prob_final_val), log_prob_final_val, jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype))
        return final_log_prob_clean

# --- START OF numpyro_model_fixed modification ---

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

    params_for_likelihood = {}
    estimated_param_names = {p_spec["name"] for p_spec in user_priors}

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
                # Numpyro's InverseGamma uses rate = 1/scale convention
                rate_param = jnp.maximum(dist_args_processed.get("rate", 1.0), 1e-7) 
                sampled_value = numpyro.sample(name, dist.InverseGamma(conc, rate=rate_param))
            elif dist_name == "uniform": sampled_value = numpyro.sample(name, dist.Uniform(dist_args_processed.get("low", 0.0), dist_args_processed.get("high", 1.0)))
            elif dist_name == "halfnormal": sampled_value = numpyro.sample(name, dist.HalfNormal(jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)))
            # --- ADDED TruncatedNormal ---
            elif dist_name == "truncnorm":
                 loc = dist_args_processed.get("loc", 0.0)
                 scale = jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)
                 low = dist_args_processed.get("low", -jnp.inf)
                 high = dist_args_processed.get("high", jnp.inf)
                 sampled_value = numpyro.sample(name, dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high))
            # --- END ADDED ---
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

    if ys is not None: 
        if H_obs is None or init_x_mean is None or init_P_cov is None:
            raise ValueError("H_obs, init_x_mean, init_P_cov are required when ys is provided.")
        
        log_prob = model_instance.log_likelihood(
            params_for_likelihood,
            ys, H_obs, init_x_mean, init_P_cov,
            static_valid_obs_idx_for_kf,
            static_n_obs_actual_for_kf
        )
        
        safe_log_prob = jax.lax.cond(jnp.isfinite(log_prob), lambda x: x, lambda _: jnp.array(-1e10, dtype=_DEFAULT_DTYPE), log_prob)
        numpyro.factor("log_likelihood", safe_log_prob)




# --- Main Execution Block (Cleaned) ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn") 
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file)

    num_sim_steps = 200
    sim_seed = 123
    sim_measurement_noise_std = 0.01 

    run_estimation_flag = True
    mcmc_seed = 456
    mcmc_chains = 1 
    mcmc_warmup = 200 
    mcmc_samples = 300 
    mcmc_target_accept = 0.8

    print(f"\n--- [1] Initializing Dynare Model ({mod_file_path}) ---")
    init_start_time = time.time()
    try:
        # Initialize with verbose=False for cleaner output
        model = DynareModelWithLambdified(mod_file_path, verbose=False) 
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {mod_file_path}"); exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize DynareModelWithLambdified: {e}")
        traceback.print_exc(); exit()
    init_end_time = time.time()
    print(f"Model initialized ({init_end_time - init_start_time:.2f} seconds).")
    print(f"  Found {len(model.all_param_names)} total parameters.")
    # Optional: print list of all_param_names if needed for debugging priors
    # print(f"  Parameters: {model.all_param_names}")

    # --- Simulation Setup ---
    sim_param_values = model.default_param_assignments.copy()
    sim_overrides = { 'b1': 0.7, 'a1': 0.6, 'g1':0.65, 'rho_L_GDP_GAP': 0.85, "sigma_SHK_RS": 0.25}
    sim_param_values.update(sim_overrides) # Apply overrides
    # Basic check for completeness
    if not all(p in sim_param_values for p in model.all_param_names):
         print("Warning: Not all parameters have values in sim_param_values. Check defaults.")
    print("\n--- [2] Simulation parameter set defined ---")

    print("\n--- [3] Simulating Data ---")
    sim_key_master = random.PRNGKey(sim_seed)
    sim_key_init, sim_key_path = random.split(sim_key_master)
    sim_solution = model.solve(sim_param_values)
    if not sim_solution["solution_valid"]:
         print("FATAL ERROR: Cannot solve model with simulation parameters."); exit()
    
    sim_initial_state_config = { "L_GDP_TREND": {"mean": 10.0, "std": 0.01}, "G_TREND": {"mean": 2.0, "std": 0.002}, "PI_TREND": {"mean": 2.0, "std": 0.01}, "RR_TREND": {"mean": 1.0, "std": 0.1} }
    s0_sim = construct_initial_state( n_aug=sim_solution["n_aug"], n_stat=model.n_stat, aug_state_vars=sim_solution["aug_state_vars"], key_init=sim_key_init, initial_state_config=sim_initial_state_config, dtype=_DEFAULT_DTYPE )
    H_obs_sim = jnp.eye(sim_solution["n_obs"], dtype=_DEFAULT_DTYPE) * (sim_measurement_noise_std**2)
    sim_start_time = time.time()
    try:
        sim_states, sim_observables = simulate_ssm_data(
            P=sim_solution["P_aug"], R=sim_solution["R_aug"], Omega=sim_solution["Omega"],
            T=num_sim_steps, key=sim_key_path, state_init=s0_sim, measurement_noise_std=sim_measurement_noise_std )
        sim_end_time = time.time()
        print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")
        print(f"  Simulated states shape: {sim_states.shape}")
        print(f"  Simulated observables shape: {sim_observables.shape}")
        # Optional: Plot simulation results (can be commented out for speed)
        # plot_simulation_with_trends_matched( sim_observables, sim_solution["ordered_obs_vars"], sim_states, sim_solution["aug_state_vars"], sim_solution["ordered_trend_state_vars"], sim_solution["contemp_trend_defs"], title=f"Simulated Data (Meas Noise Std={sim_measurement_noise_std:.2e})" )
        # plt.show(block=False) 
    except Exception as e_sim:
        print(f"FATAL ERROR during simulation: {e_sim}"); traceback.print_exc(); exit()

    # --- Estimation Setup ---
    print("\n--- [4] Defining Priors for Estimation ---")
    user_priors = [
        # Betas based on mean/std dev comments
        {"name": "b1", "prior": "beta", "args": {"concentration1": 2.975, "concentration2": 1.275}},  # mu=0.7, std=0.2
        {"name": "b4", "prior": "beta", "args": {"concentration1": 2.975, "concentration2": 1.275}},  # mu=0.7, std=0.2
        {"name": "a1", "prior": "beta", "args": {"concentration1": 2.625, "concentration2": 2.625}},  # mu=0.5, std=0.2
        {"name": "g1", "prior": "beta", "args": {"concentration1": 2.975, "concentration2": 1.275}},  # mu=0.7, std=0.2
        {"name": "g3", "prior": "beta", "args": {"concentration1": 4.4375, "concentration2": 13.3125}}, # mu=0.25, std=0.1

        # # HalfNormal / TruncatedNormal
        {"name": "a2", "prior": "halfnormal", "args": {"scale": 0.1}}, # Based on plausible scale interpretation
        {"name": "g2", "prior": "halfnormal", "args": {"scale": 0.3}}, # Truncated Normal

        # # Rho parameters (Betas centered near values)
        {"name": "rho_L_GDP_GAP", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}}, # mu=0.75
        {"name": "rho_DLA_CPI",   "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}}, # mu=0.75
        {"name": "rho_rs",        "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}}, # mu=0.75
        {"name": "rho_rs2",       "prior": "beta", "args": {"concentration1": 1.0, "concentration2": 99.0}},  # mu=0.01

        # Sigma parameters (Inverse Gamma) - Assuming all use the same prior shape/rate
        # rate = 1 / scale_commented -> rate = 1/0.2 = 5.0
        {"name": "sigma_SHK_L_GDP_GAP", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_DLA_CPI",   "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_RS",        "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_L_GDP_TREND", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_G_TREND",   "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_PI_TREND",  "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_RR_TREND",  "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
    ]
    estimated_param_names_set = {p["name"] for p in user_priors}
    fixed_params = { name: sim_param_values.get(name, model.default_param_assignments.get(name)) for name in model.all_param_names if name not in estimated_param_names_set }
    print(f"  Estimating {len(user_priors)} parameters: {[p['name'] for p in user_priors]}")
    # Optional: Print fixed params if needed
    # print(f"  Fixed parameters ({len(fixed_params)}): {list(fixed_params.keys())}")

    # --- Estimation ---
    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE:
        print(f"\n--- [5] Running Bayesian Estimation ---")
        mcmc_key = random.PRNGKey(mcmc_seed)
        H_obs_est = H_obs_sim 
        init_x_mean_est = s0_sim
        init_P_cov_est = jnp.eye(sim_solution["n_aug"], dtype=_DEFAULT_DTYPE) * 0.1

        # --- COMPUTE STATIC NaN INFO FOR MCMC ---
        if sim_observables is not None and sim_observables.shape[0] > 0:
            first_obs_slice_mcmc = onp.asarray(sim_observables[0])
            _valid_obs_idx_py_mcmc = onp.where(~onp.isnan(first_obs_slice_mcmc))[0]
            static_valid_obs_idx_mcmc_jnp = jnp.array(_valid_obs_idx_py_mcmc, dtype=jnp.int32)
            static_n_obs_actual_mcmc_py = len(_valid_obs_idx_py_mcmc)
        else: 
            static_valid_obs_idx_mcmc_jnp = jnp.array(onp.arange(model.n_obs), dtype=jnp.int32) 
            static_n_obs_actual_mcmc_py = model.n_obs
        # --- END COMPUTE STATIC NaN INFO ---

        init_values_mcmc = { p_spec["name"]: sim_param_values.get(p_spec["name"], 0.5) for p_spec in user_priors }
        print(f"  Initial MCMC values set from simulation parameters: {init_values_mcmc}")
        init_strategy = init_to_value(values=init_values_mcmc)

        # --- Kalman Integration Test Removed ---

        kernel = NUTS(numpyro_model_fixed, init_strategy=init_strategy, target_accept_prob=mcmc_target_accept)
        mcmc = MCMC(kernel, num_warmup=mcmc_warmup, num_samples=mcmc_samples, num_chains=mcmc_chains, progress_bar=True, chain_method='sequential' if mcmc_chains==1 else 'parallel')
        
        print(f"Starting MCMC ({mcmc_chains} chain(s), {mcmc_warmup} warmup, {mcmc_samples} samples)...")
        est_start_time = time.time()
        try:
            mcmc.run(
                mcmc_key, model, user_priors, fixed_params, sim_observables, 
                H_obs_est, init_x_mean_est, init_P_cov_est,
                static_valid_obs_idx_mcmc_jnp, static_n_obs_actual_mcmc_py # Pass static info
            )
            est_end_time = time.time()
            print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")
            print("\n--- [6] Estimation Summary (Estimated Parameters) ---")
            mcmc.print_summary()
            posterior_samples = mcmc.get_samples()
            try:
                import arviz as az
                print("\nGenerating trace plots...")
                az_data = az.from_numpyro(mcmc)
                az.plot_trace(az_data); plt.suptitle("Trace Plots (Estimated Parameters)", y=1.02); plt.tight_layout(); 
                # plt.show(block=False) # Keep plots non-blocking if desired
            except ImportError: print("Install arviz (`pip install arviz`) to see trace plots.")
            except Exception as e_trace: print(f"Could not generate trace plots: {e_trace}")
        except Exception as e_est:
            print(f"\n--- Estimation FAILED ---"); print(f"Error: {e_est}"); traceback.print_exc()
    else: print("\n--- [5] Skipping Estimation ---")

    print(f"\n--- Script finished ---")
    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE : 
       print("Close any plot windows to exit."); plt.show() # Single final plt.show()
# --- END OF FILE run_estimation.py (Cleaned) ---