import os
import time
import jax
import jax.numpy as jnp
import numpy as onp  # For plotting or data loading if needed
import matplotlib.pyplot as plt
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import traceback 

# --- JAX/Dynamax/Numpyro Setup ---
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
    # Ensure this import points to your revised Kalman_filter_jax.py
    from Kalman_filter_jax_prev import KalmanFilter, simulate_state_space 
    KALMAN_FILTER_JAX_AVAILABLE = True
    print("Custom KalmanFilter imported successfully.")
except ImportError:
    KALMAN_FILTER_JAX_AVAILABLE = False
    print("Warning: Kalman_filter_jax.py not found. Likelihood calculation will fail.")
    # Define a dummy simulate_state_space if KF not available, to avoid NameError later if only simulating
    def simulate_state_space(*args, **kwargs):
        raise NotImplementedError("Kalman_filter_jax.py not found, cannot simulate.")


from dynare_parser_engine import (
    parse_lambdify_and_order_model,
    build_trend_matrices as build_trend_matrices_lambdified,
    build_observation_matrix as build_observation_matrix_lambdified,
    solve_quadratic_matrix_equation_jax,
    compute_Q_jax,
    construct_initial_state,
    # simulate_ssm_data, # This was from dynare_parser_engine, we'll use the one from Kalman_filter_jax
    plot_simulation_with_trends_matched, # Keep for initial sim plot
    plot_irfs
)

# --- Dynare Model Class (Unchanged from your provided version) ---
class DynareModelWithLambdified:
    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose
        self._parsed = False
        self.func_A_stat = None
        self.func_B_stat = None
        self.func_C_stat = None
        self.func_D_stat = None
        self.func_P_trends = None
        self.func_Q_trends = None
        self.func_Omega = None
        self.param_names_for_stat_funcs = []
        self.param_names_for_trend_funcs = []
        self.param_names_for_obs_funcs = []
        self._parse_model()

    def _parse_model(self):
        if self._parsed:
            return
        if self.verbose:
            print("--- Parsing Model Structure & Lambdifying Matrices ---")
        with open(self.mod_file_path, 'r') as f:
            model_def = f.read()
        from dynare_parser_engine import (
            extract_declarations, extract_model_equations, extract_trend_declarations,
            extract_trend_equations, extract_observation_declarations, extract_measurement_equations,
            extract_stationary_shock_stderrs, extract_trend_shock_stderrs
        )
        try:
            (self.func_A_stat, self.func_B_stat, self.func_C_stat, self.func_D_stat,
             self.ordered_stat_vars, self.stat_shocks, self.param_names_for_stat_funcs,
             self.param_assignments_stat, _, self.initial_info_stat
             ) = parse_lambdify_and_order_model(model_def, verbose=self.verbose)

            self.trend_vars, self.trend_shocks = extract_trend_declarations(model_def)
            self.trend_equations = extract_trend_equations(model_def)
            self.trend_stderr_params = extract_trend_shock_stderrs(model_def)
            self.obs_vars = extract_observation_declarations(model_def)
            self.measurement_equations = extract_measurement_equations(model_def)

            current_param_names = list(self.param_names_for_stat_funcs)
            current_param_assignments = self.param_assignments_stat.copy()
            for p_name, p_val in self.trend_stderr_params.items():
                if p_name not in current_param_assignments:
                    current_param_assignments[p_name] = p_val
                if p_name not in current_param_names:
                    current_param_names.append(p_name)
            inferred_trend_sigmas = [f"sigma_{shk}" for shk in self.trend_shocks]
            for p_sigma_trend in inferred_trend_sigmas:
                if p_sigma_trend not in current_param_assignments:
                    current_param_assignments[p_sigma_trend] = 1.0
                    if self.verbose:
                        print(f"Defaulting inferred trend sigma '{p_sigma_trend}' to 1.0")
                if p_sigma_trend not in current_param_names:
                    current_param_names.append(p_sigma_trend)

            self.all_param_names = list(dict.fromkeys(current_param_names))
            self.default_param_assignments = current_param_assignments

            (self.func_P_trends, self.func_Q_trends, self.ordered_trend_state_vars,
             self.contemp_trend_defs) = build_trend_matrices_lambdified(
                self.trend_equations, self.trend_vars, self.trend_shocks,
                self.all_param_names, self.default_param_assignments, verbose=self.verbose
            )
            self.param_names_for_trend_funcs = list(self.all_param_names)

            (self.func_Omega, self.ordered_obs_vars) = build_observation_matrix_lambdified(
                self.measurement_equations, self.obs_vars, self.ordered_stat_vars,
                self.ordered_trend_state_vars, self.contemp_trend_defs,
                self.all_param_names, self.default_param_assignments, verbose=self.verbose
            )
            self.param_names_for_obs_funcs = list(self.all_param_names)

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
            if self.verbose:
                print("--- Model Structure Parsing & Lambdification Complete ---")
        except Exception as e:
            print(f"Error during model parsing: {e}")
            traceback.print_exc()
            raise

    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        if not self._parsed:
            self._parse_model()
        results = {"solution_valid": jnp.array(False, dtype=jnp.bool_)}
        try:
            stat_param_values_ordered = [param_dict.get(p, self.default_param_assignments.get(p, 1.0)) 
                                         for p in self.param_names_for_stat_funcs]
            A_num_stat = jnp.asarray(self.func_A_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            B_num_stat = jnp.asarray(self.func_B_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            C_num_stat = jnp.asarray(self.func_C_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            D_num_stat = jnp.asarray(self.func_D_stat(*stat_param_values_ordered), dtype=_DEFAULT_DTYPE)
            
            P_sol_stat, _, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500  # Increased max_iter
            )
            valid_stat_solve = converged_stat & jnp.all(jnp.isfinite(P_sol_stat))
            
            Q_sol_stat_if_valid = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
            Q_sol_stat_if_invalid = jnp.full_like(D_num_stat, jnp.nan)
            Q_sol_stat = jnp.where(valid_stat_solve, Q_sol_stat_if_valid, Q_sol_stat_if_invalid)
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat))
            
            all_param_values_ordered = [param_dict.get(p, self.default_param_assignments.get(p, 1.0)) 
                                       for p in self.all_param_names]
            P_num_trend = jnp.asarray(self.func_P_trends(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            Q_num_trend = jnp.asarray(self.func_Q_trends(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            Omega_num = jnp.asarray(self.func_Omega(*all_param_values_ordered), dtype=_DEFAULT_DTYPE)
            
            shock_std_devs = {
                f"sigma_{shk}": jnp.maximum(
                    jnp.abs(param_dict.get(f"sigma_{shk}", self.default_param_assignments.get(f"sigma_{shk}", 1.0))),
                    1e-9
                ) for shk in self.aug_shocks
            }
            stat_std_devs_arr = jnp.array([shock_std_devs[f"sigma_{shk}"] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[f"sigma_{shk}"] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE)
            
            if self.n_s_shock > 0 and Q_sol_stat.shape[1] == len(stat_std_devs_arr):
                R_sol_stat_if_q_valid = Q_sol_stat @ jnp.diag(stat_std_devs_arr)
            else:
                R_sol_stat_if_q_valid = jnp.zeros((self.n_stat, 0), dtype=_DEFAULT_DTYPE)
            
            R_sol_stat = jnp.where(
                valid_q_compute,
                R_sol_stat_if_q_valid,
                jnp.full((self.n_stat, self.n_s_shock if self.n_s_shock > 0 else 0), jnp.nan, dtype=_DEFAULT_DTYPE)
            )
            
            if self.n_t_shock > 0 and Q_num_trend.shape[1] == len(trend_std_devs_arr):
                R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr)
            else:
                R_num_trend = jnp.zeros((self.n_trend, 0), dtype=_DEFAULT_DTYPE)
            
            P_aug_if_valid = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
            P_aug = jnp.where(
                valid_stat_solve,
                P_aug_if_valid,
                jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
            )
            
            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=P_aug.dtype)
            if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock):
                R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
            if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock):
                R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)
            
            all_finite_check = (jnp.all(jnp.isfinite(P_aug)) & 
                               jnp.all(jnp.isfinite(R_aug)) & 
                               jnp.all(jnp.isfinite(Omega_num)))
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite_check
            
            results["P_aug"] = P_aug
            results["R_aug"] = R_aug
            results["Omega"] = Omega_num
            results["solution_valid"] = jnp.asarray(solution_valid_final, dtype=jnp.bool_)
            
            results["ordered_trend_state_vars"] = self.ordered_trend_state_vars
            results["contemp_trend_defs"] = self.contemp_trend_defs
            results["ordered_obs_vars"] = self.ordered_obs_vars
            results["aug_state_vars"] = self.aug_state_vars
            results["aug_shocks"] = self.aug_shocks
            results["n_aug"] = self.n_aug
            results["n_aug_shock"] = self.n_aug_shock
            results["n_obs"] = self.n_obs
            
        except Exception as e:
            print(f"[solve()] Exception: {type(e).__name__}: {e}")
            results["solution_valid"] = jnp.array(False, dtype=jnp.bool_)
            results["P_aug"], results["R_aug"], results["Omega"] = None, None, None
            results["ordered_trend_state_vars"] = getattr(self, 'ordered_trend_state_vars', [])
            results["contemp_trend_defs"] = getattr(self, 'contemp_trend_defs', {})
            results["ordered_obs_vars"] = getattr(self, 'ordered_obs_vars', [])
            results["aug_state_vars"] = getattr(self, 'aug_state_vars', [])
            results["aug_shocks"] = getattr(self, 'aug_shocks', [])
            results["n_aug"] = getattr(self, 'n_aug', -1)
            results["n_aug_shock"] = getattr(self, 'n_aug_shock', -1)
            results["n_obs"] = getattr(self, 'n_obs', -1)
        
        return results

    def log_likelihood(self, param_dict: Dict[str, float], ys: jax.Array, H_obs: jax.Array,
                      init_x_mean: jax.Array, init_P_cov: jax.Array,
                      static_valid_obs_idx: jax.Array, static_n_obs_actual: int) -> float:
        if not KALMAN_FILTER_JAX_AVAILABLE:
            raise RuntimeError("Custom KalmanFilter class is required.")
        
        LARGE_NEG_VALUE = -1e10
        desired_dtype = _DEFAULT_DTYPE
        
        def _calc_ll_branch(pd_op_branch):
            sol_inner = self.solve(pd_op_branch)
            sol_inner_valid_tracer = sol_inner.get("solution_valid", jnp.array(False, dtype=jnp.bool_))
            
            def ll_if_sol_valid():
                P_aug_in = sol_inner["P_aug"]
                R_aug_in = sol_inner["R_aug"]
                Omega_sol_in = sol_inner["Omega"]
                n_obs_full_model = self.n_obs
                
                if static_n_obs_actual == n_obs_full_model:
                    C_obs_static_kf_val = Omega_sol_in
                    H_obs_static_kf_val = H_obs
                    I_obs_static_kf_val = jnp.eye(n_obs_full_model, dtype=desired_dtype)
                elif static_n_obs_actual > 0:
                    C_obs_static_kf_val = Omega_sol_in.take(static_valid_obs_idx, axis=0)
                    H_obs_temp = H_obs.take(static_valid_obs_idx, axis=0)
                    H_obs_static_kf_val = H_obs_temp.take(static_valid_obs_idx, axis=1)
                    I_obs_static_kf_val = jnp.eye(static_n_obs_actual, dtype=desired_dtype)
                else:
                    n_aug_shape = Omega_sol_in.shape[1] if Omega_sol_in is not None and Omega_sol_in.ndim > 1 else self.n_aug
                    C_obs_static_kf_val = jnp.empty((0, n_aug_shape), dtype=desired_dtype)
                    H_obs_static_kf_val = jnp.empty((0, 0), dtype=desired_dtype)
                    I_obs_static_kf_val = jnp.empty((0, 0), dtype=desired_dtype)
                
                try:
                    kf = KalmanFilter(
                        T=P_aug_in, R=R_aug_in, C=Omega_sol_in, H=H_obs,
                        init_x=init_x_mean, init_P=init_P_cov
                    )
                    raw_log_prob = kf.log_likelihood(
                        ys, static_valid_obs_idx, static_n_obs_actual,
                        C_obs_static_kf_val, H_obs_static_kf_val, I_obs_static_kf_val
                    )
                    raw_log_prob_scalar = jnp.asarray(raw_log_prob, dtype=desired_dtype).reshape(())
                    safe_ll = jnp.where(
                        jnp.isfinite(raw_log_prob_scalar),
                        raw_log_prob_scalar,
                        jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)
                    )
                    return safe_ll
                except Exception:
                    return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)
            
            def ll_if_sol_invalid():
                return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)
            
            return jax.lax.cond(sol_inner_valid_tracer, ll_if_sol_valid, ll_if_sol_invalid)
        
        def _return_invalid_ll_branch(pd_op_branch):
            return jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)
        
        try:
            sol_outer_check = self.solve(param_dict)
            is_valid_outer = sol_outer_check.get("solution_valid", jnp.array(False, dtype=jnp.bool_))
            is_valid_outer = jnp.asarray(is_valid_outer, dtype=jnp.bool_)
        except Exception:
            is_valid_outer = jnp.array(False, dtype=jnp.bool_)
        
        log_prob_final_val = jax.lax.cond(
            pred=is_valid_outer,
            true_fun=_calc_ll_branch,
            false_fun=_return_invalid_ll_branch,
            operand=param_dict
        )
        
        final_log_prob_clean = jnp.where(
            jnp.isfinite(log_prob_final_val),
            log_prob_final_val,
            jnp.array(LARGE_NEG_VALUE, dtype=desired_dtype)
        )
        
        return final_log_prob_clean

# --- Numpyro Model (Unchanged from your provided version) ---
def numpyro_model_fixed(model_instance: DynareModelWithLambdified,
                        user_priors: List[Dict[str, Any]],
                        fixed_param_values: Dict[str, float],
                        ys: Optional[jax.Array],
                        H_obs: Optional[jax.Array],
                        init_x_mean: Optional[jax.Array],
                        init_P_cov: Optional[jax.Array],
                        static_valid_obs_idx_for_kf: jax.Array,
                        static_n_obs_actual_for_kf: int):
    if not NUMPYRO_AVAILABLE:
        raise RuntimeError("Numpyro is required for this model function.")
    
    params_for_likelihood = {}
    estimated_param_names = {p_spec["name"] for p_spec in user_priors}
    
    for prior_spec in user_priors:
        name = prior_spec["name"]
        dist_name = prior_spec.get("prior", "").lower()
        args = prior_spec.get("args", {})
        dist_args_processed = {k: jnp.asarray(v, dtype=_DEFAULT_DTYPE) for k, v in args.items()}
        sampled_value = None
        
        try:
            if dist_name == "normal":
                sampled_value = numpyro.sample(
                    name,
                    dist.Normal(
                        dist_args_processed.get("loc", 0.0),
                        jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)
                    )
                )
            elif dist_name == "beta":
                sampled_value = numpyro.sample(
                    name,
                    dist.Beta(
                        jnp.maximum(dist_args_processed.get("concentration1", 1.0), 1e-7),
                        jnp.maximum(dist_args_processed.get("concentration2", 1.0), 1e-7)
                    )
                )
            elif dist_name == "gamma":
                sampled_value = numpyro.sample(
                    name,
                    dist.Gamma(
                        jnp.maximum(dist_args_processed.get("concentration", 1.0), 1e-7),
                        rate=jnp.maximum(dist_args_processed.get("rate", 1.0), 1e-7)
                    )
                )
            elif dist_name == "invgamma":
                conc = jnp.maximum(dist_args_processed.get("concentration", 1.0), 1e-7)
                rate_param = jnp.maximum(dist_args_processed.get("rate", 1.0), 1e-7)
                sampled_value = numpyro.sample(name, dist.InverseGamma(conc, rate=rate_param))
            elif dist_name == "uniform":
                sampled_value = numpyro.sample(
                    name,
                    dist.Uniform(
                        dist_args_processed.get("low", 0.0),
                        dist_args_processed.get("high", 1.0)
                    )
                )
            elif dist_name == "halfnormal":
                sampled_value = numpyro.sample(
                    name,
                    dist.HalfNormal(
                        jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)
                    )
                )
            elif dist_name == "truncnorm":
                loc = dist_args_processed.get("loc", 0.0)
                scale = jnp.maximum(dist_args_processed.get("scale", 1.0), 1e-7)
                low = dist_args_processed.get("low", -jnp.inf)
                high = dist_args_processed.get("high", jnp.inf)
                sampled_value = numpyro.sample(
                    name,
                    dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
                )
            else:
                raise NotImplementedError(f"Prior distribution '{dist_name}' not implemented for '{name}'.")
            
            params_for_likelihood[name] = sampled_value
        except KeyError as e:
            raise ValueError(f"Missing arg for prior '{dist_name}' for '{name}': {e}")
        except Exception as e_dist:
            raise RuntimeError(f"Error sampling '{name}' with prior '{dist_name}': {e_dist}")
    
    for name, value in fixed_param_values.items():
        if name not in estimated_param_names:
            params_for_likelihood[name] = jnp.asarray(value, dtype=_DEFAULT_DTYPE)
    
    missing_keys = set(model_instance.all_param_names) - set(params_for_likelihood.keys())
    if missing_keys:
        raise RuntimeError(f"Internal Error: Parameters missing: {missing_keys}.")
    
    extra_keys = set(params_for_likelihood.keys()) - set(model_instance.all_param_names)
    if extra_keys:
        raise RuntimeError(f"Internal Error: Extra parameters found: {extra_keys}")
    
    if ys is not None:
        if H_obs is None or init_x_mean is None or init_P_cov is None:
            raise ValueError("H_obs, init_x_mean, init_P_cov are required when ys is provided.")
        
        log_prob = model_instance.log_likelihood(
            params_for_likelihood, ys, H_obs, init_x_mean, init_P_cov,
            static_valid_obs_idx_for_kf, static_n_obs_actual_for_kf
        )
        
        safe_log_prob = jax.lax.cond(
            jnp.isfinite(log_prob),
            lambda x: x,
            lambda _: jnp.array(-1e10, dtype=_DEFAULT_DTYPE),
            log_prob
        )
        
        numpyro.factor("log_likelihood", safe_log_prob)

# --- Main Execution Block ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn") 
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file)

    # --- Script Configuration ---
    num_sim_steps = 200
    sim_seed = 123
    sim_measurement_noise_std = 0.01  # Std dev for measurement noise
    sim_true_states_known = True  # Flag to indicate if we are simulating and know true states

    run_estimation_flag = True
    mcmc_seed = 456
    mcmc_chains = 1 
    mcmc_warmup = 20  # Short for testing
    mcmc_samples = 30  # Short for testing
    mcmc_target_accept = 0.8

    # Smoothing configuration
    run_smoothing_flag = True  # Control whether to run smoothing post-estimation
    num_param_draws_to_use_smoothing = 10    # Subsample from posterior MCMC draws
    num_state_draws_per_param_smoothing = 5  # Draws for D&K smoother per param set

    print(f"\n--- [1] Initializing Dynare Model ({mod_file_path}) ---")
    init_start_time = time.time()
    try:
        model = DynareModelWithLambdified(mod_file_path, verbose=False) 
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {mod_file_path}")
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize DynareModelWithLambdified: {e}")
        traceback.print_exc()
        exit()
    init_end_time = time.time()
    print(f"Model initialized ({init_end_time - init_start_time:.2f} seconds).")

    # --- Simulation Setup (to generate data for estimation) ---
    sim_param_values = model.default_param_assignments.copy()
    sim_overrides = {
        'b1': 0.7,
        'a1': 0.6,
        'g1': 0.65,
        'rho_L_GDP_GAP': 0.85,
        "sigma_SHK_RS": 0.25
    }
    sim_param_values.update(sim_overrides)
    print("\n--- [2] Simulation parameter set defined ---")

    print("\n--- [3] Simulating Data ---")
    sim_key_master = random.PRNGKey(sim_seed)
    sim_key_init_state, sim_key_path_noise = random.split(sim_key_master)
    
    sim_solution = model.solve(sim_param_values)
    if not sim_solution["solution_valid"]:
        print("FATAL ERROR: Cannot solve model with simulation parameters.")
        exit()
    
    # H_obs_sim: Covariance matrix for observation noise. Diagonal if noises are uncorrelated.
    # If sim_measurement_noise_std is scalar, assume i.i.d. noise for all observables.
    H_obs_sim = jnp.eye(sim_solution["n_obs"], dtype=_DEFAULT_DTYPE) * (sim_measurement_noise_std**2)

    # Initial state for simulation (s0_sim)
    sim_initial_state_config = {
        "L_GDP_TREND": {"mean": 10.0, "std": 0.01},
        "G_TREND": {"mean": 2.0, "std": 0.002},
        "PI_TREND": {"mean": 2.0, "std": 0.01},
        "RR_TREND": {"mean": 1.0, "std": 0.1}
    }
    s0_sim_mean = construct_initial_state(
        n_aug=sim_solution["n_aug"],
        n_stat=model.n_stat,
        aug_state_vars=sim_solution["aug_state_vars"],
        key_init=sim_key_init_state,
        initial_state_config=sim_initial_state_config,
        dtype=_DEFAULT_DTYPE
    )
    s0_sim_cov = jnp.eye(sim_solution["n_aug"], dtype=_DEFAULT_DTYPE) * 1e-4  # Small initial covariance for simulation start point

    sim_start_time = time.time()
    try:
        # Use the simulate_state_space from Kalman_filter_jax.py
        sim_states_true, sim_observables_data = simulate_state_space(
            P_aug=sim_solution["P_aug"],
            R_aug=sim_solution["R_aug"], 
            Omega=sim_solution["Omega"],
            H_obs=H_obs_sim,
            init_x=s0_sim_mean,
            init_P=s0_sim_cov,  # Provide mean and cov for initial draw
            key=sim_key_path_noise, 
            num_steps=num_sim_steps
        )
        sim_end_time = time.time()
        print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")
        # Optional: Plot initial simulation
        # plot_simulation_with_trends_matched(
        #     sim_observables_data,
        #     sim_solution["ordered_obs_vars"],
        #     sim_states_true,
        #     sim_solution["aug_state_vars"],
        #     sim_solution["ordered_trend_state_vars"],
        #     sim_solution.get("contemp_trend_defs", {}),
        #     title=f"Simulated Data (Meas Noise Std={sim_measurement_noise_std:.2e})"
        # )
        # plt.show(block=False) 
    except Exception as e_sim:
        print(f"FATAL ERROR during simulation: {e_sim}")
        traceback.print_exc()
        exit()

    # --- Estimation Setup ---
    print("\n--- [4] Defining Priors for Estimation ---")
    user_priors = [
        {"name": "b1", "prior": "beta", "args": {"concentration1": 2.975, "concentration2": 1.275}},
        {"name": "b4", "prior": "beta", "args": {"concentration1": 2.975, "concentration2": 1.275}},
        {"name": "a1", "prior": "beta", "args": {"concentration1": 2.625, "concentration2": 2.625}},
        {"name": "g1", "prior": "beta", "args": {"concentration1": 2.975, "concentration2": 1.275}},
        {"name": "g3", "prior": "beta", "args": {"concentration1": 4.4375, "concentration2": 13.3125}},
        {"name": "a2", "prior": "halfnormal", "args": {"scale": 0.1}},
        {"name": "g2", "prior": "halfnormal", "args": {"scale": 0.3}},
        {"name": "rho_L_GDP_GAP", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}},
        {"name": "rho_DLA_CPI", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}},
        {"name": "rho_rs", "prior": "beta", "args": {"concentration1": 30.0, "concentration2": 10.0}},
        {"name": "rho_rs2", "prior": "beta", "args": {"concentration1": 1.0, "concentration2": 99.0}},
        {"name": "sigma_SHK_L_GDP_GAP", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_DLA_CPI", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_RS", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_L_GDP_TREND", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_G_TREND", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_PI_TREND", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
        {"name": "sigma_SHK_RR_TREND", "prior": "invgamma", "args": {"concentration": 3.0, "rate": 5.0}},
    ]
    estimated_param_names_set = {p["name"] for p in user_priors}
    fixed_params_est = {
        name: sim_param_values.get(name, model.default_param_assignments.get(name))
        for name in model.all_param_names if name not in estimated_param_names_set
    }
    
    # --- Estimation ---
    posterior_samples = None  # Initialize
    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE:
        print(f"\n--- [5] Running Bayesian Estimation ---")
        mcmc_key = random.PRNGKey(mcmc_seed)
        
        # H_obs for estimation: Use the same H_obs_sim that generated the data
        # If estimating with real data, this would be a calibrated/estimated matrix.
        H_obs_for_estimation = H_obs_sim 
        
        # Initial state for Kalman filter in likelihood: Use s0_sim_mean and s0_sim_cov from simulation setup
        # For real data, these would be diffuse priors or from a pre-period.
        init_x_mean_for_lk = s0_sim_mean 
        init_P_cov_for_lk = s0_sim_cov  

        # Static NaN info for MCMC (derived from sim_observables_data)
        if sim_observables_data is not None and sim_observables_data.shape[0] > 0:
            first_obs_slice_mcmc = onp.asarray(sim_observables_data[0])
            _valid_obs_idx_py_mcmc = onp.where(~onp.isnan(first_obs_slice_mcmc))[0]
            static_valid_obs_idx_mcmc_jnp = jnp.array(_valid_obs_idx_py_mcmc, dtype=jnp.int32)
            static_n_obs_actual_mcmc_py = len(_valid_obs_idx_py_mcmc)
        else: 
            static_valid_obs_idx_mcmc_jnp = jnp.array(onp.arange(model.n_obs), dtype=jnp.int32) 
            static_n_obs_actual_mcmc_py = model.n_obs
        
        init_values_mcmc = {
            p_spec["name"]: sim_param_values.get(p_spec["name"], 0.5) 
            for p_spec in user_priors
        }
        init_strategy = init_to_value(values=init_values_mcmc)
        kernel = NUTS(
            numpyro_model_fixed,
            init_strategy=init_strategy,
            target_accept_prob=mcmc_target_accept
        )
        mcmc = MCMC(
            kernel,
            num_warmup=mcmc_warmup,
            num_samples=mcmc_samples,
            num_chains=mcmc_chains,
            progress_bar=True,
            chain_method='sequential' if mcmc_chains==1 else 'parallel'
        )
        
        est_start_time = time.time()
        try:
            mcmc.run(
                mcmc_key, model, user_priors, fixed_params_est, sim_observables_data, 
                H_obs_for_estimation, init_x_mean_for_lk, init_P_cov_for_lk,
                static_valid_obs_idx_mcmc_jnp, static_n_obs_actual_mcmc_py
            )
            est_end_time = time.time()
            print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")
            mcmc.print_summary()
            posterior_samples = mcmc.get_samples()
            # ... (Arviz plotting for MCMC trace - keep if desired) ...
        except Exception as e_est:
            print(f"\n--- Estimation FAILED ---")
            print(f"Error: {e_est}")
            traceback.print_exc()
    else:
        print("\n--- [5] Skipping Estimation ---")

    # --- [7] Post-Estimation: Smoothed States and Plots ---
    # (Corrected variable names from previous version)
    if run_smoothing_flag and posterior_samples is not None and KALMAN_FILTER_JAX_AVAILABLE:
        print("\n--- [7] Computing and Plotting Smoothed States ---")
        
        import arviz as az 
        from tqdm import tqdm

        key_smooth_master = random.PRNGKey(789)
        total_posterior_draws = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
        
        if num_param_draws_to_use_smoothing > total_posterior_draws:
            num_param_draws_to_use_smoothing = total_posterior_draws
        
        step_smooth = max(1, total_posterior_draws // num_param_draws_to_use_smoothing)
        indices_to_use_smoothing = onp.arange(0, total_posterior_draws, step_smooth)[:num_param_draws_to_use_smoothing]
        actual_param_draws_used_smoothing = len(indices_to_use_smoothing)
        
        print(f"Using {actual_param_draws_used_smoothing} MCMC parameter draws for smoothing.")
        print(f"Generating {num_state_draws_per_param_smoothing} state simulation draws per parameter draw.")

        ys_for_smoothing = sim_observables_data
        H_obs_for_kf_instance = H_obs_for_estimation  # This is the H_full for original data
        init_x_for_kf_instance = init_x_mean_for_lk
        init_P_for_kf_instance = init_P_cov_for_lk
        
        # Static NaN info for the ORIGINAL `ys_for_smoothing`
        _static_valid_idx_orig_data = static_valid_obs_idx_mcmc_jnp
        _static_n_actual_orig_data = static_n_obs_actual_mcmc_py
        
        if _static_n_actual_orig_data == 0:
            _static_H_obs_orig_data_sliced = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
            _static_I_obs_orig_data_sliced = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
        elif _static_n_actual_orig_data == model.n_obs:
            _static_H_obs_orig_data_sliced = H_obs_for_kf_instance  # Full H
            _static_I_obs_orig_data_sliced = jnp.eye(model.n_obs, dtype=_DEFAULT_DTYPE)
        else:
            _static_H_obs_orig_data_sliced = H_obs_for_kf_instance[jnp.ix_(_static_valid_idx_orig_data, _static_valid_idx_orig_data)]
            _static_I_obs_orig_data_sliced = jnp.eye(_static_n_actual_orig_data, dtype=_DEFAULT_DTYPE)

        kf_instance_for_smoothing = KalmanFilter(
            T=jnp.eye(model.n_aug, dtype=_DEFAULT_DTYPE),
            R=jnp.zeros((model.n_aug, model.n_aug_shock), dtype=_DEFAULT_DTYPE),
            C=jnp.zeros((model.n_obs, model.n_aug), dtype=_DEFAULT_DTYPE),
            H=H_obs_for_kf_instance,
            init_x=init_x_for_kf_instance,
            init_P=init_P_for_kf_instance
        )

        all_smoothed_state_draws_list_np = []
        all_omega_draws_for_plotting_list_np = []
        param_draw_keys_smoothing = random.split(key_smooth_master, actual_param_draws_used_smoothing)

        for i_smooth_loop, mcmc_draw_idx_smooth in enumerate(tqdm(indices_to_use_smoothing, desc="Smoothing MCMC Draws")):
            current_param_dict_smooth = {
                name: posterior_samples[name][mcmc_draw_idx_smooth] 
                for name in estimated_param_names_set
            }
            current_param_dict_smooth.update(fixed_params_est)
            solution_smooth = model.solve(current_param_dict_smooth)
            if not solution_smooth["solution_valid"]:
                continue

            Omega_draw_smooth = solution_smooth["Omega"]
            kf_instance_for_smoothing.T = solution_smooth["P_aug"]
            kf_instance_for_smoothing.R = solution_smooth["R_aug"]
            kf_instance_for_smoothing.C = Omega_draw_smooth
            
            if model.n_aug_shock > 0:
                kf_instance_for_smoothing.state_cov = solution_smooth["R_aug"] @ solution_smooth["R_aug"].T
            else:
                kf_instance_for_smoothing.state_cov = jnp.zeros(
                    (model.n_aug, model.n_aug),
                    dtype=solution_smooth["P_aug"].dtype
                )

            # C_obs for filtering original `ys` with THIS param draw's Omega
            if _static_n_actual_orig_data == 0:
                C_obs_orig_data_this_param = jnp.empty((0, model.n_aug), dtype=Omega_draw_smooth.dtype)
            elif _static_n_actual_orig_data == model.n_obs:
                C_obs_orig_data_this_param = Omega_draw_smooth
            else:
                C_obs_orig_data_this_param = Omega_draw_smooth[_static_valid_idx_orig_data, :]
            
            static_info_orig_data_for_smoother_call = {
                "static_valid_obs_idx": _static_valid_idx_orig_data,
                "static_n_obs_actual": _static_n_actual_orig_data,
                "static_C_obs": C_obs_orig_data_this_param,
                "static_H_obs": _static_H_obs_orig_data_sliced,
                "static_I_obs": _static_I_obs_orig_data_sliced
            }
            
            # Filter original data to pass to simulation_smoother
            # This uses self.filter with NaN handling for original ys
            filter_res_orig_ys_this_param = kf_instance_for_smoothing.filter(
                ys_for_smoothing,
                _static_valid_idx_orig_data,
                _static_n_actual_orig_data,
                C_obs_orig_data_this_param,
                _static_H_obs_orig_data_sliced,
                _static_I_obs_orig_data_sliced
            )
            
            _, _, smoothed_states_one_param = kf_instance_for_smoothing.simulation_smoother(
                ys=ys_for_smoothing,
                key=param_draw_keys_smoothing[i_smooth_loop],
                static_info_for_original_ys=static_info_orig_data_for_smoother_call, # MOVED to be before num_draws
                num_draws=num_state_draws_per_param_smoothing,
                filter_results_for_original_ys=filter_res_orig_ys_this_param
            )
            all_smoothed_state_draws_list_np.append(onp.asarray(smoothed_states_one_param))
            all_omega_draws_for_plotting_list_np.append(onp.asarray(Omega_draw_smooth))

        if not all_smoothed_state_draws_list_np:
            print("No valid smoothed states generated. Skipping plotting.")
        else:
            all_smoothed_state_draws_stacked_np = onp.stack(all_smoothed_state_draws_list_np, axis=0)
            total_final_state_draws = all_smoothed_state_draws_stacked_np.shape[0] * all_smoothed_state_draws_stacked_np.shape[1]
            all_smoothed_state_draws_flat_np = all_smoothed_state_draws_stacked_np.reshape(
                total_final_state_draws,
                all_smoothed_state_draws_stacked_np.shape[2],
                all_smoothed_state_draws_stacked_np.shape[3]
            )
            print(f"Generated {total_final_state_draws} smoothed state trajectories.")

            time_horizon_plot = onp.arange(ys_for_smoothing.shape[0])
            
            def plot_smoothed_var_with_hdi(ax, data_actual, smoothed_draws_var, title,
                                         trend_draws_var=None, trend_label="Trend",
                                         actual_label="Actual Data",
                                         true_state_sim=None, true_state_label="True Sim State"):
                median_smooth = onp.median(smoothed_draws_var, axis=0)
                hdi_68 = az.hdi(onp.asarray(smoothed_draws_var), hdi_prob=0.68)
                hdi_80 = az.hdi(onp.asarray(smoothed_draws_var), hdi_prob=0.80)
                
                if data_actual is not None:
                    ax.plot(time_horizon_plot, data_actual, 'k-', label=actual_label, alpha=0.5, linewidth=1.0)
                
                if true_state_sim is not None and sim_true_states_known:
                    ax.plot(time_horizon_plot, true_state_sim, 'g--', label=true_state_label, alpha=0.7, linewidth=1.2)
                
                ax.plot(time_horizon_plot, median_smooth, color='mediumblue', label="Smoothed Median", linewidth=1.5)
                ax.fill_between(time_horizon_plot, hdi_80[:,0], hdi_80[:,1], color='lightskyblue', alpha=0.4, label="80% HDI")
                ax.fill_between(time_horizon_plot, hdi_68[:,0], hdi_68[:,1], color='cornflowerblue', alpha=0.6, label="68% HDI")
                
                if trend_draws_var is not None:
                    median_trend = onp.median(trend_draws_var, axis=0)
                    ax.plot(time_horizon_plot, median_trend, color='orangered', linestyle=':', 
                           label=f"Smoothed {trend_label} (Median)", linewidth=1.5)
                
                ax.set_title(title, fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.4)
                ax.legend(fontsize='xx-small', loc='upper left')
                ax.tick_params(axis='both', which='major', labelsize=8)

            # Plot Stationary States
            num_stat_vars_plot = model.n_stat
            if num_stat_vars_plot > 0:
                cols_p = min(3, num_stat_vars_plot)
                rows_p = (num_stat_vars_plot + cols_p - 1) // cols_p
                fig_stat, axes_stat = plt.subplots(
                    rows_p, cols_p,
                    figsize=(min(5 * cols_p, 15), 2.5 * rows_p),
                    sharex=True,
                    squeeze=False
                )
                axes_stat = axes_stat.flatten()
                fig_stat.suptitle("Smoothed Stationary States", fontsize=12)
                
                for i in range(num_stat_vars_plot):
                    plot_smoothed_var_with_hdi(
                        axes_stat[i],
                        None,
                        all_smoothed_state_draws_flat_np[:, :, i],
                        model.ordered_stat_vars[i],
                        true_state_sim=sim_states_true[:, i] if sim_states_true is not None else None
                    )
                
                for jax_s in range(i + 1, len(axes_stat)):
                    axes_stat[jax_s].set_visible(False)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Plot Trend States
            num_trend_vars_plot = model.n_trend
            if num_trend_vars_plot > 0:
                cols_p = min(3, num_trend_vars_plot)
                rows_p = (num_trend_vars_plot + cols_p - 1) // cols_p
                fig_trend, axes_trend = plt.subplots(
                    rows_p, cols_p,
                    figsize=(min(5 * cols_p, 15), 2.5 * rows_p),
                    sharex=True,
                    squeeze=False
                )
                axes_trend = axes_trend.flatten()
                fig_trend.suptitle("Smoothed Trend States", fontsize=12)
                
                for i in range(num_trend_vars_plot):
                    trend_idx_aug = model.n_stat + i
                    plot_smoothed_var_with_hdi(
                        axes_trend[i],
                        None,
                        all_smoothed_state_draws_flat_np[:, :, trend_idx_aug],
                        model.ordered_trend_state_vars[i],
                        true_state_sim=sim_states_true[:, trend_idx_aug] if sim_states_true is not None else None
                    )
                
                for jax_s in range(i + 1, len(axes_trend)):
                    axes_trend[jax_s].set_visible(False)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Contemporaneous Trends (e.g., RS_TREND)
            smoothed_contemp_trends_for_plot = {}
            if ("RS_TREND" in model.trend_vars and 
                "RR_TREND" in model.ordered_trend_state_vars and 
                "PI_TREND" in model.ordered_trend_state_vars):
                idx_rr_trend = model.ordered_trend_state_vars.index("RR_TREND") + model.n_stat
                idx_pi_trend = model.ordered_trend_state_vars.index("PI_TREND") + model.n_stat
                smoothed_contemp_trends_for_plot["RS_TREND"] = (
                    all_smoothed_state_draws_flat_np[:, :, idx_rr_trend] + 
                    all_smoothed_state_draws_flat_np[:, :, idx_pi_trend]
                )
                # Plot RS_TREND itself if desired (similar to other trends)
                # fig_rs, ax_rs = plt.subplots(1, 1, figsize=(5, 2.5))
                # plot_smoothed_var_with_hdi(
                #     ax_rs, None, smoothed_contemp_trends_for_plot["RS_TREND"],
                #     "Smoothed RS_TREND (Contemporaneous)"
                # )
                # plt.tight_layout()

            # Reconstruct and Plot Observables
            all_smoothed_obs_draws_list_recalc_np = []
            for i_recalc_obs, Omega_this_param_draw_np_val in enumerate(tqdm(all_omega_draws_for_plotting_list_np, desc="Reconstructing Observables")):
                state_draws_for_this_param_np_val = all_smoothed_state_draws_list_np[i_recalc_obs]  # (S, T, n_aug)
                reconstructed_obs = onp.einsum('stk,jk->stj', state_draws_for_this_param_np_val, Omega_this_param_draw_np_val)
                all_smoothed_obs_draws_list_recalc_np.append(reconstructed_obs)
            
            if all_smoothed_obs_draws_list_recalc_np:
                all_smoothed_obs_draws_flat_np_recalc = onp.concatenate(all_smoothed_obs_draws_list_recalc_np, axis=0)
                num_obs_vars_plot = model.n_obs
                cols_p = min(2, num_obs_vars_plot)
                rows_p = (num_obs_vars_plot + cols_p - 1) // cols_p
                fig_obs, axes_obs = plt.subplots(
                    rows_p, cols_p,
                    figsize=(min(6 * cols_p, 12), 3 * rows_p),
                    sharex=True,
                    squeeze=False
                )
                axes_obs = axes_obs.flatten()
                fig_obs.suptitle("Smoothed Observable Variables", fontsize=12)
                
                obs_to_trend_map = {
                    "L_GDP_OBS": "L_GDP_TREND",
                    "DLA_CPI_OBS": "PI_TREND",
                    "PI_TREND_OBS": "PI_TREND",
                    "RS_OBS": "RS_TREND"
                }
                
                for i in range(num_obs_vars_plot):
                    obs_name = model.ordered_obs_vars[i]
                    trend_name_map = obs_to_trend_map.get(obs_name)
                    trend_draws_plot_obs = None
                    
                    if trend_name_map:
                        if trend_name_map in model.ordered_trend_state_vars:  # It's a state trend
                            trend_idx_plot_obs = model.ordered_trend_state_vars.index(trend_name_map) + model.n_stat
                            trend_draws_plot_obs = all_smoothed_state_draws_flat_np[:, :, trend_idx_plot_obs]
                        elif trend_name_map in smoothed_contemp_trends_for_plot:  # It's a reconstructed contemp trend
                            trend_draws_plot_obs = smoothed_contemp_trends_for_plot[trend_name_map]
                    
                    plot_smoothed_var_with_hdi(
                        axes_obs[i],
                        onp.asarray(ys_for_smoothing[:, i]),
                        all_smoothed_obs_draws_flat_np_recalc[:, :, i],
                        obs_name,
                        trend_draws_var=trend_draws_plot_obs,
                        trend_label=trend_name_map if trend_name_map else "N/A"
                    )
                
                for jax_s in range(i + 1, len(axes_obs)):
                    axes_obs[jax_s].set_visible(False)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        print("\n--- [7] Skipping Smoothed States Computation and Plotting ---")

    print(f"\n--- Script finished ---")
    if plt.get_fignums():
        print("Displaying all generated plots. Close plot windows to exit.")
        plt.show()