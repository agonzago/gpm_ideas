# --- START OF FILE dynare_model_wrapper.py ---

import os
import jax
import jax.numpy as jnp
import numpy as onp # For checking instance types, default values
from jax.typing import ArrayLike
from jax import random, lax # <<< Ensure lax is imported
# <<< --- ADD THIS IMPORT --- >>>
import jax.debug as jdebug
# <<< --- END ADD IMPORT --- >>>
from typing import Dict, List, Tuple, Optional, Union, Any

# --- JAX Configuration
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# --- Numpyro Imports ---
try:
    import numpyro
    import numpyro.distributions as dist
    # <<< Use init_to_uniform >>>
    from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_uniform
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    print("Warning: numpyro not found. Estimation functionality will be disabled.")

# --- Import your custom modules ---
import Dynare_parser_sda_solver as dp
# <<< Import KalmanFilter for internal use check >>>
from Kalman_filter_jax import KalmanFilter, simulate_state_space, _MACHINE_EPSILON


class DynareModel:
    """ Wrapper class for Dynare models """
    # --- __init__ remains the same ---
    def __init__(self, dynare_file_path: str):
        """ Initializes the DynareModel by parsing the .dyn file. """
        if not os.path.exists(dynare_file_path):
            raise FileNotFoundError(f"Model file not found at: {dynare_file_path}")
        self.dynare_file_path = dynare_file_path
        with open(self.dynare_file_path, 'r') as f: model_def = f.read()
        self.dtype = _DEFAULT_DTYPE
        # --- [1] PARSING AND LAMBDIFYING ---
        try:
            (self.func_A, self.func_B, self.func_C, self.func_D,
             self.ordered_stat_vars, self.stat_shocks,
             self.param_names_stat_combined,
             self.default_param_assignments_stat,
             _, _) = dp.parse_lambdify_and_order_model(model_def, verbose=False)
        except Exception as e: raise ValueError(f"Failed to parse stationary model from {dynare_file_path}") from e
        try:
            self.trend_vars, self.trend_shocks = dp.extract_trend_declarations(model_def)
            trend_equations = dp.extract_trend_equations(model_def)
            self.obs_vars = dp.extract_observation_declarations(model_def)
            measurement_equations = dp.extract_measurement_equations(model_def)
            self.trend_stderr_params = dp.extract_trend_shock_stderrs(model_def)
        except Exception as e: raise ValueError(f"Failed to parse trend/observation components from {dynare_file_path}") from e
        # --- Combine parameters ---
        self.all_param_names = list(dict.fromkeys(self.param_names_stat_combined + list(self.trend_stderr_params.keys())).keys())
        self.default_param_assignments = self.default_param_assignments_stat.copy()
        self.default_param_assignments.update(self.trend_stderr_params)
        self.aug_shocks_structure = self.stat_shocks + self.trend_shocks
        # --- Build Trend/Observation Matrices ---
        try:
            (self.func_P_trends, self.func_Q_trends,
             self.ordered_trend_state_vars, self.contemp_trend_defs) = dp.build_trend_matrices(
                trend_equations, self.trend_vars, self.trend_shocks, self.all_param_names, self.default_param_assignments, verbose=False)
            (self.func_Omega, self.ordered_obs_vars) = dp.build_observation_matrix(
                measurement_equations, self.obs_vars, self.ordered_stat_vars, self.ordered_trend_state_vars,
                self.contemp_trend_defs, self.all_param_names, self.default_param_assignments, verbose=False)
        except Exception as e: raise ValueError("Failed to build symbolic trend/observation matrices") from e
        # --- Store structure info ---
        self.aug_state_vars_structure = self.ordered_stat_vars + self.ordered_trend_state_vars
        self.n_state_aug = len(self.aug_state_vars_structure)
        self.n_shock_aug = len(self.aug_shocks_structure)
        self.n_obs = len(self.ordered_obs_vars)


    # --- solve method remains the same ---
    def solve(self, param_dict: Dict[str, float], max_iter_sda: int = 500) -> Dict[str, Any]:
        """ Solves the model using JAX functions, designed for robustness. (Attempt 7) """
        stat_args, all_args, shock_std_devs = self._prepare_params(param_dict)
        eval_params = {name: val for name, val in zip(self.all_param_names, all_args)}
        A_num_stat = jnp.asarray(self.func_A(*stat_args), dtype=self.dtype)
        B_num_stat = jnp.asarray(self.func_B(*stat_args), dtype=self.dtype)
        C_num_stat = jnp.asarray(self.func_C(*stat_args), dtype=self.dtype)
        D_num_stat = jnp.asarray(self.func_D(*stat_args), dtype=self.dtype)

        P_sol_stat, _, _, converged_flag = dp.solve_quadratic_matrix_equation_jax( # Capture flag
            A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=max_iter_sda, verbose=False)

        # <<< Check convergence flag from solver >>>
        # If not converged, P_sol_stat might be NaN. Let compute_Q handle NaN propagation.
        # jdebug.print("Solve SDA converged: {flag}", flag=converged_flag) # Optional: print convergence

        Q_sol_stat = dp.compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat, dtype=self.dtype)

        P_num_trend = jnp.asarray(self.func_P_trends(*all_args), dtype=self.dtype)
        Q_num_trend = jnp.asarray(self.func_Q_trends(*all_args), dtype=self.dtype)

        common_dtype = self.dtype
        stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=common_dtype)
        trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=common_dtype)

        n_stat_vars = len(self.ordered_stat_vars)
        n_stat_shock_expected = len(stat_std_devs_arr)
        # Scale Q_stat by sigmas: Q_stat @ diag(sigmas_stat) -> R_stat
        R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr)

        n_trend_vars = len(self.ordered_trend_state_vars)
        n_trend_shock_expected = len(trend_std_devs_arr)
        # Scale Q_trend by sigmas: Q_trend @ diag(sigmas_trend) -> R_trend
        R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr)

        # Build augmented system
        n_stat = P_sol_stat.shape[0]; n_trend = P_num_trend.shape[0]; n_aug = n_stat + n_trend
        n_stat_shock = len(self.stat_shocks); n_trend_shock = len(self.trend_shocks); n_aug_shock = n_stat_shock + n_trend_shock

        P_aug = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)

        R_aug = jnp.zeros((n_aug, n_aug_shock), dtype=common_dtype)
        R_sol_stat_shaped = R_sol_stat.reshape(n_stat, n_stat_shock) if R_sol_stat.size > 0 else jnp.zeros((n_stat, n_stat_shock))
        R_num_trend_shaped = R_num_trend.reshape(n_trend, n_trend_shock) if R_num_trend.size > 0 else jnp.zeros((n_trend, n_trend_shock))
        R_aug = R_aug.at[:n_stat, :n_stat_shock].set(R_sol_stat_shaped)
        R_aug = R_aug.at[n_stat:, n_stat_shock:].set(R_num_trend_shaped)

        Omega_num = jnp.asarray(self.func_Omega(*all_args), dtype=self.dtype)

        aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars
        aug_shocks = self.stat_shocks + self.trend_shocks
        obs_vars_ordered = self.ordered_obs_vars

        return {
            'P_aug': P_aug, 'R_aug': R_aug, 'Omega': Omega_num,
            # Also return intermediate results if needed for debug
            # 'P_sol_stat': P_sol_stat, 'R_sol_stat': R_sol_stat,
            # 'P_num_trend': P_num_trend, 'R_num_trend': R_num_trend,
            'aug_state_vars': aug_state_vars, 'aug_shocks': aug_shocks,
            'obs_vars_ordered': obs_vars_ordered,
            'param_values_used': eval_params
            }


    # --- get_irf, simulate, run_kalman remain the same ---
    def get_irf(self, param_dict: Dict[str, float], shock_name: str, horizon: int = 40) -> Dict[str, Any]:
        solution = self.solve(param_dict)
        P_aug=solution['P_aug']; R_aug=solution['R_aug']; Omega_num=solution['Omega']
        aug_shocks=solution['aug_shocks']; aug_state_vars=solution['aug_state_vars']; obs_vars_ordered=solution['obs_vars_ordered']
        try: shock_index = aug_shocks.index(shock_name)
        except ValueError: raise ValueError(f"Shock '{shock_name}' not found: {aug_shocks}")
        # Ensure irf functions return JAX arrays if needed downstream, or convert after
        irf_states_aug = dp.irf(P_aug, R_aug, shock_index=shock_index, horizon=horizon) # Assumes this is updated
        irf_observables_vals = dp.irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index, horizon=horizon) # Assumes this is updated
        return {'state_irf': irf_states_aug, 'observable_irf': irf_observables_vals, 'state_names': aug_state_vars, 'observable_names': obs_vars_ordered, 'shock_name': shock_name, 'horizon': horizon}

    def simulate(self, param_dict: Dict[str, float], H_obs: ArrayLike, init_x_mean: ArrayLike, init_P_cov: ArrayLike, key: jax.random.PRNGKey, num_steps: int) -> Dict[str, jax.Array]:
        solution = self.solve(param_dict)
        P_aug=solution['P_aug']; R_aug=solution['R_aug']; Omega_num=solution['Omega']
        n_aug=P_aug.shape[0]; n_obs=Omega_num.shape[0]
        H_obs_jax=jnp.asarray(H_obs); init_x_jax=jnp.asarray(init_x_mean); init_P_jax=jnp.asarray(init_P_cov)
        if H_obs_jax.shape!=(n_obs,n_obs): raise ValueError("H_obs shape mismatch")
        if init_x_jax.shape!=(n_aug,): raise ValueError("init_x shape mismatch")
        if init_P_jax.shape!=(n_aug,n_aug): raise ValueError("init_P shape mismatch")
        sim_states, sim_observations = simulate_state_space(P_aug, R_aug, Omega_num, H_obs_jax, init_x_jax, init_P_jax, key, num_steps)
        return {'sim_states': sim_states, 'sim_observations': sim_observations}

    def run_kalman(self, param_dict: Dict[str, float], ys: ArrayLike, H_obs: ArrayLike, init_x_mean: ArrayLike, init_P_cov: ArrayLike, smoother_key: Optional[jax.random.PRNGKey]=None, num_sim_smoother_draws: int=0) -> Dict[str, Any]:
        solution = self.solve(param_dict)
        P_aug=solution['P_aug']; R_aug=solution['R_aug']; Omega_num=solution['Omega']
        n_aug=P_aug.shape[0]; n_obs=Omega_num.shape[0]
        ys_jax=jnp.asarray(ys); H_obs_jax=jnp.asarray(H_obs); init_x_jax=jnp.asarray(init_x_mean); init_P_jax=jnp.asarray(init_P_cov)
        # Validation...
        if ys_jax.ndim!=2 or ys_jax.shape[1]!=n_obs: raise ValueError("ys shape mismatch")
        if H_obs_jax.shape!=(n_obs,n_obs): raise ValueError("H_obs shape mismatch")
        if init_x_jax.shape!=(n_aug,): raise ValueError("init_x shape mismatch")
        if init_P_jax.shape!=(n_aug,n_aug): raise ValueError("init_P shape mismatch")
        if num_sim_smoother_draws > 0 and smoother_key is None: raise ValueError("smoother_key needed")
        if num_sim_smoother_draws < 0: raise ValueError("num_sim_smoother_draws negative")

        # <<< Check for NaNs in inputs BEFORE filtering >>>
        if not (jnp.all(jnp.isfinite(P_aug)) and jnp.all(jnp.isfinite(R_aug)) and jnp.all(jnp.isfinite(Omega_num))):
            print("Warning run_kalman: Input matrices P_aug, R_aug, or Omega contain non-finite values.")
            # Depending on desired behavior, could raise error or return empty dict
            return {} # Return empty dict to indicate failure

        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega_num, H=H_obs_jax, init_x=init_x_jax, init_P=init_P_jax)
        results = {}
        filter_outs = kf.filter(ys_jax) # Uses main KF filter (Static NaN)
        results['filtered_states']=filter_outs['x_filt']; results['filtered_cov']=filter_outs['P_filt']
        # Check if filter outputs are valid before smoothing
        if not jnp.all(jnp.isfinite(filter_outs['x_filt'])):
             print("Warning run_kalman: Filtered states contain non-finite values. Skipping smoother.")
             return results # Return only filter results

        x_smooth_rts, P_smooth_rts = kf.smooth(ys_jax, filter_results=filter_outs) # Uses main KF filter if needed
        results['rts_smoothed_states']=x_smooth_rts; results['rts_smoothed_cov']=P_smooth_rts
        if not jnp.all(jnp.isfinite(x_smooth_rts)):
            print("Warning run_kalman: RTS Smoothed states contain non-finite values.")
            # Continue to simulation smoother if requested, but RTS might be unusable

        if num_sim_smoother_draws > 0:
            sim_smoother_result = kf.simulation_smoother(ys_jax, smoother_key, num_draws=num_sim_smoother_draws, filter_results=filter_outs)
            # Result handling based on num_draws
            if num_sim_smoother_draws == 1:
                 if jnp.all(jnp.isfinite(sim_smoother_result)):
                     results['sim_smoothed_draws']=sim_smoother_result[None,:,:]; results['sim_smoothed_mean']=sim_smoother_result; results['sim_smoothed_median']=sim_smoother_result
                 else: print("Warning run_kalman: Sim smoother (1 draw) contains non-finite values.")
            else: # num_draws > 1
                 mean_draws, median_draws, all_draws = sim_smoother_result
                 if jnp.all(jnp.isfinite(all_draws)):
                    results['sim_smoothed_mean']=mean_draws; results['sim_smoothed_median']=median_draws; results['sim_smoothed_draws']=all_draws
                 else: print("Warning run_kalman: Sim smoother draws contain non-finite values.")
        return results


    # --- _get_numpyro_dist remains the same ---
    def _get_numpyro_dist(self, dist_name: str, params: Union[List, Tuple]) -> numpyro.distributions.Distribution:
        # ... (implementation as before) ...
        if not NUMPYRO_AVAILABLE: raise RuntimeError("Numpyro not installed.")
        dist_name_lower = dist_name.lower()
        if dist_name_lower == "normal": return dist.Normal(loc=params[0], scale=params[1])
        elif dist_name_lower == "beta": return dist.Beta(concentration1=params[0], concentration0=params[1])
        elif dist_name_lower == "gamma": return dist.Gamma(concentration=params[0], rate=params[1])
        elif dist_name_lower == "inversegamma": return dist.InverseGamma(concentration=params[0], rate=params[1])
        elif dist_name_lower == "uniform": return dist.Uniform(low=params[0], high=params[1])
        else: raise ValueError(f"Unsupported prior distribution name: {dist_name}")


    # # --- _numpyro_model (Attempt 6 / lax.cond version) with DEBUG PRINTING ---
    # def _numpyro_model(self, ys: jax.Array, H_obs: jax.Array,
    #                    init_x_mean: jax.Array, init_P_cov: jax.Array,
    #                    priors: Dict[str, Tuple[str, Union[List, Tuple]]],
    #                    fixed_params: Dict[str, float],
    #                    verbose_solver: bool = False): # verbose_solver might not do much here
    #     """Internal numpyro model function called by MCMC. Uses lax.cond. WITH DEBUG PRINTS."""
    #     if not NUMPYRO_AVAILABLE: raise RuntimeError("Numpyro is not installed.")
    #     _LOW_LOGLIKE_PENALTY = jnp.array(-1e15, dtype=self.dtype)

    #     # --- Sample parameters from priors ---
    #     sampled_params = {}
    #     for name, (dist_name, dist_params) in priors.items():
    #         prior_dist = self._get_numpyro_dist(dist_name, dist_params)
    #         # Use sample_with_intermediates=True maybe? For now, standard sample
    #         sampled_params[name] = numpyro.sample(name, prior_dist)

    #     # Combine sampled and fixed parameters
    #     current_params = fixed_params.copy()
    #     current_params.update(sampled_params)

    #     # <<< DEBUG: Print parameters being evaluated >>>
    #     # Print a few key parameters to see their values during initialization
    #     jdebug.print("--- _numpyro_model evaluating with params (sample): ---")
    #     jdebug.print(" b1={p1}, rho_rs={p2}, sigma_SHK_RS={p3}",
    #                  p1=current_params['b1'], p2=current_params['rho_rs'], p3=current_params['sigma_SHK_RS'])


    #     # --- Solve the model for the current parameters ---
    #     def solve_and_check(params):
    #         # Initialize outputs for failure case
    #         n_aug=self.n_state_aug; n_shock=self.n_shock_aug; n_obs=self.n_obs
    #         dummy_P=jnp.full((n_aug,n_aug), jnp.nan, dtype=self.dtype)
    #         dummy_R=jnp.full((n_aug,n_shock), jnp.nan, dtype=self.dtype)
    #         dummy_O=jnp.full((n_obs,n_aug), jnp.nan, dtype=self.dtype)
    #         P_aug_s, R_aug_s, Omega_num_s = dummy_P, dummy_R, dummy_O
    #         valid_s = jnp.array(False) # Start assuming invalid

    #         try:
    #             # jdebug.print("Calling self.solve...") # Optional print before solve
    #             solution = self.solve(params) # Calls Attempt 7 solve
    #             P_aug_s = solution['P_aug']; R_aug_s = solution['R_aug']; Omega_num_s = solution['Omega']

    #             # Check for NaNs/Infs in the results
    #             valid_s = (jnp.all(jnp.isfinite(P_aug_s)) &
    #                        jnp.all(jnp.isfinite(R_aug_s)) &
    #                        jnp.all(jnp.isfinite(Omega_num_s)))

    #             # <<< DEBUG: Print solve results >>>
    #             # Printcorner elements and validity flag
    #             jdebug.print(" solve_and_check: P[0,0]={p00}, R[0,0]={r00}, O[0,0]={o00}, valid={v}",
    #                          p00=P_aug_s[0,0], r00=R_aug_s[0,0], o00=Omega_num_s[0,0], v=valid_s)

    #         except Exception as e: # Catch potential runtime errors during solve itself
    #             # <<< DEBUG: Print solve exception marker >>>
    #             jdebug.print("!!! solve_and_check EXCEPTION encountered !!!")
    #             # Output remains the dummy NaN matrices and valid_s = False

    #         return P_aug_s, R_aug_s, Omega_num_s, valid_s

    #     # Call the solver function
    #     P_aug, R_aug, Omega_num, solution_valid = solve_and_check(current_params)


    #     # --- Calculate likelihood using the robust Kalman filter ---
    #     def calculate_likelihood(kf_inputs):
    #         P, R, Omega, ys_k, H_k, init_x_k, init_P_k = kf_inputs
    #         # <<< DEBUG: Print inputs to likelihood calc >>>
    #         # Check if inputs themselves are valid before passing to KF
    #         inputs_valid = (jnp.all(jnp.isfinite(P)) &
    #                         jnp.all(jnp.isfinite(R)) &
    #                         jnp.all(jnp.isfinite(Omega)))
    #         # jdebug.print("calculate_likelihood inputs: P[0,0]={p00}, R[0,0]={r00}, O[0,0]={o00}, inputs_valid={iv}",
    #         #              p00=P[0,0], r00=R[0,0], o00=Omega[0,0], iv=inputs_valid) # Less noisy print

    #         log_likelihood_val = _LOW_LOGLIKE_PENALTY # Default to penalty
    #         likelihood_valid = jnp.array(False)      # Default to invalid

    #         # Only proceed if inputs are valid
    #         # This is a Python conditional, JAX might trace both branches?
    #         # Using lax.cond might be better here too, but let's try this first.
    #         # if inputs_valid: # This might not work as expected under JIT
    #         # Try without the if for now, let KF handle potential NaNs
    #         try:
    #             # Instantiate KF - this might fail if matrices are bad (e.g., H non-PSD)
    #             # Although H is constant here. Failure more likely in filter step.
    #             kf = KalmanFilter(T=P, R=R, C=Omega, H=H_k, init_x=init_x_k, init_P=init_P_k)
    #             filter_results = kf.filter_for_likelihood(ys_k) # Use robust filter

    #             # Extract likelihood sum
    #             log_likelihood_val = jnp.sum(filter_results['log_likelihood_contributions'])

    #             # Check if the calculated likelihood is finite
    #             likelihood_valid = jnp.isfinite(log_likelihood_val)

    #             # <<< DEBUG: Print raw likelihood value and validity >>>
    #             jdebug.print(" calculate_likelihood: raw_ll={ll}, ll_valid={v}",
    #                          ll=log_likelihood_val, v=likelihood_valid)

    #         except Exception as e_kf: # Catch potential runtime errors in filter
    #             # <<< DEBUG: Print KF exception marker >>>
    #             jdebug.print("!!! calculate_likelihood EXCEPTION encountered !!!")
    #             # Ensure likelihood remains invalid and penalized
    #             log_likelihood_val = _LOW_LOGLIKE_PENALTY
    #             likelihood_valid = jnp.array(False)
    #         # End of input validity check (removed)

    #         # Apply penalty if not valid OR if inputs were invalid
    #         # Need to ensure likelihood_valid remains False if inputs were bad
    #         final_likelihood_valid = likelihood_valid # & inputs_valid (rely on propagation now)
    #         safe_log_likelihood = jnp.where(final_likelihood_valid,
    #                                         log_likelihood_val,
    #                                         _LOW_LOGLIKE_PENALTY)

    #         # <<< DEBUG: Print safe likelihood value >>>
    #         jdebug.print(" calculate_likelihood: safe_ll={sll}", sll=safe_log_likelihood)
    #         return safe_log_likelihood


    #     # --- Use lax.cond to determine final log likelihood ---
    #     # Define the inputs tuple ONCE
    #     kalman_inputs = (P_aug, R_aug, Omega_num, ys, H_obs, init_x_mean, init_P_cov)

    #     # If solution_valid is True, call calculate_likelihood(kalman_inputs)
    #     # If solution_valid is False, call lambda operand: _LOW_LOGLIKE_PENALTY (operand is ignored)
    #     log_likelihood = lax.cond(
    #         solution_valid,
    #         calculate_likelihood,           # Function for True case
    #         lambda args: _LOW_LOGLIKE_PENALTY, # Function for False case (takes args)
    #         kalman_inputs                   # Operand passed to the chosen function
    #     )

    #     # <<< DEBUG: Print final log_likelihood value BEFORE numpyro.factor >>>
    #     jdebug.print("--- _numpyro_model reporting factor: log_likelihood={ll}, solution_valid={sv}",
    #                  ll=log_likelihood, sv=solution_valid)

    #     # Report likelihood factor to numpyro
    #     numpyro.factor("log_likelihood", log_likelihood)

    # --- _numpyro_model: Using kf.log_likelihood() ---
    def _numpyro_model(self, ys: jax.Array, H_obs: jax.Array,
                       init_x_mean: jax.Array, init_P_cov: jax.Array,
                       priors: Dict[str, Tuple[str, Union[List, Tuple]]],
                       fixed_params: Dict[str, float],
                       verbose_solver: bool = False):
        """Internal numpyro model. Uses kf.log_likelihood() which calls self.filter()."""
        if not NUMPYRO_AVAILABLE: raise RuntimeError("Numpyro is not installed.")
        _LOW_LOGLIKE_PENALTY = jnp.array(-jnp.inf, dtype=self.dtype) # Use -inf penalty

        # --- Sample parameters ---
        sampled_params = {}
        for name, (dist_name, dist_params) in priors.items():
            prior_dist = self._get_numpyro_dist(dist_name, dist_params)
            sampled_params[name] = numpyro.sample(name, prior_dist)
        current_params = fixed_params.copy()
        current_params.update(sampled_params)

        # <<< DEBUG: Print parameters >>>
        # jdebug.print("--- _numpyro_model evaluating with params (sample): ---")
        # jdebug.print(" b1={p1}, rho_rs={p2}, sigma_SHK_RS={p3}",
        #              p1=current_params['b1'], p2=current_params['rho_rs'], p3=current_params['sigma_SHK_RS'])

        # --- Solve the model ---
        def solve_and_check(params):
            # ... (solve_and_check implementation with its debug prints remains the same) ...
            n_aug=self.n_state_aug; n_shock=self.n_shock_aug; n_obs=self.n_obs
            dummy_P=jnp.full((n_aug,n_aug), jnp.nan, dtype=self.dtype)
            dummy_R=jnp.full((n_aug,n_shock), jnp.nan, dtype=self.dtype)
            dummy_O=jnp.full((n_obs,n_aug), jnp.nan, dtype=self.dtype)
            P_aug_s, R_aug_s, Omega_num_s = dummy_P, dummy_R, dummy_O
            valid_s = jnp.array(False)
            try:
                solution = self.solve(params)
                P_aug_s = solution['P_aug']; R_aug_s = solution['R_aug']; Omega_num_s = solution['Omega']
                valid_s = (jnp.all(jnp.isfinite(P_aug_s)) &
                           jnp.all(jnp.isfinite(R_aug_s)) &
                           jnp.all(jnp.isfinite(Omega_num_s)))
                # jdebug.print(" solve_and_check: P[0,0]={p00}, R[0,0]={r00}, O[0,0]={o00}, valid={v}",
                #              p00=P_aug_s[0,0], r00=R_aug_s[0,0], o00=Omega_num_s[0,0], v=valid_s)
            except Exception as e:
                # jdebug.print("!!! solve_and_check EXCEPTION encountered !!!")
                pass # Keep dummy values and valid_s=False
            return P_aug_s, R_aug_s, Omega_num_s, valid_s

        P_aug, R_aug, Omega_num, solution_valid = solve_and_check(current_params)

        # --- Calculate likelihood using kf.log_likelihood (which calls kf.filter) ---
        def calculate_likelihood(kf_inputs):
            # Unpack inputs (even though some might be unused if kf is instantiated inside)
            P, R, Omega, ys_k, H_k, init_x_k, init_P_k = kf_inputs
            log_likelihood_val = _LOW_LOGLIKE_PENALTY # Default to penalty

            # <<< Instantiate KF and call log_likelihood method >>>
            try:
                kf = KalmanFilter(T=P, R=R, C=Omega, H=H_k, init_x=init_x_k, init_P=init_P_k)
                # This now calls the log_likelihood method which uses the simple filter
                log_likelihood_val = kf.log_likelihood(ys_k)
                # log_likelihood already returns penalized value (-inf if needed)
                # jdebug.print(" calculate_likelihood: kf.log_likelihood result={ll}", ll=log_likelihood_val) # Debug
            except Exception as e_kf:
                 # jdebug.print("!!! calculate_likelihood (calling kf.log_likelihood) EXCEPTION encountered !!!")
                 log_likelihood_val = _LOW_LOGLIKE_PENALTY # Ensure penalty on exception

            return log_likelihood_val

        # --- Use lax.cond to determine final log likelihood ---
        kalman_inputs = (P_aug, R_aug, Omega_num, ys, H_obs, init_x_mean, init_P_cov)
        log_likelihood = lax.cond(
            solution_valid,
            calculate_likelihood,
            lambda args: _LOW_LOGLIKE_PENALTY, # Penalty if solve failed
            kalman_inputs
        )

        # <<< DEBUG: Print final factor >>>
        # jdebug.print("--- _numpyro_model reporting factor: log_likelihood={ll}, solution_valid={sv}",
        #              ll=log_likelihood, sv=solution_valid)

        numpyro.factor("log_likelihood", log_likelihood)

    # --- estimate method ---
    def estimate(self,
                 ys: ArrayLike,
                 H_obs: ArrayLike,
                 init_x_mean: ArrayLike,
                 init_P_cov: ArrayLike,
                 priors: Dict[str, Tuple[str, Union[List, Tuple]]],
                 mcmc_params: Dict[str, int],
                 rng_key: Optional[jax.random.PRNGKey] = None,
                 verbose_solver: bool = False,
                 # <<< USE init_param_values >>>
                 init_param_values: Optional[Dict[str, float]] = None
                 ) -> numpyro.infer.mcmc.MCMC:
        """ Estimates specified parameters using Bayesian MCMC (Numpyro NUTS). """
        if not NUMPYRO_AVAILABLE: raise RuntimeError("Numpyro not installed.")
        if not isinstance(priors, dict) or not priors: raise ValueError("`priors` empty.")
        if not isinstance(mcmc_params, dict) or 'num_warmup' not in mcmc_params or 'num_samples' not in mcmc_params: raise ValueError("`mcmc_params` missing keys.")

        num_warmup = mcmc_params['num_warmup']; num_samples = mcmc_params['num_samples']
        num_chains = mcmc_params.get('num_chains', 1); target_accept_prob = mcmc_params.get('target_accept_prob', 0.8)
        # Progress bar might be less useful with sequential chains / debug prints
        show_progress_bar = mcmc_params.get('progress_bar', True) # Allow disabling progress bar
        if verbose_solver: # If debugging solver, maybe disable progress bar too
             show_progress_bar = False

        if rng_key is None: rng_key = random.PRNGKey(0)

        estimated_params = list(priors.keys()); fixed_params = self.default_param_assignments.copy()
        for p_est in estimated_params:
            if p_est in fixed_params: del fixed_params[p_est]
            elif p_est not in self.all_param_names: raise ValueError(f"Prior for non-model param: {p_est}")
            # Ensure fixed params dict contains all non-estimated params, even if default is 0
            # This was slightly flawed before, let's fix
        for p_model in self.all_param_names:
            if p_model not in estimated_params and p_model not in fixed_params:
                # This case should not happen if defaults cover all params, but as safeguard:
                print(f"Warning: Model param '{p_model}' not in priors or defaults. Setting fixed value to 0.0.")
                fixed_params[p_model] = 0.0


        ys_jax=jnp.asarray(ys); H_obs_jax=jnp.asarray(H_obs); init_x_jax=jnp.asarray(init_x_mean); init_P_jax=jnp.asarray(init_P_cov)

        # --- Determine Initialization Strategy ---
        # Default to init_to_uniform (samples from prior support)
        init_strategy = init_to_uniform(radius=0.1) # Smaller radius might help? Default is 2

        if init_param_values:
             print("Note: Using provided init_param_values for MCMC initialization.")
             # Ensure the provided dict has values for ALL estimated parameters
             init_values_for_mcmc = {}
             missing_init = []
             for p_est in estimated_params:
                 if p_est in init_param_values:
                     init_values_for_mcmc[p_est] = init_param_values[p_est]
                 else:
                     missing_init.append(p_est)

             if not missing_init:
                 # Use init_to_value only if all estimated params are provided
                 print(" -> Using init_to_value strategy.")
                 init_strategy = init_to_value(values=init_values_for_mcmc)
             else:
                 # Fallback if some initial values are missing
                 print(f" -> WARNING: Not all estimated params found in init_param_values (missing: {missing_init}). Falling back to init_to_uniform.")
                 # Keep init_strategy as init_to_uniform
        else:
             print("Note: Using default MCMC initialization strategy (init_to_uniform).")
        # ---------------------------------------

        kernel = NUTS(self._numpyro_model,
                      init_strategy=init_strategy,
                      target_accept_prob=target_accept_prob)

        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=show_progress_bar, # Use configured setting
                    jit_model_args=True) # Try explicitly JITing model args


        # Run MCMC
        mcmc.run(rng_key, ys=ys_jax, H_obs=H_obs_jax, init_x_mean=init_x_jax, init_P_cov=init_P_jax, priors=priors, fixed_params=fixed_params, verbose_solver=verbose_solver)
        return mcmc


    # --- _prepare_params remains the same ---
    def _prepare_params(self, param_dict: Dict[str, float]) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        """ Validates and prepares parameter lists. """
        # Ensure all default parameters are considered before updating
        eval_params = self.default_param_assignments.copy()
        eval_params.update(param_dict) # Update with user-provided values

        # Check against *all* parameters required by the model instance
        missing_all = [p for p in self.all_param_names if p not in eval_params]
        if missing_all: raise ValueError(f"Missing required parameter values: {missing_all}")

        # Prepare args for stationary and full model functions based on their respective param lists
        stat_args = [eval_params[p] for p in self.param_names_stat_combined]
        all_args = [eval_params[p] for p in self.all_param_names]

        # Extract shock standard deviations specifically
        shock_std_devs = {}
        missing_sigmas = []
        for shock_name in self.aug_shocks_structure:
            sigma_param_name = f"sigma_{shock_name}"
            if sigma_param_name in eval_params:
                shock_std_devs[shock_name] = eval_params[sigma_param_name]
            else:
                # This indicates an internal inconsistency if all_param_names was checked correctly
                missing_sigmas.append(sigma_param_name)

        if missing_sigmas:
            # This should ideally not happen if the initial check passes
            raise ValueError(f"Internal Error: Missing values for shock std dev params expected to be in eval_params: {missing_sigmas}")

        return stat_args, all_args, shock_std_devs

# --- END OF FILE dynare_model_wrapper.py ---