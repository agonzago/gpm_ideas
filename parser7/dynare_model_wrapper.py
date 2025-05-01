# --- START OF FILE dynare_model_wrapper.py ---

import os
import jax
import jax.numpy as jnp
import numpy as onp # For checking instance types, default values
from jax.typing import ArrayLike
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any

#
# --- Numpyro Imports ---
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    print("Warning: numpyro not found. Estimation functionality will be disabled.")

# --- Import your custom modules ---
# Assume the parser/solver is in dynare_parser_spd7.py
import Dynare_parser_sda_solver as dp
# Assume the Kalman filter is in Kalman_filter_jax.py
from Kalman_filter_jax import KalmanFilter, simulate_state_space

##--- JAX Configuration ---
# Ensure x64 is enabled if needed (can be set externally too)
try:
    jax.config.update("jax_enable_x64", True)
    print(f"dynare_model_wrapper: JAX float64 enabled: {jax.config.jax_enable_x64}")
except Exception as e:
    print(f"dynare_model_wrapper: Warning - Could not set jax_enable_x64: {e}")


class DynareModel:
    """
    A wrapper class to parse, solve, and analyze Dynare models using JAX.

    This class parses the model definition once upon initialization and provides
    methods to solve the model, compute IRFs, simulate data, and run Kalman
    filtering/smoothing for different parameter values without re-parsing.
    """

# --- START OF CORRECTED __init__ in dynare_model_wrapper.py ---

    def __init__(self, dynare_file_path: str):
        """
        Initializes the DynareModel by parsing the .dyn file.
        # ... (rest of docstring) ...
        """
        if not os.path.exists(dynare_file_path):
            raise FileNotFoundError(f"Model file not found at: {dynare_file_path}")

        self.dynare_file_path = dynare_file_path
        print(f"--- Initializing DynareModel from: {dynare_file_path} ---")

        with open(self.dynare_file_path, 'r') as f:
            model_def = f.read()

        # --- [1] PARSING AND LAMBDIFYING (Done Once) ---
        print("   Parsing stationary model components...")
        try:
            # The parser now returns the COMBINED stationary params list and assignments dict
            # Unpack directly into the correct attribute names
            (self.func_A, self.func_B, self.func_C, self.func_D,
             self.ordered_stat_vars, self.stat_shocks,
             self.param_names_stat_combined,        # <<< CORRECT attribute name
             self.default_param_assignments_stat,   # <<< CORRECT attribute name
             _, _) = dp.parse_lambdify_and_order_model(model_def)
            # No need to extract stat_stderr_params again here, parser does it.

        except Exception as e:
            print(f"Error parsing stationary model: {e}")
            # Consider adding traceback print here for easier debugging during init
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to parse stationary model from {dynare_file_path}") from e

        print("   Parsing trend/observation components...")
        try:
            self.trend_vars, self.trend_shocks = dp.extract_trend_declarations(model_def)
            trend_equations = dp.extract_trend_equations(model_def)
            self.obs_vars = dp.extract_observation_declarations(model_def)
            measurement_equations = dp.extract_measurement_equations(model_def)
            self.trend_stderr_params = dp.extract_trend_shock_stderrs(model_def) # Dict {sigma_NAME: value}
        except Exception as e:
            print(f"Error parsing trend/observation declarations: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to parse trend/observation components from {dynare_file_path}") from e

        # --- Combine parameters: Stationary (already combined) + Trend Sigmas ---
        # self.param_names_stat_combined is now correctly set from the parser

        # Combine all param names: Stationary (already combined) + Trend Sigmas
        self.all_param_names = list(dict.fromkeys(
            self.param_names_stat_combined + list(self.trend_stderr_params.keys())
        ).keys())

        # Combine default assignments: Start with combined stationary assignments + Update with Trend sigmas
        # Start with assignments returned by the parser (already includes stat sigmas)
        self.default_param_assignments = self.default_param_assignments_stat.copy()
        # Update/add trend sigma defaults
        self.default_param_assignments.update(self.trend_stderr_params)

        # --- Store names for combined shocks structure ---
        self.aug_shocks_structure = self.stat_shocks + self.trend_shocks

        print("   Building symbolic trend and observation matrices...")
        try:
            # Build trend matrices (lambdified functions)
            # Pass the FINAL combined parameters list and assignments for structure building
            (self.func_P_trends, self.func_Q_trends,
             self.ordered_trend_state_vars, self.contemp_trend_defs) = dp.build_trend_matrices(
                trend_equations, self.trend_vars, self.trend_shocks,
                self.all_param_names, self.default_param_assignments
            )

            # Build observation matrix (lambdified function)
            (self.func_Omega, self.ordered_obs_vars) = dp.build_observation_matrix(
                measurement_equations, self.obs_vars, self.ordered_stat_vars,
                self.ordered_trend_state_vars, self.contemp_trend_defs,
                self.all_param_names, self.default_param_assignments
             )
        except Exception as e:
            print(f"Error building symbolic trend/observation matrices: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError("Failed to build symbolic trend/observation matrices") from e

        # --- Store names for augmented state structure ---
        self.aug_state_vars_structure = self.ordered_stat_vars + self.ordered_trend_state_vars

        self.n_state_aug = len(self.aug_state_vars_structure)
        self.n_shock_aug = len(self.aug_shocks_structure)
        self.n_obs = len(self.ordered_obs_vars)

        print("--- DynareModel Initialization Complete ---")
        print(f"   Augmented State Variables ({self.n_state_aug}): {self.aug_state_vars_structure}")
        print(f"   Augmented Shocks ({self.n_shock_aug}): {self.aug_shocks_structure}")
        print(f"   Observable Variables ({self.n_obs}): {self.ordered_obs_vars}")
        # Updated print statements for clarity
        print(f"   Combined Stationary Param Names ({len(self.param_names_stat_combined)}): {self.param_names_stat_combined}")
        print(f"   All Parameters ({len(self.all_param_names)}): {self.all_param_names}")
        print(f"   Default Parameter Assignments (Combined): {self.default_param_assignments}")

# --- END OF CORRECTED __init__ in dynare_model_wrapper.py ---
# Add these methods inside the DynareModel class definition

    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Solves the model for the given parameters.

        Evaluates all system matrices (A, B, C, D, P_trends, Q_trends, Omega),
        solves the stationary component (P_stat, Q_stat), scales Q by standard
        deviations to get R, and builds the augmented system (P_aug, R_aug, Omega_num).

        Args:
            param_dict: Dictionary mapping parameter names to values.

        Returns:
            A dictionary containing the solved system matrices and variable names:
            {
                'P_aug': Augmented transition matrix [n_aug, n_aug],
                'R_aug': Augmented scaled shock matrix [n_aug, n_shock_aug],
                'Omega': Observation matrix [n_obs, n_aug],
                'P_sol_stat': Stationary solution P [n_stat, n_stat],
                'R_sol_stat': Stationary scaled shock matrix [n_stat, n_stat_shock],
                'P_num_trend': Trend transition matrix [n_trend, n_trend],
                'R_num_trend': Trend scaled shock matrix [n_trend, n_trend_shock],
                'aug_state_vars': List of augmented state variable names,
                'aug_shocks': List of augmented shock names,
                'obs_vars_ordered': List of observable variable names,
                'param_values_used': The full dict of parameters used for evaluation.
            }

        Raises:
            ValueError: If parameters are missing or the solver fails.
            RuntimeError: If the stationary solver fails or Q computation fails.
        """
        print(f"--- Solving model with provided parameters ---")
        # Prepare parameters using the internal helper
        stat_args, all_args, shock_std_devs = self._prepare_params(param_dict)
        # Create the full dictionary of evaluated parameters for returning
        eval_params = {name: val for name, val in zip(self.all_param_names, all_args)}

        # --- Evaluate Stationary Matrices ---
        print("   Evaluating stationary matrices (A, B, C, D)...")
        A_num_stat = jnp.asarray(self.func_A(*stat_args))
        B_num_stat = jnp.asarray(self.func_B(*stat_args))
        C_num_stat = jnp.asarray(self.func_C(*stat_args))
        D_num_stat = jnp.asarray(self.func_D(*stat_args))

        # --- Solve Stationary Model (SDA) ---
        print("   Solving stationary model (A P^2 + B P + C = 0)...")
        # Assuming dp.solve_quadratic_matrix_equation is imported correctly
        P_sol_stat, iter_count, res_ratio_stat = dp.solve_quadratic_matrix_equation(
            A_num_stat, B_num_stat, C_num_stat, tol=1e-12, verbose=False
        )
        if P_sol_stat is None or res_ratio_stat > 1e-6:
            print(f"   WARNING: Stationary solver issue. Iter: {iter_count}, Res Ratio: {res_ratio_stat:.2e}")
            if P_sol_stat is None:
                raise RuntimeError("Stationary SDA solver failed.")
            # Allow proceeding if residual is just high, but warn

        # --- Compute Stationary Q ---
        print("   Computing stationary Q matrix...")
        # Assuming dp.compute_Q is imported correctly
        Q_sol_stat = dp.compute_Q(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
        if Q_sol_stat is None:
            raise RuntimeError("Failed to compute Q_stationary.")

        # --- Evaluate Trend and Observation Matrices ---
        print("   Evaluating trend matrices (P_trends, Q_trends)...")
        P_num_trend = jnp.asarray(self.func_P_trends(*all_args))
        Q_num_trend = jnp.asarray(self.func_Q_trends(*all_args))

        # --- Scale Q matrices to get R matrices for Kalman Filter ---
        print("   Scaling Q matrices by std devs to get R matrices...")
        # Ensure std devs are JAX arrays with correct dtype
        common_dtype = P_sol_stat.dtype if P_sol_stat.size > 0 else P_num_trend.dtype
        stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=common_dtype)
        trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=common_dtype)

        # Handle cases where Q might be empty (e.g., no shocks)
        R_sol_stat = Q_sol_stat
        if Q_sol_stat.shape[1] > 0 : # If there are stationary shocks
            if Q_sol_stat.shape[1] != len(stat_std_devs_arr):
                raise ValueError(f"Mismatch between Q_stat columns ({Q_sol_stat.shape[1]}) and stationary shock std devs ({len(stat_std_devs_arr)})")
            R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr)

        R_num_trend = Q_num_trend
        if Q_num_trend.shape[1] > 0: # If there are trend shocks
            if Q_num_trend.shape[1] != len(trend_std_devs_arr):
                raise ValueError(f"Mismatch between Q_trend columns ({Q_num_trend.shape[1]}) and trend shock std devs ({len(trend_std_devs_arr)})")
            R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr)

        # --- Build Augmented System ---
        # Note: dp.build_augmented_state_space should accept R matrices
        print("   Building augmented state-space system (P_aug, R_aug, Omega)...")
        # Assuming dp.build_augmented_state_space is imported correctly
        (P_aug, R_aug, Omega_num,
         aug_state_vars, aug_shocks, obs_vars_ordered) = dp.build_augmented_state_space(
            P_sol_stat, R_sol_stat,     # Stationary solution P, R
            P_num_trend, R_num_trend,   # Trend solution P, R
            self.func_Omega,            # Observation matrix function
            self.ordered_stat_vars, self.ordered_trend_state_vars, self.obs_vars,
            self.stat_shocks, self.trend_shocks,
            all_args                    # Parameter values for evaluating Omega
        )

        # --- Consistency Checks ---
        if aug_state_vars != self.aug_state_vars_structure:
            print(f"   Warning: State variable order discrepancy. Expected: {self.aug_state_vars_structure}, Got: {aug_state_vars}")
        if aug_shocks != self.aug_shocks_structure:
            print(f"   Warning: Shock order discrepancy. Expected: {self.aug_shocks_structure}, Got: {aug_shocks}")
        if obs_vars_ordered != self.ordered_obs_vars:
            print(f"   Warning: Observable variable order discrepancy. Expected: {self.ordered_obs_vars}, Got: {obs_vars_ordered}")

        print("--- Model Solved Successfully ---")
        # Return results in a dictionary
        return {
            'P_aug': P_aug,
            'R_aug': R_aug,
            'Omega': Omega_num,
            'P_sol_stat': P_sol_stat,
            'R_sol_stat': R_sol_stat,
            'P_num_trend': P_num_trend,
            'R_num_trend': R_num_trend,
            'aug_state_vars': aug_state_vars,
            'aug_shocks': aug_shocks,
            'obs_vars_ordered': obs_vars_ordered,
            'param_values_used': eval_params
        }

    def get_irf(self, param_dict: Dict[str, float], shock_name: str, horizon: int = 40) -> Dict[str, Any]:
        """
        Computes Impulse Response Functions (IRFs) for a specific shock.

        Args:
            param_dict: Dictionary mapping parameter names to values.
            shock_name: The name of the shock to impulse (must be in the model).
            horizon: The number of periods for the IRF.

        Returns:
            A dictionary containing the IRFs:
            {
                'state_irf': Array of state responses [horizon, n_aug],
                'observable_irf': Array of observable responses [horizon, n_obs],
                'state_names': List of augmented state variable names,
                'observable_names': List of observable variable names,
                'shock_name': The name of the impulused shock,
                'horizon': The horizon length.
            }

        Raises:
            ValueError: If parameters are missing or shock_name is invalid.
            RuntimeError: If the model solver fails.
        """
        print(f"--- Computing IRFs for shock '{shock_name}' ---")
        # Solve the model first for the given parameters
        solution = self.solve(param_dict)
        P_aug = solution['P_aug']
        R_aug = solution['R_aug'] # R_aug has std dev scaling included
        Omega_num = solution['Omega']
        aug_shocks = solution['aug_shocks']
        aug_state_vars = solution['aug_state_vars']
        obs_vars_ordered = solution['obs_vars_ordered']

        # Find the index of the shock
        try:
            shock_index = aug_shocks.index(shock_name)
            print(f"   Using shock index: {shock_index}")
        except ValueError:
            raise ValueError(f"Shock '{shock_name}' not found in model shocks: {aug_shocks}")

        # Calculate IRFs using R_aug (already scaled)
        # Assuming dp.irf and dp.irf_observables are imported correctly
        # They should handle R_aug = Q_aug @ diag(sigmas) correctly
        irf_states_aug = dp.irf(P_aug, R_aug, shock_index=shock_index, horizon=horizon)
        irf_observables_vals = dp.irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index, horizon=horizon)

        print(f"--- IRF computation complete for '{shock_name}' ---")
        # Return results in a dictionary
        return {
            'state_irf': irf_states_aug,
            'observable_irf': irf_observables_vals,
            'state_names': aug_state_vars,
            'observable_names': obs_vars_ordered,
            'shock_name': shock_name,
            'horizon': horizon
        }

    def simulate(self,
                 param_dict: Dict[str, float],
                 H_obs: ArrayLike,
                 init_x_mean: ArrayLike,
                 init_P_cov: ArrayLike,
                 key: jax.random.PRNGKey,
                 num_steps: int) -> Dict[str, jax.Array]:
        """
        Simulates data from the state-space model.

        Args:
            param_dict: Dictionary mapping parameter names to values.
            H_obs: Observation noise covariance matrix [n_obs, n_obs].
            init_x_mean: Initial state mean vector [n_aug].
            init_P_cov: Initial state covariance matrix [n_aug, n_aug].
            key: JAX random key for simulation.
            num_steps: Number of time steps to simulate.

        Returns:
            A dictionary containing the simulated data:
            {
                'sim_states': Simulated states [num_steps, n_aug],
                'sim_observations': Simulated observations [num_steps, n_obs]
            }

        Raises:
            ValueError: If parameters are missing, or input shapes are incorrect.
            RuntimeError: If the model solver fails.
        """
        print(f"--- Simulating {num_steps} steps from the model ---")
        # Solve the model first for the given parameters
        solution = self.solve(param_dict)
        P_aug = solution['P_aug']
        R_aug = solution['R_aug']
        Omega_num = solution['Omega']
        n_aug = P_aug.shape[0]
        n_obs = Omega_num.shape[0]

        # --- Input Validation and Conversion ---
        H_obs_jax = jnp.asarray(H_obs)
        init_x_jax = jnp.asarray(init_x_mean)
        init_P_jax = jnp.asarray(init_P_cov)

        if H_obs_jax.shape != (n_obs, n_obs):
            raise ValueError(f"H_obs shape mismatch: expected ({n_obs},{n_obs}), got {H_obs_jax.shape}")
        if init_x_jax.shape != (n_aug,):
            raise ValueError(f"init_x_mean shape mismatch: expected ({n_aug},), got {init_x_jax.shape}")
        if init_P_jax.shape != (n_aug, n_aug):
            raise ValueError(f"init_P_cov shape mismatch: expected ({n_aug},{n_aug}), got {init_P_jax.shape}")

        print("   Calling simulate_state_space JITted function...")
        # Assuming simulate_state_space is imported from Kalman_filter_jax
        sim_states, sim_observations = simulate_state_space(
            P_aug, R_aug, Omega_num, H_obs_jax, init_x_jax, init_P_jax, key, num_steps
        )
        print("--- Simulation complete ---")
        # Return results in a dictionary
        return {
            'sim_states': sim_states,
            'sim_observations': sim_observations
        }

    def run_kalman(self,
                   param_dict: Dict[str, float],
                   ys: ArrayLike,
                   H_obs: ArrayLike,
                   init_x_mean: ArrayLike,
                   init_P_cov: ArrayLike,
                   smoother_key: Optional[jax.random.PRNGKey] = None,
                   num_sim_smoother_draws: int = 0
                   ) -> Dict[str, Any]:
        """
        Runs Kalman Filter, RTS Smoother, and optionally Simulation Smoother.

        Args:
            param_dict: Dictionary mapping parameter names to values.
            ys: Observed data array [T, n_obs]. Use NaNs for missing values.
            H_obs: Observation noise covariance matrix [n_obs, n_obs].
            init_x_mean: Initial state mean vector [n_aug] for the filter.
            init_P_cov: Initial state covariance matrix [n_aug, n_aug] for the filter.
            smoother_key: JAX random key, required if num_sim_smoother_draws > 0.
            num_sim_smoother_draws: Number of draws for the simulation smoother.
                                      If 0 (default), only filter and RTS are run.

        Returns:
            A dictionary containing filter/smoother results:
            {
                'filtered_states': Filtered state means [T, n_aug],
                'filtered_cov': Filtered state covariances [T, n_aug, n_aug],
                'rts_smoothed_states': RTS smoothed state means [T, n_aug],
                'rts_smoothed_cov': RTS smoothed state covariances [T, n_aug, n_aug],
                # Optional simulation smoother results (if num_sim_smoother_draws > 0):
                'sim_smoothed_mean': Mean of simulation smoother draws [T, n_aug],
                'sim_smoothed_median': Median of simulation smoother draws [T, n_aug],
                'sim_smoothed_draws': All draws [num_draws, T, n_aug]
            }

        Raises:
            ValueError: If parameters are missing, input shapes are incorrect,
                        or smoother_key is missing when needed.
            RuntimeError: If the model solver fails or Kalman operations fail.
        """
        print(f"--- Running Kalman Filter/Smoothers ---")
        # Solve the model first for the given parameters
        solution = self.solve(param_dict)
        P_aug = solution['P_aug']
        R_aug = solution['R_aug']
        Omega_num = solution['Omega']
        n_aug = P_aug.shape[0]
        n_obs = Omega_num.shape[0]

        # --- Input Validation and Conversion ---
        ys_jax = jnp.asarray(ys)
        H_obs_jax = jnp.asarray(H_obs)
        init_x_jax = jnp.asarray(init_x_mean)
        init_P_jax = jnp.asarray(init_P_cov)

        if ys_jax.ndim != 2 or ys_jax.shape[1] != n_obs:
            raise ValueError(f"ys shape mismatch: expected (T, {n_obs}), got {ys_jax.shape}")
        if H_obs_jax.shape != (n_obs, n_obs):
            raise ValueError(f"H_obs shape mismatch: expected ({n_obs},{n_obs}), got {H_obs_jax.shape}")
        if init_x_jax.shape != (n_aug,):
            raise ValueError(f"init_x_mean shape mismatch: expected ({n_aug},), got {init_x_jax.shape}")
        if init_P_jax.shape != (n_aug, n_aug):
            raise ValueError(f"init_P_cov shape mismatch: expected ({n_aug},{n_aug}), got {init_P_jax.shape}")
        if num_sim_smoother_draws > 0 and smoother_key is None:
            raise ValueError("smoother_key must be provided if num_sim_smoother_draws > 0")
        if num_sim_smoother_draws < 0:
            raise ValueError("num_sim_smoother_draws cannot be negative")

        print("   Instantiating KalmanFilter...")
        # Assuming KalmanFilter is imported from Kalman_filter_jax
        kf = KalmanFilter(
            T=P_aug, R=R_aug, C=Omega_num, H=H_obs_jax,
            init_x=init_x_jax, init_P=init_P_jax
        )

        results = {} # Initialize dictionary to store results

        # --- Run Filter ---
        print("   Running Filter...")
        filter_outs = kf.filter(ys_jax)
        # Unpack filter results (assuming order: x_pred, P_pred, x_filt, P_filt)
        # Standard Python indexing: 0=x_pred, 1=P_pred, 2=x_filt, 3=P_filt
        x_filt = filter_outs[2]
        P_filt = filter_outs[3]
        results['filtered_states'] = x_filt
        results['filtered_cov'] = P_filt
        print("   Filter finished.")

        # --- Run RTS Smoother ---
        print("   Running RTS Smoother...")
        x_smooth_rts, P_smooth_rts = kf.smooth(ys_jax)
        results['rts_smoothed_states'] = x_smooth_rts
        results['rts_smoothed_cov'] = P_smooth_rts
        print("   RTS Smoother finished.")

        # --- Run Simulation Smoother (Optional) ---
        if num_sim_smoother_draws > 0:
            print(f"   Running Simulation Smoother ({num_sim_smoother_draws} draws)...")
            sim_smoother_result = kf.simulation_smoother(
                ys_jax, smoother_key, num_draws=num_sim_smoother_draws
            )
            if num_sim_smoother_draws == 1:
                # Result is a single draw [T, n_state]
                single_draw = sim_smoother_result
                results['sim_smoothed_draws'] = single_draw[None, :, :] # Add batch dim
                results['sim_smoothed_mean'] = single_draw
                results['sim_smoothed_median'] = single_draw # Mean=Median=Draw for 1
            else:
                # Result is tuple: (mean, median, all_draws)
                # Standard Python indexing: 0=mean, 1=median, 2=all_draws
                mean_smooth = sim_smoother_result[0]
                median_smooth = sim_smoother_result[1]
                all_draws = sim_smoother_result[2]
                results['sim_smoothed_mean'] = mean_smooth
                results['sim_smoothed_median'] = median_smooth
                results['sim_smoothed_draws'] = all_draws
            print("   Simulation Smoother finished.")

        print("--- Kalman operations complete ---")
        # Return the dictionary containing all results
        return results

    def _get_numpyro_dist(self, dist_name: str, params: Union[List, Tuple]) -> numpyro.distributions.Distribution:
        """Helper to map string names to numpyro distributions."""
        if not NUMPYRO_AVAILABLE:
            raise RuntimeError("Numpyro is not installed. Cannot use estimation features.")

        dist_name_lower = dist_name.lower()
        if dist_name_lower == "normal":
            if len(params) != 2: raise ValueError("Normal prior needs 2 params (loc, scale)")
            return dist.Normal(loc=params[0], scale=params[1])
        elif dist_name_lower == "beta":
            if len(params) != 2: raise ValueError("Beta prior needs 2 params (concentration1, concentration0)")
            return dist.Beta(concentration1=params[0], concentration0=params[1])
        elif dist_name_lower == "gamma":
            if len(params) != 2: raise ValueError("Gamma prior needs 2 params (concentration, rate)")
            return dist.Gamma(concentration=params[0], rate=params[1])
        elif dist_name_lower == "inversegamma":
             # Corrected argument name and error message
             if len(params) != 2: raise ValueError("InverseGamma prior needs 2 params (concentration, rate)")
             # Numpyro uses concentration (alpha, shape) and rate (beta, inverse-scale)
             return dist.InverseGamma(concentration=params[0], rate=params[1]) # <<< CORRECTED LINE
        elif dist_name_lower == "uniform":
             if len(params) != 2: raise ValueError("Uniform prior needs 2 params (low, high)")
             return dist.Uniform(low=params[0], high=params[1])
        # Add other distributions as needed (e.g., LogNormal, HalfNormal, HalfCauchy for std devs)
        # Example: HalfCauchy is often preferred for standard deviations
        # elif dist_name_lower == "halfcauchy":
        #     if len(params) != 1: raise ValueError("HalfCauchy prior needs 1 param (scale)")
        #     return dist.HalfCauchy(scale=params[0])
        else:
            raise ValueError(f"Unsupported prior distribution name: {dist_name}")    

    def _numpyro_model(self, ys: jax.Array, H_obs: jax.Array,
                       init_x_mean: jax.Array, init_P_cov: jax.Array,
                       priors: Dict[str, Tuple[str, Union[List, Tuple]]],
                       fixed_params: Dict[str, float]):
        """Internal numpyro model function called by MCMC."""
        if not NUMPYRO_AVAILABLE:
            raise RuntimeError("Numpyro is not installed.")

        # 1. Sample parameters from priors
        sampled_params = {}
        for name, (dist_name, dist_params) in priors.items():
            prior_dist = self._get_numpyro_dist(dist_name, dist_params)
            # Sample on the CPU? Or let JAX handle device placement? Let JAX handle it.
            sampled_params[name] = numpyro.sample(name, prior_dist)

        # 2. Combine sampled and fixed parameters
        current_params = fixed_params.copy()
        current_params.update(sampled_params)
        # Ensure all necessary parameters are present (should be guaranteed by _prepare_params called by solve)

        # 3. Solve the DSGE model for the current parameter draw
        #    This involves numpy/scipy calls (SDA) but returns JAX arrays needed for KF
        try:
            solution = self.solve(current_params)
            P_aug = solution['P_aug']
            R_aug = solution['R_aug']
            Omega_num = solution['Omega']
        except (RuntimeError, ValueError, jax.errors.ConcretizationTypeError, onp.linalg.LinAlgError) as e:
            # Handle cases where parameters lead to solver failure or invalid matrices
            # Option 1: Return very low likelihood
            print(f"Warning: Model solution failed for parameters {sampled_params}. Assigning low likelihood. Error: {e}")
            log_likelihood = -jnp.inf # Or a very large negative number like -1e18
            # Add penalty factor to numpyro log probability
            numpyro.factor("log_likelihood", log_likelihood)
            return # Stop further execution for this draw

        # 4. Instantiate Kalman Filter
        #    Ensure init conditions have correct dtype matching P_aug etc.
        desired_dtype = P_aug.dtype
        init_x_mean_typed = jnp.asarray(init_x_mean, dtype=desired_dtype)
        init_P_cov_typed = jnp.asarray(init_P_cov, dtype=desired_dtype)
        H_obs_typed = jnp.asarray(H_obs, dtype=desired_dtype)
        ys_typed = jnp.asarray(ys, dtype=desired_dtype)


        kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega_num, H=H_obs_typed,
                          init_x=init_x_mean_typed, init_P=init_P_cov_typed)

        # 5. Calculate log-likelihood using the Kalman Filter instance
        try:
            log_likelihood = kf.log_likelihood(ys_typed)
             # Check for non-finite likelihood which indicates issues
            if not jnp.isfinite(log_likelihood):
                 print(f"Warning: Non-finite log-likelihood ({log_likelihood}) calculated. Assigning low likelihood.")
                 log_likelihood = -jnp.inf # Or large negative penalty
        except (ValueError, jax.errors.ConcretizationTypeError, onp.linalg.LinAlgError) as e: 
             # Catch potential errors during filtering/likelihood calc (e.g., singular matrices)
             print(f"Warning: Kalman filter/likelihood failed for parameters {sampled_params}. Assigning low likelihood. Error: {e}")
             log_likelihood = -jnp.inf # Or large negative penalty


        # 6. Add log-likelihood to numpyro's log probability accumulator
        #    Use numpyro.factor for externally computed log-likelihoods
        numpyro.factor("log_likelihood", log_likelihood)


    def estimate(self,
                 ys: ArrayLike,
                 H_obs: ArrayLike, # Observation noise covariance (assumed known for now)
                 init_x_mean: ArrayLike,
                 init_P_cov: ArrayLike,
                 priors: Dict[str, Tuple[str, Union[List, Tuple]]],
                 mcmc_params: Dict[str, int],
                 rng_key: Optional[jax.random.PRNGKey] = None
                 ) -> numpyro.infer.mcmc.MCMC:
        """
        Estimates specified parameters using Bayesian MCMC (Numpyro NUTS).

        Args:
            ys: Observed data array [T, n_obs]. Use NaNs for missing values.
            H_obs: Observation noise covariance matrix [n_obs, n_obs]. Assumed known.
                   (Could be estimated too by adding priors for its elements/std devs).
            init_x_mean: Initial state mean vector [n_aug] for the filter.
            init_P_cov: Initial state covariance matrix [n_aug, n_aug] for the filter.
                       (These are typically fixed or estimated with diffuse priors).
            priors: Dictionary defining priors for parameters to be estimated.
                    Format: {'param_name': ('DistributionName', [param1, param2,...]), ...}
                    Example: {'b1': ('Normal', [0.7, 0.1]), 'sigma_SHK_RS': ('InverseGamma', [2.0, 0.1])}
            mcmc_params: Dictionary with MCMC settings. Required keys:
                         'num_warmup': Number of warmup steps.
                         'num_samples': Number of sampling steps.
                         Optional: 'num_chains': Number of chains (default 1).
                                   'target_accept_prob': Target acceptance probability for NUTS.
            rng_key: JAX random key for reproducibility. If None, a default key is used.

        Returns:
            A numpyro MCMC object containing the results. Access samples via mcmc.get_samples().
            Print summary via mcmc.print_summary().

        Raises:
            RuntimeError: If numpyro is not installed.
            ValueError: If input parameters or prior specifications are invalid.
        """
        if not NUMPYRO_AVAILABLE:
            raise RuntimeError("Numpyro is not installed. Estimation functionality requires it.")

        print("--- Starting Bayesian Estimation using Numpyro NUTS ---")

        # --- Validate Inputs ---
        if not isinstance(priors, dict) or not priors:
            raise ValueError("`priors` dictionary cannot be empty.")
        if not isinstance(mcmc_params, dict) or 'num_warmup' not in mcmc_params or 'num_samples' not in mcmc_params:
             raise ValueError("`mcmc_params` must be a dict with 'num_warmup' and 'num_samples'.")

        num_warmup = mcmc_params['num_warmup']
        num_samples = mcmc_params['num_samples']
        num_chains = mcmc_params.get('num_chains', 1)
        target_accept_prob = mcmc_params.get('target_accept_prob', 0.8)

        if rng_key is None:
            print("Warning: No rng_key provided, using default PRNGKey(0).")
            rng_key = random.PRNGKey(0)

        # Identify fixed vs estimated parameters
        estimated_params = list(priors.keys())
        fixed_params = self.default_param_assignments.copy()
        missing_defaults = []
        for p_est in estimated_params:
            if p_est not in fixed_params:
                 # This means a parameter to be estimated wasn't even in the default list
                 # This could happen if it's not declared OR inferred from shocks.
                 # It MUST be in self.all_param_names though.
                 if p_est not in self.all_param_names:
                      raise ValueError(f"Parameter '{p_est}' specified in priors is not part of the model parameters: {self.all_param_names}")
                 else:
                      # Add a placeholder value to fixed_params; it will be overwritten by sampling.
                      # But warn the user this might be unexpected.
                      print(f"Warning: Parameter '{p_est}' in priors had no default value. Using 0.0 as placeholder before sampling.")
                      fixed_params[p_est] = 0.0
            else:
                # Remove estimated parameters from the fixed list
                del fixed_params[p_est]

        print(f"Estimating parameters: {estimated_params}")
        # print(f"Fixed parameters (using defaults): {fixed_params}") # Can be long

        # Ensure data and initial conditions are JAX arrays (convert once)
        # Dtype consistency is handled inside _numpyro_model based on solved system
        ys_jax = jnp.asarray(ys)
        H_obs_jax = jnp.asarray(H_obs)
        init_x_jax = jnp.asarray(init_x_mean)
        init_P_jax = jnp.asarray(init_P_cov)


        # Instantiate the NUTS kernel
        kernel = NUTS(self._numpyro_model, target_accept_prob=target_accept_prob)

        # Set up the MCMC sampler
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=True)

        # Run MCMC
        print(f"Running NUTS sampler ({num_chains} chain(s), {num_warmup} warmup, {num_samples} samples)...")
        mcmc.run(rng_key, ys=ys_jax, H_obs=H_obs_jax, init_x_mean=init_x_jax, init_P_cov=init_P_jax,
                 priors=priors, fixed_params=fixed_params)
        print("--- MCMC Sampling Complete ---")

        # Print summary
        mcmc.print_summary()

        return mcmc

    # def _prepare_params(self, param_dict: Dict[str, float]) -> Tuple[List[float], List[float], Dict[str, float]]:
    #     """
    #     Validates and prepares parameter lists for evaluation based on the
    #     structure discovered during __init__.

    #     Args:
    #         param_dict: Dictionary mapping parameter names to values provided by the user.

    #     Returns:
    #         A tuple containing:
    #             - stat_args: Ordered list of float values for stationary functions.
    #                          Uses self.param_names_stat_combined order.
    #             - all_args: Ordered list of float values for all functions (stationary + trend + obs).
    #                         Uses self.all_param_names order.
    #             - shock_std_devs: Dictionary mapping ALL augmented shock names to their float std dev values.

    #     Raises:
    #         ValueError: If any required parameters (including sigma_ parameters
    #                     for all shocks) are missing after merging with defaults.
    #     """
    #     eval_params = self.default_param_assignments.copy()
    #     eval_params.update(param_dict) # User values overwrite defaults

    #     # Check all parameters needed for ANY function are present
    #     missing_all = [p for p in self.all_param_names if p not in eval_params]
    #     if missing_all:
    #         raise ValueError(f"Missing required parameter values for model functions: {missing_all}")

    #     # Create ordered list for stationary functions (using combined list)
    #     stat_args = []
    #     for p_name_stat in self.param_names_stat_combined:
    #         stat_args.append(float(eval_params[p_name_stat]))

    #     # Create ordered list for all functions
    #     all_args = [float(eval_params[p_name]) for p_name in self.all_param_names]

    #     # Extract shock standard deviations for ALL augmented shocks
    #     shock_std_devs = {}
    #     missing_sigmas = []
    #     for shock_name in self.aug_shocks_structure:
    #         sigma_param_name = f"sigma_{shock_name}"
    #         if sigma_param_name in eval_params:
    #             shock_std_devs[shock_name] = float(eval_params[sigma_param_name])
    #         else:
    #             # This check is important: if sigma was needed (in all_param_names) but
    #             # is still missing, it means no default and no user value provided.
    #              missing_sigmas.append(sigma_param_name)

    #     if missing_sigmas:
    #         raise ValueError(
    #             f"Missing values for required shock standard deviation parameters: {missing_sigmas}. "
    #             f"Ensure 'sigma_SHOCKNAME' is defined (e.g., in 'shocks;' or 'trend_shocks;' stderr blocks) "
    #             f"or provided explicitly. All shocks: {self.aug_shocks_structure}"
    #         )

    #     return stat_args, all_args, shock_std_devs

    def _prepare_params(self, param_dict: Dict[str, float]) -> Tuple[List[Any], List[Any], Dict[str, Any]]: # Return type changed to Any
        """
        Validates and prepares parameter lists for evaluation based on the
        structure discovered during __init__. Accepts JAX tracers during estimation.
        # ... (rest of docstring) ...
        """
        eval_params = self.default_param_assignments.copy()
        eval_params.update(param_dict) # User values (potentially tracers) overwrite defaults

        # Check all parameters needed for ANY function are present
        missing_all = [p for p in self.all_param_names if p not in eval_params]
        if missing_all:
            raise ValueError(f"Missing required parameter values for model functions: {missing_all}")

        # Create ordered list for stationary functions (using combined list)
        # NO float() conversion here
        stat_args = []
        for p_name_stat in self.param_names_stat_combined:
            stat_args.append(eval_params[p_name_stat]) # Keep as JAX type/tracer

        # Create ordered list for all functions
        # NO float() conversion here
        all_args = [eval_params[p_name] for p_name in self.all_param_names] # Keep as JAX type/tracer

        # Extract shock standard deviations for ALL augmented shocks
        # NO float() conversion here
        shock_std_devs = {}
        missing_sigmas = []
        for shock_name in self.aug_shocks_structure:
            sigma_param_name = f"sigma_{shock_name}"
            if sigma_param_name in eval_params:
                shock_std_devs[shock_name] = eval_params[sigma_param_name] # Keep as JAX type/tracer
            else:
                 missing_sigmas.append(sigma_param_name)

        if missing_sigmas:
            # This error should ideally be caught earlier if defaults are set correctly
             raise ValueError(
                 f"Missing values for required shock standard deviation parameters: {missing_sigmas}. "
                 f"All shocks: {self.aug_shocks_structure}"
             )

        return stat_args, all_args, shock_std_devs

# --- END OF FILE dynare_model_wrapper.py ---