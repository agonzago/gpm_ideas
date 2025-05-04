# --- START OF FILE dynare_model_wrapper.py ---

import os
import jax
import jax.numpy as jnp
import numpy as onp
from jax.typing import ArrayLike
from jax import random, lax
import jax.debug as jdebug
from typing import Dict, List, Tuple, Optional, Union, Any


# --- JAX Configuration ---
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# --- Numpyro Imports ---
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_uniform
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

# --- Import your custom modules ---
import Dynare_parser_sda_solver as dp

# Define machine epsilon based on JAX config
_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps if jax.config.jax_enable_x64 else jnp.finfo(jnp.float32).eps

class DynareModel:
    """ Wrapper class for Dynare models, using TFP for state-space operations. """
    # --- __init__ (Largely unchanged, just sets up attributes) ---
    def __init__(self, dynare_file_path: str):
        if not os.path.exists(dynare_file_path): 
            raise FileNotFoundError(f"Model file not found at: {dynare_file_path}")
        self.dynare_file_path = dynare_file_path
        self.dtype = _DEFAULT_DTYPE
        with open(self.dynare_file_path, 'r') as f: 
            model_def = f.read()
        try: 
            (self.func_A, self.func_B, self.func_C, self.func_D, self.ordered_stat_vars, self.stat_shocks, 
            self.param_names_stat_combined, self.default_param_assignments_stat, _, _) = dp.parse_lambdify_and_order_model(model_def, verbose=False)
            self.n_state_stat = len(self.ordered_stat_vars)
            self.n_shock_stat = len(self.stat_shocks)
        except Exception as e: 
            raise ValueError(f"Failed to parse stationary model from {dynare_file_path}") from e
        try: 
            self.trend_vars, self.trend_shocks = dp.extract_trend_declarations(model_def)
            trend_equations = dp.extract_trend_equations(model_def)
            self.obs_vars = dp.extract_observation_declarations(model_def)
            measurement_equations = dp.extract_measurement_equations(model_def)
            self.trend_stderr_params = dp.extract_trend_shock_stderrs(model_def)
        except Exception as e: 
            raise ValueError(f"Failed to parse trend/observation components from {dynare_file_path}") from e
        
        self.all_param_names = list(dict.fromkeys(self.param_names_stat_combined + list(self.trend_stderr_params.keys())).keys())
        self.default_param_assignments = self.default_param_assignments_stat.copy()
        self.default_param_assignments.update(self.trend_stderr_params)

        try: 
            (self.func_P_trends, self.func_Q_trends, self.ordered_trend_state_vars, self.contemp_trend_defs) = \
            dp.build_trend_matrices(trend_equations, self.trend_vars, self.trend_shocks, self.all_param_names, self.default_param_assignments, verbose=False)
            
            self.n_state_trend = len(self.ordered_trend_state_vars)
            self.n_shock_trend = len(self.trend_shocks)
            (self.func_Omega, self.ordered_obs_vars) = dp.build_observation_matrix(measurement_equations, self.obs_vars, self.ordered_stat_vars, self.ordered_trend_state_vars, self.contemp_trend_defs, self.all_param_names, self.default_param_assignments, verbose=False)
            self.n_obs = len(self.ordered_obs_vars)
        except Exception as e: 
            raise ValueError("Failed to build symbolic trend/observation matrices") from e
        self.aug_state_vars_structure = self.ordered_stat_vars + self.ordered_trend_state_vars; 
        self.aug_shocks_structure = self.stat_shocks + self.trend_shocks; 
        self.n_state_aug = self.n_state_stat + self.n_state_trend; 
        self.n_shock_aug = self.n_shock_stat + self.n_shock_trend

        if self.n_state_aug != len(self.aug_state_vars_structure): 
            raise RuntimeError("Internal Consistency Error: Augmented state dimension mismatch.")
        if self.n_shock_aug != len(self.aug_shocks_structure): 
            raise RuntimeError("Internal Consistency Error: Augmented shock dimension mismatch.")

    # --- _prepare_params (Unchanged) ---
    def _prepare_params(self, param_dict: Dict[str, float]) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        eval_params = self.default_param_assignments.copy(); eval_params.update(param_dict)
        missing_all = [p for p in self.all_param_names if p not in eval_params]; resolved = True
        if missing_all:
            for p_miss in missing_all:
                if p_miss in self.default_param_assignments: eval_params[p_miss] = self.default_param_assignments[p_miss]; print(f"Warning (_prepare_params): Used internal default for missing '{p_miss}'.")
                else: resolved = False
            if not resolved: truly_missing = [p for p in self.all_param_names if p not in eval_params]; raise ValueError(f"Missing required parameter values and no defaults found: {truly_missing}")
        stat_args = [eval_params[p] for p in self.param_names_stat_combined]
        all_args = [eval_params[p] for p in self.all_param_names]
        shock_std_devs = {}
        missing_sigmas = []
        for shock_name in self.aug_shocks_structure:
            sigma_param_name = f"sigma_{shock_name}"
            print(f"DEBUG (_prepare_params): Checking for shock '{shock_name}', parameter '{sigma_param_name}'") # Print check
            if sigma_param_name in eval_params:
                shock_std_devs[shock_name] = eval_params[sigma_param_name]
                print(f"  -> Found value: {eval_params[sigma_param_name]}") # Print found value
            else:
                print(f"  -> PARAMETER NOT FOUND in eval_params!") # Print if not found
                missing_sigmas.append(sigma_param_name)

        print(f"DEBUG (_prepare_params): final shock_std_devs keys: {list(shock_std_devs.keys())}") # Print final keys

        if missing_sigmas:
            raise ValueError(f"Internal Error: Missing values for shock std dev params: {missing_sigmas}.")

        return stat_args, all_args, shock_std_devs

    # --- solve  ---
    def solve(self, param_dict: Dict[str, float], max_iter_sda: int = 500) -> Dict[str, Any]:

        stat_args, all_args, shock_std_devs = self._prepare_params(param_dict)
        eval_params = {name: val for name, val in zip(self.all_param_names, all_args)}

        A_num_stat = jnp.asarray(self.func_A(*stat_args), dtype=self.dtype)
        B_num_stat = jnp.asarray(self.func_B(*stat_args), dtype=self.dtype)
        C_num_stat = jnp.asarray(self.func_C(*stat_args), dtype=self.dtype)
        D_num_stat = jnp.asarray(self.func_D(*stat_args), dtype=self.dtype)
        
        P_sol_stat, _, _, converged_flag = dp.solve_quadratic_matrix_equation_jax(A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=max_iter_sda)
        Q_sol_stat = dp.compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat, dtype=self.dtype)
        
        P_num_trend = jnp.asarray(self.func_P_trends(*all_args), dtype=self.dtype); 
        Q_num_trend = jnp.asarray(self.func_Q_trends(*all_args), dtype=self.dtype); 
        Omega_num = jnp.asarray(self.func_Omega(*all_args), dtype=self.dtype)
        
        p_stat_valid = converged_flag & jnp.all(jnp.isfinite(P_sol_stat)); 
        q_stat_valid = jnp.all(jnp.isfinite(Q_sol_stat)); 
        
        intermediate_matrices_valid = (jnp.all(jnp.isfinite(P_num_trend)) & jnp.all(jnp.isfinite(Q_num_trend)) & jnp.all(jnp.isfinite(Omega_num)))
        valid_intermediate_solution = p_stat_valid & q_stat_valid & intermediate_matrices_valid

        stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=self.dtype); 
        trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=self.dtype)
        
        R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr) if self.n_shock_stat > 0 else jnp.zeros((self.n_state_stat, 0), dtype=self.dtype)
        R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr) if self.n_shock_trend > 0 else jnp.zeros((self.n_state_trend, 0), dtype=self.dtype)
        
        P_aug_calc = jax.scipy.linalg.block_diag(P_sol_stat, P_num_trend)
        R_aug_calc = jnp.zeros((self.n_state_aug, self.n_shock_aug), dtype=self.dtype)
        
        if self.n_state_stat > 0 and self.n_shock_stat > 0: R_aug_calc = R_aug_calc.at[:self.n_state_stat, :self.n_shock_stat].set(R_sol_stat)
        if self.n_state_trend > 0 and self.n_shock_trend > 0: R_aug_calc = R_aug_calc.at[self.n_state_stat:, self.n_shock_stat:].set(R_num_trend)
        
        Omega_final_calc = Omega_num
        final_matrices_finite = (jnp.all(jnp.isfinite(P_aug_calc)) & jnp.all(jnp.isfinite(R_aug_calc)) & jnp.all(jnp.isfinite(Omega_final_calc)))
        solution_valid_flag = valid_intermediate_solution & final_matrices_finite

        nan_fill_aug = lambda shape: jnp.full(shape, jnp.nan, dtype=self.dtype)
        P_aug = jnp.where(solution_valid_flag, P_aug_calc, nan_fill_aug(P_aug_calc.shape)); 
        R_aug = jnp.where(solution_valid_flag, R_aug_calc, nan_fill_aug(R_aug_calc.shape)); 
        Omega_final = jnp.where(solution_valid_flag, Omega_final_calc, nan_fill_aug(Omega_final_calc.shape))

        return {'P_aug': P_aug, 
                'R_aug': R_aug, 
                'Omega': Omega_final, 
                'solution_valid': solution_valid_flag, 
                'param_values_used': eval_params, 
                'aug_state_vars': self.aug_state_vars_structure, 
                'aug_shocks': self.aug_shocks_structure, 
                'obs_vars_ordered': self.ordered_obs_vars}

    # --- get_irf
    def get_irf(self, param_dict: Dict[str, float], shock_name: str, horizon: int = 40) -> Dict[str, Any]:
        solution = self.solve(param_dict)
        if not solution['solution_valid']: 
            print("Warning get_irf: Model solution is invalid.") 
            return {}
        P_aug=solution['P_aug']; 
        R_aug=solution['R_aug']; 
        Omega_num=solution['Omega']; 
        aug_shocks=solution['aug_shocks']; 
        aug_state_vars=solution['aug_state_vars']; 
        obs_vars_ordered=solution['obs_vars_ordered']
        try: 
            shock_index = aug_shocks.index(shock_name)
        except ValueError: 
            raise ValueError(f"Shock '{shock_name}' not found: {aug_shocks}")
        irf_states_aug = dp.irf(P_aug, R_aug, shock_index=shock_index, horizon=horizon); 
        irf_observables_vals = dp.irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index, horizon=horizon)
        return {'state_irf': irf_states_aug, 
                'observable_irf': irf_observables_vals, 
                'state_names': aug_state_vars, 
                'observable_names': obs_vars_ordered, 
                'shock_name': shock_name, 
                'horizon': horizon}


def simulate_ssm_data(
    P: jnp.ndarray,
    R: jnp.ndarray,
    Omega: jnp.ndarray,
    T: int,
    state_init: jnp.ndarray = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate data from a linear state-space model:
      s_{t+1} = P @ s_t + R @ e_t
      y_t     = Omega @ s_t + eta_t (eta_t iid N(0, I))

    Returns (states, observables).
    """
    n_state = P.shape[0]
    n_shock = R.shape[1]
    n_obs = Omega.shape[0]
    if state_init is None:
        state_init = jnp.zeros(n_state)
    
    # Draw state and measurement noise
    key_one, key_two = jax.random.split(key)
    E = jax.random.normal(key_one, shape=(T, n_shock))  # shocks
    eta = jax.random.normal(key_two, shape=(T, n_obs))  # meas noise

    def step(s_prev, inputs):
        e_t, eta_t = inputs
        s_t = P @ s_prev + R @ e_t
        y_t = Omega @ s_t + eta_t
        return s_t, (s_t, y_t)
    
    _, (states, ys) = jax.lax.scan(
        step, state_init, (E, eta)
    )
    return states, ys


# --- END OF FILE dynare_model_wrapper.py ---