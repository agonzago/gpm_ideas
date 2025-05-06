# --- START OF MODIFIED run_estimation.py ---
import os
import time
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any
import re

# --- Force CPU Execution (Optional) ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update("jax_platforms", "cpu")
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

# --- JAX/Numpyro Setup --- (Keep as is)
# ...
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
# ... (Import numpyro, KalmanFilter) ...

# --- Import from Parser --- (Keep as is)
from dynare_parser_sda_solver_jax import (
    # ... all necessary imports ...
    parse_and_order_stationary_model_symbolic,
    generate_matrix_lambda_functions,
    generate_trend_lambda_functions,
    generate_observation_lambda_functions,
    solve_quadratic_matrix_equation_jax,
    compute_Q_jax,
    construct_initial_state,
    simulate_ssm_data,
    plot_simulation_with_trends_matched,
    plot_irfs,
    extract_trend_declarations, # Needed below
    extract_trend_shock_stderrs # Needed below
)


class DynareModelWithSympyVJPs:
    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose
        self._parsed = False
        self._symbolic_data = {}

        with open(self.mod_file_path, 'r') as f:
            self.model_def_content = f.read()

        self._parse_and_generate_lambdas()

    def _parse_and_generate_lambdas(self):
        # --- This method remains largely the same ---
        # It parses structure and calls generate_matrix_lambda_functions
        # to populate self._symbolic_data, self.all_param_names,
        # self.stat_var_perm_indices, self.stat_eq_perm_indices, etc.
        if self._parsed: return
        if self.verbose: print("--- DynareModel: Parsing and Generating Symbolic Lambdas ---")
        # 1. Stationary Part
        self.stationary_structure = parse_and_order_stationary_model_symbolic(self.model_def_content, verbose=self.verbose)
        self.all_param_names = self.stationary_structure['param_names_all']
        self.default_param_assignments = self.stationary_structure['param_assignments_default']
        self._symbolic_data['stationary'] = generate_matrix_lambda_functions(
            equations_str=self.stationary_structure['equations_processed'],
            var_names_ordered=self.stationary_structure['var_names_initial_order'],
            shock_names_ordered=self.stationary_structure['shock_names'],
            all_param_names_ordered=self.all_param_names, model_type="stationary", verbose=self.verbose)
        self.stat_var_perm_indices = jnp.array(self.stationary_structure['var_permutation_indices'])
        self.stat_eq_perm_indices = jnp.array(self.stationary_structure['eq_permutation_indices'])
        self.ordered_stat_vars = self.stationary_structure['ordered_vars_final']
        self.stat_shocks = self.stationary_structure['shock_names']
        # 2. Trend Part
        _trend_stderr_p = extract_trend_shock_stderrs(self.model_def_content)
        self.default_param_assignments.update(_trend_stderr_p)
        for p_name in _trend_stderr_p.keys():
            if p_name not in self.all_param_names: self.all_param_names.append(p_name)
        self._symbolic_data['trend'] = generate_trend_lambda_functions(self.model_def_content, self.all_param_names, verbose=self.verbose)
        self.ordered_trend_state_vars = self._symbolic_data['trend'].get('state_trend_vars', [])
        self.contemp_trend_defs = self._symbolic_data['trend'].get('contemporaneous_trend_defs', {})
        _, self.trend_shocks = extract_trend_declarations(self.model_def_content)
        # 3. Observation Part
        self._symbolic_data['observation'] = generate_observation_lambda_functions(
            self.model_def_content, self.all_param_names, self.ordered_stat_vars,
            self.ordered_trend_state_vars, self.contemp_trend_defs, verbose=self.verbose)
        self.ordered_obs_vars = self._symbolic_data['observation'].get('ordered_obs_vars', [])
        # Store dimensions
        self.n_stat, self.n_s_shock = len(self.ordered_stat_vars), len(self.stat_shocks)
        self.n_trend, self.n_t_shock = len(self.ordered_trend_state_vars), len(self.trend_shocks)
        self.n_obs = len(self.ordered_obs_vars)
        self.n_aug, self.n_aug_shock = self.n_stat + self.n_trend, self.n_s_shock + self.n_t_shock
        self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars
        self.aug_shocks = self.stat_shocks + self.trend_shocks
        self._parsed = True
        if self.verbose: print("--- DynareModel: Symbolic Setup Complete ---")


    # --- Static VJP methods for Stationary Matrices ---
    @staticmethod
    def _build_stationary_matrices_vjp_fwd_static(
        params_tuple_jax, # Differentiable inputs first
        lambdas_stat,     # Non-differentiable data
        stat_eq_perm_indices,
        stat_var_perm_indices
        ):
        # --- Forward pass logic (same as before, but NO self) ---
        num_eq = len(lambdas_stat['A']['elements'])
        num_vars = len(lambdas_stat['A']['elements'][0]) if num_eq > 0 else 0
        num_shocks = len(lambdas_stat['D']['elements'][0]) if num_eq > 0 and lambdas_stat['D']['elements'] and lambdas_stat['D']['elements'][0] else 0
        A_unord_list = [[lambdas_stat['A']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]
        B_unord_list = [[lambdas_stat['B']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]
        C_unord_list = [[lambdas_stat['C']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]
        D_unord_list = [[lambdas_stat['D']['elements'][i][j](*params_tuple_jax) for j in range(num_shocks)] for i in range(num_eq)] if num_shocks > 0 else [[] for _ in range(num_eq)]
        A_unord = jnp.array(A_unord_list, dtype=_DEFAULT_DTYPE).reshape(num_eq, num_vars) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
        B_unord = jnp.array(B_unord_list, dtype=_DEFAULT_DTYPE).reshape(num_eq, num_vars) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
        C_unord = jnp.array(C_unord_list, dtype=_DEFAULT_DTYPE).reshape(num_eq, num_vars) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
        D_unord = jnp.array(D_unord_list, dtype=_DEFAULT_DTYPE).reshape(num_eq, num_shocks) if num_shocks > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
        A_ord = A_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
        B_ord = B_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
        C_ord = C_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
        D_ord = D_unord[jnp.ix_(stat_eq_perm_indices, jnp.arange(D_unord.shape[1]))] if num_shocks > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
        # --- Residuals needed for backward pass ---
        residuals = (params_tuple_jax, lambdas_stat, A_unord, B_unord, C_unord, D_unord, stat_eq_perm_indices, stat_var_perm_indices)
        return (A_ord, B_ord, C_ord, D_ord), residuals

    @staticmethod
    def _build_stationary_matrices_vjp_bwd_static(residuals, grads_output_ordered_tuple):
        # --- Backward pass logic (same as before, but NO self) ---
        params_tuple_jax, lambdas_stat, A_unord, B_unord, C_unord, D_unord, stat_eq_perm_indices, stat_var_perm_indices = residuals
        grad_A_ord, grad_B_ord, grad_C_ord, grad_D_ord = grads_output_ordered_tuple
        num_params = len(params_tuple_jax)
        grad_A_unord, grad_B_unord, grad_C_unord, grad_D_unord = jnp.zeros_like(A_unord), jnp.zeros_like(B_unord), jnp.zeros_like(C_unord), jnp.zeros_like(D_unord)
        # --- Inverse permutation ---
        if A_unord.size > 0:
            for r_new in range(grad_A_ord.shape[0]):
                r_old_eq = stat_eq_perm_indices[r_new]
                for c_new_var in range(grad_A_ord.shape[1]):
                    c_old_var = stat_var_perm_indices[c_new_var]
                    grad_A_unord = grad_A_unord.at[r_old_eq, c_old_var].add(grad_A_ord[r_new, c_new_var])
                    grad_B_unord = grad_B_unord.at[r_old_eq, c_old_var].add(grad_B_ord[r_new, c_new_var])
                    grad_C_unord = grad_C_unord.at[r_old_eq, c_old_var].add(grad_C_ord[r_new, c_new_var])
        if D_unord.size > 0:
            for r_new in range(grad_D_ord.shape[0]):
                r_old_eq = stat_eq_perm_indices[r_new]
                for c_new_shk in range(grad_D_ord.shape[1]):
                     grad_D_unord = grad_D_unord.at[r_old_eq, c_new_shk].add(grad_D_ord[r_new, c_new_shk])
        # --- Gradient accumulation ---
        param_grads_list = [jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in range(num_params)]
        for p_idx in range(num_params):
            current_param_grad = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
            if A_unord.size > 0:
                for i,j in jnp.ndindex(A_unord.shape): current_param_grad += grad_A_unord[i,j] * lambdas_stat['A']['grads'][i][j][p_idx](*params_tuple_jax)
            if B_unord.size > 0:
                for i,j in jnp.ndindex(B_unord.shape): current_param_grad += grad_B_unord[i,j] * lambdas_stat['B']['grads'][i][j][p_idx](*params_tuple_jax)
            if C_unord.size > 0:
                for i,j in jnp.ndindex(C_unord.shape): current_param_grad += grad_C_unord[i,j] * lambdas_stat['C']['grads'][i][j][p_idx](*params_tuple_jax)
            if D_unord.size > 0:
                for i,j in jnp.ndindex(D_unord.shape): current_param_grad += grad_D_unord[i,j] * lambdas_stat['D']['grads'][i][j][p_idx](*params_tuple_jax)
            param_grads_list[p_idx] = current_param_grad
        # --- Return grads for differentiable inputs ONLY (params_tuple_jax) ---
        # The other inputs to _vjp_fwd_static (lambdas, indices) are non-differentiable.
        return (tuple(param_grads_list), None, None, None) # Match inputs of _build_stationary_matrices_symbolic_static

    # --- Static VJP Primal Function ---
    # Takes differentiable args first, then static data args
    @staticmethod
    @jax.custom_vjp
    def _build_stationary_matrices_symbolic_static(
        params_tuple_jax, # Differentiable
        lambdas_stat, stat_eq_perm_indices, stat_var_perm_indices # Non-differentiable data
        ):
        # This function just defines the primal computation signature for VJP
        (A_ord, B_ord, C_ord, D_ord), _ = DynareModelWithSympyVJPs._build_stationary_matrices_vjp_fwd_static(
             params_tuple_jax, lambdas_stat, stat_eq_perm_indices, stat_var_perm_indices
             )
        return A_ord, B_ord, C_ord, D_ord

    # --- Associate VJP rules with the static function ---
    _build_stationary_matrices_symbolic_static.defvjp(
        _build_stationary_matrices_vjp_fwd_static,
        _build_stationary_matrices_vjp_bwd_static
    )

    # --- Repeat static VJP pattern for Trend ---
    @staticmethod
    def _build_trend_matrices_vjp_fwd_static(params_tuple_jax, lambdas_trend_data):
        if not lambdas_trend_data or not lambdas_trend_data.get('lambda_matrices') or not lambdas_trend_data['lambda_matrices'].get('P_trends'): # Check if 'P_trends' exists
            P_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
            Q_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
            residuals = (params_tuple_jax, lambdas_trend_data, P_trends, Q_trends) # Need P/Q shape info even if empty
            return (P_trends, Q_trends), residuals

        lambdas_trend = lambdas_trend_data['lambda_matrices']
        num_rows_p = len(lambdas_trend['P_trends']['elements'])
        num_cols_p = len(lambdas_trend['P_trends']['elements'][0]) if num_rows_p > 0 else 0
        num_cols_q = 0
        if 'Q_trends' in lambdas_trend and lambdas_trend['Q_trends']['elements'] and lambdas_trend['Q_trends']['elements'][0]:
             num_cols_q = len(lambdas_trend['Q_trends']['elements'][0])

        P_list = [[lambdas_trend['P_trends']['elements'][i][j](*params_tuple_jax) for j in range(num_cols_p)] for i in range(num_rows_p)]
        Q_list = [[lambdas_trend['Q_trends']['elements'][i][j](*params_tuple_jax) for j in range(num_cols_q)] for i in range(num_rows_p)] if num_cols_q > 0 else [[] for _ in range(num_rows_p)]
        P_trends = jnp.array(P_list, dtype=_DEFAULT_DTYPE).reshape(num_rows_p, num_cols_p) if num_rows_p * num_cols_p > 0 else jnp.zeros((num_rows_p,0), dtype=_DEFAULT_DTYPE)
        Q_trends = jnp.array(Q_list, dtype=_DEFAULT_DTYPE).reshape(num_rows_p, num_cols_q) if num_rows_p * num_cols_q > 0 else jnp.zeros((num_rows_p,0), dtype=_DEFAULT_DTYPE)
        residuals = (params_tuple_jax, lambdas_trend_data, P_trends, Q_trends)
        return (P_trends, Q_trends), residuals

    @staticmethod
    def _build_trend_matrices_vjp_bwd_static(residuals, grads_output_tuple):
        params_tuple_jax, lambdas_trend_data, P_trends, Q_trends = residuals
        grad_P, grad_Q = grads_output_tuple
        if not lambdas_trend_data or not lambdas_trend_data.get('lambda_matrices') or not lambdas_trend_data['lambda_matrices'].get('P_trends'):
            return (tuple(jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in params_tuple_jax), None) # Grad for params, None for data

        lambdas_trend = lambdas_trend_data['lambda_matrices']
        num_params = len(params_tuple_jax)
        param_grads_list = [jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in range(num_params)]
        for p_idx in range(num_params):
            current_param_grad = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
            if P_trends.size > 0:
                for i,j in jnp.ndindex(P_trends.shape): current_param_grad += grad_P[i,j] * lambdas_trend['P_trends']['grads'][i][j][p_idx](*params_tuple_jax)
            if Q_trends.size > 0 and 'Q_trends' in lambdas_trend: # Check if Q exists
                for i,j in jnp.ndindex(Q_trends.shape): current_param_grad += grad_Q[i,j] * lambdas_trend['Q_trends']['grads'][i][j][p_idx](*params_tuple_jax)
            param_grads_list[p_idx] = current_param_grad
        return (tuple(param_grads_list), None) # Grad for params, None for data

    @staticmethod
    @jax.custom_vjp
    def _build_trend_matrices_symbolic_static(params_tuple_jax, lambdas_trend_data):
        (P, Q), _ = DynareModelWithSympyVJPs._build_trend_matrices_vjp_fwd_static(params_tuple_jax, lambdas_trend_data)
        return P, Q
    _build_trend_matrices_symbolic_static.defvjp(_build_trend_matrices_vjp_fwd_static, _build_trend_matrices_vjp_bwd_static)

    # --- Repeat static VJP pattern for Observation ---
    @staticmethod
    def _build_observation_matrix_vjp_fwd_static(params_tuple_jax, lambdas_obs_data, n_obs, n_aug): # Pass shapes needed if empty
        if not lambdas_obs_data or not lambdas_obs_data.get('lambda_matrices') or not lambdas_obs_data['lambda_matrices'].get('Omega'):
             Omega = jnp.empty((n_obs, n_aug), dtype=_DEFAULT_DTYPE) # Use passed shapes
             residuals = (params_tuple_jax, lambdas_obs_data, Omega)
             return Omega, residuals

        lambdas_omega = lambdas_obs_data['lambda_matrices']['Omega']
        num_rows = len(lambdas_omega['elements'])
        num_cols = len(lambdas_omega['elements'][0]) if num_rows > 0 else 0
        Omega_list = [[lambdas_omega['elements'][i][j](*params_tuple_jax) for j in range(num_cols)] for i in range(num_rows)]
        Omega = jnp.array(Omega_list, dtype=_DEFAULT_DTYPE).reshape(num_rows, num_cols) if num_rows * num_cols > 0 else jnp.zeros((num_rows,0), dtype=_DEFAULT_DTYPE)
        residuals = (params_tuple_jax, lambdas_obs_data, Omega)
        return Omega, residuals

    @staticmethod
    def _build_observation_matrix_vjp_bwd_static(residuals, grad_Omega):
        params_tuple_jax, lambdas_obs_data, Omega = residuals
        if not lambdas_obs_data or not lambdas_obs_data.get('lambda_matrices') or not lambdas_obs_data['lambda_matrices'].get('Omega'):
            return (tuple(jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in params_tuple_jax), None, None) # Grad params, None data, None shapes

        lambdas_omega = lambdas_obs_data['lambda_matrices']['Omega']
        num_params = len(params_tuple_jax)
        param_grads_list = [jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in range(num_params)]
        for p_idx in range(num_params):
            current_param_grad = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
            if Omega.size > 0:
                for i,j in jnp.ndindex(Omega.shape): current_param_grad += grad_Omega[i,j] * lambdas_omega['grads'][i][j][p_idx](*params_tuple_jax)
            param_grads_list[p_idx] = current_param_grad
        return (tuple(param_grads_list), None, None, None) # Match inputs: params, lambdas, n_obs, n_aug

    @staticmethod
    @jax.custom_vjp
    def _build_observation_matrix_symbolic_static(params_tuple_jax, lambdas_obs_data, n_obs, n_aug): # Add shapes
        Omega, _ = DynareModelWithSympyVJPs._build_observation_matrix_vjp_fwd_static(params_tuple_jax, lambdas_obs_data, n_obs, n_aug)
        return Omega
    _build_observation_matrix_symbolic_static.defvjp(_build_observation_matrix_vjp_fwd_static, _build_observation_matrix_vjp_bwd_static)


    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        if not self._parsed: self._parse_and_generate_lambdas()
        params_tuple_jax = tuple(jnp.asarray(param_dict[p_name], dtype=_DEFAULT_DTYPE) for p_name in self.all_param_names)

        results = {"solution_valid": jnp.array(False)}
        try: # Wrap entire solve in try-except
            # Call STATIC VJP functions, passing data from self
            A_num_stat, B_num_stat, C_num_stat, D_num_stat = DynareModelWithSympyVJPs._build_stationary_matrices_symbolic_static(
                params_tuple_jax,
                self._symbolic_data['stationary']['lambda_matrices'],
                self.stat_eq_perm_indices,
                self.stat_var_perm_indices
            )

            P_sol_stat, _, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500
            )
            valid_stat_solve = converged_stat & jnp.all(jnp.isfinite(P_sol_stat))

            Q_sol_stat = jnp.where(
                 valid_stat_solve,
                 compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat),
                 jnp.full_like(D_num_stat, jnp.nan)
            )
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat))

            P_num_trend, Q_num_trend = DynareModelWithSympyVJPs._build_trend_matrices_symbolic_static(
                params_tuple_jax,
                self._symbolic_data['trend'] # Pass the whole trend data dict
            )

            Omega_num = DynareModelWithSympyVJPs._build_observation_matrix_symbolic_static(
                params_tuple_jax,
                self._symbolic_data['observation'], # Pass the whole obs data dict
                self.n_obs, # Pass shape info
                self.n_aug  # Pass shape info
            )

            # --- Build R Matrices & Augmented System (Numerical part) ---
            shock_std_devs = {}
            for shock_name in self.aug_shocks:
                sigma_param_name = f"sigma_{shock_name}"
                param_idx = self.all_param_names.index(sigma_param_name)
                std_dev = params_tuple_jax[param_idx]
                shock_std_devs[shock_name] = jnp.maximum(jnp.abs(std_dev), 1e-9)

            stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE) if self.stat_shocks else jnp.array([], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE) if self.trend_shocks else jnp.array([], dtype=_DEFAULT_DTYPE)

            # Ensure Q matrices have correct dimensions before multiplying
            R_sol_stat = jnp.zeros((self.n_stat, self.n_s_shock), dtype=_DEFAULT_DTYPE)
            if self.n_s_shock > 0 and Q_sol_stat.shape == (self.n_stat, self.n_s_shock):
                R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr)

            R_num_trend = jnp.zeros((self.n_trend, self.n_t_shock), dtype=_DEFAULT_DTYPE)
            if self.n_t_shock > 0 and Q_num_trend.shape == (self.n_trend, self.n_t_shock):
                 R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr)

            # Handle potentially empty blocks in block_diag
            blocks = []
            if P_sol_stat.size > 0: blocks.append(P_sol_stat)
            if P_num_trend.size > 0: blocks.append(P_num_trend)
            if not blocks: P_aug = jnp.zeros((self.n_aug, self.n_aug), dtype=_DEFAULT_DTYPE)
            elif len(blocks) == 1: P_aug = blocks[0] # If only one block is non-empty
            else: P_aug = jax.scipy.linalg.block_diag(*blocks)
            # Ensure P_aug has correct final shape if one block was empty
            if P_aug.shape != (self.n_aug, self.n_aug):
                 # This might happen if e.g. n_stat > 0, n_trend = 0
                 temp_P_aug = jnp.zeros((self.n_aug, self.n_aug), dtype=_DEFAULT_DTYPE)
                 if P_sol_stat.size > 0: temp_P_aug = temp_P_aug.at[:self.n_stat, :self.n_stat].set(P_sol_stat)
                 # No need to set P_num_trend if it was empty
                 P_aug = temp_P_aug

            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=_DEFAULT_DTYPE)
            if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock):
                 R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
            if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock):
                 R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)

            all_finite = (jnp.all(jnp.isfinite(P_aug)) & jnp.all(jnp.isfinite(R_aug)) & jnp.all(jnp.isfinite(Omega_num)))
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite

            results.update({
                "P_aug": P_aug, "R_aug": R_aug, "Omega": Omega_num,
                "solution_valid": solution_valid_final,
                "ordered_trend_state_vars": self.ordered_trend_state_vars,
                "contemp_trend_defs": self.contemp_trend_defs,
                "ordered_obs_vars": self.ordered_obs_vars,
                "aug_state_vars": self.aug_state_vars,
                "aug_shocks": self.aug_shocks,
                "n_aug": self.n_aug, "n_aug_shock": self.n_aug_shock, "n_obs": self.n_obs
            })

        except Exception as e: # Catch errors during solve (incl. VJP calls, SDA, etc.)
             if self.verbose: print(f"Exception during model solve (VJP): {type(e).__name__}: {e}")
             # import traceback; traceback.print_exc() # Optional detailed traceback
             results["solution_valid"] = jnp.array(False)
             # Ensure matrices are None or NaN filled if solve fails for stability
             results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE) if hasattr(self, 'n_aug') else None
             results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE) if hasattr(self, 'n_aug') and hasattr(self, 'n_aug_shock') else None
             results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE) if hasattr(self, 'n_obs') and hasattr(self, 'n_aug') else None

        return results

    # log_likelihood method remains the same, it calls self.solve()

    def log_likelihood(self, param_dict: Dict[str, float], ys: jax.Array,
                       H_obs: jax.Array, init_x_mean: jax.Array, init_P_cov: jax.Array) -> jax.Array:
        if not KALMAN_FILTER_JAX_AVAILABLE: raise RuntimeError("Custom KalmanFilter required.")
        LARGE_NEG_VALUE = -1e10
        solution = self.solve(param_dict) # Calls the solve method using VJPs

        def _calculate_likelihood_branch(sol):
            # Check if solution seems valid before trying KF
            if not jnp.all(jnp.isfinite(sol["P_aug"])) or \
               not jnp.all(jnp.isfinite(sol["R_aug"])) or \
               not jnp.all(jnp.isfinite(sol["Omega"])):
                 # if self.verbose: print("[LogLik VJP] Non-finite matrices in valid solution branch.")
                 return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

            # Check dimensions before KF instantiation
            P_aug, R_aug, Omega_m = sol["P_aug"], sol["R_aug"], sol["Omega"]
            n_aug_kf, n_obs_kf = P_aug.shape[0], Omega_m.shape[0]
            n_aug_shock_kf = R_aug.shape[1]
            # Add checks against init_x_mean, init_P_cov, ys, H_obs shapes
            shapes_ok = (
                 P_aug.shape==(n_aug_kf, n_aug_kf) and R_aug.shape==(n_aug_kf, n_aug_shock_kf) and
                 Omega_m.shape==(n_obs_kf, n_aug_kf) and H_obs.shape==(n_obs_kf, n_obs_kf) and
                 init_x_mean.shape==(n_aug_kf,) and init_P_cov.shape==(n_aug_kf, n_aug_kf) and
                 (ys.ndim==1 and ys.shape[0]==n_obs_kf or ys.ndim==2 and ys.shape[1]==n_obs_kf) # Allow single obs row or multiple
                 )
            if not shapes_ok:
                 # if self.verbose: print("[LogLik VJP] Shape mismatch before KF.")
                 return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

            try:
                kf = KalmanFilter(T=P_aug, R=R_aug, C=Omega_m, H=H_obs, init_x=init_x_mean, init_P=init_P_cov)
                log_prob = kf.log_likelihood(ys)
                # Ensure final value is finite for Numpyro
                safe_log_prob = jnp.where(jnp.isfinite(log_prob), log_prob, jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE))
                return safe_log_prob
            except Exception as kf_e:
                # if self.verbose: print(f"[LogLik VJP] KF Exception: {kf_e}")
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

        def _invalid_likelihood_branch(_):
            return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

        # Use lax.cond on the solution_valid flag
        log_prob_final = jax.lax.cond(
            solution["solution_valid"],
            _calculate_likelihood_branch,
            _invalid_likelihood_branch,
            solution # Pass the solution dict as operand
        )
        # Final safety check (redundant if branches handle it, but safe)
        return jnp.where(jnp.isfinite(log_prob_final), log_prob_final, jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE))


# --- Numpyro Model Function (Unchanged from previous version) ---
def numpyro_model_symbolic_vjp(
    model_instance: DynareModelWithSympyVJPs,
    user_priors: List[Dict[str, Any]],
    fixed_param_values: Dict[str, float],
    ys: Optional[jax.Array] = None,
    H_obs: Optional[jax.Array] = None,
    init_x_mean: Optional[jax.Array] = None,
    init_P_cov: Optional[jax.Array] = None
):
    # --- This function remains exactly the same ---
    # It samples parameters and calls model_instance.log_likelihood
    if not NUMPYRO_AVAILABLE: raise RuntimeError("Numpyro required.")
    params_for_likelihood = {}
    estimated_param_names = {p["name"] for p in user_priors}
    for prior_spec in user_priors:
        name, dist_name, args = prior_spec["name"], prior_spec.get("prior","").lower(), prior_spec.get("args",{})
        dist_args_proc = {k: jnp.asarray(v,dtype=_DEFAULT_DTYPE) for k,v in args.items()}
        val = None
        if dist_name=="normal": val=numpyro.sample(name,dist.Normal(dist_args_proc.get("loc",0.0),jnp.maximum(dist_args_proc.get("scale",1.0),1e-7)))
        elif dist_name=="beta": val=numpyro.sample(name,dist.Beta(jnp.maximum(dist_args_proc.get("concentration1",1.0),1e-7),jnp.maximum(dist_args_proc.get("concentration2",1.0),1e-7)))
        elif dist_name=="gamma": val=numpyro.sample(name,dist.Gamma(jnp.maximum(dist_args_proc.get("concentration",1.0),1e-7),rate=jnp.maximum(dist_args_proc.get("rate",1.0),1e-7)))
        elif dist_name=="invgamma": val=numpyro.sample(name,dist.InverseGamma(jnp.maximum(dist_args_proc.get("concentration",1.0),1e-7),scale=jnp.maximum(dist_args_proc.get("scale",1.0),1e-7)))
        elif dist_name=="uniform": val=numpyro.sample(name,dist.Uniform(dist_args_proc.get("low",0.0),dist_args_proc.get("high",1.0)))
        elif dist_name=="halfnormal": val=numpyro.sample(name,dist.HalfNormal(jnp.maximum(dist_args_proc.get("scale",1.0),1e-7)))
        else: raise NotImplementedError(f"Prior '{dist_name}' not implemented for '{name}'.")
        params_for_likelihood[name] = val
    for name, value in fixed_param_values.items():
        if name not in estimated_param_names: params_for_likelihood[name] = jnp.asarray(value, dtype=_DEFAULT_DTYPE)
    if ys is not None:
        if H_obs is None or init_x_mean is None or init_P_cov is None: raise ValueError("H_obs, init_x_mean, init_P_cov required with ys.")
        log_prob = model_instance.log_likelihood(params_for_likelihood, ys, H_obs, init_x_mean, init_P_cov)
        numpyro.factor("log_likelihood", log_prob)


# --- Main Execution Block (Modified to use DynareModelWithSympyVJPs) ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file)

    num_sim_steps, sim_seed, sim_measurement_noise_std = 200, 123, 0.001
    run_estimation_flag, mcmc_seed, mcmc_chains, mcmc_warmup, mcmc_samples, mcmc_target_accept = True, 456, 1, 250, 500, 0.85 # Back to original samples

    print(f"\n--- [1] Initializing Dynare Model with Symbolic VJPs ({mod_file_path}) ---")
    model = DynareModelWithSympyVJPs(mod_file_path, verbose=False)
    print(f"Model initialized. Found {len(model.all_param_names)} parameters total.")

    sim_param_values = model.default_param_assignments.copy()
    # sim_param_values.update({'b1': 0.7, ...}) # Apply overrides if needed

    print("\n--- [3] Simulating Data ---")
    sim_key_master = random.PRNGKey(sim_seed); sim_key_init, sim_key_path = random.split(sim_key_master)
    # Need to wrap solve in try-except here in case initial solve fails
    try:
        sim_solution = model.solve(sim_param_values)
        if not sim_solution["solution_valid"]:
            print("FATAL: Cannot solve model with simulation parameters for VJP model."); exit()
    except Exception as e_solve_init:
         print(f"FATAL: Error during initial solve for simulation: {e_solve_init}"); exit()


    s0_sim = construct_initial_state(
        sim_solution["n_aug"], model.n_stat, sim_solution["aug_state_vars"], sim_key_init,
        initial_state_config={"L_GDP_TREND":{"mean":10000.0,"std":0.01},"G_TREND":{"mean":0.5},"PI_TREND":{"mean":2.0}},
        dtype=_DEFAULT_DTYPE
    )
    H_obs_sim = jnp.eye(sim_solution["n_obs"], dtype=_DEFAULT_DTYPE) * (sim_measurement_noise_std**2)
    sim_states, sim_observables = simulate_ssm_data(
        sim_solution["P_aug"], sim_solution["R_aug"], sim_solution["Omega"],
        num_sim_steps, sim_key_path, s0_sim, sim_measurement_noise_std
    )
    print(f"Simulation complete. Observables shape: {sim_observables.shape}")
    # plot_simulation_with_trends_matched(...) # Optional plot

    print("\n--- [4] Defining Priors for Estimation ---")
    user_priors = [{"name": "sigma_SHK_RS", "prior": "invgamma", "args": {"concentration": 3.0, "scale": 0.2}}]
    estimated_param_names_set = {p["name"] for p in user_priors}
    fixed_params = {name: val for name, val in model.default_param_assignments.items() if name not in estimated_param_names_set}

    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE:
        print(f"\n--- [5] Running Bayesian Estimation (Symbolic VJP Model) ---")
        mcmc_key = random.PRNGKey(mcmc_seed)
        H_obs_est, init_x_mean_est, init_P_cov_est = H_obs_sim, s0_sim, jnp.eye(sim_solution["n_aug"], dtype=_DEFAULT_DTYPE) * 1.0
        init_values_mcmc = {p["name"]: sim_param_values.get(p["name"], 0.1) for p in user_priors}

        initial_params_full = {**fixed_params, **init_values_mcmc}
        try:
            ll_test = model.log_likelihood(initial_params_full, sim_observables, H_obs_est, init_x_mean_est, init_P_cov_est)
            print(f"Log-likelihood at initial parameters (VJP Model): {ll_test}")
            if not jnp.isfinite(ll_test) or ll_test < -1e9 :
                print("ERROR: Initial log-likelihood is not valid. Exiting before MCMC.")
                exit()
        except Exception as e_ll_test:
             print(f"ERROR evaluating initial log-likelihood: {e_ll_test}"); exit()

        kernel = NUTS(numpyro_model_symbolic_vjp, init_strategy=init_to_value(values=init_values_mcmc), target_accept_prob=mcmc_target_accept)
        mcmc = MCMC(kernel, num_warmup=mcmc_warmup, num_samples=mcmc_samples, num_chains=mcmc_chains, progress_bar=True, chain_method='parallel')

        print(f"Starting MCMC (Symbolic VJP Model)...")
        est_start_time = time.time()
        try:
            mcmc.run(mcmc_key, model, user_priors, fixed_params, sim_observables, H_obs_est, init_x_mean_est, init_P_cov_est)
            est_end_time = time.time()
            print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")
            mcmc.print_summary()
        except Exception as e_mcmc:
            print(f"\n--- MCMC FAILED ---")
            print(f"{type(e_mcmc).__name__}: {e_mcmc}")
            import traceback; traceback.print_exc() # Print full traceback for MCMC errors
    else:
        print("\n--- Skipping Estimation ---")

    print(f"\n--- Script finished (Symbolic VJP Model) ---")
    # if run_estimation_flag: plt.show()

# --- END OF MODIFIED FILE run_estimation.py ---