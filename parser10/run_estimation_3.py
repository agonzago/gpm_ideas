import os
import time
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any
import re

# --- JAX/Numpyro Setup ---
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# --- Force CPU Execution (Optional) ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update("jax_platforms", "cpu")
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    NUMPYRO_AVAILABLE = True
    try:
        num_devices_to_use = jax.local_device_count()
        numpyro.set_host_device_count(num_devices_to_use)
    except Exception as e_np_config:
        print(f"Warning: Could not configure numpyro device count: {e_np_config}")
except ImportError:
    NUMPYRO_AVAILABLE = False
    print("Warning: numpyro not found. Estimation disabled.")

try:
    from Kalman_filter_jax import KalmanFilter # Assuming it's in the same directory
    KALMAN_FILTER_JAX_AVAILABLE = True
except ImportError:
    KALMAN_FILTER_JAX_AVAILABLE = False
    print("Warning: Kalman_filter_jax.py not found. Likelihood calculation will fail.")

# --- Import from Parser ---
from dynare_parser_sda_solver_jax import (
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
    extract_trend_declarations,
    extract_trend_shock_stderrs,
    extract_stationary_shock_stderrs,
    extract_declarations
)

# ==============================================================================
# == VJP HELPER FUNCTIONS (JAX-Compatible) ==
# ==============================================================================

@jax.custom_vjp
def build_stationary_matrices_with_vjp(params_tuple_jax, lambdas_stat, stat_eq_perm_indices, stat_var_perm_indices):
    # Forward pass just computes matrices
    num_eq = len(lambdas_stat['A']['elements'])
    num_vars = len(lambdas_stat['A']['elements'][0]) if num_eq > 0 else 0
    num_shocks = len(lambdas_stat['D']['elements'][0]) if num_eq > 0 and lambdas_stat['D']['elements'] and lambdas_stat['D']['elements'][0] else 0
    
    # Calculate unordered matrices using lambda functions
    A_unord = jnp.array([[lambdas_stat['A']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    B_unord = jnp.array([[lambdas_stat['B']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    C_unord = jnp.array([[lambdas_stat['C']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    D_unord = jnp.array([[lambdas_stat['D']['elements'][i][j](*params_tuple_jax) for j in range(num_shocks)] for i in range(num_eq)]) if num_shocks > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    
    # Apply permutations
    A_ord = A_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    B_ord = B_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    C_ord = C_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    D_ord = D_unord[jnp.ix_(stat_eq_perm_indices, jnp.arange(D_unord.shape[1]))] if num_shocks > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    
    return A_ord, B_ord, C_ord, D_ord

# Forward rule for custom VJP
def _build_stat_matrices_fwd(params_tuple_jax, lambdas_stat, stat_eq_perm_indices, stat_var_perm_indices):
    num_eq = len(lambdas_stat['A']['elements'])
    num_vars = len(lambdas_stat['A']['elements'][0]) if num_eq > 0 else 0
    num_shocks = len(lambdas_stat['D']['elements'][0]) if num_eq > 0 and lambdas_stat['D']['elements'] and lambdas_stat['D']['elements'][0] else 0
    
    # Calculate unordered matrices using lambda functions
    A_unord = jnp.array([[lambdas_stat['A']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    B_unord = jnp.array([[lambdas_stat['B']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    C_unord = jnp.array([[lambdas_stat['C']['elements'][i][j](*params_tuple_jax) for j in range(num_vars)] for i in range(num_eq)]) if num_vars > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    D_unord = jnp.array([[lambdas_stat['D']['elements'][i][j](*params_tuple_jax) for j in range(num_shocks)] for i in range(num_eq)]) if num_shocks > 0 else jnp.zeros((num_eq, 0), dtype=_DEFAULT_DTYPE)
    
    # Apply permutations
    A_ord = A_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    B_ord = B_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    C_ord = C_unord[jnp.ix_(stat_eq_perm_indices, stat_var_perm_indices)] if num_vars > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    D_ord = D_unord[jnp.ix_(stat_eq_perm_indices, jnp.arange(D_unord.shape[1]))] if num_shocks > 0 else jnp.zeros((num_eq,0), dtype=_DEFAULT_DTYPE)
    
    return (A_ord, B_ord, C_ord, D_ord), (params_tuple_jax, lambdas_stat, A_unord, B_unord, C_unord, D_unord, stat_eq_perm_indices, stat_var_perm_indices)

# Backward rule for custom VJP
def _build_stat_matrices_bwd(residuals_for_bwd, grads_output_ordered_tuple):
    params_tuple_jax, lambdas_stat, A_unord, B_unord, C_unord, D_unord, stat_eq_perm_indices, stat_var_perm_indices = residuals_for_bwd
    grad_A_ord, grad_B_ord, grad_C_ord, grad_D_ord = grads_output_ordered_tuple
    num_params = len(params_tuple_jax)
    
    # Initialize gradient arrays for unordered matrices
    grad_A_unord = jnp.zeros_like(A_unord)
    grad_B_unord = jnp.zeros_like(B_unord)
    grad_C_unord = jnp.zeros_like(C_unord)
    grad_D_unord = jnp.zeros_like(D_unord)
    
    # Transfer gradients from ordered to unordered matrices
    if A_unord.size > 0:
        for r_new in range(grad_A_ord.shape[0]):
            r_old_eq = stat_eq_perm_indices[r_new]
            for c_new_var in range(grad_A_ord.shape[1]):
                c_old_var = stat_var_perm_indices[c_new_var]
                grad_A_unord = grad_A_unord.at[r_old_eq, c_old_var].add(grad_A_ord[r_new, c_new_var])
                grad_B_unord = grad_B_unord.at[r_old_eq, c_old_var].add(grad_B_ord[r_new, c_new_var])
                grad_C_unord = grad_C_unord.at[r_old_eq, c_old_var].add(grad_C_ord[r_new, c_new_var])
    
    # Handle D matrix separately (different shape)
    if D_unord.size > 0:
        for r_new in range(grad_D_ord.shape[0]):
            r_old_eq = stat_eq_perm_indices[r_new]
            for c_new_shk in range(grad_D_ord.shape[1]):
                grad_D_unord = grad_D_unord.at[r_old_eq, c_new_shk].add(grad_D_ord[r_new, c_new_shk])
    
    # Calculate parameter gradients
    param_grads_list = [jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in range(num_params)]
    for p_idx in range(num_params):
        current_param_grad = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
        if A_unord.size > 0:
            for i,j in jnp.ndindex(A_unord.shape):
                current_param_grad += grad_A_unord[i,j] * lambdas_stat['A']['grads'][i][j][p_idx](*params_tuple_jax)
        if B_unord.size > 0:
            for i,j in jnp.ndindex(B_unord.shape):
                current_param_grad += grad_B_unord[i,j] * lambdas_stat['B']['grads'][i][j][p_idx](*params_tuple_jax)
        if C_unord.size > 0:
            for i,j in jnp.ndindex(C_unord.shape):
                current_param_grad += grad_C_unord[i,j] * lambdas_stat['C']['grads'][i][j][p_idx](*params_tuple_jax)
        if D_unord.size > 0:
            for i,j in jnp.ndindex(D_unord.shape):
                current_param_grad += grad_D_unord[i,j] * lambdas_stat['D']['grads'][i][j][p_idx](*params_tuple_jax)
        param_grads_list[p_idx] = current_param_grad
    
    return (tuple(param_grads_list), None, None, None)  # Grads for params, None for static data

# Register forward and backward rules
build_stationary_matrices_with_vjp.defvjp(_build_stat_matrices_fwd, _build_stat_matrices_bwd)


# --- Trend Matrices VJP ---
@jax.custom_vjp
def build_trend_matrices_with_vjp(params_tuple_jax, lambdas_trend_data):
    if not lambdas_trend_data or not lambdas_trend_data.get('lambda_matrices') or not lambdas_trend_data['lambda_matrices'].get('P_trends'):
        P_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        Q_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    else:
        lambdas_trend = lambdas_trend_data['lambda_matrices']
        num_rows_p = len(lambdas_trend['P_trends']['elements'])
        num_cols_p = len(lambdas_trend['P_trends']['elements'][0]) if num_rows_p > 0 else 0
        
        num_cols_q = 0
        if 'Q_trends' in lambdas_trend and lambdas_trend['Q_trends']['elements'] and lambdas_trend['Q_trends']['elements'][0]:
            num_cols_q = len(lambdas_trend['Q_trends']['elements'][0])
        
        P_list = [[lambdas_trend['P_trends']['elements'][i][j](*params_tuple_jax) for j in range(num_cols_p)] for i in range(num_rows_p)]
        Q_list = [[lambdas_trend['Q_trends']['elements'][i][j](*params_tuple_jax) for j in range(num_cols_q)] for i in range(num_rows_p)] if num_cols_q > 0 else [[] for _ in range(num_rows_p)]
        
        P_trends = jnp.array(P_list, dtype=_DEFAULT_DTYPE) if num_rows_p * num_cols_p > 0 else jnp.zeros((num_rows_p,0), dtype=_DEFAULT_DTYPE)
        Q_trends = jnp.array(Q_list, dtype=_DEFAULT_DTYPE) if num_rows_p * num_cols_q > 0 else jnp.zeros((num_rows_p,0), dtype=_DEFAULT_DTYPE)
    
    return P_trends, Q_trends

def _build_trend_matrices_fwd(params_tuple_jax, lambdas_trend_data):
    if not lambdas_trend_data or not lambdas_trend_data.get('lambda_matrices') or not lambdas_trend_data['lambda_matrices'].get('P_trends'):
        P_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        Q_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        return (P_trends, Q_trends), (params_tuple_jax, lambdas_trend_data, P_trends, Q_trends)
    
    lambdas_trend = lambdas_trend_data['lambda_matrices']
    num_rows_p = len(lambdas_trend['P_trends']['elements'])
    num_cols_p = len(lambdas_trend['P_trends']['elements'][0]) if num_rows_p > 0 else 0
    
    num_cols_q = 0
    if 'Q_trends' in lambdas_trend and lambdas_trend['Q_trends']['elements'] and lambdas_trend['Q_trends']['elements'][0]:
        num_cols_q = len(lambdas_trend['Q_trends']['elements'][0])
    
    P_list = [[lambdas_trend['P_trends']['elements'][i][j](*params_tuple_jax) for j in range(num_cols_p)] for i in range(num_rows_p)]
    Q_list = [[lambdas_trend['Q_trends']['elements'][i][j](*params_tuple_jax) for j in range(num_cols_q)] for i in range(num_rows_p)] if num_cols_q > 0 else [[] for _ in range(num_rows_p)]
    
    P_trends = jnp.array(P_list, dtype=_DEFAULT_DTYPE) if num_rows_p * num_cols_p > 0 else jnp.zeros((num_rows_p,0), dtype=_DEFAULT_DTYPE)
    Q_trends = jnp.array(Q_list, dtype=_DEFAULT_DTYPE) if num_rows_p * num_cols_q > 0 else jnp.zeros((num_rows_p,0), dtype=_DEFAULT_DTYPE)
    
    return (P_trends, Q_trends), (params_tuple_jax, lambdas_trend_data, P_trends, Q_trends)

def _build_trend_matrices_bwd(residuals_for_bwd, grads_output_tuple):
    params_tuple_jax, lambdas_trend_data, P_trends, Q_trends = residuals_for_bwd
    grad_P, grad_Q = grads_output_tuple
    
    if not lambdas_trend_data or not lambdas_trend_data.get('lambda_matrices') or not lambdas_trend_data['lambda_matrices'].get('P_trends'):
        return (tuple(jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in params_tuple_jax), None)
    
    lambdas_trend = lambdas_trend_data['lambda_matrices']
    num_params = len(params_tuple_jax)
    param_grads_list = [jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in range(num_params)]
    
    for p_idx in range(num_params):
        current_param_grad = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
        if P_trends.size > 0:
            for i,j in jnp.ndindex(P_trends.shape):
                current_param_grad += grad_P[i,j] * lambdas_trend['P_trends']['grads'][i][j][p_idx](*params_tuple_jax)
        if Q_trends.size > 0 and 'Q_trends' in lambdas_trend:
            for i,j in jnp.ndindex(Q_trends.shape):
                current_param_grad += grad_Q[i,j] * lambdas_trend['Q_trends']['grads'][i][j][p_idx](*params_tuple_jax)
        param_grads_list[p_idx] = current_param_grad
    
    return (tuple(param_grads_list), None)  # Grad for params_tuple_jax, None for lambdas_trend_data

build_trend_matrices_with_vjp.defvjp(_build_trend_matrices_fwd, _build_trend_matrices_bwd)


# --- Observation Matrix VJP ---
@jax.custom_vjp
def build_observation_matrix_with_vjp(params_tuple_jax, lambdas_obs_data, n_obs, n_aug):
    if not lambdas_obs_data or not lambdas_obs_data.get('lambda_matrices') or not lambdas_obs_data['lambda_matrices'].get('Omega'):
        Omega = jnp.empty((n_obs, n_aug), dtype=_DEFAULT_DTYPE)
    else:
        lambdas_omega = lambdas_obs_data['lambda_matrices']['Omega']
        num_lambda_rows = len(lambdas_omega['elements']) if lambdas_omega['elements'] else 0
        num_lambda_cols = len(lambdas_omega['elements'][0]) if num_lambda_rows > 0 and lambdas_omega['elements'][0] else 0
        
        Omega_list = [[(lambdas_omega['elements'][i][j](*params_tuple_jax) if i < num_lambda_rows and j < num_lambda_cols else 0.0)
                      for j in range(n_aug)] for i in range(n_obs)]
        Omega = jnp.array(Omega_list, dtype=_DEFAULT_DTYPE)
    
    return Omega

def _build_obs_matrix_fwd(params_tuple_jax, lambdas_obs_data, n_obs, n_aug):
    if not lambdas_obs_data or not lambdas_obs_data.get('lambda_matrices') or not lambdas_obs_data['lambda_matrices'].get('Omega'):
        Omega = jnp.zeros((n_obs, n_aug), dtype=_DEFAULT_DTYPE)
        return Omega, (params_tuple_jax, lambdas_obs_data, Omega)
    
    lambdas_omega = lambdas_obs_data['lambda_matrices']['Omega']
    num_lambda_rows = len(lambdas_omega['elements']) if lambdas_omega['elements'] else 0
    num_lambda_cols = len(lambdas_omega['elements'][0]) if num_lambda_rows > 0 and lambdas_omega['elements'][0] else 0
    
    Omega_list = [[(lambdas_omega['elements'][i][j](*params_tuple_jax) if i < num_lambda_rows and j < num_lambda_cols else 0.0)
                  for j in range(n_aug)] for i in range(n_obs)]
    Omega = jnp.array(Omega_list, dtype=_DEFAULT_DTYPE)
    
    return Omega, (params_tuple_jax, lambdas_obs_data, Omega)

def _build_obs_matrix_bwd(residuals_for_bwd, grad_Omega):
    params_tuple_jax, lambdas_obs_data, Omega_from_fwd = residuals_for_bwd
    
    if not lambdas_obs_data or not lambdas_obs_data.get('lambda_matrices') or not lambdas_obs_data['lambda_matrices'].get('Omega'):
        return (tuple(jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in params_tuple_jax), None, None, None)
    
    lambdas_omega = lambdas_obs_data['lambda_matrices']['Omega']
    num_params = len(params_tuple_jax)
    param_grads_list = [jnp.array(0.0, dtype=_DEFAULT_DTYPE) for _ in range(num_params)]
    
    num_lambda_rows = len(lambdas_omega['elements']) if lambdas_omega['elements'] else 0
    num_lambda_cols = len(lambdas_omega['elements'][0]) if num_lambda_rows > 0 and lambdas_omega['elements'][0] else 0
    rows_to_iter = min(Omega_from_fwd.shape[0], num_lambda_rows) 
    cols_to_iter = min(Omega_from_fwd.shape[1], num_lambda_cols)
    
    for p_idx in range(num_params):
        current_param_grad = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
        if Omega_from_fwd.size > 0:
            for i in range(rows_to_iter):
                for j in range(cols_to_iter):
                    if i < num_lambda_rows and j < num_lambda_cols and grad_Omega[i, j] != 0:
                        dOdP_fn = lambdas_omega['grads'][i][j][p_idx]
                        current_param_grad += grad_Omega[i, j] * dOdP_fn(*params_tuple_jax)
        param_grads_list[p_idx] = current_param_grad
    
    return (tuple(param_grads_list), None, None, None)  # Grads for params, None for static data

build_observation_matrix_with_vjp.defvjp(_build_obs_matrix_fwd, _build_obs_matrix_bwd)


# ==============================================================================
# == DynareModel Class ==
# ==============================================================================
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
        if self._parsed: return
        if self.verbose: print("--- DynareModel: Parsing and Generating Symbolic Lambdas ---")
        
        # Parse stationary model structure
        self.stationary_structure = parse_and_order_stationary_model_symbolic(self.model_def_content, verbose=self.verbose)
        self.all_param_names = self.stationary_structure['param_names_all']
        self.default_param_assignments = self.stationary_structure['param_assignments_default']
        
        # Generate lambdas for stationary model matrices
        self._symbolic_data['stationary'] = generate_matrix_lambda_functions(
            equations_str=self.stationary_structure['equations_processed'],
            var_names_ordered=self.stationary_structure['var_names_initial_order'],
            shock_names_ordered=self.stationary_structure['shock_names'],
            all_param_names_ordered=self.all_param_names, 
            model_type="stationary", 
            verbose=self.verbose
        )
        
        # Store permutation indices and ordered variable lists
        self.stat_var_perm_indices = jnp.array(self.stationary_structure['var_permutation_indices'])
        self.stat_eq_perm_indices = jnp.array(self.stationary_structure['eq_permutation_indices'])
        self.ordered_stat_vars = self.stationary_structure['ordered_vars_final']
        self.stat_shocks = self.stationary_structure['shock_names']
        
        # Extract trend shock standard errors and update parameter lists
        _trend_stderr_p = extract_trend_shock_stderrs(self.model_def_content)
        self.default_param_assignments.update(_trend_stderr_p)
        for p_name in _trend_stderr_p.keys():
            if p_name not in self.all_param_names: 
                self.all_param_names.append(p_name)
        
        # Generate lambdas for trend model matrices
        self._symbolic_data['trend'] = generate_trend_lambda_functions(
            self.model_def_content, 
            self.all_param_names, 
            verbose=self.verbose
        )
        self.ordered_trend_state_vars = self._symbolic_data['trend'].get('state_trend_vars', [])
        self.contemp_trend_defs = self._symbolic_data['trend'].get('contemporaneous_trend_defs', {})
        _, self.trend_shocks = extract_trend_declarations(self.model_def_content)
        
        # Generate lambdas for observation model matrices
        self._symbolic_data['observation'] = generate_observation_lambda_functions(
            self.model_def_content, 
            self.all_param_names, 
            self.ordered_stat_vars,
            self.ordered_trend_state_vars, 
            self.contemp_trend_defs, 
            verbose=self.verbose
        )
        self.ordered_obs_vars = self._symbolic_data['observation'].get('ordered_obs_vars', [])
        
        # Store dimensions for convenience
        self.n_stat, self.n_s_shock = len(self.ordered_stat_vars), len(self.stat_shocks)
        self.n_trend, self.n_t_shock = len(self.ordered_trend_state_vars), len(self.trend_shocks)
        self.n_obs = len(self.ordered_obs_vars)
        self.n_aug, self.n_aug_shock = self.n_stat + self.n_trend, self.n_s_shock + self.n_t_shock
        self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars
        self.aug_shocks = self.stat_shocks + self.trend_shocks
        
        self._parsed = True
        if self.verbose: print("--- DynareModel: Symbolic Setup Complete ---")

    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Solves the model using the provided parameter values.
        
        Args:
            param_dict: Dictionary mapping parameter names to values
            
        Returns:
            Dictionary containing solution matrices and metadata
        """
        if not self._parsed: 
            self._parse_and_generate_lambdas()
        
        # Convert parameter dictionary to tuple in correct order
        params_tuple_jax = tuple(jnp.asarray(param_dict[p_name], dtype=_DEFAULT_DTYPE) 
                                for p_name in self.all_param_names)

        results = {"solution_valid": jnp.array(False)}
        if self.verbose: 
            print(f"\n--- Solving with params: {param_dict} ---")

        try:
            # --- Step 1: Stationary Matrices ---
            if self.verbose: 
                print("  Building stationary matrices...")
            
            # Use module-level VJP function
            A_num_stat, B_num_stat, C_num_stat, D_num_stat = build_stationary_matrices_with_vjp(
                params_tuple_jax,
                self._symbolic_data['stationary']['lambda_matrices'],
                self.stat_eq_perm_indices,
                self.stat_var_perm_indices
            )
            
# Check for NaN/Inf in matrices
            if not (jnp.all(jnp.isfinite(A_num_stat)) and jnp.all(jnp.isfinite(B_num_stat)) and 
                   jnp.all(jnp.isfinite(C_num_stat)) and jnp.all(jnp.isfinite(D_num_stat))):
                if self.verbose: 
                    print("  ERROR: NaN/Inf in stationary matrices A,B,C,D.")
                results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                return results

            # --- Step 2: Solve Stationary Model (SDA) ---
            if self.verbose: 
                print("  Solving quadratic matrix equation...")
                
            P_sol_stat, _, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500
            )
            
            valid_stat_solve = converged_stat & jnp.all(jnp.isfinite(P_sol_stat))
            
            if not valid_stat_solve:
                if self.verbose: 
                    print("  ERROR: SDA solver failed or P_sol_stat contains NaN/Inf.")
                results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                return results

            # --- Step 3: Compute Q_stationary ---
            if self.verbose: 
                print("  Computing Q matrix...")
                
            # Ensure D_num_stat has correct shape
            fallback_D_shape = (A_num_stat.shape[0], self.n_s_shock if hasattr(self, 'n_s_shock') else 0)
            Q_sol_stat = jnp.where(
                 valid_stat_solve,
                 compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat),
                 jnp.full(fallback_D_shape, jnp.nan, dtype=_DEFAULT_DTYPE)
            )
            
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat))
            
            if not valid_q_compute:
                if self.verbose: 
                    print("  ERROR: Q_sol_stat computation resulted in NaN/Inf.")
                results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                return results

            # --- Step 4: Trend Matrices ---
            if self.verbose: 
                print("  Building trend matrices...")
                
            # Use module-level VJP function
            P_num_trend, Q_num_trend = build_trend_matrices_with_vjp(
                params_tuple_jax,
                self._symbolic_data['trend']
            )
            
            if not (jnp.all(jnp.isfinite(P_num_trend)) and jnp.all(jnp.isfinite(Q_num_trend))):
                if self.verbose: 
                    print("  ERROR: NaN/Inf in trend matrices P,Q.")
                results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                return results

            # --- Step 5: Observation Matrix ---
            if self.verbose: 
                print("  Building observation matrix...")
                
            # Use module-level VJP function
            Omega_num = build_observation_matrix_with_vjp(
                params_tuple_jax,
                self._symbolic_data['observation'],
                self.n_obs,
                self.n_aug
            )
            
            if not jnp.all(jnp.isfinite(Omega_num)):
                if self.verbose: 
                    print("  ERROR: NaN/Inf in Omega matrix.")
                results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                return results
            
            # --- Step 6: Build R matrices and augmented system ---
            if self.verbose: 
                print("  Building R matrices and augmented system...")
                
            # Extract shock standard deviations from parameter values
            shock_std_devs = {}
            for shock_name in self.aug_shocks:
                sigma_param_name = f"sigma_{shock_name}"
                param_idx = self.all_param_names.index(sigma_param_name)
                std_dev = params_tuple_jax[param_idx]
                shock_std_devs[shock_name] = jnp.maximum(jnp.abs(std_dev), 1e-9)
            
            # Convert to arrays for matrix multiplication
            stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], dtype=_DEFAULT_DTYPE) if self.stat_shocks else jnp.array([], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], dtype=_DEFAULT_DTYPE) if self.trend_shocks else jnp.array([], dtype=_DEFAULT_DTYPE)
            
            # Apply standard deviations to Q matrices to get R matrices
            R_sol_stat = jnp.zeros((self.n_stat, self.n_s_shock), dtype=_DEFAULT_DTYPE)
            if self.n_s_shock > 0 and Q_sol_stat.ndim == 2 and Q_sol_stat.shape == (self.n_stat, self.n_s_shock): 
                R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr)
                
            R_num_trend = jnp.zeros((self.n_trend, self.n_t_shock), dtype=_DEFAULT_DTYPE)
            if self.n_t_shock > 0 and hasattr(Q_num_trend, 'shape') and Q_num_trend.ndim == 2 and Q_num_trend.shape == (self.n_trend, self.n_t_shock): 
                R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr)
            
            # Build block-diagonal P_aug
            blocks_P = []
            if hasattr(P_sol_stat, 'size') and P_sol_stat.size > 0 and P_sol_stat.shape == (self.n_stat, self.n_stat): 
                blocks_P.append(P_sol_stat)
            if hasattr(P_num_trend, 'size') and P_num_trend.size > 0 and P_num_trend.shape == (self.n_trend, self.n_trend): 
                blocks_P.append(P_num_trend)
            
            if not blocks_P: 
                P_aug = jnp.zeros((self.n_aug, self.n_aug), dtype=_DEFAULT_DTYPE)
            elif len(blocks_P) == 1 and blocks_P[0].shape == (self.n_aug, self.n_aug): 
                P_aug = blocks_P[0]
            elif len(blocks_P) == 2 and (blocks_P[0].shape[0] + blocks_P[1].shape[0] == self.n_aug): 
                P_aug = jax.scipy.linalg.block_diag(*blocks_P)
            else: 
                # Manual block diagonal if needed
                P_aug = jnp.zeros((self.n_aug, self.n_aug), dtype=_DEFAULT_DTYPE)
                curr_idx = 0
                if hasattr(P_sol_stat, 'size') and P_sol_stat.size > 0 and P_sol_stat.shape == (self.n_stat, self.n_stat):
                    P_aug = P_aug.at[:self.n_stat, :self.n_stat].set(P_sol_stat)
                    curr_idx = self.n_stat
                if hasattr(P_num_trend, 'size') and P_num_trend.size > 0 and P_num_trend.shape == (self.n_trend, self.n_trend) and self.n_trend > 0:
                    P_aug = P_aug.at[curr_idx:curr_idx+self.n_trend, curr_idx:curr_idx+self.n_trend].set(P_num_trend)

            # Build R_aug by combining R_sol_stat and R_num_trend
            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=_DEFAULT_DTYPE)
            if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock): 
                R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
            if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock): 
                R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)

            # Final validations
            all_finite = (jnp.all(jnp.isfinite(P_aug)) & jnp.all(jnp.isfinite(R_aug)) & jnp.all(jnp.isfinite(Omega_num)))
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite
            
            if self.verbose: 
                print(f"  Final solution_valid: {solution_valid_final}, all_finite: {all_finite}")

            if not solution_valid_final:
                if self.verbose: 
                    print("  ERROR: One or more steps in solve failed or produced non-finite results.")
                results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
                results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
                return results

            # Store results
            results.update({
                "P_aug": P_aug, 
                "R_aug": R_aug, 
                "Omega": Omega_num,
                "solution_valid": solution_valid_final,
                "ordered_trend_state_vars": self.ordered_trend_state_vars,
                "contemp_trend_defs": self.contemp_trend_defs,
                "ordered_obs_vars": self.ordered_obs_vars,
                "aug_state_vars": self.aug_state_vars, 
                "aug_shocks": self.aug_shocks,
                "n_aug": self.n_aug, 
                "n_aug_shock": self.n_aug_shock, 
                "n_obs": self.n_obs
            })

        except Exception as e:
             if self.verbose: 
                 print(f"Exception during model solve: {type(e).__name__}: {e}")
                 import traceback
                 traceback.print_exc()
             results["solution_valid"] = jnp.array(False)
             n_aug_val = getattr(self, 'n_aug', 1)
             n_aug_val = n_aug_val if n_aug_val > 0 else 1
             n_aug_shock_val = getattr(self, 'n_aug_shock', 0)
             n_aug_shock_val = n_aug_shock_val if n_aug_shock_val >= 0 else 0
             n_obs_val = getattr(self, 'n_obs', 1)
             n_obs_val = n_obs_val if n_obs_val > 0 else 1
             results["P_aug"] = jnp.full((n_aug_val, n_aug_val), jnp.nan, dtype=_DEFAULT_DTYPE)
             results["R_aug"] = jnp.full((n_aug_val, n_aug_shock_val), jnp.nan, dtype=_DEFAULT_DTYPE)
             results["Omega"] = jnp.full((n_obs_val, n_aug_val), jnp.nan, dtype=_DEFAULT_DTYPE)
        
        # Ensure solution_valid is a JAX array
        results["solution_valid"] = jnp.asarray(results["solution_valid"], dtype=jnp.bool_)
        return results

    def log_likelihood(self, param_dict: Dict[str, float], ys: jax.Array,
                       H_obs: jax.Array, init_x_mean: jax.Array, init_P_cov: jax.Array) -> jax.Array:
        """
        Calculates log-likelihood of observations given parameters.
        
        Args:
            param_dict: Dictionary mapping parameter names to values
            ys: Observation data array [T, n_obs]
            H_obs: Observation noise covariance matrix [n_obs, n_obs]
            init_x_mean: Initial state mean [n_aug]
            init_P_cov: Initial state covariance [n_aug, n_aug]
            
        Returns:
            JAX array containing log-likelihood value
        """
        if not KALMAN_FILTER_JAX_AVAILABLE: 
            raise RuntimeError("Custom KalmanFilter required.")
        
        LARGE_NEG_VALUE = -1e10
        solution = self.solve(param_dict)  # Calls solve with VJP functions

        def _calculate_likelihood_branch(sol):
            # Check if solution is valid and finite
            if not (jnp.all(jnp.isfinite(sol["P_aug"])) and 
                    jnp.all(jnp.isfinite(sol["R_aug"])) and 
                    jnp.all(jnp.isfinite(sol["Omega"]))):
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

            # Verify matrix dimensions
            expected_P_shape = (self.n_aug, self.n_aug)
            expected_R_shape = (self.n_aug, self.n_aug_shock)
            expected_Omega_shape = (self.n_obs, self.n_aug)
            expected_H_shape = (self.n_obs, self.n_obs)
            expected_init_x_shape = (self.n_aug,)
            expected_init_P_shape = (self.n_aug, self.n_aug)
            expected_ys_shape1 = (self.n_obs,)
            expected_ys_shape2 = (ys.shape[0] if ys.ndim == 2 else 1, self.n_obs)

            # Get actual shapes from solution
            P_aug_actual_shape = sol["P_aug"].shape if hasattr(sol["P_aug"], 'shape') else (-1, -1)
            R_aug_actual_shape = sol["R_aug"].shape if hasattr(sol["R_aug"], 'shape') else (-1, -1)
            Omega_actual_shape = sol["Omega"].shape if hasattr(sol["Omega"], 'shape') else (-1, -1)

            # Check if all shapes match expected
            shapes_ok = (
                P_aug_actual_shape == expected_P_shape and
                R_aug_actual_shape == expected_R_shape and
                Omega_actual_shape == expected_Omega_shape and 
                H_obs.shape == expected_H_shape and
                init_x_mean.shape == expected_init_x_shape and 
                init_P_cov.shape == expected_init_P_shape and
                (ys.ndim == 1 and ys.shape == expected_ys_shape1 or 
                 ys.ndim == 2 and ys.shape[1] == expected_ys_shape2[1])
            )
            
            if not shapes_ok:
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
            
            try:
                # Initialize Kalman filter with solution matrices
                kf = KalmanFilter(
                    T=sol["P_aug"], 
                    R=sol["R_aug"], 
                    C=sol["Omega"], 
                    H=H_obs, 
                    init_x=init_x_mean, 
                    init_P=init_P_cov
                )
                
                # Calculate log-likelihood
                log_prob = kf.log_likelihood(ys)
                
                # Ensure finite value is returned
                safe_log_prob = jnp.where(
                    jnp.isfinite(log_prob), 
                    log_prob, 
                    jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
                )
                
                return safe_log_prob
                
            except Exception:
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

        def _invalid_likelihood_branch(_):
            return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)

        # Use jax.lax.cond for conditionally computing the log-likelihood
        log_prob_final = jax.lax.cond(
            solution["solution_valid"],
            _calculate_likelihood_branch,
            _invalid_likelihood_branch,
            solution
        )
        
        # Final safety check for NaN/Inf
        return jnp.where(
            jnp.isfinite(log_prob_final), 
            log_prob_final, 
            jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
        )


# --- Numpyro Model Function ---
def numpyro_model_symbolic_vjp(
    model_instance: DynareModelWithSympyVJPs,
    user_priors: List[Dict[str, Any]],
    fixed_param_values: Dict[str, float],
    ys: Optional[jax.Array] = None,
    H_obs: Optional[jax.Array] = None,
    init_x_mean: Optional[jax.Array] = None,
    init_P_cov: Optional[jax.Array] = None
):
    """
    Creates a NumPyro model with specified priors for Bayesian estimation.
    
    Args:
        model_instance: Instance of DynareModelWithSympyVJPs
        user_priors: List of prior specifications for parameters to be estimated
        fixed_param_values: Dictionary of values for fixed parameters
        ys: Observation data (optional)
        H_obs: Observation noise covariance matrix (optional)
        init_x_mean: Initial state mean (optional)
        init_P_cov: Initial state covariance (optional)
    """
    if not NUMPYRO_AVAILABLE: 
        raise RuntimeError("Numpyro required.")
    
    # Dictionary to hold all parameters for the likelihood
    params_for_likelihood = {}
    
    # Set of parameters being estimated
    estimated_param_names = {p["name"] for p in user_priors}
    
    # Sample from prior distributions
    for prior_spec in user_priors:
        name = prior_spec["name"]
        dist_name = prior_spec.get("prior", "").lower()
        args = prior_spec.get("args", {})
        
        # Convert all distribution arguments to JAX arrays
        dist_args_proc = {k: jnp.asarray(v, dtype=_DEFAULT_DTYPE) for k, v in args.items()}
        
        # Sample from appropriate distribution
        if dist_name == "normal":
            val = numpyro.sample(
                name,
                dist.Normal(
                    dist_args_proc.get("loc", 0.0),
                    jnp.maximum(dist_args_proc.get("scale", 1.0), 1e-7)
                )
            )
        elif dist_name == "beta":
            val = numpyro.sample(
                name,
                dist.Beta(
                    jnp.maximum(dist_args_proc.get("concentration1", 1.0), 1e-7),
                    jnp.maximum(dist_args_proc.get("concentration2", 1.0), 1e-7)
                )
            )
        elif dist_name == "gamma":
            val = numpyro.sample(
                name,
                dist.Gamma(
                    jnp.maximum(dist_args_proc.get("concentration", 1.0), 1e-7),
                    rate=jnp.maximum(dist_args_proc.get("rate", 1.0), 1e-7)
                )
            )
        elif dist_name == "invgamma":
            val = numpyro.sample(
                name,
                dist.InverseGamma(
                    jnp.maximum(dist_args_proc.get("concentration", 1.0), 1e-7),
                    scale=jnp.maximum(dist_args_proc.get("scale", 1.0), 1e-7)
                )
            )
        elif dist_name == "uniform":
            val = numpyro.sample(
                name,
                dist.Uniform(
                    dist_args_proc.get("low", 0.0),
                    dist_args_proc.get("high", 1.0)
                )
            )
        elif dist_name == "halfnormal":
            val = numpyro.sample(
                name,
                dist.HalfNormal(
                    jnp.maximum(dist_args_proc.get("scale", 1.0), 1e-7)
                )
            )
        else:
            raise NotImplementedError(f"Prior '{dist_name}' not implemented for '{name}'.")
            
        # Store sampled value
        params_for_likelihood[name] = val
    
    # Add fixed parameters
    for name, value in fixed_param_values.items():
        if name not in estimated_param_names:
            params_for_likelihood[name] = jnp.asarray(value, dtype=_DEFAULT_DTYPE)
    
    # Calculate and factor in log-likelihood if data is provided
    if ys is not None:
        if H_obs is None or init_x_mean is None or init_P_cov is None:
            raise ValueError("H_obs, init_x_mean, init_P_cov required with ys.")
            
        # Calculate log-likelihood using model
        log_prob = model_instance.log_likelihood(
            params_for_likelihood, ys, H_obs, init_x_mean, init_P_cov
        )
        
        # Add log-likelihood as factor in the model
        numpyro.factor("log_likelihood", log_prob)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Get model file path, using environment variable or default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file)

    # Configuration parameters
    num_sim_steps, sim_seed = 200, 123
    sim_measurement_noise_std = 0.001
    run_estimation_flag = True
    mcmc_seed, mcmc_chains = 456, 1
    mcmc_warmup, mcmc_samples = 250, 500
    mcmc_target_accept = 0.85

    print(f"\n--- [1] Initializing Dynare Model with Symbolic VJPs ({mod_file_path}) ---")
    try:
        model = DynareModelWithSympyVJPs(mod_file_path, verbose=True)
        print(f"Model initialized. Found {len(model.all_param_names)} parameters total.")
    except Exception as e_init:
        print(f"FATAL: Failed to initialize DynareModelWithSympyVJPs: {e_init}")
        import traceback
        traceback.print_exc()
        exit()

    # Use default parameter values for simulation
    sim_param_values = model.default_param_assignments.copy()

    print("\n--- [2] Simulating Data ---")
    sim_key_master = random.PRNGKey(sim_seed)
    sim_key_init, sim_key_path = random.split(sim_key_master)
    
    try:
        # Solve model with simulation parameters
        sim_solution = model.solve(sim_param_values)
        
        if not sim_solution["solution_valid"]:
            print("FATAL: Cannot solve model with simulation parameters. Check verbose output from solve().")
            print(f"  sim_solution content on failure: {sim_solution}")
            exit()
        else:
            print("  Solve for simulation parameters was successful.")
    except Exception as e_solve_init:
         print(f"FATAL: Exception during initial solve for simulation: {type(e_solve_init).__name__}: {e_solve_init}")
         import traceback
         traceback.print_exc()
         exit()

    # Set up initial state for simulation
    s0_sim = construct_initial_state(
        sim_solution["n_aug"], 
        model.n_stat, 
        sim_solution["aug_state_vars"], 
        sim_key_init,
        initial_state_config={
            "L_GDP_TREND": {"mean": 10000.0, "std": 0.01},
            "G_TREND": {"mean": 0.5},
            "PI_TREND": {"mean": 2.0}
        },
        dtype=_DEFAULT_DTYPE
    )
    
    # Define observation noise covariance
    H_obs_sim = jnp.eye(sim_solution["n_obs"], dtype=_DEFAULT_DTYPE) * (sim_measurement_noise_std**2)
    
    # Simulate data
    sim_states, sim_observables = simulate_ssm_data(
        sim_solution["P_aug"], 
        sim_solution["R_aug"], 
        sim_solution["Omega"],
        num_sim_steps, 
        sim_key_path, 
        s0_sim, 
        sim_measurement_noise_std
    )
    
    print(f"Simulation complete. Observables shape: {sim_observables.shape}")

    print("\n--- [3] Defining Priors for Estimation ---")
    # Example prior - adjust as needed
    user_priors = [
        {
            "name": "sigma_SHK_RS", 
            "prior": "invgamma", 
            "args": {"concentration": 3.0, "scale": 0.2}
        }
    ]
    
    # Extract fixed parameters (those not being estimated)
    estimated_param_names_set = {p["name"] for p in user_priors}
    fixed_params = {
        name: val for name, val in model.default_param_assignments.items() 
        if name not in estimated_param_names_set
    }

    # Run Bayesian estimation if requested
    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE:
        print(f"\n--- [4] Running Bayesian Estimation (Symbolic VJP Model) ---")
        mcmc_key = random.PRNGKey(mcmc_seed)
        
        # Use same matrices as in simulation
        H_obs_est = H_obs_sim
        init_x_mean_est = s0_sim
        init_P_cov_est = jnp.eye(sim_solution["n_aug"], dtype=_DEFAULT_DTYPE) * 1.0
        
        # Use simulation values as initial values for MCMC
        init_values_mcmc = {
            p["name"]: sim_param_values.get(p["name"], 0.1) for p in user_priors
        }
        
        # Combine fixed and initial parameters for testing
        initial_params_full = {**fixed_params, **init_values_mcmc}
        
        # Test log-likelihood at initial parameters
        try:
            ll_test = model.log_likelihood(
                initial_params_full, 
                sim_observables, 
                H_obs_est, 
                init_x_mean_est, 
                init_P_cov_est
            )
            
            print(f"Log-likelihood at initial parameters: {ll_test}")
            
            if not jnp.isfinite(ll_test) or ll_test < -1e9:
                print("ERROR: Initial log-likelihood is not valid. Exiting before MCMC.")
                exit()
                
        except Exception as e_ll_test:
             print(f"ERROR evaluating initial log-likelihood: {type(e_ll_test).__name__}: {e_ll_test}")
             import traceback
             traceback.print_exc()
             exit()

        # Initialize MCMC
        kernel = NUTS(
            numpyro_model_symbolic_vjp, 
            init_strategy=init_to_value(values=init_values_mcmc),
            target_accept_prob=mcmc_target_accept)
        
        mcmc = MCMC(
            kernel, 
            num_warmup=mcmc_warmup, 
            num_samples=mcmc_samples, 
            num_chains=mcmc_chains,
            progress_bar=True, 
            chain_method='parallel'
        )
        
        print(f"Starting MCMC (Symbolic VJP Model)...")
        est_start_time = time.time()
        
        try:
            # Run MCMC
            mcmc.run(
                mcmc_key, 
                model, 
                user_priors, 
                fixed_params, 
                sim_observables, 
                H_obs_est, 
                init_x_mean_est, 
                init_P_cov_est
            )
            
            est_end_time = time.time()
            print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")
            
            # Print summary statistics
            mcmc.print_summary()
            
            # Optional: visualize posterior distributions
            try:
                import arviz as az
                posterior_samples = mcmc.get_samples()
                az_data = az.from_dict(posterior=posterior_samples)
                az.plot_trace(az_data)
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("arviz not available for plotting posterior distributions.")
            
        except Exception as e_mcmc:
            print(f"\n--- MCMC FAILED ---")
            print(f"{type(e_mcmc).__name__}: {e_mcmc}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Estimation ---")
        
    print(f"\n--- Script finished (Symbolic VJP Model) ---")
    