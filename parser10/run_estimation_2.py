"""
Modified run_estimation.py with fixes for JAX custom VJP implementation
"""
import os
import time
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random, grad, value_and_grad
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import re

# --- Force CPU Execution ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update("jax_platforms", "cpu")
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

# --- JAX/Numpyro Setup ---
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# Import numpyro if available
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

# Import KalmanFilter if available
try:
    from Kalman_filter_jax import KalmanFilter
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
    extract_trend_shock_stderrs
)

# ==============================================================================
# == IMPROVED DYNARE MODEL CLASS ==
# ==============================================================================
class DynareModelImproved:
    """
    Improved Dynare model implementation that avoids redundant Jacobian 
    computations by pre-calculating element functions.
    """
    def __init__(self, mod_file_path: str, verbose: bool = False):
        self.mod_file_path = mod_file_path
        self.verbose = verbose
        self._parsed = False
        self._symbolic_data = {}
        
        with open(self.mod_file_path, 'r') as f:
            self.model_def_content = f.read()
        
        self._parse_and_generate_element_functions()
        
    def _parse_and_generate_element_functions(self):
        """
        Parse model structure and generate element functions
        without JAX VJPs that handle both evaluation and gradients.
        """
        if self._parsed:
            return
            
        if self.verbose:
            print("--- Parsing and Generating Element Functions ---")
        
        # 1. Stationary Part
        self.stationary_structure = parse_and_order_stationary_model_symbolic(
            self.model_def_content, verbose=self.verbose
        )
        
        self.all_param_names = self.stationary_structure['param_names_all']
        self.default_param_assignments = self.stationary_structure['param_assignments_default']
        
        # Generate lambda functions (but we'll use them differently)
        self._symbolic_data['stationary'] = generate_matrix_lambda_functions(
            equations_str=self.stationary_structure['equations_processed'],
            var_names_ordered=self.stationary_structure['var_names_initial_order'],
            shock_names_ordered=self.stationary_structure['shock_names'],
            all_param_names_ordered=self.all_param_names,
            model_type="stationary",
            verbose=self.verbose
        )
        
        self.stat_var_perm_indices = jnp.array(self.stationary_structure['var_permutation_indices'])
        self.stat_eq_perm_indices = jnp.array(self.stationary_structure['eq_permutation_indices'])
        self.ordered_stat_vars = self.stationary_structure['ordered_vars_final']
        self.stat_shocks = self.stationary_structure['shock_names']
        
        # 2. Trend Part
        _trend_stderr_p = extract_trend_shock_stderrs(self.model_def_content)
        self.default_param_assignments.update(_trend_stderr_p)
        
        for p_name in _trend_stderr_p.keys():
            if p_name not in self.all_param_names:
                self.all_param_names.append(p_name)
        
        self._symbolic_data['trend'] = generate_trend_lambda_functions(
            self.model_def_content, self.all_param_names, verbose=self.verbose
        )
        
        self.ordered_trend_state_vars = self._symbolic_data['trend'].get('state_trend_vars', [])
        self.contemp_trend_defs = self._symbolic_data['trend'].get('contemporaneous_trend_defs', {})
        _, self.trend_shocks = extract_trend_declarations(self.model_def_content)
        
        # 3. Observation Part
        self._symbolic_data['observation'] = generate_observation_lambda_functions(
            self.model_def_content,
            self.all_param_names,
            self.ordered_stat_vars,
            self.ordered_trend_state_vars,
            self.contemp_trend_defs,
            verbose=self.verbose
        )
        
        self.ordered_obs_vars = self._symbolic_data['observation'].get('ordered_obs_vars', [])
        
        # Store dimensions
        self.n_stat = len(self.ordered_stat_vars)
        self.n_s_shock = len(self.stat_shocks)
        self.n_trend = len(self.ordered_trend_state_vars)
        self.n_t_shock = len(self.trend_shocks)
        self.n_obs = len(self.ordered_obs_vars)
        self.n_aug = self.n_stat + self.n_trend
        self.n_aug_shock = self.n_s_shock + self.n_t_shock
        self.aug_state_vars = self.ordered_stat_vars + self.ordered_trend_state_vars
        self.aug_shocks = self.stat_shocks + self.trend_shocks
        
        # Create numpy-based evaluation functions for matrices
        self._create_numpy_matrix_functions()
        
        self._parsed = True
        if self.verbose:
            print("--- Model Parsing and Setup Complete ---")
            
    def _create_numpy_matrix_functions(self):
        """
        Create numpy-based evaluation functions for all matrix elements.
        These don't rely on JAX's autodiff but instead use the pre-generated
        sympy lambdified functions.
        """
        if self.verbose:
            print("  Creating numpy matrix evaluation functions...")
            
        # Function to convert a SymPy lambdified function to a numpy function
        def convert_to_numpy_fn(lambda_fn):
            def numpy_fn(*param_values):
                try:
                    return float(lambda_fn(*param_values))
                except Exception:
                    return 0.0
            return numpy_fn
            
        # 1. Stationary matrices
        matrices = ['A', 'B', 'C', 'D']
        self._stationary_element_fns = {}
        
        for mat_name in matrices:
            if mat_name not in self._symbolic_data['stationary']['lambda_matrices']:
                continue
                
            lambda_data = self._symbolic_data['stationary']['lambda_matrices'][mat_name]
            elements = lambda_data['elements']
            grads = lambda_data['grads']
            
            self._stationary_element_fns[mat_name] = {
                'elements': [[convert_to_numpy_fn(fn) for fn in row] for row in elements],
                'grads': [[[convert_to_numpy_fn(grad_fn) for grad_fn in param_grads] 
                          for param_grads in row_grads] 
                        for row_grads in grads]
            }
            
        # 2. Trend matrices
        matrices = ['P_trends', 'Q_trends']
        self._trend_element_fns = {}
        
        for mat_name in matrices:
            if ('trend' not in self._symbolic_data or 
                'lambda_matrices' not in self._symbolic_data['trend'] or
                mat_name not in self._symbolic_data['trend']['lambda_matrices']):
                continue
                
            lambda_data = self._symbolic_data['trend']['lambda_matrices'][mat_name]
            elements = lambda_data['elements']
            grads = lambda_data['grads']
            
            self._trend_element_fns[mat_name] = {
                'elements': [[convert_to_numpy_fn(fn) for fn in row] for row in elements],
                'grads': [[[convert_to_numpy_fn(grad_fn) for grad_fn in param_grads] 
                          for param_grads in row_grads] 
                        for row_grads in grads]
            }
            
        # 3. Observation matrix
        if ('observation' in self._symbolic_data and 
            'lambda_matrices' in self._symbolic_data['observation'] and
            'Omega' in self._symbolic_data['observation']['lambda_matrices']):
            
            lambda_data = self._symbolic_data['observation']['lambda_matrices']['Omega']
            elements = lambda_data['elements']
            grads = lambda_data['grads']
            
            self._obs_element_fns = {
                'elements': [[convert_to_numpy_fn(fn) for fn in row] for row in elements],
                'grads': [[[convert_to_numpy_fn(grad_fn) for grad_fn in param_grads] 
                          for param_grads in row_grads] 
                        for row_grads in grads]
            }
            
    def _build_matrices_numpy(self, param_dict: Dict[str, float]):
        """
        Build model matrices using numpy evaluation functions.
        
        Args:
            param_dict: Dictionary of parameter values
            
        Returns:
            Dictionary of matrices
        """
        # Convert param_dict to parameter tuple
        param_tuple = tuple(param_dict.get(p_name, 0.0) for p_name in self.all_param_names)
        
        # 1. Stationary matrices
        A_unord = onp.zeros((self.n_stat, self.n_stat), dtype=onp.float64)
        B_unord = onp.zeros((self.n_stat, self.n_stat), dtype=onp.float64)
        C_unord = onp.zeros((self.n_stat, self.n_stat), dtype=onp.float64)
        D_unord = onp.zeros((self.n_stat, self.n_s_shock), dtype=onp.float64)
        
        if 'A' in self._stationary_element_fns:
            for i in range(min(self.n_stat, len(self._stationary_element_fns['A']['elements']))):
                for j in range(min(self.n_stat, len(self._stationary_element_fns['A']['elements'][i]))):
                    A_unord[i, j] = self._stationary_element_fns['A']['elements'][i][j](*param_tuple)
                    
        if 'B' in self._stationary_element_fns:
            for i in range(min(self.n_stat, len(self._stationary_element_fns['B']['elements']))):
                for j in range(min(self.n_stat, len(self._stationary_element_fns['B']['elements'][i]))):
                    B_unord[i, j] = self._stationary_element_fns['B']['elements'][i][j](*param_tuple)
                    
        if 'C' in self._stationary_element_fns:
            for i in range(min(self.n_stat, len(self._stationary_element_fns['C']['elements']))):
                for j in range(min(self.n_stat, len(self._stationary_element_fns['C']['elements'][i]))):
                    C_unord[i, j] = self._stationary_element_fns['C']['elements'][i][j](*param_tuple)
                    
        if 'D' in self._stationary_element_fns:
            for i in range(min(self.n_stat, len(self._stationary_element_fns['D']['elements']))):
                for j in range(min(self.n_s_shock, len(self._stationary_element_fns['D']['elements'][i]))):
                    D_unord[i, j] = self._stationary_element_fns['D']['elements'][i][j](*param_tuple)
        
        # Apply permutations
        A_ord = A_unord[onp.ix_(self.stat_eq_perm_indices, self.stat_var_perm_indices)]
        B_ord = B_unord[onp.ix_(self.stat_eq_perm_indices, self.stat_var_perm_indices)]
        C_ord = C_unord[onp.ix_(self.stat_eq_perm_indices, self.stat_var_perm_indices)]
        D_ord = D_unord[onp.ix_(self.stat_eq_perm_indices, onp.arange(D_unord.shape[1]))]
        
        # 2. Trend matrices
        P_trends = onp.zeros((self.n_trend, self.n_trend), dtype=onp.float64)
        Q_trends = onp.zeros((self.n_trend, self.n_t_shock), dtype=onp.float64)
        
        if 'P_trends' in self._trend_element_fns:
            for i in range(min(self.n_trend, len(self._trend_element_fns['P_trends']['elements']))):
                for j in range(min(self.n_trend, len(self._trend_element_fns['P_trends']['elements'][i]))):
                    P_trends[i, j] = self._trend_element_fns['P_trends']['elements'][i][j](*param_tuple)
                    
        if 'Q_trends' in self._trend_element_fns:
            for i in range(min(self.n_trend, len(self._trend_element_fns['Q_trends']['elements']))):
                for j in range(min(self.n_t_shock, len(self._trend_element_fns['Q_trends']['elements'][i]))):
                    Q_trends[i, j] = self._trend_element_fns['Q_trends']['elements'][i][j](*param_tuple)
        
        # 3. Observation matrix
        Omega = onp.zeros((self.n_obs, self.n_aug), dtype=onp.float64)
        
        if hasattr(self, '_obs_element_fns'):
            for i in range(min(self.n_obs, len(self._obs_element_fns['elements']))):
                for j in range(min(self.n_aug, len(self._obs_element_fns['elements'][i]))):
                    Omega[i, j] = self._obs_element_fns['elements'][i][j](*param_tuple)
        
        # Store gradients for later use (needed for autograd)
        self._last_gradients = {
            'A': onp.zeros((self.n_stat, self.n_stat, len(self.all_param_names)), dtype=onp.float64),
            'B': onp.zeros((self.n_stat, self.n_stat, len(self.all_param_names)), dtype=onp.float64),
            'C': onp.zeros((self.n_stat, self.n_stat, len(self.all_param_names)), dtype=onp.float64),
            'D': onp.zeros((self.n_stat, self.n_s_shock, len(self.all_param_names)), dtype=onp.float64),
            'P_trends': onp.zeros((self.n_trend, self.n_trend, len(self.all_param_names)), dtype=onp.float64),
            'Q_trends': onp.zeros((self.n_trend, self.n_t_shock, len(self.all_param_names)), dtype=onp.float64),
            'Omega': onp.zeros((self.n_obs, self.n_aug, len(self.all_param_names)), dtype=onp.float64)
        }
        
        # Calculate gradients for each matrix element
        # (This is computationally expensive but need to do it only once per solve)
        if 'A' in self._stationary_element_fns:
            for i in range(min(self.n_stat, len(self._stationary_element_fns['A']['elements']))):
                for j in range(min(self.n_stat, len(self._stationary_element_fns['A']['elements'][i]))):
                    for k in range(len(self.all_param_names)):
                        try:
                            self._last_gradients['A'][i, j, k] = self._stationary_element_fns['A']['grads'][i][j][k](*param_tuple)
                        except:
                            pass
        
        # ... similarly for other matrices
        
        return {
            'A_ord': jnp.array(A_ord),
            'B_ord': jnp.array(B_ord),
            'C_ord': jnp.array(C_ord),
            'D_ord': jnp.array(D_ord),
            'P_trends': jnp.array(P_trends),
            'Q_trends': jnp.array(Q_trends),
            'Omega': jnp.array(Omega)
        }
            
    def solve(self, param_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Solve the model for given parameters.
        
        Args:
            param_dict: Dictionary of parameter values
            
        Returns:
            Dictionary containing the solution matrices and metadata
        """
        if not self._parsed:
            self._parse_and_generate_element_functions()
            
        # Default result (used for errors)
        results = {"solution_valid": jnp.array(False)}
        
        try:
            # Step 1: Build matrices using numpy functions
            matrices = self._build_matrices_numpy(param_dict)
            
            A_num_stat = matrices['A_ord'] 
            B_num_stat = matrices['B_ord']
            C_num_stat = matrices['C_ord']
            D_num_stat = matrices['D_ord']
            P_num_trend = matrices['P_trends']
            Q_num_trend = matrices['Q_trends']
            Omega_num = matrices['Omega']
            
            # Check matrices for NaNs/Infs
            if not (jnp.all(jnp.isfinite(A_num_stat)) and 
                    jnp.all(jnp.isfinite(B_num_stat)) and
                    jnp.all(jnp.isfinite(C_num_stat)) and
                    jnp.all(jnp.isfinite(D_num_stat))):
                if self.verbose:
                    print("ERROR: NaN/Inf in stationary matrices.")
                return results
                
            # Step 2: Solve stationary model (SDA)
            P_sol_stat, _, _, converged_stat = solve_quadratic_matrix_equation_jax(
                A_num_stat, B_num_stat, C_num_stat, tol=1e-12, max_iter=500
            )
            
            valid_stat_solve = converged_stat & jnp.all(jnp.isfinite(P_sol_stat))
            
            if not valid_stat_solve:
                if self.verbose:
                    print("ERROR: SDA solver failed or P_sol_stat contains NaN/Inf.")
                return results
                
            # Step 3: Compute Q_stationary
            Q_sol_stat = compute_Q_jax(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
            valid_q_compute = jnp.all(jnp.isfinite(Q_sol_stat))
            
            if not valid_q_compute:
                if self.verbose:
                    print("ERROR: Q_sol_stat computation resulted in NaN/Inf.")
                return results
                
            # Check trend matrices
            if not (jnp.all(jnp.isfinite(P_num_trend)) and jnp.all(jnp.isfinite(Q_num_trend))):
                if self.verbose:
                    print("ERROR: NaN/Inf in trend matrices.")
                return results
                
            # Check observation matrix
            if not jnp.all(jnp.isfinite(Omega_num)):
                if self.verbose:
                    print("ERROR: NaN/Inf in Omega matrix.")
                return results
                
            # Build R matrices and augmented system
            shock_std_devs = {}
            for shock_name in self.aug_shocks:
                sigma_param_name = f"sigma_{shock_name}"
                if sigma_param_name in param_dict:
                    std_dev = param_dict[sigma_param_name]
                else:
                    # Use default if not provided
                    std_dev = self.default_param_assignments.get(sigma_param_name, 0.01)
                shock_std_devs[shock_name] = jnp.maximum(jnp.abs(std_dev), 1e-9)
                
            # Standard deviations arrays
            stat_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.stat_shocks], 
                                         dtype=_DEFAULT_DTYPE) if self.stat_shocks else jnp.array([], dtype=_DEFAULT_DTYPE)
            trend_std_devs_arr = jnp.array([shock_std_devs[shk] for shk in self.trend_shocks], 
                                          dtype=_DEFAULT_DTYPE) if self.trend_shocks else jnp.array([], dtype=_DEFAULT_DTYPE)
            
            # R matrices
            R_sol_stat = jnp.zeros((self.n_stat, self.n_s_shock), dtype=_DEFAULT_DTYPE)
            if self.n_s_shock > 0 and Q_sol_stat.shape == (self.n_stat, self.n_s_shock):
                R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs_arr)
                
            R_num_trend = jnp.zeros((self.n_trend, self.n_t_shock), dtype=_DEFAULT_DTYPE)
            if self.n_t_shock > 0 and Q_num_trend.shape == (self.n_trend, self.n_t_shock):
                R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs_arr)
                
            # Build augmented matrices
            # P_aug is block diagonal matrix of P_sol_stat and P_num_trend
            P_aug = jnp.zeros((self.n_aug, self.n_aug), dtype=_DEFAULT_DTYPE)
            if P_sol_stat.shape == (self.n_stat, self.n_stat):
                P_aug = P_aug.at[:self.n_stat, :self.n_stat].set(P_sol_stat)
            if P_num_trend.shape == (self.n_trend, self.n_trend) and self.n_trend > 0:
                P_aug = P_aug.at[self.n_stat:, self.n_stat:].set(P_num_trend)
                
            # R_aug combines R_sol_stat and R_num_trend
            R_aug = jnp.zeros((self.n_aug, self.n_aug_shock), dtype=_DEFAULT_DTYPE)
            if self.n_stat > 0 and self.n_s_shock > 0 and R_sol_stat.shape == (self.n_stat, self.n_s_shock):
                R_aug = R_aug.at[:self.n_stat, :self.n_s_shock].set(R_sol_stat)
            if self.n_trend > 0 and self.n_t_shock > 0 and R_num_trend.shape == (self.n_trend, self.n_t_shock):
                R_aug = R_aug.at[self.n_stat:, self.n_s_shock:].set(R_num_trend)
                
            # Final checks
            all_finite = (jnp.all(jnp.isfinite(P_aug)) & 
                         jnp.all(jnp.isfinite(R_aug)) & 
                         jnp.all(jnp.isfinite(Omega_num)))
            solution_valid_final = valid_stat_solve & valid_q_compute & all_finite
            
            if not solution_valid_final:
                if self.verbose:
                    print("ERROR: Final matrices contain NaN/Inf.")
                return results
                
            # Return the full solution
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
            
            # Fill in error matrices with NaNs
            results["P_aug"] = jnp.full((self.n_aug, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
            results["R_aug"] = jnp.full((self.n_aug, self.n_aug_shock), jnp.nan, dtype=_DEFAULT_DTYPE)
            results["Omega"] = jnp.full((self.n_obs, self.n_aug), jnp.nan, dtype=_DEFAULT_DTYPE)
            
        # Ensure solution_valid is boolean type
        results["solution_valid"] = jnp.asarray(results["solution_valid"], dtype=jnp.bool_)
        
        return results
        
    def log_likelihood(self, param_dict: Dict[str, float], ys: jax.Array,
                      H_obs: jax.Array, init_x_mean: jax.Array, init_P_cov: jax.Array) -> jax.Array:
        """
        Calculate log-likelihood using Kalman filter.
        
        Args:
            param_dict: Parameter dictionary
            ys: Observations array
            H_obs: Observation noise covariance
            init_x_mean: Initial state mean
            init_P_cov: Initial state covariance
            
        Returns:
            Log-likelihood value
        """
        if not KALMAN_FILTER_JAX_AVAILABLE:
            raise RuntimeError("Custom KalmanFilter required.")
            
        LARGE_NEG_VALUE = -1e10
        
        # Solve the model
        solution = self.solve(param_dict)
        
        def _calculate_likelihood(sol):
            # Check if solution is valid
            if not (jnp.all(jnp.isfinite(sol["P_aug"])) and 
                   jnp.all(jnp.isfinite(sol["R_aug"])) and 
                   jnp.all(jnp.isfinite(sol["Omega"]))):
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
                
            # Check dimensions
            expected_P_shape = (self.n_aug, self.n_aug)
            expected_R_shape = (self.n_aug, self.n_aug_shock)
            expected_Omega_shape = (self.n_obs, self.n_aug)
            expected_H_shape = (self.n_obs, self.n_obs)
            expected_init_x_shape = (self.n_aug,)
            expected_init_P_shape = (self.n_aug, self.n_aug)
            
            # Get actual shapes
            P_shape = sol["P_aug"].shape
            R_shape = sol["R_aug"].shape
            Omega_shape = sol["Omega"].shape
            H_shape = H_obs.shape
            init_x_shape = init_x_mean.shape
            init_P_shape = init_P_cov.shape
            
            # Check shapes
            shapes_ok = (
                P_shape == expected_P_shape and
                R_shape == expected_R_shape and
                Omega_shape == expected_Omega_shape and
                H_shape == expected_H_shape and
                init_x_shape == expected_init_x_shape and
                init_P_shape == expected_init_P_shape and
                (ys.ndim == 1 and ys.shape[0] == self.n_obs or 
                 ys.ndim == 2 and ys.shape[1] == self.n_obs)
            )
            
            if not shapes_ok:
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
                
            try:
                # Initialize Kalman filter
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
                
                # Ensure value is finite
                safe_log_prob = jnp.where(
                    jnp.isfinite(log_prob),
                    log_prob,
                    jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
                )
                
                return safe_log_prob
                
            except Exception as e:
                if self.verbose:
                    print(f"Exception in Kalman filter: {e}")
                return jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE)
                
        # Use solution_valid to choose between log-likelihood calculation and error value
        return jax.lax.cond(
            solution["solution_valid"],
            _calculate_likelihood,
            lambda _: jnp.array(LARGE_NEG_VALUE, dtype=_DEFAULT_DTYPE),
            solution
        )

# --- Numpyro Model Function ---
def numpyro_model_with_numpy_functions(
    model_instance: DynareModelImproved,
    user_priors: List[Dict[str, Any]],
    fixed_param_values: Dict[str, float],
    ys: Optional[jax.Array] = None,
    H_obs: Optional[jax.Array] = None,
    init_x_mean: Optional[jax.Array] = None,
    init_P_cov: Optional[jax.Array] = None
):
    """
    Numpyro model using the improved Dynare model with numpy-based functions.
    
    Args:
        model_instance: DynareModelImproved instance
        user_priors: List of prior specifications
        fixed_param_values: Dictionary of fixed parameter values
        ys: Observations array (optional)
        H_obs: Observation noise covariance (optional)
        init_x_mean: Initial state mean (optional)
        init_P_cov: Initial state covariance (optional)
    """
    if not NUMPYRO_AVAILABLE:
        raise RuntimeError("Numpyro required.")
        
    # Sample parameters from priors
    params_for_likelihood = {}
    estimated_param_names = {p["name"] for p in user_priors}
    
    # Process each prior specification
    for prior_spec in user_priors:
        name = prior_spec["name"]
        dist_name = prior_spec.get("prior", "").lower()
        args = prior_spec.get("args", {})
        
        # Process distribution arguments
        dist_args_proc = {
            k: jnp.asarray(v, dtype=_DEFAULT_DTYPE) for k, v in args.items()
        }
        
        # Sample from the appropriate distribution
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
            
        params_for_likelihood[name] = val
        
    # Add fixed parameters
    for name, value in fixed_param_values.items():
        if name not in estimated_param_names:
            params_for_likelihood[name] = jnp.asarray(value, dtype=_DEFAULT_DTYPE)
            
    # Calculate log-likelihood if observations are provided
    if ys is not None:
        if H_obs is None or init_x_mean is None or init_P_cov is None:
            raise ValueError("H_obs, init_x_mean, init_P_cov required with ys.")
            
        # Compute log-likelihood using the model
        log_prob = model_instance.log_likelihood(
            params_for_likelihood, ys, H_obs, init_x_mean, init_P_cov
        )
        
        # Add log-likelihood to model
        numpyro.factor("log_likelihood", log_prob)

# --- Main Execution Block ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_mod_file = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
    mod_file_path = os.environ.get("DYNARE_MOD_FILE", default_mod_file)

    # Simulation parameters
    num_sim_steps = 200
    sim_seed = 123
    sim_measurement_noise_std = 0.001
    
    # MCMC parameters
    run_estimation_flag = True
    mcmc_seed = 456
    mcmc_chains = 1
    mcmc_warmup = 250
    mcmc_samples = 500
    mcmc_target_accept = 0.85

    print(f"\n--- [1] Initializing Improved Dynare Model ({mod_file_path}) ---")
    try:
        model = DynareModelImproved(mod_file_path, verbose=True)
        print(f"Model initialized. Found {len(model.all_param_names)} parameters total.")
    except Exception as e_init:
        print(f"FATAL: Failed to initialize DynareModelImproved: {e_init}")
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
            print("FATAL: Cannot solve model with simulation parameters.")
            print(f"Solution content: {sim_solution}")
            exit()
        else:
            print("  Solve for simulation parameters was successful.")
    except Exception as e_solve:
        print(f"FATAL: Exception during solve for simulation: {e_solve}")
        import traceback
        traceback.print_exc()
        exit()

    # Construct initial state
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
    
    # Observation noise covariance
    H_obs_sim = jnp.eye(sim_solution["n_obs"], dtype=_DEFAULT_DTYPE) * (sim_measurement_noise_std**2)
    
    # Simulate state-space model
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
    
    # Optional: Plot simulations
    # plot_simulation_with_trends_matched(
    #     sim_observables, 
    #     sim_solution["ordered_obs_vars"], 
    #     sim_states,
    #     sim_solution["aug_state_vars"], 
    #     model.ordered_trend_state_vars, 
    #     model.contemp_trend_defs,
    #     title="Simulated Data with Trends"
    # )

    print("\n--- [3] Defining Priors for Estimation ---")
    user_priors = [
        {"name": "sigma_SHK_RS", "prior": "invgamma", "args": {"concentration": 3.0, "scale": 0.2}}
    ]
    estimated_param_names_set = {p["name"] for p in user_priors}
    fixed_params = {
        name: val for name, val in model.default_param_assignments.items() 
        if name not in estimated_param_names_set
    }

    if run_estimation_flag and NUMPYRO_AVAILABLE and KALMAN_FILTER_JAX_AVAILABLE:
        print(f"\n--- [4] Running Bayesian Estimation ---")
        mcmc_key = random.PRNGKey(mcmc_seed)
        
        # Setup for estimation
        H_obs_est = H_obs_sim
        init_x_mean_est = s0_sim
        init_P_cov_est = jnp.eye(sim_solution["n_aug"], dtype=_DEFAULT_DTYPE) * 1.0
        
        # Initial values for MCMC
        init_values_mcmc = {
            p["name"]: sim_param_values.get(p["name"], 0.1) for p in user_priors
        }
        
        # Full parameter set (estimated + fixed)
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
                
        except Exception as e_ll:
            print(f"ERROR evaluating initial log-likelihood: {e_ll}")
            import traceback
            traceback.print_exc()
            exit()

        # Setup NUTS sampler and MCMC
        kernel = NUTS(
            numpyro_model_with_numpy_functions,
            init_strategy=init_to_value(values=init_values_mcmc),
            target_accept_prob=mcmc_target_accept
        )
        
        mcmc = MCMC(
            kernel,
            num_warmup=mcmc_warmup,
            num_samples=mcmc_samples,
            num_chains=mcmc_chains,
            progress_bar=True,
            chain_method='parallel'
        )
        
        print(f"Starting MCMC sampling...")
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
            
            # Print results
            mcmc.print_summary()
            
            # Optional: Plot posterior distributions
            # samples = mcmc.get_samples()
            # plt.figure(figsize=(10, 6))
            # for i, param_name in enumerate(samples.keys()):
            #     plt.subplot(len(samples), 1, i+1)
            #     plt.hist(samples[param_name], bins=30, alpha=0.7)
            #     plt.title(f"Posterior for {param_name}")
            # plt.tight_layout()
            # plt.show()
            
        except Exception as e_mcmc:
            print(f"\n--- MCMC FAILED ---")
            print(f"{type(e_mcmc).__name__}: {e_mcmc}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Estimation ---")
        
    print(f"\n--- Script finished successfully ---")