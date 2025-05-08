# --- START OF FILE Kalman_filter_jax.py (Modified Jitter) ---

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp # Using onp for clarity with NumPy operations
from typing import Tuple, Optional, Union, Sequence, Dict, Any

_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps
# Define a consistent, slightly larger jitter value for KF stability
_KF_JITTER = 1e-8

class KalmanFilter:
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype) # This is C_full
        self.H = jnp.asarray(H, dtype=desired_dtype) # This is H_full
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P, dtype=desired_dtype)

        n_state = self.T.shape[0]
        n_obs_full = self.C.shape[0] # Full number of observables
        n_shocks = self.R.shape[1] if self.R.ndim == 2 and self.R.shape[1] > 0 else 0

        # Validations use n_state and n_obs_full
        if self.T.shape != (n_state, n_state): raise ValueError(f"T shape mismatch")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks): raise ValueError(f"R shape mismatch")
        elif n_shocks == 0 and self.R.size != 0 : self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        if self.C.shape != (n_obs_full, n_state): raise ValueError(f"C shape mismatch")
        if self.H.shape != (n_obs_full, n_obs_full): raise ValueError(f"H shape mismatch")
        if self.init_x.shape != (n_state,): raise ValueError(f"init_x shape mismatch")
        if self.init_P.shape != (n_state, n_state): raise ValueError(f"init_P shape mismatch")

        self.n_state = n_state
        self.n_obs_full = n_obs_full # Store full n_obs for reference
        self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)

        if self.n_shocks > 0: self.state_cov = self.R @ self.R.T
        else: self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)
        
        # Cholesky for observation noise simulation (uses full H)
        # Use slightly larger jitter for H Cholesky as well
        try:
            # Increased jitter here too
            H_reg_chol = self.H + _KF_JITTER * jnp.eye(self.n_obs_full, dtype=desired_dtype) 
            self.L_H_full = jnp.linalg.cholesky(H_reg_chol) # For simulating full observation noise vector
            self.simulate_obs_noise_internal = self._simulate_obs_noise_chol_internal
        except Exception:
            # Ensure H_stable_full also uses consistent jitter if fallback needed
            self.H_stable_full = self.H + _KF_JITTER * jnp.eye(self.n_obs_full, dtype=desired_dtype)
            self.simulate_obs_noise_internal = self._simulate_obs_noise_mvn_internal


    def _simulate_obs_noise_chol_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        # Simulates noise for all n_obs_full components
        z_eta = random.normal(key, tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)
        return z_eta @ self.L_H_full.T

    def _simulate_obs_noise_mvn_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try:
            return random.multivariate_normal(
                key, jnp.zeros((self.n_obs_full,), dtype=self.H.dtype),
                self.H_stable_full, shape=mvn_shape, dtype=self.H.dtype
            )
        except Exception:
            return jnp.zeros(tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)


    def filter(self,
               ys: ArrayLike, # Full observations array (T, n_obs_full)
               static_valid_obs_idx: jax.Array, 
               static_n_obs_actual: int,        
               static_C_obs: jax.Array,         
               static_H_obs: jax.Array,         
               static_I_obs: jax.Array          # Identity matrix of size n_obs_actual
              ) -> Dict[str, jax.Array]:
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype) 
        T_mat, I_s = self.T, self.I_s
        state_cov = self.state_cov
        # Use the defined jitter
        kf_jitter = _KF_JITTER
        MAX_STATE_VALUE = 1e6 # Clip state values for stability
        T_steps = ys_arr.shape[0]

        if T_steps == 0:
            return { 
                'x_pred': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_pred': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'x_filt': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_filt': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'innovations': jnp.empty((0, static_n_obs_actual), dtype=self.C.dtype),
                'innovation_cov': jnp.empty((0, static_n_obs_actual, static_n_obs_actual), dtype=self.C.dtype),
                'log_likelihood_contributions': jnp.empty((0,), dtype=I_s.dtype)
            }

        # Kalman Filter Step Function (uses passed-in static matrices)
        def step_static_nan(carry, y_t_full_slice): 
            x_prev_filt, P_prev_filt = carry
            
            # --- Prediction Step ---
            x_pred_t = T_mat @ x_prev_filt
            x_pred_t = jnp.clip(x_pred_t, -MAX_STATE_VALUE, MAX_STATE_VALUE)
            
            # Ensure P_prev_filt is symmetric and add jitter before propagating
            P_prev_filt_sym = (P_prev_filt + P_prev_filt.T) / 2.0
            P_prev_filt_reg = P_prev_filt_sym + kf_jitter * I_s # Use kf_jitter
            
            P_pred_t = T_mat @ P_prev_filt_reg @ T_mat.T + state_cov
            P_pred_t = (P_pred_t + P_pred_t.T) / 2.0 # Ensure symmetry
            P_pred_t = P_pred_t + kf_jitter * I_s # Add jitter *after* prediction
            
            # --- Update Step ---
            y_obs_t = jnp.take(y_t_full_slice, static_valid_obs_idx, axis=0) if static_n_obs_actual > 0 else jnp.empty((0,), dtype=y_t_full_slice.dtype)
            
            y_pred_obs = static_C_obs @ x_pred_t
            v_obs = y_obs_t - y_pred_obs
            PCt_obs = P_pred_t @ static_C_obs.T 
            S_obs = static_C_obs @ PCt_obs + static_H_obs
            S_obs_reg = S_obs + kf_jitter * static_I_obs # Use kf_jitter for S regularization

            K = jnp.zeros((self.n_state, static_n_obs_actual), dtype=x_pred_t.dtype)
            solve_ok = jnp.array(False) # Flag for successful solve/pinv

            if static_n_obs_actual > 0:
                # Try Cholesky solve first
                try:
                    # Use assume_a='pos' for Cholesky based on regularization
                    L_S_obs = jnp.linalg.cholesky(S_obs_reg) 
                    K_T_temp = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs.T, lower=True, trans='N')
                    K = jax.scipy.linalg.solve_triangular(L_S_obs, K_T_temp, lower=True, trans='T').T
                    solve_ok = jnp.all(jnp.isfinite(K))
                except Exception:
                    solve_ok = jnp.array(False) # Cholesky failed

                # Fallback to standard solve if Cholesky failed or produced NaNs
                def fallback_solve(operand):
                    PCt_obs_op, S_obs_reg_op = operand
                    try:
                        # assume_a='pos' is reasonable given regularization
                        K_fallback = jax.scipy.linalg.solve(S_obs_reg_op, PCt_obs_op.T, assume_a='pos').T 
                        return K_fallback, jnp.all(jnp.isfinite(K_fallback))
                    except Exception:
                        return jnp.zeros_like(PCt_obs_op.T).T, jnp.array(False)
                
                # Fallback to pinv if standard solve failed
                def fallback_pinv(operand):
                     PCt_obs_op, S_obs_reg_op = operand
                     try:
                         # Use a slightly larger rcond for pinv stability maybe?
                         K_pinv = PCt_obs_op @ jnp.linalg.pinv(S_obs_reg_op, rcond=1e-6) 
                         return K_pinv, jnp.all(jnp.isfinite(K_pinv))
                     except Exception:
                         return jnp.zeros_like(PCt_obs_op.T).T, jnp.array(False)

                # Conditional execution for fallbacks
                K, solve_ok = jax.lax.cond(solve_ok, 
                                           lambda op: (op[0], op[1]), # Return K and solve_ok from Cholesky
                                           fallback_solve, 
                                           (K, solve_ok), # Pass current K and solve_ok
                                           operand=(PCt_obs, S_obs_reg))
                
                K, solve_ok = jax.lax.cond(solve_ok, 
                                           lambda op: (op[0], op[1]), # Return K and solve_ok from solve
                                           fallback_pinv, 
                                           (K, solve_ok), # Pass current K and solve_ok
                                           operand=(PCt_obs, S_obs_reg))

                # Clip gain for stability, even if solve seemed ok
                K = jnp.clip(K, -1e3, 1e3) 
            
            # Update state and covariance
            x_filt_t = x_pred_t
            P_filt_t = P_pred_t
            # Use solve_ok flag to potentially skip update if K is invalid
            def perform_update(operand):
                K_op, v_obs_op, x_pred_op, P_pred_op, C_obs_op, H_obs_op, I_s_op = operand
                x_update = K_op @ v_obs_op
                x_filt_op = x_pred_op + x_update
                x_filt_op = jnp.clip(x_filt_op, -MAX_STATE_VALUE, MAX_STATE_VALUE)
                IKC_obs = I_s_op - K_op @ C_obs_op
                P_filt_op = IKC_obs @ P_pred_op @ IKC_obs.T + K_op @ H_obs_op @ K_op.T
                P_filt_op = (P_filt_op + P_filt_op.T) / 2.0
                P_filt_op = P_filt_op + kf_jitter * I_s_op # Add jitter after update
                return x_filt_op, P_filt_op

            def skip_update(operand):
                 # Return predicted state if K is invalid
                 _, _, x_pred_op, P_pred_op, _, _, _ = operand
                 return x_pred_op, P_pred_op # Or potentially P_pred_op + more jitter?

            # Only update if n_obs > 0 AND the solve for K was successful
            update_condition = (static_n_obs_actual > 0) & solve_ok 
            x_filt_t, P_filt_t = jax.lax.cond(
                update_condition,
                perform_update,
                skip_update,
                operand=(K, v_obs, x_pred_t, P_pred_t, static_C_obs, static_H_obs, I_s)
            )

            # --- Log Likelihood Contribution ---
            ll_t = 0.0
            if static_n_obs_actual > 0:
                try:
                    # Use slogdet on the regularized S_obs_reg
                    sign, log_det_S_obs = jnp.linalg.slogdet(S_obs_reg) 
                    
                    # Use Cholesky solve for Mahalanobis distance for stability
                    L_S_obs_ll = jnp.linalg.cholesky(S_obs_reg)
                    z = jax.scipy.linalg.solve_triangular(L_S_obs_ll, v_obs, lower=True)
                    mahalanobis_dist_obs = jnp.sum(z**2)
                    
                    log_pi_term = jnp.log(2 * jnp.pi) * static_n_obs_actual
                    ll_term = -0.5 * (log_pi_term + log_det_S_obs + mahalanobis_dist_obs)
                    
                    # Penalize if slogdet sign is non-positive or if Mahalanobis calculation failed (NaN)
                    valid_ll_calc = (sign > 0) & jnp.isfinite(mahalanobis_dist_obs) & jnp.isfinite(log_det_S_obs)
                    ll_t = jnp.where(valid_ll_calc, ll_term, -1e6) # Use large penalty
                    
                except Exception: # Catch Cholesky or other errors during LL calculation
                    ll_t = jnp.array(-1e6) # Assign large penalty on exception
            
            ll_t = jnp.where(jnp.isfinite(ll_t), ll_t, -1e6) # Final safeguard

            # --- Prepare Outputs ---
            # Ensure outputs are finite, otherwise default to safe values (zeros or identity)
            outputs = {
                'x_pred': jnp.where(jnp.isfinite(x_pred_t), x_pred_t, jnp.zeros_like(x_pred_t)), 
                'P_pred': jnp.where(jnp.isfinite(P_pred_t), P_pred_t, jnp.eye(self.n_state, dtype=P_pred_t.dtype) * 1e6), # Large P if NaN
                'x_filt': jnp.where(jnp.isfinite(x_filt_t), x_filt_t, jnp.zeros_like(x_filt_t)), 
                'P_filt': jnp.where(jnp.isfinite(P_filt_t), P_filt_t, jnp.eye(self.n_state, dtype=P_filt_t.dtype) * 1e6), # Large P if NaN
                'innovations': jnp.where(jnp.isfinite(v_obs), v_obs, jnp.zeros_like(v_obs)), 
                'innovation_cov': jnp.where(jnp.isfinite(S_obs_reg), S_obs_reg, jnp.eye(static_n_obs_actual, dtype=S_obs_reg.dtype) * 1e6), # Large S if NaN
                'log_likelihood_contributions': ll_t # Already safeguarded
            }
            
            x_filt_t_safe = outputs['x_filt']
            P_filt_t_safe = outputs['P_filt']
            # Ensure P is symmetric for next step (might be redundant if already done in update)
            P_filt_t_safe = (P_filt_t_safe + P_filt_t_safe.T) / 2.0 
            
            return (x_filt_t_safe, P_filt_t_safe), outputs

        init_carry = (self.init_x, self.init_P)
        (_, _), scan_outputs = lax.scan(step_static_nan, init_carry, ys_arr) 
        
        # Final check on scan outputs (might be redundant given checks inside step)
        for key_final, val_final_arr in scan_outputs.items():
            if key_final == 'log_likelihood_contributions': 
                scan_outputs[key_final] = jnp.where(jnp.isfinite(val_final_arr), val_final_arr, jnp.full_like(val_final_arr, -1e6))
            elif hasattr(val_final_arr, 'size') and val_final_arr.size > 0: 
                # Check if it's a covariance matrix
                if val_final_arr.ndim == 3 and val_final_arr.shape[-1] == val_final_arr.shape[-2]:
                     default_val = jnp.eye(val_final_arr.shape[-1], dtype=val_final_arr.dtype) * 1e6
                     scan_outputs[key_final] = jnp.where(jnp.isfinite(val_final_arr), val_final_arr, default_val)
                else: # State or innovation vector
                     scan_outputs[key_final] = jnp.where(jnp.isfinite(val_final_arr), val_final_arr, jnp.zeros_like(val_final_arr))
        return scan_outputs

    def log_likelihood(self,
                       ys: ArrayLike,
                       static_valid_obs_idx: jax.Array,
                       static_n_obs_actual: int,
                       static_C_obs: jax.Array,
                       static_H_obs: jax.Array,
                       static_I_obs: jax.Array
                      ) -> jax.Array:
        filter_results = self.filter(ys, static_valid_obs_idx, static_n_obs_actual,
                                     static_C_obs, static_H_obs, static_I_obs)
        total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
        # Ensure final sum is also finite
        return jnp.where(jnp.isfinite(total_log_likelihood), total_log_likelihood, -1e10)


    def smooth(self,
               ys: ArrayLike,
               filter_results: Optional[Dict] = None,
               # Static NaN info, only needed if filter_results is None
               static_valid_obs_idx: Optional[jax.Array] = None,
               static_n_obs_actual: Optional[int] = None,
               static_C_obs_for_filter: Optional[jax.Array] = None, 
               static_H_obs_for_filter: Optional[jax.Array] = None, 
               static_I_obs_for_filter: Optional[jax.Array] = None
              ) -> Tuple[jax.Array, jax.Array]:
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        if filter_results is None:
            if static_valid_obs_idx is None or \
               static_n_obs_actual is None or \
               static_C_obs_for_filter is None or \
               static_H_obs_for_filter is None or \
               static_I_obs_for_filter is None:
                raise ValueError("Static NaN info must be provided to smooth() if filter_results is None.")
            filter_outs_dict = self.filter(ys_arr, static_valid_obs_idx, static_n_obs_actual,
                                           static_C_obs_for_filter, static_H_obs_for_filter, static_I_obs_for_filter)
        else:
            filter_outs_dict = filter_results

        x_pred = filter_outs_dict['x_pred']; P_pred = filter_outs_dict['P_pred']
        x_filt = filter_outs_dict['x_filt']; P_filt = filter_outs_dict['P_filt']
        T_mat = self.T; N = x_filt.shape[0]
        if N == 0: return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)
        
        x_s_next = x_filt[-1]; P_s_next = P_filt[-1]
        
        # Ensure inputs to scan are finite, replace with defaults if necessary
        P_pred_safe = jnp.where(jnp.isfinite(P_pred), P_pred, jnp.eye(self.n_state, dtype=P_pred.dtype)*1e6)
        P_filt_safe = jnp.where(jnp.isfinite(P_filt), P_filt, jnp.eye(self.n_state, dtype=P_filt.dtype)*1e6)
        x_pred_safe = jnp.where(jnp.isfinite(x_pred), x_pred, jnp.zeros_like(x_pred))
        x_filt_safe = jnp.where(jnp.isfinite(x_filt), x_filt, jnp.zeros_like(x_filt))

        P_pred_for_scan = P_pred_safe[1:N]; P_filt_for_scan = P_filt_safe[0:N-1]
        x_pred_for_scan = x_pred_safe[1:N]; x_filt_for_scan = x_filt_safe[0:N-1]
        scan_inputs = (P_pred_for_scan[::-1], P_filt_for_scan[::-1], x_pred_for_scan[::-1], x_filt_for_scan[::-1])
        
        kf_jitter = _KF_JITTER # Use consistent jitter

        def backward_step(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            
            # Ensure Pf_t is symmetric before use
            Pf_t_sym = (Pf_t + Pf_t.T) / 2.0
            
            Pp_next_reg = Pp_next_t + kf_jitter * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)
            
            Jt = jnp.zeros((self.n_state, self.n_state), dtype=Pf_t.dtype)
            solve_J_ok = jnp.array(False)
            try:
                # Try direct solve
                Jt_transpose = jax.scipy.linalg.solve(Pp_next_reg, (T_mat @ Pf_t_sym).T, assume_a='pos') # assume pos due to jitter
                Jt = Jt_transpose.T
                solve_J_ok = jnp.all(jnp.isfinite(Jt))
            except Exception:
                 solve_J_ok = jnp.array(False)

            def fallback_pinv_J(operand):
                T_mat_op, Pf_t_sym_op, Pp_next_reg_op = operand
                try:
                    Jt_pinv = (Pf_t_sym_op @ T_mat_op.T) @ jnp.linalg.pinv(Pp_next_reg_op, rcond=1e-6)
                    return Jt_pinv, jnp.all(jnp.isfinite(Jt_pinv))
                except Exception:
                    return jnp.zeros_like(Pf_t_sym_op), jnp.array(False)
            
            # Use fallback if direct solve failed
            Jt, solve_J_ok = jax.lax.cond(
                solve_J_ok,
                lambda op: (op[0], op[1]),
                fallback_pinv_J,
                (Jt, solve_J_ok),
                operand=(T_mat, Pf_t_sym, Pp_next_reg)
            )

            # Calculate smoothed state/covariance, potentially using Jt=0 if solve failed
            x_diff = x_s_next_t - xp_next_t
            x_s_t = xf_t + Jt @ x_diff
            
            P_diff = P_s_next_t - Pp_next_t
            P_s_t = Pf_t_sym + Jt @ P_diff @ Jt.T
            P_s_t = (P_s_t + P_s_t.T) / 2.0 # Ensure symmetry
            
            # Safeguard outputs
            x_s_t = jnp.where(jnp.isfinite(x_s_t), x_s_t, jnp.zeros_like(x_s_t))
            P_s_t = jnp.where(jnp.isfinite(P_s_t), P_s_t, jnp.eye(self.n_state, dtype=P_s_t.dtype)*1e6)
            P_s_t = (P_s_t + P_s_t.T) / 2.0 # Re-ensure symmetry after where

            return (x_s_t, P_s_t), (x_s_t, P_s_t)
        
        init_carry_smooth = (x_s_next, P_s_next)
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)
        
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt_safe[N-1:N]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt_safe[N-1:N]], axis=0)
        
        # Final check on smoothed outputs
        x_smooth = jnp.where(jnp.isfinite(x_smooth), x_smooth, jnp.zeros_like(x_smooth))
        P_smooth = jnp.where(jnp.isfinite(P_smooth), P_smooth, jnp.eye(self.n_state, dtype=P_smooth.dtype)*1e6)

        return x_smooth, P_smooth

    def _simulation_smoother_single_draw(self,
                                       ys: jax.Array,
                                       key: jax.random.PRNGKey,
                                       filter_results_orig: Optional[Dict] = None,
                                       # Must pass static NaN info for original ys if filter_results_orig is None
                                       static_valid_obs_idx_orig: Optional[jax.Array] = None,
                                       static_n_obs_actual_orig: Optional[int] = None,
                                       static_C_obs_orig: Optional[jax.Array] = None,
                                       static_H_obs_orig: Optional[jax.Array] = None,
                                       static_I_obs_orig: Optional[jax.Array] = None
                                       ) -> jax.Array:
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        n_s, n_eps_shocks = self.n_state, self.n_shocks
        kf_jitter = _KF_JITTER

        # --- Get RTS Smoothed States for Original Data ---
        # This section needs the static info if filter_results_orig is None
        if filter_results_orig is None or 'x_filt' not in filter_results_orig:
            if static_valid_obs_idx_orig is None or \
               static_n_obs_actual_orig is None or \
               static_C_obs_orig is None or \
               static_H_obs_orig is None or \
               static_I_obs_orig is None:
                 raise ValueError("Static NaN info for original ys must be provided to _simulation_smoother_single_draw if filter_results_orig is None.")
            filter_results_for_rts = self.filter(ys, static_valid_obs_idx_orig, static_n_obs_actual_orig,
                                                 static_C_obs_orig, static_H_obs_orig, static_I_obs_orig)
        else:
            filter_results_for_rts = filter_results_orig

        # Call smooth - it will use filter_results_for_rts
        x_smooth_rts, _ = self.smooth(ys, filter_results=filter_results_for_rts) # No need to pass static info again here

        # --- Simulate States (x_star) ---
        key_init, key_eps, key_eta = random.split(key, 3)
        try:
            init_P_reg = self.init_P + kf_jitter * jnp.eye(n_s, dtype=self.init_P.dtype) # Use jitter
            L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
            x0_star = self.init_x + L0 @ z0
        except Exception: x0_star = self.init_x # Fallback

        eps_star_sim = random.normal(key_eps, (Tsteps, n_eps_shocks), dtype=self.R.dtype) if n_eps_shocks > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
        
        def state_sim_step(x_prev_star, eps_t_star_arg):
            shock_term = self.R @ eps_t_star_arg if n_eps_shocks > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
            x_curr_star = self.T @ x_prev_star + shock_term
            # Clip simulated states for stability before generating observations
            x_curr_star = jnp.clip(x_curr_star, -1e6, 1e6) 
            return x_curr_star, x_curr_star
            
        _, x_star_path = lax.scan(state_sim_step, x0_star, eps_star_sim)

        # --- Simulate Observations (y_star) using ORIGINAL NaN pattern ---
        # Use the NaN pattern from the *original* ys data for consistency.
        # The static info for the original data was passed in or derived earlier.
        static_valid_obs_idx_sim_star_jnp = static_valid_obs_idx_orig
        static_n_obs_actual_sim_star = static_n_obs_actual_orig
        
        y_star_sim = jnp.full((Tsteps, self.n_obs_full), jnp.nan, dtype=x_star_path.dtype) 
        static_C_obs_sim_star, static_H_obs_sim_star, static_I_obs_sim_star = None, None, None

        if static_n_obs_actual_sim_star > 0:
            # Use the KF's C and H matrices (self.C, self.H) but select rows/cols based on original NaN pattern
            static_C_obs_sim_star = self.C[static_valid_obs_idx_sim_star_jnp, :]
            static_H_obs_sim_star = self.H[jnp.ix_(static_valid_obs_idx_sim_star_jnp, static_valid_obs_idx_sim_star_jnp)]
            static_I_obs_sim_star = jnp.eye(static_n_obs_actual_sim_star, dtype=_DEFAULT_DTYPE) # Use default dtype

            eta_star_full_sim = self.simulate_obs_noise_internal(key_eta, (Tsteps,)) 
            eta_star_valid_sim = eta_star_full_sim[:, static_valid_obs_idx_sim_star_jnp] if self.n_obs_full > 0 else jnp.empty((Tsteps, 0))
            
            # Generate valid observations using x_star_path and the derived C_obs_sim_star
            y_star_valid_only_sim = (x_star_path @ static_C_obs_sim_star.T) + eta_star_valid_sim 
            y_star_sim = y_star_sim.at[:, static_valid_obs_idx_sim_star_jnp].set(y_star_valid_only_sim)
        else: # All NaNs
            static_C_obs_sim_star = jnp.empty((0, self.n_state), dtype=self.C.dtype)
            static_H_obs_sim_star = jnp.empty((0,0), dtype=self.H.dtype)
            static_I_obs_sim_star = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

        # --- Filter and Smooth the Simulated Data (y_star_sim) ---
        filter_results_star = self.filter(y_star_sim, static_valid_obs_idx_sim_star_jnp, static_n_obs_actual_sim_star,
                                           static_C_obs_sim_star, static_H_obs_sim_star, static_I_obs_sim_star)
        x_smooth_star, _ = self.smooth(y_star_sim, filter_results=filter_results_star) 

        # --- Combine to get the final draw ---
        x_draw = x_star_path + (x_smooth_rts - x_smooth_star)
        # Final check for safety
        x_draw = jnp.where(jnp.isfinite(x_draw), x_draw, jnp.zeros_like(x_draw)) 
        return x_draw


    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1,
                            filter_results_initial_ys: Optional[Dict] = None, # Results from filtering original ys
                            # Static NaN info for the original `ys` data MUST be provided if filter_results_initial_ys is None
                            static_info_for_original_ys: Optional[Dict] = None
                           ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        Tsteps = ys_arr.shape[0]
        empty_state_shape = (num_draws if num_draws > 1 else 1, Tsteps, self.n_state) if Tsteps > 0 else (num_draws if num_draws > 1 else 1, 0, self.n_state)
        empty_mean_shape = (Tsteps, self.n_state) if Tsteps > 0 else (0, self.n_state)

        if Tsteps == 0: 
            if num_draws == 1: return jnp.empty(empty_mean_shape, dtype=self.init_x.dtype)
            else: return jnp.empty(empty_mean_shape, dtype=self.init_x.dtype), \
                         jnp.empty(empty_mean_shape, dtype=self.init_x.dtype), \
                         jnp.empty(empty_state_shape, dtype=self.init_x.dtype)

        # --- Prepare static info and filter results for the ORIGINAL data ---
        _filter_results_orig_for_draw = filter_results_initial_ys
        _static_valid_idx_orig = None
        _static_n_actual_orig = None
        _static_C_obs_orig = None
        _static_H_obs_orig = None
        _static_I_obs_orig = None

        if filter_results_initial_ys is None:
            if static_info_for_original_ys is None:
                 # Compute static info for original ys if not provided
                first_obs_slice_concrete = onp.asarray(ys_arr[0])
                _valid_obs_idx_py_ll = onp.where(~onp.isnan(first_obs_slice_concrete))[0]
                _static_valid_idx_orig = jnp.array(_valid_obs_idx_py_ll, dtype=jnp.int32)
                _static_n_actual_orig = len(_valid_obs_idx_py_ll)
                
                if _static_n_actual_orig == self.n_obs_full:
                    _static_C_obs_orig = self.C; _static_H_obs_orig = self.H; _static_I_obs_orig = jnp.eye(self.n_obs_full, dtype=self.T.dtype)
                elif _static_n_actual_orig > 0:
                    _static_C_obs_orig = self.C[_static_valid_idx_orig, :]; _static_H_obs_orig = self.H[jnp.ix_(_static_valid_idx_orig, _static_valid_idx_orig)]; _static_I_obs_orig = jnp.eye(_static_n_actual_orig, dtype=self.T.dtype)
                else:
                    _static_C_obs_orig = jnp.empty((0, self.n_state), dtype=self.C.dtype); _static_H_obs_orig = jnp.empty((0,0), dtype=self.H.dtype); _static_I_obs_orig = jnp.empty((0,0), dtype=self.T.dtype)
            else: # Use provided static info
                _static_valid_idx_orig = static_info_for_original_ys["static_valid_obs_idx"]
                _static_n_actual_orig = static_info_for_original_ys["static_n_obs_actual"]
                _static_C_obs_orig = static_info_for_original_ys["static_C_obs"]
                _static_H_obs_orig = static_info_for_original_ys["static_H_obs"]
                _static_I_obs_orig = static_info_for_original_ys["static_I_obs"]
            
            # Filter the original data now using the derived/provided static info
            _filter_results_orig_for_draw = self.filter(ys_arr, _static_valid_idx_orig, _static_n_actual_orig,
                                                       _static_C_obs_orig, _static_H_obs_orig, _static_I_obs_orig)
        else:
             # If filter results are provided, we still need the static info for the simulation part later
             if static_info_for_original_ys is None:
                  # Need to derive it even if filter results are passed in
                  first_obs_slice_concrete = onp.asarray(ys_arr[0])
                  _valid_obs_idx_py_ll = onp.where(~onp.isnan(first_obs_slice_concrete))[0]
                  _static_valid_idx_orig = jnp.array(_valid_obs_idx_py_ll, dtype=jnp.int32)
                  _static_n_actual_orig = len(_valid_obs_idx_py_ll)
                  # C, H, I not strictly needed here as filter isn't called, but store for consistency
                  if _static_n_actual_orig == self.n_obs_full: _static_C_obs_orig = self.C; _static_H_obs_orig = self.H; _static_I_obs_orig = jnp.eye(self.n_obs_full, dtype=self.T.dtype)
                  elif _static_n_actual_orig > 0: _static_C_obs_orig = self.C[_static_valid_idx_orig, :]; _static_H_obs_orig = self.H[jnp.ix_(_static_valid_idx_orig, _static_valid_idx_orig)]; _static_I_obs_orig = jnp.eye(_static_n_actual_orig, dtype=self.T.dtype)
                  else: _static_C_obs_orig = jnp.empty((0, self.n_state), dtype=self.C.dtype); _static_H_obs_orig = jnp.empty((0,0), dtype=self.H.dtype); _static_I_obs_orig = jnp.empty((0,0), dtype=self.T.dtype)
             else:
                 _static_valid_idx_orig = static_info_for_original_ys["static_valid_obs_idx"]
                 _static_n_actual_orig = static_info_for_original_ys["static_n_obs_actual"]
                 _static_C_obs_orig = static_info_for_original_ys["static_C_obs"]
                 _static_H_obs_orig = static_info_for_original_ys["static_H_obs"]
                 _static_I_obs_orig = static_info_for_original_ys["static_I_obs"]
        
        # --- Run the vmapped smoother ---
        keys = random.split(key, num_draws)
        
        # We need _static_valid_idx_orig and _static_n_actual_orig inside the vmapped function
        # to correctly simulate y_star. _filter_results_orig_for_draw is also needed.
        # Make sure these are available in the closure or passed explicitly.
        
        # Pass static info related to ORIGINAL ys to the single draw function
        vmapped_smoother = vmap(
            lambda k_single: self._simulation_smoother_single_draw(
                ys_arr, k_single, filter_results_orig=_filter_results_orig_for_draw,
                # Pass the static info derived/checked above for the original ys
                static_valid_obs_idx_orig=_static_valid_idx_orig, 
                static_n_obs_actual_orig=_static_n_actual_orig,
                static_C_obs_orig=_static_C_obs_orig,       # Not strictly needed by _sim_smoother if filter_res provided, but good practice
                static_H_obs_orig=_static_H_obs_orig,       # Not strictly needed by _sim_smoother if filter_res provided
                static_I_obs_orig=_static_I_obs_orig        # Not strictly needed by _sim_smoother if filter_res provided
            ), in_axes=(0,)
        )
        all_draws_jax = vmapped_smoother(keys) # Shape (num_draws, Tsteps, n_state)
        
        if num_draws == 1:
            return all_draws_jax[0] # Return (Tsteps, n_state)
        else:
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- simulate_state_space function (remains the same as before) ---
def _simulate_state_space_impl( 
    P_aug: ArrayLike, R_aug: ArrayLike, Omega: ArrayLike, H_obs: ArrayLike,
    init_x: ArrayLike, init_P: ArrayLike, key: jax.random.PRNGKey, num_steps: int
) -> Tuple[jax.Array, jax.Array]:
    
    desired_dtype = P_aug.dtype 
    P_aug_jax = jnp.asarray(P_aug, dtype=desired_dtype) 
    R_aug_jax = jnp.asarray(R_aug, dtype=desired_dtype)
    Omega_jax = jnp.asarray(Omega, dtype=desired_dtype) 
    H_obs_jax = jnp.asarray(H_obs, dtype=desired_dtype)
    init_x_jax = jnp.asarray(init_x, dtype=desired_dtype)
    init_P_jax = jnp.asarray(init_P, dtype=desired_dtype)
    n_aug = P_aug_jax.shape[0] 
    n_aug_shocks = R_aug_jax.shape[1] if R_aug_jax.ndim == 2 and R_aug_jax.shape[1] > 0 else 0
    n_obs_sim = Omega_jax.shape[0]; key_init, key_state_noise, key_obs_noise = random.split(key, 3)
    kf_jitter = _KF_JITTER # Use consistent jitter

    try:
        init_P_reg = init_P_jax + kf_jitter * jnp.eye(n_aug, dtype=desired_dtype)
        L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_aug,), dtype=desired_dtype); x0 = init_x_jax + L0 @ z0

    except Exception: x0 = init_x_jax

    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_aug_shocks), dtype=desired_dtype) if n_aug_shocks > 0 else jnp.zeros((num_steps, 0), dtype=desired_dtype)
    obs_noise_sim = jnp.zeros((num_steps, n_obs_sim), dtype=desired_dtype)

    if n_obs_sim > 0:
        try:
            # Use consistent jitter for H Cholesky/MVN simulation
            H_obs_reg = H_obs_jax + kf_jitter * jnp.eye(n_obs_sim, dtype=desired_dtype) 
            # Try Cholesky first for potentially better performance if H is sparse later
            L_H_obs = jnp.linalg.cholesky(H_obs_reg)
            z_eta = random.normal(key_obs_noise, (num_steps, n_obs_sim), dtype=desired_dtype)
            obs_noise_sim = z_eta @ L_H_obs.T
        except Exception:
            # Fallback to MVN
             try:
                 H_obs_reg_mvn = H_obs_jax + kf_jitter * 10 * jnp.eye(n_obs_sim, dtype=desired_dtype) # Maybe slightly more jitter for MVN
                 obs_noise_sim = random.multivariate_normal(key_obs_noise, jnp.zeros(n_obs_sim, dtype=desired_dtype), H_obs_reg_mvn, shape=(num_steps,), dtype=desired_dtype)
             except Exception: pass # Leave as zeros if MVN also fails

    def simulation_step(x_prev, noise_t_tuple):
        eps_t_arg, eta_t_arg = noise_t_tuple
        shock_term_sim = R_aug_jax @ eps_t_arg if n_aug_shocks > 0 else jnp.zeros(n_aug, dtype=x_prev.dtype)
        x_curr_sim = P_aug_jax @ x_prev + shock_term_sim
        y_curr_sim = Omega_jax @ x_curr_sim + eta_t_arg if n_obs_sim > 0 else jnp.empty((0,), dtype=x_curr_sim.dtype)
        return x_curr_sim, (x_curr_sim, y_curr_sim)

    combined_noise_sim = (state_shocks_std_normal, obs_noise_sim)
    _, (states_sim, observations_sim) = lax.scan(simulation_step, x0, combined_noise_sim)

    return states_sim, observations_sim

simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))
# --- END OF FILE Kalman_filter_jax.py (Modified Jitter) ---