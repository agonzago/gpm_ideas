# --- START OF FILE Kalman_filter_jax.py ---
# (Imports and __init__ as before)
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp
from typing import Tuple, Optional, Union, Sequence, Dict, Any

_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps

class KalmanFilter:
    # --- __init__, _simulate_obs_noise_chol, _simulate_obs_noise_mvn ---
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype); self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype); self.H = jnp.asarray(H, dtype=desired_dtype)
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype); self.init_P = jnp.asarray(init_P, dtype=desired_dtype)
        n_state = self.T.shape[0]; n_obs = self.C.shape[0]; n_shocks = self.R.shape[1] if self.R.ndim == 2 else 0
        if self.T.shape != (n_state, n_state): raise ValueError(f"T shape mismatch")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks): raise ValueError(f"R shape mismatch")
        elif n_shocks == 0 and self.R.size != 0: self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        if self.C.shape != (n_obs, n_state): raise ValueError(f"C shape mismatch")
        if self.H.shape != (n_obs, n_obs): raise ValueError(f"H shape mismatch")
        if self.init_x.shape != (n_state,): raise ValueError(f"init_x shape mismatch")
        if self.init_P.shape != (n_state, n_state): raise ValueError(f"init_P shape mismatch")
        self.n_state = n_state; self.n_obs = n_obs; self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)
        if self.n_shocks > 0: self.state_cov = self.R @ self.R.T
        else: self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)
        self.H_stable = self.H; self.log_det_H_term = 0.0
        try:
            H_reg = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype); self.L_H = jnp.linalg.cholesky(H_reg)
            self.simulate_obs_noise = self._simulate_obs_noise_chol; self.log_det_H_term = 2 * jnp.sum(jnp.log(jnp.diag(self.L_H)))
        except Exception:
            self.H_stable = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.simulate_obs_noise = self._simulate_obs_noise_mvn
            sign, log_det = jnp.linalg.slogdet(self.H_stable); self.log_det_H_term = log_det
    def _simulate_obs_noise_chol(self, key, shape): z_eta = random.normal(key, tuple(shape) + (self.n_obs,), dtype=self.H.dtype); return z_eta @ self.L_H.T
    def _simulate_obs_noise_mvn(self, key, shape):
        mvn_shape = tuple(shape) if len(shape) > 0 else (); eta = random.multivariate_normal(key, jnp.zeros((self.n_obs,), dtype=self.H.dtype), self.H_stable, shape=mvn_shape, dtype=self.H.dtype); return eta
    # ---------------------------------------------------------------------

    def filter_for_likelihood(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """ Uses lax.cond for NaN handling - suitable for MCMC likelihood. """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, C, H, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov # R @ R.T
        _MAX_COND_NUM = 1e12 # Define a threshold for condition number

        def step_for_likelihood(carry, y_t):
            x_prev_filt, P_prev_filt = carry

            # Prediction Step
            x_pred_t = T_mat @ x_prev_filt
            # <<< Add jitter to P prediction >>>
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov + _MACHINE_EPSILON * I_s
            P_pred_t = (P_pred_t + P_pred_t.T) / 2.0 # Ensure symmetry

            # Check for missing observation
            is_missing = jnp.all(jnp.isnan(y_t))
            y_obs = jnp.nan_to_num(y_t, nan=0.0) # Replace NaN with 0 for calculations

            # Innovation
            v = y_obs - (C @ x_pred_t)
            PCt = P_pred_t @ C.T
            S = C @ PCt + H
            S_reg = S + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=S.dtype)
            S_reg = (S_reg + S_reg.T) / 2.0 # Ensure symmetry

            # --- Perform Update Step Conditionally ---
            # Function to perform the actual Kalman update
            def perform_update():
                update_valid = jnp.array(True) # Assume valid initially
                x_filt_t_up = x_pred_t # Default to prediction if update fails
                P_filt_t_up = P_pred_t

                # <<< Check condition number of S_reg >>>
                cond_S = jnp.linalg.cond(S_reg)
                # jdebug.print(" Filter Step: Cond(S)={c}", c=cond_S) # Optional debug print
                update_valid &= cond_S < _MAX_COND_NUM
                update_valid &= jnp.all(jnp.isfinite(cond_S))

                # Attempt update only if S seems well-conditioned
                def _do_the_update():
                    try: # Use Cholesky first for stability/speed with PSD matrices
                        L_S = jnp.linalg.cholesky(S_reg)
                        # Solve L_S * Y = PCt' -> Y = solve(L_S, PCt')
                        Y_sol = jax.scipy.linalg.solve_triangular(L_S, PCt, lower=True)
                        # Solve L_S' * K = Y' -> K = solve(L_S', Y')
                        K_sol = jax.scipy.linalg.solve_triangular(L_S.T, Y_sol, lower=False)
                    except: # Fallback to general solver if Cholesky fails
                        try:
                            K_sol = jax.scipy.linalg.solve(S_reg, PCt, assume_a='pos') # Try assuming positive definite
                        except:
                             try: # Fallback to general solver
                                K_sol = jax.scipy.linalg.solve(S_reg, PCt, assume_a='gen')
                             except: # Final fallback: pseudo-inverse (often indicates problems)
                                # jdebug.print(" Filter Step: Using pinv for K") # Optional debug print
                                K_sol = jnp.linalg.pinv(S_reg) @ PCt

                    # Check if Kalman gain calculation was successful
                    kg_valid = jnp.all(jnp.isfinite(K_sol))
                    # jdebug.print(" Filter Step: KG valid={v}", v=kg_valid)

                    # Perform state and covariance update using Joseph form for P for better stability
                    x_filt_t_upd = x_pred_t + K_sol @ v
                    IKC = I_s - K_sol @ C
                    # Joseph form: P_filt = (I-KC)P_pred(I-KC)' + K H K'
                    P_filt_t_upd = IKC @ P_pred_t @ IKC.T + K_sol @ H @ K_sol.T
                    P_filt_t_upd = (P_filt_t_upd + P_filt_t_upd.T) / 2.0 # Ensure symmetry

                    # Return updated values and validity flag for this specific path
                    return x_filt_t_upd, P_filt_t_upd, kg_valid

                # Define function for failure case (return predictions)
                def _update_failed():
                    return x_pred_t, P_pred_t, jnp.array(False)

                # Choose path based on S condition number
                x_filt_t_up, P_filt_t_up, kg_valid = lax.cond(update_valid, _do_the_update, _update_failed)

                # Overall update validity depends on S condition and Kalman Gain calculation
                update_fully_valid = update_valid & kg_valid

                return x_filt_t_up, P_filt_t_up, update_fully_valid

            # Function for missing data (no update performed)
            def handle_missing():
                # State and Covariance are just the predictions
                # Update validity is trivially True (as no update needed)
                # Likelihood contribution is zero
                return x_pred_t, P_pred_t, jnp.array(True)

            # Use lax.cond: if is_missing, call handle_missing, else call perform_update
            x_filt_t, P_filt_t, update_valid_flag = lax.cond(is_missing,
                                                             handle_missing,
                                                             perform_update)

            # --- Calculate Log Likelihood Contribution ---
            log_pi_term = jnp.log(2 * jnp.pi) * self.n_obs
            sign, log_det_S = jnp.linalg.slogdet(S_reg)
            log_det_S_valid = sign > 0 # Determinant is valid only if sign is positive

            # Calculate Mahalanobis distance safely
            try:
                # Use Cholesky solve for stability if possible
                L_S_maha = jnp.linalg.cholesky(S_reg)
                v_scaled = jax.scipy.linalg.solve_triangular(L_S_maha, v, lower=True)
                mahalanobis_dist = jnp.sum(v_scaled**2)
                maha_valid = jnp.isfinite(mahalanobis_dist)
            except: # Fallback to solve (less stable but might work)
                try:
                    solved_term = jax.scipy.linalg.solve(S_reg, v, assume_a='pos')
                    mahalanobis_dist = v @ solved_term
                    maha_valid = jnp.isfinite(mahalanobis_dist)
                except: # Give up if solve fails
                    mahalanobis_dist = jnp.inf # Penalize heavily
                    maha_valid = jnp.array(False)


            # Likelihood contribution is valid only if:
            # - Update step was valid (or data was missing)
            # - Log determinant was valid
            # - Mahalanobis distance was valid
            ll_term_valid = update_valid_flag & log_det_S_valid & maha_valid

            # Calculate raw LL term (can be NaN/Inf if inputs are bad)
            ll_t_raw = -0.5 * (log_pi_term + log_det_S + mahalanobis_dist)

            # Final safe LL contribution: Use penalty if missing or any term invalid
            # Use jnp.where for conditional assignment safe for JIT
            safe_ll_t = jnp.where(is_missing, 0.0, # Zero LL for missing obs
                                  jnp.where(ll_term_valid, ll_t_raw, -jnp.inf)) # -Inf penalty if invalid

            # jdebug.print(" Filter Step: LL raw={llr}, LL safe={lls}, missing={m}, update_ok={uok}, logdet_ok={dok}, maha_ok={mok}",
            #             llr=ll_t_raw, lls=safe_ll_t, m=is_missing, uok=update_valid_flag, dok=log_det_S_valid, mok=maha_valid) # Detailed Debug

            # --- Prepare outputs for scan ---
            # Ensure outputs are always finite, use predictions if update failed
            # (This might mask where the failure occurred, but necessary for scan)
            final_x_filt = jnp.where(update_valid_flag, x_filt_t, x_pred_t)
            final_P_filt = jnp.where(update_valid_flag, P_filt_t, P_pred_t)

            outputs = {
                'x_pred': x_pred_t,
                'P_pred': P_pred_t,
                'x_filt': final_x_filt, # Return potentially corrected state
                'P_filt': final_P_filt, # Return potentially corrected cov
                'innovations': v,
                'innovation_cov': S, # Return original S, not S_reg
                'log_likelihood_contributions': safe_ll_t # Return penalized LL
            }
            return (final_x_filt, final_P_filt), outputs # Carry potentially corrected state/cov

        # --- Run the scan ---
        init_carry = (self.init_x, self.init_P)
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), scan_outputs = lax.scan(step_for_likelihood, init_carry, ys_reshaped)
        return scan_outputs

    # --- Version 2: Filter assuming Static NaN pattern (or no NaNs) ---
    def filter(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """ Applies the Kalman filter, optimized for a STATIC pattern of missing values. """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, C_full, H_full, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov
        T_steps = ys_arr.shape[0]
        if T_steps == 0: # Handle empty input
             empty_float=lambda *s:jnp.empty(s,dtype=I_s.dtype); empty_innov=lambda *s:jnp.empty(s,dtype=self.C.dtype); n_obs_actual=0
             if ys.shape[0]>0 and ys.shape[1]>0:
                 try: valid_obs_idx_np=onp.where(~onp.isnan(onp.asarray(ys[0])))[0]; n_obs_actual=len(valid_obs_idx_np)
                 except IndexError: pass
             return {'x_pred': empty_float(0,self.n_state), 'P_pred': empty_float(0,self.n_state,self.n_state), 'x_filt': empty_float(0,self.n_state), 'P_filt': empty_float(0,self.n_state,self.n_state), 'innovations': empty_innov(0,n_obs_actual), 'innovation_cov': empty_innov(0,n_obs_actual,n_obs_actual), 'log_likelihood_contributions': empty_float(0)}

        first_obs_np = onp.asarray(ys_arr[0]) # Use NumPy
        if onp.any(onp.isnan(first_obs_np)):
            valid_obs_idx_np = onp.where(~onp.isnan(first_obs_np))[0]; valid_obs_idx = jnp.array(valid_obs_idx_np)
            n_obs_actual = len(valid_obs_idx); C_obs = C_full.at[valid_obs_idx, :].get()
            H_obs = H_full.at[jnp.ix_(valid_obs_idx, valid_obs_idx)].get()
            I_obs = jnp.eye(n_obs_actual, dtype=I_s.dtype); select_obs = lambda y: y.at[valid_obs_idx].get()
        else: # No NaNs
            n_obs_actual = self.n_obs; C_obs = C_full; H_obs = H_full; I_obs = jnp.eye(self.n_obs, dtype=I_s.dtype); select_obs = lambda y: y

        if n_obs_actual == 0: # All missing
            C_obs = jnp.empty((0, self.n_state), dtype=C_full.dtype); H_obs = jnp.empty((0, 0), dtype=H_full.dtype)
            I_obs = jnp.empty((0, 0), dtype=I_s.dtype); select_obs = lambda y: jnp.empty((0,), dtype=y.dtype)

        def step_static_nan(carry, y_t_full):
            x_prev_filt, P_prev_filt = carry; x_pred_t = T_mat @ x_prev_filt
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov
            y_obs_t = select_obs(y_t_full); v_obs = y_obs_t - (C_obs @ x_pred_t)
            PCt_obs = P_pred_t @ C_obs.T; S_obs = C_obs @ PCt_obs + H_obs
            S_obs_reg = S_obs + _MACHINE_EPSILON * I_obs
            try: L_S_obs = jnp.linalg.cholesky(S_obs_reg); Y_obs = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs, lower=True); K = jax.scipy.linalg.solve_triangular(L_S_obs.T, Y_obs, lower=False)
            except Exception:
                 try: K = jnp.linalg.solve(S_obs_reg, PCt_obs)
                 except Exception:
                     try: S_obs_pinv = jnp.linalg.pinv(S_obs_reg); K = PCt_obs @ S_obs_pinv
                     except Exception: K = jnp.zeros((self.n_state, n_obs_actual), dtype=x_pred_t.dtype)
            x_filt_t = x_pred_t + K @ v_obs; IKC_obs = I_s - K @ C_obs
            P_filt_t = IKC_obs @ P_pred_t @ IKC_obs.T + K @ H_obs @ K.T
            P_filt_t = (P_filt_t + P_filt_t.T) / 2.0
            log_pi_term = jnp.log(2 * jnp.pi) * n_obs_actual
            sign, log_det_S_obs = jnp.linalg.slogdet(S_obs_reg)
            try: solved_term_obs = jax.scipy.linalg.solve(S_obs_reg, v_obs, assume_a='pos'); mahalanobis_dist_obs = v_obs @ solved_term_obs
            except Exception: mahalanobis_dist_obs = jnp.where(n_obs_actual > 0, 1e18, 0.0)
            ll_t = -0.5 * (log_pi_term + log_det_S_obs + mahalanobis_dist_obs)
            safe_ll_t = jnp.where(n_obs_actual == 0, 0.0, jnp.where(sign > 0, ll_t, -1e18))
            outputs = {'x_pred': x_pred_t, 'P_pred': P_pred_t, 'x_filt': x_filt_t, 'P_filt': P_filt_t, 'innovations': v_obs, 'innovation_cov': S_obs, 'log_likelihood_contributions': safe_ll_t}
            return (x_filt_t, P_filt_t), outputs

        init_carry = (self.init_x, self.init_P)
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), scan_outputs = lax.scan(step_static_nan, init_carry, ys_reshaped)
        return scan_outputs

    # def log_likelihood(self, ys: ArrayLike) -> jax.Array:
    #     """ Computes the log-likelihood using the robust filter_for_likelihood. """
    #     filter_results = self.filter_for_likelihood(ys) # <<< Calls robust version
    #     total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
    #     return total_log_likelihood

    def log_likelihood(self, ys: ArrayLike) -> jax.Array:
        """ Computes the log-likelihood using the STATIC NAN filter. """
        # <<< REVERT THIS LINE >>>
        filter_results = self.filter(ys) # Use the static NaN filter
        # <<< END REVERT >>>
        total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
        # Check if result is finite, maybe return penalty otherwise?
        # For now, let potential NaNs propagate.
        safe_total_ll = jnp.where(jnp.isfinite(total_log_likelihood), total_log_likelihood, -jnp.inf)
        # jdebug.print(" log_likelihood method: total_ll={ll}, safe_total_ll={sll}", ll=total_log_likelihood, sll=safe_total_ll) # Debug
        return safe_total_ll # Return potentially penalized value


    def smooth(self, ys: ArrayLike, filter_results: Optional[Dict] = None, use_likelihood_filter: bool = False) -> Tuple[jax.Array, jax.Array]:
        """ Applies the RTS smoother. Uses specified filter if results not provided. """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        if filter_results is None:
            # <<< Choose filter based on flag >>>
            filter_func = self.filter_for_likelihood if use_likelihood_filter else self.filter
            filter_outs_dict = filter_func(ys_arr)
        else:
            filter_outs_dict = filter_results
        x_pred=filter_outs_dict['x_pred']; P_pred=filter_outs_dict['P_pred']; x_filt=filter_outs_dict['x_filt']; P_filt=filter_outs_dict['P_filt']; T_mat=self.T
        N = x_filt.shape[0]
        if N == 0: return jnp.empty((0,self.n_state)), jnp.empty((0,self.n_state,self.n_state))
        x_s_next=x_filt[-1]; P_s_next=P_filt[-1]
        scan_inputs=(P_pred[1:][::-1], P_filt[:-1][::-1], x_pred[1:][::-1], x_filt[:-1][::-1])
        def backward_step(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth; Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            try: Pp_next_reg = Pp_next_t+_MACHINE_EPSILON*jnp.eye(self.n_state,dtype=Pp_next_t.dtype); Jt_transpose = jax.scipy.linalg.solve(Pp_next_reg, (T_mat @ Pf_t).T, assume_a='gen'); Jt = Jt_transpose.T
            except Exception: Pp_next_pinv = jnp.linalg.pinv(Pp_next_t); Jt = Pf_t @ T_mat.T @ Pp_next_pinv
            x_diff = x_s_next_t - xp_next_t; x_s_t = xf_t + Jt @ x_diff
            P_diff = P_s_next_t - Pp_next_t; P_s_t = Pf_t + Jt @ P_diff @ Jt.T
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            return (x_s_t, P_s_t), (x_s_t, P_s_t)
        init_carry_smooth=(x_s_next, P_s_next); (_, _),(x_s_rev,P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt[-1][None, :]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt[-1][None, :, :]], axis=0)
        return x_smooth, P_smooth

    def _simulation_smoother_single_draw(self, ys: jax.Array, key: jax.random.PRNGKey, filter_results_orig: Optional[Dict] = None) -> jax.Array:
        """ Internal simulation smoother draw. """
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        n_s = self.n_state; n_eps = self.n_shocks
        # Smooth original data using the main filter (Static NaN default)
        x_smooth_rts, _ = self.smooth(ys, filter_results=filter_results_orig, use_likelihood_filter=False) # <<< Explicitly False
        key_init, key_eps, key_eta = random.split(key, 3)
        try:
            init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(n_s, dtype=self.init_P.dtype); L0 = jnp.linalg.cholesky(init_P_reg)
            z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype); x0_star = self.init_x + L0 @ z0
        except Exception: x0_star = self.init_x
        eps_star = random.normal(key_eps, (Tsteps, n_eps), dtype=self.R.dtype) if n_eps > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
        def state_sim_step(x_prev_star, eps_t_star):
            shock_term = self.R @ eps_t_star if n_eps > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
            x_curr_star = self.T @ x_prev_star + shock_term; return x_curr_star, x_curr_star
        _, x_star = lax.scan(state_sim_step, x0_star, eps_star)
        eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
        y_star = (x_star @ self.C.T) + eta_star
        # Smooth simulated data: MUST use the likelihood filter version inside vmap
        x_smooth_star, _ = self.smooth(y_star, use_likelihood_filter=True) # <<< Explicitly True
        x_draw = x_star + (x_smooth_rts - x_smooth_star)
        return x_draw

    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1,
                            filter_results: Optional[Dict] = None
                            ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """ Runs Durbin-Koopman smoother. Uses appropriate filter internally. """
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        Tsteps = ys_arr.shape[0]
        empty_state=jnp.empty((0,self.n_state),dtype=self.init_x.dtype)
        if Tsteps == 0:
            if num_draws == 1: return empty_state
            else: return empty_state, empty_state, jnp.empty((num_draws,0,self.n_state),dtype=self.init_x.dtype)

        # Run filter ONCE on original data using the main filter if results not provided
        if filter_results is None:
            filter_results_orig = self.filter(ys_arr) # Uses main (Static NaN) filter
        else:
            filter_results_orig = filter_results

        if num_draws == 1:
            # Pass original filter results to single draw function
            single_draw = self._simulation_smoother_single_draw(ys_arr, key, filter_results_orig=filter_results_orig)
            return single_draw
        else:
            # Calculate RTS on original data ONCE using original filter results (uses main filter)
            x_smooth_rts_common, _ = self.smooth(ys_arr, filter_results=filter_results_orig, use_likelihood_filter=False)

            # Vmapped function implicitly calls _simulation_smoother_single_draw
            # which now correctly uses filter_for_likelihood for y_star
            def perform_single_dk_draw_wrapper(key_single_draw):
                # This wrapper just calls the internal method correctly
                # It doesn't need filter_results_orig passed again because x_smooth_rts_common is calculated outside
                return self._simulation_smoother_single_draw(ys_arr, key_single_draw, filter_results_orig=filter_results_orig)


            keys = random.split(key, num_draws)
            vmapped_smoother = vmap(perform_single_dk_draw_wrapper, in_axes=(0,))
            all_draws_jax = vmapped_smoother(keys)
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- Keep simulation function as is ---
def _simulate_state_space_impl( P_aug, R_aug, Omega, H_obs, init_x, init_P, key, num_steps):
    # ... (exact same implementation as before) ...
    desired_dtype = P_aug.dtype
    P_aug_jax=jnp.asarray(P_aug,dtype=desired_dtype); R_aug_jax=jnp.asarray(R_aug,dtype=desired_dtype)
    Omega_jax=jnp.asarray(Omega,dtype=desired_dtype); H_obs_jax=jnp.asarray(H_obs,dtype=desired_dtype)
    init_x_jax=jnp.asarray(init_x,dtype=desired_dtype); init_P_jax=jnp.asarray(init_P,dtype=desired_dtype)
    n_aug=P_aug_jax.shape[0]; n_aug_shocks=R_aug_jax.shape[1] if R_aug_jax.ndim==2 else 0; n_obs=Omega_jax.shape[0]
    key_init,key_state_noise,key_obs_noise=random.split(key, 3)
    try:
        init_P_reg=init_P_jax+_MACHINE_EPSILON*jnp.eye(n_aug,dtype=desired_dtype)
        L0=jnp.linalg.cholesky(init_P_reg); z0=random.normal(key_init,(n_aug,),dtype=desired_dtype); x0=init_x_jax+L0@z0
    except Exception: x0=init_x_jax
    state_shocks_std_normal=random.normal(key_state_noise,(num_steps,n_aug_shocks),dtype=desired_dtype) if n_aug_shocks > 0 else jnp.zeros((num_steps,0),dtype=desired_dtype)
    try:
        H_obs_reg=H_obs_jax+_MACHINE_EPSILON*jnp.eye(n_obs,dtype=desired_dtype)
        obs_noise=random.multivariate_normal(key_obs_noise,jnp.zeros(n_obs,dtype=desired_dtype),H_obs_reg,shape=(num_steps,),dtype=desired_dtype)
    except Exception as e: obs_noise=jnp.zeros((num_steps,n_obs),dtype=desired_dtype)
    def simulation_step(x_prev,noise_t):
        eps_t,eta_t=noise_t
        shock_term=R_aug_jax@eps_t if n_aug_shocks > 0 else jnp.zeros(n_aug,dtype=x_prev.dtype)
        x_curr=P_aug_jax@x_prev+shock_term; y_curr=Omega_jax@x_curr+eta_t
        return x_curr,(x_curr,y_curr)
    combined_noise=(state_shocks_std_normal,obs_noise)
    final_state,(states,observations)=lax.scan(simulation_step,x0,combined_noise)
    return states,observations

simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))

# --- END OF FILE Kalman_filter_jax.py ---