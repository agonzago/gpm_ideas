# --- START OF FILE Kalman_filter_jax.py ---

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp # Keep for other functions if needed (though not for filter logic)
from typing import Tuple, Optional, Union, Sequence, Dict, Any
import jax.debug as jdebug # For optional debug prints

# Use float64 epsilon if enabled, otherwise float32
_BASE_EPS = jnp.finfo(jnp.float64).eps if jax.config.jax_enable_x64 else jnp.finfo(jnp.float32).eps
_MACHINE_EPSILON = _BASE_EPS * 10.0 # Slightly increased jitter
_LOG_2PI = jnp.log(2 * jnp.pi)

class KalmanFilter:
    """
    Kalman Filter and Smoother implementation in JAX.
    Uses a robust filter method suitable for dynamic NaNs and MCMC.
    """
    # --- __init__, _simulate_obs_noise_chol, _simulate_obs_noise_mvn (Unchanged) ---
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype); self.R = jnp.asarray(R, dtype=desired_dtype); self.C = jnp.asarray(C, dtype=desired_dtype); self.H = jnp.asarray(H, dtype=desired_dtype); self.init_x = jnp.asarray(init_x, dtype=desired_dtype); self.init_P = jnp.asarray(init_P, dtype=desired_dtype)
        n_state = self.T.shape[0]; n_obs = self.C.shape[0]
        if self.R.ndim == 2: n_shocks = self.R.shape[1]
        elif self.R.size == 0: n_shocks = 0; self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        else: raise ValueError(f"R matrix has unexpected shape: {self.R.shape}")
        self.n_state = n_state; self.n_obs = n_obs; self.n_shocks = n_shocks
        if self.T.shape != (n_state, n_state): raise ValueError(f"T shape mismatch.")
        if self.R.shape != (n_state, n_shocks): raise ValueError(f"R shape mismatch.")
        if self.C.shape != (n_obs, n_state): raise ValueError(f"C shape mismatch.")
        if self.H.shape != (n_obs, n_obs): raise ValueError(f"H shape mismatch.")
        if self.init_x.shape != (n_state,): raise ValueError(f"init_x shape mismatch.")
        if self.init_P.shape != (n_state, n_state): raise ValueError(f"init_P shape mismatch.")
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype); self.I_o = jnp.eye(self.n_obs, dtype=desired_dtype)
        self.state_cov = self.R @ self.R.T
        self.H_stable = self.H + _MACHINE_EPSILON * self.I_o
        try: self.L_H = jnp.linalg.cholesky(self.H_stable); self.simulate_obs_noise = self._simulate_obs_noise_chol; self.log_det_H_term = 2 * jnp.sum(jnp.log(jnp.diag(self.L_H)))
        except Exception: self.simulate_obs_noise = self._simulate_obs_noise_mvn; sign, log_det = jnp.linalg.slogdet(self.H_stable); self.log_det_H_term = jnp.where(sign > 0, log_det, -jnp.inf)
    def _simulate_obs_noise_chol(self, key, shape): z_eta = random.normal(key, tuple(shape) + (self.n_obs,), dtype=self.H.dtype); return jnp.einsum('ij,...j->...i', self.L_H, z_eta)
    def _simulate_obs_noise_mvn(self, key, shape): mvn_shape = tuple(shape) if len(shape) > 0 else (); eta = random.multivariate_normal(key, jnp.zeros((self.n_obs,), dtype=self.H.dtype), self.H_stable, shape=mvn_shape, dtype=self.H.dtype); return eta
    # ---------------------------------------------------------------------

    # --- Filter Method: Robust for Likelihood (Handles Dynamic NaNs) ---
    def filter_for_likelihood(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """
        Kalman filter optimized for likelihood calculation within MCMC.
        Handles potentially time-varying NaN patterns using lax.cond.
        Returns penalized likelihood contribution (-inf) on numerical failure.
        Uses stable solve for K. Includes Debug Prints.
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, C, H, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov

        def step_for_likelihood(carry, y_t):
            x_prev_filt, P_prev_filt = carry # Original 2-element carry

            x_pred_t = T_mat @ x_prev_filt
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov + _MACHINE_EPSILON * I_s
            P_pred_t = (P_pred_t + P_pred_t.T) / 2.0
            is_missing = jnp.all(jnp.isnan(y_t))
            y_obs = jnp.nan_to_num(y_t, nan=0.0)
            v = y_obs - (C @ x_pred_t)
            PCt = P_pred_t @ C.T # Shape (n_state, n_obs)
            S = C @ PCt + H      # Shape (n_obs, n_obs)
            S_reg = S + _MACHINE_EPSILON * self.I_o
            S_reg = (S_reg + S_reg.T) / 2.0

            def perform_update():
                update_valid = jnp.array(True); x_filt_t_up = x_pred_t; P_filt_t_up = P_pred_t
                # Stable Kalman Gain: Solve S_reg K' = PCt'
                try: Kt = jax.scipy.linalg.solve(S_reg, PCt.T, assume_a='pos'); K_sol = Kt.T
                except Exception:
                    try: S_pinv = jnp.linalg.pinv(S_reg); K_sol = PCt @ S_pinv
                    except Exception: K_sol = jnp.full((self.n_state, self.n_obs), jnp.nan, dtype=PCt.dtype)
                kg_valid = jnp.all(jnp.isfinite(K_sol)); update_valid &= kg_valid
                x_filt_t_upd = x_pred_t + K_sol @ v
                IKC = I_s - K_sol @ C
                P_filt_t_upd = IKC @ P_pred_t @ IKC.T + K_sol @ H @ K_sol.T; P_filt_t_upd = (P_filt_t_upd + P_filt_t_upd.T) / 2.0
                x_filt_final = jnp.where(update_valid, x_filt_t_upd, x_pred_t); P_filt_final = jnp.where(update_valid, P_filt_t_upd, P_pred_t)
                # Log Likelihood
                ll_term_valid = jnp.array(True); log_pi_term = _LOG_2PI * self.n_obs
                sign, log_det_S = jnp.linalg.slogdet(S_reg); ll_term_valid &= sign > 0
                mahalanobis_dist = jnp.inf
                try:
                    solved_term = jax.scipy.linalg.solve(S_reg, v, assume_a='pos')
                    solve_valid = jnp.all(jnp.isfinite(solved_term))
                    mahalanobis_dist = jnp.where(solve_valid, jnp.dot(v, solved_term), jnp.inf)
                except Exception: mahalanobis_dist = jnp.inf
                maha_valid = jnp.isfinite(mahalanobis_dist); log_det_valid = jnp.isfinite(log_det_S); ll_term_valid &= maha_valid & log_det_valid
                ll_t_raw = -0.5 * (log_pi_term + log_det_S + mahalanobis_dist)
                safe_ll_t = jnp.where(update_valid & ll_term_valid, ll_t_raw, -jnp.inf)
                # Debug Prints (Comment out for performance)
                # jdebug.print(" Likelihood Filter: update_ok={uok}, kg_ok={kgok}, sign_ok={sok}, logdet_ok={dok}, maha_ok={mok}, ll_term_ok={llok}, ll_safe={lls:.4f}", uok=update_valid, kgok=kg_valid, sok=(sign > 0), dok=log_det_valid, mok=maha_valid, llok=ll_term_valid, lls=safe_ll_t)
                return x_filt_final, P_filt_final, safe_ll_t
            def handle_missing():
                 # jdebug.print(" Likelihood Filter: is_missing=True, ll_safe=0.0") # Comment out for performance
                 return x_pred_t, P_pred_t, jnp.array(0.0, dtype=x_pred_t.dtype)
            x_filt_t, P_filt_t, ll_t = lax.cond(is_missing, handle_missing, perform_update)
            outputs = {'x_pred': x_pred_t, 'P_pred': P_pred_t, 'x_filt': x_filt_t, 'P_filt': P_filt_t, 'innovations': v, 'innovation_cov': S, 'log_likelihood_contributions': ll_t}
            return (x_filt_t, P_filt_t), outputs # Original 2-element carry
        init_carry = (self.init_x, self.init_P); ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs)); (_, _), scan_outputs = lax.scan(step_for_likelihood, init_carry, ys_reshaped)
        return scan_outputs

    # --- Filter Method 2: REMOVED ---
    # The filter method optimized for static NaNs caused JIT/Tracer issues.
    # We will rely solely on filter_for_likelihood.

    # --- log_likelihood (Unchanged) ---
    def log_likelihood(self, ys: ArrayLike) -> jax.Array:
        """ Computes the total log-likelihood using the robust filter_for_likelihood. """
        filter_results = self.filter_for_likelihood(ys); total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions']); safe_total_ll = jnp.where(jnp.isfinite(total_log_likelihood), total_log_likelihood, -jnp.inf); return safe_total_ll

    # --- smooth (Updated: Always uses filter_for_likelihood internally) ---
    def smooth(self, ys: ArrayLike, filter_results: Optional[Dict] = None) -> Tuple[jax.Array, jax.Array]:
        """
        Applies the Rauch-Tung-Striebel (RTS) smoother.
        Always uses self.filter_for_likelihood() if filter_results not provided.
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype); T_steps = ys_arr.shape[0]
        if filter_results is None:
            # Always use the robust filter internally when results not provided
            filter_outs_dict = self.filter_for_likelihood(ys_arr)
        else: filter_outs_dict = filter_results

        x_pred=filter_outs_dict['x_pred']; P_pred=filter_outs_dict['P_pred']; x_filt=filter_outs_dict['x_filt']; P_filt=filter_outs_dict['P_filt']; T_mat=self.T; I_s = self.I_s
        if T_steps == 0: return jnp.empty((0,self.n_state), dtype=self.init_x.dtype), jnp.empty((0,self.n_state,self.n_state), dtype=self.init_P.dtype)
        x_s_next=x_filt[-1]; P_s_next=P_filt[-1]
        scan_inputs=(P_pred[1:][::-1], P_filt[:-1][::-1], x_pred[1:][::-1], x_filt[:-1][::-1])
        def backward_step(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth; Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            try: Pp_next_reg = Pp_next_t + _MACHINE_EPSILON * I_s; Jt_transpose = jax.scipy.linalg.solve(Pp_next_reg, (T_mat @ Pf_t), assume_a='gen'); Jt = Jt_transpose.T
            except Exception: Pp_next_pinv = jnp.linalg.pinv(Pp_next_t + _MACHINE_EPSILON * I_s); Jt = Pf_t @ T_mat.T @ Pp_next_pinv
            jt_valid = jnp.all(jnp.isfinite(Jt))
            x_diff = x_s_next_t - xp_next_t; x_s_t = xf_t + Jt @ x_diff
            P_diff = P_s_next_t - Pp_next_t; P_s_t = Pf_t + Jt @ P_diff @ Jt.T; P_s_t = (P_s_t + P_s_t.T) / 2.0
            x_s_t_final = jnp.where(jt_valid, x_s_t, jnp.full_like(x_s_t, jnp.nan)); P_s_t_final = jnp.where(jt_valid, P_s_t, jnp.full_like(P_s_t, jnp.nan))
            return (x_s_t_final, P_s_t_final), (x_s_t_final, P_s_t_final)
        init_carry_smooth=(x_s_next, P_s_next); (_, _),(x_s_rev, P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt[-1][None, :]], axis=0); P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt[-1][None, :, :]], axis=0)
        return x_smooth, P_smooth

    # --- _simulation_smoother_single_draw (Updated: Calls default smooth) ---
    def _simulation_smoother_single_draw(self, ys: jax.Array, key: jax.random.PRNGKey, x_smooth_rts: jax.Array) -> jax.Array:
        """ Internal function for simulation smoother. Calls self.smooth() for y_star. """
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        n_s = self.n_state; n_eps = self.n_shocks; key_init, key_eps, key_eta = random.split(key, 3)
        try: init_P_reg = self.init_P + _MACHINE_EPSILON * self.I_s; L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype); x0_star = self.init_x + L0 @ z0
        except Exception: x0_star = self.init_x
        eps_star = random.normal(key_eps, (Tsteps, n_eps), dtype=self.R.dtype) if n_eps > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
        def state_sim_step(x_prev_star, eps_t_star): shock_term = self.R @ eps_t_star if n_eps > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype); x_curr_star = self.T @ x_prev_star + shock_term; return x_curr_star, x_curr_star
        _, x_star_scan = lax.scan(state_sim_step, x0_star, eps_star); x_star = jnp.concatenate([x0_star[None, :], x_star_scan], axis=0)[:-1]
        eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
        y_star = jnp.einsum('ij,tj->ti', self.C, x_star) + eta_star
        # Smooth y_star: Always uses robust filter internally now
        x_smooth_star, _ = self.smooth(y_star) # No need for use_likelihood_filter flag
        smooth_star_valid = jnp.all(jnp.isfinite(x_smooth_star))
        x_draw = jnp.where(smooth_star_valid, x_star + (x_smooth_rts - x_smooth_star), jnp.full_like(x_star, jnp.nan))
        return x_draw

    # --- simulation_smoother (Updated: Calls default smooth for original ys) ---
    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1, filter_results: Optional[Dict] = None) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """ Runs the Durbin-Koopman simulation smoother. """
        if num_draws <= 0: raise ValueError("num_draws must be >= 1."); ys_arr = jnp.asarray(ys, dtype=self.C.dtype); Tsteps = ys_arr.shape[0]
        empty_state = lambda *s: jnp.empty(s + (self.n_state,), dtype=self.init_x.dtype)
        if Tsteps == 0:
            if num_draws == 1: return empty_state(0)
            else: return empty_state(0), empty_state(0), empty_state(num_draws, 0)
        # Smooth original data `ys` using the default smooth method
        x_smooth_rts_orig, P_smooth_rts_orig = self.smooth(ys_arr, filter_results=filter_results) # No flag needed
        if not jnp.all(jnp.isfinite(x_smooth_rts_orig)):
            print("Error: Failed to compute RTS smoothed state for original data in sim smoother."); nan_state = lambda *s: jnp.full(s + (self.n_state,), jnp.nan, dtype=self.init_x.dtype)
            if num_draws == 1: return nan_state(Tsteps)
            else: return nan_state(Tsteps), nan_state(Tsteps), nan_state(num_draws, Tsteps)
        def perform_single_dk_draw_wrapper(key_single_draw): return self._simulation_smoother_single_draw(ys_arr, key_single_draw, x_smooth_rts_orig)
        keys = random.split(key, num_draws); vmapped_smoother = vmap(perform_single_dk_draw_wrapper, in_axes=(0,)); all_draws_jax = vmapped_smoother(keys)
        if num_draws == 1: return all_draws_jax[0]
        else:
            if not jnp.all(jnp.isfinite(all_draws_jax)): print("Warning simulation_smoother: Some draws contain non-finite values.")
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0); median_smooth_sim = jnp.nanpercentile(all_draws_jax, 50.0, axis=0, method='linear') if jnp.any(jnp.isnan(all_draws_jax)) else jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- Standalone Simulation Function (Unchanged) ---
def simulate_state_space( T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike, key: jax.random.PRNGKey, num_steps: int ) -> Tuple[jax.Array, jax.Array]:
    desired_dtype = T.dtype; T_jax=jnp.asarray(T,dtype=desired_dtype); R_jax=jnp.asarray(R,dtype=desired_dtype); C_jax=jnp.asarray(C,dtype=desired_dtype); H_jax=jnp.asarray(H,dtype=desired_dtype); init_x_jax=jnp.asarray(init_x,dtype=desired_dtype); init_P_jax=jnp.asarray(init_P,dtype=desired_dtype)
    n_state=T_jax.shape[0]; n_shocks=R_jax.shape[1] if R_jax.ndim==2 else 0; n_obs=C_jax.shape[0]; key_init, key_state_noise, key_obs_noise = random.split(key, 3)
    try: init_P_reg = init_P_jax + _MACHINE_EPSILON * jnp.eye(n_state, dtype=desired_dtype); L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_state,), dtype=desired_dtype); x0 = init_x_jax + L0 @ z0
    except Exception: x0 = init_x_jax
    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_shocks), dtype=desired_dtype) if n_shocks > 0 else jnp.zeros((num_steps, 0), dtype=desired_dtype)
    try: H_obs_reg = H_jax + _MACHINE_EPSILON * jnp.eye(n_obs, dtype=desired_dtype); obs_noise = random.multivariate_normal(key_obs_noise, jnp.zeros(n_obs, dtype=desired_dtype), H_obs_reg, shape=(num_steps,), dtype=desired_dtype)
    except Exception as e: print(f"Warning simulate_state_space: Could not simulate observation noise: {e}. Using zeros."); obs_noise = jnp.zeros((num_steps, n_obs), dtype=desired_dtype)
    def simulation_step(x_prev, noise_t): eps_t, eta_t = noise_t; shock_term = R_jax @ eps_t if n_shocks > 0 else jnp.zeros(n_state, dtype=x_prev.dtype); x_curr = T_jax @ x_prev + shock_term; y_curr = C_jax @ x_curr + eta_t; return x_curr, (x_curr, y_curr)
    combined_noise = (state_shocks_std_normal, obs_noise); final_state, (states_scan, observations_scan) = lax.scan(simulation_step, x0, combined_noise)
    return states_scan, observations_scan

# --- END OF FILE Kalman_filter_jax.py ---