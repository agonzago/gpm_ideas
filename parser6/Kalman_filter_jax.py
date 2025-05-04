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
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        # ... (init code from previous response) ...
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype); self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype); self.H = jnp.asarray(H, dtype=desired_dtype)
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype); self.init_P = jnp.asarray(init_P, dtype=desired_dtype)
        n_state = self.T.shape[0]; n_obs = self.C.shape[0]; n_shocks = self.R.shape[1] if self.R.ndim == 2 else 0
        # Validation...
        self.n_state = n_state; self.n_obs = n_obs; self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)
        if self.n_shocks > 0: self.state_cov = self.R @ self.R.T
        else: self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)
        # Cholesky setup...
        self.H_stable = self.H; self.log_det_H_term = 0.0
        try:
            H_reg = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.L_H = jnp.linalg.cholesky(H_reg)
            self.simulate_obs_noise = self._simulate_obs_noise_chol
            self.log_det_H_term = 2 * jnp.sum(jnp.log(jnp.diag(self.L_H)))
        except Exception:
            self.H_stable = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.simulate_obs_noise = self._simulate_obs_noise_mvn
            sign, log_det = jnp.linalg.slogdet(self.H_stable); self.log_det_H_term = log_det

    def _simulate_obs_noise_chol(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        z_eta = random.normal(key, tuple(shape) + (self.n_obs,), dtype=self.H.dtype)
        return z_eta @ self.L_H.T

    def _simulate_obs_noise_mvn(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try:
            eta = random.multivariate_normal(key, jnp.zeros((self.n_obs,), dtype=self.H.dtype), self.H_stable, shape=mvn_shape, dtype=self.H.dtype)
            return eta
        except Exception: return jnp.zeros(tuple(shape) + (self.n_obs,), dtype=self.H.dtype)

    # --- Version 1: Filter for Likelihood Calculation (Robust to tracing NaNs in ys) ---
    def filter_for_likelihood(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """ Uses lax.cond for NaN handling - suitable for MCMC likelihood. """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, C, H, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov

        def step_for_likelihood(carry, y_t):
            x_prev_filt, P_prev_filt = carry
            x_pred_t = T_mat @ x_prev_filt
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov

            is_missing = jnp.all(jnp.isnan(y_t)) # Check all NaN
            y_obs = jnp.nan_to_num(y_t, nan=0.0) # Replace NaN with 0
            v = y_obs - (C @ x_pred_t) # Full innovation

            PCt = P_pred_t @ C.T
            S = C @ PCt + H
            S_reg = S + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=S.dtype)

            def perform_update():
                try:
                    L_S = jnp.linalg.cholesky(S_reg)
                    Y = jax.scipy.linalg.solve_triangular(L_S, PCt, lower=True)
                    K = jax.scipy.linalg.solve_triangular(L_S.T, Y, lower=False)
                except Exception:
                    try: K = jnp.linalg.solve(S_reg, PCt)
                    except Exception: S_pinv = jnp.linalg.pinv(S_reg); K = PCt @ S_pinv
                x_filt_t_up = x_pred_t + K @ v
                IKC = I_s - K @ C
                P_filt_t_up = IKC @ P_pred_t @ IKC.T + K @ H @ K.T
                P_filt_t_up = (P_filt_t_up + P_filt_t_up.T) / 2.0
                return x_filt_t_up, P_filt_t_up

            x_filt_t, P_filt_t = lax.cond(
                is_missing, lambda: (x_pred_t, P_pred_t), perform_update
            )

            # Log Likelihood (approximate for partial NaNs)
            log_pi_term = jnp.log(2 * jnp.pi) * self.n_obs
            sign, log_det_S = jnp.linalg.slogdet(S_reg)
            try:
                solved_term = jax.scipy.linalg.solve(S_reg, v, assume_a='pos')
                mahalanobis_dist = v @ solved_term
            except Exception: mahalanobis_dist = 1e18
            ll_t = -0.5 * (log_pi_term + log_det_S + mahalanobis_dist)
            safe_ll_t = jnp.where(is_missing | (sign <= 0), 0.0, ll_t)

            outputs = {
                'x_pred': x_pred_t, 'P_pred': P_pred_t,
                'x_filt': x_filt_t, 'P_filt': P_filt_t,
                'innovations': v, 'innovation_cov': S,
                'log_likelihood_contributions': safe_ll_t
            }
            return (x_filt_t, P_filt_t), outputs

        init_carry = (self.init_x, self.init_P)
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), scan_outputs = lax.scan(step_for_likelihood, init_carry, ys_reshaped)
        return scan_outputs

    # --- Version 2: Filter assuming Static NaN pattern (or no NaNs) ---
    def filter(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """
        Applies the Kalman filter, optimized for a STATIC pattern of missing values.
        Should be called outside JAX transformations or on concrete data.
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, C_full, H_full, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov

        T_steps = ys_arr.shape[0]
        if T_steps == 0: # Handle empty input
             empty_float=lambda *s:jnp.empty(s,dtype=I_s.dtype); empty_innov=lambda *s:jnp.empty(s,dtype=self.C.dtype)
             n_obs_actual=0
             if ys.shape[0]>0 and ys.shape[1]>0:
                 try: valid_obs_idx_np=onp.where(~onp.isnan(onp.asarray(ys[0])))[0]; n_obs_actual=len(valid_obs_idx_np)
                 except IndexError: pass
             return {'x_pred': empty_float(0,self.n_state), 'P_pred': empty_float(0,self.n_state,self.n_state), 'x_filt': empty_float(0,self.n_state), 'P_filt': empty_float(0,self.n_state,self.n_state), 'innovations': empty_innov(0,n_obs_actual), 'innovation_cov': empty_innov(0,n_obs_actual,n_obs_actual), 'log_likelihood_contributions': empty_float(0)}

        # --- Preprocessing using NumPy on potentially concrete ys_arr[0] ---
        first_obs_np = onp.asarray(ys_arr[0]) # Use NumPy
        if onp.any(onp.isnan(first_obs_np)):
            valid_obs_idx_np = onp.where(~onp.isnan(first_obs_np))[0]
            valid_obs_idx = jnp.array(valid_obs_idx_np) # Convert to JAX only if needed
            n_obs_actual = len(valid_obs_idx)
            C_obs = C_full.at[valid_obs_idx, :].get() # Use .at[].get() for JAX compatibility
            H_obs = H_full.at[jnp.ix_(valid_obs_idx, valid_obs_idx)].get()
            I_obs = jnp.eye(n_obs_actual, dtype=I_s.dtype)
            select_obs = lambda y: y.at[valid_obs_idx].get()
        else: # No NaNs
            n_obs_actual = self.n_obs
            C_obs = C_full; H_obs = H_full; I_obs = jnp.eye(self.n_obs, dtype=I_s.dtype)
            select_obs = lambda y: y

        if n_obs_actual == 0: # All missing
            C_obs = jnp.empty((0, self.n_state), dtype=C_full.dtype)
            H_obs = jnp.empty((0, 0), dtype=H_full.dtype)
            I_obs = jnp.empty((0, 0), dtype=I_s.dtype)
            select_obs = lambda y: jnp.empty((0,), dtype=y.dtype)

        # --- Kalman Filter Step Function ---
        def step_static_nan(carry, y_t_full):
            x_prev_filt, P_prev_filt = carry
            x_pred_t = T_mat @ x_prev_filt
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov

            y_obs_t = select_obs(y_t_full)
            v_obs = y_obs_t - (C_obs @ x_pred_t)

            PCt_obs = P_pred_t @ C_obs.T
            S_obs = C_obs @ PCt_obs + H_obs
            S_obs_reg = S_obs + _MACHINE_EPSILON * I_obs

            try:
                L_S_obs = jnp.linalg.cholesky(S_obs_reg)
                Y_obs = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs, lower=True)
                K = jax.scipy.linalg.solve_triangular(L_S_obs.T, Y_obs, lower=False)
            except Exception:
                 try: K = jnp.linalg.solve(S_obs_reg, PCt_obs)
                 except Exception:
                     try: S_obs_pinv = jnp.linalg.pinv(S_obs_reg); K = PCt_obs @ S_obs_pinv
                     except Exception: K = jnp.zeros((self.n_state, n_obs_actual), dtype=x_pred_t.dtype)

            x_filt_t = x_pred_t + K @ v_obs
            IKC_obs = I_s - K @ C_obs
            P_filt_t = IKC_obs @ P_pred_t @ IKC_obs.T + K @ H_obs @ K.T
            P_filt_t = (P_filt_t + P_filt_t.T) / 2.0

            log_pi_term = jnp.log(2 * jnp.pi) * n_obs_actual
            sign, log_det_S_obs = jnp.linalg.slogdet(S_obs_reg)
            try:
                solved_term_obs = jax.scipy.linalg.solve(S_obs_reg, v_obs, assume_a='pos')
                mahalanobis_dist_obs = v_obs @ solved_term_obs
            except Exception: mahalanobis_dist_obs = jnp.where(n_obs_actual > 0, 1e18, 0.0)
            ll_t = -0.5 * (log_pi_term + log_det_S_obs + mahalanobis_dist_obs)
            safe_ll_t = jnp.where(n_obs_actual == 0, 0.0, jnp.where(sign > 0, ll_t, -1e18))

            outputs = { # Return full dimension innovations/cov for consistency? No, return observed.
                'x_pred': x_pred_t, 'P_pred': P_pred_t,
                'x_filt': x_filt_t, 'P_filt': P_filt_t,
                'innovations': v_obs,           # Return observed innovations
                'innovation_cov': S_obs,        # Return observed innovation cov
                'log_likelihood_contributions': safe_ll_t
            }
            return (x_filt_t, P_filt_t), outputs

        init_carry = (self.init_x, self.init_P)
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), scan_outputs = lax.scan(step_static_nan, init_carry, ys_reshaped)
        return scan_outputs

    # --- log_likelihood now explicitly calls filter_for_likelihood ---
    def log_likelihood(self, ys: ArrayLike) -> jax.Array:
        """ Computes the log-likelihood using the robust filter version. """
        # print("Calculating log-likelihood...") # Quieter
        filter_results = self.filter_for_likelihood(ys) # <<< CALL LIKELIHOOD VERSION
        total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
        # print(f"Log-likelihood calculated: {total_log_likelihood}") # Quieter
        return total_log_likelihood

    # --- smooth now explicitly calls the main filter (Static NaN) by default ---
    def smooth(self, ys: ArrayLike, filter_results: Optional[Dict] = None) -> Tuple[jax.Array, jax.Array]:
        """ Applies the RTS smoother, uses main filter (Static NaN) if results not provided. """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        if filter_results is None:
            # Calls the main self.filter (Static NaN version)
            filter_outs_dict = self.filter(ys_arr)
        else:
            filter_outs_dict = filter_results
        # --- Access results by KEY from the dictionary ---
        # These keys MUST exist in the dictionary returned by self.filter
        x_pred = filter_outs_dict['x_pred']
        P_pred = filter_outs_dict['P_pred']
        x_filt = filter_outs_dict['x_filt']
        P_filt = filter_outs_dict['P_filt']
        T_mat = self.T
        # --- Smoother logic remains the same ---
        N = x_filt.shape[0]
        if N == 0: return jnp.empty((0,self.n_state),dtype=x_filt.dtype), jnp.empty((0,self.n_state,self.n_state),dtype=P_filt.dtype)
        x_s_next = x_filt[-1]; P_s_next = P_filt[-1]
        scan_inputs = (P_pred[1:][::-1], P_filt[:-1][::-1], x_pred[1:][::-1], x_filt[:-1][::-1])
        def backward_step(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            try:
                Pp_next_reg = Pp_next_t + _MACHINE_EPSILON * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)
                Jt_transpose = jax.scipy.linalg.solve(Pp_next_reg, (T_mat @ Pf_t).T, assume_a='gen')
                Jt = Jt_transpose.T
            except Exception:
                Pp_next_pinv = jnp.linalg.pinv(Pp_next_t)
                Jt = Pf_t @ T_mat.T @ Pp_next_pinv
            x_diff = x_s_next_t - xp_next_t; x_s_t = xf_t + Jt @ x_diff
            P_diff = P_s_next_t - Pp_next_t; P_s_t = Pf_t + Jt @ P_diff @ Jt.T
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            return (x_s_t, P_s_t), (x_s_t, P_s_t)
        init_carry_smooth = (x_s_next, P_s_next)
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt[-1][None, :]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt[-1][None, :, :]], axis=0)
        return x_smooth, P_smooth


    # --- simulation_smoother uses the main (Static NaN) filter by default ---
    def _simulation_smoother_single_draw(self, ys: jax.Array, key: jax.random.PRNGKey, filter_results_orig: Optional[Dict] = None) -> jax.Array:
        """ Internal simulation smoother draw. """
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        n_s = self.n_state; n_eps = self.n_shocks
        # Smooth original data using provided/calculated filter results (uses main filter if filter_results_orig=None)
        x_smooth_rts, _ = self.smooth(ys, filter_results=filter_results_orig)
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
        # Smooth simulated data (calls main filter internally)
        x_smooth_star, _ = self.smooth(y_star) # Calls main filter on y_star
        x_draw = x_star + (x_smooth_rts - x_smooth_star)
        return x_draw

    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1,
                            filter_results: Optional[Dict] = None
                            ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """ Runs Durbin-Koopman smoother, uses main filter if results not provided. """
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        Tsteps = ys_arr.shape[0]
        empty_state = jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        if Tsteps == 0:
            if num_draws == 1: return empty_state
            else: return empty_state, empty_state, jnp.empty((num_draws, 0, self.n_state), dtype=self.init_x.dtype)

        # Run filter ONCE on original data if results not provided
        if filter_results is None:
            filter_results_orig = self.filter(ys_arr) # Uses main filter
        else:
            filter_results_orig = filter_results

        if num_draws == 1:
            single_draw = self._simulation_smoother_single_draw(ys_arr, key, filter_results_orig=filter_results_orig)
            return single_draw
        else:
            # Calculate RTS on original data ONCE using provided/calculated filter results
            x_smooth_rts_common, _ = self.smooth(ys_arr, filter_results=filter_results_orig)

            # Vmapped function for steps 2-6
            def perform_single_dk_draw(key_single_draw):
                key_init, key_eps, key_eta = random.split(key_single_draw, 3)
                try:
                    init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(self.n_state, dtype=self.init_P.dtype); L0 = jnp.linalg.cholesky(init_P_reg)
                    z0 = random.normal(key_init, (self.n_state,), dtype=self.init_x.dtype); x0_star = self.init_x + L0 @ z0
                except Exception: x0_star = self.init_x
                eps_star = random.normal(key_eps, (Tsteps, self.n_shocks), dtype=self.R.dtype) if self.n_shocks > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
                def state_sim_step(x_prev_star, eps_t_star):
                     shock_term = self.R @ eps_t_star if self.n_shocks > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
                     x_curr_star = self.T @ x_prev_star + shock_term; return x_curr_star, x_curr_star
                _, x_star = lax.scan(state_sim_step, x0_star, eps_star)
                eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
                y_star = (x_star @ self.C.T) + eta_star
                x_smooth_star, _ = self.smooth(y_star) # Calls main filter internally for y_star
                x_draw = x_star + (x_smooth_rts_common - x_smooth_star)
                return x_draw

            keys = random.split(key, num_draws)
            vmapped_smoother = vmap(perform_single_dk_draw, in_axes=(0,))
            all_draws_jax = vmapped_smoother(keys)
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- Keep simulation function as is ---
def _simulate_state_space_impl( P_aug, R_aug, Omega, H_obs, init_x, init_P, key, num_steps):
    # ... (exact same implementation as before) ...
    desired_dtype = P_aug.dtype
    P_aug_jax = jnp.asarray(P_aug, dtype=desired_dtype); R_aug_jax = jnp.asarray(R_aug, dtype=desired_dtype)
    Omega_jax = jnp.asarray(Omega, dtype=desired_dtype); H_obs_jax = jnp.asarray(H_obs, dtype=desired_dtype)
    init_x_jax = jnp.asarray(init_x, dtype=desired_dtype); init_P_jax = jnp.asarray(init_P, dtype=desired_dtype)
    n_aug = P_aug_jax.shape[0]; n_aug_shocks = R_aug_jax.shape[1] if R_aug_jax.ndim == 2 else 0; n_obs = Omega_jax.shape[0]
    key_init, key_state_noise, key_obs_noise = random.split(key, 3)
    try:
        init_P_reg = init_P_jax + _MACHINE_EPSILON * jnp.eye(n_aug, dtype=desired_dtype)
        L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_aug,), dtype=desired_dtype); x0 = init_x_jax + L0 @ z0
    except Exception: x0 = init_x_jax
    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_aug_shocks), dtype=desired_dtype) if n_aug_shocks > 0 else jnp.zeros((num_steps, 0), dtype=desired_dtype)
    try:
        H_obs_reg = H_obs_jax + _MACHINE_EPSILON * jnp.eye(n_obs, dtype=desired_dtype)
        obs_noise = random.multivariate_normal(key_obs_noise, jnp.zeros(n_obs, dtype=desired_dtype), H_obs_reg, shape=(num_steps,), dtype=desired_dtype)
    except Exception as e: obs_noise = jnp.zeros((num_steps, n_obs), dtype=desired_dtype)
    def simulation_step(x_prev, noise_t):
        eps_t, eta_t = noise_t
        shock_term = R_aug_jax @ eps_t if n_aug_shocks > 0 else jnp.zeros(n_aug, dtype=x_prev.dtype)
        x_curr = P_aug_jax @ x_prev + shock_term; y_curr = Omega_jax @ x_curr + eta_t
        return x_curr, (x_curr, y_curr)
    combined_noise = (state_shocks_std_normal, obs_noise)
    final_state, (states, observations) = lax.scan(simulation_step, x0, combined_noise)
    return states, observations

simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))


# --- END OF MODIFIED FILE Kalman_filter_jax.py ---