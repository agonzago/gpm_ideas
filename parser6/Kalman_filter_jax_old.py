# --- START OF FILE Kalman_filter_jax.py ---

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp # Use numpy for potential warnings, keep core logic in JAX
from typing import Tuple, Optional, Union, Sequence, Dict, Any

# Small constant for numerical stability (jitter)
_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps # Use float64 epsilon if possible

class KalmanFilter:
    """
    Implements the standard Kalman Filter, RTS Smoother, Durbin-Koopman
    Simulation Smoother, and Log-Likelihood calculation using JAX.

    Assumes a linear Gaussian state-space model:
        x_t = T @ x_{t-1} + R @ epsilon_t,  epsilon_t ~ N(0, I)
        y_t = C @ x_t + eta_t,            eta_t ~ N(0, H)

    where `R` is the state shock transformation matrix such that the state noise
    covariance `Q = R @ R.T`.
    """
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        """
        Initializes the Kalman Filter instance.

        Args:
            T: State transition matrix [n_state, n_state].
            R: State shock transformation matrix [n_state, n_shocks].
               Assumes R @ R.T gives the state noise covariance Q.
            C: Observation matrix [n_obs, n_state].
            H: Observation noise covariance matrix [n_obs, n_obs].
            init_x: Initial state mean estimate [n_state].
            init_P: Initial state covariance estimate [n_state, n_state].
        """
        # Ensure inputs are JAX arrays and potentially convert dtype
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype)
        self.H = jnp.asarray(H, dtype=desired_dtype)
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P, dtype=desired_dtype)

        # --- Input Validation (Basic) ---
        n_state = self.T.shape[0]
        n_obs = self.C.shape[0]
        n_shocks = self.R.shape[1] if self.R.ndim == 2 else 0

        if self.T.shape != (n_state, n_state):
            raise ValueError(f"T shape mismatch: expected ({n_state},{n_state}), got {self.T.shape}")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks):
             raise ValueError(f"R shape mismatch: expected ({n_state},{n_shocks}), got {self.R.shape}")
        elif n_shocks == 0 and self.R.size != 0:
             self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        if self.C.shape != (n_obs, n_state):
            raise ValueError(f"C shape mismatch: expected ({n_obs},{n_state}), got {self.C.shape}")
        if self.H.shape != (n_obs, n_obs):
            raise ValueError(f"H shape mismatch: expected ({n_obs},{n_obs}), got {self.H.shape}")
        if self.init_x.shape != (n_state,):
            raise ValueError(f"init_x shape mismatch: expected ({n_state},), got {self.init_x.shape}")
        if self.init_P.shape != (n_state, n_state):
            raise ValueError(f"init_P shape mismatch: expected ({n_state},{n_state}), got {self.init_P.shape}")

        self.n_state = n_state
        self.n_obs = n_obs
        self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)

        if self.n_shocks > 0:
            self.state_cov = self.R @ self.R.T
        else:
            self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)

        self.H_stable = self.H
        self.log_det_H_term = 0.0
        try:
            H_reg = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.L_H = jnp.linalg.cholesky(H_reg)
            self.simulate_obs_noise = self._simulate_obs_noise_chol
            self.log_det_H_term = 2 * jnp.sum(jnp.log(jnp.diag(self.L_H)))
        except Exception:
            # print("Warning: Cholesky decomposition failed for H. Using mvn.") # Quieter
            try: min_eig = jnp.min(jnp.linalg.eigvalsh(self.H))
            except Exception: pass # print("Warning: Could not compute eigenvalues for H stability check.")
            self.H_stable = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.simulate_obs_noise = self._simulate_obs_noise_mvn
            sign, log_det = jnp.linalg.slogdet(self.H_stable)
            # if sign <= 0: print(f"Warning: slogdet(H_stable) indicates non-positive-definite matrix") # Quieter
            self.log_det_H_term = log_det

    def _simulate_obs_noise_chol(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        z_eta = random.normal(key, tuple(shape) + (self.n_obs,), dtype=self.H.dtype)
        return z_eta @ self.L_H.T

    def _simulate_obs_noise_mvn(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try:
            eta = random.multivariate_normal(
                key, jnp.zeros((self.n_obs,), dtype=self.H.dtype),
                self.H_stable, shape=mvn_shape, dtype=self.H.dtype
            )
            return eta
        except Exception as e:
            # print(f"Error during multivariate_normal simulation: {e}") # Quieter
            # print("Returning zeros for observation noise.")
            return jnp.zeros(tuple(shape) + (self.n_obs,), dtype=self.H.dtype)

    def filter(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """
        Applies the Kalman filter, optimized for a STATIC pattern of missing values.
        It assumes that if ys[t, i] is NaN, then ys[k, i] is NaN for all k.
        The pattern of missing values is determined from the first time step ys[0].
        (Static NaN version)

        Args:
            ys: Observations array, shape `[T, n_obs]`. NaN indicates missing values
                with a pattern constant over time.

        Returns:
            A dictionary containing filter results and log-likelihood contributions.
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, C_full, H_full, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov

        T_steps = ys_arr.shape[0]
        if T_steps == 0: # Handle empty input
             empty_float = lambda *s: jnp.empty(s, dtype=I_s.dtype)
             empty_innov = lambda *s: jnp.empty(s, dtype=self.C.dtype)
             n_obs_actual = 0
             try:
                  valid_obs_idx_np = onp.where(~onp.isnan(onp.asarray(ys[0])))[0]
                  n_obs_actual = len(valid_obs_idx_np)
             except IndexError: pass
             return {
                 'x_pred': empty_float(0, self.n_state), 'P_pred': empty_float(0, self.n_state, self.n_state),
                 'x_filt': empty_float(0, self.n_state), 'P_filt': empty_float(0, self.n_state, self.n_state),
                 'innovations': empty_innov(0, n_obs_actual), 'innovation_cov': empty_innov(0, n_obs_actual, n_obs_actual),
                 'log_likelihood_contributions': empty_float(0)
             }

        # --- Preprocessing: Identify observed subset (assuming static pattern) ---
        try:
            valid_obs_idx = jnp.where(~jnp.isnan(ys_arr[0]))[0]
        except IndexError:
             valid_obs_idx = jnp.array([], dtype=jnp.int32)

        n_obs_actual = len(valid_obs_idx)

        if n_obs_actual == self.n_obs:
            C_obs = C_full
            H_obs = H_full
            I_obs = jnp.eye(self.n_obs, dtype=I_s.dtype)
            select_obs = lambda y: y
        elif n_obs_actual > 0:
            # print(f"Kalman Filter: Detected static missing values. Using {n_obs_actual}/{self.n_obs} observed variables.") # Quieter
            C_obs = C_full[valid_obs_idx, :]
            H_obs = H_full[jnp.ix_(valid_obs_idx, valid_obs_idx)]
            I_obs = jnp.eye(n_obs_actual, dtype=I_s.dtype)
            select_obs = lambda y: y[valid_obs_idx]
        else:
            # print("Kalman Filter: All observations missing. Running prediction steps only.") # Quieter
            C_obs = jnp.empty((0, self.n_state), dtype=C_full.dtype)
            H_obs = jnp.empty((0, 0), dtype=H_full.dtype)
            I_obs = jnp.empty((0, 0), dtype=I_s.dtype)
            select_obs = lambda y: jnp.empty((0,), dtype=y.dtype)

        # --- Kalman Filter Step Function (using observed subset) ---
        def step_static_nan(carry, y_t_full):
            x_prev_filt, P_prev_filt = carry

            # --- Prediction Step ---
            x_pred_t = T_mat @ x_prev_filt
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov

            # --- Update Step ---
            y_obs_t = select_obs(y_t_full)
            y_pred_obs = C_obs @ x_pred_t
            v_obs = y_obs_t - y_pred_obs

            PCt_obs = P_pred_t @ C_obs.T
            S_obs = C_obs @ PCt_obs + H_obs
            S_obs_reg = S_obs + _MACHINE_EPSILON * I_obs

            # --- Kalman Gain Calculation ---
            try:
                L_S_obs = jnp.linalg.cholesky(S_obs_reg)
                Y_obs = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs, lower=True)
                K = jax.scipy.linalg.solve_triangular(L_S_obs.T, Y_obs, lower=False)
            except Exception:
                try: K = jnp.linalg.solve(S_obs_reg, PCt_obs)
                except Exception:
                    try:
                        S_obs_pinv = jnp.linalg.pinv(S_obs_reg)
                        K = PCt_obs @ S_obs_pinv
                    except Exception:
                        K = jnp.zeros((self.n_state, n_obs_actual), dtype=x_pred_t.dtype)

            # --- State and Covariance Update ---
            x_filt_t = x_pred_t + K @ v_obs
            IKC_obs = I_s - K @ C_obs
            P_filt_t = IKC_obs @ P_pred_t @ IKC_obs.T + K @ H_obs @ K.T
            P_filt_t = (P_filt_t + P_filt_t.T) / 2.0

            # --- Log Likelihood Contribution Calculation ---
            log_pi_term = jnp.log(2 * jnp.pi) * n_obs_actual
            sign, log_det_S_obs = jnp.linalg.slogdet(S_obs_reg)

            try:
                solved_term_obs = jax.scipy.linalg.solve(S_obs_reg, v_obs, assume_a='pos')
                mahalanobis_dist_obs = v_obs @ solved_term_obs
            except Exception:
                 mahalanobis_dist_obs = jnp.where(n_obs_actual > 0, 1e18, 0.0)

            ll_t = -0.5 * (log_pi_term + log_det_S_obs + mahalanobis_dist_obs)
            safe_ll_t = jnp.where(n_obs_actual == 0, 0.0,
                                  jnp.where(sign > 0, ll_t, -1e18))

            outputs = {
                'x_pred': x_pred_t, 'P_pred': P_pred_t,
                'x_filt': x_filt_t, 'P_filt': P_filt_t,
                'innovations': v_obs,
                'innovation_cov': S_obs,
                'log_likelihood_contributions': safe_ll_t
            }
            return (x_filt_t, P_filt_t), outputs

        # --- Run the Scan ---
        init_carry = (self.init_x, self.init_P)
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), scan_outputs = lax.scan(step_static_nan, init_carry, ys_reshaped)

        return scan_outputs

    # --- Keep log_likelihood method as is ---
    def log_likelihood(self, ys: ArrayLike) -> jax.Array:
        # print("Calculating log-likelihood...") # Quieter
        filter_results = self.filter(ys)
        total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
        # print(f"Log-likelihood calculated: {total_log_likelihood}") # Quieter
        return total_log_likelihood

    # --- UPDATED smooth method ---
    def smooth(self, ys: ArrayLike, filter_results: Optional[Dict] = None) -> Tuple[jax.Array, jax.Array]:
        """
        Applies the Rauch-Tung-Striebel (RTS) smoother.
        Can optionally accept pre-computed filter results to avoid re-filtering.

        Args:
            ys: Observations array, shape `[T, n_obs]`. NaN indicates missing.
            filter_results: (Optional) Dictionary output from self.filter(ys).
                            If None, self.filter(ys) will be called internally.

        Returns:
            A tuple containing:
                - x_smooth: Smoothed state means `E[x_t | y_{1:T}]` [T, n_state]
                - P_smooth: Smoothed state covariances `Cov(x_t | y_{1:T})` [T, n_state, n_state]
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)

        if filter_results is None:
            filter_outs_dict = self.filter(ys_arr)
        else:
            filter_outs_dict = filter_results

        x_pred = filter_outs_dict['x_pred']
        P_pred = filter_outs_dict['P_pred']
        x_filt = filter_outs_dict['x_filt']
        P_filt = filter_outs_dict['P_filt']
        T_mat = self.T

        N = x_filt.shape[0]
        if N == 0:
            return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)

        x_s_next = x_filt[-1]
        P_s_next = P_filt[-1]
        scan_inputs = (P_pred[1:][::-1], P_filt[:-1][::-1], x_pred[1:][::-1], x_filt[:-1][::-1])

        def backward_step(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            # TPf = Pf_t @ T_mat.T # This order might be more stable if Pf_t is better conditioned
            TPf = T_mat @ Pf_t # Revert to original order T @ P_filt_t for gain calc
            try:
                Pp_next_reg = Pp_next_t + _MACHINE_EPSILON * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)
                # Solve Pp_next_reg @ Jt' = (T @ Pf)' = Pf' @ T'
                # Note: Jt = Pf_t @ T.T @ inv(Pp_next_t)
                Jt_transpose = jax.scipy.linalg.solve(Pp_next_reg, (T_mat @ Pf_t).T, assume_a='gen') # Use 'gen'
                Jt = Jt_transpose.T
            except Exception:
                Pp_next_pinv = jnp.linalg.pinv(Pp_next_t)
                Jt = Pf_t @ T_mat.T @ Pp_next_pinv # Original gain calculation formula
            x_diff = x_s_next_t - xp_next_t
            x_s_t = xf_t + Jt @ x_diff
            P_diff = P_s_next_t - Pp_next_t
            P_s_t = Pf_t + Jt @ P_diff @ Jt.T
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            return (x_s_t, P_s_t), (x_s_t, P_s_t)

        init_carry_smooth = (x_s_next, P_s_next)
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)

        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt[-1][None, :]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt[-1][None, :, :]], axis=0)
        return x_smooth, P_smooth


    # --- UPDATED simulation smoother single draw ---
    def _simulation_smoother_single_draw(self, ys: jax.Array, key: jax.random.PRNGKey, filter_results_orig: Optional[Dict] = None) -> jax.Array:
        """ Internal: Performs one draw of Durbin-Koopman simulation smoother."""
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        n_s = self.n_state
        n_eps = self.n_shocks

        # Step 1: Use provided filter results for smoothing original data
        x_smooth_rts, _ = self.smooth(ys, filter_results=filter_results_orig)

        key_init, key_eps, key_eta = random.split(key, 3)

        # Step 2: Simulate initial state x0*
        try:
            init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(n_s, dtype=self.init_P.dtype)
            L0 = jnp.linalg.cholesky(init_P_reg)
            z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
            x0_star = self.init_x + L0 @ z0
        except Exception: x0_star = self.init_x

        # Step 3: Simulate state trajectory x*
        eps_star = random.normal(key_eps, (Tsteps, n_eps), dtype=self.R.dtype) if n_eps > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
        def state_sim_step(x_prev_star, eps_t_star):
            shock_term = self.R @ eps_t_star if n_eps > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
            x_curr_star = self.T @ x_prev_star + shock_term
            return x_curr_star, x_curr_star
        _, x_star = lax.scan(state_sim_step, x0_star, eps_star)

        # Step 4: Simulate observation trajectory y*
        eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
        y_star = (x_star @ self.C.T) + eta_star

        # Step 5: RTS smoother on simulated data y* (runs filter internally)
        x_smooth_star, _ = self.smooth(y_star)

        # Step 6: Combine results
        x_draw = x_star + (x_smooth_rts - x_smooth_star)
        return x_draw

    # --- UPDATED simulation smoother main method ---
    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1,
                            filter_results: Optional[Dict] = None
                            ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """ Runs the Durbin-Koopman simulation smoother. """
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        Tsteps = ys_arr.shape[0]
        empty_state = jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        if Tsteps == 0:
            if num_draws == 1: return empty_state
            else: return empty_state, empty_state, jnp.empty((num_draws, 0, self.n_state), dtype=self.init_x.dtype)

        # Run filter ONCE if results not provided
        if filter_results is None:
            filter_results_orig = self.filter(ys_arr)
        else:
            filter_results_orig = filter_results

        if num_draws == 1:
            single_draw = self._simulation_smoother_single_draw(ys_arr, key, filter_results_orig=filter_results_orig)
            return single_draw
        else:
            # Calculate RTS on original data ONCE
            x_smooth_rts_common, _ = self.smooth(ys_arr, filter_results=filter_results_orig)

            # Vmapped function for steps 2-6
            def perform_single_dk_draw(key_single_draw):
                # This internal call now implicitly passes the original filter results
                # via the x_smooth_rts_common calculation above.
                # It recalculates the smoother for the simulated data internally.
                key_init, key_eps, key_eta = random.split(key_single_draw, 3)
                try:
                    init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(self.n_state, dtype=self.init_P.dtype)
                    L0 = jnp.linalg.cholesky(init_P_reg)
                    z0 = random.normal(key_init, (self.n_state,), dtype=self.init_x.dtype)
                    x0_star = self.init_x + L0 @ z0
                except Exception: x0_star = self.init_x
                eps_star = random.normal(key_eps, (Tsteps, self.n_shocks), dtype=self.R.dtype) if self.n_shocks > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
                def state_sim_step(x_prev_star, eps_t_star):
                     shock_term = self.R @ eps_t_star if self.n_shocks > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
                     x_curr_star = self.T @ x_prev_star + shock_term
                     return x_curr_star, x_curr_star
                _, x_star = lax.scan(state_sim_step, x0_star, eps_star)
                eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
                y_star = (x_star @ self.C.T) + eta_star
                x_smooth_star, _ = self.smooth(y_star) # Runs filter internally for y_star
                x_draw = x_star + (x_smooth_rts_common - x_smooth_star)
                return x_draw

            keys = random.split(key, num_draws)
            # print(f" Performing {num_draws} simulation smoother draws (vectorized)...") # Quieter
            vmapped_smoother = vmap(perform_single_dk_draw, in_axes=(0,))
            all_draws_jax = vmapped_smoother(keys)
            # print(" Finished smoothing calculations.") # Quieter
            # print(" Calculating statistics (mean, median)...") # Quieter
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            # print(" Finished.") # Quieter
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- Keep simulation function as is ---
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
    n_aug_shocks = R_aug_jax.shape[1] if R_aug_jax.ndim == 2 else 0
    n_obs = Omega_jax.shape[0]
    key_init, key_state_noise, key_obs_noise = random.split(key, 3)
    try:
        init_P_reg = init_P_jax + _MACHINE_EPSILON * jnp.eye(n_aug, dtype=desired_dtype)
        L0 = jnp.linalg.cholesky(init_P_reg)
        z0 = random.normal(key_init, (n_aug,), dtype=desired_dtype)
        x0 = init_x_jax + L0 @ z0
    except Exception: x0 = init_x_jax
    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_aug_shocks), dtype=desired_dtype) if n_aug_shocks > 0 else jnp.zeros((num_steps, 0), dtype=desired_dtype)
    try:
        H_obs_reg = H_obs_jax + _MACHINE_EPSILON * jnp.eye(n_obs, dtype=desired_dtype)
        obs_noise = random.multivariate_normal(key_obs_noise, jnp.zeros(n_obs, dtype=desired_dtype), H_obs_reg, shape=(num_steps,), dtype=desired_dtype)
    except Exception as e: obs_noise = jnp.zeros((num_steps, n_obs), dtype=desired_dtype)
    def simulation_step(x_prev, noise_t):
        eps_t, eta_t = noise_t
        shock_term = R_aug_jax @ eps_t if n_aug_shocks > 0 else jnp.zeros(n_aug, dtype=x_prev.dtype)
        x_curr = P_aug_jax @ x_prev + shock_term
        y_curr = Omega_jax @ x_curr + eta_t
        return x_curr, (x_curr, y_curr)
    combined_noise = (state_shocks_std_normal, obs_noise)
    final_state, (states, observations) = lax.scan(simulation_step, x0, combined_noise)
    return states, observations

simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))


# --- END OF MODIFIED FILE Kalman_filter_jax.py ---