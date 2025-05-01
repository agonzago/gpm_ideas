# --- START OF MODIFIED FILE Kalman_filter_jax.py ---

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
        n_shocks = self.R.shape[1] # Can be 0 if no state shocks

        if self.T.shape != (n_state, n_state):
            raise ValueError(f"T shape mismatch: expected ({n_state},{n_state}), got {self.T.shape}")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks): # Allow R to be empty if n_shocks=0
             raise ValueError(f"R shape mismatch: expected ({n_state},{n_shocks}), got {self.R.shape}")
        elif n_shocks == 0 and self.R.size != 0: # If n_shocks is 0, R should be empty or representable as such
             # JAX might represent this differently, check size. Let's ensure R is correctly shaped if n_shocks=0
             self.R = jnp.zeros((n_state, 0), dtype=desired_dtype) # Ensure correct empty shape
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
        # Precompute state noise covariance Q = R @ R.T
        # Handle case with zero shocks properly
        if self.n_shocks > 0:
            self.state_cov = self.R @ self.R.T
        else:
            self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)


        # Precompute Cholesky of H if possible for observation noise simulation & stability
        self.H_stable = self.H # Default to original H
        self.log_det_H_term = 0.0 # For likelihood calculation fallback if needed
        try:
            # Add slight jitter for robustness before Cholesky
            H_reg = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.L_H = jnp.linalg.cholesky(H_reg)
            self.simulate_obs_noise = self._simulate_obs_noise_chol
            # Calculate log_det term using Cholesky: log(det(H)) = 2 * sum(log(diag(L_H)))
            self.log_det_H_term = 2 * jnp.sum(jnp.log(jnp.diag(self.L_H)))
        except Exception: # Use broader Exception for JAX linalg errors
            print("Warning: Cholesky decomposition failed for H. "
                  "Using multivariate_normal for observation noise simulation (may be slower). "
                  "Adding jitter to H for mvn stability.")
            # Ensure H is at least PSD for mvn by adding jitter
            try:
                min_eig = jnp.min(jnp.linalg.eigvalsh(self.H))
                if min_eig < -1e-9: # Check for significantly negative eigenvalues
                     print(f"Warning: H has significant negative eigenvalues ({min_eig}). Simulation/Likelihood might fail.")
            except Exception:
                print("Warning: Could not compute eigenvalues for H stability check.")

            # Add jitter for numerical stability in mvn if Cholesky failed
            self.H_stable = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=desired_dtype)
            self.simulate_obs_noise = self._simulate_obs_noise_mvn
            # Calculate log_det term using slogdet for stability
            sign, log_det = jnp.linalg.slogdet(self.H_stable)
            if sign <= 0:
                print(f"Warning: slogdet(H_stable) indicates non-positive-definite matrix (sign={sign}). Likelihood calculation may be inaccurate.")
                # Assign a large negative number or handle as error? Using logdet for now.
            self.log_det_H_term = log_det # Store log determinant

    def _simulate_obs_noise_chol(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        """Simulate observation noise using precomputed Cholesky factor."""
        # Shape needs to be a tuple for random.normal
        z_eta = random.normal(key, tuple(shape) + (self.n_obs,), dtype=self.H.dtype)
        # eta = L_H @ z_eta (if z_eta is [n_obs, ...])
        # For shape [..., n_obs], use eta = z_eta @ L_H.T
        return z_eta @ self.L_H.T

    def _simulate_obs_noise_mvn(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        """Simulate observation noise using multivariate_normal (fallback)."""
        mvn_shape = tuple(shape) if len(shape) > 0 else () # mvn needs shape for >=1 samples
        try:
            # Use the potentially stabilized H matrix
            eta = random.multivariate_normal(
                key,
                jnp.zeros((self.n_obs,), dtype=self.H.dtype),
                self.H_stable, # Use H_stable which might have jitter
                shape=mvn_shape,
                dtype=self.H.dtype
            )
            return eta
        except Exception as e:
            # Catch potential errors from mvn (e.g., if H_stable still not PSD enough)
            print(f"Error during multivariate_normal simulation: {e}")
            print("Returning zeros for observation noise.")
            return jnp.zeros(tuple(shape) + (self.n_obs,), dtype=self.H.dtype)


    def filter(self, ys: ArrayLike) -> Dict[str, jax.Array]:
        """
        Applies the Kalman filter.

        Handles missing observations (NaNs):
        - If an entire observation vector `y_t` is NaN, the update step is skipped
          (filtered state = predicted state), and likelihood contribution is 0.
        - If only *some* elements of `y_t` are NaN: This implementation replaces
          those NaNs with 0.0 for the update calculation (approximation).
          The likelihood calculation uses all dimensions, which is also an
          approximation in this case. A more correct handling would modify
          C and H based on the NaN pattern.

        Args:
            ys: Observations array, shape `[T, n_obs]`. NaN indicates missing.

        Returns:
            A dictionary containing:
                - 'x_pred': Predicted state means `E[x_t | y_{1:t-1}]` [T, n_state]
                - 'P_pred': Predicted state covariances `Cov(x_t | y_{1:t-1})` [T, n_state, n_state]
                - 'x_filt': Filtered state means `E[x_t | y_{1:t}]` [T, n_state]
                - 'P_filt': Filtered state covariances `Cov(x_t | y_{1:t})` [T, n_state, n_state]
                - 'innovations': Prediction errors `v_t = y_t - C @ x_pred_t` [T, n_obs]
                - 'innovation_cov': Innovation covariance `S_t = C @ P_pred_t @ C.T + H` [T, n_obs, n_obs]
                - 'log_likelihood_contributions': Log likelihood contribution per time step [T]
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype) # Ensure consistent dtype
        T_mat, C, H, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov # Use precomputed R @ R.T

        def step(carry, y_t):
            x_prev_filt, P_prev_filt = carry # Filtered state from t-1

            # --- Prediction Step ---
            x_pred_t = T_mat @ x_prev_filt
            P_pred_t = T_mat @ P_prev_filt @ T_mat.T + state_cov

            # --- Update Step ---
            # Check if ALL observations at time t are missing
            is_missing = jnp.all(jnp.isnan(y_t))
            # Create a mask for valid (non-NaN) observations
            valid_obs_mask = ~jnp.isnan(y_t)
            # Replace NaNs with 0.0 for computation
            y_obs = jnp.nan_to_num(y_t, nan=0.0)

            y_pred = C @ x_pred_t
            v = y_obs - y_pred              # Prediction error (innovation)

            # --- Innovation Covariance ---
            # Add jitter for robustness (especially important for likelihood calc)
            PCt = P_pred_t @ C.T
            S = C @ PCt + H
            S_reg = S + _MACHINE_EPSILON * jnp.eye(self.n_obs, dtype=S.dtype) # Regularize S

            def perform_update():
                # --- Kalman Gain Calculation (Numerically Stable) ---
                try:
                    # Solve S_reg K = PCt using Cholesky factorization of S_reg
                    L_S = jnp.linalg.cholesky(S_reg)
                    # Solve L Y = PCt for Y
                    Y = jax.scipy.linalg.solve_triangular(L_S, PCt, lower=True)
                    # Solve L.T K = Y for K
                    K = jax.scipy.linalg.solve_triangular(L_S.T, Y, lower=False)
                except Exception: # Broader catch for potential LinAlgErrors
                    # Fallback: Standard solve S K = P C.T
                    try:
                        # Use already regularized S_reg for solve
                        K = jnp.linalg.solve(S_reg, PCt)
                    except Exception:
                        # Fallback 2: Pseudo-inverse (least robust)
                        S_pinv = jnp.linalg.pinv(S_reg) # Use regularized S here too
                        K = PCt @ S_pinv

                # --- State and Covariance Update ---
                x_filt_t = x_pred_t + K @ v
                IKC = I_s - K @ C
                # Use H directly here for Joseph form, not H_stable
                P_filt_t = IKC @ P_pred_t @ IKC.T + K @ H @ K.T
                # Symmetrize P_filt
                P_filt_t = (P_filt_t + P_filt_t.T) / 2.0

                return x_filt_t, P_filt_t, K # Return K for likelihood debugging if needed

            # If observation is missing, skip update: filtered = predicted
            x_filt_t, P_filt_t, K = lax.cond(is_missing,
                                              lambda: (x_pred_t, P_pred_t, jnp.zeros((self.n_state, self.n_obs), dtype=x_pred_t.dtype)), # If missing
                                              perform_update)                   # If not missing

            # --- Log Likelihood Contribution Calculation ---
            # Contribution is 0 if observation is missing
            # Uses the full v and S_reg even if partially missing (approximation)
            # Correct handling requires modifying C, H based on valid_obs_mask
            log_pi_term = jnp.log(2 * jnp.pi) * self.n_obs # For full rank
            sign, log_det_S = jnp.linalg.slogdet(S_reg)

            # Use solve for v' S^-1 v for stability
            try:
                # Solve S_reg * x = v for x, then compute v' * x
                solved_term = jax.scipy.linalg.solve(S_reg, v, assume_a='pos')
                mahalanobis_dist = v @ solved_term
            except Exception: # If solve fails (e.g., S_reg still ill-conditioned)
                 print("Warning: Solve failed for Mahalanobis distance in filter likelihood. Using large penalty.")
                 mahalanobis_dist = 1e18 # Penalize heavily

            # Log-likelihood for this step (multivariate normal density)
            ll_t = -0.5 * (log_pi_term + log_det_S + mahalanobis_dist)

            # Zero out contribution if missing or if S was not positive definite
            safe_ll_t = jnp.where(is_missing | (sign <= 0), 0.0, ll_t)

            # Note: If sign <= 0, log_det_S might be -inf or NaN. JAX handles this,
            # but the resulting ll_t will be non-finite. The jnp.where handles this.
            # A large negative number could also be returned instead of 0.0 for non-PSD S.
            # safe_ll_t = jnp.where(is_missing, 0.0, jnp.where(sign > 0, ll_t, -1e18)) # Alt: Penalize non-PSD

            outputs = {
                'x_pred': x_pred_t, 'P_pred': P_pred_t,
                'x_filt': x_filt_t, 'P_filt': P_filt_t,
                'innovations': v, 'innovation_cov': S, # Return original S, not regularized S_reg
                'log_likelihood_contributions': safe_ll_t
            }
            return (x_filt_t, P_filt_t), outputs

        # Run the scan loop
        init_carry = (self.init_x, self.init_P)
        # Ensure ys has shape [T, n_obs] even if T=1
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), scan_outputs = lax.scan(step, init_carry, ys_reshaped)

        return scan_outputs # Dictionary of arrays

    def log_likelihood(self, ys: ArrayLike) -> jax.Array:
        """
        Computes the log-likelihood of the observations `ys`.

        Args:
            ys: Observations array, shape `[T, n_obs]`. NaN indicates missing.

        Returns:
            The total log-likelihood value.
        """
        print("Calculating log-likelihood...")
        filter_results = self.filter(ys)
        total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
        print(f"Log-likelihood calculated: {total_log_likelihood}")
        return total_log_likelihood

    def smooth(self, ys: ArrayLike) -> Tuple[jax.Array, jax.Array]:
        """
        Applies the Rauch-Tung-Striebel (RTS) smoother.

        Requires filter results first. Handles missing observations implicitly
        via the filter results.

        Args:
            ys: Observations array, shape `[T, n_obs]`. NaN indicates missing.

        Returns:
            A tuple containing:
                - x_smooth: Smoothed state means `E[x_t | y_{1:T}]` [T, n_state]
                - P_smooth: Smoothed state covariances `Cov(x_t | y_{1:T})` [T, n_state, n_state]
        """
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        # Run the filter first
        filter_outs_dict = self.filter(ys_arr)
        x_pred = filter_outs_dict['x_pred']
        P_pred = filter_outs_dict['P_pred']
        x_filt = filter_outs_dict['x_filt']
        P_filt = filter_outs_dict['P_filt']
        T_mat = self.T

        N = x_filt.shape[0]
        if N == 0:
            return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)

        # Initialize smoother recursion at the last time step
        x_s_next = x_filt[-1]
        P_s_next = P_filt[-1]

        # Stack filter results for backward pass (reverse order, exclude last step)
        # Sequence for scan: (P_pred_{t+1}, P_filt_t, x_pred_{t+1}, x_filt_t) for t = N-2 down to 0
        scan_inputs = (P_pred[1:][::-1], P_filt[:-1][::-1], x_pred[1:][::-1], x_filt[:-1][::-1])

        def backward_step(carry_smooth, scan_t):
            # carry = (x_s_{t+1}, P_s_{t+1})
            x_s_next_t, P_s_next_t = carry_smooth
            # scan_t = (P_pred_{t+1}, P_filt_t, x_pred_{t+1}, x_filt_t)
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t

            # --- Smoother Gain J_t = P_filt_t @ T' @ P_pred_{t+1}^{-1} ---
            TPf = Pf_t @ T_mat.T # P_filt_t @ T' (More stable order?) Check consistency
            # TPf = T_mat @ Pf_t # Original order: T @ P_filt_t
            try:
                # More stable: Solve P_pred_{t+1} J_t' = T @ P_filt_t for J_t'
                # Add jitter for robustness before solve
                Pp_next_reg = Pp_next_t + _MACHINE_EPSILON * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)
                # Solves Pp_next_reg @ X = TPf.T for X = Jt' -- Requires TPf shape adjustment if Pf@T' used
                # Let's use solve(A, B) solves A@X = B
                # We want Jt = Pf @ T' @ inv(Pp_next) => Pp_next @ Jt' = T @ Pf => Solve Pp_next @ X = T @ Pf
                Jt_transpose = jnp.linalg.solve(Pp_next_reg, (T_mat @ Pf_t).T) # Original T@Pf structure
                # Jt_transpose = jnp.linalg.solve(Pp_next_reg, TPf.T) # If TPf = Pf @ T'
                Jt = Jt_transpose.T
            except Exception: # Broader Exception
                # Fallback: Pseudo-inverse
                Pp_next_pinv = jnp.linalg.pinv(Pp_next_t) # Use original for pinv
                Jt = Pf_t @ T_mat.T @ Pp_next_pinv

            # --- Smoothed State Mean ---
            # x_s_t = x_filt_t + J_t @ (x_s_{t+1} - x_pred_{t+1})
            x_diff = x_s_next_t - xp_next_t
            x_s_t = xf_t + Jt @ x_diff

            # --- Smoothed State Covariance ---
            # P_s_t = P_filt_t + J_t @ (P_s_{t+1} - P_pred_{t+1}) @ J_t'
            P_diff = P_s_next_t - Pp_next_t
            P_s_t = Pf_t + Jt @ P_diff @ Jt.T

            # Symmetrize P_s_t
            P_s_t = (P_s_t + P_s_t.T) / 2.0

            return (x_s_t, P_s_t), (x_s_t, P_s_t)

        # Run the backward scan
        init_carry_smooth = (x_s_next, P_s_next)
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)

        # Combine results: Smoothed values are reversed, append the last filtered value
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt[-1][None, :]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt[-1][None, :, :]], axis=0)

        return x_smooth, P_smooth

    # --- Simulation Smoother ---
    def _simulation_smoother_single_draw(self, ys: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        """ Internal: Performs one draw of Durbin-Koopman simulation smoother."""
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)

        n_s = self.n_state
        n_eps = self.n_shocks # Number of state shocks epsilon

        # --- Step 1: Standard RTS smoother on original data y ---
        x_smooth_rts, _ = self.smooth(ys)

        key_init, key_eps, key_eta = random.split(key, 3)

        # --- Step 2: Simulate initial state x0* from p(x0) ---
        try:
            init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(n_s, dtype=self.init_P.dtype)
            L0 = jnp.linalg.cholesky(init_P_reg)
            z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
            x0_star = self.init_x + L0 @ z0
        except Exception:
            print("Warning: Cholesky failed for init_P in simulation smoother. Using init_x mean.")
            x0_star = self.init_x

        # --- Step 3: Simulate state trajectory x* using model ---
        eps_star = random.normal(key_eps, (Tsteps, n_eps), dtype=self.R.dtype) if n_eps > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)

        def state_sim_step(x_prev_star, eps_t_star):
            # Ensure R @ eps_t_star works even if eps_t_star is empty
            shock_term = self.R @ eps_t_star if n_eps > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
            x_curr_star = self.T @ x_prev_star + shock_term
            return x_curr_star, x_curr_star

        _, x_star = lax.scan(state_sim_step, x0_star, eps_star)

        # --- Step 4: Simulate observation trajectory y* using model ---
        eta_star = self.simulate_obs_noise(key_eta, (Tsteps,)) # Shape [T, n_obs]
        y_star = (x_star @ self.C.T) + eta_star # Shape [T, n_obs]

        # --- Step 5: Standard RTS smoother on simulated data y* ---
        x_smooth_star, _ = self.smooth(y_star)

        # --- Step 6: Combine results ---
        x_draw = x_star + (x_smooth_rts - x_smooth_star)
        return x_draw # Shape [T, n_state]


    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1
                            ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """ Runs the Durbin-Koopman simulation smoother. """
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")

        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        Tsteps = ys_arr.shape[0]

        empty_state = jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        if Tsteps == 0:
            if num_draws == 1: return empty_state
            else: return empty_state, empty_state, jnp.empty((num_draws, 0, self.n_state), dtype=self.init_x.dtype)

        if num_draws == 1:
            # --- Single Draw Case ---
            print("Running Simulation Smoother (1 draw)...")
            single_draw = self._simulation_smoother_single_draw(ys_arr, key)
            print("Finished smoothing.")
            return single_draw # Shape [T, n_state]
        else:
            # --- Monte Carlo Case (Multiple Draws) ---
            print(f"Running Simulation Smoother ({num_draws} draws)...")

            # Step 1 (Common): RTS on original data y
            print(" Calculating E[x|y] (RTS on original data)...")
            x_smooth_rts_common, _ = self.smooth(ys_arr)

            # Vmapped function for steps 2-6
            def perform_single_dk_draw(key_single_draw):
                key_init, key_eps, key_eta = random.split(key_single_draw, 3)
                # Step 2: Simulate x0*
                try:
                    init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(self.n_state, dtype=self.init_P.dtype)
                    L0 = jnp.linalg.cholesky(init_P_reg)
                    z0 = random.normal(key_init, (self.n_state,), dtype=self.init_x.dtype)
                    x0_star = self.init_x + L0 @ z0
                except Exception: x0_star = self.init_x # Use mean if fails
                # Step 3: Simulate x*
                eps_star = random.normal(key_eps, (Tsteps, self.n_shocks), dtype=self.R.dtype) if self.n_shocks > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
                def state_sim_step(x_prev_star, eps_t_star):
                     shock_term = self.R @ eps_t_star if self.n_shocks > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
                     x_curr_star = self.T @ x_prev_star + shock_term
                     return x_curr_star, x_curr_star
                _, x_star = lax.scan(state_sim_step, x0_star, eps_star)
                # Step 4: Simulate y*
                eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
                y_star = (x_star @ self.C.T) + eta_star
                # Step 5: RTS on y*
                x_smooth_star, _ = self.smooth(y_star)
                # Step 6: Combine
                x_draw = x_star + (x_smooth_rts_common - x_smooth_star)
                return x_draw

            keys = random.split(key, num_draws)
            print(f" Performing {num_draws} simulation smoother draws (vectorized)...")
            vmapped_smoother = vmap(perform_single_dk_draw, in_axes=(0,))
            all_draws_jax = vmapped_smoother(keys)
            print(" Finished smoothing calculations.")

            print(" Calculating statistics (mean, median)...")
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            print(" Finished.")
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- Simulation Function (ensure dtype consistency) ---
def _simulate_state_space_impl( # Use an internal name
    P_aug: ArrayLike,
    R_aug: ArrayLike,
    Omega: ArrayLike,
    H_obs: ArrayLike,
    init_x: ArrayLike,
    init_P: ArrayLike,
    key: jax.random.PRNGKey,
    num_steps: int
) -> Tuple[jax.Array, jax.Array]:
    """ Simulates data from a linear Gaussian state-space model. (Internal Impl) """
    desired_dtype = P_aug.dtype # Infer dtype from inputs
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

    # --- Initial State Simulation ---
    try:
        init_P_reg = init_P_jax + _MACHINE_EPSILON * jnp.eye(n_aug, dtype=desired_dtype)
        L0 = jnp.linalg.cholesky(init_P_reg)
        z0 = random.normal(key_init, (n_aug,), dtype=desired_dtype)
        x0 = init_x_jax + L0 @ z0
    except Exception:
        print("Warning: Cholesky failed for init_P in simulation. Using init_x mean.")
        x0 = init_x_jax

    # --- Generate Noise ---
    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_aug_shocks), dtype=desired_dtype) if n_aug_shocks > 0 else jnp.zeros((num_steps, 0), dtype=desired_dtype)
    try:
        H_obs_reg = H_obs_jax + _MACHINE_EPSILON * jnp.eye(n_obs, dtype=desired_dtype)
        obs_noise = random.multivariate_normal(
            key_obs_noise, jnp.zeros(n_obs, dtype=desired_dtype), H_obs_reg, shape=(num_steps,), dtype=desired_dtype
        )
    except Exception as e:
        print(f"Warning: Error simulating obs noise with mvn in simulate_state_space: {e}. Using zeros.")
        obs_noise = jnp.zeros((num_steps, n_obs), dtype=desired_dtype)

    # --- Simulation Loop (using lax.scan) ---
    def simulation_step(x_prev, noise_t):
        eps_t, eta_t = noise_t
        shock_term = R_aug_jax @ eps_t if n_aug_shocks > 0 else jnp.zeros(n_aug, dtype=x_prev.dtype)
        x_curr = P_aug_jax @ x_prev + shock_term
        y_curr = Omega_jax @ x_curr + eta_t
        return x_curr, (x_curr, y_curr)

    combined_noise = (state_shocks_std_normal, obs_noise)
    final_state, (states, observations) = lax.scan(simulation_step, x0, combined_noise)

    return states, observations

# Apply JIT *after* definition
simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))


# --- END OF MODIFIED FILE Kalman_filter_jax.py ---