# --- START OF FILE Kalman_filter_jax.py ---

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp # Use numpy for potential warnings, keep core logic in JAX
from typing import Tuple, Optional, Union, Sequence

# Small constant for numerical stability (jitter)
_MACHINE_EPSILON = jnp.finfo(jnp.float32).eps

class KalmanFilter:
    """
    Implements the standard Kalman Filter, RTS Smoother, and
    Durbin-Koopman Simulation Smoother using JAX.

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
        # Ensure inputs are JAX arrays
        self.T = jnp.asarray(T)
        self.R = jnp.asarray(R)
        self.C = jnp.asarray(C)
        self.H = jnp.asarray(H)
        self.init_x = jnp.asarray(init_x)
        self.init_P = jnp.asarray(init_P)

        # --- Input Validation (Basic) ---
        n_state = self.T.shape[0]
        n_obs = self.C.shape[0]
        n_shocks = self.R.shape[1]

        if self.T.shape != (n_state, n_state):
            raise ValueError(f"T shape mismatch: expected ({n_state},{n_state}), got {self.T.shape}")
        if self.R.shape[0] != n_state:
            raise ValueError(f"R shape mismatch: expected ({n_state},?), got {self.R.shape}")
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
        self.I_s = jnp.eye(self.n_state)
        self.state_cov = self.R @ self.R.T # Precompute state noise covariance Q = R @ R.T

        # Precompute Cholesky of H if possible for observation noise simulation & stability
        self.H_stable = self.H # Default to original H
        try:
            # Add slight jitter for robustness before Cholesky
            H_reg = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs)
            self.L_H = jnp.linalg.cholesky(H_reg)
            self.simulate_obs_noise = self._simulate_obs_noise_chol
        except ValueError: # Use ValueError for JAX linalg errors (e.g., not PSD)
            print("Warning: Cholesky decomposition failed for H. "
                  "Using multivariate_normal for observation noise simulation (may be slower). "
                  "Adding jitter to H for mvn stability.")
            # Ensure H is at least PSD for mvn by adding jitter
            min_eig = jnp.min(jnp.linalg.eigvalsh(self.H))
            if min_eig < -1e-9: # Check for significantly negative eigenvalues
                 print(f"Warning: H has significant negative eigenvalues ({min_eig}). Simulation might fail.")
            # Add jitter for numerical stability in mvn if Cholesky failed
            self.H_stable = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs)
            self.simulate_obs_noise = self._simulate_obs_noise_mvn

    def _simulate_obs_noise_chol(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        """Simulate observation noise using precomputed Cholesky factor."""
        # Shape needs to be a tuple for random.normal
        z_eta = random.normal(key, tuple(shape) + (self.n_obs,))
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
                jnp.zeros((self.n_obs,)),
                self.H_stable, # Use H_stable which might have jitter
                shape=mvn_shape
            )
            # No need to squeeze if shape=() was passed correctly
            return eta
        except Exception as e:
            # Catch potential errors from mvn (e.g., if H_stable still not PSD enough)
            print(f"Error during multivariate_normal simulation: {e}")
            print("Returning zeros for observation noise.")
            return jnp.zeros(tuple(shape) + (self.n_obs,))


    def filter(self, ys: ArrayLike) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Applies the Kalman filter.

        Handles missing observations (NaNs):
        - If an entire observation vector `y_t` is NaN, the update step is skipped
          (filtered state = predicted state for that step).
        - **Limitation:** If only *some* elements of `y_t` are NaN, this implementation
          replaces those NaNs with 0.0 for the update calculation. This is an
          approximation and not the theoretically correct way to handle partially
          missing data (which would involve modifying C and H).

        Args:
            ys: Observations array, shape `[T, n_obs]`. NaN indicates missing.

        Returns:
            A tuple containing:
                - x_pred: Predicted state means `E[x_t | y_{1:t-1}]` [T, n_state]
                - P_pred: Predicted state covariances `Cov(x_t | y_{1:t-1})` [T, n_state, n_state]
                - x_filt: Filtered state means `E[x_t | y_{1:t}]` [T, n_state]
                - P_filt: Filtered state covariances `Cov(x_t | y_{1:t})` [T, n_state, n_state]
        """
        ys_arr = jnp.asarray(ys)
        T, C, H, I_s = self.T, self.C, self.H, self.I_s
        state_cov = self.state_cov # Use precomputed R @ R.T

        def step(carry, y_t):
            x_prev_filt, P_prev_filt = carry # Filtered state from t-1

            # --- Prediction Step ---
            x_pred_t = T @ x_prev_filt
            P_pred_t = T @ P_prev_filt @ T.T + state_cov

            # --- Update Step ---
            # Check if ALL observations at time t are missing
            is_missing = jnp.all(jnp.isnan(y_t))

            def perform_update(_):
                # Replace NaNs with 0.0 for computation if *partially* missing.
                # If fully missing, this branch isn't taken anyway.
                y_obs = jnp.nan_to_num(y_t, nan=0.0)
                y_pred = C @ x_pred_t
                v = y_obs - y_pred              # Prediction error (innovation)

                # --- Kalman Gain Calculation (Numerically Stable) ---
                PCt = P_pred_t @ C.T
                S = C @ PCt + H                 # Innovation covariance S = C P_pred C' + H

                try:
                    # Add small diagonal jitter for Cholesky robustness
                    S_reg = S + _MACHINE_EPSILON * jnp.eye(self.n_obs)
                    L_S = jnp.linalg.cholesky(S_reg)
                    # Solve K = P C.T S^{-1} = P C.T (L L.T)^{-1}
                    # Step 1: Solve L Y = P C.T for Y
                    Y = jax.scipy.linalg.solve_triangular(L_S, PCt, lower=True)
                    # Step 2: Solve L.T K = Y for K
                    K = jax.scipy.linalg.solve_triangular(L_S.T, Y, lower=False)
                except ValueError: # Catch JAX linalg errors (e.g., not PSD)
                    # Fallback 1: Standard solve S K = P C.T
                    # print("Warning: Cholesky failed in filter update. Using standard solve.")
                    try:
                        # Add jitter here too for solve robustness
                        S_reg = S + _MACHINE_EPSILON * jnp.eye(self.n_obs)
                        K = jnp.linalg.solve(S_reg, PCt)
                    except ValueError: # Catch potential LinAlgError from solve
                        # Fallback 2: Pseudo-inverse (least robust)
                        # print("Warning: Standard solve failed in filter update. Using pinv.")
                        S_pinv = jnp.linalg.pinv(S) # Use original S for pinv
                        K = PCt @ S_pinv

                # --- State and Covariance Update ---
                x_filt_t = x_pred_t + K @ v
                # Joseph form for covariance update (more stable)
                # P_filt = (I - K C) P_pred (I - K C)' + K H K'
                IKC = I_s - K @ C
                # Use H directly here, not H_stable (H is the model parameter)
                P_filt_t = IKC @ P_pred_t @ IKC.T + K @ self.H @ K.T
                # Symmetrize P_filt to avoid numerical drift
                P_filt_t = (P_filt_t + P_filt_t.T) / 2.0

                return x_filt_t, P_filt_t

            # If observation is missing, skip update: filtered = predicted
            x_filt_t, P_filt_t = lax.cond(is_missing,
                                          lambda _: (x_pred_t, P_pred_t), # If missing
                                          perform_update,                 # If not missing
                                          operand=None)

            return (x_filt_t, P_filt_t), (x_pred_t, P_pred_t, x_filt_t, P_filt_t)

        # Run the scan loop
        init_carry = (self.init_x, self.init_P)
        # Ensure ys has shape [T, n_obs] even if T=1
        ys_reshaped = jnp.reshape(ys_arr, (-1, self.n_obs))
        (_, _), outs = lax.scan(step, init_carry, ys_reshaped)
        # outs = (x_pred, P_pred, x_filt, P_filt)
        return outs

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
        ys_arr = jnp.asarray(ys)
        # Run the filter first
        filter_outs = self.filter(ys_arr)
        if not isinstance(filter_outs, tuple) or len(filter_outs) != 4:
             raise TypeError("Internal Error: Filter did not return expected tuple.") # Changed error type
        x_pred, P_pred, x_filt, P_filt = filter_outs
        T_mat = self.T

        N = x_filt.shape[0]
        if N == 0:
            return jnp.empty((0, self.n_state)), jnp.empty((0, self.n_state, self.n_state))

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
            TPf = T_mat @ Pf_t # T @ P_filt_t
            try:
                # More stable: Solve P_pred_{t+1} J_t' = T @ P_filt_t
                # Add jitter for robustness before solve
                Pp_next_reg = Pp_next_t + _MACHINE_EPSILON * jnp.eye(self.n_state)
                # Solves Pp_next_reg @ X = TPf.T for X = Jt'
                Jt_transpose = jnp.linalg.solve(Pp_next_reg, TPf.T)
                Jt = Jt_transpose.T
            except ValueError: # Changed Exception type to ValueError
                # Fallback: Pseudo-inverse
                # print(f"Warning: Solve failed in smoother gain @ t={N-2-len(x_s_rev)}. Using pinv.")
                Pp_next_pinv = jnp.linalg.pinv(Pp_next_t) # Use original for pinv
                Jt = Pf_t @ T_mat.T @ Pp_next_pinv

            # --- Smoothed State Mean ---
            # x_s_t = x_filt_t + J_t @ (x_s_{t+1} - x_pred_{t+1})
            x_s_t = xf_t + Jt @ (x_s_next_t - xp_next_t)

            # --- Smoothed State Covariance ---
            # P_s_t = P_filt_t + J_t @ (P_s_{t+1} - P_pred_{t+1}) @ J_t'
            P_s_t = Pf_t + Jt @ (P_s_next_t - Pp_next_t) @ Jt.T

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
        """
        Internal method: Performs one draw of the Durbin-Koopman simulation smoother.

        Args:
            ys: Observations array [T, n_obs].
            key: JAX random key for this specific draw.

        Returns:
            A single smoothed state draw `x_draw ~ p(x | y)` [T, n_state].
        """
        Tsteps = ys.shape[0]
        if Tsteps == 0:
            return jnp.empty((0, self.n_state))

        n_s = self.n_state
        n_eps = self.n_shocks # Number of state shocks epsilon

        # --- Step 1: Standard RTS smoother on original data y ---
        # Calculate E[x | y] = x_smooth_rts
        x_smooth_rts, _ = self.smooth(ys)

        # Split key for different simulation components
        key_init, key_eps, key_eta = random.split(key, 3)

        # --- Step 2: Simulate initial state x0* from p(x0) ---
        try:
            # Draw from N(init_x, init_P) using Cholesky
            init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(n_s) # Add jitter
            L0 = jnp.linalg.cholesky(init_P_reg)
            z0 = random.normal(key_init, (n_s,))
            x0_star = self.init_x + L0 @ z0
        except ValueError:
            print("Warning: Cholesky failed for init_P in simulation smoother. Using init_x mean.")
            x0_star = self.init_x

        # --- Step 3: Simulate state trajectory x* using model ---
        # Generate state shocks epsilon* ~ N(0, I)
        # Shape: (Tsteps, n_eps) for states x*_1 to x*_T
        eps_star = random.normal(key_eps, (Tsteps, n_eps))

        # Simulate forward using lax.scan: x*_t = T @ x*_{t-1} + R @ epsilon*_t
        def state_sim_step(x_prev_star, eps_t_star):
            x_curr_star = self.T @ x_prev_star + self.R @ eps_t_star
            return x_curr_star, x_curr_star

        _, x_star = lax.scan(state_sim_step, x0_star, eps_star)
        # x_star result has shape [T, n_state] (x*_1 to x*_T)

        # --- Step 4: Simulate observation trajectory y* using model ---
        # Generate observation noise eta* ~ N(0, H) using the appropriate method
        eta_star = self.simulate_obs_noise(key_eta, (Tsteps,)) # Shape [T, n_obs]

        # Calculate y*_t = C @ x*_t + eta*_t
        # Efficient calculation: (C @ x*_t')' = x_t @ C'
        y_star = (x_star @ self.C.T) + eta_star # Shape [T, n_obs]

        # --- Step 5: Standard RTS smoother on simulated data y* ---
        # Calculate E[x | y*] = x_smooth_star
        x_smooth_star, _ = self.smooth(y_star)

        # --- Step 6: Combine results to get a draw from p(x | y) ---
        # x_draw = x* + E[x | y] - E[x | y*]
        x_draw = x_star + (x_smooth_rts - x_smooth_star)

        return x_draw # Shape [T, n_state]

    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1
                            ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        """
        Runs the Durbin-Koopman simulation smoother.

        Generates draws from the posterior distribution `p(x | y)`.

        Args:
            ys: Observations array [T, n_obs]. NaN indicates missing.
            key: Base JAX random key.
            num_draws: Number of draws to generate. Defaults to 1.

        Returns:
            - If num_draws=1: A single draw `x_draw` [T, n_state].
            - If num_draws>1: A tuple (mean, median, all_draws):
                - mean: Mean across draws [T, n_state].
                - median: Median across draws [T, n_state].
                - all_draws: All draws [num_draws, T, n_state].
            - Raises ValueError if num_draws <= 0.
        """
        if num_draws <= 0:
            raise ValueError("num_draws must be >= 1.")

        ys_arr = jnp.asarray(ys)
        Tsteps = ys_arr.shape[0]

        # Handle empty observation sequence
        if Tsteps == 0:
            empty_state = jnp.empty((0, self.n_state))
            if num_draws == 1: return empty_state
            else: return empty_state, empty_state, jnp.empty((num_draws, 0, self.n_state))

        if num_draws == 1:
            # --- Single Draw Case ---
            print("Running Simulation Smoother (1 draw)...")
            single_draw = self._simulation_smoother_single_draw(ys_arr, key)
            print("Finished smoothing.")
            return single_draw # Shape [T, n_state]

        else:
            # --- Monte Carlo Case (Multiple Draws) ---
            print(f"Running Simulation Smoother ({num_draws} draws)...")

            # Step 1 (Common for all draws): Standard RTS on original data y
            # Calculate E[x | y] = x_smooth_rts_common
            print(" Calculating E[x|y] (RTS on original data)...")
            x_smooth_rts_common, _ = self.smooth(ys_arr)

            # Define the function to be vmapped (takes only key, uses common E[x|y])
            # This function performs steps 2-6 of the Durbin-Koopman algorithm per draw.
            def perform_single_dk_draw(key_single_draw):
                # Split key for this draw's simulation components
                key_init, key_eps, key_eta = random.split(key_single_draw, 3)

                # Step 2: Simulate initial state x0*
                try:
                    init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(self.n_state)
                    L0 = jnp.linalg.cholesky(init_P_reg)
                    z0 = random.normal(key_init, (self.n_state,))
                    x0_star = self.init_x + L0 @ z0
                except ValueError:
                    # Warning printed only once if needed, use mean
                    x0_star = self.init_x

                # Step 3: Simulate state trajectory x*
                eps_star = random.normal(key_eps, (Tsteps, self.n_shocks))
                def state_sim_step(x_prev_star, eps_t_star):
                    x_curr_star = self.T @ x_prev_star + self.R @ eps_t_star
                    return x_curr_star, x_curr_star
                _, x_star = lax.scan(state_sim_step, x0_star, eps_star)

                # Step 4: Simulate observation trajectory y*
                eta_star = self.simulate_obs_noise(key_eta, (Tsteps,))
                y_star = (x_star @ self.C.T) + eta_star

                # Step 5: Standard RTS smoother on simulated data y*
                # Calculate E[x | y*] = x_smooth_star
                x_smooth_star, _ = self.smooth(y_star)

                # Step 6: Combine results using the *common* E[x|y]
                # x_draw = x* + E[x | y] - E[x | y*]
                x_draw = x_star + (x_smooth_rts_common - x_smooth_star)
                return x_draw

            # Generate unique keys for each draw
            keys = random.split(key, num_draws)

            # Run vmap over the keys
            print(f" Performing {num_draws} simulation smoother draws (vectorized)...")
            # Vmap the function that performs steps 2-6
            vmapped_smoother = vmap(perform_single_dk_draw, in_axes=(0,))
            all_draws_jax = vmapped_smoother(keys)
            # Result shape: (num_draws, T, n_state)
            print(" Finished smoothing calculations.")

            # Calculate summary statistics
            print(" Calculating statistics (mean, median)...")
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            # Use percentile for median (numerically stable)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear') # Use 'linear' for jax>0.4.14

            print(" Finished.")
            # Return tuple: (mean, median, all_draws)
            return mean_smooth_sim, median_smooth_sim, all_draws_jax


# --- Simulation Function ---

# Define the function WITHOUT the decorator
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
    """
    Simulates data from a linear Gaussian state-space model. (Internal Implementation)
    ... (rest of docstring) ...
    """
    # --- Function body remains exactly the same ---
    P_aug_jax = jnp.asarray(P_aug)
    R_aug_jax = jnp.asarray(R_aug)
    Omega_jax = jnp.asarray(Omega)
    H_obs_jax = jnp.asarray(H_obs)
    init_x_jax = jnp.asarray(init_x)
    init_P_jax = jnp.asarray(init_P)

    n_aug = P_aug_jax.shape[0]
    n_aug_shocks = R_aug_jax.shape[1]
    n_obs = Omega_jax.shape[0]

    key_init, key_state_noise, key_obs_noise = random.split(key, 3)

    # --- Initial State Simulation ---
    try:
        init_P_reg = init_P_jax + _MACHINE_EPSILON * jnp.eye(n_aug)
        L0 = jnp.linalg.cholesky(init_P_reg)
        z0 = random.normal(key_init, (n_aug,))
        x0 = init_x_jax + L0 @ z0
    except ValueError: # Use appropriate exception for old JAX if needed
        print("Warning: Cholesky failed for init_P in simulation. Using init_x mean.")
        x0 = init_x_jax

    # --- Generate Noise ---
    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_aug_shocks))
    try:
        H_obs_reg = H_obs_jax + _MACHINE_EPSILON * jnp.eye(n_obs)
        obs_noise = random.multivariate_normal(
            key_obs_noise, jnp.zeros(n_obs), H_obs_reg, shape=(num_steps,)
        )
    except Exception as e: # Use appropriate exception for old JAX if needed
        print(f"Error simulating obs noise with mvn in simulate_state_space: {e}. Using zeros.")
        obs_noise = jnp.zeros((num_steps, n_obs))

    # --- Simulation Loop (using lax.scan) ---
    def simulation_step(x_prev, noise_t):
        eps_t, eta_t = noise_t
        x_curr = P_aug_jax @ x_prev + R_aug_jax @ eps_t
        y_curr = Omega_jax @ x_curr + eta_t
        return x_curr, (x_curr, y_curr)

    combined_noise = (state_shocks_std_normal, obs_noise)
    final_state, (states, observations) = lax.scan(simulation_step, x0, combined_noise)

    return states, observations

# --- Apply JIT *after* definition, exporting the desired name ---
simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))

