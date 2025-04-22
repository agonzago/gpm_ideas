import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import numpy as onp

class KalmanFilter:
    # ... (__init__, filter, smooth methods remain the same) ...
    def __init__(self, T, R, C, H, init_x, init_P):
        self.T, self.R, self.C, self.H = T, R, C, H
        self.init_x, self.init_P = init_x, init_P
        self.n_state = T.shape[0]
        self.n_obs   = C.shape[0]
        self.I_s = jnp.eye(self.n_state)

    def filter(self, ys):
        """ Existing filter method """
        # ... (implementation as before) ...
        T, R, C, H = self.T, self.R, self.C, self.H
        I_s = self.I_s
        def step(carry, y):
            x_prev, P_prev = carry
            x_pred = T @ x_prev
            P_pred = T @ P_prev @ T.T + R @ R.T
            miss = jnp.all(jnp.isnan(y))
            def do_update(_):
                y0 = jnp.nan_to_num(y, nan=0.0)
                v = y0 - C @ x_pred
                S = C @ P_pred @ C.T + H
                try:
                    L = jnp.linalg.cholesky(S)
                    K_t = jnp.linalg.solve(L, P_pred @ C.T)
                    K = jnp.linalg.solve(L.T, K_t).T
                except jnp.linalg.LinAlgError:
                     K = jnp.linalg.solve(S.T, (P_pred @ C.T).T).T
                x_f = x_pred + K @ v
                P_f = (I_s - K @ C) @ P_pred
                return x_f, P_f
            x_filt, P_filt = lax.cond(miss, lambda _: (x_pred, P_pred), do_update, operand=None)
            return (x_filt, P_filt), (x_pred, P_pred, x_filt, P_filt)
        init_carry = (self.init_x, self.init_P)
        ys = jnp.reshape(ys, (-1, self.n_obs))
        (_, _), outs = lax.scan(step, init_carry, ys)
        return outs # x_pred, P_pred, x_filt, P_filt

    def smooth(self, ys):
        """ Existing smooth method """
        # ... (implementation as before) ...
        filter_outs = self.filter(ys)
        # Check if filter_outs has the expected structure before unpacking
        if not isinstance(filter_outs, tuple) or len(filter_outs) != 4:
             raise ValueError("Filter did not return expected tuple (x_pred, P_pred, x_filt, P_filt)")
        x_pred, P_pred, x_filt, P_filt = filter_outs
        T_mat, I_s = self.T, self.I_s
        N = x_filt.shape[0]
        if N == 0: return jnp.empty((0, self.n_state)), jnp.empty((0, self.n_state, self.n_state))
        if N == 1: return x_filt, P_filt
        x_pred_rev = x_pred[:-1][::-1]; P_pred_rev = P_pred[:-1][::-1]
        x_filt_rev = x_filt[:-1][::-1]; P_filt_rev = P_filt[:-1][::-1]
        def backward(carry, seq):
            x_next_s, P_next_s = carry
            Pp, Pf, xp, xf = seq
            try: J = Pf @ T_mat.T @ jnp.linalg.inv(Pp)
            except jnp.linalg.LinAlgError: J = Pf @ T_mat.T @ jnp.linalg.pinv(Pp)
            x_s = xf + J @ (x_next_s - xp); P_s = Pf + J @ (P_next_s - Pp) @ J.T
            return (x_s, P_s), (x_s, P_s)
        init = (x_filt[-1], P_filt[-1]); seq = (P_pred_rev, P_filt_rev, x_pred_rev, x_filt_rev)
        ( _, _), (x_s_rev, P_s_rev) = lax.scan(backward, init, seq)
        x_smooth = jnp.vstack([x_filt[-1][None, :], x_s_rev])[::-1]
        P_smooth = jnp.concatenate([P_filt[-1][None, :, :], P_s_rev], axis=0)[::-1]
        return x_smooth, P_smooth

    # --- Internal helper for a single draw ---
    def _simulation_smoother_single_draw(self, ys, key):
        """ Internal method: Returns ONE draw """
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state))
        n_s = self.n_state; n_eps = self.R.shape[1]; n_y = self.n_obs
        x_smooth_real, _ = self.smooth(ys)
        key0, key_eps, key_eta = random.split(key, 3)
        try:
            L0 = jnp.linalg.cholesky(self.init_P); z0 = random.normal(key0, (n_s,)); x0_star = self.init_x + L0 @ z0
        except jnp.linalg.LinAlgError: x0_star = self.init_x
        if Tsteps > 1:
             eps = random.normal(key_eps, (Tsteps - 1, n_eps))
             def sim_step(x_prev, e): x_curr = self.T @ x_prev + self.R @ e; return x_curr, x_curr
             _, xs_tail = lax.scan(sim_step, x0_star, eps); x_star = jnp.vstack([x0_star[None, :], xs_tail])
        else: x_star = x0_star[None, :]
        try: eta = random.multivariate_normal(key_eta, jnp.zeros((n_y,)), self.H, shape=(Tsteps,))
        except ValueError:
             try: L_H = jnp.linalg.cholesky(self.H); z_eta = random.normal(key_eta, (Tsteps, n_y)); eta = z_eta @ L_H.T
             except jnp.linalg.LinAlgError: eta = jnp.zeros((Tsteps, n_y))
        y_star = (x_star @ self.C.T) + eta
        x_s_star, _ = self.smooth(y_star)
        return x_smooth_real + (x_star - x_s_star)

    # --- NEW UNIFIED METHOD ---
    def simulation_smoother(self, ys, key, num_draws=1):
        """
        Runs the Durbin-Koopman simulation smoother.

        If num_draws = 1 (default), returns a single draw.
        If num_draws > 1, runs Monte Carlo draws using vmap and returns
        statistics (mean, median) and all draws.

        Args:
            ys (DeviceArray): Observations array [T, n_obs].
            key (PRNGKey): Base JAX random key.
            num_draws (int): Number of draws to generate. Defaults to 1.

        Returns:
            DeviceArray: If num_draws=1, returns single draw [T, n_state].
            tuple: If num_draws>1, returns (mean, median, all_draws):
                - mean: Mean across draws [T, n_state].
                - median: Median across draws [T, n_state].
                - all_draws: All draws [num_draws, T, n_state].
            None: If num_draws <= 0.
        """
        if num_draws == 1:
            # --- Single Draw Case ---
            print("Running Simulation Smoother (1 draw)...")
            single_draw = self._simulation_smoother_single_draw(ys, key)
            print("Finished smoothing.")
            return single_draw # Shape [T, n_state]

        elif num_draws > 1:
            # --- Monte Carlo Case ---
            print(f"Running Simulation Smoother ({num_draws} draws)...")
            keys = random.split(key, num_draws)

            # vmap the internal single-draw function
            vmapped_smoother = vmap(self._simulation_smoother_single_draw, in_axes=(None, 0))
            all_draws_jax = vmapped_smoother(ys, keys)
            # Result shape: (num_draws, T, n_state)
            print("Finished smoothing calculations.")

            # Calculate statistics
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.median(all_draws_jax, axis=0)

            # Return tuple: (mean, median, all_draws)
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

        else:
            print("Warning: num_draws must be >= 1.")
            return None # Or raise an error

    # Remove the old monte_carlo_simulation_smoother method if it exists


# ―――――――――――――――――――――――――――
# Test on a local‐level model
# ―――――――――――――――――――――――――――
if __name__ == "__main__":
    # Imports needed for execution block
    import numpy as onp
    import matplotlib.pyplot as plt
    from jax import random

    # --- Simulation Setup ---
    onp.random.seed(0)
    T = 1000
    true_state_noise_std = 0.1
    measurement_noise_std = 0.5
    sm_draws = 200 # Number of draws

    # --- Simulate True Data ---
    true_x = onp.zeros(T+1)
    for t in range(T): true_x[t+1] = true_x[t] + true_state_noise_std * onp.random.randn()
    y = true_x[:T] + measurement_noise_std * onp.random.randn(T)
    y[int(T*0.1):int(T*0.15)] = onp.nan; y[int(T*0.5)] = onp.nan
    ys_jax = jnp.array(y).reshape((-1,1))

    # --- Model Definition ---
    T_mat = jnp.array([[1.0]]); R_mat = jnp.array([[true_state_noise_std]])
    C_mat = jnp.array([[1.0]]); H_mat = jnp.array([[measurement_noise_std**2]])
    init_x = jnp.array([0.0]); init_P = jnp.array([[1.0]])

    # --- Instantiate Kalman Filter ---
    kf = KalmanFilter(T_mat, R_mat, C_mat, H_mat, init_x, init_P)

    # --- 1. Standard Kalman Filter & RTS Smoother ---
    print("Running Filter...")
    filter_outs = kf.filter(ys_jax) # Get the tuple
    print("Running RTS Smoother...")
    x_smooth_rts, p_smooth_rts = kf.smooth(ys_jax) # Assuming smooth calls filter internally correctly

    # --- 2. Simulation Smoothing (using the new unified method) ---
    main_key = random.PRNGKey(42)

    # Call the unified simulation smoother
    sim_smoother_result = kf.simulation_smoother(ys_jax, main_key, num_draws=sm_draws)

    # --- Handle the results based on num_draws ---
    if sm_draws == 1:
        # If only one draw was requested, result is the draw itself
        mean_smooth_sim = sim_smoother_result # Use the draw as the 'mean' for plotting maybe?
        median_smooth_sim = sim_smoother_result # And as the 'median'
        all_draws_jax = sim_smoother_result[None, :, :] # Add batch dim for consistency if needed later
        print("Single draw obtained.")
    elif sm_draws > 1:
        # If multiple draws, unpack the tuple
        mean_smooth_sim, median_smooth_sim, all_draws_jax = sim_smoother_result
        print("Mean, Median, and all draws obtained.")
    else:
        print("Simulation smoother not run or returned None.")
        # Handle error or exit if needed
        exit()

    # --- Convert JAX arrays to NumPy for plotting ---
    # Unpack filter results safely
    if isinstance(filter_outs, tuple) and len(filter_outs) == 4:
        _, _, x_filt, _ = filter_outs
        x_filt_np = onp.array(x_filt).flatten()
    else:
        print("Error: Filter output not as expected.")
        x_filt_np = onp.zeros(T) # Placeholder

    x_smooth_rts_np = onp.array(x_smooth_rts).flatten()
    mean_smooth_sim_np = onp.array(mean_smooth_sim).flatten()
    median_smooth_sim_np = onp.array(median_smooth_sim).flatten()
    ys_np = onp.array(ys_jax).flatten()

    # --- Plotting ---
    time_axis = onp.arange(T)

    # Plot 1: Filter vs True
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, ys_np, "kx", markersize=3, alpha=0.6, label="Observations (y)")
    plt.plot(time_axis, x_filt_np, "b-", linewidth=1.5, label="Filtered State (mean)")
    plt.plot(time_axis, true_x[:T], "g-", linewidth=1.5, label="True State (x)")
    plt.legend()
    plt.title("Kalman Filter vs. True State")
    plt.xlabel("Time step (t)")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot 2: RTS Smoother vs Simulation Smoother Mean & Median
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, true_x[:T], "g-", linewidth=1.5, label="True State (x)")
    plt.plot(time_axis, x_smooth_rts_np, "r-", linewidth=2, label="RTS Smoothed State (mean)")
    plt.plot(time_axis, mean_smooth_sim_np, "m--", linewidth=1.5, label="Simulation Smoother (Mean)")
    # Only plot median if we actually calculated it (sm_draws > 1)
    if sm_draws > 1:
        plt.plot(time_axis, median_smooth_sim_np, "c:", linewidth=1.5, label="Simulation Smoother (Median)")
    plt.legend()
    plt.title(f"RTS Smoother vs. Simulation Smoother ({sm_draws} Draw(s))")
    plt.xlabel("Time step (t)")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Optional: Plot a few individual draws (only makes sense if sm_draws > 1)
    if sm_draws > 1:
        plt.figure(figsize=(12, 6))
        n_draws_to_plot = 10
        n_draws_to_plot = min(n_draws_to_plot, sm_draws)
        # Ensure all_draws_jax has the expected shape before plotting
        if all_draws_jax is not None and all_draws_jax.ndim == 3:
             draws_to_plot_np = onp.array(all_draws_jax[:n_draws_to_plot, :, 0])
             plt.plot(time_axis, draws_to_plot_np.T, "m-", alpha=0.3, label=f"Sim. Draws (first {n_draws_to_plot})")
        else:
             # If sm_draws was 1, we might only have the single draw in mean_smooth_sim_np
              plt.plot(time_axis, mean_smooth_sim_np, "m-", alpha=0.8, label="Sim. Draw (num_draws=1)")

        plt.plot(time_axis, true_x[:T], "g-", linewidth=2, label="True State (x)")
        plt.plot(time_axis, mean_smooth_sim_np, "k--", linewidth=2, label="Simulation Smoother (mean)")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(f"A Few Simulation Smoother Draws ({sm_draws} total)")
        plt.xlabel("Time step (t)")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()