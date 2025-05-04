# --- START OF main_script.py ---
import os
print("Attempting to force JAX to use CPU...") # You might remove this forcing later
os.environ['JAX_PLATFORMS'] = 'gpu'
print(f"JAX_PLATFORMS set to: {os.environ.get('JAX_PLATFORMS')}")

# --- Add this line ---
import jax
jax.config.update("jax_enable_x64", True)
# --------------------

print(f"JAX version: {jax.__version__}") # Good to keep checking version
# Verify x64 is enabled (Optional check)
print(f"JAX float64 enabled: {jax.config.jax_enable_x64}")

import numpy as onp # Use onp for plotting and standard numpy operations
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jax import random
from jax import config

# --- Import your custom modules ---
# Assume the parser/solver is in dynare_parser_spd7.py
import Dynare_parser_sda_solver as dp
# Assume the Kalman filter is in Kalman_filter_jax.py
from parser7.Kalman_filter_jax_old import KalmanFilter, simulate_state_space

# --- Helper Function for HDR ---
def calculate_hdr(draws, level):
    """Calculates the High Density Region (HDR) using percentiles."""
    lower_perc = (100 - level) / 2
    upper_perc = 100 - lower_perc
    lower_bound = jnp.percentile(draws, lower_perc, axis=0)
    upper_bound = jnp.percentile(draws, upper_perc, axis=0)
    return lower_bound, upper_bound


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # --- Configuration ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # *** !!! Point this to your correct model file !!! ***
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        print(f"Using model file: {mod_file_path}")

        if not os.path.exists(mod_file_path):
            raise FileNotFoundError(f"Model file not found at: {mod_file_path}")

        # --- [1] PARSING AND SOLVING ---
        print("\n--- [1] Parsing and Solving Model ---")
        with open(mod_file_path, 'r') as f:
            model_def = f.read()

        # Parse stationary model
        (func_A, func_B, func_C, func_D,
         ordered_stat_vars, stat_shocks, param_names_stat, param_assignments_stat,
         _, _) = dp.parse_lambdify_and_order_model(model_def)

        # Parse trend/observation components
        trend_vars, trend_shocks = dp.extract_trend_declarations(model_def)
        trend_equations = dp.extract_trend_equations(model_def)
        obs_vars = dp.extract_observation_declarations(model_def)
        measurement_equations = dp.extract_measurement_equations(model_def)
        trend_stderr_params = dp.extract_trend_shock_stderrs(model_def)

        # Combine parameters
        all_param_names = list(dict.fromkeys(param_names_stat + list(trend_stderr_params.keys())).keys())
        all_param_assignments = param_assignments_stat.copy()
        all_param_assignments.update(trend_stderr_params)

        # --- Parameter Evaluation (Using defaults as needed) ---
        test_param_values = all_param_assignments.copy()
        default_test_values = { # Provide defaults for parameters NOT assigned in file
            'b1': 0.7, 'b4': 0.7, 'a1': 0.5, 'a2': 0.1,
            'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
            'rho_L_GDP_GAP': 0.75, 'rho_DLA_CPI': 0.75,
            'rho_rs': 0.75, 'rho_rs2': 0.01
        }
        # Assign default std devs (usually 1) only if not parsed
        all_shocks_combined = stat_shocks + trend_shocks
        shock_std_devs = {}
        for shk in all_shocks_combined:
            pname = f"sigma_{shk}"
            if pname in test_param_values:
                shock_std_devs[shk] = test_param_values[pname]
            else:
                # Default for stationary shocks often assumed 1 unless specified
                is_trend_shock = shk in trend_shocks
                default_std = 0.01 # Or 0.0 for unused shocks like SHP_PI_TREND if sigma=0
                if is_trend_shock and 'SHP_' in shk and pname not in test_param_values: 
                    default_std = 0.0 # Heuristic
                print(f" Note: Std dev for '{shk}' (param '{pname}') not found. Using default: {default_std}")
                test_param_values[pname] = default_std # Add default to values dict
                all_param_names.append(pname) # Ensure it's in the name list if added
                shock_std_devs[shk] = default_std
        all_param_names = list(dict.fromkeys(all_param_names).keys()) # Update list


        # --- Final Argument Lists ---
        test_args = []
        stat_test_args = []
        missing_params = []
        for p in all_param_names:
             val = test_param_values.get(p, default_test_values.get(p))
             if val is None:
                 missing_params.append(p)
                 test_args.append(0.0) # Fallback - will likely cause issues
             else:
                 test_args.append(val)
        if missing_params:
             raise ValueError(f"Missing parameter values: {missing_params}")

        # Arguments specifically for stationary system functions
        for p_stat in param_names_stat:
             stat_test_args.append(test_param_values[p_stat])


        # Evaluate stationary matrices
        A_num_stat = func_A(*stat_test_args)
        B_num_stat = func_B(*stat_test_args)
        C_num_stat = func_C(*stat_test_args)
        D_num_stat = func_D(*stat_test_args)

        # Solve stationary model (SDA)
        P_sol_stat, _, res_ratio_stat = dp.solve_quadratic_matrix_equation(A_num_stat, B_num_stat, C_num_stat)
        if P_sol_stat is None or res_ratio_stat > 1e-6:
            print(f"Warning: Stationary solver issue. Residual Ratio: {res_ratio_stat:.2e}")
            # Decide whether to stop or proceed with caution
            if P_sol_stat is None: raise RuntimeError("Stationary solver failed.")

        # Compute stationary Q
        Q_sol_stat = dp.compute_Q(A_num_stat, B_num_stat, D_num_stat, P_sol_stat)
        if Q_sol_stat is None: raise RuntimeError("Failed to compute Q_stationary.")

        # --- Build Trend and Observation Matrices ---
        (func_P_trends, func_Q_trends,
         ordered_trend_state_vars, contemp_trend_defs) = dp.build_trend_matrices(
            trend_equations, trend_vars, trend_shocks, all_param_names, all_param_assignments
        )

        (func_Omega, ordered_obs_vars) = dp.build_observation_matrix(
            measurement_equations, obs_vars, ordered_stat_vars,
            ordered_trend_state_vars, contemp_trend_defs,
            all_param_names, all_param_assignments
         )

        # Evaluate trend and observation matrices
        P_num_trend = func_P_trends(*test_args)
        Q_num_trend = func_Q_trends(*test_args)
        # Omega evaluated inside build_augmented_state_space

        # --- Scale Q matrices to get R matrices for Kalman Filter ---
        # Create diagonal Cholesky factor L from shock standard deviations
        # L = diag(std_devs)
        # R = Q @ L
        stat_std_devs = jnp.array([shock_std_devs[shk] for shk in stat_shocks])
        trend_std_devs = jnp.array([shock_std_devs[shk] for shk in trend_shocks])

        R_sol_stat = Q_sol_stat @ jnp.diag(stat_std_devs)
        R_num_trend = Q_num_trend @ jnp.diag(trend_std_devs)

        # --- Build Augmented System ---
        # Note: build_augmented_state_space now takes R matrices directly
        # *** Corrected function name and arguments ***
        (P_aug, R_aug, Omega_num, # Pass func_Omega, R_stat, R_trend
         aug_state_vars, aug_shocks, obs_vars_ordered) = dp.build_augmented_state_space(
            P_sol_stat, R_sol_stat,     # Stationary solution P, R
            P_num_trend, R_num_trend, # Trend solution P, R
            func_Omega,              # Observation matrix function
            ordered_stat_vars, ordered_trend_state_vars, obs_vars,
            stat_shocks, trend_shocks,
            test_args               # Parameter values for evaluating Omega
        )

        print("\nAugmented System Ready.")
        # Ensure build_augmented_state_space_with_R exists in your parser file
        # and correctly combines P's and R's (scaled Q's) into P_aug, R_aug.
        # It should build R_aug block-diagonally like Q_aug was built before.
        # R_aug = block_diag(R_sol_stat, R_num_trend) padded with zeros if needed.

        print("\nAugmented System Ready.")
        print(f" P_aug shape: {P_aug.shape}")
        print(f" R_aug shape: {R_aug.shape}") # R_aug is Q_aug scaled by std devs
        print(f" Omega_num shape: {Omega_num.shape}")
        print(f" Augmented States: {aug_state_vars}")
        print(f" Augmented Shocks: {aug_shocks}")


        # --- [2] IRF ANALYSIS ---
        print("\n--- [2] Generating Impulse Responses ---")
        horizon = 40
        # *** Select Shock for IRF ***
        shock_name_irf = "SHK_RS"          # Example Stationary Shock
        #shock_name_irf = "SHK_L_GDP_TREND" # Example Trend Shock

        if shock_name_irf not in aug_shocks:
            print(f"ERROR: IRF shock '{shock_name_irf}' not found in {aug_shocks}")
        else:
            shock_index_irf = aug_shocks.index(shock_name_irf)
            print(f" Calculating IRFs for shock: {shock_name_irf} (index {shock_index_irf})")

            # State IRFs (using R_aug for shock impact)
            # Need a modified irf function that takes R instead of Q
            # irf_states_aug = dp.irf_R(P_aug, R_aug, shock_index=shock_index_irf, horizon=horizon)
            # For now, assume dp.irf works with Q, so we pass R_aug as Q (unit shock input)
            # The magnitude comes from R_aug containing the std dev scaling.
            irf_states_aug = dp.irf(P_aug, R_aug, shock_index=shock_index_irf, horizon=horizon)

            # Observable IRFs
            irf_observables_vals = dp.irf_observables(P_aug, R_aug, Omega_num, shock_index=shock_index_irf, horizon=horizon)

            # Plotting IRFs (Optional - reusing code from parser script)
            dp.plot_irfs(irf_states_aug, aug_state_vars, horizon, f"State IRFs to {shock_name_irf}")
            dp.plot_irfs(irf_observables_vals, obs_vars_ordered, horizon, f"Observable IRFs to {shock_name_irf}")
            # Ensure plot_irfs exists in your parser file


        # --- [3] DATA SIMULATION ---
        print("\n--- [3] Simulating Data ---")
        sim_key = random.PRNGKey(99)
        num_sim_steps = 250

        # *** Define Observation Noise Covariance H ***
        # Needs to be based on assumptions about measurement error.
        # Example: Independent errors with some std dev for each observable.
        n_obs = len(obs_vars_ordered)
        # Example: 0.1 std dev for all observables
        obs_noise_std_devs = jnp.ones(n_obs) * 0.1
        # obs_noise_std_devs = obs_noise_std_devs.at[1].set(0.05) # Customize if needed
        H_obs = jnp.diag(obs_noise_std_devs**2)
        print(f" Observation Noise Covariance (H):\n{H_obs}")

        # --- Define Initial Conditions for Simulation/Filter ---
        init_x_aug = jnp.zeros(P_aug.shape[0]) # Start at zero mean
        # Use unconditional variance if possible, otherwise identity matrix
        # init_P_aug = calculate_unconditional_variance(P_aug, R_aug) # Needs function
        init_P_aug = jnp.eye(P_aug.shape[0]) * 1.0 # Simple identity matrix start
        print(f" Initial State Covariance (P0):\n{init_P_aug}")

        sim_states, sim_observations = simulate_state_space(
            P_aug, R_aug, Omega_num, H_obs, init_x_aug, init_P_aug, sim_key, num_sim_steps
        )
        print(f" Simulated {num_sim_steps} steps.")

        # Introduce Missing Values (Optional)
        sim_obs_with_nan = sim_observations.at[int(num_sim_steps*0.2):int(num_sim_steps*0.3), :].set(jnp.nan)


        # --- [4] KALMAN FILTERING AND SMOOTHING ---
        print("\n--- [4] Running Kalman Filter & Smoothers ---")
        num_sim_smoother_draws = 500 # Number of draws for simulation smoother
        hdr_levels = [68, 80]      # HDR levels to calculate

        # Instantiate Kalman Filter
        kf_augmented = KalmanFilter(
            T=P_aug,         # Transition matrix P
            R=R_aug,         # Shock matrix R (scaled Q)
            C=Omega_num,     # Observation matrix Omega
            H=H_obs,         # Observation noise covariance H
            init_x=init_x_aug, # Initial state mean
            init_P=init_P_aug  # Initial state covariance
        )

        # Run Filter
        print(" Running Filter...")
        filter_outs_aug = kf_augmented.filter(sim_obs_with_nan)
        _, _, x_filt_aug, _ = filter_outs_aug
        print(" Filter finished.")

        # Run RTS Smoother
        print(" Running RTS Smoother...")
        x_smooth_rts_aug, p_smooth_rts_aug = kf_augmented.smooth(sim_obs_with_nan)
        print(" RTS Smoother finished.")

        # Run Simulation Smoother
        print(f" Running Simulation Smoother ({num_sim_smoother_draws} draws)...")
        smoother_key = random.PRNGKey(111)
        sim_smoother_result_aug = kf_augmented.simulation_smoother(
            sim_obs_with_nan, smoother_key, num_draws=num_sim_smoother_draws
        )

        # Unpack simulation smoother results and calculate HDRs
        hdrs = {}
        if num_sim_smoother_draws == 1:
            mean_smooth_sim_aug = sim_smoother_result_aug
            all_draws_aug = sim_smoother_result_aug[None, :, :] # Add batch dim
            print(" Single draw obtained.")
        elif num_sim_smoother_draws > 1:
            mean_smooth_sim_aug, _, all_draws_aug = sim_smoother_result_aug
            print(" Calculating HDRs...")
            for level in hdr_levels:
                hdrs[level] = calculate_hdr(all_draws_aug, level)
            print(" Mean and HDRs obtained.")
        else:
             raise ValueError("num_sim_smoother_draws must be >= 1")


        # --- [5] PLOTTING SIMULATION & FILTERING RESULTS ---
        print("\n--- [5] Plotting Simulation vs. Filter/Smoothers ---")

        # Convert relevant JAX arrays to NumPy for plotting
        states_np = onp.array(sim_states)
        obs_nan_np = onp.array(sim_obs_with_nan)
        x_filt_np = onp.array(x_filt_aug)
        x_smooth_rts_np = onp.array(x_smooth_rts_aug)
        mean_smooth_sim_np = onp.array(mean_smooth_sim_aug)
        all_draws_np = onp.array(all_draws_aug) # Shape [num_draws, T, n_aug]
        hdrs_np = {level: (onp.array(lower), onp.array(upper))
                   for level, (lower, upper) in hdrs.items()}


        time_axis = onp.arange(num_sim_steps)

        for i in range(P_aug.shape[0]): # Plot each augmented state variable
            plt.figure(figsize=(12, 6))
            state_name = aug_state_vars[i]

            # Plot true state
            plt.plot(time_axis, states_np[:, i], "g-", linewidth=2.5, label="True Simulated State", zorder=5)

            # Plot filtered state
            # plt.plot(time_axis, x_filt_np[:, i], "c--", linewidth=1, label="Filtered (Mean)", zorder=4)

            # Plot RTS smoothed state
            plt.plot(time_axis, x_smooth_rts_np[:, i], "r-.", linewidth=1.5, label="RTS Smoothed (Mean)", zorder=6)

            # Plot Simulation Smoother Mean and HDRs
            if num_sim_smoother_draws > 0:
                plt.plot(time_axis, mean_smooth_sim_np[:, i], "k-", linewidth=2, label=f"Sim. Smoother (Mean, {num_sim_smoother_draws} draws)", zorder=7)
                # Plot HDR bands in reverse order (wider first)
                colors = plt.cm.viridis(onp.linspace(0.3, 0.7, len(hdr_levels))) # Color gradient for bands
                sorted_levels = sorted(hdr_levels, reverse=True)
                for idx, level in enumerate(sorted_levels):
                     if level in hdrs_np:
                         lower, upper = hdrs_np[level]
                         plt.fill_between(time_axis, lower[:, i], upper[:, i],
                                          color=colors[idx], alpha=0.3,
                                          label=f"{level}% HDR (Sim. Smoother)", zorder=idx+1)


            plt.title(f"State Variable: {state_name}")
            plt.xlabel("Time step (t)")
            plt.ylabel("Value")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # --- Error Handling ---
    except FileNotFoundError as e:
         print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    except RuntimeError as e:
         print(f"\nA runtime error occurred: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- END OF main_script.py ---