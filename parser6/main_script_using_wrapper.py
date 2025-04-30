# --- START OF main_script_using_wrapper.py ---
import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random

# Ensure JAX is configured (can also be done inside the wrapper)
os.environ['JAX_PLATFORMS'] = 'cpu' # Or 'gpu'
jax.config.update("jax_enable_x64", True)
print(f"main_script: JAX float64 enabled: {jax.config.jax_enable_x64}")
print(f"main_script: JAX version: {jax.__version__}")
print(f"main_script: JAX backend: {jax.default_backend()}")


# --- Import the Wrapper ---
from dynare_model_wrapper import DynareModel
# Import helpers if needed (e.g., plot_irfs from parser, calculate_hdr from main)
from Dynare_parser_sda_solver import plot_irfs as dp_plot_irfs # Rename to avoid clash
# Helper from original main.py
def calculate_hdr(draws, level):
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
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        print(f"Using model file: {mod_file_path}")

        # --- [1] Initialize the Model (Parses Once) ---
        # This performs the expensive parsing and lambdification step.
        model = DynareModel(mod_file_path)

        # --- [2] Define Parameter Set ---
        # Start with defaults parsed from the file
        param_values = model.default_param_assignments.copy()
        # Override or add parameters as needed
        param_values.update({
            'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12,
            'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
            'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
            'rho_rs': 0.75, 'rho_rs2': 0.05,
            # Ensure sigma_ parameters are present (use defaults or provide values)
            # Example: Make sure defaults from file or manual overrides exist
            # If sigma_SHK_L_GDP_TREND=0.01 was in the file via stderr, it's included.
            # If sigma_SHK_RS wasn't, we need to add it.
             'sigma_SHK_L_GDP_GAP': 0.1, # Assuming stationary shocks default to 1 if not specified
             'sigma_SHK_DLA_CPI': 1.0,
             'sigma_SHK_RS': 0.25,       # Specific value for RS shock
             'sigma_SHK_PERM_TARGET': 0.0, # Unused shock
             'sigma_SHP_L_GDP_TREND': 0.0, # Unused shock
             # Add any other sigma_ parameters needed based on model.aug_shocks_structure
        })
        # Add defaults for any missing sigmas found in the structure but not valued yet
        # (Safer: _prepare_params inside DynareModel will raise error if sigmas are missing)
        for shk in model.aug_shocks_structure:
            pname = f"sigma_{shk}"
            if pname not in param_values:
                print(f"Note: Param '{pname}' for shock '{shk}' not in user dict or defaults. Setting to 0.01 (example).")
                param_values[pname] = 0.01 # Or raise error, or use a better default heuristic

        print(f"\nUsing Parameter Dictionary:\n{param_values}")

        # --- [3] Solve the Model ---
        # The solve method evaluates matrices and finds the solution for *these* parameters
        solution = model.solve(param_values)
        print("\nModel Solved. Augmented state space dimensions:")
        print(f" P_aug shape: {solution['P_aug'].shape}")
        print(f" R_aug shape: {solution['R_aug'].shape}")
        print(f" Omega shape: {solution['Omega'].shape}")

        # --- [4] Calculate and Plot IRFs ---
        shock_to_plot = "SHK_RS" #"SHK_L_GDP_TREND"
        horizon = 40
        print(f"\nCalculating IRFs for shock: {shock_to_plot}")
        irf_results = model.get_irf(param_values, shock_name=shock_to_plot, horizon=horizon)

        # Plotting using the imported function
        dp_plot_irfs(irf_results['state_irf'], irf_results['state_names'], horizon,
                     f"State IRFs to {shock_to_plot} (Wrapper)")
        dp_plot_irfs(irf_results['observable_irf'], irf_results['observable_names'], horizon,
                     f"Observable IRFs to {shock_to_plot} (Wrapper)")

        # --- [5] Simulate Data ---
        sim_key = random.PRNGKey(123)
        num_sim_steps = 250

        # Define Observation Noise Covariance H
        n_obs = model.n_obs
        obs_noise_std_devs = jnp.ones(n_obs) * 0.1 # Example: 0.1 std dev
        H_obs_sim = jnp.diag(obs_noise_std_devs**2)

        # Define Initial Conditions for Simulation
        n_aug = model.n_state_aug
        init_x_sim = jnp.zeros(n_aug) # Start at zero mean
        init_P_sim = jnp.eye(n_aug) * 1.0 # Simple identity matrix start

        print("\nSimulating data...")
        simulation_output = model.simulate(param_values, H_obs_sim, init_x_sim, init_P_sim, sim_key, num_sim_steps)
        sim_states = simulation_output['sim_states']
        sim_observations = simulation_output['sim_observations']
        print(f" Simulated {num_sim_steps} steps.")

        # Introduce Missing Values (Optional)
        sim_obs_with_nan = sim_observations.at[int(num_sim_steps*0.2):int(num_sim_steps*0.3), :].set(jnp.nan)


        # --- [6] Run Kalman Filter/Smoothers ---
        filter_key = random.PRNGKey(456)
        num_draws_sim_smooth = 500
        hdr_levels_plot = [68, 90]

        # Use same H and initial conditions as simulation for this example
        # (In practice, filter initials might differ)
        H_obs_filt = H_obs_sim
        init_x_filt = init_x_sim
        init_P_filt = init_P_sim

        print("\nRunning Kalman Filter and Smoothers...")
        kalman_results = model.run_kalman(
            param_dict=param_values,
            ys=sim_obs_with_nan,
            H_obs=H_obs_filt,
            init_x_mean=init_x_filt,
            init_P_cov=init_P_filt,
            smoother_key=filter_key, # Provide key ONLY if num_draws > 0
            num_sim_smoother_draws=num_draws_sim_smooth
        )
        print("Kalman operations complete.")

        # --- [7] Plotting Simulation & Filtering Results ---
        print("\nPlotting Simulation vs. Filter/Smoothers...")

        # Convert relevant JAX arrays to NumPy for plotting
        states_np = onp.array(sim_states)
        obs_nan_np = onp.array(sim_obs_with_nan)
        x_filt_np = onp.array(kalman_results['filtered_states'])
        x_smooth_rts_np = onp.array(kalman_results['rts_smoothed_states'])

        # Simulation smoother results (check if they exist)
        mean_smooth_sim_np = None
        all_draws_np = None
        hdrs_np = {}
        if num_draws_sim_smooth > 0:
            mean_smooth_sim_np = onp.array(kalman_results['sim_smoothed_mean'])
            all_draws_np = onp.array(kalman_results['sim_smoothed_draws']) # Shape [num_draws, T, n_aug]
            print(" Calculating HDRs for plotting...")
            for level in hdr_levels_plot:
                hdrs_np[level] = calculate_hdr(all_draws_np, level) # Use helper
                hdrs_np[level] = (onp.array(hdrs_np[level][0]), onp.array(hdrs_np[level][1]))
            print(" HDRs calculated.")

        time_axis = onp.arange(num_sim_steps)
        state_names_plot = solution['aug_state_vars'] # Get names from solution

        for i in range(n_aug): # Plot each augmented state variable
            plt.figure(figsize=(12, 6))
            state_name = state_names_plot[i]

            # Plot true state
            plt.plot(time_axis, states_np[:, i], "g-", linewidth=2.5, label="True Simulated State", zorder=5)

            # Plot RTS smoothed state
            plt.plot(time_axis, x_smooth_rts_np[:, i], "r-.", linewidth=1.5, label="RTS Smoothed (Mean)", zorder=6)

            # Plot Simulation Smoother Mean and HDRs if available
            if mean_smooth_sim_np is not None:
                plt.plot(time_axis, mean_smooth_sim_np[:, i], "k-", linewidth=2, label=f"Sim. Smoother (Mean, {num_draws_sim_smooth} draws)", zorder=7)
                colors = plt.cm.viridis(onp.linspace(0.3, 0.7, len(hdr_levels_plot)))
                sorted_levels = sorted(hdr_levels_plot, reverse=True)
                for idx, level in enumerate(sorted_levels):
                     if level in hdrs_np:
                         lower, upper = hdrs_np[level]
                         plt.fill_between(time_axis, lower[:, i], upper[:, i],
                                          color=colors[idx], alpha=0.3,
                                          label=f"{level}% HDR (Sim. Smoother)", zorder=idx+1)

            plt.title(f"State Variable: {state_name} (Wrapper Example)")
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

# --- END OF main_script_using_wrapper.py ---