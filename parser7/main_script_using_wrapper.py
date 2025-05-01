# --- START OF main_script_using_wrapper.py ---
import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random


# --- Numpyro Import ---
try:
    import numpyro
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    print("main_script: Warning - numpyro not found. Estimation will not run.")

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
        # Make sure the .dyn file includes the 'shocks;' block with stderr if you want defaults
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        print(f"Using model file: {mod_file_path}")

        # --- [1] Initialize the Model ---
        model = DynareModel(mod_file_path)

        # --- [2] Define Parameter Set (for Simulation/Filter Example) ---
        # Use defaults parsed from file (including any std errs)
        sim_param_values = model.default_param_assignments.copy()
        # Override specific parameters for simulation if needed
        sim_param_values.update({
            'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12,
            'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
            'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
            'rho_rs': 0.75, 'rho_rs2': 0.05,
            # Make sure sigma values are reasonable (e.g., > 0)
            # Let's use the defaults from the file if present, else define
             'sigma_SHK_L_GDP_GAP': sim_param_values.get('sigma_SHK_L_GDP_GAP', 0.1),
             'sigma_SHK_DLA_CPI': sim_param_values.get('sigma_SHK_DLA_CPI', 0.1),
             'sigma_SHK_RS': sim_param_values.get('sigma_SHK_RS', 0.2),
             'sigma_SHK_L_GDP_TREND': sim_param_values.get('sigma_SHK_L_GDP_TREND', 0.05),
             'sigma_SHK_G_TREND': sim_param_values.get('sigma_SHK_G_TREND', 0.01),
             'sigma_SHP_PI_TREND': sim_param_values.get('sigma_SHP_PI_TREND', 0.005), # Trend shocks
             'sigma_SHK_RR_TREND': sim_param_values.get('sigma_SHK_RR_TREND', 0.01),
             # sigma_SHK_RS_TREND is derived from RR_TREND and PI_TREND, no direct shock in model def
        })
        # Add defaults for any missing sigmas found in the structure (should be covered by parser now)
        for shk in model.aug_shocks_structure:
            pname = f"sigma_{shk}"
            if pname not in sim_param_values:
                 # This case should be less likely if stderr blocks are used
                 print(f"Note: Param '{pname}' for shock '{shk}' not in user dict or defaults. Setting example 0.01.")
                 sim_param_values[pname] = 0.01

        print(f"\n--- Using Parameter Dictionary for Simulation ---\n{sim_param_values}")

        # --- [3] Solve the Model (for Simulation) ---
        # solve now returns JAX arrays where needed
        solution_sim = model.solve(sim_param_values)

        # --- [4] Simulate Data ---
        sim_key = random.PRNGKey(123)
        num_sim_steps = 200 # Shorter simulation for faster testing
        n_obs = model.n_obs
        n_aug = model.n_state_aug

        # Define Observation Noise Covariance H (Assumed diagonal, known)
        # Make H realistic but small relative to state variance
        obs_noise_variances = jnp.array([0.05**2, 0.05**2, 0.01**2, 0.05**2]) # Variance for L_GDP_OBS, DLA_CPI_OBS, PI_TREND_OBS, RS_OBS
        if len(obs_noise_variances) != n_obs:
             raise ValueError(f"Need {n_obs} obs noise variances, got {len(obs_noise_variances)}")
        H_obs_sim = jnp.diag(obs_noise_variances)

        # Define Initial Conditions for Simulation
        init_x_sim = jnp.zeros(n_aug) # Start at zero mean (steady state)
        # Use unconditional variance from model? Harder. Use identity guess.
        init_P_sim = jnp.eye(n_aug) * 0.1 # Small initial uncertainty

        print("\nSimulating data...")
        simulation_output = model.simulate(sim_param_values, H_obs_sim, init_x_sim, init_P_sim, sim_key, num_sim_steps)
        sim_states = simulation_output['sim_states']
        sim_observations = simulation_output['sim_observations']
        print(f" Simulated {num_sim_steps} steps.")

        # Introduce Missing Values (Optional)
        # sim_obs_with_nan = sim_observations.at[int(num_sim_steps*0.2):int(num_sim_steps*0.3), 0].set(jnp.nan) # Make only GDP missing
        sim_obs_with_nan = sim_observations # Use full data for estimation example

        # --- [5] Define Priors for Estimation ---
        # Select a subset of parameters to estimate
        estimation_priors = {
            # Param: (Dist Name, [params])
            'b1': ('Normal', [0.7, 0.1]),       # Prior mean 0.7, std dev 0.1
            'a1': ('Beta', [6.0, 2.0]),         # Prior mean = 6/(6+2)=0.75, implies persistence
            'g1': ('Normal', [0.7, 0.1]),
            # Standard deviations (must be positive)
            'sigma_SHK_L_GDP_GAP': ('InverseGamma', [4.0, 0.1]), # Prior mean ~ 0.1/(4-1) = 0.033
            'sigma_SHK_DLA_CPI': ('InverseGamma', [4.0, 0.1]),
            'sigma_SHK_RS': ('InverseGamma', [4.0, 0.2]),        # Prior mean ~ 0.2/3 = 0.067
        }
        # Note: We are NOT estimating trend shock sigmas here, they use defaults.
        # Note: Observation noise H is assumed KNOWN here (H_obs_sim).

        # --- [6] Run Estimation ---
        if NUMPYRO_AVAILABLE:
            print("\n--- Running Bayesian Estimation ---")
            est_key = random.PRNGKey(789)
            mcmc_config = {
                'num_warmup': 500,   # Increase for real runs
                'num_samples': 1000, # Increase for real runs
                'num_chains': 2,     # Run multiple chains
            }

            # Use simulated data (ys), known H, and filter initial conditions
            # In practice, use real data and potentially diffuse initial conditions
            # Use H_obs_sim as the known observation noise for the filter/likelihood
            H_obs_filt = H_obs_sim
            init_x_filt = init_x_sim # Use simulation start for simplicity
            init_P_filt = init_P_sim

            try:
                mcmc_results = model.estimate(
                    ys=sim_obs_with_nan,       # Data to fit
                    H_obs=H_obs_filt,          # Assumed known observation noise cov
                    init_x_mean=init_x_filt,   # Filter initial state mean
                    init_P_cov=init_P_filt,    # Filter initial state cov
                    priors=estimation_priors,  # Defined priors
                    mcmc_params=mcmc_config,   # MCMC settings
                    rng_key=est_key            # Random key
                )

                # --- [7] Analyze Estimation Results ---
                print("\n--- Estimation Summary ---")
                # Summary is printed automatically by estimate method
                # Access samples
                posterior_samples = mcmc_results.get_samples()

                # Plot posterior distributions (example for b1)
                plt.figure(figsize=(10, 4))
                plt.hist(onp.array(posterior_samples['b1']), bins=50, density=True, label='Posterior samples')
                # Plot prior density for comparison
                prior_b1_mean, prior_b1_std = estimation_priors['b1'][1]
                x_prior = onp.linspace(prior_b1_mean - 3*prior_b1_std, prior_b1_mean + 3*prior_b1_std, 200)
                from scipy.stats import norm
                plt.plot(x_prior, norm.pdf(x_prior, prior_b1_mean, prior_b1_std), 'r-', label='Prior (Normal)')
                # Plot true value used in simulation
                plt.axvline(sim_param_values['b1'], color='g', linestyle='--', label=f"True Sim Value ({sim_param_values['b1']:.2f})")
                plt.title("Posterior Distribution for 'b1'")
                plt.xlabel("Parameter Value")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True, alpha=0.5)
                plt.show()

                # You can plot other parameters similarly

            except Exception as e_est:
                 print(f"\nAn error occurred during estimation: {e_est}")
                 import traceback
                 traceback.print_exc()

        else:
            print("\n--- Skipping Estimation (numpyro not available) ---")


        # --- [8] Example: Run Filter/Smoother with POSTERIOR MEAN parameters ---
        if NUMPYRO_AVAILABLE and 'posterior_samples' in locals():
            print("\n--- Running Filter/Smoother with Posterior Mean Parameters ---")
            # Calculate posterior mean from samples
            posterior_mean_params = {k: jnp.mean(v) for k, v in posterior_samples.items()}
            print(f"Posterior Mean Parameters:\n{posterior_mean_params}")

            # Combine posterior means with fixed parameters
            final_params_post_mean = model.default_param_assignments.copy()
            # Update with fixed values used during estimation (if any were fixed but not default)
            # final_params_post_mean.update(fixed_params) # Get fixed_params from estimate scope if needed
            # Update with posterior means
            final_params_post_mean.update(posterior_mean_params)

             # Add defaults for any missing sigmas (should be covered by defaults+priors)
            for shk in model.aug_shocks_structure:
                pname = f"sigma_{shk}"
                if pname not in final_params_post_mean:
                    print(f"Note: Param '{pname}' missing after posterior mean update. Using 0.01.")
                    final_params_post_mean[pname] = 0.01

            print(f"\nFull parameter set with posterior means:\n{final_params_post_mean}")

            # Run Kalman with these parameters
            filter_key_post = random.PRNGKey(457)
            num_draws_sim_smooth_post = 50 # Fewer draws maybe
            hdr_levels_plot = [68, 90]

            try:
                kalman_results_post = model.run_kalman(
                    param_dict=final_params_post_mean,
                    ys=sim_obs_with_nan, # Use the same simulated data
                    H_obs=H_obs_filt,       # Use same H as estimation
                    init_x_mean=init_x_filt, # Use same init as estimation
                    init_P_cov=init_P_filt,
                    smoother_key=filter_key_post,
                    num_sim_smoother_draws=num_draws_sim_smooth_post
                )
                print("Kalman operations with posterior mean complete.")

                # Plot comparison: True vs Smoothed (Posterior Mean)
                print("\nPlotting True Simulation vs. Smoothed (Posterior Mean)...")
                states_np = onp.array(sim_states)
                x_smooth_rts_post_np = onp.array(kalman_results_post['rts_smoothed_states'])
                mean_smooth_sim_post_np = None
                all_draws_post_np = None
                hdrs_post_np = {}

                if num_draws_sim_smooth_post > 0 and kalman_results_post.get('sim_smoothed_draws') is not None:
                     mean_smooth_sim_post_np = onp.array(kalman_results_post['sim_smoothed_mean'])
                     all_draws_post_np = onp.array(kalman_results_post['sim_smoothed_draws'])
                     print(" Calculating HDRs for posterior mean smoothing...")
                     for level in hdr_levels_plot:
                         hdrs_post_np[level] = calculate_hdr(all_draws_post_np, level)
                         hdrs_post_np[level] = (onp.array(hdrs_post_np[level][0]), onp.array(hdrs_post_np[level][1]))
                     print(" HDRs calculated.")

                time_axis = onp.arange(num_sim_steps)
                # Get state names from the LATEST solution (based on posterior mean params)
                solution_post = model.solve(final_params_post_mean)
                state_names_plot = solution_post['aug_state_vars']

                for i in range(n_aug): # Plot each augmented state variable
                    plt.figure(figsize=(12, 6))
                    state_name = state_names_plot[i] if i < len(state_names_plot) else f"State {i}"

                    plt.plot(time_axis, states_np[:, i], "g-", linewidth=2.5, label="True Simulated State", zorder=5)
                    # Plot RTS smoothed state (from posterior mean run)
                    plt.plot(time_axis, x_smooth_rts_post_np[:, i], "b-.", linewidth=1.5, label="RTS Smoothed (Post. Mean)", zorder=6)

                    # Plot Simulation Smoother Mean and HDRs if available (from posterior mean run)
                    if mean_smooth_sim_post_np is not None:
                        plt.plot(time_axis, mean_smooth_sim_post_np[:, i], "m-", linewidth=2, label=f"Sim. Smooth (Post. Mean, {num_draws_sim_smooth_post} draws)", zorder=7)
                        colors = plt.cm.coolwarm(onp.linspace(0.3, 0.7, len(hdr_levels_plot))) # Different color scheme
                        sorted_levels = sorted(hdr_levels_plot, reverse=True)
                        for idx, level in enumerate(sorted_levels):
                             if level in hdrs_post_np:
                                 lower, upper = hdrs_post_np[level]
                                 plt.fill_between(time_axis, lower[:, i], upper[:, i],
                                                  color=colors[idx], alpha=0.3,
                                                  label=f"{level}% HDR (Post. Mean Sim Smooth)", zorder=idx+1)

                    plt.title(f"State: {state_name} (True vs. Posterior Mean Smoothed)")
                    plt.xlabel("Time step (t)")
                    plt.ylabel("Value")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            except Exception as e_post:
                print(f"\nError running Kalman filter with posterior mean: {e_post}")
                import traceback
                traceback.print_exc()


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