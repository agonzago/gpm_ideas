# --- START OF main_script_using_wrapper.py ---
import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random
import time

#--- Force CPU Execution (Optional) ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update('jax_platforms', 'cpu')
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

# Number of devices (4)

try:
    import numpyro
    NUMPYRO_AVAILABLE = True
    numpyro.set_host_device_count(4)

except ImportError:
    NUMPYRO_AVAILABLE = False
    print("main_script: Warning - numpyro not found. Estimation will not run.")

# Ensure JAX is configured
jax.config.update("jax_enable_x64", True)

# --- Import the Wrapper ---
from dynare_model_wrapper import DynareModel
# Import helpers if needed
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
        start_time_script = time.time()
        # --- Configuration ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        print(f"Using model file: {mod_file_path}")

        # --- [1] Initialize the Model ---
        # Pass verbose=False to the underlying parser functions via the wrapper's __init__
        model = DynareModel(mod_file_path)

        # --- [2] Define Parameter Set for SIMULATION ---
        sim_param_values = model.default_param_assignments.copy()
        sim_param_values.update({
            'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12,
            'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
            'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
            'rho_rs': 0.75, 'rho_rs2': 0.05,
            'sigma_SHK_L_GDP_TREND': 0.01, 'sigma_SHK_G_TREND': 0.03,
            'sigma_SHK_L_GDP_GAP': 0.10, 'sigma_SHK_DLA_CPI': 0.05,
            'sigma_SHK_RS': 0.15, 'sigma_SHP_PI_TREND': 0.01,
            'sigma_SHK_RR_TREND': 0.02, 'sigma_SHK_RS_TREND': 0.01,
        })
        missing_sim_params = [p for p in model.all_param_names if p not in sim_param_values]
        if missing_sim_params:
            raise ValueError(f"Missing parameter values in sim_param_values: {missing_sim_params}")

        # --- [4] Simulate Data ---
        sim_key = random.PRNGKey(123)
        num_sim_steps = 300
        n_obs = model.n_obs
        n_aug = model.n_state_aug
        obs_noise_variances = jnp.array([0.002**2, 0.002**2, 0.001**2, 0.001**2])
        if len(obs_noise_variances) != n_obs:
             raise ValueError(f"Need {n_obs} obs noise variances, got {len(obs_noise_variances)}")
        H_obs_sim = jnp.diag(obs_noise_variances)
        init_x_sim = jnp.zeros(n_aug)
        init_P_sim = jnp.eye(n_aug) * 0.1

        print(f"\nSimulating {num_sim_steps} steps of data...")
        sim_start_time = time.time()
        simulation_output = model.simulate(sim_param_values, H_obs_sim, init_x_sim, init_P_sim, sim_key, num_sim_steps)
        sim_states = simulation_output['sim_states']
        sim_observations = simulation_output['sim_observations']
        sim_end_time = time.time()
        print(f" Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")

        sim_obs_with_nan = sim_observations # Use data without NaNs for this run

        # --- [5] Define Priors for ESTIMATION ---
        # Using tight priors centered on true values
        estimation_priors_tight = {
            'b1': ('Beta', [75.0, 25.0]), 'b4': ('Beta', [65.0, 35.0]),
            'a1': ('Beta', [55.0, 45.0]), 'a2': ('Normal', [0.12, 0.01]),
            'g1': ('Beta', [70.0, 30.0]), 'g2': ('Normal', [0.30, 0.03]),
            'g3': ('Normal', [0.25, 0.025]), 'rho_L_GDP_GAP': ('Beta', [80.0, 20.0]),
            'rho_DLA_CPI': ('Beta', [70.0, 30.0]), 'rho_rs': ('Beta', [75.0, 25.0]),
            'rho_rs2': ('Beta', [5.0, 95.0]),
            'sigma_SHK_L_GDP_TREND': ('InverseGamma', [25.0, 0.01 * 26]),
            'sigma_SHK_G_TREND':     ('InverseGamma', [25.0, 0.03 * 26]),
            'sigma_SHK_L_GDP_GAP':   ('InverseGamma', [25.0, 0.10 * 26]),
            'sigma_SHK_DLA_CPI': ('InverseGamma', [25.0, 0.05 * 26]),
            'sigma_SHK_RS':      ('InverseGamma', [25.0, 0.15 * 26]),
            'sigma_SHP_PI_TREND':('InverseGamma', [25.0, 0.01 * 26]),
            'sigma_SHK_RR_TREND':('InverseGamma', [25.0, 0.02 * 26]),
            'sigma_SHK_RS_TREND':('InverseGamma', [25.0, 0.01 * 26]),
        }
        estimation_priors = estimation_priors_tight

        # Verify priors cover all model parameters
        prior_param_names = set(estimation_priors.keys())
        all_model_params = set(model.all_param_names)
        if prior_param_names != all_model_params:
             missing = all_model_params - prior_param_names
             extra = prior_param_names - all_model_params
             raise ValueError(f"Prior/Model parameter mismatch. Missing: {missing}, Extra: {extra}")

        # --- [6] Run Estimation ---
        if NUMPYRO_AVAILABLE:
            print("\n--- Running Bayesian Estimation ---")
            est_key = random.PRNGKey(789)
            mcmc_config = {
                'num_warmup': 1000,
                'num_samples': 1000,
                'num_chains': 2,
                'target_accept_prob': 0.90
            }
            H_obs_filt = H_obs_sim
            init_x_filt = init_x_sim
            init_P_filt = init_P_sim

            try:
                est_start_time = time.time()
                mcmc_results = model.estimate(
                    ys=sim_obs_with_nan,
                    H_obs=H_obs_filt,
                    init_x_mean=init_x_filt,
                    init_P_cov=init_P_filt,
                    priors=estimation_priors,
                    mcmc_params=mcmc_config,
                    rng_key=est_key,
                    verbose_solver=False
                    # init_param_values=sim_param_values # <<< REMOVE THIS LINE
                )
                est_end_time = time.time()
                print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")

                # --- [7] Analyze Estimation Results ---
                print("\n--- Estimation Summary ---")
                mcmc_results.print_summary()
                posterior_samples = mcmc_results.get_samples()

                # --- Plotting Posterior Distributions (Optional) ---
                # ...

            except Exception as e_est:
                 print(f"\nAn error occurred during estimation: {e_est}")
                 import traceback
                 traceback.print_exc()
                 if 'posterior_samples' in locals(): del posterior_samples

        else:
             print("\n--- Skipping Estimation (numpyro not available) ---")


        # --- [8] Example: Run Filter/Smoother with POSTERIOR MEAN parameters ---
        if NUMPYRO_AVAILABLE and 'posterior_samples' in locals():
            print("\n--- Running Filter/Smoother with Posterior Mean Parameters ---")
            posterior_mean_params = {k: jnp.mean(v) for k, v in posterior_samples.items()}

            final_params_post_mean = posterior_mean_params
            filter_key_post = random.PRNGKey(457)
            num_draws_sim_smooth_post = 100
            hdr_levels_plot = [68, 90]

            try:
                kalman_start_time = time.time()
                # This call uses the main filter/smoother which handle y_star correctly now
                kalman_results_post = model.run_kalman(
                    param_dict=final_params_post_mean,
                    ys=sim_obs_with_nan,
                    H_obs=H_obs_filt,
                    init_x_mean=init_x_filt,
                    init_P_cov=init_P_filt,
                    smoother_key=filter_key_post,
                    num_sim_smoother_draws=num_draws_sim_smooth_post
                )
                kalman_end_time = time.time()
                print(f"Kalman operations with posterior mean complete ({kalman_end_time - kalman_start_time:.2f} seconds).")

                # --- Plotting Smoothed States (Optional) ---
                # ... (plotting code remains the same, should work now) ...
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
                         hdrs_post_np[level] = calculate_hdr(jnp.array(all_draws_post_np), level)
                         hdrs_post_np[level] = (onp.array(hdrs_post_np[level][0]), onp.array(hdrs_post_np[level][1]))
                     print(" HDRs calculated.")

                time_axis = onp.arange(num_sim_steps)
                state_names_plot = model.aug_state_vars_structure
                n_states_to_plot = len(state_names_plot)
                cols_smooth = 4
                rows_smooth = (n_states_to_plot + cols_smooth - 1) // cols_smooth
                fig_smooth, axes_smooth = plt.subplots(rows_smooth, cols_smooth, figsize=(min(5*cols_smooth, 20), 3*rows_smooth), sharex=True)
                axes_smooth = axes_smooth.flatten()

                for i in range(n_states_to_plot):
                    ax = axes_smooth[i]
                    state_name = state_names_plot[i]
                    ax.plot(time_axis, states_np[:, i], "g-", linewidth=2.0, label="True Sim", zorder=5)
                    ax.plot(time_axis, x_smooth_rts_post_np[:, i], "b-.", linewidth=1.5, label="RTS (Post. Mean)", zorder=6)
                    if mean_smooth_sim_post_np is not None:
                        ax.plot(time_axis, mean_smooth_sim_post_np[:, i], "m--", linewidth=1.5, label=f"SimSm Mean ({num_draws_sim_smooth_post} draws)", zorder=7)
                        colors = plt.cm.magma(onp.linspace(0.3, 0.7, len(hdr_levels_plot)))
                        sorted_levels = sorted(hdr_levels_plot, reverse=True)
                        for idx_hdr, level in enumerate(sorted_levels):
                             if level in hdrs_post_np:
                                 lower, upper = hdrs_post_np[level]
                                 ax.fill_between(time_axis, lower[:, i], upper[:, i], color=colors[idx_hdr], alpha=0.25, label=f"{level}% HDR", zorder=idx_hdr+1)
                    ax.set_title(f"{state_name}")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    if i == 0: ax.legend(fontsize=8)
                for i in range(n_states_to_plot, len(axes_smooth)): axes_smooth[i].set_visible(False)
                fig_smooth.suptitle('State Variables: True vs. Smoothed (using Posterior Mean Params)', fontsize=16)
                fig_smooth.tight_layout(rect=[0, 0.03, 1, 0.96])
                plt.show() # <<< Uncomment to see plots


            except Exception as e_post:
                print(f"\nError running Kalman filter with posterior mean: {e_post}")
                import traceback
                traceback.print_exc()

        end_time_script = time.time()
        print(f"\n--- Script finished. Total time: {end_time_script - start_time_script:.2f} seconds ---")

    # --- Error Handling ---
    except FileNotFoundError as e: print(f"\nError: {e}")
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

