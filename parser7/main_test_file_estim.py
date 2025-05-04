# --- START OF FILE main_test_file_estimation.py ---

import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random
import time
from typing import Dict, List, Tuple, Optional, Union, Any

#--- Force CPU Execution (Optional) ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update('jax_platforms', 'cpu')
    print(f"JAX targeting CPU.")
except Exception as e_cpu:
    print(f"Warning: Could not force CPU platform: {e_cpu}")
print(f"JAX default platform: {jax.default_backend()}")

# Ensure JAX is configured for float64 if enabled
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
print(f"Using JAX with dtype: {_DEFAULT_DTYPE}")

# --- Numpyro Imports (Conditional) ---
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_uniform
    NUMPYRO_AVAILABLE = True
    # Configure numpyro for multi-chain execution if desired (set based on your CPU cores)
    # try:
    #     num_devices_to_use = min(4, jax.device_count()) # Use up to 4 devices if available
    #     numpyro.set_host_device_count(num_devices_to_use)
    #     print(f"Numpyro configured to use {numpyro.get_host_device_count()} host devices.")
    # except Exception as e_np_config:
    #     print(f"Warning: Could not configure numpyro device count: {e_np_config}")

except ImportError:
    NUMPYRO_AVAILABLE = False
    print("Warning: numpyro not found. Estimation functionality will be disabled.")


# --- Import the Wrapper ---
from dynare_model_wrapper import DynareModel
# Import helpers if needed
from Dynare_parser_sda_solver import plot_irfs as dp_plot_irfs # Rename
from Kalman_filter_jax_old import KalmanFilter # Import KF for standalone likelihood check


# Helper function to calculate HDR intervals
def calculate_hdr(draws: jax.Array, level: float) -> Tuple[onp.ndarray, onp.ndarray]:
    """Calculates Highest Density Region (HDR) using percentiles."""
    lower_perc = (100 - level) / 2
    upper_perc = 100 - lower_perc
    lower_bound = jnp.percentile(draws, lower_perc, axis=0)
    upper_bound = jnp.percentile(draws, upper_perc, axis=0)
    return onp.array(lower_bound), onp.array(upper_bound)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize variables to None or default values
    sim_states = None
    sim_observations = None
    sim_obs_for_filter = None
    kalman_results = None
    mcmc_results = None
    posterior_samples = None
    param_values = None # Will hold the parameters used for sim/filter

    try:
        start_time_script = time.time()
        # --- Configuration ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        print(f"Using model file: {mod_file_path}")
        if not os.path.exists(mod_file_path):
             raise FileNotFoundError(f"Model file not found: {mod_file_path}")

        # --- [1] Initialize the Model ---
        print("\n--- [1] Initializing DynareModel ---")
        init_start_time = time.time()
        model = DynareModel(mod_file_path)
        init_end_time = time.time()
        print(f"Model initialized ({init_end_time - init_start_time:.2f} seconds).")
        print(f"  Augmented State Vars ({model.n_state_aug}): {model.aug_state_vars_structure}")
        print(f"  Observable Vars ({model.n_obs}): {model.ordered_obs_vars}")
        print(f"  All Parameters ({len(model.all_param_names)}): {model.all_param_names}")

        # --- [2] Define Parameter Set for Simulation & Initial Checks ---
        param_values = model.default_param_assignments.copy() # Define param_values here
        param_values.update({
            'b1': 0.75, 'b4': 0.65, 'a1': 0.55, 'a2': 0.12,
            'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
            'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
            'rho_rs': 0.75, 'rho_rs2': 0.05,
            'sigma_SHK_L_GDP_TREND': 0.01, 'sigma_SHK_G_TREND': 0.03,
            'sigma_SHK_L_GDP_GAP': 0.10, 'sigma_SHK_DLA_CPI': 0.05,
            'sigma_SHK_RS': 0.15, 'sigma_SHP_PI_TREND': 0.01,
            'sigma_SHK_RR_TREND': 0.02, 'sigma_SHK_RS_TREND': 0.01,
        })
        missing_params = [p for p in model.all_param_names if p not in param_values]
        if missing_params:
            raise ValueError(f"Missing parameter values in param_values: {missing_params}")
        else:
            print("\n--- [2] Parameter set defined and verified ---")

        # --- [3] Compute and Plot IRFs (Optional Check) ---
        print("\n--- [3] Computing and Plotting Impulse Responses (Check) ---")
        # (IRF code can be kept or commented out for faster estimation test)
        # ... IRF code from previous script ...
        # print("Skipping IRF generation for faster estimation test.")

        # --- [4] Simulate Data ---
        print("\n--- [4] Simulating Data from the Model ---")
        sim_key = random.PRNGKey(123)
        num_sim_steps = 300
        n_obs = model.n_obs
        n_aug = model.n_state_aug
        obs_noise_stdevs = jnp.array([0.002, 0.002, 0.001, 0.001], dtype=_DEFAULT_DTYPE)
        if len(obs_noise_stdevs) != n_obs:
             raise ValueError(f"Number of observation noise std devs mismatch n_obs.")
        H_obs_sim = jnp.diag(obs_noise_stdevs**2)
        init_x_sim = jnp.zeros(n_aug, dtype=_DEFAULT_DTYPE)
        init_P_sim = jnp.eye(n_aug, dtype=_DEFAULT_DTYPE) * 0.1

        print(f"Simulating {num_sim_steps} steps of data...")
        sim_start_time = time.time()
        try:
            simulation_output = model.simulate(
                param_dict=param_values, H_obs=H_obs_sim, init_x_mean=init_x_sim,
                init_P_cov=init_P_sim, key=sim_key, num_steps=num_sim_steps
            )
            sim_states = simulation_output['sim_states']
            sim_observations = simulation_output['sim_observations']
            sim_end_time = time.time()
            print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")
            sim_obs_for_filter = sim_observations # Use this data for filtering and estimation

            # (Plotting simulated observations can be commented out)
            # print("Plotting Simulated Observations...")
            # plt.show(block=False)

        except Exception as e_sim:
            sim_end_time = time.time()
            print(f"An error occurred during simulation ({sim_end_time - sim_start_time:.2f} seconds): {e_sim}")
            import traceback; traceback.print_exc()
            # Crucially, ensure sim_obs_for_filter remains None if sim fails
            sim_obs_for_filter = None


        # --- [5] Run Kalman Filter/Smoother (Standalone Check) ---
        # Run this *before* estimation to ensure filter works with the true params and data
        if sim_obs_for_filter is not None and param_values is not None:
            print("\n--- [5] Running Kalman Filter and Smoother (Check) ---")
            filter_start_time = time.time()
            H_obs_filt = H_obs_sim
            init_x_filt = init_x_sim
            init_P_filt = init_P_sim
            filter_key = random.PRNGKey(457)
            num_draws_sim_smooth = 100 # Can reduce to 0 if slow/not needed for this check
            try:
                kalman_results = model.run_kalman(
                    param_dict=param_values, ys=sim_obs_for_filter, H_obs=H_obs_filt,
                    init_x_mean=init_x_filt, init_P_cov=init_P_filt,
                    smoother_key=filter_key, num_sim_smoother_draws=num_draws_sim_smooth
                )
                filter_end_time = time.time()
                print(f"Kalman Filter/Smoother check complete ({filter_end_time - filter_start_time:.2f} seconds).")
            except Exception as e_filt:
                filter_end_time = time.time()
                print(f"An error occurred during standalone filter/smoother check ({filter_end_time - filter_start_time:.2f} seconds): {e_filt}")
                import traceback; traceback.print_exc()
                kalman_results = None # Ensure results are None if check fails
        else:
             print("\n--- Skipping Standalone Filter/Smoother Check (Missing Sim Data or Params) ---")


        # --- [6] Calculate Log-Likelihood (Standalone Check) ---
        if sim_obs_for_filter is not None and param_values is not None:
            print("\n--- [6] Calculating Log-Likelihood (Standalone Check) ---")
            ll_start_time = time.time()
            try:
                # Use verified parameters, H, initial conditions
                solution = model.solve(param_values)
                P_aug_ll, R_aug_ll, Omega_ll = solution['P_aug'], solution['R_aug'], solution['Omega']
                if not (jnp.all(jnp.isfinite(P_aug_ll)) and jnp.all(jnp.isfinite(R_aug_ll)) and jnp.all(jnp.isfinite(Omega_ll))):
                    raise ValueError("Solution matrices contain non-finite values.")
                kf_ll = KalmanFilter(T=P_aug_ll, R=R_aug_ll, C=Omega_ll, H=H_obs_filt, init_x=init_x_filt, init_P=init_P_filt)
                filter_results_ll = kf_ll.filter_for_likelihood(sim_obs_for_filter)
                total_log_likelihood = jnp.sum(filter_results_ll['log_likelihood_contributions'])
                ll_end_time = time.time()
                print(f"Log-Likelihood check complete ({ll_end_time - ll_start_time:.2f} seconds).")
                print(f"  Total Log-Likelihood (True Params): {total_log_likelihood:.4f}")
                if not jnp.isfinite(total_log_likelihood):
                     print("  WARNING: Log-likelihood (True Params) is non-finite.")
            except Exception as e_ll:
                ll_end_time = time.time()
                print(f"An error occurred during standalone likelihood check ({ll_end_time - ll_start_time:.2f} seconds): {e_ll}")
                import traceback; traceback.print_exc()
        else:
             print("\n--- Skipping Standalone Log-Likelihood Check (Missing Sim Data or Params) ---")


        # # --- [7] Define Priors for ESTIMATION ---
        # # Using tight priors centered on true values (as before)
        # estimation_priors = {
        #     'b1': ('Beta', [75.0, 25.0]), 'b4': ('Beta', [65.0, 35.0]),
        #     'a1': ('Beta', [55.0, 45.0]), 'a2': ('Normal', [0.12, 0.01]),
        #     'g1': ('Beta', [70.0, 30.0]), 'g2': ('Normal', [0.30, 0.03]),
        #     'g3': ('Normal', [0.25, 0.025]), 'rho_L_GDP_GAP': ('Beta', [80.0, 20.0]),
        #     'rho_DLA_CPI': ('Beta', [70.0, 30.0]), 'rho_rs': ('Beta', [75.0, 25.0]),
        #     'rho_rs2': ('Beta', [5.0, 95.0]),
        #     'sigma_SHK_L_GDP_TREND': ('InverseGamma', [25.0, 0.01 * 26]),
        #     'sigma_SHK_G_TREND':     ('InverseGamma', [25.0, 0.03 * 26]),
        #     'sigma_SHK_L_GDP_GAP':   ('InverseGamma', [25.0, 0.10 * 26]),
        #     'sigma_SHK_DLA_CPI': ('InverseGamma', [25.0, 0.05 * 26]),
        #     'sigma_SHK_RS':      ('InverseGamma', [25.0, 0.15 * 26]),
        #     'sigma_SHP_PI_TREND':('InverseGamma', [25.0, 0.01 * 26]),
        #     'sigma_SHK_RR_TREND':('InverseGamma', [25.0, 0.02 * 26]),
        #     'sigma_SHK_RS_TREND':('InverseGamma', [25.0, 0.01 * 26]),
        # }

        # # Verify priors cover all model parameters (essential check)
        # prior_param_names = set(estimation_priors.keys())
        # all_model_params = set(model.all_param_names)
        # if prior_param_names != all_model_params:
        #      missing = all_model_params - prior_param_names
        #      extra = prior_param_names - all_model_params
        #      # Raise error immediately if mismatch
        #      raise ValueError(f"FATAL: Prior/Model parameter mismatch. Missing priors for: {missing}, Extra priors for: {extra}")
        # else:
        #     print("\n--- [7] Estimation priors verified against model parameters ---")


        # # --- [8] Run Estimation ---
        # if NUMPYRO_AVAILABLE and sim_obs_for_filter is not None:
        #     print("\n--- [8] Running Bayesian Estimation ---")
        #     est_key = random.PRNGKey(789)
        #     mcmc_config = {
        #         'num_warmup': 1000,   # Keep moderate for testing
        #         'num_samples': 1000,  # Keep moderate for testing
        #         'num_chains': 2,     # Use 2 chains for basic convergence checks
        #         'target_accept_prob': 0.90 # High acceptance target can help with tricky posteriors
        #     }
        #     # Use the same H, init_x, init_P as the standalone filter check
        #     H_obs_est = H_obs_filt
        #     init_x_est = init_x_filt
        #     init_P_est = init_P_filt

        #     try:
        #         est_start_time = time.time()
        #         mcmc_results = model.estimate(
        #             ys=sim_obs_for_filter,        # Use the simulated data
        #             H_obs=H_obs_est,
        #             init_x_mean=init_x_est,
        #             init_P_cov=init_P_est,
        #             priors=estimation_priors,
        #             mcmc_params=mcmc_config,
        #             rng_key=est_key,
        #             verbose_solver=False, # Keep verbose=False unless debugging solver itself
        #             init_param_values=None # <<< Explicitly use default initialization strategy
        #             #init_param_values=param_values # <<< OR uncomment to try initializing at true values
        #         )
        #         est_end_time = time.time()
        #         print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")

        #         # --- [9] Analyze Estimation Results ---
        #         print("\n--- [9] Estimation Summary ---")
        #         mcmc_results.print_summary()
        #         posterior_samples = mcmc_results.get_samples() # Store for potential later use

        #         # Optional: Plot trace plots for quick visual inspection
        #         # try:
        #         #     import arviz as az
        #         #     az_data = az.from_numpyro(mcmc_results)
        #         #     az.plot_trace(az_data, var_names=list(estimation_priors.keys()))
        #         #     plt.show(block=False)
        #         # except ImportError:
        #         #     print("Install arviz (`pip install arviz`) to see trace plots.")
        #         # except Exception as e_trace:
        #         #     print(f"Could not generate trace plots: {e_trace}")

        #     except Exception as e_est:
        #          est_end_time = time.time()
        #          print(f"\n--- Estimation FAILED ({est_end_time - est_start_time:.2f} seconds) ---")
        #          print(f"An error occurred during estimation: {e_est}")
        #          import traceback
        #          traceback.print_exc()
        #          # Ensure posterior_samples is None if estimation fails
        #          posterior_samples = None

        # elif not NUMPYRO_AVAILABLE:
        #      print("\n--- Skipping Estimation (numpyro not available) ---")
        # else: # sim_obs_for_filter is None
        #      print("\n--- Skipping Estimation (Simulation Failed) ---")


        # --- [10] Example: Run Filter/Smoother with POSTERIOR MEAN parameters ---
        # # This section only runs if estimation was successful
        # if NUMPYRO_AVAILABLE and posterior_samples is not None:
        #     print("\n--- [10] Running Filter/Smoother with Posterior Mean Parameters ---")
        #     post_mean_start_time = time.time()
        #     try:
        #          # Calculate posterior means
        #          posterior_mean_params = {k: jnp.mean(v) for k, v in posterior_samples.items()}
        #          print(" Posterior Mean Parameter Values:")
        #          for k, v in posterior_mean_params.items():
        #               print(f"  {k}: {v:.4f}")

        #          # Use the same filter settings as the standalone check
        #          H_obs_post = H_obs_filt
        #          init_x_post = init_x_filt
        #          init_P_post = init_P_filt
        #          post_key = random.PRNGKey(999)
        #          num_draws_post = 100 # Or 0 if not needed

        #          kalman_results_post = model.run_kalman(
        #              param_dict=posterior_mean_params,
        #              ys=sim_obs_for_filter, # Use original simulated data
        #              H_obs=H_obs_post,
        #              init_x_mean=init_x_post,
        #              init_P_cov=init_P_post,
        #              smoother_key=post_key,
        #              num_sim_smoother_draws=num_draws_post
        #          )
        #          post_mean_end_time = time.time()
        #          print(f"Kalman operations with posterior mean complete ({post_mean_end_time - post_mean_start_time:.2f} seconds).")

        #          # --- Plotting Smoothed States (Posterior Mean vs True) ---
        #          if sim_states is not None and 'rts_smoothed_states' in kalman_results_post:
        #               print("\nPlotting True Simulation vs. Smoothed (Posterior Mean)...")
        #               # (Plotting code structure similar to step [6], maybe compare true vs RTS post-mean)
        #               # ... plotting code using kalman_results_post ...
        #               print("Posterior mean smoothed state plotting skipped for brevity.")
        #               # plt.show(block=True) # Optional: block final plot
        #          else:
        #              print("Could not plot posterior mean smoothed states.")


        #     except Exception as e_post:
        #         post_mean_end_time = time.time()
        #         print(f"\nError running Kalman filter with posterior mean ({post_mean_end_time - post_mean_start_time:.2f} seconds): {e_post}")
        #         import traceback
        #         traceback.print_exc()
        # elif posterior_samples is None:
        #      print("\n--- Skipping Post-Estimation Analysis (Estimation Failed or Skipped) ---")


        # end_time_script = time.time()
        # print(f"\n--- Script finished. Total time: {end_time_script - start_time_script:.2f} seconds ---")

    # --- Error Handling ---
    except FileNotFoundError as e: print(f"\nFatal Error: Model file not found.\n{e}")
    except ValueError as e: print(f"\nFatal Error: A value error occurred.\n{e}"); import traceback; traceback.print_exc()
    except RuntimeError as e: print(f"\nFatal Error: A runtime error occurred.\n{e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nFatal Error: An unexpected error occurred.\n{e}"); import traceback; traceback.print_exc()

# --- END OF FILE main_test_file_estimation.py ---