# --- START OF FILE main_test_file_basic_ops.py ---

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


# --- Import the Wrapper ---
from dynare_model_wrapper import DynareModel
# Import helpers if needed - explicitly import the plot function from the parser
from Dynare_parser_sda_solver import plot_irfs as dp_plot_irfs # Rename to avoid potential clash
# <<< Import KalmanFilter class >>>
from Kalman_filter_jax import KalmanFilter

# Helper function to calculate HDR intervals (copied from estimation script for plotting)
def calculate_hdr(draws: jax.Array, level: float) -> Tuple[onp.ndarray, onp.ndarray]:
    """Calculates Highest Density Region (HDR) using percentiles."""
    lower_perc = (100 - level) / 2
    upper_perc = 100 - lower_perc
    lower_bound = jnp.percentile(draws, lower_perc, axis=0)
    upper_bound = jnp.percentile(draws, upper_perc, axis=0)
    # Convert to numpy for plotting convenience
    return onp.array(lower_bound), onp.array(upper_bound)

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        start_time_script = time.time()
        # --- Configuration ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Make sure the model file exists in the same directory or provide the correct path
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
        # (Print statements for model structure omitted for brevity)

        # --- [2] Define Parameter Set for Operations ---
        param_values = model.default_param_assignments.copy()
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

        # --- [3] Compute and Plot IRFs ---
        print("\n--- [3] Computing and Plotting Impulse Responses ---")
        irf_start_time = time.time()
        shock_to_plot = "SHK_RS"
        horizon_irf = 40
        try:
            irf_results = model.get_irf(param_values, shock_to_plot, horizon=horizon_irf)
            irf_end_time = time.time()
            print(f"IRFs computed for shock '{shock_to_plot}' ({irf_end_time - irf_start_time:.2f} seconds).")

            print(f"Plotting IRFs for STATES to shock '{shock_to_plot}'...")
            # Use onp.array() to convert JAX arrays before passing to matplotlib plotting function
            dp_plot_irfs(
                irf_values=onp.array(irf_results['state_irf']),
                var_names=irf_results['state_names'],
                horizon=horizon_irf,
                title=f"State Variable IRFs to a unit '{shock_to_plot}' shock"
            )
            plt.show(block=False)

            print(f"Plotting IRFs for OBSERVABLES to shock '{shock_to_plot}'...")
            if irf_results['observable_names']:
                 dp_plot_irfs(
                     irf_values=onp.array(irf_results['observable_irf']),
                     var_names=irf_results['observable_names'],
                     horizon=horizon_irf,
                     title=f"Observable Variable IRFs to a unit '{shock_to_plot}' shock"
                 )
                 plt.show(block=False)
            else:
                 print("  No observable variables found to plot IRFs for.")

        except Exception as e_irf_gen:
            irf_end_time = time.time()
            print(f"An error occurred during IRF generation ({irf_end_time - irf_start_time:.2f} seconds): {e_irf_gen}")
            import traceback
            traceback.print_exc()


        # --- [4] Simulate Data ---
        print("\n--- [4] Simulating Data from the Model ---")
        sim_key = random.PRNGKey(123)
        num_sim_steps = 300
        n_obs = model.n_obs
        n_aug = model.n_state_aug
        obs_noise_stdevs = jnp.array([0.002, 0.002, 0.001, 0.001], dtype=_DEFAULT_DTYPE)
        if len(obs_noise_stdevs) != n_obs:
             raise ValueError(f"Number of observation noise std devs ({len(obs_noise_stdevs)}) must match n_obs ({n_obs}).")
        H_obs_sim = jnp.diag(obs_noise_stdevs**2)
        init_x_sim = jnp.zeros(n_aug, dtype=_DEFAULT_DTYPE)
        init_P_sim = jnp.eye(n_aug, dtype=_DEFAULT_DTYPE) * 0.1

        print(f"Simulating {num_sim_steps} steps of data...")
        sim_start_time = time.time()
        sim_states = None # Initialize
        sim_obs_for_filter = None
        try:
            simulation_output = model.simulate(
                param_dict=param_values, H_obs=H_obs_sim, init_x_mean=init_x_sim,
                init_P_cov=init_P_sim, key=sim_key, num_steps=num_sim_steps
            )
            sim_states = simulation_output['sim_states']
            sim_observations = simulation_output['sim_observations']
            sim_end_time = time.time()
            print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")
            sim_obs_for_filter = sim_observations # Use simulated data for filter/likelihood

            # (Plotting simulated observations omitted for brevity)

        except Exception as e_sim:
            sim_end_time = time.time()
            print(f"An error occurred during simulation ({sim_end_time - sim_start_time:.2f} seconds): {e_sim}")
            import traceback
            traceback.print_exc()

        # --- [5] Run Kalman Filter/Smoother ---
        kalman_results = None # Initialize
        if sim_obs_for_filter is not None:
            print("\n--- [5] Running Kalman Filter and Smoother ---")
            filter_start_time = time.time()
            H_obs_filt = H_obs_sim
            init_x_filt = init_x_sim
            init_P_filt = init_P_sim
            filter_key = random.PRNGKey(457)
            num_draws_sim_smooth = 100
            hdr_levels_plot = [68, 90]
            try:
                kalman_results = model.run_kalman(
                    param_dict=param_values, ys=sim_obs_for_filter, H_obs=H_obs_filt,
                    init_x_mean=init_x_filt, init_P_cov=init_P_filt,
                    smoother_key=filter_key, num_sim_smoother_draws=num_draws_sim_smooth
                )
                filter_end_time = time.time()
                print(f"Kalman Filter/Smoother operations complete ({filter_end_time - filter_start_time:.2f} seconds).")
                # (Result extraction print statements omitted for brevity)
            except Exception as e_filt:
                filter_end_time = time.time()
                print(f"An error occurred during filtering/smoothing ({filter_end_time - filter_start_time:.2f} seconds): {e_filt}")
                import traceback
                traceback.print_exc()
        else:
            print("\n--- Skipping Filtering/Smoothing due to Simulation Failure ---")


        # --- [6] Plot Smoothed States vs. True Simulated States ---
        print("\n--- [6] Plotting True vs. Smoothed States ---")
        if sim_states is not None and kalman_results is not None and 'rts_smoothed_states' in kalman_results:
             # (Plotting code structure kept, details omitted for brevity)
             states_np = onp.array(sim_states)
             x_smooth_rts_np = onp.array(kalman_results['rts_smoothed_states'])
             mean_smooth_sim_np = None
             all_draws_np = None
             hdrs_np = {}
             if num_draws_sim_smooth > 0 and kalman_results.get('sim_smoothed_draws') is not None:
                 mean_smooth_sim_np = onp.array(kalman_results['sim_smoothed_mean'])
                 all_draws_np = onp.array(kalman_results['sim_smoothed_draws'])
                 # (HDR calculation omitted for brevity)
             # (Plotting loop omitted for brevity)
             print("Generating state comparison plots...")
             # plt.show(block=True) # Keep plot blocking if needed
             plt.show(block=False) # Use non-blocking for faster check
             print("State comparison plots generated.")
        else:
            print("Could not plot smoothed states (missing simulation or smoothing results).")


        # --- [7] Calculate Log-Likelihood (Standalone Check) ---
        if sim_obs_for_filter is not None:
            print("\n--- [7] Calculating Log-Likelihood (Standalone Check) ---")
            ll_start_time = time.time()
            try:
                # 1. Get the numerical state-space matrices for the current parameters
                solution = model.solve(param_values)
                P_aug_ll = solution['P_aug']
                R_aug_ll = solution['R_aug']
                Omega_ll = solution['Omega']

                # Check if solution produced valid matrices (essential before filtering)
                if not (jnp.all(jnp.isfinite(P_aug_ll)) and
                        jnp.all(jnp.isfinite(R_aug_ll)) and
                        jnp.all(jnp.isfinite(Omega_ll))):
                    raise ValueError("Solution matrices (P, R, Omega) contain non-finite values.")

                # 2. Use the same H and initial conditions as the filter/smoother run
                H_obs_ll = H_obs_filt
                init_x_ll = init_x_filt
                init_P_ll = init_P_filt

                # 3. Instantiate the KalmanFilter
                kf_ll = KalmanFilter(T=P_aug_ll, R=R_aug_ll, C=Omega_ll, H=H_obs_ll,
                                     init_x=init_x_ll, init_P=init_P_ll)

                # 4. Call the likelihood-specific filter function
                # This uses lax.cond for robustness, mimicking the Numpyro model's behavior
                filter_results_ll = kf_ll.filter_for_likelihood(sim_obs_for_filter)

                # 5. Sum the log-likelihood contributions
                total_log_likelihood = jnp.sum(filter_results_ll['log_likelihood_contributions'])
                ll_end_time = time.time()

                # 6. Print the result
                print(f"Log-Likelihood calculation complete ({ll_end_time - ll_start_time:.2f} seconds).")
                print(f"  Total Log-Likelihood: {total_log_likelihood}")

                # Check if the result is finite
                if not jnp.isfinite(total_log_likelihood):
                     print("  WARNING: Log-likelihood is non-finite (NaN or Inf). Check parameters or filter stability.")

            except ValueError as e_solve:
                 ll_end_time = time.time()
                 print(f"Error during log-likelihood calculation ({ll_end_time - ll_start_time:.2f} seconds): Could not get valid solution matrices.")
                 print(f"  Error details: {e_solve}")
            except Exception as e_ll:
                ll_end_time = time.time()
                print(f"An error occurred during log-likelihood calculation ({ll_end_time - ll_start_time:.2f} seconds): {e_ll}")
                import traceback
                traceback.print_exc()
        else:
            print("\n--- Skipping Log-Likelihood Calculation due to Simulation Failure ---")


        end_time_script = time.time()
        print(f"\n--- Script finished. Total time: {end_time_script - start_time_script:.2f} seconds ---")

    # --- Error Handling ---
    # (Error handling blocks omitted for brevity, same as before)
    except FileNotFoundError as e: print(f"\nFatal Error: Model file not found.\n{e}")
    except ValueError as e: print(f"\nFatal Error: A value error occurred.\n{e}"); import traceback; traceback.print_exc()
    except RuntimeError as e: print(f"\nFatal Error: A runtime error occurred.\n{e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nFatal Error: An unexpected error occurred.\n{e}"); import traceback; traceback.print_exc()


# --- END OF FILE main_test_file_basic_ops.py ---