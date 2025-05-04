import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random
import time
from typing import Dict, List, Tuple, Optional, Union, Any


# --- Force CPU Execution (Optional) ---
print("Attempting to force JAX to use CPU...")
try:
    jax.config.update("jax_platforms", "cpu")
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
    try:
        # Configure numpyro based on JAX devices
        num_devices_to_use = jax.local_device_count()
        numpyro.set_host_device_count(num_devices_to_use)
        print(
            f"Numpyro configured to use {num_devices_to_use} host devices (CPUs/GPUs detected by JAX)."
        )
    except Exception as e_np_config:
        print(f"Warning: Could not configure numpyro device count: {e_np_config}")
except ImportError:
    NUMPYRO_AVAILABLE = False
    print("Warning: numpyro not found. Estimation disabled.")


# --- Import the Wrapper ---
# Assumes dynare_model_wrapper.py has been updated to use TFP internally
from dynare_model_wrapper import DynareModel
from Dynare_parser_sda_solver import plot_irfs as dp_plot_irfs  # Keep plotter


# Helper function to calculate HDR intervals (still useful if plotting posteriors)
def calculate_hdr(draws: jax.Array, level: float) -> Tuple[onp.ndarray, onp.ndarray]:
    lower_perc = (100 - level) / 2
    upper_perc = 100 - lower_perc
    lower_bound = jnp.percentile(draws, lower_perc, axis=0)
    upper_bound = jnp.percentile(draws, upper_perc, axis=0)
    return onp.array(lower_bound), onp.array(upper_bound)


# Define machine epsilon based on JAX config (needed for standalone likelihood check)
_MACHINE_EPSILON = (
    jnp.finfo(jnp.float64).eps if jax.config.jax_enable_x64 else jnp.finfo(jnp.float32).eps
)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize variables
    sim_states = None
    sim_observations = None
    sim_obs_for_filter = None
    kalman_results = None
    mcmc_results = None
    posterior_samples = None
    param_values = None

    try:
        start_time_script = time.time()
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
        param_values = model.default_param_assignments.copy()
        param_values.update(
            {
                "b1": 0.75,
                "b4": 0.65,
                "a1": 0.55,
                "a2": 0.12,
                "g1": 0.7,
                "g2": 0.3,
                "g3": 0.25,
                "rho_L_GDP_GAP": 0.8,
                "rho_DLA_CPI": 0.7,
                "rho_rs": 0.75,
                "rho_rs2": 0.05,
                "sigma_SHK_L_GDP_TREND": 0.01,
                "sigma_SHK_G_TREND": 0.03,
                "sigma_SHK_L_GDP_GAP": 0.10,
                "sigma_SHK_DLA_CPI": 0.05,
                "sigma_SHK_RS": 0.15,
                "sigma_SHP_PI_TREND": 0.01,
                "sigma_SHK_RR_TREND": 0.02,
                "sigma_SHK_RS_TREND": 0.01,
            }
        )
        missing_params = [p for p in model.all_param_names if p not in param_values]
        if missing_params:
            raise ValueError(f"Missing parameter values in param_values: {missing_params}")
        else:
            print("\n--- [2] Parameter set defined and verified ---")

        # --- [3] Compute and Plot IRFs (Optional Check) ---
        print("\n--- [3] Computing and Plotting Impulse Responses (Check) ---")
        print("Skipping IRF generation for faster estimation test.")  # IRF generation itself doesn't use KF

        # --- [4] Simulating Data from the Model ---
        print("\n--- [4] Simulating Data from the Model ---")
        sim_key = random.PRNGKey(123)
        num_sim_steps = 300
        n_obs = model.n_obs
        n_aug = model.n_state_aug
        obs_noise_stdevs = jnp.array([0.002, 0.002, 0.001, 0.001], dtype=_DEFAULT_DTYPE)
        if len(obs_noise_stdevs) != n_obs:
            raise ValueError(f"Obs noise std devs mismatch n_obs.")
        
        H_obs_sim = jnp.diag(obs_noise_stdevs**2)
        init_x_sim = jnp.zeros(n_aug, dtype=_DEFAULT_DTYPE)
        init_P_sim = jnp.eye(n_aug, dtype=_DEFAULT_DTYPE) * 0.1
        print(f"Simulating {num_sim_steps} steps of data...")
        sim_start_time = time.time()
        try:
            simulation_output = model.simulate(
                param_dict=param_values,
                H_obs=H_obs_sim,
                init_x_mean=init_x_sim,
                init_P_cov=init_P_sim,
                key=sim_key,
                num_steps=num_sim_steps,
            )

            
            sim_states = simulation_output["sim_states"]  # Might be None now
            sim_observations = simulation_output["sim_observations"]
            sim_end_time = time.time()
            if sim_observations is None:
                raise RuntimeError("Simulation failed to produce observations.")
            print(f"Simulation complete ({sim_end_time - sim_start_time:.2f} seconds).")
            sim_obs_for_filter = sim_observations
            if sim_states is None:
                print("  Note: Simulated states were not returned by the TFP-based simulation.")
        except Exception as e_sim:
            sim_end_time = time.time()
            print(f"Error during simulation ({sim_end_time - sim_start_time:.2f} seconds): {e_sim}")
            import traceback

            traceback.print_exc()
            sim_obs_for_filter = None

        # --- [5] Running Kalman Smoother (Check using TFP via run_kalman) ---
        if sim_obs_for_filter is not None and param_values is not None:
            print("\n--- [5] Running Kalman Smoother (Check using TFP) ---")
            filter_start_time = time.time()
            H_obs_filt = H_obs_sim
            init_x_filt = init_x_sim
            init_P_filt = init_P_sim
            try:
                kalman_results = model.run_kalman(
                    param_dict=param_values,
                    ys=sim_obs_for_filter,
                    H_obs=H_obs_filt,
                    init_x_mean=init_x_filt,
                    init_P_cov=init_P_filt,
                    # No smoother_key or num_draws needed for TFP RTS
                )
                filter_end_time = time.time()
                print(f"TFP Kalman Smoother check complete ({filter_end_time - filter_start_time:.2f} seconds).")
                # Check smoothed results
                if "rts_smoothed_states" not in kalman_results or kalman_results[
                    "rts_smoothed_states"
                ] is None:
                    print("  WARNING: Smoothed states missing from TFP results.")
                elif not jnp.all(jnp.isfinite(kalman_results["rts_smoothed_states"])):
                    print(f"  WARNING: Smoothed states contain non-finite values.")
                if "rts_smoothed_cov" not in kalman_results or kalman_results[
                    "rts_smoothed_cov"
                ] is None:
                    print("  WARNING: Smoothed covariances missing from TFP results.")
                elif not jnp.all(jnp.isfinite(kalman_results["rts_smoothed_cov"])):
                    print(f"  WARNING: Smoothed covariances contain non-finite values.")
            except Exception as e_filt:
                filter_end_time = time.time()
                print(
                    f"Error during standalone smoother check ({filter_end_time - filter_start_time:.2f} seconds): {e_filt}"
                )
                import traceback

                traceback.print_exc()
                kalman_results = None
        else:
            print("\n--- Skipping Standalone Smoother Check ---")

        # --- [6] Calculating Log-Likelihood (Standalone Check using TFP) ---
        if sim_obs_for_filter is not None and param_values is not None:
            print("\n--- [6] Calculating Log-Likelihood (Standalone Check using TFP) ---")
            ll_start_time = time.time()
            try:
                solution = model.solve(param_values)
                if not solution["solution_valid"]:
                    raise ValueError("Solution invalid before likelihood calc.")
                # Need P, R (for scale op), Omega, H, initial conditions
                P_aug = solution["P_aug"]
                R_aug = solution["R_aug"]  # TFP uses R directly in LinearOperator
                Omega_mat = solution["Omega"]
                H_obs_ll = H_obs_filt
                init_x_ll = init_x_filt
                init_P_ll = init_P_filt

                # Build TFP model directly for the check (using the helper method)
                lgssm_check = model._build_tfp_lgssm(
                    P_aug, R_aug, Omega_mat, H_obs_ll, init_x_ll, init_P_ll, num_timesteps=sim_obs_for_filter.shape[0]
                )

                total_log_likelihood = lgssm_check.log_prob(sim_obs_for_filter)

                ll_end_time = time.time()
                print(f"Log-Likelihood check complete ({ll_end_time - ll_start_time:.2f} seconds).")
                print(f"  Total Log-Likelihood (True Params, TFP): {total_log_likelihood:.4f}")
                if not jnp.isfinite(total_log_likelihood):
                    print("  WARNING: Log-likelihood is non-finite.")

            except Exception as e_ll:
                ll_end_time = time.time()
                print(
                    f"Error during standalone likelihood check ({ll_end_time - ll_start_time:.2f} seconds): {e_ll}"
                )
                import traceback

                traceback.print_exc()
        else:
            print("\n--- Skipping Standalone Log-Likelihood Check ---")

        # --- [7] Define Priors for ESTIMATION (Using SIMPLIFIED Normal Priors for now) ---
        # Switch back to original priors once initialization works
        print("\n--- [7] USING TEMPORARY NORMAL PRIORS FOR DEBUGGING ---")
        estimation_priors_debug = {}
        for pname, true_val in param_values.items():
            std_dev = 0.05
            if "sigma_" in pname:
                std_dev = 0.02
                mean_val = max(true_val, 1e-6)  # Avoid zero mean for sigmas
            else:
                mean_val = true_val
            estimation_priors_debug[pname] = ("Normal", [mean_val, std_dev])
            print(f"  Using Debug Prior for {pname}: Normal({mean_val:.4f}, {std_dev:.4f})")
        estimation_priors = estimation_priors_debug  # Use the debug priors
        # --- END TEMPORARY PRIORS ---

        # Verify priors cover all model parameters (essential check)
        prior_param_names = set(estimation_priors.keys())
        all_model_params = set(model.all_param_names)
        if prior_param_names != all_model_params:
            missing = all_model_params - prior_param_names
            extra = prior_param_names - all_model_params
            raise ValueError(
                f"FATAL: Prior/Model param mismatch. Missing: {missing}, Extra: {extra}"
            )
        else:
            print("--- Estimation priors (DEBUG) verified against model parameters ---")

        # --- [8] Run Estimation ---
        if NUMPYRO_AVAILABLE and sim_obs_for_filter is not None:
            print(
                "\n--- [8] Running Bayesian Estimation (using TFP, simplified priors) ---"
            )
            est_key = random.PRNGKey(789)
            n_devices = jax.local_device_count()
            print(f" JAX sees {n_devices} devices. Setting num_chains = {n_devices}")
            chains_to_run = n_devices
            mcmc_config = {
                "num_warmup": 1000,
                "num_samples": 1000,
                "num_chains": chains_to_run,
                "target_accept_prob": 0.90,
                "progress_bar": True,
            }
            print(f" MCMC Configuration: {mcmc_config}")
            H_obs_est = H_obs_filt
            init_x_est = init_x_filt
            init_P_est = init_P_filt
            try:
                est_start_time = time.time()
                # Use init_to_value with the known good parameters AND simplified priors
                print(
                    "Note: Using provided init_param_values (true values) AND simplified Normal priors for MCMC initialization."
                )
                mcmc_results = model.estimate(
                    ys=sim_obs_for_filter,
                    H_obs=H_obs_est,
                    init_x_mean=init_x_est,
                    init_P_cov=init_P_est,
                    priors=estimation_priors,  # Using simplified Normal priors
                    mcmc_params=mcmc_config,
                    rng_key=est_key,
                    verbose_solver=False,
                    init_param_values=param_values,  # Start at true values
                )
                est_end_time = time.time()
                print(f"--- Estimation Complete ({est_end_time - est_start_time:.2f} seconds) ---")

                # --- [9] Analyze Estimation Results ---
                print("\n--- [9] Estimation Summary ---")
                mcmc_results.print_summary()
                posterior_samples = mcmc_results.get_samples()

                # Plotting trace plots (Optional)
                try:
                    import arviz as az

                    print(" Generating trace plots...")
                    az_data = az.from_numpyro(mcmc_results)
                    az.plot_trace(az_data, var_names=list(estimation_priors.keys()))
                    plt.show(block=False)
                    print(" Trace plots generated.")
                except ImportError:
                    print(" Install arviz (`pip install arviz`) to see trace plots.")
                except Exception as e_trace:
                    print(f" Could not generate trace plots: {e_trace}")

            except Exception as e_est:
                est_end_time = time.time()
                print(
                    f"\n--- Estimation FAILED ({est_end_time - est_start_time:.2f} seconds) ---"
                )
                print(f"An error occurred during estimation: {e_est}")
                import traceback

                traceback.print_exc()
                posterior_samples = None
        elif not NUMPYRO_AVAILABLE:
            print("\n--- Skipping Estimation (numpyro not available) ---")
        else:  # sim_obs_for_filter is None
            print("\n--- Skipping Estimation (Simulation Failed) ---")

        # --- [10] Example: Run Smoother with POSTERIOR MEAN parameters ---
        if NUMPYRO_AVAILABLE and posterior_samples is not None:
            print(
                "\n--- [10] Example: Run Smoother with POSTERIOR MEAN parameters (using TFP) ---"
            )
            post_mean_start_time = time.time()
            try:
                posterior_mean_params = {k: jnp.mean(v) for k, v in posterior_samples.items()}
                print(" Posterior Mean Parameter Values:")
                for k, v in posterior_mean_params.items():
                    true_val_str = (
                        f"(True: {param_values.get(k, 'N/A'):.4f})" if k in param_values else ""
                    )
                    print(f"  {k}: {v:.4f} {true_val_str}")
                H_obs_post = H_obs_filt
                init_x_post = init_x_filt
                init_P_post = init_P_filt

                # Call run_kalman which gives RTS smoothed results from TFP
                kalman_results_post = model.run_kalman(
                    param_dict=posterior_mean_params,
                    ys=sim_obs_for_filter,
                    H_obs=H_obs_post,
                    init_x_mean=init_x_post,
                    init_P_cov=init_P_post,
                )
                post_mean_end_time = time.time()
                print(
                    f"TFP Kalman operations with posterior mean complete ({post_mean_end_time - post_mean_start_time:.2f} seconds)."
                )

                # --- Plotting Smoothed States (Posterior Mean vs True) ---
                if sim_states is not None and kalman_results_post.get("rts_smoothed_states") is not None:
                    print("\nPlotting True Simulation vs. Smoothed (Posterior Mean)...")
                    states_np = onp.array(sim_states)
                    x_smooth_rts_post_np = onp.array(kalman_results_post["rts_smoothed_states"])
                    # No simulation smoother results to plot HDRs for
                    time_axis = onp.arange(num_sim_steps)
                    state_names_plot = model.aug_state_vars_structure
                    n_states_to_plot = len(state_names_plot)
                    cols_smooth = 4
                    rows_smooth = (n_states_to_plot + cols_smooth - 1) // cols_smooth
                    fig_smooth, axes_smooth = plt.subplots(
                        rows_smooth, cols_smooth, figsize=(min(5 * cols_smooth, 20), 3 * rows_smooth), sharex=True
                    )
                    axes_smooth = axes_smooth.flatten()
                    for i in range(n_states_to_plot):
                        ax = axes_smooth[i]
                        state_name = state_names_plot[i]
                        ax.plot(time_axis, states_np[:, i], "g-", linewidth=1.5, label="True Sim", zorder=5)
                        ax.plot(
                            time_axis,
                            x_smooth_rts_post_np[:, i],
                            "b-.",
                            linewidth=1.0,
                            label="RTS (Post. Mean)",
                            zorder=6,
                        )
                        ax.set_title(f"{state_name}")
                        ax.grid(True, linestyle="--", alpha=0.6)
                        if i == 0:
                            ax.legend(fontsize=8)
                    for i in range(n_states_to_plot, len(axes_smooth)):
                        axes_smooth[i].set_visible(False)
                    fig_smooth.suptitle(
                        "State Variables: True vs. Smoothed (using Posterior Mean Params)", fontsize=16
                    )
                    fig_smooth.tight_layout(rect=[0, 0.03, 1, 0.96])
                    plt.show(block=True)  # Block final plot
                elif sim_states is None:
                    print(
                        "Cannot plot comparison: sim_states were not generated/returned (likely due to TFP simulate)."
                    )
                    print(
                        "Smoothed states (posterior mean) are available in kalman_results_post['rts_smoothed_states']"
                    )
                else:
                    print("Could not plot posterior mean smoothed states (missing TFP results).")

            except Exception as e_post:
                post_mean_end_time = time.time()
                print(
                    f"\nError running Kalman/plotting with posterior mean ({post_mean_end_time - post_mean_start_time:.2f} seconds): {e_post}"
                )
                import traceback

                traceback.print_exc()
        elif posterior_samples is None:
            print("\n--- Skipping Post-Estimation Analysis (Estimation Failed or Skipped) ---")

        end_time_script = time.time()
        print(f"\n--- Script finished. Total time: {end_time_script - start_time_script:.2f} seconds ---")

    except FileNotFoundError as e:
        print(f"\nFatal Error: Model file not found.\n{e}")
    except ValueError as e:
        print(f"\nFatal Error: Value error.\n{e}")
        import traceback

        traceback.print_exc()
    except RuntimeError as e:
        print(f"\nFatal Error: Runtime error.\n{e}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"\nFatal Error: Unexpected error.\n{e}")
        import traceback

        traceback.print_exc()

# --- END OF FILE main_test_file_estim.py ---