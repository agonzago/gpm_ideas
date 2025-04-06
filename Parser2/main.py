# # ----- main.py -----

# if __name__ == "__main__":
#     import os
#     import sys
#     # Make sure the other scripts are importable (e.g., in the same directory or PYTHONPATH)
#     from parser_gpm import DynareParser
#     from model_solver import ModelSolver
#     from augmented_state_space import AugmentedStateSpace

#     output_dir = "model_files_gpm"  # Use a distinct directory
#     os.makedirs(output_dir, exist_ok=True)
#     dynare_file = "qpm_simpl1.dyn" # Make sure this file exists

#     # --- Step 1: Parse Model (Run once initially) ---
#     print("--- Parsing Dynare File ---")
#     try:
#         DynareParser.parse_and_generate_files(dynare_file, output_dir)
#         print(f"Model files generated in {output_dir}")
#     except Exception as e:
#         print(f"Error during parsing: {e}")
#         sys.exit(1)

#     # --- Step 2: Solve Base Model ---
#     print("\n--- Solving Base Model ---")
#     solver = ModelSolver(output_dir)
#     # Load initial parameters from the parsed model
#     initial_params = [solver.model_json['param_values'][p] for p in solver.model_json['parameters']]
#     base_ss = solver.solve(initial_params)

#     if base_ss is None:
#         print("Failed to solve the base model.")
#         sys.exit(1)


#     print("Base Model Solved. State space dimensions:")
#     print(f"  A: {base_ss['A'].shape}, B: {base_ss['B'].shape}, C: {base_ss['C'].shape}")
#     # Optional: Print eigenvalues
#     print(f"  Eigenvalues (abs): {[abs(e) for e in base_ss['eig']]}")

#     # --- Step 3: Define Observation Specs ---
#     observed_vars = ['L_GDP_GAP', 'DLA_CPI', 'RS'] # Example observables
#     trend_specs = {
#         'L_GDP_GAP': 'sd',  # Second difference trend for GDP Gap
#         'DLA_CPI': 'rw',   # Random walk for inflation
#         'RS': 'const'     # Constant mean (no stochastic trend) for interest rate
#         # Add other observed vars if they exist in model, e.g. 'RR_GAP': 'const'
#     }
#     # Filter trend_specs to only include variables that are actually observed
#     filtered_trend_specs = {k: v for k, v in trend_specs.items() if k in observed_vars}


#     # --- Step 4: Create Augmented Model ---
#     print("\n--- Building Augmented State Space ---")
#     aug_ss = AugmentedStateSpace(base_ss, observed_vars, filtered_trend_specs)
#     print("Augmented Model Built. Augmented dimensions:")
#     print(f"  A_aug: {aug_ss.augmented['A'].shape}, B_aug: {aug_ss.augmented['B'].shape}")
#     print(f"  C_aug: {aug_ss.augmented['C'].shape}, H: {aug_ss.augmented['H'].shape}")
#     print(f"  Augmented States: {aug_ss.augmented['n_states']}")
#     print(f"  Augmented Shocks: {aug_ss.augmented['n_shocks']}")
#     print(f"  Observed Variables: {aug_ss.augmented['n_observed']}")
#     print(f"  Augmented State Labels: {aug_ss.augmented['state_labels']}")
#     print(f"  Augmented Shock Labels: {aug_ss.augmented['shock_labels']}")


#     # --- Step 5: Analyze (IRFs) ---
#     print("\n--- Generating IRFs from Augmented Model ---")

#     # IRF to an original model shock
#     shock_to_plot_orig = 'SHK_RS' # Example original shock
#     irf_orig_shock = aug_ss.impulse_response(shock_to_plot_orig, shock_size=0.1, periods=40)
#     if irf_orig_shock is not None:
#         print(f"\nIRF to {shock_to_plot_orig} (Observed Vars):\n", irf_orig_shock.head())
#         solver.plot_irf(irf_orig_shock, observed_vars, title_suffix="Augmented Model")

#     # IRF to a trend shock (e.g., level shock for GDP gap)
#     shock_to_plot_trend = 'e_level_L_GDP_GAP' # Example trend shock
#     irf_trend_shock = aug_ss.impulse_response(shock_to_plot_trend, shock_size=0.1, periods=40)
#     if irf_trend_shock is not None:
#         print(f"\nIRF to {shock_to_plot_trend} (Observed Vars):\n", irf_trend_shock.head())
#         solver.plot_irf(irf_trend_shock, observed_vars, title_suffix="Augmented Model")

#     # --- Step 6: Parameter Update Example ---
#     print("\n--- Parameter Update Example ---")
#     # Simulate changing one parameter (e.g., Taylor rule coefficient)
#     new_params = initial_params.copy()
#     try:
#         # Find index of a parameter, e.g., 'gma_rs_1' if it exists
#         param_name_to_change = 'gma_rs_1' # CHANGE if needed
#         param_idx = solver.model_json['parameters'].index(param_name_to_change)
#         new_params[param_idx] = 1.6 # Change value
#         print(f"Updating parameter '{param_name_to_change}' to {new_params[param_idx]}")

#         new_base_ss = solver.solve(new_params)
#         if new_base_ss:
#             aug_ss.update_parameters(new_base_ss)
#             print("Augmented state space updated with new parameters.")
#             # Verify the A block changed in aug_ss.augmented['A']
#             # print("Updated A_aug block:\n", aug_ss.augmented['A'][:5, :5])
#             # Ready for Kalman filter with new parameters
#             kalman_A, kalman_C, kalman_Q = aug_ss.get_kalman_matrices()
#             print(f"Kalman matrices ready: A={kalman_A.shape}, C={kalman_C.shape}, Q={kalman_Q.shape}")

#     except ValueError:
#         print(f"Parameter '{param_name_to_change}' not found for update example.")
#     except Exception as e:
#         print(f"Error during parameter update: {e}")


#     print("\n--- Script Finished ---")


# ----- main.py -----
if __name__ == "__main__":
    import os
    import sys # Import sys for exit
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Make sure the other scripts are importable
    try:
        from parser_gpm import DynareParser
        from model_solver import ModelSolver
        from augmented_state_space import AugmentedStateSpace
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Ensure parser_gpm.py, model_solver.py, and augmented_state_space.py are in the same directory or Python path.")
        sys.exit(1)


    output_dir = "model_files_gpm"
    os.makedirs(output_dir, exist_ok=True)
    dynare_file = "qpm_simpl1.dyn"

    # --- Step 1: Parse Model (Run once initially) ---
    print("--- Parsing Dynare File ---")
    try:
        # Check if files exist to avoid re-parsing if not needed
        required_files = ["model.json", "jacobian_evaluator.py", "model_structure.py"]
        if not all(os.path.exists(os.path.join(output_dir, f)) for f in required_files):
            print("Generating model files...")
            DynareParser.parse_and_generate_files(dynare_file, output_dir)
            print(f"Model files generated in {output_dir}")
        else:
             print("Model files already exist.")
    except Exception as e:
        print(f"Error during parsing: {e}")
        sys.exit(1)

    # --- Step 2: Solve Base Model ---
    print("\n--- Solving Base Model ---")
    solver = ModelSolver(output_dir)
    # Load initial parameters from the parsed model
    initial_params = [solver.model_json['param_values'][p] for p in solver.model_json['parameters']]
    base_ss = solver.solve(initial_params)

    if base_ss is None:
        print("Failed to solve the base model.")
        sys.exit(1)


    print("Base Model Solved. State space dimensions:")
    print(f"  A: {base_ss['A'].shape}, B: {base_ss['B'].shape}, C: {base_ss['C'].shape}")
    # Optional: Print eigenvalues
    # print(f"  Eigenvalues (abs): {[abs(e) for e in base_ss['eig']]}")

        # --- Step 2b: Analyze Base Model IRFs (for comparison) ---
    print("\n--- Generating IRFs from Base Model (for comparison) ---")
    print("--- Refactored Code ---")
    # Print f and p from the solved base_ss
    f_refactored = base_ss['f']
    p_refactored = base_ss['A'] # Since A = p
    print("f matrix (first 5x5):\n", f_refactored[:5, :5])
    print("p matrix (first 5x5):\n", p_refactored[:5, :5])
    print("Norm of f:", np.linalg.norm(f_refactored))
    print("Norm of p:", np.linalg.norm(p_refactored))

    # Print B matrix from base_ss
    B_refactored = base_ss['B']
    print("B matrix (first 5x5):\n", B_refactored[:5, :5])
    print("Norm of B_refactored:", np.linalg.norm(B_refactored))

    base_shock_to_plot = 'SHK_RS'
    base_shock_size = 1.0

    # Calculate and print x0 from B_refactored
    try:
        shock_idx_refactored = base_ss['labels']['shock_labels'].index(base_shock_to_plot)
        x0_refactored = B_refactored[:, shock_idx_refactored] * base_shock_size
        print(f"x0 for {base_shock_to_plot}:\n", x0_refactored)
        print(f"Index of non-zero element in x0_refactored: {np.argmax(np.abs(x0_refactored))}")

        base_variables_to_plot = [
            "RR_GAP", "RS", "DLA_CPI", "L_GDP_GAP",
            "RES_RS_lag", "RES_L_GDP_GAP_lag", "RES_DLA_CPI_lag"
        ]
        valid_base_vars_to_plot = [v for v in base_variables_to_plot if v in base_ss['labels']['observable_labels']]

        # Use the ModelSolver's IRF method
        base_irf_df = solver.impulse_response(base_ss, base_shock_to_plot, shock_size=base_shock_size, periods=40)

        if base_irf_df is not None:
            print(f"\nBase Model IRF to {base_shock_to_plot} (Selected Vars):\n", base_irf_df[valid_base_vars_to_plot].head())
            solver.plot_irf(base_irf_df, valid_base_vars_to_plot, title_suffix="Base Model (Refactored Code)")
        else:
            print(f"Could not generate base IRF for {base_shock_to_plot}")

    except ValueError:
         print(f"Shock {base_shock_to_plot} not found in refactored shock_labels: {base_ss['labels']['shock_labels']}")
    except Exception as e:
         print(f"Error during refactored base IRF generation: {e}")

    # --- Step 3: Define Observation Specs ---
    observed_vars = ['L_GDP_GAP', 'DLA_CPI', 'RS'] # Example observables
    trend_specs = {
        'L_GDP_GAP': 'sd',
        'DLA_CPI': 'rw',
        'RS': 'const'
    }
    filtered_trend_specs = {k: v for k, v in trend_specs.items() if k in observed_vars}


    # --- Step 4: Create Augmented Model ---
    print("\n--- Building Augmented State Space ---")
    aug_ss = AugmentedStateSpace(base_ss, observed_vars, filtered_trend_specs)
    print("Augmented Model Built. Augmented dimensions:")
    print(f"  A_aug: {aug_ss.augmented['A'].shape}, B_aug: {aug_ss.augmented['B'].shape}")
    print(f"  C_aug: {aug_ss.augmented['C'].shape}, H: {aug_ss.augmented['H'].shape}")
    # print(f"  Augmented State Labels: {aug_ss.augmented['state_labels']}") # Optional details
    # print(f"  Augmented Shock Labels: {aug_ss.augmented['shock_labels']}")


    # --- Step 5: Analyze Augmented Model (IRFs) ---
    print("\n--- Generating IRFs from Augmented Model ---")

    # IRF to an original model shock
    shock_to_plot_orig_aug = 'SHK_RS'
    # *** Adjust shock size if needed - your plot peak was ~0.2, maybe use 0.1? Or keep 1.0? ***
    aug_shock_size = 1.0 # Let's try 0.1 to match the scale of your first plot
    irf_orig_shock_aug = aug_ss.impulse_response(shock_to_plot_orig_aug, shock_size=aug_shock_size, periods=40)
    if irf_orig_shock_aug is not None:
        print(f"\nAugmented IRF to {shock_to_plot_orig_aug} (Observed Vars):\n", irf_orig_shock_aug.head())
        solver.plot_irf(irf_orig_shock_aug, observed_vars, title_suffix="Augmented Model") # Use solver's plot

    # IRF to a trend shock
    shock_to_plot_trend = 'e_level_L_GDP_GAP'
    irf_trend_shock = aug_ss.impulse_response(shock_to_plot_trend, shock_size=0.1, periods=40)
    if irf_trend_shock is not None:
        print(f"\nAugmented IRF to {shock_to_plot_trend} (Observed Vars):\n", irf_trend_shock.head())
        solver.plot_irf(irf_trend_shock, observed_vars, title_suffix="Augmented Model") # Use solver's plot

    # --- Step 6: Parameter Update Example ---
    # ... (parameter update code remains the same) ...
    print("\n--- Parameter Update Example ---")
    new_params = initial_params.copy()
    try:
        param_name_to_change = 'gma_rs_1'
        param_idx = solver.model_json['parameters'].index(param_name_to_change)
        new_params[param_idx] = 1.6
        print(f"Updating parameter '{param_name_to_change}' to {new_params[param_idx]}")

        new_base_ss = solver.solve(new_params)
        if new_base_ss:
            aug_ss.update_parameters(new_base_ss)
            print("Augmented state space updated with new parameters.")
            kalman_A, kalman_C, kalman_Q = aug_ss.get_kalman_matrices()
            print(f"Kalman matrices ready: A={kalman_A.shape}, C={kalman_C.shape}, Q={kalman_Q.shape}")

    except ValueError:
        print(f"Parameter '{param_name_to_change}' not found for update example.")
    except Exception as e:
        print(f"Error during parameter update: {e}")


    print("\n--- Script Finished ---")