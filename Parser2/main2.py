# ----- Inside main.py -----
if __name__ == "__main__":

    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import os

    # Get the directory of the current file
    file_directory = os.path.dirname(os.path.abspath(__file__))

    # Set the working directory to the file's directory
    os.chdir(file_directory)

    # --- Imports and Setup ---
    try:
        from parser_gpm import DynareParser
        from model_solver import ModelSolver
        from augmented_state_space import AugmentedStateSpace # Keep this for later comparison
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

    output_dir = "model_files_gpm"
    os.makedirs(output_dir, exist_ok=True)
    dynare_file = "qpm_simpl1.dyn"

    # --- Step 1: Parse Model (using correct parser) ---
    print("--- Parsing Dynare File ---")
    try:
        required_files = ["model.json", "jacobian_evaluator.py", "model_structure.py"]
        if not all(os.path.exists(os.path.join(output_dir, f)) for f in required_files):
            print("Generating model files using TARGET ORDER logic...")
            # Ensure parser_gpm.py used here has the TARGET ORDER logic forced in generation methods
            DynareParser.parse_and_generate_files(dynare_file, output_dir)
        else:
            print("Model files already exist.")
    except Exception as e:
        print(f"Error during parsing: {e}")
        sys.exit(1)

    # --- Step 2: Solve Base Model (Get Klein Solution) ---
    print("\n--- Solving Base Model (Klein Solution) ---")
    solver = ModelSolver(output_dir)
    initial_params = [solver.model_json['param_values'][p] for p in solver.model_json['parameters']]

    # Set parameters (e.g., the zero rho case)
    param_dict = dict(zip(solver.model_json['parameters'], initial_params))
    param_dict['rho_rs'] = 0.0
    param_dict['rho_rs2'] = 0.0
    current_params = [param_dict[p] for p in solver.model_json['parameters']]
    print(f"Using parameters with rho_rs={param_dict['rho_rs']}, rho_rs2={param_dict['rho_rs2']}")


    klein_solution = solver.solve(current_params) # Gets P, F etc. in TARGET order

    if klein_solution is None:
         print("Failed to solve the base model.")
         sys.exit(1)

    # --- Step 3: Get Contemporaneous State Space ---
    contemp_ss = solver.get_contemporaneous_state_space(klein_solution)

    if contemp_ss is None:
         print("Failed to create contemporaneous state space.")
         sys.exit(1)

    # --- Step 4: Analyze Contemporaneous Model IRFs ---
    print("\n--- Generating IRFs from Contemporaneous State Space ---")
    shock_to_plot = 'SHK_RS'
    shock_size = 1.0 # Use shock size 1.0 to match Dynare

    # Define variables to plot (should match Dynare plot variables)
    contemp_variables_to_plot = [
         'RR_GAP', 'RS', 'DLA_CPI', 'L_GDP_GAP', # Example controls/states
         'RES_RS_lag', 'RES_L_GDP_GAP_lag', 'RES_DLA_CPI_lag', # Example lagged exo states (if needed)
         'RES_RS' # Current exo state - this is now an observable y = [c;k;z]
        ]
    # Filter to variables available in this system's observables
    valid_contemp_vars = [v for v in contemp_variables_to_plot if v in contemp_ss['observable_labels']]

    # Use the standard impulse_response method, now passing the contemp_ss
    contemp_irf_df = solver.impulse_response(contemp_ss, shock_to_plot, shock_size=shock_size, periods=40)

    if contemp_irf_df is not None:
        print(f"\nContemporaneous Model IRF to {shock_to_plot} (Selected Vars):\n", contemp_irf_df[valid_contemp_vars].head())
        # Print specific variable to compare with Dynare
        if 'RS' in contemp_irf_df.columns:
             print("\nRS column from Contemporaneous IRF (SHK_RS, size=1.0):\n", contemp_irf_df['RS'])
        if 'RES_RS' in contemp_irf_df.columns:
             print("\nRES_RS column from Contemporaneous IRF (SHK_RS, size=1.0):\n", contemp_irf_df['RES_RS'])

        solver.plot_irf(contemp_irf_df, valid_contemp_vars, title_suffix=f"Contemporaneous SS (rho_rs={param_dict['rho_rs']})")
    else:
        print(f"Could not generate contemporaneous IRF for {shock_to_plot}")


    # --- Optional: Step 5: Augment the Contemporaneous System (Advanced) ---
    # If you wanted trends ON TOP of this contemporaneous system, you'd need
    # a new AugmentedContemporaneousStateSpace class that takes contemp_ss
    # as input and builds A_aug, B_aug, C_aug, D_aug, H appropriately.
    # The state definition might change slightly depending on how you add trends.
    # For now, let's focus on getting the base contemporaneous IRF correct.

    print("\n--- Script Finished ---")