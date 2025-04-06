#!/usr/bin/env python3

"""
Improved main script for state space modeling with Klein solver and trend augmentation.
This script implements the correct state space representations as described in the
state_space_specification document.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("\n===== State Space Model Solution & IRF Generation =====\n")
    
    # --- Setup and Configuration ---
    output_dir = "model_files_gpm"
    os.makedirs(output_dir, exist_ok=True)
    dynare_file = "qpm_simpl1.dyn"
    
    # Import required modules
    try:
        from parser_gpm import DynareParser
        from model_solver import ModelSolver
        from augmented_state_space import AugmentedStateSpace
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)
    
    # --- Step 1: Parse and Generate Model Files ---
    print("\n=== Step 1: Parsing Dynare File ===")
    required_files = ["model.json", "jacobian_evaluator.py", "model_structure.py"]
    if not all(os.path.exists(os.path.join(output_dir, f)) for f in required_files):
        print("Generating model files...")
        try:
            DynareParser.parse_and_generate_files(dynare_file, output_dir)
        except Exception as e:
            print(f"Error during parsing: {e}")
            sys.exit(1)
    else:
        print("Model files already exist. Skipping generation.")
    
    # --- Step 2: Solve Base Model (Klein Solution) ---
    print("\n=== Step 2: Solving Base Model with Klein Method ===")
    solver = ModelSolver(output_dir)
    
    # Set parameters for the model
    param_dict = {p: v for p, v in zip(solver.model_json['parameters'], 
                                        solver.model_json['param_values'].values())}
    
    # Adjust specific parameters if needed (e.g., rho_rs=0 to simplify testing)
    #param_dict['rho_rs'] = 0.0
    #param_dict['rho_rs2'] = 0.0
    print(f"Using parameters: rho_rs={param_dict['rho_rs']}, rho_rs2={param_dict['rho_rs2']}")
    
    # Get ordered parameter values
    parameters = solver.model_json['parameters']
    current_params = [param_dict[p] for p in parameters]
    
    # Solve the model
    klein_solution = solver.solve(current_params)
    if klein_solution is None:
        print("ERROR: Failed to solve the base model.")
        sys.exit(1)
    
    print("Klein solution successfully obtained.")
    
    # --- Step 3: Create Contemporaneous State Space ---
    print("\n=== Step 3: Constructing Contemporaneous State Space ===")
    contemp_ss = solver.get_contemporaneous_state_space(klein_solution)
    if contemp_ss is None:
        print("ERROR: Failed to create contemporaneous state space.")
        sys.exit(1)
    
    # Print state space dimensions and structure
    print(f"State space dimensions:")
    print(f"  States: {contemp_ss['n_states']}")
    print(f"  Observables: {contemp_ss['n_observables']}")
    print(f"  Shocks: {contemp_ss['n_shocks']}")
    
    # --- Step 4: Generate IRFs for Contemporaneous Model ---
    print("\n=== Step 4: Generating Base Model IRFs ===")
    shock_to_test = 'SHK_RS'  # Interest rate shock
    shock_size = 1.0
    
    # Choose variables to plot
    base_vars_to_plot = [
        'RR_GAP', 'RS', 'DLA_CPI', 'L_GDP_GAP',  # Key macro variables
        'RES_RS'  # Shock process
    ]
    
    # Filter to variables available in this system
    valid_vars = [v for v in base_vars_to_plot if v in contemp_ss['observable_labels']]
    
    # Calculate impulse responses
    print(f"Calculating IRF for {shock_to_test} shock...")
    irf_df = solver.impulse_response(contemp_ss, shock_to_test, shock_size, periods=40)
    
    if irf_df is not None:
        # Display basic results
        print("\nImpulse Response Results (first 5 periods):")
        print(irf_df[valid_vars].head())
        
        # Plot results
        print("\nPlotting IRFs...")
        solver.plot_irf(irf_df, valid_vars, 
                        title_suffix=f"Base Model (rho_rs={param_dict['rho_rs']})")
    else:
        print("ERROR: Failed to generate impulse responses.")
    
    # --- Step 5: Create Augmented State Space with Trends ---
    print("\n=== Step 5: Building Augmented State Space with Trends ===")
    
    # Define which variables to observe in the augmented model
    # (typically a subset of all potential observables)
    observed_vars = ['RS', 'DLA_CPI', 'L_GDP_GAP', 'RES_RS']
    
    # Define trend specifications for observed variables
    # Options: 'rw' (random walk), 'sd' (second difference), 'const' (constant)
    trend_specs = {
        'RS': 'const',       # Constant trend for interest rate
        'DLA_CPI': 'rw',     # Random walk trend for inflation
        'L_GDP_GAP': 'sd'    # Second difference trend for output gap
    }
    
    # Create augmented state space
    try:
        aug_ss = AugmentedStateSpace(contemp_ss, observed_vars, trend_specs)
        print("Augmented state space successfully created.")
        
        # Print augmented dimensions
        print(f"Augmented state space dimensions:")
        print(f"  Base states: {aug_ss.augmented['n_base_states']}")
        print(f"  Trend states: {aug_ss.augmented['n_trend_states']}")
        print(f"  Total states: {aug_ss.augmented['n_states']}")
        print(f"  Observed variables: {aug_ss.augmented['n_observed']}")
        print(f"  Total shocks: {aug_ss.augmented['n_shocks']}")
        
    except Exception as e:
        print(f"ERROR: Failed to create augmented state space: {e}")
        import traceback
        traceback.print_exc()
        aug_ss = None
    
    # --- Step 6: Generate IRFs for Augmented Model ---
    if aug_ss is not None:
        print("\n=== Step 6: Generating Augmented Model IRFs ===")
        
        # Test both base model shock and trend shock
        shocks_to_test = [
            'SHK_RS',              # Base model shock
            'e_level_DLA_CPI'      # Trend shock for inflation
        ]
        
        for shock in shocks_to_test:
            print(f"\nCalculating augmented IRF for {shock} shock...")
            aug_irf = aug_ss.impulse_response(shock, shock_size=1.0, periods=40)
            
            if aug_irf is not None:
                # Display results
                print(f"Augmented IRF Results for {shock} (first 5 periods):")
                print(aug_irf.head())
                
                # Plot results
                plt.figure(figsize=(12, 8))
                for var in aug_irf.columns:
                    plt.plot(aug_irf.index, aug_irf[var], label=var)
                
                plt.title(f"Augmented IRF: {shock} Shock")
                plt.xlabel("Periods")
                plt.ylabel("Response")
                plt.legend()
                plt.grid(True)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                plt.show()
            else:
                print(f"ERROR: Failed to generate augmented IRF for {shock}.")
    
    # --- Step 7: Generate Kalman Matrices for Estimation ---
    if aug_ss is not None:
        print("\n=== Step 7: Extracting Kalman Filter Matrices ===")
        try:
            A, C, Q = aug_ss.get_kalman_matrices()
            print("Kalman filter matrices successfully extracted.")
            print(f"  A shape: {A.shape}")
            print(f"  C shape: {C.shape}")
            print(f"  Q shape: {Q.shape}")
        except Exception as e:
            print(f"ERROR: Failed to extract Kalman matrices: {e}")
    
    print("\n===== State Space Modeling Complete =====")

if __name__ == "__main__":
    main()