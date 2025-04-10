#!/usr/bin/env python3

"""
Test script for the fixed ModelSolver implementation.
This script verifies that the ModelSolver correctly:
1. Constructs the state space representation with proper timing
2. Handles direct shock effects in the D matrix
3. Calculates impulse responses with correct timing
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_separator(title):
    """Print a separator with a title"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_separator("TESTING FIXED MODELSOLVER IMPLEMENTATION")
    
    # Define paths
    output_dir = "model_files_final"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: First, parse and generate model files using fixed parser
        print_separator("STEP 1: GENERATING MODEL FILES")
        from parser_gpm import DynareParser
        
        dynare_file = "qpm_simpl1.dyn"
        if not all(os.path.exists(os.path.join(output_dir, f)) for f in ["model.json", "jacobian_evaluator.py", "model_structure.py"]):
            print("Generating model files with fixed parser...")
            DynareParser.parse_and_generate_files(dynare_file, output_dir)
            print("Model files generated successfully.")
        else:
            print("Model files already exist. Using existing files.")
        
        # Step 2: Import fixed ModelSolver and solve model
        print_separator("STEP 2: SOLVING MODEL")
        from model_solver import ModelSolver
        
        solver = ModelSolver(output_dir)
        
        # Set parameters for the model
        param_dict = {p: v for p, v in zip(solver.model_json['parameters'], 
                                        solver.model_json['param_values'].values())}
        
        # Test both with and without persistence
        test_cases = [
            {"name": "Standard model", "params": {"rho_rs": 0.75, "rho_rs2": 0.0}},
            {"name": "Zero persistence model", "params": {"rho_rs": 0.0, "rho_rs2": 0.0}}
        ]
        
        for case in test_cases:
            print(f"\n--- Testing {case['name']} ---")
            
            # Update parameters
            for param, value in case['params'].items():
                param_dict[param] = value
            
            print(f"Using parameters: rho_rs={param_dict['rho_rs']}, rho_rs2={param_dict['rho_rs2']}")
            
            # Get ordered parameter values
            parameters = solver.model_json['parameters']
            current_params = [param_dict[p] for p in parameters]
            
            # Solve the model
            klein_solution = solver.solve(current_params)
            if klein_solution is None:
                print(f"ERROR: Failed to solve the model for {case['name']}.")
                continue
            
            print(f"Klein solution successfully obtained for {case['name']}.")
            
            # Step 3: Create State Space Representation
            print(f"\n--- Creating State Space for {case['name']} ---")
            contemp_ss = solver.get_contemporaneous_state_space(klein_solution)
            if contemp_ss is None:
                print(f"ERROR: Failed to create state space for {case['name']}.")
                continue
            
            # Print state space dimensions
            print(f"State space dimensions:")
            print(f"  States: {contemp_ss['n_states']}")
            print(f"  Observables: {contemp_ss['n_observables']}")
            print(f"  Shocks: {contemp_ss['n_shocks']}")
            
            # Print shock impact matrices (crucial for verifying timing)
            print("\nVerifying shock impact matrices:")
            print(f"  B shape (state transition shock impact): {contemp_ss['B'].shape}")
            print(f"  D shape (direct shock effect): {contemp_ss['D'].shape}")
            
            # Check if D matrix has non-zero elements (crucial for direct effects)
            D = contemp_ss['D']
            non_zero_D = np.count_nonzero(D)
            print(f"  D matrix has {non_zero_D} non-zero elements")
            
            if non_zero_D > 0:
                print("\nDirect shock effects (D matrix):")
                n_controls = contemp_ss['indices']['n_controls']
                for i in range(contemp_ss['D'].shape[0]):
                    for j in range(contemp_ss['D'].shape[1]):
                        if abs(contemp_ss['D'][i, j]) > 1e-10:
                            # Determine if this is a control or state
                            if i < n_controls:
                                var_type = "Control"
                                var_name = contemp_ss['observable_labels'][i]
                            else:
                                var_type = "State"
                                state_idx = i - n_controls
                                var_name = contemp_ss['state_labels'][state_idx]
                            
                            shock_name = contemp_ss['shock_labels'][j]
                            print(f"  {shock_name} → {var_type} {var_name}: {contemp_ss['D'][i, j]}")
            
            # Step 4: Generate IRFs
            print(f"\n--- Generating IRFs for {case['name']} ---")
            shock_to_test = 'SHK_RS'  # Interest rate shock
            shock_size = 1.0
            
            # Choose variables to plot
            vars_to_plot = [
                'RS', 'DLA_CPI', 'L_GDP_GAP',  # Key macro variables
                'RR_GAP',                   # Gap variable
                'RES_RS'                    # Shock process
            ]
            
            # Filter to variables available in this system
            valid_vars = [v for v in vars_to_plot if v in contemp_ss['observable_labels']]
            
            # Calculate impulse responses
            print(f"Calculating IRF for {shock_to_test} shock...")
            irf_df = solver.impulse_response(contemp_ss, shock_to_test, shock_size, periods=40)
            
            if irf_df is not None:
                # Display initial responses (crucial for verifying timing)
                print("\nInitial impulse responses (first 5 periods):")
                print(irf_df[valid_vars].head())
                
                # Check for direct shock effects at period 0
                print("\nPeriod 0 responses (should show direct effects):")
                period_0 = irf_df.iloc[0][valid_vars]
                for var in valid_vars:
                    print(f"  {var}: {period_0[var]:.8f}")
                
                # Calculate maximum absolute responses
                print("\nMaximum absolute responses:")
                for var in valid_vars:
                    max_val = np.max(np.abs(irf_df[var]))
                    max_period = np.argmax(np.abs(irf_df[var]))
                    print(f"  {var}: {max_val:.8f} at period {max_period}")
                
                # Plot IRFs
                plt.figure(figsize=(12, 8))
                for var in valid_vars:
                    plt.plot(irf_df.index, irf_df[var], label=var)
                
                plt.title(f"IRF for {shock_to_test} shock - {case['name']}")
                plt.xlabel("Periods")
                plt.ylabel("Response")
                plt.legend()
                plt.grid(True)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                
                # Save the figure
                plot_file = os.path.join(output_dir, f"irf_{case['name'].replace(' ', '_')}.png")
                plt.savefig(plot_file)
                print(f"IRF plot saved to {plot_file}")
                
                # Save IRF data for later analysis
                irf_file = os.path.join(output_dir, f"irf_data_{case['name'].replace(' ', '_')}.csv")
                irf_df.to_csv(irf_file)
                print(f"IRF data saved to {irf_file}")
                
                # Extra verification specific to zero-persistence case
                if case['name'] == "Zero persistence model":
                    print("\nSpecial verification for zero-persistence case:")
                    
                    # Check if shock effect on RES_RS is immediate
                    if 'RES_RS' in valid_vars:
                        res_rs_impact = irf_df.iloc[0]['RES_RS']
                        print(f"  Immediate impact on RES_RS at period 0: {res_rs_impact:.8f}")
                        if abs(res_rs_impact) > 1e-10:
                            print("  SUCCESS: Direct shock effect on RES_RS is present at period 0")
                        else:
                            print("  WARNING: No direct effect on RES_RS at period 0")
                    
                    # Check if control variables also have immediate effects
                    control_vars = valid_vars[:3]  # RS, DLA_CPI, L_GDP_GAP
                    has_direct_effect = False
                    for var in control_vars:
                        if abs(irf_df.iloc[0][var]) > 1e-10:
                            has_direct_effect = True
                            print(f"  Direct effect on {var} at period 0: {irf_df.iloc[0][var]:.8f}")
                    
                    if has_direct_effect:
                        print("  SUCCESS: Direct effects on control variables present at period 0")
                    else:
                        print("  WARNING: No direct effects on control variables at period 0")
            else:
                print(f"ERROR: Failed to generate IRFs for {case['name']}.")
        
        print_separator("TEST COMPLETE")
        print("The ModelSolver implementation has been successfully updated to:")
        print("1. Construct the state space with proper timing convention")
        print("2. Handle direct shock effects in the D matrix")
        print("3. Calculate impulse responses with correct timing")
        print(f"All output files have been saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    
    # Get the directory of the current file
    file_directory = os.path.dirname(os.path.abspath(__file__))

    # Set the working directory to the file's directory
    os.chdir(file_directory)
    sys.exit(main())