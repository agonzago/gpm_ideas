#!/usr/bin/env python3
"""
Simple main function to generate IRFs with base and augmented solvers
Assumes all trends are random walks
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import our model solver classes
# Assuming they're in the current module or imported previously
from augmented_statespace import SimpleModelSolver, AugmentedStateSpace
from dynare_parser import DynareParser
def main():
    """
    Load qpm_simpl1 model files, solve model, and generate IRFs with both solvers
    """
    import os
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    dynare_file = "qpm_simpl1.dyn"
    dynare_parser = DynareParser(dynare_file)
    success = dynare_parser.parse()
    
    if success:
        print("Files generated:")
        print("  - clean_file.txt")
        print("  - clean_file_with_correct_timing.txt")
        print("  - clean_file_with_correct_timing_and_auxiliary_variables.txt")
        print("  - clean_file_with_auxiliary_variables_substituted.txt")
        print("  - model_json.json")
        print("  - jacobian_matrices.py")
        print("  - model_structure.py")
    else:
        print("Parsing failed.")
    

    # Create base model solver
    print("Creating base model solver...")
    out_folder = "model_files"
    json_file = os.path.join(out_folder, "model_json.json") 
    jacobian_file = os.path.join(out_folder,"jacobian_matrices.py")
    structure_file= os.path.join(out_folder,"model_structure.py")
    solver = SimpleModelSolver(json_file, jacobian_file, structure_file)
    
    # Solve the model
    print("Solving model...")
    solver.solve_model()
    
    if not solver.is_solved:
        print("Error: Model solution failed.")
        return
    
    # Get model variables
    observable_vars = solver.model_data.get('variables', [])[:10]  # Limit to first 10 for simplicity
    
    # Create trend specifications - all random walks
    trend_specs = {}
    for var in observable_vars:
        trend_specs[var] = 'rw'  # Random walk for all variables
    
    print(f"Using {len(observable_vars)} variables with random walk trends")
    
    # Create augmented state space model
    print("Creating augmented state space model...")
    aug_model = AugmentedStateSpace(solver, trend_specs, observable_vars)
    
    # Create output directory
    os.makedirs("irfs", exist_ok=True)
    
    # Generate IRFs for base model
    shocks = solver.model_data.get('shocks', [])
    print(f"Generating IRFs for {len(shocks)} economic shocks in base model...")
    
    for i, shock in enumerate(shocks):
        print(f"  Shock {i}: {shock}")
        solver.plot_irf(i, periods=20)
    
    # Generate IRFs for augmented model - economic shocks
    print(f"Generating IRFs for economic shocks in augmented model...")
    
    for i, shock in enumerate(shocks):
        print(f"  Shock {i}: {shock}")
        # Plot with cycle components only
        aug_model.plot_irf(i, periods=20, include_trend=False)
        # Plot with trend included
        aug_model.plot_irf(i, periods=20, include_trend=True)
        # Compare base and augmented models
        aug_model.compare_with_base_model(i, periods=20)
    
    # Generate IRFs for trend shocks
    n_base_shocks = len(shocks)
    n_aug_shocks = aug_model.B_aug.shape[1]
    n_trend_shocks = n_aug_shocks - n_base_shocks
    
    print(f"Generating IRFs for {n_trend_shocks} trend shocks...")
    
    for i in range(n_base_shocks, n_aug_shocks):
        print(f"  Trend shock {i - n_base_shocks}")
        aug_model.plot_irf(i, periods=20)
    
    print("All IRFs generated successfully!")

if __name__ == "__main__":
    main()