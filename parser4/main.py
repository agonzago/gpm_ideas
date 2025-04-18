#!/usr/bin/env python3
"""
Simple main function to generate IRFs with base and augmented solvers.
Uses a single observation mapping dictionary that contains:
  - The observable variable names,
  - The trend specification ('rw' for random walk, etc.), and
  - The model variable (cycle) to which it links.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import our model solver classes
from augmented_statespace import SimpleModelSolver, AugmentedStateSpace
from dynare_parser import DynareParser


def main():
    # Change working directory to script directory
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)
    
    # Load Dynare model and generate required files
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
        return

    # Create base model solver
    print("Creating base model solver...")
    out_folder = "model_files"
    json_file = os.path.join(out_folder, "model_json.json")
    jacobian_file = os.path.join(out_folder, "jacobian_matrices.py")
    structure_file = os.path.join(out_folder, "model_structure.py")
    solver = SimpleModelSolver(json_file, jacobian_file, structure_file)
    
    # Solve the model
    print("Solving model...")
    solver.solve_model()
    
    if not solver.is_solved:
        print("Error: Model solution failed.")
        return

    # Plot IRFs using the base model solver
    solver.plot_irf('SHK_RS', shock_size=1,
                    variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI',
                                       'RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP'],
                    periods=40)
    
    solver.plot_irf('SHK_L_GDP_GAP', shock_size=1,
                    variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI',
                                       'RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP'],
                    periods=40)
    
    solver.plot_irf('SHK_DLA_CPI', shock_size=1,
                    variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI',
                                       'RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP'],
                    periods=40)
    
    shock_size = 1
    impulse = np.zeros(len(solver.shock_names))
    impulse[0] = shock_size
    x_sim = solver.simulate_state_space(impulse, periods=40)
    solver.plot_simulation(x_sim,
                           variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI',
                                              'RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP'])
    
    # Define a single observation mapping dictionary that contains all information:
    observation_map = {
        "RS_OBS": { "trend": "rw", "model_var": "RS" },
        "DLA_CPI_OBS": { "trend": "cm", "model_var": "DLA_CPI" },
        "L_GDP_OBS": { "trend": "rw", "model_var": "L_GDP_GAP" }  # constant mean option
    }
    
    # Create an AugmentedStateSpace instance using the observation mapping.
    # The AugmentedStateSpace class should derive its trend specs and observable list from this mapping.
    
    aug_model = AugmentedStateSpace(solver, obs_mapping=observation_map)
    
    # Plot an IRF for the augmented model.
    # Here, shock index 0 corresponds to the first (base) shock.
    
    
    aug_model.plot_irf(3, variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI',
                                            'RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP', 
                                            'RS_OBS', 'DLA_CPI_OBS','L_GDP_OBS'], periods=40)

    # Compare the base model IRF (cycle only) with the augmented IRF (excluding trend parts)
    # for the same base shock.
    # aug_model.compare_with_base_model(0, variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI',
    #                                         'RS', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP', 
    #                                         'RS_OBS', 'DLA_CPI_OBS','L_GDP_OBS'], periods=40)


if __name__ == "__main__":
    main()