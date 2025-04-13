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
from augmented_statespace import SimpleModelSolver, AugmentedStateSpace
from dynare_parser import DynareParser


def generate_irf(f, p, n=None, l=None, state_names=None, control_names=None, shock_names=None, 
                 shock_index=0, periods=40, variables_to_plot=None):
    """
    Generate impulse response functions from model solution matrices
    
    Parameters:
    -----------
    f : ndarray
        Control solution matrix (policy function)
    p : ndarray
        State transition matrix
    n : ndarray, optional
        Control response to shocks
    l : ndarray, optional
        State response to shocks
    state_names : list, optional
        Names of state variables
    control_names : list, optional
        Names of control variables
    shock_names : list, optional
        Names of shock variables
    shock_index : int, default=0
        Index of the shock to analyze
    periods : int, default=40
        Number of periods for IRF
    variables_to_plot : list or str, optional
        Variable names to include in plot (defaults to all)
        
    Returns:
    --------
    irf_data : ndarray
        IRF data for all variables
    var_names : list
        Names of all variables in order
    """
    # Determine dimensions
    n_states = p.shape[0]
    n_controls = f.shape[0] if f is not None else 0
    n_vars = n_states + n_controls
    
    # Create default variable names if not provided
    if state_names is None:
        state_names = [f"s{i+1}" for i in range(n_states)]
    if control_names is None:
        control_names = [f"c{i+1}" for i in range(n_controls)]
    if shock_names is None and (n is not None or l is not None):
        n_shocks = l.shape[1] if l is not None else n.shape[1] if n is not None else 0
        shock_names = [f"e{i+1}" for i in range(n_shocks)]
    
    # Combine names
    var_names = state_names + control_names
    
    # Initialize IRF arrays
    irf_states = np.zeros((periods + 1, n_states))
    irf_controls = np.zeros((periods + 1, n_controls))
    
    # Check if we have shock matrices
    if l is not None:
        n_shocks = l.shape[1]
        if shock_index >= n_shocks:
            raise ValueError(f"Shock index {shock_index} out of range (0-{n_shocks-1})")
        
        # Set initial state responses using L matrix
        irf_states[0, :] = l[:, shock_index]
    
    # Set initial control responses using N matrix if available
    if n is not None:
        irf_controls[0, :] = n[:, shock_index]
    
    # Compute IRF dynamics using state transition and control policy
    for t in range(periods):
        # State transition: s_{t+1} = P * s_t
        irf_states[t+1, :] = p @ irf_states[t, :]
        
        # Control solution: c_t = F * s_t (for t>0)
        if t+1 <= periods and f is not None:
            irf_controls[t+1, :] = f @ irf_states[t+1, :]
    
    # Combine state and control IRFs
    irf_data = np.zeros((periods + 1, n_vars))
    irf_data[:, :n_states] = irf_states
    irf_data[:, n_states:] = irf_controls
    
    # Plot IRFs if requested
    if variables_to_plot is not None:
        plot_irf(irf_data, var_names, shock_names[shock_index] if shock_names else f"shock_{shock_index+1}", 
                 variables_to_plot, periods)
    
    return irf_data, var_names

def plot_irf(irf_data, var_names, shock_name, variables_to_plot=None, periods=40):
    """
    Plot impulse response functions
    
    Parameters:
    -----------
    irf_data : ndarray
        IRF data for all variables
    var_names : list
        Names of all variables
    shock_name : str
        Name of the shock
    variables_to_plot : list or str, optional
        Which variables to include in the plot
    periods : int
        Number of periods in the IRF
    """
    # If variables_to_plot not specified, use all variables
    if variables_to_plot is None:
        variables_to_plot = var_names
    elif isinstance(variables_to_plot, str):
        variables_to_plot = [variables_to_plot]
    
    # Filter to include only variables that exist
    variables_to_plot = [v for v in variables_to_plot if v in var_names]
    
    # Create time axis
    time = np.arange(periods + 1)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    for var in variables_to_plot:
        if var in var_names:
            var_idx = var_names.index(var)
            plt.plot(time, irf_data[:, var_idx], label=var)
    
    plt.title(f"Impulse Response to {shock_name} Shock")
    plt.xlabel("Periods")
    plt.ylabel("Deviation from Steady State")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"irf_{shock_name}.png")
    plt.show()

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

    irf_data, var_names = generate_irf(
        f=solver.f,
        p=solver.p,
        n=solver.n,
        l=solver.l,
        state_names=solver.state_names,
        control_names=solver.control_names,
        shock_names=solver.shock_names,
        shock_index=0,  # First shock
        periods=40,
        variables_to_plot=['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI']     )

#['RES_RS', 'RES_L_GDP_GAP', 'RES_DLA_CPI', 'RES_RS_lag', 'RS_lag', 'L_GDP_GAP_lag', 'DLA_CPI_lag', 'RR_GAP', 'DLA_CPI_lead2', 'DLA_CPI', 'DLA_CPI_lead1', 'RS', 'L_GDP_GAP']

    # # Get model variables
    # observable_vars = solver.model_data.get('variables', [])[:10]  # Limit to first 10 for simplicity
    
    # # Create trend specifications - all random walks
    # trend_specs = {}
    # for var in observable_vars:
    #     trend_specs[var] = 'rw'  # Random walk for all variables
    
    # print(f"Using {len(observable_vars)} variables with random walk trends")
    
    # # Create augmented state space model
    # print("Creating augmented state space model...")
    # aug_model = AugmentedStateSpace(solver, trend_specs, observable_vars)
    
    # # Create output directory
    # os.makedirs("irfs", exist_ok=True)
    
    # # Generate IRFs for base model
    # shocks = solver.model_data.get('shocks', [])
    # print(f"Generating IRFs for {len(shocks)} economic shocks in base model...")
    
    # for i, shock in enumerate(shocks):
    #     print(f"  Shock {i}: {shock}")
    #     solver.plot_irf(i, periods=20)
    
    # # Generate IRFs for augmented model - economic shocks
    # print(f"Generating IRFs for economic shocks in augmented model...")
    
    # for i, shock in enumerate(shocks):
    #     print(f"  Shock {i}: {shock}")
    #     # Plot with cycle components only
    #     aug_model.plot_irf(i, periods=20, include_trend=False)
    #     # Plot with trend included
    #     aug_model.plot_irf(i, periods=20, include_trend=True)
    #     # Compare base and augmented models
    #     aug_model.compare_with_base_model(i, periods=20)
    
    # # Generate IRFs for trend shocks
    # n_base_shocks = len(shocks)
    # n_aug_shocks = aug_model.B_aug.shape[1]
    # n_trend_shocks = n_aug_shocks - n_base_shocks
    
    # print(f"Generating IRFs for {n_trend_shocks} trend shocks...")
    
    # for i in range(n_base_shocks, n_aug_shocks):
    #     print(f"  Trend shock {i - n_base_shocks}")
    #     aug_model.plot_irf(i, periods=20)
    
    # print("All IRFs generated successfully!")

if __name__ == "__main__":
    main()