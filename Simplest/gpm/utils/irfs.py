# gpm/utils/irfs.py
import numpy as np
import matplotlib.pyplot as plt

def compute_irfs(Phi, R_ss, shock_index=None, periods=40, orig_vars=None, transformed_vars=None):
    """
    Compute impulse response functions.
    
    Args:
        Phi (ndarray): State transition matrix
        R_ss (ndarray): Shock impact matrix
        shock_index (int, optional): Index of the shock to compute IRFs for.
                                    If None, compute for all shocks.
        periods (int): Number of periods for the IRF
        orig_vars (list): Original variable names
        transformed_vars (list): Transformed variable names
    
    Returns:
        dict: Dictionary of IRFs keyed by shock and variable
    """
    n_vars = Phi.shape[0]
    n_shocks = R_ss.shape[1]
    
    # If shock_index is None, compute IRFs for all shocks
    if shock_index is None:
        shock_indices = range(n_shocks)
    else:
        shock_indices = [shock_index]
    
    # Create dictionary to store IRFs
    irfs = {}
    
    # For each shock
    for s_idx in shock_indices:
        # Initialize impulse vector
        x = np.zeros((n_vars, periods))
        
        # Initial impulse
        x[:, 0] = R_ss[:, s_idx]
        
        # Propagate through system
        for t in range(1, periods):
            x[:, t] = Phi @ x[:, t-1]
        
        # Store IRFs
        shock_name = f"shock_{s_idx}" if orig_vars is None else f"SHK_{orig_vars[s_idx]}"
        irfs[shock_name] = {}
        
        # Map back to original variables if provided
        for i, var in enumerate(transformed_vars or range(n_vars)):
            var_name = var if orig_vars is None else orig_vars[i % len(orig_vars)]
            irfs[shock_name][var_name] = x[i, :]
    
    return irfs

def plot_irfs(irfs, variables=None, shock_name=None, figsize=(12, 8)):
    """
    Plot impulse response functions.
    
    Args:
        irfs (dict): Dictionary of IRFs from compute_irfs
        variables (list, optional): List of variables to plot. If None, plot all.
        shock_name (str, optional): Name of the shock. If None, use the first shock.
        figsize (tuple): Figure size.
    """
    if not irfs:
        raise ValueError("No IRFs to plot")
    
    # If shock_name is None, use the first shock
    if shock_name is None:
        shock_name = next(iter(irfs.keys()))
    
    # Get shock IRFs
    shock_irfs = irfs.get(shock_name)
    if shock_irfs is None:
        raise ValueError(f"Shock {shock_name} not found in IRFs")
    
    # If variables is None, plot all variables
    if variables is None:
        variables = list(shock_irfs.keys())
    
    # Number of variables to plot
    n_vars = len(variables)
    
    # Create grid of subplots
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each variable
    for i, var in enumerate(variables):
        if var in shock_irfs:
            x = np.arange(len(shock_irfs[var]))
            axes[i].plot(x, shock_irfs[var])
            axes[i].set_title(var)
            axes[i].grid(True)
            axes[i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Turn off any unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"IRFs for {shock_name}")
    plt.tight_layout()
    
    return fig