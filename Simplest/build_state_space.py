# gpm/model/state_space.py
import numpy as np

def build_state_space(F, P, R):
    """
    Build state space matrices for the DSGE model
    
    Args:
        F: Decision rule (u(t) = F*k(t))
        P: Law of motion (k(t) = P*k(t-1))
        R: Shock impact matrix on state variables
    
    Returns:
        Phi: State transition matrix for full state vector s(t) = [k(t); u(t)]
        R_ss: Shock impact matrix for full state vector
    """
    nk = P.shape[0]  # Number of state variables
    nu = F.shape[0]  # Number of control variables
    n_total = nk + nu  # Total variables
    n_shocks = R.shape[1]  # Number of shocks
    
    # Build state space transition matrix
    # [k(t)]   = [P     0] [k(t-1)] + [R ] [eps(t)]
    # [u(t)]     [F*P   0] [u(t-1)]   [FR]
    Phi = np.zeros((n_total, n_total))
    Phi[:nk, :nk] = P
    Phi[nk:, :nk] = F @ P
    
    # Build shock impact matrix
    R_ss = np.zeros((n_total, n_shocks))
    R_ss[:nk, :] = R
    R_ss[nk:, :] = F @ R
    
    return Phi, R_ss