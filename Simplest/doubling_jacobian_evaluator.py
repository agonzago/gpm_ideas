import numpy as np

def evaluate_doubling_jacobians(theta):
    """
    Evaluates Jacobian matrices for the doubling algorithm
    
    For the model structure: 0 = A E_t[y_{t+1}] + B y_t + C y_{t-1} + D ε_t
    
    Args:
        theta: List or array of parameter values in the order of:
            ['b1', 'b4', 'a1', 'a2', 'g1', 'g2', 'g3', 'rho_DLA_CPI', 'rho_L_GDP_GAP', 'rho_rs', 'rho_rs2']
        
    Returns:
        A_plus: Matrix for future variables (coefficient on t+1 variables)
        A_zero: Matrix for current variables (coefficient on t variables)
        A_minus: Matrix for past variables (coefficient on t-1 variables)
        shock_impact: Matrix for shock impacts (n_vars x n_shocks)
        state_indices: Indices of state variables
        control_indices: Indices of control variables
    """
    # Unpack parameters from theta
    b1 = theta[0]
    b4 = theta[1]
    a1 = theta[2]
    a2 = theta[3]
    g1 = theta[4]
    g2 = theta[5]
    g3 = theta[6]
    rho_DLA_CPI = theta[7]
    rho_L_GDP_GAP = theta[8]
    rho_rs = theta[9]
    rho_rs2 = theta[10]

    n_vars = 17
    n_states = 7
    n_shocks = 3
    A_plus = np.zeros((n_vars, n_vars))
    A_zero = np.zeros((n_vars, n_vars))
    A_minus = np.zeros((n_vars, n_states))
    shock_impact = np.zeros((n_vars, n_shocks))

    # A_plus matrix elements (future variables)
    A_plus[0, 9] = 1 - b1
    A_plus[0, 12] = -b4
    A_plus[1, 8] = 1 - a1
    A_plus[2, 8] = 1 - g1
    A_plus[3, 8] = -1
    A_plus[7, 1] = -1
    A_plus[8, 3] = -1
    A_plus[9, 5] = -1
    A_plus[10, 0] = -1
    A_plus[11, 8] = 1
    A_plus[12, 16] = 1
    A_plus[13, 14] = 1
    A_plus[14, 4] = -1
    A_plus[15, 6] = -1
    A_plus[16, 2] = -1

    # A_zero matrix elements (current variables)
    A_zero[0, 1] = b1
    A_zero[0, 9] = -1
    A_zero[0, 10] = 1
    A_zero[1, 0] = a1
    A_zero[1, 7] = 1
    A_zero[1, 8] = -1
    A_zero[1, 9] = a2
    A_zero[2, 2] = g1
    A_zero[2, 9] = g3*(1 - g1)
    A_zero[2, 11] = 1
    A_zero[2, 13] = -1
    A_zero[2, 15] = g2*(1 - g1)
    A_zero[3, 12] = -1
    A_zero[3, 13] = 1
    A_zero[4, 3] = rho_L_GDP_GAP
    A_zero[4, 10] = -1
    A_zero[5, 5] = rho_DLA_CPI
    A_zero[5, 7] = -1
    A_zero[6, 4] = rho_rs
    A_zero[6, 6] = rho_rs2
    A_zero[6, 11] = -1
    A_zero[7, 9] = 1
    A_zero[8, 10] = 1
    A_zero[9, 7] = 1
    A_zero[10, 8] = 1
    A_zero[11, 16] = -1
    A_zero[12, 14] = -1
    A_zero[13, 15] = -1
    A_zero[14, 11] = 1
    A_zero[15, 4] = 1
    A_zero[16, 13] = 1

    # A_minus matrix elements (past variables)

    # shock_impact matrix elements (shock impacts)
    shock_impact[4, 0] = 1
    shock_impact[5, 1] = 1
    shock_impact[6, 2] = 1

    # Indices of state and control variables
    state_indices = [0, 1, 2, 3, 4, 5, 6]
    control_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    return A_plus, A_zero, A_minus, shock_impact, state_indices, control_indices