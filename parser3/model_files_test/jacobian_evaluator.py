import numpy as np

def evaluate_jacobians(theta):
    """
    Evaluates Jacobian matrices for the Klein method and VAR representation
    
    Args:
        theta: List or array of parameter values in the order of:
            ['b1', 'b4', 'a1', 'a2', 'g1', 'g2', 'g3', 'rho_DLA_CPI', 'rho_L_GDP_GAP', 'rho_rs', 'rho_rs2']
        
    Returns:
        a: Matrix ∂F/∂X_p (Jacobian with respect to future variables)
        b: Matrix -∂F/∂X (negative Jacobian with respect to current variables)
        c: Matrix -∂F/∂Z (negative Jacobian with respect to exogenous processes)
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

    a = np.zeros((15, 12))
    b = np.zeros((15, 12))
    c = np.zeros((15, 3))

    # A matrix elements
    a[0, 7] = b1 - 1
    a[0, 9] = b4
    a[1, 8] = a1 - 1
    a[2, 8] = g1 - 1
    a[3, 8] = 1
    a[4, 6] = 1
    a[5, 11] = 1
    a[6, 5] = 1
    a[7, 2] = 1
    a[8, 0] = 1
    a[9, 8] = -1
    a[12, 1] = 1
    a[13, 4] = 1
    a[14, 3] = 1

    # B matrix elements
    b[0, 2] = b1
    b[0, 6] = 1
    b[0, 7] = -1
    b[1, 0] = a1
    b[1, 7] = a2
    b[1, 8] = -1
    b[1, 11] = 1
    b[2, 1] = g1
    b[2, 5] = 1
    b[2, 7] = g3*(1 - g1)
    b[2, 10] = -1
    b[3, 9] = -1
    b[3, 10] = 1
    b[4, 6] = rho_L_GDP_GAP
    b[5, 11] = rho_DLA_CPI
    b[6, 3] = rho_rs2
    b[6, 5] = rho_rs
    b[7, 7] = 1
    b[8, 8] = 1
    b[12, 10] = 1
    b[13, 5] = 1
    b[14, 4] = 1

    # C matrix elements
    c[4, 0] = 1
    c[5, 1] = 1
    c[6, 2] = 1

    return a, b, c