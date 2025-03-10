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

    a = np.zeros((17, 17))
    b = np.zeros((17, 17))
    c = np.zeros((17, 3))

    # A matrix elements
    a[0, 7] = b4
    a[0, 15] = b1 - 1
    a[1, 14] = a1 - 1
    a[2, 14] = g1 - 1
    a[3, 14] = 1
    a[7, 1] = 1
    a[8, 5] = 1
    a[9, 0] = 1
    a[10, 14] = -1
    a[11, 11] = -1
    a[12, 13] = -1
    a[13, 4] = 1
    a[14, 3] = 1
    a[15, 6] = 1
    a[16, 2] = 1

    # B matrix elements
    b[0, 1] = b1
    b[0, 9] = 1
    b[0, 15] = -1
    b[1, 0] = a1
    b[1, 10] = 1
    b[1, 14] = -1
    b[1, 15] = a2
    b[2, 3] = g1
    b[2, 8] = g2*(1 - g1)
    b[2, 12] = 1
    b[2, 15] = g3*(1 - g1)
    b[2, 16] = -1
    b[3, 7] = -1
    b[3, 16] = 1
    b[4, 5] = rho_L_GDP_GAP
    b[4, 9] = -1
    b[5, 4] = rho_DLA_CPI
    b[5, 10] = -1
    b[6, 2] = rho_rs2
    b[6, 6] = rho_rs
    b[6, 12] = -1
    b[7, 15] = 1
    b[8, 9] = 1
    b[9, 14] = 1
    b[10, 11] = -1
    b[11, 13] = -1
    b[12, 8] = -1
    b[13, 10] = 1
    b[14, 16] = 1
    b[15, 12] = 1

    # C matrix elements
    c[4, 0] = 1
    c[5, 1] = 1
    c[6, 2] = 1

    return a, b, c