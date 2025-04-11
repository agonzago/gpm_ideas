import numpy as np

def evaluate_jacobians(theta):
    """
    Computes the Jacobian matrices A, B, C for the model.
    
    Args:
        theta: Array of parameter values in the order specified by the model.
    
    Returns:
        A, B, C: Jacobian matrices
    """
    # Unpack parameters
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

    # Initialize matrices
    A = np.zeros((13, 13))
    B = np.zeros((13, 13))
    C = np.zeros((13, 3))

    # A matrix elements (∂equations/∂x_p)
    A[0, 4] = 1 - b1
    A[0, 7] = -b4
    A[1, 5] = 1 - a1
    A[2, 5] = 1 - g1
    A[2, 12] = g2*(1 - g1)
    A[3, 5] = -1
    A[4, 8] = -1
    A[5, 9] = -1
    A[6, 10] = -1
    A[7, 0] = -1
    A[8, 1] = -1
    A[9, 5] = 1
    A[10, 11] = 1
    A[11, 2] = -1
    A[12, 3] = -1

    # B matrix elements (-∂equations/∂x)
    B[0, 0] = -b1
    B[0, 4] = 1
    B[0, 8] = -1
    B[1, 1] = -a1
    B[1, 4] = -a2
    B[1, 5] = 1
    B[1, 9] = -1
    B[2, 2] = -g1
    B[2, 4] = -g3*(1 - g1)
    B[2, 6] = 1
    B[2, 10] = -1
    B[3, 6] = -1
    B[3, 7] = 1
    B[4, 8] = -rho_L_GDP_GAP
    B[5, 9] = -rho_DLA_CPI
    B[6, 3] = -rho_rs2
    B[6, 10] = -rho_rs
    B[7, 4] = -1
    B[8, 5] = -1
    B[9, 11] = 1
    B[10, 12] = 1
    B[11, 6] = -1
    B[12, 10] = -1

    # C matrix elements (-∂equations/∂shocks)
    C[4, 0] = -1
    C[5, 1] = -1
    C[6, 2] = -1

    return A, B, C