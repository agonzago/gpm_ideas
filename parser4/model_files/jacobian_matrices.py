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
    A[0, 8] = -b4
    A[0, 11] = 1 - b1
    A[1, 12] = 1 - a1
    A[2, 10] = g2*(1 - g1)
    A[2, 12] = 1 - g1
    A[3, 12] = -1
    A[4, 0] = -1
    A[5, 1] = -1
    A[6, 2] = -1
    A[7, 6] = -1
    A[8, 3] = -1
    A[9, 12] = 1
    A[10, 9] = 1
    A[11, 5] = -1
    A[12, 4] = -1

    # B matrix elements (-∂equations/∂x)
    B[0, 0] = -1
    B[0, 6] = -b1
    B[0, 11] = 1
    B[1, 1] = -1
    B[1, 3] = -a1
    B[1, 11] = -a2
    B[1, 12] = 1
    B[2, 2] = -1
    B[2, 5] = -g1
    B[2, 7] = 1
    B[2, 11] = -g3*(1 - g1)
    B[3, 7] = -1
    B[3, 8] = 1
    B[4, 0] = -rho_L_GDP_GAP
    B[5, 1] = -rho_DLA_CPI
    B[6, 2] = -rho_rs
    B[6, 4] = -rho_rs2
    B[7, 11] = -1
    B[8, 12] = -1
    B[9, 9] = 1
    B[10, 10] = 1
    B[11, 7] = -1
    B[12, 2] = -1

    # C matrix elements (-∂equations/∂shocks)
    C[4, 0] = -1
    C[5, 1] = -1
    C[6, 2] = -1

    return A, B, C