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
    A[0, 10] = 1 - b1
    A[0, 12] = -b4
    A[1, 9] = 1 - a1
    A[2, 9] = 1 - g1
    A[2, 11] = g2*(1 - g1)
    A[3, 9] = -1
    A[4, 0] = -1
    A[5, 2] = -1
    A[6, 1] = -1
    A[7, 3] = -1
    A[8, 6] = -1
    A[9, 9] = 1
    A[10, 8] = 1
    A[11, 5] = -1
    A[12, 4] = -1

    # B matrix elements (-∂equations/∂x)
    B[0, 0] = -1
    B[0, 3] = -b1
    B[0, 10] = 1
    B[1, 2] = -1
    B[1, 6] = -a1
    B[1, 9] = 1
    B[1, 10] = -a2
    B[2, 1] = -1
    B[2, 5] = -g1
    B[2, 7] = 1
    B[2, 10] = -g3*(1 - g1)
    B[3, 7] = -1
    B[3, 12] = 1
    B[4, 0] = -rho_L_GDP_GAP
    B[5, 2] = -rho_DLA_CPI
    B[6, 1] = -rho_rs
    B[6, 4] = -rho_rs2
    B[7, 10] = -1
    B[8, 9] = -1
    B[9, 8] = 1
    B[10, 11] = 1
    B[11, 7] = -1
    B[12, 1] = -1

    # C matrix elements (-∂equations/∂shocks)
    C[4, 0] = -1
    C[5, 1] = -1
    C[6, 2] = -1

    return A, B, C