import numpy as np

def evaluate_jacobians(theta):
    """
    Evaluates Jacobian matrices for an economic model based on parameter values.
    
    Args:
        theta: List or array of parameter values in the order of:
              ['b1', 'b2', 'b4', 'a1', 'a2', 'g1', 'g2', 'g3', 'rho_DLA_CPI', 'rho_L_GDP_GAP', 'rho_rs']
        
    Returns:
        A: Matrix -∂F/∂X_p (negative Jacobian with respect to variables with '_p' suffix)
        B: Matrix ∂F/∂X (Jacobian with respect to variables without '_p' suffix)
        C: Matrix ∂F/∂W (Jacobian with respect to shock variables)
    """
    # Unpack parameters from theta
    b1_ = theta[0]
    b2_ = theta[1]
    b4_ = theta[2]
    a1_ = theta[3]
    a2_ = theta[4]
    g1_ = theta[5]
    g2_ = theta[6]
    g3_ = theta[7]
    rho_DLA_CPI_ = theta[8]
    rho_L_GDP_GAP_ = theta[9]
    rho_rs_ = theta[10]

    A = np.zeros((16, 16))
    B = np.zeros((16, 16))
    C = np.zeros((16, 3))

    A[0, 9] = b1_ - 1
    A[0, 12] = b4_
    A[2, 10] = a1_ - 1
    A[4, 10] = g1_ - 1
    A[6, 10] = 1
    A[7, 3] = -1
    A[8, 4] = -1
    A[9, 5] = -1
    A[10, 6] = -1
    A[11, 7] = -1
    A[12, 8] = -1
    A[13, 10] = -1
    A[14, 13] = -1
    A[15, 14] = -1

    B[0, 0] = 1
    B[0, 3] = b1_
    B[0, 9] = -1
    B[1, 0] = -1
    B[1, 4] = rho_L_GDP_GAP_
    B[2, 1] = 1
    B[2, 5] = a1_
    B[2, 9] = a2_
    B[2, 10] = -1
    B[3, 1] = -1
    B[3, 6] = rho_DLA_CPI_
    B[4, 2] = 1
    B[4, 7] = g1_
    B[4, 9] = g3_*(1 - g1_)
    B[4, 11] = -1
    B[4, 15] = g2_*(1 - g1_)
    B[5, 2] = -1
    B[5, 8] = rho_rs_
    B[6, 11] = 1
    B[6, 12] = -1
    B[7, 9] = -1
    B[8, 0] = -1
    B[9, 10] = -1
    B[10, 1] = -1
    B[11, 11] = -1
    B[12, 2] = -1
    B[13, 13] = -1
    B[14, 14] = -1
    B[15, 15] = -1

    C[1, 0] = 1
    C[3, 1] = 1
    C[5, 2] = 1

    return A, B, C