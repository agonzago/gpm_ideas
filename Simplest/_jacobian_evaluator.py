import numpy as np

def evaluate_jacobians(theta):
    """
    Evaluates Jacobian matrices for an economic model based on parameter values.
    
    Args:
        theta: List or array of parameter values in the order of:
              ['b1', 'b2', 'b4', 'a1', 'a2', 'g1', 'g2', 'g3']
        
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

    A = np.zeros((10, 10))
    B = np.zeros((10, 10))
    C = np.zeros((10, 3))

    A[0, 0] = b1_ - 1
    A[0, 3] = b4_
    A[1, 1] = a1_ - 1
    A[2, 1] = g1_ - 1
    A[3, 1] = 1
    A[4, 4] = -1
    A[5, 5] = -1
    A[6, 6] = -1
    A[7, 1] = -1
    A[8, 7] = -1
    A[9, 8] = -1

    B[0, 0] = -1
    B[0, 4] = b1_
    B[1, 0] = a2_
    B[1, 1] = -1
    B[1, 5] = a1_
    B[2, 0] = g3_*(1 - g1_)
    B[2, 2] = -1
    B[2, 6] = g1_
    B[2, 9] = g2_*(1 - g1_)
    B[3, 2] = 1
    B[3, 3] = -1
    B[4, 0] = -1
    B[5, 1] = -1
    B[6, 2] = -1
    B[7, 7] = -1
    B[8, 8] = -1
    B[9, 9] = -1

    C[0, 0] = 1
    C[1, 1] = 1
    C[2, 2] = 1

    return A, B, C