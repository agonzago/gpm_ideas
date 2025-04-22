import numpy as np

def evaluate_jacobians(theta):
    """Compute A,B,C,D given parameter vector theta"""
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
    stderr_SHK_L_GDP_TREND = theta[11]
    stderr_SHK_G_TREND = theta[12]
    stderr_SHP_PI_TREND = theta[13]
    stderr_SHK_RS_TREND = theta[14]
    stderr_SHK_RR_TREND = theta[15]

    A = np.zeros((13,9))
    B = np.zeros((13,9))
    C = np.zeros((13,9))
    D = np.zeros((13,3))

    # Fill A = ∂F/∂x_p
    A[0,0] = 1 - b1
    A[0,3] = -b4
    A[1,1] = 1 - a1
    A[2,1] = 1 - g1
    A[2,8] = g2*(1 - g1)
    A[3,1] = -1
    A[4,4] = -1
    A[5,5] = -1
    A[6,6] = -1
    A[9,1] = 1
    A[10,7] = 1

    # Fill B = -∂F/∂x
    B[0,0] = 1
    B[0,4] = -1
    B[1,0] = -a2
    B[1,1] = 1
    B[1,5] = -1
    B[2,0] = -g3*(1 - g1)
    B[2,2] = 1
    B[2,6] = -1
    B[3,2] = -1
    B[3,3] = 1
    B[4,4] = -rho_L_GDP_GAP
    B[5,5] = -rho_DLA_CPI
    B[6,6] = -rho_rs
    B[7,0] = -1
    B[8,1] = -1
    B[9,7] = 1
    B[10,8] = 1
    B[11,2] = -1
    B[12,6] = -1

    # Fill C = -∂F/∂x_lag
    C[0,0] = -b1
    C[1,1] = -a1
    C[2,2] = -g1
    C[6,6] = -rho_rs2

    # Fill D = -∂F/∂eps
    D[4,0] = -1
    D[5,1] = -1
    D[6,2] = -1

    return A, B, C, D