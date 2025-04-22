import numpy as np
def evaluate_jacobians(theta):
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
    stderr_SHK_L_GDP_TREND = theta[11]
    stderr_SHK_G_TREND = theta[12]
    stderr_SHP_PI_TREND = theta[13]
    stderr_SHK_RS_TREND = theta[14]
    stderr_SHK_RR_TREND = theta[15]
    A = np.zeros((13, 13))
    B = np.zeros((13, 13))
    C = np.zeros((13, 3))

    # Fill A
    A[0,7] = 1 - b1
    A[0,8] = -b4
    A[1,9] = 1 - a1
    A[2,9] = 1 - g1
    A[2,12] = g2*(1 - g1)
    A[3,9] = -1
    A[4,1] = -1
    A[5,0] = -1
    A[6,2] = -1
    A[7,5] = -1
    A[8,6] = -1
    A[9,9] = 1
    A[10,11] = 1
    A[11,3] = -1
    A[12,4] = -1

    # Fill B
    B[0,1] = -1
    B[0,5] = -b1
    B[0,7] = 1
    B[1,0] = -1
    B[1,6] = -a1
    B[1,7] = -a2
    B[1,9] = 1
    B[2,2] = -1
    B[2,3] = -g1
    B[2,7] = -g3*(1 - g1)
    B[2,10] = 1
    B[3,8] = 1
    B[3,10] = -1
    B[4,1] = -rho_L_GDP_GAP
    B[5,0] = -rho_DLA_CPI
    B[6,2] = -rho_rs
    B[6,4] = -rho_rs2
    B[7,7] = -1
    B[8,9] = -1
    B[9,11] = 1
    B[10,12] = 1
    B[11,10] = -1
    B[12,2] = -1

    # Fill C (core shocks)
    C[4,0] = -1
    C[5,1] = -1
    C[6,2] = -1

    return A, B, C