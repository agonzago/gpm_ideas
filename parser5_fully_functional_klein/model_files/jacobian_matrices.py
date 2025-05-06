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
    A[0,7] = -b4
    A[0,9] = 1 - b1
    A[1,10] = 1 - a1
    A[2,10] = 1 - g1
    A[2,11] = g2*(1 - g1)
    A[3,10] = -1
    A[4,2] = -1
    A[5,1] = -1
    A[6,0] = -1
    A[7,3] = -1
    A[8,6] = -1
    A[9,10] = 1
    A[10,8] = 1
    A[11,4] = -1
    A[12,5] = -1

    # Fill B
    B[0,2] = -1
    B[0,3] = -b1
    B[0,9] = 1
    B[1,1] = -1
    B[1,6] = -a1
    B[1,9] = -a2
    B[1,10] = 1
    B[2,0] = -1
    B[2,4] = -g1
    B[2,9] = -g3*(1 - g1)
    B[2,12] = 1
    B[3,7] = 1
    B[3,12] = -1
    B[4,2] = -rho_L_GDP_GAP
    B[5,1] = -rho_DLA_CPI
    B[6,0] = -rho_rs
    B[6,5] = -rho_rs2
    B[7,9] = -1
    B[8,10] = -1
    B[9,8] = 1
    B[10,11] = 1
    B[11,12] = -1
    B[12,0] = -1

    # Fill C (core shocks)
    C[4,0] = -1
    C[5,1] = -1
    C[6,2] = -1

    return A, B, C