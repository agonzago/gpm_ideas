import numpy as np

def evaluate_trend_jacobians(theta):
    """Return A_tr, B_tr"""
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
    A_tr = np.zeros((5,5))
    B_tr = np.zeros((5,5))
    # fill A_tr
    A_tr[0,0] = 1
    A_tr[0,4] = 1
    A_tr[1,4] = 1
    A_tr[2,1] = 1
    A_tr[3,1] = 1
    A_tr[3,3] = 1
    A_tr[4,3] = 1
    # fill B_tr
    B_tr[0,0] = 1
    B_tr[1,1] = 1
    B_tr[2,2] = 1
    B_tr[4,4] = 1
    return A_tr, B_tr