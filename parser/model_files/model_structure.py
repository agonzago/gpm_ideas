import numpy as np

indices = {'n_endogenous': 3, 'n_exo_states': 4, 'n_controls': 10, 'n_shocks': 3, 'n_states': 7, 'n_observables': 3}

R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

B_structure = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

C_structure = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

D = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

labels = {'state_labels': ['DLA_CPI_lag', 'RS_lag', 'L_GDP_GAP_lag', 'RES_RS_lag', 'RES_DLA_CPI_lag', 'RES_L_GDP_GAP_lag', 'RES_RS_lag2'], 'observable_labels': ['DLA_CPI_lead1', 'RR_GAP', 'L_GDP_GAP', 'DLA_CPI_lead3', 'DLA_CPI_lead2', 'RES_RS', 'DLA_CPI', 'RES_DLA_CPI', 'RS', 'RES_L_GDP_GAP', 'DLA_CPI_lag', 'RS_lag', 'L_GDP_GAP_lag', 'RES_RS_lag', 'RES_DLA_CPI_lag', 'RES_L_GDP_GAP_lag', 'RES_RS_lag2'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS'], 'param_labels': ['b1', 'b4', 'a1', 'a2', 'g1', 'g2', 'g3', 'rho_DLA_CPI', 'rho_L_GDP_GAP', 'rho_rs', 'rho_rs2']}

T_trend_structure = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

R_trend_structure = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

C_trend_structure = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

n_trend_states = 3

obs_mapping = {'rs_obs': {'type': 'control', 'index': 8, 'obs_index': 0}, 'dla_cpi_obs': {'type': 'control', 'index': 6, 'obs_index': 1}, 'l_gdp_obs': {'type': 'control', 'index': 2, 'obs_index': 2}}

