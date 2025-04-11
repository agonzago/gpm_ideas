import numpy as np

indices = {'n_states': 7, 'n_controls': 9, 'n_vars': 16, 'n_shocks': 3, 'n_endogenous_states': 3, 'n_exo_states_ws': 4, 'n_exo_states_wos': 0, 'zero_persistence_processes': ['RES_L_GDP_GAP', 'RES_DLA_CPI', 'RES_RS']}

R_struct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

C_selection = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

D_struct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls

labels = {'state_labels': ['DLA_CPI_lag', 'L_GDP_GAP_lag', 'RS_lag', 'RES_DLA_CPI_lag', 'RES_L_GDP_GAP_lag', 'RES_RS_lag', 'RES_RS_lag2'], 'control_labels': ['DLA_CPI', 'DLA_CPI_lead1', 'DLA_CPI_lead2', 'L_GDP_GAP', 'RES_DLA_CPI', 'RES_L_GDP_GAP', 'RES_RS', 'RR_GAP', 'RS'], 'variable_labels': ['DLA_CPI_lag', 'L_GDP_GAP_lag', 'RS_lag', 'RES_DLA_CPI_lag', 'RES_L_GDP_GAP_lag', 'RES_RS_lag', 'RES_RS_lag2', 'DLA_CPI', 'DLA_CPI_lead1', 'DLA_CPI_lead2', 'L_GDP_GAP', 'RES_DLA_CPI', 'RES_L_GDP_GAP', 'RES_RS', 'RR_GAP', 'RS'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS']}
