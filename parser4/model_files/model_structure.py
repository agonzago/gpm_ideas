import numpy as np

indices = {'n_states': 7, 'n_controls': 6, 'n_vars': 13, 'n_shocks': 3, 'n_endogenous_states': 0, 'n_exo_states_ws': 0, 'n_exo_states_wos': 0, 'zero_persistence_processes': []}

R_struct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

C_selection = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

D_struct = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls

labels = {'state_labels': ['RES_L_GDP_GAP', 'RES_RS', 'RES_DLA_CPI', 'L_GDP_GAP_lag', 'RES_RS_lag', 'RS_lag', 'DLA_CPI_lag'], 'control_labels': ['RS', 'DLA_CPI_lead1', 'DLA_CPI', 'L_GDP_GAP', 'DLA_CPI_lead2', 'RR_GAP'], 'variable_labels': ['RES_L_GDP_GAP', 'RES_RS', 'RES_DLA_CPI', 'L_GDP_GAP_lag', 'RES_RS_lag', 'RS_lag', 'DLA_CPI_lag', 'RS', 'DLA_CPI_lead1', 'DLA_CPI', 'L_GDP_GAP', 'DLA_CPI_lead2', 'RR_GAP'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS']}
shock_to_process_var_map = {'SHK_L_GDP_GAP': 'RES_L_GDP_GAP', 'SHK_DLA_CPI': 'RES_DLA_CPI', 'SHK_RS': 'RES_RS'}