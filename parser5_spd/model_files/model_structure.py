import numpy as np

indices = {'n_states': 7, 'n_controls': 6, 'n_vars': 13, 'n_shocks': 3, 'n_endogenous_states': 0, 'n_exo_states_ws': 0, 'n_exo_states_wos': 0, 'zero_persistence_processes': []}

R_struct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

C_selection = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

D_struct = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls

labels = {'state_labels': ['RES_DLA_CPI', 'RES_RS', 'RES_L_GDP_GAP', 'DLA_CPI_lag', 'RS_lag', 'L_GDP_GAP_lag', 'RES_RS_lag'], 'control_labels': ['RS', 'DLA_CPI_lead1', 'DLA_CPI_lead2', 'RR_GAP', 'L_GDP_GAP', 'DLA_CPI'], 'variable_labels': ['RES_DLA_CPI', 'RES_RS', 'RES_L_GDP_GAP', 'DLA_CPI_lag', 'RS_lag', 'L_GDP_GAP_lag', 'RES_RS_lag', 'RS', 'DLA_CPI_lead1', 'DLA_CPI_lead2', 'RR_GAP', 'L_GDP_GAP', 'DLA_CPI'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS']}
shock_to_process_var_map = {'SHK_L_GDP_GAP': 'RES_L_GDP_GAP', 'SHK_DLA_CPI': 'RES_DLA_CPI', 'SHK_RS': 'RES_RS'}