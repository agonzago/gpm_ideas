import numpy as np

indices = {'n_endogenous': 3, 'n_exo_states': 3, 'n_controls': 6, 'n_shocks': 3, 'n_states': 6, 'n_observables': 12, 'zero_persistence_processes': []}

R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

B_structure = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

C_structure = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

D = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

labels = {'state_labels': ['DLA_CPI_lag', 'RS_lag', 'L_GDP_GAP_lag', 'RES_RS_lag2', 'RES_RS_lag', 'RES_RS'], 'observable_labels': ['RES_L_GDP_GAP', 'L_GDP_GAP', 'DLA_CPI', 'RR_GAP', 'RS', 'RES_DLA_CPI', 'DLA_CPI_lag', 'RS_lag', 'L_GDP_GAP_lag', 'RES_RS_lag2', 'RES_RS_lag', 'RES_RS'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS'], 'shock_to_state_map': {'SHK_L_GDP_GAP': 'RES_L_GDP_GAP', 'SHK_DLA_CPI': 'RES_DLA_CPI', 'SHK_RS': 'RES_RS_lag'}, 'state_to_shock_map': {'RES_L_GDP_GAP': 'SHK_L_GDP_GAP', 'RES_DLA_CPI': 'SHK_DLA_CPI', 'RES_RS': 'SHK_RS', 'RES_RS_lag': 'SHK_RS'}, 'zero_persistence_processes': []}
