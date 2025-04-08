import numpy as np

indices = {'n_endogenous': 0, 'n_exo_states': 0, 'n_controls': 17, 'n_shocks': 3, 'n_states': 0, 'n_observables': 17, 'zero_persistence_processes': []}

R = np.array([])

B_structure = np.array([])

C_structure = np.array([[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []])

D = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

labels = {'state_labels': [], 'observable_labels': ['DLA_CPI_lead2', 'RES_L_GDP_GAP', 'DLA_CPI_lead3', 'RES_L_GDP_GAP_lag', 'DLA_CPI', 'L_GDP_GAP', 'L_GDP_GAP_lag', 'RES_DLA_CPI', 'RS_lag', 'RES_DLA_CPI_lag', 'RES_RS_lag2', 'DLA_CPI_lag', 'RS', 'RES_RS', 'RES_RS_lag', 'DLA_CPI_lead1', 'RR_GAP'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS'], 'shock_to_state_map': {}, 'state_to_shock_map': {}, 'zero_persistence_processes': []}
