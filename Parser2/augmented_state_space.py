# ----- augmented_state_space.py -----
import numpy as np
import pandas as pd
import scipy.linalg as la

class AugmentedStateSpace:
    def __init__(self, base_ss, observed_vars, trend_specs):
        if base_ss is None:
            raise ValueError("base_ss cannot be None. Ensure the base model was solved successfully.")
        self.base_ss = base_ss
        self.observed_vars = observed_vars
        self.trend_specs = trend_specs # dict: {'VAR': 'rw', 'VAR2': 'sd', ...}
        self.augmented = {} # To store augmented matrices, labels, indices

        self._build_augmented_system()

    def _build_augmented_system(self):
        """Constructs the augmented state-space matrices."""
        base_A = self.base_ss['A']
        base_B = self.base_ss['B']
        base_C = self.base_ss['C']
        base_labels = self.base_ss['labels']
        base_indices = self.base_ss['indices']

        trend_state_labels = []
        trend_shock_labels = []
        trend_blocks_A = []
        trend_blocks_B = []
        trend_var_map = {} # Maps base variable name -> index in trend_state_labels

        state_counter = 0
        shock_counter = 0

        potential_observables = base_labels['observable_labels']

        for var_name in potential_observables:
            if var_name in self.trend_specs:
                trend_type = self.trend_specs[var_name]
                if trend_type == 'rw':
                    # Random Walk: trend_t = trend_{t-1} + e_level
                    level_state = f"level_{var_name}"
                    level_shock = f"e_level_{var_name}"
                    trend_state_labels.append(level_state)
                    trend_shock_labels.append(level_shock)
                    trend_blocks_A.append(np.array([[1.0]])) # A_trend block
                    # B_trend block: maps level_shock to level_state equation
                    b_block = np.zeros((1, 1))
                    b_block[0, 0] = 1.0
                    trend_blocks_B.append(b_block)
                    trend_var_map[var_name] = state_counter
                    state_counter += 1
                    shock_counter += 1
                elif trend_type == 'sd':
                    # Second Difference:
                    # trend_t = trend_{t-1} + g_{t-1} + e_level
                    # g_t = g_{t-1} + e_growth
                    level_state = f"level_{var_name}"
                    growth_state = f"growth_{var_name}"
                    level_shock = f"e_level_{var_name}"
                    growth_shock = f"e_growth_{var_name}"
                    trend_state_labels.extend([level_state, growth_state])
                    trend_shock_labels.extend([level_shock, growth_shock])
                    # A_trend block: [[1, 1], [0, 1]]
                    trend_blocks_A.append(np.array([[1.0, 1.0], [0.0, 1.0]]))
                    # B_trend block: maps shocks to equations
                    b_block = np.zeros((2, 2))
                    b_block[0, 0] = 1.0 # level_shock -> level_state eq
                    b_block[1, 1] = 1.0 # growth_shock -> growth_state eq
                    trend_blocks_B.append(b_block)
                    trend_var_map[var_name] = state_counter # Index of the level state
                    state_counter += 2
                    shock_counter += 2
                elif trend_type == 'const':
                    # Constant Mean: trend_t = trend_{t-1} (no shock)
                    level_state = f"level_{var_name}"
                    trend_state_labels.append(level_state)
                    # No shock associated with constant mean trend
                    trend_blocks_A.append(np.array([[1.0]]))
                    # B_trend block is just zero for this state
                    trend_blocks_B.append(np.zeros((1,0))) # Placeholder, will adjust size later
                    trend_var_map[var_name] = state_counter
                    state_counter += 1
                    # shock_counter remains unchanged
                else:
                    print(f"Warning: Unknown trend type '{trend_type}' for variable '{var_name}'. Skipping.")

        # Assemble A_trend and B_trend
        if trend_blocks_A:
            A_trend = la.block_diag(*trend_blocks_A)
            # Adjust B_trend dimensions
            n_trend_states = A_trend.shape[0]
            n_trend_shocks = len(trend_shock_labels)
            B_trend = np.zeros((n_trend_states, n_trend_shocks))
            current_row = 0
            current_col = 0
            for block in trend_blocks_B:
                 rows, cols = block.shape
                 if cols > 0: # Skip 'const' blocks here
                      B_trend[current_row:current_row+rows, current_col:current_col+cols] = block
                      current_col += cols
                 current_row += rows

        else:
            A_trend = np.empty((0, 0))
            B_trend = np.empty((0, 0))
            n_trend_states = 0
            n_trend_shocks = 0

        # Build Augmented Matrices
        n_base_states = base_indices['n_states']
        n_base_shocks = base_indices['n_shocks']

        A_aug = la.block_diag(base_A, A_trend)
        # Ensure correct shapes even if one part is empty
        if B_trend.size == 0:
             B_aug = np.hstack([base_B, np.zeros((n_base_states, 0))])
             B_aug = np.vstack([B_aug, np.zeros((0, n_base_shocks + 0))])
        elif base_B.size == 0:
             B_aug = np.hstack([np.zeros((0, n_trend_shocks))])
             B_aug = np.vstack([B_aug, np.hstack([np.zeros((n_trend_states, 0)), B_trend])])
        else:
             B_aug = la.block_diag(base_B, B_trend)


        # Build C_aug: y_potential = C_aug * x_aug
        n_potential_obs = base_C.shape[0]
        n_aug_states = n_base_states + n_trend_states
        C_aug = np.zeros((n_potential_obs, n_aug_states))
        C_aug[:, :n_base_states] = base_C # Embed base C

        for i, var_name in enumerate(potential_observables):
            if var_name in trend_var_map:
                trend_state_idx = trend_var_map[var_name]
                aug_col_idx = n_base_states + trend_state_idx
                C_aug[i, aug_col_idx] = 1.0 # Add trend effect

        # Build H: y_observed = H * y_potential
        n_observed = len(self.observed_vars)
        H = np.zeros((n_observed, n_potential_obs))
        obs_var_indices = {var: idx for idx, var in enumerate(potential_observables)}
        for k, obs_var in enumerate(self.observed_vars):
            if obs_var in obs_var_indices:
                i = obs_var_indices[obs_var]
                H[k, i] = 1.0
            else:
                print(f"Warning: Observed variable '{obs_var}' not found in potential observables. Cannot create H matrix correctly.")


        # Store augmented system components
        self.augmented['A'] = A_aug
        self.augmented['B'] = B_aug
        self.augmented['C'] = C_aug # Maps augmented state to potential observables
        self.augmented['H'] = H     # Selects actual observables
        self.augmented['state_labels'] = base_labels['state_labels'] + trend_state_labels
        self.augmented['shock_labels'] = base_labels['shock_labels'] + trend_shock_labels
        self.augmented['observable_labels'] = self.observed_vars # Labels for H*C*x output
        self.augmented['n_states'] = n_aug_states
        self.augmented['n_shocks'] = n_base_shocks + n_trend_shocks
        self.augmented['n_observed'] = n_observed


    def update_parameters(self, new_base_ss):
        """Updates A and B blocks in augmented matrices with new base solution."""
        if new_base_ss is None:
             print("Warning: Cannot update parameters with None base_ss.")
             return

        n_base_states = new_base_ss['indices']['n_states']
        n_base_shocks = new_base_ss['indices']['n_shocks']

        if n_base_states != self.base_ss['indices']['n_states'] or \
           n_base_shocks != self.base_ss['indices']['n_shocks']:
            print("Warning: Structure mismatch during parameter update. Rebuilding augmented system.")
            self.base_ss = new_base_ss
            self._build_augmented_system() # Rebuild if structure changed
            return

        # Update blocks directly
        self.augmented['A'][:n_base_states, :n_base_states] = new_base_ss['A']
        self.augmented['B'][:n_base_states, :n_base_shocks] = new_base_ss['B']
        # Store the new base_ss for consistency
        self.base_ss = new_base_ss


    def impulse_response(self, shock_name, shock_size=1.0, periods=40):
        """Calculate IRFs for the augmented model, returning observed variable responses."""
        A_aug, B_aug = self.augmented['A'], self.augmented['B']
        C_obs = self.augmented['H'] @ self.augmented['C'] # Combined observation matrix
        aug_shock_labels = self.augmented['shock_labels']
        n_aug_states = self.augmented['n_states']
        n_observed = self.augmented['n_observed']
        obs_labels = self.augmented['observable_labels']


        try:
            shock_idx_aug = aug_shock_labels.index(shock_name)
        except ValueError:
            print(f"Error: Shock '{shock_name}' not found in augmented model shocks: {aug_shock_labels}")
            return None

        aug_states_irf = np.zeros((periods, n_aug_states))
        obs_irf = np.zeros((periods, n_observed))

        # Initial shock impact on augmented states
        x_aug = B_aug[:, shock_idx_aug] * shock_size

        for t in range(periods):
            aug_states_irf[t, :] = x_aug
            obs_irf[t, :] = C_obs @ x_aug
            x_aug = A_aug @ x_aug # Evolve augmented state

        irf_df = pd.DataFrame(obs_irf, columns=obs_labels)
        irf_df.attrs['shock_name'] = shock_name
        return irf_df

    def get_kalman_matrices(self):
        """Returns matrices needed for a standard Kalman filter."""
        A = self.augmented['A']
        B = self.augmented['B']
        C_obs = self.augmented['H'] @ self.augmented['C']
        # Assuming uncorrelated shocks with unit variance for Q
        Q = B @ B.T # State noise covariance
        # Measurement noise covariance R needs to be specified separately
        # D term (direct impact of state noise on observation) is usually zero
        D_obs = np.zeros((self.augmented['n_observed'], self.augmented['n_shocks']))

        return A, C_obs, Q # Return state transition, observation mapping, state noise cov
