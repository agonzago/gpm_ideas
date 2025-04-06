# Inside augmented_state_space.py

import numpy as np
import pandas as pd
import scipy.linalg as la

class AugmentedStateSpace:
    def __init__(self, base_ss, observed_vars, trend_specs):
        if base_ss is None:
            raise ValueError("base_ss cannot be None. Ensure the base model was solved successfully.")
        self.base_ss = base_ss
        self.observed_vars = list(observed_vars)
        self.trend_specs = trend_specs
        self.augmented = {}
        self._build_augmented_system()

    def _build_augmented_system(self):
        """Constructs the augmented state-space matrices including trends."""
        print("--- Building Augmented State Space System ---")

        try:
            base_A = self.base_ss['A']
            base_B = self.base_ss['B']
            base_C = self.base_ss['C']
            base_labels = self.base_ss['labels']
            base_indices = self.base_ss['indices']
            n_base_states = base_indices['n_states']
            n_base_shocks = base_indices['n_shocks'] # Get n_base_shocks here
            n_controls = base_indices['n_controls']
            potential_observables = list(base_labels['observable_labels'])
            base_state_labels = list(base_labels['state_labels'])
            base_shock_labels = list(base_labels['shock_labels'])
        except KeyError as e:
            raise KeyError(f"Missing expected key in base_ss dictionary: {e}")

        trend_state_labels = []
        trend_shock_labels = []
        trend_blocks_A = []
        trend_blocks_B = []
        trend_level_state_map = {}
        state_counter = 0

        for var_name in potential_observables:
            if var_name in self.trend_specs:
                trend_type = self.trend_specs[var_name]
                print(f"  Adding trend type '{trend_type}' for variable '{var_name}'")
                if trend_type == 'rw':
                    level_state = f"level_{var_name}"
                    level_shock = f"e_level_{var_name}"
                    trend_state_labels.append(level_state)
                    trend_shock_labels.append(level_shock)
                    trend_blocks_A.append(np.array([[1.0]]))
                    trend_blocks_B.append(np.array([[1.0]]))
                    trend_level_state_map[var_name] = state_counter
                    state_counter += 1
                elif trend_type == 'sd':
                    level_state = f"level_{var_name}"
                    growth_state = f"growth_{var_name}"
                    level_shock = f"e_level_{var_name}"
                    growth_shock = f"e_growth_{var_name}"
                    trend_state_labels.extend([level_state, growth_state])
                    trend_shock_labels.extend([level_shock, growth_shock])
                    trend_blocks_A.append(np.array([[1.0, 1.0], [0.0, 1.0]]))
                    b_block = np.zeros((2, 2))
                    b_block[0, 0] = 1.0
                    b_block[1, 1] = 1.0
                    trend_blocks_B.append(b_block)
                    trend_level_state_map[var_name] = state_counter
                    state_counter += 2
                elif trend_type == 'const':
                    level_state = f"level_{var_name}"
                    trend_state_labels.append(level_state)
                    trend_blocks_A.append(np.array([[1.0]]))
                    trend_blocks_B.append(np.zeros((1, 0)))
                    trend_level_state_map[var_name] = state_counter
                    state_counter += 1
                else:
                    print(f"Warning: Unknown trend type '{trend_type}' for variable '{var_name}'. Skipping.")

        if trend_blocks_A:
            A_trend = la.block_diag(*trend_blocks_A)
            n_trend_states = A_trend.shape[0]
            n_trend_shocks = len(trend_shock_labels)
            B_trend = np.zeros((n_trend_states, n_trend_shocks))
            current_row_idx = 0
            current_col_idx = 0
            for block in trend_blocks_B:
                rows, cols = block.shape
                if cols > 0:
                    if current_col_idx + cols <= n_trend_shocks:
                        B_trend[current_row_idx : current_row_idx + rows,
                                current_col_idx : current_col_idx + cols] = block
                        current_col_idx += cols
                    else:
                        print("Warning: B_trend block placement error - mismatch in shock count.")
                current_row_idx += rows
            if B_trend.shape != (n_trend_states, n_trend_shocks):
                print(f"Warning: Final B_trend shape {B_trend.shape} mismatch expected ({n_trend_states}, {n_trend_shocks})")
        else:
            A_trend = np.empty((0, 0))
            B_trend = np.empty((0, 0))
            n_trend_states = 0
            n_trend_shocks = 0

        A_aug = la.block_diag(base_A, A_trend)

        if n_base_shocks > 0 and n_trend_shocks > 0:
            B_aug = la.block_diag(base_B, B_trend)
        elif n_base_shocks > 0:
            B_aug = np.vstack([base_B, np.zeros((n_trend_states, n_base_shocks))])
        elif n_trend_shocks > 0:
            B_aug = np.vstack([np.zeros((n_base_states, n_trend_shocks)), B_trend])
        else:
            B_aug = np.zeros((n_base_states + n_trend_states, 0))

        n_aug_states = n_base_states + n_trend_states
        n_aug_shocks = n_base_shocks + n_trend_shocks
        if B_aug.shape != (n_aug_states, n_aug_shocks):
            print(f"Warning: Final B_aug shape {B_aug.shape} mismatch expected ({n_aug_states}, {n_aug_shocks})")

        # --- Checking B_aug construction ---
        print("\n--- Checking B_aug construction ---")
        print(f"Shape of B_aug: {B_aug.shape}")
        # Use n_base_shocks which is now defined in this scope
        print("B_aug (first {} rows, first {} cols - Base Shock Part):".format(n_base_states, n_base_shocks))
        print(B_aug[:n_base_states, :n_base_shocks])
        if n_base_shocks > 2: # Check if SHK_RS index exists
             print(f"Column for SHK_RS (base part, index 2): {B_aug[:n_base_states, 2]}")
             expected_b_col = np.zeros(n_base_states)
             # Set expected index 3 to 1 (for RES_RS_lag in TARGET order)
             if n_base_states > 3:
                  expected_b_col[3] = 1.0
             if not np.allclose(B_aug[:n_base_states, 2], expected_b_col):
                  print("WARNING: B_aug column for SHK_RS seems incorrect during construction!")
                  print(f"Expected: {expected_b_col}")
        print("--- End Checking B_aug construction ---\n")
        # --- End Check ---


        n_potential_obs = base_C.shape[0]
        C_aug = np.zeros((n_potential_obs, n_aug_states))
        C_aug[:, :n_base_states] = base_C

        for i, var_name in enumerate(potential_observables):
            if var_name in trend_level_state_map:
                level_trend_idx_local = trend_level_state_map[var_name]
                full_trend_col_idx = n_base_states + level_trend_idx_local
                if full_trend_col_idx < C_aug.shape[1]:
                    C_aug[i, full_trend_col_idx] = 1.0
                else:
                    print(f"Warning: Trend column index {full_trend_col_idx} out of bounds for C_aug for variable {var_name}")

        n_observed = len(self.observed_vars)
        H = np.zeros((n_observed, n_potential_obs))
        try:
            potential_obs_indices = {var: idx for idx, var in enumerate(potential_observables)}
            for k, obs_var in enumerate(self.observed_vars):
                if obs_var in potential_obs_indices:
                    i = potential_obs_indices[obs_var]
                    H[k, i] = 1.0
                else:
                    print(f"Warning: Observed variable '{obs_var}' not found in potential observables list: {potential_observables}. Row {k} in H will be zero.")
        except Exception as e:
            print(f"Error building H matrix: {e}")

        self.augmented['A'] = A_aug
        self.augmented['B'] = B_aug # Store B_aug
        self.augmented['C'] = C_aug
        self.augmented['H'] = H
        self.augmented['state_labels'] = base_state_labels + trend_state_labels
        self.augmented['shock_labels'] = base_shock_labels + trend_shock_labels
        self.augmented['observable_labels'] = self.observed_vars
        self.augmented['n_states'] = n_aug_states
        self.augmented['n_shocks'] = n_aug_shocks
        self.augmented['n_observed'] = n_observed
        self.augmented['n_base_states'] = n_base_states
        self.augmented['n_trend_states'] = n_trend_states
        # Also store n_base_shocks for convenience in IRF method
        self.augmented['n_base_shocks'] = n_base_shocks


        # --- DEBUG PRINTS for Observation Mapping ---

        # --- Final Check: Compare A_aug block with base_A ---
        print("\n--- Verifying A_aug vs base_A ---")
        aug_A_block = self.augmented['A'][:n_base_states, :n_base_states]
        base_A_orig = self.base_ss['A']
        print(f"Shape of base_A from input: {base_A_orig.shape}")
        print(f"Shape of A_aug base block: {aug_A_block.shape}")
        if aug_A_block.shape == base_A_orig.shape:
            if np.allclose(aug_A_block, base_A_orig):
                 print("OK: Base block of A_aug matches input base_A.")
            else:
                 print("ERROR: Base block of A_aug DOES NOT MATCH input base_A!")
                 diff_norm = np.linalg.norm(aug_A_block - base_A_orig)
                 print(f"Difference norm: {diff_norm}")
                 print("Input base_A (first 5x5):")
                 print(base_A_orig[:5, :5])
                 print("A_aug block (first 5x5):")
                 print(aug_A_block[:5, :5])
        else:
             print("ERROR: Shape mismatch between A_aug base block and input base_A.")
        print("--- End Verification ---\n")

        # (Keep the debug prints for H@C_aug as in the previous response)
        print("\n--- Debugging Observation Mapping ---")
        if H.shape[1] != C_aug.shape[0]:
             print(f"FATAL ERROR: H matrix columns ({H.shape[1]}) != C_aug matrix rows ({C_aug.shape[0]})")
             return

        C_obs_eff = H @ C_aug
        print(f"Effective Observation Matrix (C_obs_eff = H @ C_aug) shape: {C_obs_eff.shape}")

        for k, obs_var in enumerate(self.observed_vars):
            print(f"\nVariable: {obs_var} (Index {k} in observed list)")
            try:
                idx_potential = potential_observables.index(obs_var)
                print(f"  Index in potential list (base_C rows): {idx_potential}")
                eff_row = C_obs_eff[k, :]
                print(f"  Effective Obs Row (H@C_aug)[{k},:] (shape {eff_row.shape}):")
                print(f"    Base part (coeffs on x_base): {eff_row[:n_base_states]}")
                print(f"    Trend part (coeffs on x_trend): {eff_row[n_base_states:]}")
                base_C_row = self.base_ss['C'][idx_potential, :]
                print(f"  Base C Row base_C[{idx_potential},:] (shape {base_C_row.shape}): {base_C_row}")
                if np.allclose(eff_row[:n_base_states], base_C_row):
                    print("  --> Base parts MATCH.")
                else:
                    print("  --> !!! Base parts DO NOT MATCH !!!")
                    diff = np.linalg.norm(eff_row[:n_base_states] - base_C_row)
                    print(f"      Difference norm: {diff}")
                if obs_var in trend_level_state_map:
                    level_trend_idx_local = trend_level_state_map[obs_var]
                    full_trend_col_idx = n_base_states + level_trend_idx_local
                    if full_trend_col_idx < eff_row.shape[0]:
                        coeff_on_own_trend = eff_row[full_trend_col_idx]
                        print(f"    Coeff on own trend state ({trend_state_labels[level_trend_idx_local]} at full index {full_trend_col_idx}): {coeff_on_own_trend}")
                        if not np.isclose(coeff_on_own_trend, 1.0):
                            print(f"      WARNING: Coefficient on own trend state is not 1.0!")
                    else:
                        print(f"    WARNING: Trend index {full_trend_col_idx} out of bounds for effective row.")
                else:
                    print(f"    Variable has no specified trend state or is not in trend_level_state_map.")
                    if np.allclose(eff_row[n_base_states:], 0):
                        print("    Trend part of effective row is all zero (as expected).")
                    else:
                        print("    WARNING: Trend part of effective row is NON-ZERO even though variable has no trend state mapped.")
            except ValueError:
                print(f"  Variable '{obs_var}' not found in potential observables list.")
            except IndexError as e_idx:
                print(f"  Indexing error during debug print for {obs_var}: {e_idx}")
        print("--- End Debugging Observation Mapping / End Build ---")


    def update_parameters(self, new_base_ss):
        """Updates A and B blocks in augmented matrices with new base solution."""
        # ... (keep implementation as before) ...
        if new_base_ss is None:
             print("Warning: Cannot update parameters with None base_ss.")
             return
        n_base_states = self.augmented.get('n_base_states', -1)
        n_aug_shocks = self.augmented.get('n_shocks', -1)
        n_base_shocks_in_new = new_base_ss['indices']['n_shocks']
        if n_base_states != new_base_ss['indices']['n_states']:
            print("Warning: Base state count mismatch during parameter update. Rebuilding augmented system.")
            self.base_ss = new_base_ss
            self._build_augmented_system()
            return
        self.augmented['A'][:n_base_states, :n_base_states] = new_base_ss['A']
        if n_base_shocks_in_new <= self.augmented['B'].shape[1]:
             self.augmented['B'][:n_base_states, :n_base_shocks_in_new] = new_base_ss['B']
        else:
             print("Warning: More base shocks in new_base_ss than columns available in augmented B.")
        self.base_ss = new_base_ss


    def impulse_response(self, shock_name, shock_size=1.0, periods=40):
        """Calculate IRFs for the augmented model, returning observed variable responses."""
        print(f"\n--- Augmented IRF for {shock_name} ---")

        if not self.augmented or 'A' not in self.augmented:
             print("Error: Augmented system not built or is invalid.")
             return None

        A_aug, B_aug = self.augmented['A'], self.augmented['B']
        H, C_aug = self.augmented['H'], self.augmented['C']

        if H.shape[1] == C_aug.shape[0]:
            C_obs = H @ C_aug
        else:
            print(f"Error: Dimension mismatch for H@C_aug ({H.shape[1]} vs {C_aug.shape[0]})")
            return None

        aug_shock_labels = self.augmented['shock_labels']
        n_aug_states = self.augmented['n_states']
        n_observed = self.augmented['n_observed']
        obs_labels = self.augmented['observable_labels']

        # *** Get n_base_states and n_base_shocks from the augmented dictionary ***
        n_base_states = self.augmented.get('n_base_states', 0)
        n_base_shocks = self.augmented.get('n_base_shocks', 0) # Get base shock count


        try:
            shock_idx_aug = aug_shock_labels.index(shock_name)
            print(f"Shock index in augmented list: {shock_idx_aug} (out of {len(aug_shock_labels)})")
        except ValueError:
            print(f"Error: Shock '{shock_name}' not found in augmented model shocks: {aug_shock_labels}")
            return None
        except IndexError:
            print(f"Error: Shock index {shock_idx_aug} out of bounds for B_aug columns ({B_aug.shape[1]})")
            return None

        aug_states_irf = np.zeros((periods, n_aug_states))
        obs_irf = np.zeros((periods, n_observed))

        e0_aug = np.zeros(len(aug_shock_labels))
        if shock_idx_aug < len(e0_aug):
             e0_aug[shock_idx_aug] = shock_size
        else:
             print(f"Error: Shock index {shock_idx_aug} out of bounds for shock vector.")
             return None

        if B_aug.shape[1] != len(e0_aug):
             print(f"Error: B_aug columns ({B_aug.shape[1]}) != shock vector length ({len(e0_aug)})")
             return None

        # --- Checking B_aug before use in IRF ---
        # Use n_base_shocks which is now defined in this scope
        print("\n--- Checking B_aug before use in IRF ---")
        B_aug_in_irf = self.augmented['B'] # Get B_aug from stored dict
        print(f"Shape of B_aug_in_irf: {B_aug_in_irf.shape}")
        if B_aug_in_irf.shape[0] >= n_base_states and B_aug_in_irf.shape[1] >= n_base_shocks:
             print("B_aug_in_irf (first {} rows, first {} cols):".format(n_base_states, n_base_shocks))
             print(B_aug_in_irf[:n_base_states, :n_base_shocks])
             if n_base_shocks > 2: # Check if SHK_RS index (2) exists
                  print(f"Column for SHK_RS (base part, index 2): {B_aug_in_irf[:n_base_states, 2]}")
        else:
             print("Warning: Cannot print B_aug slice due to dimension mismatch.")
        print("--- End Checking B_aug before use ---\n")
        # --- End Check ---

        x_aug = B_aug @ e0_aug

        print("Initial Augmented State (x_aug_0):")
        if n_base_states <= n_aug_states:
            print(f"  Base states (first {n_base_states}): {x_aug[:n_base_states]}")
            print(f"  Trend states (last {n_aug_states - n_base_states}): {x_aug[n_base_states:]}")
            # Use n_base_shocks here to check if it's a base shock
            is_base_shock = shock_idx_aug < n_base_shocks
            if is_base_shock and not np.allclose(x_aug[n_base_states:], 0):
                 print("WARNING: Initial trend states are non-zero for a base shock!")
        else:
            print(f"Error: n_base_states ({n_base_states}) > n_aug_states ({n_aug_states})")
            print(f"  Full x_aug_0: {x_aug}")

        if C_obs.shape[1] != n_aug_states:
            print(f"Error: C_obs columns ({C_obs.shape[1]}) != n_aug_states ({n_aug_states})")
            return None

        for t in range(periods):
            aug_states_irf[t, :] = x_aug
            if x_aug.shape[0] == C_obs.shape[1]:
                 obs_irf[t, :] = C_obs @ x_aug
            else:
                 print(f"Error at t={t}: x_aug dimension ({x_aug.shape[0]}) mismatch with C_obs columns ({C_obs.shape[1]})")
                 obs_irf[t, :] = np.nan
                 break

            if A_aug.shape[1] != x_aug.shape[0]:
                 print(f"Error at t={t}: A_aug columns ({A_aug.shape[1]}) mismatch with x_aug rows ({x_aug.shape[0]})")
                 break

            x_aug = A_aug @ x_aug

            is_base_shock = shock_idx_aug < n_base_shocks # Check again inside loop if needed
            if t < 5 and is_base_shock:
                if n_base_states <= n_aug_states:
                     print(f"t={t+1}, Trend states = {x_aug[n_base_states:]}")
                     if not np.allclose(x_aug[n_base_states:], 0):
                          print(f"WARNING: Non-zero trend states at t={t+1} for a base shock!")
                else:
                     print(f"t={t+1}, Full x_aug = {x_aug}")

        if len(obs_labels) != obs_irf.shape[1]:
             print(f"Warning: Number of observable labels ({len(obs_labels)}) does not match IRF columns ({obs_irf.shape[1]}). Using generic labels.")
             obs_labels = [f"Obs_{i}" for i in range(obs_irf.shape[1])]

        irf_df = pd.DataFrame(obs_irf, columns=obs_labels)
        irf_df.attrs['shock_name'] = shock_name
        return irf_df

    def get_kalman_matrices(self):
         """Returns matrices needed for a standard Kalman filter."""
         # ... (keep implementation as before) ...
         if not self.augmented or 'A' not in self.augmented:
             raise RuntimeError("Augmented system has not been built.")
         A = self.augmented['A']
         B = self.augmented['B']
         H = self.augmented['H']
         C_aug = self.augmented['C']
         if H.shape[1] != C_aug.shape[0]:
              raise ValueError(f"Dimension mismatch for H@C_aug ({H.shape[1]} vs {C_aug.shape[0]})")
         C_obs = H @ C_aug
         if B.size > 0:
             Q = B @ B.T
         else:
             Q = np.zeros((A.shape[0], A.shape[0]))
         return A, C_obs, Q

# --- End of augmented_state_space.py modifications ---