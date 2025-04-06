import numpy as np
import pandas as pd
import scipy.linalg as la

class AugmentedStateSpace:
    def __init__(self, base_ss, observed_vars, trend_specs):
        """
        Initialize an augmented state space system that includes stochastic trends.
        
        Args:
            base_ss: Base state space system (contemporaneous model)
            observed_vars: List of observable variables
            trend_specs: Dictionary mapping variable names to trend types
                         ('rw' for random walk, 'sd' for second difference, 'const' for constant)
        """
        if base_ss is None:
            raise ValueError("base_ss cannot be None. Ensure the base model was solved successfully.")
        
        self.base_ss = base_ss
        self.observed_vars = list(observed_vars)
        self.trend_specs = trend_specs
        self.augmented = {}
        self._build_augmented_system()

    def _build_augmented_system(self):
        """
        Constructs the augmented state-space system:
        
        x^aug_{t+1} = A^aug x^aug_t + B^aug ε^aug_t
        y^obs_t = H C^aug x^aug_t + H D^aug ε^aug_t
        
        where x^aug_t = [x_t; x^trend_t] combines the base model states and trend states.
        
        This follows section 5 of the state_space_specification document.
        """
        print("--- Building Augmented State Space System ---")

        try:
            # Extract base state space components
            base_A = self.base_ss['A']
            base_B = self.base_ss['B']
            base_C = self.base_ss['C']
            base_D = self.base_ss['D']  # Important for direct shock effects
            base_labels = self.base_ss['observable_labels']
            base_state_labels = self.base_ss['state_labels']
            base_shock_labels = self.base_ss['shock_labels']
            
            # Get base dimensions
            n_base_states = base_A.shape[0]
            n_base_shocks = base_B.shape[1]
            n_potential_obs = base_C.shape[0]
            
            # Dictionary to track which trend states correspond to which variables
            trend_level_state_map = {}
            
            # --------- Build Trend Blocks ---------
            trend_state_labels = []
            trend_shock_labels = []
            trend_blocks_A = []  # A matrices for each trend process
            trend_blocks_B = []  # B matrices for each trend process
            state_counter = 0
            
            # Process each observable that has a trend specification
            for var_name in base_labels:
                if var_name in self.trend_specs:
                    trend_type = self.trend_specs[var_name]
                    print(f"  Adding trend type '{trend_type}' for variable '{var_name}'")
                    
                    if trend_type == 'rw':  # Random walk trend
                        # trend_t = trend_{t-1} + ε^level_t
                        level_state = f"level_{var_name}"
                        level_shock = f"e_level_{var_name}"
                        
                        trend_state_labels.append(level_state)
                        trend_shock_labels.append(level_shock)
                        
                        # A matrix: [1]
                        trend_blocks_A.append(np.array([[1.0]]))
                        
                        # B matrix: [1]
                        trend_blocks_B.append(np.array([[1.0]]))
                        
                        # Map variable to its trend state
                        trend_level_state_map[var_name] = state_counter
                        state_counter += 1
                        
                    elif trend_type == 'sd':  # Second difference trend
                        # trend_t = trend_{t-1} + g_{t-1} + ε^level_t
                        # g_t = g_{t-1} + ε^growth_t
                        level_state = f"level_{var_name}"
                        growth_state = f"growth_{var_name}"
                        level_shock = f"e_level_{var_name}"
                        growth_shock = f"e_growth_{var_name}"
                        
                        trend_state_labels.extend([level_state, growth_state])
                        trend_shock_labels.extend([level_shock, growth_shock])
                        
                        # A matrix: [1 1; 0 1]
                        trend_blocks_A.append(np.array([
                            [1.0, 1.0],
                            [0.0, 1.0]
                        ]))
                        
                        # B matrix: [1 0; 0 1]
                        trend_blocks_B.append(np.array([
                            [1.0, 0.0],
                            [0.0, 1.0]
                        ]))
                        
                        # Map variable to its trend state
                        trend_level_state_map[var_name] = state_counter
                        state_counter += 2
                        
                    elif trend_type == 'const':  # Constant mean trend
                        # trend_t = trend_{t-1}
                        level_state = f"level_{var_name}"
                        
                        trend_state_labels.append(level_state)
                        
                        # A matrix: [1]
                        trend_blocks_A.append(np.array([[1.0]]))
                        
                        # B matrix: [] (empty, no shocks)
                        trend_blocks_B.append(np.zeros((1, 0)))
                        
                        # Map variable to its trend state
                        trend_level_state_map[var_name] = state_counter
                        state_counter += 1
                        
                    else:
                        print(f"Warning: Unknown trend type '{trend_type}' for variable '{var_name}'. Skipping.")
            
            # --------- Construct Augmented Matrices ---------
            
            # Build A_trend and B_trend from the blocks
            if trend_blocks_A:
                # Combine A blocks into block diagonal matrix
                A_trend = la.block_diag(*trend_blocks_A)
                n_trend_states = A_trend.shape[0]
                n_trend_shocks = len(trend_shock_labels)
                
                # Construct B_trend by combining blocks
                B_trend = np.zeros((n_trend_states, n_trend_shocks))
                current_row = 0
                current_col = 0
                
                for block in trend_blocks_B:
                    rows, cols = block.shape
                    if cols > 0:
                        B_trend[current_row:current_row+rows, current_col:current_col+cols] = block
                        current_col += cols
                    current_row += rows
            else:
                A_trend = np.empty((0, 0))
                B_trend = np.empty((0, 0))
                n_trend_states = 0
                n_trend_shocks = 0
            
            # Construct augmented A matrix - block diagonal
            A_aug = la.block_diag(base_A, A_trend)
            
            # Construct augmented B matrix - block diagonal with proper dimensions
            if n_base_shocks > 0 and n_trend_shocks > 0:
                B_aug = la.block_diag(base_B, B_trend)
            elif n_base_shocks > 0:
                B_aug = np.vstack([base_B, np.zeros((n_trend_states, n_base_shocks))])
            elif n_trend_shocks > 0:
                B_aug = np.vstack([np.zeros((n_base_states, n_trend_shocks)), B_trend])
            else:
                B_aug = np.zeros((n_base_states + n_trend_states, 0))
            
            # Calculate augmented dimensions
            n_aug_states = n_base_states + n_trend_states
            n_aug_shocks = n_base_shocks + n_trend_shocks
            
            # Verify B_aug shape
            if B_aug.shape != (n_aug_states, n_aug_shocks):
                print(f"Warning: B_aug shape {B_aug.shape} doesn't match expected ({n_aug_states}, {n_aug_shocks})")
            
            # Construct augmented C matrix (C_aug)
            C_aug = np.zeros((n_potential_obs, n_aug_states))
            
            # Base part of C_aug (for base states)
            C_aug[:, :n_base_states] = base_C
            
            # Add trend effects to C_aug
            for i, var_name in enumerate(base_labels):
                if var_name in trend_level_state_map:
                    level_idx = trend_level_state_map[var_name]
                    full_idx = n_base_states + level_idx
                    if full_idx < C_aug.shape[1]:
                        C_aug[i, full_idx] = 1.0
                    else:
                        print(f"Warning: Trend column index {full_idx} out of bounds for C_aug")
            
            # Construct augmented D matrix (D_aug) for direct shock effects
            D_aug = np.zeros((n_potential_obs, n_aug_shocks))
            
            # Base part of D_aug (for base shocks)
            if base_D is not None and base_D.shape[1] == n_base_shocks:
                D_aug[:, :n_base_shocks] = base_D
            
            # Construct observation selection matrix H
            n_observed = len(self.observed_vars)
            H = np.zeros((n_observed, n_potential_obs))
            
            # Set up H to select observed variables
            try:
                for i, obs_var in enumerate(self.observed_vars):
                    if obs_var in base_labels:
                        var_idx = base_labels.index(obs_var)
                        H[i, var_idx] = 1.0
                    else:
                        print(f"Warning: Observed variable '{obs_var}' not found in potential observables")
            except Exception as e:
                print(f"Error building H matrix: {e}")
            
            # --------- Store Results ---------
            self.augmented['A'] = A_aug
            self.augmented['B'] = B_aug
            self.augmented['C'] = C_aug
            self.augmented['D'] = D_aug
            self.augmented['H'] = H
            self.augmented['state_labels'] = base_state_labels + trend_state_labels
            self.augmented['shock_labels'] = base_shock_labels + trend_shock_labels
            self.augmented['observable_labels'] = self.observed_vars
            self.augmented['n_states'] = n_aug_states
            self.augmented['n_shocks'] = n_aug_shocks
            self.augmented['n_observed'] = n_observed
            self.augmented['n_base_states'] = n_base_states
            self.augmented['n_trend_states'] = n_trend_states
            self.augmented['n_base_shocks'] = n_base_shocks
            
            # Store trend mapping for later use
            self.augmented['trend_level_state_map'] = trend_level_state_map
            
            # Verify the base block of A_aug matches base_A
            print("\n--- Verifying A_aug vs base_A ---")
            aug_A_block = self.augmented['A'][:n_base_states, :n_base_states]
            if aug_A_block.shape == base_A.shape:
                if np.allclose(aug_A_block, base_A):
                    print("OK: Base block of A_aug matches input base_A")
                else:
                    print("ERROR: Base block of A_aug does not match input base_A!")
                    diff_norm = np.linalg.norm(aug_A_block - base_A)
                    print(f"Difference norm: {diff_norm}")
            else:
                print("ERROR: Shape mismatch between A_aug base block and input base_A")
            
            # Print effective observation equation
            print("\n--- Checking Observation Mapping ---")
            if H.shape[1] == C_aug.shape[0]:
                C_obs_eff = H @ C_aug
                D_obs_eff = H @ D_aug
                print(f"Effective observation matrix (H @ C_aug) shape: {C_obs_eff.shape}")
                print(f"Effective direct effect matrix (H @ D_aug) shape: {D_obs_eff.shape}")
                
                # Check each observed variable
                for i, obs_var in enumerate(self.observed_vars):
                    print(f"\nVariable: {obs_var} (Row {i} in observed list)")
                    if obs_var in base_labels:
                        base_idx = base_labels.index(obs_var)
                        
                        # Print coefficients for this variable
                        eff_row = C_obs_eff[i]
                        print(f"  Base coefficients: {eff_row[:n_base_states]}")
                        print(f"  Trend coefficients: {eff_row[n_base_states:]}")
                        
                        # Check if variable has a trend
                        if obs_var in trend_level_state_map:
                            level_idx = trend_level_state_map[obs_var]
                            full_idx = n_base_states + level_idx
                            coeff = eff_row[full_idx]
                            print(f"  Coefficient on own trend: {coeff}")
                            if not np.isclose(coeff, 1.0):
                                print("  WARNING: Coefficient on own trend is not 1.0!")
                    else:
                        print(f"  Variable not found in base model observables")
            else:
                print(f"ERROR: Dimension mismatch in H@C_aug ({H.shape[1]} vs {C_aug.shape[0]})")
            
            print("--- Augmented State Space Construction Complete ---")
            
        except Exception as e:
            print(f"Error building augmented state space: {e}")
            import traceback
            traceback.print_exc()

    def update_parameters(self, new_base_ss):
        """
        Updates the base model part of the augmented system with new parameters.
        
        Args:
            new_base_ss: Updated base state space system
        """
        if new_base_ss is None:
            print("Warning: Cannot update parameters with None base_ss")
            return
        
        # Get dimensions
        n_base_states = self.augmented.get('n_base_states', 0)
        n_aug_states = self.augmented.get('n_states', 0)
        n_base_shocks = self.augmented.get('n_base_shocks', 0)
        
        # Verify dimensions match
        if n_base_states != new_base_ss['A'].shape[0]:
            print("Warning: Base state count mismatch during parameter update. Rebuilding system.")
            self.base_ss = new_base_ss
            self._build_augmented_system()
            return
        
        # Update A matrix (base part)
        self.augmented['A'][:n_base_states, :n_base_states] = new_base_ss['A']
        
        # Update B matrix (base part)
        new_n_shocks = new_base_ss['B'].shape[1]
        if new_n_shocks <= self.augmented['B'].shape[1]:
            self.augmented['B'][:n_base_states, :new_n_shocks] = new_base_ss['B']
        else:
            print("Warning: New base has more shocks than augmented B can accommodate")
        
        # Update C matrix (base part)
        if 'C' in new_base_ss and new_base_ss['C'].shape[1] == n_base_states:
            self.augmented['C'][:, :n_base_states] = new_base_ss['C']
        
        # Update D matrix (base part) for direct effects
        if 'D' in new_base_ss and new_base_ss['D'].shape[1] == n_base_shocks:
            self.augmented['D'][:, :n_base_shocks] = new_base_ss['D']
        
        # Update base_ss reference
        self.base_ss = new_base_ss

    def impulse_response(self, shock_name, shock_size=1.0, periods=40):
        """
        Calculate IRFs for the augmented model, returning observed variable responses.
        
        Args:
            shock_name: Name of the shock to apply
            shock_size: Size of the shock (default=1.0)
            periods: Number of periods to simulate (default=40)
            
        Returns:
            pd.DataFrame: DataFrame with IRF results for observed variables
        """
        print(f"\n--- Calculating Augmented IRF for {shock_name} ---")
        
        if not self.augmented or 'A' not in self.augmented:
            print("Error: Augmented system not built or is invalid")
            return None
        
        try:
            # Extract augmented matrices
            A_aug = self.augmented['A']
            B_aug = self.augmented['B']
            H = self.augmented['H']
            C_aug = self.augmented['C']
            D_aug = self.augmented['D']
            
            # Get dimensions and labels
            shock_labels = self.augmented['shock_labels']
            n_aug_states = self.augmented['n_states']
            n_observed = self.augmented['n_observed']
            obs_labels = self.augmented['observable_labels']
            n_base_states = self.augmented.get('n_base_states', 0)
            n_base_shocks = self.augmented.get('n_base_shocks', 0)
            trend_level_state_map = self.augmented.get('trend_level_state_map', {})
            
            # Compute effective observation matrices
            if H.shape[1] == C_aug.shape[0] and H.shape[1] == D_aug.shape[0]:
                C_obs = H @ C_aug
                D_obs = H @ D_aug
            else:
                print(f"Error: Dimension mismatch in observation matrices")
                return None
            
            # Find shock index
            try:
                shock_idx = shock_labels.index(shock_name)
                print(f"Shock index: {shock_idx} (out of {len(shock_labels)})")
                is_base_shock = shock_idx < n_base_shocks
                shock_type = "base" if is_base_shock else "trend"
                print(f"Shock type: {shock_type}")
            except ValueError:
                print(f"Error: Shock '{shock_name}' not found in shock labels: {shock_labels}")
                return None
            
            # Initialize arrays for IRFs
            x_irf = np.zeros((periods, n_aug_states))
            y_irf = np.zeros((periods, n_observed))
            
            # Create shock vector
            e0 = np.zeros(len(shock_labels))
            e0[shock_idx] = shock_size
            
            # Period 0: Direct shock impact 
            # For period 0, y_0 = D_obs * e_0
            # This captures the direct effect of shocks on observables
            y_irf[0] = D_obs @ e0
            
            # Compute first period state from shock
            # x_1 = B_aug * e_0
            x_irf[1] = B_aug @ e0
            
            # Period 1: First period response
            # For period 1, y_1 = C_obs * x_1
            y_irf[1] = C_obs @ x_irf[1]
            
            # Propagate dynamics for remaining periods
            for t in range(2, periods):
                # State transition: x_t = A_aug * x_{t-1}
                x_irf[t] = A_aug @ x_irf[t-1]
                
                # Observation equation: y_t = C_obs * x_t
                y_irf[t] = C_obs @ x_irf[t]
            
            # Create DataFrame with IRF results
            irf_df = pd.DataFrame(y_irf, columns=obs_labels)
            irf_df.index.name = "Period"
            irf_df.attrs['shock_name'] = shock_name
            
            # Print summary
            print("\nIRF Summary:")
            print(f"Maximum impact values:")
            for var in obs_labels:
                var_idx = obs_labels.index(var)
                max_abs = np.max(np.abs(y_irf[:, var_idx]))
                max_idx = np.argmax(np.abs(y_irf[:, var_idx]))
                print(f"  {var}: {max_abs:.6f} at period {max_idx}")
            
            return irf_df
            
        except Exception as e:
            print(f"Error calculating augmented IRF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_kalman_matrices(self):
        """
        Returns matrices needed for a standard Kalman filter.
        
        Returns:
            tuple: (A, C, Q) matrices for the Kalman filter
        """
        if not self.augmented or 'A' not in self.augmented:
            raise RuntimeError("Augmented system has not been built")
        
        # Get state transition matrix
        A = self.augmented['A']
        
        # Get observation matrix
        H = self.augmented['H']
        C_aug = self.augmented['C']
        if H.shape[1] != C_aug.shape[0]:
            raise ValueError(f"Dimension mismatch for H@C_aug ({H.shape[1]} vs {C_aug.shape[0]})")
        C_obs = H @ C_aug
        
        # Compute state covariance matrix
        B = self.augmented['B']
        if B.size > 0:
            Q = B @ B.T
        else:
            Q = np.zeros((A.shape[0], A.shape[0]))
        
        return A, C_obs, Q
        
    def simulate(self, periods=100, shocks=None, initial_state=None):
        """
        Simulate the augmented state space model for a given number of periods.
        
        Args:
            periods: Number of periods to simulate (default=100)
            shocks: Dictionary mapping shock names to time series of values,
                   or None to use random normal shocks
            initial_state: Initial state vector, or None for zero initial state
            
        Returns:
            pd.DataFrame: DataFrame with simulated observable variables
        """
        if not self.augmented or 'A' not in self.augmented:
            print("Error: Augmented system has not been built")
            return None
        
        try:
            # Extract system matrices
            A = self.augmented['A']
            B = self.augmented['B']
            H = self.augmented['H']
            C = self.augmented['C']
            D = self.augmented['D']
            
            # Get dimensions
            n_states = self.augmented['n_states']
            n_shocks = self.augmented['n_shocks']
            n_obs = self.augmented['n_observed']
            
            # Labels
            shock_labels = self.augmented['shock_labels']
            obs_labels = self.augmented['observable_labels']
            
            # Setup initial state
            if initial_state is None:
                x = np.zeros(n_states)
            else:
                x = np.array(initial_state).flatten()
                if len(x) != n_states:
                    print(f"Error: Initial state length {len(x)} doesn't match required {n_states}")
                    return None
            
            # Setup storage for results
            states = np.zeros((periods, n_states))
            obs = np.zeros((periods, n_obs))
            shock_values = np.zeros((periods, n_shocks))
            
            # Generate or use provided shocks
            if shocks is None:
                # Generate random shocks
                shock_values = np.random.normal(0, 1, (periods, n_shocks))
            else:
                # Use provided shocks
                for shock_name, values in shocks.items():
                    if shock_name in shock_labels and len(values) >= periods:
                        shock_idx = shock_labels.index(shock_name)
                        shock_values[:, shock_idx] = values[:periods]
                    else:
                        print(f"Warning: Shock {shock_name} not found or has insufficient values")
            
            # Simulate model
            for t in range(periods):
                # Get current shocks
                e_t = shock_values[t]
                
                # Store current state
                states[t] = x
                
                # Compute observation
                C_obs = H @ C
                D_obs = H @ D
                obs[t] = C_obs @ x + D_obs @ e_t
                
                # Update state
                x = A @ x + B @ e_t
            
            # Create DataFrames
            obs_df = pd.DataFrame(obs, columns=obs_labels)
            states_df = pd.DataFrame(states, columns=self.augmented['state_labels'])
            shocks_df = pd.DataFrame(shock_values, columns=shock_labels)
            
            # Return complete results
            return {
                'observables': obs_df,
                'states': states_df,
                'shocks': shocks_df
            }
            
        except Exception as e:
            print(f"Error simulating model: {e}")
            import traceback
            traceback.print_exc()
            return None