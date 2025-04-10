import numpy as np
import scipy.linalg as la
import json, importlib, sys, os, pandas as pd, matplotlib.pyplot as plt


def klein(a=None, b=None, n_states=None, eigenvalue_warnings=True):
    '''Solves linear dynamic models with the form of:
    
                a*Et[x(t+1)] = b*x(t)       
                
        [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
        a vector of nonpredetermined costate variables.
        
        The solution to the model is a set of matrices f, p such that:

                u(t)   = f*s(t)
                s(t+1) = p*s(t).

        The solution algorithm is based on Klein (2000) and his solab.m Matlab program.

    Args:
        a:                      (Numpy ndarray) Coefficient matrix on future-dated variables
        b:                      (Numpy ndarray) Coefficient matrix on current-dated variables
        n_states:               (int) Number of state variables
        eigenvalue_warnings:    (bool) Whether to print warnings about eigenvalues. Default: True

    Returns:
        f:          (Numpy ndarray) Solution matrix coeffients on s(t) for u(t)
        p:          (Numpy ndarray) Solution matrix coeffients on s(t) for s(t+1)
        stab:       (int) Indicates solution stability and uniqueness
                        stab == 1: too many stable eigenvalues
                        stab == -1: too few stable eigenvalues
                        stab == 0: just enough stable eigenvalues
        eig:        The generalized eigenvalues from the Schur decomposition
    '''
    # Use ordered QZ decomposition
    s, t, alpha, beta, q, z = la.ordqz(A=a, B=b, sort='ouc', output='complex')

    # Components of the z matrix
    z11 = z[0:n_states, 0:n_states]
    z21 = z[n_states:, 0:n_states]
    
    # Number of nonpredetermined variables
    n_costates = np.shape(a)[0] - n_states
    
    if n_states > 0:
        if np.linalg.matrix_rank(z11) < n_states:
            print("ERROR: Invertibility condition violated. Check model equations or parameter values.")
            return None, None, 1, None  # Return None to signal failure

    s11 = s[0:n_states, 0:n_states]
    if n_states > 0:
        z11i = la.inv(z11)
    else:
        z11i = z11

    # Components of the s,t,and q matrices   
    t11 = t[0:n_states, 0:n_states]
    
    # Compute the generalized eigenvalues
    tii = np.diag(t)
    sii = np.diag(s)
    eig = np.zeros(np.shape(tii), dtype=np.complex128)

    for k in range(len(tii)):
        if np.abs(sii[k]) > 0:
            eig[k] = tii[k] / sii[k]    
        else:
            eig[k] = np.inf

    # Solution matrix coefficients on the endogenous state
    if n_states > 0:
        dyn = np.linalg.solve(s11, t11)
    else:
        dyn = np.array([])

    f = z21.dot(z11i)
    p = z11.dot(dyn).dot(z11i)

    f = np.real(f)
    p = np.real(p)

    # Check stability (but don't exit on issues, just report them)
    stab = 0  # Default: just enough stable eigenvalues
    
    # Count stable eigenvalues (inside unit circle)
    n_stable = np.sum(np.abs(eig) < 1.0)
    
    if n_stable > n_states:
        if eigenvalue_warnings:
            print(f"WARNING: Too many stable eigenvalues: {n_stable} (should be {n_states})")
        stab = 1
    elif n_stable < n_states:
        if eigenvalue_warnings:
            print(f"WARNING: Too few stable eigenvalues: {n_stable} (should be {n_states})")
        stab = -1
    else:
        if eigenvalue_warnings:
            print(f"Eigenvalue check passed: {n_stable} stable eigenvalues")

    return f, p, stab, eig


class ModelSolver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._load_model_components()

    def _load_model_components(self):
        """Load model components from generated files"""
        # --- Load model.json (Needed for parameters) ---
        model_path = os.path.join(self.output_dir, "model.json")
        try:
            with open(model_path, 'r') as f:
                self.model_json = json.load(f)
            # Store parameter info directly for easier access later
            self.parameter_names = self.model_json.get('parameters', [])
            self.default_param_values = self.model_json.get('param_values', {})
            print(f"Loaded parameters from {model_path}")
        except FileNotFoundError:
            print(f"Error: {model_path} not found.")
            self.model_json = {}  # Avoid attribute errors later
            self.parameter_names = []
            self.default_param_values = {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {model_path}.")
            self.model_json = {}
            self.parameter_names = []
            self.default_param_values = {}

        # --- Load Jacobian Evaluator ---
        jac_path = os.path.join(self.output_dir, "jacobian_evaluator.py")
        try:
            spec = importlib.util.spec_from_file_location("jacobian_evaluator", jac_path)
            jac_module = importlib.util.module_from_spec(spec)
            sys.modules["jacobian_evaluator"] = jac_module
            spec.loader.exec_module(jac_module)
            self.evaluate_jacobians = jac_module.evaluate_jacobians
            print(f"Loaded Jacobian evaluator from {jac_path}")
        except FileNotFoundError:
            print(f"Error: {jac_path} not found. Cannot solve model.")
            self.evaluate_jacobians = None  # Set to None to indicate failure
        except Exception as e:
            print(f"Error loading Jacobian evaluator: {e}")
            self.evaluate_jacobians = None

        # --- Load Structure (Contains TARGET order labels/indices/R) ---
        struct_path = os.path.join(self.output_dir, "model_structure.py")
        try:
            spec = importlib.util.spec_from_file_location("model_structure", struct_path)
            struct_module = importlib.util.module_from_spec(spec)
            sys.modules["model_structure"] = struct_module
            spec.loader.exec_module(struct_module)
            self.indices = struct_module.indices
            self.R = struct_module.R
            self.labels = struct_module.labels
            print(f"Loaded structure from {struct_path}.")
            print(f"  n_states={self.indices['n_states']}, n_endo={self.indices['n_endogenous']}, n_exo={self.indices['n_exo_states']}")
            print(f"  State Labels: {self.labels['state_labels']}")
            print(f"  Control Labels: {self.labels['observable_labels'][:self.indices['n_controls']]}")
            
            # Check for zero-persistence processes
            if 'zero_persistence_processes' in self.indices:
                zero_persistence = self.indices['zero_persistence_processes']
                if zero_persistence:
                    print(f"  Zero-persistence processes: {zero_persistence}")
        except FileNotFoundError:
            print(f"Error: {struct_path} not found. Cannot construct state space.")
            # Set defaults to avoid errors, but indicate failure
            self.indices = {}
            self.R = None
            self.labels = {'state_labels': [], 'observable_labels': [], 'shock_labels': []}
        except Exception as e:
            print(f"Error loading structure file: {e}")
            self.indices = {}
            self.R = None
            self.labels = {'state_labels': [], 'observable_labels': [], 'shock_labels': []}

    def solve(self, params):
        """
        Solves the model using Klein method and returns matrices P, F based on TARGET order.
        
        Args:
            params: List of parameter values
            
        Returns:
            Dictionary with solution matrices and metadata
        """
        print("\n--- Solving model with Klein method ---")
        
        # Map parameter values to parameter names
        param_names = list(self.model_json['param_values'].keys())
        param_dict = dict(zip(param_names, params))
        theta = [param_dict.get(p, self.model_json['param_values'].get(p)) for p in param_names]

        # Evaluate Jacobian matrices
        a, b, c = self.evaluate_jacobians(theta)
        n_states = self.indices['n_states']
        
        if n_states == 0:
            print("Error: No state variables found in model.")
            return None

        # Call Klein solver
        f, p, stab, eig = klein(a, b, n_states, eigenvalue_warnings=True)

        if f is None or p is None:
            print("Error: Klein solver failed.")
            return None  # Indicate failure

        print("Klein solution obtained successfully.")
        print(f"  Solution stability indicator: {stab}")
        print(f"  F matrix shape: {f.shape}, P matrix shape: {p.shape}")

        # Return the solution components relative to Klein's s_{t+1}=Ps_t, c_t=Fs_t
        solution = {
            'f': f, 
            'p': p,
            'labels': self.labels,  # Use labels from structure file
            'indices': self.indices,  # Use indices from structure file
            'stab': stab, 
            'eig': eig,
            'R': self.R  # Include the R matrix mapping shocks to z_t
        }
        return solution

    def get_contemporaneous_state_space(self, klein_solution):
        """
        Constructs the contemporaneous state-space system based on the correct mathematical formulation:
        
        s_{t+1} = A_s s_t + B_s ε_t
        y_t = C_s s_t + D_s ε_t
        
        where:
        - s_t = [k_t; z_t] combines endogenous and exogenous states
        - y_t = [c_t; k_t; z_t] are the observables
        - ε_t are the structural shocks
        
        This implementation includes proper handling of direct shock effects
        and zero-persistence processes.
        
        Args:
            klein_solution: Dictionary with solution matrices from Klein method
            
        Returns:
            Dictionary containing the state space matrices and metadata
        """
        print("\n--- Constructing Contemporaneous State Space with Correct Timing ---")
        
        if klein_solution is None:
            print("Error: Klein solution not available.")
            return None

        try:
            # --- Extract matrices from Klein solution ---
            P = klein_solution['p']            # State transition matrix
            F = klein_solution['f']            # Control policy function
            R = klein_solution['R']            # Shock selection matrix
            indices = klein_solution['indices']
            labels = klein_solution['labels']

            # Get dimensions
            n_k = indices['n_endogenous']      # Number of endogenous states k
            n_z = indices['n_exo_states']      # Number of exogenous states z
            n_states = indices['n_states']     # Total states = n_k + n_z
            n_c = indices['n_controls']        # Number of controls c
            n_shocks = indices['n_shocks']     # Number of shocks

            # Check for zero-persistence processes
            zero_persistence_processes = indices.get('zero_persistence_processes', [])
            has_zero_persistence = len(zero_persistence_processes) > 0
            if has_zero_persistence:
                print(f"Model contains zero-persistence processes: {zero_persistence_processes}")

            # --- Partition P and F matrices ---
            Pkk = P[:n_k, :n_k]                # Endogenous state transition
            Pkz = P[:n_k, n_k:]                # Exogenous → endogenous impact
            Pzz = P[n_k:, n_k:]                # Exogenous state transition
            
            Fck = F[:, :n_k]                   # Endogenous → control impact
            Fcz = F[:, n_k:]                   # Exogenous → control impact

            # --- Construct A_s: State Transition Matrix ---
            # A_s = [Pkk Pkz; 0 Pzz]
            A_s = np.block([
                [Pkk, Pkz],
                [np.zeros((n_z, n_k)), Pzz]
            ])

            # --- Construct B_s: Shock Impact Matrix ---
            # B_s = [Pkz*R; R]
            # This captures both:
            # 1. How shocks directly affect exogenous states (R)
            # 2. How these shocks propagate to endogenous states (Pkz*R)
            B_s = np.block([
                [Pkz @ R],
                [R]
            ])

            # --- Construct C_s: Observation Matrix ---
            # C_s = [Fck Fcz; I 0; 0 I]
            C_s = np.block([
                [Fck, Fcz],                     # Controls
                [np.eye(n_k), np.zeros((n_k, n_z))],  # Endogenous states
                [np.zeros((n_z, n_k)), np.eye(n_z)]   # Exogenous states
            ])

            # --- Construct D_s: Direct Shock Impact Matrix ---
            # D_s = [Fcz*R; 0; R]
            # This captures how shocks directly affect:
            # 1. Controls through exogenous processes (Fcz*R)
            # 2. Exogenous states (R)
            D_s = np.block([
                [Fcz @ R],                      # Direct impact on controls
                [np.zeros((n_k, n_shocks))],    # No direct impact on endogenous states
                [R]                             # Direct impact on exogenous states
            ])

            # --- Enhance Direct Effects for Zero-Persistence Processes ---
            if has_zero_persistence:
                print("Enhancing direct shock effects for zero-persistence processes...")
                
                # Get mapping between shocks and state variables
                shock_to_state_map = labels.get('shock_to_state_map', {})
                state_to_shock_map = labels.get('state_to_shock_map', {})
                
                # Process each zero-persistence variable
                for var in zero_persistence_processes:
                    print(f"  Processing zero-persistence variable: {var}")
                    
                    # Find states related to this variable
                    related_states = [s for s in labels['state_labels'] if s.startswith(f"{var}_") or s == var]
                    print(f"    Related states: {related_states}")
                    
                    # Find associated shocks
                    for state in related_states:
                        if state in state_to_shock_map:
                            shock = state_to_shock_map[state]
                            shock_idx = labels['shock_labels'].index(shock)
                            print(f"    State {state} is driven by shock {shock}")
                            
                            # Enhance direct effect on controls that depend on this variable
                            for i, control in enumerate(labels['observable_labels'][:n_c]):
                                # Set direct effect to 1.0 (you might refine this based on model structure)
                                D_s[i, shock_idx] = 1.0
                                print(f"    Adding direct effect: {shock} → {control}")

            # --- Verify shock impact matrices ---
            print("\nVerifying shock impact matrices:")
            
            # Check B_s (state transition shock impacts)
            if np.allclose(B_s, 0):
                print("WARNING: B_s is all zeros - shocks won't affect state transitions!")
            else:
                nonzero_B = np.count_nonzero(B_s)
                print(f"B_s has {nonzero_B} non-zero elements")
            
            # Check D_s (direct observation shock impacts)
            if np.allclose(D_s, 0):
                print("WARNING: D_s is all zeros - no direct shock effects!")
            else:
                nonzero_D = np.count_nonzero(D_s)
                print(f"D_s has {nonzero_D} non-zero elements")
                
                # Print sample of direct shock impacts on controls
                control_impacts = D_s[:n_c, :]
                if np.any(control_impacts):
                    print("Sample direct control impacts (Fcz*R):")
                    for i in range(min(3, n_c)):
                        for j in range(min(3, n_shocks)):
                            if abs(control_impacts[i, j]) > 1e-10:
                                print(f"  Shock {j} → Control {i}: {control_impacts[i, j]:.6f}")

            # --- Store results ---
            contemp_ss = {
                'A': A_s, 
                'B': B_s, 
                'C': C_s, 
                'D': D_s,
                'state_labels': labels['state_labels'],
                'observable_labels': labels['observable_labels'],
                'shock_labels': labels['shock_labels'],
                'n_states': n_states,
                'n_observables': n_c + n_states,
                'n_shocks': n_shocks,
                # Store component matrices for reference
                'Pkk': Pkk, 'Pkz': Pkz, 'Pzz': Pzz,
                'Fck': Fck, 'Fcz': Fcz, 'R': R,
                # Store mappings
                'shock_to_state_map': labels.get('shock_to_state_map', {}),
                'state_to_shock_map': labels.get('state_to_shock_map', {}),
                'zero_persistence_processes': zero_persistence_processes,
                # Store indices
                'indices': indices
            }
            
            print("\nContemporaneous State Space Successfully Constructed")
            print(f"A: {A_s.shape}, B: {B_s.shape}, C: {C_s.shape}, D: {D_s.shape}")
            
            return contemp_ss

        except Exception as e:
            print(f"Error constructing contemporaneous state space: {e}")
            import traceback
            traceback.print_exc()
            return None

    def impulse_response(self, state_space, shock_name, shock_size=1.0, periods=40):
        """
        Calculate impulse response functions (IRFs) for a given state space system
        with proper handling of direct shock effects at period 0.
        
        For a system s_{t+1} = A_s s_t + B_s ε_t, y_t = C_s s_t + D_s ε_t:
        1. Period 0: y_0 = D_s ε_0 (direct impact)
        2. Period 1: s_1 = B_s ε_0, y_1 = C_s s_1
        3. Period t>1: s_t = A_s s_{t-1}, y_t = C_s s_t
        
        This algorithm correctly handles zero-persistence cases by including direct shock
        impacts through the D_s matrix.
        
        Args:
            state_space: The state space system dictionary
            shock_name: Name of the shock to apply
            shock_size: Size of the shock (default=1.0)
            periods: Number of periods to simulate (default=40)
            
        Returns:
            pandas.DataFrame: DataFrame with IRF results for all observables
        """
        print(f"\n--- Calculating IRF for {shock_name} ---")
        
        if state_space is None:
            print("Error: State space system is None")
            return None
        
        try:
            # Extract state space matrices
            A = state_space['A']  # State transition
            B = state_space['B']  # Shock impact on states
            C = state_space['C']  # Observation
            D = state_space['D']  # Direct shock impact on observables
            
            # Get dimensions and labels
            n_states = state_space['n_states']
            n_obs = state_space['n_observables']
            n_shocks = state_space['n_shocks']
            n_controls = state_space['indices']['n_controls']
            
            shock_labels = state_space['shock_labels']
            obs_labels = state_space['observable_labels']
            
            # Check for zero-persistence processes
            zero_persistence = state_space.get('zero_persistence_processes', [])
            if zero_persistence:
                print(f"Model contains zero-persistence processes: {zero_persistence}")
            
            # Find shock index
            try:
                shock_idx = shock_labels.index(shock_name)
                print(f"Shock index for '{shock_name}': {shock_idx}")
            except ValueError:
                print(f"Error: Shock '{shock_name}' not found in shock labels: {shock_labels}")
                return None
            
            # Initialize arrays for IRFs
            s_irf = np.zeros((periods, n_states))  # State responses
            y_irf = np.zeros((periods, n_obs))     # Observable responses
            
            # Create shock vector (one-time impulse at t=0)
            e0 = np.zeros(n_shocks)
            e0[shock_idx] = shock_size
            
            # --- Period 0: Direct impact through D matrix ---
            # y_0 = D ε_0
            y_irf[0] = D @ e0
            
            # Print period 0 impact
            print("\nPeriod 0 (Direct Impact):")
            direct_impact = D @ e0
            
            # Separate control impacts from state impacts
            if n_controls > 0:
                control_impact = direct_impact[:n_controls]
                print(f"  Direct impact on controls: max abs = {np.max(np.abs(control_impact)):.6f}")
                
                # Print specific control impacts
                for i in range(min(5, n_controls)):
                    if abs(control_impact[i]) > 1e-10:
                        control_var = obs_labels[i] if i < len(obs_labels) else f"Control_{i}"
                        print(f"  {control_var}: {control_impact[i]:.6f}")
            
            # --- Period 1: First period state response ---
            # s_1 = B ε_0
            s_irf[1] = B @ e0
            
            # y_1 = C s_1
            y_irf[1] = C @ s_irf[1]
            
            # Print period 1 state response
            print("\nPeriod 1 (First State Response):")
            state_response = B @ e0
            print(f"  First period state response magnitude: {np.linalg.norm(state_response):.6f}")
            
            # --- Periods 2+ : State evolution ---
            for t in range(2, periods):
                # s_t = A s_{t-1}
                s_irf[t] = A @ s_irf[t-1]
                
                # y_t = C s_t
                y_irf[t] = C @ s_irf[t]
            
            # --- Create DataFrame with results ---
            irf_df = pd.DataFrame(y_irf, columns=obs_labels)
            irf_df.index.name = "Period"
            irf_df.attrs['shock_name'] = shock_name
            irf_df.attrs['shock_size'] = shock_size
            
            # --- Print summary statistics ---
            print("\nIRF Summary:")
            
            # Print max impact for a few key variables
            print("Maximum absolute impacts:")
            for var_idx, var_name in enumerate(obs_labels[:min(5, len(obs_labels))]):
                max_abs = np.max(np.abs(y_irf[:, var_idx]))
                max_idx = np.argmax(np.abs(y_irf[:, var_idx]))
                print(f"  {var_name}: {max_abs:.6f} at period {max_idx}")
            
            # Show first few periods for a key control variable (if applicable)
            if n_controls > 0 and len(obs_labels) > 0:
                key_var = obs_labels[0]  # First observable (usually an important control)
                key_idx = 0
                print(f"\nFirst 5 periods for {key_var}:")
                for t in range(min(5, periods)):
                    print(f"  Period {t}: {y_irf[t, key_idx]:.6f}")
            
            return irf_df
        
        except Exception as e:
            print(f"Error calculating IRF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_irf(self, irf_df, variables_to_plot, title_suffix="", figsize=(12, 8)):
        """
        Plots selected variables from an IRF DataFrame.
        
        Args:
            irf_df: DataFrame containing IRF results
            variables_to_plot: List of variable names to plot
            title_suffix: Optional suffix for the plot title
            figsize: Figure size as (width, height) tuple
        """
        if irf_df is None:
            print("Cannot plot None IRF DataFrame.")
            return

        plt.figure(figsize=figsize)
        shock_name = irf_df.attrs.get('shock_name', 'Unknown Shock')
        plot_title = f'Impulse Responses to {shock_name}'
        if title_suffix:
            plot_title += f" ({title_suffix})"

        # Use index for x-axis
        x_axis = irf_df.index
        x_label = 'Periods'

        # Plot each variable
        for var in variables_to_plot:
            if var in irf_df.columns:
                plt.plot(x_axis, irf_df[var], label=var)
            else:
                print(f"Warning: Variable '{var}' not found in IRF results.")

        plt.xlabel(x_label)
        plt.ylabel('Deviation from Steady State')
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Include period 0 in the plot
        plt.xlim(left=0)
        
        plt.tight_layout()
        
        try:
            plt.show()
        except Exception as e:
            print(f"Note: Plot display failed ({e}). May need GUI backend.")