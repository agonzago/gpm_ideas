import numpy as np
import scipy.linalg as la
import json, importlib, sys, os, pandas as pd, matplotlib.pyplot as plt


def klein(a=None,b=None,n_states=None,eigenvalue_warnings=True):

    '''Solves linear dynamic models with the form of:
    
                a*Et[x(t+1)] = b*x(t)       
                
        [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
        a vector of nonpredetermined costate variables. z(t) is a vector of exogenous forcing variables with 
        autocorrelation matrix phi. The solution to the model is a set of matrices f, n, p, l such that:

                u(t)   = f*s(t)
                s(t+1) = p*s(t).

        The solution algorithm is based on Klein (2000) and his solab.m Matlab program.

    Args:
        a:                      (Numpy ndarray) Coefficient matrix on future-dated variables
        b:                      (Numpy ndarray) Coefficient matrix on current-dated variables
        c:                      (Numpy ndarray) Coefficient matrix on exogenous forcing variables
        n_states:               (int) Number of state variables
        eigenvalue_warnings:    (bool) Whether to print warnings that there are too many or few eigenvalues. Default: True

    Returns:
        f:          (Numpy ndarray) Solution matrix coeffients on s(t) for u(t)
        p:          (Numpy ndarray) Solution matrix coeffients on s(t) for s(t+1)
        stab:       (int) Indicates solution stability and uniqueness
                        stab == 1: too many stable eigenvalues
                        stab == -1: too few stable eigenvalues
                        stab == 0: just enoughstable eigenvalues
        eig:        The generalized eigenvalues from the Schur decomposition

    '''

    s,t,alpha,beta,q,z = la.ordqz(A=a,B=b,sort='ouc',output='complex')

    # Components of the z matrix
    z11 = z[0:n_states,0:n_states]
    
    z21 = z[n_states:,0:n_states]
    
    # number of nonpredetermined variables
    n_costates = np.shape(a)[0] - n_states
    
    if n_states>0:
        if np.linalg.matrix_rank(z11)<n_states:
            sys.exit("Invertibility condition violated. Check model equations or parameter values.")

    s11 = s[0:n_states,0:n_states];
    if n_states>0:
        z11i = la.inv(z11)

    else:
        z11i = z11


    # Components of the s,t,and q matrices   
    t11 = t[0:n_states,0:n_states]
    # Verify that there are exactly n_states stable (inside the unit circle) eigenvalues:
    stab = 0


    # Compute the generalized eigenvalues
    tii = np.diag(t)
    sii = np.diag(s)
    eig = np.zeros(np.shape(tii),dtype=np.complex128)
    # eig = np.zeros(np.shape(tii))

    for k in range(len(tii)):
        if np.abs(sii[k])>0:
            # eig[k] = np.abs(tii[k])/np.abs(sii[k])
            eig[k] = tii[k]/sii[k]    
        else:
            eig[k] = np.inf



    # Solution matrix coefficients on the endogenous state
    if n_states>0:
            dyn = np.linalg.solve(s11,t11)
    else:
        dyn = np.array([])


    f = z21.dot(z11i)
    p = z11.dot(dyn).dot(z11i)

    f = np.real(f)
    p = np.real(p)

    return f, p,stab,eig

class ModelSolver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._load_model_components()

    def _load_model_components(self):
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
                self.model_json = {} # Avoid attribute errors later
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
                self.evaluate_jacobians = None # Set to None to indicate failure
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
                print(f"Loaded structure from {struct_path}. n_states={self.indices['n_states']}, n_endo={self.indices['n_endogenous']}, n_exo={self.indices['n_exo_states']}")
                print(f"  State Labels: {self.labels['state_labels']}")
                print(f"  Control Labels: {self.labels['observable_labels'][:self.indices['n_controls']]}")
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
        """Solves using Klein and returns matrices P, F based on TARGET order."""
        # ... (solve method remains the same as the version that forces TARGET order jacobians) ...
        # It now returns f, p corresponding to the TARGET order.
        param_names = list(self.model_json['param_values'].keys())
        param_dict = dict(zip(param_names, params))
        theta = [param_dict.get(p, self.model_json['param_values'].get(p)) for p in param_names]

        a, b, c = self.evaluate_jacobians(theta)
        n_states = self.indices['n_states']
        if n_states == 0: return None

        f, p, stab, eig = klein(a, b, n_states)

        if f is None or p is None:
            print("Error: Klein solver failed.")
            return None # Indicate failure


        print("--- ModelSolver.solve ---")
        print("f matrix (from Klein, TARGET order) (first 5x5):\n", f[:5,:5])
        print("p matrix (from Klein, TARGET order) (first 5x5):\n", p[:5,:5])

        # Return the solution components relative to Klein's s_{t+1}=Ps_t, c_t=Fs_t
        solution = {
            'f': f, 'p': p,
            'labels': self.labels, # Use labels from structure file
            'indices': self.indices, # Use indices from structure file
            'stab': stab, 'eig': eig,
            'R': self.R # Include the R matrix mapping shocks to z_t
        }
        return solution

    # def get_contemporaneous_state_space(self, klein_solution):
    #     """
    #     Constructs the contemporaneous state-space system:
    #     x_{t+1} = A x_t + B ε_t,  
    #     y_t = C x_t + D ε_t
        
    #     where x_t = [k_t; z_t] and y_t = [c_t; k_t; z_t].
        
    #     This aligns with section 3 of the state_space_specification document:
    #     k_{t+1} = P_{kk}k_t + P_{kz}z_t
    #     z_{t+1} = P_{zz}z_t + R ε_{t+1}
    #     c_t = F_{ck}k_t + F_{cz}z_t
        
    #     Returns:
    #         Dict containing the state space matrices and metadata
    #     """
    #     print("\n--- Constructing Contemporaneous State Space ---")
    #     if klein_solution is None:
    #         print("Error: Klein solution not available.")
    #         return None

    #     try:
    #         # --- Extract matrices from Klein solution (TARGET order) ---
    #         P = klein_solution['p']
    #         F = klein_solution['f']
    #         R = klein_solution['R']
    #         indices = klein_solution['indices']
    #         labels = klein_solution['labels']

    #         n_k = indices['n_endogenous']    # Number of endogenous states k
    #         n_z = indices['n_exo_states']    # Number of exogenous states z
    #         n_states = indices['n_states']   # Total states = n_k + n_z
    #         n_c = indices['n_controls']      # Number of controls c
    #         n_shocks = indices['n_shocks']

    #         # Verify dimensions
    #         if n_states != n_k + n_z:
    #             print(f"Warning: n_states ({n_states}) != n_k ({n_k}) + n_z ({n_z})")
            
    #         if R.shape != (n_z, n_shocks):
    #             print(f"Warning: R matrix shape {R.shape} doesn't match expected ({n_z}, {n_shocks})")

    #         # --- Partition P and F based on k and z ---
    #         # P = [Pkk Pkz]
    #         #     [0   Pzz]  <- Klein usually has zeros in bottom left
    #         Pkk = P[:n_k, :n_k]
    #         Pkz = P[:n_k, n_k:]
    #         Pzz = P[n_k:, n_k:]
    #         # F = [Fck Fcz]
    #         Fck = F[:, :n_k]
    #         Fcz = F[:, n_k:]

    #         # --- Compute A matrix for x_{t+1} = A x_t + B ε_t ---
    #         # A = [Pkk Pkz]
    #         #     [0   Pzz]
    #         A = np.block([
    #             [Pkk, Pkz],
    #             [np.zeros((n_z, n_k)), Pzz]
    #         ])

    #         # --- Compute B matrix for x_{t+1} = A x_t + B ε_t ---
    #         # B = [Pkz*R]
    #         #     [R    ]
    #         # Note: We need Pkz*R to reflect how z_t shocks propagate to k_{t+1}
    #         B = np.block([
    #             [Pkz @ R],
    #             [R]
    #         ])

    #         # --- Compute C matrix for y_t = C x_t + D ε_t ---
    #         # C = [Fck Fcz]
    #         #     [I   0  ]
    #         #     [0   I  ]
    #         C = np.block([
    #             [Fck, Fcz],                       # Map to c_t
    #             [np.eye(n_k), np.zeros((n_k, n_z))],  # Map to k_t
    #             [np.zeros((n_z, n_k)), np.eye(n_z)]   # Map to z_t
    #         ])

    #         # --- Compute D matrix for y_t = C x_t + D ε_t ---
    #         # D = [Fcz*R] <- Important for when shocks directly affect exo processes
    #         #     [0    ]
    #         #     [R    ]
    #         D = np.block([
    #             [Fcz @ R],                      # Direct impact on controls
    #             [np.zeros((n_k, n_shocks))],    # No direct impact on endogenous states
    #             [R]                             # Direct impact on exogenous states
    #         ])
            
    #         # Print shock impact matrices for debugging
    #         print(f"\nDirect shock impact on exogenous states (R):")
    #         print(R)
    #         print(f"\nDirect shock impact on controls (Fcz*R):")
    #         print(Fcz @ R)

    #         # --- Define Labels for this system ---
    #         state_labels = labels['state_labels'][:n_k] + labels['state_labels'][n_k:]
    #         obs_labels = list(labels['observable_labels'][:n_c]) + list(labels['state_labels'][:n_k]) + list(labels['state_labels'][n_k:])
    #         shock_labels = labels['shock_labels']

    #         # --- Store Results ---
    #         contemp_ss = {
    #             'A': A, 
    #             'B': B, 
    #             'C': C, 
    #             'D': D,
    #             'state_labels': state_labels,
    #             'observable_labels': obs_labels,
    #             'shock_labels': shock_labels,
    #             'n_states': n_states,
    #             'n_observables': n_c + n_k + n_z,
    #             'n_shocks': n_shocks,
    #             # Include component matrices for reference
    #             'Pkk': Pkk, 'Pkz': Pkz, 'Pzz': Pzz,
    #             'Fck': Fck, 'Fcz': Fcz, 'R': R
    #         }
            
    #         print("Contemporaneous State Space Constructed Successfully")
    #         print(f"  A shape: {A.shape}, B shape: {B.shape}")
    #         print(f"  C shape: {C.shape}, D shape: {D.shape}")
            
    #         return contemp_ss

    #     except Exception as e:
    #         print(f"Error constructing contemporaneous state space: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None

    # def get_contemporaneous_state_space(self, klein_solution):
    #     """
    #     Constructs the contemporaneous state-space system:
        
    #     x_{t+1} = A x_t + B ε_t
    #     y_t = C x_t + D ε_t
        
    #     where x_t = [k_t; z_t] and y_t = [c_t; k_t; z_t].
        
    #     This follows section 3 of the state_space_specification document with
    #     special handling for zero-persistence cases.
        
    #     Args:
    #         klein_solution: Dictionary with solution matrices from Klein method
            
    #     Returns:
    #         Dictionary containing the state space matrices and metadata
    #     """
    #     print("\n--- Constructing Contemporaneous State Space ---")
    #     if klein_solution is None:
    #         print("Error: Klein solution not available.")
    #         return None

    #     try:
    #         # --- Extract matrices from Klein solution ---
    #         P = klein_solution['p']            # State transition matrix
    #         F = klein_solution['f']            # Control policy function
    #         R = klein_solution['R']            # Shock selection matrix
    #         indices = klein_solution['indices']
    #         labels = klein_solution['labels']

    #         # Get dimensions
    #         n_k = indices['n_endogenous']      # Number of endogenous states k
    #         n_z = indices['n_exo_states']      # Number of exogenous states z
    #         n_states = indices['n_states']     # Total states = n_k + n_z
    #         n_c = indices['n_controls']        # Number of controls c
    #         n_shocks = indices['n_shocks']     # Number of shocks

    #         # Get variable and shock labels
    #         state_labels = labels['state_labels']
    #         observable_labels = labels['observable_labels']
    #         shock_labels = labels['shock_labels']
            
    #         # Get mappings between shocks and states (crucial for zero persistence)
    #         shock_to_state_map = labels.get('shock_to_state_map', {})
    #         state_to_shock_map = labels.get('state_to_shock_map', {})

    #         # Verify dimensions
    #         if n_states != n_k + n_z:
    #             print(f"Warning: n_states ({n_states}) != n_k ({n_k}) + n_z ({n_z})")
            
    #         if R.shape != (n_z, n_shocks):
    #             print(f"Warning: R matrix shape {R.shape} doesn't match expected ({n_z}, {n_shocks})")

    #         # --- Partition P and F based on k and z ---
    #         # P = [Pkk Pkz]
    #         #     [0   Pzz]  <- Klein usually has zeros in bottom left
    #         Pkk = P[:n_k, :n_k]                # Endogenous → endogenous transition
    #         Pkz = P[:n_k, n_k:]                # Exogenous → endogenous impact
    #         Pzz = P[n_k:, n_k:]                # Exogenous process persistence
            
    #         # F = [Fck Fcz]
    #         Fck = F[:, :n_k]                   # Endogenous → control impact
    #         Fcz = F[:, n_k:]                   # Exogenous → control impact
            
    #         # --- Check for zero-persistence exogenous processes ---
    #         print("\nChecking for zero-persistence processes:")
            
    #         # Examine diagonal of Pzz - values near zero indicate no persistence
    #         has_zero_persistence = False
    #         for i in range(min(n_z, Pzz.shape[0], Pzz.shape[1])):
    #             ar_coef = Pzz[i, i]
    #             if abs(ar_coef) < 1e-6:  # Near zero
    #                 exo_var_name = state_labels[n_k + i] if n_k + i < len(state_labels) else f"ExoState_{i}"
    #                 print(f"  Zero persistence detected: {exo_var_name} (AR coef = {ar_coef})")
    #                 has_zero_persistence = True
            
    #         if has_zero_persistence:
    #             print("  Zero-persistence processes found - ensuring direct shock paths exist")
    #         else:
    #             print("  No zero-persistence processes detected")

    #         # --- Compute A matrix for x_{t+1} = A x_t + B ε_t ---
    #         # A = [Pkk Pkz]
    #         #     [0   Pzz]
    #         A = np.block([
    #             [Pkk, Pkz],
    #             [np.zeros((n_z, n_k)), Pzz]
    #         ])

    #         # --- Compute B matrix for x_{t+1} = A x_t + B ε_t ---
    #         # B = [Pkz*R]  <- This captures how shocks affect endogenous states
    #         #     [R    ]  <- This captures how shocks affect exogenous states
    #         B = np.block([
    #             [Pkz @ R],  # How shocks propagate to endogenous states
    #             [R]         # How shocks propagate to exogenous states
    #         ])
            
    #         # Verify B matrix has appropriate non-zero elements
    #         if not np.any(B):
    #             print("WARNING: B matrix has all zeros - shock transmission won't work!")
    #         else:
    #             print(f"B matrix has {np.count_nonzero(B)} non-zero elements")

    #         # --- Compute C matrix for y_t = C x_t + D ε_t ---
    #         # C = [Fck Fcz]  <- Maps states to controls
    #         #     [I   0  ]  <- Identity mapping for endogenous states
    #         #     [0   I  ]  <- Identity mapping for exogenous states
    #         C = np.block([
    #             [Fck, Fcz],                          # Map to controls
    #             [np.eye(n_k), np.zeros((n_k, n_z))],  # Map to endogenous states
    #             [np.zeros((n_z, n_k)), np.eye(n_z)]   # Map to exogenous states
    #         ])

    #         # --- Compute D matrix for y_t = C x_t + D ε_t ---
    #         # D = [Fcz*R]  <- Direct impact of shocks on controls (crucial for zero persistence)
    #         #     [0    ]  <- No direct impact on endogenous states
    #         #     [R    ]  <- Direct impact of shocks on exogenous states
    #         D = np.block([
    #             [Fcz @ R],                          # Direct impact on controls
    #             [np.zeros((n_k, n_shocks))],        # No direct impact on endogenous states
    #             [R]                                 # Direct impact on exogenous states
    #         ])
            
    #         # --- Additional verification for D matrix ---
    #         # Verify D matrix has appropriate non-zero elements for shock transmission
    #         if not np.any(D):
    #             print("WARNING: D matrix has all zeros - direct shock effects won't work!")
    #         else:
    #             print(f"D matrix has {np.count_nonzero(D)} non-zero elements")
                
    #             # Print the direct shock impacts on controls
    #             direct_control_impacts = D[:n_c, :]
    #             if np.any(direct_control_impacts):
    #                 print("\nDirect shock impacts on controls:")
    #                 for i in range(n_c):
    #                     for j in range(n_shocks):
    #                         if abs(direct_control_impacts[i, j]) > 1e-6:
    #                             control_var = observable_labels[i] if i < len(observable_labels) else f"Control_{i}"
    #                             shock_var = shock_labels[j] if j < len(shock_labels) else f"Shock_{j}"
    #                             print(f"  {shock_var} → {control_var}: {direct_control_impacts[i, j]}")

    #         # --- Define Labels for this system ---
    #         contemp_state_labels = state_labels[:n_k] + state_labels[n_k:]
    #         contemp_obs_labels = list(observable_labels[:n_c]) + list(state_labels[:n_k]) + list(state_labels[n_k:])
            
    #         # --- Store Results ---
    #         contemp_ss = {
    #             'A': A, 
    #             'B': B, 
    #             'C': C, 
    #             'D': D,
    #             'state_labels': contemp_state_labels,
    #             'observable_labels': contemp_obs_labels,
    #             'shock_labels': shock_labels,
    #             'n_states': n_states,
    #             'n_observables': n_c + n_k + n_z,
    #             'n_shocks': n_shocks,
    #             # Include component matrices for reference
    #             'Pkk': Pkk, 'Pkz': Pkz, 'Pzz': Pzz,
    #             'Fck': Fck, 'Fcz': Fcz, 'R': R,
    #             # Include mappings for reference
    #             'shock_to_state_map': shock_to_state_map,
    #             'state_to_shock_map': state_to_shock_map
    #         }
            
    #         print("\nContemporaneous State Space Successfully Constructed")
    #         print(f"  A: {A.shape}, B: {B.shape}, C: {C.shape}, D: {D.shape}")
            
    #         return contemp_ss

    #     except Exception as e:
    #         print(f"Error constructing contemporaneous state space: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None


    def get_contemporaneous_state_space(self, klein_solution):
        """
        Constructs the contemporaneous state-space system based on the mathematical formulation:
        
        s_{t+1} = A_s s_t + B_s ε_t
        y_t = C_s s_t + D_s ε_t
        
        where:
        - s_t = [k_t; z_t] combines endogenous and exogenous states
        - y_t = [c_t; k_t; z_t] are the observables
        - ε_t are the structural shocks
        
        Args:
            klein_solution: Dictionary with solution matrices from Klein method
            
        Returns:
            Dictionary containing the state space matrices and metadata
        """
        print("\n--- Constructing Contemporaneous State Space ---")
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
            zero_persistence_processes = labels.get('zero_persistence_processes', [])
            has_zero_persistence = len(zero_persistence_processes) > 0
            if has_zero_persistence:
                print(f"Zero persistence detected in: {zero_persistence_processes}")

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

            # --- Verify shock impact matrices ---
            print("\nVerifying shock impact matrices:")
            
            # Check B_s (state transition shock impacts)
            if np.allclose(B_s, 0):
                print("WARNING: B_s is all zeros - shocks won't affect state transitions!")
            else:
                nonzero_B = np.count_nonzero(B_s)
                print(f"B_s has {nonzero_B} non-zero elements")
                
                # Print sample of shock impacts
                # For endogenous states
                endo_impacts = B_s[:n_k, :]
                if np.any(endo_impacts):
                    print("Sample endogenous state impacts (Pkz*R):")
                    for i in range(min(3, n_k)):
                        for j in range(min(3, n_shocks)):
                            if abs(endo_impacts[i, j]) > 1e-10:
                                print(f"  Shock {j} → Endo State {i}: {endo_impacts[i, j]:.6f}")
                
                # For exogenous states
                exo_impacts = B_s[n_k:, :]
                if np.any(exo_impacts):
                    print("Sample exogenous state impacts (R):")
                    for i in range(min(3, n_z)):
                        for j in range(min(3, n_shocks)):
                            if abs(exo_impacts[i, j]) > 1e-10:
                                print(f"  Shock {j} → Exo State {i}: {exo_impacts[i, j]:.6f}")
            
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

            # --- Define labels for this system ---
            state_labels = labels['state_labels']
            obs_labels = labels['observable_labels']
            shock_labels = labels['shock_labels']

            # --- Store results ---
            contemp_ss = {
                'A': A_s, 
                'B': B_s, 
                'C': C_s, 
                'D': D_s,
                'state_labels': state_labels,
                'observable_labels': obs_labels,
                'shock_labels': shock_labels,
                'n_states': n_states,
                'n_observables': n_c + n_states,
                'n_shocks': n_shocks,
                # Store component matrices for reference
                'Pkk': Pkk, 'Pkz': Pkz, 'Pzz': Pzz,
                'Fck': Fck, 'Fcz': Fcz, 'R': R,
                # Store mappings
                'shock_to_state_map': labels.get('shock_to_state_map', {}),
                'state_to_shock_map': labels.get('state_to_shock_map', {}),
                'zero_persistence_processes': zero_persistence_processes
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
        Calculate impulse response functions (IRFs) for a given state space system.
        
        For a system s_{t+1} = A_s s_t + B_s ε_t, y_t = C_s s_t + D_s ε_t:
        1. Period 0: y_0 = D_s ε_0 (direct impact)
        2. Period 1: s_1 = B_s ε_0, y_1 = C_s s_1
        3. Period t>1: s_t = A_s s_{t-1}, y_t = C_s s_t
        
        This algorithm properly handles zero-persistence cases by including direct shock
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
            n_controls = state_space.get('n_controls', 0)
            
            shock_labels = state_space['shock_labels']
            obs_labels = state_space['observable_labels']
            state_labels = state_space['state_labels']
            
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
                print(f"  Direct impact on controls: {control_impact}")
                
                # Print specific control impacts
                for i in range(min(3, n_controls)):
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
    # def impulse_response(self, state_space, shock_name, shock_size=1.0, periods=40):
    #     """
    #     Calculate impulse response functions (IRFs) for a given state space system.
        
    #     For a system x_{t+1} = A x_t + B ε_t, y_t = C x_t + D ε_t:
    #     1. At t=0: x_0 = 0, ε_0 = [0,...,shock_size,...,0], y_0 = D ε_0
    #     2. At t=1: x_1 = B ε_0, y_1 = C x_1 + D ε_1 (where ε_1=0)
    #     3. At t=2: x_2 = A x_1, y_2 = C x_2 + D ε_2 (where ε_2=0)
    #     ...and so on.
        
    #     Returns:
    #         pandas.DataFrame: DataFrame with IRFs for all observables
    #     """
    #     print(f"\n--- Calculating IRF for {shock_name} ---")
        
    #     if state_space is None:
    #         print("Error: State space system is None")
    #         return None
        
    #     try:
    #         # Extract state space matrices
    #         A = state_space['A']
    #         B = state_space['B']
    #         C = state_space['C']
    #         D = state_space['D']
            
    #         # Get dimensions and labels
    #         n_states = state_space['n_states']
    #         n_obs = state_space['n_observables']
    #         n_shocks = state_space['n_shocks']
    #         shock_labels = state_space['shock_labels']
    #         obs_labels = state_space['observable_labels']
            
    #         # Find shock index
    #         try:
    #             shock_idx = shock_labels.index(shock_name)
    #             print(f"Shock index for {shock_name}: {shock_idx}")
    #         except ValueError:
    #             print(f"Error: Shock '{shock_name}' not found in shock labels: {shock_labels}")
    #             return None
            
    #         # Initialize arrays for IRFs
    #         x_irf = np.zeros((periods, n_states))  # State responses
    #         y_irf = np.zeros((periods, n_obs))     # Observable responses
            
    #         # Create shock vector at t=0
    #         e0 = np.zeros(n_shocks)
    #         e0[shock_idx] = shock_size
            
    #         # Extract components for debugging
    #         n_controls = len(state_space['observable_labels']) - n_states
    #         D_controls = D[:n_controls, :]
    #         D_states = D[n_controls:, :]
            
    #         # Print shock impact vectors for debugging
    #         print(f"Shock vector e0: {e0}")
    #         print(f"Direct impact on controls (D_controls @ e0): {D_controls @ e0}")
    #         print(f"Direct impact on states (D_states @ e0): {D_states @ e0}")
    #         print(f"Impact on next period states (B @ e0): {B @ e0}")
            
    #         # Period 0 (impact) response 
    #         # y_0 = D ε_0
    #         y_irf[0] = D @ e0
            
    #         # Period 1 state is affected by shock at t=0
    #         # x_1 = B ε_0
    #         x_irf[1] = B @ e0
            
    #         # Output at period 1 combines state effect and any lingering shock effect
    #         # y_1 = C x_1 (no shock at t=1)
    #         y_irf[1] = C @ x_irf[1]
            
    #         # Propagate the shock for remaining periods
    #         for t in range(2, periods):
    #             x_irf[t] = A @ x_irf[t-1]
    #             y_irf[t] = C @ x_irf[t]
            
    #         # Create DataFrame with IRF results
    #         irf_df = pd.DataFrame(y_irf, columns=obs_labels)
    #         irf_df.index.name = "Period"
    #         irf_df.attrs['shock_name'] = shock_name
            
    #         # Print first few periods of IRF for key variables
    #         print("\nIRF first 5 periods:")
    #         control_vars = obs_labels[:n_controls]
    #         for var in control_vars[:min(4, len(control_vars))]:  # Print max 4 control variables
    #             var_idx = obs_labels.index(var)
    #             values = y_irf[:5, var_idx]
    #             print(f"{var}: {values}")
            
    #         return irf_df
        
    #     except Exception as e:
    #         print(f"Error calculating IRF: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None


    def plot_irf(self, irf_df, variables_to_plot, title_suffix="", figsize=(12, 8)):
        """Plots selected variables from an IRF DataFrame."""
        if irf_df is None:
            print("Cannot plot None IRF DataFrame.")
            return

        plt.figure(figsize=figsize)
        shock_name = irf_df.attrs.get('shock_name', 'Unknown Shock')
        plot_title = f'Impulse Responses to {shock_name}'
        if title_suffix:
            plot_title += f" ({title_suffix})"

        # Use 'Period' column for x-axis if it exists, otherwise use index
        if 'Period' in irf_df.columns:
            x_axis = irf_df['Period']
            x_label = 'Period'
        else:
            x_axis = irf_df.index
            x_label = 'Periods (Index)'


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
        # Set x-axis limits to start from 1 if using 'Period' column
        if 'Period' in irf_df.columns:
            plt.xlim(left=1) # Start plot at period 1
        plt.tight_layout()
        try:
            plt.show()
        except Exception as e:
            print(f"Note: Plot display failed ({e}). May need GUI backend.")