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


    def get_contemporaneous_state_space(self, klein_solution):
        """
        Constructs the state-space system x_t = A'x_{t-1} + B'eps_t, y_t = C'x_t + D'eps_t
        where x_t = [k_t; z_{t-1}], y_t = [c_t; k_t; z_t].
        Requires output from solve() method.
        """
        print("\n--- Constructing Contemporaneous State Space ---")
        if klein_solution is None:
            print("Error: Klein solution not available.")
            return None

        try:
            # --- Extract matrices from Klein solution (TARGET order) ---
            P = klein_solution['p']
            F = klein_solution['f']
            R = klein_solution['R'] # Should be (n_exo x n_shocks)
            indices = klein_solution['indices']
            labels = klein_solution['labels']

            n_k = indices['n_endogenous'] # Number of endogenous states k
            n_z = indices['n_exo_states'] # Number of exogenous states z
            n_states_s = indices['n_states'] # Total states in s = n_k + n_z
            n_c = indices['n_controls']   # Number of controls c
            n_shocks = indices['n_shocks']

            if n_states_s != n_k + n_z:
                print(f"Warning: n_states ({n_states_s}) != n_k ({n_k}) + n_z ({n_z})")
                # Adjust n_z maybe? Let's assume indices are correct.
                # n_z = n_states_s - n_k

            if R.shape != (n_z, n_shocks):
                print(f"Warning: R matrix shape {R.shape} doesn't match n_z={n_z}, n_shocks={n_shocks}")
                # Attempt to proceed, but results might be wrong. Pad/truncate R? Best to fix R generation.
                # return None # Safer to stop

            # --- Partition P and F based on k and z ---
            # P = [Pkk Pkz]
            #     [Pzk Pzz]  <- Note: Klein usually gives [0 Pzz] block here
            Pkk = P[:n_k, :n_k]
            Pkz = P[:n_k, n_k:]
            # Pzk = P[n_k:, :n_k] # Should be zero if exo states are truly exo
            Pzz = P[n_k:, n_k:]
            # F = [Fck Fcz]
            Fck = F[:, :n_k]
            Fcz = F[:, n_k:]

            # Optional: Check if Pzk is close to zero
            # if n_k > 0 and n_z > 0 and not np.allclose(P[n_k:, :n_k], 0):
            #      print("Warning: Bottom-left block of P matrix (Pzk) is not zero.")

            # --- Compute matrix products ---
            PkzPzz = Pkz @ Pzz
            PkzR = Pkz @ R
            FczPzz = Fcz @ Pzz
            FczR = Fcz @ R

            # --- Construct A', B', C', D' ---
            # State vector x_t = [k_t ; z_{t-1}] (size n_k + n_z)
            n_states_x = n_k + n_z

            # A' = [Pkk   Pkz@Pzz]
            #      [ 0     Pzz   ]
            A_prime = np.zeros((n_states_x, n_states_x))
            A_prime[:n_k, :n_k] = Pkk
            A_prime[:n_k, n_k:] = PkzPzz
            # Row block for z_{t-1} update: z_t = Pzz * z_{t-1} (+ R*eps)
            # In x_{t+1}, the z_t component corresponds to the state z_{t} at t+1.
            # Wait, the state is [k_t; z_{t-1}].
            # x_{t+1} = [k_{t+1}; z_t]
            # k_{t+1} = Pkk k_t + PkzPzz z_{t-1} + PkzR eps_t
            # z_t     = 0   k_t + Pzz    z_{t-1} + R    eps_t
            # This matches the derivation.
            A_prime[n_k:, n_k:] = Pzz # Correct

            # B' = [Pkz@R]
            #      [  R  ]
            B_prime = np.zeros((n_states_x, n_shocks))
            B_prime[:n_k, :] = PkzR
            B_prime[n_k:, :] = R

            # Observation y_t = [c_t; k_t; z_t] (size n_c + n_k + n_z)
            n_obs_y = n_c + n_k + n_z

            # C' = [Fck   Fcz@Pzz]
            #      [ I     0     ]
            #      [ 0     Pzz   ]
            C_prime = np.zeros((n_obs_y, n_states_x))
            C_prime[:n_c, :n_k] = Fck      # c_t from k_t
            C_prime[:n_c, n_k:] = FczPzz   # c_t from z_{t-1}
            C_prime[n_c:n_c+n_k, :n_k] = np.eye(n_k) # k_t from k_t
            # Row block for z_t: z_t = Pzz * z_{t-1} (+ R*eps)
            C_prime[n_c+n_k:, n_k:] = Pzz # z_t from z_{t-1}


            # D' = [Fcz@R]
            #      [  0  ]
            #      [  R  ]
            D_prime = np.zeros((n_obs_y, n_shocks))
            D_prime[:n_c, :] = FczR # c_t from eps_t
            # k_t has no direct shock impact
            D_prime[n_c+n_k:, :] = R  # z_t from eps_t

            # --- Define Labels for this system ---
            state_labels_x = labels['state_labels'][:n_k] + [f"{z}_lag1" for z in labels['state_labels'][n_k:]]
            obs_labels_y = list(labels['observable_labels'][:n_c]) + list(labels['state_labels'][:n_k]) + list(labels['state_labels'][n_k:])
            shock_labels_eps = labels['shock_labels']

            # --- Store Results ---
            contemp_ss = {
                'A': A_prime, 'B': B_prime, 'C': C_prime, 'D': D_prime,
                'state_labels': state_labels_x,
                'observable_labels': obs_labels_y,
                'shock_labels': shock_labels_eps,
                'n_states': n_states_x,
                'n_observables': n_obs_y,
                'n_shocks': n_shocks,
                # Include original components for reference if needed
                'P': P, 'F': F, 'R': R,
                'Pkk': Pkk, 'Pkz': Pkz, 'Pzz': Pzz,
                'Fck': Fck, 'Fcz': Fcz
            }
            print("Contemporaneous State Space Constructed.")
            print(f"  A' shape: {A_prime.shape}, B' shape: {B_prime.shape}")
            print(f"  C' shape: {C_prime.shape}, D' shape: {D_prime.shape}")

            return contemp_ss

        except Exception as e:
            print(f"Error constructing contemporaneous state space: {e}")
            import traceback
            traceback.print_exc()
            return None


    def impulse_response(self, state_space_system, shock_name, shock_size=1.0, periods=40):
        """
        Calculate IRFs for a state-space system x_t = Ax_{t-1} + B*eps_t, y_t = Cx_t + D*eps_t.
        Matches Dynare timing convention: response at period t=1 is the initial impact.
        """
        print(f"\n--- Calculating Standard IRF for {shock_name} (Corrected) ---")
        if state_space_system is None:
             print("Error: State space system is None.")
             return None

        try:
            A = state_space_system['A']
            B = state_space_system['B']
            C = state_space_system['C']
            D = state_space_system['D']
            n_states = state_space_system['n_states']
            n_observables = state_space_system['n_observables']
            n_shocks = state_space_system['n_shocks']
            shock_labels = state_space_system['shock_labels']
            observable_labels = state_space_system['observable_labels']

            shock_idx = shock_labels.index(shock_name)
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error accessing data from state_space_system or finding shock: {e}")
            return None

        # Check dimensions before starting
        if not (A.shape == (n_states, n_states) and
                B.shape == (n_states, n_shocks) and
                C.shape == (n_observables, n_states) and
                D.shape == (n_observables, n_shocks)):
             print("Error: Matrix dimension mismatch in state_space_system.")
             print(f"  A:{A.shape} vs ({n_states},{n_states}), B:{B.shape} vs ({n_states},{n_shocks})")
             print(f"  C:{C.shape} vs ({n_observables},{n_states}), D:{D.shape} vs ({n_observables},{n_shocks})")
             return None


        # --- Simulate IRF ---
        # irf index k corresponds to response at period k+1 (index 0 = period 1)
        x_irf = np.zeros((periods, n_states))
        y_irf = np.zeros((periods, n_observables))

        # Create shock vector for period 1 (impact period)
        eps_1 = np.zeros(n_shocks)
        eps_1[shock_idx] = shock_size

        # --- Calculate Period 1 Response (Index 0) ---
        # State x_1 = B * eps_1
        x_t = B @ eps_1
        x_irf[0, :] = x_t

        # Observation y_1 = C * x_1 + D * eps_1
        y_t = C @ x_t + D @ eps_1
        y_irf[0, :] = y_t

        # --- Simulate Periods 2 to T (Indices 1 to periods-1) ---
        for t in range(1, periods):
            # x_{t+1} = A * x_t
            x_t = A @ x_irf[t-1, :]
            x_irf[t, :] = x_t

            # y_{t+1} = C * x_{t+1} (assuming shock only at t=1)
            y_t = C @ x_t
            y_irf[t, :] = y_t

        # --- Create DataFrame ---
        # Index k corresponds to response at period k+1
        irf_df = pd.DataFrame(y_irf, columns=observable_labels)
        # Create a 'Period' column starting from 1 to match Dynare plots
        irf_df.insert(0, 'Period', range(1, periods + 1))
        # Or set index starting from 1
        # irf_df.index = range(1, periods + 1)
        # irf_df.index.name = "Period (t)"

        irf_df.attrs['shock_name'] = shock_name
        return irf_df

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