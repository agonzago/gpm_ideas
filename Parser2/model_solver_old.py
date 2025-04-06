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
        # Load model.json
        model_path = os.path.join(self.output_dir, "model.json")
        with open(model_path, 'r') as f:
            self.model_json = json.load(f)

        # Load jacobian_evaluator.py
        jac_path = os.path.join(self.output_dir, "jacobian_evaluator.py")
        spec = importlib.util.spec_from_file_location("jacobian_evaluator", jac_path)
        jac_module = importlib.util.module_from_spec(spec)
        sys.modules["jacobian_evaluator"] = jac_module
        spec.loader.exec_module(jac_module)
        self.evaluate_jacobians = jac_module.evaluate_jacobians

        # Load model_structure.py
        struct_path = os.path.join(self.output_dir, "model_structure.py")
        spec = importlib.util.spec_from_file_location("model_structure", struct_path)
        struct_module = importlib.util.module_from_spec(spec)
        sys.modules["model_structure"] = struct_module
        spec.loader.exec_module(struct_module)
        self.indices = struct_module.indices
        self.R_struct = struct_module.R # Shock to exo state mapping
        self.labels = struct_module.labels
        # Note: B_structure and C_structure from the parser might be less useful now,
        # as we construct them directly from f, p, and R below.

    def solve(self, params):
        """Solves the model for given parameters and returns the base state-space."""
        theta = [self.model_json['param_values'][p] if p in self.model_json['param_values'] else None for p in self.model_json['parameters']]
        # Overwrite with input params if provided
        if params:
            # Ensure params list matches order in model_json['parameters']
            param_dict = dict(zip(self.model_json['parameters'], params))
            theta = [param_dict[p] for p in self.model_json['parameters']]


        a, b, c = self.evaluate_jacobians(theta)

        n_states = self.indices['n_states']
        n_controls = self.indices['n_controls']
        n_shocks = self.indices['n_shocks']
        n_endo_states = self.indices['n_endogenous'] # Number of purely endogenous states

        if n_states == 0:
            print("Warning: Model has no state variables.")
            # Handle static model case if necessary (returning empty/dummy matrices)
            return None

        f, p, stab, eig = klein(a, b, n_states)

        # Build state-space matrices: x_t = A x_{t-1} + B eps_t, y_t = C x_t + D eps_t
        # where x are the states, y are the observables (controls + states)

        # A matrix (State transition): From Klein's P matrix
        A = p

        # B matrix (Shock impact on states): From structural R matrix
        B = np.zeros((n_states, n_shocks))
        # R maps shocks to *exogenous* states. Exogenous states start at index n_endo_states
        B[n_endo_states:, :] = self.R_struct # R_struct should be (n_exo_states, n_shocks)

        # C matrix (Observation matrix): Maps states to observables (controls; states)
        n_observables = n_controls + n_states
        C = np.zeros((n_observables, n_states))
        C[:n_controls, :] = f             # Controls = f(states)
        C[n_controls:, :] = np.eye(n_states) # States = I * states

        # D matrix (Direct shock impact on observables): Usually zero
        D = np.zeros((n_observables, n_shocks))

        base_ss = {
            'A': A, 'B': B, 'C': C, 'D': D,
            'f': f, 'p': p, # Keep Klein solution if needed
            'labels': self.labels, # Contains state_labels, observable_labels, shock_labels
            'indices': self.indices, # Contains counts like n_states, n_controls, etc.
            'stab': stab, 'eig': eig
        }
        return base_ss

    def impulse_response(self, base_ss, shock_name, shock_size=1.0, periods=40):
        """Calculate IRFs for the base model."""
        try:
            shock_idx = base_ss['labels']['shock_labels'].index(shock_name)
        except ValueError:
            print(f"Error: Shock '{shock_name}' not found in base model shocks.")
            return None

        A, B, C = base_ss['A'], base_ss['B'], base_ss['C']
        n_states = base_ss['indices']['n_states']
        n_observables = base_ss['indices']['n_observables']

        states_irf = np.zeros((periods, n_states))
        obs_irf = np.zeros((periods, n_observables))

        # Initial shock impact on states
        x = B[:, shock_idx] * shock_size

        for t in range(periods):
            states_irf[t, :] = x
            obs_irf[t, :] = C @ x
            x = A @ x # Evolve state for next period

        irf_df = pd.DataFrame(obs_irf, columns=base_ss['labels']['observable_labels'])
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


        for var in variables_to_plot:
            if var in irf_df.columns:
                plt.plot(irf_df.index, irf_df[var], label=var)
            else:
                print(f"Warning: Variable '{var}' not found in IRF results.")

        plt.xlabel('Periods')
        plt.ylabel('Deviation from Steady State')
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        plt.tight_layout()
        plt.show()