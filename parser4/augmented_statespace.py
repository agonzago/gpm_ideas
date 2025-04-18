import json
import numpy as np
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys
import scipy.linalg as la


def klein(a=None,b=None,c=None,phi=None,n_states=None,eigenvalue_warnings=True):

    '''Solves linear dynamic models with the form of:
    
                a*Et[x(t+1)] = b*x(t) + c*z(t)

        with x(t) = [s(t); u(t)] where s(t) is a vector of predetermined (state) variables and u(t) is
        a vector of nonpredetermined costate variables. z(t) is a vector of exogenous forcing variables with 
        autocorrelation matrix phi. The solution to the model is a set of matrices f, n, p, l such that:

                u(t)   = f*s(t) + n*z(t)
                s(t+1) = p*s(t) + l*z(t).

        The solution algorithm is based on Klein (2000) and his solab.m Matlab program.

    Args:
        a:                      (Numpy ndarray) Coefficient matrix on future-dated variables
        b:                      (Numpy ndarray) Coefficient matrix on current-dated variables
        c:                      (Numpy ndarray) Coefficient matrix on exogenous forcing variables
        phi:                    (Numpy ndarray) Autocorrelation of exogenous forcing variables
        n_states:               (int) Number of state variables
        eigenvalue_warnings:    (bool) Whether to print warnings that there are too many or few eigenvalues. Default: True

    Returns:
        f:          (Numpy ndarray) Solution matrix coeffients on s(t) for u(t)
        n:          (Numpy ndarray) Solution matrix coeffients on z(t) for u(t)
        p:          (Numpy ndarray) Solution matrix coeffients on s(t) for s(t+1)
        l:          (Numpy ndarray) Solution matrix coeffients on z(t) for s(t+1)
        stab:       (int) Indicates solution stability and uniqueness
                        stab == 1: too many stable eigenvalues
                        stab == -1: too few stable eigenvalues
                        stab == 0: just enoughstable eigenvalues
        eig:        The generalized eigenvalues from the Schur decomposition

    '''

    s,t,alpha,beta,q,z = la.ordqz(A=a,B=b,sort='ouc',output='complex')

    # print('type of s,',s.dtype)
    # print('type of t,',t.dtype)

    forcingVars = False
    if len(np.shape(c))== 0:
        nz = 0
        phi = np.empty([0,0])
    else:
        forcingVars = True
        nz = np.shape(c)[1]
        

    # Components of the z matrix
    z11 = z[0:n_states,0:n_states]
    z12 = z[0:n_states,n_states:]
    z21 = z[n_states:,0:n_states]
    z22 = z[n_states:,n_states:]

    # number of nonpredetermined variables
    n_costates = np.shape(a)[0] - n_states
    
    if n_states>0:
        if np.linalg.matrix_rank(z11)<n_states:
            sys.exit("Invertibility condition violated. Check model equations or parameter values.")

    s11 = s[0:n_states,0:n_states];
    if n_states>0:
        z11i = la.inv(z11)
        s11i = la.inv(s11)
    else:
        z11i = z11
        s11i = s11

    # Components of the s,t,and q matrices
    s12 = s[0:n_states,n_states:]
    s22 = s[n_states:,n_states:]
    t11 = t[0:n_states,0:n_states]
    t12 = t[0:n_states,n_states:]
    t22 = t[n_states:,n_states:]
    q1  = q[0:n_states,:]
    q2  = q[n_states:,:]

    # Verify that there are exactly n_states stable (inside the unit circle) eigenvalues:
    stab = 0

    if n_states>0:
        if np.abs(t[n_states-1,n_states-1])>np.abs(s[n_states-1,n_states-1]):
            if eigenvalue_warnings:
                print('Warning: Too few stable eigenvalues. Check model equations or parameter values.')
            stab = -1

    if n_states<n_states+n_costates:
        if np.abs(t[n_states,n_states])<np.abs(s[n_states,n_states]):
            if eigenvalue_warnings:
                print('Warning: Too many stable eigenvalues. Check model equations or parameter values.')
            stab = 1

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

    # Solution matrix coefficients on the exogenous state
    if not forcingVars:
        n = np.empty([n_costates,0])
        l = np.empty([n_states,0])
    else:
        mat1 = np.kron(np.transpose(phi),s22) - np.kron(np.identity(nz),t22)
        mat1i = la.inv(mat1)
        q2c = q2.dot(c)
        vecq2c = q2c.flatten('C').T
        vecm = mat1i.dot(vecq2c)
        m = np.transpose(np.reshape(np.transpose(vecm),(nz,n_costates)))
        
        n = (z22 - z21.dot(z11i).dot(z12)).dot(m)
        l = -z11.dot(s11i).dot(t11).dot(z11i).dot(z12).dot(m) + z11.dot(s11i).dot(t12.dot(m) - s12.dot(m).dot(phi)+q1.dot(c)) + z12.dot(m).dot(phi)

    return f,n,p,l,stab,eig


class SimpleModelSolver:
    """
    Model solver using Klein's method based on the output of the Dynare parser
    Uses the imported klein function for the solution
    """
    
    def __init__(self, json_file, jacobian_file, structure_file):
        """Initialize the solver with the files generated by the parser"""
        self.json_file = json_file
        self.jacobian_file = jacobian_file
        self.structure_file = structure_file
        
        # Load the model data
        self.model_data = self._load_json(json_file)
        self.jacobian_module = self._load_module('jacobian_module', jacobian_file)
        self.structure_module = self._load_module('structure_module', structure_file)
        
        # Extract variable lists
        self.var_names = self.model_data.get('variables', [])
        self.state_names = self.model_data.get('states', [])
        self.control_names = self.model_data.get('controls', [])
        self.shock_names = self.model_data.get('shocks', [])
        self.shock_to_process_var_map = self.model_data.get('shock_to_process_var_map',[])

        # Initialize solution matrices
        self.f = None  # Control solution matrix (f in klein)
        self.n = None  # Control response to shocks (n in klein)
        self.p = None  # State transition matrix (p in klein)
        self.l = None  # State response to shocks (l in klein)
        self.is_solved = False
        self.T = None 
        self.R  =None

    def _load_json(self, file_path):
        """Load a JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading JSON file: {e}")
    
    def _load_module(self, module_name, file_path):
        """Dynamically load a Python module from a file"""
        try:
            spec = spec_from_file_location(module_name, file_path)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise RuntimeError(f"Error loading module from {file_path}: {e}")
    
    def compute_jacobians(self):
        """Compute the Jacobian matrices A, B, C"""
        # Get parameters from the model data
        param_names = self.model_data.get('parameters', [])
        param_values = self.model_data.get('param_values', {})
        
        # Create parameter vector in the correct order
        theta = np.array([param_values.get(param, 0.0) for param in param_names])
        
        # Compute the Jacobians using the jacobian_evaluator module
        A, B, C = self.jacobian_module.evaluate_jacobians(theta)
        
        return A, B, C
    
    def solve_klein(self, A, B, C=None, n_states=None):
        """
        Solve the model using the klein function
        
        Parameters:
        -----------
        A : 2D array
            Jacobian matrix for t+1 variables (AE_t x_{t+1})
        B : 2D array
            Jacobian matrix for t variables (Bx_t)
        C : 2D array, optional
            Jacobian matrix for shocks (C𝜀_t)
        n_states : int, optional
            Number of state variables
            
        Returns:
        --------
        f : 2D array
            Control solution matrix (policy function)
        p : 2D array
            State transition matrix
        """
        # Get number of states if not provided
        if n_states is None:
            n_states = len(self.state_names)
            
        # Get dimensions
        n_vars = A.shape[1]
        n_controls = n_vars - n_states
        n_shocks = len(self.shock_names) if C is not None else 0
        
        print(f"Klein solution - states: {n_states}, controls: {n_controls}, shocks: {n_shocks}")
        
        # Create autocorrelation matrix for exogenous shocks (phi)
        # Default to identity matrix (no autocorrelation)
        #phi = np.eye(n_shocks) if n_shocks > 0 else None
        
        # Call the klein function
        try:
            f, n, p, l, stab, eig = klein(a=A, b=B, c=None, phi=None, n_states=n_states)
            
            # Check solution stability
            if stab != 0:
                print(f"Warning: Solution stability issue (stab={stab})")
                if stab == 1:
                    print("Too many stable eigenvalues - multiple solutions possible")
                elif stab == -1:
                    print("Too few stable eigenvalues - no stable solution exists")
            
            # Store the solution matrices


            self.f = np.real(f)  # Control solution
            #self.n = np.real(n)  # Control response to shocks
            self.p = np.real(p)  # State transition
            #self.l = np.real(l)  # State response to shocks
            self.is_solved = True
            
            print(f"F matrix shape: {f.shape}, P matrix shape: {p.shape}")
            if n is not None:
                print(f"N matrix shape: {n.shape}, L matrix shape: {l.shape}")
            
            return f, p
            
        except Exception as e:
            print(f"Error in klein solver: {str(e)}")
            self.is_solved = False
            return None, None
    
    def solve_model(self):
        """Solve the model using Klein's method"""
        # Compute Jacobians
        print("Computing Jacobians...")
        A, B, C = self.compute_jacobians()
        print(f"Jacobian A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
        
        # Solve the model using Klein's method
        print("Solving model using Klein's method...")
        n_states = len(self.state_names)
        f, p = self.solve_klein(A, B, C, n_states)
        
        if f is None or p is None:
            print("Model could not be solved!")
            return None, None
        
        print("Model solved!")
        return f, p
    
    def compute_irf(self, shock_name, shock_size: float = 1.0, periods=40):
        """Compute impulse response functions for a specific shock"""
        if not self.is_solved:
            raise RuntimeError("Model must be solved before computing IRFs")
        
        # Get dimensions
        n_states = len(self.state_names)
        n_controls = len(self.control_names)
        n_shocks = len(self.shock_names)
        
        # if shock_index >= n_shocks:
        #     raise ValueError(f"Shock index {shock_index} out of range (0-{n_shocks-1})")
        
        # Get variable names
        state_vars = self.state_names
        control_vars = self.control_names
        all_vars = state_vars + control_vars
        # Find the target state variable name corresponding to the shock
        if shock_name not in self.shock_to_process_var_map:
            raise ValueError(f"Shock '{shock_name}' not found in shock_to_process_var_map.")
        exo_to_shock = self.shock_to_process_var_map[shock_name]

        # Find the index of the target state variable
        try:
            shock_state_index = state_vars.index(exo_to_shock)
            print(f"Applying shock '{shock_name}' to state variable '{exo_to_shock}' at index {shock_state_index}")
        except ValueError:
            raise ValueError(f"State variable '{exo_to_shock}' (target for shock '{shock_name}') not found in state_names list: {state_vars}")

        
        # Initialize IRF arrays
        irf_states = np.zeros((periods + 1, n_states))
        irf_controls = np.zeros((periods, n_controls))
        
        e = np.zeros(n_states)
        e[shock_state_index] = shock_size  # Apply a unit shock to the target state variable    
        # Compute IRF dynamics using the state transition and control policy functions
        irf_states[0, :] = e
        #irf_controls[0 :] = self.f @ irf_states[0, :]
        for t in range(0,periods):
            # State transition: s_{t+1} = P * s_t                        
            irf_states[t+1, :] = self.p @ irf_states[t, :]
            irf_controls[t, :] = self.f @ irf_states[t, :]
        # Combine state and control IRFs
        irf_all = np.zeros((periods, len(all_vars)))
        irf_all[:, :n_states] = irf_states[0:periods,:]
        irf_all[:, n_states:] = irf_controls
        
        print(f"IRF computed for shock: {shock_name}")
        return irf_all
    
    def plot_irf(self, shock_name, shock_size: float = 1.0, variables_to_plot=None, periods=40):
        """Plot impulse response functions for selected variables"""
        
    
        irf_data = self.compute_irf(shock_name, shock_size, periods)
        
        # If variables_to_plot not specified, use all variables
        state_vars = self.state_names
        control_vars = self.control_names
        var_names = state_vars + control_vars

        if variables_to_plot is None:
            variables_to_plot = var_names
        elif isinstance(variables_to_plot, str):
            variables_to_plot = [variables_to_plot]  # Handle single variable case
        
        # Filter to only include variables that exist
        variables_to_plot = [v for v in variables_to_plot if v in var_names]
        
        # Create time axis
        time = np.arange(periods)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for var in variables_to_plot:
            if var in var_names:
                var_idx = var_names.index(var)
                plt.plot(time, irf_data[:, var_idx], label=var)
        
        plt.title(f"Impulse Response to {shock_name} Shock")
        plt.xlabel("Periods")
        plt.ylabel("Deviation from Steady State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"irf_{shock_name}.png")
        plt.show()

    # def build_kalman_matrices(self):
    #     """
    #     Build Kalman filter matrices (T and R) following the Fortran-style routine.
    #     Maps parser‐defined constants to the required dimensions:
    #         num_cntrl_obs = number of control variables (from base_solver.control_names)
    #         num_k        = number of states in Klein’s solution (from base_solver.p.shape[0])
    #         num_exo      = number of shocks (from base_solver.shock_names)
    #     """
    #     # Map constants from the parser via base_solver
    #     num_cntrl = len(self.model_data['controls'])
    #     num_k =  len(self.model_data['states'])
    #     num_exo = len(self.model_data['shocks'])
        
    #     # Derive Fortran-style constants
    #     num_est = num_cntrl + num_k    # Total Kalman states = controls + states
    #     n_k     = num_cntrl            # observed controls block size
    #     k_ex    = num_k - num_exo           # non-exogenous state block size
    #     n_ex    = n_k + k_ex               # index end for non-exo states

    #     # Get Klein’s solution matrices from the base solver
    #     # f is the policy matrix (shape: (num_cntrl_obs, num_k))
    #     # p is the state transition matrix (shape: (num_k, num_k))
    #     F_obs = self.f
    #     P = self.p

    #     if F_obs.shape != (num_cntrl, num_k):
    #         raise ValueError(f"F_obs shape mismatch: expected ({num_cntrl_obs},{num_k}), got {F_obs.shape}")
    #     if P.shape != (num_k, num_k):
    #         raise ValueError(f"P shape mismatch: expected ({num_k},{num_k}), got {P.shape}")
    #     if not (0 <= num_exo <= num_k):
    #         raise ValueError("num_exo must be between 0 and num_k")
    #     if num_cntrl < 0:
    #         raise ValueError("num_cntrl_obs cannot be negative")
        
    #     # Partition the input matrices as in KleintoKalman
    #     F1 = F_obs[:, :k_ex]  # (n_k, k_ex)
    #     F2 = F_obs[:, k_ex:]  # (n_k, num_exo)
        
    #     P11 = P[:k_ex, :k_ex]   # (k_ex, k_ex)
    #     P12 = P[:k_ex, k_ex:]   # (k_ex, num_exo)
    #     # P21 is assumed zero
    #     P22 = P[k_ex:, k_ex:]   # (num_exo, num_exo)
        
    #     # Initialize T and R matrices
    #     T = np.zeros((num_est, num_est))
    #     R = np.zeros((num_est, num_exo))
        
    #     # Construct T matrix block by block
    #     # Block 1: Observed controls update
    #     if k_ex > 0:
    #         T[0:n_k, n_k:n_ex] = F1       # maps non-exo state variables to observed controls
    #     if num_exo > 0:
    #         T[0:n_k, n_ex:num_est] = F2 @ P22   # maps exo state changes
        
    #     # Block 2: Non-exogenous state update
    #     if k_ex > 0:
    #         T[n_k:n_ex, n_k:n_ex] = P11
    #         if num_exo > 0:
    #             T[n_k:n_ex, n_ex:num_est] = P12 @ P22
        
    #     # Block 3: Exogenous state update
    #     if num_exo > 0:
    #         T[n_ex:num_est, n_ex:num_est] = P22
        
    #     # Construct R matrix block by block (shock impact)
    #     if num_exo > 0:
    #         R[0:n_k, 0:num_exo] = F2      # Impact on observed controls
    #     if k_ex > 0 and num_exo > 0:
    #         R[n_k:n_ex, 0:num_exo] = P12    # Impact on non-exogenous states
    #     if num_exo > 0:
    #         np.fill_diagonal(R[n_ex:num_est, 0:num_exo], 1.0)  # Exogenous state shock impact
        
    #     # Save the built matrices in the instance (or return them)
    #     self.T = T
    #     self.R = R
    #     return T, R

    def build_kalman_matrices(self):
        """
        Build the augmented state space matrices using the ordering:
            [ states; controls ]
        where
            states:    x (dimension n_states)
            controls:  y = f * x (dimension n_controls)
            
        With Klein's solution, the dynamics for the states are
              xₜ₊₁ = p * xₜ + shock
        and the controls are given algebraically by
              yₜ = f * xₜ.
            
        Thus, the augmented system becomes:
            x_aug = [ x; y ]
            x_augₜ₊₁ = T * x_augₜ + R * εₜ
            
        with
                T = [ p       0
                        f * p   0 ]
                R = [ I
                        f ]
        """
        # Map constants from the parsed model
        n_states = len(self.model_data['states'])
        n_controls = len(self.model_data['controls'])
        
        # Klein's solution matrices: 
        #    p (n_states x n_states) and f (n_controls x n_states)
        if self.p.shape != (n_states, n_states):
            raise ValueError(f"State transition matrix p shape mismatch: expected ({n_states}, {n_states}), got {self.p.shape}")
        if self.f.shape != (n_controls, n_states):
            raise ValueError(f"Policy matrix f shape mismatch: expected ({n_controls}, {n_states}), got {self.f.shape}")
        
        # Build augmented T matrix
        T = np.zeros((n_states + n_controls, n_states + n_controls))
        # Upper-left block: state evolution from p
        T[:n_states, :n_states] = self.p
        # Lower-left block: controls evolve as f * p
        T[n_states:, :n_states] = self.f @ self.p
        # The other blocks remain zero since controls are algebraically determined
        
        # Build augmented R matrix
        R = np.zeros((n_states + n_controls, n_states))
        
        #Instead of assuming shocks hit every state, we only apply shocks to selected states
        shock_map = self.model_data.get('shock_to_process_var_map', {})
        n_shocks = len(self.shock_names)
        
        R = np.zeros((n_states + n_controls, n_shocks))
        # Build a selection matrix S (n_states x n_shocks_active) such that only selected states get a shock
        S = np.zeros((n_states, n_shocks))
        for j, (shock_name, target_state) in enumerate(shock_map.items()):
            if target_state not in self.state_names:
                raise ValueError(f"Target state '{target_state}' for shock '{shock_name}' not found in state_names.")
            i = self.state_names.index(target_state)
            S[i, j] = 1.0
        # Top block: states only get a shock if selected in S
        R[:n_states, :] = S
        # Bottom block: controls respond to shocks through f
        R[n_states:, :] = self.f @ S

        
        # Save the augmented matrices
        self.T = T
        self.R = R
        return T, R
    
    def simulate_state_space(self, shock, periods=40):
        """
        Simulate the state space model using the Fortran‑style matrices T and R.
        
        The simulation follows:
            x(0) = R @ shock
            x(t+1) = T @ x(t)
        
        Assumes the state vector ordering:
            [observed_controls; non_exo_states; exo_states]
        
        Returns:
        --------
        x_sim : np.ndarray
            Simulated state vectors over time (shape: (periods+1, num_est))
        y_sim : np.ndarray
            Simulated output (observables), taken as the first num_cntrl_obs elements
                    of the state vector (shape: (periods+1, num_cntrl_obs))
        """
        # Ensure that the Kalman matrices have been built
        if self.T is None or self.R is None:
            self.build_kalman_matrices()
        
        # Dimensions
        num_est = self.T.shape[0]
        num_cntrl_obs =  len(self.model_data['controls'])
        
        # Initialize state simulation array
        x_sim = np.zeros((periods + 1, num_est))
        # At time zero, the state is the shock impact
        x_sim[0, :] = self.R @ shock
        
        # Propagate the state using the transition matrix T
        for t in range(periods):
            x_sim[t + 1, :] = self.T @ x_sim[t, :]
        
        # Observed outputs taken as the first num_cntrl_obs elements of the state vector
        #y_sim = x_sim[:, :num_cntrl_obs]
        return x_sim #, y_sim

    def plot_simulation(self, x_sim, variables_to_plot=None):
        """
        Create a plot of the simulation output, mimicking the style of plot_irf.
        
        Parameters:
        -----------
        x_sim : np.ndarray
            Simulated full state vectors over time (shape: (periods+1, num_est))
        
        """
        import matplotlib.pyplot as plt
        
        # Get observable variable names from the model data.
        #obs_vars = self.get_simulation_names()
        state_vars = self.state_names
        control_vars = self.control_names
        var_names = state_vars + control_vars

        
        # Filter to only include variables that exist
        variables_to_plot = [v for v in variables_to_plot if v in var_names]
        
        # Create time axis
        periods = x_sim.shape[0]
        time = np.arange(periods)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for var in variables_to_plot:
            if var in var_names:
                var_idx = var_names.index(var)
                plt.plot(time, x_sim[:, var_idx], label=var)
        
        plt.title(f"Impulse Response to Shock")
        plt.xlabel("Periods")
        plt.ylabel("Deviation from Steady State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        #plt.savefig(f"irf_{shock_name}.png")
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

class AugmentedStateSpace:
    def __init__(self, base_solver, obs_mapping=None):
        """
        Initialize the augmented state space.
        Uses base_solver's T and R matrices and augments the state vector with stochastic trends.
        The augmented state vector is: [ base states (states+controls); trend states ]
        
        Parameters:
          base_solver : SimpleModelSolver
            The base model solver that has built the Kalman matrices T and R.
          obs_mapping : dict, optional
            Mapping from observable names to specifications.
            Example:
              {
                "RS_OBS": {"cycle": "RS", "trend": "rw", "model_var": "RS"},
                "DLA_CPI_OBS": {"cycle": "DLA_CPI", "trend": "rw", "model_var": "DLA_CPI"},
                "L_GDP_OBS": {"cycle": "L_GDP_GAP", "trend": "cm", "model_var": "L_GDP_GAP"}
              }
        """
        self.base_solver = base_solver
        
        # Ensure Kalman matrices T and R are built.
        if self.base_solver.T is None or self.base_solver.R is None:
            self.base_solver.build_kalman_matrices()
        self.T_base = self.base_solver.T   # base state-transition matrix
        self.R_base = self.base_solver.R   # base shock impact matrix
        
        # Base state labels: states plus controls.
        self.base_state_labels = self.base_solver.state_names + self.base_solver.control_names
        self.base_shock_labels = self.base_solver.shock_names
        
        # Process obs_mapping to extract observable list and trend specifications.
        if obs_mapping is not None:
            self.obs_mapping = obs_mapping
            self.observable_vars = list(obs_mapping.keys())
            self.trend_specs = {}
            for obs, spec in obs_mapping.items():
                if "trend" in spec:
                    key = spec.get("model_var", obs)
                    self.trend_specs[key] = spec["trend"]
        else:
            self.obs_mapping = {var: {"cycle": var, "trend": None} for var in self.base_solver.var_names}
            self.observable_vars = list(self.obs_mapping.keys())
            self.trend_specs = {var: None for var in self.base_solver.var_names}
        
        # Augmented matrices to be built.
        self.A_aug = None
        self.B_aug = None
        self.C_aug = None
        self.aug_state_labels = None
        self.aug_shock_labels = None
        
        self._build_augmented_state_space()
    
    def _build_trend_matrices(self):
        """
        Build trend matrices based on self.trend_specs.
        
        Returns:
          A_trend (np.ndarray): Trend state transition matrix.
          B_trend (np.ndarray): Trend shock impact matrix.
          trend_labels (list): Trend state labels.
          trend_shock_labels (list): Trend shock labels.
        """
        trend_blocks = []
        shock_blocks = []
        trend_labels = []
        trend_shock_labels = []
        for var, trend_type in self.trend_specs.items():
            if trend_type == 'rw':
                A_i = np.array([[1.0]])
                B_i = np.array([[1.0]])
                trend_blocks.append(A_i)
                shock_blocks.append(B_i)
                trend_labels.append(f"{var}_trend")
                trend_shock_labels.append(f"{var}_trend_shock")
            elif trend_type == 'sd':
                A_i = np.array([[1.0, 1.0],
                                [0.0, 1.0]])
                B_i = np.eye(2)
                trend_blocks.append(A_i)
                shock_blocks.append(B_i)
                trend_labels.extend([f"{var}_level", f"{var}_growth"])
                trend_shock_labels.extend([f"{var}_level_shock", f"{var}_growth_shock"])
            elif trend_type == 'dt':
                A_i = np.array([[1.0]])
                B_i = np.zeros((1, 1))
                trend_blocks.append(A_i)
                shock_blocks.append(B_i)
                trend_labels.append(f"{var}_trend")
                trend_shock_labels.append(f"{var}_trend_shock")
            elif trend_type == 'cm':
                # Constant mean option: state persists; no shock impact.
                A_i = np.array([[1.0]])
                B_i = np.zeros((1, 1))
                trend_blocks.append(A_i)
                shock_blocks.append(B_i)
                trend_labels.append(f"{var}_trend")
                trend_shock_labels.append(f"{var}_trend_shock")
            elif trend_type is None:
                continue
            else:
                raise ValueError(f"Unknown trend type '{trend_type}' for variable '{var}'")
        if trend_blocks:
            A_trend = la.block_diag(*trend_blocks)
            B_trend = la.block_diag(*shock_blocks)
        else:
            A_trend = np.empty((0, 0))
            B_trend = np.empty((0, 0))
        return A_trend, B_trend, trend_labels, trend_shock_labels

    def _build_augmented_state_space(self):
        """
        Build the augmented state space.
        The augmented system is:
          x_aug(t+1) = A_aug x_aug(t) + B_aug ε(t)
          y(t)       = C_aug x_aug(t)
        where:
          A_aug = block_diag(T_base, A_trend)
          B_aug = block_diag(R_base, B_trend)
        and the augmented state vector is [ base states (states+controls); trend states ].
        The observation matrix C_aug maps these augmented states as specified.
        """
        # Build trend matrices.
        A_trend, B_trend, trend_labels, trend_shock_labels = self._build_trend_matrices()
        n_trend_states = A_trend.shape[0]
        
        # Build augmented matrices.
        self.A_aug = la.block_diag(self.T_base, A_trend)
        self.B_aug = la.block_diag(self.R_base, B_trend)
        
        # Augmented state labels.
        self.aug_state_labels = self.base_state_labels + trend_labels
        self.aug_shock_labels = self.base_shock_labels + trend_shock_labels
        
        # Build augmented observation matrix C_aug.
        # For each observable in obs_mapping, load its base and trend parts based on "model_var".
        n_aug_states = len(self.aug_state_labels)
        n_obs = len(self.observable_vars)
        C_aug = np.zeros((n_obs, n_aug_states))
        for i, obs in enumerate(self.observable_vars):
            spec = self.obs_mapping.get(obs, {})
            model_var = spec.get("model_var", obs)
            # Base loading: look in the base state labels.
            if model_var in self.base_state_labels:
                idx = self.base_state_labels.index(model_var)
                C_aug[i, idx] = 1.0
            else:
                print(f"Warning: {model_var} not found in base_state_labels.")
            # Trend loading: if a trend is specified, map to the trend state.
            trend_type = spec.get("trend", None)
            if trend_type is not None:
                if trend_type in ['rw', 'dt', 'cm']:
                    trend_label = f"{model_var}_trend"
                    if trend_label in self.aug_state_labels:
                        trend_idx = self.aug_state_labels.index(trend_label)
                        C_aug[i, trend_idx] = 1.0
                elif trend_type == 'sd':
                    # For the mixed specification, load both components.
                    for suffix in ["_level", "_growth"]:
                        label = f"{model_var}{suffix}"
                        if label in self.aug_state_labels:
                            trend_idx = self.aug_state_labels.index(label)
                            C_aug[i, trend_idx] = 1.0
                else:
                    raise ValueError(f"Trend type {trend_type} not implemented.")
        self.C_aug = C_aug
        
        print("Augmented state space created:")
        print(f"  - Base states: {len(self.base_state_labels)}")
        print(f"  - Trend states: {n_trend_states}")
        print(f"  - Total states: {n_aug_states}")
        print(f"  - Total shocks: {self.B_aug.shape[1]}")
        print(f"  - Observables: {n_obs}")

    def compute_irf(self, shock_index, periods=40):
        """
        Compute impulse response functions for the augmented model.
        The state evolves as:
             x_aug(t+1) = A_aug x_aug(t) + B_aug ε(t)
        and the observables are computed as:
             y(t) = C_aug x_aug(t)
        
        We simulate x_aug and then compute y_irf = C_aug * x_aug.
        A combined IRF matrix is then constructed by concatenating x_aug and y_irf
        so that both the full augmented states and the observables can be selected by name.
        
        Returns:
           combined_irf : np.ndarray of shape (periods+1, n_aug_states + n_obs)
           combined_labels : list of labels for the columns (first the augmented state labels,
                             then the observable variable names)
        """
        n_aug_states = self.A_aug.shape[0]
        n_shocks = self.B_aug.shape[1]
        if shock_index >= n_shocks:
            raise ValueError(f"Shock index {shock_index} out of range (0 to {n_shocks-1}).")
        
        # Create a unit shock vector.
        shock_vec = np.zeros(n_shocks)
        shock_vec[shock_index] = 1.0
        
        # Simulate state trajectory for x_aug.
        x_aug = np.zeros((periods + 1, n_aug_states))
        x_aug[0, :] = self.B_aug @ shock_vec
        for t in range(periods):
            x_aug[t+1, :] = self.A_aug @ x_aug[t, :]
        
        # Compute observable IRF: y_irf = C_aug * x_aug.
        y_irf = (self.C_aug @ x_aug.T).T   # shape: (periods+1, n_obs)
        
        # Combine the full state and observable responses.
        combined_irf = np.hstack((x_aug, y_irf))
        combined_labels = self.aug_state_labels + self.observable_vars
        
        return combined_irf, combined_labels

    def plot_irf(self, shock_index, variables_to_plot=None, periods=40):
        """
        Plot impulse response functions for the augmented model.
        
        The IRF output is a combined matrix that has both:
          - the full augmented states (base states + trend states)
          - the observables computed as y = C_aug * x_aug
        
        Users can select by name the variables to plot from the combined labels.
        If variables_to_plot is not provided, all variables are plotted.
        """
        combined_irf, combined_labels = self.compute_irf(shock_index, periods)
        
        # If variables_to_plot is specified, filter the combined_labels.
        if variables_to_plot is not None:
            plot_labels = [var for var in variables_to_plot if var in combined_labels]
            if not plot_labels:
                print("None of the selected variables are available; plotting all variables.")
                plot_labels = combined_labels
        else:
            plot_labels = combined_labels
        
        # Get indices corresponding to these labels.
        indices = [combined_labels.index(var) for var in plot_labels]
        time = np.arange(periods + 1)
        
        plt.figure(figsize=(12, 8))
        for idx in indices:
            plt.plot(time, combined_irf[:, idx], label=combined_labels[idx])
        plt.title(f"Augmented Model IRF (Shock index {shock_index})")
        plt.xlabel("Periods")
        plt.ylabel("Response")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()