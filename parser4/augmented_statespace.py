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
        
        # Initialize solution matrices
        self.f = None  # Control solution matrix (f in klein)
        self.n = None  # Control response to shocks (n in klein)
        self.p = None  # State transition matrix (p in klein)
        self.l = None  # State response to shocks (l in klein)
        self.is_solved = False
    
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
        phi = np.eye(n_shocks) if n_shocks > 0 else None
        
        # Call the klein function
        try:
            f, n, p, l, stab, eig = klein(a=A, b=B, c=C, phi=phi, n_states=n_states)
            
            # Check solution stability
            if stab != 0:
                print(f"Warning: Solution stability issue (stab={stab})")
                if stab == 1:
                    print("Too many stable eigenvalues - multiple solutions possible")
                elif stab == -1:
                    print("Too few stable eigenvalues - no stable solution exists")
            
            # Store the solution matrices


            self.f = np.real(f)  # Control solution
            self.n = np.real(n)  # Control response to shocks
            self.p = np.real(p)  # State transition
            self.l = np.real(l)  # State response to shocks
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
    
    def compute_irf(self, shock_index, periods=40):
        """Compute impulse response functions for a specific shock"""
        if not self.is_solved:
            raise RuntimeError("Model must be solved before computing IRFs")
        
        # Get dimensions
        n_states = len(self.state_names)
        n_controls = len(self.control_names)
        n_shocks = len(self.shock_names)
        
        if shock_index >= n_shocks:
            raise ValueError(f"Shock index {shock_index} out of range (0-{n_shocks-1})")
        
        # Get variable names
        state_vars = self.state_names
        control_vars = self.control_names
        all_vars = state_vars + control_vars
        shock_name = self.shock_names[shock_index]
        
        # Create shock vector (unit shock)
        shock = np.zeros(n_shocks)
        shock[shock_index] = 1.0
        
        # Initialize IRF arrays
        irf_states = np.zeros((periods + 1, n_states))
        irf_controls = np.zeros((periods + 1, n_controls))
        
        # Set initial state responses using L matrix (state response to shocks)
        if self.l is not None:
            irf_states[0, :] = self.l[:, shock_index]
        
        # Set initial control responses using N matrix (control response to shocks)
        if self.n is not None:
            irf_controls[0, :] = self.n[:, shock_index]
        
        # Compute IRF dynamics using the state transition and control policy functions
        for t in range(periods):
            # State transition: s_{t+1} = P * s_t
            irf_states[t+1, :] = self.p @ irf_states[t, :]
            
            # Control solution: c_t = F * s_t (for t>0)
            if t+1 <= periods:
                irf_controls[t+1, :] = self.f @ irf_states[t+1, :]
        
        # Combine state and control IRFs
        irf_all = np.zeros((periods + 1, len(all_vars)))
        irf_all[:, :n_states] = irf_states
        irf_all[:, n_states:] = irf_controls
        
        print(f"IRF computed for shock: {shock_name}")
        return irf_all, all_vars
    
    def plot_irf(self, shock_index, variables_to_plot=None, periods=40):
        """Plot impulse response functions for selected variables"""
        irf_data, var_names = self.compute_irf(shock_index, periods)
        
        # Get shock name
        shock_name = self.shock_names[shock_index]
        
        # If variables_to_plot not specified, use all variables
        if variables_to_plot is None:
            variables_to_plot = var_names
        elif isinstance(variables_to_plot, str):
            variables_to_plot = [variables_to_plot]  # Handle single variable case
        
        # Filter to only include variables that exist
        variables_to_plot = [v for v in variables_to_plot if v in var_names]
        
        # Create time axis
        time = np.arange(periods + 1)
        
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





class AugmentedStateSpace:
    """
    Augmented state space representation with trend components
    This extends the base model to include random walk trend specifications
    """
    
    def __init__(self, base_solver, trend_specs=None, observable_vars=None):
        """
        Initialize the augmented state space with a base model solver
        
        Parameters:
        -----------
        base_solver : SimpleModelSolver
            The base model solver with solution already computed
        trend_specs : dict, optional
            Dictionary mapping variable names to trend specifications:
            - 'rw': Random walk
        observable_vars : list, optional
            List of variable names that are observable
        """
        self.base_solver = base_solver
        
        # Make sure the base model is solved
        if not base_solver.is_solved:
            raise ValueError("Base model must be solved before creating augmented state space")
        
        # Default trend specification (all random walks)
        if trend_specs is None:
            trend_specs = {}
            # Default to random walk for all variables
            for var in base_solver.var_names:
                trend_specs[var] = 'rw'
            
        self.trend_specs = trend_specs
        
        # Default all variables are observable if not specified
        if observable_vars is None:
            observable_vars = base_solver.var_names
            
        self.observable_vars = observable_vars
        
        # Initialize augmented matrices
        self.A_aug = None  # Augmented transition matrix
        self.B_aug = None  # Augmented shock impact matrix
        self.C_aug = None  # Augmented measurement matrix
        self.H = None      # Selection matrix for observable variables
        
        # Build the augmented state space
        self._build_augmented_state_space()
    
    def _build_augmented_state_space(self):
        """Build the augmented state space with random walk trend components"""
        # Get base model dimensions
        n_states = len(self.base_solver.state_names)
        n_controls = len(self.base_solver.control_names)
        n_vars = n_states + n_controls
        n_shocks = len(self.base_solver.shock_names)
        
        # Create the base model state transition matrix A
        A = np.zeros((n_vars, n_vars))
        
        # Fill the upper-left block with P (state transition)
        A[:n_states, :n_states] = self.base_solver.p
        
        # Fill the lower-left block with F (control policy)
        A[n_states:, :n_states] = self.base_solver.f
        
        # Create the base model shock impact matrix B
        B = np.zeros((n_vars, n_shocks))
        
        # Fill the state response using L matrix (from klein)
        if self.base_solver.l is not None:
            B[:n_states, :] = self.base_solver.l
        
        # Fill the control response using N matrix (from klein)
        if self.base_solver.n is not None:
            B[n_states:, :] = self.base_solver.n
        
        # Create the measurement matrix C (identity for now)
        C = np.eye(n_vars)
        
        # Count trend components needed (all random walks in this case)
        n_trend_states = 0
        n_trend_shocks = 0
        
        # Track which variables have trends
        rw_vars = []   # Random walk
        
        # Count trend states and shocks
        for var, trend_type in self.trend_specs.items():
            if var in self.base_solver.var_names:
                if trend_type == 'rw':  # Random walk
                    n_trend_states += 1
                    n_trend_shocks += 1
                    rw_vars.append(var)
        
        # Create the augmented state transition matrix A_aug
        n_aug_states = n_vars + n_trend_states
        A_aug = np.zeros((n_aug_states, n_aug_states))
        
        # Fill the upper-left block with A (base model)
        A_aug[:n_vars, :n_vars] = A
        
        # Fill the trend block (lower-right)
        trend_start = n_vars
        for i in range(n_trend_states):
            # Random walk: trend_{t} = trend_{t-1} + eps_level
            A_aug[trend_start + i, trend_start + i] = 1.0
        
        # Create the augmented shock impact matrix B_aug
        n_aug_shocks = n_shocks + n_trend_shocks
        B_aug = np.zeros((n_aug_states, n_aug_shocks))
        
        # Fill the upper-left block with B (base model)
        B_aug[:n_vars, :n_shocks] = B
        
        # Fill the trend shock impacts (diagonal ones for each trend shock)
        for i in range(n_trend_shocks):
            B_aug[n_vars + i, n_shocks + i] = 1.0
        
        # Create the augmented measurement matrix C_aug
        C_aug = np.zeros((n_vars, n_aug_states))
        
        # Fill the base model part (cycle components)
        C_aug[:, :n_vars] = C
        
        # Add trend components to observed variables
        for i, var in enumerate(self.base_solver.var_names):
            if var in self.trend_specs and var in rw_vars:
                # Find position of this trend in the state vector
                trend_pos = n_vars + rw_vars.index(var)
                C_aug[i, trend_pos] = 1.0
        
        # Create selection matrix H for observable variables
        H = np.zeros((len(self.observable_vars), n_vars))
        
        for i, var in enumerate(self.observable_vars):
            if var in self.base_solver.var_names:
                var_idx = self.base_solver.var_names.index(var)
                H[i, var_idx] = 1.0
        
        # Store augmented matrices
        self.A_aug = A_aug
        self.B_aug = B_aug
        self.C_aug = C_aug
        self.H = H
        
        # Calculate final observation matrix
        self.obs_matrix = H @ C_aug
        
        print(f"Augmented state space created:")
        print(f"  - Base states+controls: {n_vars}")
        print(f"  - Trend states: {n_trend_states}")
        print(f"  - Base shocks: {n_shocks}")
        print(f"  - Trend shocks: {n_trend_shocks}")
        print(f"  - Observable variables: {len(self.observable_vars)}")
    
    def compute_irf(self, shock_index, periods=40, include_trend=True):
        """
        Compute impulse response functions for the augmented model
        
        Parameters:
        -----------
        shock_index : int
            Index of the shock to analyze (base model shocks come first)
        periods : int, optional
            Number of periods to simulate (default: 40)
        include_trend : bool, optional
            Whether to include trend components in the IRF (default: True)
            
        Returns:
        --------
        irf_all : 2D array
            IRF data for observable variables (periods+1 x n_observables)
        obs_vars : list
            Names of observable variables
        """
        # Get dimensions
        n_base_shocks = len(self.base_solver.shock_names)
        n_aug_shocks = self.B_aug.shape[1]
        n_states = self.A_aug.shape[0]
        n_obs = self.H.shape[0]
        
        if shock_index >= n_aug_shocks:
            raise ValueError(f"Shock index {shock_index} out of range (0-{n_aug_shocks-1})")
        
        # Determine shock name
        if shock_index < n_base_shocks:
            shock_name = self.base_solver.shock_names[shock_index]
        else:
            # This is a trend shock
            trend_shock_idx = shock_index - n_base_shocks
            var_index = trend_shock_idx  # Each trend variable has one shock
            
            if var_index < len(self.base_solver.var_names):
                shock_name = f"{self.base_solver.var_names[var_index]}_trend_shock"
            else:
                shock_name = f"Trend_shock_{trend_shock_idx}"
        
        # Create shock vector (unit shock)
        shock = np.zeros(n_aug_shocks)
        shock[shock_index] = 1.0
        
        # Initialize IRF arrays
        irf_states = np.zeros((periods + 1, n_states))
        
        # Compute initial state impact
        irf_states[0, :] = self.B_aug @ shock
        
        # Compute IRF dynamics using the augmented state space
        for t in range(periods):
            # State transition: x_{t+1} = A_aug * x_t
            irf_states[t+1, :] = self.A_aug @ irf_states[t, :]
        
        # Compute observed variables: y_t = H * C_aug * x_t
        irf_obs = np.zeros((periods + 1, n_obs))
        
        for t in range(periods + 1):
            if include_trend:
                # Use the full state for observation
                irf_obs[t, :] = self.obs_matrix @ irf_states[t, :]
            else:
                # Use only the cycle component (base model part)
                n_base_vars = len(self.base_solver.var_names)
                irf_obs[t, :] = self.H @ self.C_aug[:, :n_base_vars] @ irf_states[t, :n_base_vars]
        
        print(f"Augmented IRF computed for shock: {shock_name}")
        return irf_obs, self.observable_vars
    
    def plot_irf(self, shock_index, variables_to_plot=None, periods=40, include_trend=True):
        """Plot impulse response functions for the augmented model"""
        irf_data, obs_vars = self.compute_irf(shock_index, periods, include_trend)
        
        # Get shock name
        n_base_shocks = len(self.base_solver.shock_names)
        if shock_index < n_base_shocks:
            shock_name = self.base_solver.shock_names[shock_index]
        else:
            trend_shock_idx = shock_index - n_base_shocks
            if trend_shock_idx < len(self.observable_vars):
                shock_name = f"{self.observable_vars[trend_shock_idx]}_trend_shock"
            else:
                shock_name = f"Trend_shock_{trend_shock_idx}"
        
        # If variables_to_plot not specified, use all observable variables
        if variables_to_plot is None:
            variables_to_plot = self.observable_vars
        elif isinstance(variables_to_plot, str):
            variables_to_plot = [variables_to_plot]  # Handle single variable case
        
        # Filter to only include variables that exist
        variables_to_plot = [v for v in variables_to_plot if v in obs_vars]
        
        # Create time axis
        time = np.arange(periods + 1)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for var in variables_to_plot:
            if var in obs_vars:
                var_idx = obs_vars.index(var)
                plt.plot(time, irf_data[:, var_idx], label=var)
        
        trend_str = " with Trend Components" if include_trend else " (Cycle Components Only)"
        plt.title(f"Augmented Model IRF to {shock_name} Shock{trend_str}")
        plt.xlabel("Periods")
        plt.ylabel("Deviation from Steady State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        trend_suffix = "with_trend" if include_trend else "cycle_only"
        plt.savefig(f"aug_irf_{shock_name}_{trend_suffix}.png")
        plt.show()
    
    def compare_with_base_model(self, shock_index, variables_to_plot=None, periods=40):
        """
        Compare IRFs between base model and augmented model
        for the same economic shock (excluding trend shocks)
        
        Parameters:
        -----------
        shock_index : int
            Index of the base model shock to analyze
        variables_to_plot : list or str, optional
            Variable(s) to plot (default: all observable variables)
        periods : int, optional
            Number of periods to simulate (default: 40)
        """
        # Check that this is a base model shock
        n_base_shocks = len(self.base_solver.shock_names)
        if shock_index >= n_base_shocks:
            raise ValueError(f"Only base model shocks (0-{n_base_shocks-1}) can be compared")
        
        # Get shock name
        shock_name = self.base_solver.shock_names[shock_index]
        
        # Compute IRFs for both models
        base_irf, base_vars = self.base_solver.compute_irf(shock_index, periods)
        aug_irf, aug_vars = self.compute_irf(shock_index, periods, include_trend=False)
        
        # If variables_to_plot not specified, use intersection of observable variables
        if variables_to_plot is None:
            variables_to_plot = [v for v in aug_vars if v in base_vars]
        elif isinstance(variables_to_plot, str):
            if variables_to_plot in base_vars and variables_to_plot in aug_vars:
                variables_to_plot = [variables_to_plot]
            else:
                print(f"Warning: Variable {variables_to_plot} not found in both models")
                variables_to_plot = [v for v in aug_vars if v in base_vars][:5]  # Show first 5 common variables
        
        # Create time axis
        time = np.arange(periods + 1)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        line_styles = ['-', '--']
        colors = ['b', 'r']
        
        for var in variables_to_plot:
            if var in base_vars and var in aug_vars:
                # Get indices
                base_idx = base_vars.index(var)
                aug_idx = aug_vars.index(var)
                
                # Plot both IRFs
                plt.plot(time, base_irf[:, base_idx], f'{colors[0]}{line_styles[0]}', label=f'{var} (Base Model)')
                plt.plot(time, aug_irf[:, aug_idx], f'{colors[1]}{line_styles[1]}', label=f'{var} (Augmented Model)')
        
        plt.title(f"Comparison of IRFs to {shock_name} Shock")
        plt.xlabel("Periods")
        plt.ylabel("Deviation from Steady State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"compare_irf_{shock_name}.png")
        plt.show()
    """
    Augmented state space representation with trend components
    This extends the base model to include various trend specifications
    """
    
    def __init__(self, base_solver, trend_specs=None, observable_vars=None):
        """
        Initialize the augmented state space with a base model solver
        
        Parameters:
        -----------
        base_solver : SimpleModelSolver
            The base model solver with solution already computed
        trend_specs : dict, optional
            Dictionary mapping variable names to trend specifications:
            - 'rw': Random walk
            - 'sd': Second difference
            - 'cm': Constant mean
        observable_vars : list, optional
            List of variable names that are observable
        """
        self.base_solver = base_solver
        
        # Make sure the base model is solved
        if not base_solver.is_solved:
            raise ValueError("Base model must be solved before creating augmented state space")
        
        # Default trend specification (constant mean)
        if trend_specs is None:
            trend_specs = {}
            
        self.trend_specs = trend_specs
        
        # Default all variables are observable if not specified
        if observable_vars is None:
            observable_vars = base_solver.var_names
            
        self.observable_vars = observable_vars
        
        # Initialize augmented matrices
        self.A_aug = None  # Augmented transition matrix
        self.B_aug = None  # Augmented shock impact matrix
        self.C_aug = None  # Augmented measurement matrix
        self.H = None      # Selection matrix for observable variables
        
        # Build the augmented state space
        self._build_augmented_state_space()
    
    def _build_augmented_state_space(self):
        """Build the augmented state space with trend components"""
        # Get base model dimensions
        n_states = len(self.base_solver.state_names)
        n_controls = len(self.base_solver.control_names)
        n_vars = n_states + n_controls
        n_shocks = len(self.base_solver.shock_names)
        
        # Create the base model state transition matrix A
        A = np.zeros((n_vars, n_vars))
        
        # Fill the upper-left block with P (state transition)
        A[:n_states, :n_states] = self.base_solver.p
        
        # Fill the lower-left block with F (control policy)
        A[n_states:, :n_states] = self.base_solver.f
        
        # Create the base model shock impact matrix B
        B = np.zeros((n_vars, n_shocks))
        
        # If we have an H matrix, use it for shock impacts
        if self.base_solver.h is not None:
            B[:n_states, :] = self.base_solver.h
        elif self.base_solver.R is not None:
            # Use R to map shocks to exogenous states
            n_endo_states = len(self.base_solver.endo_states)
            n_exo_states = n_states - n_endo_states
            
            if n_exo_states > 0 and self.base_solver.R.shape[0] == n_exo_states:
                B[n_endo_states:n_states, :] = self.base_solver.R
        
        # Create the measurement matrix C (identity for now)
        C = np.eye(n_vars)
        
        # Count trend components needed
        n_trend_states = 0
        n_trend_shocks = 0
        
        # Track which variables have which trend types
        rw_vars = []   # Random walk
        sd_vars = []   # Second difference
        sd_g_vars = [] # Second difference growth components
        
        # Count number of states needed for each type of trend
        for var, trend_type in self.trend_specs.items():
            if var in self.base_solver.var_names:
                if trend_type == 'rw':  # Random walk
                    n_trend_states += 1
                    n_trend_shocks += 1
                    rw_vars.append(var)
                elif trend_type == 'sd':  # Second difference
                    n_trend_states += 2  # Level and growth
                    n_trend_shocks += 2  # Level and growth shocks
                    sd_vars.append(var)
                    sd_g_vars.append(f"{var}_growth")
                # 'cm' (constant mean) doesn't need additional states
        
        # Create the augmented state transition matrix A_aug
        n_aug_states = n_vars + n_trend_states
        A_aug = np.zeros((n_aug_states, n_aug_states))
        
        # Fill the upper-left block with A (base model)
        A_aug[:n_vars, :n_vars] = A
        
        # Fill the trend block (lower-right)
        trend_start = n_vars
        for var in rw_vars:
            # Random walk: trend_{t} = trend_{t-1} + eps_level
            A_aug[trend_start, trend_start] = 1.0
            trend_start += 1
            
        for i in range(len(sd_vars)):
            # Second difference:
            # trend_{t} = trend_{t-1} + g_{t-1} + eps_level
            # g_{t} = g_{t-1} + eps_growth
            A_aug[trend_start, trend_start] = 1.0      # trend depends on past trend
            A_aug[trend_start, trend_start+1] = 1.0    # trend depends on past growth
            A_aug[trend_start+1, trend_start+1] = 1.0  # growth depends on past growth
            trend_start += 2
        
        # Create the augmented shock impact matrix B_aug
        n_aug_shocks = n_shocks + n_trend_shocks
        B_aug = np.zeros((n_aug_states, n_aug_shocks))
        
        # Fill the upper-left block with B (base model)
        B_aug[:n_vars, :n_shocks] = B
        
        # Fill the trend shock impacts
        trend_start = n_vars
        shock_start = n_shocks
        
        for var in rw_vars:
            # Random walk: trend shock impacts level
            B_aug[trend_start, shock_start] = 1.0
            trend_start += 1
            shock_start += 1
            
        for i in range(len(sd_vars)):
            # Second difference: level and growth shocks
            B_aug[trend_start, shock_start] = 1.0      # Level shock
            B_aug[trend_start+1, shock_start+1] = 1.0  # Growth shock
            trend_start += 2
            shock_start += 2
        
        # Create the augmented measurement matrix C_aug
        n_observables = len(self.observable_vars)
        C_aug = np.zeros((n_vars, n_aug_states))
        
        # Fill the base model part (cycle components)
        C_aug[:, :n_vars] = C
        
        # Add trend components to observed variables
        for i, var in enumerate(self.base_solver.var_names):
            if var in self.trend_specs:
                trend_type = self.trend_specs[var]
                
                if trend_type == 'rw':
                    # Find position of this trend in the state vector
                    trend_pos = n_vars + rw_vars.index(var)
                    C_aug[i, trend_pos] = 1.0
                    
                elif trend_type == 'sd':
                    # Find position of this trend in the state vector
                    trend_pos = n_vars + len(rw_vars) + 2 * sd_vars.index(var)
                    C_aug[i, trend_pos] = 1.0
        
        # Create selection matrix H for observable variables
        H = np.zeros((n_observables, n_vars))
        
        for i, var in enumerate(self.observable_vars):
            if var in self.base_solver.var_names:
                var_idx = self.base_solver.var_names.index(var)
                H[i, var_idx] = 1.0
        
        # Store augmented matrices
        self.A_aug = A_aug
        self.B_aug = B_aug
        self.C_aug = C_aug
        self.H = H
        
        # Calculate final observation matrix
        self.obs_matrix = H @ C_aug
        
        print(f"Augmented state space created:")
        print(f"  - Base states+controls: {n_vars}")
        print(f"  - Trend states: {n_trend_states}")
        print(f"  - Base shocks: {n_shocks}")
        print(f"  - Trend shocks: {n_trend_shocks}")
        print(f"  - Observable variables: {n_observables}")
    
    def compute_irf(self, shock_index, periods=40, include_trend=True):
        """
        Compute impulse response functions for the augmented model
        
        Parameters:
        -----------
        shock_index : int
            Index of the shock to analyze (base model shocks come first)
        periods : int, optional
            Number of periods to simulate (default: 40)
        include_trend : bool, optional
            Whether to include trend components in the IRF (default: True)
            
        Returns:
        --------
        irf_all : 2D array
            IRF data for observable variables (periods+1 x n_observables)
        obs_vars : list
            Names of observable variables
        """
        # Get dimensions
        n_base_shocks = len(self.base_solver.shock_names)
        n_aug_shocks = self.B_aug.shape[1]
        n_states = self.A_aug.shape[0]
        n_obs = self.H.shape[0]
        
        if shock_index >= n_aug_shocks:
            raise ValueError(f"Shock index {shock_index} out of range (0-{n_aug_shocks-1})")
        
        # Determine shock name
        if shock_index < n_base_shocks:
            shock_name = self.base_solver.shock_names[shock_index]
        else:
            # This is a trend shock
            trend_shock_idx = shock_index - n_base_shocks
            trend_var_idx = 0
            remaining = trend_shock_idx
            
            # Find which trend variable this shock corresponds to
            for var, trend_type in self.trend_specs.items():
                if trend_type == 'rw':
                    if remaining == 0:
                        shock_name = f"{var}_level_shock"
                        break
                    remaining -= 1
                elif trend_type == 'sd':
                    if remaining == 0:
                        shock_name = f"{var}_level_shock"
                        break
                    elif remaining == 1:
                        shock_name = f"{var}_growth_shock"
                        break
                    remaining -= 2
        
        # Create shock vector (unit shock)
        shock = np.zeros(n_aug_shocks)
        shock[shock_index] = 1.0
        
        # Initialize IRF arrays
        irf_states = np.zeros((periods + 1, n_states))
        
        # Compute initial state impact
        irf_states[0, :] = self.B_aug @ shock
        
        # Compute IRF dynamics using the augmented state space
        for t in range(periods):
            # State transition: x_{t+1} = A_aug * x_t
            irf_states[t+1, :] = self.A_aug @ irf_states[t, :]
        
        # Compute observed variables: y_t = H * C_aug * x_t
        irf_obs = np.zeros((periods + 1, n_obs))
        
        for t in range(periods + 1):
            irf_obs[t, :] = self.obs_matrix @ irf_states[t, :]
        
        print(f"Augmented IRF computed for shock: {shock_name}")
        return irf_obs, self.observable_vars
    
    def plot_irf(self, shock_index, variables_to_plot=None, periods=40, include_trend=True):
        """Plot impulse response functions for the augmented model"""
        irf_data, obs_vars = self.compute_irf(shock_index, periods, include_trend)
        
        # Get shock name
        n_base_shocks = len(self.base_solver.shock_names)
        if shock_index < n_base_shocks:
            shock_name = self.base_solver.shock_names[shock_index]
        else:
            shock_name = f"Trend Shock {shock_index - n_base_shocks + 1}"
        
        # If variables_to_plot not specified, use all observable variables
        if variables_to_plot is None:
            variables_to_plot = self.observable_vars
        elif isinstance(variables_to_plot, str):
            variables_to_plot = [variables_to_plot]
            
        # Filter to only include variables that exist
        variables_to_plot = [v for v in variables_to_plot if v in obs_vars]
        
        # Create time axis
        time = np.arange(periods + 1)
        
        # Calculate number of subplots needed
        n_plots = len(variables_to_plot)
        n_rows = int(np.ceil(n_plots / 3))
        n_cols = min(3, n_plots)
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        for i, var in enumerate(variables_to_plot):
            if var in obs_vars:
                plt.subplot(n_rows, n_cols, i+1)
                var_idx = obs_vars.index(var)
                plt.plot(time, irf_data[:, var_idx])
                plt.title(var)
                plt.grid(True)
                if i >= n_plots - n_cols:  # Bottom row
                    plt.xlabel("Periods")
                if i % n_cols == 0:  # First column
                    plt.ylabel("Deviation from SS")
        
        trend_str = " with Trend Components" if include_trend else " (Cycle Components Only)"
        plt.suptitle(f"Augmented Model IRF to {shock_name} Shock{trend_str}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save plot
        trend_suffix = "with_trend" if include_trend else "cycle_only"
        plt.savefig(f"aug_irf_{shock_name}_{trend_suffix}.png")
        plt.show()
        
        # Also create a single plot with all variables for comparison
        plt.figure(figsize=(12, 8))
        for var in variables_to_plot:
            if var in obs_vars:
                var_idx = obs_vars.index(var)
                plt.plot(time, irf_data[:, var_idx], label=var)
        
        plt.title(f"Augmented Model IRF to {shock_name} Shock{trend_str}")
        plt.xlabel("Periods")
        plt.ylabel("Deviation from Steady State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save combined plot
        plt.savefig(f"aug_irf_{shock_name}_combined_{trend_suffix}.png")
        plt.show()
    
    def compare_with_base_model(self, shock_index, variables_to_plot=None, periods=40):
        """
        Compare IRFs between base model and augmented model
        for the same economic shock (excluding trend shocks)
        
        Parameters:
        -----------
        shock_index : int
            Index of the base model shock to analyze
        variables_to_plot : list or str, optional
            Variable(s) to plot (default: all observable variables)
        periods : int, optional
            Number of periods to simulate (default: 40)
        """
        # Check that this is a base model shock
        n_base_shocks = len(self.base_solver.shock_names)
        if shock_index >= n_base_shocks:
            raise ValueError(f"Only base model shocks (0-{n_base_shocks-1}) can be compared")
        
        # Get shock name
        shock_name = self.base_solver.shock_names[shock_index]
        
        # Compute IRFs for both models
        base_irf, base_vars = self.base_solver.compute_irf(shock_index, periods)
        aug_irf, aug_vars = self.compute_irf(shock_index, periods, include_trend=False)
        
        # If variables_to_plot not specified, use intersection of observable variables
        if variables_to_plot is None:
            variables_to_plot = [v for v in aug_vars if v in base_vars]
        elif isinstance(variables_to_plot, str):
            if variables_to_plot in base_vars and variables_to_plot in aug_vars:
                variables_to_plot = [variables_to_plot]
            else:
                print(f"Warning: Variable {variables_to_plot} not found in both models")
                variables_to_plot = [v for v in aug_vars if v in base_vars][:5]  # Show first 5 common variables
        
        # Create time axis
        time = np.arange(periods + 1)
        
        # Create plots (one per variable)
        for var in variables_to_plot:
            if var in base_vars and var in aug_vars:
                plt.figure(figsize=(10, 6))
                
                # Get indices
                base_idx = base_vars.index(var)
                aug_idx = aug_vars.index(var)
                
                # Plot both IRFs
                plt.plot(time, base_irf[:, base_idx], 'b-', label='Base Model')
                plt.plot(time, aug_irf[:, aug_idx], 'r--', label='Augmented Model (cycle only)')
                
                plt.title(f"Comparison of IRF for {var} to {shock_name} Shock")
                plt.xlabel("Periods")
                plt.ylabel("Deviation from Steady State")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save plot
                plt.savefig(f"compare_irf_{var}_{shock_name}.png")
                plt.show()
        
        # Create a combined plot with all variables
        plt.figure(figsize=(15, 10))
        
        # Calculate number of subplots needed
        n_plots = len(variables_to_plot)
        n_rows = int(np.ceil(n_plots / 3))
        n_cols = min(3, n_plots)
        
        for i, var in enumerate(variables_to_plot):
            if var in base_vars and var in aug_vars:
                plt.subplot(n_rows, n_cols, i+1)
                
                # Get indices
                base_idx = base_vars.index(var)
                aug_idx = aug_vars.index(var)
                
                # Plot both IRFs
                plt.plot(time, base_irf[:, base_idx], 'b-', label='Base')
                plt.plot(time, aug_irf[:, aug_idx], 'r--', label='Aug')
                
                plt.title(var)
                plt.grid(True)
                if i >= n_plots - n_cols:  # Bottom row
                    plt.xlabel("Periods")
                if i % n_cols == 0:  # First column
                    plt.ylabel("Deviation from SS")
                if i == 0:  # Only add legend to first subplot
                    plt.legend()
        
        plt.suptitle(f"Comparison of Base and Augmented Model IRFs to {shock_name} Shock", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save plot
        plt.savefig(f"compare_irf_all_{shock_name}.png")