# parser3/model_solver.py

import numpy as np
import pandas as pd
import json
import os
import importlib.util
import sys

# Ensure linearsolve is installed and importable
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
        vecq2c = q2c.flatten(1).T
        vecm = mat1i.dot(vecq2c)
        m = np.transpose(np.reshape(np.transpose(vecm),(nz,n_costates)))
        
        n = (z22 - z21.dot(z11i).dot(z12)).dot(m)
        l = -z11.dot(s11i).dot(t11).dot(z11i).dot(z12).dot(m) + z11.dot(s11i).dot(t12.dot(m) - s12.dot(m).dot(phi)+q1.dot(c)) + z12.dot(m).dot(phi)

    return f,n,p,l,stab,eig

class ModelSolver:
    """
    Solves a DSGE model specified by parser_gpm output files,
    using analytical Jacobians and the linearsolve Klein solver.
    """

    def __init__(self, parser_output_dir):
        """
        Initializes the solver by loading model information from parser output.

        Args:
            parser_output_dir (str): Path to the directory containing
                                    model.json, jacobian_evaluator.py,
                                    and model_structure.py.
        """
        self.parser_output_dir = os.path.abspath(parser_output_dir)
        self._jacobian_file_path = os.path.join(self.parser_output_dir, "jacobian_evaluator.py")
        self._structure_file_path = os.path.join(self.parser_output_dir, "model_structure.py")
        self._json_file_path = os.path.join(self.parser_output_dir, "model.json")

        self.model_json = None
        self.model_structure = None
        self.parameters = None # Will be pandas Series
        self.param_names = None # List in specific order
        self.steady_state = None # Will be pandas Series
        self.a = None # Jacobian A (adjusted for Klein)
        self.b = None # Jacobian B (adjusted for Klein)
        self.c = None # Jacobian C (adjusted for Klein)
        self.f = None # Control solution matrix (y = f @ s)
        self.p = None # State transition matrix (s = p @ s_lag)

        self.is_loaded = False
        self.is_solved = False

        self._load_parser_output()
        self.compute_steady_state() # Assume zero SS for now

    def _load_parser_output(self):
        """Loads model definition files generated by parser_gpm."""
        print(f"Loading model files from: {self.parser_output_dir}")

        # --- Check file existence ---
        if not os.path.isdir(self.parser_output_dir):
            raise FileNotFoundError(f"Parser output directory not found: {self.parser_output_dir}")
        if not os.path.isfile(self._json_file_path):
            raise FileNotFoundError(f"model.json not found in {self.parser_output_dir}")
        if not os.path.isfile(self._jacobian_file_path):
            raise FileNotFoundError(f"jacobian_evaluator.py not found in {self.parser_output_dir}")
        if not os.path.isfile(self._structure_file_path):
            raise FileNotFoundError(f"model_structure.py not found in {self.parser_output_dir}")

        # --- Load model.json ---
        try:
            with open(self._json_file_path, 'r') as f:
                self.model_json = json.load(f)
            # Store parameters in a Series, preserving order from json
            self.param_names = list(self.model_json['parameters'])
            self.parameters = pd.Series(self.model_json['param_values'], name='value')
            # Ensure index matches the defined order
            self.parameters = self.parameters.reindex(self.param_names)
            if self.parameters.isnull().any():
                print("Warning: Some parameters listed in model.json['parameters'] "
                    "have missing values in model.json['param_values'].")

            print("Successfully loaded model.json")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse model.json: {e}")

        # --- Load model_structure.py ---
        try:
            spec = importlib.util.spec_from_file_location("model_structure", self._structure_file_path)
            structure_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(structure_module)
            # Store the relevant dicts/arrays
            self.model_structure = {
                'indices': structure_module.indices,
                'labels': structure_module.labels,
                'R': getattr(structure_module, 'R', None), # Optional
                'B_structure': getattr(structure_module, 'B_structure', None), # Optional
                'C_structure': getattr(structure_module, 'C_structure', None), # Optional
                'D': getattr(structure_module, 'D', None) # Optional
            }
            # Verify essential keys exist
            if 'indices' not in self.model_structure or 'labels' not in self.model_structure:
                raise KeyError("model_structure.py is missing 'indices' or 'labels'.")
            # Use labels from structure file as the definitive source
            self.state_names = self.model_structure['labels'].get('state_labels', [])
            self.control_names = self.model_structure['labels'].get('control_labels', [])
            self.shock_names = self.model_structure['labels'].get('shock_labels', [])

            if not self.state_names or not self.control_names:
                raise KeyError("model_structure['labels'] is missing 'state_labels' or 'control_labels'.")

            # Combine state and control labels in the standard order (states first)
            self.variable_names = self.state_names + self.control_names
            self.n_vars = len(self.variable_names)
            self.n_states = self.model_structure['indices'].get('n_states', len(self.state_names))
            self.n_controls = self.model_structure['indices'].get('n_controls', len(self.control_names))
            self.n_shocks = self.model_structure['indices'].get('n_shocks', len(self.shock_names))

            # Cross-validate counts if indices are present
            if 'indices' in self.model_structure:
                if self.n_states != self.model_structure['indices'].get('n_states'):
                    print("Warning: n_states from labels differs from indices.")
                if self.n_controls != self.model_structure['indices'].get('n_controls'):
                    print("Warning: n_controls from labels differs from indices.")
                if self.n_shocks != self.model_structure['indices'].get('n_shocks'):
                    print("Warning: n_shocks from labels differs from indices.")

            if self.n_vars != self.n_states + self.n_controls:
                print(f"Warning: n_vars ({self.n_vars}) != n_states ({self.n_states}) + n_controls ({self.n_controls}). Check model_structure.py.")

            print(f"Successfully loaded model_structure.py (States: {self.n_states}, Controls: {self.n_controls}, Shocks: {self.n_shocks})")

        except Exception as e:
            raise RuntimeError(f"Failed to load or parse model_structure.py: {e}")

        self.is_loaded = True
        print("Model loading complete.")

    def compute_steady_state(self, ss_values=None):
        """
        Sets the steady state. Currently assumes zero steady state unless provided.

        Args:
            ss_values (dict or pd.Series, optional): Dictionary or Series mapping
                variable names to steady-state values. If None, assumes zero SS.
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before computing steady state.")

        if ss_values is not None:
            if isinstance(ss_values, dict):
                self.steady_state = pd.Series(ss_values)
            elif isinstance(ss_values, pd.Series):
                self.steady_state = ss_values.copy()
            else:
                raise TypeError("ss_values must be a dictionary or pandas Series.")
            # Ensure index covers all variables, filling missing with 0? Or raise error?
            try:
                self.steady_state = self.steady_state.reindex(self.variable_names, fill_value=0.0)
            except Exception as e:
                print(f"Warning: Could not reindex provided steady state. Ensure names match variable_names. Error: {e}")
                # Fallback to zero or raise? Let's fallback for now.
                self.steady_state = pd.Series(np.zeros(self.n_vars), index=self.variable_names)

            print("Manual steady state set.")
        else:
            # Assume zero steady state
            print("Assuming zero steady state.")
            self.steady_state = pd.Series(np.zeros(self.n_vars), index=self.variable_names)

    def compute_jacobians(self, params=None):
        """
        Computes the analytical Jacobians A, B, C using the generated evaluator.

        Args:
            params (dict or pd.Series or list/np.ndarray, optional):
                Parameter values to use. If None, uses values loaded from model.json.
                If dict/Series, uses names. If list/array, assumes order
                matches self.param_names.

        Returns:
            tuple: (A, B, C) matrices adjusted for the Klein solver.
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before computing Jacobians.")

        # --- Prepare parameter vector ---
        if params is None:
            theta = self.parameters.loc[self.param_names].values
            if np.isnan(theta).any():
                print("Warning: Using parameters from model.json, but some values are NaN.")
        elif isinstance(params, (dict, pd.Series)):
            temp_params = pd.Series(params)
            try:
                # Use stored order, get values from provided dict/series
                theta = temp_params.reindex(self.param_names).values
                if np.isnan(theta).any():
                    raise ValueError("Provided params missing values for required parameters or contain NaN.")
            except KeyError as e:
                raise KeyError(f"Parameter '{e}' not found in provided params.")
        elif isinstance(params, (list, np.ndarray)):
            if len(params) != len(self.param_names):
                raise ValueError(f"Provided parameter list/array has wrong length ({len(params)}), "
                                f"expected {len(self.param_names)}.")
            theta = np.asarray(params)
            if np.isnan(theta).any():
                raise ValueError("Provided params list/array contains NaN values.")
        else:
            raise TypeError("params must be dict, Series, list, array, or None.")

        # --- Dynamically load and run evaluator ---
        try:
            spec = importlib.util.spec_from_file_location("jacobian_evaluator", self._jacobian_file_path)
            jacobian_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(jacobian_module)
            evaluate_jacobians = jacobian_module.evaluate_jacobians
            print(f"Executing jacobian_evaluator.py with {len(theta)} parameters...")
            a_calc, b_calc, c_calc = evaluate_jacobians(theta)
            print("Jacobians computed by evaluator.")

        except Exception as e:
            raise RuntimeError(f"Failed to load or execute jacobian_evaluator.py: {e}")

        # --- Sign adjustment for Klein solver ---
        # Parser assumes: a_calc = -dF/dXp, b_calc = dF/dX, c_calc = dF/dZ
        # Klein needs: A = dF/dXp, B = dF/dX for A*Xp + B*X = 0 solver input
        # So, A = -a_calc, B = b_calc. C = c_calc is used later for shock impact.
        self.a = -a_calc
        self.b = b_calc
        self.c = c_calc

        # --- Validate dimensions ---
        n_eqs = self.a.shape[0] # Number of equations should match first dim
        # Number of variables (states+controls) must match columns
        expected_cols = self.n_vars
        expected_shock_cols = self.n_shocks

        if self.a.shape != (n_eqs, expected_cols):
            raise ValueError(f"Jacobian A shape mismatch: {self.a.shape}, expected ({n_eqs}, {expected_cols}) based on {self.n_vars} total variables")
        if self.b.shape != (n_eqs, expected_cols):
            raise ValueError(f"Jacobian B shape mismatch: {self.b.shape}, expected ({n_eqs}, {expected_cols}) based on {self.n_vars} total variables")
        if self.c.shape != (n_eqs, expected_shock_cols):
            raise ValueError(f"Jacobian C shape mismatch: {self.c.shape}, expected ({n_eqs}, {expected_shock_cols}) based on {self.n_shocks} shocks")

        print("Jacobian matrices A, B, C computed and stored.")
        return self.a, self.b, self.c

    def solve(self, force_recompute_jacobians=False, params=None):
        """
        Solves the model using the Klein algorithm.

        Args:
            force_recompute_jacobians (bool): If True, recalculate Jacobians
                                            before solving. Default False.
            params (dict, pd.Series, list, np.ndarray, optional):
                Parameter values to use for Jacobian calculation if recomputing.
                If None, uses current self.parameters.
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before solving.")
        if self.steady_state is None:
            print("Warning: Steady state not computed. Assuming zero.")
            self.compute_steady_state()

        if self.a is None or self.b is None or force_recompute_jacobians:
            print("Computing Jacobians before solving...")
            self.compute_jacobians(params=params) # Use provided or stored parameters

        print("Solving model using Klein algorithm...")
        try:
            # Pass the adjusted A, B to Klein
            # Klein expects square matrices for A and B if the number of equations equals the number of variables.
            # Need to ensure the dimensions passed match Klein's expectations.
            # Klein likely solves A*E[x_t+1] + B*x_t = 0 where x is the vector of *all* variables.
            if self.a.shape[0] != self.n_vars or self.b.shape[0] != self.n_vars:
                print(f"Warning: Number of equations ({self.a.shape[0]}) does not match "
                    f"number of variables ({self.n_vars}). Klein solver might behave unexpectedly.")

            # Assuming klein takes A, B of shape (n_eqs, n_vars) and number of states n_states
            # Check klein's docstring or source for exact input requirements!
            # Let's assume it needs number of state variables to partition A, B internally.
            self.f, self.p = klein(self.a, self.b, self.n_states) # Pass n_states
            self.is_solved = True
            print("Model solved successfully.")
            print(f"  State transition matrix 'p' shape: {self.p.shape}") # Should be (n_states, n_states)
            print(f"  Control policy matrix 'f' shape: {self.f.shape}")   # Should be (n_controls, n_states)

            # --- Validate solution matrix dimensions ---
            if self.p.shape != (self.n_states, self.n_states):
                 print(f"Warning: Solution matrix 'p' shape {self.p.shape} "
                       f"unexpected, expected ({self.n_states}, {self.n_states}).")
            if self.f.shape != (self.n_controls, self.n_states):
                 print(f"Warning: Solution matrix 'f' shape {self.f.shape} "
                       f"unexpected, expected ({self.n_controls}, {self.n_states}).")

        except Exception as e:
            self.is_solved = False
            # Check for specific LinAlgError for more informative message
            if isinstance(e, np.linalg.LinAlgError):
                 print(f"ERROR: Linear algebra error during Klein solution (matrix may be singular or eigenvalues issue): {e}")
            elif isinstance(e, ValueError) and "eigenvalues" in str(e):
                 print(f"ERROR: Klein solver failed due to eigenvalue condition (Blanchard-Kahn violation?): {e}")
            else:
                 print(f"ERROR: Failed to solve model using Klein algorithm: {e}")
            # Consider re-raising or handling differently
            raise RuntimeError("Model solution failed.") from e


    def impulse_response(self, shock_name, periods=40):
        """
        Computes impulse responses for a given shock.

        Args:
            shock_name (str): Name of the shock (must be in shock_names).
            periods (int): Number of periods for the IRF.

        Returns:
            pandas.DataFrame: DataFrame containing the impulse responses for all variables.
                              Index is period (0 to periods), columns are variable names.
        """
        if not self.is_solved:
            raise RuntimeError("Model must be solved before computing IRFs.")
        if self.c is None:
             raise RuntimeError("Jacobian C matrix not available. Recompute Jacobians.")

        try:
            shock_index = self.shock_names.index(shock_name)
        except ValueError:
            raise ValueError(f"Shock '{shock_name}' not found. Available shocks: {self.shock_names}")
        except AttributeError:
             raise RuntimeError("Shock names not loaded correctly from model_structure.py.")

        print(f"Computing IRF for shock: {shock_name} ({periods} periods)...")

        # --- Construct the full system transition matrix p_full ---
        # Describes evolution of x_t = [s_t; y_t] based on x_{t-1}
        # Where x_t = [state_vars; control_vars]
        p_full = np.zeros((self.n_vars, self.n_vars))
        p_full[:self.n_states, :self.n_states] = self.p                 # s_t = p @ s_{t-1}
        p_full[self.n_states:, :self.n_states] = self.f @ self.p      # y_t = f @ s_t = f @ p @ s_{t-1}

        # --- Calculate shock impact matrix d ---
        # Solve (A*P_full + B)*X_t = -C*e_t for X_t = d @ e_t
        try:
            coeff_matrix = self.a @ p_full + self.b
            # Solve for the shock impact matrix 'd' (response of ALL vars to ALL shocks)
            d = scipy.linalg.solve(coeff_matrix, -self.c)
            if d.shape != (self.n_vars, self.n_shocks):
                 print(f"Warning: Shock impact matrix 'd' shape {d.shape} unexpected, "
                       f"expected ({self.n_vars}, {self.n_shocks}).")
        except np.linalg.LinAlgError as e:
             print(f"ERROR: Could not compute shock impact matrix 'd'. "
                   f"Matrix 'A @ P_full + B' may be singular: {e}")
             raise RuntimeError("IRF calculation failed due to linear algebra error.") from e
        except Exception as e:
             print(f"ERROR: Unexpected error calculating shock impact matrix 'd': {e}")
             raise


        # --- Simulate the IRF ---
        irf_array = np.zeros((periods + 1, self.n_vars))

        # Initial impact at t=1 (index 1 in array)
        initial_impact = d[:, shock_index] # Response of all vars to the selected shock at time t=1
        irf_array[1, :] = initial_impact

        # Subsequent periods: x_{t+1} = P_full @ x_t
        for t in range(1, periods):
            irf_array[t + 1, :] = p_full @ irf_array[t, :]

        # --- Format Output ---
        irf_df = pd.DataFrame(irf_array,
                              index=pd.RangeIndex(periods + 1, name='Period'),
                              columns=self.variable_names)

        print("IRF computed.")
        return irf_df

    def simulate(self, periods=100, shock_cov=None, initial_state=None, seed=None):
        """
        Simulates the model forward for a given number of periods.

        Args:
            periods (int): Number of simulation periods.
            shock_cov (np.ndarray, optional): Covariance matrix for the shocks.
                If None, assumes identity matrix (orthogonal shocks of variance 1).
                Must be shape (n_shocks, n_shocks).
            initial_state (np.ndarray or pd.Series, optional): Starting values for
                 *all* variables (states and controls) at t=0. Defaults to steady state (zero).
                 If array, order must match self.variable_names.
            seed (int, optional): Random seed for shock generation.

        Returns:
            pandas.DataFrame: DataFrame containing the simulated variable paths.
                              Index is period (0 to periods), columns are variable names.
        """
        if not self.is_solved:
            raise RuntimeError("Model must be solved before simulating.")
        if self.c is None:
             raise RuntimeError("Jacobian C matrix not available. Recompute Jacobians.")

        rng = np.random.default_rng(seed)

        # --- Setup Shocks ---
        if self.n_shocks == 0:
            print("Warning: Model has no shocks. Simulating deterministic path.")
            shocks = np.zeros((periods, 0)) # No shocks to generate
            shock_cov = np.zeros((0, 0))
        elif shock_cov is None:
            # Assume identity covariance matrix
            shock_cov = np.eye(self.n_shocks)
            print("Assuming identity shock covariance matrix.")
        elif not isinstance(shock_cov, np.ndarray) or shock_cov.shape != (self.n_shocks, self.n_shocks):
            raise ValueError(f"shock_cov must be a numpy array of shape ({self.n_shocks}, {self.n_shocks})")

        # Generate random shocks (mean zero) if shocks exist
        if self.n_shocks > 0:
             shocks = rng.multivariate_normal(np.zeros(self.n_shocks), shock_cov, size=periods) # Shape (periods, n_shocks)
        else:
             shocks = np.zeros((periods, 0)) # Empty array if no shocks


        # --- Construct p_full and shock impact d (same as IRF) ---
        p_full = np.zeros((self.n_vars, self.n_vars))
        p_full[:self.n_states, :self.n_states] = self.p
        p_full[self.n_states:, :self.n_states] = self.f @ self.p
        try:
            # Only calculate d if there are shocks
            if self.n_shocks > 0:
                coeff_matrix = self.a @ p_full + self.b
                d = scipy.linalg.solve(coeff_matrix, -self.c) # Shape (n_vars, n_shocks)
            else:
                d = np.zeros((self.n_vars, 0)) # Empty impact matrix if no shocks

        except Exception as e:
             print(f"ERROR: Could not compute shock impact matrix 'd' for simulation: {e}")
             raise RuntimeError("Simulation failed during setup.") from e


        # --- Setup Simulation Array ---
        simulation_array = np.zeros((periods + 1, self.n_vars))

        # Set initial state (at t=0, index 0)
        if initial_state is None:
            simulation_array[0, :] = self.steady_state.loc[self.variable_names].values
        elif isinstance(initial_state, pd.Series):
             try:
                  # Ensure index matches and fill missing with SS value (usually 0)
                  ss_dict = self.steady_state.to_dict()
                  initial_vals = initial_state.reindex(self.variable_names).fillna(pd.Series(ss_dict))
                  if initial_vals.isnull().any():
                      print("Warning: initial_state Series or steady_state contains NaNs.")
                  simulation_array[0, :] = initial_vals.values

             except KeyError:
                  raise ValueError("initial_state Series missing required variable names.")
             except Exception as e:
                  print(f"Error processing initial_state Series: {e}")
                  simulation_array[0, :] = self.steady_state.loc[self.variable_names].values # Fallback
        elif isinstance(initial_state, np.ndarray):
             if initial_state.shape == (self.n_vars,):
                 simulation_array[0, :] = initial_state
             else:
                 raise ValueError(f"initial_state array has wrong shape {initial_state.shape}, expected ({self.n_vars},)")
        else:
             raise TypeError("initial_state must be Series, array, or None.")

        # --- Run Simulation Loop ---
        print(f"Simulating model for {periods} periods...")
        for t in range(periods):
            # state_{t+1} = P_full @ state_t + d @ shock_{t+1}
            current_state_vec = simulation_array[t, :]
            # Handle case with no shocks
            shock_impact = d @ shocks[t, :] if self.n_shocks > 0 else np.zeros(self.n_vars)
            next_state_vec = (p_full @ current_state_vec) + shock_impact
            simulation_array[t + 1, :] = next_state_vec

        # --- Format Output ---
        sim_df = pd.DataFrame(simulation_array,
                              index=pd.RangeIndex(periods + 1, name='Period'),
                              columns=self.variable_names)

        print("Simulation complete.")
        return sim_df


