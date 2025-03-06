#%%
# gpm/solver/klein.py
import numpy as np
import warnings
from scipy.linalg import ordqz

# gpm/utils/irfs.py
import numpy as np
import matplotlib.pyplot as plt

# gpm/model/jacobian.py
import sympy as sy
import numpy as np
import re

"""
Simple Dynare Parser for Klein's Method
"""

import re
import os

class DynareParser:
    def __init__(self):
        self.variables = []
        self.exogenous = []
        self.parameters = []
        self.param_values = {}
        self.equations = []
        self.max_lead = 0
        self.max_lag = 0
        self.transformed_variables = []
        self.transformed_equations = []
        
    def parse_file(self, file_content):
        """Parse a Dynare file content into structured data"""
        # Extract var section
        var_match = re.search(r'var\s+(.*?);', file_content, re.DOTALL)
        if var_match:
            var_section = var_match.group(1)
            # Remove comments and extract variable names
            var_section = re.sub(r'//.*?$', '', var_section, flags=re.MULTILINE)
            self.variables = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', var_section)]
        
        # Extract varexo section
        varexo_match = re.search(r'varexo\s+(.*?);', file_content, re.DOTALL)
        if varexo_match:
            varexo_section = varexo_match.group(1)
            varexo_section = re.sub(r'//.*?$', '', varexo_section, flags=re.MULTILINE)
            self.exogenous = [v.strip() for v in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', varexo_section)]
        
        # Extract parameters
        param_match = re.search(r'parameters\s+(.*?);', file_content, re.DOTALL)
        if param_match:
            param_section = param_match.group(1)
            param_section = re.sub(r'//.*?$', '', param_section, flags=re.MULTILINE)
            self.parameters = [p.strip() for p in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', param_section)]
        
        # Extract parameter values
        for param in self.parameters:
            param_value_match = re.search(rf'{param}\s*=\s*([0-9.]+)\s*;', file_content)
            if param_value_match:
                self.param_values[param] = float(param_value_match.group(1))
        
        # Extract model equations
        model_match = re.search(r'model;(.*?)end;', file_content, re.DOTALL)
        if model_match:
            model_section = model_match.group(1)
            # Clean up comments and split into equations
            cleaned_lines = []
            for line in model_section.split(';'):
                line = re.sub(r'//.*?$', '', line, flags=re.MULTILINE).strip()
                if line:
                    cleaned_lines.append(line)
            self.equations = cleaned_lines
        
        # Find max lead and lag
        for eq in self.equations:
            # Search for leads like varname(+n)
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq)
            for var, lead in lead_matches:
                self.max_lead = max(self.max_lead, int(lead))
            
            # Search for lags like varname(-n)
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq)
            for var, lag in lag_matches:
                self.max_lag = max(self.max_lag, int(lag))
    
    def transform_model(self):
        """Transform the Dynare model into a system with only t and t+1 variables"""
        # Create the transformed variables list
        self.transformed_variables = self.variables.copy()
        
        # Track which variables have lags and leads and their maximum lag/lead
        var_max_lag = {}
        var_max_lead = {}
        
        # First pass - identify what needs transforming
        for eq in self.equations:
            # Clean the equation of any comments before processing
            eq_clean = re.sub(r'//.*$', '', eq).strip()
            
            # Find all variables with leads
            lead_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+(\d+)\)', eq_clean)
            for var, lead in lead_matches:
                lead_val = int(lead)
                if lead_val >= 1:
                    var_max_lead[var] = max(var_max_lead.get(var, 0), lead_val)
            
            # Find all variables with lags
            lag_matches = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-(\d+)\)', eq_clean)
            for var, lag in lag_matches:
                lag_val = int(lag)
                if lag_val >= 1:
                    var_max_lag[var] = max(var_max_lag.get(var, 0), lag_val)
        
        # Add lag variables to transformed variables list only for variables that have lags
        for var, max_lag in var_max_lag.items():
            for lag in range(1, max_lag + 1):
                lag_suffix = str(lag) if lag > 1 else ""
                self.transformed_variables.append(f"{var}_lag{lag_suffix}")
        
        # Add lead variables beyond +1 to transformed variables list only for variables that have leads beyond +1
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:
                for lead in range(1, max_lead + 1):
                    self.transformed_variables.append(f"{var}_lead{lead}")
        
        # Transform equations
        for i, eq in enumerate(self.equations):
            # Remove comments
            eq_clean = re.sub(r'//.*$', '', eq).strip()
            transformed_eq = eq_clean
            
            # Replace leads with corresponding variables
            for lead in range(self.max_lead, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\+' + str(lead) + r'\)'
                if lead == 1:
                    # For +1, use _p suffix (next period)
                    transformed_eq = re.sub(pattern, r'\1_p', transformed_eq)
                else:
                    # For +2 and higher, use _lead suffix
                    transformed_eq = re.sub(pattern, r'\1_lead' + str(lead), transformed_eq)
            
            # Replace lags with corresponding variables
            for lag in range(self.max_lag, 0, -1):
                pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\(\-' + str(lag) + r'\)'
                lag_suffix = str(lag) if lag > 1 else ""
                transformed_eq = re.sub(pattern, r'\1_lag' + lag_suffix, transformed_eq)
            
            # Split the equation on the equals sign
            if "=" in transformed_eq:
                lhs, rhs = transformed_eq.split("=", 1)
                transformed_eq = f"{rhs.strip()} - {lhs.strip()}"

                
            self.transformed_equations.append({f"eq{i+1}": transformed_eq.strip()})
        
        # Add transition equations for lags, only for variables that have lags
        eq_num = len(self.transformed_equations) + 1
        for var, max_lag in var_max_lag.items():
            # First lag: var_lag_p = var
            self.transformed_equations.append({f"eq{eq_num}": f"{var}_lag_p - {var}"})
            eq_num += 1
            
            # Additional lags: var_lagN_p = var_lag(N-1)
            for lag in range(2, max_lag + 1):
                prev_lag_suffix = str(lag-1) if lag-1 > 1 else ""
                self.transformed_equations.append({f"eq{eq_num}": f"{var}_lag{lag}_p - {var}_lag{prev_lag_suffix}"})
                eq_num += 1
        
        # Add transition equations for leads, only for variables that have leads beyond +1
        for var, max_lead in var_max_lead.items():
            if max_lead > 1:
                # First lead equation: var_p = var_lead1
                self.transformed_equations.append({f"eq{eq_num}": f"{var}_p - {var}_lead1"})
                eq_num += 1
                
                # Additional lead equations: var_leadN_p = var_lead(N+1)
                for lead in range(1, max_lead):
                    self.transformed_equations.append({f"eq{eq_num}": f"{var}_lead{lead}_p - {var}_lead{lead+1}"})
                    eq_num += 1
    
    def generate_output(self):
        """Generate the output script with the transformed model"""
        output = ""
        
        # Output equations
        output += "equations = {\n"
        for i, eq_dict in enumerate(self.transformed_equations):
            for key, value in eq_dict.items():
                output += f'    {{"{key}": "{value}"}}'
                if i < len(self.transformed_equations) - 1:
                    output += ","
                output += "\n"
        output += "};\n\n"
        
        # Output variables
        variables_str = ", ".join([f'"{var}"' for var in self.transformed_variables])
        output += f"variables = [{variables_str}];\n\n"
        
        # Output parameters
        parameters_str = ", ".join([f'"{param}"' for param in self.parameters])
        output += f"parameters = [{parameters_str}];\n\n"
        
        # Output parameter values
        param_values_str = "\n".join([f"{param} = {value};" for param, value in self.param_values.items()])
        output += f"{param_values_str}\n\n"
        
        # Output shocks
        shocks_str = ", ".join([f'"{shock}"' for shock in self.exogenous])
        output += f"shocks = [{shocks_str}];\n"
        
        return output


def parse_dynare_file(filename):
    """
    Parse a Dynare model file and transform it for Klein's method
    
    Args:
        filename (str): Path to the Dynare file
    
    Returns:
        dict: Dictionary with transformed model components
    """
    # Read the Dynare file
    with open(filename, 'r') as f:
        dynare_content = f.read()
    
    # Parse and transform
    parser = DynareParser()
    parser.parse_file(dynare_content)
    parser.transform_model()
    
    # Return a dictionary with all components
    return {
        'equations': parser.transformed_equations,
        'variables': parser.transformed_variables,
        'parameters': parser.parameters,
        'param_values': parser.param_values,
        'shocks': parser.exogenous,
        'output_text': parser.generate_output()
    }


def save_transformed_model(parsed_model, output_file):
    """
    Save the transformed model to a file
    
    Args:
        parsed_model (dict): Parsed model from parse_dynare_file()
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write(parsed_model['output_text'])


def solve_klein(A, B, nk):
    """
    Solve the linear rational expectations model using Klein's method.
    
    Args:
        A: The A matrix in AE[x(t+1)|t] = Bx(t)
        B: The B matrix in AE[x(t+1)|t] = Bx(t)
        nk: Number of predetermined variables
        
    Returns:
        F: Decision rule such that u(t) = F*k(t)
        P: Law of motion such that k(t+1) = P*k(t)
    """
    # QZ decomposition with reordering        
    S, T, alpha, beta, Q, Z = ordqz(A, B, sort='ouc')

    # Check if we have the correct number of stable eigenvalues
    if abs(T[nk-1,nk-1]) > abs(S[nk-1,nk-1]) or abs(T[nk,nk]) < abs(S[nk,nk]):
        warnings.warn('Wrong number of stable eigenvalues.')

    # Calculate generalized eigenvalues
    eigenvalues = []
    for i in range(len(S)):
        if abs(S[i, i]) < 1e-10:
            eigenvalues.append(float('inf'))  # Infinity for zero on diagonal of S
        else:
            eigenvalues.append(T[i, i] / S[i, i])
    
    # Extract submatrices
    Z11 = Z[:nk, :nk]
    Z21 = Z[nk:, :nk]
    
    # Check invertibility
    if np.linalg.matrix_rank(Z11) < nk:
        warnings.warn("Z11 is not invertible - unique stable solution doesn't exist")
        return None, None
    
    # Compute the solution
    S11 = S[:nk, :nk]
    T11 = T[:nk, :nk]
    
    # Compute dynamics matrix
    dyn = np.linalg.solve(S11, T11)
    
    # Compute policy and transition functions
    F = Z21 @ np.linalg.inv(Z11)
    P = Z11 @ dyn @ np.linalg.inv(Z11)
    
    # Convert to real if the model has real coefficients
    F = np.real_if_close(F)
    P = np.real_if_close(P)
    
    return F, P

def reorder_variables_for_klein(variables):
    """
    Reorder variables to put predetermined variables first, followed by controls
    
    Args:
        variables: List of variable names
        
    Returns:
        Reordered list of variables
    """
    # Identify different types of variables
    pred_endo = [var for var in variables if '_lag' in var]
    exo_states = [var for var in variables if var.startswith('RES_')]
    others = [var for var in variables if 
              '_lag' not in var and 
              not var.startswith('RES_') and
              '_p' not in var and
              '_lead' not in var]
    
    # Return reordered list
    return pred_endo + exo_states + others

def derive_jacobians_economic_model(equations_dict, variables, shocks, parameters):
    """
    Derives symbolic Jacobian matrices for an economic model with the form F(X, X_p, W, θ) = 0
    where X are variables, X_p are the same variables with "_p" suffix, and W are shocks.
    
    Computes:
    - A = -∂F/∂X_p (negative Jacobian with respect to variables with "_p" suffix)
    - B = ∂F/∂X (Jacobian with respect to variables without "_p" suffix)
    - C = ∂F/∂W (Jacobian with respect to shock variables)
    
    Args:
        equations_dict: Dictionary of equations in the format {eq_name: equation_string}
        variables: List of variable names (without "_p" suffix)
        shocks: List of shock variable names
        parameters: List of parameter names (theta vector)
        
    Returns:
        A Python script as a string that defines a function to efficiently evaluate Jacobians
    """
    # Create the list of variables with "_p" suffix
    variables_p = [var + "_p" for var in variables]
    
    # Create symbolic variables
    var_symbols = {var: sy.symbols(var) for var in variables}
    var_p_symbols = {var_p: sy.symbols(var_p) for var_p in variables_p}
    shock_symbols = {shock: sy.symbols(shock) for shock in shocks}
    param_symbols = {param: sy.symbols(param) for param in parameters}
    
    # Combine all symbols in a dictionary
    all_symbols = {**var_symbols, **var_p_symbols, **shock_symbols, **param_symbols}
    
    # Parse equations into sympy expressions
    equations = []
    for eq_name, eq_str in equations_dict.items():
        # Convert string to sympy expression
        eq_expr = eq_str
        for name, symbol in all_symbols.items():
            eq_expr = eq_expr.replace(name, str(symbol))
        
        # Less specific regex that should match more equation formats
        match = re.search(r'(.*) - (?:\()?(.*)(?:\))?', eq_expr)
        if match:
            lhs = match.group(1)
            rhs = match.group(2)
            # Create expression of form lhs - rhs = 0
            expr = sy.sympify(lhs) - sy.sympify(rhs)
            equations.append(expr)
    
    # Create system as sympy Matrix
    F = sy.Matrix(equations)
    
    # Compute Jacobians
    X_symbols = [var_symbols[var] for var in variables]
    X_p_symbols = [var_p_symbols[var_p] for var_p in variables_p]
    W_symbols = [shock_symbols[shock] for shock in shocks]
    
    # B = ∂F/∂X (with respect to variables without "_p")
    B_symbolic = F.jacobian(X_symbols)
    
    # A = -∂F/∂X_p (negative Jacobian with respect to variables with "_p")
    A_symbolic = -F.jacobian(X_p_symbols)
    
    # C = ∂F/∂W (with respect to shock variables)
    C_symbolic = F.jacobian(W_symbols)
    
    # Generate Python function that evaluates these Jacobians efficiently
    function_code = [
        "import numpy as np",
        "",
        "def evaluate_jacobians(theta):",
        "    \"\"\"",
        "    Evaluates Jacobian matrices for an economic model based on parameter values.",
        "    ",
        "    Args:",
        "        theta: List or array of parameter values in the order of:",
        f"              {parameters}",
        "        ",
        "    Returns:",
        "        A: Matrix -∂F/∂X_p (negative Jacobian with respect to variables with '_p' suffix)",
        "        B: Matrix ∂F/∂X (Jacobian with respect to variables without '_p' suffix)",
        "        C: Matrix ∂F/∂W (Jacobian with respect to shock variables)",
        "    \"\"\"",
        "    # Unpack parameters from theta",
    ]
    
    # Add parameter unpacking
    for i, param in enumerate(parameters):
        function_code.append(f"    {param}_ = theta[{i}]")
    
    # Initialize A, B, and C matrices
    function_code.extend([
        "",
        f"    A = np.zeros(({len(equations)}, {len(variables)}))",
        f"    B = np.zeros(({len(equations)}, {len(variables)}))",
        f"    C = np.zeros(({len(equations)}, {len(shocks)}))",
        ""
    ])
    
    # Add A matrix elements
    for i in range(A_symbolic.rows):
        for j in range(A_symbolic.cols):
            if A_symbolic[i, j] != 0:
                expr = str(A_symbolic[i, j])
                for param in parameters:
                    expr = expr.replace(param, f"{param}_")
                function_code.append(f"    A[{i}, {j}] = {expr}")
    
    # Add B matrix elements
    function_code.append("")
    for i in range(B_symbolic.rows):
        for j in range(B_symbolic.cols):
            if B_symbolic[i, j] != 0:
                expr = str(B_symbolic[i, j])
                for param in parameters:
                    expr = expr.replace(param, f"{param}_")
                function_code.append(f"    B[{i}, {j}] = {expr}")
    
    # Add C matrix elements
    function_code.append("")
    for i in range(C_symbolic.rows):
        for j in range(C_symbolic.cols):
            if C_symbolic[i, j] != 0:
                expr = str(C_symbolic[i, j])
                for param in parameters:
                    expr = expr.replace(param, f"{param}_")
                function_code.append(f"    C[{i}, {j}] = {expr}")
    
    # Return A, B, and C
    function_code.extend([
        "",
        "    return A, B, C"
    ])
    
    return "\n".join(function_code)


def build_state_space(F, P, num_k, num_exo, num_control_obs):
    """
    Build state space representation matching the Fortran implementation
    
    Args:
        F: Policy function matrix
        P: State transition matrix
        num_k: Number of predetermined variables
        num_exo: Number of exogenous shock variables
        num_control_obs: Number of observable control variables
    """
    # Define dimensions (as in KALMANFILTERMATRICES)
    num_est = num_control_obs + num_k
    n_k = num_control_obs  # Controls
    k_ex = num_k - num_exo  # Predetermined endogenous
    n_ex = num_est - num_exo  # All except exogenous shocks
    
    # Initialize transition and shock matrices
    T = np.zeros((num_est, num_est))
    R = np.zeros((num_est, num_exo))
    
    # Fill transition matrix T
    # Controls impacted by states
    T[:n_k, n_k:n_ex] = F[:, :k_ex]  # Impact of pred_endo on controls
    T[:n_k, n_ex:] = np.dot(F[:, k_ex:], P[k_ex:, k_ex:])  # Impact of exo_states
    
    # State transitions
    T[n_k:n_ex, n_k:n_ex] = P[:k_ex, :k_ex]  # Pred_endo dynamics
    T[n_k:n_ex, n_ex:] = np.dot(P[:k_ex, k_ex:], P[k_ex:, k_ex:])  # Exo impact on pred_endo
    T[n_ex:, n_ex:] = P[k_ex:, k_ex:]  # Exogenous shock processes
    
    # Fill shock impact matrix R
    R[:n_k, :] = F[:, k_ex:]  # Impact on controls
    R[n_k:n_ex, :] = P[:k_ex, k_ex:]  # Impact on pred_endo
    
    # Direct impact of innovations on exogenous states
    for j in range(num_exo):
        R[n_ex+j, j] = 1.0
    
    return T, R

def setup_model_and_state_space(parsed_model, jacobian_file="_jacobian_evaluator.py"):
    """
    Setup the model variables and create the state space representation
    using a pre-generated Jacobian evaluator
    """
    # Extract components from parsed model
    variables = parsed_model['variables']
    equations = {key: eq for d in parsed_model['equations'] for key, eq in d.items()}
    parameters = parsed_model['parameters']
    param_values = parsed_model['param_values']
    shocks = parsed_model['shocks']
    
    # In transformed variables, we need to identify:
    # 1. Lagged endogenous variables (e.g., X_lag)
    # 2. Exogenous state variables (e.g., RES_X)
    
    # More precise identification of variable types
    pred_endo = []
    exo_states = []
    controls = []

    # First identify the original variables from the transformed ones
    original_vars = set()
    for var in variables:
        if '_lag' in var:
            # Extract original variable name from X_lag to get X
            original_name = var.split('_lag')[0]
            original_vars.add(original_name)
        elif '_lead' in var or '_p' in var:
            # Skip leads/future vars as they're not original variables
            continue
        else:
            original_vars.add(var)
    
    # Now classify each transformed variable
    for var in variables:
        if '_lag' in var:
            pred_endo.append(var)
        elif var.startswith('RES_'):
            exo_states.append(var)
        elif var in original_vars and not var.startswith('RES_'):
            # Current-period endogenous variables
            controls.append(var)
    
    # Calculate key dimensions
    num_pred_endo = len(pred_endo)
    num_exo = len(exo_states)
    num_k = num_pred_endo + num_exo  # Total predetermined variables
    num_control_obs = len(controls)
    
    print(f"Predetermined endogenous: {num_pred_endo} - {pred_endo}")
    print(f"Exogenous states: {num_exo} - {exo_states}")
    print(f"Control variables: {num_control_obs} - {controls}")
    print(f"Total state dimension: {num_k}")
    
    # Load the Jacobian evaluator from file
    print(f"Loading Jacobian evaluator from {jacobian_file}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("jacobian_evaluator", jacobian_file)
    jacobian_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jacobian_module)
    evaluate_jacobians = jacobian_module.evaluate_jacobians
    
    # Prepare parameters and evaluate Jacobians
    print("Evaluating Jacobians...")
    theta = [param_values[param] for param in parameters]
    A, B, C = evaluate_jacobians(theta)
    
    # Check matrix dimensions
    print(f"Jacobian A shape: {A.shape}")
    print(f"Jacobian B shape: {B.shape}")
    print(f"Jacobian C shape: {C.shape}")
    
    # Solve the model with Klein's method
    print("Solving model using Klein's method...")
    F, P = solve_klein(A, B, num_k)
    
    if F is None or P is None:
        print("Failed to find stable solution")
        return None
    
    # Check solution matrix dimensions
    print(f"Solution F shape: {F.shape}")
    print(f"Solution P shape: {P.shape}")
    
    # Build state space with the correct specification
    print("Building state space representation...")
    T, R = build_state_space(F, P, num_k, num_exo, num_control_obs)
    
    print(f"State transition matrix T shape: {T.shape}")
    print(f"Shock impact matrix R shape: {R.shape}")
    
    return {
        'T': T,
        'R': R,
        'F': F,
        'P': P,
        'A': A,
        'B': B,
        'C': C,
        'variable_info': {
            'pred_endo': pred_endo,
            'exo_states': exo_states,
            'controls': controls,
            'shocks': shocks,
            'all_variables': variables
        },
        'dimensions': {
            'num_k': num_k,
            'num_exo': num_exo,
            'num_control_obs': num_control_obs
        }
    }


def process_model(dynare_file, jacobian_file="_jacobian_evaluator.py", shock_name=None, periods=40):
    """
    Process a DSGE model from a Dynare file and compute IRFs
    """
    print(f"Processing model from {dynare_file}")
    
    # Step 1: Parse the Dynare file
    parsed_model = parse_dynare_file(dynare_file)
    
    # Step 2: Setup the model and create state space
    model_solution = setup_model_and_state_space(parsed_model, jacobian_file)
    
    if model_solution is None:
        print("Failed to solve model")
        return None
    
    # Step 3: Compute IRFs
    print("Computing impulse response functions...")
    
    # Find shock index if name provided
    shock_index = None
    if shock_name:
        shocks = model_solution['variable_info']['shocks']
        if shock_name in shocks:
            shock_index = shocks.index(shock_name)
            print(f"Using shock {shock_name} (index {shock_index})")
        else:
            print(f"Shock {shock_name} not found. Available shocks: {shocks}")
            return None
    
    irfs = compute_irfs(
        model_solution['T'],
        model_solution['R'],
        model_solution['dimensions'],
        model_solution['variable_info'],
        shock_index=shock_index,
        periods=periods
    )
    
    # Step 4: Plot IRFs
    main_vars = ['L_GDP_GAP', 'DLA_CPI', 'RS']
    plot_irfs(irfs, variables=main_vars, shock_name=f"SHK_{shock_name}" if shock_name else None)
    
    return {
        'model_solution': model_solution,
        'irfs': irfs
    }



def compute_irfs(T, R, dimensions, variable_info, shock_index=None, periods=40):
    """
    Compute impulse response functions using the state space representation
    
    Args:
        T: State transition matrix
        R: Shock impact matrix
        dimensions: Dictionary with model dimensions
        variable_info: Dictionary with variable classifications
        shock_index: Index of shock to simulate (None = all shocks)
        periods: Number of periods for IRF
    
    Returns:
        Dictionary of IRFs by shock and variable
    """
    num_vars = T.shape[0]
    num_shocks = R.shape[1]
    
    # Determine which shocks to compute IRFs for
    if shock_index is None:
        shock_indices = range(num_shocks)
    else:
        shock_indices = [shock_index]
    
    # Create IRF container
    irfs = {}
    
    # Get flat list of all variables
    all_vars = (
        variable_info['controls'] + 
        variable_info['pred_endo'] + 
        variable_info['exo_states']
    )
    
    # For each shock
    for s_idx in shock_indices:
        shock_name = f"SHK_{variable_info['shocks'][s_idx]}"
        irfs[shock_name] = {}
        
        # Initialize impulse vector
        x = np.zeros((num_vars, periods))
        
        # Initial impulse from shock
        x[:, 0] = R[:, s_idx]
        
        # Propagate through system
        for t in range(1, periods):
            x[:, t] = np.dot(T, x[:, t-1])
        
        # Store IRFs by variable
        for i, var in enumerate(all_vars):
            irfs[shock_name][var] = x[i, :]
    
    return irfs

def plot_irfs(irfs, variables=None, shock_name=None, figsize=(12, 8)):
    """
    Plot impulse response functions.
    
    Args:
        irfs (dict): Dictionary of IRFs from compute_irfs
        variables (list, optional): List of variables to plot. If None, plot all.
        shock_name (str, optional): Name of the shock. If None, use the first shock.
        figsize (tuple): Figure size.
    """
    if not irfs:
        raise ValueError("No IRFs to plot")
    
    # If shock_name is None, use the first shock
    if shock_name is None:
        shock_name = next(iter(irfs.keys()))
    
    # Get shock IRFs
    shock_irfs = irfs.get(shock_name)
    if shock_irfs is None:
        raise ValueError(f"Shock {shock_name} not found in IRFs")
    
    # If variables is None, plot all variables
    if variables is None:
        variables = list(shock_irfs.keys())
    
    # Number of variables to plot
    n_vars = len(variables)
    
    # Create grid of subplots
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each variable
    for i, var in enumerate(variables):
        if var in shock_irfs:
            x = np.arange(len(shock_irfs[var]))
            axes[i].plot(x, shock_irfs[var])
            axes[i].set_title(var)
            axes[i].grid(True)
            axes[i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Turn off any unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"IRFs for {shock_name}")
    plt.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":

    dynare_file = "/Volumes/TOSHIBA EXT/main_work/Work/Projects/gpm_ideas/Simplest/qpm_simpl1.dyn"
    print(f"Processing DSGE model from {dynare_file}...")
    
   
    # Step 1: Parse the Dynare file
    print("Parsing Dynare file...")
    parsed_model = parse_dynare_file(dynare_file)

    # Step 2: Save the parsed model to a file
    import json
    parsed_model_file = "parsed_model.json"
    print(f"Saving parsed model to {parsed_model_file}...")

    # Convert sets to lists for JSON serialization if needed
    serializable_model = {}
    for key, value in parsed_model.items():
        if key == 'equations':
            # Handle the list of dictionaries structure
            serializable_model[key] = value
        else:
            serializable_model[key] = value

    with open(parsed_model_file, 'w') as f:
        json.dump(serializable_model, f, indent=2)


    #Step 4: Compute the Jacobian     
    print("Generating Jacobian evaluator...")    
    equations = {key: eq for d in parsed_model['equations'] for key, eq in d.items()}
    variables = parsed_model['variables']
    shocks = parsed_model['shocks']
    parameters = parsed_model['parameters']
    
    # Reorder variables for Klein method
    ordered_variables = reorder_variables_for_klein(variables)

    # Generate Jacobian code
    jacobian_code = derive_jacobians_economic_model(equations, ordered_variables, shocks, parameters)
    
    # Save the analytical Jacobian in the computer
    with open("_jacobian_evaluator.py", 'w') as f:
        f.write(jacobian_code) 

    # Step 4: Setup model and build state space
    # Process model using existing Jacobian evaluator
    results = process_model("qpm_simpl1.dyn", jacobian_file="_jacobian_evaluator.py", shock_name="SHK_L_GDP_GAP")
 
       

    # # Step 6: Load the jacobian
    # print("Importing Jacobian evaluation function...")
    # import importlib.util
    # spec = importlib.util.spec_from_file_location("jacobian_evaluator", jacobian_file)
   
    # jacobian_module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(jacobian_module)
    # evaluate_jacobians = jacobian_module.evaluate_jacobians

    # # Step 7: Evaluate Jacobians
    # print("Evaluating Jacobians...")
    # # Step 5: Prepare parameters for evaluation
    # theta = [param_values[param] for param in parameters]
    # A, B, C = evaluate_jacobians(theta)
    
    # # Step 7: Determine number of predetermined variables
    # nk = sum(1 for var in variables if '_lag' in var or var.endswith('_lag'))
    # print(f"Found {nk} predetermined variables")
    
    # # Step 8: Solve the model using Klein's method
    # print("Solving model using Klein's method...")
    # F, P = solve_klein(A, B, nk)

    # # Step 9: Build state space representation
    # print("Building state space representation...")
    # Phi, H = build_state_space(F, P, C)

    # # COmpute IRFs    
    # print("Computing impulse response functions...")
    # # If shock_name is provided, find its index
    # shock_name = "SHK_L_GDP_GAP"
    # shock_index = None
    # if shock_name:
    #     if shock_name in shocks:
    #         shock_index = shocks.index(shock_name)
    #         print(f"Using shock: {shock_name} (index {shock_index})")
    #     else:
    #         print(f"Warning: Shock {shock_name} not found in model shocks.")
    #         print(f"Available shocks: {shocks}")
    #         shock_index = 0  # Use first shock as default
    #         shock_name = shocks[0]
    #         print(f"Using first shock: {shock_name} instead")

    # # Compute IRFs
    
    
    # periods = 40
    # irfs = compute_irfs(Phi, H, shock_index, periods, parsed_model['variables'], variables)
   

    # # Plot IRFs    
    # print("Plotting impulse response functions...")
    # plot_irfs(irfs, None, f"{shock_name}" if shock_index is not None else None)
    # plt.tight_layout()
    # plt.show()
    # Process a DSGE model
    # results = process_dsge_model("/Volumes/TOSHIBA EXT/main_work/Work/Projects/gpm_ideas/Simplest/qpm_simpl1.dyn", shock_name="SHK_L_GDP_GAP")    
    # print( results)
    # # Print detailed solution
    # if results['success']:
    #     print_model_solution(results)
    
    # # You can also compare different policy scenarios
    # def compare_policy_scenarios(dynare_file, param_name, baseline_value, alternative_value, shock_name=None):
    #     """Compare IRFs between baseline and alternative parameter values"""
    #     print(f"\nComparing policy scenarios with different {param_name} values:")
    #     print(f"  Baseline: {param_name} = {baseline_value}")
    #     print(f"  Alternative: {param_name} = {alternative_value}")
        
    #     # Process baseline model
    #     baseline_results = process_dsge_model(dynare_file, shock_name=shock_name, plot=False)
        
    #     # Process alternative model with modified parameter
    #     # First parse the model
    #     parsed_model = parse_dynare_file(dynare_file)
        
    #     # Modify the parameter value
    #     parsed_model['param_values'][param_name] = alternative_value
        
    #     # Execute the same steps as in process_dsge_model
    #     variables = parsed_model['variables']
    #     equations = {key: eq for d in parsed_model['equations'] for key, eq in d.items()}
    #     parameters = parsed_model['parameters']
    #     param_values = parsed_model['param_values']
    #     shocks = parsed_model['shocks']
        
    #     # Derive and evaluate Jacobians
    #     jacobian_code = derive_jacobians_economic_model(equations, variables, shocks, parameters)
    #     jacobian_module = types.ModuleType('jacobian_module')
    #     exec(jacobian_code, jacobian_module.__dict__)
    #     evaluate_jacobians = jacobian_module.evaluate_jacobians
        
    #     theta = [param_values[param] for param in parameters]
    #     A, B, C = evaluate_jacobians(theta)
        
    #     nk = sum(1 for var in variables if '_lag' in var or var.endswith('_lag'))
    #     F, P = solve_klein(A, B, nk)
        
    #     if F is None or P is None:
    #         print("Warning: Alternative model could not be solved")
    #         return
        
    #     Phi, R_ss = build_state_space(F, P, C[:nk])
        
    #     # Find shock index
    #     shock_index = 0  # Default to first shock
    #     if shock_name and shock_name in shocks:
    #         shock_index = shocks.index(shock_name)
        
    #     # Compute IRFs for both models
    #     baseline_irfs = baseline_results['irfs']
    #     alt_irfs = compute_irfs(Phi, R_ss, shock_index, 40, parsed_model['variables'], variables)
        
    #     # Plot comparison
    #     shock_key = f"SHK_{shock_name}" if shock_name and shock_name in shocks else list(baseline_irfs.keys())[0]
        
    #     # Select variables to plot (original model variables, not transformed)
    #     orig_vars = parsed_model['variables'][:4]  # Just first few variables for clarity
        
    #     # Create plot
    #     fig, axes = plt.subplots(1, len(orig_vars), figsize=(15, 5))
    #     if len(orig_vars) == 1:
    #         axes = [axes]
            
    #     for i, var in enumerate(orig_vars):
    #         if var in baseline_irfs[shock_key] and var in alt_irfs[shock_key]:
    #             x = np.arange(len(baseline_irfs[shock_key][var]))
    #             axes[i].plot(x, baseline_irfs[shock_key][var], label=f'Baseline ({param_name}={baseline_value})')
    #             axes[i].plot(x, alt_irfs[shock_key][var], label=f'Alternative ({param_name}={alternative_value})')
    #             axes[i].set_title(var)
    #             axes[i].grid(True)
    #             axes[i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    #             axes[i].legend()
        
    #     plt.suptitle(f'Comparison of IRFs: Impact of changing {param_name}')
    #     plt.tight_layout()
    #     plt.show()
    
    # # Example policy comparison
    # compare_policy_scenarios("/Volumes/TOSHIBA EXT/main_work/Work/Projects/gpm_ideas/Simplest/qpm_simpl1.dyn", "g2", 0.3, 0.6, "SHK_L_GDP_GAP")


