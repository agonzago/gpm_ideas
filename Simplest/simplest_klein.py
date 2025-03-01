#%%
import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Union

def build_klein_matrices(variables: Dict[str, Dict[str, List[str]]], 
                        equations: Dict[str, str],
                        parameters: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build matrices A and B for Klein's method using symbolic differentiation.
    
    Args:
        variables: Dictionary with keys 'states' and 'controls', each containing a list of variable names
                  States are predetermined variables, controls are jump variables
        equations: Dictionary mapping equation names to equation strings in SymPy-compatible format
                   Each equation should be in the form 'expression = 0'
        parameters: Dictionary mapping parameter names to their float values
        
    Returns:
        A, B: Numpy arrays representing matrices in the system A*E[x(t+1)] = B*x(t)
    """
    # Create symbolic variables for all time periods needed
    symbolic_vars = {}
    all_vars = variables['states'] + variables['controls']
    
    # Create symbolic variables for t-1, t, and t+1 periods
    for var in all_vars:
        symbolic_vars[f"{var}(-1)"] = sp.Symbol(f"{var}(-1)")
        symbolic_vars[f"{var}"] = sp.Symbol(f"{var}")
        symbolic_vars[f"{var}(+1)"] = sp.Symbol(f"{var}(+1)")
    
    # Add parameters as symbolic constants
    for param, value in parameters.items():
        symbolic_vars[param] = sp.Symbol(param)
    
    # Parse equations into sympy expressions
    parsed_eqs = {}
    for eq_name, eq_str in equations.items():
        # Split at the equals sign
        if "=" in eq_str:
            lhs_str, rhs_str = eq_str.split("=")
            # Create expression where LHS - RHS = 0
            eq_expr = sp.sympify(lhs_str, locals=symbolic_vars) - sp.sympify(rhs_str, locals=symbolic_vars)
        else:
            # If no equals sign, assume the expression is already in the form expr = 0
            eq_expr = sp.sympify(eq_str, locals=symbolic_vars)
        
        parsed_eqs[eq_name] = eq_expr
    
    # Determine system dimensions
    n = len(all_vars)
    
    # Initialize matrices A and B
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    # Fill matrices by computing gradients of each equation
    for i, eq_name in enumerate(equations.keys()):
        eq = parsed_eqs[eq_name]
        
        # Fill matrix A with derivatives w.r.t. t+1 variables
        for j, var in enumerate(all_vars):
            var_t_plus_1 = symbolic_vars[f"{var}(+1)"]
            # Compute derivative and substitute parameter values
            derivative = sp.diff(eq, var_t_plus_1)
            for param, value in parameters.items():
                derivative = derivative.subs(symbolic_vars[param], value)
            
            # Convert to float and store in matrix A
            A[i, j] = float(derivative)
        
        # Fill matrix B with negated derivatives w.r.t. t variables
        for j, var in enumerate(all_vars):
            var_t = symbolic_vars[f"{var}"]
            # Compute derivative and substitute parameter values
            derivative = sp.diff(eq, var_t)
            for param, value in parameters.items():
                derivative = derivative.subs(symbolic_vars[param], value)
            
            # Negate, convert to float, and store in matrix B
            B[i, j] = -float(derivative)
    
    return A, B

def build_dsge_model(variables_dict: Dict[str, List[str]],
                    equations_dict: Dict[str, str],
                    parameters_dict: Dict[str, float],
                    timing_conventions: Dict[str, Dict[str, List[int]]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build matrices A and B for a DSGE model with arbitrary timing conventions.
    
    Args:
        variables_dict: Dictionary mapping variable categories to lists of variable names
                       (e.g., {'endogenous': [...], 'exogenous': [...]})
        equations_dict: Dictionary mapping equation names to equation strings
        parameters_dict: Dictionary mapping parameter names to their values
        timing_conventions: Dictionary specifying which lags and leads appear in the model
                           (e.g., {'var1': {'lags': [-1], 'leads': [1, 3]}})
        
    Returns:
        A, B: Matrices for the expanded system AE[x(t+1)] = Bx(t)
        nk: Number of predetermined variables
    """
    # Step 1: Identify all unique variable timings
    all_timings = set()
    for var, timings in timing_conventions.items():
        lags = timings.get('lags', [])
        leads = timings.get('leads', [])
        all_timings.update(lags)
        all_timings.update(leads)
    
    min_lag = min(all_timings) if all_timings else 0
    max_lead = max(all_timings) if all_timings else 0
    
    # Step 2: Expand state vector to include all necessary lags and leads
    expanded_states = []
    expanded_controls = []
    
    # Auxiliary variables for expectations
    aux_variables = []
    
    # Map from original variable and timing to position in expanded vector
    var_mapping = {}
    
    # First, add current and lagged endogenous variables
    for var in variables_dict.get('states', []):
        # Add current period
        expanded_states.append(var)
        var_mapping[(var, 0)] = len(expanded_states) - 1
        
        # Add lags
        for lag in sorted([l for l in timing_conventions.get(var, {}).get('lags', []) if l < 0]):
            expanded_states.append(f"{var}_{lag}")
            var_mapping[(var, lag)] = len(expanded_states) - 1
    
    for var in variables_dict.get('controls', []):
        # Add current period
        expanded_controls.append(var)
        var_mapping[(var, 0)] = len(expanded_states) + len(expanded_controls) - 1
    
    # Add auxiliary expectation variables for leads > 1
    for var in variables_dict.get('states', []) + variables_dict.get('controls', []):
        leads = [l for l in timing_conventions.get(var, {}).get('leads', []) if l > 1]
        if leads:
            # Create auxiliary variables e1, e2, ... for each lead
            for lead in range(1, max(leads) + 1):
                aux_name = f"e_{var}_{lead}"
                aux_variables.append(aux_name)
                var_mapping[(var, lead)] = len(expanded_states) + len(expanded_controls) + len(aux_variables) - 1
    
    # Combine all variables
    all_expanded_vars = expanded_states + expanded_controls + aux_variables
    n = len(all_expanded_vars)
    
    # Calculate number of predetermined variables
    nk = len(expanded_states)  # States and lags are predetermined
    
    # Initialize matrices with zeros
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    # Step 3: Fill in the matrices based on the model equations
    
    # TODO: Implement the symbolic differentiation for the expanded system
    # This would require parsing the equations, replacing variables with their 
    # expanded counterparts, and computing gradients
    
    return A, B, nk

def klein_solve(A, B, nk):
    """
    Solve a linear rational expectations model using Klein's method.
    
    Args:
        A: The A matrix in AE[x(t+1)] = Bx(t)
        B: The B matrix in AE[x(t+1)] = Bx(t)
        nk: Number of predetermined variables
        
    Returns:
        F: Decision rule such that u(t) = F*k(t)
        P: Law of motion such that k(t+1) = P*k(t)
    """
    from scipy.linalg import ordqz
    
    # QZ decomposition with reordering
    # 'iuc' sorts eigenvalues inside unit circle first
    AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='iuc')
    
    # Calculate and check eigenvalues properly
    # stable_count = 0
    # for i in range(len(alpha)):
    #     if abs(beta[i]) < 1e-10:
    #         # If beta is zero, the eigenvalue is infinity (unstable)
    #         # Don't count as stable
    #         continue
    #     else:
    #         # Regular case - calculate eigenvalue and check stability
    #         eigenvalue = alpha[i] / beta[i]
    #         if abs(eigenvalue) < 1.0:
    #             stable_count += 1

    # Check if we have the right number of stable eigenvalues
    # print(f"Stable eigenvalues: {stable_count}")
    # if stable_count != nk:
    #     raise ValueError(f"Found {stable_count} stable eigenvalues, expected {nk}")
    
    # Extract submatrices
    Z11 = Z[:nk, :nk]
    Z21 = Z[nk:, :nk]
    
    # Check invertibility
    if np.linalg.matrix_rank(Z11) < nk:
        raise ValueError("Z11 is not invertible - unique stable solution doesn't exist")
    
    # Compute the solution
    Z11i = np.linalg.inv(Z11)
    S11 = AA[:nk, :nk]
    T11 = BB[:nk, :nk]
    
    # Compute dynamics matrix
    dyn = np.linalg.solve(S11, T11)
    
    # Compute policy and transition functions
    F = Z21 @ Z11i
    P = Z11 @ dyn @ Z11i
    
    # Convert to real if the model has real coefficients
    F = np.real_if_close(F)
    P = np.real_if_close(P)
    
    return F, P


# Example usage with simplified QPM model
if __name__ == "__main__":
    # Define variables
    variables = {
        'states': ['L_GDP_GAP_lag', 'DLA_CPI_lag', 'RS_lag', 'e1', 'e2', 'e3'],
        'controls': ['L_GDP_GAP', 'DLA_CPI', 'RS', 'RR_GAP']
    }
    
    # Define equations (this should match your QPM model)
    equations = {
        'is_curve': "L_GDP_GAP - (1-b1)*L_GDP_GAP(+1) - b1*L_GDP_GAP_lag + b4*RR_GAP(+1)",
        'phillips': "DLA_CPI - a1*DLA_CPI_lag - (1-a1)*e1 - a2*L_GDP_GAP",
        'monetary_policy': "RS - g1*RS_lag - (1-g1)*(e1 + g2*e3 + g3*L_GDP_GAP)",
        'real_rate': "RR_GAP - RS + e1",
        'lgdp_lag_evol': "L_GDP_GAP_lag(+1) - L_GDP_GAP",
        'dlacpi_lag_evol': "DLA_CPI_lag(+1) - DLA_CPI",
        'rs_lag_evol': "RS_lag(+1) - RS",
        'e1_def': "e1 - DLA_CPI(+1)",
        'e2_def': "e2 - e1(+1)",
        'e3_def': "e3 - e2(+1)"
    }
    
    # Define parameters
    parameters = {
        'b1': 0.7,
        'b4': 0.7,
        'a1': 0.5,
        'a2': 0.1,
        'g1': 0.7,
        'g2': 0.3,
        'g3': 0.25
    }
    
    # Build matrices
    A, B = build_klein_matrices(variables, equations, parameters)
    
    # Solve model
    nk = len(variables['states'])  # Number of predetermined variables
    F, P = klein_solve(A, B, nk)
    
    print("Decision rule F:")
    print(F)
    print("\nTransition matrix P:")
    print(P)