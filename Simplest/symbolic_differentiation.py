#%%
import numpy as np
import sympy as sp
import re
import warnings
def create_klein_matrices(model_file):
    """
    Parse a model file and create the A and B matrices for Klein's method.
    
    Args:
        model_file: Path to the model specification file
    
    Returns:
        A: Matrix for forward-looking variables (multiplies E_t[x_{t+1}])
        B: Matrix for current variables (multiplies x_t)
        var_list: List of variables in order corresponding to matrix rows/columns
        predetermined_count: Number of predetermined variables
    """
    # Read the model file
    with open(model_file, 'r') as f:
        model_text = f.read()
    
    # Parse equations using regex
    equations = []
    eq_pattern = re.compile(r'\{"eq\d+"\s*:\s*"([^"]+)"\}')
    eq_matches = eq_pattern.findall(model_text)
    for i, eq_expr in enumerate(eq_matches, 1):
        equations.append({"eq" + str(i): eq_expr})
    
    # Parse variables list
    variables = []
    var_match = re.search(r'variables\s*=\s*\[(.*?)\];', model_text, re.DOTALL)
    if var_match:
        var_str = var_match.group(1)
        var_list = [v.strip().strip('"\'') for v in var_str.split(',')]
        variables = [v for v in var_list if v]  # Filter out empty strings
    
    # Parse parameter values
    parameters = {}
    param_pattern = re.compile(r'(\w+)\s*=\s*([\d\.]+);?')
    param_matches = param_pattern.findall(model_text)
    for param_name, param_value in param_matches:
        try:
            parameters[param_name] = float(param_value)
        except ValueError:
            pass  # Skip if not a valid float
    
    # Add g1, g2, g3 if not found but needed
    if 'g1' not in parameters and any('g1' in eq for eq_dict in equations for eq in eq_dict.values()):
        parameters['g1'] = 0.7  # Default value
    if 'g2' not in parameters and any('g2' in eq for eq_dict in equations for eq in eq_dict.values()):
        parameters['g2'] = 0.3  # Default value
    if 'g3' not in parameters and any('g3' in eq for eq_dict in equations for eq in eq_dict.values()):
        parameters['g3'] = 0.25  # Default value
    
    print(f"Parsed {len(equations)} equations")
    print(f"Parsed {len(variables)} variables")
    print(f"Parsed {len(parameters)} parameters: {parameters}")
    
    # Create symbolic variables
    sym_vars = {}
    for var in variables:
        # Current period variable
        sym_vars[var] = sp.Symbol(var)
        # Forward variable (t+1)
        sym_vars[f"{var}_p"] = sp.Symbol(f"{var}_p")
    
    # Parse equations into symbolic form
    sym_equations = []
    for eq_dict in equations:
        for eq_name, eq_expr in eq_dict.items():
            try:
                # Replace parameters with their values
                for param, value in parameters.items():
                    eq_expr = eq_expr.replace(param, str(value))
                
                # Handle equation format - split into LHS and RHS
                if ' - (' in eq_expr:
                    left_side, right_side = eq_expr.split(' - (', 1)
                    right_side = right_side.rstrip(')')
                elif ' - ' in eq_expr:
                    left_side, right_side = eq_expr.split(' - ', 1)
                else:
                    # Try to find another way to split
                    parts = eq_expr.split('-')
                    if len(parts) >= 2:
                        left_side = parts[0]
                        right_side = '-'.join(parts[1:])
                    else:
                        print(f"Warning: Could not parse equation: {eq_expr}")
                        continue
                
                # Clean up the expressions
                left_side = left_side.strip()
                right_side = right_side.strip()
                
                # Convert to sympy expressions
                left_expr = sp.sympify(left_side)
                right_expr = sp.sympify(right_side)
                eq = left_expr - right_expr
                
                sym_equations.append(eq)
                print(f"Parsed equation {eq_name}: {eq}")
            except Exception as e:
                print(f"Error parsing equation {eq_name}: {e}")
                continue
    
    # Check if we have the right number of equations
    if len(sym_equations) != len(variables):
        print(f"Warning: Number of equations ({len(sym_equations)}) doesn't match number of variables ({len(variables)})")
    
    # Identify current and forward variables
    current_vars = [var for var in variables]
    forward_vars = [f"{var}_p" for var in variables]
    
    # Create the A and B matrices
    n = len(variables)
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)
    
    # Compute Jacobians
    for i, eq in enumerate(sym_equations):
        if i >= n:
            break  # Ensure we don't exceed matrix dimensions
            
        # Differentiate with respect to forward variables (A matrix)
        for j, var in enumerate(forward_vars):
            if var in str(eq):
                try:
                    # Compute partial derivative with respect to forward variable
                    deriv = sp.diff(eq, sym_vars[var])
                    # Substitute zero for all variables to get the coefficient
                    subs_dict = {sym_vars[v]: 0 for v in sym_vars}
                    A[i, j] = float(deriv.subs(subs_dict))
                except Exception as e:
                    print(f"Error differentiating eq {i} w.r.t {var}: {e}")
        
        # Differentiate with respect to current variables (B matrix)
        for j, var in enumerate(current_vars):
            if var in str(eq):
                try:
                    # Compute partial derivative with respect to current variable
                    deriv = sp.diff(eq, sym_vars[var])
                    # Substitute zero for all variables to get the coefficient
                    subs_dict = {sym_vars[v]: 0 for v in sym_vars}
                    # Note: We negate the derivative for B matrix in AE[x_{t+1}] = Bx_t form
                    B[i, j] = -float(deriv.subs(subs_dict))
                except Exception as e:
                    print(f"Error differentiating eq {i} w.r.t {var}: {e}")
    
    # Count predetermined variables
    predetermined_count = sum(1 for var in variables if "_lag" in var)
    print(f"Identified {predetermined_count} predetermined variables")
    
    return A, B, variables, predetermined_count

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
    from scipy.linalg import qz, ordqz
    
    # print(f"\nSolving system with {nk} predetermined variables...")
    
    # # Check for regularity
    # if np.linalg.matrix_rank(np.vstack([A, B])) < A.shape[1]:
    #     print("Warning: Matrix pencil A*z - B may be singular!")
    
    # # QZ decomposition
    # S, T, Q, Z = qz(A, B)
    
    # # Reorder with stable generalized eigenvalues first
    # S, T, Q, Z = ordqz(S, T, Q, Z, 'udo')
    
    from scipy.linalg import ordqz
    
    # QZ decomposition with reordering        
    S, T, alpha, beta, Q, Z = ordqz(A, B, sort='ouc')


    if abs(T[nk-1,nk-1]) > abs(S[nk-1,nk-1]) or abs(T[nk,nk]) < abs(S[nk,nk]):
        warnings.warn('Wrong number of stable eigenvalues.')

    # Calculate generalized eigenvalues
    eigenvalues = []
    for i in range(len(S)):
        if abs(S[i, i]) < 1e-10:
            eigenvalues.append(float('inf'))  # Infinity for zero on diagonal of S
        else:
            eigenvalues.append(T[i, i] / S[i, i])
    
    # # Count stable eigenvalues (inside unit circle)
    # stable_count = sum(1 for eig in eigenvalues if abs(eig) < 1.0)
    
    # print(f"Eigenvalues: {[round(eig, 4) if not np.isinf(eig) else 'inf' for eig in eigenvalues]}")
    # print(f"Found {stable_count} stable eigenvalues, expected {nk}")
    
    # if stable_count != nk:
    #     print(f"Warning: Number of stable eigenvalues ({stable_count}) doesn't match predetermined variables ({nk})")
    #     print("Continuing with solution...")
    
    # Extract submatrices
    Z11 = Z[:nk, :nk]
    Z21 = Z[nk:, :nk]
    
    # Check invertibility
    if np.linalg.matrix_rank(Z11) < nk:
        print("Error: Z11 is not invertible - unique stable solution doesn't exist")
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

def solve_model_klein(model_file):
    """
    End-to-end function to parse a model file and solve it using Klein's method.
    
    Args:
        model_file: Path to the model specification file
    
    Returns:
        F: Decision rule such that u(t) = F*k(t)
        P: Law of motion such that k(t+1) = P*k(t)
        variables: List of model variables
    """
    # Get matrices and variable information
    A, B, variables, nk = create_klein_matrices(model_file)
    
    print("\nA matrix:")
    print(A)
    print("\nB matrix:")
    print(B)
    
    # Solve using Klein's method
    print("\nSolving the model...")
    print("nk =", nk)   
    F, P = solve_klein(A, B, nk)
    
    return F, P, variables

import numpy as np
import matplotlib.pyplot as plt
import re
from symbolic_differentiation import create_klein_matrices, solve_klein

def extract_shock_info(model_file):
    """Extract shock information from the model file"""
    with open(model_file, 'r') as f:
        model_text = f.read()
    
    # Parse shock names
    shock_match = re.search(r'shocks\s*=\s*\[(.*?)\];', model_text, re.DOTALL)
    if shock_match:
        shock_str = shock_match.group(1)
        shock_names = [s.strip().strip('"\'') for s in shock_str.split(',')]
    else:
        shock_names = []
    
    # Parse equations to identify which shocks appear in which equations
    eq_pattern = re.compile(r'\{"eq\d+"\s*:\s*"([^"]+)"\}')
    eq_matches = eq_pattern.findall(model_text)
    
    shock_equations = {}
    for shock in shock_names:
        shock_equations[shock] = []
        for i, eq in enumerate(eq_matches):
            if shock in eq:
                shock_equations[shock].append(i)
    
    return shock_names, shock_equations

def build_shock_matrix(shock_names, shock_equations, n_vars, nk):
    """
    Build the shock matrix R based on which equations contain which shocks
    
    Args:
        shock_names: List of shock names
        shock_equations: Dictionary mapping shock names to equation indices
        n_vars: Total number of variables
        nk: Number of predetermined variables
        
    Returns:
        R: Shock impact matrix for the state variables (k(t))
    """
    n_shocks = len(shock_names)
    R = np.zeros((nk, n_shocks))
    
    # For each shock, identify its direct impact on state variables
    for i, shock in enumerate(shock_names):
        for eq_idx in shock_equations[shock]:
            # If the shock appears in an equation for a state variable, set its impact
            if eq_idx < nk:
                R[eq_idx, i] = 1.0
    
    return R

def build_state_space(F, P, R):
    """
    Build state space matrices for the DSGE model
    
    Args:
        F: Decision rule (u(t) = F*k(t))
        P: Law of motion (k(t) = P*k(t-1))
        R: Shock impact matrix on state variables
    
    Returns:
        Phi: State transition matrix for full state vector s(t) = [k(t); u(t)]
        R_ss: Shock impact matrix for full state vector
    """
    nk = P.shape[0]  # Number of state variables
    nu = F.shape[0]  # Number of control variables
    n_total = nk + nu  # Total variables
    n_shocks = R.shape[1]  # Number of shocks
    
    # Build state space transition matrix
    # [k(t)]   = [P     0] [k(t-1)] + [R ] [eps(t)]
    # [u(t)]     [F*P   0] [u(t-1)]   [FR]
    Phi = np.zeros((n_total, n_total))
    Phi[:nk, :nk] = P
    Phi[nk:, :nk] = F @ P
    
    # Build shock impact matrix
    R_ss = np.zeros((n_total, n_shocks))
    R_ss[:nk, :] = R
    R_ss[nk:, :] = F @ R
    
    return Phi, R_ss

def compute_irf(Phi, R_ss, shock_idx, periods=40):
    """Compute impulse response function for a given shock"""
    n_variables = Phi.shape[0]
    irf = np.zeros((periods, n_variables))
    
    # Initial shock (of size 1)
    shock = np.zeros(R_ss.shape[1])
    shock[shock_idx] = 1.0
    
    # Impact effect (first period)
    irf[0, :] = R_ss @ shock
    
    # Propagation through time
    for t in range(1, periods):
        irf[t, :] = Phi @ irf[t-1, :]
    
    return irf

def plot_irfs(irf, variables, shock_name, figsize=(12, 8)):
    """Plot impulse response functions"""
    n_vars = len(variables)
    
    # Determine grid layout
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each variable's response
    for i, var in enumerate(variables):
        if i < len(axes):
            axes[i].plot(irf[:, i])
            axes[i].set_title(var)
            axes[i].grid(True)
            axes[i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Response to {shock_name} shock', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def analyze_all_shocks(model_file, periods=40):
    """Analyze IRFs for all shocks in the model"""
    # Get model matrices and variable information
    A, B, variables, nk = create_klein_matrices(model_file)
    
    # Extract shock information
    shock_names, shock_equations = extract_shock_info(model_file)
    
    # Build shock impact matrix
    R = build_shock_matrix(shock_names, shock_equations, len(variables), nk)
    
    # Solve the model
    F, P = solve_klein(A, B, nk)
    
    # Build state space representation
    Phi, R_ss = build_state_space(F, P, R)
    
    # Compute and plot IRFs for each shock
    results = {}
    for i, shock in enumerate(shock_names):
        irf = compute_irf(Phi, R_ss, i, periods)
        fig = plot_irfs(irf, variables, shock)
        
        results[shock] = {
            'irf': irf,
            'fig': fig
        }
    
    # Return matrices for further analysis
    state_space = {
        'F': F,  # Decision rule: u(t) = F*k(t)
        'P': P,  # Law of motion: k(t) = P*k(t-1)
        'Phi': Phi,  # State space transition matrix
        'R': R,  # Shock impact on state variables
        'R_ss': R_ss,  # Shock impact on full state vector
        'variables': variables,
        'nk': nk,
        'shock_names': shock_names
    }
    
    return results, state_space

if __name__ == "__main__":
    model_file = "qpm_simpl1.txt"
    
    # Analyze all shocks
    results, state_space = analyze_all_shocks(model_file)
    
    # Display information about the state space representation
    print("State space representation:")
    print(f"Number of state variables: {state_space['nk']}")
    print(f"Number of control variables: {len(state_space['variables']) - state_space['nk']}")
    print(f"State variables: {state_space['variables'][:state_space['nk']]}")
    print(f"Control variables: {state_space['variables'][state_space['nk']:]}")
    
    # Get the full transition matrix
    Phi = state_space['Phi']
    
    # Calculate the eigenvalues of the transition matrix
    eigenvalues = np.linalg.eigvals(Phi)
    print("\nEigenvalues of state transition matrix:")
    for i, eig in enumerate(eigenvalues):
        print(f"λ{i+1} = {eig:.4f} (magnitude: {abs(eig):.4f})")
    
    # Show all plots
    plt.show()

# # Example usage
# if __name__ == "__main__":
#     model_file = "/Volumes/TOSHIBA EXT/main_work/Work/Projects/iris_replacement/Simplest/qpm_simpl1.txt"
#     F, P, variables = solve_model_klein(model_file)
    
#     if F is not None and P is not None:
#         # Extract non-predetermined variables (decision rule)
#         non_predetermined = [v for v in variables if "_lag" not in v]
#         predetermined = [v for v in variables if "_lag" in v]
        
#         print("\nDecision rule (F):")
#         for i, var in enumerate(non_predetermined):
#             print(f"{var} =", end=" ")
#             for j, state_var in enumerate(predetermined):
#                 if abs(F[i, j]) > 1e-10:  # Only print non-zero coefficients
#                     print(f"{F[i, j]:.4f}*{state_var}", end=" ")
#             print()
        
#         print("\nLaw of motion (P):")
#         for i, var in enumerate(predetermined):
#             print(f"{var} evolves according to:")
#             for j, state_var in enumerate(predetermined):
#                 if abs(P[i, j]) > 1e-10:  # Only print non-zero coefficients
#                     print(f"  {P[i, j]:.4f}*{state_var}")
#     else:
#         print("Failed to compute solution.")