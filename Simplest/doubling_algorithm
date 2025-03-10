#%%
import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm

def solve_dsge_klein_representation(A_plus, A_zero, A_minus, 
                                    state_indices, control_indices,
                                    initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    """
    Solves a DSGE model using the doubling algorithm and returns the Klein representation matrices.
    
    Parameters:
    -----------
    A_plus : ndarray
        Jacobian with respect to future variables (t+1)
    A_zero : ndarray
        Jacobian with respect to current variables (t)
    A_minus : ndarray
        Jacobian with respect to past variables (t-1)
    state_indices : list
        Indices of predetermined state variables in the model
    control_indices : list
        Indices of control variables in the model
    initial_guess : ndarray, optional
        Initial guess for X
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Whether to print progress
    
    Returns:
    --------
    F : ndarray
        Policy function matrix (control variables as function of state variables)
    P : ndarray
        State transition matrix (evolution of state variables)
    converged : bool
        Whether the solution converged
    X : ndarray
        The original solution to the quadratic matrix equation
    """
    # First solve the quadratic matrix equation using the doubling algorithm
    X, iterations, residual = solve_quadratic_matrix_equation_doubling(
        A_plus, A_zero, A_minus, initial_guess, tol, max_iter, verbose
    )
    
    if residual > tol:
        if verbose:
            print(f"Solution did not converge. Residual: {residual}")
        return None, None, False, X
    
    # Construct the full transition matrix
    n_vars = A_zero.shape[0]
    full_transition = np.zeros((n_vars, n_vars))
    
    # The structure depends on how the model is arranged, but generally:
    # 1. For predetermined variables (states), we use the law of motion from the model
    # 2. For control variables, we use the policy function X
    
    # Extract state-to-state transition (P matrix)
    n_states = len(state_indices)
    P = np.zeros((n_states, n_states))
    
    # Assuming X gives the full set of relationships
    # We need to extract the parts relevant to state transitions
    for i, row_idx in enumerate(state_indices):
        for j, col_idx in enumerate(state_indices):
            # This assumes X contains the full transition structure
            # The exact indexing depends on the structure of X
            P[i, j] = X[row_idx, col_idx]
    
    # Extract control-to-state relationships (F matrix)
    n_controls = len(control_indices)
    F = np.zeros((n_controls, n_states))
    
    for i, row_idx in enumerate(control_indices):
        for j, col_idx in enumerate(state_indices):
            F[i, j] = X[row_idx, col_idx]
    
    return F, P, True, X

def solve_quadratic_matrix_equation_doubling(A, B, C, initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    """
    Solve the quadratic matrix equation A*X^2 + B*X + C = 0 using the structure-preserving doubling algorithm.
    
    Parameters:
    -----------
    A : ndarray
        Coefficient matrix for X^2 term
    B : ndarray
        Coefficient matrix for X term
    C : ndarray
        Constant term
    initial_guess : ndarray, optional
        Initial guess for X
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Whether to print progress
    
    Returns:
    --------
    X : ndarray
        Solution to the quadratic matrix equation
    iter_count : int
        Number of iterations performed
    reached_tol : float
        Final error tolerance reached
    """
    guess_provided = True
    
    if initial_guess is None or initial_guess.size == 0:
        guess_provided = False
        initial_guess = np.zeros_like(A)
    
    # Initialize matrices
    E = C.copy()
    F = A.copy()
    
    # Compute B̄ = B + A*initial_guess
    B_bar = B.copy()
    B_bar += A @ initial_guess
    
    # LU factorization of B̄
    try:
        B_lu, B_piv = lu_factor(B_bar)
    except:
        return A, 0, 1.0
    
    # Compute initial values
    E = lu_solve((B_lu, B_piv), C)
    F = lu_solve((B_lu, B_piv), A)
    
    X = -E - initial_guess
    Y = -F
    
    # Preallocate temporary matrices
    X_new = np.zeros_like(X)
    Y_new = np.zeros_like(Y)
    E_new = np.zeros_like(E)
    F_new = np.zeros_like(F)
    
    temp1 = np.zeros_like(Y)
    temp2 = np.zeros_like(Y)
    temp3 = np.zeros_like(Y)
    
    n = X.shape[0]
    II = np.eye(n)
    
    Xtol = 1.0
    Ytol = 1.0
    
    solved = False
    iter_count = max_iter
    
    # Main iteration loop
    for i in range(1, max_iter + 1):
        # Compute EI = I - Y*X
        np.matmul(Y, X, out=temp1)
        temp1 = II - temp1
        
        # Factorize EI
        try:
            EI_lu, EI_piv = lu_factor(temp1)
        except:
            return A, i, 1.0
        
        # Compute E = E * EI^(-1) * E
        temp3 = lu_solve((EI_lu, EI_piv), E)
        np.matmul(E, temp3, out=E_new)
        
        # Compute FI = I - X*Y
        np.matmul(X, Y, out=temp2)
        temp2 = II - temp2
        
        # Factorize FI
        try:
            FI_lu, FI_piv = lu_factor(temp2)
        except:
            return A, i, 1.0
        
        # Compute F = F * FI^(-1) * F
        temp3 = lu_solve((FI_lu, FI_piv), F)
        np.matmul(F, temp3, out=F_new)
        
        # Compute X_new = X + F * FI^(-1) * X * E
        np.matmul(X, E, out=temp3)
        temp3 = lu_solve((FI_lu, FI_piv), temp3)
        np.matmul(F, temp3, out=X_new)
        
        if i > 5 or guess_provided:
            Xtol = norm(X_new)
        
        X_new += X
        
        # Compute Y_new = Y + E * EI^(-1) * Y * F
        np.matmul(Y, F, out=temp3)
        temp3 = lu_solve((EI_lu, EI_piv), temp3)
        np.matmul(E, temp3, out=Y_new)
        
        if i > 5 or guess_provided:
            Ytol = norm(Y_new)
        
        Y_new += Y
        
        # Check for convergence
        if Xtol < tol:
            solved = True
            iter_count = i
            break
        
        # Update values for next iteration
        X[:] = X_new
        Y[:] = Y_new
        E[:] = E_new
        F[:] = F_new
    
    # Compute the final X
    X_new += initial_guess
    X = X_new
    
    # Compute the residual
    AXX = A @ X @ X
    AXXnorm = norm(AXX)
    AXX += B @ X
    AXX += C
    
    reached_tol = norm(AXX) / AXXnorm
    
    return X, iter_count, reached_tol

import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm

def example_test_klein_representation():
    """
    Creates a simple DSGE model and solves it using the doubling algorithm,
    returning the Klein representation matrices.
    """
    # Simple 3-equation New Keynesian model structure
    # Variables: [y_t, π_t, i_t, y_{t-1}, π_{t-1}, i_{t-1}]
    # where y is output gap, π is inflation, i is interest rate
    
    # Number of variables
    n_vars = 6
    
    # Parameters
    beta = 0.99    # Discount factor
    kappa = 0.3    # Slope of Phillips curve
    sigma = 1      # Intertemporal elasticity of substitution
    phi_pi = 1.5   # Taylor rule inflation coefficient
    phi_y = 0.5    # Taylor rule output coefficient
    rho = 0.8      # Interest rate smoothing
    
    # Indices of state and control variables
    # States: lagged variables (positions 3, 4, 5)
    state_indices = [3, 4, 5]
    # Controls: current variables (positions 0, 1, 2)
    control_indices = [0, 1, 2]
    
    # Initialize Jacobian matrices
    A_plus = np.zeros((n_vars, n_vars))  # Future
    A_zero = np.zeros((n_vars, n_vars))  # Current
    A_minus = np.zeros((n_vars, n_vars))  # Past
    
    # IS equation: y_t = E_t[y_{t+1}] - (1/sigma)*(i_t - E_t[π_{t+1}])
    A_plus[0, 0] = 1.0                # E_t[y_{t+1}]
    A_plus[0, 1] = 1.0/sigma          # E_t[π_{t+1}]
    A_zero[0, 0] = -1.0               # y_t
    A_zero[0, 2] = -1.0/sigma         # i_t
    
    # Phillips curve: π_t = beta*E_t[π_{t+1}] + kappa*y_t
    A_plus[1, 1] = beta               # E_t[π_{t+1}]
    A_zero[1, 0] = kappa              # y_t
    A_zero[1, 1] = -1.0               # π_t
    
    # Taylor rule: i_t = rho*i_{t-1} + (1-rho)*(phi_pi*π_t + phi_y*y_t)
    A_zero[2, 0] = (1-rho)*phi_y      # y_t
    A_zero[2, 1] = (1-rho)*phi_pi     # π_t
    A_zero[2, 2] = -1.0               # i_t
    A_minus[2, 5] = rho               # i_{t-1}
    
    # Transition equations for lagged variables
    A_zero[3, 0] = 1.0                # y_t = y_t
    A_zero[3, 3] = 0.0                # Placeholder
    
    A_zero[4, 1] = 1.0                # π_t = π_t
    A_zero[4, 4] = 0.0                # Placeholder
    
    A_zero[5, 2] = 1.0                # i_t = i_t
    A_zero[5, 5] = 0.0                # Placeholder
    
    # Solve the model
    F, P, converged, X = solve_dsge_klein_representation(
        A_plus, A_zero, A_minus, 
        state_indices, control_indices,
        verbose=True
    )
    
    if converged:
        print("Solution converged successfully")
        print("\nF matrix (Policy function):")
        print(F)
        print("\nP matrix (State transition):")
        print(P)
        
        # Compute eigenvalues of P to check stability
        eigenvalues = np.linalg.eigvals(P)
        print("\nEigenvalues of P:")
        print(eigenvalues)
        print("Max eigenvalue magnitude:", np.max(np.abs(eigenvalues)))
        
        # Test the impulse response to an interest rate shock
        n_periods = 20
        impulse_responses = simulate_impulse_response(F, P, shock_size=0.25, n_periods=n_periods)
        
        print("\nImpulse responses to a 25bp interest rate shock:")
        print("Output gap:", impulse_responses[0, :])
        print("Inflation:", impulse_responses[1, :])
        print("Interest rate:", impulse_responses[2, :])
    else:
        print("Solution did not converge")
    
    return F, P, X

def simulate_impulse_response(F, P, shock_size=0.25, n_periods=20):
    """
    Simulates impulse response to an interest rate shock
    
    Parameters:
    -----------
    F : ndarray
        Policy function matrix
    P : ndarray
        State transition matrix
    shock_size : float
        Size of the shock
    n_periods : int
        Number of periods to simulate
        
    Returns:
    --------
    responses : ndarray
        Matrix of impulse responses (rows=variables, cols=periods)
    """
    n_controls = F.shape[0]
    n_states = P.shape[0]
    
    # Initialize state vector with the shock
    states = np.zeros((n_states, n_periods))
    controls = np.zeros((n_controls, n_periods))
    
    # Apply shock to the interest rate (assuming it's the third state)
    states[2, 0] = shock_size
    
    # Compute responses
    for t in range(n_periods):
        # Current controls based on current states
        controls[:, t] = F @ states[:, t]
        
        # State transition for next period
        if t < n_periods - 1:
            states[:, t+1] = P @ states[:, t]
    
    return np.vstack([controls, states])


example_test_klein_representation()