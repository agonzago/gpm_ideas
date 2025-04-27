import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def compute_ABCD_new():
    """
    Computes the A, B, C, D matrices for the corrected model specification:
    Final Equation System (10):
      1 (Row 0): L_GDP_GAP - L_GDP_GAP_m1*b1 + L_GDP_GAP_p1*(b1 - 1) - RES_L_GDP_GAP + b4*(RR_GAP_p1) = 0
      2 (Row 1): DLA_CPI - DLA_CPI_m1*a1 + DLA_CPI_p1*(a1 - 1) - L_GDP_GAP*a2 - RES_DLA_CPI = 0
      3 (Row 2): -RES_RS + RS - RS_m1*g1 + (g1 - 1)*(DLA_CPI_p1 + L_GDP_GAP*g3 + aux_DLA_CPI_lead2_p1*g2) = 0
      4 (Row 3): -RR_GAP + RS - aux_DLA_CPI_lead1=0
      5 (Row 4): RES_L_GDP_GAP - RES_L_GDP_GAP_m1*rho_L_GDP_GAP - SHK_L_GDP_GAP = 0
      6 (Row 5): RES_DLA_CPI - RES_DLA_CPI_m1*rho_DLA_CPI - SHK_DLA_CPI = 0
      7 (Row 6): RES_RS - RES_RS_m1*rho_rs - SHK_RS - aux_RES_RS_lag1_m1*rho_rs2 = 0
      8 (Row 7): -DLA_CPI_p1 + aux_DLA_CPI_lead1 = 0
      9 (Row 8): -aux_DLA_CPI_lead1_p1 + aux_DLA_CPI_lead2 = 0
     10 (Row 9): -RES_RS_m1 + aux_RES_RS_lag1 = 0

    State vector order (y_t indices):
    0: RES_DLA_CPI, 1: RES_L_GDP_GAP, 2: RES_RS, 3: aux_RES_RS_lag1,
    4: DLA_CPI, 5: RR_GAP, 6: L_GDP_GAP, 7: RS,
    8: aux_DLA_CPI_lead1, 9: aux_DLA_CPI_lead2

    Shocks e_t:
    e_0 = SHK_L_GDP_GAP, e_1 = SHK_DLA_CPI, e_2 = SHK_RS

    Returns
    -------
    A, B, C : 10x10 ndarrays
    D       : 10x3 ndarray
        So that the system in linear form is:
        0 = A * E[y_{t+1}] + B * y_t + C * y_{t-1} + D * e_t
    """
    # Define the parameters (using same values as before for illustration)
    b1 = 0.7
    b4 = 0.7
    a1 = 0.5
    a2 = 0.1
    g1 = 0.6
    g2 = 0.3
    g3 = 0.25
    rho_L_GDP_GAP = 0.75
    rho_DLA_CPI = 0.75
    rho_rs = 0.8
    rho_rs2 = 0.01 # Coefficient for RES_RS_lag in RES_RS process

    # Dimensions
    n_eq = 10
    n_var = 10
    n_shock = 3
    # Initialize A, B, C, D as zeros
    A = np.zeros((n_eq, n_var))
    B = np.zeros((n_eq, n_var))
    C = np.zeros((n_eq, n_var))
    D = np.zeros((n_eq, n_shock))

    # Variable indices mapping for clarity:
    # 0: RES_DLA_CPI, 1: RES_L_GDP_GAP, 2: RES_RS, 3: aux_RES_RS_lag1,
    # 4: DLA_CPI, 5: RR_GAP, 6: L_GDP_GAP, 7: RS,
    # 8: aux_DLA_CPI_lead1, 9: aux_DLA_CPI_lead2

    # Shocks indices:
    # 0: SHK_L_GDP_GAP, 1: SHK_DLA_CPI, 2: SHK_RS

    # =======================
    # Equation 1 (Row 0): L_GDP_GAP eq
    # L_GDP_GAP - L_GDP_GAP_m1*b1 + L_GDP_GAP_p1*(b1 - 1) - RES_L_GDP_GAP + b4*(RR_GAP_p1) = 0
    B[0, 6] = 1.0           # L_GDP_GAP (t)
    C[0, 6] = -b1           # L_GDP_GAP_m1 (t-1)
    A[0, 6] = (b1 - 1)      # L_GDP_GAP_p1 (t+1)
    B[0, 1] = -1.0          # RES_L_GDP_GAP (t)
    A[0, 5] = b4            # RR_GAP_p1 (t+1)

    # =======================
    # Equation 2 (Row 1): DLA_CPI eq
    # DLA_CPI - DLA_CPI_m1*a1 + DLA_CPI_p1*(a1 - 1) - L_GDP_GAP*a2 - RES_DLA_CPI = 0
    B[1, 4] = 1.0           # DLA_CPI (t)
    C[1, 4] = -a1           # DLA_CPI_m1 (t-1)
    A[1, 4] = (a1 - 1)      # DLA_CPI_p1 (t+1)
    B[1, 6] = -a2           # L_GDP_GAP (t)
    B[1, 0] = -1.0          # RES_DLA_CPI (t)

    # =======================
    # Equation 3 (Row 2): RS eq
    # -RES_RS + RS - RS_m1*g1 + (g1 - 1)*(DLA_CPI_p1 + L_GDP_GAP*g3 + aux_DLA_CPI_lead2_p1*g2) = 0
    B[2, 2] = -1.0          # RES_RS (t)
    B[2, 7] = 1.0           # RS (t)
    C[2, 7] = -g1           # RS_m1 (t-1)
    A[2, 4] = (g1 - 1)      # DLA_CPI_p1 (t+1)
    B[2, 6] = (g1 - 1) * g3 # L_GDP_GAP (t)
    A[2, 9] = (g1 - 1) * g2 # aux_DLA_CPI_lead2_p1 (t+1)

    # =======================
    # Equation 4 (Row 3): RR_GAP definition
    # -RR_GAP + RS - aux_DLA_CPI_lead1 = 0
    B[3, 5] = -1.0          # RR_GAP (t)
    B[3, 7] = 1.0           # RS (t)
    B[3, 8] = -1.0          # aux_DLA_CPI_lead1 (t)

    # =======================
    # Equation 5 (Row 4): RES_L_GDP_GAP process
    # RES_L_GDP_GAP - RES_L_GDP_GAP_m1*rho_L_GDP_GAP - SHK_L_GDP_GAP = 0
    B[4, 1] = 1.0           # RES_L_GDP_GAP (t)
    C[4, 1] = -rho_L_GDP_GAP# RES_L_GDP_GAP_m1 (t-1)
    D[4, 0] = -1.0          # SHK_L_GDP_GAP (shock 0)

    # =======================
    # Equation 6 (Row 5): RES_DLA_CPI process
    # RES_DLA_CPI - RES_DLA_CPI_m1*rho_DLA_CPI - SHK_DLA_CPI = 0
    B[5, 0] = 1.0           # RES_DLA_CPI (t)
    C[5, 0] = -rho_DLA_CPI  # RES_DLA_CPI_m1 (t-1)
    D[5, 1] = -1.0          # SHK_DLA_CPI (shock 1)

    # =======================
    # Equation 7 (Row 6): RES_RS process
    # RES_RS - RES_RS_m1*rho_rs - SHK_RS - aux_RES_RS_lag1_m1*rho_rs2 = 0
    B[6, 2] = 1.0           # RES_RS (t)
    C[6, 2] = -rho_rs       # RES_RS_m1 (t-1)
    D[6, 2] = -1.0          # SHK_RS (shock 2)
    C[6, 3] = -rho_rs2      # aux_RES_RS_lag1_m1 (t-1)

    # =======================
    # Equation 8 (Row 7): aux_DLA_CPI_lead1 definition
    # -DLA_CPI_p1 + aux_DLA_CPI_lead1 = 0
    A[7, 4] = -1.0          # DLA_CPI_p1 (t+1)
    B[7, 8] = 1.0           # aux_DLA_CPI_lead1 (t)

    # =======================
    # Equation 9 (Row 8): aux_DLA_CPI_lead2 definition
    # -aux_DLA_CPI_lead1_p1 + aux_DLA_CPI_lead2 = 0
    A[8, 8] = -1.0          # aux_DLA_CPI_lead1_p1 (t+1)
    B[8, 9] = 1.0           # aux_DLA_CPI_lead2 (t)

    # =======================
    # Equation 10 (Row 9): aux_RES_RS_lag1 definition
    # -RES_RS_m1 + aux_RES_RS_lag1 = 0
    C[9, 2] = -1.0          # RES_RS_m1 (t-1)
    B[9, 3] = 1.0           # aux_RES_RS_lag1 (t)

    return A, B, C, D


def compute_ABCD_and_solve_new():
    """
    High-level procedure for the new model:
    1) Obtain A, B, C, D from the new DSGE model specification.
    2) Solve 0 = A P^2 + B P + C using the SPD SF1 approach.
    3) Compute Q.
    4) Return A, B, C, D, P, Q.
    """
    A, B, C, D = compute_ABCD_new() # Use the new function
    P_sol, iter_count, residual_ratio = solve_quadratic_matrix_equation(A, B, C, tol=1e-12) # Added tolerance
    if P_sol is None or residual_ratio > 1e-6: # Add a check for convergence
        print("Solver failed or did not converge well.")
        return A, B, C, D, None, None
    print(f"Solver iterations: {iter_count}, final residual ratio: {residual_ratio:.2e}")
    Q = compute_Q(A, B, D, P_sol)
    if Q is None:
        print("Failed to compute Q.")
        return A, B, C, D, P_sol, None


    # Ensure P is stable (eigenvalues < 1 in magnitude)
    eigenvalues = np.linalg.eigvals(P_sol)
    max_eig = np.max(np.abs(eigenvalues))
    print(f"Maximum eigenvalue magnitude of P: {max_eig:.4f}")
    if max_eig >= 1.0 - 1e-9: # Allow for slight numerical inaccuracy
       print("Warning: Solution P might be unstable or borderline stable.")


    return A, B, C, D, P_sol, Q

def irf_new(P, Q, shock_index, horizon=40):
    """
    Compute impulse responses for y_t = P y_{t-1} + Q e_t,
    for a specific shock index.

    Parameters
    ----------
    P : ndarray (n x n)
        Transition matrix.
    Q : ndarray (n x n_shock)
        Shock impact matrix.
    shock_index : int
        Index of the shock to simulate (0, 1, or 2).
    horizon : int, optional
        Number of periods for the IRF. The default is 40.

    Returns
    -------
    ndarray
        Array of shape (horizon, n) with responses over time.
    """
    n = P.shape[0]
    n_shock = Q.shape[1]
    if shock_index < 0 or shock_index >= n_shock:
        raise ValueError(f"shock_index must be between 0 and {n_shock-1}")

    y_resp = np.zeros((horizon, n))
    # Initial impulse: only one shock is non-zero at t=0
    e0 = np.zeros((n_shock, 1))
    e0[shock_index] = 1.0

    # y_0 = P * y_{-1} + Q * e_0. Assume y_{-1} = 0.
    y_current = Q @ e0

    y_resp[0, :] = y_current.flatten()

    # Subsequent periods: e_t = 0 for t > 0
    et = np.zeros((n_shock, 1))
    for t in range(1, horizon):
        y_current = P @ y_current # + Q @ et (which is zero)
        y_resp[t, :] = y_current.flatten()

    return y_resp


import numpy as np
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve

def solve_quadratic_matrix_equation(A, B, C, initial_guess=None, tol=1e-14, max_iter=100, verbose=False):
    """
    A Python version of a structure-preserving doubling method analogous to the Julia code snippet.
    Solves the quadratic matrix equation:
        0 = A X^2 + B X + C
    for the "stable" solution X using a doubling-type iteration. This assumes
    B + A @ initial_guess (if provided) can be factorized (i.e., is invertible) at the start.

    Parameters
    ----------
    A, B, C : 2D numpy arrays of shape (n, n), dtype float
    initial_guess : 2D numpy array of shape (n, n), optional
        If None, starts at zeros_like(A).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of doubling iterations.
    verbose : bool
        If True, print iteration details.

    Returns
    -------
    X : 2D numpy array
        The computed stable solution matrix.
    iter_count : int
        Number of iterations performed.
    residual_ratio : float
        A measure of how well the final X solves the equation, based on
        ||A X^2 + B X + C|| / ||A X^2||.
    """

    n = A.shape[0]
    if initial_guess is None or initial_guess.size == 0:
        guess_provided = False
        initial_guess = np.zeros_like(A)
    else:
        guess_provided = True

    # Copy matrices
    E = C.copy()
    F = A.copy()
    Bbar = B.copy()

    # Emulate "Bbar = Bbar + A @ initial_guess"
    Bbar += A @ initial_guess

    try:
        lu_Bbar = lu_factor(Bbar)
    except ValueError:
        # Factorization failed
        return A.copy(), 0, 1.0

    # Solve E = Bbar \ C, F = Bbar \ A
    E = lu_solve(lu_Bbar, E)
    F = lu_solve(lu_Bbar, F)

    # Initialize X, Y
    X = -E - initial_guess
    Y = -F

    # Allocate space for new iterates
    X_new = np.zeros_like(X)
    Y_new = np.zeros_like(Y)
    E_new = np.zeros_like(E)
    F_new = np.zeros_like(F)

    I = np.eye(n, dtype=A.dtype)
    solved = False
    iter_count = max_iter

    for i in range(1, max_iter + 1):
        # EI = I - Y*X
        temp1 = Y @ X
        EI = I - temp1

        # Factor EI
        try:
            lu_EI = lu_factor(EI)
        except ValueError:
            # If factorization fails, return something
            return A.copy(), i, 1.0

        # E_new = E * (EI^-1) * E
        #   We do E_new = E @ (lu_solve(lu_EI, E))
        temp1 = lu_solve(lu_EI, E)
        E_new = E @ temp1

        # FI = I - X*Y
        temp2 = X @ Y
        FI = I - temp2

        # Factor FI
        try:
            lu_FI = lu_factor(FI)
        except ValueError:
            return A.copy(), i, 1.0

        # F_new = F * (FI^-1) * F
        temp2 = lu_solve(lu_FI, F)
        F_new = F @ temp2

        # X_new = X + F * (FI^-1) * (X * E)
        temp3 = X @ E
        temp3 = lu_solve(lu_FI, temp3)
        X_new = F @ temp3
        X_new += X

        # Possibly check norm for X_new
        if i > 5 or guess_provided:
            Xtol = norm(X_new, ord='fro')
        else:
            Xtol = 1.0

        # Y_new = Y + E * (EI^-1) * (Y * F)
        temp1 = Y @ F
        temp1 = lu_solve(lu_EI, temp1)
        Y_new = E @ temp1
        Y_new += Y

        if verbose:
            print(f"Iteration {i}: Xtol={Xtol:e}")

        # Check convergence
        if Xtol < tol:
            solved = True
            iter_count = i
            break

        # Update the iterates
        X[:] = X_new
        Y[:] = Y_new
        E[:] = E_new
        F[:] = F_new

    # Incorporate initial_guess: X_new += initial_guess
    X_new += initial_guess
    X = X_new

    # Final check: compute residual = A X^2 + B X + C
    AX2 = A @ (X @ X)
    AX2_norm = norm(AX2, ord='fro')
    residual = AX2 + B @ X + C
    if AX2_norm == 0.0:
        residual_ratio = norm(residual, ord='fro')
    else:
        residual_ratio = norm(residual, ord='fro') / AX2_norm

    return X, iter_count, residual_ratio

def compute_Q(A, B, D, P):
    """
    Once P satisfies A P^2 + B P + C=0, we can solve for Q in

    (A P + B)*Q + D = 0   =>   (A P + B)*Q = -D   =>   Q = -(A P + B)^{-1} D.

    This Q is such that  y_t = P y_{t-1} + Q e_t .
    For dimension n=2, D is typically 2x1 if there's 1 shock.
    """
    APB = A @ P + B
    try:
        invAPB = np.linalg.inv(APB)
    except np.linalg.LinAlgError:
        print("Cannot invert (A P + B). Possibly singular.")
        return None
    Q = - invAPB @ D
    return Q



def main_new():
    # 1) Compute A, B, C, D and solve for P and Q for the new model
    A, B, C, D, P_sol, Q_sol = compute_ABCD_and_solve_new()

    if P_sol is None or Q_sol is None:
        print("Exiting due to solver failure.")
        return

    print("\n--- New Model Matrices ---")
    print("A:\n", np.round(A, 3))
    print("B:\n", np.round(B, 3))
    print("C:\n", np.round(C, 3))
    print("D:\n", np.round(D, 3))
    print("Solution P:\n", np.round(P_sol, 3))
    print("Solution Q:\n", np.round(Q_sol, 3))


    # 3) Generate IRFs for a specific shock (e.g., monetary policy shock, SHK_RS, index 2)
    shock_index_to_plot = 2 # 0: GDP, 1: CPI, 2: RS
    shock_names = ["SHK_L_GDP_GAP", "SHK_DLA_CPI", "SHK_RS"]
    horizon = 40
    irf_vals = irf_new(P_sol, Q_sol, shock_index=shock_index_to_plot, horizon=horizon)

    # 4) Plot IRF for selected variables
    var_indices = {
        "RES_DLA_CPI": 0, "RES_L_GDP_GAP": 1, "RES_RS": 2,
        "DLA_CPI": 4, "RR_GAP": 5, "L_GDP_GAP": 6, "RS": 7,
        "aux_DLA_CPI_lead1": 8, "aux_DLA_CPI_lead2": 9
    }
    vars_to_plot = ["L_GDP_GAP", "DLA_CPI", "RS", "RR_GAP", "RES_RS"]

    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Impulse Responses to a Unit {shock_names[shock_index_to_plot]} Shock")

    num_plots = len(vars_to_plot)
    rows = (num_plots + 1) // 2
    cols = 2

    for i, var_name in enumerate(vars_to_plot):
        idx = var_indices[var_name]
        plt.subplot(rows, cols, i + 1)
        plt.plot(irf_vals[:, idx], label=f'IRF for {var_name}')
        plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
        plt.title(f"{var_name}")
        plt.xlabel("Quarters")
        plt.ylabel("Response")
        plt.grid(True, alpha=0.5)
        #plt.legend() # Can get crowded, title is enough

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


# Assuming the helper functions are defined above as in the original post:
if __name__ == "__main__":
    # Make sure helper functions are defined or imported
    # Define sda_sf1, solve_quadratic_matrix_equation, compute_Q here or import them
    main_new()