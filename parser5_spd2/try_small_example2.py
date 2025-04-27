
import sympy
import numpy as np
import matplotlib.pyplot as plt

def compute_ABCD():
    """
    1) L_GDP_GAP
    - b1 * L_GDP_GAP(-1)
    + (b1 - 1) * L_GDP_GAP(+1)
    - RES_L_GDP_GAP
    + b4*(RS(+1) - aux_DLA_CPI_lead1(+1)) = 0

    2) DLA_CPI
    - a1 * DLA_CPI(-1)
    + (a1 - 1) * DLA_CPI(+1)
    - a2*L_GDP_GAP
    - RES_DLA_CPI = 0

    3) -RES_RS + RS - g1*RS(-1) + (g1 - 1)*(DLA_CPI(+1)+ g3*L_GDP_GAP + g2*aux_DLA_CPI_lead2(+1))=0

    4) RES_L_GDP_GAP - rho_L_GDP_GAP*RES_L_GDP_GAP(-1) - SHK_L_GDP_GAP=0

    5) RES_DLA_CPI - rho_DLA_CPI*RES_DLA_CPI(-1) - SHK_DLA_CPI=0

    6) RES_RS - rho_rs*RES_RS(-1) - SHK_RS - rho_rs2*aux_RES_RS_lag1(-1)=0

    7) -DLA_CPI(+1) + aux_DLA_CPI_lead1=0

    8) -aux_DLA_CPI_lead1(+1) + aux_DLA_CPI_lead2=0

    9) -RES_RS(-1) + aux_RES_RS_lag1=0

    State vector order (y_t indices):
    0: RES_DLA_CPI
    1: RES_L_GDP_GAP
    2: RES_RS
    3: aux_RES_RS_lag1
    4: DLA_CPI
    5: L_GDP_GAP
    6: RS
    7: aux_DLA_CPI_lead1
    8: aux_DLA_CPI_lead2

    We assume the shocks are:
    e_0 = SHK_L_GDP_GAP
    e_1 = SHK_DLA_CPI
    e_2 = SHK_RS

    Returns
    -------
    A, B, C : 9x9 ndarrays
    D       : 9x3 ndarray
        So that the system in linear form is:
        0 = A * y_{t+1} + B * y_t + C * y_{t-1} + D * e_t
    """
    # Define the parameters
    b1 = 0.7   
    b4 = 0.7         
    a1 = 0.5         
    a2 = 0.1         
    g1 = 0.7         
    g2 = 0.3         
    g3 = 0.25        
    rho_L_GDP_GAP  =0.75
    rho_DLA_CPI   =0.75
    rho_rs      =0.8
    rho_rs2      =0.01

    # Dimensions
    n_eq = 9
    n_shock = 3
    # Initialize A, B, C, D as zeros
    A = np.zeros((n_eq, n_eq))
    B = np.zeros((n_eq, n_eq))
    C = np.zeros((n_eq, n_eq))
    D = np.zeros((n_eq, n_shock))

    # For convenience, define an index mapping:
    # 0: RES_DLA_CPI
    # 1: RES_L_GDP_GAP
    # 2: RES_RS
    # 3: aux_RES_RS_lag1
    # 4: DLA_CPI
    # 5: L_GDP_GAP
    # 6: RS
    # 7: aux_DLA_CPI_lead1
    # 8: aux_DLA_CPI_lead2

    # Shocks:
    # shock0 -> SHK_L_GDP_GAP
    # shock1 -> SHK_DLA_CPI
    # shock2 -> SHK_RS

    # =======================
    # Equation 1 (row 0):
    #  L_GDP_GAP
    #  - b1 * L_GDP_GAP(-1)
    #  + (b1 - 1)* L_GDP_GAP(+1)
    #  - RES_L_GDP_GAP
    #  + b4*(RS(+1) - aux_DLA_CPI_lead1(+1)) = 0
    # => A-coeff (y_{t+1}):
    A[0, 5] = (b1 - 1)       # L_GDP_GAP(+1) : index 5
    A[0, 6] = b4             # RS(+1): index 6
    A[0, 7] = -b4            # aux_DLA_CPI_lead1(+1): index 7
    # => B-coeff (y_t):
    B[0, 5] =  1.0           # L_GDP_GAP
    B[0, 1] = -1.0           # -RES_L_GDP_GAP: index 1
    # => C-coeff (y_{t-1}):
    C[0, 5] = -b1
    # => D-coeff (shock): none => 0

    # =======================
    # Equation 2 (row 1):
    #  DLA_CPI
    #  - a1 * DLA_CPI(-1)
    #  + (a1 - 1)* DLA_CPI(+1)
    #  - a2 * L_GDP_GAP
    #  - RES_DLA_CPI = 0
    # => A:
    A[1, 4] = (a1 - 1)
    # => B:
    B[1, 4] = 1.0      # DLA_CPI
    B[1, 5] = -a2      # -a2 * L_GDP_GAP
    B[1, 0] = -1.0     # -RES_DLA_CPI
    # => C:
    C[1, 4] = -a1
    # => D: none

    # =======================
    # Equation 3 (row 2):
    # -RES_RS + RS - RS(-1)*g1
    #   + (g1 - 1)*( DLA_CPI(+1) + g3*L_GDP_GAP + g2*aux_DLA_CPI_lead2(+1)) = 0
    #
    # => A:
    # (g1 -1)*DLA_CPI(+1) => A[2,4] += (g1-1)
    # (g1 -1)*aux_DLA_CPI_lead2(+1)*g2 => A[2,8] += (g1-1)*g2
    A[2, 4] = (g1 - 1)
    A[2, 8] = (g1 - 1)*g2
    # => B:
    # -RES_RS => B[2,2] = -1
    # +RS => B[2,6] = +1
    # + (g1 -1)*g3 * L_GDP_GAP => B[2,5] += (g1 -1)*g3
    B[2, 2] = -1.0
    B[2, 6] =  1.0
    B[2, 5] += (g1 -1)*g3
    # => C:
    # - g1 * RS(-1) => C[2,6] = -g1
    C[2, 6] = -g1
    # => D: none

    # =======================
    # Equation 4 (row 3):
    # RES_L_GDP_GAP - rho_L_GDP_GAP*RES_L_GDP_GAP(-1) - SHK_L_GDP_GAP=0
    # => B:
    B[3,1] = 1.0
    # => C:
    C[3,1] = -rho_L_GDP_GAP
    # => D:
    #  - SHK_L_GDP_GAP => means + (-1)*shock0
    D[3,0] = -1.0

    # =======================
    # Equation 5 (row 4):
    # RES_DLA_CPI - rho_DLA_CPI*RES_DLA_CPI(-1) - SHK_DLA_CPI=0
    # => B:
    B[4,0] = 1.0
    # => C:
    C[4,0] = -rho_DLA_CPI
    # => D:
    D[4,1] = -1.0    # -SHK_DLA_CPI => shock1

    # =======================
    # Equation 6 (row 5):
    # RES_RS - rho_rs * RES_RS(-1)
    #  - SHK_RS
    #  - rho_rs2 * aux_RES_RS_lag1(-1) = 0
    #
    # => B:
    B[5,2] = 1.0   # RES_RS
    # => C:
    C[5,2] = -rho_rs
    # The term - rho_rs2*aux_RES_RS_lag1(-1):
    #  => y_{t-1}(3), coefficient = -rho_rs2
    C[5,3] = -rho_rs2
    # => D:
    D[5,2] = -1.0   # - SHK_RS => shock2

    # =======================
    # Equation 7 (row 6):
    # -DLA_CPI(+1) + aux_DLA_CPI_lead1 = 0
    # => A:
    A[6,4] = -1.0    # - DLA_CPI(+1)
    # => B:
    B[6,7] =  1.0    # + aux_DLA_CPI_lead1

    # =======================
    # Equation 8 (row 7):
    # -aux_DLA_CPI_lead1(+1) + aux_DLA_CPI_lead2 = 0
    # => A:
    A[7,7] = -1.0
    # => B:
    B[7,8] =  1.0

    # =======================
    # Equation 9 (row 8):
    # -RES_RS(-1) + aux_RES_RS_lag1 = 0
    # => B:
    B[8,3] =  1.0   # + aux_RES_RS_lag1
    # => C:
    C[8,2] = -1.0   # -RES_RS(-1)

    return A, B, C, D

def sda_sf1(A, B, C, tol=1e-12, max_iter=1000):
    """
    A simple Structure-Preserving Doubling Algorithm in "SF1" form to solve
    the quadratic matrix equation:
    0 = A P^2 + B P + C

    for the stable (spectral radius < 1) solution P, assuming B is invertible.

    Parameters
    ----------
    A, B, C : ndarrays (square, same dimension n)
    tol : float, convergence tolerance
    max_iter : int, iteration cap

    Returns
    -------
    P_sol : ndarray or None
        The stable solution if found, else None if breakdown or no convergence.
    """
    n = A.shape[0]
    # Invert B once (assuming it is invertible).
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        print("Error: B not invertible. SF1 cannot be directly applied.")
        return None

    # Initial values:
    # X0 = -B^-1*C, Y0 = -B^-1*A
    Xk = -B_inv @ C
    Yk = -B_inv @ A
    Ek = Xk.copy()
    Fk = Yk.copy()

    for _iter in range(max_iter):
        # I - YkXk
        I_minus_YX = np.eye(n) - Yk @ Xk
        # I - XkYk
        I_minus_XY = np.eye(n) - Xk @ Yk

        # Check invertibility
        try:
            iMYX = np.linalg.inv(I_minus_YX)
            iMXY = np.linalg.inv(I_minus_XY)
        except np.linalg.LinAlgError:
            # breakdown
            return None

        # Update
        Ek_next = Ek @ iMYX @ Ek
        Fk_next = Fk @ iMXY @ Fk
        Xk_next = Xk + Fk @ iMXY @ Xk @ Ek
        Yk_next = Yk + Ek @ iMYX @ Yk @ Fk

        # Check change in Xk
        diff_norm = np.linalg.norm(Xk_next - Xk, ord='fro')
        if diff_norm < tol:
            Xk = Xk_next
            break

        Xk, Yk, Ek, Fk = Xk_next, Yk_next, Ek_next, Fk_next

    return Xk

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

def compute_ABCD_and_solve():
    """
    High-level procedure:
    1) Obtain A, B, C, D from a toy DSGE model.
    2) Solve 0 = A P^2 + B P + C using the SPD SF1 approach.
    3) Return P, plus also the (A,B,C,D).
    """
    A, B, C, D = compute_ABCD()
#    P_sol = sda_sf1(A, B, C)
    P_sol, iter_count, residual_ratio = solve_quadratic_matrix_equation(A, B, C)
    print(f"SF1 iterations: {iter_count}, residual ratio: {residual_ratio:.2e}")
    Q = compute_Q(A, B, D, P_sol)

    return A, B, C, D, P_sol, Q

def irf(P, Q, horizon=10):
    """
    Compute impulse responses for y_t = P y_{t-1} + Q e_t,
    with an initial impulse e_0 = identity (if multiple shocks, just do a single, or do them in a loop).
    For a single shock dimension we set e_0=1, others=0.

    Return array of shape (horizon, n) with each row the response at time t.
    """
    n = P.shape[0]
    # We have e dimension typically 1 => Q is (n x 1)
    # Let e0 = 1.0 at t=0, then 0 afterwards.
    y_resp = []
    y_prev = np.zeros((n,1))
    shock = np.zeros((3,1))  #Number of shocks is 3
    shock[2]=1  # impulse at t=0
    for t in range(horizon):
        y_now = P @ y_prev + Q @ shock
        y_resp.append(y_now.flatten())
        # next step
        y_prev = y_now
        shock = np.zeros_like(shock)  # 0 after initial
    return np.array(y_resp)


def main():
    # 1) Compute A, B, C, D and solve for P
    A, B, C, D, P_sol, Q_sol = compute_ABCD_and_solve()
    print("A:\n", A)
    print("B:\n", B)
    print("C:\n", C)
    print("D:\n", D)
    print("Solution P:\n", P_sol)
    print("Solution Q:\n", Q_sol)

    # 3) Generate IRFs
    horizon = 40
    irf_vals = irf(P_sol, Q_sol, horizon=horizon)
    # 4) Plot IRF

    #   y_t = [x_t, z_t]
    #   We'll just plot them on the same figure.

    # 0: RES_DLA_CPI
    # 1: RES_L_GDP_GAP
    # 2: RES_RS
    # 3: aux_RES_RS_lag1
    # 4: DLA_CPI
    # 5: L_GDP_GAP
    # 6: RS
    # 7: aux_DLA_CPI_lead1
    # 8: aux_DLA_CPI_lead2

    plt.figure(figsize=(7,5))
    plt.plot(irf_vals[:,0], label='IRF for RES_DLA_CPI')
    plt.plot(irf_vals[:,1], label='IRF for RES_L_GDP_GAP')
    plt.plot(irf_vals[:,2], label='IRF for RES_RS')
    plt.plot(irf_vals[:,4], label='IRF for DLA_CPI')
    plt.plot(irf_vals[:,5], label='IRF for L_GDP_GAP')
    plt.plot(irf_vals[:,6], label='IRF for RS')
    plt.axhline(0, color='k', linewidth=0.8)
    plt.title("Impulse Responses for x and z")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("IRF values:\n", irf_vals)
if __name__ == "__main__":
    main()    