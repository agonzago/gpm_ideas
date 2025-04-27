    # """"
    # 0 = A E[y_{t+1}] + B y_t + C y_{t-1} + D e_t

    # where y_t = [x_t, z_t]', e_t = shk_z_t (a 1D shock).

    # Equations:
    # 1) x_t - alpha*x_{t+1} - beta*z_t = 0
    # 2) z_t - rhoz*z_{t-1} - shk_z_t = 0

    # Returns
    # -------
    # A, B, C : 2x2 ndarrays
    # D       : 2x1 ndarray
    # """


# The system is 2x2; we define each row accordingly.
# Row1 (for x_t eq):
#  0 = alpha*x_{t+1} + 0*z_{t+1} + (-1)*x_t + beta*z_t + 0*y_{t-1} + 0*shock
# Row2 (for z_t eq):
#  0 = 0*x_{t+1} + 0*z_{t+1} + (0)*x_t + (-1)*z_t + [0, rhoz]*y_{t-1} + [0,1]*shock


import sympy
import numpy as np
import matplotlib.pyplot as plt

def compute_linear_model_matrices():
    """
    Example routine to define a simple linear DSGE model in symbolic form:
    0 = A E[y_{t+1}] + B y_t + C y_{t-1} + D e_t
    and extract (A, B, C, D) numerically.
    """

    alpha = 0.1
    beta = 0.1
    rhoz = 0.9

    A = np.array([
        [ alpha,   0.0 ],
        [ 0.0,     0.0 ]
    ])  # Coeff on E[y_{t+1}]
    B = np.array([
        [ -1.0,   beta ],
        [  0.0,   -1.0 ]
    ])  # Coeff on y_t
    C = np.array([
        [  0.0,    0.0  ],
        [  0.0,   rhoz  ]
    ])  # Coeff on y_{t-1}
    # We have 1 shock (shk_z), so D is 2x1
    # eq1 has no shock, eq2 has +1*shock
    D = np.array([
        [ 0.0 ],
        [ 1.0 ]
    ])
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
    A, B, C, D = compute_linear_model_matrices()
    P_sol = sda_sf1(A, B, C)

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
    shock = np.array([[1.0]])  # impulse at t=0
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
    horizon = 10
    irf_vals = irf(P_sol, Q_sol, horizon=horizon)
    # 4) Plot IRF

    #   y_t = [x_t, z_t]
    #   We'll just plot them on the same figure.

    plt.figure(figsize=(7,5))
    plt.plot(irf_vals[:,0], label='IRF for x')
    plt.plot(irf_vals[:,1], label='IRF for z')
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