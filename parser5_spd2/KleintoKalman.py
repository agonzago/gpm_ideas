import numpy as np

def kalman_filter_matrices_fortran_style(F_obs, P, num_k, num_exo, num_cntrl_obs):
    """
    Constructs Kalman filter T and R matrices based on the logic
    in Fortran/dsoltokalman.f90:KALMANFILTERMATRICES.

    Assumes a Kalman state vector:
    [observed_controls; non_exo_states; exo_states]

    Parameters:
    -----------
    F_obs : np.ndarray
        Observed control rows from Klein's f matrix.
        Shape: (num_cntrl_obs, num_k)
    P : np.ndarray
        Klein's state transition matrix p.
        Shape: (num_k, num_k)
    num_k : int
        Total number of states in Klein's solution (s_t).
    num_exo : int
        Number of exogenous shock processes (assumed to be the last num_exo states).
    num_cntrl_obs : int
        Number of observed control variables.

    Returns:
    --------
    T : np.ndarray
        Kalman filter state transition matrix.
        Shape: (num_est, num_est)
    R : np.ndarray
        Kalman filter shock impact matrix.
        Shape: (num_est, num_exo)
    """

    # Calculate intermediate sizes based on Fortran variables
    num_est = num_cntrl_obs + num_k  # Total Kalman states
    n_k = num_cntrl_obs            # Size of observed controls block
    k_ex = num_k - num_exo         # Size of non-exogenous states block
    n_ex = n_k + k_ex              # End index for non-exogenous states block

    if F_obs.shape != (num_cntrl_obs, num_k):
        raise ValueError(f"F_obs shape mismatch: expected ({num_cntrl_obs},{num_k}), got {F_obs.shape}")
    if P.shape != (num_k, num_k):
        raise ValueError(f"P shape mismatch: expected ({num_k},{num_k}), got {P.shape}")
    if not (0 <= num_exo <= num_k):
         raise ValueError("num_exo must be between 0 and num_k")
    if num_cntrl_obs < 0:
         raise ValueError("num_cntrl_obs cannot be negative")


    # --- Partition input matrices ---
    # Partition F_obs: [F1, F2]
    F1 = F_obs[:, :k_ex]  # Shape: (n_k, k_ex)
    F2 = F_obs[:, k_ex:]  # Shape: (n_k, num_exo)

    # Partition P: [[P11, P12], [P21, P22]]
    P11 = P[:k_ex, :k_ex]      # Shape: (k_ex, k_ex)
    P12 = P[:k_ex, k_ex:]      # Shape: (k_ex, num_exo)
    P21 = P[k_ex:, :k_ex]      # Shape: (num_exo, k_ex) - Note: Fortran assumes this is zero implicitly later
    P22 = P[k_ex:, k_ex:]      # Shape: (num_exo, num_exo)

    # --- Initialize output matrices ---
    T = np.zeros((num_est, num_est))
    R = np.zeros((num_est, num_exo))

    # --- Construct T matrix block by block ---
    # State vector blocks: [obs_ctrl (n_k), s_non_exo (k_ex), s_exo (num_exo)]

    # Row block 1: obs_ctrl_t+1 = f(s_t) = f1*s_non_exo_t + f2*s_exo_t
    # Fortran T(1:N_K, ...) corresponds to T[:n_k, :]
    # T(1:N_K, 1+N_K:N_EX) = F(:,1:K_EX) -> T12 = F1
    if k_ex > 0:
        T[0:n_k, n_k:n_ex] = F1
    # T(1:N_K, 1+N_EX:NUM_EST) = F(:,K_EX+1:NUM_K) @ P(K_EX+1:NUM_K,K_EX+1:NUM_K) -> T13 = F2 @ P22
    if num_exo > 0:
        T[0:n_k, n_ex:num_est] = F2 @ P22
    # T(1:N_K, 1:N_K) = 0.0 -> T11 = 0 (already zero)

    # Row block 2: s_non_exo_t+1 = p11*s_non_exo_t + p12*s_exo_t
    # Fortran T(1+N_K:N_EX, ...) corresponds to T[n_k:n_ex, :]
    # T(1+N_K:N_EX, N_K+1:N_EX) = P(1:K_EX, 1:K_EX) -> T22 = P11 (Fortran index corrected)
    if k_ex > 0:
        T[n_k:n_ex, n_k:n_ex] = P11
    # T(1+N_K:N_EX, 1+N_EX:NUM_EST) = P(1:K_EX, K_EX+1:NUM_K) @ P(K_EX+1:NUM_K, K_EX+1:NUM_K) -> T23 = P12 @ P22
    if k_ex > 0 and num_exo > 0:
        T[n_k:n_ex, n_ex:num_est] = P12 @ P22
    # T(1+N_K:N_EX, 1:N_K) = 0.0 -> T21 = 0 (already zero)

    # Row block 3: s_exo_t+1 = p21*s_non_exo_t + p22*s_exo_t
    # Fortran T(1+N_EX:NUM_EST, ...) corresponds to T[n_ex:num_est, :]
    # T(1+N_EX:NUM_EST, 1+N_EX:NUM_EST) = P(K_EX+1:NUM_K, K_EX+1:NUM_K) -> T33 = P22
    if num_exo > 0:
        T[n_ex:num_est, n_ex:num_est] = P22
    # T(1+N_EX:NUM_EST, N_K+1:N_EX) = 0.0 -> T32 = 0 (already zero, assumes P21=0)
    # T(1+N_EX:NUM_EST, 1:N_K) = 0.0 -> T31 = 0 (already zero)

    # --- Construct R matrix block by block ---
    # Maps shocks_t (num_exo) to changes in state [obs_ctrl; s_non_exo; s_exo]

    # Row block 1: Impact on obs_ctrl
    # R(1:N_K, 1:NUM_EXO) = F(:, K_EX+1:NUM_K) -> R1 = F2
    if num_exo > 0:
        R[0:n_k, 0:num_exo] = F2

    # Row block 2: Impact on s_non_exo
    # R(N_K+1:N_EX, 1:NUM_EXO) = P(1:K_EX, K_EX+1:NUM_K) -> R2 = P12
    if k_ex > 0 and num_exo > 0:
        R[n_k:n_ex, 0:num_exo] = P12

    # Row block 3: Impact on s_exo
    # R(N_EX+1:NUM_EST, 1:NUM_EXO) = Identity(NUM_EXO)
    if num_exo > 0:
        np.fill_diagonal(R[n_ex:num_est, 0:num_exo], 1.0) # Fortran lines 29-32

    return T, R
