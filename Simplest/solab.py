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