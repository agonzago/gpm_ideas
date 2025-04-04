import numpy as np
import importlib.util

def load_state_space(filename):
    """Loads the A, B, C, and H matrices from a Python file."""
    spec = importlib.util.spec_from_file_location("model", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Assuming the matrices are named A, B, C, H
    return module.A, module.B, module.C, module.H

def compare_matrices(A1, B1, C1, H1, A2, B2, C2, H2):
    """Compares the A, B, C, and H matrices and highlights differences."""
    print("Comparing A matrices:")
    compare_matrix(A1, A2)
    print("\nComparing B matrices:")
    compare_matrix(B1, B2)
    print("\nComparing C matrices:")
    compare_matrix(C1, C2)
    print("\nComparing H matrices:")
    compare_matrix(H1, H2)

def compare_matrix(M1, M2):
    """Compares two matrices and highlights differences."""
    if M1.shape != M2.shape:
        print(f"Matrices have different shapes: {M1.shape} vs {M2.shape}")
        return

    rows, cols = M1.shape
    for i in range(rows):
        for j in range(cols):
            if not np.isclose(M1[i, j], M2[i, j]):
                print(f"Difference at ({i}, {j}): {M1[i, j]} vs {M2[i, j]}")

# Load the state-space matrices from both files
A1, B1, C1, H1 = load_state_space("parser_gpm.py")
A2, B2, C2, H2 = load_state_space("full_code_includes_parser.py")

# Compare the matrices
compare_matrices(A1, B1, C1, H1, A2, B2, C2, H2)