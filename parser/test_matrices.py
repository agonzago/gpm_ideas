import numpy as np
import importlib.util

def load_state_space(filename, matrix_names):
    """Loads the specified matrices from a Python file."""
    spec = importlib.util.spec_from_file_location("model", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    matrices = []
    for name in matrix_names:
        matrices.append(getattr(module, name))  # Get the matrix by name

    return matrices

def compare_matrices(matrices1, matrix_names1, matrices2, matrix_names2):
    """Compares the matrices and highlights differences."""
    if len(matrices1) != len(matrices2):
        print("Different number of matrices to compare.")
        return

    for i in range(len(matrices1)):
        print(f"\nComparing {matrix_names1[i]} from file 1 with {matrix_names2[i]} from file 2:")
        compare_matrix(matrices1[i], matrices2[i])

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

# Specify the matrix names for each file
matrix_names_file1 = ["A", "B", "H", "Q"]
matrix_names_file2 = ["A", "B", "H", "Q"]

# Load the state-space matrices from both files
matrices1 = None  # Initialize to None
matrices2 = None  # Initialize to None
try:
    matrices1 = load_state_space("parser_gpm.py", matrix_names_file1)
    matrices2 = load_state_space("full_code_includes_parser.py", matrix_names_file2)
except AttributeError as e:
    print(f"Error: {e}.  Check the matrix names in each file and update matrix_names_file1 and matrix_names_file2 accordingly.")
    exit()

# Compare the matrices
if matrices1 is not None and matrices2 is not None: # Only compare if loading was successful
    compare_matrices(matrices1, matrix_names_file1, matrices2, matrix_names_file2)