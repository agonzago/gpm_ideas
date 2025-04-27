
#!/usr/bin/env python3
"""
dynare_parser_main.py
A simplified Dynare-style parser with example usage.
"""

import os
import re
import sympy
import numpy as np
import pickle
import collections
import importlib.util
import datetime 
import matplotlib.pyplot as plt
import importlib.util
import sys # Import sys for sys.exit and sys.modules

# ===========================================
# Helper Functions (Solvers, IRF) - Moved to top for clarity
# ===========================================

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
        # Use overwrite_a=True and check_finite=False for potential speedup if sure about inputs
        lu_Bbar, piv_Bbar = lu_factor(Bbar, overwrite_a=False, check_finite=False)
    except ValueError:
        print("Error: LU factorization failed for Bbar (B + A @ initial_guess). Matrix might be singular.")
        return A.copy(), 0, np.inf # Indicate failure clearly
    except np.linalg.LinAlgError:
        print("Error: Bbar (B + A @ initial_guess) is singular.")
        return A.copy(), 0, np.inf

    # Solve E = Bbar \ C, F = Bbar \ A
    # Use overwrite_b=True for potential minor speedup
    try:
        E = lu_solve((lu_Bbar, piv_Bbar), E, overwrite_b=False, check_finite=False)
        F = lu_solve((lu_Bbar, piv_Bbar), F, overwrite_b=False, check_finite=False)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Error solving initial system with Bbar factorization: {e}")
        return A.copy(), 0, np.inf

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
    iter_count = 0 # Start count at 0

    for i in range(1, max_iter + 1):
        iter_count = i # Record last attempted iteration

        # EI = I - Y*X
        temp1 = Y @ X
        EI = I - temp1

        # Factor EI
        try:
            lu_EI, piv_EI = lu_factor(EI, overwrite_a=False, check_finite=False)
            # Check for singularity post-factorization if possible (depends on library version)
            # if np.any(np.diag(lu_EI) == 0): raise np.linalg.LinAlgError("Singular matrix EI encountered.")
        except (ValueError, np.linalg.LinAlgError):
            if verbose: print(f"Iteration {i}: Factorization failed for EI (I - YX). Breakdown.")
            # Return current X, iteration count, and high residual
            current_residual = norm(A @ X @ X + B @ X + C, 'fro')
            ax2_norm = norm(A @ X @ X, 'fro')
            res_ratio = current_residual / ax2_norm if ax2_norm > 1e-16 else current_residual
            return X, i, res_ratio if res_ratio > 0 else np.inf

        # E_new = E * (EI^-1) * E => E @ lu_solve(EI, E)
        try:
            temp1 = lu_solve((lu_EI, piv_EI), E, overwrite_b=False, check_finite=False)
        except (ValueError, np.linalg.LinAlgError) as e:
            if verbose: print(f"Iteration {i}: lu_solve failed for EI \\ E: {e}")
            current_residual = norm(A @ X @ X + B @ X + C, 'fro')
            ax2_norm = norm(A @ X @ X, 'fro')
            res_ratio = current_residual / ax2_norm if ax2_norm > 1e-16 else current_residual
            return X, i, res_ratio if res_ratio > 0 else np.inf
        E_new = E @ temp1

        # FI = I - X*Y
        temp2 = X @ Y
        FI = I - temp2

        # Factor FI
        try:
            lu_FI, piv_FI = lu_factor(FI, overwrite_a=False, check_finite=False)
            # if np.any(np.diag(lu_FI) == 0): raise np.linalg.LinAlgError("Singular matrix FI encountered.")
        except (ValueError, np.linalg.LinAlgError):
            if verbose: print(f"Iteration {i}: Factorization failed for FI (I - XY). Breakdown.")
            current_residual = norm(A @ X @ X + B @ X + C, 'fro')
            ax2_norm = norm(A @ X @ X, 'fro')
            res_ratio = current_residual / ax2_norm if ax2_norm > 1e-16 else current_residual
            return X, i, res_ratio if res_ratio > 0 else np.inf

        # F_new = F * (FI^-1) * F => F @ lu_solve(FI, F)
        try:
            temp2 = lu_solve((lu_FI, piv_FI), F, overwrite_b=False, check_finite=False)
        except (ValueError, np.linalg.LinAlgError) as e:
            if verbose: print(f"Iteration {i}: lu_solve failed for FI \\ F: {e}")
            current_residual = norm(A @ X @ X + B @ X + C, 'fro')
            ax2_norm = norm(A @ X @ X, 'fro')
            res_ratio = current_residual / ax2_norm if ax2_norm > 1e-16 else current_residual
            return X, i, res_ratio if res_ratio > 0 else np.inf
        F_new = F @ temp2

        # X_new = X + F * (FI^-1) * (X * E) => X + F @ lu_solve(FI, X @ E)
        temp3 = X @ E
        try:
            temp3 = lu_solve((lu_FI, piv_FI), temp3, overwrite_b=False, check_finite=False)
        except (ValueError, np.linalg.LinAlgError) as e:
            if verbose: print(f"Iteration {i}: lu_solve failed for FI \\ (X@E): {e}")
            current_residual = norm(A @ X @ X + B @ X + C, 'fro')
            ax2_norm = norm(A @ X @ X, 'fro')
            res_ratio = current_residual / ax2_norm if ax2_norm > 1e-16 else current_residual
            return X, i, res_ratio if res_ratio > 0 else np.inf
        X_new = F @ temp3
        X_new += X

        # Calculate change in X for convergence check
        # Use Frobenius norm of the *difference* X_new - X
        X_diff_norm = norm(X_new - X, 'fro')

        # Y_new = Y + E * (EI^-1) * (Y * F) => Y + E @ lu_solve(EI, Y @ F)
        temp1 = Y @ F
        try:
            temp1 = lu_solve((lu_EI, piv_EI), temp1, overwrite_b=False, check_finite=False)
        except (ValueError, np.linalg.LinAlgError) as e:
            if verbose: print(f"Iteration {i}: lu_solve failed for EI \\ (Y@F): {e}")
            current_residual = norm(A @ X_new @ X_new + B @ X_new + C, 'fro') # Use X_new for residual
            ax2_norm = norm(A @ X_new @ X_new, 'fro')
            res_ratio = current_residual / ax2_norm if ax2_norm > 1e-16 else current_residual
            return X_new, i, res_ratio if res_ratio > 0 else np.inf # Return X_new
        Y_new = E @ temp1
        Y_new += Y

        if verbose:
            # Use X_diff_norm which measures the step size
            print(f"Iteration {i}: Change in X (Frobenius norm) = {X_diff_norm:.2e}")

        # Update the iterates for the next loop or final result
        # Use np.copyto for efficiency if arrays are large, or direct assignment
        # X[:] = X_new # In-place update if needed, but direct assign is fine
        X = X_new
        Y = Y_new
        E = E_new
        F = F_new

        # Check convergence based on the change in X
        if X_diff_norm < tol:
            solved = True
            # iter_count is already set to i
            break

    if not solved and verbose:
        print(f"Warning: Solver did not converge within {max_iter} iterations.")

    # Final residual calculation using the final X
    AX2 = A @ (X @ X)
    AX2_norm = norm(AX2, ord='fro')
    residual = AX2 + B @ X + C
    # Use a small epsilon to avoid division by zero if AX2_norm is tiny
    residual_ratio = norm(residual, ord='fro') / (AX2_norm + 1e-16)

    return X, iter_count, residual_ratio

def compute_Q(A, B, D, P):
    """
    Once P satisfies A P^2 + B P + C=0, we can solve for Q in
    (A P + B)*Q + D = 0   =>   (A P + B)*Q = -D   =>   Q = -(A P + B)^{-1} D.
    """
    APB = A @ P + B
    try:
        # Use scipy.linalg.solve for better numerical stability than inv
        # Q = solve(APB, -D) # Equivalent to inv(APB) @ (-D)
        invAPB = np.linalg.inv(APB)
        Q = - invAPB @ D
    except np.linalg.LinAlgError:
        print("Error: Cannot compute Q. Matrix (A @ P + B) might be singular.")
        return None
    return Q

def irf_new(P, Q, shock_index, horizon=40):
    """
    Compute impulse responses for y_t = P y_{t-1} + Q e_t for a specific shock.
    """
    n_state = P.shape[0]
    n_shock = Q.shape[1]
    if not 0 <= shock_index < n_shock:
        raise ValueError(f"shock_index ({shock_index}) out of bounds [0, {n_shock-1}]")

    y_resp = np.zeros((horizon, n_state))
    # Initial impulse e_0 (column vector)
    e0 = np.zeros((n_shock, 1))
    e0[shock_index, 0] = 1.0

    # y_0 = P @ y_{-1} + Q @ e_0 (assume y_{-1} = 0)
    y_current = Q @ e0 # Result is (n_state x 1)

    if y_current.shape[0] != n_state:
        print(f"Warning: Unexpected shape for y_current after Q@e0: {y_current.shape}. Expected ({n_state}, 1)")
        # Attempt to reshape if it's just a flat array mismatch
        if y_current.size == n_state: y_current = y_current.reshape(n_state, 1)
        else: raise ValueError("Shape mismatch in IRF calculation")

    y_resp[0, :] = y_current.flatten() # Store as row in the result

    # Iterate for t = 1 to horizon-1
    for t in range(1, horizon):
        y_current = P @ y_current # y_t = P @ y_{t-1} (since e_t = 0 for t>0)
        y_resp[t, :] = y_current.flatten()

    return y_resp

# --- Function for Time Shifting Expressions (Keep only one definition) ---
def time_shift_expression(expr, shift, parser_symbols, var_names_set):
    """
    Shifts the time index of variables within a Sympy expression. Handles base,
    lead (_p), lag (_m), aux_lead, and aux_lag vars based on parser_symbols.
    """
    if shift == 0:
        return expr

    subs_dict = {}
    atoms = expr.free_symbols

    for atom in atoms:
        atom_name = atom.name
        base_name = None
        current_k = 0
        is_var_type = False # Includes base vars, aux_lead, aux_lag

        # Prioritize matching aux first, then base lead/lag, then base
        # Use IGNORECASE for aux matching as sometimes they might be Aux_...
        match_aux_lead = re.match(r"(aux_\w+_lead)(\d+)$", atom_name, re.IGNORECASE)
        match_aux_lag  = re.match(r"(aux_\w+_lag)(\d+)$", atom_name, re.IGNORECASE)
        match_lead = re.match(r"(\w+)_p(\d+)$", atom_name)
        match_lag  = re.match(r"(\w+)_m(\d+)$", atom_name)

        if match_aux_lead:
            base_name = match_aux_lead.group(1) # e.g., "aux_DLA_CPI_lead"
            current_k = int(match_aux_lead.group(2))
            is_var_type = True
        elif match_aux_lag:
            base_name = match_aux_lag.group(1) # e.g., "aux_RES_RS_lag"
            current_k = -int(match_aux_lag.group(2))
            is_var_type = True
        elif match_lead:
            base_name_cand = match_lead.group(1)
            # Avoid interpreting aux_..._p as base_p
            if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                base_name = base_name_cand
                current_k = int(match_lead.group(2))
                is_var_type = True
            # else: it's an aux var lead like aux_VAR_lead1_p1, or not a var
        elif match_lag:
            base_name_cand = match_lag.group(1)
            if not base_name_cand.lower().startswith("aux_") and base_name_cand in var_names_set:
                base_name = base_name_cand
                current_k = -int(match_lag.group(2))
                is_var_type = True
            # else: it's an aux var lag like aux_VAR_lag1_m1, or not a var
        elif atom_name in var_names_set: # Base variable name
            base_name = atom_name
            current_k = 0
            is_var_type = True
        # Handle potential aux var leads/lags like aux_VAR_lead1_p1 here?
        # The current logic relies on these being created later if needed.

        if is_var_type:
            new_k = current_k + shift
            # Determine the clean base name (strip aux_, _lead#, _lag#)
            clean_base_match = re.match(r"aux_(\w+)_(?:lead|lag)\d*", base_name, re.IGNORECASE)
            if clean_base_match:
                clean_base = clean_base_match.group(1)
            else:
                clean_base = base_name # Assumes base_name was the clean name if not aux

            if new_k == 0:
                # Target is the base variable name
                if clean_base in var_names_set:
                    new_sym_name = clean_base
                else:
                    # This might happen if base_name was ill-defined, fallback or error
                    print(f"Warning: Could not map '{atom_name}' to base var for t=0. Using '{clean_base}' as target.")
                    new_sym_name = clean_base # Hope clean_base exists as a symbol
            elif new_k > 0:
                # Target is a lead variable Var_pX or aux_Var_leadX_pY
                # For simplicity now, assuming target is base_pX form
                # A more robust version would need to know if 'clean_base' itself is an aux base
                is_aux_base = base_name.lower().startswith("aux_")
                prefix = "aux_" if is_aux_base else ""
                # The format should be base_pX or aux_base_pX
                new_sym_name = f"{prefix}{clean_base}_p{new_k}"
            else: # new_k < 0
                # Target is a lag variable Var_mX or aux_Var_lagX_mY
                is_aux_base = base_name.lower().startswith("aux_")
                prefix = "aux_" if is_aux_base else ""
                new_sym_name = f"{prefix}{clean_base}_m{abs(new_k)}"

            # Ensure the target symbol exists in the parser's dictionary
            if new_sym_name not in parser_symbols:
                # Create the symbol if it's missing (e.g., y_p2 needed from y_p1 shift +1)
                # print(f"Debug: Creating missing symbol '{new_sym_name}' during time shift.") # Optional debug
                parser_symbols[new_sym_name] = sympy.Symbol(new_sym_name)

            # Add to substitution dictionary
            subs_dict[atom] = parser_symbols[new_sym_name]

        # else: atom is a parameter or shock, leave unchanged

    try:
        # Use xreplace which is generally better for symbol-for-symbol replacement
        shifted_expr = expr.xreplace(subs_dict)
    except Exception as e:
        print(f"Warning: xreplace failed during time_shift_expression for expr: {expr}. Trying subs. Error: {e}")
        try:
            shifted_expr = expr.subs(subs_dict) # Fallback to subs
        except Exception as e2:
            print(f"Error: Fallback subs also failed in time_shift_expression. Error: {e2}")
            shifted_expr = expr # Return original on error
    return shifted_expr

# ===========================================
# DynareParser Class
# ===========================================
class DynareParser:

    def __init__(self, mod_file_path):
        self.mod_file_path = mod_file_path
        self.param_names = []
        self.var_names = []
        self.var_names_set = set()
        self.shock_names = []
        self.equations_str = []
        self.symbols = {} # Dictionary to hold ALL sympy symbols used/created
        self.sympy_equations_original = []
        self.var_timing_info = collections.defaultdict(lambda: {'max_lead': 0, 'min_lag': 0, 'appears_current': False})
        self.static_subs = {} # Stores Symbol: Expression for static variables
        self.equations_after_static_elim = [] # Equations excluding those used to define static vars
        self.equations_after_static_sub = [] # Equations after substituting static var expressions
        self.aux_lead_vars = {} # Stores aux_lead_name -> Symbol
        self.aux_lag_vars = {}  # Stores aux_lag_name -> Symbol
        self.aux_var_definitions = [] # List of sympy.Eq defining the aux vars
        self.equations_after_aux_sub = [] # Original dynamic eqs after subbing > +/-1 leads/lags
        self.final_dynamic_var_names = [] # Includes base dynamic + all aux vars
        self.state_vars_ordered = [] # Final ordered list of state vector symbols
        self.state_var_map = {} # Map from state symbol to its index
        self.final_equations_for_jacobian = [] # Combined list of dynamic + aux definition equations
        self.last_param_values = None # Store params used for last numerical calculation

        self._parse_mod_file()
        self.var_names_set = set(self.var_names) # Set for faster lookup
        self._create_initial_sympy_symbols()
        self._parse_equations_to_sympy()
        print(f"Parser initialized for {mod_file_path}")
        print(f" Vars: {self.var_names}")
        print(f" Params: {self.param_names}")
        print(f" Shocks: {self.shock_names}")

    def _parse_mod_file(self):
        """Parses var, varexo, parameters, and model blocks."""
        if not os.path.isfile(self.mod_file_path):
            raise FileNotFoundError(f"Mod file not found: {self.mod_file_path}")

        with open(self.mod_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove comments first
        content = re.sub(r'//.*', '', content) # Line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL) # Block comments

        # Define regex patterns (case-insensitive, multiline/dotall)
        vpat = re.compile(r'var\s+(.*?);', re.I | re.DOTALL)
        vxpat = re.compile(r'varexo\s+(.*?);', re.I | re.DOTALL)
        ppat = re.compile(r'parameters\s+(.*?);', re.I | re.DOTALL)
        # Model block: handle optional 'linear' or other qualifiers
        mpat = re.compile(r'model(?:\s*\(.*?\))?\s*;\s*(.*?)\s*end\s*;', re.I | re.DOTALL)

        vm = vpat.search(content)
        if vm:
            # Split, strip, and filter empty strings
            self.var_names = [v.strip() for v in vm.group(1).split() if v.strip()]
        else: print("Warning: 'var' block not found.")

        vxm = vxpat.search(content)
        if vxm:
            self.shock_names = [s.strip() for s in vxm.group(1).split() if s.strip()]
        else: print("Warning: 'varexo' block not found.")

        pm = ppat.search(content)
        if pm:
            # Split, strip trailing commas, filter empty
            self.param_names = [p.strip().rstrip(',') for p in pm.group(1).split() if p.strip()]
        else: print("Warning: 'parameters' block not found.")

        mm = mpat.search(content)
        if mm:
            # Split equations by semicolon, strip whitespace, filter empty lines
            eq_block = mm.group(1)
            self.equations_str = [eq.strip() for eq in eq_block.split(';') if eq.strip()]
        else:
            raise ValueError("Critical Error: 'model; ... end;' block not found.")

        if not self.var_names: raise ValueError("No variables declared in 'var' block.")
        if not self.equations_str: raise ValueError("No equations found in 'model' block.")

    def _create_initial_sympy_symbols(self):
        """Creates sympy symbols for all declared vars, params, shocks."""
        all_names = self.var_names + self.param_names + self.shock_names
        for name in all_names:
            if name not in self.symbols:
                self.symbols[name] = sympy.Symbol(name)
            else:
                # This case should ideally not happen if parsing is clean
                print(f"Warning: Duplicate name detected '{name}' during initial symbol creation.")

    def _replace_dynare_timing(self, eq_str):
        """Replaces Dynare timing like x(+1) with x_p1, y(-2) with y_m2."""
        # Pattern: word boundary, identifier, optional space, parens, optional sign, digits, parens
        pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+\-]?)(\d+)\s*\)')

        def replacer(match):
            var_name, sign, num_str = match.groups()
            # Only replace if var_name is a declared variable
            if var_name in self.var_names_set:
                num = int(num_str)
                if num == 0: # Treat x(0) as just x
                    return var_name
                elif sign == '+' or not sign: # y(1) or y(+1) -> y_p1
                    new_name = f"{var_name}_p{num}"
                elif sign == '-': # y(-1) -> y_m1
                    new_name = f"{var_name}_m{num}"
                else: # Should not happen with pattern
                    return match.group(0)

                # Ensure the symbol exists in our central dictionary
                if new_name not in self.symbols:
                    self.symbols[new_name] = sympy.Symbol(new_name)
                return new_name
            else:
                # Not a variable, maybe a function call, leave unchanged
                return match.group(0)

        return pattern.sub(replacer, eq_str)

    def _parse_equations_to_sympy(self):
        """Converts equation strings to sympy Eq objects (LHS - RHS = 0)."""
        self.sympy_equations_original = []
        for i, eq_str in enumerate(self.equations_str):
            if not eq_str.strip(): continue # Skip empty lines

            # Apply timing replacement first
            processed_eq_str = self._replace_dynare_timing(eq_str)

            try:
                # Check for '=', split, and parse
                if '=' in processed_eq_str:
                    lhs_str, rhs_str = processed_eq_str.split('=', 1)
                    # Use local_dict=self.symbols to recognize all created symbols
                    # Use evaluate=False to prevent premature evaluation (e.g., 1/2 -> 0.5)
                    lhs_expr = sympy.parse_expr(lhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    rhs_expr = sympy.parse_expr(rhs_str.strip(), local_dict=self.symbols, evaluate=False)
                    # Create equation LHS - RHS = 0
                    equation = sympy.Eq(lhs_expr - rhs_expr, 0)
                else:
                    # Assume it's already in the form Expression = 0
                    expr = sympy.parse_expr(processed_eq_str.strip(), local_dict=self.symbols, evaluate=False)
                    equation = sympy.Eq(expr, 0)

                self.sympy_equations_original.append(equation)

            except (SyntaxError, TypeError, NameError) as e:
                print(f"\n--- Error Parsing Equation {i+1} ---")
                print(f"Original: {eq_str}")
                print(f"Processed: {processed_eq_str}")
                print(f"Error: {e}")
                # Optionally, raise the error or collect problematic equations
                raise ValueError(f"Failed to parse equation {i+1}. Check syntax and symbol definitions.") from e
        print(f"Successfully parsed {len(self.sympy_equations_original)} equations into Sympy objects.")

    def _analyze_variable_timing(self, max_k=10):
        """Analyzes min/max leads/lags for each variable."""
        # This primarily populates self.var_timing_info, useful for debugging/reporting
        print("\n--- Stage 1: Analyzing Variable Timing ---")
        var_symbols = {self.symbols[v] for v in self.var_names}
        updated = False
        for eq in self.sympy_equations_original:
            free_symbols_in_eq = eq.lhs.free_symbols
            for base_var_sym in var_symbols:
                base_name = base_var_sym.name
                if base_var_sym in free_symbols_in_eq:
                    self.var_timing_info[base_name]['appears_current'] = True
                    updated = True

                # Check for leads y_p1, y_p2, ...
                for k in range(1, max_k + 1):
                    lead_sym_name = f"{base_name}_p{k}"
                    if lead_sym_name in self.symbols and self.symbols[lead_sym_name] in free_symbols_in_eq:
                        self.var_timing_info[base_name]['max_lead'] = max(self.var_timing_info[base_name]['max_lead'], k)
                        updated = True

                # Check for lags y_m1, y_m2, ...
                for k in range(1, max_k + 1):
                    lag_sym_name = f"{base_name}_m{k}"
                    if lag_sym_name in self.symbols and self.symbols[lag_sym_name] in free_symbols_in_eq:
                        # Store lag as negative number, e.g., -1 for _m1
                        self.var_timing_info[base_name]['min_lag'] = min(self.var_timing_info[base_name]['min_lag'], -k)
                        updated = True
        if updated:
            print("Variable timing analysis complete.")
            # Optional: Print summary
            # for name, info in self.var_timing_info.items():
            #    if info['min_lag'] != 0 or info['max_lead'] != 0:
            #         print(f"  {name}: Lag {info['min_lag']}, Lead {info['max_lead']}, Current: {info['appears_current']}")
        else:
            print("No leads or lags found in equations.")

    def _identify_and_eliminate_static_vars(self):
        """Identifies static vars (only appear at time t) and solves for them."""
        print("\n--- Stage 2: Identifying and Eliminating Static Variables ---")
        all_var_symbols = {self.symbols[v] for v in self.var_names}
        self.static_subs = {} # Reset results {static_symbol: expression}
        equations_to_process = list(self.sympy_equations_original)
        remaining_equations = [] # Equations not used to solve for static vars
        solved_static_symbols = set()
        max_iterations = len(all_var_symbols) + 1 # Safety break
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            found_new_static = False
            equations_kept_this_round = []
            used_indices_this_round = set()

            for i, eq in enumerate(equations_to_process):
                if i in used_indices_this_round: continue # Already used to solve

                current_vars_in_eq = set()
                has_dynamic_timing = False
                potential_static_candidates = set()

                # Analyze symbols in the equation
                for sym in eq.lhs.free_symbols:
                    sym_name = sym.name
                    # Check if it's a base variable or a lead/lag/aux version
                    match_lead = re.match(r"(\w+)_p(\d+)$", sym_name)
                    match_lag  = re.match(r"(\w+)_m(\d+)$", sym_name)
                    match_aux = sym_name.lower().startswith("aux_")

                    if match_lead or match_lag or match_aux:
                        # If any variable appears with lead/lag/aux, the eq is dynamic w.r.t that var
                        has_dynamic_timing = True
                        # Identify the base variable name if possible
                        if match_lead: base = match_lead.group(1)
                        elif match_lag: base = match_lag.group(1)
                        elif match_aux:
                            # Try to extract base from aux name (e.g., aux_Y_lead1 -> Y)
                            aux_base_match = re.match(r"aux_(\w+)_(?:lead|lag)\d*", sym_name, re.IGNORECASE)
                            if aux_base_match: base = aux_base_match.group(1)
                            else: base = None # Cannot determine base from aux name format
                        else: base = None
                        # If we identified a base var from lead/lag/aux, add it to track dynamics
                        if base and base in self.var_names_set:
                            current_vars_in_eq.add(self.symbols[base])
                    elif sym in all_var_symbols:
                        # It's a base variable symbol appearing at time t
                        current_vars_in_eq.add(sym)
                        potential_static_candidates.add(sym)

                # Filter candidates: remove those already solved or appearing dynamically elsewhere
                unsolved_candidates = potential_static_candidates - solved_static_symbols
                # Also remove any candidate if its lead/lag appears *in this specific equation*
                # (We already checked globally, but this ensures the *equation* is static for the candidate)
                static_in_this_eq = set()
                for cand_sym in unsolved_candidates:
                    cand_name = cand_sym.name
                    appears_dynamic_in_eq = False
                    for k in range(1, 15): # Check for leads/lags within the equation's symbols
                        if f"{cand_name}_p{k}" in self.symbols and self.symbols[f"{cand_name}_p{k}"] in eq.lhs.free_symbols:
                            appears_dynamic_in_eq = True; break
                        if f"{cand_name}_m{k}" in self.symbols and self.symbols[f"{cand_name}_m{k}"] in eq.lhs.free_symbols:
                            appears_dynamic_in_eq = True; break
                    if not appears_dynamic_in_eq:
                        static_in_this_eq.add(cand_sym)

                # Try to solve if exactly one unsolved static variable remains in this eq
                if len(static_in_this_eq) == 1:
                    target_static_sym = list(static_in_this_eq)[0]
                    try:
                        # Attempt to solve the equation for the target symbol
                        solutions = sympy.solve(eq.lhs, target_static_sym)

                        if len(solutions) == 1:
                            solution_expr = solutions[0]
                            # Substitute already known static vars into the solution
                            substituted_solution = solution_expr.xreplace(self.static_subs)

                            # Check if the solution is valid (target doesn't appear on RHS)
                            if target_static_sym not in substituted_solution.free_symbols:
                                print(f"  Solved for static var '{target_static_sym.name}' using Eq {i+1}: {target_static_sym.name} = {sympy.sstr(substituted_solution, full_prec=False)}")
                                self.static_subs[target_static_sym] = substituted_solution
                                solved_static_symbols.add(target_static_sym)
                                used_indices_this_round.add(i)
                                found_new_static = True
                            else:
                                # Solution still depends on the variable itself, keep eq
                                equations_kept_this_round.append(eq)
                        else:
                            # Multiple solutions or no solution, keep equation
                            equations_kept_this_round.append(eq)
                    except NotImplementedError:
                        # sympy.solve couldn't handle it (e.g., non-polynomial)
                        print(f"  Warning: sympy.solve failed for Eq {i+1} targeting '{target_static_sym.name}'. Keeping equation.")
                        equations_kept_this_round.append(eq)
                    except Exception as e:
                        print(f"  Warning: Error solving Eq {i+1} for '{target_static_sym.name}': {e}. Keeping equation.")
                        equations_kept_this_round.append(eq)
                else:
                    # 0 or >1 unsolved static vars in this eq, keep it for now
                    equations_kept_this_round.append(eq)

            # Update the list of equations to process for the next iteration
            # Only keep those that were not used to solve for a static var
            equations_to_process = [eq for i, eq in enumerate(equations_to_process) if i not in used_indices_this_round]

            # If no new static variables were found in this iteration, break the loop
            if not found_new_static:
                break

        if iteration == max_iterations:
            print("Warning: Max iterations reached in static variable elimination. May be incomplete.")

        # Final list of equations excludes those used to define static vars
        self.equations_after_static_elim = equations_to_process
        print(f"Static variable elimination complete. Solved for {len(self.static_subs)} static variables.")
        print(f" Remaining dynamic equations: {len(self.equations_after_static_elim)}")
        if self.static_subs:
            print(" Static substitutions found:")
            for var, expr in self.static_subs.items():
                print(f"  {var.name} = {sympy.sstr(expr, full_prec=False)}")

    def _substitute_static_vars(self):
        """Substitutes the solved static variable expressions into remaining equations."""
        print("\n--- Stage 3: Substituting Static Variables into Remaining Equations ---")
        if not self.static_subs:
            print("No static variables found or solved. Skipping substitution.")
            self.equations_after_static_sub = list(self.equations_after_static_elim)
            return

        # --- Build the full substitution dictionary, including time shifts ---
        full_subs_dict = {}
        max_lead_lag = 15 # Maximum lead/lag to check for substitution

        for static_sym, static_expr in self.static_subs.items():
            static_name = static_sym.name
            # Add the base substitution (time t)
            full_subs_dict[static_sym] = static_expr

            # Add substitutions for leads (t+k)
            for k in range(1, max_lead_lag + 1):
                lead_sym_name = f"{static_name}_p{k}"
                if lead_sym_name in self.symbols:
                    # Need to time-shift the expression itself
                    try:
                        shifted_expr = time_shift_expression(static_expr, k, self.symbols, self.var_names_set)
                        full_subs_dict[self.symbols[lead_sym_name]] = shifted_expr
                    except Exception as e:
                        print(f"Warning: Failed to time-shift static expression for {lead_sym_name}: {e}")

            # Add substitutions for lags (t-k)
            for k in range(1, max_lead_lag + 1):
                lag_sym_name = f"{static_name}_m{k}"
                if lag_sym_name in self.symbols:
                    try:
                        shifted_expr = time_shift_expression(static_expr, -k, self.symbols, self.var_names_set)
                        full_subs_dict[self.symbols[lag_sym_name]] = shifted_expr
                    except Exception as e:
                        print(f"Warning: Failed to time-shift static expression for {lag_sym_name}: {e}")
        # --- End building substitution dictionary ---

        # --- Apply substitutions to the remaining dynamic equations ---
        self.equations_after_static_sub = []
        print(f"Applying {len(full_subs_dict)} static substitutions (including time shifts)...")
        for i, eq in enumerate(self.equations_after_static_elim):
            try:
                # Use xreplace for symbol-based substitution
                substituted_lhs = eq.lhs.xreplace(full_subs_dict)
                # Attempt to simplify the result
                simplified_lhs = sympy.simplify(substituted_lhs)
                new_eq = sympy.Eq(simplified_lhs, 0)
                self.equations_after_static_sub.append(new_eq)
                # Optional: print comparison
                # if simplified_lhs != eq.lhs:
                #    print(f"  Eq {i+1} substituted: {sympy.sstr(new_eq.lhs, full_prec=False)} = 0")
            except Exception as e:
                print(f"Warning: Error substituting or simplifying Eq {i+1}: {e}. Keeping original form.")
                self.equations_after_static_sub.append(eq) # Keep original if error

        print(f"Substitution complete. Resulting equations: {len(self.equations_after_static_sub)}")

    def _handle_aux_vars(self, file_path=None):
        """
        Identifies leads > +1 and lags < -1, creates auxiliary variables,
        defines their equations, and substitutes them into the main equations.
        """
        print("\n--- Stage 4: Handling Leads > +1 and Lags < -1 (Auxiliary Variables) ---")
        self.aux_lead_vars = {} # Reset {aux_lead_name: symbol}
        self.aux_lag_vars = {}  # Reset {aux_lag_name: symbol}
        self.aux_var_definitions = [] # Reset [sympy.Eq]
        # Start with equations after static substitution
        current_equations = list(self.equations_after_static_sub)
        # Dictionary to store substitutions: {original_symbol: replacement_symbol}
        # Replacement symbol will be a lead/lag of an aux variable
        subs_for_long_leads_lags = {}

        # Get dynamic variable names (base names excluding static ones)
        static_var_names = {s.name for s in self.static_subs.keys()}
        dynamic_base_var_names = [v for v in self.var_names if v not in static_var_names]

        # --- 1. Identify Maximum Leads/Lags Needing Aux Vars ---
        max_leads_needed = collections.defaultdict(int) # {base_name: max_k} for k > 1
        max_lags_needed = collections.defaultdict(int)  # {base_name: max_k} for k > 1
        for eq in current_equations:
            for atom in eq.lhs.free_symbols:
                atom_name = atom.name
                match_lead = re.match(r"(\w+)_p(\d+)", atom_name)
                match_lag = re.match(r"(\w+)_m(\d+)", atom_name)

                if match_lead:
                    base, k_str = match_lead.groups()
                    k = int(k_str)
                    # Only consider leads of *dynamic* base variables requiring aux
                    if k > 1 and base in dynamic_base_var_names:
                        max_leads_needed[base] = max(max_leads_needed[base], k)
                elif match_lag:
                    base, k_str = match_lag.groups()
                    k = int(k_str)
                    # Only consider lags of *dynamic* base variables requiring aux
                    if k > 1 and base in dynamic_base_var_names:
                        max_lags_needed[base] = max(max_lags_needed[base], k)

        # --- 2. Create Auxiliary Lead Variables and Definitions ---
        if max_leads_needed:
            print("Creating auxiliary LEAD variables:")
            # Sort for deterministic order
            for base_var_name in sorted(max_leads_needed.keys()):
                max_lead = max_leads_needed[base_var_name]
                # We need aux vars aux_lead1, ..., aux_lead(max_lead - 1)
                # aux_leadK represents E[base_var(t+k)]
                for k in range(1, max_lead): # Create aux_lead1 up to aux_lead(max_lead-1)
                    aux_name = f"aux_{base_var_name}_lead{k}"
                    # Ensure the aux symbol exists
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    self.aux_lead_vars[aux_name] = self.symbols[aux_name]

                    # --- Define the auxiliary equation ---
                    # aux_leadK(t) = E_t[RHS_var(t+1)]
                    if k == 1:
                        # aux_lead1(t) = E_t[base_var(t+1)] => aux_lead1 = base_var_p1
                        rhs_var_name = f"{base_var_name}_p1"
                    else:
                        # aux_leadK(t) = E_t[aux_lead(K-1)(t+1)] => aux_leadK = aux_lead(K-1)_p1
                        prev_aux_name = f"aux_{base_var_name}_lead{k-1}"
                        rhs_var_name = f"{prev_aux_name}_p1"

                    # Ensure the RHS symbol exists
                    if rhs_var_name not in self.symbols:
                        self.symbols[rhs_var_name] = sympy.Symbol(rhs_var_name)

                    # Create the equation: aux_name - rhs_symbol = 0
                    aux_eq = sympy.Eq(self.symbols[aux_name] - self.symbols[rhs_var_name], 0)
                    # Add if not already present (shouldn't happen with sorted processing)
                    if aux_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(aux_eq)
                        print(f"  - Defined: {aux_eq.lhs} = 0")

                    # --- Create Substitution Rule ---
                    # We need to replace base_var_p(k+1) with the lead of aux_leadK
                    # E.g., replace Y_p2 with aux_Y_lead1_p1
                    #      replace Y_p3 with aux_Y_lead2_p1
                    original_lead_name = f"{base_var_name}_p{k+1}"
                    replacement_lead_name = f"{aux_name}_p1" # Lead of the k-th aux var

                    # Ensure original symbol exists (it should if max_leads_needed was accurate)
                    if original_lead_name in self.symbols:
                        # Ensure the replacement symbol exists
                        if replacement_lead_name not in self.symbols:
                            self.symbols[replacement_lead_name] = sympy.Symbol(replacement_lead_name)
                        # Store the substitution rule
                        subs_for_long_leads_lags[self.symbols[original_lead_name]] = self.symbols[replacement_lead_name]
                        print(f"    - Substitution rule: {original_lead_name} -> {replacement_lead_name}")
                    else:
                        # This indicates a potential issue in logic or symbol creation
                        print(f"Warning: Expected symbol '{original_lead_name}' not found for substitution rule.")

        # --- 3. Create Auxiliary Lag Variables and Definitions ---
        if max_lags_needed:
            print("Creating auxiliary LAG variables:")
            # Sort for deterministic order
            for base_var_name in sorted(max_lags_needed.keys()):
                max_lag = max_lags_needed[base_var_name]
                # We need aux vars aux_lag1, ..., aux_lag(max_lag - 1)
                # aux_lagK represents base_var(t-k)
                for k in range(1, max_lag): # Create aux_lag1 up to aux_lag(max_lag-1)
                    aux_name = f"aux_{base_var_name}_lag{k}"
                    # Ensure the aux symbol exists
                    if aux_name not in self.symbols:
                        self.symbols[aux_name] = sympy.Symbol(aux_name)
                    self.aux_lag_vars[aux_name] = self.symbols[aux_name]

                    # --- Define the auxiliary equation ---
                    # aux_lagK(t) = RHS_var(t-1)
                    if k == 1:
                        # aux_lag1(t) = base_var(t-1) => aux_lag1 = base_var_m1
                        rhs_var_name = f"{base_var_name}_m1"
                    else:
                        # aux_lagK(t) = aux_lag(K-1)(t-1) => aux_lagK = aux_lag(K-1)_m1
                        prev_aux_name = f"aux_{base_var_name}_lag{k-1}"
                        rhs_var_name = f"{prev_aux_name}_m1"

                    # Ensure the RHS symbol exists
                    if rhs_var_name not in self.symbols:
                        self.symbols[rhs_var_name] = sympy.Symbol(rhs_var_name)

                    # Create the equation: aux_name - rhs_symbol = 0
                    aux_eq = sympy.Eq(self.symbols[aux_name] - self.symbols[rhs_var_name], 0)
                    # Add if not already present
                    if aux_eq not in self.aux_var_definitions:
                        self.aux_var_definitions.append(aux_eq)
                        print(f"  - Defined: {aux_eq.lhs} = 0")

                    # --- Create Substitution Rule ---
                    # We need to replace base_var_m(k+1) with the lag of aux_lagK
                    # E.g., replace Y_m2 with aux_Y_lag1_m1
                    #      replace Y_m3 with aux_Y_lag2_m1
                    original_lag_name = f"{base_var_name}_m{k+1}"
                    replacement_lag_name = f"{aux_name}_m1" # Lag of the k-th aux var

                    # Ensure original symbol exists
                    if original_lag_name in self.symbols:
                        # Ensure the replacement symbol exists
                        if replacement_lag_name not in self.symbols:
                            self.symbols[replacement_lag_name] = sympy.Symbol(replacement_lag_name)
                        # Store the substitution rule
                        subs_for_long_leads_lags[self.symbols[original_lag_name]] = self.symbols[replacement_lag_name]
                        print(f"    - Substitution rule: {original_lag_name} -> {replacement_lag_name}")
                    else:
                        print(f"Warning: Expected symbol '{original_lag_name}' not found for substitution rule.")

        # --- 4. Apply Substitutions to Original Dynamic Equations ---
        self.equations_after_aux_sub = [] # Reset list
        print(f"\nApplying {len(subs_for_long_leads_lags)} long lead/lag substitutions to main equations...")
        if subs_for_long_leads_lags: # Only apply if there are rules
            for i, eq in enumerate(current_equations):
                try:
                    subbed_lhs = eq.lhs.xreplace(subs_for_long_leads_lags)
                    # Simplify after substitution
                    simplified_lhs = sympy.simplify(subbed_lhs)
                    subbed_eq = sympy.Eq(simplified_lhs, 0)
                    self.equations_after_aux_sub.append(subbed_eq)
                except Exception as e:
                    print(f"Warning: Error substituting/simplifying Eq {i+1} during aux handling: {e}. Keeping original form.")
                    self.equations_after_aux_sub.append(eq)
            print("Substitution application complete.")
        else:
            print("No long leads/lags found requiring substitution.")
            self.equations_after_aux_sub = current_equations # No changes needed

        # --- 5. Finalize State Variables ---
        # State includes original dynamic vars + all created aux vars
        self.final_dynamic_var_names = (
            dynamic_base_var_names
            + sorted(list(self.aux_lead_vars.keys())) # Sort aux names for consistency
            + sorted(list(self.aux_lag_vars.keys()))
        )
        print(f"\nFinal dynamic state variables ({len(self.final_dynamic_var_names)}):")
        print(f"  {self.final_dynamic_var_names}")
        print(f"Equations after aux substitutions: {len(self.equations_after_aux_sub)}")
        print(f"Auxiliary variable definitions: {len(self.aux_var_definitions)}")

        # Optional: Save intermediate results if file_path is provided
        if file_path:
            output_lines = ["--- Aux Handling Stage Output ---"]
            # Add more details from the print statements above if needed
            self._save_intermediate_file(file_path, output_lines,
                                        self.equations_after_aux_sub + self.aux_var_definitions,
                                        "Equations After Aux Handling (Main + Definitions)")

        # Return tuple might not be necessary if results are stored in self
        # return self.equations_after_aux_sub, self.aux_var_definitions

    def _define_state_vector(self):
        """Orders the final dynamic variables into the state vector Y_t."""
        # This ordering tries to group predetermined/lagged vars first, then mixed/forward vars
        # It roughly follows the Dynare convention which can be helpful for some solvers
        print("\n--- Stage 5: Defining State Vector Order ---")
        if not self.final_dynamic_var_names:
            raise ValueError("Cannot define state vector: final dynamic variable list is empty.")

        all_final_equations = self.equations_after_aux_sub + self.aux_var_definitions
        final_var_symbols = {self.symbols[name] for name in self.final_dynamic_var_names}

        predetermined_vars = [] # Vars only appearing at t and t-1 (state vars)
        mixed_vars = []         # Vars appearing at t+1 (jump vars / forward-looking aux)

        # Classify each final dynamic variable
        for var_sym in final_var_symbols:
            var_name = var_sym.name
            has_lead_p1 = False
            lead_p1_name = f"{var_name}_p1"

            # Check if the t+1 version appears in *any* final equation
            if lead_p1_name in self.symbols:
                lead_p1_sym = self.symbols[lead_p1_name]
                for eq in all_final_equations:
                    if lead_p1_sym in eq.lhs.free_symbols:
                        has_lead_p1 = True
                        break

            if has_lead_p1:
                mixed_vars.append(var_sym)
            else:
                predetermined_vars.append(var_sym)

        # Further sort within categories (optional, but helps consistency):
        # Predetermined: Original vars first, then aux_lag vars
        pred_orig = sorted([v for v in predetermined_vars if not v.name.lower().startswith('aux_')], key=lambda s: s.name)
        pred_aux_lags = sorted([v for v in predetermined_vars if v.name.lower().startswith('aux_') and '_lag' in v.name.lower()], key=lambda s: s.name)

        # Mixed/Forward: Original vars first, then aux_lead vars
        mixed_orig = sorted([v for v in mixed_vars if not v.name.lower().startswith('aux_')], key=lambda s: s.name)
        mixed_aux_leads = sorted([v for v in mixed_vars if v.name.lower().startswith('aux_') and '_lead' in v.name.lower()], key=lambda s: s.name)

        # Combine into the final ordered list
        self.state_vars_ordered = pred_orig + pred_aux_lags + mixed_orig + mixed_aux_leads

        # Create the mapping from symbol to index
        self.state_var_map = {sym: i for i, sym in enumerate(self.state_vars_ordered)}

        print(f"State vector defined with {len(self.state_vars_ordered)} variables.")
        print(" Order:")
        for i, sym in enumerate(self.state_vars_ordered):
            print(f"  {i}: {sym.name}")

        # Sanity check: ensure all final dynamic vars are included
        if len(self.state_vars_ordered) != len(self.final_dynamic_var_names):
            print("\n!!! Warning: Mismatch between final dynamic var count and state vector length !!!")
            print(f" Final dynamic names ({len(self.final_dynamic_var_names)}): {self.final_dynamic_var_names}")
            print(f" State vector symbols ({len(self.state_vars_ordered)}): {[s.name for s in self.state_vars_ordered]}")
            missing = set(self.final_dynamic_var_names) - {s.name for s in self.state_vars_ordered}
            extra = {s.name for s in self.state_vars_ordered} - set(self.final_dynamic_var_names)
            if missing: print(f" Missing from state vector: {missing}")
            if extra: print(f" Extra in state vector: {extra}")
            # This usually indicates a logic error in classification or final list update

    def _build_final_equations(self):
        """Combines main dynamic equations and aux definitions for Jacobian calculation."""
        print("\n--- Stage 6: Building Final Equation System for Jacobian ---")
        # The final system includes the dynamic equations (after static/aux subs)
        # AND the definitions of the auxiliary variables.
        self.final_equations_for_jacobian = (
            list(self.equations_after_aux_sub)
            + list(self.aux_var_definitions)
        )
        n_eq = len(self.final_equations_for_jacobian)
        n_state = len(self.state_vars_ordered)

        print(f"Final system built with {n_eq} equations.")

        # Check if the number of equations matches the number of state variables
        if n_eq != n_state:
            print("\n!!! Warning: Number of final equations ({n_eq}) does not match "
                f"number of state variables ({n_state}) !!!")
            print(" This may indicate an issue with static/aux variable handling or model specification.")
            # Depending on the solver, this might still work or might cause errors later.
        else:
            print("Number of equations matches number of state variables.")

    def get_numerical_ABCD(self, param_dict_values, file_path=None):
        """
        Calculates numerical A, B, C, D matrices by evaluating symbolic Jacobians.
        D = -d(eq)/d(shock).
        """
        print("\n--- Stage 7: Calculating Numerical A, B, C, D Matrices ---")
        if not self.final_equations_for_jacobian:
            raise ValueError("Final equations for Jacobian are not built. Run previous stages.")
        if not self.state_vars_ordered:
            raise ValueError("State vector is not defined. Run _define_state_vector.")

        n_state = len(self.state_vars_ordered)
        n_eq = len(self.final_equations_for_jacobian)
        n_shocks = len(self.shock_names)

        # Ensure equations and state counts match before proceeding
        if n_eq != n_state:
            print(f"Warning: Mismatch persists: {n_eq} equations, {n_state} states. Jacobian may not be square.")
            # Allow proceeding but Jacobians might not be n_state x n_state

        self.last_param_values = param_dict_values # Store for reference/function generation

        # Create substitution dictionary for parameters ONLY
        param_subs = {}
        for p_name, p_value in param_dict_values.items():
            if p_name in self.symbols:
                param_subs[self.symbols[p_name]] = p_value
            else:
                print(f"Warning: Parameter '{p_name}' provided but not found in model symbols.")

        # --- Create Symbolic Vectors for Jacobians ---
        # Vector of state variables at time t
        state_vec_t = sympy.Matrix(self.state_vars_ordered)

        # Vector of state variables at time t+1 (leads)
        state_vec_tp1_list = []
        for state_sym in self.state_vars_ordered:
            lead_name = f"{state_sym.name}_p1"
            if lead_name not in self.symbols: # Ensure symbol exists
                self.symbols[lead_name] = sympy.Symbol(lead_name)
            state_vec_tp1_list.append(self.symbols[lead_name])
        state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)

        # Vector of state variables at time t-1 (lags)
        state_vec_tm1_list = []
        for state_sym in self.state_vars_ordered:
            lag_name = f"{state_sym.name}_m1"
            if lag_name not in self.symbols: # Ensure symbol exists
                self.symbols[lag_name] = sympy.Symbol(lag_name)
            state_vec_tm1_list.append(self.symbols[lag_name])
        state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)

        # Vector of shock variables
        shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
        shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None

        # Vector of final equations (LHS, assuming form Expression = 0)
        eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])
        # --- End Symbolic Vector Creation ---

        print("Calculating symbolic Jacobians...")
        try:
            # Jacobian w.r.t. Y(t+1) -> A matrix
            A_sym = eq_vec.jacobian(state_vec_tp1)
            # Jacobian w.r.t. Y(t) -> B matrix
            B_sym = eq_vec.jacobian(state_vec_t)
            # Jacobian w.r.t. Y(t-1) -> C matrix
            C_sym = eq_vec.jacobian(state_vec_tm1)

            # Jacobian w.r.t. shocks -> D matrix ( D = - d(eq)/d(shock) )
            if shock_vec and n_shocks > 0:
                D_sym = eq_vec.jacobian(shock_vec)
            else:
                # If no shocks, D is an empty matrix with correct number of rows (n_eq)
                D_sym = sympy.zeros(n_eq, 0)
            print("Symbolic Jacobians calculated.")

        except Exception as e:
            print(f"Error during symbolic Jacobian calculation: {e}")
            print(" Check equation forms and variable definitions.")
            # You might want to inspect eq_vec and the state vectors here
            raise

        print("Substituting parameter values into Jacobians...")
        try:
            # --- Numerical Substitution using evalf ---
            # evalf is generally preferred for converting symbolic -> numeric
            # It handles precision and common functions better than direct subs+float
            A_num = np.array(A_sym.evalf(subs=param_subs).tolist(), dtype=float)
            B_num = np.array(B_sym.evalf(subs=param_subs).tolist(), dtype=float)
            C_num = np.array(C_sym.evalf(subs=param_subs).tolist(), dtype=float)
            # D matrix substitution
            if n_shocks > 0:
                D_num = np.array(D_sym.evalf(subs=param_subs).tolist(), dtype=float)
            else:
                # Ensure D has correct shape (n_eq x 0) if no shocks
                D_num = np.zeros((n_eq, 0), dtype=float)
            print("Numerical substitution complete.")
            # --- End Numerical Substitution ---

        except Exception as e:
            print(f"Error during numerical substitution (evalf): {e}")
            print(" This might happen if some symbols remain unsubstituted (e.g., missing parameters).")
            # Optional: Print symbolic matrices that might contain remaining symbols
            # print("A_sym:", A_sym); print("B_sym:", B_sym); # etc.
            raise

        # --- Final Dimension Checks ---
        # Expected shape is (n_eq x n_state) for A, B, C and (n_eq x n_shocks) for D
        expected_ABC_shape = (n_eq, n_state)
        expected_D_shape = (n_eq, n_shocks)

        valid_dims = True
        if A_num.shape != expected_ABC_shape: print(f"ERROR: A matrix shape is {A_num.shape}, expected {expected_ABC_shape}"); valid_dims=False
        if B_num.shape != expected_ABC_shape: print(f"ERROR: B matrix shape is {B_num.shape}, expected {expected_ABC_shape}"); valid_dims=False
        if C_num.shape != expected_ABC_shape: print(f"ERROR: C matrix shape is {C_num.shape}, expected {expected_ABC_shape}"); valid_dims=False
        if D_num.shape != expected_D_shape: print(f"ERROR: D matrix shape is {D_num.shape}, expected {expected_D_shape}"); valid_dims=False

        if not valid_dims:
            raise ValueError("Matrix dimension mismatch after numerical calculation.")
        if n_eq != n_state:
            print(f"Note: Jacobians have shape ({n_eq}, {n_state}) as equations != states.")

        print("Numerical matrices A, B, C, D calculated successfully.")

        # --- Save Numerical Matrices (Optional) ---
        if file_path:
            self._save_final_matrices(file_path, A_num, B_num, C_num, D_num)

        # Return numerical matrices and the names used for states/shocks
        return A_num, B_num, C_num, D_num, [s.name for s in self.state_vars_ordered], self.shock_names

    def _generate_matrix_assignments_code_helper(self, matrix_sym, matrix_name):
        """Generates Python code lines for element-wise matrix assignments."""
        # Internal helper used by generate_matrix_function_file
        try:
            rows, cols = matrix_sym.shape
        except Exception as e:
            print(f"Error getting shape for symbolic matrix {matrix_name}. Matrix: {matrix_sym}")
            raise ValueError(f"Could not determine shape for symbolic matrix {matrix_name}") from e

        indent = "    " # Standard 4 spaces indentation
        # Initialize matrix creation line
        code_lines = [f"{indent}{matrix_name} = np.zeros(({rows}, {cols}), dtype=float)"]
        assignments = []

        # Iterate through elements
        for r in range(rows):
            for c in range(cols):
                try:
                    element = matrix_sym[r, c]
                except IndexError:
                    # Should not happen if shape is correct, but defensive check
                    print(f"Warning: Index ({r},{c}) out of bounds for {matrix_name} shape {matrix_sym.shape}")
                    continue

                # Add assignment only if element is not structurally zero
                if element != 0 and element is not sympy.S.Zero:
                    try:
                        # Use sympy.sstr for standard Python string representation
                        expr_str = sympy.sstr(element, full_prec=False)
                        # Append the assignment line
                        assignments.append(f"{indent}{matrix_name}[{r}, {c}] = {expr_str}")
                    except Exception as str_e:
                        # Handle potential errors during string conversion
                        print(f"Warning: String conversion failed for {matrix_name}[{r},{c}]: {str_e}")
                        assignments.append(f"{indent}# Error generating code for {matrix_name}[{r},{c}] = {element}")

        # Add assignments block if any non-zero elements were found
        if assignments:
            code_lines.append(f"{indent}# Fill non-zero elements of {matrix_name}")
            code_lines.extend(assignments)
        else:
            # If the matrix was all zeros, add a comment indicating so
            code_lines.append(f"{indent}# Matrix {matrix_name} is structurally zero.")

        # Join lines into a single code block string
        return "\n".join(code_lines)

    def generate_matrix_function_file(self, filename="jacobian_matrices.py"):
        """
        Generates a Python file containing a function `jacobian_matrices(theta)`
        that computes and returns the numerical A, B, C, D matrices.
        """
        function_name = "jacobian_matrices"
        print(f"\n--- Generating Python Function File: {filename} ---")
        if not self.final_equations_for_jacobian or not self.state_vars_ordered:
            raise ValueError("Preprocessing (stages 1-6) must be run successfully first.")

        n_state = len(self.state_vars_ordered)
        n_eq = len(self.final_equations_for_jacobian) # Use actual equation count
        n_shocks = len(self.shock_names)

        # --- 1. Ensure Symbolic Jacobians Exist ---
        # Recalculate or retrieve symbolic Jacobians if necessary.
        # It's often cleaner to recalculate here to ensure they are correct.
        try:
            state_vec_t = sympy.Matrix(self.state_vars_ordered)
            state_vec_tp1_list = [self.symbols.get(f"{s.name}_p1", sympy.Symbol(f"{s.name}_p1")) for s in self.state_vars_ordered]
            state_vec_tm1_list = [self.symbols.get(f"{s.name}_m1", sympy.Symbol(f"{s.name}_m1")) for s in self.state_vars_ordered]
            state_vec_tp1 = sympy.Matrix(state_vec_tp1_list)
            state_vec_tm1 = sympy.Matrix(state_vec_tm1_list)
            shock_syms_list = [self.symbols[s] for s in self.shock_names if s in self.symbols]
            shock_vec = sympy.Matrix(shock_syms_list) if shock_syms_list else None
            eq_vec = sympy.Matrix([eq.lhs for eq in self.final_equations_for_jacobian])

            print("Recalculating symbolic Jacobians for function generation...")
            A_sym = eq_vec.jacobian(state_vec_tp1)
            B_sym = eq_vec.jacobian(state_vec_t)
            C_sym = eq_vec.jacobian(state_vec_tm1)
            D_sym = eq_vec.jacobian(shock_vec) if shock_vec and n_shocks > 0 else sympy.zeros(n_eq, 0)
            print("Symbolic Jacobians ready.")
        except Exception as e:
            print(f"Error recalculating symbolic Jacobians for function generation: {e}")
            raise
        # --- End Recalculate Symbolic Jacobians ---

        # --- 2. Identify Parameters Used in Jacobians ---
        # Use the parameter order from the original .dyn file parsing
        ordered_params_from_mod = self.param_names
        # Find all parameter symbols actually present in the symbolic matrices
        all_symbols_in_matrices = set().union(*(mat.free_symbols for mat in [A_sym, B_sym, C_sym, D_sym]))
        param_symbols_in_matrices = {s for s in all_symbols_in_matrices if s.name in self.param_names}
        # Create a list of used parameter names, maintaining original order
        used_params_ordered = [p for p in ordered_params_from_mod if self.symbols.get(p) in param_symbols_in_matrices]
        # Map original parameter names to their index in the full list
        param_indices = {p_name: i for i, p_name in enumerate(ordered_params_from_mod)}
        # --- End Parameter Info ---

        # --- 3. Generate Python Code Strings for Each Matrix ---
        print("Generating Python code strings for matrix assignments...")
        code_A = self._generate_matrix_assignments_code_helper(A_sym, 'A')
        code_B = self._generate_matrix_assignments_code_helper(B_sym, 'B')
        code_C = self._generate_matrix_assignments_code_helper(C_sym, 'C')
        code_D = self._generate_matrix_assignments_code_helper(D_sym, 'D')
        print("Code generation complete.")
        # --- End Generate Code Strings ---

        # --- 4. Assemble Final File Content ---
        file_lines = []
        file_lines.append(f"# Auto-generated by DynareParser for model '{os.path.basename(self.mod_file_path)}'")
        file_lines.append(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        file_lines.append("# DO NOT EDIT MANUALLY - Changes will be overwritten.")
        file_lines.append("")
        file_lines.append("import numpy as np")
        # Import math functions for expressions like sqrt(), exp(), etc.
        file_lines.append("from math import *")
        file_lines.append("")
        file_lines.append(f"def {function_name}(theta):")
        file_lines.append(f"    \"\"\"")
        file_lines.append(f"    Computes the Jacobian matrices A, B, C, D for the model '{os.path.basename(self.mod_file_path)}'.")
        file_lines.append(f"")
        file_lines.append(f"    Args:")
        file_lines.append(f"        theta (list or np.ndarray): Parameter vector of length {len(ordered_params_from_mod)}")
        file_lines.append(f"            Expected order: {ordered_params_from_mod}")
        file_lines.append(f"")
        file_lines.append(f"    Returns:")
        file_lines.append(f"        tuple: (A, B, C, D, state_names, shock_names)")
        file_lines.append(f"            A, B, C: ({n_eq} x {n_state}) numpy arrays")
        file_lines.append(f"            D: ({n_eq} x {n_shocks}) numpy array")
        file_lines.append(f"            state_names: List of state variable names (order corresponds to matrix columns)")
        file_lines.append(f"            shock_names: List of shock names (order corresponds to D matrix columns)")
        file_lines.append(f"    \"\"\"")
        file_lines.append(f"    # --- Parameter Unpacking ---")
        file_lines.append(f"    expected_len = {len(ordered_params_from_mod)}")
        file_lines.append(f"    if len(theta) != expected_len:")
        # Use correct f-string formatting for nested braces
        file_lines.append(f"        raise ValueError(f'Expected {{expected_len}} parameters, but received {{len(theta)}}.')")
        file_lines.append("")
        file_lines.append("    # Unpack only the parameters used in the Jacobians")
        # Generate unpacking lines, indented correctly
        for p_name in used_params_ordered:
            idx = param_indices[p_name]
            file_lines.append(f"    {p_name} = theta[{idx}]")
        file_lines.append("")
        file_lines.append("    # --- Matrix Calculation ---")
        # Add the generated code blocks for A, B, C, D
        file_lines.append(code_A)
        file_lines.append("")
        file_lines.append(code_B)
        file_lines.append("")
        file_lines.append(code_C)
        file_lines.append("")
        file_lines.append(code_D)
        file_lines.append("")
        file_lines.append("    # --- Return Results ---")
        # Use repr() to get a string representation of the lists
        file_lines.append(f"    state_names = {repr([s.name for s in self.state_vars_ordered])}")
        file_lines.append(f"    shock_names = {repr(self.shock_names)}")
        file_lines.append("")
        file_lines.append("    return A, B, C, D, state_names, shock_names")

        final_file_content = "\n".join(file_lines)
        # --- End Assemble File Content ---

        # --- 5. Write the Function File ---
        try:
            # Ensure the output directory exists
            dir_name = os.path.dirname(filename)
            if dir_name: # Create directory if it's not the current directory
                os.makedirs(dir_name, exist_ok=True)

            with open(filename, "w", encoding='utf-8') as f:
                f.write(final_file_content)
            print(f"Successfully generated function file: {filename}")
        except Exception as e:
            print(f"Error writing function file {filename}: {e}")
            # Optionally re-raise or handle differently

    def process_model(self, param_dict_values_or_list, output_dir_intermediate=None,
                    output_dir_final=None, generate_function=True):
        """
        Runs the full parsing and matrix generation pipeline.

        Args:
            param_dict_values_or_list: Dict or ordered list/array of parameter values.
            output_dir_intermediate: Directory for intermediate text files.
            output_dir_final: Directory for final .pkl matrix file and .py function file.
            generate_function: If True, generates the Python function file.

        Returns:
            tuple: (A, B, C, D, state_names, shock_names) on success, None on failure.
        """
        # --- 1. Parameter Input Handling ---
        param_dict_values = {}
        if isinstance(param_dict_values_or_list, (list, tuple, np.ndarray)):
            if len(param_dict_values_or_list) != len(self.param_names):
                raise ValueError(f"Input parameter list/array length ({len(param_dict_values_or_list)}) "
                                f"does not match declared parameters ({len(self.param_names)}: {self.param_names})")
            # Create dict from list using the parser's parameter order
            param_dict_values = {name: val for name, val in zip(self.param_names, param_dict_values_or_list)}
        elif isinstance(param_dict_values_or_list, dict):
            param_dict_values = param_dict_values_or_list
            # Check for missing parameters compared to the declared list
            missing_keys = set(self.param_names) - set(param_dict_values.keys())
            if missing_keys:
                print(f"Warning: Input parameter dict is missing keys declared in the model: {missing_keys}")
            # Check for extra parameters not declared (optional)
            extra_keys = set(param_dict_values.keys()) - set(self.param_names)
            if extra_keys:
                print(f"Warning: Input parameter dict contains keys not declared in the model: {extra_keys}")
        else:
            raise TypeError("param_dict_values_or_list must be a dictionary, list, tuple, or numpy array.")
        # --- End Parameter Handling ---

        # --- 2. Define Output File Paths ---
        base_name = os.path.splitext(os.path.basename(self.mod_file_path))[0]
        fpaths_inter = {} # Dictionary to hold intermediate file paths
        final_matrices_pkl = None
        function_py = None

        # Setup intermediate paths if directory provided
        if output_dir_intermediate:
            os.makedirs(output_dir_intermediate, exist_ok=True)
            inter_names = ["0_original_eqs", "1_timing", "2_static_elim", "3_static_sub",
                        "4_aux_handling", "5_state_def", "6_final_eqs",
                        "7_DEBUG_final_eqs_used"] # Add more steps if needed
            for i, name in enumerate(inter_names):
                fpaths_inter[i] = os.path.join(output_dir_intermediate, f"{i}_{base_name}_{name}.txt")

        # Setup final paths if directory provided
        if output_dir_final:
            os.makedirs(output_dir_final, exist_ok=True)
            final_matrices_pkl = os.path.join(output_dir_final, f"{base_name}_matrices.pkl")
            if generate_function:
                function_py = os.path.join(output_dir_final, f"{base_name}_jacobian_matrices.py")
        # --- End Define Paths ---

        # --- 3. Run Processing Pipeline ---
        try:
            print("\n" + "="*30 + " Starting Model Processing Pipeline " + "="*30)

            # Stage 0: Save Original Parsed Equations
            self._save_intermediate_file(fpaths_inter.get(0), ["Stage 0: Original Parsed Equations"],
                                        self.sympy_equations_original, "Original Sympy Equations")

            # Stage 1: Analyze Timing
            self._analyze_variable_timing()
            timing_lines = ["Stage 1: Variable Timing Analysis"]
            for name, info in sorted(self.var_timing_info.items()):
                timing_lines.append(f"  {name}: Lag={info['min_lag']}, Lead={info['max_lead']}, Current={info['appears_current']}")
            self._save_intermediate_file(fpaths_inter.get(1), timing_lines)

            # Stage 2: Identify & Eliminate Static Vars
            self._identify_and_eliminate_static_vars()
            static_elim_lines = ["Stage 2: Static Variable Elimination"]
            if self.static_subs:
                static_elim_lines.append(" Solved Static Vars:")
                for var, expr in self.static_subs.items(): static_elim_lines.append(f"  {var.name} = {sympy.sstr(expr, full_prec=False)}")
            else: static_elim_lines.append(" No static variables solved.")
            self._save_intermediate_file(fpaths_inter.get(2), static_elim_lines,
                                        self.equations_after_static_elim, "Equations After Static Elimination")

            # Stage 3: Substitute Static Vars
            self._substitute_static_vars()
            self._save_intermediate_file(fpaths_inter.get(3), ["Stage 3: Static Variable Substitution"],
                                        self.equations_after_static_sub, "Equations After Static Substitution")

            # Stage 4: Handle Aux Vars (pass potential save path)
            self._handle_aux_vars(file_path=fpaths_inter.get(4)) # Pass path directly to method

            # Stage 5: Define State Vector Order
            self._define_state_vector()
            state_lines = ["Stage 5: State Vector Definition"]
            state_lines.append(f" Ordered State Vector ({len(self.state_vars_ordered)}):")
            state_lines.extend([f"  {i}: {s.name}" for i, s in enumerate(self.state_vars_ordered)])
            self._save_intermediate_file(fpaths_inter.get(5), state_lines)

            # Stage 6: Build Final Equation System
            self._build_final_equations()
            self._save_intermediate_file(fpaths_inter.get(6), ["Stage 6: Final Equation System"],
                                        self.final_equations_for_jacobian, "Final Equations for Jacobian")

            # Stage 7: DEBUG Save Final Equations just before Jacobian (Redundant but safe)
            if output_dir_intermediate:
                # Use the pre-defined path from fpaths_inter if available
                eq_out_path = fpaths_inter.get(7, os.path.join(output_dir_intermediate, f"DEBUG_{base_name}_final_equations_used.txt"))
                self.save_final_equations_to_txt(filename=eq_out_path)

            # Stage 8: Get Numerical ABCD Matrices
            # Pass the DICT of parameters and the path for the final .pkl file
            A, B, C, D, state_names, shock_names = self.get_numerical_ABCD(
                param_dict_values,
                file_path=final_matrices_pkl
            )

            # Stage 9: Generate Python Function File (if requested)
            if generate_function and function_py:
                self.generate_matrix_function_file(filename=function_py)

            print("\n" + "="*32 + " Model Processing Successful " + "="*33)
            # Return the calculated numerical results
            return A, B, C, D, state_names, shock_names

        except Exception as e:
            print(f"\n--- ERROR during model processing pipeline: {type(e).__name__}: {e} ---")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return None # Indicate failure
        # --- End Processing Pipeline ---

    # --- Helper methods for saving files ---
    def _save_intermediate_file(self, file_path, lines, equations=None, equations_title="Equations"):
        """Helper to save text-based intermediate results."""
        if not file_path: return # Skip if no path provided
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)

            with open(file_path, "w", encoding='utf-8') as f:
                f.write(f"--- {os.path.basename(file_path)} ---\n")
                f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("\n".join(lines))
                if equations is not None: # Allow saving even if equations list is empty
                    f.write(f"\n\n--- {equations_title} ({len(equations)}) ---\n")
                    if equations:
                        for i, eq in enumerate(equations):
                            try:
                                # Use sstr for more standard output, limit precision
                                f.write(f"  Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
                            except Exception as write_e:
                                f.write(f"  Eq {i+1}: Error writing equation: {write_e}\n")
                    else:
                        f.write("  (No equations in this list)\n")
            # print(f"Intermediate results saved to {file_path}") # Optional: reduce verbosity
        except Exception as e:
            print(f"Warning: Could not save intermediate file {file_path}. Error: {e}")

    def _save_final_matrices(self, file_path, A, B, C, D):
        """Helper to save final numerical matrices to .pkl and .txt."""
        if not file_path: return

        # Data to save in pickle file
        matrix_data = {
            'A': A, 'B': B, 'C': C, 'D': D,
            'state_names': [s.name for s in self.state_vars_ordered],
            'shock_names': self.shock_names,
            'param_names': self.param_names, # Include params order for reference
            'timestamp': datetime.datetime.now().isoformat()
        }
        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name: os.makedirs(dir_name, exist_ok=True)

        try:
            # Save pickle file (.pkl)
            with open(file_path, "wb") as f:
                pickle.dump(matrix_data, f)
            print(f"Numerical matrices saved to: {file_path}")

            # Save human-readable text file (.txt)
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write(f"# Numerical Matrices for {os.path.basename(self.mod_file_path)}\n")
                f.write(f"# Generated: {matrix_data['timestamp']}\n\n")
                f.write("State Names Order:\n" + repr(matrix_data['state_names']) + "\n\n")
                f.write("Shock Names Order:\n" + repr(matrix_data['shock_names']) + "\n\n")
                f.write("Parameter Names Order (for reference):\n" + repr(matrix_data['param_names']) + "\n\n")

                # Configure numpy printing options for readability
                np.set_printoptions(linewidth=200, precision=5, suppress=True)
                f.write("--- A Matrix ---\n" + np.array2string(A, max_line_width=200) + "\n\n")
                f.write("--- B Matrix ---\n" + np.array2string(B, max_line_width=200) + "\n\n")
                f.write("--- C Matrix ---\n" + np.array2string(C, max_line_width=200) + "\n\n")
                f.write("--- D Matrix ---\n" + np.array2string(D, max_line_width=200) + "\n\n")
            print(f"Human-readable matrices saved to: {txt_path}")

        except Exception as e:
            print(f"Warning: Could not save matrices file {file_path} or {txt_path}. Error: {e}")

    def save_final_equations_to_txt(self, filename="final_equations.txt"):
        """Saves the final list of equations used for Jacobian calculation."""
        print(f"Saving final equations for Jacobian to: {filename}")
        if not hasattr(self, 'final_equations_for_jacobian') or not self.final_equations_for_jacobian:
            print("Warning: Final equations list is empty. File not saved.")
            return
        try:
            dir_name = os.path.dirname(filename)
            if dir_name: os.makedirs(dir_name, exist_ok=True)

            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"# Final System Equations Used for Jacobian ({len(self.final_equations_for_jacobian)} equations)\n")
                if hasattr(self, 'state_vars_ordered') and self.state_vars_ordered:
                    f.write(f"# State Variables Order ({len(self.state_vars_ordered)}): {[s.name for s in self.state_vars_ordered]}\n\n")
                else:
                    f.write("# State variable order not determined at time of saving.\n\n")

                for i, eq in enumerate(self.final_equations_for_jacobian):
                    f.write(f"Eq {i+1}: {sympy.sstr(eq.lhs, full_prec=False)} = 0\n")
            print(f"Successfully saved final equations to {filename}")
        except Exception as e:
            print(f"Error writing final equations file {filename}: {e}")

# ===========================================
# Function to Solve and Plot from Generated File
# ===========================================
def solve_and_plot_from_generated_function(
    theta,
    generated_module_path,
    shock_index_to_plot,
    horizon=40,
    vars_to_plot=None,
    solver_options=None,
    plot_options=None
):
    """
    Loads a generated Jacobian function, computes matrices for given parameters,
    solves the model, computes IRFs, and plots them.
    """
    print(f"\n--- Solving and Plotting using Generated Function ---")
    print(f"Module path: {generated_module_path}")
    print(f"Parameter vector length: {len(theta)}")
    print(f"Shock index to plot: {shock_index_to_plot}")

    # Ensure paths are absolute for importlib
    abs_module_path = os.path.abspath(generated_module_path)
    if not os.path.isfile(abs_module_path):
        print(f"Error: Generated module file not found at resolved path: {abs_module_path}")
        return None

    if solver_options is None: solver_options = {}
    if plot_options is None: plot_options = {}

    # --- 1. Load the generated module dynamically ---
    module_name = os.path.splitext(os.path.basename(abs_module_path))[0]
    spec = None
    mod_matrices = None
    try:
        # Create a spec from the file location
        spec = importlib.util.spec_from_file_location(module_name, abs_module_path)
        if spec is None:
            print(f"Error: Could not create module spec for {module_name} at {abs_module_path}")
            return None
        # Create a new module based on the spec
        mod_matrices = importlib.util.module_from_spec(spec)
        # Add the module to sys.modules *before* execution if it might have internal imports
        # sys.modules[module_name] = mod_matrices
        # Execute the module code in the new module's namespace
        spec.loader.exec_module(mod_matrices)
        print(f"Successfully loaded module '{module_name}'")
    except FileNotFoundError:
        print(f"Error: File not found during module loading: {abs_module_path}")
        return None
    except Exception as e:
        print(f"Error loading generated module '{module_name}' from {abs_module_path}:")
        print(f"  Type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 2. Get matrices using the loaded function ---
    try:
        # Check if the expected function exists in the loaded module
        if not hasattr(mod_matrices, 'jacobian_matrices'):
            print(f"Error: Function 'jacobian_matrices' not found in loaded module '{module_name}'.")
            return None

        # Call the function from the loaded module
        A, B, C, D, state_names, shock_names = mod_matrices.jacobian_matrices(theta)
        print("Successfully obtained A, B, C, D matrices from generated function.")
        print(f"  State variables ({len(state_names)}): {state_names}")
        print(f"  Shock variables ({len(shock_names)}): {shock_names}")
        print(f"  Matrix shapes: A:{A.shape}, B:{B.shape}, C:{C.shape}, D:{D.shape}")

        # Validate shock index
        if not (0 <= shock_index_to_plot < len(shock_names)):
            print(f"Error: shock_index_to_plot ({shock_index_to_plot}) is out of bounds "
                f"for available shocks ({len(shock_names)}): {shock_names}")
            return None
        current_shock_name = shock_names[shock_index_to_plot]

    except ValueError as ve: # Catch potential ValueError from parameter length check
        print(f"Error calling 'jacobian_matrices': {ve}")
        return None
    except Exception as e:
        print(f"Error calling 'jacobian_matrices' function in module '{module_name}':")
        print(f"  Type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 3. Solve for P using the provided solver ---
    print("\nSolving the Quadratic Matrix Equation for P...")
    try:
        # Pass solver options dictionary using **kwargs
        P_sol, iter_count, residual_ratio = solve_quadratic_matrix_equation(
            A, B, C, **solver_options
        )
        print(f"Solver finished in {iter_count} iterations.")
        print(f"Final residual ratio ||AX^2+BX+C||/||AX^2||: {residual_ratio:.3e}")

        # Check residual tolerance
        tol = solver_options.get('tol', 1e-12) # Get tol from options or use default
        if residual_ratio > tol * 100: # Allow some margin over tol
            print(f"Warning: Solver residual ratio ({residual_ratio:.2e}) is significantly larger than tolerance ({tol:.2e}). Solution might be inaccurate.")
            # Optionally return None here if strict convergence is required
            # return None

        # Basic stability check (optional but recommended)
        try:
            eigenvalues = np.linalg.eigvals(P_sol)
            max_eig = np.max(np.abs(eigenvalues))
            print(f"Maximum eigenvalue magnitude of P: {max_eig:.5f}")
            if max_eig >= 1.0:
                print("Warning: Solution P matrix is UNSTABLE (max |eigenvalue| >= 1).")
            elif max_eig > 0.999:
                print("Warning: Solution P matrix is close to the stability boundary.")
        except np.linalg.LinAlgError:
            print("Warning: Could not compute eigenvalues for stability check.")

    except Exception as e:
        print(f"Error during call to solve_quadratic_matrix_equation:")
        print(f"  Type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 4. Compute Q ---
    print("\nComputing the shock impact matrix Q = -(AP+B)^-1 D...")
    try:
        Q_sol = compute_Q(A, B, D, P_sol)
        if Q_sol is None:
            # compute_Q should print its own error message
            return None # Failed to compute Q
        print(f"Computed Q matrix with shape: {Q_sol.shape}")
    except Exception as e:
        print(f"Error during call to compute_Q:")
        print(f"  Type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 5. Compute IRFs ---
    print(f"\nComputing Impulse Responses for shock '{current_shock_name}' (index {shock_index_to_plot})...")
    try:
        irf_vals = irf_new(P_sol, Q_sol, shock_index=shock_index_to_plot, horizon=horizon)
        print(f"Computed IRFs with shape: {irf_vals.shape}")
    except Exception as e:
        print(f"Error during call to irf_new:")
        print(f"  Type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 6. Plotting ---
    print("\nGenerating IRF plots...")
    try:
        # Determine variables to plot
        state_indices_map = {name: i for i, name in enumerate(state_names)}
        if vars_to_plot is None:
            # Default: plot all non-auxiliary variables
            vars_to_plot_final = [v for v in state_names if not v.lower().startswith('aux_')]
            if not vars_to_plot_final: vars_to_plot_final = state_names # Fallback
            print(f"Plotting default variables: {vars_to_plot_final}")
        else:
            # Validate user-provided list
            valid_vars = [v for v in vars_to_plot if v in state_indices_map]
            invalid_vars = [v for v in vars_to_plot if v not in state_indices_map]
            if invalid_vars:
                print(f"Warning: Requested variables not found in state list and will be skipped: {invalid_vars}")
            if not valid_vars:
                print("Error: No valid variables specified for plotting.")
                return P_sol, Q_sol, irf_vals, state_names, shock_names # Return data without plot
            vars_to_plot_final = valid_vars

        num_plots = len(vars_to_plot_final)
        if num_plots == 0:
            print("No variables selected or valid for plotting.")
            return P_sol, Q_sol, irf_vals, state_names, shock_names

        # Setup plot grid
        cols = int(np.ceil(np.sqrt(num_plots))) # Aim for squarish layout
        rows = (num_plots + cols - 1) // cols

        # Get plot options
        fig_size = plot_options.get('figsize', (4 * cols, 3 * rows))
        suptitle_prefix = plot_options.get('suptitle_prefix', "")
        if suptitle_prefix: suptitle_prefix += ": "

        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False) # Ensure axes is always 2D array
        fig.suptitle(f"{suptitle_prefix}Impulse Responses to Unit '{current_shock_name}' Shock", fontsize=14)
        axes_flat = axes.flatten() # Flatten for easy iteration

        for i, var_name in enumerate(vars_to_plot_final):
            ax = axes_flat[i]
            var_index = state_indices_map[var_name]
            ax.plot(range(horizon), irf_vals[:, var_index], label=var_name)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_title(f"{var_name}")
            ax.set_xlabel("Periods")
            ax.set_ylabel("Response")
            ax.grid(True, alpha=0.4)
            # ax.legend() # Optional: often clutters multi-plots

        # Hide any unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        plt.show()
        print("Plotting complete.")

    except Exception as e:
        print(f"Error during plotting:")
        print(f"  Type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        # Still return results even if plotting failed

    # --- 7. Return results ---
    return P_sol, Q_sol, irf_vals, state_names, shock_names

# ===========================================
# Example Usage Script (Main Block)
# ===========================================
if __name__ == "__main__":

    # --- Configuration ---
    try:
        # Try to get script directory, fall back to current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd() # Likely running in interactive environment
        print(f"Warning: Could not determine script directory, using CWD: {script_dir}")

    # Change directory to script directory for relative paths to work consistently
    try:
        os.chdir(script_dir)
        print(f"Changed working directory to: {script_dir}")
    except Exception as e:
        print(f"Warning: Could not change directory to {script_dir}. Relative paths may fail. Error: {e}")

    mod_file = "qpm_model.dyn" # Your Dynare model file name
    mod_file_base = os.path.splitext(mod_file)[0]
    output_dir_inter = "model_files_intermediate_final"
    output_dir_final = "model_files_numerical_final"

    # --- Define parameters DICT ---
    # Use the exact names as in the 'parameters' block of the .dyn file
    parameter_values_dict = {
        'b1': 0.7,
        'b4': 0.7,
        'a1': 0.5,
        'a2': 0.1,
        'g1': 0.7,
        'g2': 0.3,
        'g3': 0.25,
        'rho_DLA_CPI': 0.75,
        'rho_L_GDP_GAP': 0.75,
        'rho_rs': 0.80, # Corrected based on previous manual definition
        'rho_rs2': 0.01
    }

    # --- Step 1: Instantiate parser and Process Model ---
    parser = None # Initialize parser to None
    try:
        print(f"\n--- Initializing Parser for {mod_file} ---")
        parser = DynareParser(mod_file)

        # Create theta list IN ORDER based on parser's param_names
        print("\n--- Creating Parameter Vector (theta) ---")
        # Check if all required parameters are in the dictionary
        missing_params = [p for p in parser.param_names if p not in parameter_values_dict]
        if missing_params:
            print(f"ERROR: The following parameters are declared in the model but missing from the provided dictionary:")
            for p in missing_params: print(f" - {p}")
            sys.exit(1)
        # Create the list in the correct order
        parameter_theta = [parameter_values_dict[pname] for pname in parser.param_names]
        print(f"Parameter vector created successfully (length {len(parameter_theta)}).")
        print(f"Order: {parser.param_names}")

        print("\n--- Processing Model with DynareParser ---")
        # Call process_model with the ORDERED list/vector
        parser_result = parser.process_model(
            param_dict_values_or_list=parameter_theta, # Pass the ordered vector
            output_dir_intermediate=output_dir_inter,
            output_dir_final=output_dir_final,
            generate_function=True # Ensure the function file is generated
        )

        if not parser_result:
            print("\nERROR: Model processing failed. Check parser output for details.")
            sys.exit(1)
        else:
            # Unpack results from parser if needed (e.g., for direct comparison)
            A_direct, B_direct, C_direct, D_direct, state_names_direct, shock_names_direct = parser_result
            print("\nModel processed directly by parser successfully.")
            print(f" Direct State Names: {state_names_direct}")
            print(f" Direct Shock Names: {shock_names_direct}")

    except FileNotFoundError as fnf_e:
        print(f"\nERROR: Model file '{mod_file}' not found. Make sure it's in the correct directory.")
        print(f" Details: {fnf_e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR during parser initialization or processing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Step 2: Define path to the generated function file ---
    # Construct the path based on output_dir_final and mod_file_base
    generated_func_path = os.path.join(output_dir_final, f"{mod_file_base}_jacobian_matrices.py")
    print(f"\nPath to generated function: {generated_func_path}")

    # --- Step 3: Specify analysis options ---
    # Choose shock index (based on 'varexo' order in .dyn file)
    shock_name_to_plot = "SHK_RS" # Name of the shock
    try:
        # Find the index dynamically from parser results
        shock_to_analyze = parser.shock_names.index(shock_name_to_plot)
        print(f"Analyzing shock: '{shock_name_to_plot}' (Index: {shock_to_analyze})")
    except (ValueError, AttributeError):
        print(f"Error: Shock '{shock_name_to_plot}' not found in parser's shock list: {parser.shock_names if parser else 'Parser not initialized'}. Defaulting to index 0.")
        shock_to_analyze = 0 # Fallback to the first shock

    periods_horizon = 40
    # Select variables to plot (use names from parser.state_vars_ordered)
    variables_for_plot = ["L_GDP_GAP", "DLA_CPI", "RS", "RES_RS"] # Example subset
    # Check if these variables exist in the final state vector
    if parser:
        missing_plot_vars = [v for v in variables_for_plot if v not in state_names_direct]
        if missing_plot_vars:
            print(f"Warning: Requested plot variables not in final state list: {missing_plot_vars}. They will be skipped.")
            variables_for_plot = [v for v in variables_for_plot if v in state_names_direct]

    # --- Step 4: Call the analysis function ---
    print("\n--- Calling Analysis Function with Generated Module ---")
    analysis_results = solve_and_plot_from_generated_function(
        theta=parameter_theta,                 # Pass the ORDERED parameter vector
        generated_module_path=generated_func_path, # Pass the constructed path
        shock_index_to_plot=shock_to_analyze,
        horizon=periods_horizon,
        vars_to_plot=variables_for_plot,
        solver_options={'tol': 1e-12, 'verbose': False, 'max_iter': 150}, # Example solver options
        plot_options={'figsize': (10, 6), 'suptitle_prefix': f"{mod_file_base}"} # Example plot options
    )

    # --- Step 5: Check and Summarize Analysis Results ---
    if analysis_results:
        P_solution, Q_solution, irf_data, states, shocks = analysis_results
        print("\n" + "="*30 + " Analysis Summary " + "="*30)
        print(f"Model solved and plotted successfully using generated function.")
        print(f" State vector ({len(states)}): {states}")
        print(f" Shock vector ({len(shocks)}): {shocks}")
        print(f" Policy matrix P shape: {P_solution.shape}")
        print(f" Shock matrix Q shape: {Q_solution.shape}")
        print(f" IRF data shape: {irf_data.shape}")
        # Optional: Compare results from direct parsing vs generated function
        # if 'A_direct' in locals(): # Check if direct results exist
        #     try:
        #          # Need to re-run generated function to get A_f, B_f, etc. for comparison here
        #          # Or load the generated module again (less ideal)
        #          # This comparison was already done inside the parser's __main__ block test
        #          print("\n--- Comparison (Direct vs Generated - check parser output for detailed test) ---")
        #          # assert np.allclose(A_direct, A_f)... etc.
        #     except Exception as comp_e:
        #          print(f"Could not perform comparison: {comp_e}")
    else:
        print("\n" + "="*30 + " Analysis Failed " + "="*30)
        print("Check the output messages for errors during module loading, solving, or plotting.")

    print("\n" + "="*70)
    print("Script finished.")
    print("="*70)
