# --- In main_script_using_wrapper.py or a new test script ---
import os
import jax
import jax.numpy as jnp
import numpy as onp # Use onp for comparison baseline
import matplotlib.pyplot as plt
from jax import random

# Ensure JAX is configured (can also be done inside the wrapper)
# os.environ['JAX_PLATFORMS'] = 'cpu' # Or 'gpu'
jax.config.update("jax_enable_x64", True)
print(f"main_script: JAX float64 enabled: {jax.config.jax_enable_x64}")

# --- Import the Wrapper AND the Solvers ---
from dynare_model_wrapper import DynareModel
import Dynare_parser_sda_solver as dp # Original parser/solver module
from Dynare_parser_sda_solver import solve_quadratic_matrix_equation_jax # Import JAX solver

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # --- Configuration ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
        print(f"Using model file: {mod_file_path}")

        # --- [1] Initialize the Model (Parses Once) ---
        model = DynareModel(mod_file_path) # Uses default dtype based on jax config
        print(f"Model initialized with dtype: {model.dtype}")

        # --- [2] Get Default Parameter Set ---
        param_values = model.default_param_assignments.copy()
        # !! Crucially, ensure ALL necessary sigma_ parameters have default values !!
        # Check and add defaults if missing (as done in original main)
        for shk in model.aug_shocks_structure:
            pname = f"sigma_{shk}"
            if pname not in param_values or param_values[pname] is None or onp.isnan(param_values[pname]):
                print(f"Note: Param '{pname}' for shock '{shk}' missing/invalid in defaults. Setting to 1.0 for testing.")
                param_values[pname] = 1.0 # Assign a default for testing

        print(f"\nUsing Default Parameter Dictionary for Testing:\n{param_values}")

        # --- [3] Manually Evaluate Stationary Matrices using Lambdas ---
        print("\nEvaluating stationary matrices A, B, C using lambdas...")
        # Prepare ordered args (as floats for numpy lambdas)
        stat_param_names_ordered = model.param_names_stat_combined
        stat_args = [float(param_values[p]) for p in stat_param_names_ordered]

        # Lambdas return NumPy arrays
        A_num_np = onp.array(model.func_A(*stat_args))
        B_num_np = onp.array(model.func_B(*stat_args))
        C_num_np = onp.array(model.func_C(*stat_args))
        print(" A_num_np shape:", A_num_np.shape)
        print(" B_num_np shape:", B_num_np.shape)
        print(" C_num_np shape:", C_num_np.shape)

        # --- [4] Solve using ORIGINAL NumPy/SciPy Solver ---
        print("\nSolving with NumPy SDA solver...")
        P_sol_np, iter_np, resid_np = dp.solve_quadratic_matrix_equation(
            A_num_np, B_num_np, C_num_np, tol=1e-12
        )
        if P_sol_np is not None:
            print(f" NumPy Solver: Converged in {iter_np} iterations, Residual Ratio: {resid_np:.2e}")
        else:
            print(" NumPy Solver: Failed.")

        # --- [5] Solve using NEW JAX Solver ---
        print("\nSolving with JAX SDA solver...")
        # Pass NumPy arrays, function converts to JAX internally with specified dtype
        P_sol_jax, iter_jax, resid_jax, success_jax = solve_quadratic_matrix_equation_jax(
            A_num_np, B_num_np, C_num_np, tol=1e-12, dtype=model.dtype
        )
        # Convert JAX result back to NumPy for comparison/printing
        P_sol_jax_np = onp.array(P_sol_jax)

        if success_jax:
            print(f" JAX Solver: Converged in {iter_jax} iterations, Final Diff: {resid_jax:.2e}")
            # Optional: Calculate residual for JAX solution using NumPy ops for direct comparison
            residual_jax_calc = A_num_np @ (P_sol_jax_np @ P_sol_jax_np) + B_num_np @ P_sol_jax_np + C_num_np
            resid_jax_ratio = onp.linalg.norm(residual_jax_calc, 'fro') / (onp.linalg.norm(A_num_np @ P_sol_jax_np @ P_sol_jax_np, 'fro') + onp.linalg.norm(B_num_np @ P_sol_jax_np, 'fro') + onp.linalg.norm(C_num_np, 'fro') + 1e-15)
            print(f" JAX Solver: Calculated Residual Ratio: {resid_jax_ratio:.2e}")
        else:
            print(f" JAX Solver: Failed (Success Flag False). Iterations: {iter_jax}, Final Diff: {resid_jax:.2e}")

        # --- [6] Compare Solutions ---
        print("\nComparing NumPy vs JAX solutions...")
        if P_sol_np is not None and success_jax and jnp.all(jnp.isfinite(P_sol_jax)):
            # Use a tolerance appropriate for the dtype
            comparison_tol = 1e-7 if model.dtype == jnp.float32 else 1e-10
            are_close = onp.allclose(P_sol_np, P_sol_jax_np, rtol=comparison_tol, atol=comparison_tol)
            max_abs_diff = onp.max(onp.abs(P_sol_np - P_sol_jax_np))
            print(f" Solutions are close (tol={comparison_tol}): {are_close}")
            print(f" Max absolute difference: {max_abs_diff:.2e}")
            if not are_close:
                 print(" NumPy P:\n", P_sol_np)
                 print(" JAX P:\n", P_sol_jax_np)
                 print(" Difference:\n", P_sol_np - P_sol_jax_np)

        elif P_sol_np is None and not success_jax:
            print(" Both solvers failed, consistent.")
        else:
            print(" Solvers produced different success/failure outcomes.")

        # --- Optional: Continue with rest of main script IF comparison is good ---
        # if are_close:
        #    print("\nSolvers match, proceeding with full analysis using DynareModel wrapper...")
        #    # ... rest of your original main_script code using model.solve(), etc. ...


    except FileNotFoundError as e:
         print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    except RuntimeError as e:
         print(f"\nA runtime error occurred: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()