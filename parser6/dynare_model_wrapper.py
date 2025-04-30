# --- START OF FILE dynare_model_wrapper.py ---

import os
import jax
import jax.numpy as jnp
import numpy as onp # For checking instance types, default values
from jax.typing import ArrayLike
from jax import random
from typing import Dict, List, Tuple, Optional, Union, Any

# --- Import your custom modules ---
# Assume the parser/solver is in dynare_parser_spd7.py
import Dynare_parser_sda_solver as dp
# Assume the Kalman filter is in Kalman_filter_jax.py
from Kalman_filter_jax import KalmanFilter, simulate_state_space

# --- JAX Configuration ---
# Ensure x64 is enabled if needed (can be set externally too)
try:
    jax.config.update("jax_enable_x64", True)
    print(f"dynare_model_wrapper: JAX float64 enabled: {jax.config.jax_enable_x64}")
except Exception as e:
    print(f"dynare_model_wrapper: Warning - Could not set jax_enable_x64: {e}")


class DynareModel:
    """
    A wrapper class to parse, solve, and analyze Dynare models using JAX.

    This class parses the model definition once upon initialization and provides
    methods to solve the model, compute IRFs, simulate data, and run Kalman
    filtering/smoothing for different parameter values without re-parsing.
    """

    def __init__(self, dynare_file_path: str):
        """
        Initializes the DynareModel by parsing the .dyn file.

        Args:
            dynare_file_path: Path to the .dyn model file.

        Raises:
            FileNotFoundError: If the dynare_file_path does not exist.
            ValueError: If parsing fails or the model structure is inconsistent.
        """
        if not os.path.exists(dynare_file_path):
            raise FileNotFoundError(f"Model file not found at: {dynare_file_path}")

        self.dynare_file_path = dynare_file_path
        print(f"--- Initializing DynareModel from: {dynare_file_path} ---")

        with open(self.dynare_file_path, 'r') as f:
            model_def = f.read()

        # --- [1] PARSING AND LAMBDIFYING (Done Once) ---
        print("   Parsing stationary model components...")
        try:
            (self.func_A, self.func_B, self.func_C, self.func_D,
             self.ordered_stat_vars, self.stat_shocks, self.param_names_stat,
             self.param_assignments_stat, _, _) = dp.parse_lambdify_and_order_model(model_def)
        except Exception as e:
            print(f"Error parsing stationary model: {e}")
            raise ValueError(f"Failed to parse stationary model from {dynare_file_path}") from e

        print("   Parsing trend/observation components...")
        try:
            self.trend_vars, self.trend_shocks = dp.extract_trend_declarations(model_def)
            trend_equations = dp.extract_trend_equations(model_def)
            self.obs_vars = dp.extract_observation_declarations(model_def)
            measurement_equations = dp.extract_measurement_equations(model_def)
            self.trend_stderr_params = dp.extract_trend_shock_stderrs(model_def) # Dict {sigma_NAME: value}
        except Exception as e:
            print(f"Error parsing trend/observation declarations: {e}")
            raise ValueError(f"Failed to parse trend/observation components from {dynare_file_path}") from e

        # Combine parameters
        self.all_param_names = list(dict.fromkeys(
            self.param_names_stat + list(self.trend_stderr_params.keys())
        ).keys())
        self.default_param_assignments = self.param_assignments_stat.copy()
        self.default_param_assignments.update(self.trend_stderr_params)

        # --- Store names for combined shocks structure ---
        self.aug_shocks_structure = self.stat_shocks + self.trend_shocks

        print("   Building symbolic trend and observation matrices...")
        try:
            # Build trend matrices (lambdified functions)
            (self.func_P_trends, self.func_Q_trends,
             self.ordered_trend_state_vars, self.contemp_trend_defs) = dp.build_trend_matrices(
                trend_equations, self.trend_vars, self.trend_shocks,
                self.all_param_names, self.default_param_assignments # Pass defaults for structure building
            )

            # Build observation matrix (lambdified function)
            (self.func_Omega, self.ordered_obs_vars) = dp.build_observation_matrix(
                measurement_equations, self.obs_vars, self.ordered_stat_vars,
                self.ordered_trend_state_vars, self.contemp_trend_defs,
                self.all_param_names, self.default_param_assignments # Pass defaults for structure building
             )
        except Exception as e:
            print(f"Error building symbolic trend/observation matrices: {e}")
            raise ValueError("Failed to build symbolic trend/observation matrices") from e

        # --- Store names for augmented state structure ---
        self.aug_state_vars_structure = self.ordered_stat_vars + self.ordered_trend_state_vars

        self.n_state_aug = len(self.aug_state_vars_structure)
        self.n_shock_aug = len(self.aug_shocks_structure)
        self.n_obs = len(self.ordered_obs_vars)

        print("--- DynareModel Initialization Complete ---")
        print(f"   Augmented State Variables ({self.n_state_aug}): {self.aug_state_vars_structure}")
        print(f"   Augmented Shocks ({self.n_shock_aug}): {self.aug_shocks_structure}")
        print(f"   Observable Variables ({self.n_obs}): {self.ordered_obs_vars}")
        print(f"   All Parameters ({len(self.all_param_names)}): {self.all_param_names}")
        print(f"   Default parameters parsed: {self.default_param_assignments}")

# --- END OF FILE dynare_model_wrapper.py ---