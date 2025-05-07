# In Kalman_filter_jax.py

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp # Using onp for clarity with NumPy operations
from typing import Tuple, Optional, Union, Sequence, Dict, Any

_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps

class KalmanFilter:
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype) # This is C_full
        self.H = jnp.asarray(H, dtype=desired_dtype) # This is H_full
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P, dtype=desired_dtype)

        n_state = self.T.shape[0]
        n_obs_full = self.C.shape[0] # Full number of observables
        n_shocks = self.R.shape[1] if self.R.ndim == 2 and self.R.shape[1] > 0 else 0

        # Validations use n_state and n_obs_full
        if self.T.shape != (n_state, n_state): raise ValueError(f"T shape mismatch")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks): raise ValueError(f"R shape mismatch")
        elif n_shocks == 0 and self.R.size != 0 : self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        if self.C.shape != (n_obs_full, n_state): raise ValueError(f"C shape mismatch")
        if self.H.shape != (n_obs_full, n_obs_full): raise ValueError(f"H shape mismatch")
        if self.init_x.shape != (n_state,): raise ValueError(f"init_x shape mismatch")
        if self.init_P.shape != (n_state, n_state): raise ValueError(f"init_P shape mismatch")

        self.n_state = n_state
        self.n_obs_full = n_obs_full # Store full n_obs for reference
        self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)

        if self.n_shocks > 0: self.state_cov = self.R @ self.R.T
        else: self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)
        
        # Cholesky for observation noise simulation (uses full H)
        try:
            H_reg_chol = self.H + _MACHINE_EPSILON * 1e-2 * jnp.eye(self.n_obs_full, dtype=desired_dtype)
            self.L_H_full = jnp.linalg.cholesky(H_reg_chol) # For simulating full observation noise vector
            self.simulate_obs_noise_internal = self._simulate_obs_noise_chol_internal
        except Exception:
            self.H_stable_full = self.H + _MACHINE_EPSILON * jnp.eye(self.n_obs_full, dtype=desired_dtype)
            self.simulate_obs_noise_internal = self._simulate_obs_noise_mvn_internal


    def _simulate_obs_noise_chol_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        # Simulates noise for all n_obs_full components
        z_eta = random.normal(key, tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)
        return z_eta @ self.L_H_full.T

    def _simulate_obs_noise_mvn_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try:
            return random.multivariate_normal(
                key, jnp.zeros((self.n_obs_full,), dtype=self.H.dtype),
                self.H_stable_full, shape=mvn_shape, dtype=self.H.dtype
            )
        except Exception:
            return jnp.zeros(tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)


    def filter(self,
               ys: ArrayLike, # Full observations array (T, n_obs_full)
               # Static NaN information is NOW PASSED IN:
               static_valid_obs_idx: jax.Array, # Precomputed jnp.array of valid indices
               static_n_obs_actual: int,        # Precomputed Python int (length of static_valid_obs_idx)
               static_C_obs: jax.Array,         # Precomputed C_obs (n_obs_actual, n_state)
               static_H_obs: jax.Array,         # Precomputed H_obs (n_obs_actual, n_obs_actual)
               static_I_obs: jax.Array          # Precomputed I_obs (n_obs_actual, n_obs_actual)
              ) -> Dict[str, jax.Array]:
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype) # C is C_full
        T_mat, I_s = self.T, self.I_s
        state_cov = self.state_cov
        eps_reg = _MACHINE_EPSILON * 10
        MAX_STATE_VALUE = 1e6
        T_steps = ys_arr.shape[0]

        if T_steps == 0:
            return { # Shapes now correctly use static_n_obs_actual
                'x_pred': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_pred': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'x_filt': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_filt': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'innovations': jnp.empty((0, static_n_obs_actual), dtype=self.C.dtype),
                'innovation_cov': jnp.empty((0, static_n_obs_actual, static_n_obs_actual), dtype=self.C.dtype),
                'log_likelihood_contributions': jnp.empty((0,), dtype=I_s.dtype)
            }

        # Kalman Filter Step Function (uses passed-in static matrices)
        def step_static_nan(carry, y_t_full_slice): # y_t_full_slice is a row from ys_arr
            x_prev_filt, P_prev_filt = carry
            x_pred_t = T_mat @ x_prev_filt
            x_pred_t = jnp.clip(x_pred_t, -MAX_STATE_VALUE, MAX_STATE_VALUE)
            P_prev_filt_sym = (P_prev_filt + P_prev_filt.T) / 2.0
            P_prev_filt_reg = P_prev_filt_sym + eps_reg * I_s
            P_pred_t = T_mat @ P_prev_filt_reg @ T_mat.T + state_cov
            P_pred_t = (P_pred_t + P_pred_t.T) / 2.0
            P_pred_t = P_pred_t + eps_reg * I_s

            y_obs_t = jnp.take(y_t_full_slice, static_valid_obs_idx, axis=0) if static_n_obs_actual > 0 else jnp.empty((0,), dtype=y_t_full_slice.dtype)
            
            # Use the passed static_C_obs, static_H_obs, static_I_obs
            y_pred_obs = static_C_obs @ x_pred_t
            v_obs = y_obs_t - y_pred_obs
            PCt_obs = P_pred_t @ static_C_obs.T # C_obs is (n_actual, n_state)
            S_obs = static_C_obs @ PCt_obs + static_H_obs
            S_obs_reg = S_obs + eps_reg * 10.0 * static_I_obs

            K = jnp.zeros((self.n_state, static_n_obs_actual), dtype=x_pred_t.dtype)
            if static_n_obs_actual > 0:
                try:
                    L_S_obs = jnp.linalg.cholesky(S_obs_reg)
                    K_T_temp = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs.T, lower=True, trans='N')
                    K = jax.scipy.linalg.solve_triangular(L_S_obs, K_T_temp, lower=True, trans='T').T
                except Exception:
                    try: K = jax.scipy.linalg.solve(S_obs_reg, PCt_obs.T, assume_a='pos').T
                    except Exception:
                        try: K = PCt_obs @ jnp.linalg.pinv(S_obs_reg)
                        except Exception: K = jnp.zeros((self.n_state, static_n_obs_actual), dtype=x_pred_t.dtype)
                K = jnp.clip(K, -1e3, 1e3)
            
            x_filt_t = x_pred_t
            P_filt_t = P_pred_t
            if static_n_obs_actual > 0:
                x_update = K @ v_obs
                x_filt_t = x_pred_t + x_update
                x_filt_t = jnp.clip(x_filt_t, -MAX_STATE_VALUE, MAX_STATE_VALUE)
                IKC_obs = I_s - K @ static_C_obs
                P_filt_t = IKC_obs @ P_pred_t @ IKC_obs.T + K @ static_H_obs @ K.T
                P_filt_t = (P_filt_t + P_filt_t.T) / 2.0
                P_filt_t = P_filt_t + eps_reg * I_s

            ll_t = 0.0
            if static_n_obs_actual > 0:
                log_pi_term = jnp.log(2 * jnp.pi) * static_n_obs_actual
                sign, log_det_S_obs = jnp.linalg.slogdet(S_obs_reg)
                mahalanobis_dist_obs = 0.0
                try:
                    L_S_obs_ll = jnp.linalg.cholesky(S_obs_reg)
                    z = jax.scipy.linalg.solve_triangular(L_S_obs_ll, v_obs, lower=True)
                    mahalanobis_dist_obs = jnp.sum(z**2)
                except Exception:
                    try:
                        solved_term_obs = jax.scipy.linalg.solve(S_obs_reg, v_obs, assume_a='pos')
                        mahalanobis_dist_obs = v_obs @ solved_term_obs
                    except Exception: mahalanobis_dist_obs = 1e6
                
                ll_t = -0.5 * (log_pi_term + log_det_S_obs + mahalanobis_dist_obs)
                ll_t = jnp.where(sign > 0, ll_t, -1e6)
            ll_t = jnp.where(jnp.isfinite(ll_t), ll_t, -1e6)

            outputs = {
                'x_pred': x_pred_t, 'P_pred': P_pred_t, 'x_filt': x_filt_t, 'P_filt': P_filt_t,
                'innovations': v_obs, 'innovation_cov': S_obs_reg,
                'log_likelihood_contributions': ll_t
            }
            for key_out, val_out in outputs.items():
                if key_out == 'log_likelihood_contributions': outputs[key_out] = jnp.where(jnp.isfinite(val_out), val_out, -1e6)
                elif hasattr(val_out, 'size') and val_out.size > 0: outputs[key_out] = jnp.where(jnp.isfinite(val_out), val_out, jnp.zeros_like(val_out))
            x_filt_t_safe = outputs['x_filt']; P_filt_t_safe = outputs['P_filt']
            P_filt_t_safe = (P_filt_t_safe + P_filt_t_safe.T) / 2.0
            return (x_filt_t_safe, P_filt_t_safe), outputs

        init_carry = (self.init_x, self.init_P)
        (_, _), scan_outputs = lax.scan(step_static_nan, init_carry, ys_arr) # ys_arr is (T, n_obs_full)
        
        for key_final, val_final_arr in scan_outputs.items():
            if key_final == 'log_likelihood_contributions': scan_outputs[key_final] = jnp.where(jnp.isfinite(val_final_arr), val_final_arr, jnp.full_like(val_final_arr, -1e6))
            elif val_final_arr.size > 0: scan_outputs[key_final] = jnp.where(jnp.isfinite(val_final_arr), val_final_arr, jnp.zeros_like(val_final_arr))
        return scan_outputs

    def log_likelihood(self,
                       ys: ArrayLike,
                       # Static NaN information MUST be passed:
                       static_valid_obs_idx: jax.Array,
                       static_n_obs_actual: int,
                       static_C_obs: jax.Array,
                       static_H_obs: jax.Array,
                       static_I_obs: jax.Array
                      ) -> jax.Array:
        filter_results = self.filter(ys, static_valid_obs_idx, static_n_obs_actual,
                                     static_C_obs, static_H_obs, static_I_obs)
        total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])
        return total_log_likelihood

    # Important: The `smooth` method needs to be updated to correctly pass static NaN info
    # to its internal call to `self.filter` if `filter_results` is None.
    # And `simulation_smoother` similarly.

    def smooth(self,
               ys: ArrayLike,
               filter_results: Optional[Dict] = None,
               # Static NaN info, only needed if filter_results is None
               static_valid_obs_idx: Optional[jax.Array] = None,
               static_n_obs_actual: Optional[int] = None,
               static_C_obs_for_filter: Optional[jax.Array] = None, # C_obs based on KF's C
               static_H_obs_for_filter: Optional[jax.Array] = None, # H_obs based on KF's H
               static_I_obs_for_filter: Optional[jax.Array] = None
              ) -> Tuple[jax.Array, jax.Array]:
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        if filter_results is None:
            # This branch now requires the static NaN info to be passed if filter_results is None
            if static_valid_obs_idx is None or \
               static_n_obs_actual is None or \
               static_C_obs_for_filter is None or \
               static_H_obs_for_filter is None or \
               static_I_obs_for_filter is None:
                raise ValueError("Static NaN info must be provided to smooth() if filter_results is None.")
            filter_outs_dict = self.filter(ys_arr, static_valid_obs_idx, static_n_obs_actual,
                                           static_C_obs_for_filter, static_H_obs_for_filter, static_I_obs_for_filter)
        else:
            filter_outs_dict = filter_results
        # ... (rest of the smoother logic from your file, it uses filter_outs_dict)
        x_pred = filter_outs_dict['x_pred']; P_pred = filter_outs_dict['P_pred']
        x_filt = filter_outs_dict['x_filt']; P_filt = filter_outs_dict['P_filt']
        T_mat = self.T; N = x_filt.shape[0]
        if N == 0: return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)
        x_s_next = x_filt[-1]; P_s_next = P_filt[-1]
        P_pred_for_scan = P_pred[1:N]; P_filt_for_scan = P_filt[0:N-1]
        x_pred_for_scan = x_pred[1:N]; x_filt_for_scan = x_filt[0:N-1]
        scan_inputs = (P_pred_for_scan[::-1], P_filt_for_scan[::-1], x_pred_for_scan[::-1], x_filt_for_scan[::-1])
        def backward_step(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            Pp_next_reg = Pp_next_t + _MACHINE_EPSILON * 1e-4 * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)
            try:
                Jt_transpose = jax.scipy.linalg.solve(Pp_next_reg, (T_mat @ Pf_t).T, assume_a='auto')
                Jt = Jt_transpose.T
            except Exception: Jt = Pf_t @ T_mat.T @ jnp.linalg.pinv(Pp_next_reg)
            x_diff = x_s_next_t - xp_next_t; x_s_t = xf_t + Jt @ x_diff
            P_diff = P_s_next_t - Pp_next_t; P_s_t = Pf_t + Jt @ P_diff @ Jt.T
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            return (x_s_t, P_s_t), (x_s_t, P_s_t)
        init_carry_smooth = (x_s_next, P_s_next)
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step, init_carry_smooth, scan_inputs)
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt[N-1:N]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt[N-1:N]], axis=0)
        return x_smooth, P_smooth

    # _simulation_smoother_single_draw and simulation_smoother would need similar careful
    # handling of static NaN info if they call self.filter internally without providing
    # pre-computed filter_results. For now, I'll assume they are mostly called with
    # filter_results derived from an initial filter run where static info was handled.
    # The key is that any direct call to self.filter() must now include the static NaN args.

    def _simulation_smoother_single_draw(self,
                                       ys: jax.Array,
                                       key: jax.random.PRNGKey,
                                       filter_results_orig: Optional[Dict] = None,
                                       # Must pass static NaN info for original ys if filter_results_orig is None
                                       static_valid_obs_idx_orig: Optional[jax.Array] = None,
                                       static_n_obs_actual_orig: Optional[int] = None,
                                       static_C_obs_orig: Optional[jax.Array] = None,
                                       static_H_obs_orig: Optional[jax.Array] = None,
                                       static_I_obs_orig: Optional[jax.Array] = None
                                       ) -> jax.Array:
        Tsteps = ys.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        n_s, n_eps_shocks = self.n_state, self.n_shocks

        if filter_results_orig is None or 'x_filt' not in filter_results_orig:
            if static_valid_obs_idx_orig is None: # etc.
                raise ValueError("Static NaN info for original ys must be provided to _simulation_smoother_single_draw if filter_results_orig is None.")
            filter_results_for_rts = self.filter(ys, static_valid_obs_idx_orig, static_n_obs_actual_orig,
                                                 static_C_obs_orig, static_H_obs_orig, static_I_obs_orig)
            x_smooth_rts, _ = self.smooth(ys, filter_results=filter_results_for_rts) # smooth uses its own static info for this filter call
        else:
            x_smooth_rts, _ = self.smooth(ys, filter_results=filter_results_orig)


        key_init, key_eps, key_eta = random.split(key, 3)
        try:
            init_P_reg = self.init_P + _MACHINE_EPSILON * jnp.eye(n_s, dtype=self.init_P.dtype)
            L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
            x0_star = self.init_x + L0 @ z0
        except Exception: x0_star = self.init_x

        eps_star_sim = random.normal(key_eps, (Tsteps, n_eps_shocks), dtype=self.R.dtype) if n_eps_shocks > 0 else jnp.zeros((Tsteps, 0), dtype=self.R.dtype)
        def state_sim_step(x_prev_star, eps_t_star_arg):
            shock_term = self.R @ eps_t_star_arg if n_eps_shocks > 0 else jnp.zeros(self.n_state, dtype=x_prev_star.dtype)
            x_curr_star = self.T @ x_prev_star + shock_term
            return x_curr_star, x_curr_star
        _, x_star_path = lax.scan(state_sim_step, x0_star, eps_star_sim)

        # For y_star_sim, determine its OWN static NaN pattern based on original `ys`
        first_obs_slice_sim_pattern = onp.asarray(ys[0])
        _valid_obs_idx_sim_star_py = onp.where(~onp.isnan(first_obs_slice_sim_pattern))[0]
        static_valid_obs_idx_sim_star_jnp = jnp.array(_valid_obs_idx_sim_star_py, dtype=jnp.int32)
        static_n_obs_actual_sim_star = len(_valid_obs_idx_sim_star_py)

        y_star_sim = jnp.full((Tsteps, self.n_obs_full), jnp.nan, dtype=x_star_path.dtype) # Use n_obs_full
        static_C_obs_sim_star, static_H_obs_sim_star, static_I_obs_sim_star = None, None, None

        if static_n_obs_actual_sim_star > 0:
            # C for y_star_sim is self.C (the KF's C matrix)
            static_C_obs_sim_star = self.C[static_valid_obs_idx_sim_star_jnp, :]
            # H for y_star_sim is self.H (the KF's H matrix)
            static_H_obs_sim_star = self.H[jnp.ix_(static_valid_obs_idx_sim_star_jnp, static_valid_obs_idx_sim_star_jnp)]
            static_I_obs_sim_star = jnp.eye(static_n_obs_actual_sim_star, dtype=_MACHINE_EPSILON)

            eta_star_full_sim = self.simulate_obs_noise_internal(key_eta, (Tsteps,)) # Uses self.L_H_full or self.H_stable_full
            eta_star_valid_sim = eta_star_full_sim[:, static_valid_obs_idx_sim_star_jnp] if self.n_obs_full > 0 else jnp.empty((Tsteps, 0))
            
            y_star_valid_only_sim = (x_star_path @ static_C_obs_sim_star.T) + eta_star_valid_sim # Use C_obs_sim_star
            y_star_sim = y_star_sim.at[:, static_valid_obs_idx_sim_star_jnp].set(y_star_valid_only_sim)
        else: # All NaNs for y_star_sim
            static_C_obs_sim_star = jnp.empty((0, self.n_state), dtype=self.C.dtype)
            static_H_obs_sim_star = jnp.empty((0,0), dtype=self.H.dtype)
            static_I_obs_sim_star = jnp.empty((0,0), dtype=_MACHINE_EPSILON)


        filter_results_star = self.filter(y_star_sim, static_valid_obs_idx_sim_star_jnp, static_n_obs_actual_sim_star,
                                           static_C_obs_sim_star, static_H_obs_sim_star, static_I_obs_sim_star)
        x_smooth_star, _ = self.smooth(y_star_sim, filter_results=filter_results_star) # This smooth call uses pre-filtered

        x_draw = x_star_path + (x_smooth_rts - x_smooth_star)
        return x_draw

    def simulation_smoother(self, ys: ArrayLike, key: jax.random.PRNGKey, num_draws: int = 1,
                            filter_results_initial_ys: Optional[Dict] = None, # Results from filtering original ys
                            # Static NaN info for the original `ys` data
                            static_info_for_original_ys: Optional[Dict] = None
                           ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        Tsteps = ys_arr.shape[0]
        empty_state_shape = (0, self.n_state)
        if Tsteps == 0: # ... (empty return as before) ...
            if num_draws == 1: return jnp.empty(empty_state_shape, dtype=self.init_x.dtype)
            else: return jnp.empty(empty_state_shape, dtype=self.init_x.dtype), \
                         jnp.empty(empty_state_shape, dtype=self.init_x.dtype), \
                         jnp.empty((num_draws,) + empty_state_shape, dtype=self.init_x.dtype)

        # Handle filter_results_initial_ys and static_info_for_original_ys
        _filter_results_orig_for_draw = filter_results_initial_ys
        _static_valid_idx_orig = None
        _static_n_actual_orig = None
        _static_C_obs_orig = None
        _static_H_obs_orig = None
        _static_I_obs_orig = None

        if filter_results_initial_ys is None:
            if static_info_for_original_ys is None:
                 # Compute static info for original ys if not provided
                first_obs_slice_concrete = onp.asarray(ys_arr[0])
                _valid_obs_idx_py_ll = onp.where(~onp.isnan(first_obs_slice_concrete))[0]
                _static_valid_idx_orig = jnp.array(_valid_obs_idx_py_ll, dtype=jnp.int32)
                _static_n_actual_orig = len(_valid_obs_idx_py_ll)
                if _static_n_actual_orig == self.n_obs_full:
                    _static_C_obs_orig = self.C; _static_H_obs_orig = self.H; _static_I_obs_orig = jnp.eye(self.n_obs_full, dtype=self.T.dtype)
                elif _static_n_actual_orig > 0:
                    _static_C_obs_orig = self.C[_static_valid_idx_orig, :]; _static_H_obs_orig = self.H[jnp.ix_(_static_valid_idx_orig, _static_valid_idx_orig)]; _static_I_obs_orig = jnp.eye(_static_n_actual_orig, dtype=self.T.dtype)
                else:
                    _static_C_obs_orig = jnp.empty((0, self.n_state), dtype=self.C.dtype); _static_H_obs_orig = jnp.empty((0,0), dtype=self.H.dtype); _static_I_obs_orig = jnp.empty((0,0), dtype=self.T.dtype)
            else: # Use provided static info
                _static_valid_idx_orig = static_info_for_original_ys["static_valid_obs_idx"]
                _static_n_actual_orig = static_info_for_original_ys["static_n_obs_actual"]
                _static_C_obs_orig = static_info_for_original_ys["static_C_obs"]
                _static_H_obs_orig = static_info_for_original_ys["static_H_obs"]
                _static_I_obs_orig = static_info_for_original_ys["static_I_obs"]
            
            _filter_results_orig_for_draw = self.filter(ys_arr, _static_valid_idx_orig, _static_n_actual_orig,
                                                       _static_C_obs_orig, _static_H_obs_orig, _static_I_obs_orig)
        
        if num_draws == 1:
            return self._simulation_smoother_single_draw(ys_arr, key, 
                                                       filter_results_orig=_filter_results_orig_for_draw,
                                                       static_valid_obs_idx_orig=_static_valid_idx_orig,
                                                       static_n_obs_actual_orig=_static_n_actual_orig,
                                                       static_C_obs_orig=_static_C_obs_orig,
                                                       static_H_obs_orig=_static_H_obs_orig,
                                                       static_I_obs_orig=_static_I_obs_orig)
        else:
            keys = random.split(key, num_draws)
            vmapped_smoother = vmap(
                lambda k_single: self._simulation_smoother_single_draw(
                    ys_arr, k_single, filter_results_orig=_filter_results_orig_for_draw,
                    static_valid_obs_idx_orig=_static_valid_idx_orig, # Pass these through vmap
                    static_n_obs_actual_orig=_static_n_actual_orig,
                    static_C_obs_orig=_static_C_obs_orig,
                    static_H_obs_orig=_static_H_obs_orig,
                    static_I_obs_orig=_static_I_obs_orig
                ), in_axes=(0,)
            )
            all_draws_jax = vmapped_smoother(keys)
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- simulate_state_space function (remains the same as before) ---
def _simulate_state_space_impl( 
    P_aug: ArrayLike, R_aug: ArrayLike, Omega: ArrayLike, H_obs: ArrayLike,
    init_x: ArrayLike, init_P: ArrayLike, key: jax.random.PRNGKey, num_steps: int
) -> Tuple[jax.Array, jax.Array]:
    
    desired_dtype = P_aug.dtype 
    P_aug_jax = jnp.asarray(P_aug, dtype=desired_dtype) 
    R_aug_jax = jnp.asarray(R_aug, dtype=desired_dtype)
    Omega_jax = jnp.asarray(Omega, dtype=desired_dtype) 
    H_obs_jax = jnp.asarray(H_obs, dtype=desired_dtype)
    init_x_jax = jnp.asarray(init_x, dtype=desired_dtype)
    init_P_jax = jnp.asarray(init_P, dtype=desired_dtype)
    n_aug = P_aug_jax.shape[0] 
    n_aug_shocks = R_aug_jax.shape[1] if R_aug_jax.ndim == 2 and R_aug_jax.shape[1] > 0 else 0
    n_obs_sim = Omega_jax.shape[0]; key_init, key_state_noise, key_obs_noise = random.split(key, 3)

    try:
        init_P_reg = init_P_jax + _MACHINE_EPSILON * jnp.eye(n_aug, dtype=desired_dtype)
        L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_aug,), dtype=desired_dtype); x0 = init_x_jax + L0 @ z0

    except Exception: x0 = init_x_jax

    state_shocks_std_normal = random.normal(key_state_noise, (num_steps, n_aug_shocks), dtype=desired_dtype) if n_aug_shocks > 0 else jnp.zeros((num_steps, 0), dtype=desired_dtype)
    obs_noise_sim = jnp.zeros((num_steps, n_obs_sim), dtype=desired_dtype)

    if n_obs_sim > 0:
        try:
            H_obs_reg = H_obs_jax + _MACHINE_EPSILON * 1e-2 * jnp.eye(n_obs_sim, dtype=desired_dtype)
            obs_noise_sim = random.multivariate_normal(key_obs_noise, jnp.zeros(n_obs_sim, dtype=desired_dtype), H_obs_reg, shape=(num_steps,), dtype=desired_dtype)
        except Exception: pass

    def simulation_step(x_prev, noise_t_tuple):
        eps_t_arg, eta_t_arg = noise_t_tuple
        shock_term_sim = R_aug_jax @ eps_t_arg if n_aug_shocks > 0 else jnp.zeros(n_aug, dtype=x_prev.dtype)
        x_curr_sim = P_aug_jax @ x_prev + shock_term_sim
        y_curr_sim = Omega_jax @ x_curr_sim + eta_t_arg if n_obs_sim > 0 else jnp.empty((0,), dtype=x_curr_sim.dtype)
        return x_curr_sim, (x_curr_sim, y_curr_sim)

    combined_noise_sim = (state_shocks_std_normal, obs_noise_sim)
    _, (states_sim, observations_sim) = lax.scan(simulation_step, x0, combined_noise_sim)

    return states_sim, observations_sim

simulate_state_space = jax.jit(_simulate_state_space_impl, static_argnames=('num_steps',))