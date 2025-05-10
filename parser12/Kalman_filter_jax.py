# --- START OF FILE Kalman_filter_jax.py (Corrected according to user's summary) ---

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
import numpy as onp 
from typing import Tuple, Optional, Union, Sequence, Dict, Any

_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps
_KF_JITTER = 1e-8

class KalmanFilter:
    def __init__(self, T: ArrayLike, R: ArrayLike, C: ArrayLike, H: ArrayLike, init_x: ArrayLike, init_P: ArrayLike):
        desired_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C = jnp.asarray(C, dtype=desired_dtype)
        self.H = jnp.asarray(H, dtype=desired_dtype)
        self.init_x = jnp.asarray(init_x, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P, dtype=desired_dtype)

        n_state = self.T.shape[0]
        n_obs_full = self.C.shape[0]
        n_shocks = self.R.shape[1] if self.R.ndim == 2 and self.R.shape[1] > 0 else 0

        if self.T.shape != (n_state, n_state): raise ValueError(f"T shape mismatch")
        if n_shocks > 0 and self.R.shape != (n_state, n_shocks): raise ValueError(f"R shape mismatch")
        elif n_shocks == 0 and self.R.size != 0: self.R = jnp.zeros((n_state, 0), dtype=desired_dtype)
        if self.C.shape != (n_obs_full, n_state): raise ValueError(f"C shape mismatch")
        if self.H.shape != (n_obs_full, n_obs_full): raise ValueError(f"H shape mismatch")
        if self.init_x.shape != (n_state,): raise ValueError(f"init_x shape mismatch")
        if self.init_P.shape != (n_state, n_state): raise ValueError(f"init_P shape mismatch")

        self.n_state = n_state
        self.n_obs_full = n_obs_full
        self.n_shocks = n_shocks
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)
        self.I_obs_full = jnp.eye(self.n_obs_full, dtype=desired_dtype) if self.n_obs_full > 0 else jnp.empty((0, 0), dtype=desired_dtype)

        if self.n_shocks > 0:
            self.state_cov = self.R @ self.R.T
        else:
            self.state_cov = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)
        
        try:
            H_reg_chol = self.H + _KF_JITTER * self.I_obs_full if self.n_obs_full > 0 else self.H
            self.L_H_full = jnp.linalg.cholesky(H_reg_chol) if self.n_obs_full > 0 else jnp.empty((0, 0), dtype=desired_dtype)
            self.simulate_obs_noise_internal = self._simulate_obs_noise_chol_internal
        except Exception:
            self.H_stable_full = self.H + _KF_JITTER * self.I_obs_full if self.n_obs_full > 0 else self.H
            self.simulate_obs_noise_internal = self._simulate_obs_noise_mvn_internal

    def _simulate_obs_noise_chol_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        if self.n_obs_full == 0: return jnp.empty(tuple(shape) + (0,), dtype=self.H.dtype)
        z_eta = random.normal(key, tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)
        return z_eta @ self.L_H_full.T

    def _simulate_obs_noise_mvn_internal(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        if self.n_obs_full == 0: return jnp.empty(tuple(shape) + (0,), dtype=self.H.dtype)
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try: return random.multivariate_normal(key, jnp.zeros((self.n_obs_full,), dtype=self.H.dtype),self.H_stable_full, shape=mvn_shape, dtype=self.H.dtype)
        except Exception: return jnp.zeros(tuple(shape) + (self.n_obs_full,), dtype=self.H.dtype)

    def filter(self, ys: ArrayLike, static_valid_obs_idx: jax.Array, static_n_obs_actual: int, static_C_obs: jax.Array, static_H_obs: jax.Array, static_I_obs: jax.Array) -> Dict[str, jax.Array]:
        ys_arr = jnp.asarray(ys, dtype=self.C.dtype)
        T_mat, I_s = self.T, self.I_s
        state_cov = self.state_cov
        kf_jitter = _KF_JITTER 
        MAX_STATE_VALUE = 1e6
        T_steps = ys_arr.shape[0]

        if T_steps == 0:
            return {
                'x_pred': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_pred': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'x_filt': jnp.empty((0, self.n_state), dtype=I_s.dtype),
                'P_filt': jnp.empty((0, self.n_state, self.n_state), dtype=I_s.dtype),
                'innovations': jnp.empty((0, static_n_obs_actual if static_n_obs_actual > 0 else 0), dtype=self.C.dtype),
                'innovation_cov': jnp.empty((0, static_n_obs_actual if static_n_obs_actual > 0 else 0, static_n_obs_actual if static_n_obs_actual > 0 else 0), dtype=self.C.dtype),
                'log_likelihood_contributions': jnp.empty((0,), dtype=I_s.dtype)
            }

        def step_nan_handling(carry, y_t_full_slice):
            x_prev_filt, P_prev_filt = carry
            x_pred_t = T_mat @ x_prev_filt
            x_pred_t = jnp.clip(x_pred_t, -MAX_STATE_VALUE, MAX_STATE_VALUE)
            P_prev_filt_sym = (P_prev_filt + P_prev_filt.T) / 2.0
            P_prev_filt_reg = P_prev_filt_sym + kf_jitter * I_s
            P_pred_t = T_mat @ P_prev_filt_reg @ T_mat.T + state_cov
            P_pred_t = (P_pred_t + P_pred_t.T) / 2.0
            P_pred_t = P_pred_t + kf_jitter * I_s
            y_obs_t = jnp.take(y_t_full_slice, static_valid_obs_idx, axis=0) if static_n_obs_actual > 0 else jnp.empty((0,), dtype=y_t_full_slice.dtype)
            y_pred_obs = static_C_obs @ x_pred_t
            v_obs = y_obs_t - y_pred_obs
            PCt_obs_val = P_pred_t @ static_C_obs.T 
            S_obs_val = static_C_obs @ PCt_obs_val + static_H_obs 
            S_obs_reg_val = S_obs_val + kf_jitter * static_I_obs
            
            K_res = jnp.zeros((self.n_state, static_n_obs_actual if static_n_obs_actual > 0 else 0), dtype=x_pred_t.dtype)
            solve_status = jnp.array(False)

            if static_n_obs_actual > 0:
                # Attempt 1: Cholesky solve
                try:
                    L_S_obs = jnp.linalg.cholesky(S_obs_reg_val)
                    K_T_temp = jax.scipy.linalg.solve_triangular(L_S_obs, PCt_obs_val.T, lower=True, trans='N')
                    K_chol_attempt = jax.scipy.linalg.solve_triangular(L_S_obs, K_T_temp, lower=True, trans='T').T
                    chol_ok = jnp.all(jnp.isfinite(K_chol_attempt))
                    K_res = jnp.where(chol_ok, K_chol_attempt, K_res) 
                    solve_status = chol_ok
                except Exception:
                    solve_status = jnp.array(False)
                
                # Local helper functions for lax.cond
                def _standard_solve_branch_local(operand_tuple_std_solve):
                    # operand_tuple is ((matrices_for_solve), (K_previous, status_previous))
                    matrices_for_solve, _ = operand_tuple_std_solve 
                    PCt_op, S_reg_op = matrices_for_solve
                    try:
                        K_std = jax.scipy.linalg.solve(S_reg_op, PCt_op.T, assume_a='pos').T
                        return K_std, jnp.all(jnp.isfinite(K_std))
                    except Exception:
                        return jnp.zeros_like(PCt_op.T).T, jnp.array(False)

                def _keep_previous_result_branch_local(operand_tuple_keep):
                    # operand_tuple is ((matrices_for_solve), (K_previous, status_previous))
                    _, K_and_status_prev = operand_tuple_keep
                    return K_and_status_prev[0], K_and_status_prev[1]
                
                K_res, solve_status = jax.lax.cond(
                    solve_status, 
                    _keep_previous_result_branch_local, 
                    _standard_solve_branch_local,       
                    operand=((PCt_obs_val, S_obs_reg_val), (K_res, solve_status)) 
                )

                def _pinv_solve_branch_local(operand_tuple_pinv):
                    matrices_for_pinv, _ = operand_tuple_pinv
                    PCt_op, S_reg_op = matrices_for_pinv
                    try:
                        K_pinv = PCt_op @ jnp.linalg.pinv(S_reg_op, rcond=1e-6)
                        return K_pinv, jnp.all(jnp.isfinite(K_pinv))
                    except Exception:
                        return jnp.zeros_like(PCt_op.T).T, jnp.array(False)

                K_res, solve_status = jax.lax.cond(
                    solve_status, 
                    _keep_previous_result_branch_local, 
                    _pinv_solve_branch_local,          
                    operand=((PCt_obs_val, S_obs_reg_val), (K_res, solve_status)) 
                )
                K_res = jnp.clip(K_res, -1e3, 1e3)
            
            x_filt_t, P_filt_t = x_pred_t, P_pred_t
            def p_upd(op_upd):
                K_, v_, x_, P_, C_, H_, I_s_ = op_upd
                x_up = K_ @ v_; x_f = x_ + x_up; x_f = jnp.clip(x_f, -MAX_STATE_VALUE, MAX_STATE_VALUE)
                IKC = I_s_ - K_ @ C_; P_f = IKC @ P_ @ IKC.T + K_ @ H_ @ K_.T
                P_f = (P_f + P_f.T) / 2.0; P_f = P_f + kf_jitter * I_s_
                return x_f, P_f
            def s_upd(op_upd):
                _, _, x_, P_, _, _, _ = op_upd; return x_, P_
            
            upd_cond = (static_n_obs_actual > 0) & solve_status
            x_filt_t, P_filt_t = jax.lax.cond(upd_cond,p_upd,s_upd,operand=(K_res,v_obs,x_pred_t,P_pred_t,static_C_obs,static_H_obs,I_s))
            
            ll_t = 0.0
            if static_n_obs_actual > 0:
                try:
                    sign, log_det_S = jnp.linalg.slogdet(S_obs_reg_val)
                    L_S_ll = jnp.linalg.cholesky(S_obs_reg_val)
                    z = jax.scipy.linalg.solve_triangular(L_S_ll, v_obs, lower=True); mah_d = jnp.sum(z**2)
                    log_pi = jnp.log(2 * jnp.pi) * static_n_obs_actual; ll_term = -0.5 * (log_pi + log_det_S + mah_d)
                    val_ll = (sign > 0) & jnp.isfinite(mah_d) & jnp.isfinite(log_det_S); ll_t = jnp.where(val_ll, ll_term, -1e6)
                except Exception: ll_t = jnp.array(-1e6)
            ll_t = jnp.where(jnp.isfinite(ll_t), ll_t, -1e6)
            
            out = {'x_pred':jnp.where(jnp.isfinite(x_pred_t),x_pred_t,jnp.zeros_like(x_pred_t)),'P_pred':jnp.where(jnp.isfinite(P_pred_t),P_pred_t,I_s*1e6),'x_filt':jnp.where(jnp.isfinite(x_filt_t),x_filt_t,jnp.zeros_like(x_filt_t)),'P_filt':jnp.where(jnp.isfinite(P_filt_t),P_filt_t,I_s*1e6),'innovations':jnp.where(jnp.isfinite(v_obs),v_obs,jnp.zeros_like(v_obs)),'innovation_cov':jnp.where(jnp.isfinite(S_obs_reg_val),S_obs_reg_val,static_I_obs*1e6 if static_n_obs_actual>0 else jnp.empty((0,0),dtype=S_obs_reg_val.dtype)),'log_likelihood_contributions':ll_t}
            x_f_s,P_f_s=out['x_filt'],out['P_filt'];P_f_s=(P_f_s+P_f_s.T)/2.0
            return (x_f_s,P_f_s),out

        init_carry=(self.init_x,self.init_P)
        (_, _),scan_outputs=lax.scan(step_nan_handling,init_carry,ys_arr)
        
        for key_final,val_final_arr in scan_outputs.items():
            if key_final=='log_likelihood_contributions':scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,jnp.full_like(val_final_arr,-1e6))
            elif hasattr(val_final_arr,'size') and val_final_arr.size>0:
                if val_final_arr.ndim==3 and val_final_arr.shape[-1]==val_final_arr.shape[-2]:default_val=jnp.eye(val_final_arr.shape[-1],dtype=val_final_arr.dtype)*1e6;scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,default_val[None,:,:])
                elif val_final_arr.ndim==2 and val_final_arr.shape[0]>0 and val_final_arr.shape[1]>0:scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,jnp.zeros_like(val_final_arr))
                elif val_final_arr.ndim==1:scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,jnp.full_like(val_final_arr,-1e6))
        return scan_outputs

    def _rts_smoother_backend(self, filter_results_dict: Dict) -> Tuple[jax.Array, jax.Array]:
        x_pred = filter_results_dict['x_pred']; P_pred = filter_results_dict['P_pred']
        x_filt = filter_results_dict['x_filt']; P_filt = filter_results_dict['P_filt']
        T_mat = self.T; N = x_filt.shape[0]; kf_jitter_smooth_backend = _KF_JITTER
        if N == 0: return jnp.empty((0, self.n_state), dtype=x_filt.dtype), jnp.empty((0, self.n_state, self.n_state), dtype=P_filt.dtype)
        
        x_s_next = x_filt[-1]; P_s_next = P_filt[-1]
        P_pred_safe = jnp.where(jnp.isfinite(P_pred), P_pred, jnp.eye(self.n_state, dtype=P_pred.dtype) * 1e6)
        P_filt_safe = jnp.where(jnp.isfinite(P_filt), P_filt, jnp.eye(self.n_state, dtype=P_filt.dtype) * 1e6)
        x_pred_safe = jnp.where(jnp.isfinite(x_pred), x_pred, jnp.zeros_like(x_pred))
        x_filt_safe = jnp.where(jnp.isfinite(x_filt), x_filt, jnp.zeros_like(x_filt))
        P_pred_for_scan = P_pred_safe[1:N]; P_filt_for_scan = P_filt_safe[0:N - 1]
        x_pred_for_scan = x_pred_safe[1:N]; x_filt_for_scan = x_filt_safe[0:N - 1]
        scan_inputs = (P_pred_for_scan[::-1], P_filt_for_scan[::-1], x_pred_for_scan[::-1], x_filt_for_scan[::-1])
        
        def backward_step_common(carry_smooth, scan_t):
            x_s_next_t, P_s_next_t = carry_smooth
            Pp_next_t, Pf_t, xp_next_t, xf_t = scan_t
            Pf_t_sym = (Pf_t + Pf_t.T) / 2.0
            Pp_next_reg_val = Pp_next_t + kf_jitter_smooth_backend * jnp.eye(self.n_state, dtype=Pp_next_t.dtype)
            
            Jt_res = jnp.zeros((self.n_state, self.n_state), dtype=Pf_t.dtype)
            current_J_solve_ok = jnp.array(False)
            try:
                Jt_T = jax.scipy.linalg.solve(Pp_next_reg_val, (T_mat @ Pf_t_sym).T, assume_a='pos')
                Jt_res = Jt_T.T
                current_J_solve_ok = jnp.all(jnp.isfinite(Jt_res))
            except Exception:
                current_J_solve_ok = jnp.array(False)
            
            ### MODIFIED BLOCK START (for _rts_smoother_backend) ###
            def _fb_pinv_J_common_local(operand_tuple_J_pinv):
                matrices_for_pinv, _ = operand_tuple_J_pinv # operand_tuple is ((T,Pf_sym,Pp_reg),(Jt_prev,Status_prev))
                T_loc, Pf_s_loc, Pp_n_r_loc = matrices_for_pinv
                try: 
                    J_pinv = (Pf_s_loc @ T_loc.T) @ jnp.linalg.pinv(Pp_n_r_loc, rcond=1e-6)
                    return J_pinv, jnp.all(jnp.isfinite(J_pinv))
                except Exception: 
                    return jnp.zeros_like(Pf_s_loc), jnp.array(False) # Pf_s_loc has correct shape for Jt
            
            def _keep_J_common_local(operand_tuple_J_keep):
                _, J_and_status_to_keep = operand_tuple_J_keep # operand_tuple is ((T,Pf_sym,Pp_reg),(Jt_prev,Status_prev))
                return J_and_status_to_keep[0], J_and_status_to_keep[1]

            Jt_res, current_J_solve_ok = jax.lax.cond(
                current_J_solve_ok, # Predicate: if direct solve for Jt was OK
                _keep_J_common_local,
                _fb_pinv_J_common_local,
                operand=((T_mat, Pf_t_sym, Pp_next_reg_val), (Jt_res, current_J_solve_ok))
            )
            ### MODIFIED BLOCK END ###
            
            x_d = x_s_next_t - xp_next_t; x_s_t = xf_t + Jt_res @ x_d 
            P_d = P_s_next_t - Pp_next_t; P_s_t = Pf_t_sym + Jt_res @ P_d @ Jt_res.T 
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            x_s_t = jnp.where(jnp.isfinite(x_s_t), x_s_t, jnp.zeros_like(x_s_t))
            P_s_t = jnp.where(jnp.isfinite(P_s_t), P_s_t, jnp.eye(self.n_state, dtype=P_s_t.dtype) * 1e6)
            P_s_t = (P_s_t + P_s_t.T) / 2.0
            return (x_s_t, P_s_t), (x_s_t, P_s_t)

        init_carry_smooth = (x_s_next, P_s_next)
        (_, _), (x_s_rev, P_s_rev) = lax.scan(backward_step_common, init_carry_smooth, scan_inputs)
        x_smooth = jnp.concatenate([x_s_rev[::-1], x_filt_safe[N - 1:N]], axis=0)
        P_smooth = jnp.concatenate([P_s_rev[::-1], P_filt_safe[N - 1:N]], axis=0)
        x_smooth = jnp.where(jnp.isfinite(x_smooth), x_smooth, jnp.zeros_like(x_smooth))
        P_smooth = jnp.where(jnp.isfinite(P_smooth), P_smooth, jnp.eye(self.n_state, dtype=P_smooth.dtype) * 1e6)
        return x_smooth, P_smooth

    # The rest of KalmanFilter class (smooth, log_likelihood, _simulate_state_space_impl etc.)
    # should remain as they were in your working version or my last "TARGET VERSION"
    # as they call these corrected filter and _rts_smoother_backend methods.
    # Ensure they are present and correctly defined. For example:
    def smooth(self,ys:ArrayLike,filter_results:Optional[Dict]=None,static_valid_obs_idx:Optional[jax.Array]=None,static_n_obs_actual:Optional[int]=None,static_C_obs_for_filter:Optional[jax.Array]=None,static_H_obs_for_filter:Optional[jax.Array]=None,static_I_obs_for_filter:Optional[jax.Array]=None)->Tuple[jax.Array,jax.Array]:
        ys_arr=jnp.asarray(ys,dtype=self.C.dtype)
        if filter_results is None:
            if (static_valid_obs_idx is None or static_n_obs_actual is None or static_C_obs_for_filter is None or static_H_obs_for_filter is None or static_I_obs_for_filter is None):
                raise ValueError("Static NaN info must be provided to smooth() if filter_results is None.")
            filter_outs_dict=self.filter(ys_arr,static_valid_obs_idx,static_n_obs_actual,static_C_obs_for_filter,static_H_obs_for_filter,static_I_obs_for_filter)
        else: filter_outs_dict=filter_results
        return self._rts_smoother_backend(filter_outs_dict)

# Standalone simulate_state_space (if it was part of your original Kalman_filter_jax.py)
def _simulate_state_space_impl(P_aug: ArrayLike, R_aug: ArrayLike, Omega: ArrayLike, H_obs: ArrayLike, init_x: ArrayLike, init_P: ArrayLike, key: jax.random.PRNGKey, num_steps: int) -> Tuple[jax.Array, jax.Array]:
    desired_dtype=P_aug.dtype;P_aug_jax=jnp.asarray(P_aug,dtype=desired_dtype);R_aug_jax=jnp.asarray(R_aug,dtype=desired_dtype);Omega_jax=jnp.asarray(Omega,dtype=desired_dtype);H_obs_jax=jnp.asarray(H_obs,dtype=desired_dtype);init_x_jax=jnp.asarray(init_x,dtype=desired_dtype);init_P_jax=jnp.asarray(init_P,dtype=desired_dtype)
    n_aug=P_aug_jax.shape[0];n_aug_shocks=R_aug_jax.shape[1] if R_aug_jax.ndim==2 and R_aug_jax.shape[1]>0 else 0;n_obs_sim=Omega_jax.shape[0];key_init,key_state_noise,key_obs_noise=random.split(key,3);kf_jitter_sim_impl=_KF_JITTER
    try:init_P_reg=init_P_jax+kf_jitter_sim_impl*jnp.eye(n_aug,dtype=desired_dtype);L0=jnp.linalg.cholesky(init_P_reg);z0=random.normal(key_init,(n_aug,),dtype=desired_dtype);x0=init_x_jax+L0@z0
    except Exception:x0=init_x_jax
    state_shocks_std_normal=random.normal(key_state_noise,(num_steps,n_aug_shocks),dtype=desired_dtype) if n_aug_shocks>0 else jnp.zeros((num_steps,0),dtype=desired_dtype)
    obs_noise_sim_val=jnp.zeros((num_steps,n_obs_sim),dtype=desired_dtype)
    if n_obs_sim>0:
        try:H_obs_reg=H_obs_jax+kf_jitter_sim_impl*jnp.eye(n_obs_sim,dtype=desired_dtype);L_H_obs=jnp.linalg.cholesky(H_obs_reg);z_eta=random.normal(key_obs_noise,(num_steps,n_obs_sim),dtype=desired_dtype);obs_noise_sim_val=z_eta@L_H_obs.T
        except Exception:
            try:H_obs_reg_mvn=H_obs_jax+kf_jitter_sim_impl*10*jnp.eye(n_obs_sim,dtype=desired_dtype);obs_noise_sim_val=random.multivariate_normal(key_obs_noise,jnp.zeros(n_obs_sim,dtype=desired_dtype),H_obs_reg_mvn,shape=(num_steps,),dtype=desired_dtype)
            except Exception:pass
    def sim_step(x_prev,noise_t):eps_t,eta_t=noise_t;shock_term=R_aug_jax@eps_t if n_aug_shocks>0 else jnp.zeros(n_aug,dtype=x_prev.dtype);x_curr=P_aug_jax@x_prev+shock_term;y_curr=Omega_jax@x_curr+eta_t if n_obs_sim>0 else jnp.empty((0,),dtype=x_curr.dtype);return x_curr,(x_curr,y_curr)
    _,(states_sim_res,obs_sim_res)=lax.scan(sim_step,x0,(state_shocks_std_normal,obs_noise_sim_val))
    return states_sim_res,obs_sim_res
simulate_state_space=jax.jit(_simulate_state_space_impl,static_argnames=('num_steps',))

# --- END OF FILE Kalman_filter_jax.py (CORRECTED - FINAL ATTEMPT) ---