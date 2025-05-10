# --- START OF FILE dk_simulation_smoother_jax.py ---
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax, random, vmap
from jax.typing import ArrayLike
from typing import Tuple, Optional, Union, Sequence, Dict, Any

# This file defines DKSimulationSmoother.
# It does NOT depend on the main KalmanFilter.py used for MCMC.
# It re-implements the necessary filter/smoother steps for dense data internally.

_KF_JITTER_DK = 1e-8 # Jitter for this specific smoother's internal KF steps

class DKSimulationSmoother:
    def __init__(self, T: jax.Array, R: jax.Array, C: jax.Array, H_full: jax.Array,
                 init_x_mean: jax.Array, init_P_cov: jax.Array):
        """
        Initializes the Durbin-Koopman Simulation Smoother helper.
        Args:
            T: State transition matrix (P_aug_draw)
            R: Shock impact matrix (R_aug_draw)
            C: Observation matrix (Omega_draw - full)
            H_full: Full observation noise covariance matrix (fixed for the original data problem)
            init_x_mean: Initial state mean (for the original data problem, used for x_star simulation)
            init_P_cov: Initial state covariance (for the original data problem, used for x_star simulation)
        """
        desired_dtype = T.dtype
        self.T = jnp.asarray(T, dtype=desired_dtype)
        self.R = jnp.asarray(R, dtype=desired_dtype)
        self.C_full = jnp.asarray(C, dtype=desired_dtype) # Omega_draw
        self.H_full = jnp.asarray(H_full, dtype=desired_dtype) # H from the main problem setup
        self.init_x = jnp.asarray(init_x_mean, dtype=desired_dtype)
        self.init_P = jnp.asarray(init_P_cov, dtype=desired_dtype)

        self.n_state = T.shape[0]
        self.n_obs_full = C.shape[0]
        self.n_shocks = R.shape[1] if R.ndim == 2 and R.shape[1] > 0 else 0
        self.I_s = jnp.eye(self.n_state, dtype=desired_dtype)
        self.I_obs_full = jnp.eye(self.n_obs_full, dtype=desired_dtype) if self.n_obs_full > 0 else jnp.empty((0,0), dtype=desired_dtype)

        if self.n_shocks > 0:
            self.state_cov_Q = self.R @ self.R.T
        else:
            self.state_cov_Q = jnp.zeros((self.n_state, self.n_state), dtype=desired_dtype)
        
        try:
            H_reg_chol = self.H_full + _KF_JITTER_DK * self.I_obs_full if self.n_obs_full > 0 else self.H_full
            self.L_H_full_for_y_star = jnp.linalg.cholesky(H_reg_chol) if self.n_obs_full > 0 else jnp.empty((0,0), dtype=desired_dtype)
            self._simulate_obs_noise_for_y_star = self._simulate_obs_noise_chol_internal_dk
        except Exception:
            self.H_stable_full_for_y_star = self.H_full + _KF_JITTER_DK * self.I_obs_full if self.n_obs_full > 0 else self.H_full
            self._simulate_obs_noise_for_y_star = self._simulate_obs_noise_mvn_internal_dk

    def _simulate_obs_noise_chol_internal_dk(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        if self.n_obs_full == 0: return jnp.empty(tuple(shape) + (0,), dtype=self.H_full.dtype)
        z_eta = random.normal(key, tuple(shape) + (self.n_obs_full,), dtype=self.H_full.dtype)
        return z_eta @ self.L_H_full_for_y_star.T

    def _simulate_obs_noise_mvn_internal_dk(self, key: jax.random.PRNGKey, shape: Sequence[int]) -> jax.Array:
        if self.n_obs_full == 0: return jnp.empty(tuple(shape) + (0,), dtype=self.H_full.dtype)
        mvn_shape = tuple(shape) if len(shape) > 0 else ()
        try:
            return random.multivariate_normal(key, jnp.zeros((self.n_obs_full,), dtype=self.H_full.dtype),
                                            self.H_stable_full_for_y_star, shape=mvn_shape, dtype=self.H_full.dtype)
        except Exception:
            return jnp.zeros(tuple(shape) + (self.n_obs_full,), dtype=self.H_full.dtype)

    def _filter_for_dense_data_internal(self, ys_dense: ArrayLike) -> Dict[str, jax.Array]:
        ys_arr = jnp.asarray(ys_dense, dtype=self.C_full.dtype)
        T_mat, I_s = self.T, self.I_s
        state_cov_Q_internal = self.state_cov_Q
        C_obs_dense_internal = self.C_full
        H_obs_dense_internal = self.H_full
        I_obs_dense_internal = self.I_obs_full
        
        kf_jitter = _KF_JITTER_DK
        MAX_STATE_VALUE = 1e6
        T_steps = ys_arr.shape[0]

        if T_steps == 0:
            return {'x_pred': jnp.empty((0,self.n_state),dtype=I_s.dtype),'P_pred': jnp.empty((0,self.n_state,self.n_state),dtype=I_s.dtype),'x_filt': jnp.empty((0,self.n_state),dtype=I_s.dtype),'P_filt': jnp.empty((0,self.n_state,self.n_state),dtype=I_s.dtype),'innovations': jnp.empty((0,self.n_obs_full if self.n_obs_full>0 else 0),dtype=self.C_full.dtype),'innovation_cov': jnp.empty((0,self.n_obs_full if self.n_obs_full>0 else 0,self.n_obs_full if self.n_obs_full>0 else 0),dtype=self.C_full.dtype),'log_likelihood_contributions': jnp.empty((0,),dtype=I_s.dtype)}

        def step_dense_data(carry, y_t_slice):
            x_prev_filt, P_prev_filt = carry
            x_pred_t = T_mat @ x_prev_filt; x_pred_t = jnp.clip(x_pred_t,-MAX_STATE_VALUE,MAX_STATE_VALUE)
            P_prev_filt_sym = (P_prev_filt+P_prev_filt.T)/2.0; P_prev_filt_reg = P_prev_filt_sym+kf_jitter*I_s
            P_pred_t = T_mat @ P_prev_filt_reg @ T_mat.T + state_cov_Q_internal; P_pred_t=(P_pred_t+P_pred_t.T)/2.0; P_pred_t=P_pred_t+kf_jitter*I_s
            y_obs_t = y_t_slice; y_pred_obs = C_obs_dense_internal @ x_pred_t; v_obs = y_obs_t - y_pred_obs
            PCt_obs = P_pred_t @ C_obs_dense_internal.T; S_obs = C_obs_dense_internal @ PCt_obs + H_obs_dense_internal; S_obs_reg = S_obs + kf_jitter * I_obs_dense_internal
            
            K_val = jnp.zeros((self.n_state, self.n_obs_full if self.n_obs_full > 0 else 0), dtype=x_pred_t.dtype)
            current_solve_ok = jnp.array(False)

            if self.n_obs_full > 0:
                try: L_S_obs=jnp.linalg.cholesky(S_obs_reg); K_T_temp=jax.scipy.linalg.solve_triangular(L_S_obs,PCt_obs.T,lower=True,trans='N'); K_val=jax.scipy.linalg.solve_triangular(L_S_obs,K_T_temp,lower=True,trans='T').T; current_solve_ok=jnp.all(jnp.isfinite(K_val))
                except Exception: current_solve_ok=jnp.array(False)
                
                def _fb_sol(combined_op):
                    op_data, _ = combined_op # Unpack: op_data is (PCt_op, S_reg_op)
                    PCt_op, S_reg_op = op_data
                    try: K_fb=jax.scipy.linalg.solve(S_reg_op,PCt_op.T,assume_a='pos').T;return K_fb,jnp.all(jnp.isfinite(K_fb))
                    except Exception:return jnp.zeros_like(PCt_op.T).T,jnp.array(False)
                def _keep_K(combined_op):
                    _, K_status = combined_op # Unpack: K_status is (K_to_keep, status_to_keep)
                    return K_status[0], K_status[1]
                K_val,current_solve_ok=jax.lax.cond(current_solve_ok,_keep_K,_fb_sol,operand=((PCt_obs,S_obs_reg),(K_val,current_solve_ok)))
                
                def _fb_pinv(combined_op):
                    op_data, _ = combined_op
                    PCt_op,S_reg_op=op_data
                    try: K_pinv=PCt_op@jnp.linalg.pinv(S_reg_op,rcond=1e-6);return K_pinv,jnp.all(jnp.isfinite(K_pinv))
                    except Exception:return jnp.zeros_like(PCt_op.T).T,jnp.array(False)
                K_val,current_solve_ok=jax.lax.cond(current_solve_ok,_keep_K,_fb_pinv,operand=((PCt_obs,S_obs_reg),(K_val,current_solve_ok)))
                K_val=jnp.clip(K_val,-1e3,1e3)
            
            final_solve_ok = current_solve_ok
            x_filt_t,P_filt_t=x_pred_t,P_pred_t
            def p_upd(op): K_,v_,x_,P_,C_,H_,I_s_=op;x_up=K_@v_;x_f=x_+x_up;x_f=jnp.clip(x_f,-MAX_STATE_VALUE,MAX_STATE_VALUE);IKC=I_s_-K_@C_;P_f=IKC@P_@IKC.T+K_@H_@K_.T;P_f=(P_f+P_f.T)/2.0;P_f=P_f+kf_jitter*I_s_;return x_f,P_f
            def s_upd(op): _,_,x_,P_,_,_,_=op;return x_,P_
            upd_cond=(self.n_obs_full>0)&final_solve_ok
            x_filt_t,P_filt_t=jax.lax.cond(upd_cond,p_upd,s_upd,operand=(K_val,v_obs,x_pred_t,P_pred_t,C_obs_dense_internal,H_obs_dense_internal,I_s))
            ll_t=0.0
            if self.n_obs_full > 0:
                try: sign,log_det_S=jnp.linalg.slogdet(S_obs_reg);L_S_ll=jnp.linalg.cholesky(S_obs_reg);z=jax.scipy.linalg.solve_triangular(L_S_ll,v_obs,lower=True);mah_d=jnp.sum(z**2);log_pi=jnp.log(2*jnp.pi)*self.n_obs_full;ll_term=-0.5*(log_pi+log_det_S+mah_d);val_ll=(sign>0)&jnp.isfinite(mah_d)&jnp.isfinite(log_det_S);ll_t=jnp.where(val_ll,ll_term,-1e6)
                except Exception: ll_t=jnp.array(-1e6)
            ll_t=jnp.where(jnp.isfinite(ll_t),ll_t,-1e6)
            out={'x_pred':jnp.where(jnp.isfinite(x_pred_t),x_pred_t,jnp.zeros_like(x_pred_t)),'P_pred':jnp.where(jnp.isfinite(P_pred_t),P_pred_t,I_s*1e6),'x_filt':jnp.where(jnp.isfinite(x_filt_t),x_filt_t,jnp.zeros_like(x_filt_t)),'P_filt':jnp.where(jnp.isfinite(P_filt_t),P_filt_t,I_s*1e6),'innovations':jnp.where(jnp.isfinite(v_obs),v_obs,jnp.zeros_like(v_obs)),'innovation_cov':jnp.where(jnp.isfinite(S_obs_reg),S_obs_reg,I_obs_dense_internal*1e6 if self.n_obs_full>0 else jnp.empty((0,0),dtype=S_obs_reg.dtype)),'log_likelihood_contributions':ll_t}
            x_f_s,P_f_s=out['x_filt'],out['P_filt'];P_f_s=(P_f_s+P_f_s.T)/2.0;return (x_f_s,P_f_s),out
        
        init_carry=(self.init_x,self.init_P)
        (_, _),scan_outputs=lax.scan(step_dense_data,init_carry,ys_arr)
        for key_final,val_final_arr in scan_outputs.items():
            if key_final=='log_likelihood_contributions':scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,jnp.full_like(val_final_arr,-1e6))
            elif hasattr(val_final_arr,'size') and val_final_arr.size>0:
                if val_final_arr.ndim==3 and val_final_arr.shape[-1]==val_final_arr.shape[-2]:default_val=jnp.eye(val_final_arr.shape[-1],dtype=val_final_arr.dtype)*1e6;scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,default_val[None,:,:])
                elif val_final_arr.ndim==2 and val_final_arr.shape[0]>0 and val_final_arr.shape[1]>0:scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,jnp.zeros_like(val_final_arr))
                elif val_final_arr.ndim==1:scan_outputs[key_final]=jnp.where(jnp.isfinite(val_final_arr),val_final_arr,jnp.full_like(val_final_arr,-1e6))
        return scan_outputs

    def _rts_smoother_backend_internal(self, filter_results_dict: Dict) -> Tuple[jax.Array, jax.Array]:
        x_pred=filter_results_dict['x_pred'];P_pred=filter_results_dict['P_pred'];x_filt=filter_results_dict['x_filt'];P_filt=filter_results_dict['P_filt']
        T_mat=self.T;N=x_filt.shape[0];kf_jitter=_KF_JITTER_DK
        if N==0:return jnp.empty((0,self.n_state),dtype=x_filt.dtype),jnp.empty((0,self.n_state,self.n_state),dtype=P_filt.dtype)
        x_s_next=x_filt[-1];P_s_next=P_filt[-1]
        P_pred_safe=jnp.where(jnp.isfinite(P_pred),P_pred,jnp.eye(self.n_state,dtype=P_pred.dtype)*1e6);P_filt_safe=jnp.where(jnp.isfinite(P_filt),P_filt,jnp.eye(self.n_state,dtype=P_filt.dtype)*1e6)
        x_pred_safe=jnp.where(jnp.isfinite(x_pred),x_pred,jnp.zeros_like(x_pred));x_filt_safe=jnp.where(jnp.isfinite(x_filt),x_filt,jnp.zeros_like(x_filt))
        P_pred_for_scan=P_pred_safe[1:N];P_filt_for_scan=P_filt_safe[0:N-1];x_pred_for_scan=x_pred_safe[1:N];x_filt_for_scan=x_filt_safe[0:N-1]
        scan_inputs=(P_pred_for_scan[::-1],P_filt_for_scan[::-1],x_pred_for_scan[::-1],x_filt_for_scan[::-1])
        
        def backward_step_common(carry_smooth,scan_t):
            x_s_next_t,P_s_next_t=carry_smooth;Pp_next_t,Pf_t,xp_next_t,xf_t=scan_t
            Pf_t_sym=(Pf_t+Pf_t.T)/2.0;Pp_next_reg=Pp_next_t+kf_jitter*jnp.eye(self.n_state,dtype=Pp_next_t.dtype)
            Jt_val=jnp.zeros((self.n_state,self.n_state),dtype=Pf_t.dtype);current_solve_J_ok=jnp.array(False)
            try:Jt_T=jax.scipy.linalg.solve(Pp_next_reg,(T_mat@Pf_t_sym).T,assume_a='pos');Jt_val=Jt_T.T;current_solve_J_ok=jnp.all(jnp.isfinite(Jt_val))
            except Exception:current_solve_J_ok=jnp.array(False)
            
            def _fb_pinv_J(combined_op):
                op_data, _ = combined_op
                T_op,Pf_op,Pp_reg_op=op_data
                try:Jpv=(Pf_op@T_op.T)@jnp.linalg.pinv(Pp_reg_op,rcond=1e-6);return Jpv,jnp.all(jnp.isfinite(Jpv))
                except Exception:return jnp.zeros_like(Pf_op),jnp.array(False)
            def _keep_J(combined_op):
                _,J_status=combined_op
                return J_status[0],J_status[1]
            Jt_val,current_solve_J_ok=jax.lax.cond(current_solve_J_ok,_keep_J,_fb_pinv_J,operand=((T_mat,Pf_t_sym,Pp_next_reg),(Jt_val,current_solve_J_ok)))
            
            x_d=x_s_next_t-xp_next_t;x_s_t=xf_t+Jt_val@x_d;P_d=P_s_next_t-Pp_next_t;P_s_t=Pf_t_sym+Jt_val@P_d@Jt_val.T;P_s_t=(P_s_t+P_s_t.T)/2.0
            x_s_t=jnp.where(jnp.isfinite(x_s_t),x_s_t,jnp.zeros_like(x_s_t));P_s_t=jnp.where(jnp.isfinite(P_s_t),P_s_t,jnp.eye(self.n_state,dtype=P_s_t.dtype)*1e6);P_s_t=(P_s_t+P_s_t.T)/2.0
            return (x_s_t,P_s_t),(x_s_t,P_s_t)
        
        init_carry_smooth=(x_s_next,P_s_next);(_,_),(x_s_rev,P_s_rev)=lax.scan(backward_step_common,init_carry_smooth,scan_inputs)
        x_smooth=jnp.concatenate([x_s_rev[::-1],x_filt_safe[N-1:N]],axis=0);P_smooth=jnp.concatenate([P_s_rev[::-1],P_filt_safe[N-1:N]],axis=0)
        x_smooth=jnp.where(jnp.isfinite(x_smooth),x_smooth,jnp.zeros_like(x_smooth));P_smooth=jnp.where(jnp.isfinite(P_smooth),P_smooth,jnp.eye(self.n_state,dtype=P_smooth.dtype)*1e6)
        return x_smooth,P_smooth

    def smooth_for_dense_data_internal(self, ys_dense: ArrayLike, filter_results_dense: Optional[Dict] = None) -> Tuple[jax.Array, jax.Array]:
        if filter_results_dense is None:
            filter_outs_dict = self._filter_for_dense_data_internal(ys_dense)
        else:
            filter_outs_dict = filter_results_dense
        return self._rts_smoother_backend_internal(filter_outs_dict)

    def _draw_single_simulation(self,
                                original_ys_for_smoothing: jax.Array,
                                key: jax.random.PRNGKey,
                                x_smooth_rts_original_data: jax.Array,
                                ) -> jax.Array:
        Tsteps = original_ys_for_smoothing.shape[0]
        if Tsteps == 0: return jnp.empty((0, self.n_state), dtype=self.init_x.dtype)
        
        n_s, n_eps_shocks = self.n_state, self.n_shocks
        kf_jitter = _KF_JITTER_DK

        key_init, key_eps, key_eta = random.split(key, 3)
        try:
            init_P_reg = self.init_P + kf_jitter * self.I_s
            L0 = jnp.linalg.cholesky(init_P_reg); z0 = random.normal(key_init, (n_s,), dtype=self.init_x.dtype)
            x0_star = self.init_x + L0 @ z0
        except Exception: x0_star = self.init_x

        eps_star_sim = random.normal(key_eps,(Tsteps,n_eps_shocks),dtype=self.R.dtype) if n_eps_shocks>0 else jnp.zeros((Tsteps,0),dtype=self.R.dtype)
        
        def state_sim_step(x_prev_star, eps_t_star_arg):
            shock_term=self.R@eps_t_star_arg if n_eps_shocks>0 else jnp.zeros(self.n_state,dtype=x_prev_star.dtype)
            x_curr_star=self.T@x_prev_star+shock_term;x_curr_star=jnp.clip(x_curr_star,-1e6,1e6)
            return x_curr_star,x_curr_star
        _,x_star_path=lax.scan(state_sim_step,x0_star,eps_star_sim)

        y_star_dense_sim=jnp.zeros((Tsteps,self.n_obs_full),dtype=x_star_path.dtype)
        if self.n_obs_full>0:
            eta_star_full_sim=self._simulate_obs_noise_for_y_star(key_eta,(Tsteps,))
            y_star_dense_sim=(x_star_path@self.C_full.T)+eta_star_full_sim

        filter_results_star_dense = self._filter_for_dense_data_internal(y_star_dense_sim)
        x_smooth_star_dense, _ = self.smooth_for_dense_data_internal(
            y_star_dense_sim, 
            filter_results_dense=filter_results_star_dense
        )

        x_draw = x_star_path + (x_smooth_rts_original_data - x_smooth_star_dense)
        x_draw = jnp.where(jnp.isfinite(x_draw), x_draw, jnp.zeros_like(x_draw))
        return x_draw

    def run_smoother_draws(self, 
                           original_ys: ArrayLike,
                           key: jax.random.PRNGKey, 
                           num_draws: int,
                           x_smooth_rts_original_data_for_this_param_draw: jax.Array,
                          ) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        if num_draws <= 0: raise ValueError("num_draws must be >= 1.")
        Tsteps = original_ys.shape[0]
        
        empty_state_shape=(num_draws if num_draws>1 else 1,Tsteps,self.n_state) if Tsteps>0 else (num_draws if num_draws>1 else 1,0,self.n_state)
        empty_mean_shape=(Tsteps,self.n_state) if Tsteps>0 else (0,self.n_state)
        if Tsteps == 0:
            if num_draws == 1: return jnp.empty(empty_mean_shape, dtype=self.init_x.dtype)
            else: return (jnp.empty(empty_mean_shape,dtype=self.init_x.dtype),jnp.empty(empty_mean_shape,dtype=self.init_x.dtype),jnp.empty(empty_state_shape,dtype=self.init_x.dtype))

        keys = random.split(key, num_draws)
        
        vmapped_draw_func = vmap(
            lambda k_single: self._draw_single_simulation(
                original_ys,
                k_single,
                x_smooth_rts_original_data=x_smooth_rts_original_data_for_this_param_draw
            ), 
            in_axes=(0,)
        )
        all_draws_jax = vmapped_draw_func(keys)
        
        if num_draws == 1: return all_draws_jax[0]
        else:
            mean_smooth_sim = jnp.mean(all_draws_jax, axis=0)
            median_smooth_sim = jnp.percentile(all_draws_jax, 50.0, axis=0, method='linear')
            return mean_smooth_sim, median_smooth_sim, all_draws_jax

# --- END OF FILE dk_simulation_smoother_jax.py ---