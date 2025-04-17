SUBROUTINE GETSTEADYSTATE(VP, NVAR, SSVAL, NEQ, NUM_RAT_OBS, RAT_OBS)
  USE PARAMETERS_MOD
  USE LIKELIHOOD_CONST_MOD, ONLY : RATIND
  IMPLICIT NONE 
  INTEGER, PARAMETER  :: NUMRAT=59
  INTEGER NVAR, NEQ, NUM_RAT_OBS, I
  REAL(DP) , INTENT(IN) , DIMENSION(NVAR) :: VP
  REAL(DP) , INTENT(OUT), DIMENSION(NEQ) :: SSVAL
  REAL(DP) , INTENT(OUT), DIMENSION(NUM_RAT_OBS) :: RAT_OBS
  REAL(DP) , DIMENSION(NUMRAT) :: RAT
  !PARAMETERS GET IN AS A COMMONN BLOCKS
  REAL(DP) BETTA,BETTASTAR,PSI_2

  !VARIABLES
  REAL(DP) LAG1_C_SS,LAG1_CD_SS,LAG1_CM_SS,LAG1_E_SS,LAG1_IE_SS,LAG1_INF_C_SS,LAG2_INF_C_SS,LAG3_INF_C_SS,&
       &LAG1_INF_CD_SS,LAG1_INF_E_SS,LAG1_INF_MD_SS,LAG1_INF_Q_SS,LAG1_INF_RM_SS,LAG1_INF_T_SS,LAG1_INF_XD_SS,&
       &LAG1_MD_SS,LAG1_MS_SS,LAG1_NGDP_PC_SS,LAG2_NGDP_PC_SS,LAG3_NGDP_PC_SS,LAG1_PCD_PC_SS,LAG1_PCD_PQ_SS,&
       &LAG1_PE_PQ_SS,LAG1_PMD_PC_SS,LAG1_PRM_PC_SS,LAG1_PT_PQ_SS,LAG1_PVCD_SS,LAG1_PVE_SS,LAG1_PVM_SS,&
       &LAG1_PVQ_SS,LAG1_PVRM_SS,LAG1_PVT_SS,LAG1_PVXD_SS,LAG1_PX_PC_SS,LAG1_PXD_PCD_SS,LAG1_QD_SS,LAG1_RM_SS,&
       &LAG1_S_PC_SS,LAG1_T_SS,LAG1_W_SS,LAG1_X_SS,LAG1_XD_SS,LAG1_XM_SS,LAG1_YI_SS,LAG1_CSTAR_SS,LAG1_QM_SS,&
       &LAG1_tr_SS,LAG1_F_SS,LAG1_K_SS,F_SS,K_SS,CSTAR_SS,G_SS,PI_STAR_SS,QM_SS,QMR_SS,TR_SS,Z_XDEMAND_SS,&
       &ZCD_SS,ZD_SS,ZE_SS,ZH_SS,ZI_SS,ZIE_SS,ZM_SS,ZNT_SS,ZQ_SS,ZRM_SS,ZU_SS,ZX_SS,ZXD_SS,ADJCOST_C_SS,ADJCOST_X_SS,&
       &C_SS,CD_SS,CDS_SS,CDSD_SS,CM_SS,D_C_SS,D_CD_SS,D_CM_SS,D_CSTAR_SS,D_E_SS,D_F_SS,D_K_SS,D_MD_SS,D_MS_SS,D_NGDP_PC_SS,&
       &D_QD_SS,D_RM_SS,D_SS,D_T_SS,D_TR_SS,D_W_SS,D_X_SS,D_XD_SS,D_XM_SS,DELTA_SS,E_SS,ES_SS,ESD_SS,F1_SS,F2_SS,GAM_X_SS,&
       &H_SS,I_SS,IE_SS,INF_C_SS,INF_CD_SS,INF_E_SS,INF_MD_SS,INF_MF_SS,INF_Q_SS,INF_RM_SS,INF_T_SS,INF_X_SS,INF_XD_SS,&
       &KS_SS,LAMBDA_SS,LED1_YINF_C_SS,LED2_YINF_C_SS,LED3_YINF_C_SS,LED4_YINF_C_SS,LED5_YINF_C_SS,LED6_YINF_C_SS,MD_SS,&
       &MF_SS,MS_SS,NGDP_PC_SS,NT_SS,PCD_PC_SS,PCD_PQ_SS,PCDS_PQ_SS,PE_PC_SS,PE_PQ_SS,PES_PQ_SS,PHI_SS,PHICD_SS,PHIE_SS,&
       &PHIMD_SS,PHIRM_SS,PHIXD_SS,PI_STAR_M_SS,PMD_PC_SS,PMD_PX_SS,PMF_PC_SS,PMF_PMD_SS,PMR_PC_SS,PMR_PRM_SS,POPT_PQ_SS,&
       &POPTCD_PCD_SS,POPTE_PE_SS,POPTMD_PMD_SS,POPTRM_PRM_SS,POPTT_PT_SS,POPTXD_PXD_SS,PQ_PC_SS,PRM_PC_SS,PROF_CD_SS,&
       &PROF_E_SS,PROF_M_SS,PROF_Q_SS,PROF_RM_SS,PROF_T_SS,PROF_XD_SS,PSICD_SS,PSIE_SS,PSIM_SS,PSIP_SS,PSIRM_SS,PSIT_SS,&
       &PSIXD_SS,PT_PC_SS,PT_PMD_SS,PT_PQ_SS,PTS_PQ_SS,PVCD_SS,PVE_SS,PVM_SS,PVQ_SS,PVRM_SS,PVT_SS,PVXD_SS,PX_PC_SS,&
       &PXD_PC_SS,PXD_PCD_SS,PXD_PMD_SS,PXD_PQ_SS,PXD_PX_SS,PXDS_PQ_SS,QD_SS,QE_Q_SS,QS_SS,RK_SS,RM_SS,RMF_SS,RMS_SS,&
       &S_PC_SS,T_SS,TCD_SS,TE_SS,THETACD_SS,THETAE_SS,THETAM_SS,THETAP_SS,THETARM_SS,THETAT_SS,THETAXD_SS,TM_SS,TS_SS,&
       &TXD_SS,U_SS,VQ_SS,W_SS,WOPT_SS,X_SS,XD_SS,XDS_SS,XDSD_SS,XM_SS,YI_SS,YINF_C_SS,YNGDP_PC_SS

  REAL(DP) PMR_PMD_SS, PCDS_PCD_SS, PXDS_PXD_SS, PES_PE_SS

  !DEFINE BETTA AND PHI_1 
  BETTASTAR  = (PI_STAR_BAR)/(ISTAR)
  BETTA  =  BETTASTAR*(1+GBAR)**(SIGMA_M)
  PSI_2  = ((1+UPSILON)/UPSILON)*((1-BETTA*(1+GBAR)**(-SIGMA_M)*(1-PSI_1))/(BETTA*(1+GBAR)**(-SIGMA_M)))



  !! Variables iniciales

  cds_ss=EXP(VP(1))
  cm_ss=EXP(VP(2))
  es_ss=EXP(VP(3))
  h_ss=EXP(VP(4))
  mf_ss=EXP(VP(5))
  rmf_ss=EXP(VP(6))
  rk_ss=EXP(VP(7))
  tcd_ss=EXP(VP(8))
  te_ss=EXP(VP(9))
  tm_ss=EXP(VP(10))
  txd_ss=EXP(VP(11))
  xds_ss=EXP(VP(12))
  xm_ss=EXP(VP(13))

  !  cds_ss=(VP(1))
  !  cm_ss=(VP(2))
  !  es_ss=(VP(3))
  !  h_ss=(VP(4))
  !  mf_ss=(VP(5))
  !  rmf_ss=(VP(6))
  !  rk_ss=(VP(7))
  !  tcd_ss=(VP(8))
  !  te_ss=(VP(9))
  !  tm_ss=(VP(10))
  !  txd_ss=(VP(11))
  !  xds_ss=(VP(12))
  !  xm_ss=(VP(13))

  !! SS values
  adjcost_c_ss=1.0_dp
  adjcost_x_ss=1.0_dp
  inf_c_ss=(pi_bar)
  inf_cd_ss=(pi_bar)
  inf_e_ss=(pi_bar)
  inf_md_ss=(pi_bar)
  inf_mf_ss=(pi_bar)
  inf_q_ss=(pi_bar)
  inf_rm_ss=(pi_bar)
  inf_t_ss=(pi_bar)
  inf_x_ss=(pi_bar)
  inf_xd_ss=(pi_bar)
  cstar_ss=(cstar_bar)
  G_ss=(1+gbar)
  pi_star_ss=(pi_star_bar)
  pi_star_m_ss=(pi_star_bar)
  d_ss=(pi_bar/pi_star_bar)
  ie_ss=(istar)
  i_ss=((istar*pi_bar/pi_star_bar))
  popt_pq_ss=1.0_dp
  poptcd_pcd_ss=1.0_dp
  popte_pe_ss=1.0_dp
  poptmd_pmd_ss=1.0_dp
  poptrm_prm_ss=1.0_dp
  poptt_pt_ss=1.0_dp
  poptxd_pxd_ss=1.0_dp
  pvcd_ss=1.0_dp
  pve_ss=1.0_dp
  pvm_ss=1.0_dp
  pvq_ss=1.0_dp
  pvrm_ss=1.0_dp
  pvt_ss=1.0_dp
  pvxd_ss=1.0_dp
  u_ss=1.0_dp
  zu_ss=(zu_bar)
  zh_ss=(zh_bar)
  zq_ss=(zq_bar)
  zd_ss=(zd_bar)
  znt_ss=(znt_bar)
  zm_ss=(zm_bar)
  zrm_ss=(zrm_bar)
  zx_ss=(zx_bar)
  zie_ss=(zie_bar)
  zi_ss=(zi_bar)
  qm_ss=(qm_bar)
  qmr_ss=(qmr_bar)
  tr_ss=(tr_bar)
  zcd_ss=(zcd_bar)
  zxd_ss=(zxd_bar)
  ze_ss=(ze_bar)
  z_xdemand_ss=(z_xdemand_bar)


  !! Divide & conquer
  delta_ss = ( Psi_1 + Psi_2/(1+upsilon))
  ms_ss=( (zm_ss)*( (gama_m)**(1/omega_m)*(mf_ss)**((omega_m-1)/omega_m) + (1-gama_m)**(1/omega_m)&
       &*(tm_ss)**((omega_m-1)/omega_m) )**(omega_m/(omega_m-1)))
  md_ss=ms_ss
  rms_ss=rmf_ss
  rm_ss=rms_ss
  cdsd_ss=( (zcd_ss)*( (gama_cd)**(1/omega_cd)*(cds_ss)**((omega_cd-1)/omega_cd) + (1-gama_cd)**(1/omega_cd)*&
       &(tcd_ss)**((omega_cd-1)/omega_cd) )**(omega_cd/(omega_cd-1)))
  cd_ss=cdsd_ss
  xdsd_ss=( (zxd_ss)*( (gama_xd)**(1/omega_xd)*(xds_ss)**((omega_xd-1)/omega_xd) + (1-gama_xd)**(1/omega_xd)*&
       &(txd_ss)**((omega_xd-1)/omega_xd) )**(omega_xd/(omega_xd-1)))
  xd_ss=xdsd_ss
  esd_ss=( (ze_ss)*( (gama_e)**(1/omega_e)*(es_ss)**((omega_e-1)/omega_e) + (1-gama_e)**(1/omega_e)*&
       &(te_ss)**((omega_e-1)/omega_e) )**(omega_e/(omega_e-1)))
  e_ss=esd_ss
  c_ss=((gama**(1/omega)*(cd_ss)**((omega-1)/omega) + (1-gama)**(1/omega)*( (cm_ss))**((omega-1)/omega))**(omega/(omega-1)))
  x_ss=( (zx_ss)*( gama_x**(1/omega_x)*(xd_ss)**((omega_x-1)/omega_x) + (1-gama_x)**(1/omega_x)*&
       &((xm_ss)*(adjcost_x_ss))**((omega_x-1)/omega_x) )**(omega_x/(omega_x-1)))
  t_ss=( (tcd_ss) + (txd_ss) + (te_ss) + (tm_ss))
  ts_ss=t_ss
  nt_ss=( (znt_ss)*(vc**(vn-1)*(cds_ss)**vn + vx**(vn-1)*(xds_ss)**vn + vr**(vn-1)*(ts_ss)**vn)**(1/vn))
  qd_ss=( (zd_ss)*(vnt**(v-1)*(nt_ss)**v + ve**(v-1)*(es_ss)**v)**(1/v))
  qs_ss=qd_ss
  pcds_pq_ss=( (zd_ss)*(znt_ss)*vnt**(v-1)*((zd_ss)*(nt_ss)/(qd_ss))**(v-1)*vc**(vn-1)*((znt_ss)*(cds_ss)/(nt_ss))**(vn-1))
  pxds_pq_ss=( (zd_ss)*(znt_ss)*vnt**(v-1)*((zd_ss)*(nt_ss)/(qd_ss))**(v-1)*vx**(vn-1)*((znt_ss)*(xds_ss)/(nt_ss))**(vn-1))
  pts_pq_ss=( (zd_ss)*(znt_ss)*vnt**(v-1)*((zd_ss)*(nt_ss)/(qd_ss))**(v-1)*vr**(vn-1)*((znt_ss)*(ts_ss)/(nt_ss))**(vn-1))
  pes_pq_ss=( (zd_ss)*ve**(v-1)*((zd_ss)*(es_ss)/(qd_ss))**(v-1))
  pcd_pc_ss=((gama)**(1/omega)*((c_ss)/(cd_ss))**(1/omega))
  !pmd_pc_ss=((1-gama)*((c_ss)/(cm_ss))**(1/omega))
  pmd_pc_ss=((1-gama)**(1/omega)*((c_ss)/((cm_ss)))**(1/omega))
  pxd_px_ss=( (zx_ss)*(gama_x)**(1/omega_x)*((x_ss)/((zx_ss)*(xd_ss)))**(1/omega_x))
  !pmd_px_ss=((((xm_ss)/(x_ss))*(((zx_ss)**(1-omega_x))/(1-gama_x)))**(-(1/omega_x)))
  pmd_px_ss=((zx_ss)*(1-gama_x)**(1/omega_x)*((x_ss)/((zx_ss)*(xm_ss)))**(1/omega_x))
  px_pc_ss=pmd_pc_ss / pmd_px_ss
  pxd_pc_ss=pxd_px_ss * px_pc_ss
  phimd_ss=(((theta_m-1)/theta_m)*(pmd_pc_ss))
  pmf_pc_ss=( (phimd_ss)*(zm_ss)*( gama_m*(ms_ss)/((zm_ss)*(mf_ss)) )**(1/omega_m))
  pt_pc_ss=( (phimd_ss)*(zm_ss)*( (1-gama_m)*(ms_ss)/((zm_ss)*(tm_ss)) )**(1/omega_m))
  s_pc_ss=pmf_pc_ss / qm_ss
  pmr_pc_ss = s_pc_ss * qmr_ss
  phirm_ss=pmr_pc_ss
  prm_pc_ss=((theta_rm/(theta_rm-1))*(phirm_ss))
  qe_q_ss=(((e_ss)/(cstar_ss))**(-(1/mu)))
  pe_pc_ss=s_pc_ss * qe_q_ss
  phicd_ss=(((theta_cd-1)/theta_cd)*(pcd_pc_ss))
  phixd_ss=(((theta_xd-1)/theta_xd)*(pxd_pc_ss))
  phie_ss=(((theta_e-1)/theta_e)*(pe_pc_ss))
  lambda_ss=(zu_bar*((c_ss) - hab*(c_ss))**(-sigma_m))
  w_ss=( (zh_ss)*((1-td)*(tbp))**(eta_m)*((h_ss))**(eta_m)*theta_w/((lambda_ss)*(theta_w - 1)))
  wopt_ss=w_ss
  !gam_x_ss=lambda_ss + px_pc_ss
  !gam_x_ss=((lambda_ss)*(px_pc_ss)/(z_xdemand_ss))
  gam_x_ss=((lambda_ss)*(rk_ss)/Psi_2)
  ks_ss=(((w_ss)/(rk_ss))**rhoqv*(alfav/(1-alfav))*(1-td)*tbp*(h_ss))
  k_ss=((ks_ss)*(1+nbar)*(1+gbar))
  vq_ss=(( (1.0)*(alfav**(1/rhoqv)*((ks_ss))**((rhoqv-1)/rhoqv) + (1-alfav)**(1/rhoqv)*&
       &((1-td)*(tbp)*(h_ss))**((rhoqv-1)/rhoqv))**(rhoqv/(rhoqv-1))))
  phi_ss=((w_ss)/((zq_ss)*(1.0)*(alfa*(qs_ss)/((vq_ss)*(zq_ss)))**(1/rhoq)*((1-alfav)*&
       &(vq_ss)/((1-td)*(tbp)*(h_ss)*(1.0)))**(1/rhoqv)))
  pq_pc_ss=((theta)/(theta-1)*(phi_ss))
  prof_q_ss=(((pq_pc_ss)-(phi_ss))*(qs_ss))
  !pts_pc_ss=pts_pq_ss + pq_pc_ss
  prof_m_ss=( (pmd_pc_ss)*(md_ss) - (phimd_ss)*(ms_ss))
  prof_rm_ss=( (prm_pc_ss)*(rm_ss) - (phirm_ss)*(rms_ss))
  prof_cd_ss=( (pcd_pc_ss)*(cd_ss) - (phicd_ss)*(cdsd_ss))
  prof_xd_ss=( (pxd_pc_ss)*(xd_ss) - (phixd_ss)*(xdsd_ss))
  prof_t_ss=( (pt_pc_ss)*(t_ss) - (pts_pq_ss)*(pq_pc_ss)*(ts_ss))
  prof_e_ss=( (pe_pc_ss)*(e_ss) - (phie_ss)*(esd_ss))
  ngdp_pc_ss=( (pq_pc_ss)*(qd_ss) - (prm_pc_ss)*(rm_ss) + (prof_rm_ss) + (prof_m_ss) +&
       & (prof_cd_ss) + (prof_xd_ss) + (prof_t_ss) + (prof_e_ss))
  f_ss=(fbar*(ngdp_pc_ss)/(s_pc_ss))



  ! pcds_pc_ss=pcds_pq_ss + pq_pc_ss
  ! pxds_pc_ss=pxds_pq_ss + pq_pc_ss
  ! pes_pc_ss=pes_pq_ss + pq_pc_ss

  pmf_pmd_ss=pmf_pc_ss / pmd_pc_ss
  pmr_prm_ss= pmr_pc_ss / prm_pc_ss
  pcd_pq_ss= pcd_pc_ss / pq_pc_ss
  pt_pmd_ss=pt_pc_ss / pmd_pc_ss
  pt_pq_ss=pt_pc_ss/ pq_pc_ss
  pxd_pq_ss=pxd_pc_ss / pq_pc_ss
  pe_pq_ss=pe_pc_ss / pq_pc_ss
  pxd_pcd_ss=pxd_pq_ss / pcd_pq_ss
  ! trm_ss=0
  pxd_pmd_ss=pxd_px_ss / pmd_px_ss


  ! Nuevo modelo

  ! print *,"nbar=",nbar
  ! print *,"gbar=",gbar
  yinf_c_ss=((inf_c_ss)**4)
  yngdp_pc_ss=(4*(ngdp_pc_ss))
  yi_ss  = ((zi_ss)**(1/(1-rho_i))*(((istar*pi_bar/pi_star_bar)**4.0) ))
  D_c_ss = ((G_ss)*(1+nbar))
  D_cd_ss=((G_ss)*(1+nbar))
  D_cm_ss =((G_ss)*(1+nbar))
  D_cstar_ss=((G_ss)*(1+nbar))
  D_e_ss=((G_ss)*(1+nbar))
  D_f_ss=((G_ss)*(1+nbar))
  D_k_ss=((G_ss)*(1+nbar))
  D_md_ss=((G_ss)*(1+nbar))
  D_ms_ss=((G_ss)*(1+nbar))
  D_ngdp_pc_ss=((G_ss)*(1+nbar))
  D_qd_ss=((G_ss)*(1+nbar))
  D_rm_ss=((G_ss)*(1+nbar))
  D_t_ss=((G_ss)*(1+nbar))
  D_tr_ss=((G_ss)*(1+nbar))
  D_w_ss=((G_ss))
  D_x_ss=((G_ss)*(1+nbar))
  D_xd_ss=((G_ss)*(1+nbar))
  D_xm_ss=((G_ss)*(1+nbar))


  !! Calvo pricing
  f1_ss= ( ((wopt_ss)*((1-td)*(tbp))*(lambda_ss)*(theta_w - 1)*((h_ss)*((wopt_ss)/(w_ss))**(-theta_w)))/&
       &(1 - ((wopt_ss)/(wopt_ss))**(1-theta_w)*betta*epsw*(1+nbar)*((G_ss))**(1-sigma_m)&
       &*((inf_c_ss)/(inf_c_ss))**(1-theta_w)))
  f2_ss=f1_ss


  THETAP_ss=( (phi_ss)*(qd_ss)/(1 - betta*((G_ss))**(1-sigma_m)*&
       &(1+nbar)*epsq*((inf_q_ss))**(theta)/(inf_q_ss)**(theta)*(lambda_ss)/(lambda_ss)))
  PSIP_ss=((pq_pc_ss)*(qd_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsq*((inf_q_ss))**(theta-1)&
       &/(inf_q_ss)**(theta-1)*(lambda_ss)/(lambda_ss)))


  THETACD_ss=( (cd_ss)*(phicd_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epscd*&
       &((inf_cd_ss))**(theta_cd)/(inf_cd_ss)**(theta_cd)*(lambda_ss)/(lambda_ss)))
  PSICD_ss=( (cd_ss)*(pcd_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epscd*&
       &((inf_cd_ss))**(theta_cd-1)/(inf_cd_ss)**(theta_cd-1)*(lambda_ss)/(lambda_ss)))


  THETAXD_ss=( (xd_ss)*(phixd_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsxd*&
       &((inf_xd_ss))**(theta_xd)/(inf_xd_ss)**(theta_xd)*(lambda_ss)/(lambda_ss)))
  PSIXD_ss=( (xd_ss)*(pxd_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsxd*&
       &((inf_xd_ss))**(theta_xd-1)/(inf_xd_ss)**(theta_xd-1)*(lambda_ss)/(lambda_ss)))


  THETAT_ss=( (t_ss)*(pts_pq_ss)*(pq_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*&
       &epst*((inf_t_ss))**(theta_t)/(inf_t_ss)**(theta_t)*(lambda_ss)/(lambda_ss)))
  PSIT_ss=( (t_ss)*(pt_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epst*&
       &((inf_t_ss))**(theta_t-1)/(inf_t_ss)**(theta_t-1)*(lambda_ss)/(lambda_ss)))


  THETAE_ss=( (e_ss)*(phie_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epse*&
       &((inf_e_ss))**(theta_e)/(inf_e_ss)**(theta_e)*(lambda_ss)/(lambda_ss)))
  PSIE_ss=( (e_ss)*(pe_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epse*&
       &((inf_e_ss))**(theta_e-1)/(inf_e_ss)**(theta_e-1)*(lambda_ss)/(lambda_ss)))


  THETAM_ss=( (md_ss)*(phimd_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsm*&
       &((inf_md_ss))**(theta_m)/(inf_md_ss)**(theta_m)*(lambda_ss)/(lambda_ss)))
  PSIM_ss=( (md_ss)*(pmd_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsm*&
       &((inf_md_ss))**(theta_m-1)/(inf_md_ss)**(theta_m-1)*(lambda_ss)/(lambda_ss)))


  THETARM_ss=( (rm_ss)*(phirm_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsrm*&
       &((inf_rm_ss))**(theta_rm)/(inf_rm_ss)**(theta_rm)*(lambda_ss)/(lambda_ss)))
  PSIRM_ss=( (rm_ss)*(prm_pc_ss)/(1 - betta*((G_ss))**(1-sigma_m)*(1+nbar)*epsrm*&
       &((inf_rm_ss))**(theta_rm-1)/(inf_rm_ss)**(theta_rm-1)*(lambda_ss)/(lambda_ss)))



  !! lags y leads

  lag1_c_ss = c_ss
  lag1_cd_ss = cd_ss
  lag1_cm_ss = cm_ss
  lag1_e_ss = e_ss
  lag1_ie_ss = ie_ss
  lag1_inf_c_ss = inf_c_ss
  lag2_inf_c_ss = inf_c_ss
  lag3_inf_c_ss = inf_c_ss
  lag1_inf_cd_ss = inf_cd_ss
  lag1_inf_e_ss = inf_e_ss
  lag1_inf_md_ss = inf_md_ss
  lag1_inf_q_ss = inf_q_ss
  lag1_inf_rm_ss = inf_rm_ss
  lag1_inf_t_ss = inf_t_ss
  lag1_inf_xd_ss = inf_xd_ss
  lag1_md_ss = md_ss
  lag1_ms_ss = ms_ss
  lag1_ngdp_pc_ss = ngdp_pc_ss
  lag2_ngdp_pc_ss = ngdp_pc_ss
  lag3_ngdp_pc_ss = ngdp_pc_ss
  lag1_pcd_pc_ss = pcd_pc_ss
  lag1_pcd_pq_ss = pcd_pq_ss
  lag1_pe_pq_ss = pe_pq_ss
  lag1_pmd_pc_ss = pmd_pc_ss
  lag1_prm_pc_ss = prm_pc_ss
  lag1_pt_pq_ss = pt_pq_ss
  lag1_pvcd_ss = pvcd_ss
  lag1_pve_ss = pve_ss
  lag1_pvm_ss = pvm_ss
  lag1_pvq_ss = pvq_ss
  lag1_pvrm_ss = pvrm_ss
  lag1_pvt_ss = pvt_ss
  lag1_pvxd_ss = pvxd_ss
  lag1_px_pc_ss = px_pc_ss
  lag1_pxd_pcd_ss = pxd_pcd_ss
  lag1_qd_ss = qd_ss
  lag1_rm_ss = rm_ss
  lag1_s_pc_ss = s_pc_ss
  lag1_t_ss = t_ss
  lag1_w_ss = w_ss
  lag1_x_ss = x_ss
  lag1_xd_ss = xd_ss
  lag1_xm_ss = xm_ss
  lag1_yi_ss = yi_ss
  lag1_f_ss = f_ss
  lag1_k_ss = k_ss
  lag1_cstar_ss = cstar_ss
  lag1_qm_ss = qm_ss
  lag1_tr_ss = tr_ss
  led1_yinf_c_ss = yinf_c_ss
  led2_yinf_c_ss = yinf_c_ss
  led3_yinf_c_ss = yinf_c_ss
  led4_yinf_c_ss = yinf_c_ss
  led5_yinf_c_ss = yinf_c_ss
  led6_yinf_c_ss = yinf_c_ss


  SSVAL(1) = LAG1_C_SS
  SSVAL(2) = LAG1_CD_SS
  SSVAL(3) = LAG1_CM_SS
  SSVAL(4) = LAG1_E_SS
  SSVAL(5) = LAG1_IE_SS
  SSVAL(6) = LAG1_INF_C_SS
  SSVAL(7) = LAG2_INF_C_SS
  SSVAL(8) = LAG3_INF_C_SS
  SSVAL(9) = LAG1_INF_CD_SS
  SSVAL(10) = LAG1_INF_E_SS
  SSVAL(11) = LAG1_INF_MD_SS
  SSVAL(12) = LAG1_INF_Q_SS
  SSVAL(13) = LAG1_INF_RM_SS
  SSVAL(14) = LAG1_INF_T_SS
  SSVAL(15) = LAG1_INF_XD_SS
  SSVAL(16) = LAG1_MD_SS
  SSVAL(17) = LAG1_MS_SS
  SSVAL(18) = LAG1_NGDP_PC_SS
  SSVAL(19) = LAG2_NGDP_PC_SS
  SSVAL(20) = LAG3_NGDP_PC_SS
  SSVAL(21) = LAG1_PCD_PC_SS
  SSVAL(22) = LAG1_PCD_PQ_SS
  SSVAL(23) = LAG1_PE_PQ_SS
  SSVAL(24) = LAG1_PMD_PC_SS
  SSVAL(25) = LAG1_PRM_PC_SS
  SSVAL(26) = LAG1_PT_PQ_SS
  SSVAL(27) = LAG1_PVCD_SS
  SSVAL(28) = LAG1_PVE_SS
  SSVAL(29) = LAG1_PVM_SS
  SSVAL(30) = LAG1_PVQ_SS
  SSVAL(31) = LAG1_PVRM_SS
  SSVAL(32) = LAG1_PVT_SS
  SSVAL(33) = LAG1_PVXD_SS
  SSVAL(34) = LAG1_PX_PC_SS
  SSVAL(35) = LAG1_PXD_PCD_SS
  SSVAL(36) = LAG1_QD_SS
  SSVAL(37) = LAG1_RM_SS
  SSVAL(38) = LAG1_S_PC_SS
  SSVAL(39) = LAG1_T_SS
  SSVAL(40) = LAG1_W_SS
  SSVAL(41) = LAG1_X_SS
  SSVAL(42) = LAG1_XD_SS
  SSVAL(43) = LAG1_XM_SS
  SSVAL(44) = LAG1_YI_SS
  SSVAL(45) = LAG1_CSTAR_SS
  SSVAL(46) = LAG1_QM_SS
  SSVAL(47) = LAG1_TR_SS
  SSVAL(48) = LAG1_F_SS
  SSVAL(49) = LAG1_K_SS
  SSVAL(50) = F_SS
  SSVAL(51) = K_SS
  SSVAL(52) = CSTAR_SS
  SSVAL(53) = G_SS
  SSVAL(54) = PI_STAR_SS
  SSVAL(55) = QM_SS
  SSVAL(56) = QMR_SS
  SSVAL(57) = TR_SS
  SSVAL(58) = Z_XDEMAND_SS
  SSVAL(59) = ZCD_SS
  SSVAL(60) = ZD_SS
  SSVAL(61) = ZE_SS
  SSVAL(62) = ZH_SS
  SSVAL(63) = ZI_SS
  SSVAL(64) = ZIE_SS
  SSVAL(65) = ZM_SS
  SSVAL(66) = ZNT_SS
  SSVAL(67) = ZQ_SS
  SSVAL(68) = ZRM_SS
  SSVAL(69) = ZU_SS
  SSVAL(70) = ZX_SS
  SSVAL(71) = ZXD_SS
  SSVAL(72) = ADJCOST_C_SS
  SSVAL(73) = ADJCOST_X_SS
  SSVAL(74) = C_SS
  SSVAL(75) = CD_SS
  SSVAL(76) = CDS_SS
  SSVAL(77) = CDSD_SS
  SSVAL(78) = CM_SS
  SSVAL(79) = D_C_SS
  SSVAL(80) = D_CD_SS
  SSVAL(81) = D_CM_SS
  SSVAL(82) = D_CSTAR_SS
  SSVAL(83) = D_E_SS
  SSVAL(84) = D_F_SS
  SSVAL(85) = D_K_SS
  SSVAL(86) = D_MD_SS
  SSVAL(87) = D_MS_SS
  SSVAL(88) = D_NGDP_PC_SS
  SSVAL(89) = D_QD_SS
  SSVAL(90) = D_RM_SS
  SSVAL(91) = D_SS
  SSVAL(92) = D_T_SS
  SSVAL(93) = D_TR_SS
  SSVAL(94) = D_W_SS
  SSVAL(95) = D_X_SS
  SSVAL(96) = D_XD_SS
  SSVAL(97) = D_XM_SS
  SSVAL(98) = DELTA_SS
  SSVAL(99) = E_SS
  SSVAL(100) = ES_SS
  SSVAL(101) = ESD_SS
  SSVAL(102) = F1_SS
  SSVAL(103) = F2_SS
  SSVAL(104) = GAM_X_SS
  SSVAL(105) = H_SS
  SSVAL(106) = I_SS
  SSVAL(107) = IE_SS
  SSVAL(108) = INF_C_SS
  SSVAL(109) = INF_CD_SS
  SSVAL(110) = INF_E_SS
  SSVAL(111) = INF_MD_SS
  SSVAL(112) = INF_MF_SS
  SSVAL(113) = INF_Q_SS
  SSVAL(114) = INF_RM_SS
  SSVAL(115) = INF_T_SS
  SSVAL(116) = INF_X_SS
  SSVAL(117) = INF_XD_SS
  SSVAL(118) = KS_SS
  SSVAL(119) = LAMBDA_SS
  SSVAL(120) = LED1_YINF_C_SS
  SSVAL(121) = LED2_YINF_C_SS
  SSVAL(122) = LED3_YINF_C_SS
  SSVAL(123) = LED4_YINF_C_SS
  SSVAL(124) = LED5_YINF_C_SS
  SSVAL(125) = LED6_YINF_C_SS
  SSVAL(126) = MD_SS
  SSVAL(127) = MF_SS
  SSVAL(128) = MS_SS
  SSVAL(129) = NGDP_PC_SS
  SSVAL(130) = NT_SS
  SSVAL(131) = PCD_PC_SS
  SSVAL(132) = PCD_PQ_SS
  SSVAL(133) = PCDS_PQ_SS
  SSVAL(134) = PE_PC_SS
  SSVAL(135) = PE_PQ_SS
  SSVAL(136) = PES_PQ_SS
  SSVAL(137) = PHI_SS
  SSVAL(138) = PHICD_SS
  SSVAL(139) = PHIE_SS
  SSVAL(140) = PHIMD_SS
  SSVAL(141) = PHIRM_SS
  SSVAL(142) = PHIXD_SS
  SSVAL(143) = PI_STAR_M_SS
  SSVAL(144) = PMD_PC_SS
  SSVAL(145) = PMD_PX_SS
  SSVAL(146) = PMF_PC_SS
  SSVAL(147) = PMF_PMD_SS
  SSVAL(148) = PMR_PC_SS
  SSVAL(149) = PMR_PRM_SS
  SSVAL(150) = POPT_PQ_SS
  SSVAL(151) = POPTCD_PCD_SS
  SSVAL(152) = POPTE_PE_SS
  SSVAL(153) = POPTMD_PMD_SS
  SSVAL(154) = POPTRM_PRM_SS
  SSVAL(155) = POPTT_PT_SS
  SSVAL(156) = POPTXD_PXD_SS
  SSVAL(157) = PQ_PC_SS
  SSVAL(158) = PRM_PC_SS
  SSVAL(159) = PROF_CD_SS
  SSVAL(160) = PROF_E_SS
  SSVAL(161) = PROF_M_SS
  SSVAL(162) = PROF_Q_SS
  SSVAL(163) = PROF_RM_SS
  SSVAL(164) = PROF_T_SS
  SSVAL(165) = PROF_XD_SS
  SSVAL(166) = PSICD_SS
  SSVAL(167) = PSIE_SS
  SSVAL(168) = PSIM_SS
  SSVAL(169) = PSIP_SS
  SSVAL(170) = PSIRM_SS
  SSVAL(171) = PSIT_SS
  SSVAL(172) = PSIXD_SS
  SSVAL(173) = PT_PC_SS
  SSVAL(174) = PT_PMD_SS
  SSVAL(175) = PT_PQ_SS
  SSVAL(176) = PTS_PQ_SS
  SSVAL(177) = PVCD_SS
  SSVAL(178) = PVE_SS
  SSVAL(179) = PVM_SS
  SSVAL(180) = PVQ_SS
  SSVAL(181) = PVRM_SS
  SSVAL(182) = PVT_SS
  SSVAL(183) = PVXD_SS
  SSVAL(184) = PX_PC_SS
  SSVAL(185) = PXD_PC_SS
  SSVAL(186) = PXD_PCD_SS
  SSVAL(187) = PXD_PMD_SS
  SSVAL(188) = PXD_PQ_SS
  SSVAL(189) = PXD_PX_SS
  SSVAL(190) = PXDS_PQ_SS
  SSVAL(191) = QD_SS
  SSVAL(192) = QE_Q_SS
  SSVAL(193) = QS_SS
  SSVAL(194) = RK_SS
  SSVAL(195) = RM_SS
  SSVAL(196) = RMF_SS
  SSVAL(197) = RMS_SS
  SSVAL(198) = S_PC_SS
  SSVAL(199) = T_SS
  SSVAL(200) = TCD_SS
  SSVAL(201) = TE_SS
  SSVAL(202) = THETACD_SS
  SSVAL(203) = THETAE_SS
  SSVAL(204) = THETAM_SS
  SSVAL(205) = THETAP_SS
  SSVAL(206) = THETARM_SS
  SSVAL(207) = THETAT_SS
  SSVAL(208) = THETAXD_SS
  SSVAL(209) = TM_SS
  SSVAL(210) = TS_SS
  SSVAL(211) = TXD_SS
  SSVAL(212) = U_SS
  SSVAL(213) = VQ_SS
  SSVAL(214) = W_SS
  SSVAL(215) = WOPT_SS
  SSVAL(216) = X_SS
  SSVAL(217) = XD_SS
  SSVAL(218) = XDS_SS
  SSVAL(219) = XDSD_SS
  SSVAL(220) = XM_SS
  SSVAL(221) = YI_SS
  SSVAL(222) = YINF_C_SS
  SSVAL(223) = YNGDP_PC_SS


  !  RAZONES PARA INCLUIR EL LARGO PLAZO EN LA ESTIMACION DE LOS PARAMETROS

  !  NUEVAS RATIOS INCLUIDAS EN LA ESTIMACION
  !  267-c_ngdp           268-x_ngdp      269-e_ngdp       270-md_ngdp
  !  271-t_ngdp           272-s*tr_ngdp   273-cd_c         274-cm_c
  !  275-xd_x             276-xm_x        277-xd_ngdp      278-cd_ngdp
  !  279-t_md             280-mf_md       281-xm_md        282-cm_md



  RAT_OBS=0.0
  IF (NUM_RAT_OBS .GT. 0) THEN



     PMR_PMD_ss=PMR_PC_ss / PMD_PC_ss
     PCDS_PCD_ss=PCDS_PQ_ss / PCD_PQ_ss
     PXDS_PXD_ss=PXDS_PQ_ss / PXD_PQ_ss
     PES_PE_ss=PES_PQ_ss / PE_PQ_ss

     RAT(1) = (px_pc_ss)*(x_ss)/(NGDP_PC_ss)
     RAT(2) = (pmd_px_ss)*(xm_ss)/(x_ss)
     RAT(3) = (xd_ss)*(pxd_pq_ss)/(qs_ss)
     RAT(4) = (c_ss)/(NGDP_PC_ss)
     RAT(5) = (cd_ss)*(pcd_pq_ss)/(qs_ss)
     RAT(6) = ((cm_ss)/(c_ss))*(pmd_pc_ss)
     RAT(7) = (NGDP_PC_ss)
     RAT(8) = (w_ss)
     RAT(9) = (h_ss)
     RAT(10) = (prm_pc_ss)*(rm_ss)/(pq_pc_ss)*(qs_ss)
     RAT(11) = (pmf_pmd_ss)*(mf_ss)/(md_ss)
     RAT(12) = ((pmf_pc_ss)*(mf_ss)+(pmr_pc_ss)*(rm_ss))/(NGDP_PC_ss)
     RAT(13) = (t_ss)*(pt_pq_ss)/(qs_ss)
     RAT(14) = (pes_pq_ss)*(es_ss)/(qs_ss)
     RAT(15) = (qe_q_ss)*(e_ss)/cstar_bar
     RAT(16) = (qe_q_ss)
     RAT(17) = (s_pc_ss)
     RAT(18) = (pcd_pc_ss)
     RAT(19) = (pmd_pc_ss)
     RAT(20) = (px_pc_ss)
     RAT(21) = (pe_pc_ss)
     RAT(22) = (pq_pc_ss)
     RAT(23) = (pmf_pc_ss)
     RAT(24) = (pmr_pc_ss)
     RAT(25) = (prm_pc_ss)
     RAT(26) = (pxd_pc_ss)
     RAT(27) = (pxd_px_ss)
     RAT(28) = (pmd_px_ss)
     RAT(29) = (pt_pmd_ss)
     RAT(30) = (pmf_pmd_ss)
     RAT(31) = (pmr_pmd_ss)
     RAT(32) = (pcd_pq_ss)
     RAT(33) = (pxd_pq_ss)
     RAT(34) = (pt_pq_ss)
     RAT(35) = (pe_pq_ss)
     RAT(36) = ((s_pc_ss)*tr_bar)/(NGDP_PC_ss)
     RAT(37) = (px_pc_ss)*(k_ss)/(NGDP_PC_ss)
     RAT(38) = (pcds_pcd_ss)
     RAT(39) = (pxds_pxd_ss)
     RAT(40) = (pes_pe_ss)
     RAT(41) = ((pt_pc_ss)/(pcd_pc_ss))*((tcd_ss)/(cd_ss))
     RAT(42) = ((pt_pc_ss)/(pxd_pc_ss))*((txd_ss)/(xd_ss))
     RAT(43) = ((pt_pc_ss)/(pe_pc_ss))*((te_ss)/(e_ss))
     RAT(44) = (pcds_pcd_ss)*((cds_ss)/(cd_ss))
     RAT(45) = (pxds_pxd_ss)*((xds_ss)/(xd_ss))
     RAT(46) = (pes_pe_ss)*((es_ss)/(e_ss))
     RAT(47) = ((cd_ss)*(pcd_pq_ss))/(qs_ss)
     RAT(48) = ((xd_ss)*(pxd_pq_ss))/(qs_ss)
     RAT(49) = ((pe_pq_ss)*(e_ss))/(qs_ss)
     RAT(50) = (pe_pc_ss)*(e_ss/ngdp_pc_ss)  !269
     RAT(51) = (pmd_pc_ss)*(md_ss/ngdp_pc_ss)  !270
     RAT(52) = (pt_pc_ss)*(t_ss/ngdp_pc_ss)  !271
     RAT(53) = (pcd_pc_ss)*(cd_ss/c_ss)  !273
     RAT(54) = (pxd_px_ss)*(xd_ss/x_ss)  !275
     RAT(55) = (pxd_pc_ss)*(xd_ss/ngdp_pc_ss)  !277
     RAT(56) = (pcd_pc_ss)*(cd_ss/ngdp_pc_ss)  !278
     RAT(57) = (pt_pc_ss/pmd_pc_ss)*(t_ss/md_ss)  !279
     RAT(58) = (xm_ss/md_ss)  !281
     RAT(59) = (cm_ss/md_ss)  !282

     DO I=1,NUM_RAT_OBS
        RAT_OBS(I)=RAT(RATIND(I))
     ENDDO
  ENDIF
  !  print *,"h_ss1:",h_ss
  !  print *,"h_ss2:",rat_obs(num_rat_obs)
  !  OPEN (1,FILE="ratsim.txt" )
  !  DO I = 1, 49
  !        write (1, '(ES14.7)'), RAT(I) 
  !   !     write(1,'(1X,200f10.5)') (AMAT(i,j),j=1,UBOUND(AMAT, 2)) 
  !     END DO
  !  CLOSE(1)
  if (.FALSE.) then
     OPEN (1,FILE="ratsimNew.txt" )
     DO I = 1, NUM_RAT_OBS
        write (1, '(ES14.7)'), RAT(RATIND(I)) 
        !     write(1,'(1X,200f10.5)') (AMAT(i,j),j=1,UBOUND(AMAT, 2)) 
     END DO
     CLOSE(1)

     OPEN (1,FILE="SSNew.txt" )
     DO I = 1, 223
        write (1, '(ES14.7)'), ssval(I)
        !     write(1,'(1X,200f10.5)') (AMAT(i,j),j=1,UBOUND(AMAT, 2)) 
     END DO
     CLOSE(1)
  endif
  ! STOP
  RETURN

END SUBROUTINE GETSTEADYSTATE
