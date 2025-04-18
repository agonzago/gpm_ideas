PROGRAM PATACON_V3C
  USE RANDOM
  USE HYBRID_CONST_MOD
  USE MODEL_CONST_MOD  
  USE XINI_MOD    
  USE LIKELIHOOD_CONST_MOD, ONLY : Y, P_INI, A_INI, ZIND, EXOIND, RATIND, CONS_OBS,VAREXOIND,RATMEAN,RATWEIGHT,RAT_ERROR,RAT_OBS
  USE PARAMETER_VECTOR_MOD 
  USE PRIOR_MODULE, ONLY : PRIOR_PAR, DIST
  
  
  

  IMPLICIT NONE
  !  INTEGER, PARAMETER  :: DP = SELECTED_REAL_KIND(14, 60)
  integer i,j,k,DATOS
  EXTERNAL SIMULMODEL
  REAL(4) DELAPSE, TIMEARRAY(2)
  
  REAL(DP) ALFA,ALFAV,CSTAR_BAR,EPSCD,EPSE,EPSM,EPSQ,EPSRM,EPST,EPSW,EPSXD,ETA_M,FBAR,GAMA,&
       &GAMA_CD,GAMA_E,GAMA_M,GAMA_X,GAMA_XD,GBAR,HAB,ISTAR,MU,NBAR,OMEG_S,OMEG_U,OMEGA,OMEGA_CD,&
       &OMEGA_E,OMEGA_M,OMEGA_X,OMEGA_XD,PI_BAR,PI_STAR_BAR,PIB_BAR,PSI_1,PSI_ADJCOST_C,PSI_ADJCOST_X,&
       &PSI_X,QM_BAR,QMR_BAR,RHO_CSTAR,RHO_G,RHO_I,RHO_PI,RHO_PI_STAR,RHO_QM,RHO_QMR,RHO_TR,RHO_Y,&
       &RHO_Z_XDEMAND,RHO_ZCD,RHO_ZD,RHO_ZE,RHO_ZH,RHO_ZI,RHO_ZIE,RHO_ZM,RHO_ZNT,RHO_ZQ,RHO_ZRM,RHO_ZU,&
       &RHO_ZX,RHO_ZXD,RHOQ,RHOQV,SIGMA_M,TBP,TD,THETA,THETA_CD,THETA_E,THETA_M,THETA_RM,THETA_T,&
       &THETA_W,THETA_XD,TR_BAR,UPSILON,V,VC,VE,VN,VNT,VR,VX,Z_XDEMAND_BAR,ZCD_BAR,ZD_BAR,ZE_BAR,&
       &ZH_BAR,ZI_BAR,ZIE_BAR,ZM_BAR,ZNT_BAR,ZQ_BAR,ZRM_BAR,ZU_BAR,ZX_BAR,ZXD_BAR

  REAL(DP) var_cstar,var_G,var_pi_star,var_qm,var_qmr,var_tr,var_z_xdemand,var_zcd,&
           &var_zd,var_ze,var_zh,var_zi,var_zie,var_zm,var_znt,var_zq,var_zrm,var_zu,&
           &var_zx,var_zxd


 ! INTEGER, parameter :: SIZEPAR=int(100)   !Numero de parametros del modelo
 ! INTEGER, parameter :: SIZEVARZ=20 !Numero de varianzas a estimar
  REAL(DP), ALLOCATABLE, DIMENSION(:, :) :: ASIM
  REAL(DP), ALLOCATABLE, DIMENSION(:,:) :: YSIM, PARBOUND, VARBOUND   
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: YSIMVEC
  REAL(DP) LOG_L
  INTEGER INFO, NUM_MEDIDA_EST, NUM_MEDIDA_CTRL, NUM_MEDIDA_EXO
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: CXINI   
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: PAR, PAR_SS_IND
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: FHESSVEC
 ! REAL(DP), ALLOCATABLE, DIMENSION(:) :: VAROUT
 ! REAL(DP), DIMENSION(SIZEPAR*(SIZEPAR + 30)) :: VAROUT
  !REAL(DP), DIMENSION(49) :: RAT 
  ! REAL(DP), ALLOCATABLE, DIMENSION(:) :: MEDIDAMEAN_CTRL, MEDIDAVAR_CTRL, MEDIDAMEAN_EXO, MEDIDAVAR_EXO
  INTEGER, ALLOCATABLE, DIMENSION(:) :: MEDIDA_CTRL_IND, MEDIDA_EXO_IND
!  EXTERNAL ESTIMATEOBJ
  INTEGER PAR_SS, NUM_PAR_SS

  !nteger, parameter ::  SIZEPAR=2
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: step
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: xmin

  INTEGER, PARAMETER :: KONVGE = 100 !INT(SIZEPAR*6)
  INTEGER :: KCOUNT = 5000

  INTEGER RESTART
  REAL(DP) YNEWLO
  REAL(DP), PARAMETER :: REQMIN=1.0E-02
  INTEGER ICOUNT
  INTEGER NUMRES
  INTEGER IFAULT
  REAL(DP) :: EPSFCN = 1.0E-4_DP

  LOGICAL DONE
  INTERFACE
     SUBROUTINE ESTIMATEOBJ(NP, P, FUNC)
       IMPLICIT NONE
       INTEGER, PARAMETER  :: DP = SELECTED_REAL_KIND(12, 60)
       INTEGER ,INTENT(IN) :: NP
       REAL (DP), INTENT(IN)  :: P(NP)
       REAL (DP), INTENT(OUT) :: FUNC
     END SUBROUTINE ESTIMATEOBJ
  END INTERFACE

  REAL(DP), ALLOCATABLE, DIMENSION(:,:) :: FHESS
  REAL(DP) C_SCALE  
  INTEGER ACCEPT
  
  
  INTEGER MCDRAWS
  REAL(DP), ALLOCATABLE , DIMENSION(:, :) :: DRAWS
  !integer num, len, status
  !character*7 value


  REAL(dp), ALLOCATABLE , DIMENSION(:, :) :: POINTS
  INTEGER :: NMCES,NTES,NSTEPES
  DOUBLE PRECISION T0ES
  REAL(DP) :: RHOES
  
  INTEGER NUM_CONS_IND
  INTEGER, ALLOCATABLE, DIMENSION(:) :: CONS_IND
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: PAR_CONS
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: PRIORMEAN
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: FJAC

  CHARACTER(len=20) :: VAL
  CHARACTER(len=50) :: file_out !file_in_par, file_in_seed
  INTEGER NUM_SEEDS
  INTEGER SET_UP, NUMREST

  REAL(DP), ALLOCATABLE, DIMENSION(:,:) :: PAR_SEEDS
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: PARINI
  REAL(DP), ALLOCATABLE, DIMENSION(:,:) :: XINI_SEEDS
  REAL(DP) LOG_L_fin

  CALL GET_COMMAND_ARGUMENT(1, VAL) 
  READ(VAL,'(I10)') NUM_SEEDS 


  CALL SEED_RANDOM_NUMBER() 
  !!MODEL CONSTANTS
  NPAR=100  
  NUM_N=223
  NUM_K=71
  NUM_EXO=20
  NUM_PER=51


 
  NUM_EST=NUM_CNTRL_OBS + NUM_K  !Numero de estados del filtro de Kalman
  !!HIBRID CONSTANTS
  NEQ=13  
  NMC = 1000
  NT = 50
  T0 = 1000
  RHO = 0.998_DP  
  NSTEP = 50
  IPRINT = 1000 
  NUMTRYS  = 10

 
  ALLOCATE(XINI_SEEDS(NEQ, NUM_SEEDS))
  ALLOCATE(XINI(NEQ))
  CALL GET_COMMAND_ARGUMENT(3, VAL)
  OPEN(1, FILE=VAL, STATUS='OLD')
  DO I=1,NEQ
     READ (1,*) (XINI_SEEDS(I,J),J=1,NUM_SEEDS) 
  END DO
  CLOSE(1)
  
  
  ALLOCATE(PARBOUND(NPAR,2))
  ALLOCATE(VARBOUND(NUM_EXO,2))
  ALLOCATE(PRIORMEAN(NPAR+NUM_EXO))
  
 ! ALLOCATE(XINI(NEQ))
  ALLOCATE(CXINI(NEQ))
  ALLOCATE(BL(NEQ))
  ALLOCATE(BU(NEQ))
  ALLOCATE(PARVECTOR(NPAR))
  ALLOCATE(VARVECTOR(NUM_EXO))
  
  
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!! ELECCION DE PARÁMETROS Y VARIABLES PARA INCLUIR EN LA ESTIMACIÓN !!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! PARAMETROS DEL MODELO 
  NUM_PAR_EST=INT(67)
  ALLOCATE(PARIND(NUM_PAR_EST)) !NUMERO DE PARAMETROS A ESTIMAR
 
  PARIND=(/&
      &1,&     !ALFA
      &2,&     !ALFAV
      &3,&     !CSTAR_BAR
     ! &4,&     !EPSCD
      &5,&     !EPSE
      &6,&     !EPSM
      &7,&     !EPSQ
      &8,&     !EPSRM
     ! &9,&     !EPST
      &10,&     !EPSW
     ! &11,&     !EPSXD
      &12,&     !ETA_M
     ! &13,&     !FBAR
      &14,&     !GAMA
      &15,&     !GAMA_CD
      &16,&     !GAMA_E
      &17,&     !GAMA_M
      &18,&     !GAMA_X
      &19,&     !GAMA_XD
     ! &20,&     !GBAR
      &21,&     !HAB
     ! &22,&     !ISTAR
      &23,&     !MU
     ! &24,&     !NBAR
  !    &25,&     !OMEG_S
     ! &26,&     !OMEG_U
      &27,&     !OMEGA
      &28,&     !OMEGA_CD
      &29,&     !OMEGA_E
      &30,&     !OMEGA_M
      &31,&     !OMEGA_X
      &32,&     !OMEGA_XD
   !   &33,&     !PI_BAR
   !   &34,&     !PI_STAR_BAR
  !    &35,&     !PIB_BAR
!      &36,&     !PSI_1
   !   &37,&     !PSI_ADJCOST_C
   !   &38,&     !PSI_ADJCOST_X
      &39,&     !PSI_X
   !   &40,&     !QM_BAR
   !   &41,&     !QMR_BAR
      &42,&     !RHO_CSTAR
      &43,&     !RHO_G
!      &44,&     !RHO_I
!      &45,&     !RHO_PI
      &46,&     !RHO_PI_STAR
      &47,&     !RHO_QM
      &48,&     !RHO_QMR
      &49,&     !RHO_TR
!      &50,&     !RHO_Y
   !   &51,&     !RHO_Z_XDEMAND
   !   &52,&     !RHO_ZCD
   !   &53,&     !RHO_ZD
      &54,&     !RHO_ZE
      &55,&     !RHO_ZH
   !   &56,&     !RHO_ZI
      &57,&     !RHO_ZIE
      &58,&     !RHO_ZM
   !   &59,&     !RHO_ZNT
      &60,&     !RHO_ZQ
      &61,&     !RHO_ZRM
      &62,&     !RHO_ZU
      &63,&     !RHO_ZX
   !   &64,&     !RHO_ZXD
      &65,&     !RHOQ
      &66,&     !RHOQV
      &67,&     !SIGMA_M
   !   &68,&     !TBP
   !   &69,&     !TD
      &70,&     !THETA
      &71,&     !THETA_CD
      &72,&     !THETA_E
      &73,&     !THETA_M
      &74,&     !THETA_RM
      &75,&     !THETA_T
      &76,&     !THETA_W
      &77,&     !THETA_XD
      &78,&     !TR_BAR
!      &79,&     !UPSILON
      &80,&     !V
      &81,&     !VC
      &82,&     !VE
      &83,&     !VN
      &84,&     !VNT
      &85,&     !VR
      &86,&     !VX
    !  &87,&     !Z_XDEMAND_BAR
      &88,&     !ZCD_BAR
      &89,&     !ZD_BAR
      &90,&     !ZE_BAR
      &91,&     !ZH_BAR
    !  &92,&     !ZI_BAR
    !  &93,&     !ZIE_BAR
  !    &94,&     !ZM_BAR
      &95,&     !ZNT_BAR
      &96,&     !ZQ_BAR
      &97,&     !ZRM_BAR
      &98,&     !ZU_BAR
      &99,&     !ZX_BAR
      &100&	!ZXD_BAR
        &/)      !FIN DE PARIND


 !INDICE DE LOS PARÁMETROS DE LARGO PLAZO  
  NUM_PAR_SS=INT(0)
  ALLOCATE(PAR_SS_IND(NUM_PAR_SS))
 ! PAR_SS_IND=(/1,12,23,67,81,86/) 
  
 ! VARIANZAS DE LOS CHOQUES QUE SERÁN ESTIMADAS
  NUM_VAR_EST=INT(15)
  ALLOCATE(VAREXOIND(NUM_VAR_EST))
  if (NUM_VAR_EST > 0) then
  VAREXOIND=(/&
     &1,&  !cstar_t
     &2,&  !G_t
     &3,&  !pi_star_t
     &4,&  !qm_t
     &5,&  !qmr_t
     &6,&  !tr_t
 !    &7,&  !z_xdemand_t
 !    &8,&  !zcd_t
 !    &9,&  !zd_t
     &10,&  !ze_t
     &11,&  !zh_t
     &12,&  !zi_t zmu_t
     &13,&  !zie_t
     &14,&  !zm_t
 !    &15,&  !znt_t
     &16,&  !zq_t
     &17,&  !zrm_t
     &18,&  !zu_t
     &19&  !zx_t
 !    &20&
         &/)  !zxd_t  FIN DE VAREXOIND
  endif   

  ! 1.adjcost_c_t  2.adjcost_x_t  3.c_t  4.cd_t  5.cds_t  6.cdsd_t  7.cm_t  8.D_c_t  9.D_cd_t  10.D_cm_t  
  ! 11.D_cstar_t  12.D_e_t  13.D_f_t  14.D_k_t  15.D_md_t  16.D_ms_t  17.D_ngdp_pc_t  18.D_qd_t  19.D_rm_t  
  ! 20.d_t  21.D_t_t  22.D_tr_t  23.D_w_t  24.D_x_t  25.D_xd_t  26.D_xm_t  27.delta_t  28.e_t  29.es_t  30.esd_t
  ! 31.f1_t  32.f2_t  33.gam_x_t  34.h_t  35.i_t  36.ie_t  37.inf_c_t  38.inf_cd_t  39.inf_e_t  40.inf_md_t 
  ! 41.inf_mf_t  42.inf_q_t  43.inf_rm_t  44.inf_t_t  45.inf_x_t  46.inf_xd_t  47.ks_t  48.lambda_t  
  ! 49.led1_yinf_c_t  50.led2_yinf_c_t  51.led3_yinf_c_t  52.led4_yinf_c_t  53.led5_yinf_c_t  54.led6_yinf_c_t 
  ! 55.md_t  56.mf_t  57.ms_t  58.ngdp_pc_t  59.nt_t  60.pcd_pc_t  61.pcd_pq_t  62.pcds_pq_t  63.pe_pc_t
  ! 64.pe_pq_t  65.pes_pq_t  66.phi_t  67.phicd_t  68.phie_t  69.phimd_t  70.phirm_t  71.phixd_t  72.pi_star_m_t
  ! 73.pmd_pc_t  74.pmd_px_t  75.pmf_pc_t  76.pmf_pmd_t  77.pmr_pc_t  78.pmr_prm_t  79.popt_pq_t  
  ! 80.poptcd_pcd_t  81.popte_pe_t  82.poptmd_pmd_t  83.poptrm_prm_t  84.poptt_pt_t  85.poptxd_pxd_t  
  ! 86.pq_pc_t  87.prm_pc_t  88.prof_cd_t  89.prof_e_t  90.prof_m_t  91.prof_q_t  92.prof_rm_t  93.prof_t_t 
  ! 94.prof_xd_t  95.PSICD_t  96.PSIE_t  97.PSIM_t  98.PSIP_t  99.PSIRM_t  100.PSIT_t  101.PSIXD_t  102.pt_pc_t 
  ! 103.pt_pmd_t  104.pt_pq_t  105.pts_pq_t  106.pvcd_t  107.pve_t  108.pvm_t  109.pvq_t  110.pvrm_t  111.pvt_t
  ! 112.pvxd_t  113.px_pc_t  114.pxd_pc_t  115.pxd_pcd_t  116.pxd_pmd_t  117.pxd_pq_t  118.pxd_px_t  
  ! 119.pxds_pq_t  120.qd_t  121.qe_q_t  122.qs_t  123.rk_t  124.rm_t  125.rmf_t  126.rms_t  127.s_pc_t  
  ! 128.t_t  129.tcd_t  130.te_t  131.THETACD_t  132.THETAE_t  133.THETAM_t  134.THETAP_t  135.THETARM_t  
  ! 136.THETAT_t  137.THETAXD_t  138.tm_t  139.ts_t  140.txd_t  141.u_t  142.vq_t  143.w_t  144.wopt_t  
  ! 145.x_t  146.xd_t  147.xds_t  148.xdsd_t  149.xm_t  150.yi_t  151.yinf_c_t  152.yngdp_pc_t  

  ! Indice de los controles observables


! 8.D_c_t 17.D_ngdp_pc_t 37.inf_c_t 12.D_e_t 24.D_x_t 23.D_w_t 15.D_md_t 19.D_rm_t 11.D_cstar_t 22.D_tr_t              
!  35.i_t 41.inf_mf_t 43.inf_rm_t



  NUM_CNTRL_OBS=INT(13) 
  ALLOCATE(ZIND(NUM_CNTRL_OBS))
  ALLOCATE(MEDIDAMEAN_CTRL(NUM_CNTRL_OBS))
  ALLOCATE(MEDIDAVAR_CTRL(NUM_CNTRL_OBS))


  ZIND=(/8,17,37,12,24,23,15,19,11,22,35,41,43/)
!  ZIND=(/8,17,37,12,24,15,19,11,22,35,41,43/)

!  ZIND=(/145,128,28/)!,35,37,42,55,58,102,59,34,113,114,120,124,127,146/) !169
  ! Errores de medida por estimar (medias y varianzas de las prior de las varianzas)

  MEDIDAMEAN_CTRL=1.0e-12
  MEDIDAVAR_CTRL=0.0

! MEDIDAMEAN_CTRL=(/1.0e-12,  1.0e-12, 1.0e-12/) !, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12,&
                   !&1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12,  1.0e-12,  1.0e-12/) !,  1.0e-12,&
                   !&1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12/)
!  MEDIDAVAR_CTRL= (/0,  0, 0 /)!, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0/)!, 0,&
                   !&0,  0, 0, 0, 0/)
 ! ZIND=(/3/) !169


  ! Indice de los estados observables

  ! 1.lag1_c_t  2.lag1_cd_t  3.lag1_cm_t  4.lag1_e_t  5.lag1_ie_t  6.lag1_inf_c_t  7.lag2_inf_c_t  
  ! 8.lag3_inf_c_t  9.lag1_inf_cd_t  10.lag1_inf_e_t  11.lag1_inf_md_t  12.lag1_inf_q_t  13.lag1_inf_rm_t  
  ! 14.lag1_inf_t_t  15.lag1_inf_xd_t  16.lag1_md_t  17.lag1_ms_t  18.lag1_ngdp_pc_t  19.lag2_ngdp_pc_t  
  ! 20.lag3_ngdp_pc_t  21.lag1_pcd_pc_t  22.lag1_pcd_pq_t  23.lag1_pe_pq_t  24.lag1_pmd_pc_t  25.lag1_prm_pc_t  
  ! 26.lag1_pt_pq_t  27.lag1_pvcd_t  28.lag1_pve_t  29.lag1_pvm_t  30.lag1_pvq_t  31.lag1_pvrm_t  32.lag1_pvt_t 
  ! 33.lag1_pvxd_t  34.lag1_px_pc_t  35.lag1_pxd_pcd_t  36.lag1_qd_t  37.lag1_rm_t  38.lag1_s_pc_t  39.lag1_t_t 
  ! 40.lag1_w_t  41.lag1_x_t  42.lag1_xd_t  43.lag1_xm_t  44.lag1_yi_t  45.lag1_cstar_t  46.lag1_qm_t  
  ! 47.lag1_tr_t  48.lag1_f_t  49.lag1_k_t  50.f_t  51.k_t  52.cstar_t  53.G_t  54.pi_star_t  55.qm_t  56.qmr_t  
  ! 57.tr_t  58.z_xdemand_t  59.zcd_t  60.zd_t  61.ze_t  62.zh_t  63.zi_t  64.zie_t  65.zm_t  66.znt_t  67.zq_t 
  ! 68.zrm_t  69.zu_t  70.zx_t  71.zxd_t  
   ! OJO: LA LISTA DE ARRIBA DEBE CONTENER SOLO LOS ESTADOS SEGUN COMO HAYAN SIDO ORDENADOS
  ! EN LA SOLUCION DEL MODELO (EN LA MATRIZ P)
  
  NUM_EST_OBS=INT(1)
  ALLOCATE(EXOIND(NUM_EST_OBS)) !OJO: QUE PASA SI NUM_EST_OBS=0
  ALLOCATE(MEDIDAMEAN_EXO(NUM_EST_OBS))
  ALLOCATE(MEDIDAVAR_EXO(NUM_EST_OBS))
  EXOIND=         (/54/)
  MEDIDAMEAN_EXO= (/1.0e-12/)
  MEDIDAVAR_EXO=  (/0/)
 ! RAZONES DE LARGO PLAZO PARA INCLUIR COMO OBSERVABLES EN LA ECUACIÓN DE MEDIDA

 ! RAT(1)=0.2152		!'(px_pc*x/y)'
 ! RAT(2)=0.3639		!'(pmd_px*xm/x)'
 ! RAT(3)=0.1178		!'(xds*pxds_pq)/qs'
 ! RAT(4)=0.7996		!'c/y'
 ! RAT(5)=0.601		!'(cds*pcds_pq)/qs'
 ! RAT(6)=0.1193		!'((cm/c)*pmd_pc)'
 ! RAT(7)=0.9821		!'y'		
 ! RAT(8)=3.6164		!'w'
 ! RAT(9)=0.2981		!'h'		
 ! RAT(10)=0.0956	!'(prm_pc*rm)/(pq_pc*qs)'
 ! RAT(11)=0.7299	!'(pmf_pmd*mf/md)'
 ! RAT(12)=0.233		!'(pmf_pc*mf + pmr_pc*rm)/y'				
 ! RAT(13)=0.0968	!'(t*pt_pq)/qs'	
 ! RAT(14)=0.1726	!'(pes_pq*es)/qs'
 ! RAT(15)=1.0322	!'(qe_q*e)/cstar_bar'
 ! RAT(16)=0.9835	!'qe_q'		
 ! RAT(17)=1.1908	!'s_pc'
 ! RAT(18)=1.0151	!'pcd_pc'
 ! RAT(19)=0.8591	!'pmd_pc'
 ! RAT(20)=1.1871	!'px_pc'
 ! RAT(21)=1.1861	!'pe_pc'
 ! RAT(22)=1.0861	!'pq_pc'
 ! RAT(23)=1.0039	!'pmf_pc'
 ! RAT(24)=1.202		!'pmr_pc'
 ! RAT(25)=1.202		!'prm_pc'
 ! RAT(26)=1.2994	!'pxd_pc'
 ! RAT(27)=1.1114	!'pxd_px'
 ! RAT(28)=0.7227	!'pmd_px'
 ! RAT(29)=1.0976	!'pt_pmd'
 ! RAT(30)=1.1723	!'pmf_pmd'
 ! RAT(31)=1.4136	!'pmr_pmd'
 ! RAT(32)=0.9347	!'pcd_pq'
 ! RAT(33)=1.2141	!'pxd_pq'
 ! RAT(34)=0.8623	!'pt_pq'
 ! RAT(35)=1.0921	!'pe_pq'
 ! RAT(36)=0.0351	!'(s_pc*tr)/y '				
 ! RAT(37)=6.8248	!'(px_pc)*k/y'
 ! RAT(38)=1.1032	!'pcds_pcd'
 ! RAT(39)=1.0677	!'pxds_pxd'
 ! RAT(40)=1.185		!'pes_pe'
 ! RAT(41)=0.0568	!'((pt_pcd)*tcd)_cd'
 ! RAT(42)=0.044		!'((pt_pxd)*txd)_xd'
 ! RAT(43)=0.125		!'((pt_pe)*te)_e'
 ! RAT(44)=0.9435	!'((pcds_pcd)*cds)_cd'
 ! RAT(45)=0.9567	!'((pxds_pxd)*xds)_xd'
 ! RAT(46)=0.8762	!'((pes_pe)*es)_e'
 ! RAT(47)=0.6369	!'(cd*pcd_pq)/qs'
 ! RAT(48)=0.1231	!'(xd*pxd_pq)/qs'
 ! RAT(49)=0.197		!'(pe_pq*e)/qs'



!  1. px_pc_t*x_t/y_t 2. pmd_px_t*xm_t/x_t 3. xd_t*pxd_pq_t/qs_t 4. c_t/y_t 5. cd_t*pcd_pq_t/qs_t 6. (cm_t/c_t)*pmd_pc_t 7. NGDP_T
!  8. w_t 9. h_t 10. prm_pc_t*rm_t/pq_pc_t*qs_t 11. pmf_pmd_t*mf_t/md_t 12. pmf_pc_t*mf_t+pmr_pc_t*rm_t/y_t 13. t_t*pt_pq_t/qs_t
!  14. pes_pq_t*es_t/qs_t 15. qe_q_t*e_t/cstar_bar 16. qe_q_t 17. s_pc_t 18. pcd_pc_t 19. pmd_pc_t 20. px_pc_t 21. pe_pc_t
!  22. pq_pc_t 23. pmf_pc_t 24. pmr_pc_t 25. prm_pc_t 26. pxd_pc_t 27. pxd_px_t 28. pmd_px_t 29. pt_pmd_t 30. pmf_pmd_t
!  31. pmr_pmd_t 32. pcd_pq_t 33. pxd_pq_t 34. pt_pq_t 35. pe_pq_t 36. (s_pc_t*tr_bar)/y_t 37. px_pc_t*k_t/y_t 38. pcds_pcd_t
!  39. pxds_pxd_t 40. pes_pe_t 41. (pt_pc_t/pcd_pc_t)*(tcd_t/cd_t) 42. (pt_pc_t/pxd_pc_t)*(txd_t/xd_t) 43. (pt_pc_t/pe_pc_t)*(te_t/e_t)
!  44. pcds_pcd_t*(cds_t/cd_t) 45. pxds_pxd_t*(xds_t/xd_t) 46. pes_pe_t*(es_t/e_t) 47. (cd_t*pcd_pq_t)/qs_t 48. (xd_t*pxd_pq_t)/qs_t
!  49. (pe_pq_t*e_t)/qs_t 50. (pe_pc_ss)*(e_ss/ngdp_pc_ss) 51. (pmd_pc_ss)*(md_ss/ngdp_pc_ss) 52. (pt_pc_ss)*(t_ss/ngdp_pc_ss)
!  53. (pcd_pc_ss)*(cd_ss/c_ss) 54. (pxd_px_ss)*(xd_ss/x_ss) 55. (pxd_pc_ss)*(xd_ss/ngdp_pc_ss) 56. (pcd_pc_ss)*(cd_ss/ngdp_pc_ss)
!  57. (pt_pc_ss/pmd_pc_ss)*(t_ss/md_ss) 58. (xm_ss/md_ss) 59. (cm_ss/md_ss)

!  RAZONES INCLUIDAS EN LOS DATOS: 
!   267-c_ngdp           268-x_ngdp      269-e_ngdp       270-md_ngdp
!   271-t_ngdp           272-s*tr_ngdp   273-cd_c         274-cm_c
!   275-xd_x             276-xm_x        277-xd_ngdp      278-cd_ngdp
!   279-t_md             280-mf_md       281-xm_md        282-cm_md


 ! Indice de las razones (EL ORDEN TIENE QUE COINCIDIR CON LOS DATOS)
   NUM_RAT_OBS=INT(17)
   ALLOCATE(RATIND(NUM_RAT_OBS))
   ALLOCATE(RATMEAN(NUM_RAT_OBS))
   ALLOCATE(RATWEIGHT(NUM_RAT_OBS))
   ALLOCATE(RAT_OBS(NUM_RAT_OBS))
   RATIND=(/4,1,50,51,52,36,53,6,54,2,55,56,57,11,58,59,9/)
   RATWEIGHT=(/1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0/)
   RATWEIGHT=100.0*RATWEIGHT






! CONSTANTES PARA LAS VARIABLES OBSERVABLES

  ! 8.D_c_t : 9.537427866308512e-004;
  !11.D_cstar_t : -9.649445147404956e-004;
  !12.D_e_t : 0.003880011913308;
  !15.D_md_t : 0.034027385095362;
  !17.D_ngdp_pc_t : 0.005191786117809;
  !19.D_rm_t : 0.007211737946478;
  !22.D_tr_t : 0.023222473326511;
  !23.D_w_bar : -0.003111602793206;
  !24.D_x_t : 0.035911890928105;




!  24.D_x_bar             =        0.001107485959037
!  12.D_e_bar             =        0.003776493222266
!  11.D_cstar_bar         =       -0.001615578925595
!   8.D_c_bar             =       -0.001568339836461
!  17.D_ngdp_pc_bar       =       -1.289427779849195e-005
!  23.D_w_bar             =       -0.007209086165934
!  22.D_tr_bar            =        0.033726368308278
!  15.D_md_bar            =        0.008864749601373
!  19.D_rm_bar            =       -0.004070466657969
!   9.D_cd_bar            =       -0.002221214060781
!  10.D_cm_bar            =        0.004671369605997
!  18.D_qd_bar            =       -0.001829647217888
!  21.D_t_bar             =       -0.001956523124283
!  25.D_xd_bar            =       -0.003613854535187
!  26.D_xm_bar            =        0.013869080573012




  NUM_CONS_IND=INT(9)
  ALLOCATE(CONS_IND(NUM_CONS_IND))
  ALLOCATE(PAR_CONS(NUM_CONS_IND))
 ! CONS_IND=(/24,12,11,8,17,23,22,15,19,9,10,18,21,25,26/)
   CONS_IND=(/24,12,11,8,17,23,22,15,19/)
!  CONS_IND=(/24,12,11,8,17,22,15,19/)
  PAR_CONS=(/  log(1.0+0.001107485959037_DP),&
              &log(1.0+0.003776493222266_DP),&
              &log(1.0-0.001615578925595_DP),&
              &log(1.0-0.001568339836461_DP),&
              &log(1.0-1.28942777984919E-05_DP),&
              &log(1.0-0.007209086165934_DP),&
              &log(1.0+0.033726368308278_DP),&
              &log(1.0+0.008864749601373_DP),&
              &log(1.0-0.004070466657969_DP)&
!              &log(1.0-0.002221214060781_DP),&
!              &log(1.0+0.004671369605997_DP),&
!              &log(1.0-0.001829647217888_DP),&
!              &log(1.0-0.001956523124283_DP),&
!              &log(1.0-0.003613854535187_DP),&
!              &log(1.0+0.013869080573012_DP)& 
              &/)

!  do i=1,9
!     print *,PAR_CONS(I)
!  ENDDO
  

 ! NUM_OBS=NUM_CNTRL_OBS + NUM_EST_OBS + NUM_RAT_OBS !TOTAL DEL NUMERO DE VARIABLES OBSERVABLES
  NUM_OBS=NUM_CNTRL_OBS + NUM_EST_OBS
  ALLOCATE(CONS_OBS(NUM_OBS))
  CONS_OBS=0.0
  DO I=1,NUM_CONS_IND
     DO J=1,NUM_CNTRL_OBS
        IF (CONS_IND(I) .EQ. ZIND(J)) THEN
            CONS_OBS(J) = PAR_CONS(I)
        ENDIF
     ENDDO
  ENDDO

!  CONS_OBS(I) = ???
!  CONS_OBS(J) = ???
! VALORES INICIALES PARA LOS PARÁMETROS

  !Parameter INITIAL values

  ! ESTIMACIÓN DIEGO 20/10/2009

  rhoq = 0.620968926962017_dp
  rhoqv = 0.713218910066946_dp
  alfa = 0.90280881797474_dp
  alfav = 0.598709167285649_dp
  v = 2.68600280659395_dp
  vn = 1.23201847939711_dp
  ve = 2.98662390505271_dp
  vnt = 0.241837440163619_dp
  vc = 2.38818353099015_dp
  vr = 4.51544687949928_dp
  vx = 39.987224153117_dp
  epsq = 0.75_dp
  epsw = 0.85_dp
  epse = 0.75_dp
  epsm = 0.75_dp
  epsrm = 0.75_dp
  epst = 0.75_dp
  epscd = epsq
  epsxd = epsq
  eta_m = 1.5_dp!3.19_dp
  sigma_m = 1.2_dp
  hab = 0.5_dp!0.1_dp
  tbp = 0.536963044063132_dp
  td = 0.13438005_dp
  gama_cd = 0.939812975319728_dp
  gama_e = 0.85755510127928_dp
  gama_m = 0.804731900155345_dp
!  gama_rm = 0.0103318822710792_dp
  gama_x = 0.566810887721729_dp
  gama_xd = 0.948220895932729_dp
  gama = 0.877285721950229_dp
  omega_cd = 0.715097255518238_dp
  omega_e = 0.647696073657445_dp
  omega_m = 0.107898894632587_dp
!  omega_rm = 0.100000234673128_dp
  omega_x = 0.449441398423515_dp
  omega_xd = 0.591013300918052_dp
  omega = 1.01369726080713_dp
  theta_cd = 39.9996351819494_dp
  theta_e = 39.9704487332371_dp
  theta_m = 14.6695560206419_dp
  theta_rm = 39.9963207335535_dp
  theta_t = 18.0305604924214_dp
  theta_w = 1.22906921951689_dp
  theta_xd = 39.9997035970135_dp
  theta = 4.56649716431929_dp
  gbar = 0.00598400211079619_dp
  nbar = 0.00304572904204794_dp
  mu = 2.19121677719709_dp
  OMEG_S = 0_dp
  OMEG_U = 0.0862_dp
  rho_i = 0.7_dp
  rho_pi = 2.5_dp
  rho_y = 0.8_dp
  rho_cstar = 0.75_dp
  rho_qm = 0.75_dp
  rho_qmr = 0.75_dp
  rho_zie = 0.75_dp
  rho_tr = 0.75_dp
  rho_pi_star = 0.75_dp
  rho_zx = 0.75_dp
  rho_ze = 0.75_dp
  rho_zh = 0.75_dp
  rho_zm = 0.75_dp
  rho_zq = 0.75_dp
  rho_zrm = 0.75_dp
  rho_zu = 0.75_dp
  rho_g = 0.75_dp
  rho_z_xdemand = 0_dp
  rho_zd = 0_dp
  rho_zi = 0_dp
  rho_znt = 0_dp
  rho_zxd = 0_dp
  rho_zcd = 0_dp
!  rho_n = 0_dp
!  rho_zv = 0_dp
  tr_bar = 0.0285613216705455_dp
  cstar_bar = 0.16647808281856_dp
  qm_bar = 0.8398_dp
  qmr_bar = 1.05640216964348_dp
  z_xdemand_bar = 1_dp
  zcd_bar = 1.12107972181441_dp
  zd_bar = 2.00387372737643_dp
  ze_bar = 1.15580617283505_dp
  zh_bar = 180.435332248903_dp
  zie_bar = 1_dp
  zi_bar = 1_dp
  zm_bar = 1.22279584800781_dp
  znt_bar = 1.26957417381715_dp
  zq_bar = 0.620903086275895_dp
  zrm_bar = 0.886969703469514_dp
!  zrms_bar = 0.16975493681612_dp
  zu_bar = 0.335275953406049_dp
!  zv_bar = 1_dp
  zx_bar = 0.932072646124591_dp
  zxd_bar = 1.07601429508886_dp
  fbar = 1.2_dp
  Psi_1 = 0.00505577051670529_dp
  upsilon = 0.698898065807517_dp
!  tt = 1_dp
  psi_x = 0.85_dp
  psi_adjcost_c = 0_dp
  psi_adjcost_x = 0_dp
  pib_bar = exp(1.63146_dp)
  pi_bar = (1.03_dp)**(0.25)
  pi_star_bar = exp(0.006030908647059_dp)
  istar = 1.011056324974531_dp*pi_star_bar
!  bettastar = (pi_star_bar)/(istar)
!  betta = bettastar*(1+gbar)^(sigma_m)
!  Psi_2 = ((1+upsilon)/upsilon)*((1-betta*(1+gbar)**(-sigma_m)*(1-Psi_1))/(betta*(1+gbar)**(-sigma_m)))


  epsq = 0.5_dp
  epsw = 0.5_dp
  epse = 0.5_dp
  epsm = 0.5_dp
  epsrm = 0.5_dp


  rho_cstar = 0.5_dp 
  rho_qm = 0.5_dp
  rho_qmr = 0.5_dp
  rho_zie = 0.5_dp
  rho_tr = 0.5_dp
  rho_pi_star = 0.5_dp

  rho_zx = 0.5_dp
  rho_ze = 0.5_dp
  rho_zh = 0.5_dp
  rho_zm = 0.5_dp
  rho_zq = 0.5_dp
  rho_zrm = 0.5_dp
  rho_zu = 0.5_dp
  rho_g = 0.5_dp

   var_cstar=0.00015625_dp
   var_G=0.00015625_dp
   var_pi_star=0.00015625_dp
   var_qm=0.00015625_dp
   var_qmr=0.00015625_dp
   var_tr=0.00015625_dp
   var_z_xdemand=0.0_dp
   var_zcd=0.0_dp
   var_zd=0.0_dp
   var_ze=0.00015625_dp
   var_zh=0.00015625_dp
   var_zi=0.00015625_dp
   var_zie=0.00015625_dp
   var_zm=0.00015625_dp
   var_znt=0.0_dp
   var_zq=0.00015625_dp
   var_zrm=0.00015625_dp
   var_zu=0.00015625_dp
   var_zx=0.00015625_dp
   var_zxd=0.0_dp



 PIB_BAR = 1
! BOUNDS PARA LOS PARAMETROS
! Lower Bounds
  PARBOUND(1,1) = 0.1      !ALFA
  PARBOUND(2,1) = 0.1      !ALFAV
  PARBOUND(3,1) = 0.01      !CSTAR_BAR
  PARBOUND(4,1) = 0.00      !EPSCD
  PARBOUND(5,1) = 0.00      !EPSE
  PARBOUND(6,1) = 0.00      !EPSM
  PARBOUND(7,1) = 0.00      !EPSQ
  PARBOUND(8,1) = 0.00      !EPSRM
  PARBOUND(9,1) = 0.00      !EPST
  PARBOUND(10,1) = 0.00      !EPSW
  PARBOUND(11,1) = 0.00      !EPSXD
  PARBOUND(12,1) = 0.5      !ETA_M
  PARBOUND(13,1) = 0.5      !FBAR
  PARBOUND(14,1) = 0.6      !GAMA
  PARBOUND(15,1) = 0.01      !GAMA_CD
  PARBOUND(16,1) = 0.01      !GAMA_E
  PARBOUND(17,1) = 0.01      !GAMA_M
  PARBOUND(18,1) = 0.01      !GAMA_X
  PARBOUND(19,1) = 0.01      !GAMA_XD
  PARBOUND(20,1) = 0.005      !GBAR
  PARBOUND(21,1) = 0      !HAB
  PARBOUND(22,1) = ISTAR - 0.01_dp      !ISTAR
  PARBOUND(23,1) = 1.5      !MU
  PARBOUND(24,1) = 0.003      !NBAR
  PARBOUND(25,1) = 0.000      !OMEG_S
  PARBOUND(26,1) = 0.001      !OMEG_U
  PARBOUND(27,1) = 0.1      !OMEGA
  PARBOUND(28,1) = 0.1      !OMEGA_CD
  PARBOUND(29,1) = 0.1      !OMEGA_E
  PARBOUND(30,1) = 0.1      !OMEGA_M
  PARBOUND(31,1) = 0.1      !OMEGA_X
  PARBOUND(32,1) = 0.0      !OMEGA_XD
  PARBOUND(33,1) = pi_bar - 0.01_dp      !PI_BAR
  PARBOUND(34,1) = 1      !PI_STAR_BAR
  PARBOUND(35,1) = 0.001      !PIB_BAR
  PARBOUND(36,1) = 0.000001      !PSI_1
  PARBOUND(37,1) = 0.0      !PSI_ADJCOST_C
  PARBOUND(38,1) = 0.0      !PSI_ADJCOST_X
  PARBOUND(39,1) = 0.0      !PSI_X
  PARBOUND(40,1) = 0.3      !QM_BAR
  PARBOUND(41,1) = 0.3      !QMR_BAR
  PARBOUND(42,1) = 0.00      !RHO_CSTAR
  PARBOUND(43,1) = 0.00      !RHO_G
  PARBOUND(44,1) = 0.00      !RHO_I
  PARBOUND(45,1) = 1.00      !RHO_PI
  PARBOUND(46,1) = 0.00      !RHO_PI_STAR
  PARBOUND(47,1) = 0.00      !RHO_QM
  PARBOUND(48,1) = 0.00      !RHO_QMR
  PARBOUND(49,1) = 0.00      !RHO_TR
  PARBOUND(50,1) = 0.00      !RHO_Y
  PARBOUND(51,1) = 0.00      !RHO_Z_XDEMAND
  PARBOUND(52,1) = 0.00      !RHO_ZCD
  PARBOUND(53,1) = 0.00      !RHO_ZD
  PARBOUND(54,1) = 0.00      !RHO_ZE
  PARBOUND(55,1) = 0.00      !RHO_ZH
  PARBOUND(56,1) = 0.00      !RHO_ZI
  PARBOUND(57,1) = 0.00      !RHO_ZIE
  PARBOUND(58,1) = 0.00      !RHO_ZM
  PARBOUND(59,1) = 0.00      !RHO_ZNT
  PARBOUND(60,1) = 0.00      !RHO_ZQ
  PARBOUND(61,1) = 0.00      !RHO_ZRM
  PARBOUND(62,1) = 0.00      !RHO_ZU
  PARBOUND(63,1) = 0.00      !RHO_ZX
  PARBOUND(64,1) = 0.00      !RHO_ZXD
  PARBOUND(65,1) = 0.1      !RHOQ
  PARBOUND(66,1) = 0.1      !RHOQV
  PARBOUND(67,1) = 0.1      !SIGMA_M
  PARBOUND(68,1) = 0.53      !TBP
  PARBOUND(69,1) = 0.13_dp      !TD
  PARBOUND(70,1) = 1.5      !THETA
  PARBOUND(71,1) = 1.5      !THETA_CD
  PARBOUND(72,1) = 1.5      !THETA_E
  PARBOUND(73,1) = 1.1      !THETA_M
  PARBOUND(74,1) = 1.1      !THETA_RM
  PARBOUND(75,1) = 1.5      !THETA_T
  PARBOUND(76,1) = 1.1      !THETA_W
  PARBOUND(77,1) = 1.5      !THETA_XD
  PARBOUND(78,1) = 0.01      !TR_BAR
  PARBOUND(79,1) = 0.001      !UPSILON
  PARBOUND(80,1) = 2      !V
  PARBOUND(81,1) = 0.2      !VC
  PARBOUND(82,1) = 0.2      !VE
  PARBOUND(83,1) = 1      !VN
  PARBOUND(84,1) = 0.2      !VNT
  PARBOUND(85,1) = 0.2      !VR
  PARBOUND(86,1) = 0.2      !VX
  PARBOUND(87,1) = 0.1      !Z_XDEMAND_BAR
  PARBOUND(88,1) = 0.1      !ZCD_BAR
  PARBOUND(89,1) = 0.1      !ZD_BAR
  PARBOUND(90,1) = 0.1      !ZE_BAR
  PARBOUND(91,1) = 15      !ZH_BAR
  PARBOUND(92,1) = 0.1      !ZI_BAR
  PARBOUND(93,1) = 0.1      !ZIE_BAR
  PARBOUND(94,1) = 0.1      !ZM_BAR
  PARBOUND(95,1) = 0.08      !ZNT_BAR
  PARBOUND(96,1) = 0.1      !ZQ_BAR
  PARBOUND(97,1) = 0.1      !ZRM_BAR
  PARBOUND(98,1) = 0.1      !ZU_BAR
  PARBOUND(99,1) = 0.1      !ZX_BAR
  PARBOUND(100,1) = 0.1      !ZXD_BAR


! varianzas de los choques
  VARBOUND(1,1)= 1E-6		!cstar
  VARBOUND(2,1)= 1E-6		!G
  VARBOUND(3,1)= 1E-6		!var_pi_star
  VARBOUND(4,1)= 1E-6		!var_qm
  VARBOUND(5,1)= 1E-6		!var_qmr
  VARBOUND(6,1)= 1E-6		!var_tr
  VARBOUND(7,1)= 1E-6		!var_z_xdemand
  VARBOUND(8,1)= 1E-6		!var_zcd
  VARBOUND(9,1)= 1E-6		!var_zd
  VARBOUND(10,1)= 1E-6		!var_ze
  VARBOUND(11,1)= 1E-6		!var_zh
  VARBOUND(12,1)= 1E-6		!var_zi
  VARBOUND(13,1)= 1E-6		!var_zie
  VARBOUND(14,1)= 1E-6		!var_zm
  VARBOUND(15,1)= 1E-6		!var_znt
  VARBOUND(16,1)= 1E-6		!var_zq
  VARBOUND(17,1)= 1E-6		!var_zrm
  VARBOUND(18,1)= 1E-6		!var_zu
  VARBOUND(19,1)= 1E-6		!var_zx
  VARBOUND(20,1)= 1E-6		!var_zxd


 !Uper Bounds
  PARBOUND(1,2) = 0.99      !ALFA
  PARBOUND(2,2) = 0.99      !ALFAV
  PARBOUND(3,2) = 5      !CSTAR_BAR
  PARBOUND(4,2) = 1.0      !EPSCD
  PARBOUND(5,2) = 1.0      !EPSE
  PARBOUND(6,2) = 1.0      !EPSM
  PARBOUND(7,2) = 1.0      !EPSQ
  PARBOUND(8,2) = 1.0      !EPSRM
  PARBOUND(9,2) = 1.0      !EPST
  PARBOUND(10,2) = 1.0      !EPSW
  PARBOUND(11,2) = 1.0      !EPSXD
  PARBOUND(12,2) = 5      !ETA_M
  PARBOUND(13,2) = 2      !FBAR
  PARBOUND(14,2) = 0.999      !GAMA
  PARBOUND(15,2) = 0.99      !GAMA_CD
  PARBOUND(16,2) = 0.99      !GAMA_E
  PARBOUND(17,2) = 0.99      !GAMA_M
  PARBOUND(18,2) = 0.99      !GAMA_X
  PARBOUND(19,2) = 0.99      !GAMA_XD
  PARBOUND(20,2) = 0.009      !GBAR
  PARBOUND(21,2) = 1      !HAB
  PARBOUND(22,2) = ISTAR + 0.01     !ISTAR
  PARBOUND(23,2) = 20      !MU
  PARBOUND(24,2) = 0.0031      !NBAR
  PARBOUND(25,2) = 0.2      !OMEG_S
  PARBOUND(26,2) = 0.2      !OMEG_U
  PARBOUND(27,2) = 3      !OMEGA
  PARBOUND(28,2) = 2      !OMEGA_CD
  PARBOUND(29,2) = 2      !OMEGA_E
  PARBOUND(30,2) = 2      !OMEGA_M
  PARBOUND(31,2) = 2      !OMEGA_X
  PARBOUND(32,2) = 2      !OMEGA_XD
  PARBOUND(33,2) = PI_BAR+0.01      !PI_BAR
  PARBOUND(34,2) = 1.1      !PI_STAR_BAR
  PARBOUND(35,2) = 50.0      !PIB_BAR
  PARBOUND(36,2) = 0.1      !PSI_1
  PARBOUND(37,2) = 2      !PSI_ADJCOST_C
  PARBOUND(38,2) = 2      !PSI_ADJCOST_X
  PARBOUND(39,2) = 1      !PSI_X
  PARBOUND(40,2) = 3      !QM_BAR
  PARBOUND(41,2) = 3      !QMR_BAR
  PARBOUND(42,2) = 1.0      !RHO_CSTAR
  PARBOUND(43,2) = 1.0      !RHO_G
  PARBOUND(44,2) = 1.00      !RHO_I
  PARBOUND(45,2) = 4.00      !RHO_PI
  PARBOUND(46,2) = 1.0      !RHO_PI_STAR
  PARBOUND(47,2) = 1.0      !RHO_QM
  PARBOUND(48,2) = 1.0      !RHO_QMR
  PARBOUND(49,2) = 1.0      !RHO_TR
  PARBOUND(50,2) = 1.50      !RHO_Y
  PARBOUND(51,2) = 1.0      !RHO_Z_XDEMAND
  PARBOUND(52,2) = 1.0      !RHO_ZCD
  PARBOUND(53,2) = 1.0      !RHO_ZD
  PARBOUND(54,2) = 1.0      !RHO_ZE
  PARBOUND(55,2) = 1.0      !RHO_ZH
  PARBOUND(56,2) = 1.0      !RHO_ZI
  PARBOUND(57,2) = 1.0      !RHO_ZIE
  PARBOUND(58,2) = 1.0      !RHO_ZM
  PARBOUND(59,2) = 1.0      !RHO_ZNT
  PARBOUND(60,2) = 1.0      !RHO_ZQ
  PARBOUND(61,2) = 1.0      !RHO_ZRM
  PARBOUND(62,2) = 1.0      !RHO_ZU
  PARBOUND(63,2) = 1.0      !RHO_ZX
  PARBOUND(64,2) = 1.0      !RHO_ZXD
  PARBOUND(65,2) = 0.98      !RHOQ
  PARBOUND(66,2) = 0.9      !RHOQV
  PARBOUND(67,2) = 4.0      !SIGMA_M
  PARBOUND(68,2) = 0.54      !TBP
  PARBOUND(69,2) = 0.135      !TD
  PARBOUND(70,2) = 20      !THETA
  PARBOUND(71,2) = 50      !THETA_CD
  PARBOUND(72,2) = 50      !THETA_E
  PARBOUND(73,2) = 50      !THETA_M
  PARBOUND(74,2) = 50      !THETA_RM
  PARBOUND(75,2) = 50      !THETA_T
  PARBOUND(76,2) = 50      !THETA_W
  PARBOUND(77,2) = 50      !THETA_XD
  PARBOUND(78,2) = 2      !TR_BAR
  PARBOUND(79,2) = 3      !UPSILON
  PARBOUND(80,2) = 10      !V
  PARBOUND(81,2) = 30      !VC
  PARBOUND(82,2) = 30      !VE
  PARBOUND(83,2) = 20      !VN
  PARBOUND(84,2) = 30      !VNT
  PARBOUND(85,2) = 30      !VR
  PARBOUND(86,2) = 30      !VX
  PARBOUND(87,2) = 4      !Z_XDEMAND_BAR
  PARBOUND(88,2) = 4      !ZCD_BAR
  PARBOUND(89,2) = 4      !ZD_BAR
  PARBOUND(90,2) = 4      !ZE_BAR
  PARBOUND(91,2) = 1500      !ZH_BAR
  PARBOUND(92,2) = 4      !ZI_BAR
  PARBOUND(93,2) = 4      !ZIE_BAR
  PARBOUND(94,2) = 4      !ZM_BAR
  PARBOUND(95,2) = 4      !ZNT_BAR
  PARBOUND(96,2) = 4      !ZQ_BAR
  PARBOUND(97,2) = 4      !ZRM_BAR
  PARBOUND(98,2) = 4      !ZU_BAR
  PARBOUND(99,2) = 4      !ZX_BAR
  PARBOUND(100,2) = 4      !ZXD_BAR


! varianzas de los choques
  VARBOUND(1,2)= 30.0	!cstar
  VARBOUND(2,2)= 30.0	!G
  VARBOUND(3,2)= 30.0	!var_pi_star
  VARBOUND(4,2)= 30.0	!var_qm
  VARBOUND(5,2)= 30.0	!var_qmr
  VARBOUND(6,2)= 30.0	!var_tr
  VARBOUND(7,2)= 30.0	!var_z_xdemand
  VARBOUND(8,2)= 30.0	!var_zcd
  VARBOUND(9,2)= 30.0	!var_zd
  VARBOUND(10,2)= 30.0	!var_ze
  VARBOUND(11,2)= 30.0	!var_zh
  VARBOUND(12,2)= 30.0	!var_zi
  VARBOUND(13,2)= 30.0	!var_zie
  VARBOUND(14,2)= 30.0	!var_zm
  VARBOUND(15,2)= 30.0	!var_znt
  VARBOUND(16,2)= 30.0	!var_zq
  VARBOUND(17,2)= 30.0	!var_zrm
  VARBOUND(18,2)= 30.0	!var_zu
  VARBOUND(19,2)= 30.0	!var_zx
  VARBOUND(20,2)= 30.0	!var_zxd



!SETUP BOUNDS PARA EL SS
  BL = -20.0_DP
  BU = 2.0_DP

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  NUM_EST=NUM_CNTRL_OBS+NUM_K !NUMERO DE ESTADOS DEL FILTRO DE KALMAN

  ALLOCATE(ASIM(NUM_EST, NUM_PER+1))
 ! ALLOCATE(YSIM(NUM_CNTRL_OBS+NUM_EST_OBS, NUM_PER))
  ALLOCATE(YSIMVEC((NUM_CNTRL_OBS+NUM_EST_OBS)*NUM_PER))
  ALLOCATE(Y(NUM_OBS, NUM_PER))
  ALLOCATE(A_INI(NUM_EST)) 
  ALLOCATE(P_INI(NUM_EST, NUM_EST))

  !ASIGNACION DE VALORES INICIALES AL VECTOR DE PARAMETROS
  PARVECTOR(1) = ALFA
  PARVECTOR(2) = ALFAV
  PARVECTOR(3) = CSTAR_BAR
  PARVECTOR(4) = EPSCD
  PARVECTOR(5) = EPSE
  PARVECTOR(6) = EPSM
  PARVECTOR(7) = EPSQ
  PARVECTOR(8) = EPSRM
  PARVECTOR(9) = EPST
  PARVECTOR(10) = EPSW
  PARVECTOR(11) = EPSXD
  PARVECTOR(12) = ETA_M
  PARVECTOR(13) = FBAR
  PARVECTOR(14) = GAMA
  PARVECTOR(15) = GAMA_CD
  PARVECTOR(16) = GAMA_E
  PARVECTOR(17) = GAMA_M
  PARVECTOR(18) = GAMA_X
  PARVECTOR(19) = GAMA_XD
  PARVECTOR(20) = GBAR
  PARVECTOR(21) = HAB
  PARVECTOR(22) = ISTAR
  PARVECTOR(23) = MU
  PARVECTOR(24) = NBAR
  PARVECTOR(25) = OMEG_S
  PARVECTOR(26) = OMEG_U
  PARVECTOR(27) = OMEGA
  PARVECTOR(28) = OMEGA_CD
  PARVECTOR(29) = OMEGA_E
  PARVECTOR(30) = OMEGA_M
  PARVECTOR(31) = OMEGA_X
  PARVECTOR(32) = OMEGA_XD
  PARVECTOR(33) = PI_BAR
  PARVECTOR(34) = PI_STAR_BAR
  PARVECTOR(35) = PIB_BAR
  PARVECTOR(36) = PSI_1
  PARVECTOR(37) = PSI_ADJCOST_C
  PARVECTOR(38) = PSI_ADJCOST_X
  PARVECTOR(39) = PSI_X
  PARVECTOR(40) = QM_BAR
  PARVECTOR(41) = QMR_BAR
  PARVECTOR(42) = RHO_CSTAR
  PARVECTOR(43) = RHO_G
  PARVECTOR(44) = RHO_I
  PARVECTOR(45) = RHO_PI
  PARVECTOR(46) = RHO_PI_STAR
  PARVECTOR(47) = RHO_QM
  PARVECTOR(48) = RHO_QMR
  PARVECTOR(49) = RHO_TR
  PARVECTOR(50) = RHO_Y
  PARVECTOR(51) = RHO_Z_XDEMAND
  PARVECTOR(52) = RHO_ZCD
  PARVECTOR(53) = RHO_ZD
  PARVECTOR(54) = RHO_ZE
  PARVECTOR(55) = RHO_ZH
  PARVECTOR(56) = RHO_ZI
  PARVECTOR(57) = RHO_ZIE
  PARVECTOR(58) = RHO_ZM
  PARVECTOR(59) = RHO_ZNT
  PARVECTOR(60) = RHO_ZQ
  PARVECTOR(61) = RHO_ZRM
  PARVECTOR(62) = RHO_ZU
  PARVECTOR(63) = RHO_ZX
  PARVECTOR(64) = RHO_ZXD
  PARVECTOR(65) = RHOQ
  PARVECTOR(66) = RHOQV
  PARVECTOR(67) = SIGMA_M
  PARVECTOR(68) = TBP
  PARVECTOR(69) = TD
  PARVECTOR(70) = THETA
  PARVECTOR(71) = THETA_CD
  PARVECTOR(72) = THETA_E
  PARVECTOR(73) = THETA_M
  PARVECTOR(74) = THETA_RM
  PARVECTOR(75) = THETA_T
  PARVECTOR(76) = THETA_W
  PARVECTOR(77) = THETA_XD
  PARVECTOR(78) = TR_BAR
  PARVECTOR(79) = UPSILON
  PARVECTOR(80) = V
  PARVECTOR(81) = VC
  PARVECTOR(82) = VE
  PARVECTOR(83) = VN
  PARVECTOR(84) = VNT
  PARVECTOR(85) = VR
  PARVECTOR(86) = VX
  PARVECTOR(87) = Z_XDEMAND_BAR
  PARVECTOR(88) = ZCD_BAR
  PARVECTOR(89) = ZD_BAR
  PARVECTOR(90) = ZE_BAR
  PARVECTOR(91) = ZH_BAR
  PARVECTOR(92) = ZI_BAR
  PARVECTOR(93) = ZIE_BAR
  PARVECTOR(94) = ZM_BAR
  PARVECTOR(95) = ZNT_BAR
  PARVECTOR(96) = ZQ_BAR
  PARVECTOR(97) = ZRM_BAR
  PARVECTOR(98) = ZU_BAR
  PARVECTOR(99) = ZX_BAR
  PARVECTOR(100) = ZXD_BAR

 ! ASIGNACION DE VALORES INICIALES A LAS VARIANZAS DE LOS CHOQUES
  varvector(1)= var_cstar
  varvector(2)= var_G
  varvector(3)= var_pi_star
  varvector(4)= var_qm
  varvector(5)= var_qmr
  varvector(6)= var_tr
  varvector(7)= var_z_xdemand
  varvector(8)= var_zcd
  varvector(9)= var_zd
  varvector(10)= var_ze
  varvector(11)= var_zh
  varvector(12)= var_zi
  varvector(13)= var_zie
  varvector(14)= var_zm
  varvector(15)= var_znt
  varvector(16)= var_zq
  varvector(17)= var_zrm
  varvector(18)= var_zu
  varvector(19)= var_zx
  varvector(20)= var_zxd

! do i=1,100
!    print *,"par",i,parvector(i)
! enddo

! stop

  ! PRUEBA PARA DETERMINAR SI LOS PARAMETROS INICIALES ESTÁN DENTRO DE LOS BOUNDS
!  DO I=1,NPAR
!     IF ((PARVECTOR(I)<PARBOUND(I,1)) .OR. (PARVECTOR(I)>PARBOUND(I,2))) then
!        PRINT *,'El valor inicial del parametro',I, 'no esta dentro de los bounds.'
        !return
!     end if
!  end do

  ! Asignación de la semilla al vector de parámetros


  DELAPSE =DTIME(TIMEARRAY)

  A_INI = 0.0
  P_INI = 0.0
  DO I = 1, NUM_EST
     P_INI(I,i) = 1.0E-4
  ENDDO


 
  CXINI = XINI

  !CALL SIMULMODEL(PARVECTOR, VARVECTOR, CXINI,A_INI, YSIM, ASIM)

 ! return

  !PRINT *,"YSIM=",YSIM
!  CALL DATASIMUL(YSIM)
  
!  IF (NUM_CNTRL_OBS + NUM_EST_OBS .GT. 0) THEN
!     DO I=1,NUM_CNTRL_OBS + NUM_EST_OBS
!        Y(I,:)=YSIM(I,:)
!     ENDDO
!  ENDIF


  open(1, FILE="dataparranew.txt", status='OLD')
  READ(1, *) (YSIMVEC(J), J=1, int(NUM_PER*(NUM_CNTRL_OBS + NUM_EST_OBS)))
  CLOSE(1)

  DO I=1, NUM_CNTRL_OBS + NUM_EST_OBS
     DO J=1, NUM_PER
        Y(I,J)=YSIMVEC(INT((I-1)*NUM_PER + J))
     ENDDO
  ENDDO

  !Adición de las razones de largo plazo al vector de variables observables

  open(2, FILE="razonesparraMean.txt", status='OLD')
  READ(2, *) (RATMEAN(J), J=1, int(NUM_RAT_OBS))
  CLOSE(2)

 ! DO I = NUM_CNTRL_OBS + NUM_EST_OBS + 1, NUM_OBS
 !    K = I - (NUM_CNTRL_OBS + NUM_EST_OBS)
 !    DO J=1, NUM_PER
 !       Y(I,J)=RATMEAN(INT((K-1)*NUM_PER + J))
 !    ENDDO
 ! ENDDO

  
  !NUMERO DE ERRORES DE MEDIDA A ESTIMAR
  NUM_MEDIDA_CTRL=0.0
  DO I=1, NUM_CNTRL_OBS
     IF (MEDIDAVAR_CTRL(I) .NE. 0.0) THEN
        NUM_MEDIDA_CTRL=NUM_MEDIDA_CTRL+1
     ENDIF
  ENDDO
  NUM_MEDIDA_EXO=0.0
  DO I=1, NUM_EST_OBS
     IF (MEDIDAVAR_EXO(I) .NE. 0.0) THEN
        NUM_MEDIDA_EXO=NUM_MEDIDA_EXO+1
     ENDIF
  ENDDO
  NUM_MEDIDA_EST= NUM_MEDIDA_CTRL + NUM_MEDIDA_EXO
  
  !CONSTRUCCION DE UN INDICE PARA LOS ERRORES DE MEDIDA A ESTIMAR

  ALLOCATE(MEDIDA_CTRL_IND(NUM_MEDIDA_CTRL))
  ALLOCATE(MEDIDA_EXO_IND(NUM_MEDIDA_EXO))
  J=INT(0)
  DO I=1, NUM_CNTRL_OBS
     IF (MEDIDAVAR_CTRL(I) .NE. 0.0) THEN
        J=J+1
        MEDIDA_CTRL_IND(J)=I
     ENDIF
  ENDDO

  J=INT(0)
  DO I=1, NUM_EST_OBS
     IF (MEDIDAVAR_EXO(I) .NE. 0.0) THEN
        J=J+1
        MEDIDA_EXO_IND(J)=I
     ENDIF
  ENDDO

  !CONSTRUCCION DEL VECTOR PAR QUE INCLUYE TODOS LOS PARAMETROS DE LA ESTIMACION

  ESTIMSIZE=NUM_PAR_EST + NUM_VAR_EST + NUM_MEDIDA_EST
  ALLOCATE(PAR(ESTIMSIZE))
  ALLOCATE(FJAC(ESTIMSIZE))

  ALLOCATE(PAR_SEEDS(ESTIMSIZE,NUM_SEEDS))
  ALLOCATE(PARINI(ESTIMSIZE))
  CALL GET_COMMAND_ARGUMENT(2, VAL)
  OPEN(2, FILE=VAL, STATUS='OLD')
  DO I=1,ESTIMSIZE
     READ (2,*) (PAR_SEEDS(I,J),J=1,NUM_SEEDS)
  END DO 
  CLOSE(2)
  
  CALL GET_COMMAND_ARGUMENT(4, VAL)
  READ(VAL,'(I10)') SET_UP
  print*, SET_UP
  XINI = XINI_SEEDS(:,SET_UP)
  PARINI = PAR_SEEDS(:,SET_UP)

  deallocate(XINI_SEEDS)
  deallocate(PAR_SEEDS)

  PRINT *, "NUMERO DE PARAMETROS A ESTIMAR:",ESTIMSIZE
  PAR=PARINI

!  DO I = 1, NUM_PAR_EST
!     PAR(I)=PARVECTOR(PARIND(I))
!  END DO
!
!  DO I = NUM_PAR_EST+1, NUM_PAR_EST + NUM_VAR_EST
!     PAR(I)=VARVECTOR(VAREXOIND(I-NUM_PAR_EST))
!  END DO
!
!  DO I = NUM_PAR_EST + NUM_VAR_EST+1,ESTIMSIZE
!     J= I - (NUM_PAR_EST + NUM_VAR_EST)
!     IF (J <= NUM_MEDIDA_CTRL) THEN
!        PAR(I)=MEDIDAMEAN_CTRL(MEDIDA_CTRL_IND(J))
!     ELSE
!        PAR(I)=MEDIDAMEAN_EXO(MEDIDA_EXO_IND(J-NUM_MEDIDA_CTRL))
!     ENDIF
!  END DO

  

 ! BOUNDS PARA TODOS LOS PARAMETROS DE LA ESTIMACIÓN EN UN SOLO VECTOR
  ALLOCATE(PARBOUNDS(ESTIMSIZE,2))
  ALLOCATE(PRIOR_PAR(ESTIMSIZE,2))
  ALLOCATE(DIST(ESTIMSIZE))
!  DO I=1,NUM_PAR_EST
!     J=PARIND(I)
!     PARBOUNDS(I,1)=PARBOUND(J,1)
!     PARBOUNDS(I,2)=PARBOUND(J,2)
!     PRIOR_PAR(I,1)=(PARBOUNDS(I,1)+PARBOUNDS(I,2))/2.0
!     PRIOR_PAR(I,2)=((PARBOUNDS(I,2)-PARBOUNDS(I,1))**2)/2.0
!     DIST(I) = "DUNI"
!  ENDDO



!PRIORS BETA
  DO I=1,NUM_PAR_EST
     J=PARIND(I)
     PARBOUNDS(I,1)=PARBOUND(J,1)
     PARBOUNDS(I,2)=PARBOUND(J,2)
     PRIOR_PAR(I,1)=PRIORMEAN(J)
     PRIOR_PAR(I,2)=(0.15*(PARBOUNDS(I,2)-PARBOUNDS(I,1)))**2
     DIST(I) = "DBET"
  ENDDO

!  ALLOCATE(PAR_SS(NUM_PAR_SS))
  DO I=1,NUM_PAR_SS
     DO J=1,NUM_PAR_EST
        IF (PAR_SS_IND(I) .EQ. PARIND(J)) THEN
           PAR_SS=J
        PRINT *,"PARPP=",PAR_SS
        ENDIF
     ENDDO
     PRIOR_PAR(PAR_SS,1)=PAR(PAR_SS)
     PRIOR_PAR(PAR_SS,2)=0.05
     DIST(PAR_SS) = "DGAM"
  ENDDO



!  DO I=NUM_PAR_EST+1,NUM_PAR_EST+NUM_VAR_EST
!     J=VAREXOIND(I-NUM_PAR_EST)
!     PARBOUNDS(I,1)=VARBOUND(J,1)
!     PARBOUNDS(I,2)=VARBOUND(J,2)
!     PRIOR_PAR(I,1)=(PARBOUNDS(I,1)+PARBOUNDS(I,2))/2.0
!     PRIOR_PAR(I,2)=((PARBOUNDS(I,2)-PARBOUNDS(I,1))**2)/2.0
!     DIST(I) = "DUNI"
!  ENDDO

! INVERSE GAMA PRIORS
  DO I=NUM_PAR_EST+1,NUM_PAR_EST+NUM_VAR_EST
     J=VAREXOIND(I-NUM_PAR_EST)
     PARBOUNDS(I,1)=VARBOUND(J,1)
     PARBOUNDS(I,2)=VARBOUND(J,2)
    ! PRIOR_PAR(I,1)=PAR(I)
     PRIOR_PAR(I,2)=10.0
     DIST(I) = "DIGA"
  ENDDO

  DO I=NUM_PAR_EST+NUM_VAR_EST+1,ESTIMSIZE
     J=I-(NUM_PAR_EST+NUM_VAR_EST)
     IF (J <= NUM_MEDIDA_CTRL) THEN
        PRIOR_PAR(I,1)=MEDIDAMEAN_CTRL(MEDIDA_CTRL_IND(J))
        PRIOR_PAR(I,2)=MEDIDAVAR_CTRL(MEDIDA_CTRL_IND(J))
     ELSE
        PRIOR_PAR(I,1)=MEDIDAMEAN_EXO(MEDIDA_EXO_IND(J-NUM_MEDIDA_CTRL))
        PRIOR_PAR(I,2)=MEDIDAVAR_EXO(MEDIDA_EXO_IND(J-NUM_MEDIDA_CTRL))
     ENDIF
     PARBOUNDS(I,1)=DBLE(0.0_DP)
     PARBOUNDS(I,2)=DBLE(20.0)   !OJO: ESTO HAY QUE HACERLO MAS GENERAL. ESTÁ MAL
     DIST(I) = "DIGA"
  ENDDO

  ! PAR(1)=ETA_M  !  12 eta_m=3.19;     
!   PAR(2)=omega !  28 omeg=1.01369726080713;          
!   PAR(3)=sigma_m !  69 sigma_m=1.2;      
!   PAR(4)=omega_x !  33 omega_x=0.449441398423515;;      
!   PAR(5)=epsq
!   PAR(6)=rho_cstar

  ALLOCATE(STEP(ESTIMSIZE))
  ALLOCATE(XMIN(ESTIMSIZE))

  i=int(1)
  if (i .eq. 1) then

 !Parámetros para el SA de la estimación de los parámetros

  !NEQ=12  
  NMCES = INT(5000)
  NTES = INT(100)
  T0ES = 400.0
  RHOES = 0.985_DP  
  NSTEPES = INT(200)
 
  ALLOCATE(POINTS(ESTIMSIZE,2))
  NUMTRYS  = 10

!  CALL HYBRIDSA_EST(ESTIMATEOBJ,ESTIMSIZE,PARBOUNDS(:,1),PARBOUNDS(:,2),T0ES,NTES,NMCES,RHOES,NSTEPES,PAR,POINTS)
!  PAR=POINTS(:,2)
   !    HYBRIDSA_EST(FUNOBJ     ,NUMPAR,   BL,            BU,            T0,  NT,  NMC,  RHO,  NSTEP,  PARAMIN,POINTS)

!   OPEN (1,FILE="parestimhib.txt" )
!   DO J = 1, estimsize
!         write (1, '(ES14.7)'), par(J) 
  !     write(1,'(1X,200f10.5)') (AMAT(i,j),j=1,UBOUND(AMAT, 2)) 
!      END DO
!   CLOSE(1)

  NMC = 1000
   NT = 200
   T0 = 500
   RHO = 0.992_DP  
   NSTEP = 100   
   NUMTRYS  = 0

! xini=1.0
!   CALL ESTIMATEOBJ(ESTIMSIZE,PAR,LOG_L)
!   PRINT *, "LOG_L=",LOG_L
 !  CALL FDJAC1(ESTIMATEOBJ,ESTIMSIZE,PAR,LOG_L,FJAC,EPSILON(1.0_dp))
 !  DO I=1,ESTIMSIZE
 !  PRINT *, "FJAC=",I,FJAC(I)
 !  ENDDO
!  STOP

 !  NMC = 10000
 !  NT = 500
 !  T0 = 10000
 !  RHO = 0.998_DP  
 !  NSTEP = 50   
 !  NUMTRYS  = 1

   NUMREST=INT(3)
   print *, "Using Nelder..."
   ifault = 1
   done = .true. 
   DO WHILE (done) 
      DO J =1, ESTIMSIZE

         IF (ABS(PAR(J)) .LE. 0.00001_dp) THEN 
            STEP(J) = 0.0025_dp
         ELSE         
            STEP(J) = 0.05_dp*PAR(J)           
         END IF
         IF ((PAR(J)+STEP(J))>PARBOUNDS(J,2)) THEN
             STEP(J)=PARBOUNDS(J,2)-PAR(J)-1E-5
         END IF
      END DO

      CALL NELMIN(ESTIMATEOBJ, ESTIMSIZE, PAR, XMIN, YNEWLO, REQMIN, &
           &     STEP, KONVGE, KCOUNT, ICOUNT, NUMRES, IFAULT )  
      
   
      IF ( IFAULT .EQ. 2 .and.  restart < NUMREST) THEN
         restart=restart+1
         PAR = xmin
         KCOUNT = KCOUNT + KCOUNT
         KCOUNT = int(KCOUNT)
         DO j =1, ESTIMSIZE           
            IF (abs(PAR(J)) .LE. 0.00001_dp) THEN 
               !STEP(j) = 1000.5
               STEP(j) = 0.0025_dp
            ELSE
               STEP(j) = 0.05_dp*PAR(J)         
               !STEP(j) = 1000.5*PAR(j,1)           
            END IF
            IF ((PAR(J)+STEP(J))>PARBOUNDS(J,2)) THEN
             STEP(J)=PARBOUNDS(J,2)-PAR(J)-1E-5
            END IF
         END DO
      ELSE 
         if ( restart .ge. NUMREST ) then 
            print *, "no converegence"
            done = .false.
         else 
            print *, "converegence"
            done = .false.
         end if
      END IF
   end DO
 
  call  ESTIMATEOBJ(ESTIMSIZE,xmin,LOG_L_fin)
   print *, "Termino la maximizacion de la funcion de verosimilitud:",LOG_L_fin
  file_out='estimacion_mv_w_100_h1_'//trim(VAL)//'.txt'
  
   OPEN (1,FILE=trim(file_out))
   write (1,'(E12.6)'), LOG_L_fin
   write (1,'(E12.6)'), RAT_ERROR
   write (1,'(I4)'), NUM_PAR_EST
   DO J = 1, NUM_PAR_EST
         write (1,'(I4)'), PARIND(J) 
   END DO
   write (1,'(I4)'), NUM_VAR_EST
   DO J = 1, NUM_VAR_EST
         write (1,'(I4)'), VAREXOIND(J) 
   END DO
   write (1,'(I4)'), estimsize
   DO J = 1, estimsize
         write (1,'(ES18.10)'), xmin(J) 
   END DO
   write (1,'(I4)'), NEQ
   DO J = 1, NEQ
         write (1,'(ES18.10)'), XINI(J) 
   END DO
   write (1,'(I4)'), NUM_RAT_OBS
   DO J = 1, NUM_RAT_OBS
         write (1,'(ES18.10)'), RAT_OBS(J) 
   END DO
   CLOSE(1)

 !      OPEN (1,FILE="new_xini.txt",status='OLD')
 !      DO I = 1, NEQ
 !            write (1, '(ES14.7)'), XINI(I) 
 !      END DO
 !     CLOSE(1)
  else
    open(1, FILE="parestim2.txt", status='OLD')
    READ(1, *) (xMIN(J), J=1, estimsize)
    CLOSE(1)
    !PRINT *, xmin
  endif

 ! print *, "POSTERIOR MODE"
 ! write(*,'(a,f14.7)'),'epsq',xmin(1)
 ! write(*,'(a,f14.7)'),'eta_m',xmin(2)
 ! write(*,'(a,f14.7)'),'omega',xmin(3)
  !write(*,'(a,f14.7)'),'omega_x',xmin(4)
  !write(*,'(a,f14.7)'),'rho_cstar',xmin(5)
  !write(*,'(a,f14.7)'),'sigma_m',xmin(6)
!  DO I=1

! XMIN=PAR

  DELAPSE=DTIME(TIMEARRAY)
  print *
  print *, DELAPSE

 ! CALL FDJAC1(ESTIMATEOBJ,ESTIMSIZE,XMIN,FVEC,FJAC,IFLAG,EPSFCN)
  return

  ALLOCATE(DRAWS(MCDRAWS, ESTIMSIZE))
  !Estimobj is in estim_bayes.f90
  ALLOCATE(FHESS(ESTIMSIZE, ESTIMSIZE))
  ALLOCATE(FHESSVEC(ESTIMSIZE*ESTIMSIZE))
!  CALL HESSIAN(ESTIMATEOBJ, XMIN, ESTIMSIZE, 0.5*EPSFCN, FHESS)
       !HESSIAN(FCN,         X,    N,         GSTEP,      HESSIAN_MAT)
       !  CALL HESSIANMAT(FHESS)

  open(1, FILE="HESSIANMAT.txt", status='OLD')
  READ(1, *) (FHESSVEC(J), J=1, int(estimsize*estimsize))
  CLOSE(1)

  DO J=1, ESTIMSIZE
     DO I=1, ESTIMSIZE
        FHESS(I,J)=FHESSVEC(INT((J-1)*ESTIMSIZE + I))
     ENDDO
  ENDDO

!  OPEN (1,FILE="HESSIANMAT.txt" )
!  DO J = 1, ESTIMSIZE
!     DO I = 1, ESTIMSIZE
!        write (1, '(ES14.7)'), FHESS(I,J) 
!     END DO
!  END DO
!  CLOSE(1)


  PRINT *
  PRINT *, "POSTERIOR HESSIAN"
  DO I = 1, ESTIMSIZE
     WRITE(*, '(6ES15.7)') , FHESS(I, 1:ESTIMSIZE)
  END DO
 
!  RETURN 
!  C_SCALE =10.0_Dp!0.01_DP!0.03
  C_SCALE = 0.4_DP
  FHESS = C_SCALE*FHESS

  CALL DPOTRF( 'L', ESTIMSIZE,FHESS ,ESTIMSIZE, INFO )  
  CALL DPOTRI( 'L', ESTIMSIZE,FHESS, ESTIMSIZE, INFO )
  PRINT *, "IF EQUAL ZERO HESSIAN IS PD"
  PRINT *, INFO
  PRINT *
  ! DO I = 1, ESTIMSIZE
!      WRITE(*, '(6ES15.7)') , FHESS(I, 1:ESTIMSIZE)
!   END DO



  
  CALL MCMCMODEL(MCDRAWS, C_SCALE, FHESS, XMIN,ESTIMSIZE, ACCEPT, DRAWS)
     
  PRINT *
  write(*, '(a20, f14.4)'), "accepted", dble(accept)/MCDRAWS
  
  print * 
!  open(1, file='draw.txt')
!  DO I = 1, MCDRAWS
!     WRITE(1, '(6ES15.7)') , DRAWS(I, 1:ESTIMSIZE)
!  END DO
!  close(1)

  OPEN (1,FILE="draw.txt" )
  DO J = 1, ESTIMSIZE
     DO I = 1, MCDRAWS
        write (1, '(ES14.7)'), DRAWS(I,J) 
     END DO
  END DO
  CLOSE(1)
  
  
  DELAPSE=DTIME(TIMEARRAY)
  print *
  print *, DELAPSE

  DEALLOCATE(ZIND)
  DEALLOCATE(XINI)
  DEALLOCATE(BL)
  DEALLOCATE(BU)

END PROGRAM PATACON_V3C

