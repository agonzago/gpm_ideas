SUBROUTINE ABMATRIX(NVAR,AMAT, BMAT, SSVAL)
  USE PARAMETERS_MOD
  IMPLICIT NONE 
  INTEGER, INTENT(IN) :: NVAR
  DOUBLE PRECISION, DIMENSION(NVAR, NVAR) , INTENT(OUT) :: AMAT, BMAT
  DOUBLE PRECISION, DIMENSION(NVAR) , INTENT(IN) :: SSVAL
  integer::i,j

  !NOMBRES DE PARAMETROS
  DOUBLE PRECISION BETTA,BETTASTAR,PSI_2

  !NOMBRE DE LAS VARIABLES
  DOUBLE PRECISION  LAG1_C_SS,LAG1_CD_SS,LAG1_CM_SS,LAG1_E_SS,LAG1_IE_SS,LAG1_INF_C_SS,LAG2_INF_C_SS,LAG3_INF_C_SS,&
       &LAG1_INF_CD_SS,LAG1_INF_E_SS,LAG1_INF_MD_SS,LAG1_INF_Q_SS,LAG1_INF_RM_SS,LAG1_INF_T_SS,LAG1_INF_XD_SS,&
       &LAG1_MD_SS,LAG1_MS_SS,LAG1_NGDP_PC_SS,LAG2_NGDP_PC_SS,LAG3_NGDP_PC_SS,LAG1_PCD_PC_SS,LAG1_PCD_PQ_SS,&
       &LAG1_PE_PQ_SS,LAG1_PMD_PC_SS,LAG1_PRM_PC_SS,LAG1_PT_PQ_SS,LAG1_PVCD_SS,LAG1_PVE_SS,LAG1_PVM_SS,&
       &LAG1_PVQ_SS,LAG1_PVRM_SS,LAG1_PVT_SS,LAG1_PVXD_SS,LAG1_PX_PC_SS,LAG1_PXD_PCD_SS,LAG1_QD_SS,&
       &LAG1_RM_SS,LAG1_S_PC_SS,LAG1_T_SS,LAG1_W_SS,LAG1_X_SS,LAG1_XD_SS,LAG1_XM_SS,LAG1_YI_SS,LAG1_CSTAR_SS,&
       &LAG1_QM_SS,LAG1_TR_SS,LAG1_F_SS,LAG1_K_SS,F_SS,K_SS,CSTAR_SS,G_SS,PI_STAR_SS,QM_SS,QMR_SS,TR_SS,&
       &Z_XDEMAND_SS,ZCD_SS,ZD_SS,ZE_SS,ZH_SS,ZI_SS,ZIE_SS,ZM_SS,ZNT_SS,ZQ_SS,ZRM_SS,ZU_SS,ZX_SS,ZXD_SS,&
       &ADJCOST_C_SS,ADJCOST_X_SS,C_SS,CD_SS,CDS_SS,CDSD_SS,CM_SS,D_C_SS,D_CD_SS,D_CM_SS,D_CSTAR_SS,&
       &D_E_SS,D_F_SS,D_K_SS,D_MD_SS,D_MS_SS,D_NGDP_PC_SS,D_QD_SS,D_RM_SS,D_SS,D_T_SS,D_TR_SS,D_W_SS,D_X_SS,&
       &D_XD_SS,D_XM_SS,DELTA_SS,E_SS,ES_SS,ESD_SS,F1_SS,F2_SS,GAM_X_SS,H_SS,I_SS,IE_SS,INF_C_SS,INF_CD_SS,&
       &INF_E_SS,INF_MD_SS,INF_MF_SS,INF_Q_SS,INF_RM_SS,INF_T_SS,INF_X_SS,INF_XD_SS,KS_SS,LAMBDA_SS,LED1_YINF_C_SS,&
       &LED2_YINF_C_SS,LED3_YINF_C_SS,LED4_YINF_C_SS,LED5_YINF_C_SS,LED6_YINF_C_SS,MD_SS,MF_SS,MS_SS,NGDP_PC_SS,&
       &NT_SS,PCD_PC_SS,PCD_PQ_SS,PCDS_PQ_SS,PE_PC_SS,PE_PQ_SS,PES_PQ_SS,PHI_SS,PHICD_SS,PHIE_SS,PHIMD_SS,PHIRM_SS,&
       &PHIXD_SS,PI_STAR_M_SS,PMD_PC_SS,PMD_PX_SS,PMF_PC_SS,PMF_PMD_SS,PMR_PC_SS,PMR_PRM_SS,POPT_PQ_SS,POPTCD_PCD_SS,&
       &POPTE_PE_SS,POPTMD_PMD_SS,POPTRM_PRM_SS,POPTT_PT_SS,POPTXD_PXD_SS,PQ_PC_SS,PRM_PC_SS,PROF_CD_SS,PROF_E_SS,&
       &PROF_M_SS,PROF_Q_SS,PROF_RM_SS,PROF_T_SS,PROF_XD_SS,PSICD_SS,PSIE_SS,PSIM_SS,PSIP_SS,PSIRM_SS,PSIT_SS,&
       &PSIXD_SS,PT_PC_SS,PT_PMD_SS,PT_PQ_SS,PTS_PQ_SS,PVCD_SS,PVE_SS,PVM_SS,PVQ_SS,PVRM_SS,PVT_SS,PVXD_SS,&
       &PX_PC_SS,PXD_PC_SS,PXD_PCD_SS,PXD_PMD_SS,PXD_PQ_SS,PXD_PX_SS,PXDS_PQ_SS,QD_SS,QE_Q_SS,QS_SS,&
       &RK_SS,RM_SS,RMF_SS,RMS_SS,S_PC_SS,T_SS,TCD_SS,TE_SS,THETACD_SS,THETAE_SS,THETAM_SS,THETAP_SS,THETARM_SS,&
       &THETAT_SS,THETAXD_SS,TM_SS,TS_SS,TXD_SS,U_SS,VQ_SS,W_SS,WOPT_SS,X_SS,XD_SS,XDS_SS,XDSD_SS,XM_SS,YI_SS,&
       &YINF_C_SS,YNGDP_PC_SS



  !ASIGNACION DE VALOR DE LOS PARAMETROS  


  LAG1_C_SS=SSVAL(1)
  LAG1_CD_SS=SSVAL(2)
  LAG1_CM_SS=SSVAL(3)
  LAG1_E_SS=SSVAL(4)
  LAG1_IE_SS=SSVAL(5)
  LAG1_INF_C_SS=SSVAL(6)
  LAG2_INF_C_SS=SSVAL(7)
  LAG3_INF_C_SS=SSVAL(8)
  LAG1_INF_CD_SS=SSVAL(9)
  LAG1_INF_E_SS=SSVAL(10)
  LAG1_INF_MD_SS=SSVAL(11)
  LAG1_INF_Q_SS=SSVAL(12)
  LAG1_INF_RM_SS=SSVAL(13)
  LAG1_INF_T_SS=SSVAL(14)
  LAG1_INF_XD_SS=SSVAL(15)
  LAG1_MD_SS=SSVAL(16)
  LAG1_MS_SS=SSVAL(17)
  LAG1_NGDP_PC_SS=SSVAL(18)
  LAG2_NGDP_PC_SS=SSVAL(19)
  LAG3_NGDP_PC_SS=SSVAL(20)
  LAG1_PCD_PC_SS=SSVAL(21)
  LAG1_PCD_PQ_SS=SSVAL(22)
  LAG1_PE_PQ_SS=SSVAL(23)
  LAG1_PMD_PC_SS=SSVAL(24)
  LAG1_PRM_PC_SS=SSVAL(25)
  LAG1_PT_PQ_SS=SSVAL(26)
  LAG1_PVCD_SS=SSVAL(27)
  LAG1_PVE_SS=SSVAL(28)
  LAG1_PVM_SS=SSVAL(29)
  LAG1_PVQ_SS=SSVAL(30)
  LAG1_PVRM_SS=SSVAL(31)
  LAG1_PVT_SS=SSVAL(32)
  LAG1_PVXD_SS=SSVAL(33)
  LAG1_PX_PC_SS=SSVAL(34)
  LAG1_PXD_PCD_SS=SSVAL(35)
  LAG1_QD_SS=SSVAL(36)
  LAG1_RM_SS=SSVAL(37)
  LAG1_S_PC_SS=SSVAL(38)
  LAG1_T_SS=SSVAL(39)
  LAG1_W_SS=SSVAL(40)
  LAG1_X_SS=SSVAL(41)
  LAG1_XD_SS=SSVAL(42)
  LAG1_XM_SS=SSVAL(43)
  LAG1_YI_SS=SSVAL(44)
  LAG1_CSTAR_SS=SSVAL(45)
  LAG1_QM_SS=SSVAL(46)
  LAG1_TR_SS=SSVAL(47)
  LAG1_F_SS=SSVAL(48)
  LAG1_K_SS=SSVAL(49)
  F_SS=SSVAL(50)
  K_SS=SSVAL(51)
  CSTAR_SS=SSVAL(52)
  G_SS=SSVAL(53)
  PI_STAR_SS=SSVAL(54)
  QM_SS=SSVAL(55)
  QMR_SS=SSVAL(56)
  TR_SS=SSVAL(57)
  Z_XDEMAND_SS=SSVAL(58)
  ZCD_SS=SSVAL(59)
  ZD_SS=SSVAL(60)
  ZE_SS=SSVAL(61)
  ZH_SS=SSVAL(62)
  ZI_SS=SSVAL(63)
  ZIE_SS=SSVAL(64)
  ZM_SS=SSVAL(65)
  ZNT_SS=SSVAL(66)
  ZQ_SS=SSVAL(67)
  ZRM_SS=SSVAL(68)
  ZU_SS=SSVAL(69)
  ZX_SS=SSVAL(70)
  ZXD_SS=SSVAL(71)
  ADJCOST_C_SS=SSVAL(72)
  ADJCOST_X_SS=SSVAL(73)
  C_SS=SSVAL(74)
  CD_SS=SSVAL(75)
  CDS_SS=SSVAL(76)
  CDSD_SS=SSVAL(77)
  CM_SS=SSVAL(78)
  D_C_SS=SSVAL(79)
  D_CD_SS=SSVAL(80)
  D_CM_SS=SSVAL(81)
  D_CSTAR_SS=SSVAL(82)
  D_E_SS=SSVAL(83)
  D_F_SS=SSVAL(84)
  D_K_SS=SSVAL(85)
  D_MD_SS=SSVAL(86)
  D_MS_SS=SSVAL(87)
  D_NGDP_PC_SS=SSVAL(88)
  D_QD_SS=SSVAL(89)
  D_RM_SS=SSVAL(90)
  D_SS=SSVAL(91)
  D_T_SS=SSVAL(92)
  D_TR_SS=SSVAL(93)
  D_W_SS=SSVAL(94)
  D_X_SS=SSVAL(95)
  D_XD_SS=SSVAL(96)
  D_XM_SS=SSVAL(97)
  DELTA_SS=SSVAL(98)
  E_SS=SSVAL(99)
  ES_SS=SSVAL(100)
  ESD_SS=SSVAL(101)
  F1_SS=SSVAL(102)
  F2_SS=SSVAL(103)
  GAM_X_SS=SSVAL(104)
  H_SS=SSVAL(105)
  I_SS=SSVAL(106)
  IE_SS=SSVAL(107)
  INF_C_SS=SSVAL(108)
  INF_CD_SS=SSVAL(109)
  INF_E_SS=SSVAL(110)
  INF_MD_SS=SSVAL(111)
  INF_MF_SS=SSVAL(112)
  INF_Q_SS=SSVAL(113)
  INF_RM_SS=SSVAL(114)
  INF_T_SS=SSVAL(115)
  INF_X_SS=SSVAL(116)
  INF_XD_SS=SSVAL(117)
  KS_SS=SSVAL(118)
  LAMBDA_SS=SSVAL(119)
  LED1_YINF_C_SS=SSVAL(120)
  LED2_YINF_C_SS=SSVAL(121)
  LED3_YINF_C_SS=SSVAL(122)
  LED4_YINF_C_SS=SSVAL(123)
  LED5_YINF_C_SS=SSVAL(124)
  LED6_YINF_C_SS=SSVAL(125)
  MD_SS=SSVAL(126)
  MF_SS=SSVAL(127)
  MS_SS=SSVAL(128)
  NGDP_PC_SS=SSVAL(129)
  NT_SS=SSVAL(130)
  PCD_PC_SS=SSVAL(131)
  PCD_PQ_SS=SSVAL(132)
  PCDS_PQ_SS=SSVAL(133)
  PE_PC_SS=SSVAL(134)
  PE_PQ_SS=SSVAL(135)
  PES_PQ_SS=SSVAL(136)
  PHI_SS=SSVAL(137)
  PHICD_SS=SSVAL(138)
  PHIE_SS=SSVAL(139)
  PHIMD_SS=SSVAL(140)
  PHIRM_SS=SSVAL(141)
  PHIXD_SS=SSVAL(142)
  PI_STAR_M_SS=SSVAL(143)
  PMD_PC_SS=SSVAL(144)
  PMD_PX_SS=SSVAL(145)
  PMF_PC_SS=SSVAL(146)
  PMF_PMD_SS=SSVAL(147)
  PMR_PC_SS=SSVAL(148)
  PMR_PRM_SS=SSVAL(149)
  POPT_PQ_SS=SSVAL(150)
  POPTCD_PCD_SS=SSVAL(151)
  POPTE_PE_SS=SSVAL(152)
  POPTMD_PMD_SS=SSVAL(153)
  POPTRM_PRM_SS=SSVAL(154)
  POPTT_PT_SS=SSVAL(155)
  POPTXD_PXD_SS=SSVAL(156)
  PQ_PC_SS=SSVAL(157)
  PRM_PC_SS=SSVAL(158)
  PROF_CD_SS=SSVAL(159)
  PROF_E_SS=SSVAL(160)
  PROF_M_SS=SSVAL(161)
  PROF_Q_SS=SSVAL(162)
  PROF_RM_SS=SSVAL(163)
  PROF_T_SS=SSVAL(164)
  PROF_XD_SS=SSVAL(165)
  PSICD_SS=SSVAL(166)
  PSIE_SS=SSVAL(167)
  PSIM_SS=SSVAL(168)
  PSIP_SS=SSVAL(169)
  PSIRM_SS=SSVAL(170)
  PSIT_SS=SSVAL(171)
  PSIXD_SS=SSVAL(172)
  PT_PC_SS=SSVAL(173)
  PT_PMD_SS=SSVAL(174)
  PT_PQ_SS=SSVAL(175)
  PTS_PQ_SS=SSVAL(176)
  PVCD_SS=SSVAL(177)
  PVE_SS=SSVAL(178)
  PVM_SS=SSVAL(179)
  PVQ_SS=SSVAL(180)
  PVRM_SS=SSVAL(181)
  PVT_SS=SSVAL(182)
  PVXD_SS=SSVAL(183)
  PX_PC_SS=SSVAL(184)
  PXD_PC_SS=SSVAL(185)
  PXD_PCD_SS=SSVAL(186)
  PXD_PMD_SS=SSVAL(187)
  PXD_PQ_SS=SSVAL(188)
  PXD_PX_SS=SSVAL(189)
  PXDS_PQ_SS=SSVAL(190)
  QD_SS=SSVAL(191)
  QE_Q_SS=SSVAL(192)
  QS_SS=SSVAL(193)
  RK_SS=SSVAL(194)
  RM_SS=SSVAL(195)
  RMF_SS=SSVAL(196)
  RMS_SS=SSVAL(197)
  S_PC_SS=SSVAL(198)
  T_SS=SSVAL(199)
  TCD_SS=SSVAL(200)
  TE_SS=SSVAL(201)
  THETACD_SS=SSVAL(202)
  THETAE_SS=SSVAL(203)
  THETAM_SS=SSVAL(204)
  THETAP_SS=SSVAL(205)
  THETARM_SS=SSVAL(206)
  THETAT_SS=SSVAL(207)
  THETAXD_SS=SSVAL(208)
  TM_SS=SSVAL(209)
  TS_SS=SSVAL(210)
  TXD_SS=SSVAL(211)
  U_SS=SSVAL(212)
  VQ_SS=SSVAL(213)
  W_SS=SSVAL(214)
  WOPT_SS=SSVAL(215)
  X_SS=SSVAL(216)
  XD_SS=SSVAL(217)
  XDS_SS=SSVAL(218)
  XDSD_SS=SSVAL(219)
  XM_SS=SSVAL(220)
  YI_SS=SSVAL(221)
  YINF_C_SS=SSVAL(222)
  YNGDP_PC_SS=SSVAL(223)

  PIB_BAR=exp(SSVAL(223))


  !DEFINE BETTA AND PHI_1 
  BETTASTAR  = (PI_STAR_BAR)/(ISTAR)
  BETTA  =  BETTASTAR*(1+GBAR)**(SIGMA_M)
  PSI_2  = ((1+UPSILON)/UPSILON)*((1-BETTA*(1+GBAR)**(-SIGMA_M)*(1-PSI_1))/(BETTA*(1+GBAR)**(-SIGMA_M)))



  ! CREAR LA MATRICES A, B

  BMAT = dble(0.0) 
  AMAT = dble(0.0)


  Amat(3, 53)=& 
       betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*sigma_m*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)
  Amat(3, 72)=& 
       betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)/omega
  Amat(3, 74)=& 
       -betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)/omega
  Amat(3, 75)=& 
       betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-2*cm_ss-2*cd_ss+cd_ss)+betta* &
       (exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)
  Amat(3, 78)=& 
       -betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-2*cm_ss-2*cd_ss+cd_ss)-betta* &
       (exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*(2-1/ &
       omega)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)
  Amat(3, 108)=& 
       -betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)
  Amat(3, 119)=& 
       -betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)
  Amat(4, 53)=& 
       -betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*sigma_m*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)
  Amat(4, 72)=& 
       -betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)/omega
  Amat(4, 74)=& 
       betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)/omega
  Amat(4, 75)=& 
       -betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-3*cm_ss-2*cd_ss+2* &
       cd_ss)-betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)
  Amat(4, 78)=& 
       betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-3*cm_ss-2*cd_ss+2* &
       cd_ss)+betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*(2-1/ &
       omega)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)
  Amat(4, 108)=& 
       betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)
  Amat(4, 119)=& 
       betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2*cm_ss-cd_ss+cd_ss)
  Amat(7, 51)=& 
       -exp(k_ss)
  Amat(10, 53)=& 
       betta*epsw*(nbar+1)*(1-sigma_m)*exp((1-theta_w)* &
       (wopt_ss-wopt_ss)+(inf_c_ss-inf_c_ss)*(1-theta_w)+G_ss* &
       (1-sigma_m)+f1_ss)
  Amat(10, 102)=& 
       betta*epsw*(nbar+1)*exp((1-theta_w)* &
       (wopt_ss-wopt_ss)+(inf_c_ss-inf_c_ss)*(1-theta_w)+G_ss* &
       (1-sigma_m)+f1_ss)
  Amat(10, 108)=& 
       betta*epsw*(nbar+1)*(theta_w-1)*exp((1-theta_w)* &
       (wopt_ss-wopt_ss)+(inf_c_ss-inf_c_ss)*(1-theta_w)+G_ss* &
       (1-sigma_m)+f1_ss)
  Amat(10, 215)=& 
       betta*epsw*(nbar+1)*(theta_w-1)*exp((1-theta_w)* &
       (wopt_ss-wopt_ss)+(inf_c_ss-inf_c_ss)*(1-theta_w)+G_ss* &
       (1-sigma_m)+f1_ss)
  Amat(11, 53)=& 
       betta*epsw*(nbar+1)*(1-sigma_m)*exp(-(eta_m+1)*theta_w* &
       (-wopt_ss+wopt_ss-inf_c_ss+inf_c_ss)+G_ss*(1-sigma_m)+f2_ss)
  Amat(11, 103)=& 
       betta*epsw*(nbar+1)*exp(-(eta_m+1)*theta_w* &
       (-wopt_ss+wopt_ss-inf_c_ss+inf_c_ss)+G_ss*(1-sigma_m)+f2_ss)
  Amat(11, 108)=& 
       betta*epsw*(eta_m+1)*(nbar+1)*theta_w*exp(-(eta_m+1)*theta_w* &
       (-wopt_ss+wopt_ss-inf_c_ss+inf_c_ss)+G_ss*(1-sigma_m)+f2_ss)
  Amat(11, 215)=& 
       betta*epsw*(eta_m+1)*(nbar+1)*theta_w*exp(-(eta_m+1)*theta_w* &
       (-wopt_ss+wopt_ss-inf_c_ss+inf_c_ss)+G_ss*(1-sigma_m)+f2_ss)
  Amat(13, 53)=& 
       betta*(nbar+1)*(1-sigma_m)*exp(-x_ss+G_ss*(1-sigma_m)+lambda_ss)* &
       (psi_x*exp(-x_ss)*(exp(x_ss)-exp(x_ss))**2/ &
       2.0+psi_x*(exp(x_ss)-exp(x_ss)))
  Amat(13, 119)=& 
       betta*(nbar+1)*exp(-x_ss+G_ss*(1-sigma_m)+lambda_ss)*(psi_x* &
       exp(-x_ss)*(exp(x_ss)-exp(x_ss))**2/ &
       2.0+psi_x*(exp(x_ss)-exp(x_ss)))
  Amat(13, 216)=& 
       betta*(nbar+1)*exp(-x_ss+G_ss*(1-sigma_m)+lambda_ss)*(psi_x* &
       (exp(x_ss)-exp(x_ss))*exp(x_ss-x_ss)+psi_x*exp(x_ss))
  Amat(14, 53)=& 
       -betta*sigma_m*exp(u_ss-G_ss*sigma_m+rk_ss+lambda_ss)-betta* &
       (1-exp(delta_ss))*sigma_m*exp(gam_x_ss-G_ss*sigma_m)
  Amat(14, 104)=& 
       betta*(1-exp(delta_ss))*exp(gam_x_ss-G_ss*sigma_m)
  Amat(14, 119)=& 
       betta*exp(u_ss-G_ss*sigma_m+rk_ss+lambda_ss)
  Amat(14, 194)=& 
       betta*exp(u_ss-G_ss*sigma_m+rk_ss+lambda_ss)
  Amat(14, 212)=& 
       betta*exp(u_ss-G_ss*sigma_m+rk_ss+lambda_ss)
  Amat(16, 53)=& 
       -betta*sigma_m*exp(-G_ss*sigma_m+lambda_ss+i_ss-inf_c_ss)
  Amat(16, 108)=& 
       -betta*exp(-G_ss*sigma_m+lambda_ss+i_ss-inf_c_ss)
  Amat(16, 119)=& 
       betta*exp(-G_ss*sigma_m+lambda_ss+i_ss-inf_c_ss)
  Amat(17, 53)=& 
       -betta*sigma_m*exp(-G_ss*sigma_m+lambda_ss-inf_c_ss+ie_ss+d_ss)
  Amat(17, 91)=& 
       betta*exp(-G_ss*sigma_m+lambda_ss-inf_c_ss+ie_ss+d_ss)
  Amat(17, 108)=& 
       -betta*exp(-G_ss*sigma_m+lambda_ss-inf_c_ss+ie_ss+d_ss)
  Amat(17, 119)=& 
       betta*exp(-G_ss*sigma_m+lambda_ss-inf_c_ss+ie_ss+d_ss)
  Amat(18, 50)=& 
       -exp(s_pc_ss+f_ss)
  Amat(21, 53)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x*sigma_m* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)
  Amat(21, 70)=& 
       -betta*(1-gama_x)**(1/omega_x)*(nbar+1)*(1-1/ &
       omega_x)*psi_adjcost_x*(exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)
  Amat(21, 73)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)/ &
       omega_x
  Amat(21, 116)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)
  Amat(21, 119)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)
  Amat(21, 216)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)/ &
       omega_x
  Amat(21, 217)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-2*xm_ss-2*xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)+betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)
  Amat(21, 220)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-2*xm_ss-2*xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*(2-1/ &
       omega_x)*psi_adjcost_x*(exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss*sigma_m+lambda_ss+inf_x_ss)
  Amat(22, 53)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x*sigma_m* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)
  Amat(22, 70)=& 
       betta*(1-gama_x)**(1/omega_x)*(nbar+1)*(1-1/ &
       omega_x)*psi_adjcost_x*(exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)
  Amat(22, 73)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)/omega_x
  Amat(22, 116)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)
  Amat(22, 119)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)
  Amat(22, 216)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)/omega_x
  Amat(22, 217)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xm_ss-2*xd_ss+2*xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)
  Amat(22, 220)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xm_ss-2*xd_ss+2*xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)+betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*(2-1/ &
       omega_x)*psi_adjcost_x*(exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)
  Amat(35, 53)=& 
       betta*epsq*(nbar+1)*(1-sigma_m)*exp(THETAP_ss+inf_q_ss* &
       theta-inf_q_ss*theta+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(35, 113)=& 
       betta*epsq*(nbar+1)*theta*exp(THETAP_ss+inf_q_ss*theta-inf_q_ss* &
       theta+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(35, 119)=& 
       betta*epsq*(nbar+1)*exp(THETAP_ss+inf_q_ss*theta-inf_q_ss* &
       theta+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(35, 205)=& 
       betta*epsq*(nbar+1)*exp(THETAP_ss+inf_q_ss*theta-inf_q_ss* &
       theta+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(36, 53)=& 
       betta*epsq*(nbar+1)*(1-sigma_m)*exp(inf_q_ss*(theta-1)-inf_q_ss* &
       (theta-1)+G_ss*(1-sigma_m)+PSIP_ss+lambda_ss-lambda_ss)
  Amat(36, 113)=& 
       betta*epsq*(nbar+1)*(theta-1)*exp(inf_q_ss*(theta-1)-inf_q_ss* &
       (theta-1)+G_ss*(1-sigma_m)+PSIP_ss+lambda_ss-lambda_ss)
  Amat(36, 119)=& 
       betta*epsq*(nbar+1)*exp(inf_q_ss*(theta-1)-inf_q_ss* &
       (theta-1)+G_ss*(1-sigma_m)+PSIP_ss+lambda_ss-lambda_ss)
  Amat(36, 169)=& 
       betta*epsq*(nbar+1)*exp(inf_q_ss*(theta-1)-inf_q_ss* &
       (theta-1)+G_ss*(1-sigma_m)+PSIP_ss+lambda_ss-lambda_ss)
  Amat(48, 53)=& 
       betta*epscd*(nbar+1)*(1-sigma_m)*exp(inf_cd_ss*theta_cd-inf_cd_ss* &
       theta_cd+THETACD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(48, 109)=& 
       betta*epscd*(nbar+1)*theta_cd*exp(inf_cd_ss*theta_cd-inf_cd_ss* &
       theta_cd+THETACD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(48, 119)=& 
       betta*epscd*(nbar+1)*exp(inf_cd_ss*theta_cd-inf_cd_ss* &
       theta_cd+THETACD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(48, 202)=& 
       betta*epscd*(nbar+1)*exp(inf_cd_ss*theta_cd-inf_cd_ss* &
       theta_cd+THETACD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(49, 53)=& 
       betta*epscd*(nbar+1)*(1-sigma_m)*exp(inf_cd_ss* &
       (theta_cd-1)-inf_cd_ss*(theta_cd-1)+G_ss* &
       (1-sigma_m)+PSICD_ss+lambda_ss-lambda_ss)
  Amat(49, 109)=& 
       betta*epscd*(nbar+1)*(theta_cd-1)*exp(inf_cd_ss* &
       (theta_cd-1)-inf_cd_ss*(theta_cd-1)+G_ss* &
       (1-sigma_m)+PSICD_ss+lambda_ss-lambda_ss)
  Amat(49, 119)=& 
       betta*epscd*(nbar+1)*exp(inf_cd_ss*(theta_cd-1)-inf_cd_ss* &
       (theta_cd-1)+G_ss*(1-sigma_m)+PSICD_ss+lambda_ss-lambda_ss)
  Amat(49, 166)=& 
       betta*epscd*(nbar+1)*exp(inf_cd_ss*(theta_cd-1)-inf_cd_ss* &
       (theta_cd-1)+G_ss*(1-sigma_m)+PSICD_ss+lambda_ss-lambda_ss)
  Amat(57, 53)=& 
       betta*epsxd*(nbar+1)*(1-sigma_m)*exp(inf_xd_ss*theta_xd-inf_xd_ss* &
       theta_xd+THETAXD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(57, 117)=& 
       betta*epsxd*(nbar+1)*theta_xd*exp(inf_xd_ss*theta_xd-inf_xd_ss* &
       theta_xd+THETAXD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(57, 119)=& 
       betta*epsxd*(nbar+1)*exp(inf_xd_ss*theta_xd-inf_xd_ss* &
       theta_xd+THETAXD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(57, 208)=& 
       betta*epsxd*(nbar+1)*exp(inf_xd_ss*theta_xd-inf_xd_ss* &
       theta_xd+THETAXD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(58, 53)=& 
       betta*epsxd*(nbar+1)*(1-sigma_m)*exp(inf_xd_ss* &
       (theta_xd-1)-inf_xd_ss*(theta_xd-1)+G_ss* &
       (1-sigma_m)+PSIXD_ss+lambda_ss-lambda_ss)
  Amat(58, 117)=& 
       betta*epsxd*(nbar+1)*(theta_xd-1)*exp(inf_xd_ss* &
       (theta_xd-1)-inf_xd_ss*(theta_xd-1)+G_ss* &
       (1-sigma_m)+PSIXD_ss+lambda_ss-lambda_ss)
  Amat(58, 119)=& 
       betta*epsxd*(nbar+1)*exp(inf_xd_ss*(theta_xd-1)-inf_xd_ss* &
       (theta_xd-1)+G_ss*(1-sigma_m)+PSIXD_ss+lambda_ss-lambda_ss)
  Amat(58, 172)=& 
       betta*epsxd*(nbar+1)*exp(inf_xd_ss*(theta_xd-1)-inf_xd_ss* &
       (theta_xd-1)+G_ss*(1-sigma_m)+PSIXD_ss+lambda_ss-lambda_ss)
  Amat(63, 53)=& 
       betta*epst*(nbar+1)*(1-sigma_m)*exp(inf_t_ss*theta_t-inf_t_ss* &
       theta_t+THETAT_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(63, 115)=& 
       betta*epst*(nbar+1)*theta_t*exp(inf_t_ss*theta_t-inf_t_ss* &
       theta_t+THETAT_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(63, 119)=& 
       betta*epst*(nbar+1)*exp(inf_t_ss*theta_t-inf_t_ss* &
       theta_t+THETAT_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(63, 207)=& 
       betta*epst*(nbar+1)*exp(inf_t_ss*theta_t-inf_t_ss* &
       theta_t+THETAT_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(64, 53)=& 
       betta*epst*(nbar+1)*(1-sigma_m)*exp(inf_t_ss*(theta_t-1)-inf_t_ss* &
       (theta_t-1)+G_ss*(1-sigma_m)+PSIT_ss+lambda_ss-lambda_ss)
  Amat(64, 115)=& 
       betta*epst*(nbar+1)*(theta_t-1)*exp(inf_t_ss*(theta_t-1)-inf_t_ss* &
       (theta_t-1)+G_ss*(1-sigma_m)+PSIT_ss+lambda_ss-lambda_ss)
  Amat(64, 119)=& 
       betta*epst*(nbar+1)*exp(inf_t_ss*(theta_t-1)-inf_t_ss* &
       (theta_t-1)+G_ss*(1-sigma_m)+PSIT_ss+lambda_ss-lambda_ss)
  Amat(64, 171)=& 
       betta*epst*(nbar+1)*exp(inf_t_ss*(theta_t-1)-inf_t_ss* &
       (theta_t-1)+G_ss*(1-sigma_m)+PSIT_ss+lambda_ss-lambda_ss)
  Amat(72, 53)=& 
       betta*epse*(nbar+1)*(1-sigma_m)*exp(inf_e_ss*theta_e-inf_e_ss* &
       theta_e+THETAE_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(72, 110)=& 
       betta*epse*(nbar+1)*theta_e*exp(inf_e_ss*theta_e-inf_e_ss* &
       theta_e+THETAE_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(72, 119)=& 
       betta*epse*(nbar+1)*exp(inf_e_ss*theta_e-inf_e_ss* &
       theta_e+THETAE_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(72, 203)=& 
       betta*epse*(nbar+1)*exp(inf_e_ss*theta_e-inf_e_ss* &
       theta_e+THETAE_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(73, 53)=& 
       betta*epse*(nbar+1)*(1-sigma_m)*exp(inf_e_ss*(theta_e-1)-inf_e_ss* &
       (theta_e-1)+G_ss*(1-sigma_m)+PSIE_ss+lambda_ss-lambda_ss)
  Amat(73, 110)=& 
       betta*epse*(nbar+1)*(theta_e-1)*exp(inf_e_ss*(theta_e-1)-inf_e_ss* &
       (theta_e-1)+G_ss*(1-sigma_m)+PSIE_ss+lambda_ss-lambda_ss)
  Amat(73, 119)=& 
       betta*epse*(nbar+1)*exp(inf_e_ss*(theta_e-1)-inf_e_ss* &
       (theta_e-1)+G_ss*(1-sigma_m)+PSIE_ss+lambda_ss-lambda_ss)
  Amat(73, 167)=& 
       betta*epse*(nbar+1)*exp(inf_e_ss*(theta_e-1)-inf_e_ss* &
       (theta_e-1)+G_ss*(1-sigma_m)+PSIE_ss+lambda_ss-lambda_ss)
  Amat(87, 53)=& 
       betta*epsm*(nbar+1)*(1-sigma_m)*exp(inf_md_ss*theta_m-inf_md_ss* &
       theta_m+THETAM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(87, 111)=& 
       betta*epsm*(nbar+1)*theta_m*exp(inf_md_ss*theta_m-inf_md_ss* &
       theta_m+THETAM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(87, 119)=& 
       betta*epsm*(nbar+1)*exp(inf_md_ss*theta_m-inf_md_ss* &
       theta_m+THETAM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(87, 204)=& 
       betta*epsm*(nbar+1)*exp(inf_md_ss*theta_m-inf_md_ss* &
       theta_m+THETAM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(88, 53)=& 
       betta*epsm*(nbar+1)*(1-sigma_m)*exp(inf_md_ss* &
       (theta_m-1)-inf_md_ss*(theta_m-1)+G_ss* &
       (1-sigma_m)+PSIM_ss+lambda_ss-lambda_ss)
  Amat(88, 111)=& 
       betta*epsm*(nbar+1)*(theta_m-1)*exp(inf_md_ss* &
       (theta_m-1)-inf_md_ss*(theta_m-1)+G_ss* &
       (1-sigma_m)+PSIM_ss+lambda_ss-lambda_ss)
  Amat(88, 119)=& 
       betta*epsm*(nbar+1)*exp(inf_md_ss*(theta_m-1)-inf_md_ss* &
       (theta_m-1)+G_ss*(1-sigma_m)+PSIM_ss+lambda_ss-lambda_ss)
  Amat(88, 168)=& 
       betta*epsm*(nbar+1)*exp(inf_md_ss*(theta_m-1)-inf_md_ss* &
       (theta_m-1)+G_ss*(1-sigma_m)+PSIM_ss+lambda_ss-lambda_ss)
  Amat(100, 53)=& 
       betta*epsrm*(nbar+1)*(1-sigma_m)*exp(inf_rm_ss*theta_rm-inf_rm_ss* &
       theta_rm+THETARM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(100, 114)=& 
       betta*epsrm*(nbar+1)*theta_rm*exp(inf_rm_ss*theta_rm-inf_rm_ss* &
       theta_rm+THETARM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(100, 119)=& 
       betta*epsrm*(nbar+1)*exp(inf_rm_ss*theta_rm-inf_rm_ss* &
       theta_rm+THETARM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(100, 206)=& 
       betta*epsrm*(nbar+1)*exp(inf_rm_ss*theta_rm-inf_rm_ss* &
       theta_rm+THETARM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Amat(101, 53)=& 
       betta*epsrm*(nbar+1)*(1-sigma_m)*exp(inf_rm_ss* &
       (theta_rm-1)-inf_rm_ss*(theta_rm-1)+G_ss* &
       (1-sigma_m)+PSIRM_ss+lambda_ss-lambda_ss)
  Amat(101, 114)=& 
       betta*epsrm*(nbar+1)*(theta_rm-1)*exp(inf_rm_ss* &
       (theta_rm-1)-inf_rm_ss*(theta_rm-1)+G_ss* &
       (1-sigma_m)+PSIRM_ss+lambda_ss-lambda_ss)
  Amat(101, 119)=& 
       betta*epsrm*(nbar+1)*exp(inf_rm_ss*(theta_rm-1)-inf_rm_ss* &
       (theta_rm-1)+G_ss*(1-sigma_m)+PSIRM_ss+lambda_ss-lambda_ss)
  Amat(101, 170)=& 
       betta*epsrm*(nbar+1)*exp(inf_rm_ss*(theta_rm-1)-inf_rm_ss* &
       (theta_rm-1)+G_ss*(1-sigma_m)+PSIRM_ss+lambda_ss-lambda_ss)
  Amat(108, 223)=& 
       abs(istar)**(4*(1-rho_i))*abs(pi_bar)**(4*(1-rho_i)-4*(1-rho_i)* &
       rho_pi)*(pib_bar**rho_y)**(rho_i-1)*(1-rho_i)*rho_y* &
       exp(zi_ss+(1-rho_i)*(rho_y*yngdp_pc_ss+rho_pi* &
       yinf_c_ss)+lag1_yi_ss*rho_i)/abs(pi_star_bar)**(4*(1-rho_i))
  Amat(109, 50)=& 
       istar*OMEG_U*exp((exp(s_pc_ss-ngdp_pc_ss+f_ss)-fbar)* &
       OMEG_U-(exp(d_ss+d_ss)*pi_star_bar**2/ &
       pi_bar**2-1)*OMEG_S+zie_ss+s_pc_ss-ngdp_pc_ss+f_ss)
  Amat(109, 91)=& 
       -istar*pi_star_bar**2*OMEG_S* &
       exp((exp(s_pc_ss-ngdp_pc_ss+f_ss)-fbar)*OMEG_U-(exp(d_ss+d_ss)* &
       pi_star_bar**2/pi_bar**2-1)*OMEG_S+zie_ss+d_ss+d_ss)/ &
       pi_bar**2
  Amat(131, 1)=& 
       -exp(lag1_c_ss)
  Amat(132, 2)=& 
       -exp(lag1_cd_ss)
  Amat(133, 3)=& 
       -exp(lag1_cm_ss)
  Amat(134, 4)=& 
       -exp(lag1_e_ss)
  Amat(135, 5)=& 
       -exp(lag1_ie_ss)
  Amat(136, 6)=& 
       -exp(lag1_inf_c_ss)
  Amat(137, 7)=& 
       -exp(lag2_inf_c_ss)
  Amat(138, 8)=& 
       -exp(lag3_inf_c_ss)
  Amat(139, 9)=& 
       -exp(lag1_inf_cd_ss)
  Amat(140, 10)=& 
       -exp(lag1_inf_e_ss)
  Amat(141, 11)=& 
       -exp(lag1_inf_md_ss)
  Amat(142, 12)=& 
       -exp(lag1_inf_q_ss)
  Amat(143, 13)=& 
       -exp(lag1_inf_rm_ss)
  Amat(144, 14)=& 
       -exp(lag1_inf_t_ss)
  Amat(145, 15)=& 
       -exp(lag1_inf_xd_ss)
  Amat(146, 16)=& 
       -exp(lag1_md_ss)
  Amat(147, 17)=& 
       -exp(lag1_ms_ss)
  Amat(148, 18)=& 
       -exp(lag1_ngdp_pc_ss)
  Amat(149, 19)=& 
       -exp(lag2_ngdp_pc_ss)
  Amat(150, 20)=& 
       -exp(lag3_ngdp_pc_ss)
  Amat(151, 21)=& 
       -exp(lag1_pcd_pc_ss)
  Amat(152, 22)=& 
       -exp(lag1_pcd_pq_ss)
  Amat(153, 23)=& 
       -exp(lag1_pe_pq_ss)
  Amat(154, 24)=& 
       -exp(lag1_pmd_pc_ss)
  Amat(155, 25)=& 
       -exp(lag1_prm_pc_ss)
  Amat(156, 26)=& 
       -exp(lag1_pt_pq_ss)
  Amat(157, 27)=& 
       -exp(lag1_pvcd_ss)
  Amat(158, 28)=& 
       -exp(lag1_pve_ss)
  Amat(159, 29)=& 
       -exp(lag1_pvm_ss)
  Amat(160, 30)=& 
       -exp(lag1_pvq_ss)
  Amat(161, 31)=& 
       -exp(lag1_pvrm_ss)
  Amat(162, 32)=& 
       -exp(lag1_pvt_ss)
  Amat(163, 33)=& 
       -exp(lag1_pvxd_ss)
  Amat(164, 34)=& 
       -exp(lag1_px_pc_ss)
  Amat(165, 35)=& 
       -exp(lag1_pxd_pcd_ss)
  Amat(166, 36)=& 
       -exp(lag1_qd_ss)
  Amat(167, 37)=& 
       -exp(lag1_rm_ss)
  Amat(168, 38)=& 
       -exp(lag1_s_pc_ss)
  Amat(169, 39)=& 
       -exp(lag1_t_ss)
  Amat(170, 40)=& 
       -exp(lag1_w_ss)
  Amat(171, 41)=& 
       -exp(lag1_x_ss)
  Amat(172, 42)=& 
       -exp(lag1_xd_ss)
  Amat(173, 43)=& 
       -exp(lag1_xm_ss)
  Amat(174, 44)=& 
       -exp(lag1_yi_ss)
  Amat(175, 48)=& 
       -exp(lag1_f_ss)
  Amat(176, 49)=& 
       -exp(lag1_k_ss)
  Amat(177, 45)=& 
       -exp(lag1_cstar_ss)
  Amat(178, 46)=& 
       -exp(lag1_qm_ss)
  Amat(179, 47)=& 
       -exp(lag1_tr_ss)
  Amat(180, 222)=& 
       exp(yinf_c_ss)
  Amat(181, 120)=& 
       exp(led1_yinf_c_ss)
  Amat(182, 121)=& 
       exp(led2_yinf_c_ss)
  Amat(183, 122)=& 
       exp(led3_yinf_c_ss)
  Amat(184, 123)=& 
       exp(led4_yinf_c_ss)
  Amat(185, 124)=& 
       exp(led5_yinf_c_ss)
  Amat(204, 69)=& 
       -1
  Amat(205, 62)=& 
       -1
  Amat(206, 67)=& 
       -1
  Amat(207, 60)=& 
       -1
  Amat(208, 66)=& 
       -1
  Amat(209, 65)=& 
       -1
  Amat(210, 68)=& 
       -1
  Amat(211, 70)=& 
       -1
  Amat(212, 64)=& 
       -1
  Amat(213, 63)=& 
       -1
  Amat(214, 55)=& 
       -1
  Amat(215, 56)=& 
       -1
  Amat(216, 57)=& 
       -1
  Amat(217, 52)=& 
       -1
  Amat(218, 54)=& 
       -1
  Amat(219, 59)=& 
       -1
  Amat(220, 71)=& 
       -1
  Amat(221, 61)=& 
       -1
  Amat(222, 58)=& 
       -1
  Amat(223, 53)=& 
       -1

  Bmat(1, 72)=& 
       -(1-gama)**(1/omega)*exp((cm_ss+adjcost_c_ss)*(omega-1)/ &
       omega)*((1-gama)**(1/ &
       omega)*exp((cm_ss+adjcost_c_ss)*(omega-1)/omega)+gama**(1/ &
       omega)*exp(cd_ss*(omega-1)/omega))**(omega/(omega-1)-1)
  Bmat(1, 74)=& 
       exp(c_ss)
  Bmat(1, 75)=& 
       -gama**(1/omega)*exp(cd_ss*(omega-1)/omega)*((1-gama)**(1/ &
       omega)*exp((cm_ss+adjcost_c_ss)*(omega-1)/omega)+gama**(1/ &
       omega)*exp(cd_ss*(omega-1)/omega))**(omega/(omega-1)-1)
  Bmat(1, 78)=& 
       -(1-gama)**(1/omega)*exp((cm_ss+adjcost_c_ss)*(omega-1)/ &
       omega)*((1-gama)**(1/ &
       omega)*exp((cm_ss+adjcost_c_ss)*(omega-1)/omega)+gama**(1/ &
       omega)*exp(cd_ss*(omega-1)/omega))**(omega/(omega-1)-1)
  Bmat(2, 2)=& 
       (exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c
  Bmat(2, 3)=& 
       -(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c
  Bmat(2, 72)=& 
       exp(adjcost_c_ss)
  Bmat(2, 75)=& 
       -(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c
  Bmat(2, 78)=& 
       (exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c
  Bmat(3, 2)=& 
       -(1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c-(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-2*lag1_cm_ss+2*lag1_cd_ss+3*cm_ss-3*cd_ss)* &
       psi_adjcost_c
  Bmat(3, 3)=& 
       (1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c+(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-2*lag1_cm_ss+2*lag1_cd_ss+3*cm_ss-3*cd_ss)* &
       psi_adjcost_c
  Bmat(3, 72)=& 
       (1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c/omega
  Bmat(3, 74)=& 
       -(1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c/omega-gama**(1/omega)*exp((c_ss-cd_ss)/ &
       omega+lambda_ss)/omega
  Bmat(3, 75)=& 
       betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-2*cm_ss-2*cd_ss+cd_ss)+2* &
       (1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c+(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-2*lag1_cm_ss+2*lag1_cd_ss+3*cm_ss-3*cd_ss)* &
       psi_adjcost_c+gama**(1/omega)*exp((c_ss-cd_ss)/ &
       omega+lambda_ss)/omega
  Bmat(3, 78)=& 
       -betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-2*cm_ss-2*cd_ss+cd_ss)-betta* &
       (exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-cm_ss-cd_ss)-(1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)*(2-1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c-(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-2*lag1_cm_ss+2*lag1_cd_ss+3*cm_ss-3*cd_ss)* &
       psi_adjcost_c
  Bmat(3, 119)=& 
       -(1-gama)**(1/ &
       omega)*(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss-lag1_cm_ss+lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c+exp(pcd_pc_ss+lambda_ss)-gama**(1/ &
       omega)*exp((c_ss-cd_ss)/omega+lambda_ss)
  Bmat(3, 131)=& 
       exp(pcd_pc_ss+lambda_ss)
  Bmat(4, 2)=& 
       -(1-gama)**(1/omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)*(-(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c-exp(-2* &
       lag1_cm_ss+2*lag1_cd_ss+2*cm_ss-2*cd_ss)*psi_adjcost_c)
  Bmat(4, 3)=& 
       -(1-gama)**(1/omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)*((exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c+exp(-2* &
       lag1_cm_ss+2*lag1_cd_ss+2*cm_ss-2*cd_ss)*psi_adjcost_c)
  Bmat(4, 72)=& 
       (1-gama)**(1/omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)* &
       (exp(adjcost_c_ss)-(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c)/ &
       omega-(1-gama)**(1/omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+adjcost_c_ss)
  Bmat(4, 74)=& 
       -(1-gama)**(1/omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)* &
       (exp(adjcost_c_ss)-(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c)/omega
  Bmat(4, 75)=& 
       -betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-3*cm_ss-2*cd_ss+2* &
       cd_ss)-betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2* &
       cm_ss-cd_ss+cd_ss)-(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)*((exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c+exp(-2* &
       lag1_cm_ss+2*lag1_cd_ss+2*cm_ss-2*cd_ss)*psi_adjcost_c)
  Bmat(4, 78)=& 
       betta*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+3*cm_ss-3*cm_ss-2*cd_ss+2*cd_ss)+2* &
       betta*(exp(cm_ss-cm_ss-cd_ss+cd_ss)-1)*(1-gama)**(1/ &
       omega)*(nbar+1)*psi_adjcost_c*exp(-G_ss* &
       sigma_m+(c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss+inf_c_ss+2*cm_ss-2* &
       cm_ss-cd_ss+cd_ss)-(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)*(-(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c-exp(-2* &
       lag1_cm_ss+2*lag1_cd_ss+2*cm_ss-2*cd_ss)* &
       psi_adjcost_c)+(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)* &
       (exp(adjcost_c_ss)-(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c)/omega
  Bmat(4, 119)=& 
       exp(pmd_pc_ss+lambda_ss)-(1-gama)**(1/ &
       omega)*exp((c_ss-cm_ss-adjcost_c_ss)/ &
       omega+lambda_ss)* &
       (exp(adjcost_c_ss)-(exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)-1)* &
       exp(-lag1_cm_ss+lag1_cd_ss+cm_ss-cd_ss)*psi_adjcost_c)
  Bmat(4, 144)=& 
       exp(pmd_pc_ss+lambda_ss)
  Bmat(5, 21)=& 
       exp(pcd_pc_ss-lag1_pcd_pc_ss)
  Bmat(5, 108)=& 
       -exp(inf_cd_ss-inf_c_ss)
  Bmat(5, 109)=& 
       exp(inf_cd_ss-inf_c_ss)
  Bmat(5, 131)=& 
       -exp(pcd_pc_ss-lag1_pcd_pc_ss)
  Bmat(6, 98)=& 
       exp(delta_ss)
  Bmat(6, 212)=& 
       -Psi_2*exp((upsilon+1)*u_ss)
  Bmat(7, 51)=& 
       -(1-exp(delta_ss))*exp(k_ss-G_ss)/(nbar+1)
  Bmat(7, 53)=& 
       (1-exp(delta_ss))*exp(k_ss-G_ss)/(nbar+1)
  Bmat(7, 58)=& 
       -exp(z_xdemand_ss+x_ss)
  Bmat(7, 98)=& 
       exp(k_ss-G_ss+delta_ss)/(nbar+1)
  Bmat(7, 216)=& 
       -exp(z_xdemand_ss+x_ss)
  Bmat(8, 1)=& 
       -hab*(exp(c_ss)-hab*exp(lag1_c_ss))**(-sigma_m-1)*sigma_m* &
       exp(zu_ss+lag1_c_ss)
  Bmat(8, 69)=& 
       -exp(zu_ss)/(exp(c_ss)-hab*exp(lag1_c_ss))**sigma_m
  Bmat(8, 74)=& 
       (exp(c_ss)-hab*exp(lag1_c_ss))**(-sigma_m-1)*sigma_m*exp(zu_ss+c_ss)
  Bmat(8, 119)=& 
       exp(lambda_ss)
  Bmat(9, 6)=& 
       -epsw*exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)*(1-theta_w))*((1-epsw)* &
       exp((1-theta_w)*wopt_ss)+epsw* &
       exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)*(1-theta_w)))**(1/ &
       (1-theta_w)-1)
  Bmat(9, 40)=& 
       -epsw*exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)*(1-theta_w))*((1-epsw)* &
       exp((1-theta_w)*wopt_ss)+epsw* &
       exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)*(1-theta_w)))**(1/ &
       (1-theta_w)-1)
  Bmat(9, 108)=& 
       -epsw*(theta_w-1)*exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)* &
       (1-theta_w))*((1-epsw)*exp((1-theta_w)*wopt_ss)+epsw* &
       exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)*(1-theta_w)))**(1/ &
       (1-theta_w)-1)/(1-theta_w)
  Bmat(9, 214)=& 
       exp(w_ss)
  Bmat(9, 215)=& 
       -(1-epsw)*exp((1-theta_w)*wopt_ss)*((1-epsw)*exp((1-theta_w)* &
       wopt_ss)+epsw*exp((lag1_w_ss+lag1_inf_c_ss-inf_c_ss)* &
       (1-theta_w)))**(1/(1-theta_w)-1)
  Bmat(10, 102)=& 
       exp(f1_ss)
  Bmat(10, 105)=& 
       -tbp*(1-td)*(theta_w-1)*exp(-theta_w* &
       (wopt_ss-w_ss)+wopt_ss+lambda_ss+h_ss)
  Bmat(10, 108)=& 
       -betta*epsw*(nbar+1)*(1-theta_w)*exp((1-theta_w)* &
       (wopt_ss-wopt_ss)+(inf_c_ss-inf_c_ss)*(1-theta_w)+G_ss* &
       (1-sigma_m)+f1_ss)
  Bmat(10, 119)=& 
       -tbp*(1-td)*(theta_w-1)*exp(-theta_w* &
       (wopt_ss-w_ss)+wopt_ss+lambda_ss+h_ss)
  Bmat(10, 214)=& 
       -tbp*(1-td)*(theta_w-1)*theta_w*exp(-theta_w* &
       (wopt_ss-w_ss)+wopt_ss+lambda_ss+h_ss)
  Bmat(10, 215)=& 
       -tbp*(1-td)*(1-theta_w)*(theta_w-1)*exp(-theta_w* &
       (wopt_ss-w_ss)+wopt_ss+lambda_ss+h_ss)-betta*epsw*(nbar+1)* &
       (1-theta_w)*exp((1-theta_w)* &
       (wopt_ss-wopt_ss)+(inf_c_ss-inf_c_ss)*(1-theta_w)+G_ss* &
       (1-sigma_m)+f1_ss)
  Bmat(11, 62)=& 
       -(tbp*(1-td))**(eta_m+1)*theta_w*exp(zh_ss+(eta_m+1)*(h_ss-theta_w* &
       (wopt_ss-w_ss)))
  Bmat(11, 103)=& 
       exp(f2_ss)
  Bmat(11, 105)=& 
       -(eta_m+1)*(tbp*(1-td))**(eta_m+1)*theta_w*exp(zh_ss+(eta_m+1)* &
       (h_ss-theta_w*(wopt_ss-w_ss)))
  Bmat(11, 108)=& 
       betta*epsw*(eta_m+1)*(nbar+1)*theta_w*exp(-(eta_m+1)*theta_w* &
       (-wopt_ss+wopt_ss-inf_c_ss+inf_c_ss)+G_ss*(1-sigma_m)+f2_ss)
  Bmat(11, 214)=& 
       -(eta_m+1)*(tbp*(1-td))**(eta_m+1)*theta_w**2*exp(zh_ss+(eta_m+1)* &
       (h_ss-theta_w*(wopt_ss-w_ss)))
  Bmat(11, 215)=& 
       (eta_m+1)*(tbp*(1-td))**(eta_m+1)*theta_w**2*exp(zh_ss+(eta_m+1)* &
       (h_ss-theta_w*(wopt_ss-w_ss)))+betta*epsw*(eta_m+1)*(nbar+1)* &
       theta_w*exp(-(eta_m+1)*theta_w* &
       (-wopt_ss+wopt_ss-inf_c_ss+inf_c_ss)+G_ss*(1-sigma_m)+f2_ss)
  Bmat(12, 102)=& 
       exp(f1_ss)
  Bmat(12, 103)=& 
       -exp(f2_ss)
  Bmat(13, 41)=& 
       -psi_x*exp(-x_ss+lambda_ss+lag1_x_ss)
  Bmat(13, 58)=& 
       -exp(z_xdemand_ss+gam_x_ss)
  Bmat(13, 104)=& 
       -exp(z_xdemand_ss+gam_x_ss)
  Bmat(13, 119)=& 
       psi_x*exp(lambda_ss-x_ss)* &
       (exp(x_ss)-exp(lag1_x_ss))+exp(px_pc_ss+lambda_ss)
  Bmat(13, 184)=& 
       exp(px_pc_ss+lambda_ss)
  Bmat(13, 216)=& 
       betta*(nbar+1)*exp(-x_ss+G_ss*(1-sigma_m)+lambda_ss)*(psi_x* &
       exp(-x_ss)*(exp(x_ss)-exp(x_ss))**2/ &
       2.0+psi_x*(exp(x_ss)-exp(x_ss)))-betta*(nbar+1)* &
       exp(-x_ss+G_ss*(1-sigma_m)+lambda_ss)*(-psi_x*exp(-x_ss)* &
       (exp(x_ss)-exp(x_ss))**2/ &
       2.0-psi_x*(exp(x_ss)-exp(x_ss))-psi_x*exp(x_ss))-psi_x* &
       exp(lambda_ss-x_ss)*(exp(x_ss)-exp(lag1_x_ss))+exp(lambda_ss)* &
       psi_x
  Bmat(14, 98)=& 
       betta*exp(-G_ss*sigma_m+gam_x_ss+delta_ss)
  Bmat(14, 104)=& 
       exp(gam_x_ss)
  Bmat(15, 104)=& 
       -Psi_2*exp(upsilon*u_ss-lambda_ss+gam_x_ss)
  Bmat(15, 119)=& 
       Psi_2*exp(upsilon*u_ss-lambda_ss+gam_x_ss)
  Bmat(15, 194)=& 
       exp(rk_ss)
  Bmat(15, 212)=& 
       -Psi_2*upsilon*exp(upsilon*u_ss-lambda_ss+gam_x_ss)
  Bmat(16, 106)=& 
       -betta*exp(-G_ss*sigma_m+lambda_ss+i_ss-inf_c_ss)
  Bmat(16, 119)=& 
       exp(lambda_ss)
  Bmat(17, 107)=& 
       -betta*exp(-G_ss*sigma_m+lambda_ss-inf_c_ss+ie_ss+d_ss)
  Bmat(17, 119)=& 
       exp(lambda_ss)
  Bmat(18, 5)=& 
       -exp(s_pc_ss-pi_star_ss+lag1_ie_ss-G_ss+f_ss)/(nbar+1)
  Bmat(18, 50)=& 
       -exp(s_pc_ss-pi_star_ss+lag1_ie_ss-G_ss+f_ss)/(nbar+1)
  Bmat(18, 53)=& 
       exp(s_pc_ss-pi_star_ss+lag1_ie_ss-G_ss+f_ss)/(nbar+1)
  Bmat(18, 54)=& 
       exp(s_pc_ss-pi_star_ss+lag1_ie_ss-G_ss+f_ss)/(nbar+1)
  Bmat(18, 57)=& 
       exp(tr_ss+s_pc_ss)
  Bmat(18, 74)=& 
       -exp(c_ss)
  Bmat(18, 129)=& 
       exp(ngdp_pc_ss)
  Bmat(18, 184)=& 
       -exp(x_ss+px_pc_ss)
  Bmat(18, 198)=& 
       exp(tr_ss+s_pc_ss)-exp(s_pc_ss-pi_star_ss+lag1_ie_ss-G_ss+f_ss)/ &
       (nbar+1)+exp(s_pc_ss+f_ss)
  Bmat(18, 216)=& 
       -exp(x_ss+px_pc_ss)
  Bmat(19, 70)=& 
       -((1-gama_x)**(1/omega_x)*exp((omega_x-1)*(xm_ss+adjcost_x_ss)/ &
       omega_x)+gama_x**(1/omega_x)*exp((omega_x-1)*xd_ss/ &
       omega_x))**(omega_x/(omega_x-1))*exp(zx_ss)
  Bmat(19, 73)=& 
       -(1-gama_x)**(1/omega_x)*((1-gama_x)**(1/ &
       omega_x)*exp((omega_x-1)*(xm_ss+adjcost_x_ss)/ &
       omega_x)+gama_x**(1/omega_x)*exp((omega_x-1)*xd_ss/ &
       omega_x))**(omega_x/ &
       (omega_x-1)-1)*exp(zx_ss+(omega_x-1)*(xm_ss+adjcost_x_ss)/ &
       omega_x)
  Bmat(19, 216)=& 
       exp(x_ss)
  Bmat(19, 217)=& 
       -gama_x**(1/omega_x)*((1-gama_x)**(1/ &
       omega_x)*exp((omega_x-1)*(xm_ss+adjcost_x_ss)/ &
       omega_x)+gama_x**(1/omega_x)*exp((omega_x-1)*xd_ss/ &
       omega_x))**(omega_x/ &
       (omega_x-1)-1)*exp(zx_ss+(omega_x-1)*xd_ss/omega_x)
  Bmat(19, 220)=& 
       -(1-gama_x)**(1/omega_x)*((1-gama_x)**(1/ &
       omega_x)*exp((omega_x-1)*(xm_ss+adjcost_x_ss)/ &
       omega_x)+gama_x**(1/omega_x)*exp((omega_x-1)*xd_ss/ &
       omega_x))**(omega_x/ &
       (omega_x-1)-1)*exp(zx_ss+(omega_x-1)*(xm_ss+adjcost_x_ss)/ &
       omega_x)
  Bmat(20, 42)=& 
       psi_adjcost_x*(exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(20, 43)=& 
       -psi_adjcost_x*(exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(20, 73)=& 
       exp(adjcost_x_ss)
  Bmat(20, 217)=& 
       -psi_adjcost_x*(exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(20, 220)=& 
       psi_adjcost_x*(exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(21, 42)=& 
       -(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xd_ss+lambda_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)-(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(21, 43)=& 
       (1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xd_ss+lambda_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)+(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(21, 70)=& 
       -(1-gama_x)**(1/omega_x)*(1-1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)-gama_x** &
       (1/omega_x)*(1-1/omega_x)*exp(zx_ss+(-zx_ss+x_ss-xd_ss)/ &
       omega_x+lambda_ss)
  Bmat(21, 73)=& 
       (1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)/omega_x
  Bmat(21, 119)=& 
       -(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)-gama_x** &
       (1/omega_x)*exp(zx_ss+(-zx_ss+x_ss-xd_ss)/ &
       omega_x+lambda_ss)+exp(pxd_px_ss+lambda_ss)
  Bmat(21, 189)=& 
       exp(pxd_px_ss+lambda_ss)
  Bmat(21, 216)=& 
       -(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)/ &
       omega_x-gama_x**(1/omega_x)*exp(zx_ss+(-zx_ss+x_ss-xd_ss)/ &
       omega_x+lambda_ss)/omega_x
  Bmat(21, 217)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-2*xm_ss-2*xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)+(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xd_ss+lambda_ss-2*lag1_xm_ss+2*lag1_xd_ss)+2* &
       (1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)+gama_x** &
       (1/omega_x)*exp(zx_ss+(-zx_ss+x_ss-xd_ss)/omega_x+lambda_ss)/ &
       omega_x
  Bmat(21, 220)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-2*xm_ss-2*xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-xm_ss-xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-(1-gama_x)**(1/ &
       omega_x)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xd_ss+lambda_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)-(1-gama_x)**(1/omega_x)*(2-1/ &
       omega_x)*psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xd_ss+lambda_ss-lag1_xm_ss+lag1_xd_ss)
  Bmat(22, 42)=& 
       -(1-gama_x)**(1/ &
       omega_x)*(-psi_adjcost_x*exp(2*xm_ss-2*xd_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)
  Bmat(22, 43)=& 
       -(1-gama_x)**(1/ &
       omega_x)*(psi_adjcost_x*exp(2*xm_ss-2*xd_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)+psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)
  Bmat(22, 70)=& 
       -(1-gama_x)**(1/omega_x)*(1-1/ &
       omega_x)*(exp(adjcost_x_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)
  Bmat(22, 73)=& 
       (1-gama_x)**(1/ &
       omega_x)*(exp(adjcost_x_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)/ &
       omega_x-(1-gama_x)**(1/ &
       omega_x)*exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+lambda_ss+adjcost_x_ss)
  Bmat(22, 119)=& 
       exp(pmd_px_ss+lambda_ss)-(1-gama_x)**(1/ &
       omega_x)*(exp(adjcost_x_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)
  Bmat(22, 145)=& 
       exp(pmd_px_ss+lambda_ss)
  Bmat(22, 216)=& 
       -(1-gama_x)**(1/ &
       omega_x)*(exp(adjcost_x_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)/ &
       omega_x
  Bmat(22, 217)=& 
       -betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xm_ss-2*xd_ss+2*xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-(1-gama_x)**(1/ &
       omega_x)*(psi_adjcost_x*exp(2*xm_ss-2*xd_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)+psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)
  Bmat(22, 220)=& 
       betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+3*xm_ss-3*xm_ss-2*xd_ss+2*xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)+2*betta*(1-gama_x)**(1/ &
       omega_x)*(nbar+1)*psi_adjcost_x* &
       (exp(xm_ss-xm_ss-xd_ss+xd_ss)-1)* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+2*xm_ss-2*xm_ss-xd_ss+xd_ss-G_ss* &
       sigma_m+lambda_ss+inf_x_ss)-(1-gama_x)**(1/ &
       omega_x)*(-psi_adjcost_x*exp(2*xm_ss-2*xd_ss-2*lag1_xm_ss+2* &
       lag1_xd_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/ &
       omega_x+lambda_ss)+(1-gama_x)**(1/ &
       omega_x)*(exp(adjcost_x_ss)-psi_adjcost_x* &
       (exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss)-1)* &
       exp(xm_ss-xd_ss-lag1_xm_ss+lag1_xd_ss))* &
       exp(zx_ss+(-zx_ss+x_ss-xm_ss-adjcost_x_ss)/omega_x+lambda_ss)/ &
       omega_x
  Bmat(23, 34)=& 
       exp(px_pc_ss-lag1_px_pc_ss)
  Bmat(23, 108)=& 
       -exp(inf_x_ss-inf_c_ss)
  Bmat(23, 116)=& 
       exp(inf_x_ss-inf_c_ss)
  Bmat(23, 184)=& 
       -exp(px_pc_ss-lag1_px_pc_ss)
  Bmat(24, 157)=& 
       -exp(-px_pc_ss+pxd_pq_ss+pq_pc_ss)
  Bmat(24, 184)=& 
       exp(-px_pc_ss+pxd_pq_ss+pq_pc_ss)
  Bmat(24, 188)=& 
       -exp(-px_pc_ss+pxd_pq_ss+pq_pc_ss)
  Bmat(24, 189)=& 
       exp(pxd_px_ss)
  Bmat(25, 144)=& 
       -exp(pmd_pc_ss-px_pc_ss)
  Bmat(25, 145)=& 
       exp(pmd_px_ss)
  Bmat(25, 184)=& 
       exp(pmd_pc_ss-px_pc_ss)
  Bmat(26, 67)=& 
       -exp(zq_ss)*((1-alfa)**(1/rhoq)*exp(zrm_ss+(rhoq-1)*rm_ss/ &
       rhoq)+alfa**(1/rhoq)*exp((rhoq-1)*vq_ss/rhoq))**(rhoq/ &
       (rhoq-1))
  Bmat(26, 68)=& 
       -(1-alfa)**(1/rhoq)*rhoq*((1-alfa)**(1/ &
       rhoq)*exp(zrm_ss+(rhoq-1)*rm_ss/rhoq)+alfa**(1/ &
       rhoq)*exp((rhoq-1)*vq_ss/rhoq))**(rhoq/ &
       (rhoq-1)-1)*exp(zrm_ss+zq_ss+(rhoq-1)*rm_ss/rhoq)/(rhoq-1)
  Bmat(26, 193)=& 
       exp(qs_ss)
  Bmat(26, 195)=& 
       -(1-alfa)**(1/rhoq)*((1-alfa)**(1/rhoq)*exp(zrm_ss+(rhoq-1)*rm_ss/ &
       rhoq)+alfa**(1/rhoq)*exp((rhoq-1)*vq_ss/rhoq))**(rhoq/ &
       (rhoq-1)-1)*exp(zrm_ss+zq_ss+(rhoq-1)*rm_ss/rhoq)
  Bmat(26, 213)=& 
       -alfa**(1/rhoq)*exp(zq_ss+(rhoq-1)*vq_ss/rhoq)*((1-alfa)**(1/ &
       rhoq)*exp(zrm_ss+(rhoq-1)*rm_ss/rhoq)+alfa**(1/ &
       rhoq)*exp((rhoq-1)*vq_ss/rhoq))**(rhoq/(rhoq-1)-1)
  Bmat(27, 105)=& 
       -(1-alfav)**(1/rhoqv)*exp(h_ss*(rhoqv-1)/rhoqv)*((1-alfav)**(1/ &
       rhoqv)*exp(h_ss*(rhoqv-1)/rhoqv)*(tbp*(1-td))**((rhoqv-1)/ &
       rhoqv)+alfav**(1/rhoqv)*exp(ks_ss*(rhoqv-1)/rhoqv))**(rhoqv/ &
       (rhoqv-1)-1)*(tbp*(1-td))**((rhoqv-1)/rhoqv)
  Bmat(27, 118)=& 
       -alfav**(1/rhoqv)*exp(ks_ss*(rhoqv-1)/rhoqv)*((1-alfav)**(1/ &
       rhoqv)*exp(h_ss*(rhoqv-1)/rhoqv)*(tbp*(1-td))**((rhoqv-1)/ &
       rhoqv)+alfav**(1/rhoqv)*exp(ks_ss*(rhoqv-1)/rhoqv))**(rhoqv/ &
       (rhoqv-1)-1)
  Bmat(27, 213)=& 
       exp(vq_ss)
  Bmat(28, 12)=& 
       epsq*theta*exp(lag1_pvq_ss-(lag1_inf_q_ss-inf_q_ss)*theta)
  Bmat(28, 30)=& 
       -epsq*exp(lag1_pvq_ss-(lag1_inf_q_ss-inf_q_ss)*theta)
  Bmat(28, 113)=& 
       -epsq*theta*exp(lag1_pvq_ss-(lag1_inf_q_ss-inf_q_ss)*theta)
  Bmat(28, 150)=& 
       (1-epsq)*theta*exp(-popt_pq_ss*theta)
  Bmat(28, 180)=& 
       exp(pvq_ss)
  Bmat(29, 180)=& 
       exp(qd_ss+pvq_ss)
  Bmat(29, 191)=& 
       exp(qd_ss+pvq_ss)
  Bmat(29, 193)=& 
       -exp(qs_ss)
  Bmat(30, 67)=& 
       -alfa**(1/rhoq)*(1-1/rhoq)*((1-alfav)/(tbp*(1-td)))**(1/ &
       rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-h_ss)/ &
       rhoqv+phi_ss)
  Bmat(30, 105)=& 
       alfa**(1/rhoq)*((1-alfav)/(tbp*(1-td)))**(1/ &
       rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-h_ss)/ &
       rhoqv+phi_ss)/rhoqv
  Bmat(30, 137)=& 
       -alfa**(1/rhoq)*((1-alfav)/(tbp*(1-td)))**(1/ &
       rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-h_ss)/ &
       rhoqv+phi_ss)
  Bmat(30, 193)=& 
       -alfa**(1/rhoq)*((1-alfav)/(tbp*(1-td)))**(1/ &
       rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-h_ss)/ &
       rhoqv+phi_ss)/rhoq
  Bmat(30, 213)=& 
       -alfa**(1/rhoq)*(1/rhoqv-1/rhoq)*((1-alfav)/(tbp*(1-td)))**(1/ &
       rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-h_ss)/ &
       rhoqv+phi_ss)
  Bmat(30, 214)=& 
       exp(w_ss)
  Bmat(31, 67)=& 
       -alfa**(1/rhoq)*alfav**(1/rhoqv)*(1-1/ &
       rhoq)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-ks_ss)/ &
       rhoqv+phi_ss)
  Bmat(31, 118)=& 
       alfa**(1/rhoq)*alfav**(1/rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/ &
       rhoq+(vq_ss-ks_ss)/rhoqv+phi_ss)/rhoqv
  Bmat(31, 137)=& 
       -alfa**(1/rhoq)*alfav**(1/rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/ &
       rhoq+(vq_ss-ks_ss)/rhoqv+phi_ss)
  Bmat(31, 193)=& 
       -alfa**(1/rhoq)*alfav**(1/rhoqv)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/ &
       rhoq+(vq_ss-ks_ss)/rhoqv+phi_ss)/rhoq
  Bmat(31, 194)=& 
       exp(rk_ss)
  Bmat(31, 213)=& 
       -alfa**(1/rhoq)*alfav**(1/rhoqv)*(1/rhoqv-1/ &
       rhoq)*exp(zq_ss+(-zq_ss-vq_ss+qs_ss)/rhoq+(vq_ss-ks_ss)/ &
       rhoqv+phi_ss)
  Bmat(32, 67)=& 
       -(1-alfa)**(1/rhoq)*(1-1/rhoq)*exp(zrm_ss+zq_ss+(-zq_ss-rm_ss+qs_ss)/ &
       rhoq+phi_ss)
  Bmat(32, 68)=& 
       -(1-alfa)**(1/rhoq)*exp(zrm_ss+zq_ss+(-zq_ss-rm_ss+qs_ss)/rhoq+phi_ss)
  Bmat(32, 137)=& 
       -(1-alfa)**(1/rhoq)*exp(zrm_ss+zq_ss+(-zq_ss-rm_ss+qs_ss)/rhoq+phi_ss)
  Bmat(32, 158)=& 
       exp(prm_pc_ss)
  Bmat(32, 193)=& 
       -(1-alfa)**(1/rhoq)*exp(zrm_ss+zq_ss+(-zq_ss-rm_ss+qs_ss)/rhoq+phi_ss)/ &
       rhoq
  Bmat(32, 195)=& 
       (1-alfa)**(1/rhoq)*exp(zrm_ss+zq_ss+(-zq_ss-rm_ss+qs_ss)/rhoq+phi_ss)/ &
       rhoq
  Bmat(33, 51)=& 
       -exp(u_ss+k_ss-G_ss)/(nbar+1)
  Bmat(33, 53)=& 
       exp(u_ss+k_ss-G_ss)/(nbar+1)
  Bmat(33, 118)=& 
       exp(ks_ss)
  Bmat(33, 212)=& 
       -exp(u_ss+k_ss-G_ss)/(nbar+1)
  Bmat(34, 150)=& 
       exp(popt_pq_ss)
  Bmat(34, 169)=& 
       theta*exp(THETAP_ss-PSIP_ss)/(theta-1)
  Bmat(34, 205)=& 
       -theta*exp(THETAP_ss-PSIP_ss)/(theta-1)
  Bmat(35, 113)=& 
       betta*epsq*(nbar+1)*theta*exp(THETAP_ss+inf_q_ss*theta-inf_q_ss* &
       theta+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(35, 119)=& 
       betta*epsq*(nbar+1)*exp(THETAP_ss+inf_q_ss*theta-inf_q_ss* &
       theta+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(35, 137)=& 
       -exp(qd_ss+phi_ss)
  Bmat(35, 191)=& 
       -exp(qd_ss+phi_ss)
  Bmat(35, 205)=& 
       exp(THETAP_ss)
  Bmat(36, 113)=& 
       -betta*epsq*(nbar+1)*(1-theta)*exp(inf_q_ss*(theta-1)-inf_q_ss* &
       (theta-1)+G_ss*(1-sigma_m)+PSIP_ss+lambda_ss-lambda_ss)
  Bmat(36, 119)=& 
       betta*epsq*(nbar+1)*exp(inf_q_ss*(theta-1)-inf_q_ss* &
       (theta-1)+G_ss*(1-sigma_m)+PSIP_ss+lambda_ss-lambda_ss)
  Bmat(36, 157)=& 
       -exp(qd_ss+pq_pc_ss)
  Bmat(36, 169)=& 
       exp(PSIP_ss)
  Bmat(36, 191)=& 
       -exp(qd_ss+pq_pc_ss)
  Bmat(37, 12)=& 
       -epsq*exp(lag1_inf_q_ss*(1-theta))*(epsq*exp(lag1_inf_q_ss* &
       (1-theta))+(1-epsq)*exp(popt_pq_ss*(1-theta)+inf_q_ss* &
       (1-theta)))**(1/(1-theta)-1)
  Bmat(37, 113)=& 
       exp(inf_q_ss)-(1-epsq)*exp(popt_pq_ss*(1-theta)+inf_q_ss*(1-theta))* &
       (epsq*exp(lag1_inf_q_ss*(1-theta))+(1-epsq)*exp(popt_pq_ss* &
       (1-theta)+inf_q_ss*(1-theta)))**(1/(1-theta)-1)
  Bmat(37, 150)=& 
       -(1-epsq)*exp(popt_pq_ss*(1-theta)+inf_q_ss*(1-theta))*(epsq* &
       exp(lag1_inf_q_ss*(1-theta))+(1-epsq)*exp(popt_pq_ss* &
       (1-theta)+inf_q_ss*(1-theta)))**(1/(1-theta)-1)
  Bmat(38, 60)=& 
       -(exp(nt_ss*v)*vnt**(v-1)+exp(es_ss*v)*ve**(v-1))**(1/v)*exp(zd_ss)
  Bmat(38, 100)=& 
       -ve**(v-1)*(exp(nt_ss*v)*vnt**(v-1)+exp(es_ss*v)*ve**(v-1))**(1/ &
       v-1)*exp(zd_ss+es_ss*v)
  Bmat(38, 130)=& 
       -vnt**(v-1)*(exp(nt_ss*v)*vnt**(v-1)+exp(es_ss*v)*ve**(v-1))**(1/ &
       v-1)*exp(zd_ss+nt_ss*v)
  Bmat(38, 191)=& 
       exp(qd_ss)
  Bmat(39, 66)=& 
       -(vx**(vn-1)*exp(vn*xds_ss)+exp(ts_ss*vn)*vr**(vn-1)+vc**(vn-1)* &
       exp(cds_ss*vn))**(1/vn)*exp(znt_ss)
  Bmat(39, 76)=& 
       -vc**(vn-1)*(vx**(vn-1)*exp(vn*xds_ss)+exp(ts_ss*vn)*vr**(vn-1)+vc** &
       (vn-1)*exp(cds_ss*vn))**(1/vn-1)*exp(znt_ss+cds_ss*vn)
  Bmat(39, 130)=& 
       exp(nt_ss)
  Bmat(39, 210)=& 
       -vr**(vn-1)*(vx**(vn-1)*exp(vn*xds_ss)+exp(ts_ss*vn)*vr**(vn-1)+vc** &
       (vn-1)*exp(cds_ss*vn))**(1/vn-1)*exp(znt_ss+ts_ss*vn)
  Bmat(39, 218)=& 
       -vx**(vn-1)*(vx**(vn-1)*exp(vn*xds_ss)+exp(ts_ss*vn)*vr**(vn-1)+vc** &
       (vn-1)*exp(cds_ss*vn))**(1/vn-1)*exp(znt_ss+vn*xds_ss)
  Bmat(40, 60)=& 
       -v*vc**(vn-1)*vnt**(v-1)*exp((vn-1)* &
       (znt_ss-nt_ss+cds_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(40, 66)=& 
       -vc**(vn-1)*vn*vnt**(v-1)*exp((vn-1)* &
       (znt_ss-nt_ss+cds_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(40, 76)=& 
       -vc**(vn-1)*(vn-1)*vnt**(v-1)*exp((vn-1)* &
       (znt_ss-nt_ss+cds_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(40, 130)=& 
       -vc**(vn-1)*(v-vn)*vnt**(v-1)*exp((vn-1)* &
       (znt_ss-nt_ss+cds_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(40, 133)=& 
       exp(pcds_pq_ss)
  Bmat(40, 191)=& 
       -(1-v)*vc**(vn-1)*vnt**(v-1)*exp((vn-1)* &
       (znt_ss-nt_ss+cds_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(41, 60)=& 
       -v*vnt**(v-1)*vx**(vn-1)*exp((vn-1)* &
       (znt_ss+xds_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(41, 66)=& 
       -vn*vnt**(v-1)*vx**(vn-1)*exp((vn-1)* &
       (znt_ss+xds_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(41, 130)=& 
       -(v-vn)*vnt**(v-1)*vx**(vn-1)*exp((vn-1)* &
       (znt_ss+xds_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(41, 190)=& 
       exp(pxds_pq_ss)
  Bmat(41, 191)=& 
       -(1-v)*vnt**(v-1)*vx**(vn-1)*exp((vn-1)* &
       (znt_ss+xds_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(41, 218)=& 
       -(vn-1)*vnt**(v-1)*vx**(vn-1)*exp((vn-1)* &
       (znt_ss+xds_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(42, 60)=& 
       -v*vnt**(v-1)*vr**(vn-1)*exp((vn-1)* &
       (znt_ss+ts_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(42, 66)=& 
       -vn*vnt**(v-1)*vr**(vn-1)*exp((vn-1)* &
       (znt_ss+ts_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(42, 130)=& 
       -(v-vn)*vnt**(v-1)*vr**(vn-1)*exp((vn-1)* &
       (znt_ss+ts_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(42, 176)=& 
       exp(pts_pq_ss)
  Bmat(42, 191)=& 
       -(1-v)*vnt**(v-1)*vr**(vn-1)*exp((vn-1)* &
       (znt_ss+ts_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(42, 210)=& 
       -(vn-1)*vnt**(v-1)*vr**(vn-1)*exp((vn-1)* &
       (znt_ss+ts_ss-nt_ss)+znt_ss+(v-1)*(zd_ss-qd_ss+nt_ss)+zd_ss)
  Bmat(43, 60)=& 
       -v*ve**(v-1)*exp((v-1)*(zd_ss-qd_ss+es_ss)+zd_ss)
  Bmat(43, 100)=& 
       -(v-1)*ve**(v-1)*exp((v-1)*(zd_ss-qd_ss+es_ss)+zd_ss)
  Bmat(43, 136)=& 
       exp(pes_pq_ss)
  Bmat(43, 191)=& 
       -(1-v)*ve**(v-1)*exp((v-1)*(zd_ss-qd_ss+es_ss)+zd_ss)
  Bmat(44, 59)=& 
       -((1-gama_cd)**(1/omega_cd)*exp((omega_cd-1)*tcd_ss/ &
       omega_cd)+gama_cd**(1/omega_cd)*exp(cds_ss*(omega_cd-1)/ &
       omega_cd))**(omega_cd/(omega_cd-1))*exp(zcd_ss)
  Bmat(44, 76)=& 
       -gama_cd**(1/omega_cd)*((1-gama_cd)**(1/ &
       omega_cd)*exp((omega_cd-1)*tcd_ss/omega_cd)+gama_cd**(1/ &
       omega_cd)*exp(cds_ss*(omega_cd-1)/omega_cd))**(omega_cd/ &
       (omega_cd-1)-1)*exp(zcd_ss+cds_ss*(omega_cd-1)/omega_cd)
  Bmat(44, 77)=& 
       exp(cdsd_ss)
  Bmat(44, 200)=& 
       -(1-gama_cd)**(1/omega_cd)*((1-gama_cd)**(1/ &
       omega_cd)*exp((omega_cd-1)*tcd_ss/omega_cd)+gama_cd**(1/ &
       omega_cd)*exp(cds_ss*(omega_cd-1)/omega_cd))**(omega_cd/ &
       (omega_cd-1)-1)*exp(zcd_ss+(omega_cd-1)*tcd_ss/omega_cd)
  Bmat(45, 59)=& 
       -gama_cd**(1/omega_cd)*(1-1/ &
       omega_cd)*exp(zcd_ss+(-zcd_ss-cds_ss+cdsd_ss)/omega_cd+phicd_ss)
  Bmat(45, 76)=& 
       gama_cd**(1/omega_cd)*exp(zcd_ss+(-zcd_ss-cds_ss+cdsd_ss)/ &
       omega_cd+phicd_ss)/omega_cd
  Bmat(45, 77)=& 
       -gama_cd**(1/omega_cd)*exp(zcd_ss+(-zcd_ss-cds_ss+cdsd_ss)/ &
       omega_cd+phicd_ss)/omega_cd
  Bmat(45, 133)=& 
       exp(pq_pc_ss+pcds_pq_ss)
  Bmat(45, 138)=& 
       -gama_cd**(1/omega_cd)*exp(zcd_ss+(-zcd_ss-cds_ss+cdsd_ss)/ &
       omega_cd+phicd_ss)
  Bmat(45, 157)=& 
       exp(pq_pc_ss+pcds_pq_ss)
  Bmat(46, 59)=& 
       -(1-gama_cd)**(1/omega_cd)*(1-1/ &
       omega_cd)*exp(zcd_ss+(-zcd_ss-tcd_ss+cdsd_ss)/omega_cd+phicd_ss)
  Bmat(46, 77)=& 
       -(1-gama_cd)**(1/omega_cd)*exp(zcd_ss+(-zcd_ss-tcd_ss+cdsd_ss)/ &
       omega_cd+phicd_ss)/omega_cd
  Bmat(46, 138)=& 
       -(1-gama_cd)**(1/omega_cd)*exp(zcd_ss+(-zcd_ss-tcd_ss+cdsd_ss)/ &
       omega_cd+phicd_ss)
  Bmat(46, 173)=& 
       exp(pt_pc_ss)
  Bmat(46, 200)=& 
       (1-gama_cd)**(1/omega_cd)*exp(zcd_ss+(-zcd_ss-tcd_ss+cdsd_ss)/ &
       omega_cd+phicd_ss)/omega_cd
  Bmat(47, 151)=& 
       exp(poptcd_pcd_ss)
  Bmat(47, 166)=& 
       exp(THETACD_ss-PSICD_ss)*theta_cd/(theta_cd-1)
  Bmat(47, 202)=& 
       -exp(THETACD_ss-PSICD_ss)*theta_cd/(theta_cd-1)
  Bmat(48, 75)=& 
       -exp(phicd_ss+cd_ss)
  Bmat(48, 109)=& 
       betta*epscd*(nbar+1)*theta_cd*exp(inf_cd_ss*theta_cd-inf_cd_ss* &
       theta_cd+THETACD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(48, 119)=& 
       betta*epscd*(nbar+1)*exp(inf_cd_ss*theta_cd-inf_cd_ss* &
       theta_cd+THETACD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(48, 138)=& 
       -exp(phicd_ss+cd_ss)
  Bmat(48, 202)=& 
       exp(THETACD_ss)
  Bmat(49, 75)=& 
       -exp(pcd_pc_ss+cd_ss)
  Bmat(49, 109)=& 
       -betta*epscd*(nbar+1)*(1-theta_cd)*exp(inf_cd_ss* &
       (theta_cd-1)-inf_cd_ss*(theta_cd-1)+G_ss* &
       (1-sigma_m)+PSICD_ss+lambda_ss-lambda_ss)
  Bmat(49, 119)=& 
       betta*epscd*(nbar+1)*exp(inf_cd_ss*(theta_cd-1)-inf_cd_ss* &
       (theta_cd-1)+G_ss*(1-sigma_m)+PSICD_ss+lambda_ss-lambda_ss)
  Bmat(49, 131)=& 
       -exp(pcd_pc_ss+cd_ss)
  Bmat(49, 166)=& 
       exp(PSICD_ss)
  Bmat(50, 9)=& 
       -epscd*exp(lag1_inf_cd_ss*(1-theta_cd))*(epscd*exp(lag1_inf_cd_ss* &
       (1-theta_cd))+(1-epscd)*exp(poptcd_pcd_ss* &
       (1-theta_cd)+inf_cd_ss*(1-theta_cd)))**(1/(1-theta_cd)-1)
  Bmat(50, 109)=& 
       exp(inf_cd_ss)-(1-epscd)*exp(poptcd_pcd_ss*(1-theta_cd)+inf_cd_ss* &
       (1-theta_cd))*(epscd*exp(lag1_inf_cd_ss* &
       (1-theta_cd))+(1-epscd)*exp(poptcd_pcd_ss* &
       (1-theta_cd)+inf_cd_ss*(1-theta_cd)))**(1/(1-theta_cd)-1)
  Bmat(50, 151)=& 
       -(1-epscd)*exp(poptcd_pcd_ss*(1-theta_cd)+inf_cd_ss*(1-theta_cd))* &
       (epscd*exp(lag1_inf_cd_ss*(1-theta_cd))+(1-epscd)* &
       exp(poptcd_pcd_ss*(1-theta_cd)+inf_cd_ss*(1-theta_cd)))**(1/ &
       (1-theta_cd)-1)
  Bmat(51, 75)=& 
       -exp(pvcd_ss+cd_ss)
  Bmat(51, 77)=& 
       exp(cdsd_ss)
  Bmat(51, 177)=& 
       -exp(pvcd_ss+cd_ss)
  Bmat(52, 9)=& 
       epscd*theta_cd*exp(lag1_pvcd_ss-(lag1_inf_cd_ss-inf_cd_ss)* &
       theta_cd)
  Bmat(52, 27)=& 
       -epscd*exp(lag1_pvcd_ss-(lag1_inf_cd_ss-inf_cd_ss)*theta_cd)
  Bmat(52, 109)=& 
       -epscd*theta_cd*exp(lag1_pvcd_ss-(lag1_inf_cd_ss-inf_cd_ss)* &
       theta_cd)
  Bmat(52, 151)=& 
       (1-epscd)*theta_cd*exp(-poptcd_pcd_ss*theta_cd)
  Bmat(52, 177)=& 
       exp(pvcd_ss)
  Bmat(53, 71)=& 
       -(gama_xd**(1/omega_xd)*exp((omega_xd-1)*xds_ss/ &
       omega_xd)+(1-gama_xd)**(1/omega_xd)*exp((omega_xd-1)*txd_ss/ &
       omega_xd))**(omega_xd/(omega_xd-1))*exp(zxd_ss)
  Bmat(53, 211)=& 
       -(1-gama_xd)**(1/omega_xd)*(gama_xd**(1/ &
       omega_xd)*exp((omega_xd-1)*xds_ss/omega_xd)+(1-gama_xd)**(1/ &
       omega_xd)*exp((omega_xd-1)*txd_ss/omega_xd))**(omega_xd/ &
       (omega_xd-1)-1)*exp(zxd_ss+(omega_xd-1)*txd_ss/omega_xd)
  Bmat(53, 218)=& 
       -gama_xd**(1/omega_xd)*(gama_xd**(1/ &
       omega_xd)*exp((omega_xd-1)*xds_ss/omega_xd)+(1-gama_xd)**(1/ &
       omega_xd)*exp((omega_xd-1)*txd_ss/omega_xd))**(omega_xd/ &
       (omega_xd-1)-1)*exp(zxd_ss+(omega_xd-1)*xds_ss/omega_xd)
  Bmat(53, 219)=& 
       exp(xdsd_ss)
  Bmat(54, 71)=& 
       -gama_xd**(1/omega_xd)*(1-1/ &
       omega_xd)*exp(zxd_ss+(-zxd_ss-xds_ss+xdsd_ss)/omega_xd+phixd_ss)
  Bmat(54, 142)=& 
       -gama_xd**(1/omega_xd)*exp(zxd_ss+(-zxd_ss-xds_ss+xdsd_ss)/ &
       omega_xd+phixd_ss)
  Bmat(54, 157)=& 
       exp(pxds_pq_ss+pq_pc_ss)
  Bmat(54, 190)=& 
       exp(pxds_pq_ss+pq_pc_ss)
  Bmat(54, 218)=& 
       gama_xd**(1/omega_xd)*exp(zxd_ss+(-zxd_ss-xds_ss+xdsd_ss)/ &
       omega_xd+phixd_ss)/omega_xd
  Bmat(54, 219)=& 
       -gama_xd**(1/omega_xd)*exp(zxd_ss+(-zxd_ss-xds_ss+xdsd_ss)/ &
       omega_xd+phixd_ss)/omega_xd
  Bmat(55, 71)=& 
       -(1-gama_xd)**(1/omega_xd)*(1-1/ &
       omega_xd)*exp(zxd_ss+(-zxd_ss+xdsd_ss-txd_ss)/omega_xd+phixd_ss)
  Bmat(55, 142)=& 
       -(1-gama_xd)**(1/omega_xd)*exp(zxd_ss+(-zxd_ss+xdsd_ss-txd_ss)/ &
       omega_xd+phixd_ss)
  Bmat(55, 173)=& 
       exp(pt_pc_ss)
  Bmat(55, 211)=& 
       (1-gama_xd)**(1/omega_xd)*exp(zxd_ss+(-zxd_ss+xdsd_ss-txd_ss)/ &
       omega_xd+phixd_ss)/omega_xd
  Bmat(55, 219)=& 
       -(1-gama_xd)**(1/omega_xd)*exp(zxd_ss+(-zxd_ss+xdsd_ss-txd_ss)/ &
       omega_xd+phixd_ss)/omega_xd
  Bmat(56, 156)=& 
       exp(poptxd_pxd_ss)
  Bmat(56, 172)=& 
       exp(THETAXD_ss-PSIXD_ss)*theta_xd/(theta_xd-1)
  Bmat(56, 208)=& 
       -exp(THETAXD_ss-PSIXD_ss)*theta_xd/(theta_xd-1)
  Bmat(57, 117)=& 
       betta*epsxd*(nbar+1)*theta_xd*exp(inf_xd_ss*theta_xd-inf_xd_ss* &
       theta_xd+THETAXD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(57, 119)=& 
       betta*epsxd*(nbar+1)*exp(inf_xd_ss*theta_xd-inf_xd_ss* &
       theta_xd+THETAXD_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(57, 142)=& 
       -exp(xd_ss+phixd_ss)
  Bmat(57, 208)=& 
       exp(THETAXD_ss)
  Bmat(57, 217)=& 
       -exp(xd_ss+phixd_ss)
  Bmat(58, 117)=& 
       -betta*epsxd*(nbar+1)*(1-theta_xd)*exp(inf_xd_ss* &
       (theta_xd-1)-inf_xd_ss*(theta_xd-1)+G_ss* &
       (1-sigma_m)+PSIXD_ss+lambda_ss-lambda_ss)
  Bmat(58, 119)=& 
       betta*epsxd*(nbar+1)*exp(inf_xd_ss*(theta_xd-1)-inf_xd_ss* &
       (theta_xd-1)+G_ss*(1-sigma_m)+PSIXD_ss+lambda_ss-lambda_ss)
  Bmat(58, 172)=& 
       exp(PSIXD_ss)
  Bmat(58, 185)=& 
       -exp(xd_ss+pxd_pc_ss)
  Bmat(58, 217)=& 
       -exp(xd_ss+pxd_pc_ss)
  Bmat(59, 15)=& 
       -epsxd*exp(lag1_inf_xd_ss*(1-theta_xd))*(epsxd*exp(lag1_inf_xd_ss* &
       (1-theta_xd))+(1-epsxd)*exp(poptxd_pxd_ss* &
       (1-theta_xd)+inf_xd_ss*(1-theta_xd)))**(1/(1-theta_xd)-1)
  Bmat(59, 117)=& 
       exp(inf_xd_ss)-(1-epsxd)*exp(poptxd_pxd_ss*(1-theta_xd)+inf_xd_ss* &
       (1-theta_xd))*(epsxd*exp(lag1_inf_xd_ss* &
       (1-theta_xd))+(1-epsxd)*exp(poptxd_pxd_ss* &
       (1-theta_xd)+inf_xd_ss*(1-theta_xd)))**(1/(1-theta_xd)-1)
  Bmat(59, 156)=& 
       -(1-epsxd)*exp(poptxd_pxd_ss*(1-theta_xd)+inf_xd_ss*(1-theta_xd))* &
       (epsxd*exp(lag1_inf_xd_ss*(1-theta_xd))+(1-epsxd)* &
       exp(poptxd_pxd_ss*(1-theta_xd)+inf_xd_ss*(1-theta_xd)))**(1/ &
       (1-theta_xd)-1)
  Bmat(60, 183)=& 
       -exp(xd_ss+pvxd_ss)
  Bmat(60, 217)=& 
       -exp(xd_ss+pvxd_ss)
  Bmat(60, 219)=& 
       exp(xdsd_ss)
  Bmat(61, 15)=& 
       epsxd*theta_xd*exp(lag1_pvxd_ss-(lag1_inf_xd_ss-inf_xd_ss)* &
       theta_xd)
  Bmat(61, 33)=& 
       -epsxd*exp(lag1_pvxd_ss-(lag1_inf_xd_ss-inf_xd_ss)*theta_xd)
  Bmat(61, 117)=& 
       -epsxd*theta_xd*exp(lag1_pvxd_ss-(lag1_inf_xd_ss-inf_xd_ss)* &
       theta_xd)
  Bmat(61, 156)=& 
       (1-epsxd)*theta_xd*exp(-poptxd_pxd_ss*theta_xd)
  Bmat(61, 183)=& 
       exp(pvxd_ss)
  Bmat(62, 155)=& 
       exp(poptt_pt_ss)
  Bmat(62, 171)=& 
       exp(THETAT_ss-PSIT_ss)*theta_t/(theta_t-1)
  Bmat(62, 207)=& 
       -exp(THETAT_ss-PSIT_ss)*theta_t/(theta_t-1)
  Bmat(63, 115)=& 
       betta*epst*(nbar+1)*theta_t*exp(inf_t_ss*theta_t-inf_t_ss* &
       theta_t+THETAT_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(63, 119)=& 
       betta*epst*(nbar+1)*exp(inf_t_ss*theta_t-inf_t_ss* &
       theta_t+THETAT_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(63, 157)=& 
       -exp(t_ss+pts_pq_ss+pq_pc_ss)
  Bmat(63, 176)=& 
       -exp(t_ss+pts_pq_ss+pq_pc_ss)
  Bmat(63, 199)=& 
       -exp(t_ss+pts_pq_ss+pq_pc_ss)
  Bmat(63, 207)=& 
       exp(THETAT_ss)
  Bmat(64, 115)=& 
       -betta*epst*(nbar+1)*(1-theta_t)*exp(inf_t_ss* &
       (theta_t-1)-inf_t_ss*(theta_t-1)+G_ss* &
       (1-sigma_m)+PSIT_ss+lambda_ss-lambda_ss)
  Bmat(64, 119)=& 
       betta*epst*(nbar+1)*exp(inf_t_ss*(theta_t-1)-inf_t_ss* &
       (theta_t-1)+G_ss*(1-sigma_m)+PSIT_ss+lambda_ss-lambda_ss)
  Bmat(64, 171)=& 
       exp(PSIT_ss)
  Bmat(64, 173)=& 
       -exp(t_ss+pt_pc_ss)
  Bmat(64, 199)=& 
       -exp(t_ss+pt_pc_ss)
  Bmat(65, 14)=& 
       -epst*exp(lag1_inf_t_ss*(1-theta_t))*(epst*exp(lag1_inf_t_ss* &
       (1-theta_t))+(1-epst)*exp(poptt_pt_ss*(1-theta_t)+inf_t_ss* &
       (1-theta_t)))**(1/(1-theta_t)-1)
  Bmat(65, 115)=& 
       exp(inf_t_ss)-(1-epst)*exp(poptt_pt_ss*(1-theta_t)+inf_t_ss* &
       (1-theta_t))*(epst*exp(lag1_inf_t_ss*(1-theta_t))+(1-epst)* &
       exp(poptt_pt_ss*(1-theta_t)+inf_t_ss*(1-theta_t)))**(1/ &
       (1-theta_t)-1)
  Bmat(65, 155)=& 
       -(1-epst)*exp(poptt_pt_ss*(1-theta_t)+inf_t_ss*(1-theta_t))*(epst* &
       exp(lag1_inf_t_ss*(1-theta_t))+(1-epst)*exp(poptt_pt_ss* &
       (1-theta_t)+inf_t_ss*(1-theta_t)))**(1/(1-theta_t)-1)
  Bmat(66, 182)=& 
       -exp(t_ss+pvt_ss)
  Bmat(66, 199)=& 
       -exp(t_ss+pvt_ss)
  Bmat(66, 210)=& 
       exp(ts_ss)
  Bmat(67, 14)=& 
       epst*theta_t*exp(lag1_pvt_ss-(lag1_inf_t_ss-inf_t_ss)*theta_t)
  Bmat(67, 32)=& 
       -epst*exp(lag1_pvt_ss-(lag1_inf_t_ss-inf_t_ss)*theta_t)
  Bmat(67, 115)=& 
       -epst*theta_t*exp(lag1_pvt_ss-(lag1_inf_t_ss-inf_t_ss)*theta_t)
  Bmat(67, 155)=& 
       (1-epst)*theta_t*exp(-poptt_pt_ss*theta_t)
  Bmat(67, 182)=& 
       exp(pvt_ss)
  Bmat(68, 61)=& 
       -((1-gama_e)**(1/omega_e)*exp((omega_e-1)*te_ss/ &
       omega_e)+gama_e**(1/omega_e)*exp(es_ss*(omega_e-1)/ &
       omega_e))**(omega_e/(omega_e-1))*exp(ze_ss)
  Bmat(68, 100)=& 
       -gama_e**(1/omega_e)*((1-gama_e)**(1/ &
       omega_e)*exp((omega_e-1)*te_ss/omega_e)+gama_e**(1/ &
       omega_e)*exp(es_ss*(omega_e-1)/omega_e))**(omega_e/ &
       (omega_e-1)-1)*exp(ze_ss+es_ss*(omega_e-1)/omega_e)
  Bmat(68, 101)=& 
       exp(esd_ss)
  Bmat(68, 201)=& 
       -(1-gama_e)**(1/omega_e)*((1-gama_e)**(1/ &
       omega_e)*exp((omega_e-1)*te_ss/omega_e)+gama_e**(1/ &
       omega_e)*exp(es_ss*(omega_e-1)/omega_e))**(omega_e/ &
       (omega_e-1)-1)*exp(ze_ss+(omega_e-1)*te_ss/omega_e)
  Bmat(69, 61)=& 
       -gama_e**(1/omega_e)*(1-1/omega_e)*exp(ze_ss+(-ze_ss-es_ss+esd_ss)/ &
       omega_e+phie_ss)
  Bmat(69, 100)=& 
       gama_e**(1/omega_e)*exp(ze_ss+(-ze_ss-es_ss+esd_ss)/omega_e+phie_ss)/ &
       omega_e
  Bmat(69, 101)=& 
       -gama_e**(1/omega_e)*exp(ze_ss+(-ze_ss-es_ss+esd_ss)/omega_e+phie_ss)/ &
       omega_e
  Bmat(69, 136)=& 
       exp(pq_pc_ss+pes_pq_ss)
  Bmat(69, 139)=& 
       -gama_e**(1/omega_e)*exp(ze_ss+(-ze_ss-es_ss+esd_ss)/omega_e+phie_ss)
  Bmat(69, 157)=& 
       exp(pq_pc_ss+pes_pq_ss)
  Bmat(70, 61)=& 
       -(1-gama_e)**(1/omega_e)*(1-1/ &
       omega_e)*exp(ze_ss+(-ze_ss-te_ss+esd_ss)/omega_e+phie_ss)
  Bmat(70, 101)=& 
       -(1-gama_e)**(1/omega_e)*exp(ze_ss+(-ze_ss-te_ss+esd_ss)/ &
       omega_e+phie_ss)/omega_e
  Bmat(70, 139)=& 
       -(1-gama_e)**(1/omega_e)*exp(ze_ss+(-ze_ss-te_ss+esd_ss)/ &
       omega_e+phie_ss)
  Bmat(70, 173)=& 
       exp(pt_pc_ss)
  Bmat(70, 201)=& 
       (1-gama_e)**(1/omega_e)*exp(ze_ss+(-ze_ss-te_ss+esd_ss)/ &
       omega_e+phie_ss)/omega_e
  Bmat(71, 152)=& 
       exp(popte_pe_ss)
  Bmat(71, 167)=& 
       exp(THETAE_ss-PSIE_ss)*theta_e/(theta_e-1)
  Bmat(71, 203)=& 
       -exp(THETAE_ss-PSIE_ss)*theta_e/(theta_e-1)
  Bmat(72, 99)=& 
       -exp(phie_ss+e_ss)
  Bmat(72, 110)=& 
       betta*epse*(nbar+1)*theta_e*exp(inf_e_ss*theta_e-inf_e_ss* &
       theta_e+THETAE_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(72, 119)=& 
       betta*epse*(nbar+1)*exp(inf_e_ss*theta_e-inf_e_ss* &
       theta_e+THETAE_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(72, 139)=& 
       -exp(phie_ss+e_ss)
  Bmat(72, 203)=& 
       exp(THETAE_ss)
  Bmat(73, 99)=& 
       -exp(pe_pc_ss+e_ss)
  Bmat(73, 110)=& 
       -betta*epse*(nbar+1)*(1-theta_e)*exp(inf_e_ss* &
       (theta_e-1)-inf_e_ss*(theta_e-1)+G_ss* &
       (1-sigma_m)+PSIE_ss+lambda_ss-lambda_ss)
  Bmat(73, 119)=& 
       betta*epse*(nbar+1)*exp(inf_e_ss*(theta_e-1)-inf_e_ss* &
       (theta_e-1)+G_ss*(1-sigma_m)+PSIE_ss+lambda_ss-lambda_ss)
  Bmat(73, 134)=& 
       -exp(pe_pc_ss+e_ss)
  Bmat(73, 167)=& 
       exp(PSIE_ss)
  Bmat(74, 10)=& 
       -epse*exp(lag1_inf_e_ss*(1-theta_e))*(epse*exp(lag1_inf_e_ss* &
       (1-theta_e))+(1-epse)*exp(popte_pe_ss*(1-theta_e)+inf_e_ss* &
       (1-theta_e)))**(1/(1-theta_e)-1)
  Bmat(74, 110)=& 
       exp(inf_e_ss)-(1-epse)*exp(popte_pe_ss*(1-theta_e)+inf_e_ss* &
       (1-theta_e))*(epse*exp(lag1_inf_e_ss*(1-theta_e))+(1-epse)* &
       exp(popte_pe_ss*(1-theta_e)+inf_e_ss*(1-theta_e)))**(1/ &
       (1-theta_e)-1)
  Bmat(74, 152)=& 
       -(1-epse)*exp(popte_pe_ss*(1-theta_e)+inf_e_ss*(1-theta_e))*(epse* &
       exp(lag1_inf_e_ss*(1-theta_e))+(1-epse)*exp(popte_pe_ss* &
       (1-theta_e)+inf_e_ss*(1-theta_e)))**(1/(1-theta_e)-1)
  Bmat(75, 99)=& 
       -exp(pve_ss+e_ss)
  Bmat(75, 101)=& 
       exp(esd_ss)
  Bmat(75, 178)=& 
       -exp(pve_ss+e_ss)
  Bmat(76, 10)=& 
       epse*theta_e*exp(lag1_pve_ss-(lag1_inf_e_ss-inf_e_ss)*theta_e)
  Bmat(76, 28)=& 
       -epse*exp(lag1_pve_ss-(lag1_inf_e_ss-inf_e_ss)*theta_e)
  Bmat(76, 110)=& 
       -epse*theta_e*exp(lag1_pve_ss-(lag1_inf_e_ss-inf_e_ss)*theta_e)
  Bmat(76, 152)=& 
       (1-epse)*theta_e*exp(-popte_pe_ss*theta_e)
  Bmat(76, 178)=& 
       exp(pve_ss)
  Bmat(77, 199)=& 
       exp(t_ss)
  Bmat(77, 200)=& 
       -exp(tcd_ss)
  Bmat(77, 201)=& 
       -exp(te_ss)
  Bmat(77, 209)=& 
       -exp(tm_ss)
  Bmat(77, 211)=& 
       -exp(txd_ss)
  Bmat(78, 23)=& 
       exp(pe_pq_ss-lag1_pe_pq_ss)
  Bmat(78, 110)=& 
       exp(inf_e_ss-inf_q_ss)
  Bmat(78, 113)=& 
       -exp(inf_e_ss-inf_q_ss)
  Bmat(78, 135)=& 
       -exp(pe_pq_ss-lag1_pe_pq_ss)
  Bmat(79, 52)=& 
       -exp(cstar_ss-mu*qe_q_ss)
  Bmat(79, 99)=& 
       exp(e_ss)
  Bmat(79, 192)=& 
       mu*exp(cstar_ss-mu*qe_q_ss)
  Bmat(80, 131)=& 
       -exp(pcd_pc_ss-pq_pc_ss)
  Bmat(80, 132)=& 
       exp(pcd_pq_ss)
  Bmat(80, 157)=& 
       exp(pcd_pc_ss-pq_pc_ss)
  Bmat(81, 135)=& 
       exp(pe_pq_ss)
  Bmat(81, 157)=& 
       exp(s_pc_ss+qe_q_ss-pq_pc_ss)
  Bmat(81, 192)=& 
       -exp(s_pc_ss+qe_q_ss-pq_pc_ss)
  Bmat(81, 198)=& 
       -exp(s_pc_ss+qe_q_ss-pq_pc_ss)
  Bmat(82, 78)=& 
       -exp(cm_ss)
  Bmat(82, 126)=& 
       exp(md_ss)
  Bmat(82, 220)=& 
       -exp(xm_ss)
  Bmat(83, 65)=& 
       -((1-gama_m)**(1/omega_m)*exp((omega_m-1)*tm_ss/ &
       omega_m)+gama_m**(1/omega_m)*exp(mf_ss*(omega_m-1)/ &
       omega_m))**(omega_m/(omega_m-1))*exp(zm_ss)
  Bmat(83, 127)=& 
       -gama_m**(1/omega_m)*((1-gama_m)**(1/ &
       omega_m)*exp((omega_m-1)*tm_ss/omega_m)+gama_m**(1/ &
       omega_m)*exp(mf_ss*(omega_m-1)/omega_m))**(omega_m/ &
       (omega_m-1)-1)*exp(zm_ss+mf_ss*(omega_m-1)/omega_m)
  Bmat(83, 128)=& 
       exp(ms_ss)
  Bmat(83, 209)=& 
       -(1-gama_m)**(1/omega_m)*((1-gama_m)**(1/ &
       omega_m)*exp((omega_m-1)*tm_ss/omega_m)+gama_m**(1/ &
       omega_m)*exp(mf_ss*(omega_m-1)/omega_m))**(omega_m/ &
       (omega_m-1)-1)*exp(zm_ss+(omega_m-1)*tm_ss/omega_m)
  Bmat(84, 65)=& 
       -gama_m**(1/omega_m)*(1-1/omega_m)*exp(zm_ss+(-zm_ss+ms_ss-mf_ss)/ &
       omega_m+phimd_ss)
  Bmat(84, 127)=& 
       gama_m**(1/omega_m)*exp(zm_ss+(-zm_ss+ms_ss-mf_ss)/omega_m+phimd_ss)/ &
       omega_m
  Bmat(84, 128)=& 
       -gama_m**(1/omega_m)*exp(zm_ss+(-zm_ss+ms_ss-mf_ss)/omega_m+phimd_ss)/ &
       omega_m
  Bmat(84, 140)=& 
       -gama_m**(1/omega_m)*exp(zm_ss+(-zm_ss+ms_ss-mf_ss)/omega_m+phimd_ss)
  Bmat(84, 146)=& 
       exp(pmf_pc_ss)
  Bmat(85, 65)=& 
       -(1-gama_m)**(1/omega_m)*(1-1/ &
       omega_m)*exp(zm_ss+(-zm_ss-tm_ss+ms_ss)/omega_m+phimd_ss)
  Bmat(85, 128)=& 
       -(1-gama_m)**(1/omega_m)*exp(zm_ss+(-zm_ss-tm_ss+ms_ss)/ &
       omega_m+phimd_ss)/omega_m
  Bmat(85, 140)=& 
       -(1-gama_m)**(1/omega_m)*exp(zm_ss+(-zm_ss-tm_ss+ms_ss)/ &
       omega_m+phimd_ss)
  Bmat(85, 173)=& 
       exp(pt_pc_ss)
  Bmat(85, 209)=& 
       (1-gama_m)**(1/omega_m)*exp(zm_ss+(-zm_ss-tm_ss+ms_ss)/ &
       omega_m+phimd_ss)/omega_m
  Bmat(86, 153)=& 
       exp(poptmd_pmd_ss)
  Bmat(86, 168)=& 
       exp(THETAM_ss-PSIM_ss)*theta_m/(theta_m-1)
  Bmat(86, 204)=& 
       -exp(THETAM_ss-PSIM_ss)*theta_m/(theta_m-1)
  Bmat(87, 111)=& 
       betta*epsm*(nbar+1)*theta_m*exp(inf_md_ss*theta_m-inf_md_ss* &
       theta_m+THETAM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(87, 119)=& 
       betta*epsm*(nbar+1)*exp(inf_md_ss*theta_m-inf_md_ss* &
       theta_m+THETAM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(87, 126)=& 
       -exp(phimd_ss+md_ss)
  Bmat(87, 140)=& 
       -exp(phimd_ss+md_ss)
  Bmat(87, 204)=& 
       exp(THETAM_ss)
  Bmat(88, 111)=& 
       -betta*epsm*(nbar+1)*(1-theta_m)*exp(inf_md_ss* &
       (theta_m-1)-inf_md_ss*(theta_m-1)+G_ss* &
       (1-sigma_m)+PSIM_ss+lambda_ss-lambda_ss)
  Bmat(88, 119)=& 
       betta*epsm*(nbar+1)*exp(inf_md_ss*(theta_m-1)-inf_md_ss* &
       (theta_m-1)+G_ss*(1-sigma_m)+PSIM_ss+lambda_ss-lambda_ss)
  Bmat(88, 126)=& 
       -exp(pmd_pc_ss+md_ss)
  Bmat(88, 144)=& 
       -exp(pmd_pc_ss+md_ss)
  Bmat(88, 168)=& 
       exp(PSIM_ss)
  Bmat(89, 11)=& 
       -epsm*exp(lag1_inf_md_ss*(1-theta_m))*(epsm*exp(lag1_inf_md_ss* &
       (1-theta_m))+(1-epsm)*exp(poptmd_pmd_ss*(1-theta_m)+inf_md_ss* &
       (1-theta_m)))**(1/(1-theta_m)-1)
  Bmat(89, 111)=& 
       exp(inf_md_ss)-(1-epsm)*exp(poptmd_pmd_ss*(1-theta_m)+inf_md_ss* &
       (1-theta_m))*(epsm*exp(lag1_inf_md_ss*(1-theta_m))+(1-epsm)* &
       exp(poptmd_pmd_ss*(1-theta_m)+inf_md_ss*(1-theta_m)))**(1/ &
       (1-theta_m)-1)
  Bmat(89, 153)=& 
       -(1-epsm)*exp(poptmd_pmd_ss*(1-theta_m)+inf_md_ss*(1-theta_m))* &
       (epsm*exp(lag1_inf_md_ss*(1-theta_m))+(1-epsm)* &
       exp(poptmd_pmd_ss*(1-theta_m)+inf_md_ss*(1-theta_m)))**(1/ &
       (1-theta_m)-1)
  Bmat(90, 126)=& 
       -exp(pvm_ss+md_ss)
  Bmat(90, 128)=& 
       exp(ms_ss)
  Bmat(90, 179)=& 
       -exp(pvm_ss+md_ss)
  Bmat(91, 11)=& 
       epsm*theta_m*exp(lag1_pvm_ss-(lag1_inf_md_ss-inf_md_ss)*theta_m)
  Bmat(91, 29)=& 
       -epsm*exp(lag1_pvm_ss-(lag1_inf_md_ss-inf_md_ss)*theta_m)
  Bmat(91, 111)=& 
       -epsm*theta_m*exp(lag1_pvm_ss-(lag1_inf_md_ss-inf_md_ss)*theta_m)
  Bmat(91, 153)=& 
       (1-epsm)*theta_m*exp(-poptmd_pmd_ss*theta_m)
  Bmat(91, 179)=& 
       exp(pvm_ss)
  Bmat(92, 144)=& 
       exp(pt_pq_ss+pq_pc_ss-pmd_pc_ss)
  Bmat(92, 157)=& 
       -exp(pt_pq_ss+pq_pc_ss-pmd_pc_ss)
  Bmat(92, 174)=& 
       exp(pt_pmd_ss)
  Bmat(92, 175)=& 
       -exp(pt_pq_ss+pq_pc_ss-pmd_pc_ss)
  Bmat(93, 55)=& 
       -exp(s_pc_ss+qm_ss-pmd_pc_ss)
  Bmat(93, 144)=& 
       exp(s_pc_ss+qm_ss-pmd_pc_ss)
  Bmat(93, 147)=& 
       exp(pmf_pmd_ss)
  Bmat(93, 198)=& 
       -exp(s_pc_ss+qm_ss-pmd_pc_ss)
  Bmat(94, 91)=& 
       -exp(pi_star_m_ss+d_ss)
  Bmat(94, 112)=& 
       exp(inf_mf_ss)
  Bmat(94, 143)=& 
       -exp(pi_star_m_ss+d_ss)
  Bmat(95, 46)=& 
       exp(qm_ss-lag1_qm_ss)
  Bmat(95, 54)=& 
       -exp(pi_star_m_ss-pi_star_ss)
  Bmat(95, 55)=& 
       -exp(qm_ss-lag1_qm_ss)
  Bmat(95, 143)=& 
       exp(pi_star_m_ss-pi_star_ss)
  Bmat(96, 24)=& 
       exp(pmd_pc_ss-lag1_pmd_pc_ss)
  Bmat(96, 108)=& 
       -exp(inf_md_ss-inf_c_ss)
  Bmat(96, 111)=& 
       exp(inf_md_ss-inf_c_ss)
  Bmat(96, 144)=& 
       -exp(pmd_pc_ss-lag1_pmd_pc_ss)
  Bmat(97, 196)=& 
       -exp(rmf_ss)
  Bmat(97, 197)=& 
       exp(rms_ss)
  Bmat(98, 141)=& 
       -exp(phirm_ss)
  Bmat(98, 148)=& 
       exp(pmr_pc_ss)
  Bmat(99, 154)=& 
       exp(poptrm_prm_ss)
  Bmat(99, 170)=& 
       exp(THETARM_ss-PSIRM_ss)*theta_rm/(theta_rm-1)
  Bmat(99, 206)=& 
       -exp(THETARM_ss-PSIRM_ss)*theta_rm/(theta_rm-1)
  Bmat(100, 114)=& 
       betta*epsrm*(nbar+1)*theta_rm*exp(inf_rm_ss*theta_rm-inf_rm_ss* &
       theta_rm+THETARM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(100, 119)=& 
       betta*epsrm*(nbar+1)*exp(inf_rm_ss*theta_rm-inf_rm_ss* &
       theta_rm+THETARM_ss+G_ss*(1-sigma_m)+lambda_ss-lambda_ss)
  Bmat(100, 141)=& 
       -exp(rm_ss+phirm_ss)
  Bmat(100, 195)=& 
       -exp(rm_ss+phirm_ss)
  Bmat(100, 206)=& 
       exp(THETARM_ss)
  Bmat(101, 114)=& 
       -betta*epsrm*(nbar+1)*(1-theta_rm)*exp(inf_rm_ss* &
       (theta_rm-1)-inf_rm_ss*(theta_rm-1)+G_ss* &
       (1-sigma_m)+PSIRM_ss+lambda_ss-lambda_ss)
  Bmat(101, 119)=& 
       betta*epsrm*(nbar+1)*exp(inf_rm_ss*(theta_rm-1)-inf_rm_ss* &
       (theta_rm-1)+G_ss*(1-sigma_m)+PSIRM_ss+lambda_ss-lambda_ss)
  Bmat(101, 158)=& 
       -exp(rm_ss+prm_pc_ss)
  Bmat(101, 170)=& 
       exp(PSIRM_ss)
  Bmat(101, 195)=& 
       -exp(rm_ss+prm_pc_ss)
  Bmat(102, 13)=& 
       -epsrm*exp(lag1_inf_rm_ss*(1-theta_rm))*(epsrm*exp(lag1_inf_rm_ss* &
       (1-theta_rm))+(1-epsrm)*exp(poptrm_prm_ss* &
       (1-theta_rm)+inf_rm_ss*(1-theta_rm)))**(1/(1-theta_rm)-1)
  Bmat(102, 114)=& 
       exp(inf_rm_ss)-(1-epsrm)*exp(poptrm_prm_ss*(1-theta_rm)+inf_rm_ss* &
       (1-theta_rm))*(epsrm*exp(lag1_inf_rm_ss* &
       (1-theta_rm))+(1-epsrm)*exp(poptrm_prm_ss* &
       (1-theta_rm)+inf_rm_ss*(1-theta_rm)))**(1/(1-theta_rm)-1)
  Bmat(102, 154)=& 
       -(1-epsrm)*exp(poptrm_prm_ss*(1-theta_rm)+inf_rm_ss*(1-theta_rm))* &
       (epsrm*exp(lag1_inf_rm_ss*(1-theta_rm))+(1-epsrm)* &
       exp(poptrm_prm_ss*(1-theta_rm)+inf_rm_ss*(1-theta_rm)))**(1/ &
       (1-theta_rm)-1)
  Bmat(103, 181)=& 
       -exp(rm_ss+pvrm_ss)
  Bmat(103, 195)=& 
       -exp(rm_ss+pvrm_ss)
  Bmat(103, 197)=& 
       exp(rms_ss)
  Bmat(104, 13)=& 
       epsrm*theta_rm*exp(lag1_pvrm_ss-(lag1_inf_rm_ss-inf_rm_ss)* &
       theta_rm)
  Bmat(104, 31)=& 
       -epsrm*exp(lag1_pvrm_ss-(lag1_inf_rm_ss-inf_rm_ss)*theta_rm)
  Bmat(104, 114)=& 
       -epsrm*theta_rm*exp(lag1_pvrm_ss-(lag1_inf_rm_ss-inf_rm_ss)* &
       theta_rm)
  Bmat(104, 154)=& 
       (1-epsrm)*theta_rm*exp(-poptrm_prm_ss*theta_rm)
  Bmat(104, 181)=& 
       exp(pvrm_ss)
  Bmat(105, 25)=& 
       -exp(prm_pc_ss-lag1_prm_pc_ss)
  Bmat(105, 108)=& 
       exp(inf_rm_ss-inf_c_ss)
  Bmat(105, 114)=& 
       -exp(inf_rm_ss-inf_c_ss)
  Bmat(105, 158)=& 
       exp(prm_pc_ss-lag1_prm_pc_ss)
  Bmat(106, 148)=& 
       -exp(pmr_pc_ss-prm_pc_ss)
  Bmat(106, 149)=& 
       exp(pmr_prm_ss)
  Bmat(106, 158)=& 
       exp(pmr_pc_ss-prm_pc_ss)
  Bmat(107, 56)=& 
       -exp(s_pc_ss+qmr_ss)
  Bmat(107, 148)=& 
       exp(pmr_pc_ss)
  Bmat(107, 198)=& 
       -exp(s_pc_ss+qmr_ss)
  Bmat(108, 44)=& 
       -abs(istar)**(4*(1-rho_i))*abs(pi_bar)**(4*(1-rho_i)-4*(1-rho_i)* &
       rho_pi)*(pib_bar**rho_y)**(rho_i-1)*rho_i* &
       exp(zi_ss+(1-rho_i)*(rho_y*yngdp_pc_ss+rho_pi* &
       yinf_c_ss)+lag1_yi_ss*rho_i)/abs(pi_star_bar)**(4*(1-rho_i))
  Bmat(108, 63)=& 
       -abs(istar)**(4*(1-rho_i))*abs(pi_bar)**(4*(1-rho_i)-4*(1-rho_i)* &
       rho_pi)*(pib_bar**rho_y)**(rho_i-1)*exp(zi_ss+(1-rho_i)* &
       (rho_y*yngdp_pc_ss+rho_pi*yinf_c_ss)+lag1_yi_ss*rho_i)/ &
       abs(pi_star_bar)**(4*(1-rho_i))
  Bmat(108, 221)=& 
       exp(yi_ss)
  Bmat(108, 222)=& 
       -abs(istar)**(4*(1-rho_i))*abs(pi_bar)**(4*(1-rho_i)-4*(1-rho_i)* &
       rho_pi)*(pib_bar**rho_y)**(rho_i-1)*(1-rho_i)*rho_pi* &
       exp(zi_ss+(1-rho_i)*(rho_y*yngdp_pc_ss+rho_pi* &
       yinf_c_ss)+lag1_yi_ss*rho_i)/abs(pi_star_bar)**(4*(1-rho_i))
  Bmat(109, 64)=& 
       -istar*exp((exp(s_pc_ss-ngdp_pc_ss+f_ss)-fbar)* &
       OMEG_U-(exp(d_ss+d_ss)*pi_star_bar**2/ &
       pi_bar**2-1)*OMEG_S+zie_ss)
  Bmat(109, 91)=& 
       istar*pi_star_bar**2*OMEG_S* &
       exp((exp(s_pc_ss-ngdp_pc_ss+f_ss)-fbar)*OMEG_U-(exp(d_ss+d_ss)* &
       pi_star_bar**2/pi_bar**2-1)*OMEG_S+zie_ss+d_ss+d_ss)/ &
       pi_bar**2
  Bmat(109, 107)=& 
       exp(ie_ss)
  Bmat(109, 129)=& 
       istar*OMEG_U*exp((exp(s_pc_ss-ngdp_pc_ss+f_ss)-fbar)* &
       OMEG_U-(exp(d_ss+d_ss)*pi_star_bar**2/ &
       pi_bar**2-1)*OMEG_S+zie_ss+s_pc_ss-ngdp_pc_ss+f_ss)
  Bmat(109, 198)=& 
       -istar*OMEG_U*exp((exp(s_pc_ss-ngdp_pc_ss+f_ss)-fbar)* &
       OMEG_U-(exp(d_ss+d_ss)*pi_star_bar**2/ &
       pi_bar**2-1)*OMEG_S+zie_ss+s_pc_ss-ngdp_pc_ss+f_ss)
  Bmat(110, 38)=& 
       exp(s_pc_ss-lag1_s_pc_ss)
  Bmat(110, 54)=& 
       exp(pi_star_ss-inf_c_ss+d_ss)
  Bmat(110, 91)=& 
       exp(pi_star_ss-inf_c_ss+d_ss)
  Bmat(110, 108)=& 
       -exp(pi_star_ss-inf_c_ss+d_ss)
  Bmat(110, 198)=& 
       -exp(s_pc_ss-lag1_s_pc_ss)
  Bmat(111, 35)=& 
       -exp(pxd_pcd_ss-lag1_pxd_pcd_ss)
  Bmat(111, 109)=& 
       exp(inf_xd_ss-inf_cd_ss)
  Bmat(111, 117)=& 
       -exp(inf_xd_ss-inf_cd_ss)
  Bmat(111, 186)=& 
       exp(pxd_pcd_ss-lag1_pxd_pcd_ss)
  Bmat(112, 22)=& 
       -exp(pcd_pq_ss-lag1_pcd_pq_ss)
  Bmat(112, 109)=& 
       -exp(inf_cd_ss-inf_q_ss)
  Bmat(112, 113)=& 
       exp(inf_cd_ss-inf_q_ss)
  Bmat(112, 132)=& 
       exp(pcd_pq_ss-lag1_pcd_pq_ss)
  Bmat(113, 26)=& 
       -exp(pt_pq_ss-lag1_pt_pq_ss)
  Bmat(113, 113)=& 
       exp(inf_t_ss-inf_q_ss)
  Bmat(113, 115)=& 
       -exp(inf_t_ss-inf_q_ss)
  Bmat(113, 175)=& 
       exp(pt_pq_ss-lag1_pt_pq_ss)
  Bmat(114, 145)=& 
       exp(pxd_px_ss-pmd_px_ss)
  Bmat(114, 187)=& 
       exp(pxd_pmd_ss)
  Bmat(114, 189)=& 
       -exp(pxd_px_ss-pmd_px_ss)
  Bmat(115, 132)=& 
       exp(pxd_pq_ss-pcd_pq_ss)
  Bmat(115, 186)=& 
       exp(pxd_pcd_ss)
  Bmat(115, 188)=& 
       -exp(pxd_pq_ss-pcd_pq_ss)
  Bmat(116, 184)=& 
       -exp(px_pc_ss+pxd_px_ss)
  Bmat(116, 185)=& 
       exp(pxd_pc_ss)
  Bmat(116, 189)=& 
       -exp(px_pc_ss+pxd_px_ss)
  Bmat(117, 157)=& 
       -exp(pt_pq_ss+pq_pc_ss)
  Bmat(117, 173)=& 
       exp(pt_pc_ss)
  Bmat(117, 175)=& 
       -exp(pt_pq_ss+pq_pc_ss)
  Bmat(118, 144)=& 
       -exp(pmf_pmd_ss+pmd_pc_ss)
  Bmat(118, 146)=& 
       exp(pmf_pc_ss)
  Bmat(118, 147)=& 
       -exp(pmf_pmd_ss+pmd_pc_ss)
  Bmat(119, 134)=& 
       exp(pe_pc_ss)
  Bmat(119, 135)=& 
       -exp(pq_pc_ss+pe_pq_ss)
  Bmat(119, 157)=& 
       -exp(pq_pc_ss+pe_pq_ss)
  Bmat(120, 137)=& 
       exp(qd_ss+pvq_ss+phi_ss)
  Bmat(120, 157)=& 
       -exp(qd_ss+pq_pc_ss)
  Bmat(120, 162)=& 
       exp(prof_q_ss)
  Bmat(120, 180)=& 
       exp(qd_ss+pvq_ss+phi_ss)
  Bmat(120, 191)=& 
       exp(qd_ss+pvq_ss+phi_ss)-exp(qd_ss+pq_pc_ss)
  Bmat(121, 126)=& 
       -exp(pmd_pc_ss+md_ss)
  Bmat(121, 128)=& 
       exp(phimd_ss+ms_ss)
  Bmat(121, 140)=& 
       exp(phimd_ss+ms_ss)
  Bmat(121, 144)=& 
       -exp(pmd_pc_ss+md_ss)
  Bmat(121, 161)=& 
       exp(prof_m_ss)
  Bmat(122, 141)=& 
       exp(rms_ss+phirm_ss)
  Bmat(122, 158)=& 
       -exp(rm_ss+prm_pc_ss)
  Bmat(122, 163)=& 
       exp(prof_rm_ss)
  Bmat(122, 195)=& 
       -exp(rm_ss+prm_pc_ss)
  Bmat(122, 197)=& 
       exp(rms_ss+phirm_ss)
  Bmat(123, 75)=& 
       -exp(pcd_pc_ss+cd_ss)
  Bmat(123, 77)=& 
       exp(phicd_ss+cdsd_ss)
  Bmat(123, 131)=& 
       -exp(pcd_pc_ss+cd_ss)
  Bmat(123, 138)=& 
       exp(phicd_ss+cdsd_ss)
  Bmat(123, 159)=& 
       exp(prof_cd_ss)
  Bmat(124, 142)=& 
       exp(xdsd_ss+phixd_ss)
  Bmat(124, 165)=& 
       exp(prof_xd_ss)
  Bmat(124, 185)=& 
       -exp(xd_ss+pxd_pc_ss)
  Bmat(124, 217)=& 
       -exp(xd_ss+pxd_pc_ss)
  Bmat(124, 219)=& 
       exp(xdsd_ss+phixd_ss)
  Bmat(125, 157)=& 
       exp(ts_ss+pts_pq_ss+pq_pc_ss)
  Bmat(125, 164)=& 
       exp(prof_t_ss)
  Bmat(125, 173)=& 
       -exp(t_ss+pt_pc_ss)
  Bmat(125, 176)=& 
       exp(ts_ss+pts_pq_ss+pq_pc_ss)
  Bmat(125, 199)=& 
       -exp(t_ss+pt_pc_ss)
  Bmat(125, 210)=& 
       exp(ts_ss+pts_pq_ss+pq_pc_ss)
  Bmat(126, 99)=& 
       -exp(pe_pc_ss+e_ss)
  Bmat(126, 101)=& 
       exp(phie_ss+esd_ss)
  Bmat(126, 134)=& 
       -exp(pe_pc_ss+e_ss)
  Bmat(126, 139)=& 
       exp(phie_ss+esd_ss)
  Bmat(126, 160)=& 
       exp(prof_e_ss)
  Bmat(127, 129)=& 
       exp(ngdp_pc_ss)
  Bmat(127, 157)=& 
       -exp(qd_ss+pq_pc_ss)
  Bmat(127, 158)=& 
       exp(rm_ss+prm_pc_ss)
  Bmat(127, 159)=& 
       -exp(prof_cd_ss)
  Bmat(127, 160)=& 
       -exp(prof_e_ss)
  Bmat(127, 161)=& 
       -exp(prof_m_ss)
  Bmat(127, 163)=& 
       -exp(prof_rm_ss)
  Bmat(127, 164)=& 
       -exp(prof_t_ss)
  Bmat(127, 165)=& 
       -exp(prof_xd_ss)
  Bmat(127, 191)=& 
       -exp(qd_ss+pq_pc_ss)
  Bmat(127, 195)=& 
       exp(rm_ss+prm_pc_ss)
  Bmat(128, 6)=& 
       -exp(lag3_inf_c_ss+lag2_inf_c_ss+lag1_inf_c_ss+inf_c_ss)
  Bmat(128, 7)=& 
       -exp(lag3_inf_c_ss+lag2_inf_c_ss+lag1_inf_c_ss+inf_c_ss)
  Bmat(128, 8)=& 
       -exp(lag3_inf_c_ss+lag2_inf_c_ss+lag1_inf_c_ss+inf_c_ss)
  Bmat(128, 108)=& 
       -exp(lag3_inf_c_ss+lag2_inf_c_ss+lag1_inf_c_ss+inf_c_ss)
  Bmat(128, 222)=& 
       exp(yinf_c_ss)
  Bmat(129, 18)=& 
       -exp(lag1_ngdp_pc_ss)
  Bmat(129, 19)=& 
       -exp(lag2_ngdp_pc_ss)
  Bmat(129, 20)=& 
       -exp(lag3_ngdp_pc_ss)
  Bmat(129, 129)=& 
       -exp(ngdp_pc_ss)
  Bmat(129, 223)=& 
       exp(yngdp_pc_ss)
  Bmat(130, 106)=& 
       -4*exp(4*i_ss)
  Bmat(130, 221)=& 
       exp(yi_ss)
  Bmat(131, 74)=& 
       -exp(c_ss)
  Bmat(132, 75)=& 
       -exp(cd_ss)
  Bmat(133, 78)=& 
       -exp(cm_ss)
  Bmat(134, 99)=& 
       -exp(e_ss)
  Bmat(135, 107)=& 
       -exp(ie_ss)
  Bmat(136, 108)=& 
       -exp(inf_c_ss)
  Bmat(137, 6)=& 
       -exp(lag1_inf_c_ss)
  Bmat(138, 7)=& 
       -exp(lag2_inf_c_ss)
  Bmat(139, 109)=& 
       -exp(inf_cd_ss)
  Bmat(140, 110)=& 
       -exp(inf_e_ss)
  Bmat(141, 111)=& 
       -exp(inf_md_ss)
  Bmat(142, 113)=& 
       -exp(inf_q_ss)
  Bmat(143, 114)=& 
       -exp(inf_rm_ss)
  Bmat(144, 115)=& 
       -exp(inf_t_ss)
  Bmat(145, 117)=& 
       -exp(inf_xd_ss)
  Bmat(146, 126)=& 
       -exp(md_ss)
  Bmat(147, 128)=& 
       -exp(ms_ss)
  Bmat(148, 129)=& 
       -exp(ngdp_pc_ss)
  Bmat(149, 18)=& 
       -exp(lag1_ngdp_pc_ss)
  Bmat(150, 19)=& 
       -exp(lag2_ngdp_pc_ss)
  Bmat(151, 131)=& 
       -exp(pcd_pc_ss)
  Bmat(152, 132)=& 
       -exp(pcd_pq_ss)
  Bmat(153, 135)=& 
       -exp(pe_pq_ss)
  Bmat(154, 144)=& 
       -exp(pmd_pc_ss)
  Bmat(155, 158)=& 
       -exp(prm_pc_ss)
  Bmat(156, 175)=& 
       -exp(pt_pq_ss)
  Bmat(157, 177)=& 
       -exp(pvcd_ss)
  Bmat(158, 178)=& 
       -exp(pve_ss)
  Bmat(159, 179)=& 
       -exp(pvm_ss)
  Bmat(160, 180)=& 
       -exp(pvq_ss)
  Bmat(161, 181)=& 
       -exp(pvrm_ss)
  Bmat(162, 182)=& 
       -exp(pvt_ss)
  Bmat(163, 183)=& 
       -exp(pvxd_ss)
  Bmat(164, 184)=& 
       -exp(px_pc_ss)
  Bmat(165, 186)=& 
       -exp(pxd_pcd_ss)
  Bmat(166, 191)=& 
       -exp(qd_ss)
  Bmat(167, 195)=& 
       -exp(rm_ss)
  Bmat(168, 198)=& 
       -exp(s_pc_ss)
  Bmat(169, 199)=& 
       -exp(t_ss)
  Bmat(170, 214)=& 
       -exp(w_ss)
  Bmat(171, 216)=& 
       -exp(x_ss)
  Bmat(172, 217)=& 
       -exp(xd_ss)
  Bmat(173, 220)=& 
       -exp(xm_ss)
  Bmat(174, 221)=& 
       -exp(yi_ss)
  Bmat(175, 50)=& 
       -exp(f_ss)
  Bmat(176, 51)=& 
       -exp(k_ss)
  Bmat(177, 52)=& 
       -exp(cstar_ss)
  Bmat(178, 55)=& 
       -exp(qm_ss)
  Bmat(179, 57)=& 
       -exp(tr_ss)
  Bmat(180, 120)=& 
       exp(led1_yinf_c_ss)
  Bmat(181, 121)=& 
       exp(led2_yinf_c_ss)
  Bmat(182, 122)=& 
       exp(led3_yinf_c_ss)
  Bmat(183, 123)=& 
       exp(led4_yinf_c_ss)
  Bmat(184, 124)=& 
       exp(led5_yinf_c_ss)
  Bmat(185, 125)=& 
       exp(led6_yinf_c_ss)
  Bmat(186, 1)=& 
       exp(-lag1_c_ss+G_ss+c_ss)*(nbar+1)
  Bmat(186, 53)=& 
       -exp(-lag1_c_ss+G_ss+c_ss)*(nbar+1)
  Bmat(186, 74)=& 
       -exp(-lag1_c_ss+G_ss+c_ss)*(nbar+1)
  Bmat(186, 79)=& 
       exp(D_c_ss)
  Bmat(187, 2)=& 
       exp(-lag1_cd_ss+G_ss+cd_ss)*(nbar+1)
  Bmat(187, 53)=& 
       -exp(-lag1_cd_ss+G_ss+cd_ss)*(nbar+1)
  Bmat(187, 75)=& 
       -exp(-lag1_cd_ss+G_ss+cd_ss)*(nbar+1)
  Bmat(187, 80)=& 
       exp(D_cd_ss)
  Bmat(188, 3)=& 
       exp(-lag1_cm_ss+G_ss+cm_ss)*(nbar+1)
  Bmat(188, 53)=& 
       -exp(-lag1_cm_ss+G_ss+cm_ss)*(nbar+1)
  Bmat(188, 78)=& 
       -exp(-lag1_cm_ss+G_ss+cm_ss)*(nbar+1)
  Bmat(188, 81)=& 
       exp(D_cm_ss)
  Bmat(189, 45)=& 
       exp(-lag1_cstar_ss+G_ss+cstar_ss)*(nbar+1)
  Bmat(189, 52)=& 
       -exp(-lag1_cstar_ss+G_ss+cstar_ss)*(nbar+1)
  Bmat(189, 53)=& 
       -exp(-lag1_cstar_ss+G_ss+cstar_ss)*(nbar+1)
  Bmat(189, 82)=& 
       exp(D_cstar_ss)
  Bmat(190, 4)=& 
       exp(-lag1_e_ss+G_ss+e_ss)*(nbar+1)
  Bmat(190, 53)=& 
       -exp(-lag1_e_ss+G_ss+e_ss)*(nbar+1)
  Bmat(190, 83)=& 
       exp(D_e_ss)
  Bmat(190, 99)=& 
       -exp(-lag1_e_ss+G_ss+e_ss)*(nbar+1)
  Bmat(191, 48)=& 
       exp(-lag1_f_ss+G_ss+f_ss)*(nbar+1)
  Bmat(191, 50)=& 
       -exp(-lag1_f_ss+G_ss+f_ss)*(nbar+1)
  Bmat(191, 53)=& 
       -exp(-lag1_f_ss+G_ss+f_ss)*(nbar+1)
  Bmat(191, 84)=& 
       exp(D_f_ss)
  Bmat(192, 49)=& 
       exp(-lag1_k_ss+k_ss+G_ss)*(nbar+1)
  Bmat(192, 51)=& 
       -exp(-lag1_k_ss+k_ss+G_ss)*(nbar+1)
  Bmat(192, 53)=& 
       -exp(-lag1_k_ss+k_ss+G_ss)*(nbar+1)
  Bmat(192, 85)=& 
       exp(D_k_ss)
  Bmat(193, 16)=& 
       exp(md_ss-lag1_md_ss+G_ss)*(nbar+1)
  Bmat(193, 53)=& 
       -exp(md_ss-lag1_md_ss+G_ss)*(nbar+1)
  Bmat(193, 86)=& 
       exp(D_md_ss)
  Bmat(193, 126)=& 
       -exp(md_ss-lag1_md_ss+G_ss)*(nbar+1)
  Bmat(194, 17)=& 
       exp(ms_ss-lag1_ms_ss+G_ss)*(nbar+1)
  Bmat(194, 53)=& 
       -exp(ms_ss-lag1_ms_ss+G_ss)*(nbar+1)
  Bmat(194, 87)=& 
       exp(D_ms_ss)
  Bmat(194, 128)=& 
       -exp(ms_ss-lag1_ms_ss+G_ss)*(nbar+1)
  Bmat(195, 18)=& 
       (nbar+1)*exp(ngdp_pc_ss-lag1_ngdp_pc_ss+G_ss)
  Bmat(195, 53)=& 
       -(nbar+1)*exp(ngdp_pc_ss-lag1_ngdp_pc_ss+G_ss)
  Bmat(195, 88)=& 
       exp(D_ngdp_pc_ss)
  Bmat(195, 129)=& 
       -(nbar+1)*exp(ngdp_pc_ss-lag1_ngdp_pc_ss+G_ss)
  Bmat(196, 36)=& 
       (nbar+1)*exp(qd_ss-lag1_qd_ss+G_ss)
  Bmat(196, 53)=& 
       -(nbar+1)*exp(qd_ss-lag1_qd_ss+G_ss)
  Bmat(196, 89)=& 
       exp(D_qd_ss)
  Bmat(196, 191)=& 
       -(nbar+1)*exp(qd_ss-lag1_qd_ss+G_ss)
  Bmat(197, 37)=& 
       (nbar+1)*exp(rm_ss-lag1_rm_ss+G_ss)
  Bmat(197, 53)=& 
       -(nbar+1)*exp(rm_ss-lag1_rm_ss+G_ss)
  Bmat(197, 90)=& 
       exp(D_rm_ss)
  Bmat(197, 195)=& 
       -(nbar+1)*exp(rm_ss-lag1_rm_ss+G_ss)
  Bmat(198, 39)=& 
       (nbar+1)*exp(t_ss-lag1_t_ss+G_ss)
  Bmat(198, 53)=& 
       -(nbar+1)*exp(t_ss-lag1_t_ss+G_ss)
  Bmat(198, 92)=& 
       exp(D_t_ss)
  Bmat(198, 199)=& 
       -(nbar+1)*exp(t_ss-lag1_t_ss+G_ss)
  Bmat(199, 47)=& 
       (nbar+1)*exp(tr_ss-lag1_tr_ss+G_ss)
  Bmat(199, 53)=& 
       -(nbar+1)*exp(tr_ss-lag1_tr_ss+G_ss)
  Bmat(199, 57)=& 
       -(nbar+1)*exp(tr_ss-lag1_tr_ss+G_ss)
  Bmat(199, 93)=& 
       exp(D_tr_ss)
  Bmat(200, 40)=& 
       exp(w_ss-lag1_w_ss+G_ss)
  Bmat(200, 53)=& 
       -exp(w_ss-lag1_w_ss+G_ss)
  Bmat(200, 94)=& 
       exp(D_w_ss)
  Bmat(200, 214)=& 
       -exp(w_ss-lag1_w_ss+G_ss)
  Bmat(201, 41)=& 
       (nbar+1)*exp(x_ss-lag1_x_ss+G_ss)
  Bmat(201, 53)=& 
       -(nbar+1)*exp(x_ss-lag1_x_ss+G_ss)
  Bmat(201, 95)=& 
       exp(D_x_ss)
  Bmat(201, 216)=& 
       -(nbar+1)*exp(x_ss-lag1_x_ss+G_ss)
  Bmat(202, 42)=& 
       (nbar+1)*exp(xd_ss-lag1_xd_ss+G_ss)
  Bmat(202, 53)=& 
       -(nbar+1)*exp(xd_ss-lag1_xd_ss+G_ss)
  Bmat(202, 96)=& 
       exp(D_xd_ss)
  Bmat(202, 217)=& 
       -(nbar+1)*exp(xd_ss-lag1_xd_ss+G_ss)
  Bmat(203, 43)=& 
       (nbar+1)*exp(xm_ss-lag1_xm_ss+G_ss)
  Bmat(203, 53)=& 
       -(nbar+1)*exp(xm_ss-lag1_xm_ss+G_ss)
  Bmat(203, 97)=& 
       exp(D_xm_ss)
  Bmat(203, 220)=& 
       -(nbar+1)*exp(xm_ss-lag1_xm_ss+G_ss)
  Bmat(204, 69)=& 
       -rho_zu
  Bmat(205, 62)=& 
       -rho_zh
  Bmat(206, 67)=& 
       -rho_zq
  Bmat(207, 60)=& 
       -rho_zd
  Bmat(208, 66)=& 
       -rho_znt
  Bmat(209, 65)=& 
       -rho_zm
  Bmat(210, 68)=& 
       -rho_zrm
  Bmat(211, 70)=& 
       -rho_zx
  Bmat(212, 64)=& 
       -rho_zie
  Bmat(213, 63)=& 
       -rho_zi
  Bmat(214, 55)=& 
       -rho_qm
  Bmat(215, 56)=& 
       -rho_qmr
  Bmat(216, 57)=& 
       -rho_tr
  Bmat(217, 52)=& 
       -rho_cstar
  Bmat(218, 54)=& 
       -rho_pi_star
  Bmat(219, 59)=& 
       -rho_zcd
  Bmat(220, 71)=& 
       -rho_zxd
  Bmat(221, 61)=& 
       -rho_ze
  Bmat(222, 58)=& 
       -rho_z_xdemand
  Bmat(223, 53)=& 
       -rho_g

  do i=1,nvar
     do j=1,nvar
        if (isnan(amat(i,j))) then
        print *,"anan:",i,j
        endif
        if (isnan(bmat(i,j))) then
        print *,"bnan:",i,j
        endif
      enddo
  enddo


  RETURN

END SUBROUTINE ABMATRIX
