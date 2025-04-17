SUBROUTINE ESTIMATEOBJ(SIZEPAR,PAR,LOG_L)
  USE PARAMETER_VECTOR_MOD 
  USE MODEL_CONST_MOD, ONLY : NPAR
  IMPLICIT NONE
  
  INTEGER , INTENT(IN) :: SIZEPAR
  DOUBLE PRECISION , INTENT(IN) , DIMENSION(SIZEPAR) :: PAR
  DOUBLE PRECISION, INTENT(OUT) :: LOG_L
  DOUBLE PRECISION , DIMENSION(NPAR) :: CPARVECTOR
  INTEGER INFO
  DOUBLE PRECISION MUNKRESFX
  EXTERNAL MUNKRESFX
  
!  print *, "Estoy en ..."
!  print *, par
  
  CPARVECTOR = PARVECTOR    
  cparvector(12)=MUNKRESFX(par(1), DBLE(1.0),DBLE(5.00))  !  12 eta_m=3.19;       
  cparvector(28)=MUNKRESFX(par(2), DBLE(0.1),DBLE(2.00)) !  28 omeg=1.01369726080713;     
  cparvector(69)=MUNKRESFX(par(3), DBLE(0.1),DBLE(2.00)) !  69 sigma_m=1.2;      
  cparvector(33)=MUNKRESFX(par(4), DBLE(0.1),DBLE(2.00)) !  33 omega_x=0.449441398423515;  
  
 EE=345345
  ! cparvector(4) =MUNKRESFX(par(1), DBLE(0.01),DBLE(0.99))
!   cparvector(10)=MUNKRESFX(par(2), DBLE(0.01),DBLE(0.99))
!   cparvector(5) =MUNKRESFX(par(3), DBLE(0.01),DBLE(0.99))
!   cparvector(6) =MUNKRESFX(par(4), DBLE(0.01),DBLE(0.99))
!   cparvector(7) =MUNKRESFX(par(5), DBLE(0.01),DBLE(0.99))
!   cparvector(8) =MUNKRESFX(par(6), DBLE(0.01),DBLE(0.99))
!   cparvector(9) =MUNKRESFX(par(7), DBLE(0.01),DBLE(0.99))
!   cparvector(11)=MUNKRESFX(par(8), DBLE(0.01),DBLE(0.99))
!  !  cparvector(12)=MUNKRESFX(par(9), DBLE(1.0),DBLE(5.00))  !  12 eta_m=3.19;       
! !   cparvector(28)=MUNKRESFX(par(10), DBLE(0.1),DBLE(2.00)) !  28 omeg=1.01369726080713;     
! !   cparvector(69)=MUNKRESFX(par(11), DBLE(0.1),DBLE(2.00)) !  69 sigma_m=1.2;      
! !   cparvector(33)=MUNKRESFX(par(12), DBLE(0.1),DBLE(2.00)) !  33 omega_x=0.449441398423515;  
!   cparvector(43)=MUNKRESFX(par(9), DBLE(0.01),DBLE(0.99)) ! 48 rho_pi_star=0.5;    
!   cparvector(48)=MUNKRESFX(par(10), DBLE(0.01),DBLE(0.99)) ! 49 rho_qm=0.8;         
!   cparvector(49)=MUNKRESFX(par(11), DBLE(0.01),DBLE(0.99)) ! 50 rho_qmr=0.8;        
!   cparvector(50)=MUNKRESFX(par(12), DBLE(0.01),DBLE(0.99)) ! 51 rho_tr=0.8;         
!   cparvector(51)=MUNKRESFX(par(13), DBLE(0.01),DBLE(0.99)) ! 53 rho_z_xdemand=0.55; 
!   cparvector(53)=MUNKRESFX(par(14), DBLE(0.01),DBLE(0.99)) ! 54 rho_zcd=0.8;        
!   cparvector(54)=MUNKRESFX(par(15), DBLE(0.01),DBLE(0.99)) ! 55 rho_zd=0.8;         
!   cparvector(55)=MUNKRESFX(par(16), DBLE(0.01),DBLE(0.99)) ! 56 rho_ze=0.8;         
!   cparvector(56)=MUNKRESFX(par(17), DBLE(0.01),DBLE(0.99)) ! 57 rho_zh=0.8;         
!   cparvector(57)=MUNKRESFX(par(18), DBLE(0.01),DBLE(0.99)) ! 60 rho_zm=0.8;         
!   cparvector(60)=MUNKRESFX(par(19), DBLE(0.01),DBLE(0.99)) ! 61 rho_znt=0.8;        
!   cparvector(61)=MUNKRESFX(par(20), DBLE(0.01),DBLE(0.99)) ! 62 rho_zq=0.8;         
!   cparvector(62)=MUNKRESFX(par(21), DBLE(0.01),DBLE(0.99)) ! 63 rho_zrm=0.8;        
!   cparvector(63)=MUNKRESFX(par(22), DBLE(0.01),DBLE(0.99)) ! 64 rho_zu=0.65;        
!   cparvector(64)=MUNKRESFX(par(23), DBLE(0.01),DBLE(0.99)) ! 65 rho_zx=0.8;         
!   cparvector(65)=MUNKRESFX(par(24), DBLE(0.01),DBLE(0.99)) ! 66 rho_zxd=0.8;        
!   cparvector(66)=MUNKRESFX(par(25), DBLE(0.01),DBLE(0.99)) ! 66 rho_zxd=0.8; 
!   cparvector(12)=MUNKRESFX(par(26), DBLE(1.0),DBLE(5.00))  !  12 eta_m=3.19;       
!  print *,cparvector(12) 
                                                        
  CALL LOGLIK_PAT(CPARVECTOR,LOG_L,INFO)    
 !   IF ( INFO .NE. 0 ) THEN 
!      CALL PASSPAR(PARVECTOR,NPAR)
!   END IF 
  !LOG_L = LOG_L
!   print *
   print *, log_l
!   write( *,'(a, f14.7)'), 'epsq = ',   cparvector(4) 
!   write( *,'(a, f14.7)'), 'epsw = ',   cparvector(10)
!   write( *,'(a, f14.7)'), 'epscd= ',   cparvector(5) 
!   write( *,'(a, f14.7)'), 'epse = ',   cparvector(6) 
!   write( *,'(a, f14.7)'), 'epsm = ',   cparvector(7)
!   write( *,'(a, f14.7)'), 'epsrm= ',   cparvector(8) 
!   write( *,'(a, f14.7)'), 'epst = ',   cparvector(9) 
!   write( *,'(a, f14.7)'), 'epsxd= ',   cparvector(11)
!   write(*,'(a,f14.7)'),'eta_m',        cparvector(12)
!   write(*,'(a,f14.7)'),'omeg',           cparvector(28)
  !   write(*,'(a,f14.7)'),'sigma_m',           cparvector(69)
!   write(*,'(a,f14.7)'),'omega_x',        cparvector(33)
!   write(*,'(a,f14.7)'),'rho_pi_star',      cparvector(43)
!   write(*,'(a,f14.7)'),'rho_qm',           cparvector(48)
!   write(*,'(a,f14.7)'),'rho_qmr',         cparvector(49)
!   write(*,'(a,f14.7)'),'rho_tr',           cparvector(50)
!   write(*,'(a,f14.7)'),'rho_z_xdeman',    cparvector(51)
!   write(*,'(a,f14.7)'),'rho_zcd=',       cparvector(53)
!   write(*,'(a,f14.7)'),'rho_zd=',         cparvector(54)
!   write(*,'(a,f14.7)'),'rho_ze=',         cparvector(55)
!   write(*,'(a,f14.7)'),'rho_zh=',         cparvector(56)
!   write(*,'(a,f14.7)'),'rho_zm=',         cparvector(57)
!   write(*,'(a,f14.7)'),'rho_znt=',       cparvector(60)
!   write(*,'(a,f14.7)'),'rho_zq=',         cparvector(61)
!   write(*,'(a,f14.7)'),'rho_zrm=',       cparvector(62)
!   write(*,'(a,f14.7)'),'rho_zu=',         cparvector(63)
!   write(*,'(a,f14.7)'),'rho_zx=',         cparvector(64)
!   write(*,'(a,f14.7)'),'rho_zxd=',       cparvector(65)
!   write(*,'(a,f14.7)'),'rho_zxd=',       cparvector(66)  
! write( *,'(a, es14.7)'), 'log_l=', log_l
  
  

  
END SUBROUTINE ESTIMATEOBJ
