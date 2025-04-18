SUBROUTINE SOLVESS(PAR, NPAR, XINI, BST,INFO )  
  USE HYBRID_CONST_MOD
  !USE CSOLVE_MOD
  IMPLICIT NONE 
  INTEGER, PARAMETER  :: dp = SELECTED_REAL_KIND(12, 60)
  INTEGER, INTENT(IN):: NPAR
  REAL (dp), intent(INOUT),DIMENSION(NEQ) ::  XINI
  REAL (dp),  intent(OUT) :: BST
  REAL (dp),  iNTENT(IN), DIMENSION(NPAR)  :: PAR
  INTEGER, INTENT(OUT)::   INFO 
  
  REAL (dp),  DIMENSION(NEQ) ::  CXINI, FVEC, XBST  
  INTEGER, PARAMETER:: LONG = SELECTED_REAL_KIND(9,99)
  REAL (dp),  PARAMETER ::  TOL = 0.5*SQRT(EPSILON(1.0_LONG))
  REAL (dp)  BSTALL, FOBJ
  INTEGER NUMTRYSI,NP, I,J,R
  
  REAL (dp),  DIMENSION(NEQ, NMC+1) :: POINTS

  !!para nelder!!!!
  REAL(DP), DIMENSION(NEQ) :: step
  REAL(DP), DIMENSION(NEQ) :: CXINIMIN
  INTEGER, PARAMETER :: KONVGE = 100 !INT(SIZEPAR*6)
  INTEGER :: KCOUNT = 7500
  INTEGER RESTART
  REAL(DP) YNEWLO
  REAL(DP), PARAMETER :: REQMIN=1.0E-02
  INTEGER ICOUNT
  INTEGER :: NUMRES
  INTEGER IFAULT
  REAL(DP) :: EPSFCN = 1.0E-4_DP
  REAL(DP) :: TOL_SS  
  EXTERNAL FCNSA, HYBRIDSA, PASSPAR, WRCSOLVE, FCNNM
  
  CALL PASSPAR(PAR,NPAR)
  
  CXINI = XINI   
  CALL WRHBRD(CXINI, NEQ, FVEC, INFO) !fsolve
  
  TOL_SS=1.0E-14_DP
  !CALL WRCSOLVE(CXINI, NEQ, FVEC, INFO)
  !PRINT *,"NUMTRIES:",NUMTRYS
  !print *,"fsolve:",SUM(FVEC**2.0_dp)
  IF ( (( INFO .EQ. 1 ) .OR. (INFO .eq. 4))  .AND. ( SUM(FVEC**2.0_dp) .lt. TOL_SS) ) THEN      
     BST = SUM(FVEC**2.0)  
     XINI = CXINI
     INFO = 0     
     RETURN
  ELSE     
     !COMIENZA EL HIBRIDO
     BSTALL =  HUGE(0.0)
     NUMTRYSI  = 0
     INFO = 1     
     DO WHILE ( ( BSTALL .GT. TOL_SS ) .AND. ( NUMTRYSI .LT. NUMTRYS ) ) 
        print *, ' Searching for a new steady state ...', NUMTRYSI
        NUMTRYSI = NUMTRYSI +1        
        BST = HUGE(0.0)   
        CALL HYBRIDSA(FCNSA,NP,POINTS) 
        I=int(1) 
        DO WHILE ( ( BST .GT. TOL_SS ) .AND. ( I .LT. NP+1 ) )               
           CXINI = POINTS(1:NEQ, I)

           CALL WRHBRD(CXINI, NEQ, FVEC, INFO)                                 
           IF ( .NOT. (ANY(ISNAN(FVEC)) ) ) THEN               
              FOBJ = SUM(FVEC**2.0_dp)
         !   print *,"FOBJ_FSOLVE:",FOBJ
              IF (FOBJ .LT. BST) THEN                  
                 BST = FOBJ
                 XBST = CXINI
              END IF
           END IF
        I=I+1
        END DO
        BSTALL = BST  
      END DO
     
     if ( NUMTRYSI .GE. NUMTRYS)  then 
        INFO = 1       
        XINI = CXINI
!        print *, ' Got a problem with the steady state ... at ', NUMTRYSI 
!       OPEN (1,FILE="no_ss_par.txt" )
!       DO I = 1, NPAR
!     !     DO I = 1, NUM_CNTRL_OBS+NUM_K
!             write (1, '(ES14.7)'), PAR(I) 
!    !   !     write(1,'(1X,200f10.5)') (AMAT(i,j),j=1,UBOUND(AMAT, 2)) 
!     !     END DO
!       END DO
!      CLOSE(1)     
!
!       OPEN (1,FILE="no_ss_xini.txt" )
!       DO I = 1, NEQ
!     !     DO I = 1, NUM_CNTRL_OBS+NUM_K
!             write (1, '(ES14.7)'), XINI(I) 
!    !   !     write(1,'(1X,200f10.5)') (AMAT(i,j),j=1,UBOUND(AMAT, 2)) 
!     !     END DO
!       END DO
!      CLOSE(1)    

     else 
        INFO = 0
        XINI = XBST
        BST = BSTALL
        print *, ' Got a new steady state ...', NUMTRYSI 
        DO I=1,NEQ
           PRINT *,"XINI=",XINI(I)
        ENDDO
     end if
  END IF
  
  
  

END SUBROUTINE SOLVESS
