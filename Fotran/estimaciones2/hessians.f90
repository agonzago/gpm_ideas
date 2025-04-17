SUBROUTINE HESSIAN(FCN, X, N, GSTEP, HESSIAN_MAT)
  IMPLICIT NONE

  INTEGER,INTENT(IN) :: N
  INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(12, 60)
  REAL(DP), DIMENSION(N), INTENT(IN) :: X
  REAL(DP), INTENT(IN) :: GSTEP    
  REAL(DP), PARAMETER :: EPS= EPSILON(1.0_DP)
  REAL(DP), INTENT(OUT) , DIMENSION(N,N) :: HESSIAN_MAT

  !  EXTERNAL FNC
  INTERFACE
     SUBROUTINE FCN(N, X, FVEC)
       IMPLICIT NONE
       INTEGER, PARAMETER  ::  DP = SELECTED_REAL_KIND(12, 60)
       INTEGER, INTENT(IN)      :: N
       REAL (DP), INTENT(IN), DIMENSION(N) :: X
       REAL (DP), INTENT(OUT)   :: FVEC
     END SUBROUTINE FCN
  END INTERFACE

  !WORKING VARIABLES
  REAL(DP), DIMENSION(N):: H1, H_1, XH1, XH_1
  REAL(DP), DIMENSION(N) :: F1, F_1,TEMP
  REAL(DP) F0, F_11,F11, FEV 

  INTEGER I, J

  H1=MAX(ABS(X),SQRT(GSTEP))*EPS**(1.0_DP/4.0_DP)
  !H1(4) = 1.0E-4_DP


  H_1=H1  
  XH1=X+H1  
  H1=XH1-X  
  XH1=X-H_1  
  H_1=X-XH1


  !WRITE (*, '(4F10.5)'), H1
  !  STOP
  XH1=X
  HESSIAN_MAT=0.0_DP

  CALL FCN(N, X, F0)    

  !PRINT *, H1
  !STOP
  F1= 0.0_DP 
  F_1=F1  
  DO I=1,N    
     XH1(I)=X(I)+H1(I)
     CALL FCN(N, XH1, FEV)
     F1(I) = FEV     
     XH1(I)=X(I)-H_1(I)     
     CALL FCN(N, XH1, FEV)
     F_1(I) = FEV
     XH1(I)=X(I)     
  END DO

  XH_1=XH1 
  DO I=1,N    
     HESSIAN_MAT(I,I) = (F1(I)+F_1(I)-2*F0)/(H1(I)*H_1(I))
     TEMP=F1+F_1-F0 !*ONES(1,N)
     DO J=I+1,N        
        XH1(I)=X(I)+H1(I)
        XH1(J)=X(J)+H_1(J)
        CALL FCN(N, XH1, FEV)
        F11 = FEV        

        XH_1(I)=X(I)-H1(I)
        XH_1(J)=X(J)-H_1(J)
        CALL FCN(N, XH_1, FEV)
        F_11 = FEV        

        HESSIAN_MAT(I, J) = -( -F11 - F_11 + TEMP(I) + TEMP(J))/(2.0_DP*H1(I)*H_1(J))
        HESSIAN_MAT(J, I) =  HESSIAN_MAT(I, J)

        XH1(I)=X(I)
        XH1(J)=X(J)
        XH_1(I)=X(I)
        XH_1(J)=X(J)
     END DO
  END DO

  !   DO I = 1, N
  !      WRITE(*, '(3ES15.7)') , HESSIAN_MAT(I, 1:N)
  !   END DO
END SUBROUTINE HESSIAN


SUBROUTINE HESSIAN_RICH(FCN, X,N, R, FHESS)
  IMPLICIT NONE 

  INTEGER,INTENT(IN) :: N
  INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(12, 60)
  REAL(DP), DIMENSION(N), INTENT(IN) :: X

  REAL(DP), PARAMETER :: DTOL=0.00001_DP    
  REAL(DP), PARAMETER :: ZTOL= SQRT(EPSILON(1.0_DP)/7.0E-7_DP)   !SQRT(EPSILON(1.0_DP)/7.0E-7_DP)  
  REAL(DP), PARAMETER ::  EPS=1.0E-4_DP !EPSILON(1.0_DP)  
  INTEGER,  INTENT(IN) :: R
  REAL(DP) , PARAMETER :: V = 2.0_DP  
  REAL(DP), INTENT(OUT), DIMENSION(N, N) :: FHESS

  EXTERNAL FNC
  INTERFACE
     SUBROUTINE FCN(N, X, FVEC)
       IMPLICIT NONE
       INTEGER, PARAMETER  ::  DP = SELECTED_REAL_KIND(12, 60)
       INTEGER, INTENT(IN)      :: N
       REAL (DP), INTENT(IN)    :: X(N)
       REAL (DP), INTENT(OUT)   :: FVEC
     END SUBROUTINE FCN
  END INTERFACE

  REAL(DP) , DIMENSION(N) :: H0, H
  REAL(DP)  FEV, F0, F1, F2, TEMPI, TEMPJ 
  REAL(DP) , DIMENSION(1,INT(N*(N+3)/2)) :: D
  REAL(DP) , DIMENSION(1,R) :: DAPROX, HAPROX
  REAL(DP) , DIMENSION(1,N) :: DDIAG, HDIAG
  INTEGER I,K,M, J, U,  P
  REAL(DP), DIMENSION(N) :: XTEMP

  XTEMP = X
  CALL FCN(N, XTEMP, FEV)
  F0 = FEV  
  P = N
  H0 = ABS(DTOL*XTEMP)+EPS*(MIN(ABS(XTEMP) ,ZTOL))  
  !  PRINT *, H0
  !   PRINT *, H0
  ! XTEMP = X+H0
  !   H0 = XTEMP-X
  !   PRINT *, H0
  D = 0.0_DP
  DDIAG =0.0_DP
  DAPROX = 0.0_DP
  HAPROX = 0.0_DP
  DO I = 1, P ! TAMAÑO DEL VECTOR
     ! EACH PARAMETER  - FIRST DERIV. & HESSIAN DIAGONAL
     H = H0
     TEMPI = XTEMP(I)
     DO K = 1, R  !NUMERO DE REDUCCIONES                      
        XTEMP(I) = TEMPI + H(I)        
        CALL FCN(N, XTEMP, FEV)
        F1 = FEV
        XTEMP(I) = TEMPI - H(I)
        CALL FCN(N, XTEMP, FEV)
        F2 = FEV        
        DAPROX(1,K) = (F1 - F2)/(2.0_DP*H(I))
        HAPROX(1,K) = (F1 - 2*F0 + F2)/(H(I)*H(I))        
        XTEMP(I) = TEMPI         
        H = H/V            
     END DO !R   

     DO M = 1, (R-1)
        DO K = 1, (R-M)
           DAPROX(1,K) = (DAPROX(1,(K+1))*(4.0_DP**DBLE(M))-DAPROX(1,K))/(4.0_DP**DBLE(M)-1.0_DP)         
           HAPROX(1,K) = (HAPROX(1,(K+1))*(4.0_DP**DBLE(M))-HAPROX(1,K))/(4.0_DP**DBLE(M)-1.0_DP)
        END DO!K        
     END DO !M       
     D(1,I) = DAPROX(1,1)     
     HDIAG(1,I) = HAPROX(1,1)     
  END DO !I


  U = P
  DO I =1, P
     TEMPI = XTEMP(I)
     DO J =1, I
        TEMPJ = XTEMP(J)
        U = U +1 
        IF ( I .EQ. J) THEN
           D(1,U) = HDIAG(1,I)
        ELSE 
           H = H0            
           DO K = 1, R
              XTEMP(I) = TEMPI + H(I)
              XTEMP(J) = TEMPJ + H(J)
              CALL FCN(N, XTEMP, FEV)
              F1 = FEV
              XTEMP(I) = TEMPI
              XTEMP(J) = TEMPJ

              XTEMP(I) = TEMPI - H(I)
              XTEMP(J) = TEMPJ - H(J)
              CALL FCN(N, XTEMP, FEV)
              F2 = FEV              
              XTEMP(I) = TEMPI
              XTEMP(J) = TEMPJ
              DAPROX(1,K) =  (F1 - 2*F0 + F2 - HDIAG(1,I)*H(I)**2.0_DP - HDIAG(1,J)*H(J)**2.0_DP)&
                   &/(2.0_DP*H(I)*H(J))  !# F''(I,J)  
              H = H/V             
           END DO


           DO M = 1, (R-1)
              DO K = 1, (R - M)
                 DAPROX(1,K) = (DAPROX(1,K+1)*(4.0_DP**M)-DAPROX(1,K))/(4**M-1)
              END DO
              D(1,U) = DAPROX(1,1)
           END DO !M
        END IF ! II = J
        XTEMP(J) = TEMPJ
     END DO !J
     XTEMP(I) = TEMPI
  END DO !I

  FHESS = 0.0_DP 
  U = N
  DO I = 1, N
     DO J = 1, I
        U = U + 1
        FHESS(I, J ) = D(1, U)
     END DO
  END DO
  FHESS = FHESS + TRANSPOSE(FHESS)
  DO I = 1, N 
     FHESS(I,I) = 0.5_DP*FHESS(I,I)
  END DO

  RETURN

END SUBROUTINE HESSIAN_RICH
