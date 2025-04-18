SUBROUTINE KALMANSTATES(Y,Z,T, H,R, Q,A0,P0, PDIM, MDIM, RDIM, NDIM,S_AT, S_PT,S_VT, LIK,INFO)
  !KALMAN FILTER DURBIN AND KOOPMAN
  ! Este filtro es muy ineficiente. x dos cosas:
  ! 1. Calcula mas matrices que las necesarias
  ! 2. El filtro usando Ricatti puede ser mas rápido.
 
  INTEGER, INTENT(IN) :: PDIM, MDIM, RDIM, NDIM
  DOUBLE PRECISION , INTENT(IN), DIMENSION(PDIM, NDIM) :: Y
  DOUBLE PRECISION , INTENT(IN), DIMENSION(PDIM,MDIM) :: Z
  DOUBLE PRECISION , INTENT(IN), DIMENSION(MDIM,MDIM) :: T
  DOUBLE PRECISION , INTENT(IN), DIMENSION(PDIM,PDIM) :: H
  DOUBLE PRECISION , INTENT(IN), DIMENSION(MDIM,RDIM) :: R
  DOUBLE PRECISION , INTENT(IN), DIMENSION(RDIM,RDIM) :: Q  
  DOUBLE PRECISION , INTENT(IN), DIMENSION(MDIM) :: A0
  DOUBLE PRECISION , INTENT(IN), DIMENSION(MDIM,MDIM) :: P0
  DOUBLE PRECISION, INTENT(OUT) :: LIK  
  !WORKING VARIABLES
  INTEGER I 
  DOUBLE PRECISION, DIMENSION(MDIM) ::  AT, AT_NEW, ATT
  DOUBLE PRECISION, DIMENSION(PDIM) ::  VT, ZAT ,VTIFT
  DOUBLE PRECISION, DIMENSION(MDIM,MDIM) :: PT,PT_NEW,PTT
  DOUBLE PRECISION, DIMENSION(PDIM,PDIM) :: FT,IFT
  DOUBLE PRECISION, DIMENSION(MDIM,PDIM) :: KT
  DOUBLE PRECISION, DIMENSION(MDIM,MDIM) :: LT
  DOUBLE PRECISION, DIMENSION(MDIM,PDIM) :: MT
  DOUBLE PRECISION, DIMENSION(MDIM,MDIM) :: TPT
  DOUBLE PRECISION, DIMENSION(MDIM,MDIM) :: TMT
  DOUBLE PRECISION, DIMENSION(MDIM,RDIM) :: RQ

  
  DOUBLE PRECISION, intent(out) ,DIMENSION(MDIM, 0:NDIM) :: S_AT
  DOUBLE PRECISION, intent(out) , DIMENSION(MDIM*MDIM,0:NDIM) :: S_PT
  DOUBLE PRECISION, intent(out) , DIMENSION(PDIM, NDIM) :: S_VT
  
  DOUBLE PRECISION PI, L2PI, DETFT
  INTEGER INFO
  
  DOUBLE PRECISION DDOT
  EXTERNAL DDOT

  integer p1, p2
  double precision sum

  PI = ACOS(-1.0)
  L2PI = LOG(2*PI)

  !KALMAN FILTER OF XT WITH Y_T-1, (ONE PERIOD AHEAD FORECAST)
  !1. Evitar multiplicar por cero
  !2. Usar las simetricas
  !3. Usar el Fast Kalman Filter de Koop
  AT=A0
  PT=P0
  LIK = 0.0
  S_AT(:,0) = AT
  S_PT(:,0) = RESHAPE(PT, (/MDIM*MDIM/))
  DO I = 1, NDIM
     CALL DGEMV('N', PDIM, MDIM, DBLE(1.0),Z ,PDIM, AT, INT(1), DBLE(0.0), ZAT, INT(1)) 
     VT = Y(:,I) - ZAT     
!      print *, VT
!      print *
     
     CALL DGEMM('N','T',MDIM,PDIM,MDIM,DBLE(1.0),PT,MDIM,Z,PDIM, DBLE(0.0),MT,MDIM)       
    
     FT = H

     
     CALL DGEMM('N','N',PDIM,PDIM,MDIM,DBLE(1.0),Z,PDIM,MT,MDIM, DBLE(1.0),FT,PDIM)       
   !   sum= 0.0
!      do p1 = 1, PDIM
!         sum = sum+ FT(p1,p1)*FT(p1,p1)
!      end do
!      print *,sum
          
     CALL DINVERTLU(FT, IFT, DETFT, INFO)
     IF ( INFO .NE. 0 ) THEN 
        PRINT *, "MATRIX FT NOT INVERTIBLE"        
        RETURN
     END IF

     CALL DSYMV('U',  PDIM,DBLE(1.0),IFT,PDIM, VT, INT(1.0), DBLE(0.0), VTIFT, int(1))       
     
     LIK = LIK + LOG(DETFT) + DDOT(PDIM,VTIFT,INT(1),VT,INT(1)) 
     
     CALL DGEMM('N','N',MDIM,PDIM,MDIM,DBLE(1.0),T, MDIM,MT,MDIM,DBLE(0.0),TMT,MDIM)       
     CALL DSYMM('R','U', MDIM, PDIM,DBLE(1.0),IFT, PDIM, TMT,MDIM, DBLE(0.0), KT, MDIM)

     LT = T
     CALL DGEMM('N','N',MDIM,MDIM,PDIM,DBLE(-1.0),KT, MDIM,Z,PDIM,DBLE(1.0),LT,MDIM)       
     
     CALL DGEMV('N', MDIM, MDIM, DBLE(1.0), T, MDIM, AT,INT(1),DBLE(0.0), AT_NEW, INT(1))
     CALL DGEMV('N', MDIM, PDIM, DBLE(1.0), KT, MDIM, VT, INT(1), DBLE(1.0), AT_NEW, INT(1))

     CALL DGEMM('N','N',MDIM,MDIM,MDIM,DBLE(1.0),T, MDIM,PT,MDIM,DBLE(0.0),TPT,MDIM)
     CALL DGEMM('N','T',MDIM,MDIM,MDIM,DBLE(1.0),TPT, MDIM,LT,MDIM,DBLE(0.0),PT_NEW,MDIM)
     
     
     CALL DGEMM('N','N',MDIM,RDIM,RDIM,DBLE(1.0),R,MDIM,Q,RDIM,DBLE(0.0),RQ,MDIM)     
     CALL DGEMM('N','T',MDIM,MDIM,RDIM,DBLE(1.0),RQ, MDIM,R,MDIM,DBLE(1.0),PT_NEW,MDIM)

     !ATT = AT
     !CALL DGEMV('N','T',MDIM,MDIM,RDIM,DBLE(1.0),IFT, MDIM,MT,MDIM,DBLE(1.0),ATT,INT(1.0))
     ATT  = AT + MATMUL(MATMUL(MT, IFT), VT)
     PTT = PT - MATMUL(MATMUL(MT, IFT), TRANSPOSE(MT))
     
     S_VT(:,I) = VT
     S_AT(:,I) = ATT
     S_PT(:,I) = RESHAPE(PTT, (/MDIM*MDIM/))
     
     !!ACTUALIZA LOS VALORES
     AT = AT_NEW
     PT = PT_NEW        
     
  END DO

  !  LIK=-0.5*(NDIM*PDIM*L2PI)-0.5*LIK
  LIK=0.5*(NDIM*PDIM*L2PI)+0.5*LIK
!  LIK = 0.5*LIK

CONTAINS 

  SUBROUTINE CHOLINVERSE(A, AINV, DET, INFO)
    !USING THE CHOLESKY FACTORIZATIO
    IMPLICIT NONE

    INTEGER, INTENT(OUT) :: INFO
    DOUBLE PRECISION , INTENT(IN), DIMENSION(:,:) :: A
    DOUBLE PRECISION , INTENT(OUT), DIMENSION(:,:) :: AINV
    DOUBLE PRECISION DET
    INTEGER N, I

    DET = 0.0
    AINV = A
    N = SIZE(AINV,1)
    IF ( SIZE(AINV,1) .NE. SIZE(AINV,2) ) THEN 
       PRINT *, "MATRIX IS NOT SQUARED"
       RETURN
    END IF

    CALL DPOTRF('U', N, AINV, N, INFO )
    IF (INFO .NE. 0) THEN 
       PRINT *, "QR FACTORIZATION FAIL"
       RETURN
    END IF

    ! COMPUTES THE DETERMINANT
    DET = 1
    DO I = 1,N 
       DET = DET*AINV(I,I)
    ENDDO
    DET = DET**2


    CALL DPOTRI('U', N, AINV, N, INFO )
    IF (INFO .NE. 0) THEN 
       PRINT *, "QR INVERSE FAIL"
       RETURN
    END IF


    RETURN
  END SUBROUTINE CHOLINVERSE


  SUBROUTINE DINVERTLU(A, AINV, DET, OK)
    ! RETURNS THE INVERSE OF A MATRIX AND THE LOG DETERMINANT
    IMPLICIT NONE
    DOUBLE PRECISION, DIMENSION(:,:), INTENT( IN ) :: A
    DOUBLE PRECISION, DIMENSION(:,:), INTENT( OUT ) :: AINV
    INTEGER, INTENT(OUT) ::  OK
    DOUBLE PRECISION, DIMENSION(SIZE(A,1)) :: PIVOT, WORK      
    INTEGER :: N, M
    DOUBLE PRECISION, INTENT(OUT) :: DET    
    DOUBLE PRECISION SDET(2)


    N = SIZE(A,1)
    M = SIZE(A,2)
    IF ( M .NE. N ) THEN          
       WRITE(*,*) "ERROR: MATRIX IS NOT SQUARE"
       RETURN
    END IF
    AINV = A
    CALL DGETRF( N , M, AINV, N, PIVOT, OK )  
    IF ( OK .NE. 0 ) THEN          
       WRITE(*,*) "ERROR: MATRIX IS NOT LU FACTORIZABLE"
       RETURN
    END IF

    !    COMPUTES THE DETERMINANT
    CALL DGEDI( AINV, N, N, PIVOT, SDET, WORK, INT(10) )
    DET = SDET(1) * 10.0D0**SDET(2)


    CALL DGETRI( N, AINV, N, PIVOT, WORK, N, OK )        
    IF ( OK .NE. 0 ) THEN   
       WRITE(*,*) "ERROR: MATRIX IS NOT INVERTIBLE"
       RETURN 
    END IF


  END SUBROUTINE DINVERTLU


END SUBROUTINE KALMANSTATES








