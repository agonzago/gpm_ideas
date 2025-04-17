MODULE CSOLVE_MOD

  PUBLIC  :: CSOLVEF90 
  CONTAINS
  
  SUBROUTINE CSOLVEF90(FNC,X,N,CRIT,ITMAX, VERBOSE, RC) 
    USE RANDOM
    IMPLICIT NONE  
    !INTEGER, PARAMETER  :: DP = SELECTED_REAL_KIND(14, 60)
    REAL(DP) ,intent(in) :: CRIT
    LOGICAL, INTENT(IN) :: VERBOSE
    INTEGER ,INTENT(IN) :: ITMAX,N
    INTEGER ,INTENT(OUT) :: RC
    REAL(DP) , INTENT(INOUT) , DIMENSION(N) :: X  
    REAL(DP) DELTA, ALPHA,FACTOR  
    LOGICAL DONE, SUBDONE, SHRINK, RANDOMIZE        
    REAL(DP), DIMENSION(N) :: WA1, WA2
    REAL(DP), PARAMETER :: EPSFCN = 10_dp*EPSILON(1.0_DP) !EPSILON(1.0_DP) !
    REAL(DP), PARAMETER :: EPSI= 0.5_DP*SQRT(EPSILON(1.0_DP))

    INTEGER  ITCT, ML, MU, IFLAG, RANK, ITERN, INFO
    REAL(DP), DIMENSION(N) :: F0, F, FMIN !, FVEC
    REAL(DP), DIMENSION(N) :: DX0, DX, XMIN
    REAL(DP) :: LAMBDA, LAMBDAMIN, DXSIZE, AF, AFMIN, AF0, AF00,NORMX 
    REAL(DP) , DIMENSION(N, N) :: GRAD
    REAL(DP) :: ANORM, RCOND
    REAL(DP) :: tmp
    integer IterG

    EXTERNAL FNC
    !SOLVELIN
    INTERFACE
       SUBROUTINE FNC(N, X, FVEC, IFLAG)
         IMPLICIT NONE
         INTEGER, PARAMETER  :: DP = SELECTED_REAL_KIND(14, 60)
         INTEGER, INTENT(IN)      :: N
         REAL (DP), INTENT(IN)    :: X(N)
         REAL (DP), INTENT(OUT)   :: FVEC(N)
         INTEGER, INTENT(IN OUT)  :: IFLAG
       END SUBROUTINE FNC
    END INTERFACE

    INTERFACE
       SUBROUTINE SOLVELIN(YIN, X, BETA, INFO, RANK)
         IMPLICIT NONE     
         INTEGER, INTENT(OUT) :: RANK, INFO
         DOUBLE PRECISION, INTENT(IN), DIMENSION(:) :: YIN
         DOUBLE PRECISION, INTENT(IN), DIMENSION(:,:) :: X
         DOUBLE PRECISION, INTENT(OUT), DIMENSION(SIZE(X,2),1) :: BETA
       END SUBROUTINE SOLVELIN
    END INTERFACE

    !------------ ALPHA ------------------
    ! TOLERANCE ON RATE OF DESCENT
    ALPHA=1E-3_DP    
    FACTOR = 0.6_DP
    DELTA = 1.0e-6_dp
    DONE=.FALSE.
    ML = N - 1
    MU = N - 1
    IFLAG = 1

    CALL FNC(N, X, F0, IFLAG)     
    AF0=SUM(ABS(F0))
    AF00=AF0
    ITCT=0
    DO WHILE ( .NOT. DONE )      
       IF ( isnan(af0) .or. ((ITCT > 3) .AND. (AF00-AF0<CRIT*MAX(1.0,AF0)) .AND. (MOD(ITCT,2) .EQ. 1) )) THEN
          !print *, "hola"
          RANDOMIZE=.TRUE.
       ELSE        
          CALL FDJAC1(FNC, N, X, F0, GRAD, N, IFLAG, ML, MU, DELTA,WA1, WA2)              
          IF ( ( .NOT. ANY(ISNAN(GRAD)) ) ) then ! .AND. ( SUM(ABS(GRAD)) .GT. 4.0_DP*(DBLE(N)**2.0_DP)*EPSI ) ) THEN
             !CALL SVD_CALCULOS(GRAD, N, INFO)            
             CALL RCOND_SUB(GRAD, N, ANORM, RCOND, 'A', INFO)
             IF ( RCOND .LT. 1E-12_DP ) THEN
                DO ITERG = 1, N
                   GRAD(ITERG,ITERG) = GRAD(ITERG,ITERG) +  DELTA
                END DO
             END IF
             CALL SOLVELIN(F0, GRAD, DX0, INFO, RANK) 
             !print *, DX0
             DX0 = -DX0         
             RANDOMIZE=.FALSE.
          ELSE
             IF (VERBOSE) THEN 
                PRINT *, 'GRADIENT IMAGINARY'
             END IF
             RANDOMIZE=.FALSE.
          END IF
       END IF
       IF ( RANDOMIZE ) THEN 
          IF( VERBOSE) THEN 
             PRINT *, 'RANDOM SEARCH'
          END IF
          NORMX = SQRT(SUM(X**2.0_DP))
          DO ITERN = 1,N
             tmp = RANDOM_NORMAL()! REAL(0.0), REAL(1.0))           
             DX0(ITERN) = NORMX/tmp

          END DO
       END IF

       LAMBDA=1.0_DP
       LAMBDAMIN=1.0_DP
       FMIN=F0
       XMIN=X
       AFMIN=AF0
       DXSIZE=SQRT(SUM(DX0**2.0_DP))
       FACTOR=0.6_DP
       SHRINK=.TRUE.
       SUBDONE=.FALSE.

       DO WHILE (.NOT. SUBDONE )
          DX=LAMBDA*DX0       
          !print *, dx
          CALL FNC(N, X+DX, F, IFLAG)        
          AF=SUM(ABS(F))

          IF ( (isnan(af0).and. .not. isnan(af)) .or. ( AF<AFMIN )) THEN 
             AFMIN=AF
             FMIN=F
             LAMBDAMIN=LAMBDA
             XMIN=X+DX
          ENDIF
          IF (((LAMBDA >0) .AND. (isnan(af) .or. (AF0-AF < ALPHA*LAMBDA&
               &*AF0))) .OR. ((LAMBDA<0) .and. (isnan(af) .or. (AF0-AF &
               &< 0) ) ) ) THEN
             IF (.NOT. SHRINK) THEN
                FACTOR=FACTOR**0.6_DP;
                SHRINK=.TRUE.
             END IF
             IF ( ABS(LAMBDA*(1-FACTOR))*DXSIZE > 0.1_dp*DELTA ) THEN
                LAMBDA = FACTOR*LAMBDA
             ELSEIF  ( (LAMBDA > 0.0_DP) .AND. (FACTOR .EQ. 0.6_DP) ) then !I.E., WE'VE ONLY BEEN SHRINKING
                LAMBDA=-0.3_DP
             ELSE 
                SUBDONE=.TRUE.
                IF ( LAMBDA > 0.0_DP ) THEN
                   IF ( FACTOR .EQ. 0.6_DP ) THEN 
                      RC = 2
                   ELSE
                      RC = 1
                   END IF
                ELSE
                   RC=3
                END IF
             END IF
          ELSEIF ( (LAMBDA >0) .AND. (AF-AF0 > (1-ALPHA)*LAMBDA*AF0)) THEN
             IF ( SHRINK ) THEN 
                FACTOR=FACTOR**0.6_DP
                SHRINK=.FALSE.
             END IF
             LAMBDA=LAMBDA/FACTOR
          ELSE ! GOOD VALUE FOUND
             SUBDONE=.TRUE.
             RC=0
          END IF
          !WRITE(*, '(A5, I10,A5,f15.5,A10,f15.5,A5,I4)'), 'ITCT', ITCT,'AF',AFMIN,'LAMBDA', LAMBDAMIN,'RC', RC
       END DO !~SUBDONE
       ITCT=ITCT+1;
       IF(VERBOSE) THEN 
          WRITE(*, '(A5, I10,A5,f15.5,A10,f15.5,A5,I4)'), 'ITCT', ITCT,'AF',AFMIN,'LAMBDA', LAMBDAMIN,'RC', RC
          ! DO ITERN = 1, N
          !            WRITE(*, '(A5, I4, A5, f15.5, A20, I4, A5,f15.5)'), 'XMIN(',ITERN,')=', XMIN(ITERN), '     FMIN(',ITERN,')=', FMIN(ITERN)
          !         END DO
       END IF
       X=XMIN   
       F0=FMIN
       AF00=AF0
       AF0=AFMIN
       !PRINT*, af, aFMIN , CRIT
       IF ( ITCT .GE. ITMAX ) THEN 
          DONE=.TRUE.
          RC=4
       ELSEIF (AF0<CRIT ) THEN
          DONE=.TRUE.
          RC=0
       ENDIF
    END DO
  END SUBROUTINE CSOLVEF90



  SUBROUTINE SVD_CALCULOS(GRAD, N, INFO)
    IMPLICIT NONE 
    INTEGER , INTENT(IN) :: N
    INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(14,60)
    REAL(DP), PARAMETER ::EPSFCN = EPSILON(1.0_DP)
    REAL(DP), INTENT(INOUT) , DIMENSION(N,N) :: GRAD
    INTEGER , INTENT(OUT) :: INFO
    INTEGER LWORK 
    REAL(DP),DIMENSION(:), ALLOCATABLE :: WORK            
    REAL(DP),DIMENSION(1) :: TEMPWORK
    REAL(DP),DIMENSION(N) :: SVDD
    REAL(DP), DIMENSION(N,N):: SVDU, SVDVT 
    INTEGER  LDVT
    INTEGER  LDU
    INTEGER , DIMENSION(8*N) :: IWORK
    REAL(DP) MAX_SVDD
    INTEGER I

    LDVT = N
    LDU  = N
    LWORK = -1        
    CALL DGESDD( 'A', int(N), int(N), GRAD, int(N), SVDD, SVDU, int(LDU), SVDVT, int(LDVT), TEMPWORK, LWORK, int(IWORK), INFO )
    IF (INFO .NE. 0) THEN 
       PRINT *, 'SVD: FAILS '
       RETURN
    END IF

    LWORK = INT(TEMPWORK(1))
    ALLOCATE(WORK(LWORK))
    CALL DGESDD( 'A', int(N), int(N), GRAD, int(N), SVDD, SVDU, int(LDU), SVDVT, int(LDVT), WORK, LWORK, int(IWORK), INFO )
    !CALL DGESDD( 'A', N, N, GRAD,N, SVDD, SVDU, LDU, SVDVT, LDVT, WORK, LWORK, IWORK, INFO )
    IF (INFO .NE. 0) THEN 
       PRINT *, 'SVD: FAILS '
       RETURN
    END IF

    IF ( ( .NOT. (MINVAL(SVDD) .GT. 0) )  .OR.  ((MAXVAL(SVDD)/MINVAL(SVDD))>100.0_DP*EPSFCN) ) THEN  
       MAX_SVDD = MAXVAL(SVDD)*1.0E-13_DP
       SVDD= MAX(SVDD, MAX_SVDD)
       DO I = 1, N
          SVDU(:, I) = SVDU(:, I)*SVDD(I)
       ENDDO
       CALL DGEMM ('N', 'T', N, N, N, DBLE(1.0), SVDU, N, SVDVT, N, DBLE(0.0), GRAD, N)
    END IF
  END SUBROUTINE SVD_CALCULOS

  FUNCTION NORM_VECT(BETA) 
    IMPLICIT NONE
    INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(14,60)
    REAL(DP) , INTENT(IN), DIMENSION(:) :: BETA 
    REAL(DP) NORM_VECT
    NORM_VECT = SQRT(SUM(BETA**2))
  END FUNCTION NORM_VECT

  SUBROUTINE SOLVELIN(YIN, X, BETA, INFO, RANK)
    IMPLICIT NONE
    
    INTEGER, INTENT(OUT) :: RANK, INFO
    DOUBLE PRECISION, INTENT(IN), DIMENSION(:) :: YIN
    DOUBLE PRECISION, INTENT(IN), DIMENSION(:,:) :: X
    DOUBLE PRECISION, INTENT(OUT), DIMENSION(SIZE(X,2),1) :: BETA
    !WORKING VARIABLES 

    !DOUBLE PRECISION, DIMENSION(:,1), ALL
    INTEGER LWORK  
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: YC
    INTEGER, DIMENSION(:), ALLOCATABLE :: JPVT 
    DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: WORK
    DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: XC
    DOUBLE PRECISION, PARAMETER :: RCOND = 1E-12
    EXTERNAL DGELSY  
    INTEGER :: K, N 
    K = SIZE(X, 2) 
    N = SIZE(X,1) 

    ALLOCATE( JPVT(K) )
    ALLOCATE( YC(N, 1) )
    ALLOCATE( XC(N, K) )
    ALLOCATE(WORK(1))  
    JPVT=0
    YC = RESHAPE(YIN, (/N, 1 /))
    XC = X
    !PRINT *, K, N
    !  !WORKSPACE QUERY IS ASSUMED;  
    LWORK = -1
    CALL DGELSY(N, K, INT(1), XC, N, YC, N, JPVT,RCOND, RANK, WORK, LWORK, INFO)  
    IF (INFO .NE. 0) THEN     
       STOP 'SOLVELIN : DEGELSY FAILS AT QUERY WORK SPACE'
    ENDIF

    !   !ALLOCATE MEMORY FOR WORK
    LWORK = WORK(1)
    DEALLOCATE(WORK)
    ALLOCATE(WORK(LWORK))  
    CALL DGELSY(N, K, INT(1), XC, N, YC, N, JPVT,&
         &     RCOND, RANK, WORK, LWORK, INFO)

    IF (INFO .NE. 0) THEN     
       STOP 'SOLVELIN : DEGELSY FAILS'
    ENDIF
    BETA(1:K,1) = YC(1:K,1)

    DEALLOCATE(WORK)
    DEALLOCATE(JPVT)
    DEALLOCATE(YC)
    DEALLOCATE(XC)

  END SUBROUTINE SOLVELIN


  SUBROUTINE RCOND_SUB(A, N, ANORM, RCOND, FLAG, INFO)
    !Si flag = 'n' solo calcula la norma
    !Si flag = 'a' calcula anorm y rcond
    IMPLICIT NONE 
    INTEGER , INTENT(IN) :: N
    INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(14,60) 
    REAL(DP), INTENT(IN) , DIMENSION(N,N) :: A
    REAL(DP) ,DIMENSION(N,N) :: ACOPY 
    INTEGER , DIMENSION(N) :: IPIV(N)
    INTEGER , INTENT(OUT) :: INFO
    REAL(8) , INTENT(OUT) :: RCOND, ANORM
    REAL(8) , DIMENSION (4*N) :: WORK   
    INTEGER , DIMENSION (N) :: IWORK 
    CHARACTER FLAG  
    logical all_var

    REAL(8) DLANGE
    EXTERNAL DLANGE

    all_var = flag.EQ.'N' .OR. flag.EQ.'n' 

    ACOPY = A
    IF (all_var) THEN 
       ANORM = DLANGE( '1', N, N, ACOPY, N, WORK )     
       RCOND = 0.0_dp
    ELSE
       ANORM = DLANGE( '1', N, N, ACOPY, N, WORK )     
       CALL DGETRF( N, N, ACOPY, N, IPIV, INFO )     
       IF ( INFO .EQ. 0 ) THEN 
          CALL DGECON('1', N, ACOPY, N, ANORM, RCOND, WORK, IWORK,INFO )
       END IF
    END IF

  END SUBROUTINE RCOND_SUB

  !SUBRUTINA PARA CALCULAR LAS DERIVADAS NUMERICAS  
  SUBROUTINE FDJAC1(FCN, N, X, FVEC, FJAC, LDFJAC, IFLAG, ML, MU, EPSFCN,   &
       WA1, WA2)

    INTEGER, PARAMETER  :: DP = SELECTED_REAL_KIND(14, 60)
    INTEGER, INTENT(IN)        :: N
    REAL (DP), INTENT(IN OUT)  :: X(N)
    REAL (DP), INTENT(IN)      :: FVEC(N)
    INTEGER, INTENT(IN)        :: LDFJAC
    REAL (DP), INTENT(OUT)     :: FJAC(LDFJAC,N)
    INTEGER, INTENT(IN OUT)    :: IFLAG
    INTEGER, INTENT(IN)        :: ML
    INTEGER, INTENT(IN)        :: MU
    REAL (DP), INTENT(IN)      :: EPSFCN
    REAL (DP), INTENT(IN OUT)  :: WA1(N)
    REAL (DP), INTENT(OUT)     :: WA2(N)

    ! EXTERNAL FCN
    INTERFACE
       SUBROUTINE FCN(N, X, FVEC, IFLAG)
         IMPLICIT NONE
         INTEGER, PARAMETER  :: DP = SELECTED_REAL_KIND(14, 60)
         INTEGER, INTENT(IN)      :: N
         REAL (DP), INTENT(IN)    :: X(N)
         REAL (DP), INTENT(OUT)   :: FVEC(N)
         INTEGER, INTENT(IN OUT)  :: IFLAG
       END SUBROUTINE FCN
    END INTERFACE

    !   **********

    !   SUBROUTINE FDJAC1

    !   THIS SUBROUTINE COMPUTES A FORWARD-DIFFERENCE APPROXIMATION TO THE N BY N
    !   JACOBIAN MATRIX ASSOCIATED WITH A SPECIFIED PROBLEM OF N FUNCTIONS IN N
    !   VARIABLES.  IF THE JACOBIAN HAS A BANDED FORM, THEN FUNCTION EVALUATIONS
    !   ARE SAVED BY ONLY APPROXIMATING THE NONZERO TERMS.

    !   THE SUBROUTINE STATEMENT IS

    !     SUBROUTINE FDJAC1(FCN,N,X,FVEC,FJAC,LDFJAC,IFLAG,ML,MU,EPSFCN,
    !                       WA1,WA2)

    !   WHERE

    !     FCN IS THE NAME OF THE USER-SUPPLIED SUBROUTINE WHICH CALCULATES
    !       THE FUNCTIONS.  FCN MUST BE DECLARED IN AN EXTERNAL STATEMENT IN
    !       THE USER CALLING PROGRAM, AND SHOULD BE WRITTEN AS FOLLOWS.

    !       SUBROUTINE FCN(N,X,FVEC,IFLAG)
    !       INTEGER N,IFLAG
    !       REAL X(N),FVEC(N)
    !       ----------
    !       CALCULATE THE FUNCTIONS AT X AND
    !       RETURN THIS VECTOR IN FVEC.
    !       ----------
    !       RETURN
    !       END

    !       THE VALUE OF IFLAG SHOULD NOT BE CHANGED BY FCN UNLESS
    !       THE USER WANTS TO TERMINATE EXECUTION OF FDJAC1.
    !       IN THIS CASE SET IFLAG TO A NEGATIVE INTEGER.

    !     N IS A POSITIVE INTEGER INPUT VARIABLE SET TO THE NUMBER
    !       OF FUNCTIONS AND VARIABLES.

    !     X IS AN INPUT ARRAY OF LENGTH N.

    !     FVEC IS AN INPUT ARRAY OF LENGTH N WHICH MUST CONTAIN THE
    !       FUNCTIONS EVALUATED AT X.

    !     FJAC IS AN OUTPUT N BY N ARRAY WHICH CONTAINS THE
    !       APPROXIMATION TO THE JACOBIAN MATRIX EVALUATED AT X.

    !     LDFJAC IS A POSITIVE INTEGER INPUT VARIABLE NOT LESS THAN N
    !       WHICH SPECIFIES THE LEADING DIMENSION OF THE ARRAY FJAC.

    !     IFLAG IS AN INTEGER VARIABLE WHICH CAN BE USED TO TERMINATE
    !       THE EXECUTION OF FDJAC1.  SEE DESCRIPTION OF FCN.

    !     ML IS A NONNEGATIVE INTEGER INPUT VARIABLE WHICH SPECIFIES
    !       THE NUMBER OF SUBDIAGONALS WITHIN THE BAND OF THE
    !       JACOBIAN MATRIX. IF THE JACOBIAN IS NOT BANDED, SET
    !       ML TO AT LEAST N - 1.

    !     EPSFCN IS AN INPUT VARIABLE USED IN DETERMINING A SUITABLE
    !       STEP LENGTH FOR THE FORWARD-DIFFERENCE APPROXIMATION. THIS
    !       APPROXIMATION ASSUMES THAT THE RELATIVE ERRORS IN THE
    !       FUNCTIONS ARE OF THE ORDER OF EPSFCN. IF EPSFCN IS LESS
    !       THAN THE MACHINE PRECISION, IT IS ASSUMED THAT THE RELATIVE
    !       ERRORS IN THE FUNCTIONS ARE OF THE ORDER OF THE MACHINE PRECISION.

    !     MU IS A NONNEGATIVE INTEGER INPUT VARIABLE WHICH SPECIFIES
    !       THE NUMBER OF SUPERDIAGONALS WITHIN THE BAND OF THE
    !       JACOBIAN MATRIX. IF THE JACOBIAN IS NOT BANDED, SET
    !       MU TO AT LEAST N - 1.

    !     WA1 AND WA2 ARE WORK ARRAYS OF LENGTH N.  IF ML + MU + 1 IS AT
    !       LEAST N, THEN THE JACOBIAN IS CONSIDERED DENSE, AND WA2 IS
    !       NOT REFERENCED.

    !   SUBPROGRAMS CALLED

    !     MINPACK-SUPPLIED ... SPMPAR

    !     FORTRAN-SUPPLIED ... ABS,MAX,SQRT

    !   ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. MARCH 1980.
    !   BURTON S. GARBOW, KENNETH E. HILLSTROM, JORGE J. MORE

    !   **********
    INTEGER    :: I, J, K, MSUM
    REAL (DP)  :: EPS, EPSMCH, H, TEMP
    REAL (DP), PARAMETER  :: ZERO = 0.0_DP

    !     EPSMCH IS THE MACHINE PRECISION.

    EPSMCH = EPSILON(1.0_DP)

    EPS = SQRT(MAX(EPSFCN, EPSMCH))
    MSUM = ML + MU + 1
    IF (MSUM >= N) THEN

       !        COMPUTATION OF DENSE APPROXIMATE JACOBIAN.

       DO  J = 1, N
          TEMP = X(J)
          H = EPS * ABS(TEMP)
          IF (H == ZERO) H = EPS
          X(J) = TEMP + H
          CALL FCN(N, X, WA1, IFLAG)        
          IF (IFLAG < 0) EXIT
          X(J) = TEMP
          DO  I = 1, N           
             FJAC(I,J) = (WA1(I)-FVEC(I)) / H
          END DO
       END DO
    ELSE

       !        COMPUTATION OF BANDED APPROXIMATE JACOBIAN.

       DO  K = 1, MSUM
          DO  J = K, N, MSUM
             WA2(J) = X(J)
             H = EPS * ABS(WA2(J))
             IF (H == ZERO) H = EPS
             X(J) = WA2(J) + H
          END DO
          CALL FCN(N, X, WA1, IFLAG)
          IF (IFLAG < 0) EXIT
          DO  J = K, N, MSUM
             X(J) = WA2(J)
             H = EPS * ABS(WA2(J))
             IF (H == ZERO) H = EPS
             DO  I = 1, N
                FJAC(I,J) = ZERO
                IF (I >= J-MU .AND. I <= J+ML) FJAC(I,J) = (WA1(I)-FVEC(I)) / H
             END DO
          END DO
       END DO
    END IF
    RETURN

    !     LAST CARD OF SUBROUTINE FDJAC1.

  END SUBROUTINE FDJAC1



END MODULE CSOLVE_MOD
