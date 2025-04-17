SUBROUTINE HYBRIDSA(FUNOBJ,NP,POINTS)
  USE HYBRID_CONST_MOD
  IMPLICIT NONE    
  INTEGER, INTENT(OUT) ::  NP
  DOUBLE PRECISION, INTENT(OUT) ,DIMENSION(NEQ, NMC+1) :: POINTS
  
  
  DOUBLE PRECISION, DIMENSION(NEQ) :: PARAM, EPSI, INBL, INBU , DPARAM  
  DOUBLE PRECISION BST,DIFF,NEWOBJ,OBJF, P, U, T
  INTEGER I,I1,I2,INFO

  EXTERNAL FUNIDRAW, MINVECT, MAXVECT,HIBPRIN
  EXTERNAL FUNOBJ, RNDSTART,RNDEND
  INTEGER ACCEPT, ISCAPE
  LOGICAL FLAG

  BST = 1E16
  POINTS = 0.0
  !     INITIALISE THE TEMPERATURE  
  T = T0
  EPSI = (BU-BL)/NSTEP

  FLAG = .TRUE.
  ISCAPE = 0
  
 ! do I1=1,nmc
 ! CALL FUNIDRAW(BL, BU, NEQ, PARAM)
 ! POINTS(1:NEQ,I1)=PARAM
 ! ENDDO
 ! np=NMC
 ! RETURN

  DO WHILE (FLAG .EQV. .TRUE. )
     CALL FUNIDRAW(BL, BU, NEQ, PARAM)     
     CALL FUNOBJ(PARAM, NEQ, OBJF)     
     FLAG = ISNAN(OBJF)          
  END DO
! PRINT *, "OBJ",OBJF
! objf = HUGE(DBLE(1.0))
  FLAG = .TRUE.
!  PRINT *, OBJF
  
  DO I1 = 1,NT         
     NP = 1
     ACCEPT  = 0
     
     DO  I2 = 1,NMC
        
        POINTS(1:NEQ,  (NP+1)) = PARAM
        
        CALL MAXVECT((POINTS(1:NEQ,NP+1) - EPSI(1:NEQ)),BL,NEQ, INBL)
        CALL MINVECT((POINTS(1:NEQ,NP+1) +EPSI(1:NEQ)),BU,NEQ, INBU)
        
        !DPARAM = PARAM
        DO WHILE (FLAG .EQV. .TRUE. )           
           CALL FUNIDRAW(INBL, INBU, NEQ, PARAM)            
           CALL FUNOBJ(PARAM,NEQ,NEWOBJ)                         
         !  PRINT *, "NEWOBJ",NEWOBJ
           FLAG = ISNAN(NEWOBJ)
           !PRINT *,FLAG
        END DO
        !PRINT *, NEWOBJ
        !PARAM = DPARAM        
        FLAG = .TRUE.
        
        DIFF = (NEWOBJ-OBJF)
      ! print *,"DIFF:",DIFF
        IF (DIFF.LT.0.0) THEN
           ACCEPT = ACCEPT +1
           NP = NP + 1               
           POINTS(1:NEQ, NP) = PARAM
           
           OBJF = NEWOBJ
           IF (NEWOBJ.LT.BST) THEN
              BST = NEWOBJ
              POINTS(1:NEQ, 1) = PARAM
           END IF
           
        ELSE
           
           P = EXP(-DIFF/T)               
           CALL FUNIDRAW(DBLE(0.0), DBLE(1.0), INT(1.0), U )
           IF (U.LT.P) THEN
              NP = NP + 1
              ACCEPT = ACCEPT +1
              OBJF = NEWOBJ
              POINTS(1:NEQ, NP) = PARAM
           ELSE
              PARAM(1:NEQ) = POINTS(1:NEQ,NP+1)
           END IF
        END IF
        
     ENDDO
     IF ( MOD(I1, IPRINT ) .EQ. 0 ) THEN 
        CALL HIBPRIN(T, I1, DBLE(ACCEPT)/DBLE(NMC), BST, OBJF, NP)
     END IF
     
     T = T*RHO     
  
     
  END DO
  

  RETURN
END SUBROUTINE HYBRIDSA
   
   
SUBROUTINE MAXVECT(X, Y,N, NEWX)
  IMPLICIT NONE
  INTEGER N, I
  DOUBLE PRECISION X(N,1), Y(N,1), NEWX(N,1)      

  DO I = 1, N         
     IF (X(I,1).LE.Y(I,1) ) THEN
        NEWX(I,1) = Y(I,1)
     ELSE
        NEWX(I,1) = X(I,1)
     END IF

  END DO
  RETURN
END SUBROUTINE MAXVECT

SUBROUTINE MINVECT(X, Y,N, NEWX)
  IMPLICIT NONE
  INTEGER N, I
  DOUBLE PRECISION X(N,1), Y(N,1), NEWX(N,1)      

  DO I = 1, N         
     IF (X(I,1).LE.Y(I,1) ) THEN
        NEWX(I,1) = X(I,1)
     ELSE
        NEWX(I,1) = Y(I,1)
     END IF
  END DO
  RETURN
END SUBROUTINE MINVECT


SUBROUTINE FUNIDRAW(LB, UB, K, DRAWS )
  !USE RANDOM_UNIFORM
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(12, 60) 
  INTEGER K,I
  real(dp):: LB(K), UB(K), DRAWS(K)
  real(dp):: tmp
  !EXTERNAL UNIDRAW
!  CALL RANDOM_NUMBER()

  DO I =1, K
     CALL RANDOM_NUMBER(tmp)
     !tmp=
     DRAWS(I) = LB(I)  + (UB(I) - LB(I))*tmp !tmp RANDOM_UNIFORM(REAL(LB(I)), REAL(UB(I)))
  ENDDO

END SUBROUTINE FUNIDRAW


SUBROUTINE HIBPRIN(TEMP, TEMSTEP, ACCEPT, BEST, CURRENT, NP )  
  IMPLICIT NONE 
  
  INTEGER, INTENT(IN) :: TEMSTEP, NP 
  DOUBLE PRECISION, INTENT(IN) :: TEMP, BEST, CURRENT , ACCEPT
  PRINT * 
  PRINT '(A16, I10, A11, F10.5)', "TEMPERATURE STEP", TEMSTEP,"TEMPERATURE" ,TEMP 
  PRINT '(A14,F10.5)', "ACCEPTACE RATE", ACCEPT 
  PRINT '(A24, I5)', "NUMBER OF INITIAL VALUES", NP 
  PRINT '(A26, F14.7)', "CURRENT OBJECTIVE FUNCTION", CURRENT 
  PRINT '(A13, F14.7)', "BEST FUNCTION" ,BEST 
  PRINT *
  
END SUBROUTINE
  

