!program uno
!  implicit none
  
!  integer, parameter :: N = 2
!  real(8) A(N,N)
!  integer info
!  real(8) rcond, anorm
      
!  A = reshape((/1,2, 3,5/), (/2,2/))
!  CALL RCOND_SUB(A, N, ANORM, RCOND, 'A', INFO)
!  print *, anorm
!  print *, rcond
  !print *, A
   

!end program uno

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
