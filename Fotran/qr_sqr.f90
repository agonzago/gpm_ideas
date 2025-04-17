subroutine QR_sqr(n,A,Q,R,info1,info2)
  implicit none
  integer, intent(in):: n
  double precision, intent(in), dimension(n,n) :: A
  double precision, intent(out), dimension(n,n) :: Q,R
  integer, intent(out):: info1, info2
  double precision, allocatable :: TEMPWORK(:), WORK(:)
  double precision, dimension(n) :: TAU
  integer:: lwork,m
  INTEGER,DIMENSION(N)::JPVT
  JPVT=INT(0)
  allocate(tempwork(1))
  !CALL DGEQRF( n, n, A, n, TAU, TEMPWORK, -1, INFO1)
  M=INT(N)
  CALL DGEQP3( M, M, A, N, JPVT, TAU, TEMPWORK, -1, INFO1)
  LWORK=INT(TEMPWORK(1))
  allocate(WORK(LWORK))
!  CALL DGEQRF( n, n, A, n, TAU, WORK, LWORK, INFO1)
  JPVT=INT(0)
  CALL DGEQP3( M, M, A, M, JPVT, TAU, WORK, LWORK, INFO1)
  deallocate(work)
  deallocate(tempwork)
  allocate(tempwork(1))
  R=A
  CALL DORGQR( M, M, M, A, M, TAU, TEMPWORK, -1, INFO2 )
  LWORK=INT(TEMPWORK(1))
  allocate(WORK(LWORK))
  CALL DORGQR( M, M, M, A, M, TAU, WORK, LWORK, INFO2 )
  deallocate(tempwork)
  allocate(tempwork(1))  
  Q=A
end subroutine
