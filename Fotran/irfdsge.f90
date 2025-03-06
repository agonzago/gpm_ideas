SUBROUTINE IRFDGSE(IRF, F, P,stderr, NUM_IRF, N, K, var)
  IMPLICIT NONE 
  INTEGER NUM_IRF, N, K, VAR
  DOUBLE PRECISION STDERR      
  DOUBLE PRECISION IRF(NUM_IRF, N), MX(N,K), X(K,1),X0(K,1) 
  DOUBLE PRECISION XNULL(N, 1)
  DOUBLE PRECISION P(K,K), F(N-K, K)
  INTEGER I, J, lb


  !EXTERNAL DGEMM
  !  C$$$      print *, N, K
  MX =0
  MX(1:(N-K),1:K) = F
  lb = (N-K+1)
  DO I = lb,N 
     MX(I,I+1-lb) = 1.0
  ENDDO

  IRF = 0.0
  X = 0
  X(var,1) = STDERR  
  ! C     MX (N x K)
  ! C     X (K X 1)
  DO I = 1, NUM_IRF
     CALL DGEMM('N','N',N,int(1.0),K, DBLE(1.0),MX, N,X,K,DBLE(0.0), XNULL, N)
     
     CALL DGEMM('N','N',K,int(1.0),K, DBLE(1.0),P,K, X,K,DBLE(0.0), X0,K)     
     X = X0         
     IRF(I:I, 1:N) = TRANSPOSE(XNULL)         
  ENDDO


END SUBROUTINE IRFDGSE





