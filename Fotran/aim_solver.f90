! AIM algorithm - Anderson (2000) - Anderson & Moore (1986)
subroutine AIM(AA,BB,L,KK,lags,leads,F,P,aimcode)
  implicit none
  integer, intent(in):: L, kk, lags, leads
  !double precision, dimension(L,L*(lags+leads+1)) :: H
  double precision, intent(in), dimension(L,L) :: AA,BB
  double precision, dimension(L,L) :: B
  double precision, intent(out), dimension(KK,KK) :: P
  double precision, intent(out), dimension(L-KK,KK) :: F
  integer, intent(out)::aimcode
  double precision, dimension(L,L*(lags+leads+1)) :: hs,HS_COPY
  double precision, dimension(L,L) :: QQ, R
  integer :: bcols, nexact, qcols, qrows, nz, iq, zerorows_sz, nnumeric, info1, info2, info, i_c, js_sz, lgroots_sz, test,I,J,HCOLS
  double precision:: condn, eig_mod, RCOND,qnorm
  integer, dimension(L*(lags+leads)) :: left
  integer, dimension(L) :: right, zerorows, hrows
  logical, dimension(L) :: zerowslog
  double precision, dimension(L*(lags+leads),L*(lags+leads)) :: A
  double precision, dimension(L*(lags+leads-1),L*(lags+leads-1)) :: EYE 
  double precision, dimension(L*lags,L*(lags+leads))::q
  integer,dimension(L*(lags+leads))::lgroots,js
  integer,dimension(L*lags)::IPIVq
  integer,dimension(L*(lags+leads-1))::eyerows,eyecols
  double precision, allocatable:: copy(:,:)
  DOUBLE PRECISION,DIMENSION(L*LAGS)::THEMAX
  REAL DELAPSE, TIMEARRAY(2)
  eig_mod=1.0-(1.0e-6) !tolerance for the eigenvalues
  bcols=int(L*lags)
  nexact = int(0)
  qcols=int(L*(lags+leads))
  qrows=int(L*lags)
  nz=int(0)
  iq=int(0)
  aimcode=int(0)
  left   = (/(i_c,i_c=1,qcols)/)
  right  = (/(i_c,i_c=qcols+1,qcols+int(L))/)
  !hs=dble(H)
   HS(:,2*L+1:3*L)=AA  
  !H_P1(:,1:NUM_K)=0.0
   HS(:,2*L+1:2*L+KK)=0.0
  !H_0
   HS(:,L+1:2*L)=AA
  !H_0(:,NUM_K+1:NUM_N)=BMAT(:,NUM_K+1:NUM_N)
   HS(:,L+1+KK:L+L)=-BB(:,KK+1:L)
  !H_M1=B
   HS(:,1:L)=-BB
  !H_M1(:,NUM_K+1:NUM_N)=0.0
   HS(:,KK+1:L)=0.0   

  q=dble(0.0)
  condn=dble(1e-10)
  zerorows_sz=int(0)
  B=dble(0.0)
  HCOLS=int(L*(LAGS+LEADS+1))
! Rows with all its elelments equal to 0 in hs(i,right) 
   zerowslog = sum(abs( hs(:,right) ),2) .eq. 0
   CALL find(zerowslog,int(L),zerorows,int(zerorows_sz)) 
  do 
     if ((zerorows_sz .eq. 0) .or. (iq>qrows) ) exit
     q(iq+1:iq+zerorows_sz,:) = hs(zerorows(1:zerorows_sz),left)
     allocate(copy(zerorows_sz,HCOLS))
     copy=hs(zerorows(1:zerorows_sz),:)
     call shiftright(copy,int(zerorows_sz),HCOLS,int(L))
     hs(zerorows(1:zerorows_sz),:)=copy
     deallocate(copy)
     iq=iq+zerorows_sz
     nexact=nexact + zerorows_sz
     zerowslog = sum(abs( hs(:,right) ),2) .eq. 0
     CALL find(zerowslog,int(L),zerorows,zerorows_sz)
  end do
  if (iq>qrows) then
     print *,'Aim: too many exact shiftrights.'
     aimcode=-1
     return
  end if
  nnumeric = int(0);


  CALL QR_sqr(int(L),hs(:,right),QQ,R,info1,info2)
  if (info1<0 .or. info2<0) then
     print *,'Aim: QR decomposition can not be done.'
     aimcode=-1
     return
  end if

  CALL find_diag(R,int(L),int(L),condn,zerorows,zerorows_sz)
  nnumeric=int(0)
  do 
     if (zerorows_sz .eq. 0 .or. iq > qrows) exit
      condn=dble(1.0e-6)
      HS_COPY=HS
!      QQ=transpose(QQ)
!      HS=MATMUL(QQ,HS)
      CALL DGEMM('T','N',int(L),HCOLS,INT(L),DBLE(1.0),QQ,INT(L),HS_COPY,INT(L),DBLE(0.0),HS,INT(L))
      !call sparse_mult(int(1),QQ,HS_COPY,int(L),int(L),int(L),int(hcols),condn,HS)
     q(iq+1:iq+zerorows_sz,:)=hs(zerorows(1:zerorows_sz),left)
     allocate(copy(zerorows_sz,HCOLS))
     copy=hs(zerorows(1:zerorows_sz),:)
     CALL shiftright(copy,zerorows_sz,HCOLS,int(L))
     hs(zerorows(1:zerorows_sz),:)=copy
     deallocate(copy)
     iq = iq +zerorows_sz
     nnumeric = nnumeric + zerorows_sz;
     CALL QR_sqr(int(L),hs(:,right),QQ,R,info1,info2)
     if (info1<0 .or. info2<0) then
        print *,'Aim: QR decomposition can not be done.'
        aimcode=-1
        return
     end if
     CALL find_diag(R,int(L),int(L),condn,zerorows,zerorows_sz)
  end do
  if (iq>qrows) then
     print *,'Aim: too many numeric shiftrights.'
     aimcode=-1 
     return
  end if

  allocate(copy(int(L),qcols))
  copy=hs(:,left)
  CALL DGESV(int(L),QCOLS, -hs(:,right), int(L), IPIVq(1:L), copy, int(L), INFO )
  hs(:,left)=copy
  deallocate(copy)
    if (info .ne. 0) then
       print *,'Aim: inv(-hs(:,right))*hs(:,left) cannot be done.'
       aimcode=-1 !OJO
       return
    end if


! Build the big transition matrix.
  A=0.0
  if (qcols > L) then
     eyerows=(/(i_c, i_c=1,qcols-int(L))/)
     eyecols=(/(i_c, i_c=int(L)+1,qcols)/)
     CALL didentity(int(qcols-int(L)),EYE)
     A(eyerows,eyecols)=EYE
  end if

  hrows=(/(i_c, i_c=qcols-int(L)+1,qcols)/)
  A(hrows,:) = hs(:,left)

  if ((ISNAN(sum(A)))) then
     print *,'A is NAN or INF'
     aimcode=-1 
     return 
  end if 
 
 
 CALL build_a_del(qcols,A,eig_mod,lgroots,lgroots_sz,js,js_sz,info)
   if (info .ne. 0) then
     print *,'Matrix A can not be reduced'
     aimcode=-1 
     return 
  end if 
  
  if(iq < qrows) then
    q(iq+1:qrows,js(1:js_sz)) = transpose (A(1:js_sz,1:qrows-iq))
    !allocate(copy(int(qrows-iq),int(js_sz)))
    !call sparse_transpose(A(1:js_sz,1:qrows-iq),int(js_sz),int(qrows-iq),copy)
    !call sparse_transpose(A(1:js_sz,1:qrows-iq),int(js_sz),int(qrows-iq),q(iq+1:qrows,js(1:js_sz)))
    !print *,maxval(abs(copy-q(iq+1:qrows,js(1:js_sz))))
    !q(iq+1:qrows,js(1:js_sz))=copy
    !deallocate(copy)
  end if


  test=nexact+nnumeric+lgroots_sz;
  if (test > qrows) then
      aimcode = -1
      print *,'Aim: too many big roots.'
      return
  else if (test < qrows) then
      aimcode = -1
      print *,'Aim: too few big roots.'
      return
  end if

  CALL RCOND_SUB(q(:,qcols-qrows+1:qcols),qrows,qnorm,RCOND,'A',INFO)


  IF (RCOND > 1.0E-6) THEN
     !left = (/(i_c,i_c=1,qcols-qrows)/)
     !right = (/(i_c,i_c=qcols-qrows+1,qcols)/)
     allocate(copy(qrows,qcols-qrows))
     copy=q(:,1:qcols-qrows)
     CALL DGESV(qrows,qcols-qrows,-q(:,qcols-qrows+1:qcols),qrows,IPIVq,copy,qrows,info)
     q(:,1:qcols-qrows)=copy
     deallocate(copy)
       if (info .ne. 0) then
          print *,'Aim: inv(-q(:,right))*q(:,left) cannot be done.'
          aimcode=-1 !OJO
          return
       end if
     B=Q(1:L,1:L*lags)
  ELSE !rescale by dividing row by maximal qr element
     THEMAX=MAXVAL(ABS(q(:,qcols-qrows+1:qcols)),2)
     THEMAX=1.0/THEMAX
     !ONEOVER=0.0
     DO I=1,QROWS
        Q(I,:)=THEMAX(I)*Q(I,:)
     END DO
     CALL RCOND_SUB(q(:,qcols-qrows+1:qcols),qrows,qnorm,RCOND,'A',INFO)
     IF (RCOND > 1.0E-6) THEN
        allocate(copy(qrows,qcols-qrows))
        copy=q(:,1:qcols-qrows)
        CALL DGESV(qrows,qcols-qrows,-q(:,qcols-qrows+1:qcols),qrows,IPIVq,copy,qrows,info)
        q(:,1:qcols-qrows)=copy
        deallocate(copy)
             if (info .ne. 0) then
                print *,'Aim: inv(-q(:,right))*q(:,left) cannot be done.'
                aimcode=-1 !OJO
                return
             end if
        B=Q(1:L,1:L*lags)
     END IF
  END IF
  
  if ( (RCOND > 1.0E-6) .AND. aimcode .EQ. 0) THEN
   !   PRINT *,'Aim: unique solution.'
  elseif ((RCOND <= 1.0E-6) .AND. aimcode .EQ. 0) THEN
      aimcode =  -1
      PRINT *,'Aim: q(:,right) is singular.'
 ! elseif ((RCOND <= 1.0E-6) .AND. aimcode .EQ. 3) THEN
 !     aimcode = 35
 !     PRINT *,'Aim: too many big roots, and q(:,right) is singular.'
 ! elseif ((RCOND <= 1.0E-6) .AND. aimcode .EQ. 4) THEN
 !     aimcode = 45
 !     PRINT *,'Aim: too few big roots, and q(:,right) is singular.'
  end IF

 P=B(1:KK,1:KK)
 F=B(KK+1:L,1:KK)

end subroutine

  

