subroutine build_a_del(n,a,eig_mod,lgroots,lgroots_sz,zerocols,zerocols_sz,info)
  implicit none
  integer, intent(in):: n
  double precision, intent(inout),dimension(n,n):: a
  double precision, intent(in):: eig_mod
  integer, intent(out),dimension(n):: zerocols, lgroots
  integer,intent(out)::lgroots_sz,zerocols_sz, info
  integer:: i, lwork, zerocols_tmp,nn
  logical, dimension(n):: zerocols_log, noz
  double precision, allocatable :: TEMPWORK(:), WORK(:),WR(:),WI(:),VL(:,:),VR(:,:)
  logical, dimension(n)::lgroots_log
  double precision,dimension(n)::wr_o, wi_o
  double precision:: pru
  double precision, allocatable::copy(:,:)
!  Delete inessential lags and build index array js.  js indexes the
!  columns in the big transition matrix that correspond to the
!  essential lags in the model.  They are the columns of q that will
!  get the unstable left eigenvectors. 

!js       = (/(i_c, i_c=1,qcols)/)
!js =left
  nn=int(n)
  zerocols_log = (sum(abs(a),1) .eq. 0.0)
  zerocols_tmp=0
  CALL find(zerocols_log,nn,zerocols,zerocols_sz)
  do while (zerocols_sz .ne. zerocols_tmp)
      !if (zerocols_sz==zerocols_tmp) exit
      a(zerocols(1:zerocols_sz),:) =0.0
      zerocols_log = (sum(abs(a),1) .eq. 0.0)
      zerocols_tmp=zerocols_sz
      CALL find(zerocols_log,nn,zerocols,zerocols_sz)
  enddo
   noz=.not.zerocols_log
  CALL find(noz,nn,zerocols,zerocols_sz)
!    print *,'A=',A(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz))
 ! a(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz))=reshape((/0.0,-0.6,-4.0,-2.0,0.0,0.26,0.4,&
   !                        &0.2,0.0,0.6,3.0,2.0,0.0,-0.3,-2.0,-1.0/),(/4,4/))
  !print *,'A=',A(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz))
! Build matrix q
  allocate(tempwork(1))
  allocate(WR(zerocols_sz))
  allocate(WI(zerocols_sz))
  allocate(VL(zerocols_sz,zerocols_sz))
  allocate(VR(zerocols_sz,zerocols_sz))
  allocate(copy(zerocols_sz,zerocols_sz))
  copy=A(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz))
  CALL DGEEV( 'V', 'N',INT(zerocols_sz),copy, INT(zerocols_sz), WR, WI, VL, INT(zerocols_sz), &
       & VR,INT(zerocols_sz), TEMPWORK, int(-1), INFO )
!  CALL DGEEV( 'V', 'N',zerocols_sz,A(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz)), zerocols_sz, WR, WI, VL, zerocols_sz, &
!       & VR,zerocols_sz, TEMPWORK, int(-1), INFO )
  LWORK=INT(TEMPWORK(1))
  allocate(WORK(LWORK))
  CALL DGEEV( 'V', 'N',INT(zerocols_sz),copy, INT(zerocols_sz), WR, WI, VL, INT(zerocols_sz),&
     & VR,INT(zerocols_sz), WORK, LWORK, INFO )
  !CALL DGEEV( 'V', 'N',zerocols_sz,A(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz)), zerocols_sz, WR, WI, VL, zerocols_sz,&
  !   & VR,zerocols_sz, WORK, LWORK, INFO )
  A(zerocols(1:zerocols_sz),zerocols(1:zerocols_sz))=copy
  deallocate(copy)
  deallocate(tempwork)
  deallocate(work)
  lgroots_log(1:zerocols_sz)=SQRT(WR*WR+WI*WI)>eig_mod
  CALL find(lgroots_log(1:zerocols_sz),zerocols_sz,lgroots(1:zerocols_sz),lgroots_sz)
  wr_o(1:lgroots_sz)=WR(1:lgroots_sz)
  wi_o(1:lgroots_sz)=WI(1:lgroots_sz)
  i=1
  do 
     if (i>lgroots_sz) exit
     pru=abs(WI(lgroots(i)))
     if (pru .gt. 1e-10) then
        a(1:zerocols_sz,i)=VL(:,lgroots(i))+VL(:,lgroots(i)+1)
        a(1:zerocols_sz,i+1)=VL(:,lgroots(i))-VL(:,lgroots(i)+1)
        i=i+2
     else
        a(1:zerocols_sz,i)=VL(:,lgroots(i))
        i=i+1
     end if
  end do
end subroutine

