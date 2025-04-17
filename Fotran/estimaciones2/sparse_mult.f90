subroutine sparse_mult(T,A,B,ma,na,mb,nb,eps,C)
  integer, intent(in):: ma,na,mb,nb,T
  double precision, intent(in):: A(ma,na),B(mb,nb),eps
  double precision, intent(out):: C(ma,nb)
  double precision:: ra(int(ma*na)),rb(int(mb*nb)),rc(int(ma*nb)),rao(int(ma*na))
  integer:: ja(ma*na),jb(mb*nb),jc(ma*nb),ia(ma+1),ib(mb+1),ic(ma+1),jao(int(ma*na)),iao(na+1)
  integer, dimension(int(nb)):: wi
  integer:: nnza,nnzb,ierr

 ! call csr(a,ra,ja,ia,int(ma),int(na),nnza,eps)
 ! call csr(b,rb,jb,ib,int(mb),int(nb),nnzb,eps)

 call dnscsr(int(ma),int(na), int(ma*na), a, int(ma), ra, ja, ia, ierr )
 call dnscsr(int(mb),int(nb), int(mb*nb), b, int(mb), rb, jb, ib, ierr )
!print *,ma,na
!print *,nnza
!print *,ia(ma+1)
!print *,mb,nb
!print *,nnzb
!print *,ib(mb+1)
!print *,eps
!print *,rb(1:nnzb)
 if (T .eq. 1) then
  call csrcsc2 (int(ma), int(na), int(1),int(1),ra,ja,ia,rao,jao,iao)
  call amub(int(na), int(nb), int(1), rao, jao, iao, rb, jb, ib, rc, jc, ic, int(na*nb), wi, ierr)
  call csrdns(int(na), int(nb), rc, jc, ic, c, INT(na), ierr )
 else
  call amub(int(ma), int(nb), int(1), ra, ja, ia, rb, jb, ib, rc, jc, ic, int(ma*nb), wi, ierr)
  call csrdns(int(ma), int(nb), rc, jc, ic, c, INT(ma), ierr )
 endif
 !call amubb(int(ma), int(na), nnza, nnzb, int(1), ra(1:nnza), ja(1:nnza), ia, &
 !           rb(1:nnzb), jb(1:nnzb), ib, rc, jc, ic, int(ma*nb), ierr)
 !call amubb(int(ma), int(na), int(1), ra, ja, ia, &
 !           rb, jb, ib, rc, jc, ic, int(ma*nb), ierr)
!Print *,'hasta aqui si'
  !call csrdns(int(ma), int(nb), rc(1:ic(ma+1)-1), jc(1:ic(ma+1)-1), ic, c, INT(ma), ierr )
 
end subroutine
