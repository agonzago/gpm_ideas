subroutine find_diag(A,m,n,condn,ind_cero,size_ind)
  implicit none
  integer, intent(in):: m,n
  double precision, intent(in) :: condn
  double precision, intent(in), dimension(m,n) :: A
  integer, intent(out), dimension(m) :: ind_cero
  integer, intent(out):: size_ind
  integer:: i
  
  size_ind=0
  do i=1,m
     if (abs(A(i,i)) < condn) then
        size_ind=size_ind+1
        ind_cero(size_ind)=i
     end if
  end do
end subroutine
