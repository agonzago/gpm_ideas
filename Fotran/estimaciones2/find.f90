subroutine find(log_array,size_array,ind_array,size_ind)
  implicit none
  integer, intent(in):: size_array
  logical, intent(in), dimension(size_array) :: log_array
  integer, intent(out), dimension(size_array) :: ind_array
  integer, intent(out):: size_ind
  integer:: i
  
  size_ind=0
  do i=1,size_array
     if (log_array(i)) then
        size_ind=size_ind+1
        ind_array(size_ind)=i
     end if
  end do
end subroutine
