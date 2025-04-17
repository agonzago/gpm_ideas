subroutine didentity(n,Id)
implicit none
integer, intent(in):: n
double precision, intent(out), dimension(n,n):: Id
integer:: i

Id=0
do i=1,n
Id(i,i)=1.0
end do

end subroutine
