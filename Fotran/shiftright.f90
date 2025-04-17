subroutine shiftright(x,rx,cx,n)
  implicit none
  integer, intent(in):: rx,cx,n
  double precision, intent(inout), dimension(rx,cx):: x
  
  double precision, dimension(rx,cx):: y
  integer, dimension(cx-n):: left, right
  integer:: i,j
  left =(/(i,i=1,cx-n)/)
  right =(/(j,j=n+1,cx)/) 
  y=0.0
  !print *, left
  !print *, right
  y(1:rx,right)=x(1:rx,left)
  x=y
end subroutine

