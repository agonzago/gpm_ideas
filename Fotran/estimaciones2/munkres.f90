function Munkresfx(x, a, b)
  implicit none 
  INTEGER, PARAMETER  :: dp = SELECTED_REAL_KIND(12, 60)
  REAL (dp) Munkresfx  
  REAL (dp), intent(in) :: x, a, b
  REAL (dp) , parameter :: rho = 0.01_dp
  
  Munkresfx = 0.5*(a+b)+ (0.5*(b-a))*(2.0*rho*x/(1+sqrt((1+4.0*rho*rho*x*x))))
  !Munkresfx = x
end function Munkresfx

function MunkresfxInv(x, a, b)
  implicit none 
  INTEGER, PARAMETER  :: dp = SELECTED_REAL_KIND(12, 60)
  REAL (dp) MunkresfxInv 
  REAL (dp), intent(in) :: x, a, b
  REAL (dp)  kte
  REAL (dp) , parameter :: rho = 0.01_dp
   kte = (((2*x)/(b-a))- ((a + b)/(b-a)))
   MunkresfxInv = (kte/( 1.0 - (kte)**2.0 ))/rho
  !MunkresfxInv = x
  
end function MunkresfxInv


