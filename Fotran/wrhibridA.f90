SUBROUTINE WRHYBRIDSA(NPARAM, NMC,NT,T0,RHO,BL,BU,NP,POINTS,NSTEP,IPRINT)
  
  IMPLICIT NONE 
  
  INTEGER, INTENT(IN) :: NPARAM, NMC,NT, NSTEP,IPRINT
  INTEGER, INTENT(OUT) :: NP 
  DOUBLE PRECISION, INTENT(IN):: T0,RHO
  DOUBLE PRECISION, INTENT(IN), DIMENSION(NPARAM) :: BL,BU
  DOUBLE PRECISION, INTENT(OUT), DIMENSION(NPARAM, NMC+1) :: POINTS
  
  EXTERNAL FCNSA, HYBRIDSA
  
  CALL HYBRIDSA(FCNSA,NPARAM, NMC,NT,T0,RHO,BL,BU,NP, POINTS,NSTEP,IPRINT)
  

END SUBROUTINE WRHYBRIDSA
