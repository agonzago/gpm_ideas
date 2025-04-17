program drawvector
  double precision dimension(3800000):draw
  CHARACTER STRING*10

double precision dimension(100000): draw1,draw2,draw3,draw4,draw5,draw6,draw7,draw8,draw9,draw10,&
                                    &draw11,draw12,draw13,draw14,draw15,draw16,draw17,draw18,draw19,&
                                    &draw20,draw21,draw22,draw23,draw24,draw25,draw26,draw27,draw28,&
draw29,
draw30,
draw31,
draw32,
draw33,
draw34,
draw35,
draw36,
draw37,
draw38,

open(39, FILE="parestim3.txt", status='OLD')
  READ(1, *) (xMIN(J), J=1, estimsize)
  CLOSE(39)


  OPEN (1,FILE="draw1.txt" )
  DO J = (I-1)*100000+1, 100000*I
        write (1, '(ES14.7)'), draw(J) 
     END DO
  END DO
  CLOSE(1)

