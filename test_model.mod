
var Y C K;
varexo RES_RS SHK_Y;
parameters rho_rs rho_rs2 beta delta;

beta = 0.99;
delta = 0.025;
rho_rs = 0.7;
rho_rs2 = 0.2;

model;
Y = K(-1)^alpha * C;
Y(+1) = beta*(Y + RES_RS);
RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) + SHK_RS;
end;
