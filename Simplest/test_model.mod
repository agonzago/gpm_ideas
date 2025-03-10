

var
    
    L_GDP_GAP             
    DLA_CPI         
    RS               
    RR_GAP                  
    RES_L_GDP_GAP      
    RES_DLA_CPI
    RES_RS
;

varexo
    SHK_L_GDP_GAP     
    SHK_DLA_CPI
    SHK_RS               
;   

parameters b1, b4, a1, a2, g1, g2, g3, rho_DLA_CPI, rho_L_GDP_GAP, rho_rs, rho_rs2;

    
    b1 = 0.7;          
    b4 = 0.7;          
    a1 = 0.5;          
    a2 = 0.1;          
    g1 = 0.7;         
    g2 = 0.3;          
    g3 = 0.25;         
    rho_L_GDP_GAP  =0.1;
    rho_DLA_CPI   =0.2; 
    rho_rs      =0.3;
    rho_rs2      =0.4;
model;
        
    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*RR_GAP(+1) + RES_L_GDP_GAP;
    RES_L_GDP_GAP(+1) = rho_L_GDP_GAP*RES_L_GDP_GAP + SHK_L_GDP_GAP;

    
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    RES_DLA_CPI(+1) = rho_DLA_CPI*RES_DLA_CPI + SHK_DLA_CPI;

    
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS;
    RES_RS(+1) = rho_rs*RES_RS + rho_rs2*RES_RS(-1) + SHK_RS;

    RR_GAP = RS - DLA_CPI(+1);

end;

initval;
    L_GDP_GAP = 0;    
    DLA_CPI = 0;
    RS = 0;
    RR_GAP = 0;
    RES_L_GDP_GAP =0;
    RES_DLA_CPI =0;
    RES_RS =0;  
end;

steady;
check;

// Specify the shocks
shocks;
    var SHK_L_GDP_GAP = 1;    
end;

stoch_simul(order=1, irf=40) L_GDP_GAP DLA_CPI RS;

