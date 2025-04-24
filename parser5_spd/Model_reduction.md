# Initial model

    // Aggregate demand
    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*RR_GAP(+1) + RES_L_GDP_GAP;
    // Core Inflation
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    // Monetary policy reaction function
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS;
    RR_GAP = RS - DLA_CPI(+1);
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI;
    RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) + SHK_RS;

# Identification of equations/variables

1.  Dynamic: A variable that contains leads and lags of its own
    variable.

2.  Static: Equations/variables that at time t are defined purely by
    other variables at time t, or the expected value of other variables.
    The variable itself does not appear with a lead or lag in its own
    definition.

#### identification 

    L_GDP_GAP -> Dynamic (depends on L_GDP_GAP(+1) and L_GDP_GAP(-1));
    DLA_CPI -> Dynamic (depends on DLA_CPI(-1), DLA_CPI(+1));
    RS -> (Depends on  RS(-1) and DLA_CPI(+1), DLA_CPI(+3));
    RR_GAP -> Static (at time t depends on RS and the expected value of DLA_CPI, that are known at time t)
    RES_L_GDP_GAP(+1) ->  dynamic (exogenous state), depends on RES_L_GDP_GAP and SHK_L_GDP_GAP;
    RES_DLA_CPI(+1) ->  dynamic (exogenous state), depends on RES_DLA_CPI and SHK_L_GDP_GAP;
    RES_RS(+1) ->  dynamic (exogenous state), depends on RES_RS(-1) and SHK_L_GDP_GAP;

# Eliminate Static Equations

    // Aggregate demand
    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*(RS(+1) - DLA_CPI(+2)) + RES_L_GDP_GAP;
    // Core Inflation
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    // Monetary policy reaction function
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS;
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI;
    RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS + SHK_RS;

# Identify lead and lag structure. 

Variables in the system

    L_GDP_GAP enters in t, t+1, t-1
    DLA_CPI  enters in t, t+1, t+3, t-1
    RS: enters  t, t-1, t+1
    RES_L_GDP_GAP: enters in t, t-1
    RES_DLA_CPI: enters in t+1, t-1
    RES_RS: enters in t,t-1, t-2

    The system should be written for variables in t+1, t and t-1. So variables with longer leads and lags should be expressed through auxiliary variables.

    aux_DLA_CPI_lead(t) = DLA_CPI(t+1)
    aux_DLA_CPI_lead2(t) = aux_DLA_CPI_lead(t+1)
    aux_DLA_CPI_lead3(t) = aux_DLA_CPI_lead2(t+1) 

    Similar longer lags (longer than (-1)) should also imply auxiliary variables. This will imply 
    RES_RS_lag = RES_RS(-1)
    RES_RS_lag2 = RES_RS_lag(-1)

# Rewrite the system of equations in terms of the auxiliary variables


    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*(RS(+1) - aux_DLA_CPI_lead2(t)) + RES_L_GDP_GAP;
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*aux_DLA_CPI_lead3(t) + g3*L_GDP_GAP) + RES_RS;
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI + SHK_DLA_CPI;
    RES_RS = rho_rs*RES_RS(-1) + rho_rs2*aux_RES_RS_lag(-1) + SHK_RS;
    aux_DLA_CPI_lead(t) = DLA_CPI(t+1)
    aux_DLA_CPI_lead2(t) = aux_DLA_CPI_lead(t+1)
    aux_DLA_CPI_lead3(t) = aux_DLA_CPI_lead(t+1)
    aux_RES_RS_lag = RES_RS(-1)

# Reduce the system again.


    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*(RS(+1) - aux_DLA_CPI_lead(t+1)) + RES_L_GDP_GAP;
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*aux_DLA_CPI_lead2(t+1) + g3*L_GDP_GAP) + RES_RS;
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI;
    RES_RS  = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) + SHK_RS;
    aux_DLA_CPI_lead(t) = DLA_CPI(t+1)
    aux_DLA_CPI_lead2(t) = aux_DLA_CPI_lead(t+1)
    aux_RES_RS_lag = RES_RS(-1)

    10 equations on 10 variables

    List of contempotaneous variables 
    L_GDP_GAP
    DLA_CPI
    RS
    RES_L_GDP_GAP
    RES_DLA_CPI
    RES_RS
    aux_DLA_CPI_lead
    aux_DLA_CPI_lead2
    aux_DLA_CPI_lead3
    aux_RES_RS_lag

    List of lagged variables 
    L_GDP_GAP_m1
    DLA_CPI_m1
    RS_m1
    RES_L_GDP_GAP_m1
    RES_DLA_CPI_m1
    RES_RS_m1
    aux_DLA_CPI_lead_m1
    aux_DLA_CPI_lead2_m1
    aux_DLA_CPI_lead3_m1
    aux_RES_RS_lag_m1

    List of forward variables 
    L_GDP_GAP_p1
    DLA_CPI_p1
    RS_p1
    RES_L_GDP_GAP_p1
    RES_DLA_CPI_p1
    RES_RS_p1
    aux_DLA_CPI_lead_p1
    aux_DLA_CPI_lead2_p1
    aux_DLA_CPI_lead3_p1
    aux_RES_RS_lag_p1

    Final set of equations:
    0 = (1-b1)*L_GDP_GAP_p1 + b1*L_GDP_GAP_m1 - b4*(RS_p1 - aux_DLA_CPI_lead2) + RES_L_GDP_GAP -L_GDP_GAP;
    0 = a1*DLA_CPI_m1 + (1-a1)*DLA_CPI_p1 + a2*L_GDP_GAP + RES_DLA_CPI -DLA_CPI;
    0 = g1*RS_m1 + (1-g1)*(DLA_CPI_p1 + g2*aux_DLA_CPI_lead3 + g3*L_GDP_GAP) + RES_RS -RS ;
    0 = RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;
    0 = rho_DLA_CPI*RES_DLA_CPI_m1 + SHK_DLA_CPI - RES_DLA_CPI;
    0 = rho_rs*RES_RS_m1 + rho_rs2*aux_RES_RS_lag_m1 + SHK_RS - RES_RS;
    0 = DLA_CPI_p1 - aux_DLA_CPI_lead
    0 = aux_DLA_CPI_lead_p1 - aux_DLA_CPI_lead2
    0 = aux_DLA_CPI_lead2_p1 - aux_DLA_CPI_lead3
    0 = RES_RS_m1 - aux_RES_RS_lag 
