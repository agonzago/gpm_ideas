# Initial model

var

// Main variables

L_GDP_GAP // Output Gap

DLA_CPI // QoQ Core Inflation

RS // MP Rate

RR_GAP // Real Interest Rate Gap

RES_L_GDP_GAP // exogenous states

RES_DLA_CPI

RES_RS

;

varexo

SHK_L_GDP_GAP // Output gap shock

SHK_DLA_CPI

SHK_RS // Foreign interest rate shock

;

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

# Clean Equations


    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*(RR_GAP(+1)) + RES_L_GDP_GAP;
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    RR_GAP = RS - DLA_CPI(+1)  
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
    RES_DLA_CPI: enters in t, t-1
    RES_RS: enters in t,t-1, t-2

    The system should be written for variables in t+1, t and t-1. So variables with longer leads and lags should be expressed through auxiliary variables.

    aux_DLA_CPI_lead(t) = DLA_CPI(t+1)
    aux_DLA_CPI_lead2(t) = aux_DLA_CPI_lead(t+1)
    aux_DLA_CPI_lead3(t) = aux_DLA_CPI_lead2(t+1) 

    Similar longer lags (longer than (-1)) should also imply auxiliary variables. This will imply 
    RES_RS_lag = RES_RS(-1)
    RES_RS_lag2 = RES_RS_lag(-1)

# Rewrite the system of equations in terms of the auxiliary variables


    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*(RR_GAP(+1)) + RES_L_GDP_GAP;
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*aux_DLA_CPI_lead(+1) + g3*L_GDP_GAP) + RES_RS;
    RR_GAP = RS - DLA_CPI(+1);
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI + SHK_DLA_CPI;
    RES_RS = rho_rs*RES_RS(-1) + rho_rs2*aux_RES_RS_lag(-1) + SHK_RS;
    aux_DLA_CPI_lead = DLA_CPI(+1)
    aux_DLA_CPI_lead2 = aux_DLA_CPI_lead(+1)
    aux_RES_RS_lag = RES_RS(-1)

# Replace by \_mk and pk.


    L_GDP_GAP = (1-b1)*L_GDP_GAP_p1 + b1*L_GDP_GAP_m1 - b4*(RR_GAP_p1) + RES_L_GDP_GAP;
    DLA_CPI = a1*DLA_CPI_m1 + (1-a1)*DLA_CPI_p1 + a2*L_GDP_GAP + RES_DLA_CPI;
    RS = g1*RS_m1 + (1-g1)*(DLA_CPI_p1 + g2*aux_DLA_CPI_lead_p1 + g3*L_GDP_GAP) + RES_RS;
    RR_GAP = RS - DLA_CPI_p1;
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI + SHK_DLA_CPI;
    RES_RS = rho_rs*RES_RS_m1 + rho_rs2*aux_RES_RS_lag_m1 + SHK_RS;
    aux_DLA_CPI_lead = DLA_CPI_p1
    aux_DLA_CPI_lead2 = aux_DLA_CPI_lead_p1
    aux_RES_RS_lag = RES_RS_m1

    10 equations on 10 variables

    List of contempotaneous variables 
    L_GDP_GAP
    DLA_CPI
    RS
    RR_GAP
    RES_L_GDP_GAP
    RES_DLA_CPI
    RES_RS
    aux_DLA_CPI_lead
    aux_DLA_CPI_lead2
    aux_RES_RS_lag

    The order of the variables should be
    forward_looking = 
    L_GDP_GA
    DLA_CPI
    RS
    aux_DLA_CPI_lead
    aux_DLA_CPI_lead2


    backward_looking_list
    RES_L_GDP_GAP
    RES_DLA_CPI
    RES_RS
    aux_RES_RS_lag

    exo_var_list 
    SHK_L_GDP_GAP
    SHK_DLA_CPI
    SHK_RS 

# How to clasify variables 

model;

y = a\*y(-1) + b\*y(+2) - c\*w(+1) + zy;

x = alpha\*x(+1) + beta\*y + zx;

w = x(+1) - zw;

zy = rhozy\*zy(-1) + shk_zy;

zx = rhozx\*zx(-1) + shk_zx;

zw = rhozw\*zw(-1) + rhozw\*zw(-2) + shk_zw;

end;

parsed model

0 = a\*y_m1 + b\*aux_y_lead_p1 - c\*w_p1 + zy-y;

0 = alpha\*x_1 + beta\*y + zx -x;

0 = x_p1 - zw-w;

0 = rhozy\*zy_m1 + shk_zy -zy;

0 = rhozx\*zx_m1 + shk_zx -zx;

0 = rhozw\*zw_m1 + rhozw1\*aux_zw_lag_m1 + shk_zw-zw;

0 = -aux_zw_lag + zw_m1;

0 = -aux_y_lead + y_p1;

Variables initial order y,w, x, zy, zx, zw, aux_zw_lag, aux_y_lead,

$$A=\left[\begin{array}{cccccccc}
b & -c & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & \alpha & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{array}\right];B=\left[\begin{array}{cccccccc}
-1 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
\beta & -1 & 0 & 0 & 1 & 0 & 0 & 0\\
0 & -1 & 0 & 0 & 0 & -1 & 0 & 0\\
0 & 0 & 0 & -1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & -1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & -1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & -1 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
\end{array}\right];C=\left[\begin{array}{cccccccc}
a & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & \rho_{zy} & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & \rho_{zx} & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & \rho_{zw} & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{array}\right];$$

$$D=\left[\begin{array}{ccc}
0 & 0 & 0\\
0 & 0 & 0\\
0 & 0 & 0\\
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
0 & 0 & 0\\
0 & 0 & 0
\end{array}\right]$$

How to order the variables:

**Backward looking exogenous variables:**

All columns in A for these variables are zero, and

All columns of other variables on these variables are zero in B and C.

**Backward looking endogenous variables:**

All columns in A for those variables are zero.

**Forward looking variables:**

All columns in C for those variables are zero.

**Static variables:**

All columns in A, B for those variables are zero.

The idea is then

1 ) Eliminate static variables by substituting out them:

2\) On the remaining variables order them as: backward looking exogenous
states zy, zx, zw follow by forward/backward endogenous variables y,w,
x.

3\) Order equations for the backward exogenous variables first. In this
case, we don;t have the other equations variables in any way. Just move
the equations for the exogenous variables first.

# Jacoobians of the ordered equations and variables

$x_{t}=\left(mix_{t},back_{t}\right)$,$eps_{t}$=exo_var_list

$$A=\frac{\partial F}{\partial x_{t+1}}$$

$$B=\frac{\partial F}{\partial x_{t}}$$

$$C=\frac{\partial F}{\partial x_{t-1}}$$

$$D=\frac{\partial F}{\partial\epsilon_{t}}$$

# Build the complete state space 

## elements

The dyn files has sections

    varexo_trends
    SHK_L_GDP_TREND,
    SHK_G_TREND,
    SHP_PI_TREND,
    SHK_RS_TREND,
    SHK_RR_TREND
    ;
    trends_vars
    L_GDP_TREND,
    PI_TREND,
    RS_TREND,
    RR_TREND,
    G_TREND
    ;
    trend_model;
    L_GDP_TREND = L_GDP_TREND(-1) + G_TREND(-1) + SHK_L_GDP_TREND;
    G_TREND = G_TREND(-1) + SHK_G_TREND;
    PI_TREND = PI_TREND(-1) + SHP_PI_TREND;
    RS_TREND = RR_TREND + PI_TREND;
    RR_TREND = RR_TREND(-1) + SHK_RR_TREND;
    end;
    varobs
    L_GDP_OBS
    DLA_CPI_OBS
    PI_TREND_OBS
    RS_OBS
    ;
    measument_equations;
    L_GDP_OBS = L_GDP_TREND + L_GDP_GAP;
    DLA_CPI_OBS = DLA_CPI + PI_TREND;
    PI_TREND_OBS = PI_TREND;
    RS_OBS = RS_TREND + RS;
    end;

In this section you have defined: The list of trends in trends_vars, the
trend_model, and the varexo_trends that are the list of shocks to the
stochastic trends. You also have varobs with the list of observable
variables and measument_equations; end; that links how observable
variables are related with the statee vector.

## Building the state space

The SPD algorithm gives you the solution

$$y_{t}=Py_{t-1}+Q\epsilon_{t}$$

$$\epsilon_{t}\sim N(0,\Sigma)$$ this soluction should be augmented to
include the trends and the link with the observable variables.

Let's define an augmented state vector
$y_{t}^{a}=$$\left(y_{t},trends_{t}\right).$ The the augmented
state-space would be

$$y_{t}^{a}=\left[\begin{array}{cc}
P & 0\\
0 & P_{trends}
\end{array}\right]y_{t-1}^{a}+\left[\begin{array}{cc}
Q & 0\\
0 & Q_{trends}
\end{array}\right]\epsilon_{t}^{a}$$ shocks $\epsilon_{t}^{a}$ include
shocks to trends.

The measurement equation would be

$$y_{t}^{obs}=\Omega y_{t}^{a}+H\xi_{t}$$ $$\xi_{t}\sim N(0,I)$$ and H
is a diagonal matrix with measurement errors standard errors. $\Omega$
is the measurement matrix/observation matrix an relects the
relationships presented in measument_equations.

## How to build $Q_{trends}$, [$\Sigma$ and P_trends,$\Omega$?]{.medium} 

If $trend_{t}$is a vector with L_GDP_TREND,
PI_TREND,RS_TREND,RR_TREND,G_TREND and then the parsed

    L_GDP_TREND = L_GDP_TREND_m1 + G_TREND_m1 + SHK_L_GDP_TREND;
    G_TREND = G_TREND_m1 + SHK_G_TREND;
    PI_TREND = PI_TREND_m1 + SHP_PI_TREND;
    RS_TREND = RR_TREND + PI_TREND;
    RR_TREND = RR_TREND_m1 + SHK_RR_TREND;

    the P_trend is the derivarive of the RHS of each equation with respect to _m coefficients

$$trend_{t}=P_{trend}trend_{t-1}=\left[\begin{array}{cccccc}
1 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 1 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 1 & 0\\
0 & 0 & 0 & 1 & 0 & 0
\end{array}\right]trend_{t-1}$$

$$Q_{trends}=\text{diag}\left(\sigma_{trend1},\dots,\sigma_{trendx}\right)$$

$$\Sigma=\text{diag}\left(\sigma_{sh1},\dots,\sigma_{shx}\right)$$

- P_trend: Could depend on parameters also.

- The vector of parameters (full) should include sigma's for shocks
  (place holders), parameters for trends.

SImilarly $\Omega$ is the derivative (jacobbian) of the observation
measument_equations with respect to $y_{t}^{obs}$
