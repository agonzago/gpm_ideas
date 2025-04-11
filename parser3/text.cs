I wan't to create a parser from a dynare file that has the following structure, not model. So the parser has two be general
but sufficient to deal with the following model structure

/*
 * Smallest QPM in dynare
 */

var
    // Main variables
    L_GDP_GAP          // Output Gap
    DLA_CPI         // QoQ Core Inflation
    RS                 // MP Rate
    RR_GAP             // Real Interest Rate Gap
    RES_L_GDP_GAP      // exogenous states
    RES_DLA_CPI
    RES_RS
;

varexo
    SHK_L_GDP_GAP     // Output gap shock
    SHK_DLA_CPI
    SHK_RS            // Foreign interest rate shock
;

parameters b1, b4, a1, a2, g1, g2, g3, rho_DLA_CPI, rho_L_GDP_GAP, rho_rs, rho_rs2;

    // Parameters from readmodel.m
    b1 = 0.7;          // Output persistence
    b4 = 0.7;          // MCI weight
    a1 = 0.5;          // Inflation persistence
    a2 = 0.1;          // RMC passthrough
    g1 = 0.7;         // Interest rate smoothing
    g2 = 0.3;          // Inflation response
    g3 = 0.25;         // Output gap response
    rho_L_GDP_GAP  =0.75;
    rho_DLA_CPI   =0.75;
    rho_rs      =0.9;
    rho_rs2      =0.1;
model;
    // Aggregate demand
    L_GDP_GAP = (1-b1)*L_GDP_GAP(+1) + b1*L_GDP_GAP(-1) - b4*RR_GAP(+1) + RES_L_GDP_GAP;


    // Core Inflation
    DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*L_GDP_GAP + RES_DLA_CPI;


    // Monetary policy reaction function
    RS = g1*RS(-1) + (1-g1)*(DLA_CPI(+1) + g2*DLA_CPI(+3) + g3*L_GDP_GAP) + RES_RS;


    RR_GAP = RS - DLA_CPI(+1);
    RES_L_GDP_GAP = rho_L_GDP_GAP*RES_L_GDP_GAP(-1) + SHK_L_GDP_GAP;
    RES_DLA_CPI = rho_DLA_CPI*RES_DLA_CPI(-1) + SHK_DLA_CPI;
    RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) +  SHK_RS;

end;


I need it to generate several auxiliary files to document the different steps of the parsing process. 
1. The first step is to read the dynare file and clean it from comments and empty lines. We only need
     Parameteres, 
     Variables, 
     exogenous,
     Model equations 
  This should be saved in a clean txt file 
2. Read the clean file generate in step 1 and extract the follwoing information 
        Identify the exogenous variables as thouse that variables that contains shock contains in varexo
        Lead forward the equation, that is variables like RES_RS = rho_rs*RES_RS(-1) + rho_rs2*RES_RS(-2) +  SHK_RS; 
        get identified and there is a link between the variable and the shock.  Also, the equation should be lead forward
        and written as RES_RS(+1) = rho_rs*RES_RS + rho_rs2*RES_RS(-1) +  SHK_RS;

        Save the file with the new timing in a second file called clean_file_with_correct_timing.txt

3. Load and parse the file clean_file_with_correct_timing.txt. 
       1. For each variable identified lead and lags so that RES_RS[ (+1),0, (-1)], or DLA_CPI[ (+3),0, (-1)]
       2. Generare auxiliary variables and equations to express all variables as variables in t and t+1. 
          That is,for example, for lags RES_RS(-1) we need one auxiliry variable RES_RS_lag and one auxiliary 
          equation RES_RS_lag_p = RES_RS; For lead larger that one DLA_CPI(+3), for example you will need two auxiliary 
          variables and two equations, DLA_CPI_lead = DLA_CPI(+1), DLA_CPI_lead2 = DLA_CPI_lead_p;   
       3. Save this updated file including the auxiliary variables and equations in a file called 
          clean_file_with_correct_timing_and_auxiliary_variables.txt

3. Load the file,  and replace (-1) with _lag (-2) with _lag2, (+2) with _lead_p (+3) with _lead2_p, etc. and save the file again
4. Load the file and parse it to generate the final output. The final output will have a json file with the model equations,
   a list of all variables, the list of states (all variables with lags), a list of shocks and the mapping the the exogenous variables
   and the auxiliary variables. The final output will be saved in a file called final_output.json 
5. Save the file as model_json.json
6. Load the json file and parse it to generates matrices A, B, C using the jacobbian. 
        A = \partial{equation}\partial x_p, B = -\partial{Equations}\partial{x}, C = -\partial{equations}/\partial{exogenous_shocks}
        create the jacobbians using symbolic differenciation and save the matrices in a file called jacobian_matrices.py, 
        this function takes as input the parameters of the model.   
7. You should also create a file called model_structure.py

import numpy as np

indices = {'n_states': 4, 'n_controls': 9, 'n_vars': 13, 'n_shocks': 3, 'n_endogenous_states': 3, 'n_exo_states_ws': 3, 'n_exo_states_wos': 3, 'zero_persistence_processes': []}

R_struct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

C_selection = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

D_struct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

# R(shock->state direct)=0; C(selects states); D(shock->var direct)=hits controls

labels = {'state_labels': ['DLA_CPI_lag', 'L_GDP_GAP_lag', 'RS_lag', 'RES_DLA_CPI_lag', 'RES_L_GDP_GAP_lag', 'RES_RS_lag', 'RES_RS_lag2'], 'control_labels': ['DLA_CPI', 'DLA_CPI_lead1', 'DLA_CPI_lead2', 'L_GDP_GAP', 'RES_DLA_CPI', 'RES_L_GDP_GAP', 'RES_RS', 'RR_GAP', 'RS'], 'variable_labels': ['DLA_CPI_lag', 'L_GDP_GAP_lag', 'RS_lag', 'RES_DLA_CPI_lag', 'RES_L_GDP_GAP_lag', 'RES_RS_lag', 'RES_RS_lag2', 'DLA_CPI', 'DLA_CPI_lead1', 'DLA_CPI_lead2', 'L_GDP_GAP', 'RES_DLA_CPI', 'RES_L_GDP_GAP', 'RES_RS', 'RR_GAP', 'RS'], 'shock_labels': ['SHK_L_GDP_GAP', 'SHK_DLA_CPI', 'SHK_RS']}


For the model above the json file should look like this 
{
  "parameters": [
    "b1",
    "b4",
    "a1",
    "a2",
    "g1",
    "g2",
    "g3",
    "rho_DLA_CPI",
    "rho_L_GDP_GAP",
    "rho_rs",
    "rho_rs2"
  ],
  "param_values": {
    "b1": 0.7,
    "b4": 0.7,
    "a1": 0.5,
    "a2": 0.1,
    "g1": 0.7,
    "g2": 0.3,
    "g3": 0.25,
    "rho_DLA_CPI": 0.75,
    "rho_L_GDP_GAP": 0.75,
    "rho_rs": 0.9,
    "rho_rs2": 0.1
  },
  "states": [
    "DLA_CPI_lag",
    "L_GDP_GAP_lag",
    "RS_lag",
    "RES_RS_lag",    
  ],
  "controls": [
    "DLA_CPI",
    "DLA_CPI_lead1",
    "DLA_CPI_lead2",
    "L_GDP_GAP",
    "RES_DLA_CPI",
    "RES_L_GDP_GAP",
    "RES_RS",
    "RR_GAP",
    "RS"
  ],
  "all_variables": [
    "DLA_CPI_lag",
    "L_GDP_GAP_lag",
    "RS_lag",
    "RES_RS_lag",    
    "DLA_CPI",
    "DLA_CPI_lead1",
    "DLA_CPI_lead2",
    "L_GDP_GAP",
    "RES_DLA_CPI",
    "RES_L_GDP_GAP",
    "RES_RS",
    "RR_GAP",
    "RS"
  ],
  "shocks": [
    "SHK_L_GDP_GAP",
    "SHK_DLA_CPI",
    "SHK_RS"
  ],
  "equations": [
    {
      "eq1": "(1-b1)*L_GDP_GAP_p + b1*L_GDP_GAP_lag - b4*RR_GAP_p + RES_L_GDP_GAP - (L_GDP_GAP)"
    },
    {
      "eq2": "a1*DLA_CPI_lag + (1-a1)*DLA_CPI_p + a2*L_GDP_GAP + RES_DLA_CPI - (DLA_CPI)"
    },
    {
      "eq3": "g1*RS_lag + (1-g1)*(DLA_CPI_p + g2*DLA_CPI_lead2_p + g3*L_GDP_GAP) + RES_RS - (RS)"
    },
    {
      "eq4": "RS - DLA_CPI_p - (RR_GAP)"
    },
    {
      "eq5": "rho_L_GDP_GAP*RES_L_GDP_GAP + SHK_L_GDP_GAP - (RES_L_GDP_GAP_p)"
    },
    {
      "eq6": "rho_DLA_CPI*RES_DLA_CPI + SHK_DLA_CPI - (RES_DLA_CPI_p)"
    },
    {
      "eq7": "rho_rs*RES_RS + rho_rs2*RES_RS_lag + SHK_RS - (RES_RS_p)"
    },
    {
      "eq8": "L_GDP_GAP - (L_GDP_GAP_lag_p)"
    },
    {
      "eq9": "RS - (RS_lag_p)"
    },
    {
      "eq10": "DLA_CPI - (DLA_CPI_lag_p)"
    },
    {
      "eq11": "DLA_CPI_p - (DLA_CPI_lead1)"
    },
    {
      "eq12": "DLA_CPI_lead1_p - (DLA_CPI_lead2)"
    },
    {
      "eq13": "RES_RS - (RES_RS_lag_p)"
    }    
    
  ],
  "shock_to_process_var_map": {
    "SHK_L_GDP_GAP": "RES_L_GDP_GAP",
    "SHK_DLA_CPI": "RES_DLA_CPI",
    "SHK_RS": "RES_RS"
  }
}
