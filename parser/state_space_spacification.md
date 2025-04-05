1.  I have a code that construct the state space representation starting from the solution of the DSGE model by Klein.

    $s_{t+1}=Ps_{t}$

    $c_{t}=Fs_{t}$

    For simplicity it is easier to split the matrices of the system taking into account the kind of variable we have (endogenous states, exogenous state, and controls).

2.  When computing the solution of the model variables are ordered as follows

        variables = self.state_variables + self.control_variables

    where

        self.state_variables = endogenous_states + exo_with_shocks + exo_without_shocks

    So we can use this order to group variables in the F and P matrices cumming from Klein solution

    Lets call $c_{t}$ controls, $k_{t}$ endogenous states and $z_{t}$ exogenous states. That $c_{t}$ is the vector containing self.control_variables, $k_{t}$ a vector containing endogenous_states and $z_{t}$ ordered as exo_with_shocks + exo_without_shocks. With this notation is possible to expressed Klein's solution as

    $$\left(\begin{array}{c}
    k_{t+1}\\
    z_{t+1}
    \end{array}\right)=\left(\begin{array}{cc}
    P_{kk} & P_{kz}\\
    0 & P_{zz}
    \end{array}\right)\left(\begin{array}{c}
    k_{t}\\
    z_{t}
    \end{array}\right)$$

    $$c_{t}=\left(\begin{array}{cc}
    F_{ck} & F_{cz}\end{array}\right)\left(\begin{array}{c}
    k_{t}\\
    z_{t}
    \end{array}\right)$$ so that the state space representation would be $$\begin{aligned}
    k_{t+1} & =P_{kk}k_{t}+P_{kz}z_{t}\\
    z_{t+1} & =P_{zz}z_{t}+R\epsilon_{t+1}\\
    c_{t} & =F_{ck}k_{t}+F_{cz}z_{t}
    \end{aligned}$$

    where $\epsilon_{t+1}$ is the vector of shocks consistent with self.varexo_list. R is a selection matrix that tells how each shock affects the $z_{t}$. Given the information on the parser $z_{t}$ follows a particular order and each variable has it own shock. Here the issue is that some exogenous states may have more than one lag and hence in the representation they imply more states (current and lag). Only current states will have shock. This is typical of a companion form representation of AR process.

3.  With this notation we can write the state space model for the Kalman filter, under the assumption that all controls and states are observable

    $$\begin{aligned}
    y_{t} & =Cx_{t}\\
    x_{t} & =Ax_{t-1}+B\epsilon_{t}
    \end{aligned}$$ where $$\begin{aligned}
    C & =\left[\begin{array}{c}
    F\\
    I
    \end{array}\right]
    \end{aligned}$$

    $$A=P$$

    $$B=\left[\begin{array}{c}
    0_{(\text{n\_endogenos x n\_shock)}}\\
    R_{(\text{n\_exogenous x n\_shock)}}
    \end{array}\right]$$

    matrix dimensions are $B$: (n_states x n_shock), A (n_states x n_states) C (n_obs x n_states). Since we assume in this representation that all variables are potentially observable the n_obs = n_states. Also note that n_states = len(variables) in the solution of the DSGE. n_endogenous = n_controls + n_endogenous_states, n_exogenous = exo_with_shocks + exo_without_shocks.

4.  The code parse_gpm.py contains all this transformation.

5.  In the code full_code_includes_parser.py I'm working on an implementation that augments the state space by including stochastic trends and also has other caracteristics.

    1.  Only a subset of the variables is observable, as defined by the user

    2.  The code is written in a way the maximizes efficiency and limits the regeneration of matrices that are constant across parameter changes. That is, matrices, or submatrices that remain constant when parameters are changed. This is to generae a code that is efficient when computing the likelihood function or carry out mcmc.

    3.  The model state space representation remains similar but includes trends so the number of states is larger by definition. The augmented state space model should look like $$\begin{aligned}
        y_{t}^{obs} & =HC_{aug}x_{t}^{aug}\\
        x_{t}^{aug} & =A_{aug}x_{t-1}^{aug}+B_{aug}\epsilon_{t}^{aug}
        \end{aligned}$$ where H: is a selection matrix with zeros and ones (or may be replace by an index) that selects the rows of $C_{aug}x_{t}^{aug}$ that are observable, $$A_{aug}=\left[\begin{array}{cc}
        A & 0\\
        0 & A_{trend}
        \end{array}\right]$$

        $$B_{aug}=\left[\begin{array}{cc}
        B & 0\\
        0 & B_{trend}
        \end{array}\right]$$ and $\epsilon_{t}^{aug}=\left(\epsilon_{t},\epsilon_{t}^{trends}\right)$.

    4.  Not all observable variables have the same trend specification. The trend specification of a variables can be

        1.  Random walk:

            $$trend_{t}=trend_{t-1}+\epsilon_{t}^{level}$$

        2.  Second difference $$\begin{aligned}
            trend_{t} & =trend_{t-1}+g_{t-1}+\epsilon_{t}^{level}\\
            g_{t} & =g_{t-1}+\epsilon_{t}^{growth}
            \end{aligned}$$

        3.  Constant mean

            $$trend_{t}=trend_{t-1}$$

    5.  The code is not producing the right IRFs but I think is because, 1) is not creating the state space representation in an organized for. I'm adding this information to seek you help and produce a state space represenation for the augmented model that matches the description above and repreduces the IRFS accordingly.

    6.  In a future specification we may consider situations where the trends can interact.

6.  The code is structure as follows:

    1.  DynareParser just reads the file and produce some ausiliary function and constants needed for the solution

    2.  ModelSover. Loads files created by the parser and computes the klain solution

    3.  AugmentedState Space creates the augmented representation.

7.  There are several changes needed. ModelSolver has los of function not needed and is missing Klein solution (that is available in the code, just copy that into the class). Remove all not necesary functions.

8.  Check the augmented class and link it to the model solver. Check the the filtration is working and the irfs are correct.
