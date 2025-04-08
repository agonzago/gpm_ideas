1.  I'm working on a code that reads a dynare file (of a linear model).
    It can be any text file (so dynare is not fundamental). The idea is
    that, I read the dyanre-like file and create a json file with the
    equations written in timing t and t+1 as will be expected by Klein's
    method. The model has long leads and lags so I need to create
    auxiliary equations and variables to accommodate this feature. After
    creating the json file I compute the Jacobbian the equation with
    respect to the variables in t and t+1 using analytical methods and
    save the jacobbian's in the file that letter is loaded in memory.
    The procedure loads these files and evaluate them a parameter value
    and call Klein's solution method. The solution is working fine.

2.  The system of equations I'm working on is

    $$AE_{t}x_{t+1}=Bx_{t}+C\epsilon_{t}$$ where $x_{t}$ is a vector
    organized such that you have state first and control after. Let's
    call controls $c_{t}$ and states $s_{t}$. States included lagged
    values of the endogenous variables and lagged values of the
    exogenous variables.

3.  As mentioned before any variable in the model could have long lead
    and lags. For these variables the parser generates auxiliary
    variables and equations to write the model in the standard Klein
    format. So if there is a lead variable with orrder 2 for example the
    parser will create the following variables and equations. Say there
    is a variable in the system with y(+2) that is with two leads in
    this case,

    $$\begin{aligned}
    x_{t}^{lead1} & =y(+1)\\
    x_{t}^{lead2} & =x_{t+1}^{lead1}
    \end{aligned}$$ then effectively $x_{t}^{lead2}=y(+1)$, longer leads
    will follow the same approach. If you longer lags, say y(-2), you
    will also create auxiliary variables, in this case

    $$x_{t+1}^{lag}=y$$ so $x_{t}^{lag}=y(-1),$similarly longer lags can
    be accommodated.

4.  One aspect that deserves attention is the following: In the model,
    not all exogenous variables have the same lag structure. For
    example, you could you have two exogenous variables
    $$z_{1t}=\rho_{10}z_{1t-1}+\rho_{11}z_{1t-2}+\epsilon_{t}^{z1}$$

    $$z_{2t}=\rho_{20}z_{2t-1}+\epsilon_{t}^{z2}$$ so in this case, the
    $z_{1t}$ variable will be represented in Klein's Canonical
    representation with the equations $$\begin{aligned}
    x_{t+1}^{lag} & =z_{1t}
    \end{aligned}$$ so the representation would be $$\begin{aligned}
    z_{1t+1} & =\rho_{10}z_{t}+\rho_{11}x_{t}^{lag}+\epsilon_{t+1}^{z1}\\
    x_{t+1}^{lag} & =z_{1t}
    \end{aligned}$$ that leads to the following state vector
    $\left(z_{1t},x_{t}^{lag},x_{t}^{lag2}\right)^{\prime}$ and
    associated Klein's Canonical representation
    $$\left[\begin{array}{ccc}
    1 & 0 & 0\\
    0 & 1 & 0\\
    0 & 0 & 1
    \end{array}\right]\left[\begin{array}{c}
    z_{1t+1}\\
    x_{t+1}^{lag}\\
    z_{2t+1}
    \end{array}\right]=\left[\begin{array}{ccc}
    \rho_{10} & \rho_{11} & 0\\
    1 & 0 & 0\\
    0 & 0 & \rho_{20}
    \end{array}\right]\left[\begin{array}{c}
    z_{1t}\\
    x_{t}^{lag}\\
    z_{2t}
    \end{array}\right]+\left[\begin{array}{cc}
    1 & 0\\
    0 & 0\\
    0 & 1
    \end{array}\right]\left[\begin{array}{c}
    \epsilon_{t+1}^{z1}\\
    \epsilon_{t+1}^{z2}
    \end{array}\right]$$ that is, two important things emerge:

    1.  Even if the model in dynare syntax is written as
        $z_{1t}=\rho_{10}z_{1t-1}+\rho_{11}z_{1t-1}+\epsilon_{t}^{z1}$
        the parser should understand it as
        $z_{1t+1}=\rho_{10}z_{1t}+\rho_{11}z_{1t-1}+\epsilon_{t+1}^{z1}$
        and $z_{t}$ is still the contemporaneous exogenous variable. So
        that at shock at time $z_{t}$ moves $z_{t}$ and all other
        variables in the system. Say
        $$\tilde{y}_{t}=\alpha\tilde{y}_{t+1}+z_{t}$$ where
        $\tilde{y}_{t}$ is an endogenous control.

    2.  In the system above (with the Canonical representation for the
        shocks) there is matrix R multiplying the vector of exogenous
        shocks in the example $$R=\left[\begin{array}{cc}
        1 & 0\\
        0 & 0\\
        0 & 1
        \end{array}\right]$$ this matrix is part of matrix $C$ in the
        Canonical representation and should map shocks to exogenous
        states. Not all states will get a shock by construction.

5.  The next task is to build a state space representation for the
    Kalman filter. The way I'm doing this is

    $s_{t+1}=Ps_{t}$

    $c_{t}=Fs_{t}$

    For simplicity it is easier to split the matrices of the system
    taking into account the kind of variable we have (endogenous states,
    exogenous state, and controls).

6.  When computing the jacobbian of the model equations variables are
    ordered as follows

        variables = self.state_variables + self.control_variables

    where

        self.state_variables = endogenous_states + exo_with_shocks + exo_without_shocks

    So we can use this order to group variables in the F and P matrices
    cumming from Klein's solution

    Lets call $c_{t}$ controls, $k_{t}$ endogenous states and $\xi_{t}$
    exogenous states. That $c_{t}$ is the vector containing
    self.control_variables, $k_{t}$ a vector containing
    endogenous_states and $\xi_{t}$ ordered as exo_with_shocks +
    exo_without_shocks. With this notation is possible to expressed
    Klein's solution as

    $$\left(\begin{array}{c}
    k_{t+1}\\
    \xi_{t+1}
    \end{array}\right)=\left(\begin{array}{cc}
    P_{kk} & P_{k\xi}\\
    0 & P_{\xi\xi}
    \end{array}\right)\left(\begin{array}{c}
    k_{t}\\
    \xi_{t}
    \end{array}\right)+H\epsilon_{t+1}$$ where $\epsilon_{t}$ contains
    the exogenous shocks and $H$ holds 1 and zeros selecting and it's
    function of $R$. I'm using $\xi_{t}$ to denote the exogenous states
    to make the point that given the lag structure of the exiguous
    process $\xi$ is not equal to $z_{t}$ but it could include auxiliary
    states.

    $$c_{t}=\left(\begin{array}{cc}
    F_{ck} & F_{c\xi}\end{array}\right)\left(\begin{array}{c}
    k_{t}\\
    \xi_{t}
    \end{array}\right)$$

7.  How to get the steady space representation $$\begin{aligned}
    k_{t+1} & =p_{kk}k_{t}+p_{k\xi}\xi_{t}\\
    c_{t} & =F_{ck}k_{t}+F_{\xi\xi}\xi_{t}
    \end{aligned}$$ if I lag $\xi_{t}$ then,

    $$\begin{aligned}
    k_{t+1} & =p_{kk}k_{t}+p_{k\xi}p_{\xi\xi}\xi_{t-1}+p_{k\xi}R\epsilon_{t}\\
    c_{t}= & F_{ck}k_{t}+F_{\xi\xi}p_{\xi\xi}\xi_{t-1}+F_{\xi\xi}R\epsilon_{t}\\
    \xi_{t}= & p_{\xi\xi}\xi_{t-1}+R\epsilon_{t}
    \end{aligned}$$ in this representation a shock today can affect, the
    endogenous states tomorrow, controls and exogenous states
    contemporaneously.

8.  I'm working on an implementation that augments the state space by
    including stochastic trends and also has other characteristics.

    1.  Only a subset of the variables is observable, as defined by the
        user

    2.  The code is written in a way the maximizes efficiency and limits
        the regeneration of matrices that are constant across parameter
        changes. That is, matrices, or submatrices that remain constant
        when parameters are changed. This is to generate a code that is
        efficient when computing the likelihood function or carry out
        mcmc.

    3.  The model state space representation remains similar but
        includes trends so the number of states is larger by definition.
        The augmented state space model should look like
        $$\begin{aligned}
        y_{t}^{obs} & =HC_{aug}x_{t}^{aug}\\
        x_{t}^{aug} & =A_{aug}x_{t-1}^{aug}+B_{aug}\epsilon_{t}^{aug}
        \end{aligned}$$ where H: is a selection matrix with zeros and
        ones (or may be replace by an index) that selects the rows of
        $C_{aug}x_{t}^{aug}$ that are observable,
        $$A_{aug}=\left[\begin{array}{cc}
        A & 0\\
        0 & A_{trend}
        \end{array}\right]$$

        $$B_{aug}=\left[\begin{array}{cc}
        B & 0\\
        0 & B_{trend}
        \end{array}\right]$$ and
        $\epsilon_{t}^{aug}=\left(\epsilon_{t},\epsilon_{t}^{trends}\right)$.

    4.  Not all observable variables have the same trend specification.
        The trend specification of a variables can be

        1.  Random walk:

            $$trend_{t}=trend_{t-1}+\epsilon_{t}^{level}$$

        2.  Second difference $$\begin{aligned}
            trend_{t} & =trend_{t-1}+g_{t-1}+\epsilon_{t}^{level}\\
            g_{t} & =g_{t-1}+\epsilon_{t}^{growth}
            \end{aligned}$$

        3.  Constant mean

            $$trend_{t}=trend_{t-1}$$

9.  The code is structure as follows:

    1.  DynareParser just reads the file and produce some auxiliary
        function and constants needed for the solution

    2.  ModelSover. Loads files created by the parser and computes the
        klain solution

    3.  AugmentedState Space creates the augmented representation.

10. Tasks

    1.  Check the parser and make sure it correctly creates the
        auxiliary equations as explained before.

    2.  Given this jaccobians check that the state space representation
        is correct

    3.  Check the augmented class and link it to the model solver.

    4.  Create a function that computes the IRFs of the simple model and
        the augmented model (they should produce the same results for
        shocks in the econonomic model)
