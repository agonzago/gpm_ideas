function g1 = dynamic_g1(T, y, x, params, steady_state, it_, T_flag)
% function g1 = dynamic_g1(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   g1
%

if T_flag
    T = qpm_simpl1.dynamic_g1_tt(T, y, x, params, steady_state, it_);
end
g1 = zeros(9, 24);
g1(1,1)=(-params(1));
g1(1,5)=1;
g1(1,14)=(-(1-params(1)));
g1(1,16)=params(2);
g1(1,9)=(-1);
g1(2,5)=(-params(4));
g1(2,2)=(-params(3));
g1(2,6)=1;
g1(2,15)=(-(1-params(3)));
g1(2,10)=(-1);
g1(3,5)=(-((1-params(5))*params(7)));
g1(3,15)=(-(1-params(5)));
g1(3,3)=(-params(5));
g1(3,7)=1;
g1(3,11)=(-1);
g1(3,21)=(-((1-params(5))*params(6)));
g1(4,15)=1;
g1(4,7)=(-1);
g1(4,8)=1;
g1(5,9)=(-params(9));
g1(5,17)=1;
g1(5,22)=(-1);
g1(6,10)=(-params(8));
g1(6,18)=1;
g1(6,23)=(-1);
g1(7,4)=(-params(11));
g1(7,11)=(-params(10));
g1(7,19)=1;
g1(7,24)=(-1);
g1(8,15)=(-1);
g1(8,12)=1;
g1(9,20)=(-1);
g1(9,13)=1;

end
