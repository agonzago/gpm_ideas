function [residual, T_order, T] = static_resid(y, x, params, T_order, T)
if nargin < 5
    T_order = -1;
    T = NaN(0, 1);
end
[T_order, T] = qpm_simpl1.sparse.static_resid_tt(y, x, params, T_order, T);
residual = NaN(9, 1);
    residual(1) = (y(1)) - (y(1)*(1-params(1))+y(1)*params(1)-params(2)*y(4)+y(5));
    residual(2) = (y(2)) - (y(2)*params(3)+y(2)*(1-params(3))+y(1)*params(4)+y(6));
    residual(3) = (y(3)) - (y(7)+y(3)*params(5)+(1-params(5))*(y(1)*params(7)+y(2)+params(6)*y(9)));
    residual(4) = (y(4)) - (y(3)-y(2));
    residual(5) = (y(5)) - (y(5)*params(9)+x(1));
    residual(6) = (y(6)) - (y(6)*params(8)+x(2));
    residual(7) = (y(7)) - (y(7)*params(10)+y(7)*params(11)+x(3));
    residual(8) = (y(8)) - (y(2));
    residual(9) = (y(9)) - (y(8));
end
