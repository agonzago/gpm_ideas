function [residual, T_order, T] = dynamic_resid(y, x, params, steady_state, T_order, T)
if nargin < 6
    T_order = -1;
    T = NaN(0, 1);
end
[T_order, T] = qpm_simpl1.sparse.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
residual = NaN(9, 1);
    residual(1) = (y(10)) - ((1-params(1))*y(19)+params(1)*y(1)-params(2)*y(22)+y(14));
    residual(2) = (y(11)) - (params(3)*y(2)+(1-params(3))*y(20)+y(10)*params(4)+y(15));
    residual(3) = (y(12)) - (y(16)+params(5)*y(3)+(1-params(5))*(y(10)*params(7)+y(20)+params(6)*y(27)));
    residual(4) = (y(13)) - (y(12)-y(20));
    residual(5) = (y(14)) - (params(9)*y(5)+x(1));
    residual(6) = (y(15)) - (params(8)*y(6)+x(2));
    residual(7) = (y(16)) - (params(10)*y(7)+x(3));
    residual(8) = (y(17)) - (y(20));
    residual(9) = (y(18)) - (y(26));
end
