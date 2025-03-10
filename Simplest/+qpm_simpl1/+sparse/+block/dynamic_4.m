function [y, T, residual, g1] = dynamic_4(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(6, 1);
  residual(1)=(y(12))-(y(16)+params(5)*y(3)+(1-params(5))*(y(10)*params(7)+y(20)+params(6)*y(27)));
  residual(2)=(y(10))-((1-params(1))*y(19)+params(1)*y(1)-params(2)*y(22)+y(14));
  residual(3)=(y(11))-(params(3)*y(2)+(1-params(3))*y(20)+y(10)*params(4)+y(15));
  residual(4)=(y(13))-(y(12)-y(20));
  residual(5)=(y(17))-(y(20));
  residual(6)=(y(18))-(y(26));
if nargout > 3
    g1_v = NaN(20, 1);
g1_v(1)=(-params(5));
g1_v(2)=(-params(1));
g1_v(3)=(-params(3));
g1_v(4)=1;
g1_v(5)=(-1);
g1_v(6)=(-((1-params(5))*params(7)));
g1_v(7)=1;
g1_v(8)=(-params(4));
g1_v(9)=1;
g1_v(10)=1;
g1_v(11)=1;
g1_v(12)=1;
g1_v(13)=(-(1-params(1)));
g1_v(14)=(-(1-params(5)));
g1_v(15)=(-(1-params(3)));
g1_v(16)=1;
g1_v(17)=(-1);
g1_v(18)=params(2);
g1_v(19)=(-1);
g1_v(20)=(-((1-params(5))*params(6)));
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 6, 18);
end
end
