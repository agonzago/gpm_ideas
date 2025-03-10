function [y, T, residual, g1] = static_4(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(6, 1);
  residual(1)=(y(4))-(y(3)-y(2));
  residual(2)=(y(1))-(y(1)*(1-params(1))+y(1)*params(1)-params(2)*y(4)+y(5));
  residual(3)=(y(2))-(y(2)*params(3)+y(2)*(1-params(3))+y(1)*params(4)+y(6));
  residual(4)=(y(3))-(y(7)+y(3)*params(5)+(1-params(5))*(y(1)*params(7)+y(2)+params(6)*y(9)));
  residual(5)=(y(8))-(y(2));
  residual(6)=(y(9))-(y(8));
if nargout > 3
    g1_v = NaN(13, 1);
g1_v(1)=(-1);
g1_v(2)=1-params(5);
g1_v(3)=1;
g1_v(4)=params(2);
g1_v(5)=(-params(4));
g1_v(6)=(-((1-params(5))*params(7)));
g1_v(7)=1;
g1_v(8)=(-(1-params(5)));
g1_v(9)=(-1);
g1_v(10)=1;
g1_v(11)=(-1);
g1_v(12)=(-((1-params(5))*params(6)));
g1_v(13)=1;
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 6, 6);
end
end
