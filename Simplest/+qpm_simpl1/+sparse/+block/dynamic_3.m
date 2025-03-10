function [y, T, residual, g1] = dynamic_3(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(25))-(y(16)*params(10)+params(11)*y(7)+x(3));
if nargout > 3
    g1_v = NaN(3, 1);
g1_v(1)=(-params(11));
g1_v(2)=(-params(10));
g1_v(3)=1;
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 3);
end
end
