function [g1, T_order, T] = dynamic_g1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T_order, T)
if nargin < 9
    T_order = -1;
    T = NaN(0, 1);
end
[T_order, T] = qpm_simpl1.sparse.dynamic_g1_tt(y, x, params, steady_state, T_order, T);
g1_v = NaN(32, 1);
g1_v(1)=(-params(1));
g1_v(2)=(-params(3));
g1_v(3)=(-params(5));
g1_v(4)=(-params(9));
g1_v(5)=(-params(8));
g1_v(6)=(-params(10));
g1_v(7)=1;
g1_v(8)=(-params(4));
g1_v(9)=(-((1-params(5))*params(7)));
g1_v(10)=1;
g1_v(11)=1;
g1_v(12)=(-1);
g1_v(13)=1;
g1_v(14)=(-1);
g1_v(15)=1;
g1_v(16)=(-1);
g1_v(17)=1;
g1_v(18)=(-1);
g1_v(19)=1;
g1_v(20)=1;
g1_v(21)=1;
g1_v(22)=(-(1-params(1)));
g1_v(23)=(-(1-params(3)));
g1_v(24)=(-(1-params(5)));
g1_v(25)=1;
g1_v(26)=(-1);
g1_v(27)=params(2);
g1_v(28)=(-1);
g1_v(29)=(-((1-params(5))*params(6)));
g1_v(30)=(-1);
g1_v(31)=(-1);
g1_v(32)=(-1);
if ~isoctave && matlab_ver_less_than('9.8')
    sparse_rowval = double(sparse_rowval);
    sparse_colval = double(sparse_colval);
end
g1 = sparse(sparse_rowval, sparse_colval, g1_v, 9, 30);
end
