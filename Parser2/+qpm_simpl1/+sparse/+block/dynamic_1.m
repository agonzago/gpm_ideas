function [y, T] = dynamic_1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(14)=params(9)*y(5)+x(1);
  y(15)=params(8)*y(6)+x(2);
  y(16)=params(10)*y(7)+x(3);
end
