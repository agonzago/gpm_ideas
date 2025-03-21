function SparseStaticResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(residual) == 10
    @assert length(y) == 10
    @assert length(x) == 3
    @assert length(params) == 11
@inbounds begin
    residual[1] = (y[1]) - (y[1]*(1-params[1])+y[1]*params[1]-params[2]*y[4]+y[5]);
    residual[2] = (y[2]) - (y[2]*params[3]+y[2]*(1-params[3])+y[1]*params[4]+y[6]);
    residual[3] = (y[3]) - (y[7]+y[3]*params[5]+(1-params[5])*(y[1]*params[7]+y[2]+params[6]*y[9]));
    residual[4] = (y[4]) - (y[3]-y[2]);
    residual[5] = (y[5]) - (y[5]*params[9]+x[1]);
    residual[6] = (y[6]) - (y[6]*params[8]+x[2]);
    residual[7] = (y[7]) - (x[3]+y[7]*params[10]+params[11]*y[10]);
    residual[8] = (y[8]) - (y[2]);
    residual[9] = (y[9]) - (y[8]);
    residual[10] = (y[10]) - (y[7]);
end
    if ~isreal(residual)
        residual = real(residual)+imag(residual).^2;
    end
    return nothing
end

