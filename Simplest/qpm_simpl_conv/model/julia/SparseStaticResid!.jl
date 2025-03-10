function SparseStaticResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(residual) == 17
    @assert length(y) == 17
    @assert length(x) == 3
    @assert length(params) == 11
@inbounds begin
    residual[1] = (y[1]) - (y[1]*(1-params[1])+params[1]*y[10]-params[2]*y[4]+y[5]);
    residual[2] = (y[5]) - (params[9]*y[14]+x[1]);
    residual[3] = (y[2]) - (params[3]*y[8]+y[2]*(1-params[3])+y[1]*params[4]+y[6]);
    residual[4] = (y[6]) - (params[8]*y[15]+x[2]);
    residual[5] = (y[3]) - (params[5]*y[9]+(1-params[5])*(y[2]+params[6]*y[13]+y[1]*params[7])+y[7]);
    residual[6] = (y[7]) - (params[10]*y[16]+params[11]*y[17]+x[3]);
    residual[7] = (y[4]) - (y[3]-y[2]);
    residual[8] = (y[8]) - (y[2]);
    residual[9] = (y[9]) - (y[3]);
    residual[10] = (y[10]) - (y[1]);
    residual[11] = (y[16]) - (y[7]);
    residual[12] = (y[17]) - (y[16]);
    residual[13] = (y[14]) - (y[5]);
    residual[14] = (y[15]) - (y[6]);
    residual[15] = (y[11]) - (y[2]);
    residual[16] = (y[12]) - (y[11]);
    residual[17] = (y[13]) - (y[12]);
end
    if ~isreal(residual)
        residual = real(residual)+imag(residual).^2;
    end
    return nothing
end

