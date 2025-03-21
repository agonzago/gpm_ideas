function SparseDynamicResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(residual) == 10
    @assert length(y) == 30
    @assert length(x) == 3
    @assert length(params) == 11
@inbounds begin
    residual[1] = (y[11]) - ((1-params[1])*y[21]+params[1]*y[1]-params[2]*y[24]+y[15]);
    residual[2] = (y[12]) - (params[3]*y[2]+(1-params[3])*y[22]+y[11]*params[4]+y[16]);
    residual[3] = (y[13]) - (y[17]+params[5]*y[3]+(1-params[5])*(y[11]*params[7]+y[22]+params[6]*y[29]));
    residual[4] = (y[14]) - (y[13]-y[22]);
    residual[5] = (y[15]) - (params[9]*y[5]+x[1]);
    residual[6] = (y[16]) - (params[8]*y[6]+x[2]);
    residual[7] = (y[17]) - (x[3]+params[10]*y[7]+params[11]*y[10]);
    residual[8] = (y[18]) - (y[22]);
    residual[9] = (y[19]) - (y[28]);
    residual[10] = (y[20]) - (y[7]);
end
    return nothing
end

