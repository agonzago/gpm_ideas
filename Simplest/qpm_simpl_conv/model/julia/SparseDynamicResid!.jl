function SparseDynamicResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(residual) == 17
    @assert length(y) == 51
    @assert length(x) == 3
    @assert length(params) == 11
@inbounds begin
    residual[1] = (y[18]) - ((1-params[1])*y[35]+params[1]*y[27]-params[2]*y[38]+y[22]);
    residual[2] = (y[22]) - (params[9]*y[31]+x[1]);
    residual[3] = (y[19]) - (params[3]*y[25]+(1-params[3])*y[36]+y[18]*params[4]+y[23]);
    residual[4] = (y[23]) - (params[8]*y[32]+x[2]);
    residual[5] = (y[20]) - (params[5]*y[26]+(1-params[5])*(y[36]+params[6]*y[30]+y[18]*params[7])+y[24]);
    residual[6] = (y[24]) - (params[10]*y[33]+params[11]*y[34]+x[3]);
    residual[7] = (y[21]) - (y[20]-y[36]);
    residual[8] = (y[25]) - (y[2]);
    residual[9] = (y[26]) - (y[3]);
    residual[10] = (y[27]) - (y[1]);
    residual[11] = (y[33]) - (y[7]);
    residual[12] = (y[34]) - (y[16]);
    residual[13] = (y[31]) - (y[5]);
    residual[14] = (y[32]) - (y[6]);
    residual[15] = (y[28]) - (y[36]);
    residual[16] = (y[29]) - (y[45]);
    residual[17] = (y[30]) - (y[46]);
end
    return nothing
end

