function SparseDynamicG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(g1_v) == 35
    @assert length(y) == 30
    @assert length(x) == 3
    @assert length(params) == 11
@inbounds begin
g1_v[1]=(-params[1]);
g1_v[2]=(-params[3]);
g1_v[3]=(-params[5]);
g1_v[4]=(-params[9]);
g1_v[5]=(-params[8]);
g1_v[6]=(-params[10]);
g1_v[7]=(-1);
g1_v[8]=(-params[11]);
g1_v[9]=1;
g1_v[10]=(-params[4]);
g1_v[11]=(-((1-params[5])*params[7]));
g1_v[12]=1;
g1_v[13]=1;
g1_v[14]=(-1);
g1_v[15]=1;
g1_v[16]=(-1);
g1_v[17]=1;
g1_v[18]=(-1);
g1_v[19]=1;
g1_v[20]=(-1);
g1_v[21]=1;
g1_v[22]=1;
g1_v[23]=1;
g1_v[24]=1;
g1_v[25]=(-(1-params[1]));
g1_v[26]=(-(1-params[3]));
g1_v[27]=(-(1-params[5]));
g1_v[28]=1;
g1_v[29]=(-1);
g1_v[30]=params[2];
g1_v[31]=(-1);
g1_v[32]=(-((1-params[5])*params[6]));
g1_v[33]=(-1);
g1_v[34]=(-1);
g1_v[35]=(-1);
end
    return nothing
end

