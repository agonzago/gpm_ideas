function SparseStaticG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 0
    @assert length(g1_v) == 22
    @assert length(y) == 10
    @assert length(x) == 3
    @assert length(params) == 11
@inbounds begin
g1_v[1]=(-params[4]);
g1_v[2]=(-((1-params[5])*params[7]));
g1_v[3]=(-(1-params[5]));
g1_v[4]=1;
g1_v[5]=(-1);
g1_v[6]=1-params[5];
g1_v[7]=(-1);
g1_v[8]=params[2];
g1_v[9]=1;
g1_v[10]=(-1);
g1_v[11]=1-params[9];
g1_v[12]=(-1);
g1_v[13]=1-params[8];
g1_v[14]=(-1);
g1_v[15]=1-params[10];
g1_v[16]=(-1);
g1_v[17]=1;
g1_v[18]=(-1);
g1_v[19]=(-((1-params[5])*params[6]));
g1_v[20]=1;
g1_v[21]=(-params[11]);
g1_v[22]=1;
end
    if ~isreal(g1_v)
        g1_v = real(g1_v)+2*imag(g1_v);
    end
    return nothing
end

