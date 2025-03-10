function set_auxiliary_variables!(y, x, params)
#
# Computes auxiliary variables of the static model
#
@inbounds begin
y[8]=y[2];
y[9]=y[8];
y[10]=y[7];
end
end
