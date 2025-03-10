function dynamic_set_auxiliary_series!(ds, params)
#
# Computes auxiliary variables of the dynamic model
#
@inbounds begin
ds.AUX_ENDO_LEAD_81 .=lag(ds.DLA_CPI,-1);
ds.AUX_ENDO_LEAD_44 .=lag(ds.AUX_ENDO_LEAD_81,-1);
ds.AUX_ENDO_LAG_6_1 .=lag(ds.RES_RS);
end
end
