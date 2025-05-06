function ds = dynamic_set_auxiliary_series(ds, params)
%
% Computes auxiliary variables of the dynamic model
%
ds.AUX_ENDO_LEAD_81=ds.DLA_CPI(1);
ds.AUX_ENDO_LEAD_44=ds.AUX_ENDO_LEAD_81(1);
ds.AUX_ENDO_LAG_6_1=ds.RES_RS(-1);
end
