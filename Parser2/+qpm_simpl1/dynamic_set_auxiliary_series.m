function ds = dynamic_set_auxiliary_series(ds, params)
%
% Computes auxiliary variables of the dynamic model
%
ds.AUX_ENDO_LEAD_76=ds.DLA_CPI(1);
ds.AUX_ENDO_LEAD_44=ds.AUX_ENDO_LEAD_76(1);
end
