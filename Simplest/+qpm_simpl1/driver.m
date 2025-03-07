%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'qpm_simpl1';
M_.dynare_version = '5.5';
oo_.dynare_version = '5.5';
options_.dynare_version = '5.5';
%
% Some global variables initialization
%
global_initialization;
M_.exo_names = cell(3,1);
M_.exo_names_tex = cell(3,1);
M_.exo_names_long = cell(3,1);
M_.exo_names(1) = {'SHK_L_GDP_GAP'};
M_.exo_names_tex(1) = {'SHK\_L\_GDP\_GAP'};
M_.exo_names_long(1) = {'SHK_L_GDP_GAP'};
M_.exo_names(2) = {'SHK_DLA_CPI'};
M_.exo_names_tex(2) = {'SHK\_DLA\_CPI'};
M_.exo_names_long(2) = {'SHK_DLA_CPI'};
M_.exo_names(3) = {'SHK_RS'};
M_.exo_names_tex(3) = {'SHK\_RS'};
M_.exo_names_long(3) = {'SHK_RS'};
M_.endo_names = cell(9,1);
M_.endo_names_tex = cell(9,1);
M_.endo_names_long = cell(9,1);
M_.endo_names(1) = {'L_GDP_GAP'};
M_.endo_names_tex(1) = {'L\_GDP\_GAP'};
M_.endo_names_long(1) = {'L_GDP_GAP'};
M_.endo_names(2) = {'DLA_CPI'};
M_.endo_names_tex(2) = {'DLA\_CPI'};
M_.endo_names_long(2) = {'DLA_CPI'};
M_.endo_names(3) = {'RS'};
M_.endo_names_tex(3) = {'RS'};
M_.endo_names_long(3) = {'RS'};
M_.endo_names(4) = {'RR_GAP'};
M_.endo_names_tex(4) = {'RR\_GAP'};
M_.endo_names_long(4) = {'RR_GAP'};
M_.endo_names(5) = {'RES_L_GDP_GAP'};
M_.endo_names_tex(5) = {'RES\_L\_GDP\_GAP'};
M_.endo_names_long(5) = {'RES_L_GDP_GAP'};
M_.endo_names(6) = {'RES_DLA_CPI'};
M_.endo_names_tex(6) = {'RES\_DLA\_CPI'};
M_.endo_names_long(6) = {'RES_DLA_CPI'};
M_.endo_names(7) = {'RES_RS'};
M_.endo_names_tex(7) = {'RES\_RS'};
M_.endo_names_long(7) = {'RES_RS'};
M_.endo_names(8) = {'AUX_ENDO_LEAD_76'};
M_.endo_names_tex(8) = {'AUX\_ENDO\_LEAD\_76'};
M_.endo_names_long(8) = {'AUX_ENDO_LEAD_76'};
M_.endo_names(9) = {'AUX_ENDO_LEAD_56'};
M_.endo_names_tex(9) = {'AUX\_ENDO\_LEAD\_56'};
M_.endo_names_long(9) = {'AUX_ENDO_LEAD_56'};
M_.endo_partitions = struct();
M_.param_names = cell(11,1);
M_.param_names_tex = cell(11,1);
M_.param_names_long = cell(11,1);
M_.param_names(1) = {'b1'};
M_.param_names_tex(1) = {'b1'};
M_.param_names_long(1) = {'b1'};
M_.param_names(2) = {'b2'};
M_.param_names_tex(2) = {'b2'};
M_.param_names_long(2) = {'b2'};
M_.param_names(3) = {'b4'};
M_.param_names_tex(3) = {'b4'};
M_.param_names_long(3) = {'b4'};
M_.param_names(4) = {'a1'};
M_.param_names_tex(4) = {'a1'};
M_.param_names_long(4) = {'a1'};
M_.param_names(5) = {'a2'};
M_.param_names_tex(5) = {'a2'};
M_.param_names_long(5) = {'a2'};
M_.param_names(6) = {'g1'};
M_.param_names_tex(6) = {'g1'};
M_.param_names_long(6) = {'g1'};
M_.param_names(7) = {'g2'};
M_.param_names_tex(7) = {'g2'};
M_.param_names_long(7) = {'g2'};
M_.param_names(8) = {'g3'};
M_.param_names_tex(8) = {'g3'};
M_.param_names_long(8) = {'g3'};
M_.param_names(9) = {'rho_DLA_CPI'};
M_.param_names_tex(9) = {'rho\_DLA\_CPI'};
M_.param_names_long(9) = {'rho_DLA_CPI'};
M_.param_names(10) = {'rho_L_GDP_GAP'};
M_.param_names_tex(10) = {'rho\_L\_GDP\_GAP'};
M_.param_names_long(10) = {'rho_L_GDP_GAP'};
M_.param_names(11) = {'rho_rs'};
M_.param_names_tex(11) = {'rho\_rs'};
M_.param_names_long(11) = {'rho_rs'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 3;
M_.endo_nbr = 9;
M_.param_nbr = 11;
M_.orig_endo_nbr = 7;
M_.aux_vars(1).endo_index = 8;
M_.aux_vars(1).type = 0;
M_.aux_vars(1).orig_expr = 'DLA_CPI(1)';
M_.aux_vars(2).endo_index = 9;
M_.aux_vars(2).type = 0;
M_.aux_vars(2).orig_expr = 'AUX_ENDO_LEAD_76(1)';
M_ = setup_solvers(M_);
M_.Sigma_e = zeros(3, 3);
M_.Correlation_matrix = eye(3, 3);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
M_.surprise_shocks = [];
M_.heteroskedastic_shocks.Qvalue_orig = [];
M_.heteroskedastic_shocks.Qscale_orig = [];
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
M_.orig_eq_nbr = 7;
M_.eq_nbr = 9;
M_.ramsey_eq_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 1;
M_.orig_maximum_endo_lead = 3;
M_.orig_maximum_exo_lag = 0;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 1;
M_.orig_maximum_lead = 3;
M_.orig_maximum_lag_with_diffs_expanded = 1;
M_.lead_lag_incidence = [
 1 7 16;
 2 8 17;
 3 9 0;
 0 10 18;
 4 11 0;
 5 12 0;
 6 13 0;
 0 14 19;
 0 15 20;]';
M_.nstatic = 0;
M_.nfwrd   = 3;
M_.npred   = 4;
M_.nboth   = 2;
M_.nsfwrd   = 5;
M_.nspred   = 6;
M_.ndynamic   = 9;
M_.dynamic_tmp_nbr = [0; 0; 0; 0; ];
M_.model_local_variables_dynamic_tt_idxs = {
};
M_.equations_tags = {
  1 , 'name' , 'L_GDP_GAP' ;
  2 , 'name' , 'RES_L_GDP_GAP' ;
  3 , 'name' , 'DLA_CPI' ;
  4 , 'name' , 'RES_DLA_CPI' ;
  5 , 'name' , 'RS' ;
  6 , 'name' , 'RES_RS' ;
  7 , 'name' , 'RR_GAP' ;
};
M_.mapping.L_GDP_GAP.eqidx = [1 3 5 ];
M_.mapping.DLA_CPI.eqidx = [3 5 7 ];
M_.mapping.RS.eqidx = [5 7 ];
M_.mapping.RR_GAP.eqidx = [1 7 ];
M_.mapping.RES_L_GDP_GAP.eqidx = [1 2 ];
M_.mapping.RES_DLA_CPI.eqidx = [3 4 ];
M_.mapping.RES_RS.eqidx = [5 6 ];
M_.mapping.SHK_L_GDP_GAP.eqidx = [2 ];
M_.mapping.SHK_DLA_CPI.eqidx = [4 ];
M_.mapping.SHK_RS.eqidx = [6 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.state_var = [1 2 3 5 6 7 ];
M_.exo_names_orig_ord = [1:3];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(9, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(3, 1);
M_.params = NaN(11, 1);
M_.endo_trends = struct('deflator', cell(9, 1), 'log_deflator', cell(9, 1), 'growth_factor', cell(9, 1), 'log_growth_factor', cell(9, 1));
M_.NNZDerivatives = [32; -1; -1; ];
M_.static_tmp_nbr = [0; 0; 0; 0; ];
M_.model_local_variables_static_tt_idxs = {
};
M_.params(1) = 0.7;
b1 = M_.params(1);
M_.params(2) = 0.2;
b2 = M_.params(2);
M_.params(3) = 0.7;
b4 = M_.params(3);
M_.params(4) = 0.5;
a1 = M_.params(4);
M_.params(5) = 0.1;
a2 = M_.params(5);
M_.params(6) = 0.7;
g1 = M_.params(6);
M_.params(7) = 0.3;
g2 = M_.params(7);
M_.params(8) = 0.25;
g3 = M_.params(8);
M_.params(10) = 0.0;
rho_L_GDP_GAP = M_.params(10);
M_.params(9) = 0.0;
rho_DLA_CPI = M_.params(9);
M_.params(11) = 0.0;
rho_rs = M_.params(11);
%
% INITVAL instructions
%
options_.initval_file = false;
oo_.steady_state(1) = 0;
oo_.steady_state(2) = 0;
oo_.steady_state(3) = 0;
oo_.steady_state(4) = 0;
oo_.steady_state(5) = 0;
oo_.steady_state(6) = 0;
oo_.steady_state(7) = 0;
oo_.steady_state(8)=oo_.steady_state(2);
oo_.steady_state(9)=oo_.steady_state(8);
if M_.exo_nbr > 0
	oo_.exo_simul = ones(M_.maximum_lag,1)*oo_.exo_steady_state';
end
if M_.exo_det_nbr > 0
	oo_.exo_det_simul = ones(M_.maximum_lag,1)*oo_.exo_det_steady_state';
end
steady;
oo_.dr.eigval = check(M_,options_,oo_);
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = 1;
options_.irf = 40;
options_.order = 1;
var_list_ = {'L_GDP_GAP';'DLA_CPI';'RS'};
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_);


oo_.time = toc(tic0);
disp(['Total computing time : ' dynsec2hms(oo_.time) ]);
if ~exist([M_.dname filesep 'Output'],'dir')
    mkdir(M_.dname,'Output');
end
save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'oo_recursive_', '-append');
end
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
