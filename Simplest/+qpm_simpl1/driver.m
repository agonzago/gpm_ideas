%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

clearvars -global
clear_persistent_variables(fileparts(which('dynare')), false)
tic0 = tic;
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info
options_ = [];
M_.fname = 'qpm_simpl1';
M_.dynare_version = '6.0';
oo_.dynare_version = '6.0';
options_.dynare_version = '6.0';
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
M_.endo_names = cell(10,1);
M_.endo_names_tex = cell(10,1);
M_.endo_names_long = cell(10,1);
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
M_.endo_names(8) = {'AUX_ENDO_LEAD_81'};
M_.endo_names_tex(8) = {'AUX\_ENDO\_LEAD\_81'};
M_.endo_names_long(8) = {'AUX_ENDO_LEAD_81'};
M_.endo_names(9) = {'AUX_ENDO_LEAD_44'};
M_.endo_names_tex(9) = {'AUX\_ENDO\_LEAD\_44'};
M_.endo_names_long(9) = {'AUX_ENDO_LEAD_44'};
M_.endo_names(10) = {'AUX_ENDO_LAG_6_1'};
M_.endo_names_tex(10) = {'AUX\_ENDO\_LAG\_6\_1'};
M_.endo_names_long(10) = {'AUX_ENDO_LAG_6_1'};
M_.endo_partitions = struct();
M_.param_names = cell(11,1);
M_.param_names_tex = cell(11,1);
M_.param_names_long = cell(11,1);
M_.param_names(1) = {'b1'};
M_.param_names_tex(1) = {'b1'};
M_.param_names_long(1) = {'b1'};
M_.param_names(2) = {'b4'};
M_.param_names_tex(2) = {'b4'};
M_.param_names_long(2) = {'b4'};
M_.param_names(3) = {'a1'};
M_.param_names_tex(3) = {'a1'};
M_.param_names_long(3) = {'a1'};
M_.param_names(4) = {'a2'};
M_.param_names_tex(4) = {'a2'};
M_.param_names_long(4) = {'a2'};
M_.param_names(5) = {'g1'};
M_.param_names_tex(5) = {'g1'};
M_.param_names_long(5) = {'g1'};
M_.param_names(6) = {'g2'};
M_.param_names_tex(6) = {'g2'};
M_.param_names_long(6) = {'g2'};
M_.param_names(7) = {'g3'};
M_.param_names_tex(7) = {'g3'};
M_.param_names_long(7) = {'g3'};
M_.param_names(8) = {'rho_DLA_CPI'};
M_.param_names_tex(8) = {'rho\_DLA\_CPI'};
M_.param_names_long(8) = {'rho_DLA_CPI'};
M_.param_names(9) = {'rho_L_GDP_GAP'};
M_.param_names_tex(9) = {'rho\_L\_GDP\_GAP'};
M_.param_names_long(9) = {'rho_L_GDP_GAP'};
M_.param_names(10) = {'rho_rs'};
M_.param_names_tex(10) = {'rho\_rs'};
M_.param_names_long(10) = {'rho_rs'};
M_.param_names(11) = {'rho_rs2'};
M_.param_names_tex(11) = {'rho\_rs2'};
M_.param_names_long(11) = {'rho_rs2'};
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 3;
M_.endo_nbr = 10;
M_.param_nbr = 11;
M_.orig_endo_nbr = 7;
M_.aux_vars(1).endo_index = 8;
M_.aux_vars(1).type = 0;
M_.aux_vars(1).orig_expr = 'DLA_CPI(1)';
M_.aux_vars(2).endo_index = 9;
M_.aux_vars(2).type = 0;
M_.aux_vars(2).orig_expr = 'AUX_ENDO_LEAD_81(1)';
M_.aux_vars(3).endo_index = 10;
M_.aux_vars(3).type = 1;
M_.aux_vars(3).orig_index = 7;
M_.aux_vars(3).orig_lead_lag = -1;
M_.aux_vars(3).orig_expr = 'RES_RS(-1)';
M_.Sigma_e = zeros(3, 3);
M_.Correlation_matrix = eye(3, 3);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = true;
M_.det_shocks = [];
M_.surprise_shocks = [];
M_.learnt_shocks = [];
M_.learnt_endval = [];
M_.heteroskedastic_shocks.Qvalue_orig = [];
M_.heteroskedastic_shocks.Qscale_orig = [];
M_.matched_irfs = {};
M_.matched_irfs_weights = {};
options_.linear = false;
options_.block = false;
options_.bytecode = false;
options_.use_dll = false;
options_.ramsey_policy = false;
options_.discretionary_policy = false;
M_.eq_nbr = 10;
M_.ramsey_orig_eq_nbr = 0;
M_.ramsey_orig_endo_nbr = 0;
M_.set_auxiliary_variables = exist(['./+' M_.fname '/set_auxiliary_variables.m'], 'file') == 2;
M_.epilogue_names = {};
M_.epilogue_var_list_ = {};
M_.orig_maximum_endo_lag = 2;
M_.orig_maximum_endo_lead = 3;
M_.orig_maximum_exo_lag = 0;
M_.orig_maximum_exo_lead = 0;
M_.orig_maximum_exo_det_lag = 0;
M_.orig_maximum_exo_det_lead = 0;
M_.orig_maximum_lag = 2;
M_.orig_maximum_lead = 3;
M_.orig_maximum_lag_with_diffs_expanded = 2;
M_.lead_lag_incidence = [
 1 8 18;
 2 9 19;
 3 10 0;
 0 11 20;
 4 12 0;
 5 13 0;
 6 14 0;
 0 15 21;
 0 16 22;
 7 17 0;]';
M_.nstatic = 0;
M_.nfwrd   = 3;
M_.npred   = 5;
M_.nboth   = 2;
M_.nsfwrd   = 5;
M_.nspred   = 7;
M_.ndynamic   = 10;
M_.dynamic_tmp_nbr = [0; 0; 0; 0; ];
M_.equations_tags = {
  1 , 'name' , 'L_GDP_GAP' ;
  2 , 'name' , 'DLA_CPI' ;
  3 , 'name' , 'RS' ;
  4 , 'name' , 'RR_GAP' ;
  5 , 'name' , 'RES_L_GDP_GAP' ;
  6 , 'name' , 'RES_DLA_CPI' ;
  7 , 'name' , 'RES_RS' ;
};
M_.mapping.L_GDP_GAP.eqidx = [1 2 3 ];
M_.mapping.DLA_CPI.eqidx = [2 3 4 ];
M_.mapping.RS.eqidx = [3 4 ];
M_.mapping.RR_GAP.eqidx = [1 4 ];
M_.mapping.RES_L_GDP_GAP.eqidx = [1 5 ];
M_.mapping.RES_DLA_CPI.eqidx = [2 6 ];
M_.mapping.RES_RS.eqidx = [3 7 ];
M_.mapping.SHK_L_GDP_GAP.eqidx = [5 ];
M_.mapping.SHK_DLA_CPI.eqidx = [6 ];
M_.mapping.SHK_RS.eqidx = [7 ];
M_.static_and_dynamic_models_differ = false;
M_.has_external_function = false;
M_.block_structure.time_recursive = false;
M_.block_structure.block(1).Simulation_Type = 1;
M_.block_structure.block(1).endo_nbr = 2;
M_.block_structure.block(1).mfs = 2;
M_.block_structure.block(1).equation = [ 5 6];
M_.block_structure.block(1).variable = [ 5 6];
M_.block_structure.block(1).is_linear = true;
M_.block_structure.block(1).NNZDerivatives = 4;
M_.block_structure.block(1).bytecode_jacob_cols_to_sparse = [1 2 3 4 ];
M_.block_structure.block(2).Simulation_Type = 6;
M_.block_structure.block(2).endo_nbr = 2;
M_.block_structure.block(2).mfs = 2;
M_.block_structure.block(2).equation = [ 7 10];
M_.block_structure.block(2).variable = [ 7 10];
M_.block_structure.block(2).is_linear = true;
M_.block_structure.block(2).NNZDerivatives = 5;
M_.block_structure.block(2).bytecode_jacob_cols_to_sparse = [0 0 1 2 ];
M_.block_structure.block(3).Simulation_Type = 8;
M_.block_structure.block(3).endo_nbr = 6;
M_.block_structure.block(3).mfs = 6;
M_.block_structure.block(3).equation = [ 3 1 2 4 8 9];
M_.block_structure.block(3).variable = [ 3 1 2 4 8 9];
M_.block_structure.block(3).is_linear = true;
M_.block_structure.block(3).NNZDerivatives = 20;
M_.block_structure.block(3).bytecode_jacob_cols_to_sparse = [1 2 3 7 8 9 10 11 12 14 15 16 17 18 ];
M_.block_structure.block(1).g1_sparse_rowval = int32([]);
M_.block_structure.block(1).g1_sparse_colval = int32([]);
M_.block_structure.block(1).g1_sparse_colptr = int32([]);
M_.block_structure.block(2).g1_sparse_rowval = int32([1 2 ]);
M_.block_structure.block(2).g1_sparse_colval = int32([1 2 ]);
M_.block_structure.block(2).g1_sparse_colptr = int32([1 2 3 ]);
M_.block_structure.block(3).g1_sparse_rowval = int32([1 2 3 1 4 1 2 3 3 4 5 6 2 1 3 4 5 2 6 1 ]);
M_.block_structure.block(3).g1_sparse_colval = int32([1 2 3 7 7 8 8 8 9 10 11 12 14 15 15 15 15 16 17 18 ]);
M_.block_structure.block(3).g1_sparse_colptr = int32([1 2 3 4 4 4 4 6 9 10 11 12 13 13 14 18 19 20 21 ]);
M_.block_structure.variable_reordered = [ 5 6 7 10 3 1 2 4 8 9];
M_.block_structure.equation_reordered = [ 5 6 7 10 3 1 2 4 8 9];
M_.block_structure.incidence(1).lead_lag = -1;
M_.block_structure.incidence(1).sparse_IM = [
 1 1;
 2 2;
 3 3;
 5 5;
 6 6;
 7 7;
 7 10;
 10 7;
];
M_.block_structure.incidence(2).lead_lag = 0;
M_.block_structure.incidence(2).sparse_IM = [
 1 1;
 1 5;
 2 1;
 2 2;
 2 6;
 3 1;
 3 3;
 3 7;
 4 3;
 4 4;
 5 5;
 6 6;
 7 7;
 8 8;
 9 9;
 10 10;
];
M_.block_structure.incidence(3).lead_lag = 1;
M_.block_structure.incidence(3).sparse_IM = [
 1 1;
 1 4;
 2 2;
 3 2;
 3 9;
 4 2;
 8 2;
 9 8;
];
M_.block_structure.dyn_tmp_nbr = 0;
M_.state_var = [5 6 7 10 3 1 2 ];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(10, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(3, 1);
M_.params = NaN(11, 1);
M_.endo_trends = struct('deflator', cell(10, 1), 'log_deflator', cell(10, 1), 'growth_factor', cell(10, 1), 'log_growth_factor', cell(10, 1));
M_.NNZDerivatives = [35; -1; -1; ];
M_.dynamic_g1_sparse_rowval = int32([1 2 3 5 6 7 10 7 1 2 3 2 3 4 4 1 5 2 6 3 7 8 9 10 1 2 3 4 8 1 9 3 5 6 7 ]);
M_.dynamic_g1_sparse_colval = int32([1 2 3 5 6 7 7 10 11 11 11 12 13 13 14 15 15 16 16 17 17 18 19 20 21 22 22 22 22 24 28 29 31 32 33 ]);
M_.dynamic_g1_sparse_colptr = int32([1 2 3 4 4 5 6 8 8 8 9 12 13 15 16 18 20 22 23 24 25 26 30 30 31 31 31 31 32 33 33 34 35 36 ]);
M_.lhs = {
'L_GDP_GAP'; 
'DLA_CPI'; 
'RS'; 
'RR_GAP'; 
'RES_L_GDP_GAP'; 
'RES_DLA_CPI'; 
'RES_RS'; 
'AUX_ENDO_LEAD_81'; 
'AUX_ENDO_LEAD_44'; 
'AUX_ENDO_LAG_6_1'; 
};
M_.static_tmp_nbr = [0; 0; 0; 0; ];
M_.block_structure_stat.block(1).Simulation_Type = 3;
M_.block_structure_stat.block(1).endo_nbr = 1;
M_.block_structure_stat.block(1).mfs = 1;
M_.block_structure_stat.block(1).equation = [ 5];
M_.block_structure_stat.block(1).variable = [ 5];
M_.block_structure_stat.block(2).Simulation_Type = 3;
M_.block_structure_stat.block(2).endo_nbr = 1;
M_.block_structure_stat.block(2).mfs = 1;
M_.block_structure_stat.block(2).equation = [ 6];
M_.block_structure_stat.block(2).variable = [ 6];
M_.block_structure_stat.block(3).Simulation_Type = 6;
M_.block_structure_stat.block(3).endo_nbr = 2;
M_.block_structure_stat.block(3).mfs = 2;
M_.block_structure_stat.block(3).equation = [ 7 10];
M_.block_structure_stat.block(3).variable = [ 7 10];
M_.block_structure_stat.block(4).Simulation_Type = 6;
M_.block_structure_stat.block(4).endo_nbr = 6;
M_.block_structure_stat.block(4).mfs = 6;
M_.block_structure_stat.block(4).equation = [ 3 4 1 2 8 9];
M_.block_structure_stat.block(4).variable = [ 2 3 4 1 8 9];
M_.block_structure_stat.variable_reordered = [ 5 6 7 10 2 3 4 1 8 9];
M_.block_structure_stat.equation_reordered = [ 5 6 7 10 3 4 1 2 8 9];
M_.block_structure_stat.incidence.sparse_IM = [
 1 4;
 1 5;
 2 1;
 2 6;
 3 1;
 3 2;
 3 3;
 3 7;
 3 9;
 4 2;
 4 3;
 4 4;
 5 5;
 6 6;
 7 7;
 7 10;
 8 2;
 8 8;
 9 8;
 9 9;
 10 7;
 10 10;
];
M_.block_structure_stat.tmp_nbr = 0;
M_.block_structure_stat.block(1).g1_sparse_rowval = int32([1 ]);
M_.block_structure_stat.block(1).g1_sparse_colval = int32([1 ]);
M_.block_structure_stat.block(1).g1_sparse_colptr = int32([1 2 ]);
M_.block_structure_stat.block(2).g1_sparse_rowval = int32([1 ]);
M_.block_structure_stat.block(2).g1_sparse_colval = int32([1 ]);
M_.block_structure_stat.block(2).g1_sparse_colptr = int32([1 2 ]);
M_.block_structure_stat.block(3).g1_sparse_rowval = int32([1 2 1 2 ]);
M_.block_structure_stat.block(3).g1_sparse_colval = int32([1 1 2 2 ]);
M_.block_structure_stat.block(3).g1_sparse_colptr = int32([1 3 5 ]);
M_.block_structure_stat.block(4).g1_sparse_rowval = int32([1 2 5 1 2 2 3 1 4 5 6 1 6 ]);
M_.block_structure_stat.block(4).g1_sparse_colval = int32([1 1 1 2 2 3 3 4 4 5 5 6 6 ]);
M_.block_structure_stat.block(4).g1_sparse_colptr = int32([1 4 6 8 10 12 14 ]);
M_.static_g1_sparse_rowval = int32([2 3 3 4 8 3 4 1 4 1 5 2 6 3 7 10 8 9 3 9 7 10 ]);
M_.static_g1_sparse_colval = int32([1 1 2 2 2 3 3 4 4 5 5 6 6 7 7 7 8 8 9 9 10 10 ]);
M_.static_g1_sparse_colptr = int32([1 3 6 8 10 12 14 17 19 21 23 ]);
M_.params(1) = 0.7;
b1 = M_.params(1);
M_.params(2) = 0.7;
b4 = M_.params(2);
M_.params(3) = 0.5;
a1 = M_.params(3);
M_.params(4) = 0.1;
a2 = M_.params(4);
M_.params(5) = 0.7;
g1 = M_.params(5);
M_.params(6) = 0.3;
g2 = M_.params(6);
M_.params(7) = 0.25;
g3 = M_.params(7);
M_.params(9) = 0.1;
rho_L_GDP_GAP = M_.params(9);
M_.params(8) = 0.2;
rho_DLA_CPI = M_.params(8);
M_.params(10) = 0.8;
rho_rs = M_.params(10);
M_.params(11) = 0.0;
rho_rs2 = M_.params(11);
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
oo_.steady_state(10)=oo_.steady_state(7);
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
M_.Sigma_e(3, 3) = 1;
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
if exist('options_mom_', 'var') == 1
  save([M_.dname filesep 'Output' filesep 'qpm_simpl1_results.mat'], 'options_mom_', '-append');
end
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
