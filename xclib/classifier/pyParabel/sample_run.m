addpath( genpath( '../Tools' ) );

dataset = 'EUR-Lex';
data_dir = fullfile( '..', 'Sandbox', 'Data', dataset );
results_dir = fullfile( '..', 'Sandbox', 'Results', dataset );
model_dir = fullfile( '..', 'Sandbox', 'Results', dataset, 'model' );

trn_ft_mat = read_text_mat( fullfile( data_dir, 'trn_X_Xf.txt' ) );
trn_lbl_mat = read_text_mat( fullfile( data_dir, 'trn_X_Y.txt' ) );
tst_ft_mat = read_text_mat( fullfile( data_dir, 'tst_X_Xf.txt' ) );
tst_lbl_mat = read_text_mat( fullfile( data_dir, 'tst_X_Y.txt' ) );

% create the model folder
create_recur_dir( model_dir );

% training
% Reads training features (into trn_ft_mat), training labels (into trn_lbl_mat), and writes FastXML model (to model_dir);
param = [];
param.num_thread = 1;
param.start_tree = 0;
param.num_tree = 3;
param.bias_feat = 1.0;
param.classifier_cost = 1.0;
param.max_leaf = 100;
param.classifier_threshold = 0.1;
param.centroid_threshold = 0;
param.clustering_eps = 0.0001;
param.classifier_maxitr = 20;
param.classifier_kind = 0;
param.quiet = 0;
parabel_train( trn_ft_mat, trn_lbl_mat, model_dir, param );

% testing
% Reads test features (into tst_ft_mat), FastXML model (in model_dir), and writes test label scores (into score_mat)
param = [];
param.num_tree = 3;
score_mat = parabel_predict( tst_ft_mat, model_dir, param );

% performance evaluation 
wts = inv_propensity(trn_lbl_mat,0.55,1.5);
get_all_metrics(score_mat, tst_lbl_mat, wts);