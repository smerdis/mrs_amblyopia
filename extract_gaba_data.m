% extract_gaba_data.m
%
% script to extract data from the various matlab structs that have
% intermediate results computed by Kelly. getting this data and outputting
% it so it can be read by pandas/R/etc is a useful first step, since it
% forces me to understand what the data represents while putting it in a
% format suitable for modeling.

%% useful variables, condition definitions
clear all;

data_dir = '~/code/MRS_amblyopia/analysis/gaba' ;
gaba_fn = [data_dir '/MRS_group_summary_n31_110614.mat'] ;
%gaba_fn = [data_dir '/MRS_norm_group_summary_n31_110614.mat'] ;

output_fn = 'gaba_data_2021.txt';

%% Surround suppression task
load(gaba_fn) ; % should yield 'DAV' and some other (ignored) vars

% Call the parsing function
% [gaba_col_str, gaba_txt] = parse_gaba(DAV.GABA.norm.ratio_CrOff) ;
% [gaba_col_str, gaba_txt] = parse_gaba(DAV.GABA.raw.ratio_CrOff) ;
[gaba_col_str, gaba_txt] = parse_gaba(DAV.GABA.raw.ratio_CrSum) ;

%% Evaluate, merge

final_txt = [gaba_col_str gaba_txt];
[fid, msg] = fopen(output_fn, 'w');
fprintf(fid, final_txt);
