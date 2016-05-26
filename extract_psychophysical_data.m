% extract_psychophysical_data.m
%
% script to extract data from the various matlab structs that have
% intermediate results computed by Kelly. getting this data and outputting
% it so it can be read by pandas/R/etc is a useful first step, since it
% forces me to understand what the data represents while putting it in a
% format suitable for modeling.

%% useful variables, condition definitions
clear all;

data_dir = '~/silver/MRS_amblyopia/analysis/psychophysics' ;
ss_fn = [data_dir '/SS_stair_summary_n35_032116_3.5Bin_none.mat'] ;
os_fn = [data_dir '/OS_stair_summary_n36_052316_3.5Bin_none.mat'] ;

output_fn = 'supp_data.txt';

%% Surround suppression task
load(ss_fn) ; % should yield 'SS'

% Call the parsing function
[ss_col_str, ss_txt] = parse_suppression(SS.all, 'SS') ;

%% Overlay suppression task
load(os_fn) ; % should yield 'OS'

% Call the parsing function
[os_col_str, os_txt] = parse_suppression(OS.all, 'OS') ;

%% Evaluate, merge

assert(strcmp(ss_col_str,os_col_str),'Suppression tasks have different columns!'); % both tasks should have the same columns
final_txt = [ss_col_str ss_txt os_txt];
[fid, msg] = fopen(output_fn, 'w');
fprintf(fid, final_txt);
