% extract_fmri.m
%
% script to extract data from the various matlab structs that have
% intermediate results computed by Kelly. getting this data and outputting
% it so it can be read by pandas/R/etc is a useful first step, since it
% forces me to understand what the data represents while putting it in a
% format suitable for modeling.

%% useful variables, condition definitions
clear all;

load '~/silver/MRS_amblyopia/mrs_amblyopia/ForArjun/collDiffsN31_5.1.16.mat'
load '~/silver/MRS_amblyopia/mrs_amblyopia/ForArjun/subjDemosN31_5.1.16.mat'

output_fn = 'fmri_data.txt';

% Call the parsing function
[fmri_col_str, fmri_txt] = parse_fmri(inits, diffB_D) ;

%% Evaluate, merge

final_txt = [fmri_col_str fmri_txt];
[fid, msg] = fopen(output_fn, 'w');
fprintf(fid, final_txt);