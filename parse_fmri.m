function [col_str, output_txt ] = parse_fmri(inits, diffs)
% Accepts a strucutre of subj initials and per-subject values
% (e.g. binocular-dichoptic BOLD amplitude) and formats it for
% easy pandas/R/etc parsing.
assert(size(inits, 1)==size(diffs.V1,2)) % this is how the original files are structured

output_cols = {'subjName', 'BinSumDiffV1', 'BinSumDiffV2d', 'BinSumDiffV2v', 'BinSumDiffV3d', 'BinSumDiffV3v'};
col_str = strjoin(output_cols,'\t') ;
col_str = sprintf('%s\n',col_str);
output_txt = '' ;

nr = size(inits, 1); % number of rows of data
for i_r = 1:nr
    subjName = lower(inits{i_r}); % initials [unique identifier]
    obs_txt = sprintf('%s\t%.03f\t%.03f\t%.03f\t%.03f\t%.03f\n',subjName,diffs.V1(i_r),diffs.V2d(i_r),diffs.V2v(i_r),diffs.V3d(i_r),diffs.V3v(i_r));
    output_txt = [output_txt obs_txt] ; 
end
end

