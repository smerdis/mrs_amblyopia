%%
% A function to parse a gaba task data
% matrix into a long-format columnar text file.

function [col_str, output_txt] = parse_gaba(d)
% d: data structure (i.e. DAV.GABA.raw.ratio_CrOff)

population_conditions = {'Control', 'Amblyope'};
presentation_conditions = {'occ_binoc', 'occ_none'}; % which conditions (columns of allSub) will be used

% expected column names for the data in each of the above conditions
expected_lbls = {'motor','occ_binoc','occ_dichop','occ_none','mean_occ_all','mean_occ_stim','normBinoc','normDichop'};
sub_col = 1;
contrast_col = 2;
thresh_col = 3;
sem_col = 4;
relcontrast_col = 5; % ultimately our IV of interest

% define output columns
output_cols = {'subjName', 'Population', 'Presentation', 'GABA'};
col_str = strjoin(output_cols,'\t') ;
col_str = sprintf('%s\n',col_str);
output_txt = '' ;

% make sure the data is what we think it is!
dat_lbls = d.condOrder;
assert(isequal(dat_lbls,expected_lbls),'Columns of allSub differ from what was expected!') ;
assert(length(d.ambGroup)==length(d.allSub_clean), 'ambGroup and allSub have different length!') ;

df = d.allSub ;
[nr, ~] = size(df); % number of rows of data
for i_r = 1:nr
    subjName = lower(d.subID{i_r}); % initials [unique identifier]
    Population = population_conditions{d.ambGroup(i_r)+1};
    for i_p = 1:length(presentation_conditions)
        Presentation = presentation_conditions{i_p};
        pres_col = find(ismember(d.condOrder,Presentation));
        GABA = df(i_r, pres_col);
        obs_txt = sprintf('%s\t%s\t%s\t%.03f\n',subjName,Population,Presentation,GABA) ;
        output_txt = [output_txt obs_txt] ;
    end
end

end