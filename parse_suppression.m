%%
% It would be best to write a function to parse a suppression task data
% matrix into a long-format columnar text file, then use it on OS and SS.

function [col_str, output_txt] = parse_suppression(d, task_id, grouped)
% d: data structure (i.e. SS.all, OS.all)
% task_id: code to use for this task (string) - provided by calling
% script, which is aware of multiple tasks. this function only cares about
% one.
% grouped: 0|1, where 0 = get individual subjects and 1 = get group means

population_conditions = {'Control', 'Amblyope'};
presentation_conditions = {'nMono', 'nDicho'};
orient_conds = {'Iso', 'Cross'};
eye_conds = {'Nde', 'De'};

% define output columns
if grouped
    output_cols = {'Task', 'Presentation', 'Orientation', 'Eye', 'Population', 'Trace', 'Bin', 'ThreshElev', 'ThreshElev_SE', 'RelMaskContrast'};
else % if we want data for each subject, add a column to identify them (initials)
    output_cols = {'Subject', 'Task', 'Presentation', 'Orientation', 'Eye', 'Population', 'MaskContrast', 'ThreshElev', 'ThreshElev_SE', 'RelMaskContrast', 'Trace', 'BaselineThresh'};
    
    % also let's define what we expect the individual subject data to look like
    expected_lbls = {'subj'  'maskCtr'  'thresh'  'stderr'  'relMaskCtr'};
    sub_col = 1;
    contrast_col = 2;
    thresh_col = 3;
    sem_col = 4;
    rel_contrast_col = 5;
end
col_str = strjoin(output_cols,'\t') ;
col_str = sprintf('%s\n',col_str) ;
output_txt = '' ;

for i_pc = presentation_conditions
    % select the data corresponding to the presentation condition
    if strcmp(i_pc, 'nMono') dat_p = d.nMono;
    elseif strcmp(i_pc, 'nDicho') dat_p = d.nDicho; end
    
    % select the appropriate BaselineThresh sub-struct
    if any(strcmp('avgBase', fieldnames(d))) % if d.avgBase exists, use it (it should exist for OS)
        dat_base = d.avgBase ;
    else % otherwise use d.base1, which should exist for SS.
        dat_base = d.base1 ;
    end
    
    for i_oc = orient_conds
        % select the data corresponding to the orientation condition
        if strcmp(i_oc, 'Iso') dat_po = dat_p.Iso;
        elseif strcmp(i_oc, 'Cross') dat_po = dat_p.Cross; end
        
        for i_ec = eye_conds
            % select the data corresponding to the eye that viewed
            if strcmp(i_ec, 'Nde')
                dat_poe = dat_po.Nde;
                dat_base_eye = dat_base.Nde ;
            elseif strcmp(i_ec, 'De')
                dat_poe = dat_po.De;
                dat_base_eye = dat_base.De ;
            end
            
            % update 11/10/16: want this function to handle individual data
            % (grouped = 0) as well as get means like it has been doing.
            if grouped
                % This is where the changes due to fitting the updated data
                % structs Eunice provided on 6/13/16 begin.
                for i_pop = population_conditions
                    if strcmp(i_pop, 'Control') dat_poep = dat_poe.con_binMean.relMaskCtr ;
                    elseif strcmp(i_pop, 'Amblyope') dat_poep = dat_poe.amb_binMean.relMaskCtr ; end
                    
                    % at this point we have a 6 x n_bins data structure, with
                    % the relevant mask contrast value (x-axis) in row #6 (per
                    % EY email 6/13/16). Also thresh (y-axis) is #2, sem #3.
                    df = dat_poep.data ;
                    [nr, nbins] = size(df); % number of rows of data
                    for i_bin = 1:nbins
                        Presentation = i_pc{1} ;
                        Orientation = i_oc{1} ;
                        Eye = i_ec{1} ;
                        Population = i_pop{1} ;
                        trace = [Population '-' Eye];
                        ThreshElev = df(2, i_bin);
                        ThreshElev_SE = df(3, i_bin);
                        RelMaskContrast = df(6, i_bin);
                        obs_txt = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%d\t%.03f\t%.03f\t%.03f\n',task_id,Presentation,Orientation,Eye,Population,trace,i_bin,ThreshElev,ThreshElev_SE,RelMaskContrast);
                        output_txt = [output_txt obs_txt] ;
                    end
                end
            else
                % make sure the data is what we think it is!
                dat_lbls = dat_poe.datLabels;
                assert(isequal(dat_lbls,expected_lbls),'Columns of allSub differ from what was expected!') ;
                assert(length(dat_poe.ambOrder)==length(dat_poe.subNms), 'ambOrder and subNms have different length!') ;

                df = dat_poe.allSub ;
                [nr, ~] = size(df); % number of rows of data
                for i_r = 1:nr
                    subjID = df(i_r, sub_col) ;
                    subjName = dat_poe.subNms{subjID} ;% initials [unique identifier], order in subNms corresponds to first col (subjID) of df
                    subj_idx_in_baseline_struct = find(strcmp(dat_base_eye.subNms, subjName)) ;
                    subj_baseline_thresh = dat_base_eye.allSub(subj_idx_in_baseline_struct, 3) ;
                    Population = population_conditions{dat_poe.ambOrder(subjID)+1} ;
                    Presentation = i_pc{1} ;
                    Orientation = i_oc{1} ;
                    Eye = i_ec{1} ;
                    MaskContrast = df(i_r, contrast_col);
                    ThreshElev = df(i_r, thresh_col);
                    ThreshElev_SE = df(i_r, sem_col);
                    RelMaskContrast = df(i_r, rel_contrast_col);
                    trace = [Population '-' Eye];
                    obs_txt = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%.03f\t%.03f\t%.03f\t%.03f\t%s\t%.03f\n',subjName,task_id,Presentation,Orientation,Eye,Population,MaskContrast,ThreshElev,ThreshElev_SE,RelMaskContrast,trace,subj_baseline_thresh);
                    output_txt = [output_txt obs_txt] ;
                end
            end
        end
    end
end

end