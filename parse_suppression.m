%%
% It would be best to write a function to parse a suppression task data
% matrix into a long-format columnar text file, then use it on OS and SS.

function [col_str, output_txt] = parse_suppression(d, task_id)
% d: data structure (i.e. SS.all, OS.all)
% task_id: code to use for this task (string) - provided by calling
% script, which is aware of multiple tasks. this function only cares about
% one.

population_conditions = {'Control', 'Amblyope'};
presentation_conditions = {'nMono', 'nDicho'};
orient_conds = {'Iso', 'Cross'};
eye_conds = {'Nde', 'De'};

% expected column names for the data in each of the above conditions
expected_lbls = {'subj'  'maskCtr'  'thresh'  'stderr'  'relMaskCtr'};
sub_col = 1;
contrast_col = 2;
thresh_col = 3;
sem_col = 4;
relcontrast_col = 5; % ultimately our IV of interest

% define output columns
output_cols = {'subjName', 'Population', 'Task', 'Presentation', 'Orientation', 'Eye', 'MaskContrast', 'ThreshElev', 'ThreshElev_SE', 'RelMaskContrast'};
col_str = strjoin(output_cols,'\t') ;
col_str = sprintf('%s\n',col_str) ;
output_txt = '' ;

for i_pc = presentation_conditions
    % select the data corresponding to the presentation condition
    if strcmp(i_pc, 'nMono') dat_p = d.nMono;
    elseif strcmp(i_pc, 'nDicho') dat_p = d.nDicho; end
    
    for i_oc = orient_conds
        % select the data corresponding to the orientation condition
        if strcmp(i_oc, 'Iso') dat_po = dat_p.Iso;
        elseif strcmp(i_oc, 'Cross') dat_po = dat_p.Cross; end
        
        for i_ec = eye_conds
            % select the data corresponding to the eye that viewed
            if strcmp(i_ec, 'Nde') dat_poe = dat_po.Nde;
            elseif strcmp(i_ec, 'De') dat_poe = dat_po.De; end
                        
            % make sure the data is what we think it is!
            dat_lbls = dat_poe.datLabels;
            assert(isequal(dat_lbls,expected_lbls),'Columns of allSub differ from what was expected!') ;
            assert(length(dat_poe.ambOrder)==length(dat_poe.subNms), 'ambOrder and subNms have different length!') ;
            
            df = dat_poe.allSub ;
            [nr, ~] = size(df); % number of rows of data
            for i_r = 1:nr
                subjID = df(i_r, sub_col) ;
                subjName = dat_poe.subNms{subjID} ;% initials [unique identifier], order in subNms corresponds to first col (subjID) of df
                Population = population_conditions{dat_poe.ambOrder(subjID)+1} ;
                Presentation = i_pc{1} ;
                Orientation = i_oc{1} ;
                Eye = i_ec{1} ;
                MaskContrast = df(i_r, contrast_col);
                ThreshElev = df(i_r, thresh_col);
                ThreshElev_SE = df(i_r, sem_col);
                RelMaskContrast = df(i_r, relcontrast_col);
                obs_txt = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%.03f\t%.03f\t%.03f\t%.03f\n',subjName,Population,task_id,Presentation,Orientation,Eye,MaskContrast,ThreshElev,ThreshElev_SE,RelMaskContrast);
                output_txt = [output_txt obs_txt] ;
            end
        end
    end
end

end