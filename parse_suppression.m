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

% define output columns
output_cols = {'Task', 'Presentation', 'Orientation', 'Eye', 'Population', 'Trace', 'Bin', 'ThreshElev', 'ThreshElev_SE', 'RelMaskContrast'};
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
        end
    end
end

end