clear all; close all;
[fileNm, pathNm] = uigetfile('*.mat','Select MRS summary mat file');
load([pathNm fileNm]);

%% subjects to plot
subID= {'AC', 'AS', 'BI', 'CH', 'CS', 'EM', 'GD', 'JV', 'KS', 'LI', 'MG', 'PS', 'RN', 'SD', 'AI', 'CR', 'CT', 'DL', 'EI', 'ES', 'GM', 'JD', 'JM', 'KW', 'LS', 'MK', 'MS', 'NL', 'YL'};
numSub=size(subID,2);

%% parameters for plotting
xaxRange=[220 600];
maxFreq = 2000/123.2556;
nPoints=4096;
freqRange =fliplr((nPoints+1-(1:1:nPoints))/nPoints*maxFreq+4.7-maxFreq/2.0);

%% normalize MOTOR spectra to creatine and average across subjects
 ambDat=[]; conDat=[]; clear freqRangeInd;
 for wSub=1:numSub
     specSubInd= find(strcmpi(subID{wSub},DAV.spectra.motor.subID));
     spec= fliplr(real(DAV.spectra.motor.diff(specSubInd,:)));
     xrange= DAV.spectra.motor.dav_xrange(specSubInd,:);

     freqRangeInd(wSub,:)=freqRange(xrange); %compile for group average

     crCond= find(strcmpi('motor',DAV.Cr.raw.auc_sum.condOrder));
     crSubInd=find(strcmpi(subID{wSub},DAV.Cr.raw.auc_sum.subID));
     Cr= DAV.Cr.raw.auc_sum.allSub(crSubInd,crCond);

     if DAV.spectra.motor.ambGroup(specSubInd)==1 %amb
         ambDat=[ambDat; spec(xrange)./Cr];
     else
         conDat=[conDat; spec(xrange)./Cr];
     end
 end
 nDiffSpec.motor.avgFreqRange= mean(freqRangeInd,1);

 nDiffSpec.motor.con.avg= mean(conDat);
 nDiffSpec.motor.amb.avg= mean(ambDat);
 nDiffSpec.motor.con.ste= std(conDat)./sqrt(size(conDat,1));
 nDiffSpec.motor.amb.ste= std(ambDat)./sqrt(size(ambDat,1));

 %% normalize all OCC spectra to creatine and average across OCC conditions AND subjects
 occCond={'occ_binoc','occ_none','occ_dichop'};
 ambDat=[]; conDat=[]; freqRangeInd=[];
 for wCond=1:size(occCond,2)
    for wSub=1:numSub
        specCond= occCond{wCond};
        specSubInd= find(strcmpi(subID{wSub},DAV.spectra.(specCond).subID));
        spec= fliplr(real(DAV.spectra.(specCond).diff(specSubInd,:)));
        xrange= DAV.spectra.(specCond).dav_xrange(specSubInd,:);
        freqRangeInd=[freqRangeInd; freqRange(xrange)]; %compile for group average

        crCond= find(strcmpi(occCond{wCond},DAV.Cr.raw.auc_sum.condOrder));
        crSubInd=find(strcmpi(subID{wSub},DAV.Cr.raw.auc_sum.subID));
        Cr= DAV.Cr.raw.auc_sum.allSub(crSubInd,crCond);

        if DAV.spectra.(specCond).ambGroup(specSubInd)==1 %amb
            ambDat=[ambDat; spec(xrange)./Cr];
        else
            conDat=[conDat; spec(xrange)./Cr];
        end
    end
 end
 nDiffSpec.all_occ.avgFreqRange= mean(freqRangeInd,1);

 nDiffSpec.all_occ.con.avg= mean(conDat);
 nDiffSpec.all_occ.amb.avg= mean(ambDat);

 nDiffSpec.all_occ.con.ste= std(conDat)./sqrt(size(conDat,1)/3); %divide by 3 since there are 3 occ conds (only want to use numSubs, not numSubs*numConds)
 nDiffSpec.all_occ.amb.ste= std(ambDat)./sqrt(size(ambDat,1)/3);

%% plot normalized spectra- compare groups for each condition (motor vs allOcc)
%cmap=[1.0 0.498 0.055; 0.121 0.466 0.706]; %blue for AMB, orange for CON
cmap=[0.121 0.466 0.706; 1.0 0.498 0.055];
cond={'motor','all_occ'};
for wCond=1:size(cond,2)
    figure(wCond);
    conDat=nDiffSpec.(cond{wCond}).con.avg;
    conErr=nDiffSpec.(cond{wCond}).con.ste;
    
    ambDat=nDiffSpec.(cond{wCond}).amb.avg;
    ambErr=nDiffSpec.(cond{wCond}).amb.ste;
    tFreqRange= nDiffSpec.(cond{wCond}).avgFreqRange;

    %boundedline(101:601,conDat(101:601),conErr(101:601),'cmap',cmap(1,:),'alpha');
    boundedline(1:601,conDat,conErr,'cmap',cmap(2,:),'alpha');
    %boundedline(101:601,ambDat(101:601),ambErr(101:601),'cmap',cmap(2,:),'alpha');
    boundedline(1:601,ambDat,ambErr,'cmap',cmap(1,:),'alpha')
    
    axis([xaxRange -.001 .004])
    %legend('PWA','NSP'); legend('boxoff');
    %title([cond{wCond}],'interpreter','none');
    xlabel('Frequency (ppm)', 'FontSize', 16); ylabel('Relative intensity', 'FontSize', 16);

    t=get(gca,'XTick');
    xticValues=tFreqRange(t);
    xticValues_round = round(xticValues, 2);
    set(gca,'xticklabel',xticValues_round);
end