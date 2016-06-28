% write example data for modeling to text file
load MeeseBaker09_dhb4cpd4hz ;

assert(length(dichoMask)==length(dichoTest));
dfn = 'dicho_egdata.txt' ;
[fid, msg] = fopen(dfn,'w') ;
doutput = sprintf('dichoMask\tdichoTest\n');
for i=1:length(dichoMask)
    doutput = [doutput sprintf('%.03f\t%.03f\n',dichoMask(i),dichoTest(i))];
end
fprintf(fid,doutput)

%% monocular

assert(length(monoMask)==length(monoTest));
dfn = 'mono_egdata.txt' ;
[fid, msg] = fopen(dfn,'w') ;
doutput = sprintf('monoMask\tmonoTest\n');
for i=1:length(monoMask)
    doutput = [doutput sprintf('%.03f\t%.03f\n',monoMask(i),monoTest(i))];
end
fprintf(fid,doutput)