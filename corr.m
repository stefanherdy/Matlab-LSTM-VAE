% ePath =  'C:\Users\stefa\Desktop\Masterarbeit\Code\KellerVibroV4-1_1\mcodeKeller\maxloss';
% for k=1:23
%     Files = dir(fullfile(ePath,'*.mat'));
%     FileName = Files(k).name;
%     path = strcat(ePath, '\', FileName);
%     load(path)

maxLossListnew=[];
RList = [];
for i=1:10000
    Thresh = 0.8 +i/100;
    for j=1:length(maxLossList);
        if maxLossList(j) > Thresh
            maxLossListnew(j) = 1;
        else
            maxLossListnew(j) = 0;
        end
    end

        R = corrcoef(maxLossListnew,outl);
        RList(i) = R(1,2);
        v = i;
        Cov = R(1,2);

end

Maximum = max(RList)
