myDir = 'C:\Users\stefa\Desktop\Masterarbeit\Code\sites\SeestadtAspern'
path = fullfile(myDir, '\pointData\mat');
myFiles = dir(fullfile(path,'*.mat'));

segmentMetaData.PullDownForceMin = 5;
segmentMetaData.VibratorAmperageMin = 30;
segmentMetaData.startDepthMax = 0.5;
outl = zeros(1,5);

for j = 1:length(myFiles)
%for j = 1:5

    n = string(myFiles(j).name);
    s = strcat(path,'\',n);

    % Load the files
    file = load(s);
    
    phase = 2;
    phases = segmentPhases( file.data, segmentMetaData );

    values = file.data.Depth;
    time = file.data.Time;
    
    if phase == 1;
        values = values(phases.penetrationStart:phases.compactionStart);
        time = time(phases.penetrationStart:phases.compactionStart);
    end
    if phase == 2;
        values = values(phases.compactionStart:phases.processEnd);
        time = time(phases.compactionStart:phases.processEnd);
    end

    
    values = values';
    time = time';
    if maxLossList(j) > 1.2
        plot(time,values,'r')
        grid on
        outlval = input('Outlier or not')
        if outlval == 1
            outl(1,j) = 1;
        end
        if outlval == 0
            outl(1,j) = 0;
        end
    elseif maxLossList(j) < 0.1.23
        plot(time,values,'b')
        grid on
        outlval = input('Outlier or not')
        if outlval == 1
            outl(1,j) = 1;
        end
        if outlval == 0
            outl(1,j) = 0;
        end
    else
        plot(time,values,'k')
        grid on
        outlval = input('Outlier or not')
        if outlval == 1
            outl(1,j) = 1;
        end
        if outlval == 0
            outl(1,j) = 0;
        end
    end
end
save('Outlierlist', 'outl')