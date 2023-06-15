%% LSTM Prediction
% 
% Description : This script is made to predict the future timesteps step by
% step. The computed loss is then used to have an "outlierness" of every
% point
% For every time stamp n  the next points n to n+t are predicted depending 
% on a specific number of previous datapoints n-t, where t is an empirical 
% number and the number of the previous time stamps that influence 
% the prediction
%
% Author : 
%    Stefan Herdy
%    m01610562
%
% Date: 30.04.2020
%  --------------------------------------------------
% (c) 2020, Stefan Herdy
%  Chair of Automation, University of Leoben, Austria
%  email: stefan.herdy@stud.unileoben.ac.at
%  --------------------------------------------------
%
%% Prepare Workspace
close all;
%clear;
% Add the path to the used functions

addpath(genpath(['..',filesep,'mcodeKellerLib']));

%% Load Data
% Ask user for site folder
%myDir = uigetdir( cd, 'Select the folder for the site'); 

myDir = 'C:\Users\stefa\Desktop\Masterarbeit\Code\sites\SeestadtAspern'


            
%% Settings
% LoadNet describes if the a trained net should be used or if the network
% should be trained on the training data
LoadNet = true;
% SaveNet describes if the trained network should be saved or not
SaveNet = false;

timelist = [12, 12, 12, 15, 15, 15, 18, 18, 18, 21, 21, 3, 3, 3, 3, 3, 3, 6, 6, 6, 9, 9, 9]
% Define how long the predicted sequences should be
for k=1:length(timelist)

    TimeStep = timelist(k);
    % Define a Threshold for the maximum loss. If the max Loss for a point is

    % above this Threshold, the data gets plotted for visual inspection
    LossThreshDepth = 0;
    LossThreshDisc = 0;
    % Discontinuity defines wheter the discontinuity data should be used for the
    % prediction or not
    Discontinuity = true;
    % The pahase defines wich phase should be analysed. 1 = compaction phase, 2 =
    % penetration phase.
    phase = 2;

    if Discontinuity == true
        LossThresh = LossThreshDisc;
    else
        LossThresh = LossThreshDepth;
    end

    % Call generatePredInput to load the train data
    [XTrain, YTrain] = generatePredInputDisc(phase, myDir, TimeStep, Discontinuity)

    %% Define LSTM Network Architecture
    % An LSTM regression is used to predict a sequence of timesteps based on
    % previous timesteps

    inputSize = 1;
    numHiddenUnits = 1500;
    numResponses = 1;

    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits)
        %bilstmLayer(numHiddenUnits)
        %bilstmLayer(numHiddenUnits)
        %fullyConnectedLayer(10)
        fullyConnectedLayer(numResponses)
        regressionLayer]

    %% Specify the training options. 

    maxEpochs = 1;
    miniBatchSize = 8;

    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'MiniBatchSize',miniBatchSize, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',10, ...
        'LearnRateDropFactor',0.1, ...
        'Verbose',0, ...
        'Plots','training-progress');
    %% Train the network
    % Train the network based on the above defined settings
    if LoadNet == false;
        prednet = trainNetwork(XTrain,YTrain,layers,options);
    end
    %if LoadNet == true;
    %    load prednet;
    %end

    s1 = int2str(TimeStep);
    s2 = int2str(numHiddenUnits);
    s3 = int2str(maxEpochs);

    name = strcat('PredDisc_highdrop','_',s1,'_',s2,'_',s3);
    name = convertCharsToStrings(name);
    name = string(name);

    ePath =  'C:\Users\stefa\Desktop\Masterarbeit\Code\KellerVibroV4-1_1\mcodeKeller\prednets';

    Files = dir(fullfile(ePath,'*.mat'));

    
    FileName = Files(k).name;
%   %dFileName = dFiles(k).name;
    path = strcat(ePath, '\', FileName);
%   %dpath = strcat(dPath, '\', dFileName);
    load(path)
%   %load(dpath)



    if SaveNet == true;
        save(name, 'prednet');

    end

    %% Test LSTM Network
    % call makePrediction to test the trained or loaded LSTM network
    makePredictionDisc(phase, myDir, TimeStep, prednet, LossThresh, Discontinuity, FileName);

    %makePrediction(phase, myDir, TimeStep, prednet, LossThresh);

    
 end



