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
clear;
% Add the path to the used functions

addpath(genpath(['..',filesep,'mcodeKellerLib']));

%% Settings
% LoadNet describes if the a trained net should be used or if the network
% should be trained on the training data
LoadNet = false;
% SaveNet describes if the trained network should be saved or not
SaveNet = true;
% Define how long the predicted sequences should be
TimeStep = 10;
% Define a Threshold for the maximum loss. If the max Loss for a point is
% above this Threshold, the data gets plotted for visual inspection
LossThresh = 0.8;

%% Load Data
% Ask user for site folder
myDir = uigetdir( cd, 'Select the folder for the site'); 

% Call generatePredInput to load the train data
[XTrain, YTrain] = generatePredInput(2, myDir, TimeStep)

%% Define LSTM Network Architecture
% An LSTM regression is used to predict a sequence of timesteps based on
% previous timesteps

inputSize = 1;
numHiddenUnits = 1000;
numResponses = 1;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer]

%% Specify the training options. 

maxEpochs = 5;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',miniBatchSize, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train the network
% Train the network based on the above defined settings
if LoadNet == false;
    prednet = trainNetwork(XTrain,YTrain,layers,options);
end
if LoadNet == true;
    load prednet;
end
if SaveNet == true;
    save prednet;
end

%% Test LSTM Network
% call makePrediction to test the trained or loaded LSTM network
makePrediction(2, myDir, TimeStep, prednet, LossThresh);






