%% First LSTM Tests
% 
% Description : This script is made for first tests for applying LSTM Nets
% on our data and perform site classification
%
% Author : 
%    Stefan Herdy
%    m01610562
%
% Date: 10.04.2020
%  --------------------------------------------------
% (c) 2020, Stefan Herdy
%  Chair of Automation, University of Leoben, Austria
%  email: stefan.herdy@stud.unileoben.ac.at
%  --------------------------------------------------
%
%% Prepare Workspace
close all;
clear;
%
%% Load Data
[DataMatrix,Labels] = generateSiteInput()

%% Prepare data
X = {};
Y = {};

[s1, s2] = size(DataMatrix);

% Shuffle the data
rand = randperm(s1);

for i = 1:s1;
    plc = rand(i);
    X{plc,1} = DataMatrix{i,1};
    Y{plc,1} = Labels{i,1};
    
end

% Split into train and test data
[s1 s2] = size(X);
split = 0.9*s1;
split = round(split);



X_Train = X(1:split,1);
X_Test = X(split:end,1);

Y_Train = Y(1:split,1);
Y_Test = Y(split:end,1);

Y_Train = categorical(Y_Train);
Y_Test = categorical(Y_Test);

%% Define LSTM Network Architecture

inputSize = 9;
numHiddenUnits = 50;
numClasses = 4;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%% Specify the training options. 

maxEpochs = 10;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train the network

net = trainNetwork(X_Train,Y_Train,layers,options);
%% Test LSTM Network

% Classify the test data.


Y_Pred = classify(net,X_Test, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

%% Calculate the classification accuracy of the predictions.

acc = sum(Y_Pred == Y_Test)./numel(Y_Test)

C = confusionmat(Y_Test,Y_Pred);
CC = confusionchart(C)
CC.Title = 'Site Classification using LSTM Net';
%CC.RowSummary = 'row-normalized';
%CC.ColumnSummary = 'column-normalized';
