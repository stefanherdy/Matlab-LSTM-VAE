%% First LSTM Tests
% 
% Description : This script is made for first tests for applying LSTM Nets
% on our data and perform site classification
%
% Author : 
%    Stefan Herdy
%    m01610562
%
% Date: 16.04.2020
%  --------------------------------------------------
% (c) 2020, Stefan Herdy
%  Chair of Automation, University of Leoben, Austria
%  email: stefan.herdy@stud.unileoben.ac.at
%  --------------------------------------------------
%
%% Prepare Workspace
close all;
clear;
addpath(genpath(['..',filesep,'mcodeKellerLib']));
%% Settings
% LoadNet describes if the a trained net should be used or if the network
% should be trained on the training data
LoadNet = false;
% SaveNet describes if the trained network should be saved or not
SaveNet = true;
%% Load Data
[DataMatrix,Outlier] = generateOutlInput()


%% Prepare data
X = {};
Y = {};

% Shuffle the data
[r, c] = size(DataMatrix);
rand = randperm(r);

for i = 1:r;
    plc = rand(i);
    X{plc,1} = DataMatrix{i,1};
    Y{plc,1} = Outlier{i,1};
    
end

% Define the proportion between test an train data
TTsplit = 0.6

% Split into train and test data
[s1 s2] = size(X);
split = TTsplit*s1;
split = round(split);



X_Train = X(1:split,1);
X_Test = X(split:end,1);

Y_Train = Y(1:split,1);
Y_Test = Y(split:end,1);

Y_Train = categorical(Y_Train);
Y_Test = categorical(Y_Test);

%% Define LSTM Network Architecture

inputSize = 9;
numHiddenUnits = 100;
numClasses = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%% Specify the training options. 

maxEpochs = 5;
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
% Train the network based on the above defined settings
if LoadNet == false;
    classnet = trainNetwork(X_Train,Y_Train,layers,options);
end
if LoadNet == true;
    load classnet;
end
if SaveNet == true;
    save classnet;
end

%% Test LSTM Network

% Classify the test data.
Y_Pred = classify(classnet,X_Test, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

%% Calculate the classification accuracy of the predictions.

acc = sum(Y_Pred == Y_Test)./numel(Y_Test)

% Plot the results as confusion matrix
C = confusionmat(Y_Test,Y_Pred);
CC = confusionchart(C)
CC.Title = 'Site Classification using LSTM Net';
%CC.RowSummary = 'row-normalized';
%CC.ColumnSummary = 'column-normalized';
