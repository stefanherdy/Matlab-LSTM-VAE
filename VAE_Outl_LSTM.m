%% Variational Autoencoders
% 
% Description : This script is made to apply a Variational Autoencoder to
% the Depth Data to find outliers in an unsupervised way.
%
% Author : 
%    Stefan Herdy
%    m01610562
%
% Date: 13.05.2020
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

%% Initial Settings
% Visualization of the latent space
% If plotLatentPoints is set to true, the user can click on a point in the
% latent space and the corresponding depth data gets plotted. numPoints is
% the number of the points the user wants to plot.
plotLatentPoints = false;
numPoints = 20;

% Visualisation of the VAE reconstruction and the real depth data
% If plotHighLoss is set to true, all points with a maximum loss or a
% mean absolute loss above the defined thresholds are plotted
plotHighLoss = false;
MAEThresh = 0.1;
MaxThresh = 0.1;

% Visualization of the outliers in the Latent space
% If plotLatentOutliers is set to true, the outliers with an outlierness 
% above LatentThresh in the latent space
% are plotted
plotLatentOutliers = false;
LatentThresh = 1;

% Define the phase that should be analysed.
% Is the phase is set to 1, the penetration phase is loaded. If the phase
% is set to 2, the compaction phase is loaded.
phase = 2;

% derivative: boolean value that defines wether the 1st derivative
% should be also imported or not
derivative = false;
% discont: boolean value that defines wether the nth discontinuity
% should be also imported or not 
discont = false;
% CxCont: int value that defines the nth discontinuity
CxCont = 2;
% degree: degree of the Vandermonde matrix for local smoothing
degree = 3;
% kernelSize: size of the convolution matrix
kernelSize = 2;


TSLength = 400;


%% Load Data


[DataMatrix,Labels,Outlier] = generateVAEInputLSTM(phase, derivative, discont, CxCont, degree, kernelSize, TSLength);


% Split into train and test data
TTsplit = 0.6;

[s1, s2] = size(DataMatrix);
split = TTsplit*s1;
split = round(split);


X_Train = DataMatrix(:,:);
X_Test = DataMatrix(:,:);



Y_Train = Labels(:,:);
Y_Test = Labels(:,:);



Outl_Test = Outlier(:,:);

YTrain = categorical(Y_Train);
YTest = categorical(Y_Test);

% Reshape the input to a 4D matrix


if derivative == true && discont == true 
    X_Train = reshape(X_Train', [TSLength,3, size(X_Train,1)]);
    X_Test = reshape(X_Test', [TSLength,3, size(X_Test,1)]);
   
end

if derivative == true && discont == false 
    X_Train = reshape(X_Train', [TSLength,2, size(X_Train,1)]);
    X_Test = reshape(X_Test', [TSLength,2, size(X_Test,1)]);
end

if derivative == false && discont == true 
    X_Train = reshape(X_Train', [TSLength,2, size(X_Train,1)]);
    X_Test = reshape(X_Test', [TSLength,2, size(X_Test,1)]);
end

if derivative == false && discont == false 
    X_Train = reshape(X_Train', [TSLength,1, size(X_Train,1)]);
    X_Test = reshape(X_Test', [TSLength,1, size(X_Test,1)]);
end


% Change the input to a dlarray
XTrain = dlarray(X_Train, 'TCB');
XTest = dlarray(X_Test, 'TCB');

%% Construct Network
% Autoencoders have two parts: the encoder and the decoder. The encoder takes 
% an input and outputs a compressed representation in the latent space (the encoding), which 
% is a vector of size latent_dim.

latentDim = 2;


encoderLG = layerGraph([
    sequenceInputLayer(1, 'Name', 'input1')
    lstmLayer(100, 'Name', 'lstm1')
    %lstmLayer(50, 'Name', 'lstm2')
    %lstmLayer(100, 'Name', 'lstm3')
    %lstmLayer(100, 'Name', 'lstm2', 'OutputMode', 'last')
    fullyConnectedLayer(2*latentDim, 'Name', 'fc1')
    ]);

decoderLG = layerGraph([
    sequenceInputLayer(latentDim, 'Name', 'input2')
    %imageInputLayer([1 1 latentDim], 'Name', 'input2', 'Normalization','none')
    lstmLayer(10, 'Name', 'lstm21')
    %lstmLayer(50, 'Name', 'lstm22')
    %lstmLayer(100, 'Name', 'lstm23')
    %fullyConnectedLayer(2*latentDim, 'Name', 'fc2')
    fullyConnectedLayer(1, 'Name', 'fc3')
    ]);
    


%% 
% To train both networks with a custom training loop , 
% convert the layer graphs to |dlnetwork| objects.

encoderNet = dlnetwork(encoderLG);
decoderNet = dlnetwork(decoderLG);


%% Specify Training Options
% Train on a GPU 

executionEnvironment = "auto";
%% 
% Set the training options for the network. 

numEpochs = 1;
miniBatchSize = 10;
lr = 1e-3;
numIterations = floor(split/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];
%% Train Model
% Train the model using a custom training loop.
% 
% For each iteration in an epoch:

for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = XTrain(:,idx,:);
        XBatch = dlarray(single(XBatch), 'CBT');
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);           
        end 
            
        [infGrad, genGrad] = dlfeval(...
            @modelGradients, encoderNet, decoderNet, XBatch);
        
        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
                genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
                infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
    end
    elapsedTime = toc;
    
    [z, zMean, zLogvar] = sampling_LSTM(encoderNet, XTest);
    xPred = sigmoid(forward(decoderNet, z));
    %[x, y, z] = size(XTest);
    %xPred = reshape(xPred, [x, y, z]);
    %testpred = xPred(1,7,:);
    elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
    elbo = mean(abs(elbo));
    disp("Epoch : "+epoch+" Test loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s") 
    
    
end

%% Compute an visualize the Results
sizev = 300;
%MAEloss = computeLoss(xPred, XTest, plotHighLoss,MAEThresh,MaxThresh,sizev);



[zMean, zLogVar] = visualizeLatentSpace_LSTM_Clif(XTest, encoderNet, decoderNet, TSLength);

%classifySVM(zMean,Outl_Test)
latentOutlierness = latentOutlier(XTest, encoderNet, plotLatentOutliers, LatentThresh,sizev);


%% Functions
% Compute the gradients of the loss with respect to the learnable paramaters 
% of both networks by calling the |dlgradient| function.

function [infGrad, genGrad] = modelGradients(encoderNet, decoderNet, x)
[z, zMean, zLogvar] = sampling_LSTM(encoderNet, x);
xPred = sigmoid(forward(decoderNet, z));
%xPred = reshape(xPred, [1 10 300]);
loss = ELBOloss(x, xPred, zMean, zLogvar);
loss = mean(abs(loss));
[genGrad, infGrad] = dlgradient(loss, decoderNet.Learnables, ...
    encoderNet.Learnables);
end

%% 
% The |ELBOloss| function takes the encodings of the means and the variances 
% returned by the |sampling| function, and uses them to compute the ELBO loss.

function elbo = ELBOloss(x, xPred, zMean, zLogvar)
squares = 0.5*(xPred-x).^2;
reconstructionLoss  = sum(squares, [1,2,3]);

KL = -.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);

elbo = mean(reconstructionLoss + KL); 
end
% Visualization Functions

function visualizeReconstruction(XTest,YTest, encoderNet, decoderNet)
f = figure;
figure(f)
title("Example ground truth image vs. reconstructed image")
for i = 1:2
    for c=0:9
        idx = iRandomIdxOfClass(YTest,c);
        X = XTest(:,:,:,idx);

        [z, ~, ~] = sampling(encoderNet, X);
        XPred = sigmoid(forward(decoderNet, z));
        
        X = gather(extractdata(X));
        XPred = gather(extractdata(XPred));

        comparison = [X, ones(size(X,1),1), XPred];
        subplot(4,5,(i-1)*10+c+1), imshow(comparison,[]),
    end
end
end

function idx = iRandomIdxOfClass(T,c)
idx = T == categorical(c);
idx = find(idx);
idx = idx(randi(numel(idx),1));
end


%% The |Generate| function tests the generative capabilities of the VAE. It initializes 
% a |dlarray| object containing 25 randomly generated encodings, passes them through 
% the decoder network, and plots the outputs.

function generate(decoderNet, latentDim)
randomNoise = dlarray(randn(1,1,latentDim,25),'SSCB');
generatedImage = sigmoid(predict(decoderNet, randomNoise));
generatedImage = extractdata(generatedImage);

f3 = figure;
figure(f3)
imshow(imtile(generatedImage, "ThumbnailSize", [100,100]))
title("Generated samples of digits")
drawnow
end
