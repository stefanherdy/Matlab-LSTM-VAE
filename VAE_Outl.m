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
numPoints = 5;

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
discont = true;
% CxCont: int value that defines the nth discontinuity
CxCont = 1;
% degree: degree of the Vandermonde matrix for local smoothing
degree = 3;
% kernelSize: size of the convolution matrix
kernelSize = 2;

% If partial is set to true, the single discontinuity or derivative bits
% are imported and analysed
partial = false;


%% Load Data

if partial == true
    sizev = 30;
    [DataMatrix,Labels,Outlier,TestArray] = generateVAEPartialInput(phase, derivative, discont, CxCont, degree, kernelSize,sizev);
else
    [DataMatrix,Labels,Outlier] = generateVAEInput(phase, derivative, discont, CxCont, degree, kernelSize);
end

% Split into train and test data
TTsplit = 0.6;

[s1, s2] = size(DataMatrix);
split = TTsplit*s1;
split = round(split);


%X_Train = DataMatrix(1:split,:);
%X_Test = DataMatrix(split:end,:);

X_Train = DataMatrix(:,:);
X_Test = DataMatrix(:,:);

%Y_Train = Labels(1:split,:);
%Y_Test = Labels(split:end,:);

Y_Train = Labels(:,:);
Y_Test = Labels(:,:);

%Outl_Test = Outlier(split:end,:);

Outl_Test = Outlier(:,:);

YTrain = categorical(Y_Train);
YTest = categorical(Y_Test);

% Reshape the input to a 4D matrix

if partial == true
    d = size(X_Train,2)/sizev;
    X_Train = reshape(X_Train', [sizev,1,1,d]);
    X_Test = reshape(X_Test', [sizev,1,1,d]);
    X_Train = X_Train(:,:,:,1:500);
    imageSize = [sizev 1 1];
end

if derivative == true && discont == true && partial ==false
    X_Train = reshape(X_Train', [300,3,1, size(X_Train,1)]);
    X_Test = reshape(X_Test', [300,3,1, size(X_Test,1)]);
    sizev = 300;
    imageSize = [sizev 3 1];
end

if derivative == true && discont == false && partial ==false
    X_Train = reshape(X_Train', [300,2,1, size(X_Train,1)]);
    X_Test = reshape(X_Test', [300,2,1, size(X_Test,1)]);
    sizev = 300;
    imageSize = [sizev 2 1];
end

if derivative == false && discont == true && partial ==false
    X_Train = reshape(X_Train', [300,2,1, size(X_Train,1)]);
    X_Test = reshape(X_Test', [300,2,1, size(X_Test,1)]);
    sizev = 300;
    imageSize = [sizev 2 1];
end

if derivative == false && discont == false && partial ==false
    X_Train = reshape(X_Train', [300,1,1, size(X_Train,1)]);
    X_Test = reshape(X_Test', [300,1,1, size(X_Test,1)]);
    sizev = 300;
    imageSize = [sizev 1 1];
end


% Change the input to a dlarray
XTrain = dlarray(X_Train, 'SSCB');
XTest = dlarray(X_Test, 'SSCB');

%% Construct Network
% Autoencoders have two parts: the encoder and the decoder. The encoder takes 
% an input and outputs a compressed representation in the latent space (the encoding), which 
% is a vector of size latent_dim.

latentDim = 3;

if partial == true
    convsize =30;
    stride1 = 5;
    stride2 = 6;
else
    convsize = 7;
    stride1 = 15;
    stride2 = 20;
end

encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer([convsize 1], 2000, 'Padding','same', 'Name', 'conv1') %, 'Stride', [stride1 1])
    reluLayer('Name','relu1')
    convolution2dLayer([convsize 1], 1000, 'Padding','same', 'Name', 'conv2') %, 'Stride', [stride1 1])
    reluLayer('Name','relu2')
    convolution2dLayer([convsize 1], 1000, 'Padding','same', 'Name', 'conv3') %, 'Stride', [stride1 1])
    reluLayer('Name','relu5')
    convolution2dLayer([convsize 1], 1000, 'Padding','same', 'Name', 'conv4') %, 'Stride', [stride1 1])
    reluLayer('Name','relu6')
    convolution2dLayer([convsize 1], 1000, 'Padding','same', 'Name', 'conv5') %, 'Stride', [stride1 1])
    reluLayer('Name','relu7')
    convolution2dLayer([convsize 1], 1000, 'Padding','same', 'Name', 'conv6') %, 'Stride', [stride1 1])
    reluLayer('Name','relu8')
    convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv7') %, 'Stride', [stride1 1])
    reluLayer('Name','relu9')
    convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv8') %, 'Stride', [stride1 1])
    reluLayer('Name','relu10')
    convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv9') %, 'Stride', [stride1 1])
    reluLayer('Name','relu11')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv10') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu12')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv11') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu13')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv12') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu14')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv13') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu15')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv14') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu16')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv15') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu17')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv16') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu18')
    %convolution2dLayer([convsize 1], 500, 'Padding','same', 'Name', 'conv17') %, 'Stride', [stride1 1])
    %reluLayer('Name','relu19')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder1')
    ]);

decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    transposedConv2dLayer([convsize 1], 1000, 'Cropping', 'same', 'Name', 'transpose1', 'Stride', [4 1])
    reluLayer('Name','relu21')
    %transposedConv2dLayer([convsize 1], 1000, 'Cropping', 'same', 'Name', 'transpose2', 'Stride', [2 1])
    %reluLayer('Name','relu22')
    transposedConv2dLayer([convsize 1], 500, 'Cropping', 'same', 'Name', 'transpose3', 'Stride', [3 1])
    reluLayer('Name','relu23')
    transposedConv2dLayer([convsize 1], 500, 'Cropping', 'same', 'Name', 'transpose4', 'Stride', [5 1])
    reluLayer('Name','rel24')
    transposedConv2dLayer([convsize 1], 300, 'Cropping', 'same', 'Name', 'transpose5', 'Stride', [5 1])
    reluLayer('Name','relu25')
    transposedConv2dLayer([convsize 1], 1, 'Cropping', 'same', 'Name', 'transpose6')
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

numEpochs = 5;

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
        XBatch = XTrain(:,:,:,idx);
        XBatch = dlarray(single(XBatch), 'SSCB');
        
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
    
    [z, zMean, zLogvar] = sampling(encoderNet, XTest);
    xPred = sigmoid(forward(decoderNet, z));
    elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
    disp("Epoch : "+epoch+" Test loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s") 
    
    
end

%% Compute an visualize the Results

MAEloss = computeLoss(xPred, XTest, plotHighLoss,MAEThresh,MaxThresh,sizev)

if partial == true
    for i =1:length(TestArray);
        XTest = TestArray{i};
        d = size(XTest,2)/sizev;
        XTest = reshape(XTest, [sizev,1,1,d]);
        XTest = dlarray(XTest, 'SSCB');
        l = Labels(i);
        YTest = [];
        YTest(1:d) = l+1;
        [zMean, zLogVar] = visualizeLatentSpace(XTest, YTest, encoderNet, Labels, plotLatentPoints, numPoints,sizev);
        input('Press Enter to see the next point')
    end
else
    [zMean, zLogVar] = visualizeLatentSpace(XTest, YTest, encoderNet, Labels, plotLatentPoints, numPoints,sizev);
end
    %classifySVM(zMean,Outl_Test)
latentOutlierness = latentOutlier(XTest, encoderNet, plotLatentOutliers, LatentThresh,sizev);


%% Functions
% Compute the gradients of the loss with respect to the learnable paramaters 
% of both networks by calling the |dlgradient| function.

function [infGrad, genGrad] = modelGradients(encoderNet, decoderNet, x)
[z, zMean, zLogvar] = sampling(encoderNet, x);
xPred = sigmoid(forward(decoderNet, z));
loss = ELBOloss(x, xPred, zMean, zLogvar);
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
