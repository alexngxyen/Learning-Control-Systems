% MAE 277 Project Learning Control Systems
% Final Project
% Description: K-Nearest Neighbor for wine dataset.
% Author: Alex Nguyen

clc; clear; close all;

%% Define Data Set:
addpath Data

% Wine Attributes
Text = textread('Wine_Attributes.txt','%s');
Attributes = char(Text);

% Collect Wine Data
FID = fopen('wine.txt');
C_data0 = textscan(FID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f', 200, 'Delimiter',',');
fclose('all');

% Target Data 
N = length(C_data0{1});
Class = C_data0{1};
X = cell2mat(C_data0(:, 2:14));  % Input
Y = Class;                       % Output

[trainInd, testInd1, testInd2] = dividerand(N);
testInd = [testInd1, testInd2];
xTrain = X(trainInd, :); xTest = X(testInd, :);
yTrain = Y(trainInd); yTest = Y(testInd);

% Data Size
Nx = size(X);
Ny = size(Y);

%% K-Nearest Neighbor:

% Train KNN Classifier 
rng(10);
K = ceil(sqrt(length(xTrain))/2);
Mdl = fitcknn(xTrain, yTrain, 'NumNeighbors', K, 'Standardize', 1);

% Predict
[label, score, cost] = predict(Mdl, xTest);

%% Classification Analysis;

% Random Subset of True and Predicted Labels
idxR = randsample(length(yTest), 10);
T = table(yTest(idxR), label(idxR),...
    'VariableNames', {'True Labels', 'Predicted Labels'});

% Test Data Classification
ctest = (yTest == label);
C = sum(ctest) / length(ctest);

% Resubstitution Loss
rloss = resubLoss(Mdl);

% Cross-Validation Loss 
CVMd1 = crossval(Mdl);
kloss = kfoldLoss(CVMd1);

% Print Results
fprintf('K-Nearest Neighbors (KNN)\n');
fprintf('Random Subset of Test Data:\n');
disp(T);
fprintf('Test Data: \n\t Correct Classification %4.4f %% \n\t Incorrect Classification %4.4f %% \n\n', 100 * C, 100 * (1 - C)); 
fprintf('Resubstitution Loss: \n\tClassifier Predicts Incorrectly for %4.4f %% of the Training Data \n\n', rloss * 100);
fprintf('Cross-Validated Loss: \n\tGeneralized Classification Error %4.4f %% of the Training Data \n', kloss * 100);
