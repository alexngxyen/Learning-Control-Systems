% MAE 277 Project Learning Control Systems
% Final Project
% Description: Feed-Forward Neural Network (FFNN) for wine dataset.
% Author: Alex Nguyen

clc; 
clear; 
close all;

%% Define Data Set:

% Add Folder to Path
addpath Data

% Wine Attributes
Text = textread('Wine_Attributes.txt','%s');
Attributes = char(Text);

% Wine Data
FID = fopen('wine.txt');
C_data0 = textscan(FID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f', 200, 'Delimiter',',');
fclose('all');

% Target Data 
N = length(C_data0{1});
Class = C_data0{1};
X = cell2mat(C_data0(:, 2:14))';                           % Input
Y =  double([(Class == 1), (Class == 2), (Class == 3)])';  % Output

% Data Size
Nx = size(X);
Ny = size(Y);
Nc = size(Y, 1);

%% Feed-Forward Neural Network:

% Pattern Recognition Network
rng(10);
var1 = mean([Ny(1), Nx(1)]);      % # of Hidden Layer Neurons
var2 = 'traingdm';                % Training Function (Gradient Descent with Momentum)
var3 = 'mse';                     % Performance Function
Net = patternnet(var1, var2, var3);

% Train FFNN
rng(10);
[NET, TR] = train(Net, X, Y);

% FFNN Performance (Training, Validation, and Test Sets)
figure;
plotperform(TR)

%% Test Feed-Forward Neural Network:
% Target Data
testX = X(:, TR.testInd);  
testT = Y(:, TR.testInd);

% Test Data
testY = NET(testX);     

% Confusion Plot
figure;
plotconfusion(testT, testY)

% Classification Results
N_testY = testY ./ sum(testY, 1);
[~, iffnn] = max(N_testY, [], 1);
Ytest = iffnn';
ctest = (Ytest == Class(TR.testInd));
C = sum(ctest(:)) / length(ctest(:));

% Print Results
fprintf('Feed-Forward Neural Network \n');
fprintf('Test Data: \n\t Correct Classification %4.4f %% \n\t Incorrect Classification %4.4f %% \n\n', 100 * C, 100 * (1 - C)); 
