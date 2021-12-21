% MAE 277 Project Learning Control Systems
% Final Project
% Description: Multiclass Model and EM Algorithms with PCA Analysis for 
% wine dataset.
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
Y = cell(length(Class), 1);      % Output
for ii = 1:length(Class)  
    if Class(ii) == 1
    Y{ii} = 'Class 1';
    elseif Class(ii) == 2
    Y{ii} = 'Class 2';
    elseif Class(ii) == 3
    Y{ii} = 'Class 3';
    end
end              

% Data Size
Nx = size(X);
Ny = size(Y);
Nc = size(Y, 1);

%% Support Vector Machine:
% SVM Template
t = templateSVM('Standardize', true);

% Train Multiclass ECOC Model
rng(10);
PMdl = fitcecoc(X, Y, 'Holdout', 0.30, 'Learners', t, 'ClassNames', ...
    {'Class 1', 'Class 2', 'Class 3'});
Mdl = PMdl.Trained{1};  

% Predict Test-Sample Labels
testInds = test(PMdl.Partition);  
XTest = X(testInds, :); YTest = Y(testInds, :);
labels = predict(Mdl, XTest);

% Resubsitution Classification Error
rng(10);
Mdl = fitcecoc(X, Y);
error = resubLoss(Mdl);

%% Classification Analysis:
% Random Subset of True and Predicted Labels
idx = randsample(sum(testInds), 10);
T = table(YTest(idx), labels(idx),...
    'VariableNames', {'True Labels', 'Predicted Labels'});

% Test Data Classification
ctest = strcmp(YTest, labels);
C = sum(ctest) / length(ctest);

%% Cross-Validation:

% Train ECOC Classifier
rng(10);
Mdl_cv = fitcecoc(X, Y, 'Learners', t, ...
    'ClassNames', {'Class 1', 'Class 2', 'Class 3'});

% Generalized Classification Error
CVMdl = crossval(Mdl_cv);           % 10-Fold Cross Validation
genError = kfoldLoss(CVMdl);

%% Print Results:
fprintf('Multi-Class Support Vector Machines (SVMs) \n');
fprintf('Random Subset of Test Data for SVM Classification:\n');
disp(T);
fprintf('Test Data: \n\t Correct Classification %4.4f %% \n\t Incorrect Classification %4.4f %% \n\n', 100 * C, 100 * (1 - C)) 
fprintf('Resubstitution Loss: \n\tClassifier Predicts Incorrectly for %4.4f %% of the Training Data \n\n', error * 100);
fprintf('Cross-Validated: \n\tGeneralized Classification Error is %4.4f %% of the Training Data\n\n', 100 * genError);

