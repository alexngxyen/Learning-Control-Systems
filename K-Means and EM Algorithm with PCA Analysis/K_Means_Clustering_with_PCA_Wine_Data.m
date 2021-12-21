% MAE 277 Project Learning Control Systems
% Final Project
% Description: K-Means and EM Algorithms with PCA for wine dataset.
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
X = cell2mat(C_data0(:, 2:14))';                          % Input
Y = double([(Class == 1), (Class == 2), (Class == 3)])';  % Output

% Data Size
Nx = size(X);
Ny = size(Y);

%% Principal Component Analysis (PCA):
% Normalize Data
Z = (X - mean(X, 2)) ./ std(X, 0, 2);

% PCA of Raw Data
[coeff, score, latent, tsquared, explained, mu] = pca(Z');

% Orthonormal Eigenvectors for p Largest Eigenvalues 
p = 2; 
Up = coeff(:, 1:p);

% Analysis (Low-Dimensional Representation)
y = Up' * Z;

% Synthesis (Reconstruction)
Zhat = Up * y;

% Percentage of Total Variance
figure;
bar(explained);
xlabel('Principal Components'); ylabel('Variance %');
title('Percentage of Total Variance');

%% K-Means Clustering Algorithm:

% Perform K-Means Clustering
rng(10);
K = 3;                                 % Number of Clusters
X_Data = y';                           % Low Dimensional Data Matrix
[Centroid, C_mu] = kmeans(X_Data, K);

% Plot Clusters
figure;
hold on;
% gscatter(y(1, :), y(2, :), Centroid)
plot(X_Data(Centroid == 1, 1), X_Data(Centroid == 1, 2), 'r.', 'markersize', 12);
plot(X_Data(Centroid == 2, 1), X_Data(Centroid == 2, 2), 'b.', 'markersize', 12);
plot(X_Data(Centroid == 3, 1), X_Data(Centroid == 3, 2), 'g.', 'markersize', 12);
plot(C_mu(:, 1), C_mu(:, 2), 'kx', 'markersize', 12, 'linewidth', 3);
hold off;
legend('Cluster 1', 'Cluster 2', 'Cluster 3', ' Centroids', 'location', 'best')
xlabel('First Principal Component'); ylabel('Second Principal Component');
title('Principal Component Scatter Plot');    

%% Classification Analysis:

% Ensure Classified Clusters Match Training Data
i1 = find(Centroid ~= Centroid(1));
i2 = find((Centroid ~= Centroid(1)) & (Centroid ~= Centroid(i1(1))));
ind1 = Centroid(1); 
ind2 = Centroid(i1(1)); 
ind3 = Centroid(i2(1));

C_pca = Centroid;
C_pca(Centroid == ind1) = 1; 
C_pca(Centroid == ind2) = 2; 
C_pca(Centroid == ind3) = 3;

% Random Subset of True and Predicted Clusters
idxR = randsample(length(Class), 10);
T = table(Class(idxR), C_pca(idxR),...
    'VariableNames', {'True Labels', 'Predicted Labels'});

% Analyze Results
ctest = (C_pca == Class);
C = sum(ctest(:)) / length(ctest(:));

% Print Results
fprintf('K-Means Clustering with PCA \n');
fprintf('%d Principle Components reduced to %d Principle Components (p < d) \n\n', Nx(1), p);
fprintf('The First %d Principal Components Account for %4.4f %% of the Variance \n\n', p, sum(explained(1:p)))
fprintf('Random Subset of Cluster Data:\n');
disp(T);
fprintf('Cluster Data: \n\t Correct Classification %4.4f %% \n\t Incorrect Classification %4.4f %% \n\n', 100 * C, 100 * (1 - C)); 
