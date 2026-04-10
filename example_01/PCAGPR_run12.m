% Reproducibility script for Example 1 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the PCA+GPR results for the high-speed-link
% benchmark (Example 1), including:
% 1) the PCA+GPR entry corresponding to model #24 in Table 3,
% 2) the scatter plot in Fig. 3,
%
% The script:
% 1) loads the training and test datasets,
% 2) performs PCA on the training outputs via SVD,
% 3) truncates the PCA basis according to a relative tolerance,
% 4) trains one scalar GPR model for each retained PCA coefficient,
% 5) predicts the PCA coefficients on the test set,
% 6) reconstructs the full outputs,
% 7) computes relative error metrics,
% 8) displays a summary of the trained model,
% 9) generates the figures.

clear all;
clc;
close all;

% Plot/font settings used in the figures
FS = 15;
FN = 'Times';
FW = 'normal';
FA = 'normal';

% Set the random seed to ensure reproducibility
rng('default');

% Load normalized inputs, training/test outputs, and frequency vector
load('example_01_DATA.mat');

%% Modeling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PCA is computed through the singular value decomposition (SVD)
% of the output matrix.
%
% The transpose is used because the output matrix must be arranged as:
% [number of output components] x [number of training samples]
%
% In other words:
% - columns correspond to training realizations
% - rows correspond to output samples/components

% Start timing the overall PCA+GPR training phase
t_GPR = tic;

% Relative truncation threshold used for PCA
RelTol = 1e-3;

% Compute the sample mean of the outputs across the training set
MU = mean(Y_ED.', 2);

% Mean-center the output matrix before applying SVD
Ytilde = Y_ED.' - MU;

% Singular value decomposition of the centered output matrix
[U,S,V] = svd(Ytilde, 'econ');

% Determine the number of retained PCA modes.
% Only the singular values larger than RelTol times the largest one
% are retained.
nbar = find(diag(S)/S(1,1) > RelTol, 1, 'last');

% Retained PCA basis
Un = sparse(U(:,1:nbar));

% Compute the PCA coefficients associated with the training outputs.
% Each row of z_train contains the retained latent coordinates
% for one training sample.
z_train = (Un' * Ytilde).';

% Train one independent scalar GPR model for each retained PCA coefficient
for n = 1:nbar
    % Fit scalar Gaussian process regressor for the n-th latent coefficient
    M_gpr{n} = fitrgp( ...
        X_ED_norm, z_train(:,n), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true, ...
        'ConstantSigma', false);
end

% Model evaluation on the test set:
% predict all retained PCA coefficients for all test samples
Z_GPR = zeros(nbar, size(X_TEST_norm,1));

for n = 1:nbar
    % Predict the n-th latent coefficient on the test set
    Z_GPR(n,:) = predict(M_gpr{n}, X_TEST_norm);
end

% Stop timing
t_GPR = toc(t_GPR);

% Reconstruct the predicted outputs in the original output space:
% mean + retained PCA basis times predicted latent coefficients
Y_GPR = MU + Un * Z_GPR;
Y_pred = Y_GPR.';

%% Compute error metrics

% Small constant to avoid division by zero in relative errors
eps_norm = 1e-12;

% Relative L2 error over the full output matrix, expressed in percent
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / (norm(Y_TEST, 'fro') + eps_norm) * 100;

% Relative Linf error over all entries, expressed in percent
Linf_rel_Error = norm(abs(Y_TEST(:) - Y_pred(:)), inf) / norm(abs(Y_TEST(:)), inf) * 100;

%% Display summary in Command Window
fprintf('\n\n---------------------\n');
fprintf('--- PCA+GPR MODEL SUMMARY ---\n');

% Number of available training samples
fprintf('Number of training samples: %d\n', size(X_ED_norm,1));

% Kernel name used in the scalar GPR models
fprintf('Input kernel kx: ardmatern52\n');

% PCA truncation threshold
fprintf('PCA tolerance: %.1e\n', RelTol);

% Number of retained PCA modes
fprintf('Retained PCA modes: %d\n', nbar);

% Total number of scalar coefficients to be learned/stored,
% approximated here as retained PCA modes times number of training samples
fprintf('Number of retained coefficients: %d\n', nbar * size(X_ED_norm,1));

% Total training time
fprintf('Training time: %.4f seconds\n', t_GPR);

% Relative error metrics
fprintf('L2 relative error: %.4e\n', L2_rel_Error);
fprintf('Linf relative error: %.4e\n', Linf_rel_Error);
fprintf('----------------------\n');

%% Scatter plot
figure;
hold on;
set(gca, 'FontSize', FS, 'FontName', FN, 'FontWeight', FW, 'FontAngle', FA);

% Scatter plot of predicted vs actual values over the whole test set
scatter(Y_TEST(:), Y_pred(:), 30, 'g', 'x', 'LineWidth', 1.2);

% Reference line y = x, corresponding to perfect prediction
min_val = min([Y_TEST(:); Y_pred(:)]);
max_val = max([Y_TEST(:); Y_pred(:)]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.2);

% Axis labels and formatting
xlabel('Actual Values', 'FontSize', FS+3, 'FontName', FN);
ylabel('Model Prediction', 'FontSize', FS+3, 'FontName', FN);
legend({'Model #24'}, 'FontSize', FS+3, 'Location', 'southeast', 'FontName', 'Arial');

% Styling
axis square;
axis tight;
grid on;
box on;
set(gca, 'LineWidth', 1.2);
set(gcf, 'Color', 'w');

%% Parametric comparison between reference and predicted responses
% for a subset of test samples
figure;
hold on;
set(gca, 'FontSize', FS, 'LineWidth', 1.2, 'FontName', FN, 'FontWeight', FW, 'FontAngle', FA);

% Plot reference and prediction for the first test example
plot(freq/1e9, Y_TEST(1,:), 'k-', 'LineWidth', 2.5);
plot(freq/1e9, Y_pred(1,:), 'm--', 'LineWidth', 2);

% Plot reference and predictions for additional test examples
% sampled every 4 rows up to row 20
plot(freq/1e9, Y_TEST(1:4:20,:), 'k-', 'LineWidth', 2.5);
plot(freq/1e9, Y_pred(1:4:20,:), 'm--', 'LineWidth', 2);

% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('Magnitude (dB)', 'FontSize', FS, 'FontName', FN);

% Legend
legend({'Reference', 'Model #24'}, 'FontSize', FS, 'Location', 'best', 'FontName', 'Arial');

% Main plot aesthetics
axis tight;
grid on;
box on;
set(gcf, 'Color', 'w');