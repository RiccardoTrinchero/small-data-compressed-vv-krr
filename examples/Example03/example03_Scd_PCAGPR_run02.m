% Reproducibility script for Example 3 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression" for a single run.
%
% This script reproduces the PCA+GPR results for Example 3 S_cd, including:
% 1) the PCA+GPR entry corresponding to model #20 in Table 6,
% 2) the scatter plot for S_cd21,
% 3) the parametric plot for S_cd21.
%
% By changing the scalar GPR kernel, it is also possible to reproduce the
% other PCA+reg configurations reported in Table 6 of the paper.
%
% Note: because this example uses a random train/test split, different
% random seeds may lead to slightly different prediction errors.

clear all;
clc;
close all;

% Plot/font settings used in the figures
FS = 15;
FN = 'Times';
FW = 'normal';
FA = 'normal';

%% Load dataset for Example 3

% The file Example03_Scd_dataset.mat contains:
%   X_MC : normalized input samples for the full dataset;
%   Y_MC : output responses for S_cd21 for the full dataset;
%   freq : frequency vector.

load('Example03_Scd_dataset.mat')

% Number of available samples in the full dataset
N_MC = size(X_MC,1);

%% Define random train/test split

% Number of training and test samples used in one split.
% In the paper, this splitting procedure is repeated five times and the
% errors are reported as mean value plus/minus standard deviation.
N_ED = 190;
N_TEST = 10;

% Random permutation of the available samples
index = randperm(N_MC,N_MC);

% Training set
X_ED = X_MC(index(1:N_ED),:);
Y_ED = Y_MC(index(1:N_ED),:);

% Test set
X_TEST = X_MC(index((end-N_TEST+1):end),:);
Y_TEST = Y_MC(index((end-N_TEST+1):end),:);

%% Plot test-set responses

% Plot the S_cd21 responses selected for the test set.
% This plot is useful to visualize the variability of the test samples.

figure
plot(freq,Y_TEST)

xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('S_{cd21}', 'FontSize', FS, 'FontName', FN);

title('Example 3: S_{cd21} responses on the test set')

axis tight;
grid on;
box on;
set(gcf, 'Color', 'w');

%% Modeling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PCA is computed through the singular value decomposition (SVD)
% of the output matrix.
%
% The transpose is used because the output matrix must be arranged as:
% [number of output components] x [number of training samples].
%
% In other words:
% - columns correspond to training realizations;
% - rows correspond to output samples/components.

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
    
    % Model #20 uses an ARD Matérn 5/2 kernel.
    M_gpr{n} = fitrgp( ...
        X_ED, z_train(:,n), ...
        'KernelFunction', 'ardmatern52', ...
        'Standardize', true, ...
        'ConstantSigma', false);
end

% Model evaluation on the test set:
% predict all retained PCA coefficients for all test samples.
Z_GPR = zeros(nbar, size(X_TEST,1));

for n = 1:nbar
    
    % Predict the n-th latent coefficient on the test set
    Z_GPR(n,:) = predict(M_gpr{n}, X_TEST);
end

% Stop timing
t_GPR = toc(t_GPR);

% Reconstruct the predicted outputs in the original output space:
% mean + retained PCA basis times predicted latent coefficients.
Y_GPR = MU + Un * Z_GPR;
Y_pred = Y_GPR.';

%% Compute error metrics

% Small constant to avoid division by zero in relative errors
eps_norm = 1e-12;

% Relative L2 error over the full output matrix, expressed as a percentage
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / ...
               (norm(Y_TEST, 'fro') + eps_norm) * 100;

% Relative Linf error over all output entries, expressed as a percentage
Linf_rel_Error = norm(abs(Y_TEST(:)-Y_pred(:)), inf) / ...
                 norm(abs(Y_TEST(:)), inf) * 100;

%% Display summary in Command Window

fprintf('\n\n---------------------\n');
fprintf('--- PCA+GPR MODEL SUMMARY ---\n');

% Number of available training samples
fprintf('Number of training samples: %d\n', size(X_ED,1));

% Kernel name used in the scalar GPR models
fprintf('Input kernel kx: ardmatern52\n');

% PCA truncation threshold
fprintf('PCA tolerance: %.1e\n', RelTol);

% Number of retained PCA modes
fprintf('Retained PCA modes: %d\n', nbar);

% Total number of scalar coefficients to be learned/stored,
% approximated here as retained PCA modes times number of training samples.
fprintf('Number of retained coefficients: %d\n', nbar * size(X_ED,1));

% Total training time
fprintf('Training time: %.4f seconds\n', t_GPR);

% Relative error metrics
fprintf('L2 relative error: %.1f\n', L2_rel_Error);
fprintf('Linf relative error: %.1f\n', Linf_rel_Error);
fprintf('----------------------\n');

%% Scatter plot for S_cd21

% Scatter plot of predicted vs actual S_cd21 values over the whole test set.

figure
hold on
set(gca, 'FontSize', FS)

% Scatter plot: predictions vs actual values
scatter(Y_TEST(:), Y_pred(:), 30, 'm', 'x')

% Reference line y = x, corresponding to perfect prediction
min_val = min([Y_TEST(:); Y_pred(:)]);
max_val = max([Y_TEST(:); Y_pred(:)]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.2)

xlabel('Actual Values', 'FontSize', FS)
ylabel('Model Prediction', 'FontSize', FS)
title('S_{cd21}: prediction vs. reference', 'FontSize', FS)

legend('Model #20', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

axis square
axis tight
grid on
box on
set(gca, 'LineWidth', 1.2)
set(gcf, 'Color', 'w')

%% Parametric comparison for S_cd21

% Compare reference and predicted S_cd21 responses for a subset of
% representative test samples.

figure
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

% Plot reference and prediction for one selected test example
plot(freq, Y_TEST(4,:), 'k-', 'LineWidth', 2.5)
plot(freq, Y_pred(4,:), 'm--', 'LineWidth', 2)

% Plot additional representative test examples
plot(freq, Y_TEST([2,4:2:10],:), 'k-', 'LineWidth', 2.5)
plot(freq, Y_pred([2,4:2:10],:), 'm--', 'LineWidth', 2)

ylabel('S_{cd21}', 'FontSize', FS)

legend({'Reference', 'Model #20'}, ...
    'FontSize', FS, 'Location', 'northwest', 'FontName', 'Arial')

xlim([min(freq),max(freq)])

grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])