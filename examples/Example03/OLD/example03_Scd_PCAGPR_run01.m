% Reproducibility script for Example 3 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the PCA+GPR results for the high-speed-link
% benchmark (Example 3), including:
% 1) the PCA+GPR entry corresponding to model #20 in Table 7,

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
%rng('default');

%% Load dataset for Example 3

% x_mc and S11_mc+S21_mc collect the overall dataset: input and complex output samples
% %
% % The frequency points are collected in the vector freq.


load('Example03_Scd_dataset.mat')

N_MC = size(X_MC,1);



%% data-splitting
N_ED = 190;
N_TEST = 10;

index = randperm(N_MC,N_MC);

X_ED = X_MC(index(1:N_ED),:);
Y_ED = Y_MC(index(1:N_ED),:);


X_TEST = X_MC(index((end-N_TEST+1):end),:);
Y_TEST = Y_MC(index((end-N_TEST+1):end),:);

%% MC plot
figure
plot(freq,Y_TEST)

% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('Magnitude (dB)', 'FontSize', FS, 'FontName', FN);

title('Scd: MC example 1 computed on the test set')
% Main plot aesthetics
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
        X_ED, z_train(:,n), ...
        'KernelFunction', 'ardmatern52', ...
        'Standardize', true, ...
        'ConstantSigma', false);
end

% Model evaluation on the test set:
% predict all retained PCA coefficients for all test samples
Z_GPR = zeros(nbar, size(X_TEST,1));

for n = 1:nbar
    % Predict the n-th latent coefficient on the test set
    Z_GPR(n,:) = predict(M_gpr{n}, X_TEST);
end

% Stop timing
t_GPR = toc(t_GPR);

% Reconstruct the predicted outputs in the original output space:
% mean + retained PCA basis times predicted latent coefficients
Y_GPR = MU + Un * Z_GPR;
Y_pred = Y_GPR.';


%%%% computing error

eps_norm = 1e-12;

% L2 rel
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / (norm(Y_TEST, 'fro') + eps_norm)*100

%Linf rel
Linf_rel_Error = norm(abs(Y_TEST(:)-Y_pred(:)),inf)/norm(abs(Y_TEST(:)),inf)*100

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
% approximated here as retained PCA modes times number of training samples
fprintf('Number of retained coefficients: %d\n', nbar * size(X_ED,1));

% Total training time
fprintf('Training time: %.4f seconds\n', t_GPR);

% Relative error metrics
fprintf('L2 relative error: %.1f\n', L2_rel_Error);
fprintf('Linf relative error: %.1f\n', Linf_rel_Error);
fprintf('----------------------\n');

%% Scatter plot Real
figure
hold on
set(gca, 'FontSize', FS)

% Scatter plot: predictions vs actual (magenta 'x' markers)
scatter(Y_TEST(:), Y_pred(:), 30, 'm', 'x')
hold on


% Reference line y = x (black dashed)
min_val = min([Y_TEST(:); Y_pred(:)]);
max_val = max([Y_TEST(:); Y_pred(:)]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.2)

% Axis labels and formatting
xlabel('Actual Values', 'FontSize', FS)
ylabel('Model Prediction', 'FontSize', FS)
title('Scd21: Prediction vs. Ground Truth','FontSize', FS)
legend('Model #10', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')


% Styling
axis square
axis tight
grid on
box on
set(gca, 'LineWidth', 1.2)
set(gcf, 'Color', 'w')



%% parametric plot
figure
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

% Plot reference and model predictions
plot(freq, Y_TEST(4,:), 'k-', 'LineWidth', 2.5)
plot(freq, Y_pred(4,:), 'm--', 'LineWidth', 2)

plot(freq, Y_TEST([2,4:2:11],:), 'k-', 'LineWidth', 2.5)
plot(freq, Y_pred([2,4:2:11],:), 'm--', 'LineWidth', 2)

% Axis labels
%xlabel('Frequency (GHz)', 'FontSize', FS)
ylabel('Scd21', 'FontSize', FS)
legend({'Reference', 'Model #10'}, 'FontSize', FS, 'Location', 'southwest', 'FontName', 'Arial')

xlim([min(freq),max(freq)])
% Main plot aesthetics
%axis tight
grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])

