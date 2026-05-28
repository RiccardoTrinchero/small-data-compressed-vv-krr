% Reproducibility script for Example 2 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the PCA+GPR results for the low-noise amplifier
% benchmark considered in Example 2. In particular, it reproduces:
% 1) the PCA+GPR entry corresponding to model #20 in Table 4
%    for L = 400 training samples;
% 2) the scatter plot of the real and imaginary parts of S11;
% 3) the parametric plots of the real and imaginary parts of S11.
%
% By changing the input covariance function, it is also possible to 
% reproduce the other PCA+reg configurations reported in Table 4 of the paper.

clear all;
clc;
close all;

% Plot/font settings used in the figures
FS = 15;
FN = 'Times';
FW = 'normal';
FA = 'normal';

%% Load dataset for Example 2

% The file Example02_dataset.mat contains:
%   x_mc   : normalized input samples for the full dataset;
%   S11_mc : complex-valued S11 responses for the full dataset;
%   S21_mc : complex-valued S21 responses for the full dataset;
%   freq   : frequency vector.
%
% The responses S11_mc and S21_mc are complex-valued and are therefore
% converted into a real-valued output representation before training.

load("Example02_dataset.mat")

%% Build real-valued output matrix with zero padding

% Number of Monte Carlo samples in the dataset
N_MC = size(S21_mc,2);

% Number of zero-padding samples inserted between output blocks
Npadding = 10;

% The output vector is constructed as:
% [Re(S11), padding, Im(S11), padding, Re(S21), padding, Im(S21)].
%
% This preprocessing allows the PCA+GPR model to operate on a single
% real-valued output vector while preserving the block structure of the
% complex-valued S-parameter responses.

for ii = 1:N_MC
    
    XX = [real(S11_mc(:,ii)); ...
          zeros(Npadding,1); ...
          imag(S11_mc(:,ii)); ...
          zeros(Npadding,1); ...
          real(S21_mc(:,ii)); ...
          zeros(Npadding,1); ...
          imag(S21_mc(:,ii))];
      
    Y_ZP(ii,:) = XX.';
        
end

% Number of frequency samples
N_freq = size(freq,1);

%% Define training and test samples

% Number of training samples used for this experiment
N_ED = 400; 

% The first N_ED samples are used for training.
% The samples from index 501 to the end are used for testing, consistently
% with the fixed train/test split adopted in the paper.
index = 1:N_MC;

X_ED = x_mc(index(1:N_ED),:);
Y_ED = Y_ZP(index(1:N_ED),:);

X_TEST = x_mc(index(500+1:end),:);
Y_TEST = Y_ZP(index(500+1:end),:);

% Complex-valued reference responses on the test set
Y_gold_S11_TEST = S11_mc(:,index(500+1:end)).';
Y_gold_S21_TEST = S21_mc(:,index(500+1:end)).';

% Number of test samples
N_TEST = size(Y_TEST,1);

%% Plot test-set responses

% Plot the magnitude of the S11 and S21 responses over the full dataset
% to show the variability induced by the parameter variations.

figure

subplot(2,1,1)
plot(freq/1e9, abs(S11_mc))
title('Example 2: S_{11} responses')
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('|S_{11}|', 'FontSize', FS, 'FontName', FN);
axis tight;
grid on;
box on;

subplot(2,1,2)
plot(freq/1e9, abs(S21_mc))
title('Example 2: S_{21} responses')
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('|S_{21}|', 'FontSize', FS, 'FontName', FN);
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
    
    % Fit scalar Gaussian process regressor for the n-th latent coefficient.
    % The ARD Matérn 5/2 kernel corresponds to model #20 in Table 4.
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

% Reconstruct the predicted outputs in the original real-valued output space:
% mean + retained PCA basis times predicted latent coefficients.
Y_GPR = MU + Un * Z_GPR;
Y_pred_tmp = Y_GPR.';

%% Convert reconstructed outputs back to complex-valued S-parameters

% Extract Re(S11), Im(S11), Re(S21), and Im(S21) from the zero-padded
% real-valued output vector and reconstruct the complex-valued responses.

Y_pred_S11 = ...
    Y_pred_tmp(:,1:N_freq) + ...
    1j * Y_pred_tmp(:,(N_freq+1+Npadding):(2*N_freq+Npadding));

Y_pred_S21 = ...
    Y_pred_tmp(:,(2*N_freq+1+2*Npadding):(3*N_freq+2*Npadding)) + ...
    1j * Y_pred_tmp(:,(3*N_freq+1+3*Npadding):end);

% Stack S11 and S21 responses to compute global error metrics
Y_pred = [Y_pred_S11; Y_pred_S21];
Y_TEST = [Y_gold_S11_TEST; Y_gold_S21_TEST];

%% Compute error metrics

% Small constant to avoid division by zero in relative errors
eps_norm = 1e-12;

% Relative L2 error over the full complex-valued output matrix,
% expressed as a percentage.
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / ...
               (norm(Y_TEST, 'fro') + eps_norm) * 100;

% Relative Linf error over all complex-valued output entries,
% expressed as a percentage.
Linf_rel_Error = norm(abs(Y_TEST(:)-Y_pred(:)), inf) / ...
                 norm(abs(Y_TEST(:)), inf) * 100;

%% Display summary in Command Window

fprintf('\n\n---------------------\n');
fprintf('--- PCA+GPR MODEL SUMMARY ---\n');

% Number of available training samples
fprintf('Number of training samples: %d\n', size(X_ED,1));

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

%% Scatter plot for S11

% Scatter plots of predicted vs actual values for the real and imaginary
% parts of S11 over the whole test set.

figure

% ---- Subplot 1: Real part ----
subplot(1, 2, 1)
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

scatter(real(Y_gold_S11_TEST(:)), real(Y_pred_S11(:)), ...
    30, 'm', 'x', 'LineWidth', 1.2)

% Reference line y = x, corresponding to perfect prediction
min_val_real = min([real(Y_gold_S11_TEST(:)); real(Y_pred_S11(:))]);
max_val_real = max([real(Y_gold_S11_TEST(:)); real(Y_pred_S11(:))]);
plot([min_val_real, max_val_real], ...
     [min_val_real, max_val_real], ...
     'k--', 'LineWidth', 1.2)

xlabel('Actual Values', 'FontSize', FS)
ylabel('Model Prediction', 'FontSize', FS)
title('Re\{S_{11}\}', 'FontSize', FS)
legend('Model #20', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

axis square
axis tight
grid on
box on

% ---- Subplot 2: Imaginary part ----
subplot(1, 2, 2)
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

scatter(imag(Y_gold_S11_TEST(:)), imag(Y_pred_S11(:)), ...
    30, 'm', 'x', 'LineWidth', 1.2)

% Reference line y = x, corresponding to perfect prediction
min_val_imag = min([imag(Y_gold_S11_TEST(:)); imag(Y_pred_S11(:))]);
max_val_imag = max([imag(Y_gold_S11_TEST(:)); imag(Y_pred_S11(:))]);
plot([min_val_imag, max_val_imag], ...
     [min_val_imag, max_val_imag], ...
     'k--', 'LineWidth', 1.2)

xlabel('Actual Values', 'FontSize', FS)
ylabel('Model Prediction', 'FontSize', FS)
title('Im\{S_{11}\}', 'FontSize', FS)

axis square
axis tight
grid on
box on

set(gcf, 'Color', 'w')

%% Parametric comparison: real part of S11

% Compare reference and predicted real parts of S11 for a subset of
% representative test samples.

figure
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

% Plot reference and prediction for the first test example
plot(freq/1e9, real(Y_gold_S11_TEST(1,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, real(Y_pred_S11(1,:)), 'm--', 'LineWidth', 2)

% Plot additional representative test examples
plot(freq/1e9, real(Y_gold_S11_TEST(1:6:30,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, real(Y_pred_S11(1:6:30,:)), 'm--', 'LineWidth', 2)

ylabel('Re\{S_{11}\}', 'FontSize', FS)

legend({'Reference', 'Model #20'}, ...
    'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

axis tight
grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])

%% Parametric comparison: imaginary part of S11

% Compare reference and predicted imaginary parts of S11 for the same
% subset of representative test samples.

figure
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

% Plot reference and prediction for the first test example
plot(freq/1e9, imag(Y_gold_S11_TEST(1,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, imag(Y_pred_S11(1,:)), 'm--', 'LineWidth', 2)

% Plot additional representative test examples
plot(freq/1e9, imag(Y_gold_S11_TEST(1:6:30,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, imag(Y_pred_S11(1:6:30,:)), 'm--', 'LineWidth', 2)

xlabel('Frequency (GHz)', 'FontSize', FS)
ylabel('Im\{S_{11}\}', 'FontSize', FS)

axis tight
grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])