% Reproducibility script for Example 2 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the VV-KRR results for Example 2, including:
% 1) the VV-KRR entry corresponding to model #4 in Table 3,
% 2) the scatter plot in Fig. 5,
% 3) the parametric plot in Fig. 6.
%
% By changing the input kernel kx and the output kernel ko,
% it is also possible to reproduce the other VV-KRR configurations
% reported in Table 4 of the paper.

clear all;
clc;
close all;

% Add the VV-KRR toolbox folder and all its subfolders.
% The folder "tool_VVKRR_v0" is assumed to be located relative to this
% main script according to the path below.
addpath(genpath('../../tool_VVKRR_v0'));

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
% This preprocessing allows the VV-KRR model to operate on a single
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

%% Optional check: plot a few real-valued output responses

% This plot shows the real part of S11 for a subset of samples after the
% zero-padding preprocessing. It is only used as a quick visual check.

figure
plot(freq, Y_ZP(1:5:100,1:N_freq), 'b')
title('Check plot: Re\{S_{11}\} after preprocessing')
xlabel('Frequency', 'FontSize', FS, 'FontName', FN);
ylabel('Re\{S_{11}\}', 'FontSize', FS, 'FontName', FN);
axis tight;
grid on;
box on;
set(gcf, 'Color', 'w');

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

% Compression / truncation tolerance used in the VV-KRR training
tol = 1e-6;

% Build the data-driven output kernel matrix from the empirical correlation
% of the training outputs.
%
% This corresponds to the output-kernel construction proposed in the paper:
%   B = corr(Y_ED).
B = corr(Y_ED);

% Replace NaN entries, which may appear if some output components have
% zero variance in the training set. This can occur, for example, in the
% zero-padding regions.
index_B = find(isnan(B));
B(index_B) = 0;

% Enforce symmetry of the output kernel matrix
B = 0.5*(B+B.');

% Start training timer
t1 = tic;

% Train the compressed VV-KRR model.
%
% X_ED : normalized training inputs
% Y_ED : real-valued training outputs
% kx   : input kernel type
% ko   : output kernel matrix or analytic output kernel type
% []   : optional extra arguments not used here
% 5    : number of cross-validation folds used for hyperparameter tuning
% tol  : compression tolerance
%
% By changing kx and ko, it is possible to reproduce the VV-KRR results
% reported in Table 4 of the paper.
%
% To reproduce model #4 in Table 4, use:
kx = 'ardRBF';   % or 'ardMatern52';'Matern52'; 'RBF'; 'Matern12'; 'ardMatern12'
ko = B;               % or 'Matern52'; 'RBF'; 'Matern12'

MODEL = train_VVRR_CV_learn_v20(X_ED,Y_ED,kx,ko,[],5,tol,'s','L2');

% Stop training timer
time_training = toc(t1);

% Start prediction timer
t2 = tic;

% Predict real-valued outputs on the test set
Y_pred_tmp = predict_VVRR_model_v7(MODEL,X_TEST,[]);

% Stop prediction timer
time_prediction = toc(t2);

%% Convert predicted outputs back to complex-valued S-parameters

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
fprintf('--- VV-KRR MODEL SUMMARY ---\n');

% Number of training samples
fprintf('Number of training samples: %d\n', size(X_ED,1));

% Input and output kernel types actually stored in the trained model
fprintf('Input kernel kx: %s\n', MODEL.kx_kernel.type);
fprintf('Output kernel ko: %s\n', MODEL.ko_kernel.type);

% Compression tolerance
fprintf('Tolerance: %.1e\n', MODEL.tol);

% Compression summary
fprintf('Retained input modes nx: %d\n', MODEL.nx);
fprintf('Retained output modes no: %d\n', MODEL.no);
fprintf('Number of retained coefficients: %d\n', MODEL.nx * MODEL.no);

% Timing information
fprintf('Training time: %.4f seconds\n', time_training);
fprintf('Prediction time: %.4f seconds\n', time_prediction);

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
legend('VV-KRR model', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

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

legend({'Reference', 'VV-KRR model'}, ...
    'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

axis tight
grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])


%% Parametric comparison: imaginary part of S11

% Compare reference and predicted imaginary parts of S11 for the same
% subset of representative test samples.
%
% The output-kernel matrix is also shown as an inset in this figure.

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

%% Output-kernel matrix for inset plot

% This block reconstructs the output kernel matrix used by the trained
% VV-KRR model.
%
% If the output kernel is data-driven, MODEL.ko_kernel.type is 'GIVEN'
% and the matrix is directly stored in MODEL.ko_kernel.B.
%
% If an analytic output kernel is used, such as RBF or Matérn, the output
% kernel matrix is reconstructed from the optimized output-kernel
% hyperparameters.

D_OUT_TEST = [];

if strcmp(MODEL.ko_kernel.type, 'GIVEN')

    % Data-driven output kernel
    B_TEST = MODEL.ko_kernel.B;

else

    % Analytic output kernel
    if isempty(D_OUT_TEST)
        D_OUT_TEST = MODEL.D_OUT_ED;
    end

    D_OUT_ED = MODEL.D_OUT_ED;

    switch MODEL.ko_kernel.type

        case 'RBF' 
            sigma_f = 1;
            B_TEST = kernel_sqexp( ...
                D_OUT_ED, D_OUT_TEST, sigma_f, MODEL.params.ko_sigma_l);

        case 'Matern52'
            sigma_f = 1;
            B_TEST = kernel_matern52( ...
                D_OUT_ED, D_OUT_TEST, sigma_f, MODEL.params.ko_sigma_l);

        case 'Matern12'
            sigma_f = 1;
            B_TEST = kernel_matern12( ...
                D_OUT_ED, D_OUT_TEST, sigma_f, MODEL.params.ko_sigma_l);

        otherwise
            error('Unsupported output kernel type: %s', MODEL.ko_kernel.type);
    end

end

%% Inset plot: output kernel / B matrix

% Create a small inset showing the output-kernel matrix B.
% For the data-driven case, this is the empirical output-correlation matrix.

inset_pos = [0.29, 0.25, 0.30, 0.30];  % [x, y, width, height]
inset_axes = axes('Position', inset_pos);

% For Example 2, B_TEST includes all real/imaginary S11 and S21 blocks
% plus zero-padding regions. Therefore, the image represents the full
% output-correlation structure used by the VV-KRR model.
imagesc(B_TEST)

set(inset_axes, 'FontSize', FS-2, 'LineWidth', 1)
colormap(inset_axes, parula)
axis square tight
box on

xlabel('', 'FontSize', FS-2)
ylabel('', 'FontSize', FS-2)
title('B Matrix', 'FontSize', FS-2)

% Turn off x/y ticks in the inset to keep it clean
set(inset_axes, 'XTick', [], 'YTick', [])

% Add a small colorbar for the inset
colorbar('Position', [0.56, 0.25, 0.01, 0.30]);

