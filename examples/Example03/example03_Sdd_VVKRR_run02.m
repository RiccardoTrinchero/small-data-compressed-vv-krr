% Reproducibility script for Example 3 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression" for a single run.
%
% This script reproduces the VV-KRR results for Example 3 S_dd, including:
% 1) the VV-KRR entry corresponding to model #10 in Table 6,
% 2) the scatter plot in Fig. 8 (left panel),
% 3) the parametric plot in Fig. 9 (top panel).
%
% By changing the input kernel kx and the output kernel ko,
% it is also possible to reproduce the other VV-KRR configurations
% reported in Table 6 of the paper.
%
% Note: because this example uses a random train/test split, different
% random seeds may lead to slightly different prediction errors.

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

%% Load dataset for Example 3

% The file Example03_Sdd_dataset.mat contains:
%   X_MC : normalized input samples for the full dataset;
%   Y_MC : output responses for S_dd21 for the full dataset;
%   freq : frequency vector.

load('Example03_Sdd_dataset.mat')

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

% Plot the S_dd21 responses selected for the test set.
% This plot is useful to visualize the variability of the test samples.

figure
plot(freq, Y_TEST)

xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('S_{dd21}', 'FontSize', FS, 'FontName', FN);

title('Example 3: S_{dd21} responses on the test set')

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
% zero variance in the training set.
index_B = find(isnan(B));
B(index_B) = 0;

% Enforce symmetry of the output kernel matrix
B = 0.5*(B+B.');

% Start training timer
t1 = tic;

% Train the compressed VV-KRR model.
%
% X_ED : normalized training inputs
% Y_ED : training outputs
% kx   : input kernel type
% ko   : output kernel matrix or analytic output kernel type
% []   : optional extra arguments not used here
% 5    : number of cross-validation folds used for hyperparameter tuning
% tol  : compression tolerance
%
%
% To reproduce other configurations in Table 6, change kx and/or ko.
kx = 'ardRBF';   % or 'ardMatern52'; 'Matern52'; 'RBF'; 'Matern12'; 'ardMatern12'
ko = 'RBF';               % or B; 'Matern52'; 'Matern12'

MODEL = train_VVRR_CV_learn_v20(X_ED,Y_ED,kx,ko,[],5,tol,'s','L2');

% Stop training timer
time_training = toc(t1);

% Start prediction timer
t2 = tic;

% Predict outputs on the test set
Y_pred = predict_VVRR_model_v7(MODEL,X_TEST,[]);

% Stop prediction timer
time_prediction = toc(t2);

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

%% Scatter plot for S_dd21

% Scatter plot of predicted vs actual S_dd21 values over the whole test set.

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
title('S_{dd21}: prediction vs. reference', 'FontSize', FS)

legend('Model #10', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

axis square
axis tight
grid on
box on
set(gca, 'LineWidth', 1.2)
set(gcf, 'Color', 'w')

%% Parametric comparison for S_dd21

% Compare reference and predicted S_dd21 responses for a subset of
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

ylabel('S_{dd21}', 'FontSize', FS)

legend({'Reference', 'Model #10'}, ...
    'FontSize', FS, 'Location', 'northwest', 'FontName', 'Arial')

xlim([min(freq), max(freq)])

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

inset_pos = [0.63, 0.25, 0.30, 0.30];  % [x, y, width, height]
inset_axes = axes('Position', inset_pos);

imagesc(freq, freq, B_TEST)

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
colorbar('Position', [0.65, 0.25, 0.01, 0.30]);