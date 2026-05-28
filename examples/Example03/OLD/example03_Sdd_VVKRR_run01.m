clear all;
clc;
close all;

% Add the VV-KRR toolbox folder and all its subfolders.
% The folder "tool_VVKRR_v0" is assumed to be located in the same
% directory as this main script.
addpath(genpath('tool_VVKRR_v0'));


% Plot/font settings used in the figures
FS = 15;
FN = 'Times';
FW = 'normal';
FA = 'normal';

% Set the random seed to ensure reproducibility
rng('default');

%% Load dataset for Example 3

% x_mc and S11_mc+S21_mc collect the overall dataset: input and complex output samples
% %
% % The frequency points are collected in the vector freq.


load('Example03_Sdd_dataset.mat')

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

title('Sdd: MC example 1 computed on the test set')
% Main plot aesthetics
axis tight;
grid on;
box on;
set(gcf, 'Color', 'w');


%% Modeling

% Compression / truncation tolerance used in the VV-KRR training
tol = 1e-6;

% Build the output kernel matrix from the empirical correlation
% of the training outputs
B = corr(Y_ED);

% Replace NaN entries, which may appear if some output components
% have zero variance in the training set
index_B = find(isnan(B));
B(index_B) = 0;

% Enforce symmetry of the output kernel matrix
B = 0.5*(B+B.');

% Start training timer
t1 = tic;

% Train the VV-KRR model.
%
% X_ED_norm : normalized training inputs
% Y_ED      : training outputs
% kx        : input kernel type
% ko        : output kernel matrix or analytic output kernel type
% []        : optional extra arguments not used here
% 5         : number of CV folds
% tol       : compression tolerance
%
% By changing kx and ko, it is possible to reproduce the VV-KRR
% results reported in Table 3 of the paper.
kx = 'ardMatern52';   % or 'Matern52'; 'RBF'; 'ardRBF'; 'Matern12'; 'ardMatern12'
ko = 'Matern52';               % or 'Matern52'; 'RBF'; 'Matern12'

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

% Relative L2 error over the full output matrix, expressed in percent
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / (norm(Y_TEST, 'fro') + eps_norm) * 100;

% Relative Linf error over all output entries, expressed in percent
Linf_rel_Error = norm(abs(Y_TEST(:)-Y_pred(:)), inf) / norm(abs(Y_TEST(:)), inf) * 100;

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

% Error metrics
fprintf('L2 relative error: %.1f\n', L2_rel_Error);
fprintf('Linf relative error: %.1f\n', Linf_rel_Error);


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
title('Sdd21: Prediction vs. Ground Truth','FontSize', FS)
legend('Model #10', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')


% Styling
axis square
axis tight
grid on
box on
set(gca, 'LineWidth', 1.2)
set(gcf, 'Color', 'w')



%% imag
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
ylabel('Sdd21', 'FontSize', FS)
legend({'Reference', 'Model #10'}, 'FontSize', FS, 'Location', 'southwest', 'FontName', 'Arial')


ylim([-0.75,0.05])
xlim([min(freq),max(freq)])
% Main plot aesthetics
%axis tight
grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])

%% plot ko
D_OUT_TEST = [];

if strcmp(MODEL.ko_kernel.type, 'GIVEN')

    B_TEST = MODEL.ko_kernel.B;%(1-MODEL.params.ko_sigma_l)*MODEL.ko_kernel.B+eye(size(MODEL.ko_kernel.B))*MODEL.params.ko_sigma_l;%MODEL.ko_kernel.B;
else

    if isempty(D_OUT_TEST)
        D_OUT_TEST = MODEL.D_OUT_ED;
    end

    D_OUT_ED = MODEL.D_OUT_ED;

    switch MODEL.ko_kernel.type

        case 'RBF' 
            sigma_f = 1;
            B_TEST = kernel_sqexp(D_OUT_ED,D_OUT_TEST,sigma_f,MODEL.params.ko_sigma_l);

        case 'Matern52'
            sigma_f = 1;
            B_TEST = kernel_matern52(D_OUT_ED,D_OUT_TEST,sigma_f,MODEL.params.ko_sigma_l);

        case 'Matern12'
            sigma_f = 1;
            B_TEST = kernel_matern12(D_OUT_ED,D_OUT_TEST,sigma_f,MODEL.params.ko_sigma_l);
    end

end

% --------------------------
% Inset plot (B Matrix)
% --------------------------
inset_pos = [0.63, 0.55, 0.30, 0.30];  % [x, y, width, height] in normalized units
inset_axes = axes('Position', inset_pos);

imagesc(freq, freq, (B_TEST))
set(inset_axes, 'FontSize', FS-2, 'LineWidth', 1)
colormap(inset_axes, parula)
axis square tight
box on

xlabel('','FontSize', FS-2)
ylabel('','FontSize', FS-2)
title('B Matrix', 'FontSize', FS-2)

% Optional: turn off x/y ticks in the inset to keep it clean
set(inset_axes, 'XTick', [], 'YTick', [])

% Add colorbar if you want (small)
cb = colorbar(inset_axes, 'Position', [0.65, 0.55, 0.01, 0.30]);



