% Reproducibility script for Example 2 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the PCA+GPR results for the high-speed-link
% benchmark (Example 2), including:
% 1) the PCA+GPR entry corresponding to model #20 in Table 4 for L =400 
% training samples,

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

%% Load dataset for Example 2
% The file Example01_dataset.mat contains:
%   x_mc        : normalized input samples (whole dataset);
%   S11_mc      : complex-valued output responses for S11 (whole dataset);
%   S21_mc      : complex-valued output responses for S21 (whole dataset);
%   freq        : frequency vector.

load("Example02_dataset.mat")


%% performing the zero padding
N_MC = size(S21_mc,2);

Npadding = 10;

for ii=1:N_MC
    
    XX = [real(S11_mc(:,ii));zeros(Npadding,1);imag(S11_mc(:,ii));zeros(Npadding,1); real(S21_mc(:,ii));zeros(Npadding,1);imag(S21_mc(:,ii))];
    Y_ZP(ii,:) = XX.';
        
end


N_freq = size(freq,1);


%% Defining test and training samples

N_ED = 400; 

index = 1:N_MC;

X_ED = x_mc(index(1:N_ED),:);
Y_ED = Y_ZP(index(1:N_ED),:);

X_TEST = x_mc(index(500+1:end),:);
Y_TEST = Y_ZP(index(500+1:end),:);

Y_gold_S11_TEST = S11_mc(:,index(500+1:end)).';
Y_gold_S21_TEST = S21_mc(:,index(500+1:end)).';

N_TEST = size(Y_TEST,1);

%% MC plot
figure
subplot(2,1,1)
plot(freq,abs(S11_mc))

% title sub 1
title('S11: MC example 2 computed on the test set')
% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('S11 (linear)', 'FontSize', FS, 'FontName', FN);


subplot(2,1,2)
plot(freq,abs(S21_mc))

% title sub 1
title('S21: MC example 2 computed on the test set')

% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS, 'FontName', FN);
ylabel('S21 (linear)', 'FontSize', FS, 'FontName', FN);

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
Y_pred_tmp = Y_GPR.';

%%

% Building complex valued from zero-padding
Y_pred_S11 = Y_pred_tmp(:,1:N_freq)+1j*Y_pred_tmp(:,(N_freq+1+Npadding):(2*N_freq+Npadding));
Y_pred_S21 = Y_pred_tmp(:,(2*N_freq+1+2*Npadding):(3*N_freq+2*Npadding))+1j*Y_pred_tmp(:,(3*N_freq+1+3*Npadding):end);

Y_pred = [Y_pred_S11;Y_pred_S21];
Y_TEST = [Y_gold_S11_TEST;Y_gold_S21_TEST];

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
fprintf('L2 relative error: %.1f\n', L2_rel_Error);
fprintf('Linf relative error: %.1f\n', Linf_rel_Error);
fprintf('----------------------\n');

%% Scatter plot
figure
% ---- Subplot 1: Real Part ----
subplot(1, 2, 1)
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

scatter(real(Y_gold_S11_TEST(:)), real(Y_pred_S11(:)), 30, 'm', 'x', 'LineWidth', 1.2)

min_val_real = min([real(Y_gold_S11_TEST(:)); real(Y_pred_S11(:))]);
max_val_real = max([real(Y_gold_S11_TEST(:)); real(Y_pred_S11(:))]);
plot([min_val_real, max_val_real], [min_val_real, max_val_real], 'k--', 'LineWidth', 1.2)

xlabel('Actual Values', 'FontSize', FS)
ylabel('Model Prediction', 'FontSize', FS)
title('Re\{S_{11}\}', 'FontSize', FS)
legend('Model #20', 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

axis square
axis tight
grid on
box on

% ---- Subplot 2: Imaginary Part ----
subplot(1, 2, 2)
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

scatter(imag(Y_gold_S11_TEST(:)), imag(Y_pred_S11(:)), 30, 'm', 'x', 'LineWidth', 1.2)

min_val_imag = min([imag(Y_gold_S11_TEST(:)); imag(Y_pred_S11(:))]);
max_val_imag = max([imag(Y_gold_S11_TEST(:)); imag(Y_pred_S11(:))]);
plot([min_val_imag, max_val_imag], [min_val_imag, max_val_imag], 'k--', 'LineWidth', 1.2)

xlabel('Actual Values', 'FontSize', FS)
ylabel('Model Prediction', 'FontSize', FS)
title('Im\{S_{11}\}', 'FontSize', FS)

axis square
axis tight
grid on
box on

% ---- Global settings ----
set(gcf, 'Color', 'w')


%% Parametric comparison between reference and predicted responses
% for a subset of test samples

% real
figure
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

% Plot reference and model predictions
plot(freq/1e9, real(Y_gold_S11_TEST(1,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, real(Y_pred_S11(1,:)), 'm--', 'LineWidth', 2)

plot(freq/1e9, real(Y_gold_S11_TEST(1:6:30,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, real(Y_pred_S11(1:6:30,:)), 'm--', 'LineWidth', 2)

% Axis labels
%xlabel('Frequency (GHz)', 'FontSize', FS)
ylabel('Re\{S11\}', 'FontSize', FS)

% Legend
legend({'Reference', 'Model #20'}, 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')


% Main plot aesthetics
axis tight
grid on
box on
set(gcf, 'Color', 'w')
pbaspect([2 1.2 1])


% imag
figure
hold on
set(gca, 'FontSize', FS, 'LineWidth', 1.2)

% Plot reference and model predictions
plot(freq/1e9, imag(Y_gold_S11_TEST(1,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, imag(Y_pred_S11(1,:)), 'm--', 'LineWidth', 2)

plot(freq/1e9, imag(Y_gold_S11_TEST(1:6:30,:)), 'k-', 'LineWidth', 2.5)
plot(freq/1e9, imag(Y_pred_S11(1:6:30,:)), 'm--', 'LineWidth', 2)

% Axis labels
xlabel('Frequency (GHz)', 'FontSize', FS)
ylabel('Im\{S11\}', 'FontSize', FS)

% Main plot aesthetics
axis tight
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
%inset_pos = [0.60, 0.60, 0.30, 0.30];  % [x, y, width, height] in normalized units

inset_pos = [0.29, 0.25, 0.30, 0.30];  % [x, y, width, height] in normalized units
inset_axes = axes('Position', inset_pos);

imagesc(freq/1e9, freq/1e9, B_TEST)
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
colorbar('Position', [0.56, 0.25, 0.01, 0.30]);




