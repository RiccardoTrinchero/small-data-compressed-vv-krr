% Reproducibility script for Example 1 of the paper:
% "Small-Data Modeling of Electromagnetic Structures via
% Compressed Vector-Valued Kernel Ridge Regression"
%
% This script reproduces the DNN baseline results for the high-speed-link
% benchmark (Example 1), including:
% 1) the DNN entry corresponding to model #25 in Table 3,
%
% The script:
% 1) loads the training and test datasets,
% 2) defines fixed training/validation subsets,
% 3) uses Bayesian optimization to tune the DNN hyperparameters,
% 4) retrains the best network,
% 5) evaluates it on the test set,
% 6) computes relative error metrics,
% 7) displays a summary of the trained model,
% 8) generates the figures.


clear all;
clc;
close all;

% Plot/font settings used in the figures
FS=15;
FN='Times';
FW='normal';
FA='normal';

% Set the random seed to ensure reproducibility
rng('default')

% Load normalized training/test inputs and corresponding outputs
load('example_01_DATA.mat')

%% Modeling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Multi-output regression with DNN + Bayesian Hyperparameter Optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Training and test data:
% L = 300 training samples
% X_ED_norm: normalized training inputs (L x p)
% Y_ED: training outputs (L x D)
%
% L_test = 1000 test samples
% X_TEST_norm: normalized test inputs (L_test x p)
% Y_TEST: test outputs (L_test x D)

% In the original version of the script, the available training data were
% split into:
% - an internal training subset
% - a validation subset
%
% That split was originally generated using cvpartition. To avoid
% reproducibility issues related to the random partitioning, the indices
% index_train and index_validation were stored explicitly in the data file
% and are reused here.

XTrain = X_ED_norm(index_train,:);
YTrain = Y_ED(index_train,:);

XVal   = X_ED_norm(index_validation,:);
YVal   = Y_ED(index_validation,:);

% Extract the number of input features and output dimensions
numFeatures = size(XTrain,2);
numOutputs  = size(YTrain,2);

% Start timing the overall DNN training phase
t_DNN = tic;

%% Objective function for Bayesian optimization
function rmse = objectiveFcn(optVars,XTrain,YTrain,XVal,YVal,numFeatures,numOutputs)
    
    % Convert categorical hyperparameters to strings
    act1Str = string(optVars.Activation1);
    act2Str = string(optVars.Activation2);

    % Activation for first hidden layer
    switch act1Str
        case "relu"
            act1 = reluLayer;
        case "leakyrelu"
            act1 = leakyReluLayer(0.01);
        otherwise
            act1 = reluLayer;
    end

    % Activation for second hidden layer
    switch act2Str
        case "relu"
            act2 = reluLayer;
        case "leakyrelu"
            act2 = leakyReluLayer(0.01);
        otherwise
            act2 = reluLayer;
    end

    % Define the DNN architecture
    layers = [
        featureInputLayer(numFeatures,"Normalization","zscore")
        fullyConnectedLayer(optVars.NumHiddenUnits1)
        act1
        dropoutLayer(optVars.Dropout1)
        fullyConnectedLayer(optVars.NumHiddenUnits2)
        act2
        dropoutLayer(optVars.Dropout2)
        fullyConnectedLayer(numOutputs)
        regressionLayer
    ];

    % Training options used inside the Bayesian optimization loop
    options = trainingOptions('adam', ...
        'InitialLearnRate',optVars.LearnRate, ...
        'MaxEpochs',1000, ...
        'MiniBatchSize',optVars.MiniBatchSize, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{XVal,YVal}, ...
        'Verbose',false, ...
        'ValidationPatience',10);

    % Train the network for the current hyperparameter configuration
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Compute the validation RMSE used by Bayesian optimization
    YPred = predict(net,XVal);
    rmse = sqrt(mean((YVal - YPred).^2,"all"));
end

%% Hyperparameter search space
vars = [
    optimizableVariable('NumHiddenUnits1',[64,256],'Type','integer')
    optimizableVariable('NumHiddenUnits2',[32,128],'Type','integer')
    optimizableVariable('Dropout1',[0.1,0.5])
    optimizableVariable('Dropout2',[0.1,0.5])
    optimizableVariable('LearnRate',[1e-4,1e-2],'Transform','log')
    optimizableVariable('MiniBatchSize',[16,128],'Type','integer')
    optimizableVariable('Activation1',["relu","leakyrelu"],'Type','categorical')
    optimizableVariable('Activation2',["relu","leakyrelu"],'Type','categorical')
];

%% Run Bayesian optimization
results = bayesopt(@(optVars)objectiveFcn(optVars,XTrain,YTrain,XVal,YVal,numFeatures,numOutputs), ...
    vars, ...
    'MaxObjectiveEvaluations',100, ...
    'AcquisitionFunctionName','expected-improvement-plus');

bestParams = bestPoint(results);

fprintf('Best Hyperparameters:\n');
disp(bestParams);

%% Train the final network with the best hyperparameters

% Convert categorical hyperparameters to strings
act1Str = string(bestParams.Activation1);
act2Str = string(bestParams.Activation2);

switch act1Str
    case "relu"
        act1 = reluLayer;
    case "leakyrelu"
        act1 = leakyReluLayer(0.01);
    otherwise
        act1 = reluLayer;
end

switch act2Str
    case "relu"
        act2 = reluLayer;
    case "leakyrelu"
        act2 = leakyReluLayer(0.01);
    otherwise
        act2 = reluLayer;
end

finalLayers = [
    featureInputLayer(numFeatures,"Normalization","zscore")
    fullyConnectedLayer(bestParams.NumHiddenUnits1)
    act1
    dropoutLayer(bestParams.Dropout1)
    fullyConnectedLayer(bestParams.NumHiddenUnits2)
    act2
    dropoutLayer(bestParams.Dropout2)
    fullyConnectedLayer(numOutputs)
    regressionLayer
];

finalOptions = trainingOptions('adam', ...
    'InitialLearnRate',bestParams.LearnRate, ...
    'MaxEpochs',1000, ...
    'MiniBatchSize',bestParams.MiniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',20, ...
    'Plots','training-progress', ...
    'Verbose',false);

finalNet = trainNetwork(XTrain,YTrain,finalLayers,finalOptions);

%% Evaluate final network

% Compute validation RMSE for the final tuned network
YPred = predict(finalNet,XVal);
finalRmse = sqrt(mean((YVal - YPred).^2,"all"));
fprintf("Final tuned DNN Validation RMSE = %.4f\n",finalRmse);

%% Test the final network

% The same validation error is recomputed here.
% This block is left unchanged with respect to the original script.
YPred = predict(finalNet,XVal);
finalRmse = sqrt(mean((YVal - YPred).^2,"all"));
fprintf("Final tuned DNN Validation RMSE = %.4f\n",finalRmse);

% Generate predictions on the external test set
Y_pred = predict(finalNet,X_TEST_norm);

% Stop timing
t_DNN = toc(t_DNN);

%% Compute error metrics

eps_norm = 1e-12;

% Relative L2 error in dB, expressed in percent
L2_rel_Error = norm((Y_TEST - Y_pred), 'fro') / (norm(Y_TEST, 'fro') + eps_norm)*100;

% Relative L2 error in linear scale, expressed in percent
L2_rel_Error_lin = norm((10.^(Y_TEST/20) - 10.^(Y_pred/20)), 'fro') / ...
    (norm(10.^(Y_TEST/20), 'fro') + eps_norm)*100;

% Relative Linf error in dB, expressed in percent
Linf_rel_Error = norm(abs(Y_TEST(:)-Y_pred(:)),inf) / norm(abs(Y_TEST(:)),inf)*100;

% Relative Linf error in linear scale, expressed in percent
Linf_rel_Error_lin = norm(abs(10.^(Y_TEST(:)/20)-10.^(Y_pred(:)/20)),inf) / ...
    (norm(abs(10.^(Y_TEST(:)/20)),inf)+eps_norm)*100;

%% Display summary in Command Window
fprintf('\n\n---------------------\n');
fprintf('--- DNN MODEL SUMMARY ---\n');

fprintf('Number of training samples: %d\n', size(X_ED_norm,1));
fprintf('Number of input features: %d\n', numFeatures);
fprintf('Number of output components: %d\n', numOutputs);

fprintf('Best NumHiddenUnits1: %d\n', bestParams.NumHiddenUnits1);
fprintf('Best NumHiddenUnits2: %d\n', bestParams.NumHiddenUnits2);
fprintf('Best Dropout1: %.4f\n', bestParams.Dropout1);
fprintf('Best Dropout2: %.4f\n', bestParams.Dropout2);
fprintf('Best LearnRate: %.4e\n', bestParams.LearnRate);
fprintf('Best MiniBatchSize: %d\n', bestParams.MiniBatchSize);
fprintf('Best Activation1: %s\n', string(bestParams.Activation1));
fprintf('Best Activation2: %s\n', string(bestParams.Activation2));

fprintf('Training time: %.4f seconds\n', t_DNN);

fprintf('L2 relative error (dB): %.4e\n', L2_rel_Error);
fprintf('Linf relative error (dB): %.4e\n', Linf_rel_Error);
fprintf('----------------------\n');

%% Scatter plot
figure
plot(Y_pred(:),Y_TEST(:),'xg')
hold all;

set(gca,'FontSize',FS,'FontName',FN,'FontWeight',FW,'FontAngle',FA)

axis square

xlabel('Model Prediction','Fontsize',FS,'FontName',FN)
ylabel('Actual Values','Fontsize',FS,'FontName',FN)
legend({'Model #25'}, 'FontSize', FS, 'Location', 'southeast', 'FontName', 'Arial')

grid on
box on
set(gca,'LineWidth',1.2)
set(gcf,'Color','w')

%% Parametric comparison between reference and predicted responses
figure
plot(freq,Y_TEST(1,:),'k','linewidth',4)
hold all
plot(freq,Y_pred(1,:),'g:','linewidth',3)

plot(freq,Y_TEST(1:3:10,:),'k','linewidth',4)
plot(freq,Y_pred(1:3:10,:),'g:','linewidth',3)

set(gca,'FontSize',FS,'FontName',FN,'FontWeight',FW,'FontAngle',FA)
l=legend({'Reference','Model #25'},'FontSize',FS,'Location','best','FontName','Arial');

xlabel('Frequency (GHz)','Fontsize',FS,'FontName',FN);
ylabel('Magnitude (dB)','Fontsize',FS,'FontName',FN);

grid on
box on
set(gca,'LineWidth',1.2)
set(gcf,'Color','w')