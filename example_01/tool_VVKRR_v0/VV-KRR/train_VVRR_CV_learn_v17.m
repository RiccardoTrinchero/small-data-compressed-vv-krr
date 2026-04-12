function [MODEL] = train_VVRR_CV_learn_v17(X,Ygiven,kx_type,ko_type,D_OUT_ED,Nfold,tol,normalized,error_type,plot_debug)

% train_VVRR_BX_learn_v0 trains a Vector Valued Kernel Ridge regression 
% using the training pairs X and Y via Cross Validation and provides as output the model 
% coefficients and kenrel parameters.
% X: is a matrix collecting the input training samples of dimension (num. realizations x num. input parameters).
% Y: is a matrix collecting the output training samples of dimension (num., realizations x num. output components).
% Nfold: number of folds to be using in the cross-validation 
% Nite: number of iteration BO
% tol: tolerance used for the eigenvector decomposition
% kx: string to define the kernel used for the input parameters
% mu_only: mean value used to normilize the ouputs
% s_only: standard devition used to normlized the ouputs 
% REMOVED optimVars: optimization parameters defining the hyperparameters to be
% optimized by the BO


% TO BE DEFINED OUTSIDE
% defining range of variation for the model hyperparameters
%optimVars = [
%optimizableVariable('sigma2',[1e-2 1e6],'Transform','log')
%optimizableVariable('lambda',[1e-12 1e-1],'Transform','log')
%optimizableVariable('ntilde',[1 N_comp],'Type','integer')];

%% initialization
% defining the num. of training samples N_ED, input parameters N_param, and
% output dimensions N_OUT

disp('********************************************')
disp('Model Initialization')

%% Sizes

tStart = tic;

%% Sizes
[N_ED,N_param] = size(X);
N_OUT = size(Ygiven,2);

%% 1) Normalize outputs

switch lower(normalized)

    % original
    case 'o'
         Y=Ygiven;
         mu_only = zeros(1,N_OUT);
         s_only = ones(1,N_OUT);

    % standardization
    case 's'
        [Y,mu_only,s_only] = normalize(Ygiven, 'zscore');
        index_NAN = find(isnan(Y));
        Y(index_NAN) = 0;

    otherwise
        disp('NORMALIZATION NOT DEFINED, WAKE UP!!!');
end


%% OUTPUT KERNEL

if isnumeric(ko_type) && ismatrix(ko_type)

    % N_param_ko = 0;
    %% I must check this function
    [T,Mu] = eig_compr_psd_v2(ko_type,tol);

    if isvector(Mu)
        mu_vec = Mu(:);
    else
        mu_vec = diag(Mu);
    end

     D_OUT_ED = [];
     ko_optvars = [];

     N_param_ko = 0; 
else

    % no data driven kernel
    T= [];
    Mu = [];

     if isempty(D_OUT_ED)
        D_OUT_ED = linspace(-1,1,N_OUT).';
    end
    
    switch ko_type
        case 'RBF' 

            N_param_ko = 1; % Define number of hyperparameters
            
        case 'Matern52'
                       
            N_param_ko = 1; % Define number of hyperparameters            

        case 'Matern12'
             
            N_param_ko = 1; % Define number of hyperparameters            
        
    end     

end


%% PARAMETER KERNEL
% defining hyperparameters

switch kx_type
        case 'RBF' 
            N_param_kx = 1; % Define number of hyperparameters
                        
        case 'ardRBF'

            N_param_kx = N_param; % Define number of hyperparameters            
            
        case 'Matern52'
                       
            N_param_kx = 1; % Define number of hyperparameters            

        case 'Matern12'
                   
            N_param_kx = 1; % Define number of hyperparameters
                       
        case 'ardMatern52'
            
            N_param_kx = N_param; % Define number of hyperparameters
                        
        case 'ardMatern12'
            
            N_param_kx = N_param; % Define number of hyperparameters
                        
        otherwise
            disp('KERNEL NOT DEFINED, WAKE UP!!!');
end

% Warm start for hyperopt
sigma_l_ko_log_0 = log10(1 * ones(N_param_ko, 1)); % this is perfect for Matern 52

% modified
%sigma_l_ko_log_0 = log10(0.001 * ones(N_param_ko, 1));
sigma_l_kx_log_0 = log10(10 * ones(N_param_kx, 1));
eta_log_0 = -3;%-2; % it mean 10^-9 in log10 scale

hyperparam_log_0 = [sigma_l_ko_log_0;sigma_l_kx_log_0;eta_log_0];

%% optimizer setup 
opts = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'MaxIter', 150, ...
    'MaxFunctionEvaluations', 1500, ...
    'FiniteDifferenceType', 'central', ...
    'FiniteDifferenceStepSize', 1e-2);


disp('********************************************')
disp('Hyperparameters estimation')

% training samples partitioning via K-fold cross-validation
c_part = cvpartition(N_ED,'Kfold',Nfold);


% defining the optimization object for the hyperparameters learning
[ObjFcn] = @(hyperparam_log)EigDec_Discrete_Sylvestre_CV_fmin_v18(X,Y,kx_type,ko_type,D_OUT_ED,T,Mu,c_part,mu_only,s_only,tol,error_type,plot_debug,hyperparam_log);

% Optimize hyperparameters
[hyperparam_log_opt, fval_tmp] = fminunc(ObjFcn, hyperparam_log_0, opts);
  

sigma_l_ko_opt = hyperparam_log_opt(1:N_param_ko);
sigma_l_kx_opt = hyperparam_log_opt(N_param_ko+(1:N_param_kx));
eta_opt = hyperparam_log_opt(N_param_ko+N_param_kx+1);

disp('********************************************')
disp('Optimal Hyperparameters')


% %% assigning the hyperparamters
% if N_param_ko == 0
%     opt_sigma_l_ko = [];
% else
%     opt_sigma_l_ko = opt_sigma_l(1:N_param_ko);
%     opt_sigma_l_kx = opt_sigma_l((N_param_ko+1):end);
% end

%% model training
disp('********************************************')
disp('Model Training')

%defining model structure
MODEL.X = X;
% MODEL.params.mu = [];
% MODEL.params.nout_compr = [];
MODEL.kx_kernel.type = kx_type;
MODEL.kx_kernel.U = [];
MODEL.kx_kernel.Lambda = [];

MODEL.params.eta = 10.^(eta_opt);

if isnumeric(ko_type) && ismatrix(ko_type)
    MODEL.ko_kernel.type = 'GIVEN';
    MODEL.ko_kernel.B = ko_type;
    MODEL.ko_kernel.T = T;
    MODEL.ko_kernel.Mu = Mu;

    
    MODEL.params.ko_sigma_l = [];%opt_sigma_l(1:N_param_ko);
    MODEL.params.kx_sigma_l = 10.^(sigma_l_kx_opt);
    MODEL.D_OUT_ED = [];
    
else
    MODEL.ko_kernel.type = ko_type;
    MODEL.ko_kernel.B = [];
    %MODEL.ko_kernel.T = T;
    %MODEL.ko_kernel.Mu = Mu;

    MODEL.params.ko_sigma_l = 10.^(sigma_l_ko_opt);
    MODEL.params.kx_sigma_l = 10.^(sigma_l_kx_opt);
    MODEL.D_OUT_ED = D_OUT_ED;
    
end
MODEL.tol = tol;
MODEL.normalization.type = normalized;
MODEL.normalization.mu = mu_only;
MODEL.normalization.sigma = s_only;
MODEL.coeff = [];

[MODEL] = build_VVRR_model_v10(MODEL,Y,tol);


model_training_cost = toc;

MODEL.training_cost = model_training_cost;

disp('********************************************')
disp('Model TRAINED!!!!!!')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ObjFcn] = EigDec_Discrete_Sylvestre_CV_fmin_v18(X_ED_norm,Y_ED,kx_type,ko_type,D_OUT_ED,T,Mu,c_part,mu_only,s_only,tol,error_type,plot_debug,hyperparam_log)

%EigDec_Discrete_Sylvestre_CV_v11(X_ED_norm,Bor,T,Mu,kx_type,Y_ED,c_part,mu_only,s_only,tol,error_type,plot_debug)


           
        % gathering the number of K-fold actually used
        Kfold = c_part.NumTestSets;
        ObjFcn = 0;
            
        
        eps_norm = 1e-9;

        N_param_x = size(X_ED_norm,2);

        eta = 10.^(hyperparam_log(end));

        %% OUTPUT KERNEL

        % 🧠 Shrinkage Methods: The Basic Idea

% Shrinkage combines the empirical correlation matrix \hat{\bm{B}} with a “target” matrix (typically the identity matrix), to reduce estimation variance:
% 
% \bm{B}_{\text{shrink}} = (1 - \lambda) \hat{\bm{B}} + \lambda \mathbf{I},
% where:
% 	•	\lambda \in [0,1] is the shrinkage intensity (can be learned from the data),
% 	•	\mathbf{I} is the identity matrix (assumes independence among outputs),
% 	•	\bm{B}_{\text{shrink}} is the regularized output kernel.

        if isnumeric(ko_type) && ismatrix(ko_type)
        
            N_param_ko = 0; 
            B_full = ko_type;
            
        else
            
            switch ko_type
                case 'RBF' 
                    N_param_ko = 1;
                    sigma_f = 1;

                    sigma_ko_l  = 10.^(hyperparam_log(1:N_param_ko));
                    sigma_ko_l = sigma_ko_l(:);
                    
                    B_full = kernel_sqexp(D_OUT_ED,D_OUT_ED,sigma_f,sigma_ko_l);
    
                    % validation
                    %B_VAL =kernel_sqexp(D_VAL_norm,D_ED_norm,sigma_f,sigma_l);
                    
                    
                case 'Matern52'
                    
                    N_param_ko = 1; 
                    sigma_f = 1;

                    sigma_ko_l  = 10.^(hyperparam_log(1:N_param_ko));
                    sigma_ko_l = sigma_ko_l(:);

                    B_full = kernel_matern52(D_OUT_ED,D_OUT_ED,sigma_f,sigma_ko_l);
                    
                    % validation
                    %B = kernel_matern52(D_VAL_norm,D_ED_norm,sigma_f,sigma_l);
                            

                case 'Matern12'
                     
                    N_param_ko = 1; 
                    sigma_f = 1;

                    sigma_ko_l  = 10.^(hyperparam_log(1:N_param_ko));
                    sigma_ko_l = sigma_ko_l(:);

                    B_full = kernel_matern12(D_OUT_ED,D_OUT_ED,sigma_f,sigma_ko_l);
    
                    % validation
                    %B = kernel_matern52(D_VAL_norm,D_ED_norm,sigma_f,sigma_l);            
            end
                                
            % compute the T and Mu matrices if closed-form kernel is selected
            %% TO BE CHECKED
            [T,Mu] = eig_compr_psd_v2(B_full,tol);
        end

        

        
        
        for KK=1:Kfold
                       

            % defining training set at the KK-fold
            X_ED_K_norm = X_ED_norm(c_part.training(KK),:);
            Y_ED_K = Y_ED(c_part.training(KK),:);

            % defining validation set at the KK-fold
            X_VAL_K_norm = X_ED_norm(c_part.test(KK),:);
            Y_VAL_K = Y_ED(c_part.test(KK),:);


            switch kx_type
                

                case 'RBF' 

                    N_param_kx = 1;

                    sigma_f = 1;                    

                    sigma_kx_l  = 10.^(hyperparam_log(N_param_ko+(1:N_param_kx)));
                    sigma_kx_l = sigma_kx_l(:);

                    Kx = kernel_sqexp(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                    % validation
                    Kx_VAL = kernel_sqexp(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
                
                case 'ardRBF'

                    N_param_kx = N_param_x;

                    sigma_f = 1;                    

                    sigma_kx_l  = 10.^(hyperparam_log(N_param_ko+(1:N_param_kx)));
                    sigma_kx_l = sigma_kx_l(:);

                    Kx = kernel_ardsqexp(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
                    
                    % validation
                    Kx_VAL = kernel_ardsqexp(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                case 'Matern52'
                                                    
                    N_param_kx = 1;

                    sigma_f = 1;                    

                    sigma_kx_l  = 10.^(hyperparam_log(N_param_ko+(1:N_param_kx)));
                    sigma_kx_l = sigma_kx_l(:);                    
                    
                    Kx = kernel_matern52(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                    % validation
                    Kx_VAL = kernel_matern52(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
                

                case 'ardMatern52'
                    
                    N_param_kx = N_param_x;

                    sigma_f = 1;                    

                    sigma_kx_l  = 10.^(hyperparam_log(N_param_ko+(1:N_param_kx)));
                    sigma_kx_l = sigma_kx_l(:);

                    Kx = kernel_ardmatern52(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                    % validation
                    Kx_VAL = kernel_ardmatern52(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                case 'Matern12'
                                    
                    N_param_kx = 1;

                    sigma_f = 1;                    

                    sigma_kx_l  = 10.^(hyperparam_log(N_param_ko+(1:N_param_kx)));
                    sigma_kx_l = sigma_kx_l(:);

                    Kx = kernel_matern12(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                    % validation
                    Kx_VAL = kernel_matern12(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
       
                case 'ardMatern12'
                    
                    N_param_kx = N_param_x;

                    sigma_f = 1;                    

                    sigma_kx_l  = 10.^(hyperparam_log(N_param_ko+(1:N_param_kx)));
                    sigma_kx_l = sigma_kx_l(:);

                    Kx = kernel_ardmatern12(X_ED_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);

                    % validation
                    Kx_VAL = kernel_ardmatern12(X_VAL_K_norm,X_ED_K_norm,sigma_f,sigma_kx_l);
                
            end

            %% TO BE CHECKED
            [U,Lambda] = eig_compr_psd_v2(0.5*(Kx+Kx'),tol);

            %%
            % y tilde
            Y_ED_tilde = U'*Y_ED_K*T;

            if isempty(Lambda) || isempty(Mu)                
                ObjFcn = Inf;
                return;
            else
                lambda_reg = eta * Lambda(end,end) * Mu(end,end);
            end
            
               
            % compute c tilde
            c_tilde = Y_ED_tilde ./ (diag(Lambda) .* diag(Mu)' + lambda_reg);
    
            % going back to C            
            C = U*c_tilde*T';
            
            % evaluate the model on the validation set
            %Y_VAL_MOKK = Kx_VAL*C*B;
            Y_VAL_MOKK = (Kx_VAL*U)*c_tilde*(T'*B_full);
      
            
            % unormalize
            Y_VAL_MOKK_UN = Y_VAL_MOKK.*s_only + mu_only;
            Y_VAL_K_UN = Y_VAL_K.*s_only + mu_only;


            %Y_VAL_MOKK_tilde = Y_VAL_MOKK_UN(:)reshape(Y_VAL_MOKK_UN,[N_VAL_K*N_OUT,1]);
            %Y_VAL_tilde = reshape(Y_VAL_K_UN,[N_VAL_K*N_OUT,1]);


           
            % plot for the debug
            switch plot_debug
                case 'S'

                %% check norm and accuracy of the solution of the sylvester equation
                %norm_error_Sylv = norm(Kx*C*Bor+C*optimVars.lambda-Y_ED_K,'fro')/norm(Y_ED_K,'fro')*100;
                norm_error_Sylv_tilde = norm(Lambda*c_tilde*Mu+c_tilde*lambda_reg-Y_ED_tilde,'fro')/norm(Y_ED_tilde,'fro')*100;
    
                %if norm_error_Sylv>50
                %    keyboard
                %end
    
                figure(1000)
                subplot(1,2,1)
                plot((Kx*C*B)','b')
                hold on
                plot(Y_ED_K.','--r')
                title(['Error Sylv.: ',num2str(norm_error_Sylv), 'Error tilde: ',num2str(norm_error_Sylv_tilde)])
                hold off 
    
                figure(1000)
                subplot(1,2,2)
                plot(Y_VAL_MOKK_UN.','b')
                hold on
                plot(Y_VAL_K_UN.','--r')
                hold off
            end

            % summing up the error computed over each fold
            eps_norm = 1e-9;  % Small value to prevent division by zero
            
            
            % Fold error
            switch error_type
                case 'L2'

                    ObjFcn = ObjFcn + norm(Y_VAL_K_UN(:) - Y_VAL_MOKK_UN(:), 'fro').^2 / ...
                              (norm(Y_VAL_K_UN(:), 'fro') + eps_norm).^2;
        
                case 'L1'
                    ObjFcn = ObjFcn + sum(abs(Y_VAL_K_UN(:) - Y_VAL_MOKK_UN(:))) / ...
                                      (sum(abs(Y_VAL_K_UN(:))) + eps_norm);
        
                otherwise               
                    disp('ERROR NOT DEFINED, WAKE UP!!!');
            end

        end

        ObjFcn = ObjFcn / Kfold;   
            
   end

       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MODEL]=build_VVRR_model_v10(MODEL,Y,tol)

    %defining model structure
    % MODEL.X = X;
    % MODEL.D_OUT = [];
    % MODEL.params.lambda = opt_lambda;
    % MODEL.params.sigma_l = opt_sigma_l;
    % MODEL.params.kB_sigma = [];
    % MODEL.params.mu = [];
    % MODEL.params.nout_compr = size(T,2);
    % MODEL.k_kernel.type = kernel_type;
    % MODEL.B_kernel.type = 'GIVEN';
    % MODEL.B_kernel.tilde = [];%T_B(:,1:ntilde_opt_Syl);
    % MODEL.coeff = [];
    
    % reading model structure
    X = MODEL.X;

    

    sigma_f = 1;

    %% defining output kenrnel
    
    if strcmp(MODEL.ko_kernel.type, 'GIVEN')

        T = MODEL.ko_kernel.T;
        Mu = MODEL.ko_kernel.Mu;

        B_full = MODEL.ko_kernel.B;

    else

        D_OUT_ED = MODEL.D_OUT_ED;

        switch MODEL.ko_kernel.type

                case 'RBF' 
        
                    sigma_f = 1;
                    B_full = kernel_sqexp(D_OUT_ED,D_OUT_ED,sigma_f,MODEL.params.ko_sigma_l);
    
                    % validation
                    %B_VAL =kernel_sqexp(D_VAL_norm,D_ED_norm,sigma_f,sigma_l);                    
                    
                case 'Matern52'
                    
                    sigma_f = 1;
                    B_full = kernel_matern52(D_OUT_ED,D_OUT_ED,sigma_f,MODEL.params.ko_sigma_l);
                    
                    % validation
                    %B = kernel_matern52(D_VAL_norm,D_ED_norm,sigma_f,sigma_l);                           

                case 'Matern12'
                     
                    sigma_f = 1;
                    B_full = kernel_matern12(D_OUT_ED,D_OUT_ED,sigma_f,MODEL.params.ko_sigma_l);
    
            end
    
           % compute the T and Mu matrices if closed-form kernel is selected
           %% TO BE CHECKED
           [T,Mu] = eig_compr_psd_v2(B_full,tol);
    end

    


    %% defining input kernel
    switch MODEL.kx_kernel.type
        case 'RBF' 
            
            K_kernel = kernel_sqexp(X,X,sigma_f,MODEL.params.kx_sigma_l);
            %Kx = kernel_RBF_v1(X_ED_K_norm,X_ED_K_norm,optimVars);
    
        case 'ardRBF'
            
            K_kernel = kernel_ardsqexp(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
       case 'Matern52'
                                                    
            K_kernel = kernel_matern52(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
    
        case 'ardMatern52'
                                
            K_kernel = kernel_ardmatern52(X,X,sigma_f,MODEL.params.kx_sigma_l);

        case 'Matern12'
                                                    
            K_kernel = kernel_matern12(X,X,sigma_f,MODEL.params.kx_sigma_l);
    
    
        case 'ardMatern12'
                                
            K_kernel = kernel_ardmatern12(X,X,sigma_f,MODEL.params.kx_sigma_l);
     
    end
    
    
    % eigen decomposition Kx
    %% TO BE CHECKED
    [U,Lambda] = eig_compr_psd_v2(0.5*(K_kernel+K_kernel'),tol);
    

    %% tranformed version of the ouput training samples
    Y_tilde = U'*Y*T;
    
    %% computing the compressed coefficients
    
    % c_tilde = zeros(size(Lambda,1),ntilde);
    % 
    % for ii=1:size(Lambda,1)
    %     for jj=1:ntilde
    % 
    %         c_tilde(ii,jj) = Y_tilde(ii,jj)/(Lambda(ii,ii)*Mu(jj,jj)+lambda);
    %     end
    % end
    
    %% full C
    % going back to C            
    %C = U*c_tilde*T';

    %%
    %lambda_reg = sum(diag(Lambda))*MODEL.params.eta/size(Lambda,1);

    lambda_reg = MODEL.params.eta*Lambda(end,end)*Mu(end,end);
               
    % compute c tilde
    c_tilde = Y_tilde ./ (diag(Lambda) .* diag(Mu)' + lambda_reg);

    
    MODEL.coeff = c_tilde;
    MODEL.ko_kernel.T = T;
    MODEL.ko_kernel.Mu = Mu;
    MODEL.kx_kernel.U = U;
    MODEL.kx_kernel.Lambda = Lambda;

    MODEL.nx = size(U,2);
    MODEL.no = size(T,2);
    MODEL.compression_percentage = (1-(MODEL.nx*MODEL.no)/(size(U,1)*size(T,1)))*100;

    %norm_error_Sylv = norm((K_kernel*U)*MODEL.coeff*(T'*B_full)+U*MODEL.coeff*T'*MODEL.params.lambda-Y,'fro')/norm(Y,'fro')*100
    %norm_error_Sylv_tilde = norm(Lambda*c_tilde*Mu+c_tilde*MODEL.params.lambda-Y_tilde,'fro')/norm(Y,'fro')*100

    MODEL.Syl_Error = 0;%norm_error_Sylv;


end