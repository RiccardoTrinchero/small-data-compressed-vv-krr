function [H_9]= function_highspeed_tf_v2(freq,C1,C2,L1_1,L1_2,eps_r,ww,tt,hh,Len1,Len2,Len3)


%% Circuit parameters

%number of stochastic variables
d = 11;

G0 = 1/50;
G1 = 1/75;
G2 = 1/5;
G3 = 1/25;
C1n = 1e-12;
C2n = 0.5e-12;
L1n = 10e-9;
L2n = 6e-9;

% Transmission line
mu0  = 4*pi*1e-7;
eps0 = 8.8542e-12;
%Len1 = 5e-2; % m
%Len2 = 3e-2; % m
%Len3 = 3e-2; % m

%eps_r = 4 ;      % 4; % relative permittivity
%ww = 252e-6 ;   % 0.1e-3; % width in meters 
%tt = 35e-6 ;    % 0.035e-3; % thickness in meters
%hh = 60e-6 ;    % 0.060e-3; % height in meters


%%

%% =========================================================================================
% Simulation as a function of 11 paramaters (C1,C2,L1_1,L1_2,eps_r,ww,tt,hh,Len1,Len2,Len3)
% ==========================================================================================

E = 1;


dim_MNA = 13;
MNA = zeros(dim_MNA);


% dumping term
sigma = 0;


    
% Excitation in Laplace-domian
s = sigma+1j*2*pi*freq;            

%excitation
E_s = 1;    
    
%--------------------------------------------------------------------------
% MNA    
%--------------------------------------------------------------------------
        
%     %MC  parameters        
%     C1(K) = C1n *(1+delta*x1(K));
%     C2(K) = C2n *(1+delta*x2(K));
%     L1_1(K) = L1n * (1+delta*x3(K));
%     L1_2(K) = L1n *(1+delta*x4(K));              
        
        
    %-----------------------------------------------------------------
    % TRANSMISSION LINES DEFINED IN TERMS PHYSICAL-BASED FORMULAS
    % FOR THE uSTRUP STRUCTURE
    %-----------------------------------------------------------------
  

    [ZC,c,l]=microstrip_single(eps_r,ww,tt,hh);

    Z01  = ZC; % e.g., 50 Ohm char. impedance
    Z02  = ZC;
    Z03  = ZC;

    %-----------------------------------------------------------------

    gamma = sqrt(s^2*l*c);

    gamma1 = gamma;
    gamma2 = gamma;
    gamma3 = gamma;


    Y1_11 = 1./Z01./tanh(gamma1*Len1);
    Y1_21 = 1./Z01./sinh(gamma1*Len1);
    Y1_12 = 1./Z01./sinh(gamma1*Len1);
    Y1_22 = 1./Z01./tanh(gamma1*Len1);

    Y2_11 = 1./Z02./tanh(gamma2*Len2);
    Y2_12 = 1./Z02./sinh(gamma2*Len2);
    Y2_21 = 1./Z02./sinh(gamma2*Len2);
    Y2_22 = 1./Z02./tanh(gamma2*Len2);

    Y3_11 = 1./Z02./tanh(gamma2*Len3);
    Y3_12 = 1./Z02./sinh(gamma2*Len3);
    Y3_21 = 1./Z02./sinh(gamma2*Len3);
    Y3_22 = 1./Z02./tanh(gamma2*Len3);

    
      
                
        %Circuit excitation
        A = zeros(13,1);
        A(1)= E_s * G1;

        MNA(1,1) =  G1;        
        MNA(1,10) = 1;
        
        MNA(2,2) =  s*C1n+G3+Y1_11;
        MNA(2,3) = -G3;
        MNA(2,6) = -Y1_21;
        MNA(2,10) = -1;
        
        MNA(3,2) = -G3;
        MNA(3,3) = G3;
        MNA(3,11) = 1;
        
        
        MNA(4,4) = s*C1n+Y2_11;
        MNA(4,5) = -Y2_21;
        MNA(4,11) = -1;
        
        MNA(5,4) = -Y2_12;
        MNA(5,5) = Y2_22+G2+s*C2n;
        
        MNA(6,2) = -Y1_12;
        MNA(6,6) = Y1_22;
        MNA(6,12) = 1;
        
        
        MNA(7,7) = s*C1;
        MNA(7,12) = -1;
        MNA(7,13) = 1;       
        
        
        MNA(8,8) = Y3_11;
        MNA(8,9) = -Y3_21;
        MNA(8,13) = -1;
        
        MNA(9,8) = -Y3_12;
        MNA(9,9) = Y3_22+s*C2+G0;
        
        MNA(10,1) = 1;
        MNA(10,2) = -1;
        MNA(10,10) = -s*L1_1;
        
        MNA(11,3) = 1;
        MNA(11,4) = -1;
        MNA(11,11) = -s*L2n;
        
        MNA(12,6) = 1;
        MNA(12,7) = -1;
        MNA(12,12) = -s*L1n;
        
        MNA(13,7) = 1;       
        MNA(13,8) = -1;
        MNA(13,13) = -s*L1_2;
        
               
        tmp = inv(MNA) * A; 

        H_9= tmp(9);  
        
      
    end
    %progressbar(J/Nf);

    






