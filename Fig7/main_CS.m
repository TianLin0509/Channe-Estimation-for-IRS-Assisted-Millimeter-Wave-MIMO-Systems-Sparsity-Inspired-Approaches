addpath(genpath('manopt'),'APIs')
clear all; clc;
load('s.mat')   % fix random seeds
% rng_seed = rng(0)
rng(rng_seed);
%% Basic Parameters
NBS = 36;
NUE = 16;
My = 6;
Mz =6;
M = My * Mz;
Nloop = 20;
T = 200; % training overhead
Ns = 3; % data streams
C = 3; % the number of paths of channels
Ttol = 2000;
SNR = 10;
Sigma_d2 = 1 / db2pow(SNR);
PNR_set = 10;
GBS = 64;
Gy = 16;
Gz = 16;
GI = Gy * Gz;
GUE = 64;
K = gen_commutation(NBS, NUE);
% dictionaries
f = @(u, N) exp(1j*pi*(0:N-1)'*u) / sqrt(N);
ABS = f((-1: 2/GBS: 1 - 2/GBS), NBS);
AUE = f((-1: 2/GUE: 1 - 2/GUE), NUE);
Ay =  f((-1: 2/Gy: 1 - 2/Gy), My);
Az =  f((-1: 2/Gz: 1 - 2/Gz), Mz);
AI = kron(Ay, Az);
T1 = ceil(T / 4);
options.verbosity = 0;
tic
%% Simulation
for PNR_idx = 1 : length(PNR_set)
    sigma2 = 1 / db2pow(PNR_set(PNR_idx));
    for n = 1 : Nloop
        % channel generation
        HAOA = unifrnd(-1, 1, C, 2); %azimuth and elevation
         HAOD = unifrnd(-1, 1, C, 1); % azimuth
        beta = (randn(C,1) + 1j * randn(C,1)) / sqrt(2);
        beta(2:end) = beta(2:end) * 10^(-0.25);
         GAOA = unifrnd(-1, 1, C, 1);
        GAOD = unifrnd(-1, 1, C, 2);
        alpha = (randn(C,1) + 1j * randn(C,1)) / sqrt(2);
        alpha(2:end) = alpha(2:end)  * 10^(-0.25);
        H = 0; G = 0;
        for c = 1 : C
            H = H + beta(c) * kron(f(HAOA(c, 1),My), f(HAOA(c, 2),Mz)) * f(HAOD(c), NUE)';
            G = G + alpha(c) * f(GAOA(c), NBS) * (kron(f(GAOD(c, 1),My), f(GAOD(c, 2),Mz)))';
        end
        H = H * sqrt(NUE * M / C);
        G = G * sqrt(NBS * M / C);
        % uplink pilots;
        S = exp(1j * unifrnd(0, 2*pi, NUE, T)) / sqrt(NUE); %[s_1, ..., s_T]
        V = exp(1j * unifrnd(0, 2*pi, M, T)); %[v_1, ..., v_T]
        V(:, 1 : T1) = repmat(V(:,1), 1, T1);
        Z = (randn(NBS, T) + 1j * randn(NBS, T)) * sqrt(sigma2 / 2); % [z_1, ..., z_T]
        R = zeros(NBS, T);
        F = zeros(M, T);
        for t = 1 : T
            R(:, t) = G * diag(V(:, t)) * H * S(:, t) + Z(:, t);
        end
        % CS-EST
        idx = OMP_col_fun(R, C, ABS);
        bar_ABS = ABS(:, idx);
        x = (-1:2/GBS:1 - 2/GBS);
%         x(idx)        
        R1 = R(:, 1 : T1);
        S1 = S(:, 1: T1);
        idx = OMP_col_fun(R1', C, S1' * AUE);
        bar_AUE = AUE(:, idx);
        x = (-1:2/GUE:1 - 2/GUE);
%         x(idx)  
        obs = zeros(T * NBS, C^2*GI);
        for t = 1 : T  
            obs((t-1) * NBS + 1 : t * NBS, :) = kron(V(:,t).' * AI, kron(S(:, t).' * conj(bar_AUE), bar_ABS));
        end
        lambda = BOMP(R(:), obs, 1, 0);
        bar_Lambda = reshape(lambda, C^2, GI);
        % Mse
        Hc = khatri_pro(H.', G);
        Hc_hat = kron(conj(bar_AUE), bar_ABS) * bar_Lambda * AI.';
        mse(PNR_idx, n) = norm(Hc_hat - Hc, 'fro')^2 / norm(Hc, 'fro')^2;
        r_CS(PNR_idx, n)  = get_erate(Hc_hat, K, Ns, NUE, NBS, Sigma_d2, M, T, Ttol, H, G);
%         r(PNR_idx, n)  = get_BFrate(G, H, Hc_hat, Ns, Sigma_d2);
        % Rate        
    end   
end
toc
mean(mse,2)
mean(r_CS, 2)




















