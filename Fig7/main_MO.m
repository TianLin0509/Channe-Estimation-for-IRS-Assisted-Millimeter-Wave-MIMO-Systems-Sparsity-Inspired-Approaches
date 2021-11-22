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
Nloop = 10;
T = 200; % training overhead
Ttol = 2000;
Ns = 3; % data streams
C = 3; % the number of paths of channels
SNR = 10;
Sigma_d2 = 1 / db2pow(SNR);
PNR_set = 10;
GBS = NBS;
Gy = My;
Gz = Mz;
GUE = NUE;
K = gen_commutation(NBS, NUE);
miu_G = 0;  % Lagrange multipliers, should be adjusted for different settings
miu_H = 0;

% dictionaries
f = @(u, N) exp(1j*pi*(0:N-1)'*u) / sqrt(N);
ABS = f((-1: 2/GBS: 1 - 2/GBS), NBS);
AUE = f((-1: 2/GUE: 1 - 2/GUE), NUE);
Ay =  f((-1: 2/Gy: 1 - 2/Gy), My);
Az =  f((-1: 2/Gz: 1 - 2/Gz), Mz);
AI = kron(Ay, Az);

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
        Z = (randn(NBS, T) + 1j * randn(NBS, T)) * sqrt(sigma2 / 2); % [z_1, ..., z_T]
        R = zeros(NBS, T);
        F = zeros(M, T);
        F2 = zeros(NBS * T, M*NUE);
        for t = 1 : T
            R(:, t) = G * diag(V(:, t)) * H * S(:, t) + Z(:, t);
        end
        % MO-EST
        % random initilization
        H_hat = (randn(M, C) + 1j * randn(M, C)) * (randn(C, NUE) + 1j * randn(C, NUE));
        G_hat = (randn(NBS, C) + 1j * randn(NBS, C)) * (randn(C, M) + 1j * randn(C, M));
        k = 0;  k_max = 20;  eps1 = 10^-3;
        while k < k_max
            % Optimize G_hat with given H_hat
            k = k + 1;
            for t = 1 : T
                F(:, t) =  diag(V(:, t)) * H_hat * S(:, t);
            end
%             if k == 1
%                 G_hat = G_init(R, F);
%             end
            problem_G.M = fixedrankembeddedfactory(NBS, M, C);
            problem_G.cost = @(X)f1(X, R, F, miu_G, ABS, AI);
            problem_G.egrad = @(X)egrad_f1(X, R, F, miu_G, ABS, AI);
            [X0.U, X0.S, X0.V] = svds(G_hat, C);
            [X, f1_cost, info1] = conjugategradient(problem_G, X0, options);
            G_hat = X.U*X.S*X.V';
            problem_H.M = fixedrankembeddedfactory(M, NUE, C);
            problem_H.cost = @(X)f2(X, R, V, S, G_hat, miu_H, AUE, AI);
            problem_H.egrad = @(X)egrad_f2(X, R, V, S, G_hat, miu_H, AUE, AI);
            [X0.U, X0.S, X0.V] = svds(H_hat, C);
            [X, f_loss(k+1), info2] = conjugategradient(problem_H, X0, options);
            H_hat = X.U*X.S*X.V';
            if k>1 && f_loss(k) - f_loss(k+1) < eps1
                break
            end
        end
        iter_num(PNR_idx, n) = k;
        % Mse
        Hc = khatri_pro(H.', G);
        Hc_hat = khatri_pro(H_hat.', G_hat);
        mse(PNR_idx, n) = norm(Hc_hat - Hc, 'fro')^2 / norm(Hc, 'fro')^2;
        r_MO(PNR_idx, n)  = get_erate(Hc_hat, K, Ns, NUE, NBS, Sigma_d2, M, T, Ttol, H, G);
%         r_Perfect(PNR_idx, n)  = get_erate(Hc, K, Ns, NUE, NBS, Sigma_d2, M, T, Ttol, H, G);
        % Rate        
    end   
end
toc
mean(mse,2)
mean(r_MO, 2)
function cost = f1(X, R, F, miu_G, ABS, AI)
    X = X.U*X.S*X.V'; % map SVD back to X
    c1 = norm(R - X * F, 'fro')^2;
    c2 = miu_G * norm(vec(ABS' * X * AI), 1);
    cost = c1 + c2;
end

function egrad = egrad_f1(X, R, F, miu_G, ABS, AI)
    X = X.U*X.S*X.V';
    egrad1 = - R * F' + X * F * F';
    Y = ABS' * X * AI;
    Y = Y ./ abs(Y);
    egrad = egrad1 + miu_G / 2 * ABS * Y * AI';
end

function cost = f2(X, R, V, S, G_hat, miu_H, AUE, AI)
    X = X.U*X.S*X.V'; % map SVD back to X
    cost = 0;
    for t = 1 : size(R, 2)
        cost = cost + norm(R(:,t) - G_hat * diag(V(:,t)) * X * S(:,t))^2;
    end
    cost = cost + miu_H * norm(vec(AI' * X * AUE), 1);
end

function egrad = egrad_f2(X, R, V, S, G_hat, miu_H, AUE, AI)
    X = X.U*X.S*X.V';
    egrad = 0;
    for t = 1 : size(R, 2)
        phit = diag(V(:,t));
        egrad = egrad - phit' * G_hat' * R(:,t) * S(:,t)' + phit' * G_hat' * G_hat * phit * X * S(:,t) * S(:,t)';
    end
    Y2 = AI' * X * AUE;
    Y2 = Y2 ./ abs(Y2);
    egrad = egrad + miu_H / 2 * AI * Y2 * AUE';
end


function G_hat = G_init(R, F)
     [u, v, e] = svds(R, 3);
     G_hat = u * v * e' * pinv(F);
end
    





















