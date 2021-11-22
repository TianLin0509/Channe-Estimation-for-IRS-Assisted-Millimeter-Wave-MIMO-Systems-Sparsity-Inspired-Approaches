clear all;
clc;
%% Parameter Settings
NUE = 16;
NBS = 36;
My =6;
Mz =6;
M = My * Mz;
P = 3;
Q = 3;
Ns = 3;
SNR_set = 0;
load('s.mat') % fix random seeds
rng(rng_seed);
Nloop = 100;

for snr_idx = 1 : length(SNR_set)
    snr = 10^(SNR_set(snr_idx) / 10);
    Vn = 1 / snr;
    for  n = 1 : Nloop
        n
        %% Generate Channel
        [H, G, Pow] = channel_generation(My, Mz, NUE, Q, NBS, P, Ns, Vn);
        
        %% Random Benchmark
        v_random = exp(1j * unifrnd(0, 2 * pi, M, 1));  % random IRS reflection
        He = H * diag(v_random) * G;   % equivalent channel
        [F_random, ~] = SVD_method(He, Ns);   % regard the equivalent channel as traditional MIMO
        rate_random(snr_idx, n) = get_rate(He, F_random, Vn);
        
        %% TSVD Benchmark
        [v_TSVD, b] = TSVD(Pow, conj(v_random), Ns);  % use v_random as its initialization for TSVD
        v_TSVD = conj(v_TSVD);
        He = H * diag(v_TSVD) * G;
        [F, ~] = SVD_method(He, Ns);
        rate_TSVD(snr_idx, n) = get_rate(He, F, Vn);
        
        %% ALT-WMMSE
        [v_ALT, F] = ALT_WMMSE(H, G, v_random, F_random, Ns, NUE, NBS, Vn);
        He = H * diag(v_ALT) * G;
        rate_ALT(snr_idx, n) = get_rate(He, F, Vn);
    end
end
mean(rate_random, 2)
mean(rate_TSVD, 2)
mean(rate_ALT, 2)

