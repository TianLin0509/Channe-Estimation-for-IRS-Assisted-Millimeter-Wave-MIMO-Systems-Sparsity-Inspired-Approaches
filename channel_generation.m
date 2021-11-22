function [H, G, Pow] = channel_generation(My, Mz, NUE, Q, NBS, P, Ns, Vn)

beta(1) = (randn(1) + 1j * randn(1)) / sqrt(2); % gain of the LoS after normalization
beta(2:Q) = 10^(-0.25)*(randn(1,Q-1)+1i*randn(1,Q-1))/sqrt(2); % gain of the NLoS
beta = sort(beta, 'descend');
H = zeros(NUE, My * Mz);
avec = @(t, y) exp(1j*pi*(0:y-1)'*t) / sqrt(y); % array response of ULA
for l=1:Q
    ar = avec(unifrnd(-1, 1), NUE);
    at = array_response_UPA(unifrnd(0, pi),unifrnd(-pi/2, pi/2), My, Mz);
    H = H + sqrt(NUE * My * Mz) * beta(l) * ar * at';
    A(:, l) = at;  % for further use
end

H = H / sqrt(Q);

alpha(1) = (randn(1) + 1j * randn(1)) / sqrt(2); % gain of the LoS
alpha(2:P) = 10^(-0.25)*(randn(1,P-1)+1i*randn(1,P-1))/sqrt(2); % gain of the NLoS
alpha = sort(alpha, 'descend');
G = zeros(My * Mz, NBS);
for l=1:P
    ar = array_response_UPA(unifrnd(0, pi),unifrnd(-pi/2, pi/2), My, Mz);
    at = avec(unifrnd(-1, 1), NBS);
    G = G + sqrt(NBS * My * Mz) * alpha(l) * ar * at';
    B(:, l) = ar;
end

G = G / sqrt(P);

% for further use
for i = 1 : Ns
    tmp = 1 / Ns / Vn * (abs(beta(1) * alpha(1)))^2 * sqrt(NBS * My * Mz)...
        * sqrt(NUE * My * Mz);
    Pow(:, i) =  (conj(A(:, i))) .* B(:,i) * sqrt(tmp);
end

