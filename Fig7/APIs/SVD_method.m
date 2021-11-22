function [F, W] = SVD_method(H, Ns)
% traditional SVD algorithm for rate maximization

[U,~,V] = svd(H);
V_ropt = V(:,1:Ns);
%power constraint
F = V_ropt / norm(V_ropt,'fro');
W = U(:,1:Ns);
