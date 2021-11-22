function erate = get_erate(Hc, K, Ns, NUE, NBS, Sigma_d2, M, T, Ttol, H, G)
%% Initialize
i = 0;
v = exp(1j * unifrnd(0, 2 * pi, M, 1));  % random IRS reflection
He = reshape(K * conj(Hc) * v, NUE, NBS);
[F, ~] = SVD_method(He, Ns);
Wcs = @(He, F, Vn) (He * F * F' * He' + Vn * eye(NUE))^(-1) * He * F; % cs : closed-form
Ecs = @(He, F, W) eye(Ns) - F' * He' * W -  W' * He * F + Sigma_d2 * W' * W +  W' * He * F * F' * He' * W;
W = Wcs(He, F, Sigma_d2);
E = Ecs(He, F, W);
Omega = E^-1;
g(1) = trace(Omega * E) - log2(det(Omega));
while i < 1 || (g(i) - g(i+1))> 1e-3
    %% Optimize vd
    problem.cost = @(x)v_cost(x, Omega, F, Sigma_d2, Hc);
    problem.egrad = @(x)v_egrad(x, Omega, F, Sigma_d2, Hc);
    problem.init = v;
    result = manifold_opt(problem);
    v = result.x;
    %% F, W, Omega
    He = reshape(K * conj(Hc) * v, NUE, NBS);
    W = Wcs(He, F, Sigma_d2);
    E = Ecs(He, F, W);
    Omega = E^-1;
    psi = trace(Omega * W' * W);
    i = i + 1;
    g(i+1) = trace(Omega * E) - log2(det(Omega));
    F_hat = (He' * W * Omega * W' * He + Sigma_d2 * psi * eye(NBS))^-1 * He' * W * Omega;
    F = F_hat / norm(F_hat, 'fro');
    E = Ecs(He, F, W);
end
He = H' * diag(v) * G';
tmp = det(eye(Ns) + 1 / Sigma_d2   *  F' * He' * He * F);
erate = log2(tmp) * ( 1 - T / Ttol);


    function g = v_egrad(v, Omega, F, Vn, Hc)
        He_v = reshape(K * conj(Hc) * v, NUE, NBS);
        T2 = Omega^-1 + 1/Vn * Omega^-1 * F' * He_v' * He_v * F;
        m = (He_v * F * T2^-2 * Omega^-1 * F').';
        m = m(:);
        g = - Hc.' * m / Vn;
    end

    function f = v_cost(v, Omega, F, Vn, Hc)
        He_v = reshape(K * conj(Hc) * v, NUE, NBS);
        f = trace((Omega^-1 + 1 / Vn * Omega^-1 * F' * He_v' * He_v * F)^-1);
    end
end
