function [v, b] = TSVD(P, v, Ns)


problem.cost = @(x)v_cost(x, P, Ns);
problem.egrad = @(x)v_egrad(x, P, Ns);
problem.init = v;
result = manifold_opt(problem);
v = result.x;
b = result.cost;
    function f = v_cost(v, P, Ns)
        f = 0;
        for i = 1 : Ns
            p = P(:, i);
            f = f + log2(1 + v' * p * p' * v);
        end
        f = -f;
    end

    function g = v_egrad(v, P, Ns)
        g = 0;
        for i = 1 : Ns
            p = P(:, i);
            g = g + 1 / log(2) * 2 * p * p' * v / (1+v' * p * p' * v);
        end        
        g = -g;
    end
end