function [newx, newcost] = Armijo_linesearch(x, mgrad, f0, problem)

alpha = 1 / norm(mgrad);  % Initialize stepsize alpha;
step = 0;

while true
    newx = x - alpha * mgrad;
    newx = newx ./ abs(newx);
    newcost = problem.cost(newx);
%     if newcost > f0 - 0.5 * alpha * mgrad' * mgrad
    if newcost > f0 - 1e-4 * alpha * mgrad' * mgrad
        alpha = 0.5 * alpha;
        step = step + 1;
        if step > 20
            break
        end
    else
        break
    end
end
  a = 0;  
    
    