function [ idx, FRF, FBB ] = OMP_col_fun( Fopt, NRF, At )

FRF = [];
Fres = Fopt;
for k = 1:NRF
    PU = At' * Fres;
%     [aa,bb] = max(diag(PU * PU'));
    [aa,bb] = max(sum( abs(PU).^2, 2 ));
    FRF = [FRF , At(:,bb)];
    FBB = pinv(FRF) * Fopt; %use pseudoinverse to avoid the inverse of a possible singular matrix
%     Fres = (Fopt - FRF * FBB) / norm(Fopt - FRF * FBB,'fro');
    Fres = (Fopt - FRF * FBB);
    idx(k) = bb;
end

end

