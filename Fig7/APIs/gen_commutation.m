function K = gen_commutation(m,n)

K = 0;
for j = 1 : n
   ej= zeros(n, 1);
   ej(j) = 1;
   K = K + (kron(kron(ej.', eye(m)), ej));
end