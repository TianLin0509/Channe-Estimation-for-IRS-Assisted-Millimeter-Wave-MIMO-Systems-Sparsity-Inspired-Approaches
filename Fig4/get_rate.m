function rate = get_rate(H, F, Vn)

Ns = size(F,2);
tmp = det(eye(Ns) + 1 / Vn   *  F' * H' * H * F);
rate = log2(tmp);