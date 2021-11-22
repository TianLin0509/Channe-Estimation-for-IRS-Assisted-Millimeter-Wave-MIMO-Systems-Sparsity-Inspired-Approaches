function rate = get_rate2(Nr, H, F, Vn)

tmp = det(eye(Nr) + 1 / Vn  * H * F * F' * H');
rate = log2(tmp);