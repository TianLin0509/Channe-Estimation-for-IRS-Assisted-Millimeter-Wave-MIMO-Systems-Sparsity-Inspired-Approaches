function a = array_response_UPA(theta, phi, Ny, Nz)

ay = 1 / sqrt(Ny) * exp((0 : Ny - 1)' * 1j * pi * sin(theta) * sin(phi));
az = 1 / sqrt(Nz) * exp((0 : Nz -1)' * 1j * pi * cos(phi));

a = kron(ay, az);

