% Compute Rotation Matrix and Translation matrix
syms r1 r2 r3 t1 t2 t3 theta
skew = [0, -r3, r2; r3, 0, -r1; -r2, r1, 0]
R = eye(3) + sin(theta) .* skew + (1-cos(theta)) * skew * skew

M = sym('M', [3, 1])
CameraM = R * M + [t1; t2; t3]
A = sym('A%d%d', [3 3])
HomogenousPixel = A * CameraM
NonHomogeneousPixel = [HomogenousPixel(1) / HomogenousPixel(3); HomogenousPixel(2) / HomogenousPixel(3)]

jacobian(NonHomogeneousPixel, [r1; r2; r3; t1; t2; t3; theta])

% [ ((A32*(M3*sin(theta) + M1*r2*(cos(theta) - 1) - 2*M2*r1*(cos(theta) -
% 1)) - A33*(M2*sin(theta) - M1*r3*(cos(theta) - 1) + 2*M3*r1*(cos(theta) -
% 1)) + A31*(M2*r2*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))*(A11*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A12*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A13*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1)))^2 -
% (A12*(M3*sin(theta) + M1*r2*(cos(theta) - 1) - 2*M2*r1*(cos(theta) - 1))
% - A13*(M2*sin(theta) - M1*r3*(cos(theta) - 1) + 2*M3*r1*(cos(theta) - 1))
% + A11*(M2*r2*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))),
% ((A33*(M1*sin(theta) + M2*r3*(cos(theta) - 1) - 2*M3*r2*(cos(theta) - 1))
% - A31*(M3*sin(theta) + 2*M1*r2*(cos(theta) - 1) - M2*r1*(cos(theta) - 1))
% + A32*(M1*r1*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))*(A11*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A12*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A13*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1)))^2 -
% (A13*(M1*sin(theta) + M2*r3*(cos(theta) - 1) - 2*M3*r2*(cos(theta) - 1))
% - A11*(M3*sin(theta) + 2*M1*r2*(cos(theta) - 1) - M2*r1*(cos(theta) - 1))
% + A12*(M1*r1*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))),
% ((A31*(M2*sin(theta) - 2*M1*r3*(cos(theta) - 1) + M3*r1*(cos(theta) - 1))
% - A32*(M1*sin(theta) + 2*M2*r3*(cos(theta) - 1) - M3*r2*(cos(theta) - 1))
% + A33*(M1*r1*(cos(theta) - 1) + M2*r2*(cos(theta) - 1)))*(A11*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A12*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A13*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1)))^2 -
% (A11*(M2*sin(theta) - 2*M1*r3*(cos(theta) - 1) + M3*r1*(cos(theta) - 1))
% - A12*(M1*sin(theta) + 2*M2*r3*(cos(theta) - 1) - M3*r2*(cos(theta) - 1))
% + A13*(M1*r1*(cos(theta) - 1) + M2*r2*(cos(theta) - 1)))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))), (A11 + A12 +
% A13)/(A31*(t1 - M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) +
% M3*(r2*sin(theta) - r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) -
% r1*r2*(cos(theta) - 1)) - M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) +
% M2*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 -
% M1*(r2*sin(theta) + r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) -
% r2*r3*(cos(theta) - 1)) + M3*((cos(theta) - 1)*r1^2 + (cos(theta) -
% 1)*r2^2 + 1))) - ((A31 + A32 + A33)*(A11*(t1 - M2*(r3*sin(theta) +
% r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) - r1*r3*(cos(theta) - 1)) +
% M1*((cos(theta) - 1)*r2^2 + (cos(theta) - 1)*r3^2 + 1)) + A12*(t1 +
% M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) - M3*(r1*sin(theta) +
% r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A13*(t1 - M1*(r2*sin(theta) + r1*r3*(cos(theta) - 1)) +
% M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) + M3*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 - M2*(r3*sin(theta) +
% r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) - r1*r3*(cos(theta) - 1)) +
% M1*((cos(theta) - 1)*r2^2 + (cos(theta) - 1)*r3^2 + 1)) + A32*(t1 +
% M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) - M3*(r1*sin(theta) +
% r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) + r1*r3*(cos(theta) - 1)) +
% M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) + M3*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r2^2 + 1)))^2, 0, 0]


% [ ((A32*(M3*sin(theta) + M1*r2*(cos(theta) - 1) - 2*M2*r1*(cos(theta) -
% 1)) - A33*(M2*sin(theta) - M1*r3*(cos(theta) - 1) + 2*M3*r1*(cos(theta) -
% 1)) + A31*(M2*r2*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))*(A21*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A22*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A23*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1)))^2 -
% (A22*(M3*sin(theta) + M1*r2*(cos(theta) - 1) - 2*M2*r1*(cos(theta) - 1))
% - A23*(M2*sin(theta) - M1*r3*(cos(theta) - 1) + 2*M3*r1*(cos(theta) - 1))
% + A21*(M2*r2*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))),
% ((A33*(M1*sin(theta) + M2*r3*(cos(theta) - 1) - 2*M3*r2*(cos(theta) - 1))
% - A31*(M3*sin(theta) + 2*M1*r2*(cos(theta) - 1) - M2*r1*(cos(theta) - 1))
% + A32*(M1*r1*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))*(A21*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A22*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A23*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1)))^2 -
% (A23*(M1*sin(theta) + M2*r3*(cos(theta) - 1) - 2*M3*r2*(cos(theta) - 1))
% - A21*(M3*sin(theta) + 2*M1*r2*(cos(theta) - 1) - M2*r1*(cos(theta) - 1))
% + A22*(M1*r1*(cos(theta) - 1) + M3*r3*(cos(theta) - 1)))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))),
% ((A31*(M2*sin(theta) - 2*M1*r3*(cos(theta) - 1) + M3*r1*(cos(theta) - 1))
% - A32*(M1*sin(theta) + 2*M2*r3*(cos(theta) - 1) - M3*r2*(cos(theta) - 1))
% + A33*(M1*r1*(cos(theta) - 1) + M2*r2*(cos(theta) - 1)))*(A21*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A22*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A23*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1)))^2 -
% (A21*(M2*sin(theta) - 2*M1*r3*(cos(theta) - 1) + M3*r1*(cos(theta) - 1))
% - A22*(M1*sin(theta) + 2*M2*r3*(cos(theta) - 1) - M3*r2*(cos(theta) - 1))
% + A23*(M1*r1*(cos(theta) - 1) + M2*r2*(cos(theta) - 1)))/(A31*(t1 -
% M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) -
% r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) -
% M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) +
% r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) +
% M3*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r2^2 + 1))), (A21 + A22 +
% A23)/(A31*(t1 - M2*(r3*sin(theta) + r1*r2*(cos(theta) - 1)) +
% M3*(r2*sin(theta) - r1*r3*(cos(theta) - 1)) + M1*((cos(theta) - 1)*r2^2 +
% (cos(theta) - 1)*r3^2 + 1)) + A32*(t1 + M1*(r3*sin(theta) -
% r1*r2*(cos(theta) - 1)) - M3*(r1*sin(theta) + r2*r3*(cos(theta) - 1)) +
% M2*((cos(theta) - 1)*r1^2 + (cos(theta) - 1)*r3^2 + 1)) + A33*(t1 -
% M1*(r2*sin(theta) + r1*r3*(cos(theta) - 1)) + M2*(r1*sin(theta) -
% r2*r3*(cos(theta) - 1)) + M3*((cos(theta) - 1)*r1^2 + (cos(theta) -
% 1)*r2^2 + 1))) - ((A31 + A32 + A33)*(A21*(t1 - M2*(r3*sin(theta) +
% r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) - r1*r3*(cos(theta) - 1)) +
% M1*((cos(theta) - 1)*r2^2 + (cos(theta) - 1)*r3^2 + 1)) + A22*(t1 +
% M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) - M3*(r1*sin(theta) +
% r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A23*(t1 - M1*(r2*sin(theta) + r1*r3*(cos(theta) - 1)) +
% M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) + M3*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r2^2 + 1))))/(A31*(t1 - M2*(r3*sin(theta) +
% r1*r2*(cos(theta) - 1)) + M3*(r2*sin(theta) - r1*r3*(cos(theta) - 1)) +
% M1*((cos(theta) - 1)*r2^2 + (cos(theta) - 1)*r3^2 + 1)) + A32*(t1 +
% M1*(r3*sin(theta) - r1*r2*(cos(theta) - 1)) - M3*(r1*sin(theta) +
% r2*r3*(cos(theta) - 1)) + M2*((cos(theta) - 1)*r1^2 + (cos(theta) -
% 1)*r3^2 + 1)) + A33*(t1 - M1*(r2*sin(theta) + r1*r3*(cos(theta) - 1)) +
% M2*(r1*sin(theta) - r2*r3*(cos(theta) - 1)) + M3*((cos(theta) - 1)*r1^2 +
% (cos(theta) - 1)*r2^2 + 1)))^2, 0, 0]
