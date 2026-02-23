%
% velocity_from_angvel -- given joint angles (phi) and their angular velocities (phidot)
%                         compute the velocity at the ball locations
%
% phi is the bending angle, and psi is the angle of segment
%
function U=velocity_from_angvel(phi,phidot,params)
      
U = zeros(params.N,3);
psi = phi(1);
psidot = phidot(1);
U(1,:) = params.L(1) * psidot .* [cos(psi), 0, -sin(psi)];

for k = 2:params.N
  psi = psi + phi(k);
  psidot = psidot + phidot(k);    
  U(k,:) = U(k-1,:) + params.L(k) * psidot .* [cos(psi), 0, -sin(psi)];
end

