%
% position_from_angle -- given the bending angels at each joint construct the positions of
%                        balls at the joints and the tip
%
% Segements are in the xz-plane (y=0)
% phi=0 is straight up, i.e. positive z-direction
% positive angle direction is clockwise; i.e. x-direction is pi/2
%
% phi is the bending angle of the at the joint at the base of segment indexed k
%
function X=position_from_angle(phi,params);

X = zeros(params.N,3);

% psi is the angle of segment
%
psi = phi(1);
X(1,:) = params.X0 + params.L(1).*[sin(psi),0,cos(psi)];

for k = 2:params.N
     psi = psi + phi(k);
     X(k,:) = X(k-1,:) + params.L(k).*[sin(psi),0,cos(psi)];
end

