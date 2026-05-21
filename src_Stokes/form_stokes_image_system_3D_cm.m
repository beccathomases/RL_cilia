%
%  Forms the regularized stokelets method of images matrix.
%  X:  The Nt x 3 array of target points (where the velocity is to be evaluated)
%  X0: The Ns x 3 array of source points (where the forces are located)
%  plane_vec = [a; b; c; d]: The surface is given by the plane ax + by + cz = d
%  This implementation uses column-major vectorization of forces and velocities
%
%  Five stokeslets and stokeslet-like objects contribute to the image system:
%   1) Primary stokeslet
%   2) Image stokeslet
%   3) Potential dipole
%   4) Stokeslet doublet
%   5) Rotlet(s)
%
function M = form_stokes_image_system_3D_cm(X,X0,epsilon,mu,plane_vec);

% Target and source points
Nt = size(X ,1);
Ns = size(X0,1);

% Normalize plane so that n has unit length
n = plane_vec(1:3);
d = plane_vec(4)/norm(n);
n = n/norm(n);

% Initialize stokeslet system matrix
M = zeros(3*Nt,3*Ns);

% (1) ------------------- Primary stokeslet matrix ----------------------------

% Displacement arrays for fast outer product calculations
X0m = X0(:, ones(1,Nt));
Xm  =  X(:, ones(1,Ns));

Y0m = X0(:,2*ones(1,Nt));
Ym  =  X(:,2*ones(1,Ns));

Z0m = X0(:,3*ones(1,Nt));
Zm  =  X(:,3*ones(1,Ns));

XX = (Xm-X0m').^2;
YY = (Ym-Y0m').^2;
ZZ = (Zm-Z0m').^2;

XY = (Xm-X0m').*(Ym-Y0m');
XZ = (Xm-X0m').*(Zm-Z0m');
YZ = (Ym-Y0m').*(Zm-Z0m');

% H1(r) and H2(r)
r  = sqrt(XX + YY + ZZ);
re = sqrt(r.^2 + epsilon.^2);
H2 = 1./re.^3;
H1 = (r.*r + 2*epsilon^2).*H2;

% Forms the primary stokeslet matrix
M_s = [[H1+H2.*XX,    H2.*XY,    H2.*XZ];
       [   H2.*XY, H1+H2.*YY,    H2.*YZ];
       [   H2.*XZ,    H2.*YZ, H1+H2.*ZZ];
      ];

M_s = M_s/(8*pi*mu);

% Adds the primary stokeslet matrix
M = M + M_s;

% (2) ------------------- Image stokeslet matrix -----------------------------

% Reflects source points across the plane
X0 = X0 - 2*(X0*n-d).*(n.');

% These calculations are recomputed
X0m = X0(:, ones(1,Nt));
Xm  =  X(:, ones(1,Ns));

Y0m = X0(:,2*ones(1,Nt));
Ym  =  X(:,2*ones(1,Ns));

Z0m = X0(:,3*ones(1,Nt));
Zm  =  X(:,3*ones(1,Ns));

XX = (Xm-X0m').^2;
YY = (Ym-Y0m').^2;
ZZ = (Zm-Z0m').^2;

XY = (Xm-X0m').*(Ym-Y0m');
XZ = (Xm-X0m').*(Zm-Z0m');
YZ = (Ym-Y0m').*(Zm-Z0m');

% H1(r) and H2(r) are recomputed as well
r  = sqrt(XX + YY + ZZ);
re = sqrt(r.^2 + epsilon.^2);
H2 = 1./re.^3;
H1 = (r.*r + 2*epsilon^2).*H2;

% Forms the image stokeslet matrix
M_s_im = [[H1+H2.*XX,    H2.*XY,    H2.*XZ];
          [   H2.*XY, H1+H2.*YY,    H2.*YZ];
          [   H2.*XZ,    H2.*YZ, H1+H2.*ZZ];
         ];

M_s_im = -M_s_im/(8*pi*mu);

% Adds the image stokeslet matrix
M = M + M_s_im;

% (3) ------------------- Potential dipole matrix -----------------------------

% Computes the distances from each source point to the plane
h = abs(X0*n - d);

% Householder operator for the potential dipole (PD) and stokeslet doublet (SD)
P = kron(eye(3) - 2*n*n', eye(Ns));

% Scaling operator for the PD, SD, and the rotlet
h = kron(eye(3), diag(h));

% D1(r) and D2(r)
D2 = 1./re.^5;
D1 = (r.*r - 2*epsilon^2).*D2;
D2 = -3.*D2;

% Forms the potential dipole matrix
%
M_pd = [[D1+D2.*XX,    D2.*XY,    D2.*XZ];
        [   D2.*XY, D1+D2.*YY,    D2.*YZ];
        [   D2.*XZ,    D2.*YZ, D1+D2.*ZZ];
       ];

% Right muiltiplies M_pd with Householder and h^2 scaling operators
M_pd = M_pd*(h.^2)*P/(4*pi*mu);

% Adds the potential dipole matrix
M = M + M_pd;

% (4) ------------------- Stokeslet doublet matrix ----------------------------

% Forms displacement vectors
Xhat = Xm-X0m';
Yhat = Ym-Y0m';
Zhat = Zm-Z0m';

% H3(r) = H1'(r)/r, H4 = H2'(r)/r = D2
H3 = -(r.*r + 4*epsilon^2)./re.^5;
H4 = D2;

% The SD matrix is constructed from four terms
% SD1
SD1 = [[H2.*Xhat.*n(1), H2.*Xhat.*n(2), H2.*Xhat.*n(3)];
       [H2.*Yhat.*n(1), H2.*Yhat.*n(2), H2.*Yhat.*n(3)];
       [H2.*Zhat.*n(1), H2.*Zhat.*n(2), H2.*Zhat.*n(3)];
      ];

% SD2
XdotN = Xhat.*n(1)+Yhat.*n(2)+Zhat.*n(3);
D = H2.*XdotN;
SD2 = blkdiag(D,D,D);

% SD3
SD3 = [[H3.*Xhat.*n(1), H3.*Yhat.*n(1), H3.*Zhat.*n(1)];
       [H3.*Xhat.*n(2), H3.*Yhat.*n(2), H3.*Zhat.*n(2)];
       [H3.*Xhat.*n(3), H3.*Yhat.*n(3), H3.*Zhat.*n(3)];
      ];

% SD4
SD4 = [[H4.*XdotN.*XX, H4.*XdotN.*XY, H4.*XdotN.*XZ];
       [H4.*XdotN.*XY, H4.*XdotN.*YY, H4.*XdotN.*YZ];
       [H4.*XdotN.*XZ, H4.*XdotN.*YZ, H4.*XdotN.*ZZ];
      ];

% Right muiltiplies M_sd with Householder and -2h scaling operators
M_sd = (SD1+SD2+SD3+SD4)*(-2*h*P)/(8*pi*mu);

% Adds the stokeslet doublet matrix
M = M + M_sd;

% (5) ------------------- Rotlet matrix ---------------------------------------

% H5 = H1'(r)/r + H2(r)
H5 = H3 + H2;

% The rotlet matrix is constructed from two terms
% R1
R1 = [[H5.*Xhat.*n(1), H5.*Yhat.*n(1), H5.*Zhat.*n(1)];
      [H5.*Xhat.*n(2), H5.*Yhat.*n(2), H5.*Zhat.*n(2)];
      [H5.*Xhat.*n(3), H5.*Yhat.*n(3), H5.*Zhat.*n(3)];
     ];

% R2
D = -H5.*XdotN;
R2 = blkdiag(D,D,D);

% Right muiltiplies M_rot with 2h scaling operators
M_rot = (R1+R2)*(2*h)/(8*pi*mu);

% Adds the rotlet matrix
M = M + M_rot;
