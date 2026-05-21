function P = setdefaultparams_ciliaball
% set default parameters
P.N = 2;                  % number of BALLS
P.L = [0.5,0.5];          % legnth of segment connecting balls
P.mu = 1;                 % fluid viscosity
P.X0 = [0 0 0];           % origin in space
P.phimax = [pi/4, pi/2];  % max of abs of angle at each hinge
P.Nstates = [11,21];      % number of discrete states to divide span of [-phimax,phimax] into 
P.a = 0.05;               % theoretical BALL radius 
P.dt = 1;                 % time step

% parameters computed from inputs
%
P.epsilon = 3/2*P.a;                % regularization parameter 
P.dphi = 2*P.phimax./(P.Nstates-1); % angle spacing between states

end 
