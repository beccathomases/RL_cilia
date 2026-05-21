%
% cilia_ball_reward
%  state is an n-tuple of joint angles
%  action = n-nuples of actions, which are {-1,0,1} indicating the increment to the angle 
%
%  returns:
%    reward -- flux in the x1-direction
%    next_state -- state after taking the action
function [reward,next_state] = cilia_ball_reward(state,action,params)

   state  = state(:);
   action = action(:);
   dphi   = params.dphi(:);

   % compute the state in between the initial and final state
   phi0 = state + 0.5*dphi.*action;

   % compute the ball positions
   X = position_from_angle(phi0,params);

   % compute the angular velocity from the actions
   phidot = action .* dphi / params.dt;

   % compute the velocities of the balls
   U = velocity_from_angvel(phi0,phidot,params);

   % form regularized stokeslets matrix with images
   plane_vec = [0,0,1,0]';
   M = form_stokes_image_system_3D_cm(X, X, params.epsilon, params.mu, plane_vec);

   % solve F = M\U
   F_vec = M\U(:);
   F = reshape(F_vec,params.N,3,1);

   % calculate flux
   reward = params.dt/(pi*params.mu) * dot(X(:,3),F(:,1));

   % compute next state
   next_state = state + dphi.*action;
end