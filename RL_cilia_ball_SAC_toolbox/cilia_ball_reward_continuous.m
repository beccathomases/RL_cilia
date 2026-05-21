function [reward,next_state] = cilia_ball_reward_continuous(state, action, params, stepScale)
% Continuous-action version of the tabular flux reward.

   state     = state(:);
   action    = action(:);
   stepScale = stepScale(:);

   % midpoint state for 1-point Gaussian quadrature
   phi0 = state + 0.5 * stepScale .* action;

   % ball positions
   X = position_from_angle(phi0, params);

   % angular velocity from continuous action
   phidot = action .* stepScale / params.dt;

   % ball velocities
   U = velocity_from_angvel(phi0, phidot, params);

   % regularized Stokeslets matrix with images
   plane_vec = [0,0,1,0]';
   M = form_stokes_image_system_3D_cm(X, X, params.epsilon, params.mu, plane_vec);

   % solve for forces
   F_vec = M \ U(:);
   F = reshape(F_vec, params.N, 3, 1);

   % flux reward
   reward = params.dt/(pi*params.mu) * dot(X(:,3), F(:,1));

   % next state
   next_state = state + stepScale .* action;
   next_state = max(-params.phimax(:), min(params.phimax(:), next_state));
end