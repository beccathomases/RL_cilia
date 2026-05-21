function [reward,next_state] = cilia_ball_reward_continuous(state, action, params)

   state  = state(:);
   action = action(:);

   % midpoint state for 1-point quadrature
   phi0 = state + 0.5 * params.stepScale(:) .* action;

   % compute ball positions
   X = position_from_angle(phi0, params);

   % angular velocity from continuous action
   phidot = angvel_from_action_continuous(action, params);

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
   next_state = state + params.stepScale(:) .* action;

   % clip to hinge bounds
   next_state = max(-params.phimax(:), min(params.phimax(:), next_state));
end