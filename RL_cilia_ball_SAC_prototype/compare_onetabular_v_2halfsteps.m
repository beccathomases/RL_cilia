% compare_one_tabular_step_vs_two_halfsteps.m
%
% Check:
%   one full tabular step
% versus
%   two continuous half-steps

clear; clc;

P = setdefaultparams_ciliaball;

% Half-step scale
stepScale = 0.5 * P.dphi(:);

fprintf('Using half-step scale = 0.5 * params.dphi\n');
fprintf('params.dphi   = [%g, %g]\n', P.dphi(1), P.dphi(2));
fprintf('stepScale     = [%g, %g]\n', stepScale(1), stepScale(2));

% Pick a test state
phi = [0.10; -0.40];

% Try a few tabular actions
actionList = [
    -1  -1
    -1   0
    -1   1
     0  -1
     0   1
     1  -1
     1   0
     1   1
];

fprintf('\n');
fprintf('%10s %14s %14s %14s %14s\n', ...
    'action', 'tabReward', '2halfReward', 'rewardDiff', 'stateDiff');
fprintf('%10s %14s %14s %14s %14s\n', ...
    '------', '---------', '----------', '----------', '---------');

for k = 1:size(actionList,1)
    a = actionList(k,:).';
    
    % ------------------------------------
    % one full tabular step
    % ------------------------------------
    [r_tab, next_tab] = cilia_ball_reward(phi, a, P);
    next_tab = next_tab(:);

    % ------------------------------------
    % two continuous half-steps
    % ------------------------------------
    [r1, phi_half] = cilia_ball_reward_continuous(phi, a, P, stepScale);
    [r2, next_half2] = cilia_ball_reward_continuous(phi_half, a, P, stepScale);

    r_half_total = r1 + r2;
    next_half2 = next_half2(:);

    % ------------------------------------
    % compare
    % ------------------------------------
    rewardDiff = abs(r_tab - r_half_total);
    stateDiff  = norm(next_tab - next_half2);

    fprintf('[%2d,%2d] %14.8f %14.8f %14.8e %14.8e\n', ...
        a(1), a(2), r_tab, r_half_total, rewardDiff, stateDiff);
end

fprintf('\nInterpretation:\n');
fprintf('  - stateDiff should be tiny (ideally near machine precision)\n');
fprintf('  - rewardDiff may be small but not exactly zero\n');
fprintf('    because one full-step midpoint is not identical to two half-step midpoints\n');

% ============================================================
% continuous version of the tabular reward
% ============================================================

function [reward,next_state] = cilia_ball_reward_continuous(state, action, params, stepScale)

   state     = state(:);
   action    = action(:);
   stepScale = stepScale(:);

   % midpoint state for 1-point Gaussian quadrature
   phi0 = state + 0.5 * stepScale .* action;

   % compute the ball positions
   X = position_from_angle(phi0, params);

   % continuous angular velocity corresponding to this move
   phidot = action .* stepScale / params.dt;

   % compute velocities of the balls
   U = velocity_from_angvel(phi0, phidot, params);

   % regularized Stokeslets matrix with images
   plane_vec = [0,0,1,0]';
   M = form_stokes_image_system_3D_cm(X, X, params.epsilon, params.mu, plane_vec);

   % solve F = M\U
   F_vec = M \ U(:);
   F = reshape(F_vec, params.N, 3, 1);

   % flux reward
   reward = params.dt/(pi*params.mu) * dot(X(:,3), F(:,1));

   % next state
   next_state = state + stepScale .* action;

   % clip to admissible hinge range
   next_state = max(-params.phimax(:), min(params.phimax(:), next_state));
end