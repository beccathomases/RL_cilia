% compare_tabular_and_continuous_reward.m
%
% Sanity check:
% Compare the original tabular reward with the continuous-action reward
% on matching moves.
%
% Idea:
%   If we choose stepScale = params.dphi and use a continuous action equal
%   to a tabular action in {-1,0,1}^2, then the two rewards should match
%   (up to roundoff).

clear; clc;

% ------------------------------------------------------------
% parameters
% ------------------------------------------------------------

P = setdefaultparams_ciliaball;

% For this comparison, force the continuous control scale to match the
% tabular grid spacing exactly.
stepScale = P.dphi(:);

fprintf('Using stepScale = params.dphi for exact comparison.\n');
fprintf('params.dphi = [%g, %g]\n', P.dphi(1), P.dphi(2));

% ------------------------------------------------------------
% choose a test state
% ------------------------------------------------------------
%
% Pick any admissible hinge-angle state strictly inside the bounds.
% You can change this.

phi = [0.10; -0.40];

fprintf('Test state phi = [%g, %g]\n', phi(1), phi(2));

% ------------------------------------------------------------
% list of tabular actions to test
% ------------------------------------------------------------

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

nTests = size(actionList,1);

fprintf('\n');
fprintf('%10s %14s %14s %14s\n', ...
    'action', 'tabReward', 'contReward', 'absDiff');
fprintf('%10s %14s %14s %14s\n', ...
    '------', '---------', '----------', '-------');

for k = 1:nTests
    a_tab = actionList(k,:).';

    % Original tabular reward
    [r_tab, next_tab] = cilia_ball_reward(phi, a_tab, P);

    % Continuous reward with matching action and stepScale = dphi
    [r_cont, next_cont] = cilia_ball_reward_continuous(phi, a_tab, P, stepScale);

    absDiff = abs(r_tab - r_cont);

    fprintf('[%2d,%2d] %14.8f %14.8f %14.8e\n', ...
        a_tab(1), a_tab(2), r_tab, r_cont, absDiff);

    % Optional next-state check
    nextDiff = norm(next_tab(:) - next_cont(:));
    if nextDiff > 1e-12
        warning('Next states differ for action [%d,%d] by %g', ...
            a_tab(1), a_tab(2), nextDiff);
    end
end

fprintf('\nDone.\n');

% ============================================================
% continuous version of the tabular reward
% ============================================================

function [reward,next_state] = cilia_ball_reward_continuous(state, action, params, stepScale)
% Continuous-action version of cilia_ball_reward
%
% state     = current hinge angles
% action    = continuous control vector
% stepScale = physical angular increment corresponding to action = 1

   state     = state(:);
   action    = action(:);
   stepScale = stepScale(:);

   % midpoint state for 1-point Gaussian quadrature
   phi0 = state + 0.5 * stepScale .* action;

   % compute the ball positions
   X = position_from_angle(phi0, params);

   % continuous angular velocity corresponding to this move
   phidot = angvel_from_action_continuous(action, params, stepScale);

   % compute the velocities of the balls
   U = velocity_from_angvel(phi0, phidot, params);

   % form regularized Stokeslets matrix with images
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

% ============================================================
% continuous angular velocity helper
% ============================================================

function phidot = angvel_from_action_continuous(action, params, stepScale)
% Continuous analogue of the tabular helper
%
% Original tabular version:
%   phidot = action .* params.dphi / params.dt

    action    = action(:);
    stepScale = stepScale(:);

    phidot = action .* stepScale / params.dt;
end