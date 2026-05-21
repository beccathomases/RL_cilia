% ciliaBallContinuousEnv_fluxReward_demo.m
%
% MATLAB-style environment skeleton for a continuous-control version
% of the 2-ball cilia problem, using the same flux reward idea as the
% tabular code.
%
% This is NOT a full SAC implementation.
% It is just the environment / reward side.

clear; clc;

% ------------------------------------------------------------
% parameters
% ------------------------------------------------------------

P = setdefaultparams_ciliaball;

opts = struct();
opts.maxSteps  = 100;                 % episode length
opts.stepScale = 0.5 * P.dphi(:);     % physical angular increment for action = 1
opts.resetMode = 'random';            % 'fixed' or 'random'

% Build continuous environment
env = make_cilia2ball_continuous_env(P, opts);

% Example usage
[obs, internal] = reset_env(env);

fprintf('Initial obs = [%g %g %g %g]\n', obs);

for t = 1:10
    % Example random continuous action in [-1,1]^2
    action = 2*rand(2,1) - 1;

    [nextObs, reward, done, info, internal] = step_env(env, internal, action);

    fprintf('t=%d, action=[%.3f %.3f], reward=%.6f, phi=[%.3f %.3f]\n', ...
        t, action(1), action(2), reward, internal.phi(1), internal.phi(2));

    if done
        break
    end
end

% ============================================================
% environment constructor
% ============================================================

function env = make_cilia2ball_continuous_env(P, opts)

    env = struct();

    env.P = P;
    env.maxSteps = opts.maxSteps;
    env.stepScale = opts.stepScale(:);     % 2x1 vector
    env.resetMode = opts.resetMode;

    % dimensions
    env.obsDim = 4;       % [phi1; phi2; dphi1; dphi2]
    env.actDim = 2;       % continuous [u1; u2]

    % action bounds for the actor
    env.actionLow  = -ones(2,1);
    env.actionHigh =  ones(2,1);

    % observation helper
    env.get_state = @get_state_from_internal;
end

% ============================================================
% reset
% ============================================================

function [obs, internal] = reset_env(env)

    P = env.P;

    switch lower(env.resetMode)
        case 'fixed'
            % Change this if you want a different default start
            phi0 = [0; -P.phimax(2)];

        case 'random'
            phi0 = -P.phimax(:) + 2*P.phimax(:).*rand(2,1);

        otherwise
            error('Unknown reset mode.');
    end

    internal = struct();
    internal.t = 0;

    % previous and current angles
    internal.phi_prev = phi0;
    internal.phi      = phi0;

    obs = get_state_from_internal(internal);
end

% ============================================================
% step
% ============================================================

function [nextObs, reward, done, info, internal] = step_env(env, internal, action)

    P = env.P;

    % ----------------------------------------
    % 1. clip action into valid range [-1,1]^2
    % ----------------------------------------
    action = max(env.actionLow, min(env.actionHigh, action(:)));

    % ----------------------------------------
    % 2. compute reward and next state
    % ----------------------------------------
    [reward, phi_new] = cilia_ball_reward_continuous(internal.phi, action, P, env.stepScale);

    % optional stuck penalty
    if norm(phi_new - internal.phi) < 1e-10
        reward = reward - 0.01;
    end

    % ----------------------------------------
    % 3. update internal state
    % ----------------------------------------
    phi_old = internal.phi;

    internal.t = internal.t + 1;
    internal.phi_prev = phi_old;
    internal.phi      = phi_new;

    % ----------------------------------------
    % 4. build next observation
    % ----------------------------------------
    nextObs = get_state_from_internal(internal);

    % ----------------------------------------
    % 5. done flag
    % ----------------------------------------
    done = internal.t >= env.maxSteps;

    % ----------------------------------------
    % 6. optional info
    % ----------------------------------------
    info = struct();
    info.phi = phi_new;
    info.action = action;
end

% ============================================================
% state function
% ============================================================

function obs = get_state_from_internal(internal)

    phi  = internal.phi(:);
    dphi = internal.phi(:) - internal.phi_prev(:);

    obs = [phi; dphi];
end

% ============================================================
% continuous version of the tabular reward
% ============================================================

function [reward,next_state] = cilia_ball_reward_continuous(state, action, params, stepScale)
% Continuous-action version of cilia_ball_reward
%
% state     = current hinge angles
% action    = continuous control in [-1,1]^n
% stepScale = physical angular increment corresponding to action = 1
%
% returns:
%   reward     = flux in x1-direction
%   next_state = updated hinge angles after the action

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
   plane_vec = [0,0,1,0]';  % z=0 plane
   M = form_stokes_image_system_3D_cm(X, X, params.epsilon, params.mu, plane_vec);

   % solve F = M\U
   F_vec = M \ U(:);
   F = reshape(F_vec, params.N, 3, 1);

   % calculate flux reward (same form as tabular version)
   reward = params.dt/(pi*params.mu) * dot(X(:,3), F(:,1));

   % next state after taking the action
   next_state = state + stepScale .* action;

   % clip to admissible hinge range
   next_state = max(-params.phimax(:), min(params.phimax(:), next_state));
end

% ============================================================
% continuous angular velocity helper
% ============================================================

function phidot = angvel_from_action_continuous(action, params, stepScale)
% Continuous analogue of:
%   phidot = action .* params.dphi / params.dt

    action    = action(:);
    stepScale = stepScale(:);

    phidot = action .* stepScale / params.dt;
end