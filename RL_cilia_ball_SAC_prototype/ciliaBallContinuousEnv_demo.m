% ciliaBallContinuousEnv_demo.m
%
% MATLAB-style environment skeleton for a continuous-control version
% of the 2-ball cilia problem.
%
% This is meant as a starting point for trying actor-critic methods
% such as SAC. It does NOT implement SAC itself.

clear; clc;

% ------------------------------------------------------------
% parameters
% ------------------------------------------------------------

P = setdefaultparams_ciliaball;

opts = struct();
opts.maxSteps = 100;                 % episode length
opts.stepScale = 0.5 * P.dphi(:);    % continuous control step size
opts.resetMode = 'random';           % 'fixed' or 'random'

% Build continuous environment
env = make_cilia2ball_continuous_env(P, opts);

% Example usage:
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
            % You can change this if you want a different default start
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

    % keep track of reward-related quantities if useful
    internal.prevMetric = compute_metric(internal.phi, P);

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
    % 2. continuous hinge update
    % ----------------------------------------
    phi_old = internal.phi;

    phi_new = phi_old + env.stepScale .* action;

    % clip to admissible hinge range
    phi_new = max(-P.phimax(:), min(P.phimax(:), phi_new));

    % ----------------------------------------
    % 3. compute reward
    % ----------------------------------------
    %
    % Replace this with the reward you trust from the tabular experiments.
    % Right now this uses a placeholder "metric gain" reward.
    %
    oldMetric = internal.prevMetric;
    newMetric = compute_metric(phi_new, P);

    reward = newMetric - oldMetric;

    % optional stuck penalty
    if norm(phi_new - phi_old) < 1e-10
        reward = reward - 0.01;
    end

    % ----------------------------------------
    % 4. update internal state
    % ----------------------------------------
    internal.t = internal.t + 1;
    internal.phi_prev = phi_old;
    internal.phi      = phi_new;
    internal.prevMetric = newMetric;

    % ----------------------------------------
    % 5. build next observation
    % ----------------------------------------
    nextObs = get_state_from_internal(internal);

    % ----------------------------------------
    % 6. done flag
    % ----------------------------------------
    done = internal.t >= env.maxSteps;

    % ----------------------------------------
    % 7. optional info
    % ----------------------------------------
    info = struct();
    info.metric = newMetric;
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
% reward metric placeholder
% ============================================================

function metric = compute_metric(phi, P)
%
% This is a PLACEHOLDER.
%
% Replace this with whatever scalar quantity your tabular reward was based on.
% For example, if your task was side sweep / span, you could compute the
% current body configuration from phi and then measure that quantity here.
%
% For now, I am just using the x-span of the 2-ball configuration as an example.

    X = position_from_angle(phi, P);   % assumes your existing helper exists
    XX = [P.X0; X];

    xcoords = XX(:,1);
    metric = max(xcoords) - min(xcoords);
end