% train_SAC_cilia2ball.m
%
% Minimal SAC training skeleton for the continuous 2-ball cilia environment.
%
% Current status:
%   - replay buffer works
%   - actor/critic forward passes work
%   - critic update is implemented
%   - actor update is currently a placeholder
%
% So this script is mostly for wiring/testing right now.

clear; clc; close all;

% ------------------------------------------------------------
% environment
% ------------------------------------------------------------

P = setdefaultparams_ciliaball;

opts = struct();
opts.maxSteps  = 100;
opts.stepScale = 0.5 * P.dphi(:);   % first continuous-control step size
opts.resetMode = 'random';

env = make_cilia2ball_continuous_env(P, opts);

obsDim = env.obsDim;
actDim = env.actDim;

% ------------------------------------------------------------
% SAC hyperparameters
% ------------------------------------------------------------

hiddenSize = 32;

nEpisodes = 50;
maxSteps = env.maxSteps;

bufferSize = 50000;
batchSize = 64;

warmupSteps = 500;     % use random actions at the very start
updateAfter = 500;     % don't update until replay buffer has some data
updateEvery = 1;       % update every environment step once started

gamma = 0.99;
tau = 0.005;
alphaEntropy = 0.2;

actorLR  = 1e-3;
criticLR = 1e-3;

% ------------------------------------------------------------
% initialize networks and replay buffer
% ------------------------------------------------------------

actor = init_actor(obsDim, actDim, hiddenSize);

critic1 = init_critic(obsDim, actDim, hiddenSize);
critic2 = init_critic(obsDim, actDim, hiddenSize);

targetCritic1 = critic1;
targetCritic2 = critic2;

buffer = init_replay_buffer(bufferSize, obsDim, actDim);

% ------------------------------------------------------------
% training loop
% ------------------------------------------------------------

globalStep = 0;
epReturnHist = zeros(nEpisodes,1);

for ep = 1:nEpisodes
    [obs, internal] = reset_env(env);
    epReturn = 0;

    for t = 1:maxSteps
        globalStep = globalStep + 1;

        % ----------------------------------------
        % action selection
        % ----------------------------------------
        if globalStep <= warmupSteps
            action = 2*rand(actDim,1) - 1;   % random action in [-1,1]^2
        else
            [action, ~, ~] = sample_action_from_actor(actor, obs, false);
        end

        % ----------------------------------------
        % environment step
        % ----------------------------------------
        [nextObs, reward, done, info, internal] = step_env(env, internal, action); %#ok<ASGLU>

        % ----------------------------------------
        % store transition
        % ----------------------------------------
        buffer = add_transition(buffer, obs, action, reward, nextObs, done);

        obs = nextObs;
        epReturn = epReturn + reward;

        % ----------------------------------------
        % learning updates
        % ----------------------------------------
        if buffer.count >= batchSize && globalStep >= updateAfter && mod(globalStep, updateEvery) == 0

            batch = sample_minibatch(buffer, batchSize);

            [critic1, critic2, criticStats] = update_critics( ...
                critic1, critic2, targetCritic1, targetCritic2, actor, ...
                batch, gamma, alphaEntropy, criticLR);

            [actor, actorStats] = update_actor( ...
                actor, critic1, critic2, batch, alphaEntropy, actorLR); %#ok<NASGU>

            [targetCritic1, targetCritic2] = soft_update_targets( ...
                targetCritic1, targetCritic2, critic1, critic2, tau);

            if mod(globalStep, 100) == 0
                fprintf(['step %d | critic losses = %.4e, %.4e | ' ...
                         'actor loss = %.4e | buffer = %d\n'], ...
                        globalStep, criticStats.loss1, criticStats.loss2, ...
                        actorStats.loss, buffer.count);
            end
        end

        if done
            break
        end
    end

    epReturnHist(ep) = epReturn;
    fprintf('Episode %d / %d, return = %.6f\n', ep, nEpisodes, epReturn);
end

% ------------------------------------------------------------
% simple diagnostic plot
% ------------------------------------------------------------

figure;
plot(epReturnHist, 'LineWidth', 1.5);
xlabel('Episode');
ylabel('Return');
title('SAC training returns (actor update still placeholder)');
grid on;


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