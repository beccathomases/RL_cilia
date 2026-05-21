% 04/20/26
% Tiny-NN Q-learning version of the cilia ball experiment script.
% Trains one small neural network per random seed and saves results.
% Input includes direction information via delta-phi:
%    x = [phi1; phi2; dphi1; dphi2]
% Also includes a small penalty if the state gets stuck (s2 == s).

clear; clc; close all;

P = setdefaultparams_ciliaball;

% Random reset for training
envOpts = struct();
envOpts.reset_mode = 'random';
envOpts.precompute = true;
envOpts.boundary = 'clip_penalty';
envOpts.invalid_penalty = -0.1;

env = ciliaBallTabularEnv(P, envOpts);

% ====================
% suggested defaults
% ====================

nEpisodes    = 25000;
alpha0       = 0.01;
epsilon0     = 0.3;
gamma        = 0.99;
m            = 16;
stuckPenalty = 0.0; % part of environment now
nseeds       = 1:5;

% keep these fixed 
alphafloor   = 0.001;
epsilonfloor = 0.02;
decay        = 0.999;
maxSteps = 100;   


for seeds = nseeds
    rng(seeds);

    % --- initialize tiny network ---
    d  = 4;              % [phi1; phi2; dphi1; dphi2]
    nA = env.nActions;   % one output per action

    net.W1 = 0.1 * randn(m, d);
    net.b1 = zeros(m, 1);
    net.W2 = 0.1 * randn(nA, m);
    net.b2 = zeros(nA, 1);

    epReturn = zeros(nEpisodes, 1);

    for ep = 1:nEpisodes
        s = env.reset();
        prev_s = s;   % so first-step dphi = 0
        G = 0;

        epsilon = max(epsilonfloor, epsilon0 * decay^(ep-1));
        alpha   = max(alphafloor,  alpha0   * decay^(ep-1));

        for t = 1:maxSteps
            % ----- current state -> NN input -----
            x = state_to_x_dphi(env, prev_s, s);

            % ----- forward pass -----
            [q, h] = nn_forward(net, x);

            % ----- epsilon-greedy -----
            if rand < epsilon
                a = randi(env.nActions);
            else
                [~, a] = max(q);
            end

            % ----- environment step -----
            [s2, r] = env.step(s, a);

            % ----- small penalty for getting stuck -----
            %if s2 == s
            %    r = r - stuckPenalty;
            %end

            % ----- next-state forward pass -----
            x2 = state_to_x_dphi(env, s, s2);
            [q2, ~] = nn_forward(net, x2);

            % ----- Q-learning target -----
            target = r + gamma * max(q2);

            % ----- TD error for chosen action -----
            err = target - q(a);

            % Save chosen action row before output-layer update
            W2a_old = net.W2(a,:);

            % ----- output-layer update (chosen action only) -----
            net.W2(a,:) = net.W2(a,:) + alpha * err * h';
            net.b2(a)   = net.b2(a)   + alpha * err;

            % ----- hidden-layer backprop -----
            delta1 = (W2a_old' * err) .* h .* (1 - h);   % sigmoid derivative

            net.W1 = net.W1 + alpha * (delta1 * x');
            net.b1 = net.b1 + alpha * delta1;

            % ----- move on -----
            prev_s = s;
            s = s2;
            G = G + r;
        end

        epReturn(ep) = G;
    end

    fname = sprintf(['run_tinyNN_dphi_%s_pen%0.2f_seed%d_g%1.3f_eps0%1.2f_' ...
                     'alp0%1.3f_nEpisode%d_m%d.mat'], ...
                     envOpts.boundary, envOpts.invalid_penalty, ...
                     seeds, gamma, epsilon0, alpha0, nEpisodes, m);

    save(fname, 'net', 'epReturn', 'gamma', 'epsilon0', 'alpha0', ...
         'nEpisodes', 'maxSteps', 'm', 'seeds', 'alphafloor', ...
         'epsilonfloor', 'decay', 'stuckPenalty', 'envOpts');
    fprintf('Saved %s\n', fname);
end


function x = state_to_x_dphi(env, s_prev, s_curr)
% Convert previous/current states into normalized input:
%   x = [phi1; phi2; dphi1; dphi2]

    sub_prev = env.state2sub(s_prev);
    phi_prev = env.sub2phi(sub_prev);

    sub_curr = env.state2sub(s_curr);
    phi_curr = env.sub2phi(sub_curr);

    phimax = env.P.phimax(:);

    phi_norm  = phi_curr(:) ./ phimax;
    dphi_norm = (phi_curr(:) - phi_prev(:)) ./ phimax;

    x = [phi_norm; dphi_norm];
end


function [q, h] = nn_forward(net, x)
% One forward pass through the tiny network
    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));   % sigmoid hidden layer
    q  = net.W2 * h + net.b2;   % linear outputs = Q-values
end