% 04/13/26
% Tiny-NN Q-learning version of the cilia ball experiment script.
% Trains one small neural network per random seed and saves results.
% Cycle detection can be run afterward on the trained net.

clear; clc; close all;

P = setdefaultparams_ciliaball;

% Precompute is nice for speed once physics is "frozen"
env = ciliaBallTabularEnv(P, struct('reset_mode','random','precompute',true));

maxSteps = 100;   % fixed for these experiments
% Vary nEpisodes if desired
nEpisodes = 10000;   % try 5000, 25000, 50000

% Vary alpha0 if desired
alpha0 = 0.01;       % for NN, usually much smaller than tabular
% try 0.05, 0.01, 0.005, 0.001
alphafloor = 0.001;  % do not change unless needed

% Vary epsilon0 if desired
epsilon0 = 0.3;      % try 1, 0.2, 0.1
epsilonfloor = 0.02;  % do not change unless needed

% Vary gamma if desired
gamma = 0.99;        % try 0.999, 0.98, 0.97

% Tiny hidden layer size
m = 32;               % try 4, 8, 16, 32

% Decay factor for alpha and epsilon
decay = 0.999;

for seeds = 1:15
    rng(seeds);

    % --- initialize tiny network ---
    d  = 2;              % two hinge angles as inputs
    nA = env.nActions;   % one output per action

    net.W1 = 0.1 * randn(m, d);
    net.b1 = zeros(m, 1);
    net.W2 = 0.1 * randn(nA, m);
    net.b2 = zeros(nA, 1);

    epReturn = zeros(nEpisodes, 1);

    for ep = 1:nEpisodes
        s = env.reset();
        G = 0;

        epsilon = max(epsilonfloor, epsilon0 * decay^(ep-1));
        alpha   = max(alphafloor,  alpha0   * decay^(ep-1));

        for t = 1:maxSteps
            % ----- current state -> NN input -----
            x = state_to_x(env, s);

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

            % ----- next-state forward pass -----
            x2 = state_to_x(env, s2);
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
            s = s2;
            G = G + r;
        end

        epReturn(ep) = G;
    end

    fname = sprintf(['run_tinyNN_seed%d_g%1.3f_eps0%1.2f_' ...
                     'alp0%1.3f_nEpisode%d_m%d.mat'], ...
                     seeds, gamma, epsilon0, alpha0, nEpisodes, m);

    save(fname, 'net', 'epReturn', 'gamma', 'epsilon0', 'alpha0', ...
         'nEpisodes', 'maxSteps', 'm', 'seeds', 'alphafloor', ...
         'epsilonfloor', 'decay');

    fprintf('Saved %s\n', fname);
end


function x = state_to_x(env, s)
% Convert tabular state index -> angle pair -> normalized column vector
    sub = env.state2sub(s);
    phi = env.sub2phi(sub);
    x   = phi(:) ./ env.P.phimax(:);
end


function [q, h] = nn_forward(net, x)
% One forward pass through the tiny network
    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));   % sigmoid hidden layer
    q  = net.W2 * h + net.b2;   % linear outputs = Q-values
end