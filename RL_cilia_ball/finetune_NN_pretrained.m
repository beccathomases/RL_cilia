% 04/20/26
% Fine-tune a pretrained tiny NN with RL on the cilia-ball problem.
%
% The pretrained net should come from tabular cycle imitation, e.g.
%   pretrained_tabularCycleImitation_seeds1_3_4_g0.99_eps00.75_alp00.99_nEpisode50000_m32.mat
%
% Input to NN:
%   x = [phi1; phi2; dphi1; dphi2]
%
% This script:
%   - loads the pretrained NN once
%   - for each random seed, starts from that same pretrained NN
%   - fine-tunes with Q-learning-style TD updates
%   - saves the fine-tuned net and training returns

clear; clc; close all;

% ====================
% pretrained net to load
% ====================

%pretrainFile = 'pretrained_tabularCycleImitation_seeds1_to_10_g0.99_eps00.75_alp00.99_nEpisode50000_m32.mat';
pretrainFile = 'pretrained_tabularCycleImitation_seeds1_2_3_4_5_6_7_8_9_10_g0.99_eps00.75_alp00.99_nEpisode50000_m32.mat';
% Short label to include in saved filenames
pretrainTag = 'tabCycle1_10';

% ====================
% training environment
% ====================

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','random','precompute',true));

% ====================
% fine-tuning parameters
% ====================

maxSteps = 100;

nEpisodes = 10000;      % try 5000 or 10000 first
alpha0 = 0.005;         % smaller than before, to preserve pretrained behavior
alphafloor = 0.0005;

epsilon0 = 0.10;        % gentle exploration
epsilonfloor = 0.01;

gamma = 0.99;

stuckPenalty = 0.05;    % keep same penalty idea as before
decay = 0.999;

nseeds = 1:5;           % each seed starts from the same pretrained net

% ====================
% load pretrained net
% ====================

Spre = load(pretrainFile);

if ~isfield(Spre,'net')
    error('Pretraining file does not contain variable "net".');
end

net0 = Spre.net;

% infer dimensions from pretrained net
[m, d] = size(net0.W1);
nA = size(net0.W2, 1);

if d ~= 4
    error('Expected pretrained net to have input dimension d = 4.');
end
if size(net0.W2, 2) ~= m
    error('Mismatch between W1 and W2 dimensions in pretrained net.');
end
if nA ~= env.nActions
    error('Pretrained net action count does not match environment.');
end

fprintf('Loaded pretrained net from %s\n', pretrainFile);
fprintf('Network dims: d = %d, m = %d, nA = %d\n', d, m, nA);

% ====================
% fine-tune from pretrained net
% ====================

for seeds = nseeds
    rng(seeds);

    % Start each run from the SAME pretrained net
    net = net0;

    epReturn = zeros(nEpisodes,1);

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

            % ----- stuck penalty -----
            if s2 == s
                r = r - stuckPenalty;
            end

            % ----- next-state forward pass -----
            x2 = state_to_x_dphi(env, s, s2);
            [q2, ~] = nn_forward(net, x2);

            % ----- target -----
            target = r + gamma * max(q2);

            % ----- TD error -----
            err = target - q(a);

            % Save chosen row before update
            W2a_old = net.W2(a,:);

            % ----- output-layer update (chosen action only) -----
            net.W2(a,:) = net.W2(a,:) + alpha * err * h';
            net.b2(a)   = net.b2(a)   + alpha * err;

            % ----- hidden-layer backprop -----
            delta1 = (W2a_old' * err) .* h .* (1 - h);

            net.W1 = net.W1 + alpha * (delta1 * x');
            net.b1 = net.b1 + alpha * delta1;

            % ----- move on -----
            prev_s = s;
            s = s2;
            G = G + r;
        end

        epReturn(ep) = G;

        if mod(ep,1000) == 0
            fprintf('seed %d, episode %d / %d, return = %.6f, alpha = %.5f, epsilon = %.5f\n', ...
                seeds, ep, nEpisodes, G, alpha, epsilon);
        end
    end

    fout = sprintf(['run_tinyNN_dphi_finetune_%s_seed%d_g%1.3f_eps0%1.2f_' ...
                    'alp0%1.4f_nEpisode%d_m%d_sp%1.2f.mat'], ...
                    pretrainTag, seeds, gamma, epsilon0, alpha0, ...
                    nEpisodes, m, stuckPenalty);

    save(fout, 'net', 'epReturn', 'gamma', 'epsilon0', 'alpha0', ...
         'alphafloor', 'epsilonfloor', 'decay', 'nEpisodes', ...
         'maxSteps', 'm', 'd', 'stuckPenalty', 'seeds', ...
         'pretrainFile', 'pretrainTag');

    fprintf('Saved fine-tuned run to %s\n', fout);
end


% ====================
% helper functions
% ====================

function x = state_to_x_dphi(env, s_prev, s_curr)
% x = [phi1; phi2; dphi1; dphi2], normalized by phimax

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
    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));
    q  = net.W2 * h + net.b2;
end