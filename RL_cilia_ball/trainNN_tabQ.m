% 04/20/26
% Pretrain tiny NN from a few GOOD TABULAR saved cycle files.
%
% Input:
%   x = [phi1; phi2; dphi1; dphi2]
%
% Target:
%   y = preference-style action target:
%       -1 for all actions, +1 for the demonstrated action
%
% This uses TABULAR cycle files like:
%   cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat

clear; clc; close all;

% ====================
% parameters matching TABULAR saved cycle files
% ====================

nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

% Put your good TABULAR seeds here
goodSeeds = 1:10;    % <-- change to your actual good tabular seeds

% How heavily to oversample each cycle
cycleWeight = 10;       % try 5, 10, 20

% NN architecture / supervised training
m = 32;                 % NN hidden size
d = 4;                  % [phi1; phi2; dphi1; dphi2]
nEpochs = 4000;         % try 2000, 4000, 8000
eta = 0.01;             % supervised learning rate

rng(1);

% ====================
% environment
% ====================

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));

nA = env.nActions;

% ====================
% build demonstration dataset from TABULAR cycles
% ====================

X = [];
Y = [];

for seed = goodSeeds
    fin = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
        seed, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

    if ~isfile(fin)
        fprintf('Could not find %s -- skipping.\n', fin);
        continue
    end

    S = load(fin);

    if ~isfield(S,'cycle_states')
        fprintf('File %s does not contain cycle_states -- skipping.\n', fin);
        continue
    end

    cycle_states = S.cycle_states(:)';

    % If stored as [s1 ... sK s1], remove repeated closing state
    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    K = numel(cycle_core);

    if K < 2
        fprintf('Cycle in %s is too short -- skipping.\n', fin);
        continue
    end

    fprintf('Using tabular seed %d with cycle length %d\n', seed, K);

    % Build one training example per phase in cycle
    % prev -> curr gives dphi
    % curr -> next gives demonstrated action
    for rep = 1:cycleWeight
        for j = 1:K
            if j == 1
                s_prev = cycle_core(K);   % wrap around
            else
                s_prev = cycle_core(j-1);
            end

            s_curr = cycle_core(j);

            if j == K
                s_next = cycle_core(1);   % wrap around
            else
                s_next = cycle_core(j+1);
            end

            x = state_to_x_dphi(env, s_prev, s_curr);
            a = infer_action_index(env, s_curr, s_next);

            y = -ones(nA,1);
            y(a) = 1;

            X(:,end+1) = x;
            Y(:,end+1) = y;
        end
    end
end

nSamples = size(X,2);

if nSamples == 0
    error('No training samples were created. Check file names and goodSeeds.');
end

fprintf('Built %d imitation samples from tabular cycles.\n', nSamples);

% ====================
% initialize NN
% ====================

net.W1 = 0.1 * randn(m, d);
net.b1 = zeros(m, 1);
net.W2 = 0.1 * randn(nA, m);
net.b2 = zeros(nA, 1);

% ====================
% supervised imitation training
% ====================

lossHist = zeros(nEpochs,1);
accHist  = zeros(nEpochs,1);

for ep = 1:nEpochs
    idx = randperm(nSamples);

    totalLoss = 0;
    nCorrect = 0;

    for k = 1:nSamples
        i = idx(k);

        x = X(:,i);
        y = Y(:,i);

        [qhat, h] = nn_forward(net, x);

        e = qhat - y;
        totalLoss = totalLoss + 0.5 * sum(e.^2);

        [~, a_pred] = max(qhat);
        [~, a_true] = max(y);
        if a_pred == a_true
            nCorrect = nCorrect + 1;
        end

        delta2 = e;
        delta1 = (net.W2' * delta2) .* h .* (1 - h);

        net.W2 = net.W2 - eta * (delta2 * h');
        net.b2 = net.b2 - eta * delta2;

        net.W1 = net.W1 - eta * (delta1 * x');
        net.b1 = net.b1 - eta * delta1;
    end

    lossHist(ep) = totalLoss / nSamples;
    accHist(ep)  = nCorrect / nSamples;

    if mod(ep,200) == 0
        fprintf('Epoch %d / %d, loss = %.6e, acc = %.2f%%\n', ...
            ep, nEpochs, lossHist(ep), 100*accHist(ep));
    end
end

% ====================
% final diagnostics
% ====================

nCorrect = 0;
for i = 1:nSamples
    qhat = nn_forward(net, X(:,i));
    [~, a_pred] = max(qhat);
    [~, a_true] = max(Y(:,i));
    if a_pred == a_true
        nCorrect = nCorrect + 1;
    end
end
trainAcc = nCorrect / nSamples;

fprintf('Final imitation accuracy = %.2f%%\n', 100*trainAcc);

figure;
plot(lossHist, 'LineWidth', 1.5);
xlabel('epoch');
ylabel('loss');
title('Tabular cycle imitation loss');
grid on;

figure;
plot(accHist, 'LineWidth', 1.5);
xlabel('epoch');
ylabel('accuracy');
title('Tabular cycle imitation accuracy');
grid on;

% ====================
% save pretrained net
% ====================

seedStr = sprintf('%d_', goodSeeds);
seedStr = seedStr(1:end-1);

fout = sprintf(['pretrained_tabularCycleImitation_seeds%s_g%1.2f_eps0%1.2f_' ...
                'alp0%1.2f_nEpisode%d_m%d.mat'], ...
                seedStr, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab, m);

save(fout, 'net', 'lossHist', 'accHist', 'trainAcc', ...
     'goodSeeds', 'cycleWeight', 'nEpochs', 'eta', ...
     'gamma_tab', 'epsilon0_tab', 'alpha0_tab', 'nEpisodes_tab', 'm');

fprintf('Saved pretrained net to %s\n', fout);

% ====================
% helper functions
% ====================

function x = state_to_x_dphi(env, s_prev, s_curr)
    sub_prev = env.state2sub(s_prev);
    phi_prev = env.sub2phi(sub_prev);

    sub_curr = env.state2sub(s_curr);
    phi_curr = env.sub2phi(sub_curr);

    phimax = env.P.phimax(:);

    phi_norm  = phi_curr(:) ./ phimax;
    dphi_norm = (phi_curr(:) - phi_prev(:)) ./ phimax;

    x = [phi_norm; dphi_norm];
end

function aIdx = infer_action_index(env, s_curr, s_next)
    sub_curr = env.state2sub(s_curr);
    sub_next = env.state2sub(s_next);

    aEff = sub_next - sub_curr;

    rowMatch = ismember(env.actions, aEff, 'rows');
    idx = find(rowMatch, 1);

    if isempty(idx)
        error('Could not infer action index from transition.');
    end

    aIdx = idx;
end

function [q, h] = nn_forward(net, x)
    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));
    q  = net.W2 * h + net.b2;
end