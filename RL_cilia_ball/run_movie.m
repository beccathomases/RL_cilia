% 04/20/26
% Pretrain tiny NN from a saved tabular Q-table.
%
% targetMode options:
%   'centeredQ' : target is centered/scaled Q(s,:)
%   'policy'    : target is a preference vector with +1 on best action, -1 elsewhere
%
% Recommended:
%   start with targetMode = 'centeredQ'
%   use a strong/better tabular teacher if possible

clear; clc; close all;

% ====================
% choose tabular file
% ====================

nEpisodes_tab = 1000;   % use your longer/better tabular run here
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;
seed_tab      = 1;

fin = sprintf('run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
    seed_tab, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

S = load(fin);

if ~isfield(S,'Q')
    error('Loaded file does not contain Q.');
end

Qtab = S.Q;

% ====================
% environment
% ====================

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));

nS = env.nStates;
nA = env.nActions;

if ~isequal(size(Qtab), [nS, nA])
    error('Size of Q does not match env.nStates x env.nActions.');
end

% ====================
% NN architecture
% ====================

m = 32;      % try 16 or 32
d = 4;       % [phi1; phi2; dphi1; dphi2]
rng(1);

net.W1 = 0.1 * randn(m, d);
net.b1 = zeros(m, 1);
net.W2 = 0.1 * randn(nA, m);
net.b2 = zeros(nA, 1);

% ====================
% dataset options
% ====================

targetMode = 'centeredQ';   % 'centeredQ' or 'policy'

X = zeros(d, nS);
Y = zeros(nA, nS);

for s = 1:nS
    X(:,s) = state_to_x_dphi_zero(env, s);

    q = Qtab(s,:)';

    switch lower(targetMode)
        case 'centeredq'
            q = q - mean(q);                    % remove offset
            q = q / (max(abs(q)) + 1e-8);      % scale to about [-1,1]
            Y(:,s) = q;

        case 'policy'
            [~, a_star] = max(q);
            y = -ones(nA,1);
            y(a_star) = 1;
            Y(:,s) = y;

        otherwise
            error('Unknown targetMode.');
    end
end

% ====================
% supervised pretraining
% ====================

nEpochs = 8000;      % try 4000 or 8000
eta     = 0.01;
lossHist = zeros(nEpochs,1);

for ep = 1:nEpochs
    dW1 = zeros(size(net.W1));
    db1 = zeros(size(net.b1));
    dW2 = zeros(size(net.W2));
    db2 = zeros(size(net.b2));

    totalLoss = 0;

    for i = 1:nS
        x = X(:,i);
        y = Y(:,i);

        [qhat, h] = nn_forward(net, x);

        e = qhat - y;
        totalLoss = totalLoss + 0.5 * sum(e.^2);

        delta2 = e;   % linear output
        delta1 = (net.W2' * delta2) .* h .* (1 - h);

        dW2 = dW2 + delta2 * h';
        db2 = db2 + delta2;

        dW1 = dW1 + delta1 * x';
        db1 = db1 + delta1;
    end

    dW2 = dW2 / nS;
    db2 = db2 / nS;
    dW1 = dW1 / nS;
    db1 = db1 / nS;

    net.W2 = net.W2 - eta * dW2;
    net.b2 = net.b2 - eta * db2;
    net.W1 = net.W1 - eta * dW1;
    net.b1 = net.b1 - eta * db1;

    lossHist(ep) = totalLoss / nS;

    if mod(ep,200) == 0
        fprintf('Epoch %d / %d, loss = %.6e\n', ep, nEpochs, lossHist(ep));
    end
end

% ====================
% diagnostics
% ====================

Yhat = zeros(nA, nS);
for s = 1:nS
    Yhat(:,s) = nn_forward(net, X(:,s));
end

if strcmpi(targetMode,'centeredQ')
    relErr = norm(Yhat - Y, 'fro') / max(norm(Y, 'fro'), 1e-12);
    fprintf('Relative Frobenius fit error = %.6e\n', relErr);
else
    relErr = NaN;
end

[~, a_tab] = max(Qtab, [], 2);
[~, a_nn]  = max(Yhat, [], 1);
policyMatch = mean(a_tab(:)' == a_nn);
fprintf('Greedy policy agreement = %.2f%%\n', 100 * policyMatch);

figure;
plot(lossHist, 'LineWidth', 1.5);
xlabel('epoch');
ylabel('supervised loss');
title(sprintf('Tabular-to-NN pretraining loss (%s)', targetMode));
grid on;

% ====================
% save pretrained network
% ====================

fout = sprintf(['pretrained_%s_from_tabular_seed%d_g%1.2f_eps0%1.2f_' ...
                'alp0%1.2f_nEpisode%d_m%d.mat'], ...
                targetMode, seed_tab, gamma_tab, epsilon0_tab, ...
                alpha0_tab, nEpisodes_tab, m);

save(fout, 'net', 'lossHist', 'relErr', 'policyMatch', 'targetMode', ...
     'seed_tab', 'gamma_tab', 'epsilon0_tab', 'alpha0_tab', ...
     'nEpisodes_tab', 'm');

fprintf('Saved pretrained net to %s\n', fout);


% ====================
% helper functions
% ====================

function x = state_to_x_dphi_zero(env, s)
% Input for dphi-network pretraining:
% x = [phi1; phi2; 0; 0]
    sub = env.state2sub(s);
    phi = env.sub2phi(sub);
    phi_norm = phi(:) ./ env.P.phimax(:);
    x = [phi_norm; 0; 0];
end

function [q, h] = nn_forward(net, x)
    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));
    q  = net.W2 * h + net.b2;
end