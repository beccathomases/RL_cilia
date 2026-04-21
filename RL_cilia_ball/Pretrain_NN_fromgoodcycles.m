% 04/20/26
% PRETRAIN A TINY NN BY IMITATION FROM GOOD TABULAR CYCLES
%
% Big picture
% -----------
% We already have some "good" strokes from tabular Q-learning runs.
% This script uses those tabular cycles as demonstration data and trains
% a small neural net to imitate the tabular action choices.
%
% In other words, this script does:
%
%     good tabular cycle data  --->  pretrained neural net
%
% The hope is that the neural net captures the stroke pattern in a compact
% way, so that we can later test that controller in other settings.
%
% -------------------------------------------------------------------------
% WHAT GOES IN
% -------------------------------------------------------------------------
% Students should give:
%   - the tabular parameters that match the saved cycle files
%   - a list of good tabular seeds in goodSeeds
%
% The script expects files named like:
%   cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat
%
% -------------------------------------------------------------------------
% WHAT COMES OUT
% -------------------------------------------------------------------------
% The script saves:
%   pretrained_tabularCycleImitation_....mat
%
% This saved file contains:
%   - net        : the pretrained neural network
%   - lossHist   : training loss over epochs
%   - accHist    : imitation accuracy over epochs
%   - trainAcc   : final imitation accuracy
%
% -------------------------------------------------------------------------
% WHAT THE NN SEES
% -------------------------------------------------------------------------
% Each training example has input
%
%   x = [phi1; phi2; dphi1; dphi2]
%
% where
%   phi1, phi2   = current hinge angles
%   dphi1, dphi2 = angle change from previous state to current state
%
% So the NN is not just seeing the current shape; it is also seeing local
% direction information.
%
% -------------------------------------------------------------------------
% WHAT THE NN IS TRYING TO PREDICT
% -------------------------------------------------------------------------
% At each point in a demonstrated tabular cycle, the tabular run takes a
% certain action. We train the NN to reproduce that action choice.
%
% The target vector y is:
%   -1 in every action slot
%   +1 in the action slot that the demonstrated tabular cycle took
%
% So this is imitation learning, not RL.
%
% -------------------------------------------------------------------------
% WHAT STUDENTS SHOULD FIDDLE WITH
% -------------------------------------------------------------------------
% The most natural things to change are:
%
%   1. goodSeeds
%      Which tabular runs are used as demonstrations?
%
%   2. cycleWeight
%      How strongly do we emphasize each demonstrated cycle?
%      Larger = the same cycle is repeated more often in training.
%
%   3. m
%      Hidden-layer size of the NN.
%
%   4. nEpochs
%      Number of supervised training epochs.
%
%   5. eta
%      Supervised learning rate.
%
% -------------------------------------------------------------------------
% WHAT TO LOOK FOR IN THE OUTPUT
% -------------------------------------------------------------------------
% The main printed quantity is:
%
%   Final imitation accuracy = ...
%
% Rough interpretation:
%   - High accuracy (for example around 90% or above) means the NN is doing
%     a good job matching the demonstrated tabular actions on this dataset.
%   - Lower accuracy means the NN is not reproducing the tabular stroke data
%     as well.
%
% Also look at:
%   - lossHist plot: should generally go down
%   - accHist plot : should generally go up and level off
%
% Note:
% High imitation accuracy means the NN has learned the demonstrated data
% well. It does NOT automatically mean that later RL fine-tuning will work.
%
% -------------------------------------------------------------------------
% PARAMETERS TO EDIT
% -------------------------------------------------------------------------

clear; clc; close all;

% These parameters should match the tabular cycle files you want to use.
nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

% Which tabular runs should be used as demonstrations?
% Examples:
%   goodSeeds = [1 3 4];
%   goodSeeds = 1:10;
goodSeeds = 1:10;

% How many times to repeat each demonstrated cycle in the training set.
% Larger values emphasize the demonstrated strokes more strongly.
cycleWeight = 10;       % try 5, 10, 20

% NN size and supervised training settings
m = 32;                 % hidden layer size
d = 4;                  % fixed here: [phi1; phi2; dphi1; dphi2]
nEpochs = 4000;         % try 2000, 4000, 8000
eta = 0.01;             % supervised learning rate

% Fix the random seed for reproducibility.
rng(1);

% ====================
% environment
% ====================

% We build the environment so we can convert between:
%   state index <-> hinge angles <-> action indices
P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));

% Number of possible discrete actions in the environment
nA = env.nActions;

% ====================
% build demonstration dataset from TABULAR cycles
% ====================

% X will store input columns x = [phi1; phi2; dphi1; dphi2]
% Y will store target action-preference columns
X = [];
Y = [];

for seed = goodSeeds

    % Build the expected tabular cycle filename for this seed.
    fin = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
        seed, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

    % Skip if the file is missing.
    if ~isfile(fin)
        fprintf('Could not find %s -- skipping.\n', fin);
        continue
    end

    % Load the saved cycle file.
    S = load(fin);

    % Make sure the file actually contains cycle data.
    if ~isfield(S,'cycle_states')
        fprintf('File %s does not contain cycle_states -- skipping.\n', fin);
        continue
    end

    % Read the saved cycle.
    cycle_states = S.cycle_states(:)';

    % If the cycle was stored as [s1 s2 ... sK s1], then remove the last
    % repeated state so we only keep the core cycle once around.
    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    % Number of states in this cycle
    K = numel(cycle_core);

    % Skip cycles that are too short to be meaningful.
    if K < 2
        fprintf('Cycle in %s is too short -- skipping.\n', fin);
        continue
    end

    fprintf('Using tabular seed %d with cycle length %d\n', seed, K);

    % Build one training example per phase of the cycle.
    %
    % For each phase j:
    %   previous state  -> current state gives dphi
    %   current state   -> next state gives the demonstrated action
    %
    % We repeat the cycle cycleWeight times so the network sees it more often.
    for rep = 1:cycleWeight
        for j = 1:K

            % Wrap around to define previous state
            if j == 1
                s_prev = cycle_core(K);
            else
                s_prev = cycle_core(j-1);
            end

            % Current state
            s_curr = cycle_core(j);

            % Wrap around to define next state
            if j == K
                s_next = cycle_core(1);
            else
                s_next = cycle_core(j+1);
            end

            % Build input x = [phi1; phi2; dphi1; dphi2]
            x = state_to_x_dphi(env, s_prev, s_curr);

            % Infer which discrete action took us from s_curr to s_next
            a = infer_action_index(env, s_curr, s_next);

            % Build target vector y:
            % start with -1 everywhere, then put +1 at the demonstrated action
            y = -ones(nA,1);
            y(a) = 1;

            % Add this training example as a new column
            X(:,end+1) = x;
            Y(:,end+1) = y;
        end
    end
end

% Total number of imitation training examples
nSamples = size(X,2);

if nSamples == 0
    error('No training samples were created. Check file names and goodSeeds.');
end

fprintf('Built %d imitation samples from tabular cycles.\n', nSamples);

% ====================
% initialize NN
% ====================

% The NN has:
%   input dimension d = 4
%   hidden layer size m
%   output dimension nA
%
% We initialize the weights randomly.
net.W1 = 0.1 * randn(m, d);
net.b1 = zeros(m, 1);
net.W2 = 0.1 * randn(nA, m);
net.b2 = zeros(nA, 1);

% ====================
% supervised imitation training
% ====================

% lossHist will track average squared loss
% accHist  will track how often the NN predicts the demonstrated action
lossHist = zeros(nEpochs,1);
accHist  = zeros(nEpochs,1);

for ep = 1:nEpochs

    % Shuffle the training examples each epoch
    idx = randperm(nSamples);

    totalLoss = 0;
    nCorrect = 0;

    for k = 1:nSamples
        i = idx(k);

        % Current training example
        x = X(:,i);
        y = Y(:,i);

        % Forward pass through the NN
        [qhat, h] = nn_forward(net, x);

        % Squared-error loss against target y
        e = qhat - y;
        totalLoss = totalLoss + 0.5 * sum(e.^2);

        % Track imitation accuracy:
        % compare the action with largest predicted score to the
        % demonstrated action
        [~, a_pred] = max(qhat);
        [~, a_true] = max(y);
        if a_pred == a_true
            nCorrect = nCorrect + 1;
        end

        % Backpropagation
        delta2 = e;
        delta1 = (net.W2' * delta2) .* h .* (1 - h);

        % Gradient step
        net.W2 = net.W2 - eta * (delta2 * h');
        net.b2 = net.b2 - eta * delta2;

        net.W1 = net.W1 - eta * (delta1 * x');
        net.b1 = net.b1 - eta * delta1;
    end

    % Store average loss and accuracy for this epoch
    lossHist(ep) = totalLoss / nSamples;
    accHist(ep)  = nCorrect / nSamples;

    % Print progress occasionally
    if mod(ep,200) == 0
        fprintf('Epoch %d / %d, loss = %.6e, acc = %.2f%%\n', ...
            ep, nEpochs, lossHist(ep), 100*accHist(ep));
    end
end

% ====================
% final diagnostics
% ====================

% Recompute final training accuracy on the full dataset
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

% Plot training loss
figure;
plot(lossHist, 'LineWidth', 1.5);
xlabel('epoch');
ylabel('loss');
title('Tabular cycle imitation loss');
grid on;

% Plot training accuracy
figure;
plot(accHist, 'LineWidth', 1.5);
xlabel('epoch');
ylabel('accuracy');
title('Tabular cycle imitation accuracy');
grid on;

% ====================
% save pretrained net
% ====================

% Build a label based on the tabular seeds used
seedStr = sprintf('%d_', goodSeeds);
seedStr = seedStr(1:end-1);

% Saved pretrained-network filename
fout = sprintf(['pretrained_tabularCycleImitation_seeds%s_g%1.2f_eps0%1.2f_' ...
                'alp0%1.2f_nEpisode%d_m%d.mat'], ...
                seedStr, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab, m);

% Save the pretrained network and some useful metadata
save(fout, 'net', 'lossHist', 'accHist', 'trainAcc', ...
     'goodSeeds', 'cycleWeight', 'nEpochs', 'eta', ...
     'gamma_tab', 'epsilon0_tab', 'alpha0_tab', 'nEpisodes_tab', 'm');

fprintf('Saved pretrained net to %s\n', fout);

% ====================
% helper functions
% ====================

function x = state_to_x_dphi(env, s_prev, s_curr)
% Convert two states into the 4-entry NN input:
%   x = [phi1; phi2; dphi1; dphi2]
%
% Here phi is normalized by phimax, and dphi is the normalized difference
% between current and previous angle vectors.

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
% Figure out which discrete action takes s_curr to s_next.
%
% We convert both states to subscripts, take the difference, and then look
% for the matching action row in env.actions.

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
% One forward pass through the tiny NN.
%
% Input:
%   x = 4-by-1 vector [phi1; phi2; dphi1; dphi2]
%
% Output:
%   q = action scores
%   h = hidden-layer activations

    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));
    q  = net.W2 * h + net.b2;
end