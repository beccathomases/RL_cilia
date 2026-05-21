% 04/20/26
% Make a movie directly from the PRETRAINED imitation net.
%
% This script:
%   - loads the pretrained imitation net
%   - runs a few greedy rollout trials
%   - detects a cycle
%   - keeps the best cycle found
%   - saves a movie of that cycle

clear; clc; close all;

% ====================
% pretrained net to test
% ====================

pretrainFile = 'pretrained_tabularCycleImitation_seeds1_2_3_4_5_6_7_8_9_10_g0.99_eps00.75_alp00.99_nEpisode50000_m32.mat';
pretrainTag  = 'tabCycle1_10';

% ====================
% rollout / detection settings
% ====================

nTrials      = 5;         % 1 is probably enough, but 5 is fine
rolloutSteps = 5000;
minCycleLen  = 2;
startMode    = 'demo';    % 'demo', 'fixed', or 'random'

% ====================
% movie settings
% ====================

nreps     = 3;       % how many times to repeat the cycle
pausetime = 0.01;    % pause between frames while displaying
saveMovie = true;    % true = save mp4
fps       = 10;

movieFile = sprintf('pretrained_%s_bestCycle_start%s.mp4', pretrainTag, startMode);

% ====================
% demo cycle files (for startMode = 'demo')
% ====================

goodSeeds_tab = 1:10;
nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

% ====================
% environment
% ====================

switch lower(startMode)
    case 'fixed'
        evalResetMode = 'fixed';
    case 'random'
        evalResetMode = 'random';
    case 'demo'
        evalResetMode = 'fixed';   % not used directly, but fine as default
    otherwise
        error('Unknown startMode. Use ''demo'', ''fixed'', or ''random''.');
end

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode', evalResetMode, 'precompute', false));

% ====================
% load pretrained net
% ====================

Spre = load(pretrainFile);

if ~isfield(Spre,'net')
    error('Pretraining file does not contain variable "net".');
end

net = Spre.net;

fprintf('Loaded pretrained net from %s\n', pretrainFile);

% ====================
% load demonstrated tabular cycles
% ====================

demoCycles = {};

if strcmpi(startMode,'demo')
    for seed_tab = goodSeeds_tab
        fin_demo = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
            seed_tab, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

        if ~isfile(fin_demo)
            fprintf('Could not find demo cycle file %s -- skipping.\n', fin_demo);
            continue
        end

        Sdemo = load(fin_demo);

        if ~isfield(Sdemo,'cycle_states')
            fprintf('Demo file %s does not contain cycle_states -- skipping.\n', fin_demo);
            continue
        end

        cyc = Sdemo.cycle_states(:)';

        if numel(cyc) >= 2 && cyc(1) == cyc(end)
            cyc = cyc(1:end-1);
        end

        if numel(cyc) < 2
            fprintf('Demo cycle in %s is too short -- skipping.\n', fin_demo);
            continue
        end

        demoCycles{end+1} = cyc; %#ok<SAGROW>
    end

    if isempty(demoCycles)
        error('startMode = demo, but no demo cycles were loaded.');
    end
end

% ====================
% search for best cycle
% ====================

bestFound   = false;
bestCycle   = [];
bestPhis    = [];
bestLen     = -Inf;
bestAvgR    = -Inf;
bestTrial   = NaN;
bestRewards = [];
bestActions = [];

for trial = 1:nTrials
    rng(trial);

    % rollout start
    switch lower(startMode)
        case 'demo'
            k = randi(numel(demoCycles));
            cyc = demoCycles{k};
            K = numel(cyc);

            j = randi(K);
            if j == 1
                jprev = K;
            else
                jprev = j - 1;
            end

            prev_s = cyc(jprev);
            s = cyc(j);

        case 'fixed'
            s = env.reset();
            prev_s = s;

        case 'random'
            s = env.reset();
            prev_s = s;
    end

    state_history  = [];
    action_history = [];
    reward_history = [];

    found_cycle = false;

    for t = 1:rolloutSteps
        first_index = find(state_history == s, 1, 'first');

        if ~isempty(first_index)
            cycle_states  = [state_history(first_index:end), s];
            cycle_actions = action_history(first_index:end);
            cycle_rewards = reward_history(first_index:end);
            cycle_len     = length(cycle_rewards);

            if cycle_len >= minCycleLen
                found_cycle = true;
                avg_cycle_reward = mean(cycle_rewards);

                cycle_phis = zeros(length(cycle_states), 2);
                for jj = 1:length(cycle_states)
                    sub = env.state2sub(cycle_states(jj));
                    phi = env.sub2phi(sub);
                    cycle_phis(jj,:) = phi(:)';
                end

                fprintf('Trial %d: cycle length %d, avg reward %.6f\n', ...
                    trial, cycle_len, avg_cycle_reward);

                % choose best by avg reward first, then by length
                if (~bestFound) || ...
                   (avg_cycle_reward > bestAvgR) || ...
                   (abs(avg_cycle_reward - bestAvgR) < 1e-12 && cycle_len > bestLen)

                    bestFound   = true;
                    bestCycle   = cycle_states;
                    bestPhis    = cycle_phis;
                    bestLen     = cycle_len;
                    bestAvgR    = avg_cycle_reward;
                    bestTrial   = trial;
                    bestRewards = cycle_rewards;
                    bestActions = cycle_actions;
                end
            else
                fprintf('Trial %d: first loop length %d < %d\n', ...
                    trial, cycle_len, minCycleLen);
            end

            break
        end

        state_history(end+1) = s;

        x = state_to_x_dphi(env, prev_s, s);
        [q, ~] = nn_forward(net, x);

        [~, a] = max(q);
        action_history(end+1) = a;

        [s2, r] = env.step(s, a);
        reward_history(end+1) = r;

        prev_s = s;
        s = s2;
    end

    if ~found_cycle
        fprintf('Trial %d: no cycle found within %d rollout steps.\n', ...
            trial, rolloutSteps);
    end
end

if ~bestFound
    error('No usable cycle was found.');
end

fprintf('\nBest pretrained cycle:\n');
fprintf('  Trial %d, length %d, avg reward %.6f\n', ...
    bestTrial, bestLen, bestAvgR);

% optional quick static plots
figure;
plot(bestPhis(:,1), bestPhis(:,2), 'o-', 'LineWidth', 2);
xlabel('\phi_1');
ylabel('\phi_2');
title(sprintf('Pretrained net stroke (trial %d, length %d, avg reward %.6f)', ...
    bestTrial, bestLen, bestAvgR));
grid on;
axis equal;

figure;
plot(1:length(bestRewards), bestRewards, 'o-', 'LineWidth', 2);
xlabel('Step in cycle');
ylabel('Reward');
title(sprintf('Pretrained net rewards (trial %d, length %d, avg %.6f)', ...
    bestTrial, bestLen, bestAvgR));
grid on;

% save best cycle info too
save(sprintf('pretrained_%s_bestCycle_start%s.mat', pretrainTag, startMode), ...
    'bestCycle', 'bestPhis', 'bestLen', 'bestAvgR', 'bestTrial', ...
    'bestRewards', 'bestActions', 'pretrainFile', 'pretrainTag', 'startMode');

% make movie
playCycleMovie(env, bestCycle, nreps, pausetime, saveMovie, movieFile, fps);

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

function [q, h] = nn_forward(net, x)
    z1 = net.W1 * x + net.b1;
    h  = 1 ./ (1 + exp(-z1));
    q  = net.W2 * h + net.b2;
end

function playCycleMovie(env, cycle_states, nCyclesToShow, pauseTime, saveMovie, movieFile, fps)

    if nargin < 3 || isempty(nCyclesToShow)
        nCyclesToShow = 10;
    end
    if nargin < 4 || isempty(pauseTime)
        pauseTime = 0.1;
    end
    if nargin < 5 || isempty(saveMovie)
        saveMovie = false;
    end
    if nargin < 6 || isempty(movieFile)
        movieFile = 'cycle_movie.mp4';
    end
    if nargin < 7 || isempty(fps)
        fps = 10;
    end

    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    if isempty(cycle_core)
        error('cycle_states is empty.');
    end

    nPhase = length(cycle_core);

    if saveMovie
        v = VideoWriter(movieFile, 'MPEG-4');
        v.FrameRate = fps;
        open(v);
    end

    figure;
    for k = 1:nCyclesToShow
        for j = 1:nPhase
            s = cycle_core(j);

            sub = env.state2sub(s);
            phi = env.sub2phi(sub);

            X  = position_from_angle(phi, env.P);
            XX = [env.P.X0; X];

            clf;
            plot(XX(:,1), XX(:,3), 'k-', 'LineWidth', 3); hold on;
            plot(X(:,1),  X(:,3),  'r.', 'MarkerSize', 30);
            plot([-1 1], [0 0], 'k', 'LineWidth', 5);

            xlim([-1 1]);
            ylim([-0.25 1.5]);
            axis equal;
            grid on;

            title(sprintf('Pretrained net cycle: phase %02d/%02d, period %02d/%02d', ...
                j, nPhase, k, nCyclesToShow));

            drawnow;

            if saveMovie
                frame = getframe(gcf);
                writeVideo(v, frame);
            end

            if pauseTime > 0
                pause(pauseTime);
            end
        end
    end

    if saveMovie
        close(v);
        fprintf('Saved movie to %s\n', movieFile);
    end
end