% compare_tabular_cycles_with_valueIter_nice.m
%
% Overlay greedy cycles from several tabular Q-learning runs together with
% the value-iteration cycle.
%
% Plots:
%   1) phase-plane plot:
%        - all tabular cycles in medium gray
%        - value-iteration cycle in black
%
%   2) reward-vs-step plot:
%        - all tabular reward traces in medium gray
%        - mean tabular reward trace in blue
%        - value-iteration reward trace in black

clear; clc; close all;

% ============================================================
% USER SETTINGS
% ============================================================

goodseeds   = 1:10;

% tabular Q-learning file settings
nEpisodes   = 1000;
maxSteps = 500;
alpha0      = 0.99;
epsilon0    = 0.75;
gamma       = 0.99;

% greedy rollout settings for extracting the tabular cycle
rolloutSteps = 500;

% value iteration file
viFile    = 'valueIter_cilia2ball_clip_penalty_pen-0.10_g0.990.mat';
viCycleID = 1;   % which ranked cycle to use

% ============================================================
% BUILD ENVIRONMENT
% ============================================================

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

% ============================================================
% LOAD VALUE-ITERATION CYCLE
% ============================================================

Svi = load(viFile);

if isfield(Svi,'rankedCycles')
    if iscell(Svi.rankedCycles)
        viCycleStruct = Svi.rankedCycles{viCycleID};
    else
        viCycleStruct = Svi.rankedCycles(viCycleID);
    end
else
    error(['Could not find "rankedCycles" in %s. ', ...
        'Change the loading line to match your saved variable name.'], viFile);
end

viCycleStates  = viCycleStruct.states(:)';
viCyclePhis    = states_to_phis(env, viCycleStates);
viCycleRewards = viCycleStruct.rewards(:);
viAvgReward    = viCycleStruct.avg_reward;

fprintf('\nLoaded value-iteration cycle %d from %s\n', viCycleID, viFile);
fprintf('  length = %d\n', length(viCycleRewards));
fprintf('  avg reward = %g\n', viAvgReward);

% ============================================================
% LOAD TABULAR CYCLES
% ============================================================

allCyclePhis    = cell(numel(goodseeds),1);
allCycleRewards = cell(numel(goodseeds),1);
allCycleStates  = cell(numel(goodseeds),1);
allFound        = false(numel(goodseeds),1);
allAvgReward    = nan(numel(goodseeds),1);
allLen          = nan(numel(goodseeds),1);

for k = 1:numel(goodseeds)
    seeds = goodseeds(k);

    fname = sprintf('run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d_maxSteps%d.mat', ...
        seeds, gamma, epsilon0, alpha0, nEpisodes,maxSteps)

    if ~isfile(fname)
        fprintf('Could not find %s -- skipping.\n', fname);
        continue
    end

    S = load(fname);

    if ~isfield(S,'Q')
        fprintf('File %s does not contain Q -- skipping.\n', fname);
        continue
    end

    Q = S.Q;

    [found_cycle, cycle_states, cycle_rewards] = ...
        find_greedy_cycle_from_Q(env, Q, rolloutSteps);

    if found_cycle
        cycle_phis = states_to_phis(env, cycle_states);

        % ------------------------------------------------------------
        % INDIVIDUAL PHASE PLOT FOR THIS SEED
        % ------------------------------------------------------------
        % figure('Color','w');
        % plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', ...
        %     'Color', [0.35 0.35 0.35], ...
        %     'LineWidth', 2, ...
        %     'MarkerSize', 6);
        % xlabel('\phi_1');
        % ylabel('\phi_2');
        % title(sprintf('Tabular seed %d cycle | len=%d | avg reward=%.6g', ...
        %     seeds, length(cycle_rewards), mean(cycle_rewards)));
        % grid on;
        % axis equal;

        % ------------------------------------------------------------
        % INDIVIDUAL REWARD PLOT FOR THIS SEED
        % ------------------------------------------------------------
        % figure('Color','w');
        % plot(1:length(cycle_rewards), cycle_rewards, 'o-', ...
        %     'Color', [0.35 0.35 0.35], ...
        %     'LineWidth', 2, ...
        %     'MarkerSize', 6);
        % xlabel('Step in cycle');
        % ylabel('Reward');
        % title(sprintf('Tabular seed %d reward trace | len=%d | avg reward=%.6g', ...
        %     seeds, length(cycle_rewards), mean(cycle_rewards)));
        % grid on;



        allCycleStates{k}  = cycle_states;
        allCyclePhis{k}    = cycle_phis;
        allCycleRewards{k} = cycle_rewards;
        allFound(k)        = true;
        allAvgReward(k)    = mean(cycle_rewards);
        allLen(k)          = length(cycle_rewards);

        fprintf('Seed %d: cycle length = %d, avg reward = %g\n', ...
            seeds, length(cycle_rewards), mean(cycle_rewards));
    else
        fprintf('Seed %d: no cycle found within %d steps.\n', seeds, rolloutSteps);
    end
end

% ============================================================
% PREPARE MEAN TABULAR REWARD TRACE
% ============================================================

foundIdx = find(allFound);

if isempty(foundIdx)
    warning('No tabular cycles found.');
    meanTabReward = [];
    meanTabX = [];
else
    lens = cellfun(@length, allCycleRewards(foundIdx));
    maxCommonLen = min(lens);

    rewardMat = nan(numel(foundIdx), maxCommonLen);
    for j = 1:numel(foundIdx)
        rr = allCycleRewards{foundIdx(j)};
        rewardMat(j,:) = rr(1:maxCommonLen);
    end

    meanTabReward = mean(rewardMat, 1);
    meanTabX = 1:maxCommonLen;
end

% ============================================================
% PLOT 1: PHASE PLANE OVERLAY
% ============================================================

figure('Color','w');
hold on;

hTab = gobjects(1);
firstTab = true;

for k = 1:numel(goodseeds)
    if allFound(k)
        ph = allCyclePhis{k};

        if firstTab
            hTab = plot(ph(:,1), ph(:,2), 'o-', ...
                'Color', [0.50 0.50 0.50], ...
                'LineWidth', 1.8, ...
                'MarkerSize', 5, ...
                'DisplayName', 'tabular cycles');
            firstTab = false;
        else
            plot(ph(:,1), ph(:,2), 'o-', ...
                'Color', [0.50 0.50 0.50], ...
                'LineWidth', 1.8, ...
                'MarkerSize', 5, ...
                'HandleVisibility', 'off');
        end
    end
end

hVI = plot(viCyclePhis(:,1), viCyclePhis(:,2), 'o-', ...
    'Color', 'k', ...
    'LineWidth', 3.2, ...
    'MarkerSize', 6, ...
    'DisplayName', 'value-iteration cycle');

xlabel('\phi_1');
ylabel('\phi_2');
title('Tabular Q-learning cycles (gray) with value-iteration cycle (black)');
grid on;
axis equal;

legend([hTab, hVI], ...
    {'tabular cycles', 'value-iteration cycle'}, ...
    'Location', 'best');

% ============================================================
% PLOT 2: REWARD TRACE OVERLAY
% ============================================================

figure('Color','w');
hold on;

hTabR = gobjects(1);
firstTabR = true;

for k = 1:numel(goodseeds)
    if allFound(k)
        rr = allCycleRewards{k};
        xx = 1:numel(rr);

        if firstTabR
            hTabR = plot(xx, rr, 'o-', ...
                'Color', [0.55 0.55 0.55], ...
                'LineWidth', 1.8, ...
                'MarkerSize', 5, ...
                'DisplayName', 'tabular cycles');
            firstTabR = false;
        else
            plot(xx, rr, 'o-', ...
                'Color', [0.55 0.55 0.55], ...
                'LineWidth', 1.8, ...
                'MarkerSize', 5, ...
                'HandleVisibility', 'off');
        end
    end
end

if ~isempty(meanTabReward)
    hMean = plot(meanTabX, meanTabReward, 'o-', ...
        'Color', [0.00 0.20 0.90], ...
        'LineWidth', 3.0, ...
        'MarkerSize', 6, ...
        'DisplayName', 'mean tabular');
else
    hMean = gobjects(1);
end

hVIr = plot(1:length(viCycleRewards), viCycleRewards, 'o-', ...
    'Color', 'k', ...
    'LineWidth', 3.2, ...
    'MarkerSize', 6, ...
    'DisplayName', 'value-iteration cycle');

xlabel('Step in cycle');
ylabel('Reward');
title('Cycle reward traces: tabular (gray), mean tabular (blue), value iteration (black)');
grid on;

if ~isempty(meanTabReward)
    legend([hTabR, hMean, hVIr], ...
        {'tabular cycles', 'mean tabular', 'value-iteration cycle'}, ...
        'Location', 'best');
else
    legend([hTabR, hVIr], ...
        {'tabular cycles', 'value-iteration cycle'}, ...
        'Location', 'best');
end

% ============================================================
% OPTIONAL SUMMARY TABLE
% ============================================================

fprintf('\n================ SUMMARY ================\n');
fprintf('Value iteration: len = %d, avg reward = %g\n', ...
    length(viCycleRewards), viAvgReward);

for k = 1:numel(goodseeds)
    if allFound(k)
        fprintf('Tabular seed %d: len = %d, avg reward = %g\n', ...
            goodseeds(k), allLen(k), allAvgReward(k));
    end
end
fprintf('=========================================\n');

% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function [found_cycle, cycle_states, cycle_rewards] = find_greedy_cycle_from_Q(env, Q, rolloutSteps)

s = env.reset();

state_history  = [];
reward_history = [];

found_cycle  = false;
cycle_states = [];
cycle_rewards = [];

for t = 1:rolloutSteps
    first_index = find(state_history == s, 1, 'first');

    if ~isempty(first_index)
        found_cycle = true;
        cycle_states  = [state_history(first_index:end), s];
        cycle_rewards = reward_history(first_index:end);
        return
    end

    state_history(end+1) = s; %#ok<AGROW>

    [~, a] = max(Q(s,:));
    [s, r] = env.step(s, a);

    reward_history(end+1) = r; %#ok<AGROW>
end
end

function cycle_phis = states_to_phis(env, cycle_states)

cycle_phis = zeros(length(cycle_states), 2);

for j = 1:length(cycle_states)
    sub = env.state2sub(cycle_states(j));
    phi = env.sub2phi(sub);
    cycle_phis(j,:) = phi(:)';
end
end