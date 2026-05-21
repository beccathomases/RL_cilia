% plot_valueIter_cycle.m
%
% Load a saved value-iteration result, inspect the best cycle,
% plot it in (phi1,phi2), and plot per-step rewards around the cycle.

clear; clc; close all;

% ============================================================
% USER SETTINGS
% ============================================================

fname = 'valueIter_cilia2ball_clip_penalty_pen-0.10_g0.990.mat';
cycleIndex = 1;   % which ranked cycle to inspect

% ============================================================
% LOAD SAVED VALUE-ITERATION RESULT
% ============================================================

S = load(fname);

if ~isfield(S,'rankedCycles')
    error('File does not contain rankedCycles.');
end
if ~isfield(S,'P')
    error('File does not contain P.');
end
if ~isfield(S,'envOpts')
    error('File does not contain envOpts.');
end

P = S.P;
envOpts = S.envOpts;
rankedCycles = S.rankedCycles;

if cycleIndex < 1 || cycleIndex > numel(rankedCycles)
    error('cycleIndex out of range.');
end

C = rankedCycles{cycleIndex};

% rebuild env so we can convert states to angles if needed
env = ciliaBallTabularEnv(P, envOpts);

cycle_states  = C.states(:).';
cycle_actions = C.actions(:).';
cycle_rewards = C.rewards(:).';
avg_cycle_reward = C.avg_reward;
cycle_len = numel(cycle_states);

fprintf('Loaded cycle %d\n', cycleIndex);
fprintf('  cycle length      = %d\n', cycle_len);
fprintf('  avg cycle reward  = %.8f\n', avg_cycle_reward);
fprintf('  states            = ');
disp(cycle_states);

% ============================================================
% CONVERT STATES TO ANGLE PAIRS
% ============================================================

cycle_states_closed = [cycle_states, cycle_states(1)];
cycle_phis = zeros(numel(cycle_states_closed), numel(P.Nstates));

for j = 1:numel(cycle_states_closed)
    sub = env.state2sub(cycle_states_closed(j));
    phi = env.sub2phi(sub);
    cycle_phis(j,:) = phi(:).';
end

% ============================================================
% PLOT STROKE IN (phi1, phi2)
% ============================================================

if size(cycle_phis,2) ~= 2
    warning('Cycle plot is set up for 2-angle states. Skipping phase plot.');
else
    figure;
    plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', 'LineWidth', 2);
    xlabel('\phi_1');
    ylabel('\phi_2');
    title(sprintf('Value-iteration cycle %d | len=%d | avg reward=%.6g', ...
        cycleIndex, cycle_len, avg_cycle_reward));
    grid on;
end

% ============================================================
% PLOT REWARDS OVER THE CYCLE
% ============================================================

figure;
plot(1:cycle_len, cycle_rewards, 'o-', 'LineWidth', 2);
xlabel('Step in cycle');
ylabel('Reward');
title(sprintf('Cycle rewards | len=%d | avg=%.6g', cycle_len, avg_cycle_reward));
grid on;

% ============================================================
% OPTIONAL: DISPLAY TABLE OF STATES/ACTIONS/REWARDS
% ============================================================

T = table((1:cycle_len)', cycle_states(:), cycle_actions(:), cycle_rewards(:), ...
    'VariableNames', {'step','state','action','reward'});

disp(T);