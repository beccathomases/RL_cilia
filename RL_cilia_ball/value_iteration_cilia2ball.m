% value_iteration_cilia2ball.m
%
% Solve the small tabular 2-ball cilia problem by value iteration,
% then extract the optimal deterministic policy and analyze its cycles.
%
% Requires:
%   - setdefaultparams_ciliaball.m
%   - ciliaBallTabularEnv.m
%   - cilia_ball_reward.m
%
% Optional:
%   - position_from_angle.m (only for render/plots if you add them later)

clear; clc;
addpath('../RL_cilia_ball_tabular/');
% ============================================================
% USER SETTINGS
% ============================================================

gamma   = 0.99;      % discount factor
tol     = 1e-10;     % stopping tolerance for value iteration
maxIter = 10000;     % safety cap

% environment options
P = setdefaultparams_ciliaball;

envOpts = struct();
envOpts.reset_mode = 'fixed';
envOpts.precompute = true;

% choose one:
% envOpts.boundary = 'clip';
% envOpts.boundary = 'clip_penalty';
% envOpts.boundary = 'stay_penalty';

envOpts.boundary = 'clip_penalty';
envOpts.invalid_penalty = -0.1;

env = ciliaBallTabularEnv(P, envOpts);

fprintf('Built env with %d states and %d actions.\n', env.nStates, env.nActions);

if isempty(env.Rtable) || isempty(env.Ntable)
    error('Expected precomputed Rtable and Ntable. Set envOpts.precompute = true.');
end

R = env.Rtable;   % nStates x nActions
N = env.Ntable;   % nStates x nActions

nS = env.nStates;
nA = env.nActions;

% ============================================================
% VALUE ITERATION
% ============================================================

V = zeros(nS,1);
policy = ones(nS,1);

for it = 1:maxIter
    Vnew = zeros(nS,1);

    for s = 1:nS
        q_sa = zeros(nA,1);
        for a = 1:nA
            s2 = N(s,a);
            q_sa(a) = R(s,a) + gamma * V(s2);
        end

        [Vnew(s), policy(s)] = max(q_sa);
    end

    err = norm(Vnew - V, inf);

    if mod(it,50) == 0 || it == 1
        fprintf('Iter %4d | ||Vnew - V||_inf = %.3e\n', it, err);
    end

    V = Vnew;

    if err < tol
        fprintf('Value iteration converged at iter %d with err %.3e\n', it, err);
        break
    end
end

if it == maxIter
    warning('Reached maxIter before meeting tolerance.');
end

% ============================================================
% COMPUTE Q* FROM FINAL V
% ============================================================

Qstar = zeros(nS, nA);
for s = 1:nS
    for a = 1:nA
        s2 = N(s,a);
        Qstar(s,a) = R(s,a) + gamma * V(s2);
    end
end

% ============================================================
% FIND ALL UNIQUE CYCLES OF THE GREEDY POLICY
% ============================================================

cycles = find_all_cycles_from_policy(env, policy);
rankedCycles = rank_cycles_from_policy(env, policy, cycles);

fprintf('\nFound %d unique cycles under optimal greedy policy.\n', numel(rankedCycles));

disp('Top 10 cycles by average reward:');
for k = 1:min(10, numel(rankedCycles))
    C = rankedCycles{k};
    fprintf('\nCycle %d\n', k);
    fprintf('  avg reward   = %.8f\n', C.avg_reward);
    fprintf('  cycle length = %d\n', numel(C.states));
    fprintf('  states       = ');
    disp(C.states(:).');
end

% ============================================================
% SAVE RESULTS
% ============================================================

saveName = sprintf('valueIter_cilia2ball_%s_pen%0.2f_g%0.3f.mat', ...
    envOpts.boundary, envOpts.invalid_penalty, gamma);

save(saveName, ...
    'V', 'Qstar', 'policy', 'rankedCycles', ...
    'gamma', 'tol', 'maxIter', 'envOpts', 'P');

fprintf('\nSaved %s\n', saveName);

% ============================================================
% OPTIONAL: show best cycle in angle space
% ============================================================

if ~isempty(rankedCycles)
    bestCycle = rankedCycles{1}.states(:).';
    bestCycleClosed = [bestCycle, bestCycle(1)];

    cycle_phis = zeros(numel(bestCycleClosed), numel(P.Nstates));
    for j = 1:numel(bestCycleClosed)
        sub = env.state2sub(bestCycleClosed(j));
        phi = env.sub2phi(sub);
        cycle_phis(j,:) = phi(:).';
    end

    if size(cycle_phis,2) == 2
        figure;
        plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', 'LineWidth', 2);
        xlabel('\phi_1');
        ylabel('\phi_2');
        title(sprintf('Best value-iteration cycle | len=%d | avg=%.6g', ...
            numel(bestCycle), rankedCycles{1}.avg_reward));
        grid on;
    end
end


% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function cycles = find_all_cycles_from_policy(env, policy)
% Find all unique cycles under a deterministic policy on tabular states.

    nS = env.nStates;
    seenCycleKeys = containers.Map('KeyType','char','ValueType','int32');
    cycles = {};

    for s0 = 1:nS
        cyc = find_cycle_from_start(env, policy, s0);
        key = canonical_cycle_key(cyc);

        if ~isKey(seenCycleKeys, key)
            seenCycleKeys(key) = 1;
            cycles{end+1} = cyc(:).'; %#ok<SAGROW>
        end
    end
end

function cyc = find_cycle_from_start(env, policy, s0)
% Follow deterministic policy until a state repeats.

    visitedStep = zeros(env.nStates,1);
    traj = [];

    s = s0;
    t = 1;

    while visitedStep(s) == 0
        visitedStep(s) = t;
        traj(end+1) = s; %#ok<SAGROW>

        a = policy(s);
        s = env.Ntable(s,a);
        t = t + 1;
    end

    cycStart = visitedStep(s);
    cyc = traj(cycStart:end);
end

function key = canonical_cycle_key(cyc)
% Canonical representation of a cycle up to cyclic rotation.

    cyc = cyc(:).';
    L = numel(cyc);

    best = [];

    for k = 1:L
        rot = cyc([k:L, 1:k-1]);

        if isempty(best) || lex_less(rot, best)
            best = rot;
        end
    end

    key = sprintf('%d_', best);
end

function tf = lex_less(a, b)
% True if row vector a is lexicographically smaller than b.

    idx = find(a ~= b, 1, 'first');
    if isempty(idx)
        tf = false;
    else
        tf = a(idx) < b(idx);
    end
end

function ranked = rank_cycles_from_policy(env, policy, cycles)
% Rank cycles by average reward under the deterministic policy.

    ranked = cell(size(cycles));

    for i = 1:numel(cycles)
        cyc = cycles{i};
        L = numel(cyc);

        rewards = zeros(L,1);
        actions = zeros(L,1);

        for j = 1:L
            s = cyc(j);
            a = policy(s);
            actions(j) = a;
            rewards(j) = env.Rtable(s,a);
        end

        C.states = cyc(:).';
        C.actions = actions(:).';
        C.rewards = rewards(:).';
        C.avg_reward = mean(rewards);

        ranked{i} = C;
    end

    avgVals = cellfun(@(C) C.avg_reward, ranked);
    [~, perm] = sort(avgVals, 'descend');
    ranked = ranked(perm);
end