% analyze_tinyNN_dphi_all_cycles.m
%
% Exhaustively analyze all recurrent cycles of a trained tiny-NN dphi policy.
% For dphi policies, the effective deterministic state is the PAIR:
%   z = (prev_s, s)
%
% We start from all diagonal starts z0 = (s0, s0), follow the greedy policy,
% collect every unique recurrent cycle, then save/rank/filter them.
%
% A "good" cycle can be defined by:
%   cycle_len >= minKeepLen
%   avg_cycle_reward > minKeepAvgReward

clear; clc;

nseeds       = 1:5;
nEpisodes    = 25000;
alpha0       = 0.01;
epsilon0     = 0.3;
gamma        = 0.99;
m            = 16;

boundary = 'clip_penalty';
invalid_penalty = -0.1;

% ------------------------------------------------------------
% Cycle filtering thresholds
% ------------------------------------------------------------
minKeepLen = 10;          % try 10, 18, or 20
minKeepAvgReward = 0;     % require positive average reward
plotTopK = 3;             % plot top K good cycles

for seeds = nseeds

    fname = sprintf(['run_tinyNN_dphi_%s_pen%0.2f_seed%d_g%1.3f_eps0%1.2f_' ...
                     'alp0%1.3f_nEpisode%d_m%d.mat'], ...
                     boundary, invalid_penalty, ...
                     seeds, gamma, epsilon0, alpha0, nEpisodes, m);

    if ~isfile(fname)
        fprintf('Could not find %s -- skipping.\n', fname);
        continue
    end

    S = load(fname);
    if ~isfield(S, 'net')
        fprintf('File %s does not contain net -- skipping.\n', fname);
        continue
    end
    net = S.net;

    P = setdefaultparams_ciliaball;

    envOpts = struct();
    envOpts.reset_mode = 'fixed';
    envOpts.precompute = false;
    envOpts.boundary = boundary;
    envOpts.invalid_penalty = invalid_penalty;

    env = ciliaBallTabularEnv(P, envOpts);

    nS = env.nStates;

    % --------------------------------------------------------
    % Explore all starts z0 = (s0, s0)
    % --------------------------------------------------------
    visitedGlobal = false(nS, nS);

    % store unique cycles
    allCycles = struct( ...
        'seed', {}, ...
        'start_state', {}, ...
        'pair_cycle', {}, ...
        'curr_states', {}, ...
        'actions', {}, ...
        'rewards', {}, ...
        'cycle_len', {}, ...
        'avg_cycle_reward', {}, ...
        'n_unique_curr_states', {} );

    cycleKeyMap = containers.Map('KeyType', 'char', 'ValueType', 'int32');

    for s0 = 1:nS

        z_prev = s0;
        z_curr = s0;

        if visitedGlobal(z_prev, z_curr)
            continue
        end

        % local trace for this start
        localStep = zeros(nS, nS);   % 0 means unseen on this walk
        path_prev = [];
        path_curr = [];
        path_actions = [];
        path_rewards = [];

        while true

            % repeated augmented state => recurrent cycle found
            if localStep(z_prev, z_curr) > 0
                idx = localStep(z_prev, z_curr);

                cyc_prev    = path_prev(idx:end);
                cyc_curr    = path_curr(idx:end);
                cyc_actions = path_actions(idx:end);
                cyc_rewards = path_rewards(idx:end);

                pairCycle = [cyc_prev(:), cyc_curr(:)];
                key = canonical_pair_cycle(pairCycle);

                if ~isKey(cycleKeyMap, key)
                    C.seed = seeds;
                    C.start_state = s0;
                    C.pair_cycle = pairCycle;
                    C.curr_states = [cyc_curr(:); cyc_curr(1)];
                    C.actions = cyc_actions(:);
                    C.rewards = cyc_rewards(:);
                    C.cycle_len = numel(cyc_actions);
                    C.avg_cycle_reward = mean(cyc_rewards);
                    C.n_unique_curr_states = numel(unique(cyc_curr));

                    allCycles(end+1) = C; %#ok<SAGROW>
                    cycleKeyMap(key) = numel(allCycles);
                end

                break
            end

            % first visit of this augmented state on this walk
            localStep(z_prev, z_curr) = numel(path_prev) + 1;
            path_prev(end+1) = z_prev; %#ok<SAGROW>
            path_curr(end+1) = z_curr; %#ok<SAGROW>

            % greedy policy on x = [phi; dphi]
            x = state_to_x_dphi(env, z_prev, z_curr);
            [q, ~] = nn_forward(net, x);
            [~, a] = max(q);

            [s2, r] = env.step(z_curr, a);

            path_actions(end+1) = a; %#ok<SAGROW>
            path_rewards(end+1) = r; %#ok<SAGROW>

            z_prev_next = z_curr;
            z_curr_next = s2;

            z_prev = z_prev_next;
            z_curr = z_curr_next;
        end

        % mark whole explored path as globally visited
        for kk = 1:numel(path_prev)
            visitedGlobal(path_prev(kk), path_curr(kk)) = true;
        end
    end

    % --------------------------------------------------------
    % Build summary / rank cycles
    % --------------------------------------------------------
    if isempty(allCycles)
        fprintf('\nSeed %d: no recurrent cycles found.\n', seeds);
        continue
    end

    cycle_len = [allCycles.cycle_len]';
    avg_rew   = [allCycles.avg_cycle_reward]';
    n_unique  = [allCycles.n_unique_curr_states]';
    start_s   = [allCycles.start_state]';

    summaryTable = table( ...
        (1:numel(allCycles))', ...
        cycle_len, avg_rew, n_unique, start_s, ...
        'VariableNames', {'cycle_id','cycle_len','avg_cycle_reward', ...
                          'n_unique_curr_states','start_state'} );

    summaryTable = sortrows(summaryTable, {'cycle_len','avg_cycle_reward'}, ...
                                      {'descend','descend'});

    goodMask = summaryTable.cycle_len >= minKeepLen & ...
               summaryTable.avg_cycle_reward > minKeepAvgReward;

    goodSummaryTable = summaryTable(goodMask, :);

    fprintf('\n=====================================================\n');
    fprintf('Seed %d\n', seeds);
    fprintf('Found %d unique recurrent cycles in augmented state.\n', numel(allCycles));
    fprintf('Keeping %d cycles with len >= %d and avg reward > %.6g\n', ...
        height(goodSummaryTable), minKeepLen, minKeepAvgReward);
    disp(summaryTable);

    % --------------------------------------------------------
    % Plot top K good cycles
    % --------------------------------------------------------
    nPlot = min(plotTopK, height(goodSummaryTable));

    for jj = 1:nPlot
        cid = goodSummaryTable.cycle_id(jj);
        C = allCycles(cid);

        cycle_phis = zeros(numel(C.curr_states), 2);
        for k = 1:numel(C.curr_states)
            sub = env.state2sub(C.curr_states(k));
            phi = env.sub2phi(sub);
            cycle_phis(k,:) = phi(:)';
        end

        figure;
        plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', 'LineWidth', 2);
        xlabel('\phi_1');
        ylabel('\phi_2');
        title(sprintf('Seed %d | cycle %d | len=%d | avg=%.6g', ...
            seeds, cid, C.cycle_len, C.avg_cycle_reward));
        grid on;

        figure;
        plot(1:numel(C.rewards), C.rewards, 'o-', 'LineWidth', 2);
        xlabel('Step in cycle');
        ylabel('Reward');
        title(sprintf('Seed %d | cycle %d rewards | len=%d | avg=%.6g', ...
            seeds, cid, C.cycle_len, C.avg_cycle_reward));
        grid on;
    end

    % --------------------------------------------------------
    % Save everything
    % --------------------------------------------------------
    fout = sprintf(['allCycles_tinyNN_dphi_%s_pen%0.2f_seed%d_g%1.3f_eps0%1.2f_' ...
                    'alp0%1.3f_nEpisode%d_m%d.mat'], ...
                    boundary, invalid_penalty, ...
                    seeds, gamma, epsilon0, alpha0, nEpisodes, m);

    save(fout, 'allCycles', 'summaryTable', 'goodSummaryTable', ...
         'boundary', 'invalid_penalty', ...
         'minKeepLen', 'minKeepAvgReward', ...
         'gamma', 'epsilon0', 'alpha0', 'nEpisodes', 'm', 'seeds', 'envOpts');

    fprintf('Saved %s\n', fout);
end


function x = state_to_x_dphi(env, s_prev, s_curr)
% x = [phi1; phi2; dphi1; dphi2]

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


function key = canonical_pair_cycle(pairCycle)
% Return a canonical string key for a cycle in augmented states,
% invariant under cyclic rotation but NOT reversal.

    L = size(pairCycle, 1);
    bestVec = [];

    for k = 1:L
        rot = pairCycle([k:L, 1:k-1], :);
        vec = reshape(rot.', 1, []);

        if isempty(bestVec) || lex_less(vec, bestVec)
            bestVec = vec;
        end
    end

    key = sprintf('%d_', bestVec);
end


function tf = lex_less(a, b)
% Lexicographic comparison for row vectors
    idx = find(a ~= b, 1, 'first');
    if isempty(idx)
        tf = false;
    else
        tf = a(idx) < b(idx);
    end
end