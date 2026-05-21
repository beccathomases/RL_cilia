% 04/20/26
% Sweep the PRETRAINED imitation net over many demonstrated starts.
%
% NO RL / NO FINE-TUNING.
%
% This script:
%   - loads the pretrained imitation net
%   - loads tabular cycle files from many seeds
%   - loops over all phase starts in each cycle
%   - maps each demonstrated start to either:
%         (a) the original coarse grid, or
%         (b) a denser grid
%   - does greedy rollout only
%   - detects the first repeated-state loop
%   - stores a summary table of cycle length / average reward
%
% Recommended use:
%   First run with testMode = 'coarse'
%   Then run with testMode = 'fine'

clear; clc; close all;

% ====================
% pretrained net
% ====================

pretrainFile = 'pretrained_tabularCycleImitation_seeds1_2_3_4_5_6_7_8_9_10_g0.99_eps00.75_alp00.99_nEpisode50000_m32.mat';
pretrainTag  = 'tabCycle1_10';

% ====================
% which tabular cycles to sweep over
% ====================

goodSeeds_tab = 1:10;
nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

% ====================
% choose test mode
% ====================

testMode = 'fine';   % 'coarse' or 'fine'

% ====================
% coarse and fine environments
% ====================

Pcoarse = setdefaultparams_ciliaball;
Pcoarse.Nstates = [11, 21];
Pcoarse.dphi = 2*Pcoarse.phimax./(Pcoarse.Nstates-1);
envCoarse = ciliaBallTabularEnv(Pcoarse, struct('reset_mode','fixed','precompute',false));

Pfine = setdefaultparams_ciliaball;
Pfine.Nstates = [21, 41];   % first denser test
Pfine.dphi = 2*Pfine.phimax./(Pfine.Nstates-1);
envFine = ciliaBallTabularEnv(Pfine, struct('reset_mode','fixed','precompute',false));

fprintf('Coarse Nstates = [%d %d]\n', Pcoarse.Nstates(1), Pcoarse.Nstates(2));
fprintf('Fine   Nstates = [%d %d]\n', Pfine.Nstates(1), Pfine.Nstates(2));

switch lower(testMode)
    case 'coarse'
        envEval = envCoarse;
        nRepeat = 1;
    case 'fine'
        envEval = envFine;
        nRepeat = 2;   % for [11,21] -> [21,41], try repeat 2
    otherwise
        error('Unknown testMode. Use ''coarse'' or ''fine''.');
end

fprintf('Testing mode: %s\n', testMode);
fprintf('Using action repeat nRepeat = %d\n', nRepeat);

% ====================
% rollout settings
% ====================

maxFineSteps = 5000;
minCycleLen  = 2;

% suppress figures by default
makePlots = false;
plotRewardThreshold = 1;

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
% summary storage
% ====================

summary_seed      = [];
summary_phase     = [];
summary_found     = [];
summary_len       = [];
summary_avgR      = [];

% optional: save best example for plotting later
bestFound = false;
bestSeed  = NaN;
bestPhase = NaN;
bestLen   = -Inf;
bestAvgR  = -Inf;
bestCycle = [];
bestPhis  = [];
bestRewards = [];

% ====================
% sweep over tabular seeds and phase starts
% ====================

for seed_tab = goodSeeds_tab

    cycleFile = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
        seed_tab, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

    if ~isfile(cycleFile)
        fprintf('Could not find %s -- skipping.\n', cycleFile);
        continue
    end

    Scyc = load(cycleFile);
    if ~isfield(Scyc,'cycle_states')
        fprintf('File %s does not contain cycle_states -- skipping.\n', cycleFile);
        continue
    end

    cycle_states = Scyc.cycle_states(:)';

    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    K = numel(cycle_core);
    if K < 2
        fprintf('Cycle in %s is too short -- skipping.\n', cycleFile);
        continue
    end

    fprintf('Seed %d: sweeping %d phase starts\n', seed_tab, K);

    for j = 1:K
        if j == 1
            jprev = K;
        else
            jprev = j - 1;
        end

        % coarse-grid demonstrated pair
        s_prev_coarse = cycle_core(jprev);
        s_curr_coarse = cycle_core(j);

        phi_prev = envCoarse.sub2phi(envCoarse.state2sub(s_prev_coarse));
        phi_curr = envCoarse.sub2phi(envCoarse.state2sub(s_curr_coarse));

        % map to evaluation grid
        sub_prev_eval = envEval.phi2sub(phi_prev);
        sub_curr_eval = envEval.phi2sub(phi_curr);

        prev_s = envEval.sub2state(sub_prev_eval);
        s      = envEval.sub2state(sub_curr_eval);

        % greedy rollout
        state_history  = [];
        action_history = [];
        reward_history = [];

        found_cycle = false;
        cycle_len = NaN;
        avg_cycle_reward = NaN;
        cycle_states_eval = [];
        cycle_rewards_eval = [];
        cycle_phis_eval = [];

        currentAction = NaN;

        for t = 1:maxFineSteps
            first_index = find(state_history == s, 1, 'first');

            if ~isempty(first_index)
                cycle_states_eval  = [state_history(first_index:end), s];
                cycle_actions_eval = action_history(first_index:end); %#ok<NASGU>
                cycle_rewards_eval = reward_history(first_index:end);
                cycle_len          = length(cycle_rewards_eval);

                if cycle_len >= minCycleLen
                    found_cycle = true;
                    avg_cycle_reward = mean(cycle_rewards_eval);

                    cycle_phis_eval = zeros(length(cycle_states_eval), 2);
                    for jj = 1:length(cycle_states_eval)
                        sub = envEval.state2sub(cycle_states_eval(jj));
                        phi = envEval.sub2phi(sub);
                        cycle_phis_eval(jj,:) = phi(:)';
                    end
                end
                break
            end

            state_history(end+1) = s;

            % choose new action every nRepeat steps
            if mod(t-1, nRepeat) == 0
                x = state_to_x_dphi(envEval, prev_s, s);
                [q, ~] = nn_forward(net, x);
                [~, currentAction] = max(q);
            end

            action_history(end+1) = currentAction;

            [s2, r] = envEval.step(s, currentAction);
            reward_history(end+1) = r;

            prev_s = s;
            s = s2;
        end

        summary_seed(end+1,1)  = seed_tab;
        summary_phase(end+1,1) = j;
        summary_found(end+1,1) = found_cycle;

        if found_cycle
            summary_len(end+1,1)  = cycle_len;
            summary_avgR(end+1,1) = avg_cycle_reward;

            if (~bestFound) || ...
               (avg_cycle_reward > bestAvgR) || ...
               (abs(avg_cycle_reward - bestAvgR) < 1e-12 && cycle_len > bestLen)

                bestFound  = true;
                bestSeed   = seed_tab;
                bestPhase  = j;
                bestLen    = cycle_len;
                bestAvgR   = avg_cycle_reward;
                bestCycle  = cycle_states_eval;
                bestPhis   = cycle_phis_eval;
                bestRewards = cycle_rewards_eval;
            end
        else
            summary_len(end+1,1)  = NaN;
            summary_avgR(end+1,1) = NaN;
        end
    end
end

% ====================
% final summary
% ====================

summaryTable = table(summary_seed, summary_phase, summary_found, summary_len, summary_avgR, ...
    'VariableNames', {'Seed','Phase','FoundCycle','CycleLength','AvgReward'});

fprintf('\n');
fprintf('==================== SWEEP SUMMARY ====================\n');

nFound = sum(summary_found);
nTotal = length(summary_found);

fprintf('Mode: %s\n', testMode);
fprintf('Found cycles in %d out of %d starts.\n', nFound, nTotal);

if nFound > 0
    validIdx = find(summary_found);
    [~, order] = sort(summary_avgR(validIdx), 'descend');
    bestRows = validIdx(order);

    fprintf('%6s %8s %10s %14s\n', 'Seed', 'Phase', 'Length', 'AvgReward');
    fprintf('%6s %8s %10s %14s\n', '----', '-----', '------', '---------');

    nPrint = min(15, length(bestRows));
    for kk = 1:nPrint
        ii = bestRows(kk);
        fprintf('%6d %8d %10d %14.6f\n', ...
            summary_seed(ii), summary_phase(ii), summary_len(ii), summary_avgR(ii));
    end

    fprintf('\nBest start found:\n');
    fprintf('  Seed %d, phase %d, length %d, avg reward %.6f\n', ...
        bestSeed, bestPhase, bestLen, bestAvgR);
else
    fprintf('No cycles found meeting minCycleLen = %d.\n', minCycleLen);
end

fprintf('=======================================================\n');

% save summary
outBase = sprintf('pretrained_%s_sweep_%s', pretrainTag, testMode);
save([outBase '.mat'], 'summaryTable', 'bestSeed', 'bestPhase', 'bestLen', 'bestAvgR');
writetable(summaryTable, [outBase '.csv']);
fprintf('Saved summary files: %s.mat and %s.csv\n', outBase, outBase);

% optional plot for best example
if bestFound && makePlots && bestAvgR > plotRewardThreshold
    figure;
    plot(bestPhis(:,1), bestPhis(:,2), 'o-', 'LineWidth', 2);
    xlabel('\phi_1');
    ylabel('\phi_2');
    title(sprintf('Best %s-grid pretrained stroke: seed %d, phase %d, length %d, avg reward %.6f', ...
        testMode, bestSeed, bestPhase, bestLen, bestAvgR));
    grid on;
    axis equal;

    figure;
    plot(1:length(bestRewards), bestRewards, 'o-', 'LineWidth', 2);
    xlabel('Step in cycle');
    ylabel('Reward');
    title(sprintf('Best %s-grid rewards: seed %d, phase %d, length %d, avg %.6f', ...
        testMode, bestSeed, bestPhase, bestLen, bestAvgR));
    grid on;
end

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