% 04/20/26
% Detect cycles in checkpointed fine-tuned pretrained tiny-NN runs.
%
% This script:
%   - loads checkpoint files saved during fine-tuning
%   - starts rollout from either:
%         'demo'   = demonstrated tabular cycle pair
%         'fixed'  = env.reset() with fixed reset mode
%         'random' = env.reset() with random reset mode
%   - does a greedy rollout using x = [phi1; phi2; dphi1; dphi2]
%   - detects the first repeated-state loop
%   - saves the cycle information
%   - makes stroke/reward figures when a cycle is found
%   - prints a final summary table at the end

clear; clc; close all;

% ====================
% parameters matching checkpoint files
% ====================

pretrainTag   = 'tabCycle1_10';

gamma         = 0.99;
epsilon0      = 0.050;
alpha0        = 0.0010;
m             = 32;
stuckPenalty  = 0.05;

seedList      = 1:5;

% Which checkpoints to inspect
checkpointEpisodes = 25:25:500;

% ====================
% cycle detection options
% ====================

rolloutSteps  = 5000;
minCycleLen   = 2;          % later try 3 if you want to reject 2-cycles
startMode     = 'demo';     % 'demo', 'fixed', or 'random'

% Plotting filter
%ploton = false;
plotOnlyPositiveReward = true;
rewardPlotThreshold    = 1;   % only plot if avg_cycle_reward > this

% Save summary table?
saveSummaryTable = true;

% ====================
% demo cycle files (for startMode = 'demo')
% ====================

goodSeeds_tab = [1 3 4];
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

        % remove repeated closing state if present
        if numel(cyc) >= 2 && cyc(1) == cyc(end)
            cyc = cyc(1:end-1);
        end

        if numel(cyc) < 2
            fprintf('Demo cycle in %s is too short -- skipping.\n', fin_demo);
            continue
        end

        demoCycles{end+1} = cyc; %#ok<SAGROW>
        fprintf('Loaded demo cycle from seed %d with length %d\n', seed_tab, numel(cyc));
    end

    if isempty(demoCycles)
        error('startMode = demo, but no demo cycles were loaded.');
    end
end

% ====================
% summary storage
% ====================

summary_seed     = [];
summary_ep       = [];
summary_len      = [];
summary_avgR     = [];
summary_plotted  = [];

% ====================
% detect cycles across checkpoints
% ====================

for seeds = seedList
    for ep = checkpointEpisodes

        fin = sprintf(['ckpt_tinyNN_dphi_finetune_%s_seed%d_' ...
                       'ep%d_g%1.3f_eps0%1.3f_alp0%1.4f_' ...
                       'm%d_sp%1.2f.mat'], ...
                       pretrainTag, seeds, ep, gamma, epsilon0, alpha0, ...
                       m, stuckPenalty);

        if ~isfile(fin)
            fprintf('Could not find %s -- skipping.\n', fin);
            continue
        end

        S = load(fin);

        if ~isfield(S,'net')
            fprintf('File %s does not contain "net" -- skipping.\n', fin);
            continue
        end

        net = S.net;

        % --------------------
        % rollout start
        % --------------------
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

        % --------------------
        % greedy rollout
        % --------------------
        state_history  = [];
        action_history = [];
        reward_history = [];

        found_cycle = false;
        cycle_len = NaN;
        avg_cycle_reward = NaN;

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

                    fprintf('Seed %d, checkpoint %d: found cycle of length %d, avg reward = %g\n', ...
                        seeds, ep, cycle_len, avg_cycle_reward);

                    cycle_phis = zeros(length(cycle_states), 2);
                    for jj = 1:length(cycle_states)
                        sub = env.state2sub(cycle_states(jj));
                        phi = env.sub2phi(sub);
                        cycle_phis(jj,:) = phi(:)';
                    end

                    % --------------------
                    % make figures only if reward passes threshold
                    % --------------------
                    shouldPlot = (~plotOnlyPositiveReward) || ...
                                 (avg_cycle_reward > rewardPlotThreshold);

                    % store summary info
                    summary_seed(end+1,1)    = seeds;
                    summary_ep(end+1,1)      = ep;
                    summary_len(end+1,1)     = cycle_len;
                    summary_avgR(end+1,1)    = avg_cycle_reward;
                    summary_plotted(end+1,1) = shouldPlot;

                    if shouldPlot
                        % stroke plot
                        figure;
                        plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', 'LineWidth', 2);
                        xlabel('\phi_1');
                        ylabel('\phi_2');
                        title(sprintf('Seed %d, ep %d stroke (length = %d, avg reward = %.6f)', ...
                            seeds, ep, cycle_len, avg_cycle_reward));
                        grid on;
                        axis equal;

                        % cycle rewards plot
                        figure;
                        plot(1:length(cycle_rewards), cycle_rewards, 'o-', 'LineWidth', 2);
                        xlabel('Step in cycle');
                        ylabel('Reward');
                        title(sprintf('Seed %d, ep %d rewards (length = %d, avg = %.6f)', ...
                            seeds, ep, cycle_len, avg_cycle_reward));
                        grid on;
                    else
                        fprintf('Seed %d, checkpoint %d: cycle found but not plotted because avg reward = %.6f <= %.6f\n', ...
                            seeds, ep, avg_cycle_reward, rewardPlotThreshold);
                    end

                else
                    fprintf('Seed %d, checkpoint %d: first loop length %d < %d, not saving.\n', ...
                        seeds, ep, cycle_len, minCycleLen);
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

        fout = sprintf(['cycle_ckpt_tinyNN_dphi_finetune_%s_seed%d_' ...
                        'ep%d_g%1.3f_eps0%1.3f_alp0%1.4f_' ...
                        'm%d_sp%1.2f_start%s.mat'], ...
                        pretrainTag, seeds, ep, gamma, epsilon0, alpha0, ...
                        m, stuckPenalty, startMode);

        if found_cycle
            save(fout, 'cycle_states', 'cycle_actions', 'cycle_rewards', ...
                 'cycle_phis', 'cycle_len', 'avg_cycle_reward', ...
                 'gamma', 'epsilon0', 'alpha0', 'rolloutSteps', ...
                 'm', 'seeds', 'stuckPenalty', 'pretrainTag', ...
                 'startMode', 'minCycleLen', 'ep');

            fprintf('Saved cycle file %s\n', fout);
        else
            fprintf('Seed %d, checkpoint %d: no cycle found within %d rollout steps.\n', ...
                seeds, ep, rolloutSteps);
        end
    end
end

% ====================
% final summary
% ====================

fprintf('\n');
fprintf('==================== FINAL SUMMARY ====================\n');

if isempty(summary_seed)
    fprintf('No cycles meeting minCycleLen = %d were found.\n', minCycleLen);
else
    % sort by average reward, largest first
    [~, idx] = sort(summary_avgR, 'descend');

    fprintf('%6s %8s %10s %14s %10s\n', ...
        'Seed', 'Ep', 'Length', 'AvgReward', 'Plotted');
    fprintf('%6s %8s %10s %14s %10s\n', ...
        '----', '----', '------', '---------', '-------');

    for k = 1:length(idx)
        ii = idx(k);
        fprintf('%6d %8d %10d %14.6f %10d\n', ...
            summary_seed(ii), summary_ep(ii), summary_len(ii), ...
            summary_avgR(ii), summary_plotted(ii));
    end

    best = idx(1);
    fprintf('\nBest cycle found:\n');
    fprintf('  Seed %d, checkpoint %d, length %d, avg reward %.6f, plotted %d\n', ...
        summary_seed(best), summary_ep(best), summary_len(best), ...
        summary_avgR(best), summary_plotted(best));
end

fprintf('=======================================================\n');

% optional save of summary table
if saveSummaryTable && ~isempty(summary_seed)
    summaryTable = table(summary_seed, summary_ep, summary_len, summary_avgR, summary_plotted, ...
        'VariableNames', {'Seed','CheckpointEpisode','CycleLength','AvgReward','Plotted'});

    save('cycle_checkpoint_summary.mat', 'summaryTable');
    writetable(summaryTable, 'cycle_checkpoint_summary.csv');
    fprintf('Saved summary files: cycle_checkpoint_summary.mat and .csv\n');
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