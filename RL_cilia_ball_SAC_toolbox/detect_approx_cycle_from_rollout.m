% detect_approx_cycle_from_rollout.m
%
% Detect an approximate cycle from a saved long rollout.
% Output filename is built automatically from the rollout filename.

clear; clc; close all;

% ============================================================
% user settings
% ============================================================

rolloutFile = fullfile("saved_agents","cilia_sac_seed001_ep300_steps300_random_ssf0.50_longrollout.mat");

stateTol = 0.20;     % tolerance in normalized 4D state
minLag   = 10;       % ignore tiny immediate returns

% ============================================================
% build output filename automatically
% ============================================================

[rolloutFolder, rolloutBase, ~] = fileparts(rolloutFile);

if endsWith(rolloutBase, "_longrollout")
    baseStem = extractBefore(rolloutBase, strlength(rolloutBase) - strlength("_longrollout") + 1);
else
    baseStem = rolloutBase;
end

cycleFile = fullfile(rolloutFolder, baseStem + "_approxCycle.mat");

% ============================================================
% load rollout
% ============================================================

load(rolloutFile, "obsHist", "phiHist", "actHist", "rewardHist", "P", "optsEval", "agentFile", "agentBase");

T = size(obsHist,2);

% normalized state
Z = zeros(size(obsHist));
Z(1:2,:) = obsHist(1:2,:) ./ P.phimax(:);
Z(3:4,:) = obsHist(3:4,:) ./ optsEval.stepScale(:);

% ============================================================
% search for approximate cycle candidates
% ============================================================

cand_i    = [];
cand_j    = [];
cand_len  = [];
cand_dist = [];
cand_avgR = [];
cand_sumR = [];

for j = 1:T
    for i = 1:(j-minLag)
        dist = norm(Z(:,j) - Z(:,i));

        if dist < stateTol
            cycLen = j - i;
            cycRewards = rewardHist(i:j-1);
            avgR = mean(cycRewards);
            sumR = sum(cycRewards);

            cand_i(end+1,1)    = i; %#ok<SAGROW>
            cand_j(end+1,1)    = j; %#ok<SAGROW>
            cand_len(end+1,1)  = cycLen; %#ok<SAGROW>
            cand_dist(end+1,1) = dist; %#ok<SAGROW>
            cand_avgR(end+1,1) = avgR; %#ok<SAGROW>
            cand_sumR(end+1,1) = sumR; %#ok<SAGROW>
        end
    end
end

if isempty(cand_i)
    fprintf('No approximate cycle found with stateTol = %.3f and minLag = %d\n', ...
        stateTol, minLag);
    return
end

% ============================================================
% choose a best candidate
% prefer avg reward, then length, then closure
% ============================================================

scoreMat = [cand_avgR, cand_len, -cand_dist];
[~, order] = sortrows(scoreMat, [-1 -2 -3]);

bestIdx = order(1);

iBest    = cand_i(bestIdx);
jBest    = cand_j(bestIdx);
lenBest  = cand_len(bestIdx);
distBest = cand_dist(bestIdx);
avgRBest = cand_avgR(bestIdx);
sumRBest = cand_sumR(bestIdx);

fprintf('Approximate cycle found.\n');
fprintf('Agent file       = %s\n', agentFile);
fprintf('Start index      = %d\n', iBest);
fprintf('End index        = %d\n', jBest);
fprintf('Cycle length     = %d\n', lenBest);
fprintf('Closure distance = %.6f\n', distBest);
fprintf('Average reward   = %.6f\n', avgRBest);
fprintf('Total reward     = %.6f\n', sumRBest);

% ============================================================
% extract cycle segment
% ============================================================

cycle_obsHist    = obsHist(:,iBest:jBest);
cycle_phiHist    = phiHist(:,iBest:jBest);
cycle_actHist    = actHist(:,iBest:jBest-1);
cycle_rewardHist = rewardHist(iBest:jBest-1);

candidateTable = table(cand_i, cand_j, cand_len, cand_dist, cand_avgR, cand_sumR, ...
    'VariableNames', {'i','j','len','dist','avgReward','sumReward'});

% ============================================================
% plots
% ============================================================

figure;
plot(phiHist(1,:), phiHist(2,:), '-', 'LineWidth', 1.0); hold on;
plot(cycle_phiHist(1,:), cycle_phiHist(2,:), 'o-', 'LineWidth', 2);
plot(cycle_phiHist(1,1), cycle_phiHist(2,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
plot(cycle_phiHist(1,end), cycle_phiHist(2,end), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\phi_1');
ylabel('\phi_2');
title(sprintf('Approximate cycle in phase plane (%s), len=%d, avgR=%.4f, dist=%.4f', ...
    baseStem, lenBest, avgRBest, distBest), 'Interpreter', 'none');
legend('full rollout','detected cycle','cycle start','cycle end');
grid on;
axis equal;

figure;
plot(1:length(cycle_rewardHist), cycle_rewardHist, 'o-', 'LineWidth', 1.2);
xlabel('Step in approximate cycle');
ylabel('Reward');
title(sprintf('Approximate cycle rewards (%s), avg=%.6f, total=%.6f', ...
    baseStem, avgRBest, sumRBest), 'Interpreter', 'none');
grid on;

% ============================================================
% save cycle segment
% ============================================================

save(cycleFile, ...
    "iBest", "jBest", "lenBest", "distBest", "avgRBest", "sumRBest", ...
    "cycle_obsHist", "cycle_phiHist", "cycle_actHist", "cycle_rewardHist", ...
    "stateTol", "minLag", "candidateTable", ...
    "P", "optsEval", "rolloutFile", "agentFile", "agentBase", "baseStem");

fprintf('Saved approximate cycle data to %s\n', cycleFile);