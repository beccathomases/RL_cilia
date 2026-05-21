% plot_detected_approx_cycle.m
%
% Load a detected approximate cycle and make:
%   1. phase-plane plot
%   2. reward plot
%
% Expected input file:
%   saved_agents/approx_cycle_from_sac_rollout.mat

clear; clc; close all;

cycleFile = fullfile("saved_agents","approx_cycle_from_sac_rollout.mat");

load(cycleFile, ...
    "cycle_phiHist", "cycle_rewardHist", ...
    "avgRBest", "sumRBest", "lenBest", "distBest");

if ~exist('cycle_phiHist','var') || isempty(cycle_phiHist)
    error('cycle_phiHist not found or empty in %s', cycleFile);
end

if ~exist('cycle_rewardHist','var')
    error('cycle_rewardHist not found in %s', cycleFile);
end

fprintf('Loaded approximate cycle from %s\n', cycleFile);
fprintf('Cycle length     = %d\n', lenBest);
fprintf('Average reward   = %.6f\n', avgRBest);
fprintf('Total reward     = %.6f\n', sumRBest);
fprintf('Closure distance = %.6f\n', distBest);

% ------------------------------------------------------------
% Phase-plane plot
% ------------------------------------------------------------
figure;
plot(cycle_phiHist(1,:), cycle_phiHist(2,:), 'o-', 'LineWidth', 2);
hold on;
plot(cycle_phiHist(1,1), cycle_phiHist(2,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
plot(cycle_phiHist(1,end), cycle_phiHist(2,end), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\phi_1');
ylabel('\phi_2');
title(sprintf('Approximate SAC cycle in phase plane (len=%d, avg reward=%.6f, closure dist=%.6f)', ...
    lenBest, avgRBest, distBest));
legend('cycle','start','end','Location','best');
grid on;
axis equal;

% ------------------------------------------------------------
% Reward plot
% ------------------------------------------------------------
figure;
plot(1:length(cycle_rewardHist), cycle_rewardHist, 'o-', 'LineWidth', 2);
xlabel('Step in cycle');
ylabel('Reward');
title(sprintf('Approximate SAC cycle rewards (avg=%.6f, total=%.6f)', ...
    avgRBest, sumRBest));
grid on;