% make_movie_from_detected_approx_cycle.m
%
% Make a movie from a detected approximate cycle.
% Output filename is built automatically from the cycle filename.

clear; clc; close all;

% ============================================================
% user settings
% ============================================================

cycleFile = fullfile("saved_agents","cilia_sac_seed001_ep300_steps300_random_ssf0.50_approxCycle.mat");

nreps = 3;
pausetime = 0.02;
saveMovie = true;
fps = 10;

% ============================================================
% build output filename automatically
% ============================================================

[cycleFolder, cycleBase, ~] = fileparts(cycleFile);

if endsWith(cycleBase, "_approxCycle")
    baseStem = extractBefore(cycleBase, strlength(cycleBase) - strlength("_approxCycle") + 1);
else
    baseStem = cycleBase;
end

movieFile = fullfile(cycleFolder, baseStem + "_approxCycle.mp4");

% ============================================================
% load detected cycle
% ============================================================

load(cycleFile, ...
    "cycle_phiHist", "cycle_rewardHist", ...
    "avgRBest", "sumRBest", "lenBest", "distBest", "P", "baseStem");

if ~exist('cycle_phiHist','var') || isempty(cycle_phiHist)
    error('cycle_phiHist not found or empty in %s', cycleFile);
end

nFrames = size(cycle_phiHist, 2);

fprintf('Loaded approximate cycle from %s\n', cycleFile);
fprintf('Cycle length     = %d\n', lenBest);
fprintf('Average reward   = %.6f\n', avgRBest);
fprintf('Total reward     = %.6f\n', sumRBest);
fprintf('Closure distance = %.6f\n', distBest);

% ============================================================
% useful plots
% ============================================================

figure;
plot(cycle_phiHist(1,:), cycle_phiHist(2,:), 'o-', 'LineWidth', 2);
hold on;
plot(cycle_phiHist(1,1), cycle_phiHist(2,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
plot(cycle_phiHist(1,end), cycle_phiHist(2,end), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\phi_1');
ylabel('\phi_2');
title(sprintf('Approximate SAC cycle in phase plane (%s), len=%d, avg reward=%.6f, closure dist=%.6f', ...
    baseStem, lenBest, avgRBest, distBest), 'Interpreter', 'none');
legend('cycle','start','end','Location','best');
grid on;
axis equal;

figure;
plot(1:length(cycle_rewardHist), cycle_rewardHist, 'o-', 'LineWidth', 2);
xlabel('Step in cycle');
ylabel('Reward');
title(sprintf('Approximate SAC cycle rewards (%s), avg=%.6f, total=%.6f', ...
    baseStem, avgRBest, sumRBest), 'Interpreter', 'none');
grid on;

% ============================================================
% optional video writer
% ============================================================

if saveMovie
    v = VideoWriter(movieFile, 'MPEG-4');
    v.FrameRate = fps;
    open(v);
end

% ============================================================
% animate cycle
% ============================================================

figure;

for rep = 1:nreps
    for j = 1:nFrames
        phi = cycle_phiHist(:,j);

        X  = position_from_angle(phi, P);
        XX = [P.X0; X];

        clf;
        plot(XX(:,1), XX(:,3), 'k-', 'LineWidth', 3); hold on;
        plot(X(:,1),  X(:,3),  'r.', 'MarkerSize', 30);
        plot([-1 1], [0 0], 'k', 'LineWidth', 5);

        xlim([-1 1]);
        ylim([-0.25 1.5]);
        axis equal;
        grid on;

        title(sprintf(['Approximate SAC cycle (%s): frame %d/%d, repeat %d/%d\n' ...
                       'len=%d, avg reward=%.6f, total reward=%.6f, closure dist=%.6f'], ...
                       baseStem, j, nFrames, rep, nreps, ...
                       lenBest, avgRBest, sumRBest, distBest), ...
                       'Interpreter', 'none');

        drawnow;

        if saveMovie
            frame = getframe(gcf);
            writeVideo(v, frame);
        end

        if pausetime > 0
            pause(pausetime);
        end
    end
end

if saveMovie
    close(v);
    fprintf('Saved movie to %s\n', movieFile);
end