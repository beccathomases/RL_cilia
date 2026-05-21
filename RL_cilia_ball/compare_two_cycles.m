% compare_two_cilia_cycles_interactive.m
%
% Compare two saved cilia cycles side by side, aligned in phase.
% Advances one frame at a time via keyboard.
%
% Required saved variables in each file:
%   cycle_states
%
% Optional saved variables:
%   cycle_rewards
%   avg_cycle_reward
%
% Requires on path:
%   setdefaultparams_ciliaball.m
%   ciliaBallTabularEnv.m
%   position_from_angle.m

clear; clc; close all;

% ============================================================
% USER SETTINGS
% ============================================================

fileA = 'cycle4_run_1_g0.99_eps00.75_alp00.99_nEpisode1000.mat';

fileB = 'cycle4_run_7_g0.99_eps00.75_alp00.99_nEpisode1000.mat';

% if true, force same cycle length by truncating to min length
forceSameLength = true;

% if true, automatically align B to A by cyclic shift
alignByPhase = true;

% body plot styling
lwBody = 3;
msJoint = 6;

% ============================================================
% LOAD / BUILD ENV
% ============================================================

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

A = local_load_cycle(fileA, env, P);
B = local_load_cycle(fileB, env, P);

fprintf('\nLoaded A: %s\n', A.shortName);
fprintf('  cycle length = %d\n', A.n);
if ~isempty(A.avgReward), fprintf('  avg reward   = %.8g\n', A.avgReward); end

fprintf('\nLoaded B: %s\n', B.shortName);
fprintf('  cycle length = %d\n', B.n);
if ~isempty(B.avgReward), fprintf('  avg reward   = %.8g\n', B.avgReward); end

% ============================================================
% OPTIONALLY MATCH LENGTHS
% ============================================================

if forceSameLength
    nCommon = min(A.n, B.n);
    A = local_truncate_cycle(A, nCommon);
    B = local_truncate_cycle(B, nCommon);
else
    nCommon = min(A.n, B.n);
end

% ============================================================
% ALIGN B TO A BY CYCLIC SHIFT
% ============================================================

bestShift = 0;
bestScore = inf;

if alignByPhase
    [bestShift, bestScore] = local_best_cyclic_shift(A.cycle_phis, B.cycle_phis);
    fprintf('\nBest cyclic shift for B relative to A: %d\n', bestShift);
    fprintf('Alignment score (mean phi-distance): %.6g\n', bestScore);

    B = local_shift_cycle(B, bestShift);
end

% ============================================================
% AXIS LIMITS FOR BODY PLOTS
% ============================================================

allx = [cell2mat(A.bodyX); cell2mat(B.bodyX)];
allz = [cell2mat(A.bodyZ); cell2mat(B.bodyZ)];

xmin = min(allx(:)); xmax = max(allx(:));
zmin = min(allz(:)); zmax = max(allz(:));

padx = 0.10 * max(1e-6, xmax - xmin);
padz = 0.10 * max(1e-6, zmax - zmin);

% phase limits
allPhi1 = [A.cycle_phis(:,1); B.cycle_phis(:,1)];
allPhi2 = [A.cycle_phis(:,2); B.cycle_phis(:,2)];

phi1min = min(allPhi1); phi1max = max(allPhi1);
phi2min = min(allPhi2); phi2max = max(allPhi2);

padphi1 = 0.10 * max(1e-6, phi1max - phi1min);
padphi2 = 0.10 * max(1e-6, phi2max - phi2min);

% reward limits
hasRA = ~isempty(A.cycle_rewards);
hasRB = ~isempty(B.cycle_rewards);

if hasRA || hasRB
    rr = [];
    if hasRA, rr = [rr; A.cycle_rewards(:)]; end
    if hasRB, rr = [rr; B.cycle_rewards(:)]; end
    rmin = min(rr); rmax = max(rr);
    padr = 0.10 * max(1e-6, rmax - rmin);
else
    rmin = -1; rmax = 1; padr = 0.1;
end

% ============================================================
% FIGURE SETUP
% ============================================================

fig = figure('Color','w', ...
    'Name','Cilia cycle comparison', ...
    'NumberTitle','off');

tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile(1);  % body A
ax2 = nexttile(2);  % body B
ax3 = nexttile(3);  % phase plane
ax4 = nexttile(4);  % rewards

% ============================================================
% INTERACTIVE STEP-THROUGH
% ============================================================

frame = 1;

while true

    % ----------------------------
    % BODY A
    % ----------------------------
    cla(ax1); hold(ax1, 'on');
    plot(ax1, A.bodyX{frame}, A.bodyZ{frame}, 'b-', 'LineWidth', lwBody);
    plot(ax1, A.bodyX{frame}, A.bodyZ{frame}, 'bo', ...
        'MarkerFaceColor', 'b', 'MarkerSize', msJoint);
    plot(ax1, [xmin-padx, xmax+padx], [0 0], 'k-', 'LineWidth', 2);
    axis(ax1, 'equal');
    xlim(ax1, [xmin-padx, xmax+padx]);
    ylim(ax1, [min(-0.25, zmin-padz), zmax+padz]);
    xlabel(ax1, 'x');
    ylabel(ax1, 'z');
    title(ax1, sprintf('%s | frame %d/%d', A.shortName, frame, A.n), ...
        'Interpreter', 'none');

    % ----------------------------
    % BODY B
    % ----------------------------
    cla(ax2); hold(ax2, 'on');
    plot(ax2, B.bodyX{frame}, B.bodyZ{frame}, 'm-', 'LineWidth', lwBody);
    plot(ax2, B.bodyX{frame}, B.bodyZ{frame}, 'mo', ...
        'MarkerFaceColor', 'm', 'MarkerSize', msJoint);
    plot(ax2, [xmin-padx, xmax+padx], [0 0], 'k-', 'LineWidth', 2);
    axis(ax2, 'equal');
    xlim(ax2, [xmin-padx, xmax+padx]);
    ylim(ax2, [min(-0.25, zmin-padz), zmax+padz]);
    xlabel(ax2, 'x');
    ylabel(ax2, 'z');
    title(ax2, sprintf('%s | frame %d/%d', B.shortName, frame, B.n), ...
        'Interpreter', 'none');

    % ----------------------------
    % PHASE PLANE
    % ----------------------------
    cla(ax3); hold(ax3, 'on');

    plot(ax3, A.cycle_phis(:,1), A.cycle_phis(:,2), 'b-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', A.shortName);
    plot(ax3, B.cycle_phis(:,1), B.cycle_phis(:,2), 'm-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', B.shortName);

    plot(ax3, A.cycle_phis(frame,1), A.cycle_phis(frame,2), 'bo', ...
        'MarkerFaceColor', 'b', 'MarkerSize', 10, 'HandleVisibility', 'off');
    plot(ax3, B.cycle_phis(frame,1), B.cycle_phis(frame,2), 'mo', ...
        'MarkerFaceColor', 'm', 'MarkerSize', 10, 'HandleVisibility', 'off');

    xlabel(ax3, '\phi_1');
    ylabel(ax3, '\phi_2');
    xlim(ax3, [phi1min-padphi1, phi1max+padphi1]);
    ylim(ax3, [phi2min-padphi2, phi2max+padphi2]);
    grid(ax3, 'on');
    legend(ax3, 'Location', 'best', 'Interpreter', 'none');
    title(ax3, sprintf('Phase comparison | B shifted by %d', bestShift), ...
        'Interpreter', 'none');

    % ----------------------------
    % REWARD TRACES
    % ----------------------------
    cla(ax4); hold(ax4, 'on');

    if hasRA
        plot(ax4, 1:A.n, A.cycle_rewards, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
            'DisplayName', [A.shortName ' reward']);
        plot(ax4, frame, A.cycle_rewards(frame), 'bo', ...
            'MarkerFaceColor', 'b', 'MarkerSize', 10, 'HandleVisibility', 'off');
    end

    if hasRB
        plot(ax4, 1:B.n, B.cycle_rewards, 'm-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
            'DisplayName', [B.shortName ' reward']);
        plot(ax4, frame, B.cycle_rewards(frame), 'mo', ...
            'MarkerFaceColor', 'm', 'MarkerSize', 10, 'HandleVisibility', 'off');
    end

    xlabel(ax4, 'Step in cycle');
    ylabel(ax4, 'Reward');
    xlim(ax4, [1, max(A.n,B.n)]);
    ylim(ax4, [rmin-padr, rmax+padr]);
    grid(ax4, 'on');
    legend(ax4, 'Location', 'best', 'Interpreter', 'none');

    if hasRA && hasRB
        title(ax4, sprintf('Rewards | avg A = %.6g, avg B = %.6g', ...
            mean(A.cycle_rewards), mean(B.cycle_rewards)), ...
            'Interpreter', 'none');
    else
        title(ax4, 'Reward traces', 'Interpreter', 'none');
    end

    drawnow;

    % ----------------------------
    % USER INPUT
    % ----------------------------
    fprintf('\nFrame %d/%d\n', frame, nCommon);
    fprintf('Press ENTER/SPACE for next frame, b for back, q to quit.\n');

    w = waitforbuttonpress;
    if w == 1
        ch = get(fig, 'CurrentCharacter');

        if strcmp(ch, 'q')
            fprintf('Quitting comparison.\n');
            break
        elseif strcmp(ch, 'b')
            frame = max(1, frame-1);
        else
            frame = frame + 1;
            if frame > nCommon
                frame = 1;  % wrap around
            end
        end
    else
        frame = frame + 1;
        if frame > nCommon
            frame = 1;
        end
    end
end


% ============================================================
% HELPERS
% ============================================================

function C = local_load_cycle(infile, env, P)

    S = load(infile);

    if ~isfield(S, 'cycle_states')
        error('File %s does not contain cycle_states.', infile);
    end

    cycle_states = S.cycle_states(:).';

    % remove duplicated closing state if present
    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_states = cycle_states(1:end-1);
    end

    n = numel(cycle_states);

    cycle_phis = zeros(n, 2);
    bodyX = cell(n,1);
    bodyZ = cell(n,1);

    for j = 1:n
        sub = env.state2sub(cycle_states(j));
        phi = env.sub2phi(sub);
        cycle_phis(j,:) = phi(:)';

        X = position_from_angle(phi, P);
        XX = [P.X0; X];

        bodyX{j} = XX(:,1);
        bodyZ{j} = XX(:,3);
    end

    C = struct();
    C.infile = infile;
    [~, base, ~] = fileparts(infile);
    C.base = base;
    C.shortName = local_short_name(base);
    C.cycle_states = cycle_states;
    C.cycle_phis = cycle_phis;
    C.bodyX = bodyX;
    C.bodyZ = bodyZ;
    C.n = n;

    if isfield(S, 'cycle_rewards')
        cr = S.cycle_rewards(:);
        if numel(cr) > n
            cr = cr(1:n);
        end
        C.cycle_rewards = cr;
    else
        C.cycle_rewards = [];
    end

    if isfield(S, 'avg_cycle_reward')
        C.avgReward = S.avg_cycle_reward;
    elseif ~isempty(C.cycle_rewards)
        C.avgReward = mean(C.cycle_rewards);
    else
        C.avgReward = [];
    end
end

function C = local_truncate_cycle(C, nKeep)
    C.cycle_states = C.cycle_states(1:nKeep);
    C.cycle_phis   = C.cycle_phis(1:nKeep,:);
    C.bodyX        = C.bodyX(1:nKeep);
    C.bodyZ        = C.bodyZ(1:nKeep);
    if ~isempty(C.cycle_rewards)
        C.cycle_rewards = C.cycle_rewards(1:nKeep);
    end
    C.n = nKeep;
end

function C = local_shift_cycle(C, shift)
    C.cycle_states = circshift(C.cycle_states, -shift);
    C.cycle_phis   = circshift(C.cycle_phis,   -shift, 1);
    C.bodyX        = circshift(C.bodyX,        -shift);
    C.bodyZ        = circshift(C.bodyZ,        -shift);
    if ~isempty(C.cycle_rewards)
        C.cycle_rewards = circshift(C.cycle_rewards, -shift);
    end
end

function [bestShift, bestScore] = local_best_cyclic_shift(phiA, phiB)
% Find cyclic shift of phiB that best aligns with phiA

    n = min(size(phiA,1), size(phiB,1));
    phiA = phiA(1:n,:);
    phiB = phiB(1:n,:);

    bestShift = 0;
    bestScore = inf;

    for s = 0:n-1
        phiBs = circshift(phiB, -s, 1);
        d = sqrt(sum((phiA - phiBs).^2, 2));
        score = mean(d);

        if score < bestScore
            bestScore = score;
            bestShift = s;
        end
    end
end

function s = local_short_name(base)

    s = strrep(base, '.mat', '');

    s = strrep(s, 'cycle4_run_', 'seed');
    s = strrep(s, 'cycle_tinyNN_dphi_', 'tinyNN_');
    s = strrep(s, 'valueIter_', 'VI_');

    maxLen = 40;
    if numel(s) > maxLen
        s = [s(1:maxLen-3), '...'];
    end
end