% make_cilia_cycle_movies.m
%
% Make MP4 movies from saved cycle .mat files containing cycle_states.
%
% Expected saved variables:
%   cycle_states
%
% Requires on path:
%   setdefaultparams_ciliaball.m
%   ciliaBallTabularEnv.m
%   position_from_angle.m
%
% Optional:
%   cycle_rewards, avg_cycle_reward (used only for titles if present)

clear; clc; close all;

% ============================================================
% USER SETTINGS
% ============================================================

cycleFiles = { ...
    'cycle4_run_1_g0.99_eps00.75_alp00.99_nEpisode1000.mat' ...
    % add more files here
    % 'cycle4_run_2_g0.99_eps00.75_alp00.99_nEpisode1000.mat'
    % 'cycle_tinyNN_dphi_clip_penalty_pen-0.10_seed4_g0.990_eps00.30_alp00.010_nEpisode25000_m16.mat'
    };

nRepeats   = 10;      % how many times to repeat the cycle in the movie
fps        = 2;      % frames per second
doTrail    = true;   % show tail of previous body positions
trailAlpha = 0.20;   % not true alpha in basic MATLAB, used via lighter color
lineWidth  = 3;

% ============================================================
% SET UP ENV / PARAMETERS
% ============================================================

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

if exist('position_from_angle','file') ~= 2
    error('position_from_angle.m must be on the MATLAB path.');
end

% ============================================================
% LOOP OVER FILES
% ============================================================

for fidx = 1:numel(cycleFiles)

    infile = cycleFiles{fidx};

    if ~isfile(infile)
        fprintf('Could not find %s -- skipping.\n', infile);
        continue
    end

    S = load(infile);

    if ~isfield(S, 'cycle_states')
        fprintf('File %s does not contain cycle_states -- skipping.\n', infile);
        continue
    end

    cycle_states = S.cycle_states(:).';

    % if saved with repeated closing state, remove final duplicate
    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_states = cycle_states(1:end-1);
    end

    cycle_len = numel(cycle_states);

    if cycle_len < 1
        fprintf('File %s has empty cycle_states -- skipping.\n', infile);
        continue
    end

    % Convert cycle states to phi and body coordinates
    cycle_phis = zeros(cycle_len, 2);
    bodyX = cell(cycle_len,1);
    bodyZ = cell(cycle_len,1);

    for j = 1:cycle_len
        sub = env.state2sub(cycle_states(j));
        phi = env.sub2phi(sub);
        cycle_phis(j,:) = phi(:)';

        X = position_from_angle(phi, P);   % expected columns include x and z
        XX = [P.X0; X];

        bodyX{j} = XX(:,1);
        bodyZ{j} = XX(:,3);
    end

    % Output movie name
    [~, base, ~] = fileparts(infile);
    outfile = [base, '_movie.mp4'];

    fprintf('\nMaking movie for %s\n', infile);
    fprintf('  cycle length = %d\n', cycle_len);
    fprintf('  output file  = %s\n', outfile);

    v = VideoWriter(outfile, 'MPEG-4');
    v.FrameRate = fps;
    open(v);

    % Determine nice fixed axis limits over whole cycle
    allx = cell2mat(bodyX);
    allz = cell2mat(bodyZ);

    xmin = min(allx(:)); xmax = max(allx(:));
    zmin = min(allz(:)); zmax = max(allz(:));

    padx = 0.10 * max(1e-6, xmax - xmin);
    padz = 0.10 * max(1e-6, zmax - zmin);

    fig = figure('Color','w');
    ax = axes(fig);
    hold(ax, 'on');

    for rep = 1:nRepeats
        for j = 1:cycle_len
            cla(ax);
            hold(ax, 'on');

            % optional trail: draw previous configurations in lighter gray
            if doTrail
                maxTrail = min(cycle_len, 8);
                for kk = max(1, j-maxTrail):j-1
                    plot(ax, bodyX{kk}, bodyZ{kk}, '-', ...
                        'Color', [0.75 0.75 0.75], ...
                        'LineWidth', 1.2);
                end
            end

            % current body
            plot(ax, bodyX{j}, bodyZ{j}, 'k-', 'LineWidth', lineWidth);
            plot(ax, bodyX{j}, bodyZ{j}, 'ro', ...
                'MarkerFaceColor', 'r', 'MarkerSize', 6);

            % floor / reference line
            plot(ax, [xmin-padx, xmax+padx], [0,0], 'k-', 'LineWidth', 2);

            axis(ax, 'equal');
            xlim(ax, [xmin-padx, xmax+padx]);
            ylim(ax, [min(-0.25, zmin-padz), zmax+padz]);

            xlabel(ax, 'x');
            ylabel(ax, 'z');

            shortName = local_short_name(base);

            if isfield(S, 'avg_cycle_reward')
                ttl = sprintf('%s | frame %d/%d | phi = [%.3f, %.3f] | avg reward = %.6g', ...
                    shortName, j, cycle_len, cycle_phis(j,1), cycle_phis(j,2), S.avg_cycle_reward);
            else
                ttl = sprintf('%s | frame %d/%d | phi = [%.3f, %.3f]', ...
                    shortName, j, cycle_len, cycle_phis(j,1), cycle_phis(j,2));
            end

            title(ttl, 'Interpreter', 'none');

            drawnow;
            frame = getframe(fig);
            writeVideo(v, frame);
        end
    end

    close(v);
    fprintf('Saved %s\n', outfile);
end

function s = local_short_name(base)
% Make long file names more readable in titles

s = strrep(base, '.mat', '');

% optional simplifications
s = strrep(s, 'cycle4_run_', 'seed');
s = strrep(s, 'cycle_tinyNN_dphi_', 'tinyNN_');
s = strrep(s, 'valueIter_', 'VI_');

% truncate if still too long
maxLen = 60;
if numel(s) > maxLen
    s = [s(1:maxLen-3), '...'];
end
end