% 04/20/26
% Make a movie of the pretrained NN transferred to the fine grid.
%
% NO RL / NO FINE-TUNING.
%
% This script:
%   - loads the pretrained imitation net
%   - loads one good coarse-grid tabular cycle
%   - picks a chosen phase start from that cycle
%   - maps that start to the fine grid
%   - runs a greedy rollout on the fine grid with repeated actions
%   - detects the first cycle
%   - makes a movie of that cycle

clear; clc; close all;

% ====================
% pretrained net
% ====================

pretrainFile = 'pretrained_tabularCycleImitation_seeds1_2_3_4_5_6_7_8_9_10_g0.99_eps00.75_alp00.99_nEpisode50000_m32.mat';
pretrainTag  = 'tabCycle1_10';

% ====================
% coarse tabular cycle to define the demonstrated start
% ====================

seed_tab      = 1;
nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

cycleFile = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
    seed_tab, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

% Best phase you found from the sweep
j = 6;

% ====================
% coarse and fine environments
% ====================

Pcoarse = setdefaultparams_ciliaball;
Pcoarse.Nstates = [11, 21];
Pcoarse.dphi = 2*Pcoarse.phimax./(Pcoarse.Nstates-1);
envCoarse = ciliaBallTabularEnv(Pcoarse, struct('reset_mode','fixed','precompute',false));

Pfine = setdefaultparams_ciliaball;
Pfine.Nstates = [21, 41];
Pfine.dphi = 2*Pfine.phimax./(Pfine.Nstates-1);
envFine = ciliaBallTabularEnv(Pfine, struct('reset_mode','fixed','precompute',false));

fprintf('Coarse Nstates = [%d %d]\n', Pcoarse.Nstates(1), Pcoarse.Nstates(2));
fprintf('Fine   Nstates = [%d %d]\n', Pfine.Nstates(1), Pfine.Nstates(2));

% ====================
% action repeat
% ====================

nRepeat = 2;
fprintf('Using action repeat nRepeat = %d on fine grid.\n', nRepeat);

% ====================
% movie settings
% ====================

nreps     = 3;       % number of times to repeat the cycle in the movie
pausetime = 0.01;    % pause while displaying
saveMovie = true;    % true = save mp4
fps       = 10;

movieFile = sprintf('pretrained_%s_fineGrid_seed%d_phase%d_repeat%d.mp4', ...
    pretrainTag, seed_tab, j, nRepeat);

% ====================
% rollout settings
% ====================

maxFineSteps = 5000;
minCycleLen  = 2;

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
% load demonstrated coarse-grid cycle
% ====================

Scyc = load(cycleFile);
if ~isfield(Scyc,'cycle_states')
    error('Cycle file does not contain cycle_states.');
end

cycle_states = Scyc.cycle_states(:)';

if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
    cycle_core = cycle_states(1:end-1);
else
    cycle_core = cycle_states;
end

K = numel(cycle_core);
if K < 2
    error('Cycle is too short.');
end

if j < 1 || j > K
    error('Chosen phase j=%d is out of range 1..%d.', j, K);
end

fprintf('Loaded demonstrated coarse-grid cycle from seed %d with length %d\n', seed_tab, K);

% ====================
% choose phase and map to fine grid
% ====================

if j == 1
    jprev = K;
else
    jprev = j - 1;
end

s_prev_coarse = cycle_core(jprev);
s_curr_coarse = cycle_core(j);

phi_prev = envCoarse.sub2phi(envCoarse.state2sub(s_prev_coarse));
phi_curr = envCoarse.sub2phi(envCoarse.state2sub(s_curr_coarse));

sub_prev_fine = envFine.phi2sub(phi_prev);
sub_curr_fine = envFine.phi2sub(phi_curr);

prev_s = envFine.sub2state(sub_prev_fine);
s      = envFine.sub2state(sub_curr_fine);

fprintf('Mapped demonstrated start (seed %d, phase %d) to fine grid.\n', seed_tab, j);

% ====================
% greedy rollout on fine grid
% ====================

state_history  = [];
action_history = [];
reward_history = [];

found_cycle = false;
cycle_len = NaN;
avg_cycle_reward = NaN;

currentAction = NaN;

for t = 1:maxFineSteps
    first_index = find(state_history == s, 1, 'first');

    if ~isempty(first_index)
        cycle_states_fine  = [state_history(first_index:end), s];
        cycle_actions_fine = action_history(first_index:end);
        cycle_rewards_fine = reward_history(first_index:end);
        cycle_len          = length(cycle_rewards_fine);

        if cycle_len >= minCycleLen
            found_cycle = true;
            avg_cycle_reward = mean(cycle_rewards_fine);

            fprintf('Found cycle on fine grid.\n');
            fprintf('Cycle length = %d\n', cycle_len);
            fprintf('Average cycle reward = %.6f\n', avg_cycle_reward);

            cycle_phis_fine = zeros(length(cycle_states_fine), 2);
            for jj = 1:length(cycle_states_fine)
                sub = envFine.state2sub(cycle_states_fine(jj));
                phi = envFine.sub2phi(sub);
                cycle_phis_fine(jj,:) = phi(:)';
            end

            % quick plots
            figure;
            plot(cycle_phis_fine(:,1), cycle_phis_fine(:,2), 'o-', 'LineWidth', 2);
            xlabel('\phi_1');
            ylabel('\phi_2');
            title(sprintf('Fine-grid transferred stroke (seed %d, phase %d, length %d, avg reward %.6f)', ...
                seed_tab, j, cycle_len, avg_cycle_reward));
            grid on;
            axis equal;

            figure;
            plot(1:length(cycle_rewards_fine), cycle_rewards_fine, 'o-', 'LineWidth', 2);
            xlabel('Fine-grid step in cycle');
            ylabel('Reward');
            title(sprintf('Fine-grid rewards (seed %d, phase %d, length %d, avg %.6f)', ...
                seed_tab, j, cycle_len, avg_cycle_reward));
            grid on;
        else
            fprintf('First loop found, but length %d < %d\n', cycle_len, minCycleLen);
        end

        break
    end

    state_history(end+1) = s;

    % choose a new action every nRepeat fine steps
    if mod(t-1, nRepeat) == 0
        x = state_to_x_dphi(envFine, prev_s, s);
        [q, ~] = nn_forward(net, x);
        [~, currentAction] = max(q);
    end

    action_history(end+1) = currentAction;

    [s2, r] = envFine.step(s, currentAction);
    reward_history(end+1) = r;

    prev_s = s;
    s = s2;
end

if ~found_cycle
    error('No usable cycle found on the fine grid.');
end

% save cycle data
save(sprintf('pretrained_%s_fineGrid_seed%d_phase%d_repeat%d.mat', ...
    pretrainTag, seed_tab, j, nRepeat), ...
    'cycle_states_fine', 'cycle_actions_fine', 'cycle_rewards_fine', ...
    'cycle_phis_fine', 'cycle_len', 'avg_cycle_reward', ...
    'seed_tab', 'j', 'nRepeat', 'pretrainFile');

% make movie
playCycleMovie(envFine, cycle_states_fine, nreps, pausetime, saveMovie, movieFile, fps);

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

function playCycleMovie(env, cycle_states, nCyclesToShow, pauseTime, saveMovie, movieFile, fps)

    if nargin < 3 || isempty(nCyclesToShow)
        nCyclesToShow = 10;
    end
    if nargin < 4 || isempty(pauseTime)
        pauseTime = 0.1;
    end
    if nargin < 5 || isempty(saveMovie)
        saveMovie = false;
    end
    if nargin < 6 || isempty(movieFile)
        movieFile = 'cycle_movie.mp4';
    end
    if nargin < 7 || isempty(fps)
        fps = 10;
    end

    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    if isempty(cycle_core)
        error('cycle_states is empty.');
    end

    nPhase = length(cycle_core);

    if saveMovie
        v = VideoWriter(movieFile, 'MPEG-4');
        v.FrameRate = fps;
        open(v);
    end

    figure;
    for k = 1:nCyclesToShow
        for jj = 1:nPhase
            s = cycle_core(jj);

            sub = env.state2sub(s);
            phi = env.sub2phi(sub);

            X  = position_from_angle(phi, env.P);
            XX = [env.P.X0; X];

            clf;
            plot(XX(:,1), XX(:,3), 'k-', 'LineWidth', 3); hold on;
            plot(X(:,1),  X(:,3),  'r.', 'MarkerSize', 30);
            plot([-1 1], [0 0], 'k', 'LineWidth', 5);

            xlim([-1 1]);
            ylim([-0.25 1.5]);
            axis equal;
            grid on;

            title(sprintf('Fine-grid transferred cycle: phase %02d/%02d, period %02d/%02d', ...
                jj, nPhase, k, nCyclesToShow));

            drawnow;

            if saveMovie
                frame = getframe(gcf);
                writeVideo(v, frame);
            end

            if pauseTime > 0
                pause(pauseTime);
            end
        end
    end

    if saveMovie
        close(v);
        fprintf('Saved movie to %s\n', movieFile);
    end
end