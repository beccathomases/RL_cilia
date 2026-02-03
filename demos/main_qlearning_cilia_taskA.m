%% Cilia Task A: Tabular Q-learning over 3 Chebyshev curvature coefficients
% Task A: maximize forward reach x(L) while keeping y(s) >= 0 (soft penalty)
%
% State: integer coefficient triple c = [c0,c1,c2], each in [-4,4]
% Action: +/- 1 step on one coefficient (6 actions)
%
% Requires:
%   coeffs_to_curve_cheb.m
%   cilia_reward_forward.m
%   cilia_state_index.m
%   cilia_index_state.m
%   cilia_step.m
%   cilia_plot_shape.m

clear; clc; close all; rng(0);
addpath(fullfile(fileparts(mfilename('fullpath')),'..','src'));

%% ---- Geometry / curve model ----
L    = 1.0;
Npts = 400;
N    = 3;        %#ok<NASGU>  % number of coefficients (fixed at 3 here)

%% ---- Coefficient grid ----
cmin = -4;
cmax =  4;
vals = cmin:cmax;
nVals = numel(vals);           % 9
S = nVals^3;                   % 729 states
A = 6;                         % +/- on each of 3 coeffs

%% ---- RL hyperparameters ----
N_EPISODES = 4000;
H = 25;                        % steps per episode (coefficient updates)

alpha0   = 0.40;
alphaMin = 0.05;
gamma    = 0.98;

eps0   = 0.50;
epsMin = 0.05;

%% ---- Reward weights ----
params.wx = 1.0;               % reward on x(L)
params.wb = 0.05;              % bending penalty weight
params.ww = 1.0;               % wall penalty weight multiplier
params.mu_wall = 200.0;        % wall penalty strength
params.terminate_minY = -0.05; % hard terminate if y goes below this

%% ---- Initialize Q-table ----
Q = zeros(S, A);

% Track learning
ep_return = zeros(N_EPISODES,1);


bestReturn = -Inf;
bestC = [0;0;0];

%% ---- Training ----
for ep = 1:N_EPISODES

    alpha = max(alphaMin, alpha0 * (0.995)^(ep-1));
    eps   = max(epsMin,   eps0   * (0.995)^(ep-1));

    c = [0;0;0];  % start each episode at straight filament
    s = cilia_state_index(c, cmin, cmax);

    G = 0;

    for t = 1:H
        % epsilon-greedy action
        if rand < eps
            a = randi(A);
        else
            [~, a] = max(Q(s,:));
        end

        % step
        [s2, r, done, c] = cilia_step(s, a, c, cmin, cmax, L, Npts, params);

        % Q-learning update
        td_target = r + gamma * max(Q(s2,:));
        Q(s,a) = Q(s,a) + alpha * (td_target - Q(s,a));

        s = s2;
        G = G + r;

        if done
            break;
        end
    end
    % Track best episode (by total return)
    if G > bestReturn
        bestReturn = G;
        bestC = c;   % c is the final coefficient vector at episode end
    end


    ep_return(ep) = G;

    if mod(ep, 200) == 0
        recent = ep_return(max(1,ep-199):ep);
        fprintf('Ep %4d | avg return(200): %+0.3f | eps=%0.3f | alpha=%0.3f\n', ...
            ep, mean(recent), eps, alpha);
    end
end

disp('Training done.');

%% ---- Evaluate greedy policy from start state ----
c = [0;0;0];
s = cilia_state_index(c, cmin, cmax);

trajC = zeros(3, H+1);
trajC(:,1) = c;

trajR = zeros(H,1);

for t = 1:H
    [~, a] = max(Q(s,:));
    [s, r, done, c] = cilia_step(s, a, c, cmin, cmax, L, Npts, params);
    trajC(:,t+1) = c;
    trajR(t) = r;
    if done, break; end
end

fprintf('\nGreedy rollout coefficients (columns):\n');
disp(trajC(:,1:(t+1)));

%% ---- Plot learning curve ----
figure; plot(ep_return, 'LineWidth', 1);
xlabel('Episode'); ylabel('Return');
title('Cilia Task A: Episode return');

%% ---- Plot the final greedy shape ----
out = coeffs_to_curve_cheb(c, L, Npts, 0, [0;0]);
[~, info] = cilia_reward_forward(out, params);

figure;
cilia_plot_shape(out);
title(sprintf('Greedy final shape | c=[%d,%d,%d], xL=%.3f, minY=%.3f', ...
    c(1), c(2), c(3), info.xL, info.minY));

%% ---- Show curvature too ----
figure; plot(out.s, out.kappa, 'LineWidth', 2); grid on;
xlabel('s'); ylabel('\kappa(s)');
title('Curvature of greedy final shape');

%% show training

cilia_plot_best_shape(bestC, bestReturn, L, Npts, params);


%% plot slice

cilia_plot_value_slice(Q, cmin, cmax, 0);   % slice at c2 = 0

%% plot trajectory 

cilia_plot_coeff_trajectory(trajC);
