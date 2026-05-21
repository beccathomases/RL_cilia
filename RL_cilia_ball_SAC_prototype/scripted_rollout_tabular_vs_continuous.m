% scripted_rollout_tabular_vs_continuous.m
%
% Sanity check:
% Replay one saved tabular cycle as an action sequence, and compare
% the original tabular dynamics/reward to the continuous version.
%
% Goal:
%   If stepScale = params.dphi and the "continuous" actions are exactly
%   the same {-1,0,1} actions from the tabular cycle, then the two rollouts
%   should match step-by-step.

clear; clc; close all;

% ============================================================
% choose a saved tabular cycle file
% ============================================================

seed_tab      = 1;
nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

cycleFile = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
    seed_tab, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

% ============================================================
% parameters / environment
% ============================================================

P = setdefaultparams_ciliaball;

% For exact consistency with the tabular code, use stepScale = dphi
stepScale = P.dphi(:);

% Build env only for helper conversions state <-> sub <-> phi <-> action
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

fprintf('Using cycle file: %s\n', cycleFile);
fprintf('Using stepScale = params.dphi = [%g, %g]\n', stepScale(1), stepScale(2));

% ============================================================
% load saved cycle
% ============================================================

S = load(cycleFile);
if ~isfield(S,'cycle_states')
    error('Cycle file does not contain cycle_states.');
end

cycle_states = S.cycle_states(:)';

% If stored as [s1 ... sK s1], remove the repeated closing state
if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
    cycle_core = cycle_states(1:end-1);
else
    cycle_core = cycle_states;
end

K = numel(cycle_core);
if K < 2
    error('Cycle is too short.');
end

fprintf('Loaded cycle of length %d\n', K);

% ============================================================
% reconstruct action sequence from cycle transitions
% ============================================================

actionSeq = zeros(K, env.nActions*0 + 2);  % 2 hinge controls here

for j = 1:K
    s_curr = cycle_core(j);

    if j == K
        s_next = cycle_core(1);
    else
        s_next = cycle_core(j+1);
    end

    actionSeq(j,:) = infer_action_vector(env, s_curr, s_next).';
end

fprintf('Reconstructed %d actions from saved cycle.\n', K);

% ============================================================
% initialize both rollouts from the same state
% ============================================================

s0 = cycle_core(1);
phi0 = env.sub2phi(env.state2sub(s0));
phi0 = phi0(:);

phi_tab  = phi0;
phi_cont = phi0;

% storage
phiTabHist   = zeros(2, K+1);
phiContHist  = zeros(2, K+1);
rTabHist     = zeros(K,1);
rContHist    = zeros(K,1);
nextErrHist  = zeros(K,1);
rewardErrHist = zeros(K,1);

phiTabHist(:,1)  = phi_tab;
phiContHist(:,1) = phi_cont;

% ============================================================
% replay the same action sequence in both versions
% ============================================================

for j = 1:K
    a = actionSeq(j,:).';   % action in {-1,0,1}^2

    % --- original tabular reward/update ---
    [r_tab, next_tab] = cilia_ball_reward(phi_tab, a, P);
    next_tab = next_tab(:);

    % --- continuous version with matching action and stepScale = dphi ---
    [r_cont, next_cont] = cilia_ball_reward_continuous(phi_cont, a, P, stepScale);
    next_cont = next_cont(:);

    % store
    rTabHist(j) = r_tab;
    rContHist(j) = r_cont;

    nextErrHist(j)   = norm(next_tab - next_cont);
    rewardErrHist(j) = abs(r_tab - r_cont);

    phi_tab  = next_tab;
    phi_cont = next_cont;

    phiTabHist(:,j+1)  = phi_tab;
    phiContHist(:,j+1) = phi_cont;
end

% ============================================================
% summary
% ============================================================

fprintf('\n');
fprintf('================ ROLLOUT CONSISTENCY SUMMARY ================\n');
fprintf('Cycle length                    : %d\n', K);
fprintf('Max next-state difference       : %.3e\n', max(nextErrHist));
fprintf('Max step reward difference      : %.3e\n', max(rewardErrHist));
fprintf('Total tabular reward            : %.10f\n', sum(rTabHist));
fprintf('Total continuous reward         : %.10f\n', sum(rContHist));
fprintf('Absolute total reward diff      : %.3e\n', abs(sum(rTabHist)-sum(rContHist)));
fprintf('Final state difference          : %.3e\n', norm(phiTabHist(:,end)-phiContHist(:,end)));
fprintf('============================================================\n');

% ============================================================
% optional plots
% ============================================================

figure;
plot(0:K, phiTabHist(1,:), 'o-', 'LineWidth', 1.5); hold on;
plot(0:K, phiContHist(1,:), 'x--', 'LineWidth', 1.5);
xlabel('Step');
ylabel('\phi_1');
title('Tabular vs continuous replay: \phi_1');
legend('tabular','continuous');
grid on;

figure;
plot(0:K, phiTabHist(2,:), 'o-', 'LineWidth', 1.5); hold on;
plot(0:K, phiContHist(2,:), 'x--', 'LineWidth', 1.5);
xlabel('Step');
ylabel('\phi_2');
title('Tabular vs continuous replay: \phi_2');
legend('tabular','continuous');
grid on;

figure;
plot(1:K, rTabHist, 'o-', 'LineWidth', 1.5); hold on;
plot(1:K, rContHist, 'x--', 'LineWidth', 1.5);
xlabel('Step');
ylabel('Reward');
title('Tabular vs continuous replay: step rewards');
legend('tabular','continuous');
grid on;

figure;
plot(1:K, rewardErrHist, 'o-', 'LineWidth', 1.5);
xlabel('Step');
ylabel('|r_{tab} - r_{cont}|');
title('Step reward absolute difference');
grid on;

% save a summary struct too
summary = struct();
summary.seed_tab = seed_tab;
summary.cycle_length = K;
summary.max_next_state_diff = max(nextErrHist);
summary.max_step_reward_diff = max(rewardErrHist);
summary.total_tabular_reward = sum(rTabHist);
summary.total_continuous_reward = sum(rContHist);
summary.total_reward_diff = abs(sum(rTabHist)-sum(rContHist));
summary.final_state_diff = norm(phiTabHist(:,end)-phiContHist(:,end));

save(sprintf('rollout_consistency_seed%d.mat', seed_tab), ...
    'summary', 'phiTabHist', 'phiContHist', 'rTabHist', 'rContHist', ...
    'nextErrHist', 'rewardErrHist', 'actionSeq');

% ============================================================
% helper functions
% ============================================================

function aVec = infer_action_vector(env, s_curr, s_next)
% Infer the discrete action vector in {-1,0,1}^2 from a state transition

    sub_curr = env.state2sub(s_curr);
    sub_next = env.state2sub(s_next);

    aVec = sub_next - sub_curr;
    aVec = aVec(:);
end

function [reward,next_state] = cilia_ball_reward_continuous(state, action, params, stepScale)
% Continuous-action version of cilia_ball_reward
% Here we use it only in the special case where:
%   stepScale = params.dphi
%   action is one of the tabular actions in {-1,0,1}^2

   state     = state(:);
   action    = action(:);
   stepScale = stepScale(:);

   % midpoint state for 1-point Gaussian quadrature
   phi0 = state + 0.5 * stepScale .* action;

   % compute the ball positions
   X = position_from_angle(phi0, params);

   % continuous angular velocity corresponding to this move
   phidot = angvel_from_action_continuous(action, params, stepScale);

   % compute the velocities of the balls
   U = velocity_from_angvel(phi0, phidot, params);

   % form regularized Stokeslets matrix with images
   plane_vec = [0,0,1,0]';
   M = form_stokes_image_system_3D_cm(X, X, params.epsilon, params.mu, plane_vec);

   % solve F = M\U
   F_vec = M \ U(:);
   F = reshape(F_vec, params.N, 3, 1);

   % flux reward
   reward = params.dt/(pi*params.mu) * dot(X(:,3), F(:,1));

   % next state
   next_state = state + stepScale .* action;

   % clip to admissible hinge range
   next_state = max(-params.phimax(:), min(params.phimax(:), next_state));
end

function phidot = angvel_from_action_continuous(action, params, stepScale)
% Continuous analogue of the tabular angular-velocity helper

    action    = action(:);
    stepScale = stepScale(:);

    phidot = action .* stepScale / params.dt;
end