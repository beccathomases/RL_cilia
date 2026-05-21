% evaluate_trained_cilia_sac_longrollout.m
%
% Run one long deterministic rollout of a trained SAC agent from a fixed
% initial condition and save the trajectory.
%
% Output filenames are built automatically from the agent filename.

clear; clc; close all;

% ============================================================
% user settings
% ============================================================

agentFile = fullfile("saved_agents","cilia_sac_seed001_ep300_steps300_random_ssf0.50.mat");
rolloutSteps = 1000;   % long horizon for cycle search

% ============================================================
% build output filename automatically
% ============================================================

[agentFolder, agentBase, ~] = fileparts(agentFile);
rolloutFile = fullfile(agentFolder, agentBase + "_longrollout.mat");

% ============================================================
% load trained agent
% ============================================================

load(agentFile, "agent", "P", "opts");

agent.UseExplorationPolicy = false;

% ============================================================
% build evaluation environment
% fixed reset, long rollout
% ============================================================

optsEval = opts;
optsEval.resetMode = 'fixed';
optsEval.maxSteps = rolloutSteps;

[env, obsInfo, actInfo, P2, opts2] = make_cilia2ball_rltoolbox_env(optsEval); %#ok<ASGLU>

% ============================================================
% rollout
% ============================================================

obs = reset(env);

obsHist    = zeros(4, rolloutSteps+1);
phiHist    = zeros(2, rolloutSteps+1);
actHist    = zeros(2, rolloutSteps);
rewardHist = zeros(1, rolloutSteps);

obsHist(:,1) = obs;
phiHist(:,1) = obs(1:2);

done = false;
kFinal = rolloutSteps;

for k = 1:rolloutSteps
    actionOut = getAction(agent, obs);

    if iscell(actionOut)
        action = actionOut{1};
    else
        action = actionOut;
    end

    [nextObs, reward, done, info] = step(env, action); %#ok<NASGU>

    obsHist(:,k+1) = nextObs;
    phiHist(:,k+1) = nextObs(1:2);
    actHist(:,k)   = action(:);
    rewardHist(k)  = reward;

    obs = nextObs;

    if done
        kFinal = k;
        break
    end
end

% trim to actual used length
obsHist    = obsHist(:,1:kFinal+1);
phiHist    = phiHist(:,1:kFinal+1);
actHist    = actHist(:,1:kFinal);
rewardHist = rewardHist(1:kFinal);

totalReward = sum(rewardHist);

fprintf('Long rollout finished.\n');
fprintf('Agent file: %s\n', agentFile);
fprintf('Steps used: %d\n', kFinal);
fprintf('Total reward: %.6f\n', totalReward);

% ============================================================
% plots
% ============================================================

figure;
plot(phiHist(1,:), phiHist(2,:), 'o-', 'LineWidth', 1.2);
xlabel('\phi_1');
ylabel('\phi_2');
title(sprintf('Long deterministic SAC rollout (%s), total reward = %.6f', ...
    agentBase, totalReward), 'Interpreter', 'none');
grid on;
axis equal;

figure;
plot(1:kFinal, rewardHist, 'o-', 'LineWidth', 1.2);
xlabel('Step');
ylabel('Reward');
title(sprintf('Step rewards during long deterministic rollout (%s)', ...
    agentBase), 'Interpreter', 'none');
grid on;

figure;
plot(1:kFinal, actHist(1,:), 'LineWidth', 1.2); hold on;
plot(1:kFinal, actHist(2,:), 'LineWidth', 1.2);
xlabel('Step');
ylabel('Action');
title(sprintf('Agent actions during long deterministic rollout (%s)', ...
    agentBase), 'Interpreter', 'none');
legend('u_1','u_2');
grid on;

% ============================================================
% save rollout
% ============================================================

save(rolloutFile, ...
    "obsHist", "phiHist", "actHist", "rewardHist", ...
    "totalReward", "kFinal", "P", "optsEval", "agentFile", "agentBase");

fprintf('Saved rollout to %s\n', rolloutFile);