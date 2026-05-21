function [critic1, critic2, stats] = update_critics( ...
    critic1, critic2, targetCritic1, targetCritic2, actor, batch, gamma, alphaEntropy, criticLR)
% UPDATE_CRITICS
%
% SAC critic update:
%   y = r + gamma*(1-done)*( min(Q1_targ,Q2_targ)(s',a') - alpha*logpi(a'|s') )
%
% This version uses the current hand-rolled 1-hidden-layer critic structs.

    % ------------------------------------------------------------
    % 1. Build SAC target
    % ------------------------------------------------------------
    [nextAction, nextLogProb] = sample_action_and_logprob_from_actor(actor, batch.nextObs, false);

    [q1Targ, ~] = critic_forward(targetCritic1, batch.nextObs, nextAction);
    [q2Targ, ~] = critic_forward(targetCritic2, batch.nextObs, nextAction);

    minQTarg = min(q1Targ, q2Targ);

    doneMask = double(batch.done);  % 1 if done, 0 otherwise

    y = batch.rew + gamma * (1 - doneMask) .* (minQTarg - alphaEntropy * nextLogProb);

    % ------------------------------------------------------------
    % 2. Current critic predictions
    % ------------------------------------------------------------
    [q1, h1] = critic_forward(critic1, batch.obs, batch.act);
    [q2, h2] = critic_forward(critic2, batch.obs, batch.act);

    % ------------------------------------------------------------
    % 3. Critic losses
    % ------------------------------------------------------------
    % Mean squared Bellman error
    e1 = q1 - y;
    e2 = q2 - y;

    loss1 = mean(e1.^2);
    loss2 = mean(e2.^2);

    % ------------------------------------------------------------
    % 4. Backprop for critic1
    % ------------------------------------------------------------
    x = [batch.obs; batch.act];   % input to critic hidden layer
    B = size(x,2);

    dL_dq1 = (2/B) * e1;   % 1 x B

    % Save old W2 before update if you want exact sequential gradients;
    % here one-step use is fine.
    dW2_1 = dL_dq1 * h1.';
    db2_1 = sum(dL_dq1, 2);

    delta1_1 = (critic1.W2.' * dL_dq1) .* (1 - h1.^2);
    dW1_1 = delta1_1 * x.';
    db1_1 = sum(delta1_1, 2);

    critic1.W2 = critic1.W2 - criticLR * dW2_1;
    critic1.b2 = critic1.b2 - criticLR * db2_1;
    critic1.W1 = critic1.W1 - criticLR * dW1_1;
    critic1.b1 = critic1.b1 - criticLR * db1_1;

    % ------------------------------------------------------------
    % 5. Backprop for critic2
    % ------------------------------------------------------------
    dL_dq2 = (2/B) * e2;   % 1 x B

    dW2_2 = dL_dq2 * h2.';
    db2_2 = sum(dL_dq2, 2);

    delta1_2 = (critic2.W2.' * dL_dq2) .* (1 - h2.^2);
    dW1_2 = delta1_2 * x.';
    db1_2 = sum(delta1_2, 2);

    critic2.W2 = critic2.W2 - criticLR * dW2_2;
    critic2.b2 = critic2.b2 - criticLR * db2_2;
    critic2.W1 = critic2.W1 - criticLR * dW1_2;
    critic2.b1 = critic2.b1 - criticLR * db1_2;

    % ------------------------------------------------------------
    % 6. Stats
    % ------------------------------------------------------------
    stats.loss1 = loss1;
    stats.loss2 = loss2;
    stats.meanTarget = mean(y);
    stats.meanQ1 = mean(q1);
    stats.meanQ2 = mean(q2);
end