function [actor, stats] = update_actor(actor, critic1, critic2, batch, alphaEntropy, actorLR)
% UPDATE_ACTOR
%
% SAC actor objective:
%   L_actor = mean( alpha*logpi(a|s) - min(Q1,Q2)(s,a) )
%
% IMPORTANT:
% This is currently a placeholder. It computes the actor loss for
% monitoring, but does NOT yet update the actor parameters.

    %#ok<*INUSD>
    persistent hasWarned

    % Forward pass pieces you need:
    [action, logProb, ~, ~] = sample_action_and_logprob_from_actor(actor, batch.obs, false);
    [q1, ~] = critic_forward(critic1, batch.obs, action);
    [q2, ~] = critic_forward(critic2, batch.obs, action);

    qMin = min(q1, q2);
    lossActor = mean(alphaEntropy * logProb - qMin);

    stats.loss = lossActor;
    stats.meanLogProb = mean(logProb);
    stats.meanQ = mean(qMin);

    if isempty(hasWarned) || ~hasWarned
        warning(['update_actor.m is currently a placeholder. ' ...
                 'Critic updates are implemented, but true SAC actor gradients ' ...
                 'should be done with autodiff (recommended: dlnetwork).']);
        hasWarned = true;
    end

    % No actor update yet:
    % actor = actor;
end