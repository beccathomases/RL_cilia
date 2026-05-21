function [mu, logStd, h] = actor_forward(actor, obs)
% ACTOR_FORWARD
% Forward pass for the actor.
%
% Returns:
%   mu      : actDim x batchSize
%   logStd  : actDim x batchSize
%   h       : hidden activations

    obs = obs(:,:);   % obsDim x batchSize

    z1 = actor.W1 * obs + actor.b1;
    h  = tanh(z1);    % could also use sigmoid or relu

    out = actor.W2 * h + actor.b2;

    actDim = actor.actDim;
    mu = out(1:actDim, :);
    logStd = out(actDim+1:end, :);

    % clamp logStd to a reasonable range
    logStd = max(min(logStd, 2), -5);
end