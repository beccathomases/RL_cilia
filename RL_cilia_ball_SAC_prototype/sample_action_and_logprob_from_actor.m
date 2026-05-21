function [action, logProb, mu, logStd, u] = sample_action_and_logprob_from_actor(actor, obs, deterministic)
% SAMPLE_ACTION_AND_LOGPROB_FROM_ACTOR
%
% Inputs:
%   actor
%   obs           : obsDim x batchSize
%   deterministic : true/false
%
% Outputs:
%   action  : actDim x batchSize, squashed to [-1,1] via tanh
%   logProb : 1 x batchSize
%   mu      : actDim x batchSize
%   logStd  : actDim x batchSize
%   u       : pre-tanh action

    if nargin < 3
        deterministic = false;
    end

    [mu, logStd] = actor_forward(actor, obs);
    std = exp(logStd);

    if deterministic
        u = mu;
    else
        eps = randn(size(mu));
        u = mu + std .* eps;
    end

    action = tanh(u);

    % log-prob of tanh-squashed Gaussian action
    % log N(u; mu, std) - sum log(1 - tanh(u)^2)
    %
    % This is the standard SAC correction term.
    log2pi = log(2*pi);

    logNormal = -0.5 * (((u - mu)./std).^2 + 2*logStd + log2pi);
    logNormal = sum(logNormal, 1);

    squashCorrection = sum(log(1 - action.^2 + 1e-6), 1);

    logProb = logNormal - squashCorrection;
end