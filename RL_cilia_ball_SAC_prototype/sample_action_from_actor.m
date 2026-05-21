function [action, mu, logStd] = sample_action_from_actor(actor, obs, deterministic)
% SAMPLE_ACTION_FROM_ACTOR
%
% If deterministic = true:
%   action = tanh(mu)
%
% If deterministic = false:
%   sample Gaussian action and squash with tanh

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
end