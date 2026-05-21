function [q, h] = critic_forward(critic, obs, act)
% CRITIC_FORWARD
% Forward pass for the critic.
%
% Inputs:
%   obs : obsDim x batchSize
%   act : actDim x batchSize
%
% Output:
%   q   : 1 x batchSize

    obs = obs(:,:);
    act = act(:,:);

    x = [obs; act];   % (obsDim + actDim) x batchSize

    z1 = critic.W1 * x + critic.b1;
    h  = tanh(z1);

    q = critic.W2 * h + critic.b2;
end