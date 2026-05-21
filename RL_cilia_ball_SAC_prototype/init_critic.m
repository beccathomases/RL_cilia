function critic = init_critic(obsDim, actDim, hiddenSize)
% INIT_CRITIC
% Tiny 1-hidden-layer critic network for SAC-style continuous control.
%
% Input:
%   [obs; act] in R^{obsDim + actDim}
%
% Output:
%   scalar Q-value

    critic.obsDim = obsDim;
    critic.actDim = actDim;
    critic.hiddenSize = hiddenSize;

    inDim = obsDim + actDim;

    critic.W1 = 0.1 * randn(hiddenSize, inDim);
    critic.b1 = zeros(hiddenSize, 1);

    critic.W2 = 0.1 * randn(1, hiddenSize);
    critic.b2 = 0;
end