function actor = init_actor(obsDim, actDim, hiddenSize)
% INIT_ACTOR
% Tiny 1-hidden-layer actor network for SAC-style continuous control.
%
% Input:
%   obs in R^{obsDim}
%
% Output:
%   mu      in R^{actDim}
%   logStd  in R^{actDim}

    actor.obsDim = obsDim;
    actor.actDim = actDim;
    actor.hiddenSize = hiddenSize;

    % hidden layer
    actor.W1 = 0.1 * randn(hiddenSize, obsDim);
    actor.b1 = zeros(hiddenSize, 1);

    % output layer:
    % first actDim entries are mu
    % next  actDim entries are logStd
    actor.W2 = 0.1 * randn(2*actDim, hiddenSize);
    actor.b2 = zeros(2*actDim, 1);
end