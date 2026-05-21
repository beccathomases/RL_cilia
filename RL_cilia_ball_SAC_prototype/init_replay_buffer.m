function buffer = init_replay_buffer(bufferSize, obsDim, actDim)
% INIT_REPLAY_BUFFER
% Create a simple circular replay buffer for SAC-style training.
%
% Stored fields:
%   obs      : obsDim x bufferSize
%   act      : actDim x bufferSize
%   rew      : 1 x bufferSize
%   nextObs  : obsDim x bufferSize
%   done     : 1 x bufferSize
%
% Bookkeeping:
%   ptr      : next insertion index
%   count    : number of stored transitions so far

    buffer.obs     = zeros(obsDim, bufferSize);
    buffer.act     = zeros(actDim, bufferSize);
    buffer.rew     = zeros(1, bufferSize);
    buffer.nextObs = zeros(obsDim, bufferSize);
    buffer.done    = false(1, bufferSize);

    buffer.ptr = 1;
    buffer.count = 0;
    buffer.maxSize = bufferSize;
end
%% 