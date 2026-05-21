function buffer = add_transition(buffer, obs, act, rew, nextObs, done)
% ADD_TRANSITION
% Add one transition to the replay buffer.

    i = buffer.ptr;

    buffer.obs(:,i)     = obs(:);
    buffer.act(:,i)     = act(:);
    buffer.rew(i)       = rew;
    buffer.nextObs(:,i) = nextObs(:);
    buffer.done(i)      = logical(done);

    % advance circular pointer
    buffer.ptr = i + 1;
    if buffer.ptr > buffer.maxSize
        buffer.ptr = 1;
    end

    % increase count up to maxSize
    buffer.count = min(buffer.count + 1, buffer.maxSize);
end