function batch = sample_minibatch(buffer, batchSize)
% SAMPLE_MINIBATCH
% Randomly sample a batch of transitions from the replay buffer.

    if buffer.count < batchSize
        error('Not enough samples in replay buffer.');
    end

    idx = randi(buffer.count, [1, batchSize]);

    batch.obs     = buffer.obs(:,idx);
    batch.act     = buffer.act(:,idx);
    batch.rew     = buffer.rew(idx);
    batch.nextObs = buffer.nextObs(:,idx);
    batch.done    = buffer.done(idx);
end