obsDim = 4;
actDim = 2;
hiddenSize = 32;
bufferSize = 1000;
batchSize = 8;

actor = init_actor(obsDim, actDim, hiddenSize);
critic1 = init_critic(obsDim, actDim, hiddenSize);
critic2 = init_critic(obsDim, actDim, hiddenSize);

buffer = init_replay_buffer(bufferSize, obsDim, actDim);

% fake data
for k = 1:20
    obs = randn(obsDim,1);
    [act,~,~] = sample_action_from_actor(actor, obs, false);
    rew = randn();
    nextObs = randn(obsDim,1);
    done = rand() < 0.1;

    buffer = add_transition(buffer, obs, act, rew, nextObs, done);
end

batch = sample_minibatch(buffer, batchSize);

[q1, ~] = critic_forward(critic1, batch.obs, batch.act);
[q2, ~] = critic_forward(critic2, batch.obs, batch.act);

disp(size(batch.obs))   % should be [4, batchSize]
disp(size(batch.act))   % should be [2, batchSize]
disp(size(q1))          % should be [1, batchSize]
disp(size(q2))          % should be [1, batchSize]