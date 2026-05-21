P = setdefaultparams_ciliaball;

% Precompute is nice for speed once physics is "frozen"
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));

nEpisodes = 1000;
maxSteps = 100;
alpha = .2;
gamma = .95;
epsilon = 0.2;

[Q, G] = qlearn_tabular(env, nEpisodes, maxSteps, alpha, gamma, epsilon);

% visualize a greedy rollout
s = env.reset();
for t = 1:100
  env.render(s);
  pause(.1)
  [~, a] = max(Q(s,:));
  [s, r] = env.step(s, a);
  fprintf('t=%d, r=%g\n', t, r);
end

