function [Q, epReturn] = qlearn_tabular(env, nEpisodes, maxSteps, alpha, gamma, epsilon)
% Minimal tabular Q-learning for an env with:
%   s0 = env.reset()
%   [s2, r] = env.step(s, aIdx)

  Q = zeros(env.nStates, env.nActions);
  epReturn = zeros(nEpisodes,1);

  for ep = 1:nEpisodes
    s = env.reset();
    G = 0;

    for t = 1:maxSteps
      % epsilon-greedy
      if rand < epsilon
        a = randi(env.nActions);
      else
        [~, a] = max(Q(s,:));
      end

      [s2, r] = env.step(s, a);

      % Q-learning update
      Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma*max(Q(s2,:)));

      s = s2;
      G = G + r;
    end

    epReturn(ep) = G;
  end
end