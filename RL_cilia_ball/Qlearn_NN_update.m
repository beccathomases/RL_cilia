P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));
env_train = ciliaBallTabularEnv(P, struct('reset_mode','random','precompute',true));
env_eval  = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));


nEpisodes = 10000;
maxSteps  = 100;
alpha     = 0.01;   % smaller than tabular
gamma     = 0.999;
epsilon   = 0.1;
m         = 16;      % tiny hidden layer

[net, G] = qlearn_tiny_nn(env_train, nEpisodes, maxSteps, alpha, gamma, epsilon, m);



% % visualize a greedy rollout
% s = env.reset();
% for t = 1:0
%     env_eval.render(s);
%     pause(0.1)
% 
%     % compute NN Q-values at current state
%     sub = env_eval.state2sub(s);
%     phi = env_eval.sub2phi(sub);
%     x   = phi(:) ./ env.P.phimax(:);   % normalize to roughly [-1,1]
% 
%     z1 = net.W1*x + net.b1;
%     h  = 1 ./ (1 + exp(-z1));          % sigmoid hidden layer
%     q  = net.W2*h + net.b2;            % linear outputs = Q-values
% 
%     [~, a] = max(q);
%     [s, r] = env_eval.step(s, a);
%     fprintf('t=%d, a=%d, r=%g, maxQ=%g\n', t, a, r, max(q));
% end