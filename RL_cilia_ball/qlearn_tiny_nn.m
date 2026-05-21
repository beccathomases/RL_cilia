function [net, epReturn] = qlearn_tiny_nn(env, nEpisodes, maxSteps, alpha, gamma, epsilon, m)
% qlearn_tiny_nn
% Q-learning with a tiny 1-hidden-layer neural network.
%
% Input to NN:
%   x = normalized angle pair [phi1; phi2]
%
% Network:
%   z1 = W1*x + b1
%   h  = sigmoid(z1)
%   q  = W2*h + b2
%
% Output:
%   q(a) approximates Q(s,a) for each discrete action a
%
% We keep the old tabular environment:
%   s0 = env.reset()
%   [s2, r] = env.step(s, a)
%
% but convert s -> phi -> x before feeding the NN.

    if nargin < 7 || isempty(m)
        m = 4;  % tiny hidden layer
    end

    d  = 2;              % two input angles
    nA = env.nActions;   % one output per action

    % Small random initialization
    net.W1 = 0.1 * randn(m, d);
    net.b1 = zeros(m, 1);
    net.W2 = 0.1 * randn(nA, m);
    net.b2 = zeros(nA, 1);

    epReturn = zeros(nEpisodes, 1);

    for ep = 1:nEpisodes
        s = env.reset();
        G = 0;

        for t = 1:maxSteps
            % ----- current state -> NN input -----
            x = state_to_x(env, s);

            % ----- forward pass -----
            z1 = net.W1*x + net.b1;        % m x 1
            h  = 1 ./ (1 + exp(-z1));      % sigmoid
            q  = net.W2*h + net.b2;        % nA x 1

            % ----- epsilon-greedy action -----
            if rand < epsilon
                a = randi(nA);
            else
                [~, a] = max(q);
            end

            % ----- environment step -----
            [s2, r] = env.step(s, a);  % s2 new state, r is the reward

            % ----- target -----
            x2  = state_to_x(env, s2);
            z1_2 = net.W1*x2 + net.b1;
            h2   = 1 ./ (1 + exp(-z1_2));
            q2   = net.W2*h2 + net.b2;

            target = r + gamma * max(q2);

            % ----- TD error for chosen action -----
            % loss = 1/2 * (q(a) - target)^2
            err = target - q(a);

            % Save row before updating output layer
            W2a_old = net.W2(a,:);

            % ----- output-layer update (only chosen action row) -----
            net.W2(a,:) = net.W2(a,:) + alpha * err * h';
            net.b2(a)   = net.b2(a)   + alpha * err;

            % ----- hidden-layer backprop -----
            delta1 = (W2a_old' * err) .* h .* (1 - h);   % m x 1

            net.W1 = net.W1 + alpha * (delta1 * x');
            net.b1 = net.b1 + alpha * delta1;

            % move on
            s = s2;
            G = G + r;
        end

        epReturn(ep) = G;
    end
end

function x = state_to_x(env, s)
% Convert tabular state index -> angle pair -> normalized column vector
    sub = env.state2sub(s);
    phi = env.sub2phi(sub);
    x   = phi(:) ./ env.P.phimax(:);
end