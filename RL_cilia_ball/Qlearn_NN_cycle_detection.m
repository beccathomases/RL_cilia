% notes : use q-table somehow? 

nEpisodes = 10000;
alpha0 = 0.01;
epsilon0 = 0.3;
gamma = 0.99;
m = 32;
stroke_penalty = .05;

for seeds = 1:6
    fname = sprintf(['run_tinyNN_seed%d_g%1.3f_eps0%1.2f_' ...
                     'alp0%1.3f_nEpisode%d_m%d.mat'], ...
                     seeds, gamma, epsilon0, alpha0, nEpisodes, m);
    load(fname);

    P = setdefaultparams_ciliaball;
    env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

    % greedy rollout with cycle detection, cycle reward, plotting
    s = env.reset();
    state_history = [];
    reward_history = [];
    found_cycle = false;
    rolloutSteps = 5000;

    for t = 1:rolloutSteps
        first_index = find(state_history == s, 1);
        if ~isempty(first_index)
            found_cycle = true;
            fprintf('t=%d, state=%d (repeat)\n', t, s);
            fprintf('found a cycle!\n');
            cycle_states = [state_history(first_index:end) s];
            cycle_rewards = reward_history(first_index:end);
            avg_cycle_reward = mean(cycle_rewards);
            fprintf('cycle states:\n');
            disp(cycle_states);
            fprintf('cycle rewards:\n');
            disp(cycle_rewards);
            fprintf('average reward over cycle = %g\n', avg_cycle_reward);
            % convert cycle states to angle pairs
            cycle_phis = zeros(length(cycle_states), 2); % matrix to store the angles for each state in cycle
            for j = 1:length(cycle_states) % loop through each state in cycle
                sub = env.state2sub(cycle_states(j)); % convert state index to subscripts
                phi = env.sub2phi(sub); % subscripts to hinge angles
                cycle_phis(j,:) = phi; % store [phi1 phi2] in row j of cycle_phis
            end
            % plot stroke
            figure;
            plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', 'LineWidth', 2);
            xlabel('\phi_1');
            ylabel('\phi_2');
            title('Stroke plot');
            grid on;
            % plot rewards over cycle
            figure;
            plot(1:length(cycle_rewards), cycle_rewards, 'o-', 'LineWidth', 2);
            xlabel('Step in cycle');
            ylabel('Reward');
            title(sprintf('Cycle rewards (avg = %.6f)', avg_cycle_reward));
            grid on;
            break
        end

        state_history(end+1) = s;

        % compute NN Q-values at current state
        sub = env.state2sub(s);
        phi = env.sub2phi(sub);
        x   = phi(:) ./ env.P.phimax(:);   % normalize to roughly [-1,1]

        z1 = net.W1*x + net.b1;
        h  = 1 ./ (1 + exp(-z1));          % sigmoid hidden layer
        q  = net.W2*h + net.b2;            % linear outputs = Q-values

        [~, a] = max(q);
        [s, r] = env.step(s, a);
        reward_history(end+1) = r;
        fprintf('t=%d, state=%d, a=%d, r=%g, maxQ=%g\n', t, state_history(end), a, r, max(q));
    end

    fout = sprintf(['cycle_tinyNN_seed%d_g%1.3f_eps0%1.2f_' ...
                    'alp0%1.3f_nEpisode%d_m%d.mat'], ...
                    seeds, gamma, epsilon0, alpha0, nEpisodes, m);

    if found_cycle
        save(fout, 'cycle_rewards', 'cycle_states', 'cycle_phis', ...
             'avg_cycle_reward', 'gamma', 'epsilon0', 'alpha0', ...
             'nEpisodes', 'maxSteps', 'm', 'seeds');
    else
        fprintf('No cycle found within %d steps.\n', rolloutSteps);
    end
end
