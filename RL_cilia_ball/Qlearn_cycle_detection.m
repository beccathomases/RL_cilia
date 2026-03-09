% This code will load a training run and find a cycle and save the
% information needed to make a movie of that cycle
% change the parameters below to load the file you want to look at

nEpisodes = 10000; 
alpha0 = .99; 
epsilon0 = 0.5; 
gamma = .99;  
seeds = 3;

fname = sprintf('run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat',seeds,gamma,epsilon0,alpha0,nEpisodes);
load(fname);

P = setdefaultparams_ciliaball;
% Precompute is nice for speed once physics is "frozen"
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

% greedy rollout 2.0 with cycle detection, cycle reward, plotting
s = env.reset();
state_history = []; % visited states in order
reward_history = []; % rewards for each action
found_cycle = false;

for t = 1:500
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
    fprintf('t=%d, state=%d\n', t, s);
    [~, a] = max(Q(s,:));
    [s, r] = env.step(s, a);
    reward_history(end+1) = r;
end

fout = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
    seeds, gamma, epsilon0, alpha0, nEpisodes);

if found_cycle
    save(fout,'cycle_rewards','cycle_states','avg_cycle_reward');
else
    fprintf('No cycle found within 500 steps, so nothing was saved.\n');
end