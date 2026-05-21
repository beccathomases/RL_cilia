% Load trained tiny-NN dphi runs, do greedy rollout, detect a cycle,
% and save the cycle information.

clear; clc;

nseeds       = 1:5;
nEpisodes    = 25000;
alpha0       = 0.01;
epsilon0     = 0.3;
gamma        = 0.99;
m            = 16;
stuckPenalty = 0.05;

rolloutSteps = 1000;
minCycleLen  = 3;   % set to 2 if you want to allow 2-cycles

for seeds = nseeds
    fname = sprintf(['run_tinyNN_dphi_seed%d_g%1.3f_eps0%1.2f_' ...
                     'alp0%1.3f_nEpisode%d_m%d_sp%1.2f.mat'], ...
                     seeds, gamma, epsilon0, alpha0, nEpisodes, m, stuckPenalty);

    if ~isfile(fname)
        fprintf('Could not find %s -- skipping.\n', fname);
        continue
    end

    S = load(fname);

    if ~isfield(S,'net')
        fprintf('File %s does not contain "net" -- skipping.\n', fname);
        continue
    end

    net = S.net;

    P = setdefaultparams_ciliaball;
    env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

    % Greedy rollout with cycle detection
    s = env.reset();
    prev_s = s;   % so initial dphi = 0

    state_history  = [];
    action_history = [];
    reward_history = [];

    found_cycle = false;

    for t = 1:rolloutSteps
        first_index = find(state_history == s, 1, 'first');

        if ~isempty(first_index)
            cycle_states  = [state_history(first_index:end), s];
            cycle_actions = action_history(first_index:end);
            cycle_rewards = reward_history(first_index:end);
            cycle_len     = length(cycle_rewards);

            if cycle_len >= minCycleLen
                found_cycle = true;
                avg_cycle_reward = mean(cycle_rewards);

                fprintf('Seed %d: found cycle of length %d\n', seeds, cycle_len);
                fprintf('Average reward over cycle = %g\n', avg_cycle_reward);

                % Convert cycle states to angle pairs
                cycle_phis = zeros(length(cycle_states), 2);
                for j = 1:length(cycle_states)
                    sub = env.state2sub(cycle_states(j));
                    phi = env.sub2phi(sub);
                    cycle_phis(j,:) = phi(:)';
                end

                % Plot stroke
                figure;
                plot(cycle_phis(:,1), cycle_phis(:,2), 'o-', 'LineWidth', 2);
                xlabel('\phi_1');
                ylabel('\phi_2');
                title(sprintf('Seed %d stroke plot', seeds));
                grid on;

                % Plot rewards over cycle
                figure;
                plot(1:length(cycle_rewards), cycle_rewards, 'o-', 'LineWidth', 2);
                xlabel('Step in cycle');
                ylabel('Reward');
                title(sprintf('Seed %d cycle rewards (avg = %.6f)', ...
                    seeds, avg_cycle_reward));
                grid on;

                break
            else
                fprintf('Seed %d: ignoring short cycle of length %d\n', ...
                    seeds, cycle_len);
                % keep going if you want to look for a longer one
            end
        end

        state_history(end+1) = s;

        % Compute NN Q-values using dphi input
        x = state_to_x_dphi(env, prev_s, s);

        z1 = net.W1 * x + net.b1;
        h  = 1 ./ (1 + exp(-z1));
        q  = net.W2 * h + net.b2;

        [~, a] = max(q);
        action_history(end+1) = a;

        [s2, r] = env.step(s, a);
        reward_history(end+1) = r;

        fprintf('t=%d, state=%d, a=%d, r=%g, maxQ=%g\n', ...
            t, s, a, r, max(q));

        prev_s = s;
        s = s2;
    end

    fout = sprintf(['cycle_tinyNN_dphi_seed%d_g%1.3f_eps0%1.2f_' ...
                    'alp0%1.3f_nEpisode%d_m%d_sp%1.2f.mat'], ...
                    seeds, gamma, epsilon0, alpha0, nEpisodes, m, stuckPenalty);

    if found_cycle
        save(fout, 'cycle_states', 'cycle_actions', 'cycle_rewards', ...
             'cycle_phis', 'cycle_len', 'avg_cycle_reward', ...
             'gamma', 'epsilon0', 'alpha0', 'nEpisodes', ...
             'rolloutSteps', 'm', 'seeds', 'stuckPenalty', 'minCycleLen');
    else
        fprintf('Seed %d: no cycle of length >= %d found within %d steps.\n', ...
            seeds, minCycleLen, rolloutSteps);
    end
end


function x = state_to_x_dphi(env, s_prev, s_curr)
% Convert previous/current states into normalized input
% x = [phi1; phi2; dphi1; dphi2]

    sub_prev = env.state2sub(s_prev);
    phi_prev = env.sub2phi(sub_prev);

    sub_curr = env.state2sub(s_curr);
    phi_curr = env.sub2phi(sub_curr);

    phimax = env.P.phimax(:);

    phi_norm  = phi_curr(:) ./ phimax;
    dphi_norm = (phi_curr(:) - phi_prev(:)) ./ phimax;

    x = [phi_norm; dphi_norm];
end