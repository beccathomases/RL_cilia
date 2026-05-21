clear; clc;

seedA = 1;
seedB = 7;

nEpisodes = 1000;
alpha0 = 0.99;
epsilon0 = 0.75;
gamma = 0.99;

fnameA = sprintf('run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
    seedA, gamma, epsilon0, alpha0, nEpisodes);
fnameB = sprintf('run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
    seedB, gamma, epsilon0, alpha0, nEpisodes);

SA = load(fnameA);
SB = load(fnameB);

Q1 = SA.Q;
Q2 = SB.Q;

P = setdefaultparams_ciliaball;
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

% extract cycle from seedA as reference
[foundA, cycle_statesA, cycle_rewardsA] = find_greedy_cycle_from_Q(env, Q1, 500);
[foundB, cycle_statesB, cycle_rewardsB] = find_greedy_cycle_from_Q(env, Q2, 500);

disp('Seed A cycle states:');
disp(cycle_statesA);

disp('Seed B cycle states:');
disp(cycle_statesB);

fprintf('Same cycle states? %d\n', isequal(cycle_statesA(:), cycle_statesB(:)));

% compare greedy actions on the states of cycle A (excluding repeated closing state)
cyc = cycle_statesA;
if cyc(1) == cyc(end)
    cyc = cyc(1:end-1);
end

fprintf('\nState   actionA   actionB\n');
for k = 1:numel(cyc)
    s = cyc(k);
    [~, a1] = max(Q1(s,:));
    [~, a2] = max(Q2(s,:));
    fprintf('%4d     %4d     %4d\n', s, a1, a2);
end


function [found_cycle, cycle_states, cycle_rewards] = find_greedy_cycle_from_Q(env, Q, rolloutSteps)
    s = env.reset();
    state_history = [];
    reward_history = [];
    found_cycle = false;
    cycle_states = [];
    cycle_rewards = [];

    for t = 1:rolloutSteps
        first_index = find(state_history == s, 1, 'first');
        if ~isempty(first_index)
            found_cycle = true;
            cycle_states = [state_history(first_index:end), s];
            cycle_rewards = reward_history(first_index:end);
            return
        end
        state_history(end+1) = s; %#ok<AGROW>
        [~, a] = max(Q(s,:));
        [s, r] = env.step(s, a);
        reward_history(end+1) = r; %#ok<AGROW>
    end
end