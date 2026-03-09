% 03/09/26 -
%  Updated Qlearning code to vary training rates
%  and saving the data from each random seed trial

P = setdefaultparams_ciliaball;
% Precompute is nice for speed once physics is "frozen" leave it "true"
% here
env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',true));

maxSteps = 100; % fixed for these experiments

% Tanya to vary nEpisodes
nEpisodes = 10000; % change to 5000, 25000, 50000 
% default 10000;

% Kyla to vary alpha0
alpha0 = .99; % Initial value for how quickly do i update my table
% default 0.99;
% change to 0.75, 0.5, 0.2 
% will decay like alpha0*(.999)^(episodes-1) with a floor of 0.05
alphafloor = 0.05;  % do not change

% Charlotte to vary epsilon0
epsilon0 = 0.5; % initial exploration vs exploitation with a floor of 0.1
% default 0.5;
% change to 1, 0.2 and 0.1 
% will decay like alpha0*(.999)^(episodes-1)
epsilonfloor = 0.1; % do not change

% Lily to vary gamma
gamma = .99;  % change to 0.999, 0.98, 0.97
% default .99;

for seeds = 1:10
    % Set the random seed for reproducibility
    rng(seeds);
    Q = zeros(env.nStates, env.nActions);
    epReturn = zeros(nEpisodes,1);

    for ep = 1:nEpisodes
        s = env.reset();
        G = 0;
        epsilon = max(epsilonfloor, epsilon0*(0.999)^(ep-1)); % Geometric decay with floor
        alpha   = max(alphafloor,  alpha0*(0.999)^(ep-1));   % Geometric decay with floor
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
    % Store the return for each seed trial
    fname = sprintf('run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat',seeds,gamma,epsilon0,alpha0,nEpisodes);
    save(fname,'Q','epReturn','gamma','epsilon0','alpha0','nEpisodes','maxSteps','seeds');
    

end