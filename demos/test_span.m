addpath(fullfile(fileparts(mfilename('fullpath')),'..','src'));
%% ---- Reward weights ----
params.wx = 1.0;               % reward on span = max(x)-min(x)
params.wb = 0.05;              % bending penalty weight
params.ww = 1.0;               % wall penalty weight multiplier
params.mu_wall = 800.0;        % INCREASED wall penalty strength (was 200)
params.theta0 = pi/2;        % clamped angle at wall y=0
params.step_cost = 0.01;       % try 0.01 or 0.02 cost to eliminate coeff bouncing

params.terminate_minY = -1e-6;   % effectively y>=0
params.crash_penalty  = -100;


L=1;
Npts = 400;
params.theta0=pi/2;
%cc = zeros(729,3); % Initialize cc to store coefficients
span = zeros(1, 81); % Preallocate span array for efficiency
reward = zeros(1, 81); % Preallocate reward array for efficiency

idx = 1;
for c0 = -4:4
    for c1= -4 :4
        for c2 = -4:4
            c=[c0;c1;c2];

            cc(idx,:) = c;
            out = coeffs_to_curve_cheb(c, L, Npts, params.theta0, [0;0]);
            [rFull, info] = cilia_reward_forward(out, params);
            allx(idx,:) = out.x;
            ally(idx,:) = out.y;
            span(idx) = info.span;
            reward(idx) = rFull;
            idx = idx + 1;
        end
    end
end

