function [s2, r, done, c] = cilia_step(s, a, c, cmin, cmax, L, Npts, params)
%CILIA_STEP Apply action a to coefficient vector c and compute reward.

    %#ok<NASGU> s  % (not used; kept for symmetry with other envs)

    % Decode action: 6 actions = +/- on each coefficient
    % 1: c1 +=1, 2: c1 -=1, 3: c2 +=1, 4: c2 -=1, 5: c3 +=1, 6: c3 -=1
    idx = ceil(a/2);           % 1..3
    sgn = +1;
    if mod(a,2) == 0
        sgn = -1;
    end

    % Apply update + clip
    c(idx) = c(idx) + sgn;
    c(idx) = min(max(c(idx), cmin), cmax);

    % Build curve from coefficients (Chebyshev curvature model)
    out = coeffs_to_curve_cheb(c, L, Npts, 0, [0;0]);

    % Reward for Task A
    [r, info] = cilia_reward_forward(out, params);

    % Termination: big wall violation
    done = false;
    if info.minY < params.terminate_minY
        done = true;
        % optional extra penalty for "crash"
        r = r - 5.0;
    end

    % Next state index
    s2 = cilia_state_index(c, cmin, cmax);
end
