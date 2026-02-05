function [s2, r, done, c] = cilia_step(s, a, c, cmin, cmax, L, Npts, params, t, H)
%CILIA_STEP Apply action a to coefficient vector c and compute reward.
% Terminal-reward design version WITH hard wall feasibility:
%   - Episodes run up to H steps, unless the curve violates the wall constraint.
%   - If minY < terminate_minY: terminate immediately with crash_penalty.
%   - Otherwise: small step_cost each nonterminal step, terminal reward at t==H.
%
% Requires params fields:
%   params.theta0
%   params.step_cost
%   params.terminate_minY   (e.g., -1e-6)
%   params.crash_penalty    (e.g., -50)

    % Decode action: 6 actions = +/- on each coefficient
    % 1: c0 +=1, 2: c0 -=1, 3: c1 +=1, 4: c1 -=1, 5: c2 +=1, 6: c2 -=1
    idx = ceil(a/2);           % 1..3
    sgn = +1;
    if mod(a,2) == 0
        sgn = -1;
    end

    % Apply update + clip
    c(idx) = c(idx) + sgn;
    c(idx) = min(max(c(idx), cmin), cmax);

    % Build curve and compute full objective + diagnostics
    out = coeffs_to_curve_cheb(c, L, Npts, params.theta0, [0;0]);
    [rFull, info] = cilia_reward_forward(out, params);

    % ---- Hard wall feasibility (terminate on violation) ----
    if info.minY < params.terminate_minY
        done = true;
        r = params.crash_penalty;
        s2 = cilia_state_index(c, cmin, cmax);
        return;
    end

    % Episode ends after H steps
    done = (t >= H);

    % Reward: terminal objective + small per-step cost
    if done
        r = rFull;
    else
        r = -params.step_cost;
    end

    % Next state index
    s2 = cilia_state_index(c, cmin, cmax);
end
