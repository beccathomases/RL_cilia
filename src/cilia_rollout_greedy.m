function [trajC, trajR, finalOut, finalInfo] = cilia_rollout_greedy(Q, c0, H, cmin, cmax, L, Npts, params)
%CILIA_ROLLOUT_GREEDY Roll out greedy policy from initial coefficients c0.

    c = c0(:);
    s = cilia_state_index(c, cmin, cmax);

    trajC = zeros(3, H+1);
    trajC(:,1) = c;

    trajR = zeros(H,1);

    done = false;
    for t = 1:H
        [~, a] = max(Q(s,:));
        [s2, r, done, c] = cilia_step(s, a, c, cmin, cmax, L, Npts, params);

        trajC(:,t+1) = c;
        trajR(t) = r;

        s = s2;
        if done, break; end
    end

    finalOut = coeffs_to_curve_cheb(c, L, Npts, params.theta0, [0;0]);
    [~, finalInfo] = cilia_reward_forward(finalOut, params);

    % Trim unused tail if ended early
    if done
        trajC = trajC(:,1:(t+1));
        trajR = trajR(1:t);
    end
end
