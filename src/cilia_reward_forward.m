function [R, info] = cilia_reward_forward(out, params)
%CILIA_REWARD_FORWARD Task A: maximize x(L) with bend + wall penalties.

    s = out.s; x = out.x; y = out.y; kappa = out.kappa;

    % Wall penalty: quadratic hinge on y<0
    yneg = max(0, -y);
    Pwall = params.mu_wall * trapz(s, yneg.^2);

    % Bending energy
    Ebend = trapz(s, kappa.^2);

    % Forward reach
    xL = x(end);

    R = params.wx*xL - params.wb*Ebend - params.ww*Pwall;

    info.xL = xL;
    info.Ebend = Ebend;
    info.Pwall = Pwall;
    info.minY = min(y);
end
