function [R, info] = cilia_reward_forward(out, params)
%CILIA_REWARD_FORWARD  Task A (side sweep): maximize lateral span of x(s).
%
% Main objective:
%   span = max_s x(s) - min_s x(s)
% Reward:
%   R = wx*span - wb*Ebend - ww*Pwall
%
% Penalties:
%   Ebend = integral kappa(s)^2 ds
%   Pwall = mu_wall * integral max(0, -y(s))^2 ds

    s = out.s; x = out.x; y = out.y; kappa = out.kappa;

    % Wall penalty: quadratic hinge on y<0
    yneg  = max(0, -y);
    Pwall = params.mu_wall * trapz(s, yneg.^2);

    % Bending energy
    Ebend = trapz(s, kappa.^2);

    % Side-sweep objective: lateral span of the filament
    span = max(x) - min(x);

    R = params.wx*span - params.wb*Ebend - params.ww*Pwall;

    % Diagnostics for plotting / logging
    info.span  = span;
    info.xMin  = min(x);
    info.xMax  = max(x);
    info.xL    = x(end);
    info.yL    = y(end);
    info.Ebend = Ebend;
    info.Pwall = Pwall;
    info.minY  = min(y);
end
