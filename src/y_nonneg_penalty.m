function [P, violated] = y_nonneg_penalty(s, y, mu, tol)
% Penalize y(s) < 0
    if nargin < 4, tol = 0; end
    yneg = max(0, -(y - tol));
    P = mu * trapz(s, yneg.^2);
    violated = any(y < tol);
end
