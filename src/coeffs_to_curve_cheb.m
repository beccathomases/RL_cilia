function out = coeffs_to_curve_cheb(c, L, Npts, theta0, r0)
%COEFFS_TO_CURVE_CHEB Chebyshev curvature basis -> 2D inextensible curve.
%
% xi = 2*s/L - 1 in [-1,1]
% kappa(s) = (1 - xi) * sum_{n=0}^{N-1} c_n T_n(xi)
% => kappa(L)=0 because xi(L)=1.
%
% theta'(s)=kappa(s); x'(s)=cos(theta); y'(s)=sin(theta)

    if nargin < 4 || isempty(theta0), theta0 = 0; end
    if nargin < 5 || isempty(r0), r0 = [0; 0]; end

    c = c(:);
    N = numel(c);

    s  = linspace(0, L, Npts).';
    xi = 2*s/L - 1;

    % Chebyshev series S(xi) = sum c_n T_n(xi)
    S = zeros(Npts,1);

    Tnm1 = ones(Npts,1);       % T0
    S = S + c(1) * Tnm1;

    if N >= 2
        Tn = xi;               % T1
        S = S + c(2) * Tn;

        for n = 2:(N-1)        % build T2..T_{N-1}
            Tnp1 = 2*xi.*Tn - Tnm1;
            S = S + c(n+1) * Tnp1;
            Tnm1 = Tn;
            Tn = Tnp1;
        end
    end

    kappa = (1 - xi) .* S;

    theta = theta0 + cumtrapz(s, kappa);

    dx = cos(theta);
    dy = sin(theta);

    x = r0(1) + cumtrapz(s, dx);
    y = r0(2) + cumtrapz(s, dy);

    speed = sqrt(dx.^2 + dy.^2);

    out.s = s;
    out.xi = xi;
    out.kappa = kappa;
    out.theta = theta;
    out.x = x;
    out.y = y;
    out.tip = [x(end); y(end)];

    out.checks.kappa_at_L = kappa(end);
    out.checks.mean_speed = mean(speed);
    out.checks.max_speed_dev = max(abs(speed-1));

    out.meta.L = L;
    out.meta.N = N;
    out.meta.Npts = Npts;
    out.meta.theta0 = theta0;
    out.meta.r0 = r0;
end
