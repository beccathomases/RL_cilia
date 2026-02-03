%% Demo: Chebyshev curvature basis -> 2D curve
clear; clc; close all;
addpath(fullfile(fileparts(mfilename('fullpath')),'..','src'));


L = 1.0;
Npts = 500;

% Coeffs correspond to Chebyshev modes T0..T_{N-1} in the series
% inside kappa(s) = (1-xi)*sum c_n T_n(xi).
c = [ 4.0; -2.0; 0.5; 0.0; 0.0 ];  % N=5 is often plenty

out = coeffs_to_curve_cheb(c, L, Npts, 0, [0;0]);
mu = 100;  % tune
[Pwall, violated] = y_nonneg_penalty(out.s, out.y, mu, 0);

% Example total cost (to minimize)
Ebend = trapz(out.s, out.kappa.^2);  % (up to constants)
J = Ebend + Pwall;



fprintf('kappa(L) = %+0.3e (should be ~0)\n', out.checks.kappa_at_L);
fprintf('mean |r''(s)| = %0.6f (should be 1)\n', out.checks.mean_speed);
fprintf('max | |r''(s)|-1 | = %0.3e\n', out.checks.max_speed_dev);

figure;
plot(out.x, out.y, 'LineWidth', 2); axis equal; grid on;
xlabel('x'); ylabel('y');
title('Curve from Chebyshev curvature basis (free tip: \kappa(L)=0)');
hold on;
plot(out.x(1), out.y(1), 'ko', 'MarkerFaceColor','k');
plot(out.x(end), out.y(end), 'ro', 'MarkerFaceColor','r');
legend('curve','base','tip','Location','best');

figure;
plot(out.s, out.kappa, 'LineWidth', 2); grid on;
xlabel('s'); ylabel('\kappa(s)');
title('Curvature (Chebyshev basis with (1-\xi) factor)');

figure;
plot(out.s, out.theta, 'LineWidth', 2); grid on;
xlabel('s'); ylabel('\theta(s)');
title('Tangent angle');
