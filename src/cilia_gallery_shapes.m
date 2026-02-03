function cilia_gallery_shapes(cmin, cmax, fixed_c2, L, Npts, params, valsToShow)
%CILIA_GALLERY_SHAPES  Gallery of centerlines over (c0,c1) grid at fixed c2.
%
% Inputs:
%   cmin,cmax    : coefficient bounds (e.g., -4, +4)
%   fixed_c2     : fixed value for c2 (e.g., 0)
%   L, Npts      : curve parameters
%   params       : reward params struct (used to compute minY, xL if desired)
%   valsToShow   : optional vector of coefficient values to include (subset)
%                 e.g., [-4 -2 0 2 4]. If omitted, uses cmin:cmax (all 9).
%
% Makes a figure with subplots: rows = c1, cols = c0.

    if nargin < 7 || isempty(valsToShow)
        valsToShow = cmin:cmax;
    end

    vals = valsToShow(:).';
    n = numel(vals);

    figure('Name', sprintf('Cilia shape gallery (c2=%d)', fixed_c2), ...
           'Color', 'w');

    % Precompute some axis limits for consistent view
    % (optional: you can tighten these after a quick run)
    xMin = 0; xMax = 1.2*L;
    yMin = -0.2*L; yMax = 1.2*L;

    for i1 = 1:n
        for i0 = 1:n
            c0 = vals(i0);
            c1 = vals(i1);
            c2 = fixed_c2;

            c = [c0; c1; c2];
            out = coeffs_to_curve_cheb(c, L, Npts, 0, [0;0]);

            % Compute quick diagnostics (optional)
            [~, info] = cilia_reward_forward(out, params);

            ax = subplot(n, n, (i1-1)*n + i0); %#ok<LAXES>
            plot(out.x, out.y, 'LineWidth', 1); hold on;
            yline(0, ':');
            plot(out.x(1), out.y(1), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 3);
            plot(out.x(end), out.y(end), 'ro', 'MarkerFaceColor','r', 'MarkerSize', 3);

            axis equal;
            xlim([xMin xMax]);
            ylim([yMin yMax]);
            grid on;
            set(ax, 'XTick', [], 'YTick', []);

            % Panel title: (c0,c1) and minY/xL (tiny)
            title(sprintf('c0=%d,c1=%d\nxL=%.2f, minY=%.2f', ...
                c0, c1, info.xL, info.minY), 'FontSize', 7);

            hold off;
        end
    end

    sgtitle(sprintf('Shape gallery at fixed c2=%d (Chebyshev curvature, free tip \\kappa(L)=0)', fixed_c2));
end
