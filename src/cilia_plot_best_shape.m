function cilia_plot_best_shape(bestC, bestSpanStored, L, Npts, params)
    out = coeffs_to_curve_cheb(bestC, L, Npts, params.theta0, [0;0]);
    [~, info] = cilia_reward_forward(out, params);

    figure;
    cilia_plot_shape(out);
    title(sprintf('Best by span | c=[%d,%d,%d] | span=%.3f (stored=%.3f) | minY=%.3f', ...
        bestC(1), bestC(2), bestC(3), info.span, bestSpanStored, info.minY));
end
