function cilia_plot_best_shape(bestC, bestReturn, L, Npts, params)
%CILIA_PLOT_BEST_SHAPE Plot the best shape found during training.

    out = coeffs_to_curve_cheb(bestC, L, Npts, 0, [0;0]);
    [R, info] = cilia_reward_forward(out, params);

    figure;
    cilia_plot_shape(out);
    title(sprintf('Best found | c=[%d,%d,%d] | R=%.3f (stored=%.3f) | xL=%.3f | minY=%.3f', ...
        bestC(1), bestC(2), bestC(3), R, bestReturn, info.xL, info.minY));
end
